#!/usr/bin/env python3
"""Agibot X2 자체 MuJoCo 시뮬 노드.

공식 시뮬 바이너리(aima-sim-app)가 AVX 의존으로 Apple Silicon(Rosetta)에서
실행 불가라, 공식 MJCF 모델을 직접 물리 시뮬하고 실기와 동일한 토픽/타입으로
노출한다. 에이전트/브릿지 쪽 코드는 실기 전환 시 수정이 필요 없다.

퍼블리시 (실기와 동일):
  /aima/hal/joint/{arm,leg,waist,head}/state   aimdk_msgs/JointStateArray
  /aima/hal/imu/{chest,torso}/state            sensor_msgs/Imu
구독:
  /aima/mc/locomotion/velocity                 aimdk_msgs/McLocomotionVelocity
"""
import io
import math
import threading

import mujoco
import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu
from std_srvs.srv import Trigger

from aimdk_msgs.msg import JointState, JointStateArray, McLocomotionVelocity

from x2_sim.mc_services import McMock

DEFAULT_MODEL = ('/sdk/aimdk-1.0.0/extra/x2_rl_deploy/x2_rl_deploy_mujoco/'
                 'configuration/robot/lx2501_3_t2d5/model_info/scene.xml')

# 관절 이름 → 실기 토픽 그룹 매핑
GROUP_KEYWORDS = {
    'arm': ('shoulder', 'elbow', 'wrist'),
    'leg': ('hip', 'knee', 'ankle'),
    'waist': ('waist',),
    'head': ('head',),
}


def group_of(joint_name: str):
    n = joint_name.lower()
    for group, keys in GROUP_KEYWORDS.items():
        if any(k in n for k in keys):
            return group
    return None


CAMERA_NAME = 'rgbd_head_front'


def load_model_with_camera_and_props(model_path: str):
    """공식 MJCF를 로드하되, 벤더 파일 수정 없이 스펙 단계에서
    head 카메라와 테스트 오브젝트를 추가한다.

    - 카메라: head_pitch_link(머리) 전면, 로봇 +X 방향 응시, 수직 FOV 55°
      (매뉴얼 2.5의 RGB-D 수직 FOV와 동일)
    - 오브젝트: 테이블/빨간 박스/파란 컵/노란 공 — "앞에 뭐 보여?" 검증용
    """
    spec = mujoco.MjSpec.from_file(model_path)

    head = spec.body('head_pitch_link')
    cam = head.add_camera()
    cam.name = CAMERA_NAME
    cam.pos = [0.06, 0.0, 0.12]
    # 카메라 프레임: -Z가 시선 방향. 로봇 +X를 보되 pitch_down만큼
    # 아래로 기울인다 (테이블 위 물체가 화면 중앙에 오도록).
    p = math.radians(18.0)
    s, c = math.sin(p), math.cos(p)
    rot = np.array([[0.0, s, -c],
                    [-1.0, 0.0, 0.0],
                    [0.0, c, s]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    cam.quat = quat
    cam.fovy = 55.0

    # 씬 조명 보강 (VLM이 볼 이미지이므로 밝게)
    light = spec.worldbody.add_light()
    light.pos = [1.0, 0.0, 2.5]
    light.dir = [0.1, 0.0, -1.0]
    light.diffuse = [0.9, 0.9, 0.9]
    light.castshadow = False

    wb = spec.worldbody
    props = [
        # (name, type, size, pos, rgba)
        ('table', mujoco.mjtGeom.mjGEOM_BOX,
         [0.30, 0.40, 0.36], [1.30, 0.0, 0.36], [0.55, 0.35, 0.20, 1.0]),
        ('red_box', mujoco.mjtGeom.mjGEOM_BOX,
         [0.05, 0.05, 0.05], [1.20, 0.18, 0.77], [0.85, 0.10, 0.10, 1.0]),
        ('blue_cup', mujoco.mjtGeom.mjGEOM_CYLINDER,
         [0.035, 0.06, 0.0], [1.30, -0.12, 0.78], [0.10, 0.25, 0.85, 1.0]),
        ('yellow_ball', mujoco.mjtGeom.mjGEOM_SPHERE,
         [0.06, 0.0, 0.0], [0.85, -0.55, 0.06], [0.95, 0.85, 0.10, 1.0]),
    ]
    for name, gtype, size, pos, rgba in props:
        g = wb.add_geom()
        g.name = name
        g.type = gtype
        g.size = size
        g.pos = pos
        g.rgba = rgba

    return spec.compile()


class X2SimNode(Node):
    def __init__(self):
        super().__init__('x2_sim')
        self.declare_parameter('model_path', DEFAULT_MODEL)
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('kp', 120.0)
        self.declare_parameter('kd', 4.0)
        # 베이스 고정 모드: 보행 컨트롤러가 없으므로 기본 ON.
        # 이동 명령은 고정 베이스의 목표 위치/방향을 옮기는 방식으로 처리.
        self.declare_parameter('hold_base', True)

        # CPU(OSMesa) 렌더가 프레임당 ~0.7s라 실효 1~2fps.
        # 대화형 에이전트에는 충분. (Phase 3: look()은 필요 시 단발 렌더)
        self.declare_parameter('camera_rate', 2.0)
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('jpeg_quality', 80)

        model_path = self.get_parameter('model_path').value
        self.model = load_model_with_camera_and_props(model_path)
        self.data = mujoco.MjData(self.model)
        self.get_logger().info(
            f'모델 로드: {model_path} (nq={self.model.nq}, nu={self.model.nu}, '
            f'timestep={self.model.opt.timestep})')

        # 액추에이터가 무는 관절 인덱스 정리
        self.act_joint_ids = [
            self.model.actuator_trnid[i][0] for i in range(self.model.nu)]
        self.act_qpos_adr = [
            self.model.jnt_qposadr[j] for j in self.act_joint_ids]
        self.act_qvel_adr = [
            self.model.jnt_dofadr[j] for j in self.act_joint_ids]
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in self.act_joint_ids]

        # 그룹별 (actuator_index) 목록
        self.groups = {g: [] for g in GROUP_KEYWORDS}
        for ai, name in enumerate(self.joint_names):
            g = group_of(name or '')
            if g:
                self.groups[g].append(ai)
        for g, idxs in self.groups.items():
            self.get_logger().info(f'그룹 {g}: {len(idxs)}개 관절')

        # 초기 자세 = PD 홀드 목표 (모델 기본 qpos)
        mujoco.mj_forward(self.model, self.data)
        self.q_ref = np.array(
            [self.data.qpos[a] for a in self.act_qpos_adr])
        self.kp = float(self.get_parameter('kp').value)
        self.kd = float(self.get_parameter('kd').value)

        # 이동 명령 상태
        self.cmd_vel = np.zeros(3)   # vx, vy, wz (body frame)
        self.cmd_time = 0.0

        # 베이스 고정 목표 (x, y, z, yaw) — 초기 pose에서 시작
        self.hold_base = bool(self.get_parameter('hold_base').value)
        self.base_xyz = np.array(self.data.qpos[0:3])
        qw, qx, qy, qz = self.data.qpos[3:7]
        self.base_yaw = math.atan2(2 * (qw * qz + qx * qy),
                                   1 - 2 * (qy * qy + qz * qz))

        # 퍼블리셔 (실기 토픽명)
        self.pub_joint = {
            g: self.create_publisher(
                JointStateArray, f'/aima/hal/joint/{g}/state',
                qos_profile_sensor_data)
            for g in GROUP_KEYWORDS}
        self.pub_imu = {
            name: self.create_publisher(
                Imu, f'/aima/hal/imu/{name}/state', qos_profile_sensor_data)
            for name in ('chest', 'torso')}
        self.create_subscription(
            McLocomotionVelocity, '/aima/mc/locomotion/velocity',
            self.on_cmd_vel, 10)

        # MC 목 서비스 (실기 프로토콜 강제: 모드/소스 게이트)
        self.mc = McMock(self)
        # 시뮬 전용 리셋 서비스 (웹 버튼/CLI에서 호출)
        self.create_service(Trigger, '/x2_sim/reset', self.on_reset)
        # 프리셋 모션 애니메이션 상태: (이름, 시작 sim time) 또는 None
        self.motion = None
        self.jname_to_ai = {n: i for i, n in enumerate(self.joint_names)}

        self.prev_base_linvel = np.zeros(3)
        self.seq = 0

        # ---------- 카메라 (Phase 2: 로봇이 보는 화면) ----------
        # 실기 토픽명 그대로. compressed가 주 소비 경로(실기 WiFi 대역폭 대비),
        # raw는 디버깅/후처리용.
        w = int(self.get_parameter('camera_width').value)
        h = int(self.get_parameter('camera_height').value)
        self.cam_size = (w, h)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        base = f'/aima/hal/sensor/{CAMERA_NAME}'
        self.pub_img_comp = self.create_publisher(
            CompressedImage, f'{base}/rgb_image/compressed',
            qos_profile_sensor_data)
        self.pub_img_raw = self.create_publisher(
            Image, f'{base}/rgb_image', qos_profile_sensor_data)
        self.pub_cam_info = self.create_publisher(
            CameraInfo, f'{base}/rgb_camera_info', qos_profile_sensor_data)
        self.cam_info = self.make_camera_info(w, h, fovy_deg=55.0)
        cam_rate = float(self.get_parameter('camera_rate').value)
        # 렌더(수십 ms)가 물리 틱을 막지 않도록 별도 콜백 그룹 + 상태 락
        self.state_lock = threading.Lock()
        self.cam_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / cam_rate, self.tick_camera,
                          callback_group=self.cam_group)

        # 물리 스텝: publish_rate 주기마다 실시간 분량만큼 서브스텝
        rate = float(self.get_parameter('publish_rate').value)
        self.substeps = max(1, round(1.0 / rate / self.model.opt.timestep))
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f'시뮬 시작: {rate}Hz 퍼블리시, 스텝당 {self.substeps} 서브스텝, '
            f'카메라 {w}x{h}@{cam_rate}Hz → {base}/rgb_image[/compressed]')

    # ---------- 콜백 ----------
    def on_reset(self, req, res):
        """시뮬을 최초 상태로 완전 초기화 (물리/베이스/모션/MC 상태)."""
        with self.state_lock:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.q_ref = np.array(
                [self.data.qpos[a] for a in self.act_qpos_adr])
            self.cmd_vel = np.zeros(3)
            self.cmd_time = 0.0
            self.motion = None
            self.base_xyz = np.array(self.data.qpos[0:3])
            qw, qx, qy, qz = self.data.qpos[3:7]
            self.base_yaw = math.atan2(2 * (qw * qz + qx * qy),
                                       1 - 2 * (qy * qy + qz * qz))
            self.prev_base_linvel = np.zeros(3)
        self.mc.reset()
        self.get_logger().info('시뮬 초기화 완료 (초기 자세/위치/MC 상태)')
        res.success = True
        res.message = '시뮬 초기화 완료'
        return res

    def on_cmd_vel(self, msg: McLocomotionVelocity):
        # 실기 의미론 게이트: locomotion 모드 + 등록된 소스가 아니면 무시
        if not self.mc.gate_velocity(msg.source):
            return
        self.cmd_vel = np.array([
            msg.forward_velocity, msg.lateral_velocity, msg.angular_velocity])
        self.cmd_time = self.data.time
        self.get_logger().info(
            f'이동 명령 수신: vx={msg.forward_velocity:.2f} '
            f'vy={msg.lateral_velocity:.2f} wz={msg.angular_velocity:.2f} '
            f'(source={msg.source!r})')

    def clear_locomotion(self):
        """비상(damping/zero-torque) 진입 시 이동 명령 즉시 소거."""
        self.cmd_vel = np.zeros(3)

    def play_motion(self, name: str):
        """프리셋 모션 시작 (McMock에서 검증 후 호출됨)."""
        self.motion = (name, self.data.time)

    def motion_offsets(self):
        """진행 중인 프리셋 모션의 관절 오프셋 계산 (q_ref에 가산)."""
        if self.motion is None:
            return None
        name, t0 = self.motion
        t = self.data.time - t0
        dur = 2.5
        if t >= dur:
            self.motion = None
            return None
        # 부드러운 시작/끝 (0→1→0 envelope)
        env = math.sin(math.pi * min(t / dur, 1.0))
        off = {}
        if name in ('POINT_HEAD',):
            off['head_pitch_joint'] = 0.3 * env * math.sin(2 * math.pi * 1.5 * t)
        elif name in ('SHAKE_HEAD',):
            off['head_yaw_joint'] = 0.35 * env * math.sin(2 * math.pi * 1.5 * t)
        else:
            # 손 계열 기본 동작: 오른팔 들어 흔들기
            off['right_shoulder_pitch_joint'] = -1.3 * env
            off['right_shoulder_roll_joint'] = -0.3 * env
            off['right_elbow_joint'] = (-0.6 + 0.4 * math.sin(
                2 * math.pi * 2.0 * t)) * env
        return off

    # ---------- 메인 루프 ----------
    def tick(self):
        with self.state_lock:
            for _ in range(self.substeps):
                self.apply_pd()
                self.apply_locomotion()
                mujoco.mj_step(self.model, self.data)
        self.publish_joints()
        self.publish_imu()
        self.seq += 1

    def apply_pd(self):
        q = self.data.qpos[self.act_qpos_adr]
        v = self.data.qvel[self.act_qvel_adr]
        q_target = self.q_ref
        offs = self.motion_offsets()
        if offs:
            q_target = self.q_ref.copy()
            for jname, delta in offs.items():
                ai = self.jname_to_ai.get(jname)
                if ai is not None:
                    q_target[ai] += delta
        tau = self.kp * (q_target - q) - self.kd * v
        self.data.ctrl[:] = np.clip(
            tau, self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1]) \
            if self.model.actuator_ctrllimited.any() else tau

    def apply_locomotion(self):
        """키네마틱 베이스 이동 (보행 컨트롤러 대체).

        hold_base 모드: 베이스 pose를 목표값으로 매 스텝 고정하고,
        이동 명령은 목표값(x, y, yaw)을 적분해 옮긴다. 로봇이 넘어질 수
        없어 안정적이며, 에이전트 입장에선 '명령 → 이동'이라는 관측
        결과가 실기와 동일하므로 Phase 2~4 개발에 충분하다.
        """
        dt = self.model.opt.timestep
        active = (self.data.time - self.cmd_time <= 0.5) and self.cmd_vel.any()
        if active:
            c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
            self.base_xyz[0] += (c * self.cmd_vel[0] - s * self.cmd_vel[1]) * dt
            self.base_xyz[1] += (s * self.cmd_vel[0] + c * self.cmd_vel[1]) * dt
            self.base_yaw += self.cmd_vel[2] * dt

        if self.hold_base:
            self.data.qpos[0:3] = self.base_xyz
            half = self.base_yaw / 2.0
            self.data.qpos[3:7] = (math.cos(half), 0.0, 0.0, math.sin(half))
            self.data.qvel[0:6] = 0.0
            if active:
                # IMU 각속도/가속에 명령이 반영되게 수평 속도만 남긴다
                c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
                self.data.qvel[0] = c * self.cmd_vel[0] - s * self.cmd_vel[1]
                self.data.qvel[1] = s * self.cmd_vel[0] + c * self.cmd_vel[1]
                self.data.qvel[5] = self.cmd_vel[2]
        elif active:
            c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
            self.data.qvel[0] = c * self.cmd_vel[0] - s * self.cmd_vel[1]
            self.data.qvel[1] = s * self.cmd_vel[0] + c * self.cmd_vel[1]
            self.data.qvel[5] = self.cmd_vel[2]

    # ---------- 카메라 ----------
    def make_camera_info(self, w, h, fovy_deg):
        info = CameraInfo()
        info.width = w
        info.height = h
        fy = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        cx, cy = w / 2.0, h / 2.0
        info.k = [fy, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.p = [fy, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.header.frame_id = CAMERA_NAME
        return info

    def tick_camera(self):
        with self.state_lock:
            self.renderer.update_scene(self.data, camera=CAMERA_NAME)
        rgb = self.renderer.render()          # (H, W, 3) uint8
        stamp = self.get_clock().now().to_msg()

        buf = io.BytesIO()
        PILImage.fromarray(rgb).save(buf, format='JPEG',
                                     quality=self.jpeg_quality)
        comp = CompressedImage()
        comp.header.stamp = stamp
        comp.header.frame_id = CAMERA_NAME
        comp.format = 'jpeg'
        comp.data = buf.getvalue()
        self.pub_img_comp.publish(comp)

        raw = Image()
        raw.header = comp.header
        raw.height, raw.width = rgb.shape[0], rgb.shape[1]
        raw.encoding = 'rgb8'
        raw.step = rgb.shape[1] * 3
        raw.data = rgb.tobytes()
        self.pub_img_raw.publish(raw)

        self.cam_info.header.stamp = stamp
        self.pub_cam_info.publish(self.cam_info)

    # ---------- 퍼블리시 ----------
    def stamp_header(self, header):
        t = self.get_clock().now().to_msg()
        header.stamp = t
        header.meas_stamp = t
        header.sequence = self.seq % (2 ** 32)

    def publish_joints(self):
        q = self.data.qpos
        v = self.data.qvel
        tau = self.data.actuator_force
        for g, idxs in self.groups.items():
            msg = JointStateArray()
            self.stamp_header(msg.header)
            for ai in idxs:
                js = JointState()
                js.name = self.joint_names[ai] or f'joint_{ai}'
                js.position = float(q[self.act_qpos_adr[ai]])
                js.velocity = float(v[self.act_qvel_adr[ai]])
                js.effort = float(tau[ai])
                msg.joints.append(js)
            self.pub_joint[g].publish(msg)

    def publish_imu(self):
        # freejoint 상태로 IMU 근사 (chest/torso 동일 값 사용)
        quat = self.data.qpos[3:7]        # w x y z
        angvel = self.data.qvel[3:6]
        linvel = np.array(self.data.qvel[0:3])
        dt = self.model.opt.timestep * self.substeps
        accel = (linvel - self.prev_base_linvel) / dt
        accel[2] += 9.81
        self.prev_base_linvel = linvel

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.orientation.w = float(quat[0])
        msg.orientation.x = float(quat[1])
        msg.orientation.y = float(quat[2])
        msg.orientation.z = float(quat[3])
        msg.angular_velocity.x = float(angvel[0])
        msg.angular_velocity.y = float(angvel[1])
        msg.angular_velocity.z = float(angvel[2])
        msg.linear_acceleration.x = float(accel[0])
        msg.linear_acceleration.y = float(accel[1])
        msg.linear_acceleration.z = float(accel[2])
        for pub in self.pub_imu.values():
            pub.publish(msg)


def main():
    rclpy.init()
    node = X2SimNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
