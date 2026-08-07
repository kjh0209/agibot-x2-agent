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
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu
from std_msgs.msg import String
from std_srvs.srv import Trigger

from aimdk_msgs.msg import (
    JointState,
    JointStateArray,
    McLocomotionVelocity,
    PmuState,
)

from x2_sim.mc_services import McMock
from x2_sim.worlds import DEFAULT_WORLD, WORLDS

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


GEOM_TYPES = {
    'box': mujoco.mjtGeom.mjGEOM_BOX,
    'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER,
    'sphere': mujoco.mjtGeom.mjGEOM_SPHERE,
}


def load_model_with_camera_and_props(model_path: str,
                                     world_name: str = DEFAULT_WORLD):
    """공식 MJCF 로드 + 벤더 파일 무수정으로 카메라·월드 소품 주입.

    - 카메라: head_pitch_link 전면, 수직 FOV 55°(매뉴얼 RGB-D 스펙), 18° 하향
    - 소품/조명: worlds.py의 월드 정의에 따름
    반환: (compiled model, prop 이름 목록)
    """
    world = WORLDS[world_name]
    spec = mujoco.MjSpec.from_file(model_path)

    head = spec.body('head_pitch_link')
    cam = head.add_camera()
    cam.name = CAMERA_NAME
    cam.pos = [0.06, 0.0, 0.12]
    p = math.radians(18.0)
    s, c = math.sin(p), math.cos(p)
    rot = np.array([[0.0, s, -c],
                    [-1.0, 0.0, 0.0],
                    [0.0, c, s]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    cam.quat = quat
    cam.fovy = 55.0

    wb = spec.worldbody
    for pos, direction, diffuse in world['lights']:
        light = wb.add_light()
        light.pos = pos
        light.dir = direction
        light.diffuse = diffuse
        light.castshadow = False

    prop_names = []
    for name, gtype, size, pos, rgba in world['props']:
        g = wb.add_geom()
        g.name = name
        g.type = GEOM_TYPES[gtype]
        g.size = size
        g.pos = pos
        g.rgba = rgba
        prop_names.append(name)

    return spec.compile(), prop_names


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

        self.declare_parameter('world', DEFAULT_WORLD)
        self.kp = float(self.get_parameter('kp').value)
        self.kd = float(self.get_parameter('kd').value)
        self.hold_base = bool(self.get_parameter('hold_base').value)
        w = int(self.get_parameter('camera_width').value)
        h = int(self.get_parameter('camera_height').value)
        self.cam_size = (w, h)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.renderer = None
        self.seq = 0

        # 월드 로드 (모델/매핑/베이스/소품/렌더러 전부 이 안에서 구성 —
        # 런타임 월드 전환 시에도 같은 메서드 재사용)
        self._build_world(str(self.get_parameter('world').value))

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
        # 위치(odometry) — 실기의 SLAM odom 토픽명 그대로
        self.pub_odom = self.create_publisher(
            Odometry, '/slam/lidar_odom', qos_profile_sensor_data)

        # PMU 목 — 감쇠 배터리 (만충 54.6V → 방전한계 39V, 기본 2시간)
        # "배터리 30% 미만 경고 / 15% 미만 세션 종료" 가드를 시뮬에서
        # 테스트할 수 있게 함. battery_minutes 파라미터로 감쇠 속도 조절.
        self.declare_parameter('battery_minutes', 120.0)
        self.batt_v_full, self.batt_v_empty = 54.6, 39.0
        self.batt_start_time = self.data.time
        self.pub_pmu = self.create_publisher(
            PmuState, '/aima/hal/pmu/state', qos_profile_sensor_data)
        self.create_timer(1.0, self.publish_pmu)
        self.create_subscription(
            McLocomotionVelocity, '/aima/mc/locomotion/velocity',
            self.on_cmd_vel, 10)

        # MC 목 서비스 (실기 프로토콜 강제: 모드/소스 게이트)
        self.mc = McMock(self)
        # 시뮬 전용 리셋 서비스 (웹 버튼/CLI에서 호출)
        self.create_service(Trigger, '/x2_sim/reset', self.on_reset)
        # 월드 전환: /x2_sim/set_world 구독, 현재 월드는 latched 발행
        from rclpy.qos import (DurabilityPolicy, QoSProfile)
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub_world = self.create_publisher(
            String, '/x2_sim/world', latched)
        self.create_subscription(
            String, '/x2_sim/set_world', self.on_set_world, 10)
        self.pub_world.publish(String(data=self.world_name))

        # ---------- 카메라 (Phase 2: 로봇이 보는 화면) ----------
        # 실기 토픽명 그대로. compressed가 주 소비 경로(실기 WiFi 대역폭 대비),
        # raw는 디버깅/후처리용.
        base = f'/aima/hal/sensor/{CAMERA_NAME}'
        self.pub_img_comp = self.create_publisher(
            CompressedImage, f'{base}/rgb_image/compressed',
            qos_profile_sensor_data)
        self.pub_img_raw = self.create_publisher(
            Image, f'{base}/rgb_image', qos_profile_sensor_data)
        self.pub_cam_info = self.create_publisher(
            CameraInfo, f'{base}/rgb_camera_info', qos_profile_sensor_data)
        self.cam_info = self.make_camera_info(
            self.cam_size[0], self.cam_size[1], fovy_deg=55.0)
        cam_rate = float(self.get_parameter('camera_rate').value)
        # 렌더(수십 ms)가 물리 틱을 막지 않도록 별도 콜백 그룹 + 상태 락
        # render_lock: 월드 전환 시 렌더러 교체와 렌더링의 경합 방지
        self.state_lock = threading.Lock()
        self.render_lock = threading.Lock()
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

    # ---------- 월드 구성/전환 ----------
    def _build_world(self, world_name: str):
        """월드(씬) 로드 — 초기화와 런타임 전환 양쪽에서 사용.

        모델/데이터/액추에이터 매핑/PD 목표/베이스 홀드/소품 충돌 목록/
        렌더러/배터리를 전부 새로 구성한다. 호출측에서 락을 잡는다
        (초기화 시엔 스레드가 없어 락 불필요).
        """
        model_path = self.get_parameter('model_path').value
        self.model, prop_names = load_model_with_camera_and_props(
            model_path, world_name)
        self.data = mujoco.MjData(self.model)
        self.world_name = world_name

        self.act_joint_ids = [
            self.model.actuator_trnid[i][0] for i in range(self.model.nu)]
        self.act_qpos_adr = [
            self.model.jnt_qposadr[j] for j in self.act_joint_ids]
        self.act_qvel_adr = [
            self.model.jnt_dofadr[j] for j in self.act_joint_ids]
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in self.act_joint_ids]
        self.groups = {g: [] for g in GROUP_KEYWORDS}
        for ai, name in enumerate(self.joint_names):
            g = group_of(name or '')
            if g:
                self.groups[g].append(ai)
        self.jname_to_ai = {n: i for i, n in enumerate(self.joint_names)}

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
        self.batt_start_time = self.data.time

        self.prop_gids = set()
        for name in prop_names:
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.prop_gids.add(gid)
        self._contact_prev = False
        self._blocked_logged = False

        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
        self.renderer = mujoco.Renderer(
            self.model, height=self.cam_size[1], width=self.cam_size[0])
        self.get_logger().info(
            f'월드 로드: {world_name} — 소품 {len(prop_names)}개, '
            f'nq={self.model.nq}, nu={self.model.nu}')

    def on_set_world(self, msg: String):
        name = msg.data.strip()
        if name not in WORLDS:
            self.get_logger().warning(
                f'미등록 월드 {name!r} — 사용 가능: {list(WORLDS)}')
            return
        with self.render_lock:
            with self.state_lock:
                self._build_world(name)
        self.mc.reset()
        self.pub_world.publish(String(data=name))
        self.get_logger().info(f'월드 전환 완료: {name}')

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
            self.batt_start_time = self.data.time   # 배터리 만충 복원
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
        self._blocked_logged = False   # 새 명령마다 차단 로그 1회 허용
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
    def _robot_prop_contact(self) -> bool:
        """로봇 geom과 소품 geom 사이의 접촉 존재 여부."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a, b = c.geom1, c.geom2
            a_prop, b_prop = a in self.prop_gids, b in self.prop_gids
            if a_prop == b_prop:
                continue              # 둘 다 소품이거나 둘 다 비소품
            other = b if a_prop else a
            if self.model.geom_bodyid[other] == 0:
                continue              # 바닥 등 worldbody geom 제외
            return True               # 상대가 로봇 바디 → 접촉
        return False

    def tick(self):
        with self.state_lock:
            for _ in range(self.substeps):
                self.apply_pd()
                prev_xyz = self.base_xyz.copy()
                prev_yaw = self.base_yaw
                self.apply_locomotion()
                mujoco.mj_step(self.model, self.data)
                hit = self._robot_prop_contact()
                # 차단 기준은 "마지막 승인 위치의 접촉 여부"다.
                # (직전 스텝이 아니라 승인 위치 기준이어야 접촉이 지속되는
                #  동안에도 계속 차단된다 — 진입-순간만 보면 뚫고 지나감)
                if hit and not self._contact_prev:
                    self.base_xyz[:] = prev_xyz
                    self.base_yaw = prev_yaw
                    self.cmd_vel[:] = 0.0
                    # 승인 위치는 그대로(접촉 없음) → _contact_prev 유지
                    if not self._blocked_logged:
                        self.get_logger().warning(
                            '장애물 접촉 — 이동 차단 (베이스 롤백). '
                            '실기라면 충돌 직전 상황')
                        self._blocked_logged = True
                else:
                    # 이동을 승인 — 승인 위치의 접촉 상태 갱신
                    # (시작부터 접촉 상태인 경우 탈출 이동 허용)
                    self._contact_prev = hit
        self.publish_joints()
        self.publish_imu()
        self.publish_odom()
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
        with self.render_lock:
            with self.state_lock:
                self.renderer.update_scene(self.data, camera=CAMERA_NAME)
            rgb = self.renderer.render()      # (H, W, 3) uint8
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

    def publish_odom(self):
        od = Odometry()
        od.header.stamp = self.get_clock().now().to_msg()
        od.header.frame_id = 'odom'
        od.child_frame_id = 'base_link'
        od.pose.pose.position.x = float(self.data.qpos[0])
        od.pose.pose.position.y = float(self.data.qpos[1])
        od.pose.pose.position.z = float(self.data.qpos[2])
        od.pose.pose.orientation.w = float(self.data.qpos[3])
        od.pose.pose.orientation.x = float(self.data.qpos[4])
        od.pose.pose.orientation.y = float(self.data.qpos[5])
        od.pose.pose.orientation.z = float(self.data.qpos[6])
        od.twist.twist.linear.x = float(self.data.qvel[0])
        od.twist.twist.linear.y = float(self.data.qvel[1])
        od.twist.twist.angular.z = float(self.data.qvel[5])
        self.pub_odom.publish(od)

    def publish_pmu(self):
        mins = float(self.get_parameter('battery_minutes').value)
        elapsed_min = (self.data.time - self.batt_start_time) / 60.0
        frac = max(0.0, 1.0 - elapsed_min / mins)
        msg = PmuState()
        msg.battery_voltage = self.batt_v_empty + frac * (
            self.batt_v_full - self.batt_v_empty)
        msg.battery_pack_voltage = msg.battery_voltage
        msg.battery_current = -3.0 if self.cmd_vel.any() else -1.0
        msg.battery_temperature = 32.0
        self.pub_pmu.publish(msg)


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
