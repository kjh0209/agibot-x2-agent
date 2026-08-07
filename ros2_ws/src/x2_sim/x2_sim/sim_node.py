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
import math

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from aimdk_msgs.msg import JointState, JointStateArray, McLocomotionVelocity

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

        model_path = self.get_parameter('model_path').value
        self.model = mujoco.MjModel.from_xml_path(model_path)
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

        self.prev_base_linvel = np.zeros(3)
        self.seq = 0

        # 물리 스텝: publish_rate 주기마다 실시간 분량만큼 서브스텝
        rate = float(self.get_parameter('publish_rate').value)
        self.substeps = max(1, round(1.0 / rate / self.model.opt.timestep))
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f'시뮬 시작: {rate}Hz 퍼블리시, 스텝당 {self.substeps} 서브스텝')

    # ---------- 콜백 ----------
    def on_cmd_vel(self, msg: McLocomotionVelocity):
        self.cmd_vel = np.array([
            msg.forward_velocity, msg.lateral_velocity, msg.angular_velocity])
        self.cmd_time = self.data.time
        self.get_logger().info(
            f'이동 명령 수신: vx={msg.forward_velocity:.2f} '
            f'vy={msg.lateral_velocity:.2f} wz={msg.angular_velocity:.2f} '
            f'(source={msg.source!r})')

    # ---------- 메인 루프 ----------
    def tick(self):
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
        tau = self.kp * (self.q_ref - q) - self.kd * v
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
