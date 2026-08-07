#!/usr/bin/env python3
"""로봇 에이전트 노드 — 컨텍스트 수집 + 채팅 인터페이스.

토픽:
  구독  /agent/user_msg   std_msgs/String   사용자 메시지 (CLI/웹에서)
  발행  /agent/reply      std_msgs/String   에이전트 최종 답변
  발행  /agent/event      std_msgs/String   툴 실행 이벤트 (UI 표시용)
  발행  /agent/say        std_msgs/String   speak() 출력 (음성 말풍선)
  발행  /agent/emoji      std_msgs/String   show_emoji() 출력
  구독  /aima/hal/sensor/rgbd_head_front/rgb_image/compressed (프레임 캐시)
  구독  /aima/hal/joint/*/state, /aima/hal/imu/chest/state    (상태 캐시)
"""
import collections
import math
import re
import threading

import rclpy
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Imu
from std_msgs.msg import String

STOP_PATTERN = re.compile(r'(멈춰|멈추|정지|그만|스톱|stop)', re.IGNORECASE)

from aimdk_msgs.msg import JointStateArray

from robot_agent.llm import MODEL, AgentBrain
from robot_agent.tools import ToolExecutor


class AgentNode(Node):
    def __init__(self):
        super().__init__('robot_agent')
        # ---------- 컨텍스트 수집기 ----------
        self.frames = collections.deque(maxlen=3)   # 최근 JPEG 프레임
        self.joint_pos = {}                          # group → n joints
        self.imu_quat = (1.0, 0.0, 0.0, 0.0)
        self.create_subscription(
            CompressedImage,
            '/aima/hal/sensor/rgbd_head_front/rgb_image/compressed',
            lambda m: self.frames.append(bytes(m.data)),
            qos_profile_sensor_data)
        for g in ('arm', 'leg', 'waist', 'head'):
            self.create_subscription(
                JointStateArray, f'/aima/hal/joint/{g}/state',
                lambda m, g=g: self.joint_pos.__setitem__(g, len(m.joints)),
                qos_profile_sensor_data)
        self.create_subscription(
            Imu, '/aima/hal/imu/chest/state',
            lambda m: setattr(self, 'imu_quat', (
                m.orientation.w, m.orientation.x,
                m.orientation.y, m.orientation.z)),
            qos_profile_sensor_data)
        self.odom = None                              # (x, y)
        self.odom_origin = None                       # 세션 시작 위치
        self.create_subscription(
            Odometry, '/slam/lidar_odom', self.on_odom,
            qos_profile_sensor_data)

        # ---------- 출력 ----------
        self.pub_reply = self.create_publisher(String, '/agent/reply', 10)
        self.pub_event = self.create_publisher(String, '/agent/event', 10)
        self.pub_say = self.create_publisher(String, '/agent/say', 10)
        self.pub_emoji = self.create_publisher(String, '/agent/emoji', 10)

        # ---------- 두뇌 ----------
        self.executor_tools = ToolExecutor(self)
        self.brain = AgentBrain(
            self.executor_tools,
            on_event=lambda kind, text: self.pub_event.publish(
                String(data=f'[{kind}] {text}')))
        self._busy = threading.Lock()

        # 채팅 처리는 전용 콜백 그룹 (툴의 서비스 대기 동안 다른 콜백 허용)
        chat_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            String, '/agent/user_msg', self.on_user_msg, 10,
            callback_group=chat_group)

        self.get_logger().info(f'에이전트 준비 완료 (model={MODEL})')

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.odom = (p.x, p.y)
        if self.odom_origin is None:
            self.odom_origin = (p.x, p.y)

    # ---------- ToolExecutor가 쓰는 인터페이스 ----------
    def latest_frame(self):
        return self.frames[-1] if self.frames else None

    def status_summary(self):
        w, x, y, z = self.imu_quat
        yaw = math.degrees(math.atan2(2 * (w * z + x * y),
                                      1 - 2 * (y * y + z * z)))
        # roll/pitch로 직립 여부를 데이터로 판정 (이미지 추측 방지)
        roll = math.degrees(math.atan2(2 * (w * x + y * z),
                                       1 - 2 * (x * x + y * y)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
        st = {
            'joints': dict(self.joint_pos),
            'yaw_deg': round(yaw, 1),
            'upright': abs(roll) < 30 and abs(pitch) < 30,
            'battery': '시뮬에는 배터리 없음 (실기: /aima/hal/pmu/state)',
        }
        if self.odom:
            st['position'] = {'x': round(self.odom[0], 2),
                              'y': round(self.odom[1], 2)}
            if self.odom_origin:
                dx = self.odom[0] - self.odom_origin[0]
                dy = self.odom[1] - self.odom_origin[1]
                st['from_start'] = {
                    'dx': round(dx, 2), 'dy': round(dy, 2),
                    'distance_m': round(math.hypot(dx, dy), 2)}
        return st

    def publish_say(self, text):
        self.pub_say.publish(String(data=text))

    def publish_emoji(self, emoji):
        self.pub_emoji.publish(String(data=emoji))

    # ---------- 채팅 ----------
    def on_user_msg(self, msg: String):
        text = msg.data.strip()
        if not text:
            return
        # ★멈춰 fast-path: LLM/락을 거치지 않고 즉시 정지 (안전 최우선)
        if STOP_PATTERN.search(text):
            self.executor_tools.stop()
            self.get_logger().warning(f'fast-path 정지 실행: {text!r}')
            self.pub_reply.publish(String(data='바로 멈췄어!'))
            if self._busy.acquire(blocking=False):   # LLM 유휴 시 맥락 기록
                try:
                    self.brain.history.append(
                        {'role': 'user', 'content': text})
                    self.brain.history.append(
                        {'role': 'assistant',
                         'content': '(즉시 정지 실행 완료)'})
                finally:
                    self._busy.release()
            return
        if not self._busy.acquire(blocking=False):
            self.pub_reply.publish(String(
                data='(이전 메시지를 아직 처리 중이에요 — 잠시만요)'))
            return
        try:
            self.get_logger().info(f'사용자: {text}')
            reply = self.brain.chat(text)
            self.get_logger().info(f'에이전트: {reply}')
            self.pub_reply.publish(String(data=reply))
        except Exception as e:
            self.get_logger().error(f'에이전트 오류: {e}')
            self.pub_reply.publish(String(
                data=f'(오류가 났어요: {e})'))
        finally:
            self._busy.release()


def main():
    rclpy.init()
    node = AgentNode()
    executor = MultiThreadedExecutor(num_threads=4)
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
