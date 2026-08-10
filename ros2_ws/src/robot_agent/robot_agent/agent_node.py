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
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Imu
from std_msgs.msg import String

STOP_PATTERN = re.compile(r'(멈춰|멈추|정지|그만|스톱|stop)', re.IGNORECASE)

from aimdk_msgs.msg import JointStateArray, PmuState

BATT_FULL_V, BATT_EMPTY_V = 54.6, 39.0

from robot_agent.llm import MODEL, AgentBrain
from robot_agent.tools import ToolExecutor


class AgentNode(Node):
    def __init__(self):
        super().__init__('robot_agent')
        # ---------- 컨텍스트 수집기 ----------
        self.frames = collections.deque(maxlen=3)   # 최근 JPEG 프레임
        self.frame_count = 0
        self.frame_cond = threading.Condition()
        self.joint_pos = {}                          # group → n joints
        self.imu_quat = (1.0, 0.0, 0.0, 0.0)
        self.create_subscription(
            CompressedImage,
            '/aima/hal/sensor/rgbd_head_front/rgb_image/compressed',
            self.on_frame, qos_profile_sensor_data)
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
        self.battery_pct = None
        self.create_subscription(
            PmuState, '/aima/hal/pmu/state', self.on_pmu,
            qos_profile_sensor_data)
        # 현재 월드(장소) — latched 구독 (go_to_place 도착 확인에도 사용)
        from rclpy.qos import DurabilityPolicy, QoSProfile
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.current_world = None
        self.create_subscription(
            String, '/x2_sim/world',
            lambda m: setattr(self, 'current_world', m.data), latched)
        # 주변 물체 + 들고 있는 물체
        import json as _json
        self.objects = {}
        self.landmarks = {}

        def _on_objects(m):
            d = _json.loads(m.data)
            self.objects = d.get('objects', {})
            self.landmarks = d.get('landmarks', {})

        self.carrying = ''
        self.create_subscription(
            String, '/x2_sim/objects', _on_objects, 10)
        latched3 = QoSProfile(depth=1)
        latched3.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String, '/x2_sim/carrying',
            lambda m: setattr(self, 'carrying', m.data), latched3)

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

        # 채팅 처리는 ★Reentrant 그룹 — MutuallyExclusive면 LLM 턴이
        # 도는 동안 "멈춰"의 fast-path 콜백까지 큐에서 대기하게 되어
        # '정지는 무엇에도 의존 금지' 원칙이 깨진다 (스모크 테스트 [6]이
        # 실제로 잡아낸 결함). 동시 실행 제어는 _busy 락이 담당.
        chat_group = ReentrantCallbackGroup()
        self.create_subscription(
            String, '/agent/user_msg', self.on_user_msg, 10,
            callback_group=chat_group)

        self.get_logger().info(f'에이전트 준비 완료 (model={MODEL})')

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.odom = (p.x, p.y)
        if self.odom_origin is None:
            self.odom_origin = (p.x, p.y)

    def on_pmu(self, msg: PmuState):
        frac = (msg.battery_voltage - BATT_EMPTY_V) / (BATT_FULL_V - BATT_EMPTY_V)
        self.battery_pct = round(max(0.0, min(1.0, frac)) * 100)

    def notify_event(self, text: str):
        """툴의 비동기 완료 통지 → UI 이벤트로 전달."""
        self.pub_event.publish(String(data=f'[알림] {text}'))

    def on_frame(self, msg: CompressedImage):
        with self.frame_cond:
            self.frames.append(bytes(msg.data))
            self.frame_count += 1
            self.frame_cond.notify_all()

    def yaw_rad(self) -> float:
        w, x, y, z = self.imu_quat
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    # ---------- ToolExecutor가 쓰는 인터페이스 ----------
    def latest_frame(self, fresh: bool = True, timeout: float = 1.5):
        """최신 프레임 반환. fresh=True면 '호출 이후 도착한' 프레임을
        기다린다 (2Hz 스트림 기준 최대 ~0.5s) — 이동/회전 직후의
        낡은 프레임으로 묘사하는 것을 방지 (묘사 품질 이슈 대응)."""
        if not fresh:
            return self.frames[-1] if self.frames else None
        with self.frame_cond:
            n = self.frame_count
            self.frame_cond.wait_for(
                lambda: self.frame_count > n, timeout=timeout)
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
            'battery_pct': self.battery_pct,
            'place': self.current_world,
            'holding': self.carrying or None,
        }
        if self.objects and self.odom:
            rx, ry = self.odom
            nearby = {}
            for name, p in self.objects.items():
                d = math.hypot(p['x'] - rx, p['y'] - ry)
                nearby[name] = round(d, 2)
            st['nearby_objects_dist_m'] = dict(
                sorted(nearby.items(), key=lambda kv: kv[1]))
        if self.landmarks and self.odom:
            rx, ry = self.odom
            lm = {}
            for name, p in self.landmarks.items():
                lm[name] = round(math.hypot(p['x'] - rx, p['y'] - ry), 2)
            # 가구/랜드마크 거리 — 접근 동작(쓰다듬기 등)의 근거 데이터
            st['nearby_landmarks_dist_m'] = dict(
                sorted(lm.items(), key=lambda kv: kv[1]))
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
    from x2_sim.single_instance import claim
    _lock = claim('robot_agent')       # noqa: F841 — 수명 유지용
    # SIGTERM 즉시 종료 — 기본 동작은 rclpy 컨텍스트만 반파된 채
    # 프로세스가 살아남아 'cannot use Destroyable' 반좀비가 됨 (실측).
    # 에이전트는 플러시할 영속 상태가 없어(감사 로그는 라인 단위 append)
    # 즉시 종료가 안전하다.
    import os as _os
    import signal as _signal
    _signal.signal(_signal.SIGTERM, lambda *_: _os._exit(0))
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
