#!/usr/bin/env python3
"""에이전트 툴 실행기 — 실기 프로토콜 준수 + 안전 가드 + 감사 로그.

설계 원칙 (Phase 3 리뷰 반영):
- 모드는 캐시하지 않는다: move/play_motion 실행 직전마다 GetMcAction으로
  조회하고, 필요 시 전환하고, 거부되면 그 사실을 툴 결과로 반환해
  에이전트가 대화로 풀게 한다 (실기에서 리모컨이 언제든 모드를 뺏는
  상황과 동일한 코드 경로).
- move()는 재발행 루프: 명령 유효기간이 0.5초이므로 duration 동안
  10Hz로 velocity를 재발행한다 (실기 동일 패턴).
- 모든 툴 호출은 jsonl 감사 로그에 남긴다 (타임스탬프+인자+결과).
"""
import base64
import json
import os
import threading
import time
from datetime import datetime, timezone

from aimdk_msgs.msg import McAction, McInputAction, McLocomotionVelocity
from std_msgs.msg import String
from aimdk_msgs.srv import (
    GetMcAction,
    PlayEmoji,
    PlayTts,
    SetMcAction,
    SetMcInputSource,
    SetMcPresetMotion,
)

from x2_sim.mc_services import ACTION_NAMES, KNOWN_MOTIONS

# ---------- 안전 상한 (하드코딩 — 프롬프트가 아니라 코드가 지킨다) ----------
MAX_VX = 0.3      # m/s  (실기 일상 권장 0.8보다 보수적)
MAX_VY = 0.2      # m/s
MAX_WZ = 0.5      # rad/s
MAX_DURATION = 3.0  # s  (한 번의 move 상한)
SOURCE = 'agent'
SOURCE_PRIORITY = 50
MIN_BATTERY_PCT = 15   # 이 미만이면 이동/모션 거절 (충전 안내)

AUDIT_PATH = os.environ.get('AGENT_AUDIT_LOG', '/ws/log/tool_audit.jsonl')

# sim/real 전환 (로드맵 원칙: 설정만으로 전환)
#   sim  — speak/emoji를 UI 토픽으로 (시뮬에는 스피커/표정화면 없음)
#   real — 실기 PlayTts/PlayEmoji 서비스 호출
ROBOT_TARGET = os.environ.get('ROBOT_TARGET', 'sim')

# 표정 문자열 → 실기 PlayEmoji emotion_id (PlayEmoji.srv 상수에서 확인)
EMOJI_ID = {
    'blink': 1, 'calm': 10, 'cute': 30, 'sleepy': 80, 'happy': 90,
    'very_happy': 100, 'sad': 110, 'confused': 130, 'surprised': 140,
    'thinking': 170, 'angry': 180, 'adore': 200, 'charging': 220,
}

STATE_SUCCESS = 1


class ToolExecutor:
    """rclpy 노드 위에서 동작하는 툴 구현체. 전부 블로킹(단순 우선)."""

    def __init__(self, node):
        self.node = node
        self.log = node.get_logger()
        os.makedirs(os.path.dirname(AUDIT_PATH), exist_ok=True)

        self.cli_get_action = node.create_client(
            GetMcAction, '/aimdk_5Fmsgs/srv/GetMcAction')
        self.cli_set_action = node.create_client(
            SetMcAction, '/aimdk_5Fmsgs/srv/SetMcAction')
        self.cli_set_source = node.create_client(
            SetMcInputSource, '/aimdk_5Fmsgs/srv/SetMcInputSource')
        self.cli_preset = node.create_client(
            SetMcPresetMotion, '/aimdk_5Fmsgs/srv/SetMcPresetMotion')
        # 실기 전용 서비스 (sim에서는 미사용 — Phase 6에서 검증)
        self.cli_tts = node.create_client(
            PlayTts, '/aimdk_5Fmsgs/srv/PlayTts')
        self.cli_emoji = node.create_client(
            PlayEmoji, '/aimdk_5Fmsgs/srv/PlayEmoji')
        self._move_thread = None
        self.pub_vel = node.create_publisher(
            McLocomotionVelocity, '/aima/mc/locomotion/velocity', 10)
        self._stop_requested = False
        # 하드와이어 정지 수신 (UI 버튼 등 외부 독립 경로) — 기본 콜백
        # 그룹이라 채팅 처리 중에도 즉시 실행됨
        node.create_subscription(
            String, '/agent/hard_stop', self._on_hard_stop, 10)
        # 장소 이동(시뮬: 월드 전환) 발행자
        self.pub_set_world = node.create_publisher(
            String, '/x2_sim/set_world', 10)

    def _on_hard_stop(self, msg: String):
        self._stop_requested = True
        vel = McLocomotionVelocity()
        vel.source = SOURCE
        self.pub_vel.publish(vel)   # 속도 0
        self.log.warning(f'하드 정지 수신 (from={msg.data!r}) — 이동 루프 중단')
        self.audit('hard_stop', {'from': msg.data}, True, 'interrupted')

    # ---------- 공통 유틸 ----------
    def audit(self, tool, args, ok, result):
        rec = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'tool': tool, 'args': args, 'ok': ok,
            'result': str(result)[:500],
        }
        try:
            with open(AUDIT_PATH, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except OSError as e:
            self.log.warning(f'감사 로그 기록 실패: {e}')

    def _call(self, client, req, timeout=3.0):
        """서비스 동기 호출 (MultiThreadedExecutor 전제)."""
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'서비스 없음: {client.srv_name}')
        fut = client.call_async(req)
        deadline = time.monotonic() + timeout
        while not fut.done():
            if time.monotonic() > deadline:
                raise RuntimeError(f'서비스 응답 시간 초과: {client.srv_name}')
            time.sleep(0.02)
        return fut.result()

    # ---------- MC 프로토콜 (등록 → 조회 → 전환) ----------
    def ensure_source(self):
        req = SetMcInputSource.Request()
        req.action.value = McInputAction.INPUTACTION_ADD
        req.input_source.name = SOURCE
        req.input_source.priority = SOURCE_PRIORITY
        req.input_source.timeout = 1000
        res = self._call(self.cli_set_source, req)
        if res.response.state.value != STATE_SUCCESS:
            raise RuntimeError('입력 소스 등록 거부됨')

    def current_action(self) -> int:
        res = self._call(self.cli_get_action, GetMcAction.Request())
        return int(res.info.current_action.value)

    def set_action(self, target: int):
        req = SetMcAction.Request()
        req.source = SOURCE
        req.command.action.value = target
        res = self._call(self.cli_set_action, req)
        if res.response.status.value != STATE_SUCCESS:
            raise RuntimeError(
                f'모드 전환 거부: {res.response.message} '
                f'(현재 제어권이 다른 소스에 있거나 전이 불가 상태)')

    def ensure_mode(self, target: int):
        """행동 직전 모드 확인 → 필요 시 전환. 캐시 없음."""
        cur = self.current_action()
        if cur == target:
            return
        # LOCOMOTION은 STAND 계열에서만 진입 → 중간 단계 자동 경유
        if target == McAction.LOCOMOTION_DEFAULT and cur not in (
                McAction.STAND_DEFAULT, McAction.STAND_BODY_CONTROL):
            self.set_action(int(McAction.STAND_DEFAULT))
        if target in (McAction.STAND_DEFAULT,) and cur == target:
            return
        self.set_action(target)

    # ---------- 툴 구현 ----------
    def look(self):
        frame = self.node.latest_frame()
        if frame is None:
            self.audit('look', {}, False, 'no frame')
            return {'error': '카메라 프레임이 아직 없습니다.'}
        self.audit('look', {}, True, f'{len(frame)} bytes jpeg')
        return {'image_b64': base64.standard_b64encode(frame).decode()}

    def status_snapshot(self):
        """감사 로그 없이 상태 요약 (매 턴 자동 주입용)."""
        st = self.node.status_summary()
        try:
            st['mode'] = ACTION_NAMES.get(self.current_action(), '?')
        except RuntimeError as e:
            st['mode'] = f'조회 실패: {e}'
        return st

    def get_status(self):
        st = self.status_snapshot()
        self.audit('get_status', {}, True, st)
        return st

    def _battery_guard(self):
        pct = self.node.battery_pct
        if pct is not None and pct < MIN_BATTERY_PCT:
            return (f'배터리 {pct}% — {MIN_BATTERY_PCT}% 미만이라 '
                    f'움직임을 거절함. 충전이 필요함 (안전 가드)')
        return None

    def move(self, vx=0.0, vy=0.0, wz=0.0, duration=1.0):
        args = {'vx': vx, 'vy': vy, 'wz': wz, 'duration': duration}
        # 하드 클램프 (LLM이 뭐라 하든 코드가 지킨다)
        vx = max(-MAX_VX, min(MAX_VX, float(vx)))
        vy = max(-MAX_VY, min(MAX_VY, float(vy)))
        wz = max(-MAX_WZ, min(MAX_WZ, float(wz)))
        duration = max(0.1, min(MAX_DURATION, float(duration)))
        guard = self._battery_guard()
        if guard:
            self.audit('move', args, False, guard)
            return {'error': guard}
        try:
            self.ensure_source()
            self.ensure_mode(int(McAction.LOCOMOTION_DEFAULT))
        except RuntimeError as e:
            self.audit('move', args, False, str(e))
            return {'error': str(e)}

        # 이전 이동이 진행 중이면 중단시키고 교체
        if self._move_thread and self._move_thread.is_alive():
            self._stop_requested = True
            self._move_thread.join(timeout=0.5)
        self._stop_requested = False

        # ★비동기 실행 (리뷰 B): 즉시 반환하고 완료는 이벤트로 알림.
        #   재발행 루프(10Hz)는 백그라운드 스레드에서 계속된다.
        def run():
            start_pos = self.node.odom
            msg = McLocomotionVelocity()
            msg.source = SOURCE
            msg.forward_velocity = vx
            msg.lateral_velocity = vy
            msg.angular_velocity = wz
            t_end = time.monotonic() + duration
            while time.monotonic() < t_end and not self._stop_requested:
                self.pub_vel.publish(msg)
                time.sleep(0.1)
            msg.forward_velocity = 0.0
            msg.lateral_velocity = 0.0
            msg.angular_velocity = 0.0
            self.pub_vel.publish(msg)
            stopped = (' (stop으로 중단됨)' if self._stop_requested else '')
            # 실제 이동 거리 — 명령과 크게 다르면 차단/제어권 상실 신호
            actual = ''
            if start_pos and self.node.odom:
                dist = ((self.node.odom[0] - start_pos[0]) ** 2
                        + (self.node.odom[1] - start_pos[1]) ** 2) ** 0.5
                expect = (vx * vx + vy * vy) ** 0.5 * duration
                actual = f', 실제 이동 {dist:.2f}m'
                if expect > 0.05 and dist < expect * 0.3:
                    actual += (' ← 명령보다 훨씬 적음! 장애물 차단 또는 '
                               '제어권 상실(리모컨 개입) 가능성')
            done = f'이동 종료: vx={vx} vy={vy} wz={wz} {duration}s{stopped}{actual}'
            self.audit('move_done', args, True, done)
            self.node.notify_event(done)

        self._move_thread = threading.Thread(target=run, daemon=True)
        self._move_thread.start()
        result = (f'이동 시작함: vx={vx} vy={vy} wz={wz}, {duration}초간. '
                  f'완료는 이벤트로 통지됨 — 사용자에게는 출발했다고만 알리면 됨')
        self.audit('move', args, True, result)
        return {'ok': result}

    def play_motion(self, motion: str):
        args = {'motion': motion}
        name_to_id = {v: k for k, v in KNOWN_MOTIONS.items()}
        mid = name_to_id.get(str(motion).upper())
        if mid is None:
            msg = (f'미등록 모션 {motion!r}. 사용 가능: '
                   f'{sorted(name_to_id)}')
            self.audit('play_motion', args, False, msg)
            return {'error': msg}
        guard = self._battery_guard()
        if guard:
            self.audit('play_motion', args, False, guard)
            return {'error': guard}
        try:
            self.ensure_source()
            self.ensure_mode(int(McAction.STAND_DEFAULT))
        except RuntimeError as e:
            self.audit('play_motion', args, False, str(e))
            return {'error': str(e)}
        req = SetMcPresetMotion.Request()
        req.motion.value = mid
        res = self._call(self.cli_preset, req)
        if res.response.state.value != STATE_SUCCESS:
            msg = f'모션 거부됨 (state={res.response.state.value})'
            self.audit('play_motion', args, False, msg)
            return {'error': msg}
        time.sleep(2.5)  # 동작 완료까지 대기 (블로킹 단순화)
        self.audit('play_motion', args, True, 'done')
        return {'ok': f'{motion} 동작 완료'}

    def speak(self, text: str):
        if ROBOT_TARGET == 'real':
            # 실기: PlayTts 서비스 (Phase 6에서 실물 검증 예정)
            req = PlayTts.Request()
            req.tts_req.text = text
            try:
                res = self._call(self.cli_tts, req, timeout=5.0)
                ok = res.header.status.value == STATE_SUCCESS
                self.audit('speak', {'text': text}, ok, 'real_tts')
                return ({'ok': f'말함: {text}'} if ok
                        else {'error': 'TTS 서비스가 거부함'})
            except RuntimeError as e:
                self.audit('speak', {'text': text}, False, str(e))
                return {'error': f'TTS 실패: {e}'}
        # 시뮬: 채팅 UI에 음성 말풍선으로 표시
        self.node.publish_say(text)
        self.audit('speak', {'text': text}, True, 'sim_ui')
        return {'ok': f'말함: {text}'}

    def show_emoji(self, emoji: str):
        emoji = str(emoji).lower()
        if ROBOT_TARGET == 'real':
            eid = EMOJI_ID.get(emoji)
            if eid is None:
                return {'error': f'미지원 표정 {emoji!r}. '
                                 f'지원: {sorted(EMOJI_ID)}'}
            req = PlayEmoji.Request()
            req.emotion_id = eid
            try:
                self._call(self.cli_emoji, req, timeout=5.0)
                self.audit('show_emoji', {'emoji': emoji}, True, 'real')
                return {'ok': f'표정 표시: {emoji}'}
            except RuntimeError as e:
                self.audit('show_emoji', {'emoji': emoji}, False, str(e))
                return {'error': f'표정 실패: {e}'}
        self.node.publish_emoji(emoji)
        self.audit('show_emoji', {'emoji': emoji}, True, 'sim_ui')
        return {'ok': f'표정 표시: {emoji}'}

    # 장소 이름 → 시뮬 월드 매핑 (실기에서는 SLAM 내비게이션 목적지가 됨)
    PLACES = {
        '스튜디오': 'studio', 'studio': 'studio',
        '거실': 'living_room', 'living_room': 'living_room',
        '사무실': 'office', 'office': 'office',
        '주방': 'kitchen', '부엌': 'kitchen', 'kitchen': 'kitchen',
    }

    def go_to_place(self, place: str):
        args = {'place': place}
        key = str(place).strip().lower()
        world = self.PLACES.get(key) or self.PLACES.get(str(place).strip())
        if world is None:
            msg = f'모르는 장소 {place!r}. 갈 수 있는 곳: 스튜디오/거실/사무실/주방'
            self.audit('go_to_place', args, False, msg)
            return {'error': msg}
        if ROBOT_TARGET == 'real':
            # 실기: SLAM 내비게이션(NaviPoint)으로 구현 예정 — Phase 6+
            msg = '실기 내비게이션은 아직 연결 전 (Phase 6+에서 SLAM 경로로 구현)'
            self.audit('go_to_place', args, False, msg)
            return {'error': msg}
        if self.node.current_world == world:
            self.audit('go_to_place', args, True, 'already there')
            return {'ok': f'이미 {place}에 있음'}
        # 발행 → latched 월드 토픽으로 도착 확인 (유실 대비 재시도)
        arrived = False
        for _ in range(3):
            self.pub_set_world.publish(String(data=world))
            deadline = time.monotonic() + 2.5
            while time.monotonic() < deadline:
                if self.node.current_world == world:
                    arrived = True
                    break
                time.sleep(0.1)
            if arrived:
                break
        if not arrived:
            msg = '월드 전환 응답 없음 (시뮬 노드 상태 확인 필요)'
            self.audit('go_to_place', args, False, msg)
            return {'error': msg}
        self.node.latest_frame(fresh=True, timeout=2.0)  # 새 씬 프레임 대기
        self.audit('go_to_place', args, True, f'world={world}')
        return {'ok': (f'{place}에 도착함 (시뮬: 씬 전환 방식이라 위치가 '
                       f'초기화됨). look으로 새 장소를 확인해 알려줄 것')}

    def stop(self):
        self._stop_requested = True
        msg = McLocomotionVelocity()
        msg.source = SOURCE
        self.pub_vel.publish(msg)  # 속도 0
        self.audit('stop', {}, True, 'stopped')
        return {'ok': '정지 명령 발행 (속도 0, 모드 유지)'}
