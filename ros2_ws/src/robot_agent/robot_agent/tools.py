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
# sim/real 판정의 단일 소스는 기능 레지스트리 (A-0)
from robot_agent.features import ROBOT_TARGET

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
        # look() 온디맨드 고해상 렌더 요청 (시뮬 전용 — 실기는 상시 스트림)
        self.pub_render_hq = node.create_publisher(
            String, '/x2_sim/render_hq', 1)
        # 집기/놓기 (시뮬 전용 — 실기는 핸드 하드웨어 + HandCommand)
        self.pub_grasp = node.create_publisher(String, '/x2_sim/grasp', 10)
        self.pub_release = node.create_publisher(
            String, '/x2_sim/release', 10)
        self._grasp_results = []
        node.create_subscription(
            String, '/x2_sim/grasp_result',
            lambda m: self._grasp_results.append(m.data), 10)

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

    def _wait_mode(self, target: int, timeout: float = 5.0):
        """A-2②: 모드 전환 '완료'까지 폴링 대기.

        시뮬 McMock은 전환이 즉시라 첫 폴에서 통과하지만, 실기의
        STAND→LOCOMOTION은 자세를 잡는 시간이 걸린다 — 완료 전에
        velocity를 쏘면 무시되거나 위험. GetMcAction 폴링으로 실제
        도달을 확인한다 (sim/real 동일 경로)."""
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if self.current_action() == target:
                return
            time.sleep(0.2)
        raise RuntimeError(
            f'모드 전환 미완료: 목표 {target}, 현재 {self.current_action()} '
            f'({timeout}s 초과)')

    def ensure_mode(self, target: int):
        """행동 직전 모드 확인 → 필요 시 전환 + 완료 대기. 캐시 없음."""
        cur = self.current_action()
        if cur == target:
            return
        # LOCOMOTION은 STAND 계열에서만 진입 → 중간 단계 자동 경유
        if target == McAction.LOCOMOTION_DEFAULT and cur not in (
                McAction.STAND_DEFAULT, McAction.STAND_BODY_CONTROL):
            self.set_action(int(McAction.STAND_DEFAULT))
            self._wait_mode(int(McAction.STAND_DEFAULT))
        if target in (McAction.STAND_DEFAULT,) and cur == target:
            return
        self.set_action(target)
        self._wait_mode(target)

    # ---------- 툴 구현 ----------
    def look(self):
        if ROBOT_TARGET == 'sim':
            # 고해상 렌더는 온디맨드 — 요청 후 신선 프레임 대기
            self.pub_render_hq.publish(String(data='look'))
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

    def move(self, vx=0.0, vy=0.0, wz=0.0, duration=1.0, wait=False):
        args = {'vx': vx, 'vy': vy, 'wz': wz, 'duration': duration,
                'wait': wait}
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
        if wait:
            # 복합 작업(접근 후 집기 등): 이동 완료까지 동기 대기
            self._move_thread.join(timeout=duration + 1.0)
            st = self.node.status_summary()
            result = (f'이동 완료(wait). 현재 위치 참고: '
                      f'{st.get("nearby_objects_dist_m", {})}')
            self.audit('move', args, True, result)
            return {'ok': result}
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
        # A-2③: 고정 sleep 대신 상태 기반 완료 판정 — 실기는 모션별
        # 소요가 다르다. 모드가 프리셋 실행으로 바뀌었다 STAND로
        # 돌아오면 완료. 모드 변화가 안 보이는 구현(시뮬 목)은
        # 애니메이션 길이(2.5s) 폴백.
        t0 = time.monotonic()
        saw_active = False
        while time.monotonic() - t0 < 12.0:
            cur = self.current_action()
            if cur != int(McAction.STAND_DEFAULT):
                saw_active = True
            elif saw_active:
                break                       # 실행 후 STAND 복귀 = 완료
            elif time.monotonic() - t0 >= 2.5:
                break                       # 모드 무변화 구현 폴백
            time.sleep(0.3)
        self.audit('play_motion', args, True,
                   f'done ({time.monotonic() - t0:.1f}s)')
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

    # 장소 이름 별칭 (home 월드 안의 방 / 별도 월드 공통)
    PLACES = {
        '스튜디오': 'studio', 'studio': 'studio',
        '거실': 'living_room', 'living_room': 'living_room',
        '사무실': 'office', 'office': 'office',
        '주방': 'kitchen', '부엌': 'kitchen', 'kitchen': 'kitchen',
        '현관': 'entrance', 'entrance': 'entrance',
        '소파': 'sofa', 'sofa': 'sofa',
        '냉장고': 'fridge', 'fridge': 'fridge',
        '집': 'home', 'home': 'home',
        '나무': 'plant', '화분': 'plant', 'plant': 'plant',
        '티비': 'tv', '텔레비전': 'tv', 'tv': 'tv',
        '티테이블': 'tea_table', '테이블': 'tea_table',
        'tea_table': 'tea_table',
        '아일랜드': 'island', 'island': 'island',
        '조리대': 'counter', '카운터': 'counter', 'counter': 'counter',
        '책상': 'office', 'desk': 'office',
    }

    def _rotate_to(self, target_yaw, timeout=10.0):
        """제자리 회전으로 목표 방향 정렬 (소파 등지기 등)."""
        import math as m
        msg = McLocomotionVelocity()
        msg.source = SOURCE
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end and not self._stop_requested:
            err = (target_yaw - self.node.yaw_rad() + m.pi) % (2 * m.pi) - m.pi
            if abs(err) < 0.12:
                break
            msg.angular_velocity = max(-0.5, min(0.5, 1.2 * err))
            self.pub_vel.publish(msg)
            time.sleep(0.1)
        msg.angular_velocity = 0.0
        self.pub_vel.publish(msg)

    def _dock(self, max_dist=0.6, forward=False, timeout=6.0):
        """천천히 전/후진해 가구(소파·테이블 등)에 밀착. 충돌 차단으로
        위치가 더 이상 변하지 않으면(=닿음) 정지."""
        import math as m
        start = self.node.odom
        if start is None:
            return
        msg = McLocomotionVelocity()
        msg.source = SOURCE
        last = start
        stall = 0
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end and not self._stop_requested:
            pos = self.node.odom
            if m.hypot(pos[0] - start[0], pos[1] - start[1]) >= max_dist:
                break
            if m.hypot(pos[0] - last[0], pos[1] - last[1]) < 0.005:
                stall += 1
                if stall >= 5:      # 0.5초간 정지 = 밀착 완료
                    break
            else:
                stall = 0
            last = pos
            msg.forward_velocity = 0.12 if forward else -0.12
            self.pub_vel.publish(msg)
            time.sleep(0.1)
        msg.forward_velocity = 0.0
        self.pub_vel.publish(msg)

    # ---------- 보행 내비게이션 (home 월드: 진짜 걸어서 이동) ----------
    def _navigate(self, waypoints, timeout=90.0):
        """웨이포인트를 향해 회전→직진 반복 (velocity 명령만 사용 —
        실기에서 SLAM 목적지 없이 걷는 방식과 동일). 걷는 과정 전체가
        카메라에 보인다. stop/하드정지로 언제든 중단 가능."""
        import math as m
        self.ensure_source()
        self.ensure_mode(int(McAction.LOCOMOTION_DEFAULT))
        self._stop_requested = False
        msg = McLocomotionVelocity()
        msg.source = SOURCE
        t_end = time.monotonic() + timeout
        best = None
        best_t = time.monotonic()
        n_wp = len(waypoints)
        for i, (wx, wy) in enumerate(waypoints):
            # 중간 웨이포인트는 느슨하게 통과(0.5m), 최종 목적지만 정밀(0.35m)
            radius = 0.35 if i == n_wp - 1 else 0.5
            retried = False
            best_err = None
            while True:
                if self._stop_requested:
                    return False, 'stop 요청으로 중단'
                if time.monotonic() > t_end:
                    return False, '내비게이션 시간 초과'
                pos = self.node.odom
                if pos is None:
                    return False, '위치 정보 없음'
                dx, dy = wx - pos[0], wy - pos[1]
                dist = m.hypot(dx, dy)
                if dist < radius:
                    break
                # 진행 감시: 5초간 거리·방향 모두 개선 없으면 막힌 것
                # (회전 중엔 거리가 안 줄므로 방향 오차 감소도 진행으로 인정)
                _b = m.atan2(dy, dx)
                _e = abs((_b - self.node.yaw_rad() + m.pi) % (2 * m.pi) - m.pi)
                if best is None or dist < best - 0.05:
                    best, best_t = dist, time.monotonic()
                elif best_err is None or _e < best_err - 0.05:
                    best_err, best_t = _e, time.monotonic()
                elif time.monotonic() - best_t > 5.0:
                    if not retried:
                        # 언스턱: 1.5초 후진 후 같은 웨이포인트 1회 재시도
                        retried = True
                        msg.forward_velocity = -0.2
                        msg.angular_velocity = 0.0
                        t_back = time.monotonic() + 1.5
                        while (time.monotonic() < t_back
                               and not self._stop_requested):
                            self.pub_vel.publish(msg)
                            time.sleep(0.1)
                        msg.forward_velocity = 0.0
                        self.pub_vel.publish(msg)
                        best, best_t = None, time.monotonic()
                        best_err = None
                        continue
                    msg.forward_velocity = 0.0
                    msg.angular_velocity = 0.0
                    self.pub_vel.publish(msg)
                    return False, (f'진행 불가 (장애물 차단 또는 제어권 '
                                   f'상실) — 남은 거리 {dist:.2f}m')
                bearing = m.atan2(dy, dx)
                err = (bearing - self.node.yaw_rad() + m.pi) % (2 * m.pi) - m.pi
                # 블렌드 제어: 방향 오차에 비례해 감속하며 회전 —
                # '정지 후 회전 ↔ 전진' 이분법(0.35rad 문턱)이 경계에서
                # 덜컹임을 만들던 것을 연속 제어로 교체
                speed = MAX_VX * max(0.0, 1.0 - abs(err) / 0.7)
                # 목적지 근접 시 감속 (마지막 0.5m 구간)
                speed = min(speed, MAX_VX * max(0.4, dist / 0.5))
                msg.forward_velocity = speed
                msg.angular_velocity = max(-0.5, min(0.5, 1.2 * err))
                self.pub_vel.publish(msg)
                time.sleep(0.1)
            best, best_t = None, time.monotonic()
        msg.forward_velocity = 0.0
        msg.angular_velocity = 0.0
        self.pub_vel.publish(msg)
        return True, '도착'

    def go_to_place(self, place: str):
        args = {'place': place}
        key = str(place).strip().lower()
        world = self.PLACES.get(key) or self.PLACES.get(str(place).strip())
        if world is None:
            msg = (f'모르는 장소 {place!r}. 갈 수 있는 곳: '
                   f'거실/주방/현관(집 안 — 걸어서), 스튜디오/사무실(별도 공간)')
            self.audit('go_to_place', args, False, msg)
            return {'error': msg}
        if ROBOT_TARGET == 'real':
            # 실기: SLAM 내비게이션(NaviPoint)으로 구현 예정 — Phase 6+
            msg = '실기 내비게이션은 아직 연결 전 (Phase 6+에서 SLAM 경로로 구현)'
            self.audit('go_to_place', args, False, msg)
            return {'error': msg}

        # ★home 월드 안의 방이면 순간이동이 아니라 실제로 걸어간다.
        #   다른 공간(스튜디오 등)에 있어도 먼저 home으로 복귀 후 걷는다
        #   — 모든 장소가 항상 연결되어 있게.
        from x2_sim.worlds import WORLDS
        home = WORLDS.get('home', {})
        places = home.get('places', {})
        if world in places and self.node.current_world != 'home':
            held = self.node.carrying
            if not self._switch_world('home'):
                msg = '집(home)으로 복귀 실패 (시뮬 노드 상태 확인 필요)'
                self.audit('go_to_place', args, False, msg)
                return {'error': msg}
            if held:
                self.node.notify_event(
                    f'공간 이동으로 들고 있던 {held}은(는) 원래 자리로 '
                    f'돌아감')
        if self.node.current_world == 'home' and world in places:
            entry = places[world]
            tx, ty = entry['pos']
            pos = self.node.odom or (0.0, 0.0)
            dist0 = ((tx - pos[0]) ** 2 + (ty - pos[1]) ** 2) ** 0.5
            if dist0 < 0.5:
                # 이미 근처여도 방향 정렬·가구 밀착(도킹)은 수행 —
                # 놓기/앉기 등 후속 동작의 위치 정밀도를 보장
                face = entry.get('yaw')
                if face is not None:
                    self._rotate_to(face)
                if entry.get('dock_back'):
                    self._dock(max_dist=float(entry['dock_back']))
                elif entry.get('dock_forward'):
                    self._dock(max_dist=float(entry['dock_forward']),
                               forward=True)
                self.audit('go_to_place', args, True, 'already there')
                return {'ok': f'이미 {place} 근처에 있음 (정렬 완료)'}
            # ★방 그래프 라우팅: 현재 방 → (BFS 최단 방 경로) → 목적 방.
            #   각 전이는 [이 방 허브] → 문 → [다음 방 허브] 순 — 허브는
            #   가구 없는 지점이라 티테이블/아일랜드 등을 자연 우회한다.
            rooms = home.get('rooms', {})
            doors = home.get('doors', {})

            def room_of(x, y):
                if x >= 2.0:
                    return 'living' if y > 0 else 'kitchen'
                if x >= -1.0:
                    return 'hall'
                return 'studio' if y > 0 else 'office'

            def door_between(a, b):
                return doors.get((a, b)) or doors.get((b, a))

            s, t = room_of(*pos), room_of(tx, ty)
            # BFS 최단 방 경로
            path = [s]
            if s != t:
                adj = {}
                for (a, b) in doors:
                    adj.setdefault(a, []).append(b)
                    adj.setdefault(b, []).append(a)
                seen, queue = {s: None}, [s]
                while queue:
                    cur = queue.pop(0)
                    if cur == t:
                        break
                    for nxt in adj.get(cur, []):
                        if nxt not in seen:
                            seen[nxt] = cur
                            queue.append(nxt)
                if t not in seen:
                    msg = f'{place}까지 가는 경로를 찾을 수 없음'
                    self.audit('go_to_place', args, False, msg)
                    return {'error': msg}
                path = []
                cur = t
                while cur is not None:
                    path.append(cur)
                    cur = seen[cur]
                path.reverse()
            waypoints = []
            for i, room in enumerate(path):
                if i > 0:
                    waypoints.append(tuple(door_between(path[i - 1], room)))
                if room in rooms and (i > 0 or len(path) > 1):
                    waypoints.append(tuple(rooms[room]['hub']))
            waypoints += [tuple(v) for v in entry.get('via', [])]
            waypoints.append((tx, ty))
            # 현재 위치와 겹치거나 연속으로 겹치는 웨이포인트 정리
            pruned = []
            px, py = pos
            for wx, wy in waypoints:
                if ((wx - px) ** 2 + (wy - py) ** 2) ** 0.5 < 0.45 \
                        and (wx, wy) != (tx, ty):
                    continue
                if pruned and ((wx - pruned[-1][0]) ** 2
                               + (wy - pruned[-1][1]) ** 2) ** 0.5 < 0.3:
                    continue
                pruned.append((wx, wy))
            t0 = time.monotonic()
            ok, detail = self._navigate(pruned)
            took = time.monotonic() - t0
            if ok:
                face = entry.get('yaw')
                if face is not None:
                    self._rotate_to(face)   # 예: 소파를 등지고 정렬
                # 가구에 닿을 때까지 전/후진 밀착 — 시뮬 충돌 차단이
                # 자연스러운 정지 역할을 한다 (진행 정지 감지로 종료)
                if entry.get('dock_back'):
                    self._dock(max_dist=float(entry['dock_back']))
                elif entry.get('dock_forward'):
                    self._dock(max_dist=float(entry['dock_forward']),
                               forward=True)
                held = f", 들고 있는 것: {self.node.carrying}" \
                    if self.node.carrying else ''
                result = (f'{place}까지 걸어서 도착 ({dist0:.1f}m, '
                          f'{took:.0f}초 소요{held}). look으로 주변 확인 가능')
                self.audit('go_to_place', args, True, result)
                return {'ok': result}
            self.audit('go_to_place', args, False, detail)
            return {'error': f'{place}로 가는 중 문제: {detail}'}

        if self.node.current_world == world:
            self.audit('go_to_place', args, True, 'already there')
            return {'ok': f'이미 {place}에 있음'}
        if world not in ('studio', 'office', 'home'):
            return {'error': f'{place}는 갈 수 없는 곳 — 집 안 장소'
                             f'(거실/주방/현관/소파/냉장고)는 걸어서, '
                             f'별도 공간은 스튜디오/사무실만 가능'}
        dropped = ''
        if self.node.carrying:
            dropped = ' ★공간 전환으로 들고 있던 물건은 원래 자리로 돌아감'
        if not self._switch_world(world):
            msg = '월드 전환 응답 없음 (시뮬 노드 상태 확인 필요)'
            self.audit('go_to_place', args, False, msg)
            return {'error': msg}
        self.node.latest_frame(fresh=True, timeout=2.0)  # 새 씬 프레임 대기
        self.audit('go_to_place', args, True, f'world={world}')
        return {'ok': (f'{place}(별도 공간)으로 전환됨. 위치 초기화됨.'
                       f'{dropped} look으로 확인해 알려줄 것')}

    def _switch_world(self, world: str) -> bool:
        """월드 전환 발행 → latched 토픽으로 도착 확인.

        전환 처리(비캐시 월드 컴파일 ~2s + 렌더 락 대기)를 감안해
        시도당 6초 대기 (2.5s 대기 시절 실측 사고: 시뮬이 8~16초 뒤에
        처리해 성공한 전환을 실패로 보고 — 렌더 락 굶주림이 원인,
        카메라 1Hz 완화와 함께 이중 대응)."""
        for _ in range(2):
            self.pub_set_world.publish(String(data=world))
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                if self.node.current_world == world:
                    return True
                time.sleep(0.1)
        return False

    def sit_down(self):
        """앉기 — 실기 McAction SIT_DOWN_DEFAULT(2000)와 동일 의미론."""
        try:
            self.ensure_source()
            self.ensure_mode(int(McAction.STAND_DEFAULT))
            self.set_action(int(McAction.SIT_DOWN_DEFAULT))
        except RuntimeError as e:
            self.audit('sit_down', {}, False, str(e))
            return {'error': str(e)}
        held = f' (들고 있는 것: {self.node.carrying})' \
            if self.node.carrying else ' (빈손)'
        self.audit('sit_down', {}, True, 'seated')
        return {'ok': f'앉았음.{held} 앉은 동안 이동 불가 — stand_up으로 일어남'}

    def stand_up(self):
        """일어서기 — 실기 McAction STAND_UP_DEFAULT(2005)."""
        try:
            self.ensure_source()
            self.set_action(int(McAction.STAND_UP_DEFAULT))
        except RuntimeError as e:
            self.audit('stand_up', {}, False, str(e))
            return {'error': str(e) + ' (앉아있지 않으면 실패함)'}
        self.audit('stand_up', {}, True, 'standing')
        return {'ok': '일어섰음 (STAND_DEFAULT)'}

    def pick_up(self, object_name: str):
        args = {'object_name': object_name}
        if ROBOT_TARGET == 'real':
            msg = ('실기 집기는 핸드 하드웨어(옵션) + HandCommand 연동 필요 '
                   '— Phase 6+에서 구현')
            self.audit('pick_up', args, False, msg)
            return {'error': msg}
        name = str(object_name).strip()
        if self.node.carrying:
            msg = f'이미 {self.node.carrying}을(를) 들고 있음 — 먼저 put_down'
            self.audit('pick_up', args, False, msg)
            return {'error': msg}
        if name not in self.node.objects:
            msg = (f'주변에 {name!r} 없음. 현재 집을 수 있는 물체: '
                   f'{list(self.node.objects)}')
            self.audit('pick_up', args, False, msg)
            return {'error': msg}
        n0 = len(self._grasp_results)
        self.pub_grasp.publish(String(data=name))
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if len(self._grasp_results) > n0:
                result = self._grasp_results[-1]
                break
            time.sleep(0.05)
        else:
            msg = '집기 응답 없음 (시뮬 노드 확인 필요)'
            self.audit('pick_up', args, False, msg)
            return {'error': msg}
        if result.startswith('ok:'):
            self.audit('pick_up', args, True, result)
            return {'ok': f'{name}을(를) 집었음. 들고 이동 가능'}
        if 'too_far' in result:
            dist = result.rsplit(':', 1)[-1]
            msg = (f'{name}이(가) 너무 멂 (거리 {dist}m, 도달 반경 0.95m) '
                   f'— 먼저 move로 가까이 접근 후 재시도')
        elif 'hands_full' in result:
            msg = '손이 차 있음 — put_down 먼저'
        else:
            msg = f'집기 실패: {result}'
        self.audit('pick_up', args, False, msg)
        return {'error': msg}

    def put_down(self):
        if ROBOT_TARGET == 'real':
            return {'error': '실기 집기/놓기는 Phase 6+ 구현 예정'}
        if not self.node.carrying:
            self.audit('put_down', {}, False, 'not carrying')
            return {'error': '들고 있는 물체가 없음'}
        held = self.node.carrying
        self.pub_release.publish(String(data=''))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and self.node.carrying:
            time.sleep(0.05)
        # 물리 안착 대기 후 실제 착지 결과를 데이터로 보고 —
        # "테이블에 내려놨어" 같은 추측성 보고 방지 (실측 사고:
        # 문 앞 바닥에 떨어뜨리고 테이블에 놨다고 답한 confabulation)
        time.sleep(1.0)
        obj = self.node.objects.get(held)
        landed = ''
        if obj:
            z = obj.get('z', 0.0)
            surface = '바닥' if z < 0.3 else f'높이 {z:.2f}m의 표면 위'
            landed = (f' → 실제 착지: {surface} '
                      f'(x={obj["x"]:.2f}, y={obj["y"]:.2f})')
        self.audit('put_down', {'was': held}, True, f'released{landed}')
        return {'ok': (f'{held}을(를) 손에서 놓음.{landed} — 바닥에 '
                       f'떨어졌다면 의도한 곳이 아닐 수 있으니 사용자에게 '
                       f'사실대로 말할 것. 테이블 위에 놓으려면 먼저 '
                       f'테이블 바로 앞(0.5m 이내)까지 접근해야 함')}

    def stop(self):
        self._stop_requested = True
        msg = McLocomotionVelocity()
        msg.source = SOURCE
        self.pub_vel.publish(msg)  # 속도 0
        self.audit('stop', {}, True, 'stopped')
        return {'ok': '정지 명령 발행 (속도 0, 모드 유지)'}
