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
import math
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
        # 자세 직접 전환 (눕기 — 시뮬 전용)
        self.pub_posture = node.create_publisher(
            String, '/x2_sim/posture', 10)
        # 문 조작
        self.pub_door = node.create_publisher(String, '/x2_sim/door', 10)
        self._lying = False
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
        sp = getattr(self.node, 'sim_posture', 'stand')
        if sp != 'stand':
            st['posture'] = sp + (
                ' (누움 — upright=false 정상, 이동 불가)'
                if sp == 'lie' else '')
        # 탐색 모드(미지 공간): 지도 사전 지식을 주지 않는다 —
        # 물체는 2m 근접에서만 감지되고, 장소는 학습한 것만 노출
        from x2_sim.worlds import WORLDS as _W
        if _W.get(self.node.current_world, {}).get('explore'):
            st['explore_mode'] = True
            if 'nearby_objects_dist_m' in st:
                st['nearby_objects_dist_m'] = {
                    k: v for k, v in st['nearby_objects_dist_m'].items()
                    if v <= 2.0}
            st.pop('nearby_landmarks_dist_m', None)
            if 'doors' in st:      # 문도 2.5m 근접에서만 감지
                st['doors'] = {k: v for k, v in st['doors'].items()
                               if v['dist_m'] <= 2.5}
            lm = self._learned(self.node.current_world)
            st['learned_places'] = sorted(lm['places'].keys())
            if lm.get('objects_seen'):
                st['objects_seen_during_explore'] = lm['objects_seen']
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

    def turn(self, angle_deg: float):
        """제자리 회전 (상대 각도, 폐루프) — "뒤돌아"류 요청 전용.

        move의 wz는 상한(0.5rad/s × 3s)에 가속 램프까지 겹쳐 한 번에
        ~80°가 한계라, 큰 회전을 시키면 30~50°에서 끝나는 문제가
        있었음(사용자 제보). IMU 요 기준 목표각까지 도는 폐루프."""
        args = {'angle_deg': angle_deg}
        guard = self._battery_guard()
        if guard:
            return {'error': guard}
        try:
            self.ensure_source()
            self.ensure_mode(int(McAction.LOCOMOTION_DEFAULT))
        except RuntimeError as e:
            self.audit('turn', args, False, str(e))
            return {'error': str(e)}
        a = max(-360.0, min(360.0, float(angle_deg)))
        start = self.node.yaw_rad()
        target = start + math.radians(a)
        self._stop_requested = False
        # 180° 초과 회전은 중간 경유각으로 나눠 최단경로 오해 방지
        steps = [target] if abs(a) <= 200 else [
            start + math.radians(a / 2.0), target]
        for t in steps:
            if self._stop_requested:
                break
            self._rotate_to(t, timeout=12.0)
        turned = math.degrees(
            (self.node.yaw_rad() - start + math.pi) % (2 * math.pi)
            - math.pi)
        self.audit('turn', args, True, f'turned {turned:.0f}deg')
        return {'ok': f'회전 완료: 요청 {a:.0f}°, 실제 {turned:.0f}° '
                      f'(look으로 새 방향 확인 가능)'}

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

    # ---------- 탐색·학습 (미지 공간: explore 월드) ----------
    # 처음 보는 공간에서 로봇이 지도를 스스로 만든다:
    #   · 프론티어 탐색 — 가본 셀(0.8m 격자)의 미방문 이웃으로 계속 전진,
    #     충돌 차단을 범퍼 삼아 막힌 셀은 제외
    #   · 지점마다 4방향 VLM 스캔 — 방/가구/물체 자가 인식
    #   · 인식한 가구에 접근해 그 지점을 장소로 자동 등록
    #   · '가본 셀'만으로 길 찾기 (BFS) — 배운 길만 안다
    # 결과는 /ws/log/learned/<world>.json 영속화. 사람 교육은
    # remember_place(보정용)로 최소화. 실기에서는 SLAM 탐색으로 대체.
    CELL = 0.8

    # ★개방 어휘 — 후보 목록 제한 없음. 어떤 공간(집/사무실/카페 등)이든
    # VLM이 보이는 대로 명명한다. 예시는 형식 안내일 뿐 제한이 아님.
    VLM_SCAN_PROMPT = (
        '실내를 다니는 로봇의 카메라 프레임이다. 단순 도형으로 만든 '
        '시뮬레이션이므로 색·형태·크기·배치로 추정하라.\n'
        '다음 JSON만 출력: {"room": "이 공간의 종류를 한 단어로 '
        '(모르면 \\"모름\\")", "furniture": ["보이는 큰 가구/설비 각각 '
        '한 단어씩"], "objects": ["보이는 소형 물체 각각 한 단어씩"]}\n'
        '규칙: 명사 한 단어로 통일(예: 침대, 회의탁자, 정수기, 세탁기, '
        '노트북 — 어디까지나 예시이며 어떤 명사든 가능). 벽/바닥/문은 '
        '제외. 확실한 것만. 없으면 빈 배열.')

    def _learned(self, world):
        if not hasattr(self, '_learned_cache'):
            self._learned_cache = {}
        if world not in self._learned_cache:
            try:
                with open(f'/ws/log/learned/{world}.json') as f:
                    d = json.load(f)
                d['cells'] = set(tuple(c) for c in d.get('cells', []))
            except Exception:
                d = {'places': {}, 'cells': set(),
                     'objects_seen': {}, 'rooms': {}}
            self._learned_cache[world] = d
        return self._learned_cache[world]

    def _save_learned(self, world):
        d = self._learned(world)
        os.makedirs('/ws/log/learned', exist_ok=True)
        out = dict(d)
        out['cells'] = sorted(list(c) for c in d['cells'])
        with open(f'/ws/log/learned/{world}.json', 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)

    def _cell(self, x, y):
        return (int(math.floor(x / self.CELL + 0.5)),
                int(math.floor(y / self.CELL + 0.5)))

    # 4방향만 — 대각 이동은 문틀/벽 모서리를 깎아 지나가려다 걸린다
    # (newhouse 문턱 스턱 실측)
    _NB8 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def _bfs_cells(self, cells, start, goal):
        from collections import deque
        if start == goal:
            return [start]
        seen = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            for dx, dy in self._NB8:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in seen or nxt not in cells:
                    continue
                seen[nxt] = cur
                if nxt == goal:
                    path = [nxt]
                    while seen[path[-1]] is not None:
                        path.append(seen[path[-1]])
                    return list(reversed(path))
                q.append(nxt)
        return None

    def _nearest_frontier(self, cells, blocked, start):
        """가본 셀에서 BFS로 가장 가까운 '미방문 이웃 셀'을 찾는다."""
        from collections import deque
        seen = {start}
        q = deque([start])
        while q:
            cur = q.popleft()
            for dx, dy in self._NB8:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in blocked:
                    continue
                if nxt in cells:
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
                else:
                    return nxt
        return None

    def _vlm_look(self):
        """현재 시야 1프레임을 VLM(Haiku 비전)으로 분류. 실패 시 None.

        시뮬에서는 프리뷰(320x240) 프레임 사용 — 고해상 온디맨드 렌더는
        라이브 영상을 스캔마다 0.5s씩 얼리는 부작용(사용자 제보)."""
        frame = None
        if ROBOT_TARGET == 'sim':
            time.sleep(0.15)            # 회전 직후 최신 프리뷰 확보
            frame = getattr(self.node, 'latest_preview', None)
        if frame is None:
            if ROBOT_TARGET == 'sim':
                self.pub_render_hq.publish(String(data='explore'))
            frame = self.node.latest_frame(fresh=True, timeout=2.5)
        if frame is None:
            return None
        try:
            import anthropic
            if not hasattr(self, '_vlm'):
                self._vlm = anthropic.Anthropic()
            r = self._vlm.messages.create(
                model='claude-haiku-4-5', max_tokens=250,
                messages=[{'role': 'user', 'content': [
                    {'type': 'image', 'source': {
                        'type': 'base64', 'media_type': 'image/jpeg',
                        'data': base64.standard_b64encode(frame).decode()}},
                    {'type': 'text', 'text': self.VLM_SCAN_PROMPT}]}])
            txt = ''.join(b.text for b in r.content if b.type == 'text')
            txt = txt[txt.find('{'):txt.rfind('}') + 1]
            out = json.loads(txt)
            self.audit('vlm_scan', {}, True,
                       json.dumps(out, ensure_ascii=False))
            return out
        except Exception as e:
            self.audit('vlm_scan', {}, False, str(e))
            return None

    def _approach_and_register(self, lm, name, yaw):
        """인식한 가구를 향해 전진(막히거나 1.6m)한 지점을 장소로 등록."""
        self._rotate_to(yaw)
        start = self.node.odom
        if start is None:
            return
        msg = McLocomotionVelocity()
        msg.source = SOURCE
        last, stall = start, 0
        t0 = time.monotonic()
        t_end = t0 + 8.0
        while time.monotonic() < t_end and not self._stop_requested:
            pos = self.node.odom
            lm['cells'].add(self._cell(*pos))
            # 상한 3.2m — 1.6m는 문 너머로 본 가구(2m+)에 못 미쳐
            # '침대'가 방 밖 중간 지점에 등록되던 원인이었음
            if math.hypot(pos[0] - start[0], pos[1] - start[1]) >= 3.2:
                break
            # ★가속 램프 유예 1초 — 출발 직후 저속 구간을 '막힘'으로
            # 오판해 그 자리(문간 등)를 장소로 등록하던 버그의 수정.
            # ('침대' 장소가 침대에서 1m 떨어진 문 축에 박히던 원인)
            if time.monotonic() - t0 > 1.0 and \
                    math.hypot(pos[0] - last[0], pos[1] - last[1]) < 0.005:
                stall += 1
                if stall >= 3:
                    # 닫힌 문에 막힌 것이면 열고 계속 접근 ('침대'가
                    # 문 앞 위치로 등록되던 사고 방지)
                    if self._try_open_door():
                        stall = 0
                        t0 = time.monotonic()   # 램프 유예 재시작
                        continue
                    break              # 가구 앞 도달 = 접근 완료
            else:
                stall = 0
            last = pos
            msg.forward_velocity = 0.15   # 저속 — '들이받는' 인상 완화
            self.pub_vel.publish(msg)
            time.sleep(0.1)
        msg.forward_velocity = 0.0
        self.pub_vel.publish(msg)
        pos = self.node.odom
        lm['places'][name] = {'pos': [round(pos[0], 2), round(pos[1], 2)],
                              'yaw': round(self.node.yaw_rad(), 2)}
        self.node.notify_event(f'탐색: {name} 발견 → 장소로 기억함')
        self._dock(max_dist=0.5, forward=False)   # 물러나 탐색 재개

    # 구조물/무의미 라벨 — 장소로 등록하지 않음 (개방 어휘의 노이즈 필터)
    LABEL_BLACKLIST = {'벽', '바닥', '문', '천장', '기둥', '창문', '모름',
                       '복도', '코너', '구석', '모서리'}

    @staticmethod
    def _clean_label(s):
        """개방 어휘 라벨 살균 — 첫 단어만, 10자 상한."""
        s = str(s).strip().split()[0] if str(s).strip() else ''
        return s[:10]

    def _scan_and_register(self, lm, world):
        """제자리 4방향 VLM 스캔 → 방/가구/물체 기록. 새 가구면 접근·등록."""
        base_yaw = self.node.yaw_rad()
        for k in range(4):
            if self._stop_requested:
                return
            yaw = base_yaw + k * (math.pi / 2.0)
            self._rotate_to(yaw, timeout=7.0)
            res = self._vlm_look()
            if not res:
                continue
            pos = self.node.odom or (0.0, 0.0)
            room = self._clean_label(res.get('room') or '모름')
            if room and room != '모름' and room not in lm['rooms']:
                lm['rooms'][room] = [round(pos[0], 2), round(pos[1], 2)]
                lm['places'].setdefault(
                    room, {'pos': [round(pos[0], 2), round(pos[1], 2)],
                           'yaw': round(yaw, 2)})
                self.node.notify_event(f'탐색: 여기는 {room}인 것 같음')
            for o in (res.get('objects') or []):
                o = self._clean_label(o)
                if o and o not in lm['objects_seen']:
                    lm['objects_seen'][o] = {
                        'near': [round(pos[0], 2), round(pos[1], 2)]}
            for fname in (res.get('furniture') or []):
                fname = self._clean_label(fname)
                if fname in self.LABEL_BLACKLIST:
                    continue
                if fname and fname not in lm['places']:
                    self._approach_and_register(lm, fname, yaw)
                    self._save_learned(world)
                    return                 # 위치가 변했으니 스캔 종료
        self._save_learned(world)

    def explore(self, duration_s: float = 180):
        """미지 공간 자율 탐색 — 프론티어 이동 + VLM 자가 인식 + 장소 등록."""
        from x2_sim.worlds import WORLDS
        world = self.node.current_world
        args = {'duration_s': duration_s}
        if not WORLDS.get(world, {}).get('explore'):
            msg = '이 공간은 이미 지도가 있는 곳 — 탐색 모드 공간에서만 사용'
            self.audit('explore', args, False, msg)
            return {'error': msg}
        guard = self._battery_guard()
        if guard:
            return {'error': guard}
        try:
            self.ensure_source()
            self.ensure_mode(int(McAction.LOCOMOTION_DEFAULT))
        except RuntimeError as e:
            return {'error': str(e)}
        duration_s = max(30.0, min(300.0, float(duration_s)))
        self._stop_requested = False
        lm = self._learned(world)
        blocked = set()
        t_end = time.monotonic() + duration_s
        last_scan = None
        while time.monotonic() < t_end and not self._stop_requested:
            pos = self.node.odom
            if pos is None:
                break
            cell = self._cell(*pos)
            lm['cells'].add(cell)
            # 근처 문 학습 (탐색이 문 존재·상태까지 파악)
            for dn, dd in (getattr(self.node, 'doors_info', {}) or {}).items():
                if math.hypot(dd['x'] - pos[0], dd['y'] - pos[1]) < 2.5:
                    lm.setdefault('doors', {})[dn] = {
                        'x': dd['x'], 'y': dd['y']}
            if last_scan is None or (abs(cell[0] - last_scan[0])
                                     + abs(cell[1] - last_scan[1])) >= 3:
                self._scan_and_register(lm, world)
                last_scan = self._cell(*(self.node.odom or pos))
                continue
            target = self._nearest_frontier(lm['cells'], blocked, cell)
            if target is None:
                break                      # 갈 수 있는 미방문 셀 없음
            path = self._bfs_cells(lm['cells'] | {target}, cell, target)
            step = path[1] if path and len(path) > 1 else target
            ok, detail = self._navigate(
                [(step[0] * self.CELL, step[1] * self.CELL)], timeout=12.0)
            if not ok:
                if 'stop' in detail:
                    break
                blocked.add(step)
        self._save_learned(world)
        summary = (f'탐색 종료 — 방 {sorted(lm["rooms"])}, '
                   f'배운 장소 {sorted(lm["places"])}, '
                   f'본 물체 {sorted(lm["objects_seen"])}, '
                   f'가본 셀 {len(lm["cells"])}개. '
                   f'이제 배운 장소는 go_to_place로 갈 수 있음')
        self.audit('explore', args, True, summary)
        return {'ok': summary}

    def _go_to_learned(self, place, args):
        """탐색 모드: 배운 장소로, 가본 셀만 밟아 이동."""
        cur = self.node.current_world
        lm = self._learned(cur)
        key = str(place).strip()
        entry = lm['places'].get(key)
        if entry is None:
            msg = (f'아직 모르는 장소 {place!r} — 지금까지 배운 곳: '
                   f'{sorted(lm["places"]) or "없음"}. explore로 더 '
                   f'탐색하거나, 그 자리에서 remember_place로 가르쳐줘야 함')
            self.audit('go_to_place', args, False, msg)
            return {'error': msg}
        tx, ty = entry['pos']
        pos = self.node.odom or (0.0, 0.0)
        cells = set(lm['cells'])
        start_c = self._cell(*pos)
        cells.add(start_c)
        goal_c = self._cell(tx, ty)
        if goal_c not in cells:
            goal_c = min(cells, key=lambda c: (c[0] - goal_c[0]) ** 2
                         + (c[1] - goal_c[1]) ** 2)
        path = self._bfs_cells(cells, start_c, goal_c)
        wps = []
        if path is None:
            # 현재 위치가 학습 셀과 단절(탐색 때 안 가본 곳) —
            # 최근접 학습 셀로 먼저 이동(앵커) 후 그 셀에서 경로 탐색.
            # 직선 폴백은 문틀을 깎다 스턱되는 원인이라 금지.
            anchor = min(lm['cells'],
                         key=lambda c: (c[0] * self.CELL - pos[0]) ** 2
                         + (c[1] * self.CELL - pos[1]) ** 2, default=None)
            if anchor is not None:
                wps.append((anchor[0] * self.CELL, anchor[1] * self.CELL))
                path = self._bfs_cells(cells, anchor, goal_c)
        if path:
            # 전 셀 경유 — 건너뛰기(직선화)는 문틀 모서리를 깎아 스턱됨
            wps += [(c[0] * self.CELL, c[1] * self.CELL)
                    for c in path[1:]]
        wps.append((float(tx), float(ty)))
        t0 = time.monotonic()
        ok, detail = self._navigate(wps, timeout=120.0)
        if ok:
            if entry.get('yaw') is not None:
                self._rotate_to(entry['yaw'])
            result = (f'{place}까지 배운 길로 걸어서 도착 '
                      f'({time.monotonic() - t0:.0f}초). look으로 확인 가능')
            self.audit('go_to_place', args, True, result)
            return {'ok': result}
        self.audit('go_to_place', args, False, detail)
        return {'error': f'{place}로 가는 중 문제: {detail}'}

    def _nearest_door(self, max_dist=1.35, state=None):
        doors = getattr(self.node, 'doors_info', {}) or {}
        pos = self.node.odom
        if pos is None:
            return None
        best = None
        for name, d in doors.items():
            if state and d.get('state') != state:
                continue
            dist = math.hypot(d['x'] - pos[0], d['y'] - pos[1])
            if dist < max_dist and (best is None or dist < best[0]):
                best = (dist, name, d)
        return best

    def _try_open_door(self):
        """닫힌 문이 1.4m 내면 열고 True — 내비 실패 시 자동 재시도용."""
        hit = self._nearest_door(max_dist=1.4, state='closed')
        if hit is None:
            return False
        self.pub_door.publish(String(data='open'))
        self.node.notify_event(f'닫힌 문({hit[1]}) 발견 — 열고 계속 진행')
        time.sleep(1.3)
        return True

    def open_door(self):
        if ROBOT_TARGET == 'real':
            return {'error': '실기 문 조작은 매니퓰레이션 필요 — 미지원'}
        hit = self._nearest_door(max_dist=1.35)
        if hit is None:
            return {'error': '근처(1.35m)에 문이 없음 — 문 앞까지 이동 필요'}
        self.pub_door.publish(String(data='open'))
        time.sleep(1.3)
        self.audit('open_door', {'door': hit[1]}, True, 'open')
        return {'ok': f'{hit[1]} 문을 열었음 — 이제 지나갈 수 있음'}

    def close_door(self):
        if ROBOT_TARGET == 'real':
            return {'error': '실기 문 조작은 매니퓰레이션 필요 — 미지원'}
        hit = self._nearest_door(max_dist=1.35)
        if hit is None:
            return {'error': '근처(1.35m)에 문이 없음'}
        self.pub_door.publish(String(data='close'))
        time.sleep(1.3)
        self.audit('close_door', {'door': hit[1]}, True, 'closed')
        return {'ok': f'{hit[1]} 문을 닫았음'}

    def forget_map(self):
        """현재 탐색 월드의 학습 기억을 전부 삭제 — '둘러봐 이전'으로.

        탐색 품질을 백지에서 판정하고 싶을 때 사용 (UI 🧠 버튼과 동일).
        파일 삭제 + 메모리 캐시 제거를 함께 해야 진짜 백지가 된다."""
        from x2_sim.worlds import WORLDS
        world = self.node.current_world
        if not WORLDS.get(world, {}).get('explore'):
            return {'error': '이 공간은 학습 지도를 쓰지 않음 (탐색 모드 전용)'}
        self.stop()                          # 진행 중 탐색이 있으면 중단
        if hasattr(self, '_learned_cache'):
            self._learned_cache.pop(world, None)
        try:
            os.remove(f'/ws/log/learned/{world}.json')
        except FileNotFoundError:
            pass
        self.audit('forget_map', {'world': world}, True, 'wiped')
        return {'ok': (f'{world}에 대한 기억을 전부 지웠음 — 이제 이 '
                       f'공간은 처음 보는 곳. explore로 다시 배울 수 있음')}

    def remember_place(self, name: str):
        """사용자가 알려준 이름으로 현재 위치를 장소로 저장 (최소 교육)."""
        from x2_sim.worlds import WORLDS
        world = self.node.current_world
        if not WORLDS.get(world, {}).get('explore'):
            return {'error': '지도가 이미 있는 공간 — 탐색 모드에서만 사용'}
        pos = self.node.odom
        if pos is None:
            return {'error': '위치 정보 없음'}
        lm = self._learned(world)
        key = str(name).strip()
        lm['places'][key] = {'pos': [round(pos[0], 2), round(pos[1], 2)],
                             'yaw': round(self.node.yaw_rad(), 2)}
        lm['cells'].add(self._cell(*pos))
        self._save_learned(world)
        self.audit('remember_place', {'name': key}, True, str(lm['places'][key]))
        return {'ok': f'기억했음: {key} = 현재 위치 '
                      f'({pos[0]:.1f}, {pos[1]:.1f})'}

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
            retried = 0
            door_tried = False
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
                    if retried < 3:
                        # 언스턱 3단: ①후진 1.5s ②좌측 이동 1.2s
                        # ③우측 이동 2.0s(좌측 복귀분 포함) — 문틀이
                        # 어느 쪽에 있는지 모르므로 양방향 시도
                        msg.forward_velocity = (-0.2 if retried == 0
                                                else 0.0)
                        msg.lateral_velocity = (0.0, 0.15, -0.15)[retried]
                        msg.angular_velocity = 0.0
                        t_back = time.monotonic() + (1.5, 1.2, 2.0)[retried]
                        while (time.monotonic() < t_back
                               and not self._stop_requested):
                            self.pub_vel.publish(msg)
                            time.sleep(0.1)
                        msg.forward_velocity = 0.0
                        msg.lateral_velocity = 0.0
                        self.pub_vel.publish(msg)
                        retried += 1
                        best, best_t = None, time.monotonic()
                        best_err = None
                        continue
                    if not door_tried and self._try_open_door():
                        # 닫힌 문에 막힌 것 — 열고 같은 웨이포인트 재시도
                        door_tried = True
                        retried = 0
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
        from x2_sim.worlds import WORLDS as _WORLDS
        # 탐색 모드 공간은 별칭 사전을 쓰지 않는다 — 학습 장소명 그대로
        if _WORLDS.get(self.node.current_world, {}).get('explore'):
            return self._go_to_learned(place, args)
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

        from x2_sim.worlds import WORLDS
        # ★home 월드 안의 방이면 순간이동이 아니라 실제로 걸어간다.
        #   다른 공간(스튜디오 등)에 있어도 먼저 home으로 복귀 후 걷는다
        #   — 모든 장소가 항상 연결되어 있게.
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
        """일어서기 — 실기 McAction STAND_UP_DEFAULT(2005). 누운 상태
        (시뮬 lie)에서는 자세 토픽으로 기립."""
        if ROBOT_TARGET == 'sim' \
                and getattr(self.node, 'sim_posture', 'stand') == 'lie':
            self.pub_posture.publish(String(data='stand'))
            time.sleep(1.8)             # 역글라이드+기립 램프 완료 대기
            try:                        # MC 모드가 SIT에 남아있으면 정리
                if self.current_action() == int(McAction.SIT_DOWN_DEFAULT):
                    self.ensure_source()
                    self.set_action(int(McAction.STAND_UP_DEFAULT))
            except RuntimeError:
                pass
            self.audit('stand_up', {}, True, 'stood from lie')
            return {'ok': '누운 데서 일어나 침대/바닥에서 내려왔음'}
        try:
            self.ensure_source()
            self.set_action(int(McAction.STAND_UP_DEFAULT))
        except RuntimeError as e:
            self.audit('stand_up', {}, False, str(e))
            return {'error': str(e) + ' (앉아있지 않으면 실패함)'}
        self.audit('stand_up', {}, True, 'standing')
        return {'ok': '일어섰음 (STAND_DEFAULT)'}

    def lie_down(self):
        """눕기 (시뮬 전용 자세) — 정면 0.55m 지점이 침대/소파처럼 낮은
        가구면 그 위로 올라가 눕고, 아니면 바닥에 눕는다."""
        if ROBOT_TARGET == 'real':
            return {'error': '실기 눕기는 미지원 (실기 v1 제외 — 안전 검토 필요)'}
        if getattr(self.node, 'sim_posture', 'stand') == 'lie':
            return {'ok': '이미 누워 있음'}
        if self.node.carrying:
            return {'error': f'{self.node.carrying}을(를) 들고는 누울 수 '
                             f'없음 — 먼저 put_down'}
        # 앉은 상태에서 눕기: MC 모드가 SIT에 잠긴 채 자세만 바뀌면
        # 이후 이동이 전부 게이트에 막힘 — 먼저 정식으로 일어난다
        try:
            if self.current_action() == int(McAction.SIT_DOWN_DEFAULT):
                self.ensure_source()
                self.set_action(int(McAction.STAND_UP_DEFAULT))
                self._wait_mode(int(McAction.STAND_DEFAULT))
                time.sleep(1.6)          # 기립 램프 완료
        except RuntimeError:
            pass
        self.pub_posture.publish(String(data='lie'))
        time.sleep(1.8)                 # 글라이드+눕기 램프 완료 대기
        self.audit('lie_down', {}, True, 'lying')
        return {'ok': ('누웠음 — 시야가 천장을 향한다. 누운 동안 이동 '
                       '불가, stand_up으로 일어남. 누운 상태에서 '
                       'upright=false는 정상')}

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
