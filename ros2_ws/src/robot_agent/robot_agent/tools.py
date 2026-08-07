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
import time
from datetime import datetime, timezone

from aimdk_msgs.msg import McAction, McInputAction, McLocomotionVelocity
from std_msgs.msg import String
from aimdk_msgs.srv import (
    GetMcAction,
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

AUDIT_PATH = os.environ.get('AGENT_AUDIT_LOG', '/ws/log/tool_audit.jsonl')

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
        self.pub_vel = node.create_publisher(
            McLocomotionVelocity, '/aima/mc/locomotion/velocity', 10)
        self._stop_requested = False
        # 하드와이어 정지 수신 (UI 버튼 등 외부 독립 경로) — 기본 콜백
        # 그룹이라 채팅 처리 중에도 즉시 실행됨
        node.create_subscription(
            String, '/agent/hard_stop', self._on_hard_stop, 10)

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

    def move(self, vx=0.0, vy=0.0, wz=0.0, duration=1.0):
        args = {'vx': vx, 'vy': vy, 'wz': wz, 'duration': duration}
        # 하드 클램프 (LLM이 뭐라 하든 코드가 지킨다)
        vx = max(-MAX_VX, min(MAX_VX, float(vx)))
        vy = max(-MAX_VY, min(MAX_VY, float(vy)))
        wz = max(-MAX_WZ, min(MAX_WZ, float(wz)))
        duration = max(0.1, min(MAX_DURATION, float(duration)))
        try:
            self.ensure_source()
            self.ensure_mode(int(McAction.LOCOMOTION_DEFAULT))
        except RuntimeError as e:
            self.audit('move', args, False, str(e))
            return {'error': str(e)}
        # 재발행 루프: 유효기간 0.5s 대응, 10Hz
        self._stop_requested = False
        msg = McLocomotionVelocity()
        msg.source = SOURCE
        msg.forward_velocity = vx
        msg.lateral_velocity = vy
        msg.angular_velocity = wz
        t_end = time.monotonic() + duration
        while time.monotonic() < t_end and not self._stop_requested:
            self.pub_vel.publish(msg)
            time.sleep(0.1)
        # 정지 명령으로 마무리
        msg.forward_velocity = 0.0
        msg.lateral_velocity = 0.0
        msg.angular_velocity = 0.0
        self.pub_vel.publish(msg)
        stopped = ' (중간에 stop 요청으로 중단됨)' if self._stop_requested else ''
        result = f'이동 완료: vx={vx} vy={vy} wz={wz} {duration}s{stopped}'
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
        # 시뮬: PlayTts 서비스가 없으므로 UI 이벤트로 대체 (Phase 4에서 표시)
        self.node.publish_say(text)
        self.audit('speak', {'text': text}, True, 'emitted')
        return {'ok': f'말함: {text}'}

    def show_emoji(self, emoji: str):
        self.node.publish_emoji(emoji)
        self.audit('show_emoji', {'emoji': emoji}, True, 'emitted')
        return {'ok': f'표정 표시: {emoji}'}

    def stop(self):
        self._stop_requested = True
        msg = McLocomotionVelocity()
        msg.source = SOURCE
        self.pub_vel.publish(msg)  # 속도 0
        self.audit('stop', {}, True, 'stopped')
        return {'ok': '정지 명령 발행 (속도 0, 모드 유지)'}
