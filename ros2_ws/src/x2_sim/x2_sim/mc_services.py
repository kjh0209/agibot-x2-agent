#!/usr/bin/env python3
"""MC(모션 컨트롤) 목 서비스 — 실기 프로토콜 강제 계층.

실기에서 locomotion 명령은 (1) 올바른 McAction 모드일 때, (2) 등록된
입력 소스에서 왔을 때만 수용된다. 자체 시뮬이 아무 명령이나 받아주면
에이전트가 "모드 확인 없이 velocity부터 쏘는" 코드로 완성되고, 실기
전환 시 조용히 무시되는 버그가 된다. 이 계층이 그 의미론을 시뮬에서
강제한다: 에이전트는 시뮬 단계부터
  소스 등록(SetMcInputSource) → 모드 전환(SetMcAction) → 명령
시퀀스를 지켜야 한다.

서비스 이름은 실기(topics_and_services)와 동일:
  /aimdk_5Fmsgs/srv/SetMcAction, GetMcAction, SetMcPresetMotion,
  /aimdk_5Fmsgs/srv/SetMcInputSource, GetCurrentInputSource
"""
from aimdk_msgs.msg import McAction, McInputAction
from aimdk_msgs.srv import (
    GetCurrentInputSource,
    GetMcAction,
    SetMcAction,
    SetMcInputSource,
    SetMcPresetMotion,
)

# CommonState 값 (aimdk_msgs/msg/CommonState.msg)
STATE_SUCCESS = 1
STATE_FAILURE = 2
STATE_INVALID = 5

ACTION_NAMES = {
    1: 'PASSIVE_DEFAULT', 2: 'SOFT_EMERGENCY_STOP', 3: 'DAMPING_DEFAULT',
    4: 'ZERO_TORQUE_DEFAULT', 100: 'JOINT_DEFAULT', 101: 'JOINT_FREEZE',
    200: 'STAND_DEFAULT', 201: 'STAND_BODY_CONTROL',
    300: 'LOCOMOTION_DEFAULT', 301: 'RUN_DEFAULT', 302: 'LOCOMOTION_STEP',
    2000: 'SIT_DOWN_DEFAULT', 2005: 'STAND_UP_DEFAULT',
}

# 허용 전이(간략화된 실기 규칙): 목표 액션 → 허용되는 현재 액션 집합
# (비상 계열은 어디서든 진입 가능)
EMERGENCY = {McAction.DAMPING_DEFAULT, McAction.ZERO_TORQUE_DEFAULT,
             McAction.SOFT_EMERGENCY_STOP, McAction.PASSIVE_DEFAULT}
TRANSITIONS = {
    McAction.STAND_DEFAULT: {McAction.PASSIVE_DEFAULT,
                             McAction.DAMPING_DEFAULT,
                             McAction.ZERO_TORQUE_DEFAULT,
                             McAction.JOINT_DEFAULT,
                             McAction.STAND_BODY_CONTROL,
                             McAction.LOCOMOTION_DEFAULT},
    McAction.STAND_BODY_CONTROL: {McAction.STAND_DEFAULT},
    McAction.LOCOMOTION_DEFAULT: {McAction.STAND_DEFAULT,
                                  McAction.STAND_BODY_CONTROL},
    McAction.JOINT_DEFAULT: {McAction.PASSIVE_DEFAULT,
                             McAction.DAMPING_DEFAULT,
                             McAction.ZERO_TORQUE_DEFAULT},
}

# 알려진 프리셋 모션 ID (McPresetMotion.msg 상수 — 화이트리스트)
KNOWN_MOTIONS = {
    1001: 'RAISE_HAND', 1002: 'WAVE_HAND', 1003: 'SHAKE_HAND',
    1004: 'FLYING_KISS_HAND', 1008: 'CLAP_HAND', 1009: 'CLIPFIST',
    1013: 'SALUTE', 2001: 'TURN_WAVE_HAND', 3001: 'INTERACTION_BOW',
    3002: 'INTERACTION_LIKE', 3003: 'INTERACTION_YE',
    3004: 'INTERACTION_SWEATHEART', 3006: 'INTERACTION_SAD',
    3007: 'INTERACTION_LIGHTWAVE', 3008: 'INTERACTION_HUG',
    3009: 'INTERACTION_HANDX', 3010: 'INTERACTION_CHESTWAVE',
    3011: 'INTERACTION_CHEER', 3012: 'INTERACTION_BLOWKISS',
    3015: 'HITCLAP', 3016: 'INTERACTION_SPEAK', 4001: 'POINT_HEAD',
    4002: 'SHAKE_HEAD',
}


class McMock:
    """sim 노드에 실기 MC 의미론을 붙이는 목 구현."""

    def __init__(self, node):
        self.node = node
        self.log = node.get_logger()
        # 입력 소스 레지스트리: name → {'priority': int, 'enabled': bool}
        # 실기의 기본 소스(리모컨)는 항상 존재한다고 가정
        self.sources = {'rc': {'priority': 100, 'enabled': True}}
        # 시작 상태: 위치제어 서기 (시뮬 로봇이 서 있는 상태와 일치)
        self.current_action = int(McAction.STAND_DEFAULT)

        node.create_service(
            SetMcAction, '/aimdk_5Fmsgs/srv/SetMcAction', self.set_action)
        node.create_service(
            GetMcAction, '/aimdk_5Fmsgs/srv/GetMcAction', self.get_action)
        node.create_service(
            SetMcInputSource, '/aimdk_5Fmsgs/srv/SetMcInputSource',
            self.set_input_source)
        node.create_service(
            GetCurrentInputSource, '/aimdk_5Fmsgs/srv/GetCurrentInputSource',
            self.get_input_source)
        node.create_service(
            SetMcPresetMotion, '/aimdk_5Fmsgs/srv/SetMcPresetMotion',
            self.set_preset_motion)
        self.log.info(
            'MC 목 서비스 활성 — 시작 액션 STAND_DEFAULT(200), 소스 rc')

    # ---------- 게이트 (sim_node가 velocity 콜백에서 호출) ----------
    def gate_velocity(self, source: str) -> bool:
        """실기 의미론: locomotion 모드 + 등록/활성 소스일 때만 True."""
        if self.current_action not in (McAction.LOCOMOTION_DEFAULT,
                                       McAction.RUN_DEFAULT,
                                       McAction.LOCOMOTION_STEP):
            self.log.warning(
                f'velocity 무시: 현재 액션 '
                f'{ACTION_NAMES.get(self.current_action, self.current_action)}'
                f' — LOCOMOTION_DEFAULT(300)로 전환 필요 (실기 동일 거동)')
            return False
        src = self.sources.get(source)
        if not src or not src['enabled']:
            self.log.warning(
                f'velocity 무시: 미등록/비활성 소스 {source!r} — '
                f'SetMcInputSource(ADD)로 등록 필요 (등록된 소스: '
                f'{list(self.sources)})')
            return False
        return True

    def motion_allowed(self) -> bool:
        return self.current_action in (McAction.STAND_DEFAULT,
                                       McAction.STAND_BODY_CONTROL)

    # ---------- 서비스 구현 ----------
    def set_action(self, req, res):
        target = int(req.command.action.value)
        source = req.source
        name = ACTION_NAMES.get(target, str(target))
        src = self.sources.get(source)
        if not src or not src['enabled']:
            res.response.status.value = STATE_INVALID
            res.response.message = (
                f'unregistered/disabled source: {source!r}')
            self.log.warning(
                f'SetMcAction({name}) 거부: 소스 {source!r} 미등록')
            return res
        allowed = (target in EMERGENCY
                   or self.current_action in TRANSITIONS.get(target, set()))
        if not allowed:
            res.response.status.value = STATE_FAILURE
            res.response.message = (
                f'invalid transition '
                f'{ACTION_NAMES.get(self.current_action)} -> {name}')
            self.log.warning(f'SetMcAction 거부: {res.response.message}')
            return res
        prev = self.current_action
        self.current_action = target
        res.response.status.value = STATE_SUCCESS
        res.response.message = 'ok'
        self.log.info(
            f'액션 전환: {ACTION_NAMES.get(prev, prev)} → {name} '
            f'(source={source!r})')
        # 비상 계열이면 이동 명령 즉시 소거
        if target in EMERGENCY:
            self.node.clear_locomotion()
        return res

    def get_action(self, req, res):
        res.info.current_action.value = self.current_action
        res.info.action_desc = ACTION_NAMES.get(
            self.current_action, str(self.current_action))
        res.info.status.value = STATE_SUCCESS
        return res

    def set_input_source(self, req, res):
        act = int(req.action.value)
        s = req.input_source
        if act == McInputAction.INPUTACTION_ADD:
            self.sources[s.name] = {
                'priority': int(s.priority), 'enabled': True}
            self.log.info(
                f'입력 소스 등록: {s.name!r} (priority={s.priority})')
        elif act == McInputAction.INPUTACTION_MODIFY and s.name in self.sources:
            self.sources[s.name]['priority'] = int(s.priority)
        elif act == McInputAction.INPUTACTION_DELETE:
            self.sources.pop(s.name, None)
            self.log.info(f'입력 소스 삭제: {s.name!r}')
        elif act == McInputAction.INPUTACTION_ENABLE and s.name in self.sources:
            self.sources[s.name]['enabled'] = True
        elif act == McInputAction.INPUTACTION_DISABLE and s.name in self.sources:
            self.sources[s.name]['enabled'] = False
        else:
            res.response.state.value = STATE_INVALID
            return res
        res.response.state.value = STATE_SUCCESS
        return res

    def get_input_source(self, req, res):
        # 활성 소스 중 최고 우선순위를 현재 소스로 보고
        active = [(v['priority'], k) for k, v in self.sources.items()
                  if v['enabled']]
        if active:
            prio, name = max(active)
            res.input_source.name = name
            res.input_source.priority = prio
        res.response.state.value = STATE_SUCCESS
        return res

    def set_preset_motion(self, req, res):
        mid = int(req.motion.value)
        name = KNOWN_MOTIONS.get(mid)
        if name is None:
            res.response.state.value = STATE_INVALID
            self.log.warning(f'프리셋 모션 거부: 미등록 ID {mid}')
            return res
        if not self.motion_allowed():
            res.response.state.value = STATE_FAILURE
            self.log.warning(
                f'프리셋 모션 {name} 거부: 현재 액션 '
                f'{ACTION_NAMES.get(self.current_action)} — '
                f'STAND 계열에서만 허용 (매뉴얼: locomotion 중 상지 동작 불가)')
            return res
        self.log.info(f'프리셋 모션 실행: {name} ({mid})')
        self.node.play_motion(name)
        res.response.state.value = STATE_SUCCESS
        return res
