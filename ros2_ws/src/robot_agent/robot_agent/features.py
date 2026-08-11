#!/usr/bin/env python3
"""기능 레지스트리 — sim/real 전환의 단일 소스 (A-0).

"config 전환" 원칙이 프롬프트·UI 레벨에서 침식되는 것을 막기 위해,
기능별 가용성을 여기 한 곳에 선언하고 아래 세 곳이 전부 이 파일에서
파생된다:
  1. LLM 툴 정의 목록 (llm.py TOOLS 필터링)
  2. 프롬프트의 "몸에 대한 사실" 절 (llm.py body_facts)
  3. UI 노출 (backend /features → 프론트 버튼/드롭다운 표시 여부)

tier 의미:
  parity          — 시뮬/실기 동일 계약. 전환 시 무수정.
  real_needs_work — 실기 구현이 필요. real_available이 True가 되기
                    전까지 real 모드에서 자동 숨김.
  sim_only        — 시뮬 전용. real 모드에서 항상 숨김/비활성.

실기 구성이 확정되면(예: 핸드 옵션 포함 여부) 이 파일의
real_available 값만 바꾸면 툴·프롬프트·UI가 함께 전환된다.
"""
import os

ROBOT_TARGET = os.environ.get('ROBOT_TARGET', 'sim')

FEATURES = {
    # ---------- LLM 툴 ----------
    'look':        {'kind': 'tool', 'tier': 'parity'},
    'get_status':  {'kind': 'tool', 'tier': 'parity'},
    'move':        {'kind': 'tool', 'tier': 'parity'},
    'turn':        {'kind': 'tool', 'tier': 'parity'},
    'face':        {'kind': 'tool', 'tier': 'parity'},
    'play_motion': {'kind': 'tool', 'tier': 'parity'},
    'speak':       {'kind': 'tool', 'tier': 'parity'},
    'show_emoji':  {'kind': 'tool', 'tier': 'parity'},
    'stop':        {'kind': 'tool', 'tier': 'parity'},
    # sit/stand: McAction 코드(2000/2005) 동일 — 인터페이스는 parity.
    # 실기 착석의 물리 전제(갠트리 금지 등)는 Phase 6 안전 체크리스트.
    'sit_down':    {'kind': 'tool', 'tier': 'parity'},
    'stand_up':    {'kind': 'tool', 'tier': 'parity'},
    # 핸드 하드웨어 필요 — 견적 회신 전 기본값: 미탑재(정책 A-3 (a)).
    # 핸드 포함 확정 시 real_available=True로 바꾸고 HandCommand 구현.
    'pick_up':     {'kind': 'tool', 'tier': 'real_needs_work',
                    'real_available': False},
    'put_down':    {'kind': 'tool', 'tier': 'real_needs_work',
                    'real_available': False},
    # SLAM 맵핑 + NaviPoint 연결(Phase 6.5) 전까지 real에서 숨김.
    'go_to_place': {'kind': 'tool', 'tier': 'real_needs_work',
                    'real_available': False},
    'go_to_object': {'kind': 'tool', 'tier': 'real_needs_work',
                     'real_available': False},
    # 미지 공간 탐색·학습 — 실기는 SLAM 자율 탐색으로 재구현 (Phase 6.5+)
    'explore':        {'kind': 'tool', 'tier': 'real_needs_work',
                       'real_available': False},
    'remember_place': {'kind': 'tool', 'tier': 'real_needs_work',
                       'real_available': False},
    # 눕기 — 실기 v1 제외 (낙상 유사 자세라 안전 검토 필요)
    'lie_down':       {'kind': 'tool', 'tier': 'real_needs_work',
                       'real_available': False},
    'forget_map':     {'kind': 'tool', 'tier': 'real_needs_work',
                       'real_available': False},
    # 문 조작 — 실기는 핸드 매니퓰레이션 필요 (v1 제외)
    'open_door':      {'kind': 'tool', 'tier': 'real_needs_work',
                       'real_available': False},
    'close_door':     {'kind': 'tool', 'tier': 'real_needs_work',
                       'real_available': False},

    # ---------- UI ----------
    'world_switch':  {'kind': 'ui', 'tier': 'sim_only'},  # 🌍 드롭다운
    'sim_reset':     {'kind': 'ui', 'tier': 'sim_only'},  # 🔄 리셋 버튼
    'memory_forget': {'kind': 'ui', 'tier': 'sim_only'},  # 🧠 기억 삭제

    # ---------- 내부 동작 (문서화 + 점검용) ----------
    'collision_block': {'kind': 'internal', 'tier': 'sim_only'},
    'velocity_ramp':   {'kind': 'internal', 'tier': 'sim_only'},
    'ondemand_render': {'kind': 'internal', 'tier': 'sim_only'},
    'kinematic_carry': {'kind': 'internal', 'tier': 'sim_only'},
    'dock_maneuver':   {'kind': 'internal', 'tier': 'sim_only'},
}


def is_enabled(name: str, target: str = None) -> bool:
    target = target or ROBOT_TARGET
    f = FEATURES.get(name)
    if f is None:
        return False
    if target == 'sim':
        return True
    if f['tier'] == 'parity':
        return True
    if f['tier'] == 'real_needs_work':
        return bool(f.get('real_available'))
    return False    # sim_only


def enabled_tools(target: str = None) -> set:
    return {n for n, f in FEATURES.items()
            if f['kind'] == 'tool' and is_enabled(n, target)}


def ui_flags(target: str = None) -> dict:
    return {n: is_enabled(n, target)
            for n, f in FEATURES.items() if f['kind'] == 'ui'}


def body_facts(target: str = None) -> str:
    """프롬프트 '몸에 대한 사실' 절 — 가용 기능에 따라 생성.

    비활성 기능은 '없다'고 명시해, 실기 구성과 프롬프트 사실이
    어긋나는 것(예: 핸드 없는 실기에 '집을 수 있다')을 차단한다."""
    lines = [
        '- 너는 팔 2개(7자유도), 다리 2개, 허리, 머리를 가진 휴머노이드다.',
    ]
    if is_enabled('pick_up', target):
        lines.append(
            '- 작은 물체는 집고(pick_up) 내려놓을(put_down) 수 있다. 집을 수 있는\n'
            '  물체와 거리는 상태의 nearby_objects_dist_m에 있다(영문 이름 사용:\n'
            '  apple, mug 등 — 사용자가 "사과"라고 하면 apple로 매핑). 1.05m보다\n'
            '  멀면 먼저 move로 접근한다. 들고 있는 것은 holding에 표시된다.\n'
            '  한 번에 하나만 들 수 있다.')
    else:
        lines.append(
            '- 현재 구성에는 물체를 집는 핸드가 없다. 집기/놓기 요청은 솔직히\n'
            '  안 된다고 말한다.')
    lines.append(
        '- 고개 돌리기는 preset(POINT_HEAD/SHAKE_HEAD)에는 있지만 특정 각도로\n'
        '  돌리는 기능은 아직 없다 — 대신 몸 전체를 turn(각도)으로 돌린다.')
    if is_enabled('go_to_place', target):
        lines.append(
            '- 기본 공간은 집(home)이며 거실/주방/현관/소파/냉장고/스튜디오/\n'
            '  사무실로 걸어서 이동한다 (go_to_place — 물건을 든 채로도 이동\n'
            '  가능). "소파에 앉아" 요청 흐름: go_to_place(\'소파\') → 도착하면\n'
            '  자동으로 소파를 등지고 서므로 → sit_down. 들고/앉은 상태는 반드시\n'
            '  상태값(holding, mode)으로 확인하고 추측으로 말하지 않는다.')
    else:
        lines.append(
            '- 장소 단위 자동 이동 기능은 아직 없다. 이동은 move(방향+시간)로만\n'
            '  가능하다. "주방으로 가" 같은 요청은 아직 지원하지 않는다고\n'
            '  솔직히 말한다.')
    lines.append(
        '- 넘어졌는지는 화면 느낌으로 추측하지 말고 반드시 get_status의\n'
        '  upright 값으로 판단한다. upright=true면 서 있는 것이다. 화면이\n'
        '  이상하면 대부분 장애물에 부딪혀 이동이 차단됐거나 고개가 밀린 것.')
    lines.append(
        '- 장애물(테이블 등)에 닿으면 이동이 자동 차단된다. 이동했는데 화면이\n'
        '  안 변하면 막힌 것이니 다른 방향을 시도하거나 사용자에게 알린다.')
    lines.append(
        '- get_status의 position(현재 좌표)과 from_start(시작점 기준 변위)로\n'
        '  자기 위치를 안다. "처음 위치로 돌아가"는 from_start의 dx, dy를\n'
        '  역방향으로 이동해서 수행할 수 있다 (yaw_deg를 고려해 몸 기준\n'
        '  방향으로 환산할 것).')
    return '\n'.join(lines)
