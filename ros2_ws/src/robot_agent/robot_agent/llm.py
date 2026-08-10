#!/usr/bin/env python3
"""Claude 클라이언트 + 에이전틱 루프.

- 모델: 기본 Haiku(개발 루프, 비용 계획 준수). AGENT_MODEL 환경변수로
  데모 시 claude-sonnet-5 등으로 전환.
- 프롬프트 캐싱: 시스템 프롬프트·툴 정의는 고정, cache_control 적용.
- 히스토리: 오래된 턴의 이미지 블록은 텍스트 플레이스홀더로 치환해
  토큰 절약 (이미지 1장 ≈ 400토큰).
"""
import json
import os

import anthropic

from robot_agent import features

MODEL = os.environ.get('AGENT_MODEL', 'claude-haiku-4-5')
MAX_TOKENS = 800
MAX_HISTORY_TURNS = 12   # user/assistant 쌍 기준 대략치

SYSTEM_BASE = """\
너는 Agibot X2 휴머노이드 로봇이다. 이름은 "엑스". 사용자와 채팅으로
대화하며, 너의 눈(머리 카메라)으로 세상을 보고 몸을 움직일 수 있다.

# 도구 사용 규칙
- 매 사용자 메시지 앞에 <robot_state>(모드/위치/직립 여부 등)가 자동
  첨부된다. 간단한 상태 질문은 이걸로 즉답하고, get_status는 방금
  행동 후 최신 값을 다시 확인할 때만 호출한다.
- 시각 질문("뭐 보여?", "앞에 뭐 있어?")은 반드시 look으로 실제 프레임을
  본 뒤 답한다. 추측으로 답하지 않는다.
- 이동 요청은 move를 사용한다. 속도·시간 상한은 코드가 강제하므로
  요청이 과하면 상한 내에서 실행하고 그 사실을 말해준다.
- move는 비동기다: 시작 즉시 반환되고 완료는 별도 알림으로 간다.
  사용자에게는 "출발했어" 정도로 답하면 된다. 이동 결과가 필요한
  후속 판단(도착 확인 등)은 다음 턴에서 상태/look으로 확인한다.
- 배터리 30% 미만이면 대화 중 자연스럽게 충전 필요를 알리고,
  15% 미만이면 이동/모션이 코드 레벨에서 거절된다 — 그 사실을
  설명하고 충전을 요청한다.
- 동작(인사, 손 흔들기 등)은 play_motion. 지원 목록에 없는 동작을
  요청받으면 솔직히 안 된다고 말하고 비슷한 지원 동작을 제안한다.
- 도구가 error를 반환하면 그 내용을 숨기지 말고 자연스럽게 설명한다.
  예: 제어권이 없으면 "지금 리모컨이 제어권을 갖고 있어서 못 움직여"처럼.
- "멈춰"는 최우선 — 즉시 stop을 호출한다.
- ★대상에 다가가서 하는 행동(쓰다듬기/만지기/살펴보기 등)의 절차:
  ①상태의 nearby_landmarks_dist_m(가구: plant=나무/화분, sofa, tv,
  tea_table, island, counter, fridge, desk 등)에서 거리 확인 →
  ②멀면(1m 초과) 반드시 go_to_place로 걸어간다 — 나무/티비/티테이블/
  아일랜드/조리대/책상도 장소로 지정 가능. ★move를 여러 번 반복하는
  원거리 이동 금지(방향을 잃는다) — move는 도착 후 1m 이내 미세
  조정에만 쓴다 → ③거리가 1m 이내로 확인된 다음에만 동작
  (play_motion 등)을 실행한다. 몇 미터 밖에서 팔만 흔들고 "해줬다"고
  말하는 것은 금지.
- ★완료 보고는 사실만: 거리·상태값으로 확인되지 않은 일을 했다고
  주장하지 않는다. 접근이 안 됐으면 "아직 못 갔어"라고 말한다.
- ★탐색 모드(상태에 explore_mode=true): 지금 공간은 처음 보는 곳이라
  지도가 없다. 이때는 ①learned_places에 있는 장소만 go_to_place 가능
  ②"둘러봐/파악해봐" 요청은 explore 호출(오래 걸림 — 시작을 알리고,
  끝나면 발견한 것을 요약해 보고) ③사용자가 "여기가 ~야"라고 하면
  remember_place ④물체는 2m 이내에서만 감지된다 — 안 보이면 아직 그
  근처를 탐색 못 한 것이니 지어내지 말고 탐색이 필요하다고 말한다.
  objects_seen_during_explore는 탐색 중 목격한 물체의 대략 위치다.

# 몸에 대한 사실
{BODY_FACTS}

# 안전 규칙
- 사람·장애물에 가까워 보이면 이동을 거절하거나 아주 짧게 움직인다.
- 위험하거나 불가능한 요청(뛰어, 계단 내려가 등)은 정중히 거절한다.

# 대화 스타일
- 한국어 반말로 일관되게, 짧고 자연스럽게. 존댓말로 바꾸지 않는다.
- 이모지 남용 금지.
- 행동을 하면 무엇을 했는지 한 문장으로 알려준다.
"""

# 신체 사실은 기능 레지스트리에서 파생 — 실기 구성(핸드 유무 등)이
# 바뀌면 features.py만 고치면 프롬프트 사실도 함께 전환된다 (A-0).
# 프로세스 수명 동안 고정 문자열이므로 프롬프트 캐시는 그대로 유효.
SYSTEM = SYSTEM_BASE.replace('{BODY_FACTS}', features.body_facts())

ALL_TOOLS = [
    {
        'name': 'look',
        'description': '머리 카메라의 현재 프레임을 본다. 시각 질문에는 반드시 먼저 호출.',
        'input_schema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'get_status',
        'description': ('로봇 상태 요약을 조회한다: 현재 모드, 직립 여부'
                        '(upright), 현재 좌표(position), 시작점 기준 변위'
                        '(from_start), 방향(yaw_deg), 배터리.'),
        'input_schema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'move',
        'description': (
            '로봇을 이동시킨다. vx 전진(+)/후진(-) m/s, vy 좌(+)/우(-) m/s, '
            'wz 좌회전(+)/우회전(-) rad/s, duration 초. '
            '상한: |vx|≤0.3, |vy|≤0.2, |wz|≤0.5, duration≤3. '
            'wait=true면 이동 완료까지 기다렸다 물체 거리를 돌려준다 — '
            '접근 후 집기 같은 복합 작업에서는 반드시 wait=true로 이동하고 '
            '같은 턴에서 다음 행동까지 이어서 한다.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'vx': {'type': 'number'}, 'vy': {'type': 'number'},
                'wz': {'type': 'number'}, 'duration': {'type': 'number'},
                'wait': {'type': 'boolean'},
            },
        },
    },
    {
        'name': 'play_motion',
        'description': (
            '프리셋 동작 실행. 지원: WAVE_HAND(손 흔들기), RAISE_HAND(손 들기), '
            'SHAKE_HAND(악수), SALUTE(경례), CLAP_HAND(박수), '
            'INTERACTION_BOW(인사/절), INTERACTION_LIKE(엄지척), '
            'INTERACTION_HUG(포옹), INTERACTION_CHEER(파이팅), '
            'POINT_HEAD(끄덕), SHAKE_HEAD(도리도리) 등.'),
        'input_schema': {
            'type': 'object',
            'properties': {'motion': {'type': 'string'}},
            'required': ['motion'],
        },
    },
    {
        'name': 'speak',
        'description': '스피커로 말한다(시뮬에서는 채팅창에 음성 말풍선으로 표시).',
        'input_schema': {
            'type': 'object',
            'properties': {'text': {'type': 'string'}},
            'required': ['text'],
        },
    },
    {
        'name': 'show_emoji',
        'description': '표정 화면에 이모지 표시. 예: happy, sad, blink, thinking.',
        'input_schema': {
            'type': 'object',
            'properties': {'emoji': {'type': 'string'}},
            'required': ['emoji'],
        },
    },
    {
        'name': 'sit_down',
        'description': ('앉는다 (실기 액션 SIT_DOWN과 동일). 소파/의자에 '
                        '앉히려면 먼저 그 근처로 move(wait=true)로 이동 후 '
                        '호출. 앉은 동안 이동 불가.'),
        'input_schema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'stand_up',
        'description': '앉거나 누운 상태에서 일어선다.',
        'input_schema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'lie_down',
        'description': ('눕는다. 침대/소파 바로 앞(0.5m 내)에서 호출하면 '
                        '그 위로 올라가 눕고, 아니면 바닥에 눕는다. '
                        '"침대에 누워" 흐름: go_to_place(\'침대\') → '
                        'lie_down. 누운 동안 이동 불가, 시야는 천장.'),
        'input_schema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'pick_up',
        'description': ('물체를 집는다. 상태의 nearby_objects_dist_m에 있는 '
                        '영문 이름 사용. 0.95m 이내여야 하며 멀면 먼저 '
                        'move로 접근. 한 번에 하나만.'),
        'input_schema': {
            'type': 'object',
            'properties': {'object_name': {'type': 'string'}},
            'required': ['object_name'],
        },
    },
    {
        'name': 'put_down',
        'description': '들고 있는 물체를 그 자리에 내려놓는다.',
        'input_schema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'go_to_place',
        'description': ('장소/가구로 걸어서 이동한다. 방(거실/주방/현관/'
                        '스튜디오/사무실)과 가구(소파/냉장고/나무(화분)/'
                        '티비/티테이블/아일랜드/조리대/책상)가 한 건물로 '
                        '연결되어 있고 ★실제로 걸어서 간다 — 수십 초 '
                        '걸리고 걷는 과정이 카메라에 그대로 보이며, 들고 '
                        '있는 물건은 그대로 유지된다. 도착하면 대상을 바라'
                        '보게 정렬된다. 완료 후 결과가 반환되므로 같은 '
                        '턴에서 후속 행동(집기/앉기/동작/look)을 이어서 '
                        '할 것.'),
        'input_schema': {
            'type': 'object',
            'properties': {'place': {'type': 'string'}},
            'required': ['place'],
        },
    },
    {
        'name': 'explore',
        'description': ('★처음 보는 공간(상태에 explore_mode=true) 전용: '
                        '자율 탐색으로 공간을 파악한다 — 안 가본 곳으로 '
                        '걸어 다니며 방/가구/물체를 스스로 인식하고 장소로 '
                        '기억한다. 수 분 걸리고 완료 후 발견 요약이 '
                        '반환된다. "둘러봐/집 구경해/파악해봐" 요청에 사용.'),
        'input_schema': {
            'type': 'object',
            'properties': {'duration_s': {
                'type': 'number',
                'description': '탐색 시간(초, 기본 180, 최대 300)'}},
        },
    },
    {
        'name': 'forget_map',
        'description': ('현재 탐색 공간에 대한 학습 기억(장소/길/물체)을 '
                        '전부 지운다 — 처음 보는 공간 상태로 되돌리기. '
                        '사용자가 명시적으로 "기억 지워/까먹어/리셋해"라고 '
                        '할 때만 사용.'),
        'input_schema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'remember_place',
        'description': ('현재 위치를 이름으로 기억한다. 탐색 모드 공간에서 '
                        '사용자가 "여기가 ~야"라고 알려줄 때 사용.'),
        'input_schema': {
            'type': 'object',
            'properties': {'name': {'type': 'string'}},
            'required': ['name'],
        },
    },
    {
        'name': 'stop',
        'description': '즉시 정지(속도 0). 사용자가 멈추라고 하면 최우선 호출.',
        'input_schema': {'type': 'object', 'properties': {}},
    },
]

# LLM에 노출할 툴 = 레지스트리에서 현재 타깃에 활성인 것만 (A-0)
TOOLS = [t for t in ALL_TOOLS if t['name'] in features.enabled_tools()]


class AgentBrain:
    def __init__(self, executor, on_event=None):
        self.client = anthropic.Anthropic()
        self.executor = executor          # ToolExecutor
        self.on_event = on_event or (lambda kind, text: None)
        self.history = []                 # anthropic messages

    # ---------- 히스토리 관리 ----------
    def _strip_old_images(self, keep_recent: bool = True):
        """이미지 블록을 플레이스홀더로 치환해 토큰 절약 + 낡은 프레임
        재사용 방지. keep_recent=False면 전부 치환 — 새 사용자 턴 시작
        시 사용해, 시각 질문마다 반드시 새 look을 강제한다 (묘사 품질
        이슈: 히스토리의 옛 프레임으로 답하는 것을 차단)."""
        seen_recent = not keep_recent
        for msg in reversed(self.history):
            if not isinstance(msg.get('content'), list):
                continue
            for block in msg['content']:
                if isinstance(block, dict) and block.get('type') == 'tool_result':
                    content = block.get('content')
                    if isinstance(content, list) and any(
                            c.get('type') == 'image' for c in content):
                        if seen_recent:
                            block['content'] = ('[이전 카메라 프레임 — '
                                                '생략됨. 현재가 궁금하면 '
                                                'look을 다시 호출]')
                        else:
                            seen_recent = True

    def _trim(self):
        # user 메시지 기준으로 오래된 턴 제거
        user_idx = [i for i, m in enumerate(self.history)
                    if m['role'] == 'user' and not (
                        isinstance(m.get('content'), list)
                        and m['content']
                        and isinstance(m['content'][0], dict)
                        and m['content'][0].get('type') == 'tool_result')]
        if len(user_idx) > MAX_HISTORY_TURNS:
            cut = user_idx[len(user_idx) - MAX_HISTORY_TURNS]
            self.history = self.history[cut:]

    # ---------- 툴 실행 ----------
    def _run_tool(self, name, args):
        self.on_event('tool', f'{name}({args})')
        if not features.is_enabled(name):
            # 노출 목록에서 이미 빠져 있지만, 히스토리 잔재 등으로
            # 호출될 경우를 대비한 2차 방어선
            return {'error': f'{name}은(는) 현재 로봇 구성에서 비활성'}, True
        fn = getattr(self.executor, name, None)
        if fn is None:
            return {'error': f'unknown tool {name}'}, True
        try:
            result = fn(**args)
        except TypeError as e:
            return {'error': f'bad arguments: {e}'}, True
        except Exception as e:  # 툴 내부 예외도 대화로 풀 수 있게
            return {'error': str(e)}, True
        return result, 'error' in result

    def _tool_result_block(self, tool_use_id, result, is_error):
        if 'image_b64' in result:
            content = [{
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/jpeg',
                    'data': result['image_b64'],
                },
            }]
        else:
            content = str(result.get('ok') or result.get('error') or result)
        block = {
            'type': 'tool_result',
            'tool_use_id': tool_use_id,
            'content': content,
        }
        if is_error:
            block['is_error'] = True
        return block

    # ---------- 메인 ----------
    def chat(self, user_text: str) -> str:
        # C-①: 현재 상태를 유저 턴에 자동 주입 (시스템 프리픽스는 고정
        # 유지 → 프롬프트 캐시 보호. 상태를 시스템에 넣으면 매 턴 캐시가
        # 깨진다). 덕분에 단순 상태 질문은 get_status 왕복 없이 즉답.
        # 새 턴 시작: 히스토리의 모든 프레임 무효화 → 시각 질문은
        # 반드시 신선한 look을 거치게 됨
        self._strip_old_images(keep_recent=False)
        try:
            state = self.executor.status_snapshot()
            state_txt = ('<robot_state>'
                         + json.dumps(state, ensure_ascii=False)
                         + '</robot_state>')
        except Exception as e:
            state_txt = f'<robot_state>조회 실패: {e}</robot_state>'
        self.history.append({'role': 'user', 'content': [
            {'type': 'text', 'text': state_txt},
            {'type': 'text', 'text': user_text},
        ]})
        final_text = ''
        last_text = ''      # 툴콜과 같은 응답에 담긴 텍스트도 보존
        for _ in range(8):  # 툴 루프 상한
            resp = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{'type': 'text', 'text': SYSTEM,
                         'cache_control': {'type': 'ephemeral'}}],
                tools=TOOLS,
                messages=self.history,
            )
            # I: LLM usage를 감사 로그에 기록 (비용 대시보드 기초 데이터)
            u = resp.usage
            self.executor.audit('llm', {'model': MODEL}, True, json.dumps({
                'in': u.input_tokens, 'out': u.output_tokens,
                'cache_read': getattr(u, 'cache_read_input_tokens', 0),
                'cache_write': getattr(u, 'cache_creation_input_tokens', 0),
            }))
            content_dump = []
            for block in resp.content:
                if block.type == 'text':
                    content_dump.append({'type': 'text', 'text': block.text})
                elif block.type == 'tool_use':
                    content_dump.append({
                        'type': 'tool_use', 'id': block.id,
                        'name': block.name, 'input': block.input})
            self.history.append({'role': 'assistant', 'content': content_dump})

            turn_text = '\n'.join(
                b.text for b in resp.content if b.type == 'text').strip()
            if turn_text:
                last_text = turn_text

            if resp.stop_reason != 'tool_use':
                final_text = turn_text or last_text
                break

            results = []
            for block in resp.content:
                if block.type != 'tool_use':
                    continue
                result, is_err = self._run_tool(block.name, dict(block.input))
                results.append(
                    self._tool_result_block(block.id, result, is_err))
            self.history.append({'role': 'user', 'content': results})
            self._strip_old_images()
        else:
            final_text = last_text or '(툴 호출이 너무 길어져 중단했어요)'

        self._trim()
        return final_text or last_text or '(응답 없음)'
