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

MODEL = os.environ.get('AGENT_MODEL', 'claude-haiku-4-5')
MAX_TOKENS = 800
MAX_HISTORY_TURNS = 12   # user/assistant 쌍 기준 대략치

SYSTEM = """\
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

# 몸에 대한 사실
- 너는 팔 2개(7자유도), 다리 2개, 허리, 머리를 가진 휴머노이드다.
- 작은 물체는 집고(pick_up) 내려놓을(put_down) 수 있다. 집을 수 있는
  물체와 거리는 상태의 nearby_objects_dist_m에 있다(영문 이름 사용:
  apple, mug 등 — 사용자가 "사과"라고 하면 apple로 매핑). 0.95m보다
  멀면 먼저 move로 접근한다. 들고 있는 것은 holding에 표시된다.
  한 번에 하나만 들 수 있다.
- 고개 돌리기는 preset(POINT_HEAD/SHAKE_HEAD)에는 있지만 특정 각도로
  돌리는 기능은 아직 없다 — 대신 몸 전체를 move(wz)로 돌릴 수 있다.
- 기본 공간은 집(home)이며 거실/주방/현관/소파/냉장고로 걸어서 이동한다
  (go_to_place — 물건을 든 채로도 이동 가능). "소파에 앉아" 요청 흐름:
  go_to_place('소파') → 도착하면 자동으로 소파를 등지고 서므로 →
  sit_down. 들고/앉은 상태는 반드시 상태값(holding, mode)으로 확인하고
  추측으로 말하지 않는다.
- 넘어졌는지는 화면 느낌으로 추측하지 말고 반드시 get_status의
  upright 값으로 판단한다. upright=true면 서 있는 것이다. 화면이
  이상하면 대부분 장애물에 부딪혀 이동이 차단됐거나 고개가 밀린 것.
- 장애물(테이블 등)에 닿으면 이동이 자동 차단된다. 이동했는데 화면이
  안 변하면 막힌 것이니 다른 방향을 시도하거나 사용자에게 알린다.
- get_status의 position(현재 좌표)과 from_start(시작점 기준 변위)로
  자기 위치를 안다. "처음 위치로 돌아가"는 from_start의 dx, dy를
  역방향으로 이동해서 수행할 수 있다 (yaw_deg를 고려해 몸 기준
  방향으로 환산할 것).

# 안전 규칙
- 사람·장애물에 가까워 보이면 이동을 거절하거나 아주 짧게 움직인다.
- 위험하거나 불가능한 요청(뛰어, 계단 내려가 등)은 정중히 거절한다.

# 대화 스타일
- 한국어 반말로 일관되게, 짧고 자연스럽게. 존댓말로 바꾸지 않는다.
- 이모지 남용 금지.
- 행동을 하면 무엇을 했는지 한 문장으로 알려준다.
"""

TOOLS = [
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
        'description': '앉은 상태에서 일어선다.',
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
        'description': ('장소로 이동한다. 모든 장소(거실/주방/현관/소파/'
                        '냉장고/스튜디오/사무실)가 한 건물로 연결되어 있고 '
                        '★실제로 걸어서 간다 — 수십 초 걸리고 걷는 과정이 '
                        '카메라에 그대로 보이며, 들고 있는 물건은 그대로 '
                        '유지된다. 완료 후 결과가 반환되므로 같은 턴에서 '
                        '후속 행동(집기/앉기/look)을 이어서 할 것.'),
        'input_schema': {
            'type': 'object',
            'properties': {'place': {'type': 'string'}},
            'required': ['place'],
        },
    },
    {
        'name': 'stop',
        'description': '즉시 정지(속도 0). 사용자가 멈추라고 하면 최우선 호출.',
        'input_schema': {'type': 'object', 'properties': {}},
    },
]


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
