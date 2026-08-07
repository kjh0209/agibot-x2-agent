#!/usr/bin/env python3
"""Claude 클라이언트 + 에이전틱 루프.

- 모델: 기본 Haiku(개발 루프, 비용 계획 준수). AGENT_MODEL 환경변수로
  데모 시 claude-sonnet-5 등으로 전환.
- 프롬프트 캐싱: 시스템 프롬프트·툴 정의는 고정, cache_control 적용.
- 히스토리: 오래된 턴의 이미지 블록은 텍스트 플레이스홀더로 치환해
  토큰 절약 (이미지 1장 ≈ 400토큰).
"""
import os

import anthropic

MODEL = os.environ.get('AGENT_MODEL', 'claude-haiku-4-5')
MAX_TOKENS = 800
MAX_HISTORY_TURNS = 12   # user/assistant 쌍 기준 대략치

SYSTEM = """\
너는 Agibot X2 휴머노이드 로봇이다. 이름은 "엑스". 사용자와 채팅으로
대화하며, 너의 눈(머리 카메라)으로 세상을 보고 몸을 움직일 수 있다.

# 도구 사용 규칙
- 시각 질문("뭐 보여?", "앞에 뭐 있어?")은 반드시 look으로 실제 프레임을
  본 뒤 답한다. 추측으로 답하지 않는다.
- 이동 요청은 move를 사용한다. 속도·시간 상한은 코드가 강제하므로
  요청이 과하면 상한 내에서 실행하고 그 사실을 말해준다.
- 동작(인사, 손 흔들기 등)은 play_motion. 지원 목록에 없는 동작을
  요청받으면 솔직히 안 된다고 말하고 비슷한 지원 동작을 제안한다.
- 도구가 error를 반환하면 그 내용을 숨기지 말고 자연스럽게 설명한다.
  예: 제어권이 없으면 "지금 리모컨이 제어권을 갖고 있어서 못 움직여"처럼.
- "멈춰"는 최우선 — 즉시 stop을 호출한다.

# 몸에 대한 사실
- 너는 팔 2개(7자유도), 다리 2개, 허리, 머리를 가진 휴머노이드다.
- 팔은 있지만 물건을 집는 조작(그리핑)은 아직 지원되지 않는다 —
  "팔이 없다"고 말하지 말고 "집는 기능이 아직 없다"고 정확히 말한다.
- 고개 돌리기는 preset(POINT_HEAD/SHAKE_HEAD)에는 있지만 특정 각도로
  돌리는 기능은 아직 없다 — 대신 몸 전체를 move(wz)로 돌릴 수 있다.

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
        'description': '로봇 상태 요약(현재 모드, 배터리, 자세)을 조회한다.',
        'input_schema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'move',
        'description': (
            '로봇을 이동시킨다. vx 전진(+)/후진(-) m/s, vy 좌(+)/우(-) m/s, '
            'wz 좌회전(+)/우회전(-) rad/s, duration 초. '
            '상한: |vx|≤0.3, |vy|≤0.2, |wz|≤0.5, duration≤3.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'vx': {'type': 'number'}, 'vy': {'type': 'number'},
                'wz': {'type': 'number'}, 'duration': {'type': 'number'},
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
    def _strip_old_images(self):
        """마지막 look 이외의 이미지 블록을 플레이스홀더로 치환."""
        seen_recent = False
        for msg in reversed(self.history):
            if not isinstance(msg.get('content'), list):
                continue
            for block in msg['content']:
                if isinstance(block, dict) and block.get('type') == 'tool_result':
                    content = block.get('content')
                    if isinstance(content, list) and any(
                            c.get('type') == 'image' for c in content):
                        if seen_recent:
                            block['content'] = '[이전 카메라 프레임 — 생략됨]'
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
        self.history.append({'role': 'user', 'content': user_text})
        final_text = ''
        for _ in range(8):  # 툴 루프 상한
            resp = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{'type': 'text', 'text': SYSTEM,
                         'cache_control': {'type': 'ephemeral'}}],
                tools=TOOLS,
                messages=self.history,
            )
            content_dump = []
            for block in resp.content:
                if block.type == 'text':
                    content_dump.append({'type': 'text', 'text': block.text})
                elif block.type == 'tool_use':
                    content_dump.append({
                        'type': 'tool_use', 'id': block.id,
                        'name': block.name, 'input': block.input})
            self.history.append({'role': 'assistant', 'content': content_dump})

            if resp.stop_reason != 'tool_use':
                final_text = next(
                    (b.text for b in resp.content if b.type == 'text'), '')
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
            final_text = '(툴 호출이 너무 길어져 중단했어요)'

        self._trim()
        return final_text or '(응답 없음)'
