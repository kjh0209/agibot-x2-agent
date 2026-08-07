#!/usr/bin/env python3
"""Anthropic API 키 동작 확인 (Phase 0 완료 기준용).

.env의 ANTHROPIC_API_KEY로 Haiku에 최소 호출 1회 — 비용 1원 미만.
사용법:  python3 scripts/check_api_key.py
"""
import os
import sys
from pathlib import Path

# .env 로드 (별도 패키지 없이)
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

if not os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-"):
    sys.exit("ANTHROPIC_API_KEY가 없습니다. .env.example을 .env로 복사해 키를 넣으세요.")

try:
    import anthropic
except ImportError:
    sys.exit("anthropic 패키지가 없습니다:  pip3 install anthropic")

client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=16,
    messages=[{"role": "user", "content": "OK라고만 답해"}],
)
text = next((b.text for b in resp.content if b.type == "text"), "")
print(f"응답: {text!r}")
print(f"토큰: 입력 {resp.usage.input_tokens} / 출력 {resp.usage.output_tokens}")
print("API 키 정상 동작 ✓")
