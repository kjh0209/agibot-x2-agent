#!/usr/bin/env python3
"""단일 인스턴스 강제 (A-4).

배경: pkill 패턴 매칭 실패로 시뮬 3중 실행, SIGTERM을 무시한 좀비
백엔드의 이중 저장 등 '중복 프로세스' 사고가 반복됨. 시작 시점에
센티널 포트를 점유해 두 번째 인스턴스가 뜨는 것 자체를 막는다.
(PID 파일 방식과 달리 프로세스가 죽으면 OS가 포트를 자동 회수 —
스테일 락 문제 없음)
"""
import socket
import sys

# 서비스별 센티널 포트 (127.0.0.1 전용 — 외부 노출 없음)
PORTS = {
    'x2_sim': 19011,
    'robot_agent': 19012,
    'chat_backend': 19014,
    'mjpeg_server': 19015,
}


def claim(name: str) -> socket.socket:
    """센티널 포트 점유. 실패(=이미 실행 중)면 명시적으로 종료.

    반환된 소켓은 호출측이 프로세스 수명 동안 참조를 유지해야 한다."""
    port = PORTS[name]
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('127.0.0.1', port))
        s.listen(1)
    except OSError:
        print(f'[{name}] 이미 실행 중인 인스턴스가 있음 '
              f'(센티널 포트 {port} 점유됨) — 중복 실행 차단, 종료합니다. '
              f'기존 프로세스를 먼저 종료하세요.', file=sys.stderr)
        sys.exit(1)
    return s
