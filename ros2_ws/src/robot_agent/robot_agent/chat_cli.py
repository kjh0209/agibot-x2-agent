#!/usr/bin/env python3
"""터미널 채팅 CLI — 에이전트 노드와 대화한다.

사용:
  docker exec -it agibot-dev bash -c \
    "source /scripts/env_x2.sh && ros2 run robot_agent chat_cli"
"""
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

CYAN, YELLOW, GREEN, DIM, RESET = (
    '\033[96m', '\033[93m', '\033[92m', '\033[2m', '\033[0m')


class ChatCli(Node):
    def __init__(self):
        super().__init__('chat_cli')
        self.pub = self.create_publisher(String, '/agent/user_msg', 10)
        self.done = threading.Event()
        self.create_subscription(String, '/agent/reply', self.on_reply, 10)
        self.create_subscription(String, '/agent/event', self.on_event, 10)
        self.create_subscription(String, '/agent/say', self.on_say, 10)
        self.create_subscription(String, '/agent/emoji', self.on_emoji, 10)

    def on_reply(self, msg):
        print(f'\n{CYAN}엑스{RESET}  {msg.data}')
        self.done.set()

    def on_event(self, msg):
        print(f'{DIM}  {msg.data}{RESET}')

    def on_say(self, msg):
        print(f'{GREEN}🔊 (음성){RESET} {msg.data}')

    def on_emoji(self, msg):
        print(f'{GREEN}😀 (표정){RESET} {msg.data}')


def main():
    # 터미널 IME가 조합/백스페이스 중 깨진 바이트를 흘려도 input()이
    # 죽지 않도록 stdin/stdout을 관대한 디코딩으로 재구성 (근본 방어)
    try:
        sys.stdin.reconfigure(errors='replace')
        sys.stdout.reconfigure(errors='replace')
    except (AttributeError, ValueError):
        pass
    rclpy.init()
    node = ChatCli()
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print(f'{YELLOW}=== Agibot X2 채팅 (종료: Ctrl+C 또는 /q) ==={RESET}')
    print(f'{DIM}예: 안녕 / 지금 뭐 보여? / 앞으로 조금 가봐 / 인사해봐 / 멈춰{RESET}')
    try:
        while True:
            try:
                text = input(f'\n{YELLOW}나{RESET}  ')
            except EOFError:
                break
            if text.strip() in ('/q', '/quit', 'exit'):
                break
            # 살균: 서로게이트 제거 + 치환문자(�) 제거
            text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
            text = text.replace('�', '').strip()
            if not text:
                continue
            node.done.clear()
            try:
                node.pub.publish(String(data=text))
            except (UnicodeEncodeError, ValueError) as e:
                print(f'{DIM}(입력 인코딩 문제로 전송 실패 — 다시 입력해주세요: {e}){RESET}')
                continue
            # 답변 올 때까지 대기 (툴 실행 이벤트는 그 사이에 출력됨)
            if not node.done.wait(timeout=120):
                print(f'{DIM}(응답 시간 초과 — 에이전트 노드가 떠 있는지 확인){RESET}')
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        print('\n채팅 종료')


if __name__ == '__main__':
    main()
