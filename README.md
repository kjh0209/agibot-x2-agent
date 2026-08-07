# Agibot X2 메신저 에이전트 시스템

로봇(Agibot X2 Ultra)의 시점 영상과 상태를 맥락으로 갖고, 자체 채팅 앱에서
사람과 직접 대화하며 툴 콜로 행동하는 에이전트 시스템.
시뮬레이션(MuJoCo)에서 완성한 코드를 설정 전환만으로 실기에 연결하는 것이 목표.

- 전체 계획: [`agibot_x2_메신저_시스템_로드맵.md`](./agibot_x2_메신저_시스템_로드맵.md)
- 비용 계획: [`agibot_x2_단계별_비용_정리.txt`](./agibot_x2_단계별_비용_정리.txt)
- SDK 문서: https://x2-aimdk.agibot.com

## 저장소 구조 (예정)

```
ros2_ws/src/
  x2_sim/         # 자체 MuJoCo 시뮬 (공식 MJCF·토픽·메시지, 카메라, MC 목,
                  #   충돌 차단, PMU 목, 리셋 서비스) + MJPEG 뷰어(:8080)
  robot_agent/    # Claude 두뇌 (툴 실행기·안전 가드·감사 로그) + chat_cli
chat/             # 자체 채팅 앱 — FastAPI+WS 백엔드 / 무빌드 프론트 (:8000)
scripts/          # env_x2.sh(환경), setup_sim.sh(x86용), chaos_mc.sh(카오스)
deploy/           # Phase 8: systemd 유닛, requirements.lock
docs/             # 실기 준비 체크리스트/네트워크/견적 초안/데모 각본
docker/           # Ubuntu 22.04 + ROS 2 Humble 개발 컨테이너 (arm64)
sdk/              # AimDK SDK/URDF (downloads/·해제본은 git 제외)
```

## 개발 환경

개발 머신이 Apple Silicon Mac이므로 ROS 2 Humble은 Docker 컨테이너
(Ubuntu 22.04 arm64)에서 구동한다. `docker/README.md` 참고.

주의: 실기(X2 Ultra) 연결 단계에서는 네이티브 Ubuntu 22.04 PC 권장
(DDS 멀티캐스트/카메라 대역폭 이슈 회피).

## 비밀키

`.env` 파일에 보관하며 git에 커밋하지 않는다 (`.gitignore` 처리됨).

```
ANTHROPIC_API_KEY=sk-ant-...
```
