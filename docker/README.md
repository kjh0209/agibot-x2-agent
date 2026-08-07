# 개발 컨테이너 (Ubuntu 22.04 + ROS 2 Humble)

Apple Silicon Mac에서는 ROS 2를 네이티브로 못 돌리므로 Docker 컨테이너를 쓴다.
전부 무료 도구만 사용한다 (Homebrew + Colima + Docker CLI).

## 1회 설치 (터미널에서 직접 실행 필요 — 관리자 암호 입력)

```bash
# 1) Homebrew 설치 (이미 있으면 생략)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2) Colima(무료 컨테이너 런타임) + Docker CLI + compose
brew install colima docker docker-compose

# 3) 컨테이너 VM 시작 — Rosetta 가속 필수!
#    (공식 시뮬 바이너리가 x86-64 전용이라 amd64 컨테이너를 돌린다)
colima start --cpu 4 --memory 8 --vm-type vz --vz-rosetta
```

## 컨테이너 빌드 및 진입

```bash
cd docker
docker compose build        # 최초 1회 (수 분 소요)
docker compose up -d        # 백그라운드 기동
docker exec -it agibot-dev bash   # 개발 셸 진입
```

## 완료 기준 확인 (Phase 0)

컨테이너 안에서:

```bash
ros2 doctor
python3 -c "import mujoco, onnxruntime, anthropic; print('deps ok')"
```

## 주의

- 이 컨테이너는 시뮬 구간(Phase 0~4) 전용. 실기 연결(Phase 5~)에서는
  DDS 멀티캐스트/카메라 대역폭 문제를 피하기 위해 네이티브 Ubuntu 22.04
  PC를 권장 (로드맵 Phase 5 참고).
- MuJoCo는 GUI 없이 offscreen(OSMesa) 렌더로 사용한다. Phase 2 카메라
  브릿지는 원래 offscreen 렌더 → 이미지 토픽 퍼블리시 구조라 지장 없음.
  시뮬 화면을 눈으로 보고 싶을 때는 브라우저 뷰어(web_video_server의
  MJPEG, 포트 8080)로 확인한다.
