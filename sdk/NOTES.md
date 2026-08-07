# SDK 확보 기록 (Phase 0)

## 다운로드 (2026-08-07)

| 항목 | 파일 | 출처 |
|---|---|---|
| AimDK SDK v1.0.0 (`aimdk v1.0.0-ga424add`) | `downloads/aimdk-aarch64-a424add7-artifacts.zip` (157MB) | https://x2-aimdk.agibot.com/downloads/aimdk-aarch64-a424add7-artifacts.zip |
| X2 URDF v1.3.0 | `downloads/X2_URDF-v1.3.0.zip` (48MB) | 문서 사이트 get_sdk 페이지 `_downloads/` 링크 |

전체 버전 목록: https://x2-aimdk.agibot.com/sdk-downloads.json
(v1.0.0 / v0.9.0 / v0.8.2 / v0.8.1)

압축 해제 위치: `sdk/aimdk-1.0.0/` (git 제외 — zip이 원본)

## SDK 구조

```
aimdk-1.0.0/
  src/aimdk_msgs      # 메시지/서비스 정의
  src/examples        # C++ 예제
  src/py_examples     # Python 예제 (Phase 3 에이전트의 참고 코드)
  extra/x2_rl_deploy/ # RL 배포 예제 + MuJoCo 시뮬레이터
    x2_rl_deploy_mujoco/bin/start_sim.sh   # 시뮬 실행 (로드맵 Phase 1)
  topics_and_services # 전체 토픽/서비스 목록
```

## 핵심 확인 사항

- **시뮬 바이너리(aima-sim-app)는 x86-64 리눅스 전용** (클로즈드소스).
  Apple Silicon에서는 amd64 컨테이너 + Rosetta로 실행 (docker/README.md).
- **펌웨어 업/다운그레이드 시 로봇 디스크 리포맷** (SDK README 원문).
  `$HOME`(/agibot/data/home/agi)만 보존되고 `$HOME/aimdk*`는 시스템 예약.
  v1.0 펌웨어 플래싱 시 `$HOME` 전체가 `.old_agi_home.bak`으로 이동됨.
- 개발은 반드시 PC2(10.0.1.41)에서. **PC1(10.0.1.40)에서 빌드/실행 금지**
  (공식 문서 명시). 유선 개발 시 호스트 PC는 10.0.1.2/24 고정 IP.
- onnxruntime v1.23.2 (x64 tgz)를 /usr/local에 수동 설치 필요 (Phase 1 빌드 시).

## Phase 3에서 쓸 주요 인터페이스 (topics_and_services 발췌)

- 카메라: `/aima/hal/sensor/rgbd_head_front/rgb_image[/compressed]`,
  `stereo_head_front_{left,right}`, `rgb_head_rear`
- 상태: `/aima/hal/joint/*/state`, `/aima/hal/imu/{chest,torso}/state`,
  `/aima/hal/pmu/state` (배터리)
- 이동: `/aima/mc/locomotion/velocity`
- 행동: `SetMcAction` / `GetMcAction` / `SetMcPresetMotion` /
  `SetMcInputSource` / `GetCurrentInputSource`
- 발화/표정: `PlayTts` / `PlayEmoji` / `PlayAudioFile` / `SetVolume`
- 오디오 입력: `/aima/hal/audio/capture` (Phase 8 STT용)
