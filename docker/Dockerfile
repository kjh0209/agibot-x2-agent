# Agibot X2 개발 컨테이너 — Ubuntu 22.04 (jammy) + ROS 2 Humble
# Apple Silicon Mac에서 arm64로 동작. AimDK RL Deployment 문서의
# Prerequisites(Ubuntu 22.04 / ROS 2 Humble / rmw_cyclonedds_cpp)를 따른다.
FROM ros:humble-ros-base-jammy

# 필수 시스템 패키지 + ROS 2 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    python3-pip \
    python3-colcon-common-extensions \
    git wget curl unzip vim \
    libgl1 libglfw3 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
#  - mujoco: 시뮬 + Phase 2 offscreen 렌더 (EGL/OSMesa 헤드리스)
#  - onnxruntime: 시뮬 locomotion 예제 구동
#  - anthropic: Phase 3 에이전트 노드
RUN pip3 install --no-cache-dir \
    mujoco \
    onnxruntime \
    anthropic \
    fastapi "uvicorn[standard]" websockets pillow

# 한글 입출력을 위한 UTF-8 로케일 (미설정 시 IME 깨진 바이트가 서로게이트로 유입)
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONIOENCODING=utf-8

# 기본 DDS를 실기와 동일하게 CycloneDDS로
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# 헤드리스 MuJoCo 렌더 (GUI 없는 컨테이너)
ENV MUJOCO_GL=osmesa

# ROS 환경 자동 소싱
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> /root/.bashrc

WORKDIR /ws
CMD ["bash"]
