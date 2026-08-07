#!/bin/bash
# Phase 1: 컨테이너 안에서 x2_rl_deploy 빌드 준비 (1회)
# 사용법(컨테이너 내부):  bash /sdk/../scripts/setup_sim.sh  또는 경로 마운트에 맞게 실행
set -e

ONNX_VER=1.23.2
ONNX_TGZ=onnxruntime-linux-x64-${ONNX_VER}.tgz

echo "== 1) RL deploy 의존성 설치 =="
apt-get update
apt-get install -y --no-install-recommends \
    libgflags-dev libyaml-cpp-dev \
    ros-humble-joy ros-humble-grid-map-msgs ros-humble-grid-map-ros

echo "== 2) onnxruntime ${ONNX_VER} (x64) 설치 =="
if [ ! -f /usr/local/include/onnxruntime/onnxruntime_c_api.h ]; then
    cd /tmp
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/${ONNX_TGZ}"
    tar -zxf "${ONNX_TGZ}"
    mkdir -p /usr/local/include/onnxruntime /usr/local/lib64
    cp -r onnxruntime-linux-x64-${ONNX_VER}/include/* /usr/local/include/onnxruntime/
    cp -r onnxruntime-linux-x64-${ONNX_VER}/lib/* /usr/local/lib/
    cp -r onnxruntime-linux-x64-${ONNX_VER}/lib/* /usr/local/lib64/
    ldconfig
    rm -rf /tmp/onnxruntime-linux-x64-${ONNX_VER} /tmp/${ONNX_TGZ}
else
    echo "이미 설치됨 — 건너뜀"
fi

echo "== 3) x2_rl_deploy colcon 빌드 =="
source /opt/ros/humble/setup.bash
cd /sdk/aimdk-1.0.0/extra/x2_rl_deploy
colcon build --event-handlers console_cohesion+
echo "빌드 완료. 시뮬 실행:"
echo "  source /sdk/aimdk-1.0.0/extra/x2_rl_deploy/install/local_setup.bash"
echo "  cd /sdk/aimdk-1.0.0/extra/x2_rl_deploy/x2_rl_deploy_mujoco/bin && ./start_sim.sh -s"
