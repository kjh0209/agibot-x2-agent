# X2 개발 환경 변수 (컨테이너 안에서 source)
#   source /scripts/env_x2.sh
source /opt/ros/humble/setup.bash
# 프리빌트 aimdk_msgs (aarch64) — 실기와 동일한 메시지 타입
_P=/sdk/aimdk-1.0.0/src/aimdk_msgs/prebuilt_aarch64
export AMENT_PREFIX_PATH=$_P:$AMENT_PREFIX_PATH
export LD_LIBRARY_PATH=$_P/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$_P/local/lib/python3.10/dist-packages:$PYTHONPATH
# 워크스페이스 오버레이
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export MUJOCO_GL=osmesa
