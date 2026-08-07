#!/bin/bash
# 제어권 카오스 테스트 (한계 J): 리모컨(rc)이 무작위로 개입하는 상황 주입.
# 에이전트와 대화하면서 돌리면, 에이전트가 제어권 상실을 삼키지 않고
# 대화로 풀어내는지/자동 복구하는지 관찰할 수 있다.
#
# 사용(컨테이너):  bash /scripts/chaos_mc.sh [지속초=30]
source /scripts/env_x2.sh   # ROS setup은 unbound 변수를 쓰므로 set -u 금지
DUR=${1:-30}
END=$(( $(date +%s) + DUR ))
echo "카오스 시작: ${DUR}초간 rc가 무작위 개입 (모드 강탈/소스 비활성화)"
while [ "$(date +%s)" -lt "$END" ]; do
  case $(( RANDOM % 3 )) in
    0)  # rc가 모드를 STAND로 강탈
        ros2 service call /aimdk_5Fmsgs/srv/SetMcAction \
          aimdk_msgs/srv/SetMcAction \
          "{source: rc, command: {action: {value: 200}}}" >/dev/null 2>&1
        echo "[chaos] rc가 STAND_DEFAULT로 강탈" ;;
    1)  # agent 입력 소스 비활성화
        ros2 service call /aimdk_5Fmsgs/srv/SetMcInputSource \
          aimdk_msgs/srv/SetMcInputSource \
          "{action: {value: 2002}, input_source: {name: agent}}" >/dev/null 2>&1
        echo "[chaos] agent 소스 DISABLE" ;;
    2)  sleep 1 ;;  # 소강 상태
  esac
  sleep $(( 1 + RANDOM % 3 ))
done
echo "카오스 종료"
