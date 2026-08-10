#!/usr/bin/env bash
# 시뮬 수퍼바이저 — 세그폴트 등 비정상 종료 시 자동 재기동.
# (OSMesa 동시 렌더 세그폴트 2회 실측 후 도입. 렌더 직렬화로 원인을
#  막았지만, 남은 미지의 크래시에도 데모가 끊기지 않게 하는 안전망.)
# 사용: docker exec -d agibot-dev bash /scripts/supervise_sim.sh
source /scripts/env_x2.sh
while true; do
  ros2 run x2_sim sim_node >> /tmp/sim.log 2>&1
  code=$?
  echo "[supervisor] sim_node 종료(exit $code) — 2s 후 재기동" >> /tmp/sim.log
  sleep 2
done
