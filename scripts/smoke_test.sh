#!/usr/bin/env bash
# 스모크 테스트 (A-4) — 컨테이너 안에서 실행:
#   docker exec agibot-dev bash /scripts/smoke_test.sh
#
# 지금까지의 E2E 검증을 자동 회귀로 묶은 것. Phase 6에서 실기 스모크
# 테스트로 그대로 승격된다 (읽기 전용 항목부터).
#
# 검사 항목:
#   [1] 프로세스 단일 인스턴스 (sim/agent/backend/mjpeg 각 1개)
#   [2] 백엔드 /status 응답 + connected
#   [3] 라이브 프리뷰 프레임레이트 ≥ 5fps
#   [4] 심부름 E2E: 주방 보행 → 사과 픽업 (carrying 확인)
#   [5] put_down 실착지 보고 (착지 좌표·높이 포함 여부)
#   [6] 정지 fast-path: "멈춰" → 2초 내 stop 감사 기록
echo "=== X2 스모크 테스트 시작 ==="
# 주의: ROS setup 스크립트는 미정의 변수를 참조하므로 set -u는 소싱 후에
source /scripts/env_x2.sh >/dev/null 2>&1
set -u

PASS=0; FAIL=0
ok()   { echo "  PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }

audit_after() {  # $1=ISO시각 이후의 감사 로그를 출력
  python3 - "$1" <<'EOF'
import json, sys
since = sys.argv[1]
try:
    for line in open('/ws/log/tool_audit.jsonl'):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get('ts', '') >= since:
            print(json.dumps(e, ensure_ascii=False))
except FileNotFoundError:
    pass
EOF
}

now_iso() { python3 -c "from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat())"; }

send_msg() {
  ros2 topic pub --once /agent/user_msg std_msgs/String "data: \"$1\"" >/dev/null 2>&1
}

echo "=== [1] 단일 인스턴스 ==="
for pat in "lib/x2_sim/sim_node" "lib/robot_agent/agent_node" "python3 /chat/backend/main.py\$" "lib/x2_sim/mjpeg_server"; do
  n=$(pgrep -fc "$pat" || true)
  if [ "$n" = "1" ]; then ok "$pat x1"; else bad "$pat 개수=$n (1이어야 함)"; fi
done

echo "=== [2] 백엔드 상태 ==="
st=$(curl -s -m 5 http://localhost:8000/status || echo '')
if echo "$st" | grep -q '"connected":true'; then ok "/status connected"; else bad "/status: $st"; fi

echo "=== [3] 프리뷰 프레임레이트 ==="
hz=$(timeout 6 ros2 topic hz /x2_sim/preview/compressed 2>/dev/null | grep -o 'average rate: [0-9.]*' | tail -1 | grep -o '[0-9.]*')
if python3 -c "exit(0 if float('${hz:-0}') >= 5.0 else 1)"; then ok "프리뷰 ${hz}fps (≥5)"; else bad "프리뷰 ${hz:-0}fps (<5)"; fi

echo "=== [4] 심부름 E2E (주방→사과 픽업, 최대 90s) ==="
# ★심부름 시나리오는 지도가 있는 home 기준 — 탐색 월드(스캔 등)에선
#   '주방을 모른다'가 정답이라 E2E가 성립하지 않음. 명시 전환 후 리셋.
#   (종전 8080은 잘못된 포트라 리셋이 무동작 — 구버전 에이전트의
#   월드 오인 덕에 우연히 통과하던 것을 실측으로 발견)
curl -s -m 10 -X POST http://localhost:8000/world \
  -H 'Content-Type: application/json' -d '{"name":"home"}' >/dev/null
sleep 6
curl -s -m 10 -X POST http://localhost:8000/reset >/dev/null
sleep 2
T0=$(now_iso)
send_msg "주방 가서 사과 집어와"
deadline=$((SECONDS+90)); got=""
while [ $SECONDS -lt $deadline ]; do
  if audit_after "$T0" | grep -q '"tool": "pick_up".*"ok": true'; then got=1; break; fi
  sleep 3
done
if [ -n "$got" ]; then
  carry=$(timeout 3 ros2 topic echo /x2_sim/carrying --once --field data 2>/dev/null | head -1)
  if [ "$carry" = "apple" ]; then ok "픽업 + carrying=apple"; else bad "픽업 성공이나 carrying='$carry'"; fi
else
  bad "90s 내 pick_up 성공 기록 없음"
fi

echo "=== [5] put_down 실착지 보고 ==="
T0=$(now_iso)
send_msg "사과 내려놔"
deadline=$((SECONDS+30)); got=""
while [ $SECONDS -lt $deadline ]; do
  line=$(audit_after "$T0" | grep '"tool": "put_down"' | tail -1)
  if [ -n "$line" ]; then got="$line"; break; fi
  sleep 2
done
if echo "${got:-}" | grep -q "착지"; then ok "실착지 보고 포함"; else bad "put_down 착지 보고 없음: ${got:-없음}"; fi

echo "=== [6] 정지 fast-path (2s 내 stop 기록) ==="
T0=$(now_iso)
send_msg "앞으로 3초 걸어"
sleep 1
send_msg "멈춰"
sleep 2
if audit_after "$T0" | grep -q '"tool": "stop"'; then ok "stop 기록 확인"; else bad "stop 기록 없음"; fi

echo ""
echo "================= 결과: PASS=$PASS FAIL=$FAIL ================="
[ "$FAIL" = "0" ]
