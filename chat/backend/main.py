#!/usr/bin/env python3
"""Phase 4: 자체 채팅 앱 백엔드 (FastAPI + rclpy 브릿지).

- WS /ws          채팅 양방향 (사용자 메시지 ↔ 답변/툴 이벤트/음성/표정/상태)
- GET /stream     로봇 1인칭 MJPEG
- GET /history    저장된 대화 이력 (SQLite)
- POST /stop      ★하드와이어 정지 — LLM·에이전트 큐와 완전 독립 경로
- POST /reset     시뮬 초기화
- /               프론트엔드 정적 파일

실행(컨테이너): source /scripts/env_x2.sh && python3 /chat/backend/main.py
"""
import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

import rclpy
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Imu
from std_msgs.msg import String
from std_srvs.srv import Trigger

from aimdk_msgs.msg import McLocomotionVelocity, PmuState
from x2_sim.worlds import WORLDS

FRONT_DIR = Path(__file__).resolve().parent.parent / 'frontend'
DB_PATH = Path('/ws/log/chat_history.db')

BATT_FULL_V, BATT_EMPTY_V = 54.6, 39.0   # 매뉴얼: 충전기 54.6V / 방전한계 <39V


# ---------------------------------------------------------------- DB
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, kind TEXT, text TEXT)''')
    return conn


def save_msg(kind, text):
    with db() as conn:
        conn.execute('INSERT INTO messages(ts, kind, text) VALUES(?,?,?)',
                     (time.time(), kind, text))


# ---------------------------------------------------------------- ROS
class Bridge(Node):
    def __init__(self):
        super().__init__('chat_backend')
        self.frame = None
        self.frame_n = 0
        self.frame_cond = threading.Condition()
        self.state = {'mode': '?', 'connected': False}
        self.subscribers = set()    # 연결별 asyncio.Queue (브로드캐스트)
        self.loop = None

        self.pub_user = self.create_publisher(String, '/agent/user_msg', 10)
        self.pub_vel = self.create_publisher(
            McLocomotionVelocity, '/aima/mc/locomotion/velocity', 10)
        self.pub_hard_stop = self.create_publisher(
            String, '/agent/hard_stop', 10)
        self.reset_cli = self.create_client(Trigger, '/x2_sim/reset')
        # 월드 전환 (latched 구독으로 현재 월드 추적)
        from rclpy.qos import DurabilityPolicy, QoSProfile
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub_set_world = self.create_publisher(
            String, '/x2_sim/set_world', 10)
        self.create_subscription(
            String, '/x2_sim/world',
            lambda m: self.state.__setitem__('world', m.data), latched)

        self.create_subscription(
            CompressedImage,
            '/aima/hal/sensor/rgbd_head_front/rgb_image/compressed',
            self.on_frame, qos_profile_sensor_data)
        for topic, kind in (('/agent/reply', 'reply'),
                            ('/agent/event', 'event'),
                            ('/agent/say', 'say'),
                            ('/agent/emoji', 'emoji')):
            self.create_subscription(
                String, topic,
                lambda m, k=kind: self.emit(k, m.data), 10)
        self.create_subscription(
            Odometry, '/slam/lidar_odom', self.on_odom,
            qos_profile_sensor_data)
        self.create_subscription(
            Imu, '/aima/hal/imu/chest/state', self.on_imu,
            qos_profile_sensor_data)
        self.create_subscription(
            PmuState, '/aima/hal/pmu/state', self.on_pmu,
            qos_profile_sensor_data)
        # 모드는 실기와 동일하게 GetMcAction 서비스 폴링 (sim/real 패리티)
        from aimdk_msgs.srv import GetMcAction
        self.mode_cli = self.create_client(
            GetMcAction, '/aimdk_5Fmsgs/srv/GetMcAction')
        self._GetMcAction = GetMcAction
        threading.Thread(target=self._poll_mode, daemon=True).start()

    # ---- 수신 콜백 → 프론트로 릴레이 ----
    def emit(self, kind, text):
        if kind in ('reply', 'say', 'emoji'):
            save_msg(kind, text)
        if self.loop:
            item = {'kind': kind, 'text': text}
            for q in list(self.subscribers):   # 모든 연결에 브로드캐스트
                self.loop.call_soon_threadsafe(q.put_nowait, item)

    def on_frame(self, msg):
        with self.frame_cond:
            self.frame = bytes(msg.data)
            self.frame_n += 1
            self.frame_cond.notify_all()

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.state['position'] = {'x': round(p.x, 2), 'y': round(p.y, 2)}

    def on_imu(self, msg):
        import math
        q = msg.orientation
        roll = math.degrees(math.atan2(2 * (q.w * q.x + q.y * q.z),
                                       1 - 2 * (q.x * q.x + q.y * q.y)))
        pitch = math.degrees(math.asin(
            max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))))
        self.state['upright'] = abs(roll) < 30 and abs(pitch) < 30

    def on_pmu(self, msg):
        v = msg.battery_voltage
        pct = max(0.0, min(1.0, (v - BATT_EMPTY_V) / (BATT_FULL_V - BATT_EMPTY_V)))
        self.state['battery_pct'] = round(pct * 100)

    def _poll_mode(self):
        from x2_sim.mc_services import ACTION_NAMES
        while True:
            try:
                if self.mode_cli.wait_for_service(timeout_sec=1.0):
                    fut = self.mode_cli.call_async(
                        self._GetMcAction.Request())
                    t_end = time.monotonic() + 2
                    while not fut.done() and time.monotonic() < t_end:
                        time.sleep(0.05)
                    if fut.done():
                        v = int(fut.result().info.current_action.value)
                        self.state['mode'] = ACTION_NAMES.get(v, str(v))
                        self.state['connected'] = True
                else:
                    self.state['connected'] = False
            except Exception:
                self.state['connected'] = False
            time.sleep(1.0)

    # ---- 프론트 → 로봇 ----
    def send_user_msg(self, text):
        save_msg('user', text)
        self.pub_user.publish(String(data=text))

    def hard_stop(self):
        vel = McLocomotionVelocity()
        vel.source = 'rc'
        self.pub_vel.publish(vel)
        self.pub_hard_stop.publish(String(data='chat_ui_button'))
        self.get_logger().warning('하드와이어 정지 (채팅 UI 버튼)')

    def call_reset(self):
        if not self.reset_cli.wait_for_service(timeout_sec=2.0):
            return False, '시뮬 노드 응답 없음'
        fut = self.reset_cli.call_async(Trigger.Request())
        t_end = time.monotonic() + 5
        while not fut.done():
            if time.monotonic() > t_end:
                return False, '시간 초과'
            time.sleep(0.05)
        r = fut.result()
        return r.success, r.message


# ---------------------------------------------------------------- app
bridge: Bridge = None
app = FastAPI()


@app.on_event('startup')
async def startup():
    bridge.loop = asyncio.get_event_loop()


@app.get('/')
async def index():
    return FileResponse(FRONT_DIR / 'index.html')


@app.get('/history')
async def history():
    with db() as conn:
        rows = conn.execute(
            'SELECT ts, kind, text FROM messages ORDER BY id DESC LIMIT 200'
        ).fetchall()
    return [{'ts': r[0], 'kind': r[1], 'text': r[2]} for r in reversed(rows)]


@app.get('/status')
async def status():
    return bridge.state


@app.post('/stop')
async def stop():
    bridge.hard_stop()
    return Response('정지 발행 완료 (LLM 독립 경로)', media_type='text/plain')


@app.get('/worlds')
async def worlds():
    return {
        'current': bridge.state.get('world'),
        'worlds': {name: w['desc'] for name, w in WORLDS.items()},
    }


@app.post('/world')
async def set_world(payload: dict):
    name = str(payload.get('name', ''))
    if name not in WORLDS:
        return Response(f'미등록 월드: {name}', status_code=400,
                        media_type='text/plain')
    bridge.pub_set_world.publish(String(data=name))
    return Response(f'월드 전환 요청: {name}', media_type='text/plain')


@app.post('/reset')
async def reset():
    ok, msg = await asyncio.to_thread(bridge.call_reset)
    return Response(msg, media_type='text/plain',
                    status_code=200 if ok else 500)


@app.get('/stream')
async def stream():
    def gen():
        n = -1
        while True:
            with bridge.frame_cond:
                bridge.frame_cond.wait_for(
                    lambda: bridge.frame_n != n, timeout=2.0)
                frame, n = bridge.frame, bridge.frame_n
            if frame is None:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n'
                   + f'Content-Length: {len(frame)}\r\n\r\n'.encode()
                   + frame + b'\r\n')
    return StreamingResponse(
        gen(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.websocket('/ws')
async def ws(sock: WebSocket):
    await sock.accept()
    inbox = asyncio.Queue()
    bridge.subscribers.add(inbox)

    async def pump_out():
        while True:
            item = await inbox.get()
            await sock.send_text(json.dumps(item, ensure_ascii=False))

    async def pump_status():
        while True:
            await sock.send_text(json.dumps(
                {'kind': 'status', 'state': bridge.state},
                ensure_ascii=False))
            await asyncio.sleep(1.0)

    out_task = asyncio.create_task(pump_out())
    st_task = asyncio.create_task(pump_status())
    try:
        while True:
            raw = await sock.receive_text()
            data = json.loads(raw)
            if data.get('kind') == 'user':
                text = str(data.get('text', '')).strip()
                if text:
                    bridge.send_user_msg(text)
    except WebSocketDisconnect:
        pass
    finally:
        bridge.subscribers.discard(inbox)
        out_task.cancel()
        st_task.cancel()


app.mount('/static', StaticFiles(directory=FRONT_DIR), name='static')


def main():
    global bridge
    rclpy.init()
    bridge = Bridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(bridge)
    threading.Thread(target=executor.spin, daemon=True).start()
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='warning')


if __name__ == '__main__':
    main()
