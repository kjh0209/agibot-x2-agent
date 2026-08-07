#!/usr/bin/env python3
"""로봇 1인칭 카메라 MJPEG 웹 뷰어 (Phase 2 확인용).

/aima/hal/sensor/rgbd_head_front/rgb_image/compressed 를 구독해
http://localhost:8080 에서 브라우저로 스트리밍한다.
  /            간단한 뷰어 페이지
  /stream      multipart MJPEG 스트림
  /snapshot.jpg 최신 프레임 1장
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import Trigger

from aimdk_msgs.msg import McLocomotionVelocity

TOPIC = '/x2_sim/preview/compressed'   # 라이브 프리뷰(저해상 고빈도)
PORT = 8080

INDEX_HTML = """<!doctype html>
<meta charset="utf-8">
<title>X2 로봇 시점</title>
<style>
 body{background:#111;color:#eee;font-family:sans-serif;text-align:center}
 img{max-width:95%;border:2px solid #444;border-radius:8px;margin-top:12px}
 code{background:#222;padding:2px 6px;border-radius:4px}
 button{border:0;border-radius:8px;padding:10px 22px;font-size:15px;
        cursor:pointer;margin-top:14px;color:#fff}
 .reset{background:#c0392b}.reset:hover{background:#e74c3c}
 .stop{background:#f39c12;font-weight:bold;font-size:17px;
       padding:12px 30px;margin-right:14px}
 .stop:hover{background:#e67e22}
 #st{margin-left:12px;color:#8f8}
</style>
<h2>Agibot X2 — 로봇이 보는 화면 (시뮬)</h2>
<img src="/stream" alt="robot view">
<div>
 <button class="stop" onclick="doStop()">🛑 즉시 정지</button>
 <button class="reset" onclick="doReset()">🔄 시뮬 초기화 (처음 위치로)</button>
 <span id="st"></span>
</div>
<p style="color:#888;font-size:12px">즉시 정지는 LLM·에이전트를 거치지 않는
독립 경로입니다 (채팅의 "멈춰"는 별도의 두 번째 경로)</p>
<script>
function doStop(){
  const st=document.getElementById('st');
  fetch('/stop',{method:'POST'})
    .then(r=>r.text()).then(t=>{st.textContent=t;
      setTimeout(()=>st.textContent='',3000);})
    .catch(e=>{st.textContent='실패: '+e;});
}
function doReset(){
  const st=document.getElementById('st');
  st.textContent='초기화 중...';
  fetch('/reset',{method:'POST'})
    .then(r=>r.text()).then(t=>{st.textContent=t;
      setTimeout(()=>st.textContent='',4000);})
    .catch(e=>{st.textContent='실패: '+e;});
}
</script>
"""


class FrameStore:
    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._n = 0

    def put(self, data: bytes):
        with self._cond:
            self._frame = data
            self._n += 1
            self._cond.notify_all()

    def get(self, last_n):
        with self._cond:
            self._cond.wait_for(lambda: self._n != last_n, timeout=2.0)
            return self._frame, self._n


store = FrameStore()


class ViewerNode(Node):
    def __init__(self):
        super().__init__('mjpeg_server')
        self.create_subscription(
            CompressedImage, TOPIC,
            lambda m: store.put(bytes(m.data)), qos_profile_sensor_data)
        self.reset_cli = self.create_client(Trigger, '/x2_sim/reset')
        # 하드와이어 정지 경로 (LLM·에이전트 큐와 완전 독립):
        #  ① rc 소스로 velocity 0 직접 발행 (시뮬/실기 즉시 감속)
        #  ② /agent/hard_stop 으로 에이전트 move 재발행 루프 인터럽트
        self.pub_vel = self.create_publisher(
            McLocomotionVelocity, '/aima/mc/locomotion/velocity', 10)
        self.pub_hard_stop = self.create_publisher(
            String, '/agent/hard_stop', 10)
        self.get_logger().info(f'구독: {TOPIC} → http://0.0.0.0:{PORT}')

    def hard_stop(self):
        vel = McLocomotionVelocity()
        vel.source = 'rc'          # 기본 등록 소스 — 게이트 통과 보장
        self.pub_vel.publish(vel)  # 속도 0
        self.pub_hard_stop.publish(String(data='ui_button'))
        self.get_logger().warning('하드와이어 정지 발행 (UI 버튼)')
        return True, '정지 발행 완료 (LLM 독립 경로)'

    def call_reset(self, timeout=5.0):
        if not self.reset_cli.wait_for_service(timeout_sec=2.0):
            return False, '시뮬 노드가 응답하지 않아요 (sim_node 실행 중인지 확인)'
        fut = self.reset_cli.call_async(Trigger.Request())
        deadline = time.monotonic() + timeout
        while not fut.done():
            if time.monotonic() > deadline:
                return False, '초기화 응답 시간 초과'
            time.sleep(0.05)
        r = fut.result()
        return r.success, r.message


viewer_node = None  # main()에서 주입


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 요청 로그 소음 제거
        pass

    def do_POST(self):
        if self.path == '/stop':
            ok, msg = viewer_node.hard_stop()
            body = msg.encode()
            self.send_response(200 if ok else 500)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/reset':
            ok, msg = viewer_node.call_reset()
            body = msg.encode()
            self.send_response(200 if ok else 500)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == '/':
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == '/snapshot.jpg':
            frame, _ = store.get(-1)
            if frame is None:
                self.send_error(503, 'no frame yet')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header(
                'Content-Type',
                'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            n = -1
            try:
                while True:
                    frame, n = store.get(n)
                    if frame is None:
                        continue
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(
                        f'Content-Length: {len(frame)}\r\n\r\n'.encode())
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main():
    global viewer_node
    rclpy.init()
    node = ViewerNode()
    viewer_node = node
    server = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
