#!/usr/bin/env python3
"""로봇 1인칭 카메라 MJPEG 웹 뷰어 (Phase 2 확인용).

/aima/hal/sensor/rgbd_head_front/rgb_image/compressed 를 구독해
http://localhost:8080 에서 브라우저로 스트리밍한다.
  /            간단한 뷰어 페이지
  /stream      multipart MJPEG 스트림
  /snapshot.jpg 최신 프레임 1장
"""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

TOPIC = '/aima/hal/sensor/rgbd_head_front/rgb_image/compressed'
PORT = 8080

INDEX_HTML = """<!doctype html>
<meta charset="utf-8">
<title>X2 로봇 시점</title>
<style>
 body{background:#111;color:#eee;font-family:sans-serif;text-align:center}
 img{max-width:95%;border:2px solid #444;border-radius:8px;margin-top:12px}
 code{background:#222;padding:2px 6px;border-radius:4px}
</style>
<h2>Agibot X2 — 로봇이 보는 화면 (시뮬)</h2>
<img src="/stream" alt="robot view">
<p>이동 테스트 (다른 터미널에서):</p>
<p><code>docker exec -it agibot-dev bash -c "source /scripts/env_x2.sh &&
ros2 topic pub --once /aima/mc/locomotion/velocity
aimdk_msgs/msg/McLocomotionVelocity
'{source: agent, forward_velocity: 0.3}'"</code></p>
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
        self.get_logger().info(f'구독: {TOPIC} → http://0.0.0.0:{PORT}')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 요청 로그 소음 제거
        pass

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
    rclpy.init()
    node = ViewerNode()
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
