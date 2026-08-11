#!/usr/bin/env python3
"""점유 격자 SLAM 맵핑 — 인지 계층의 parity 구현.

배경: 종전에는 시뮬이 월드 기하에서 차단 구역을 뽑아 '정답 지도'를
발행했다 — 실기에서 SLAM이 해야 할 일을 시뮬에선 특권 정보 열람으로
때우는 구조라, 인지 계층의 sim/real 괴리가 남아 있었다.

이 모듈은 로봇 스스로 센서로 지도를 만든다:
  입력: /x2_sim/lidar_scan — {"o":[x,y], "rays":[[az,h,dist,hit],...]}
        (시뮬: mj_ray 시뮬레이션(sim_only). 실기: 라이다/깊이 어댑터가
         동일 스키마로 발행 — 이 파일은 무수정으로 동작)
  알고리즘: 로그오즈 점유 격자 (히트 +1.2 / 최저 높이 빔 경로 -0.4)
        — 표준 occupancy grid mapping. 문이 열리면 빔이 통과해 해당
        셀이 자유로 감쇠(동적 환경 자동 반영). 셀별 최고 히트 높이를
        추적해 낮은 가구(앉는 면)/키 큰 벽을 구분.
  출력: /slam/costmap — 기존 코스트맵과 동일한 구역 JSON
        [[cx,cy,hx,hy,margin,is_door,top],...] (에이전트 A*/접근/착석
        로직이 무수정으로 소비). 월드별 저장/복원(/ws/log/slam/).

포즈는 /slam/lidar_odom(실기: 온보드 SLAM 오돔)과 동일 값이 스캔
메시지 origin에 실려 온다 — '알려진 포즈 하의 맵핑'으로, 포즈 추정
자체는 실기 온보드 SLAM의 몫(시뮬은 오돔이 그 역할).
"""
import json
import math
import os

import numpy as np

SLAM_DIR = '/ws/log/slam'

RES = 0.10            # 격자 해상도 (m)
HALF = 15.0           # 커버리지 반경 (30x30m)
L_HIT = 1.2           # 히트 로그오즈 증분
L_FREE = 0.4          # 자유 감쇠 (최저 높이 빔만)
L_MIN, L_MAX = -4.0, 6.0
OCC_TH = 2.0          # 점유 판정 문턱 — 히트 2회 이상 (단발 노이즈 배제)
MARGIN_TALL = 0.15
MARGIN_LOW = 0.10


class SlamMapper:
    """점유 격자 유지 + 구역 병합 발행. 노드에 기생(스레드/프로세스
    추가 없음) — 코드 자체는 입력 토픽만 있으면 어디서든 동작."""

    def __init__(self, node, latched_qos, sensor_qos):
        from std_msgs.msg import String
        self.node = node
        self.N = int(2 * HALF / RES)
        self.logodds = np.zeros((self.N, self.N), np.float32)
        self.top = np.zeros((self.N, self.N), np.float32)
        self.world = None
        self._dirty = False
        self._scan_count = 0
        os.makedirs(SLAM_DIR, exist_ok=True)
        self.pub = node.create_publisher(String, '/slam/costmap',
                                         latched_qos)
        node.create_subscription(String, '/x2_sim/lidar_scan',
                                 self.on_scan, sensor_qos)
        node.create_timer(1.0, self._tick)
        self.zones = []

    # ---------- 월드 전환/저장 ----------
    def set_world(self, world):
        if world == self.world:
            return
        if self.world is not None:
            self.save()
        self.world = world
        self.logodds[:] = 0.0
        self.top[:] = 0.0
        self.zones = []
        path = self._path(world)
        if os.path.exists(path):
            try:
                with open(path) as fp:
                    d = json.load(fp)
                for i, j, lo, tz in d.get('cells', []):
                    if 0 <= i < self.N and 0 <= j < self.N:
                        self.logodds[i, j] = lo
                        self.top[i, j] = tz
                self.node.get_logger().info(
                    f'SLAM 지도 복원: {world} ({len(d.get("cells", []))}셀)')
            except Exception as e:
                self.node.get_logger().warning(f'SLAM 지도 복원 실패: {e}')
        self._dirty = True

    def _path(self, world):
        return os.path.join(SLAM_DIR, f'{world}.json')

    def save(self):
        if self.world is None:
            return
        occ = np.abs(self.logodds) > 0.2
        idx = np.argwhere(occ)
        cells = [[int(i), int(j), round(float(self.logodds[i, j]), 2),
                  round(float(self.top[i, j]), 2)] for i, j in idx]
        try:
            with open(self._path(self.world), 'w') as fp:
                json.dump({'res': RES, 'cells': cells}, fp)
        except Exception as e:
            self.node.get_logger().warning(f'SLAM 지도 저장 실패: {e}')

    def forget(self, world=None):
        w = world or self.world
        try:
            os.remove(self._path(w))
        except FileNotFoundError:
            pass
        if w == self.world:
            self.logodds[:] = 0.0
            self.top[:] = 0.0
            self.zones = []
            self._publish()

    # ---------- 스캔 반영 ----------
    def _cell(self, x, y):
        return (int((x + HALF) / RES), int((y + HALF) / RES))

    def on_scan(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:
            return
        # 최신 스캔 보관 — 주행 안전 반사층(_obstacle_ahead)이 사용
        import time as _t
        self.last_scan = d
        self.last_scan_t = _t.monotonic()
        ox, oy = d['o'][0], d['o'][1]
        h_min = min((r[1] for r in d.get('rays', [])), default=0.0)
        for az, h, dist, hit in d.get('rays', []):
            ca, sa = math.cos(az), math.sin(az)
            # 자유 감쇠는 최저 높이 빔만 — 높은 빔이 소파 위를 지나며
            # 소파 셀을 자유로 지우는 것 방지
            if h <= h_min + 1e-6 and dist > 0.3:
                n = int((dist - 0.15) / RES)
                if n > 0:
                    ts = np.linspace(0.3, dist - 0.15, n)
                    xs = ((ox + ts * ca + HALF) / RES).astype(int)
                    ys = ((oy + ts * sa + HALF) / RES).astype(int)
                    ok = (xs >= 0) & (xs < self.N) & (ys >= 0) & \
                         (ys < self.N)
                    self.logodds[xs[ok], ys[ok]] = np.maximum(
                        L_MIN, self.logodds[xs[ok], ys[ok]] - L_FREE)
            if hit and dist >= 0.2:
                # 0.2m 미만 히트 제외 — 라이다 최소 유효 거리(실기 스펙)
                # + 자기 위치 셀이 점유로 오염되는 것 방지
                i, j = self._cell(ox + dist * ca, oy + dist * sa)
                if 0 <= i < self.N and 0 <= j < self.N:
                    self.logodds[i, j] = min(
                        L_MAX, self.logodds[i, j] + L_HIT)
                    if self.top[i, j] < h:
                        self.top[i, j] = h + 0.05
        self._scan_count += 1
        self._dirty = True

    # ---------- 완전 탐사(survey) 지원 ----------
    def frontiers(self):
        """프론티어 셀 좌표 목록 — 자유 셀 중 미지 셀과 인접한 것.

        완전 탐사의 수렴 판정: 도달 가능한 프론티어가 없으면 로봇이
        갈 수 있는 모든 공간이 관측 완료된 것."""
        free = self.logodds < -0.5
        unk = np.abs(self.logodds) < 0.3
        adj = np.zeros_like(unk)
        adj[1:, :] |= unk[:-1, :]
        adj[:-1, :] |= unk[1:, :]
        adj[:, 1:] |= unk[:, :-1]
        adj[:, :-1] |= unk[:, 1:]
        idx = np.argwhere(free & adj)
        return [(float(i * RES - HALF + RES / 2),
                 float(j * RES - HALF + RES / 2)) for i, j in idx]

    def free_cells(self, sub=4):
        """알려진 자유 셀 좌표 (sub칸 간격 서브샘플) — 스캔 커버리지
        판정용."""
        free = self.logodds < -0.5
        idx = np.argwhere(free)
        out = []
        for i, j in idx:
            if i % sub or j % sub:
                continue
            out.append((float(i * RES - HALF + RES / 2),
                        float(j * RES - HALF + RES / 2)))
        return out

    def known_area_m2(self):
        known = (np.abs(self.logodds) > 0.3).sum()
        return float(known) * RES * RES

    # ---------- 구역 병합·발행 ----------
    def _tick(self):
        if not self._dirty:
            return
        self._dirty = False
        self._rebuild()
        self._publish()
        if self._scan_count % 60 == 0 and self._scan_count:
            self.save()

    def _rebuild(self):
        occ = self.logodds > OCC_TH
        # 자유 감쇠로 점유가 풀린 셀은 top도 초기화
        self.top[self.logodds < -0.5] = 0.0
        doors = [(v.get('x', 0), v.get('y', 0))
                 for v in (getattr(self.node, 'doors_info', {}) or
                           {}).values()]
        tall = self.top >= 0.5
        # 행 런 → 세로 동일 런 병합 (변환기와 동일 접근) — 구역 수를
        # 수백 → 수십으로 줄여 A* 비용을 억제
        runs = {}
        for j in range(self.N):
            col = occ[:, j]
            if not col.any():
                continue
            i = 0
            while i < self.N:
                if not col[i]:
                    i += 1
                    continue
                k = i
                t = bool(tall[i, j])
                zt = float(self.top[i, j])
                while k + 1 < self.N and col[k + 1] and \
                        bool(tall[k + 1, j]) == t:
                    k += 1
                    zt = max(zt, float(self.top[k, j]))
                runs.setdefault((i, k, t), []).append((j, zt))
                i = k + 1
        zones = []
        for (i0, i1, t), cols in runs.items():
            cols.sort()
            s = 0
            while s < len(cols):
                e = s
                zt = cols[s][1]
                while e + 1 < len(cols) and \
                        cols[e + 1][0] == cols[e][0] + 1:
                    e += 1
                    zt = max(zt, cols[e][1])
                j0, j1 = cols[s][0], cols[e][0]
                cx = (i0 + i1 + 1) / 2 * RES - HALF
                cy = (j0 + j1 + 1) / 2 * RES - HALF
                hx = (i1 - i0 + 1) / 2 * RES
                hy = (j1 - j0 + 1) / 2 * RES
                margin = MARGIN_TALL if t else MARGIN_LOW
                is_door = 0
                for dx_, dy_ in doors:
                    if abs(cx - dx_) < 0.7 and abs(cy - dy_) < 0.7:
                        is_door = 1
                        break
                zones.append([round(cx, 2), round(cy, 2), round(hx, 2),
                              round(hy, 2), margin, is_door,
                              round(zt, 2)])
                s = e + 1
        self.zones = zones

    def _publish(self):
        from std_msgs.msg import String
        self.pub.publish(String(data=json.dumps(self.zones)))
