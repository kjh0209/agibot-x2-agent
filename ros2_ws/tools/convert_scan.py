#!/usr/bin/env python3
"""사진측량 스캔(GLB) → MuJoCo 월드 변환기 (범용).

어떤 실내 스캔이든 동일 파이프라인으로 처리한다 — 씬별 하드코딩 없음:
  1. 상하축 자동 감지 (수평면 법선 면적 투표) → Z-up 회전
  2. 바닥 높이 자동 감지 (상향면 z 히스토그램 최저 피크) → z=0 정렬, XY 중심화
  3. 트라이앵글 예산 초과 시 UV 보존 데시메이션 (fast_simplification
     collapse 리플레이로 원본 정점→축약 정점 매핑을 얻어 UV 이전)
  4. 파트별 OBJ(v/vt/f) + 텍스처 PNG(≤max_tex) 내보내기
  5. 점유 구역 자동 추출: 로봇 몸통 높이 슬라이스(0.12~1.6m)의
     삼각형을 2D 격자에 래스터화 → 행 단위 런 병합 → 차단 박스 목록
     (기존 기하 게이팅·코스트맵 A*가 그대로 소비하는 형식)
  6. 자유 공간(바닥 있고 장애물 침식 여유 확보) → 스폰 지점 + 소품
     배치 후보 목록
  7. meta.json 출력: {parts, zones, spawn, free, bbox}

실기 전환 관점: 이 전부는 시뮬 시각·충돌 계층(sim_only)이다. 실기에선
스캔 메시 자리에 실제 집이, 점유 구역 자리에 SLAM 코스트맵이 들어간다.
에이전트 계약(코스트맵 JSON/툴/토픽)은 변하지 않는다.
"""
import json
import math
import os
import sys

import numpy as np
import trimesh
from PIL import Image

try:
    from fast_simplification import simplify
    HAVE_SIMPLIFY = True
except Exception:
    HAVE_SIMPLIFY = False


def load_parts(glb_path):
    """씬 그래프 변환을 구운(baked) (이름, Trimesh) 목록."""
    scene = trimesh.load(glb_path, force='scene')
    parts = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname].copy()
        g.apply_transform(T)
        parts.append((gname, g))
    return parts


def detect_up_axis(parts):
    """수평면(바닥/천장) 법선이 정렬된 축 = 수직축. 면적 가중 투표."""
    score = np.zeros(3)
    for _, g in parts:
        n = g.face_normals
        a = g.area_faces
        for ax in range(3):
            m = np.abs(n[:, ax]) > 0.95
            score[ax] += float(a[m].sum())
    return int(np.argmax(score))


def rotation_to_z_up(axis):
    if axis == 2:
        return np.eye(4)
    R = np.eye(4)
    if axis == 1:      # Y-up → Z-up: (x,y,z)→(x,-z,y)
        R[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    else:              # X-up → Z-up
        R[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
    return R


def detect_floor_z(parts):
    """상향면 중심 z의 면적 가중 히스토그램에서 최저 유의 피크."""
    zs, ws = [], []
    for _, g in parts:
        up = g.face_normals[:, 2] > 0.85
        if not up.any():
            continue
        zs.append(g.triangles_center[up][:, 2])
        ws.append(g.area_faces[up])
    z = np.concatenate(zs)
    w = np.concatenate(ws)
    lo, hi = float(z.min()), float(z.max())
    nb = max(8, int((hi - lo) / 0.05))
    hist, edges = np.histogram(z, bins=nb, range=(lo, hi), weights=w)
    peak = hist.max()
    for i in range(len(hist)):        # 최저 높이부터: 유의(피크 20%↑) 첫 빈
        if hist[i] >= 0.2 * peak:
            return float((edges[i] + edges[i + 1]) / 2)
    return lo


def decimate_with_uv(g, target_tris):
    """UV 보존 데시메이션 — 복셀 정점 클러스터링.

    quadric 계열(fast_simplification 직접 산출·collapse 리플레이 둘 다)은
    사진측량 메시(UV 솔기로 위상이 잘게 끊긴 삼각형 수프)에서 삼각형
    파쇄를 일으켰다 — 실측 2회. 정점 클러스터링은 '정점을 복셀 단위로
    병합'만 하므로 구조적으로 파쇄가 불가능하고, UV는 클러스터 대표
    정점(첫 등장) 것을 그대로 쓴다. 복셀 크기를 키워가며 예산에 맞춘다."""
    v = np.asarray(g.vertices, np.float64)
    f = np.asarray(g.faces, np.int64)
    uv = np.asarray(g.visual.uv, np.float32) if g.visual.uv is not None \
        else None
    voxel = 0.015
    for _ in range(30):
        key = np.floor(v / voxel).astype(np.int64)
        if uv is not None:
            # UV 근접성도 키에 포함 — 아틀라스 차트(솔기) 경계를 넘는
            # 병합을 금지해 텍스처 줄무늬 오염 방지 (병합 후 UV 오차
            # ≤ 아틀라스의 ~4%로 유계)
            key = np.concatenate(
                [key, np.floor(uv / 0.04).astype(np.int64)], axis=1)
        _, firsts, inv = np.unique(key, axis=0, return_index=True,
                                   return_inverse=True)
        f2 = inv[f]
        good = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & \
               (f2[:, 0] != f2[:, 2])
        f2 = f2[good]
        # 중복 면 제거 — 정렬 키로 판별하되 원본 행을 유지해 와인딩
        # (앞뒷면 방향) 보존
        keys = np.sort(f2, axis=1)
        _, uidx = np.unique(keys, axis=0, return_index=True)
        f2 = f2[np.sort(uidx)]
        if len(f2) <= target_tris:
            break
        voxel *= 1.25
    v2 = v[firsts].astype(np.float32)
    uv2 = uv[firsts] if uv is not None else None
    return v2, f2, uv2


def export_obj(path, v, f, uv):
    with open(path, 'w') as fp:
        for p in v:
            fp.write(f'v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n')
        if uv is not None:
            for t in uv:
                fp.write(f'vt {t[0]:.6f} {t[1]:.6f}\n')
            for tri in f:
                a, b, c = tri + 1
                fp.write(f'f {a}/{a} {b}/{b} {c}/{c}\n')
        else:
            for tri in f:
                a, b, c = tri + 1
                fp.write(f'f {a} {b} {c}\n')


def part_texture(g):
    m = getattr(g.visual, 'material', None)
    img = getattr(m, 'baseColorTexture', None)
    if img is None:
        img = getattr(m, 'image', None)
    rgba = None
    fac = getattr(m, 'baseColorFactor', None)
    if fac is not None:
        fac = np.asarray(fac, float)
        rgba = (fac / 255.0 if fac.max() > 1.0 else fac).tolist()
    return img, rgba


def build_occupancy(parts, cell, slice_lo=0.12, slice_hi=1.60):
    """몸통 높이 슬라이스 삼각형 → 2D 점유 격자 + 자유(바닥) 격자
    + ★좌석면 격자 (0.15~0.60m 상향 수평면 — 등받이 의자 대응).

    셀 최고 높이만으로 '낮은 가구'를 찾으면 등받이/셸이 있는 의자는
    전부 키 큰 구역으로 분류돼 착석 감지에서 탈락한다 (회의 의자
    실측). 앉을 수 있음의 물리적 정의는 '앉는 높이의 상향 수평면이
    존재함'이므로 별도 격자로 추출한다."""
    allv = np.concatenate([np.asarray(g.vertices) for _, g in parts])
    xy_min = allv[:, :2].min(axis=0) - cell
    xy_max = allv[:, :2].max(axis=0) + cell
    nx = int(np.ceil((xy_max[0] - xy_min[0]) / cell))
    ny = int(np.ceil((xy_max[1] - xy_min[1]) / cell))
    occ = np.zeros((nx, ny), bool)
    top = np.zeros((nx, ny), float)
    floor = np.zeros((nx, ny), bool)
    seat = np.zeros((nx, ny), float)

    def cell_range(lo, hi):
        i0 = max(0, int((lo - xy_min[0]) / cell))
        i1 = min(nx - 1, int((hi - xy_min[0]) / cell))
        return i0, i1

    for _, g in parts:
        tris = g.triangles
        zmin = tris[:, :, 2].min(axis=1)
        zmax = tris[:, :, 2].max(axis=1)
        nz = g.face_normals[:, 2]
        blk = (zmax > slice_lo) & (zmin < slice_hi) & \
              ~((nz > 0.7) & (zmax < 0.30))       # 낮은 상향면=바닥/단차 제외
        flr = (nz > 0.7) & (zmin < 0.25)
        zc_tri = tris[:, :, 2].mean(axis=1)
        st = (nz > 0.8) & (zc_tri >= 0.15) & (zc_tri <= 0.60)
        for ti in np.nonzero(st)[0]:
            t = tris[ti]
            i0 = max(0, int((t[:, 0].min() - xy_min[0]) / cell))
            i1 = min(nx - 1, int((t[:, 0].max() - xy_min[0]) / cell))
            j0 = max(0, int((t[:, 1].min() - xy_min[1]) / cell))
            j1 = min(ny - 1, int((t[:, 1].max() - xy_min[1]) / cell))
            zt = float(zc_tri[ti])
            sub = seat[i0:i1 + 1, j0:j1 + 1]
            np.maximum(sub, zt, out=sub)
        for sel, grid, keep_top in ((blk, occ, True), (flr, floor, False)):
            idx = np.nonzero(sel)[0]
            for ti in idx:
                t = tris[ti]
                i0 = max(0, int((t[:, 0].min() - xy_min[0]) / cell))
                i1 = min(nx - 1, int((t[:, 0].max() - xy_min[0]) / cell))
                j0 = max(0, int((t[:, 1].min() - xy_min[1]) / cell))
                j1 = min(ny - 1, int((t[:, 1].max() - xy_min[1]) / cell))
                if keep_top:
                    # ★차단 격자는 정밀 래스터화 — AABB 전체 마킹은
                    # 문간 옆 벽의 큰 삼각형이 문 개구부까지 차단해
                    # '들어갈 수 있는 방을 못 들어가는' 실측 사고.
                    # 셀 중심이 삼각형 2D 투영에서 0.4셀 이내일 때만.
                    zc = min(2.2, float(zmax[ti]))
                    for ii in range(i0, i1 + 1):
                        for jj in range(j0, j1 + 1):
                            px = xy_min[0] + (ii + 0.5) * cell
                            py = xy_min[1] + (jj + 0.5) * cell
                            if _tri_dist2d(t, px, py) <= 0.4 * cell:
                                grid[ii, jj] = True
                                if top[ii, jj] < zc:
                                    top[ii, jj] = zc
                else:
                    # 바닥(자유 공간 후보)은 과대 마킹이 무해 — AABB 유지
                    grid[i0:i1 + 1, j0:j1 + 1] = True
    return occ, top, floor, seat, xy_min


def merge_seats(seat, xy_min, cell):
    """좌석면 셀 → 높이 버킷(0.1m) 행 런 병합 박스 [[cx,cy,hx,hy,h]]."""
    boxes = []
    nx, ny = seat.shape
    for j in range(ny):
        i = 0
        while i < nx:
            if seat[i, j] <= 0:
                i += 1
                continue
            k = i
            hb = round(seat[i, j], 1)
            hmax = seat[i, j]
            while k + 1 < nx and seat[k + 1, j] > 0 and \
                    round(seat[k + 1, j], 1) == hb:
                k += 1
                hmax = max(hmax, seat[k, j])
            boxes.append([round(xy_min[0] + (i + k + 1) / 2 * cell, 3),
                          round(xy_min[1] + (j + 0.5) * cell, 3),
                          round((k - i + 1) / 2 * cell, 3),
                          round(cell / 2, 3), round(float(hmax), 2)])
            i = k + 1
    # 세로 병합 (동일 x-범위·높이 버킷)
    merged = []
    boxes.sort(key=lambda b: (b[0], b[2], b[1]))
    for b in boxes:
        if merged and merged[-1][0] == b[0] and merged[-1][2] == b[2] \
                and abs((merged[-1][1] + merged[-1][3]) - (b[1] - b[3])) \
                < 1e-6 and abs(merged[-1][4] - b[4]) < 0.1:
            m = merged[-1]
            new_low = m[1] - m[3]
            new_high = b[1] + b[3]
            m[1] = round((new_low + new_high) / 2, 3)
            m[3] = round((new_high - new_low) / 2, 3)
            m[4] = max(m[4], b[4])
        else:
            merged.append(list(b))
    return merged


def _tri_dist2d(t, px, py):
    """점(px,py)에서 삼각형 2D 투영(XY)까지의 거리.

    수직 벽 삼각형은 투영이 선분으로 퇴화 — 내부 판정+변 거리로 양쪽
    모두 처리한다."""
    a, b, c = t[0][:2], t[1][:2], t[2][:2]
    # 내부면 0 (부호 면적 3개의 부호 일치)
    d1 = (px - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (py - b[1])
    d2 = (px - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (py - c[1])
    d3 = (px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])
    if not (((d1 < 0) or (d2 < 0) or (d3 < 0)) and
            ((d1 > 0) or (d2 > 0) or (d3 > 0))):
        return 0.0
    best = float('inf')
    for p, q in ((a, b), (b, c), (c, a)):
        vx, vy = q[0] - p[0], q[1] - p[1]
        L2 = vx * vx + vy * vy
        s = 0.0 if L2 == 0 else max(0.0, min(
            1.0, ((px - p[0]) * vx + (py - p[1]) * vy) / L2))
        ex, ey = p[0] + s * vx - px, p[1] + s * vy - py
        d = math.hypot(ex, ey)
        if d < best:
            best = d
    return best


def merge_zones(occ, top, xy_min, cell):
    """행 단위 런 병합(같은 높이 계급) → 세로 방향 동일 런 병합."""
    tall = top >= 0.5
    runs = {}          # (j0, i0, i1, is_tall) 후보 → 세로 병합
    boxes = []
    nx, ny = occ.shape
    for j in range(ny):
        i = 0
        while i < nx:
            if not occ[i, j]:
                i += 1
                continue
            k = i
            t = tall[i, j]
            zt = top[i, j]
            while k + 1 < nx and occ[k + 1, j] and tall[k + 1, j] == t:
                k += 1
                zt = max(zt, top[k, j])
            runs.setdefault((i, k, bool(t)), []).append((j, zt))
            i = k + 1
    for (i0, i1, t), cols in runs.items():
        cols.sort()
        s = 0
        while s < len(cols):
            e = s
            zt = cols[s][1]
            while e + 1 < len(cols) and cols[e + 1][0] == cols[e][0] + 1:
                e += 1
                zt = max(zt, cols[e][1])
            j0, j1 = cols[s][0], cols[e][0]
            cx = xy_min[0] + (i0 + i1 + 1) / 2 * cell
            cy = xy_min[1] + (j0 + j1 + 1) / 2 * cell
            hx = (i1 - i0 + 1) / 2 * cell
            hy = (j1 - j0 + 1) / 2 * cell
            boxes.append([round(cx, 3), round(cy, 3), round(hx, 3),
                          round(hy, 3), round(zt, 2)])
            s = e + 1
    return boxes


def free_space(occ, floor, xy_min, cell, erode_m=0.35):
    """바닥 있고 장애물에서 침식 여유가 있는 자유 셀 → 스폰/배치 후보."""
    er = max(1, int(round(erode_m / cell)))
    free = floor & ~occ
    nx, ny = occ.shape
    # 침식: 주변 er 셀 내 점유 존재 시 제외 (누적합으로 O(1) 창 검사)
    ip = np.pad(occ.astype(np.int32), er + 1)
    cs = ip.cumsum(0).cumsum(1)
    good = np.zeros_like(free)
    for i in range(nx):
        for j in range(ny):
            if not free[i, j]:
                continue
            a, b = i + 1, j + 1                      # 패딩 보정
            s = cs[a + 2 * er + 1, b + 2 * er + 1] - \
                cs[a, b + 2 * er + 1] - cs[a + 2 * er + 1, b] + cs[a, b]
            good[i, j] = (s == 0)
    pts = np.argwhere(good)
    if len(pts) == 0:
        pts = np.argwhere(free)
    coords = xy_min + (pts + 0.5) * cell
    center = coords.mean(axis=0)
    spawn = coords[np.argmin(((coords - center) ** 2).sum(axis=1))]
    # farthest-point 샘플링으로 배치 후보 8곳 분산 추출
    chosen = [spawn]
    for _ in range(8):
        d = np.min([((coords - c) ** 2).sum(axis=1) for c in chosen],
                   axis=0)
        chosen.append(coords[int(np.argmax(d))])
    return spawn.round(3).tolist(), \
        [c.round(3).tolist() for c in chosen[1:]]


def interior_mask(occ, floor, xy_min, cell, spawn, dilate_m=1.2):
    """실내 마스크 — 스폰에서 바닥 연결된 보행 영역 + 팽창.

    사진측량 스캔엔 창/개구부 너머의 실외 지형(언덕·마당 등)이 함께
    잡혀 있어 그대로 두면 '집 밖 월드'가 보인다 (실측: 아파트 스캔의
    언덕). 실내의 물리적 정의 = 로봇이 걸어서 닿을 수 있는 바닥과
    그것을 둘러싼 구조(팽창 반경 내 벽/천장). 그 밖은 잘라낸다."""
    free = floor & ~occ
    nx, ny = occ.shape
    si = int((spawn[0] - xy_min[0]) / cell)
    sj = int((spawn[1] - xy_min[1]) / cell)
    mask = np.zeros_like(free)
    stack = [(si, sj)]
    mask[si, sj] = True
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[a, b] \
                    and not mask[a, b]:
                mask[a, b] = True
                stack.append((a, b))
    if dilate_m <= 0:
        return mask
    r = max(1, int(round(dilate_m / cell)))
    # 팽창 (누적합 창 검사)
    ip = np.pad(mask.astype(np.int32), r + 1)
    cs = ip.cumsum(0).cumsum(1)
    out = np.zeros_like(mask)
    for i in range(nx):
        for j in range(ny):
            a, b = i + 1, j + 1
            s = cs[a + 2 * r + 1, b + 2 * r + 1] - cs[a, b + 2 * r + 1] \
                - cs[a + 2 * r + 1, b] + cs[a, b]
            out[i, j] = s > 0
    return out


def plug_holes(inmask, occ, xy_min, cell):
    """실내 경계의 미촬영 구멍 탐지 → 막을 벽 박스 목록.

    보행 가능 실내(inmask)에 인접한 셀이 '벽(occ)도 실내도 아닌' 경우
    = 스캔 구멍. 그대로 두면 재구성물의 바깥 껍데기(울퉁불퉁한 외피)
    가 '집 밖 언덕'처럼 보이고 로봇이 밖으로 샐 수도 있다 (실측).
    문간·복도는 양쪽 다 실내라 여기 걸리지 않는다 — 범용 규칙."""
    nx, ny = occ.shape
    rim = np.zeros_like(occ)
    for i in range(nx):
        for j in range(ny):
            if not inmask[i, j]:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and not inmask[a, b] \
                        and not occ[a, b]:
                    rim[a, b] = True
    boxes = []
    for j in range(ny):
        i = 0
        while i < nx:
            if not rim[i, j]:
                i += 1
                continue
            k = i
            while k + 1 < nx and rim[k + 1, j]:
                k += 1
            boxes.append([round(xy_min[0] + (i + k + 1) / 2 * cell, 3),
                          round(xy_min[1] + (j + 0.5) * cell, 3),
                          round((k - i + 1) / 2 * cell, 3),
                          round(cell / 2, 3)])
            i = k + 1
    # 세로 병합
    merged = []
    boxes.sort(key=lambda b: (b[0], b[2], b[1]))
    for b in boxes:
        if merged and merged[-1][0] == b[0] and merged[-1][2] == b[2] \
                and abs((merged[-1][1] + merged[-1][3]) -
                        (b[1] - b[3])) < 1e-6:
            m = merged[-1]
            lo, hi = m[1] - m[3], b[1] + b[3]
            m[1] = round((lo + hi) / 2, 3)
            m[3] = round((hi - lo) / 2, 3)
        else:
            merged.append(list(b))
    return merged


def filter_parts_to_mask(parts, mask, xy_min, cell):
    """마스크 밖 삼각형 제거 + 정점 재색인. 빈 파트는 탈락."""
    kept = []
    removed = 0
    for name, g in parts:
        cen = g.triangles_center
        ii = ((cen[:, 0] - xy_min[0]) / cell).astype(int)
        jj = ((cen[:, 1] - xy_min[1]) / cell).astype(int)
        nx, ny = mask.shape
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        inside = np.zeros(len(cen), bool)
        inside[ok] = mask[ii[ok], jj[ok]]
        removed += int((~inside).sum())
        if not inside.any():
            continue
        f = np.asarray(g.faces)[inside]
        used, inv = np.unique(f, return_inverse=True)
        g2 = trimesh.Trimesh(
            vertices=np.asarray(g.vertices)[used],
            faces=inv.reshape(-1, 3), process=False)
        if getattr(g.visual, 'uv', None) is not None:
            g2.visual = trimesh.visual.TextureVisuals(
                uv=np.asarray(g.visual.uv)[used],
                material=g.visual.material)
        else:
            g2.visual = g.visual
        kept.append((name, g2))
    print(f'실내 마스킹: 삼각형 {removed}개 제거(실외), '
          f'파트 {len(parts)}→{len(kept)}')
    return kept


def main():
    glb, outdir = sys.argv[1], sys.argv[2]
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 150000
    max_tex = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    # 상하축: auto = 수평면 면적 투표. 천장 없는 스캔은 벽 면적이
    # 바닥을 이겨 오검출될 수 있음(반 고흐 방 실측) — x/y/z로 강제 가능
    # (스캔별 메타데이터이지 로직 하드코딩이 아님)
    axis_arg = sys.argv[5] if len(sys.argv) > 5 else 'auto'
    cell = 0.15
    os.makedirs(outdir, exist_ok=True)

    parts = load_parts(glb)
    up = detect_up_axis(parts) if axis_arg == 'auto' \
        else {'x': 0, 'y': 1, 'z': 2}[axis_arg]
    R = rotation_to_z_up(up)
    for _, g in parts:
        g.apply_transform(R)
    z0 = detect_floor_z(parts)
    allv = np.concatenate([np.asarray(g.vertices) for _, g in parts])
    cxy = (allv[:, :2].min(axis=0) + allv[:, :2].max(axis=0)) / 2
    T = np.eye(4)
    T[:3, 3] = [-cxy[0], -cxy[1], -z0]
    for _, g in parts:
        g.apply_transform(T)

    # ---- 실내 마스킹: 1차 점유 분석으로 실내를 정의하고 실외 절단 ----
    occ0, top0, floor0, seat0, xym0 = build_occupancy(parts, cell)
    spawn0, _ = free_space(occ0, floor0, xym0, cell)
    mask = interior_mask(occ0, floor0, xym0, cell, spawn0)
    parts = filter_parts_to_mask(parts, mask, xym0, cell)
    # 실내 경계 구멍 → 막을 벽 박스 (BFS 원시 마스크 기준)
    inmask0 = interior_mask(occ0, floor0, xym0, cell, spawn0, dilate_m=0.0)
    plugs = plug_holes(inmask0, occ0, xym0, cell)
    print(f'구멍 마개 벽: {len(plugs)}개')

    total = sum(len(g.faces) for _, g in parts)
    print(f'up_axis={up} floor_z={z0:.3f} tris={total}')

    meta_parts = []
    for i, (name, g) in enumerate(parts):
        img, rgba = part_texture(g)
        v = np.asarray(g.vertices, np.float32)
        f = np.asarray(g.faces, np.int64)
        uv = np.asarray(g.visual.uv, np.float32) \
            if getattr(g.visual, 'uv', None) is not None else None
        if total > budget and HAVE_SIMPLIFY and len(f) > 2000:
            tgt = max(2000, int(budget * len(f) / total))
            v, f, uv = decimate_with_uv(g, tgt)
            print(f'  part{i} {name}: {len(g.faces)} -> {len(f)} tris')
        obj = f'part_{i}.obj'
        export_obj(os.path.join(outdir, obj), v, f, uv)
        entry = {'obj': obj}
        if img is not None and uv is not None:
            im = img.convert('RGB')
            if max(im.size) > max_tex:
                r = max_tex / max(im.size)
                im = im.resize((int(im.size[0] * r), int(im.size[1] * r)))
            tex = f'part_{i}.png'
            im.save(os.path.join(outdir, tex))
            entry['tex'] = tex
        elif rgba is not None:
            entry['rgba'] = rgba
        meta_parts.append(entry)

    # 점유/자유 분석은 (데시메이션 전) 원본 지오메트리로 — 이미 변환 적용됨
    occ, top, floor, seat, xy_min = build_occupancy(parts, cell)
    zones = merge_zones(occ, top, xy_min, cell)
    seats = merge_seats(seat, xy_min, cell)
    spawn, free = free_space(occ, floor, xy_min, cell)
    allv = np.concatenate([np.asarray(g.vertices) for _, g in parts])
    bbox = [allv[:, 0].min(), allv[:, 1].min(),
            allv[:, 0].max(), allv[:, 1].max()]
    meta = {'parts': meta_parts, 'zones': zones, 'seats': seats,
            'plugs': plugs, 'spawn': spawn, 'free': free, 'cell': cell,
            'bbox': [round(float(b), 3) for b in bbox]}
    with open(os.path.join(outdir, 'meta.json'), 'w') as fp:
        json.dump(meta, fp)
    print(f'zones={len(zones)} seats={len(seats)} spawn={spawn} '
          f'free={len(free)} bbox={meta["bbox"]}')


if __name__ == '__main__':
    main()
