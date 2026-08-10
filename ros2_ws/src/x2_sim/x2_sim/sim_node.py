#!/usr/bin/env python3
"""Agibot X2 자체 MuJoCo 시뮬 노드.

공식 시뮬 바이너리(aima-sim-app)가 AVX 의존으로 Apple Silicon(Rosetta)에서
실행 불가라, 공식 MJCF 모델을 직접 물리 시뮬하고 실기와 동일한 토픽/타입으로
노출한다. 에이전트/브릿지 쪽 코드는 실기 전환 시 수정이 필요 없다.

퍼블리시 (실기와 동일):
  /aima/hal/joint/{arm,leg,waist,head}/state   aimdk_msgs/JointStateArray
  /aima/hal/imu/{chest,torso}/state            sensor_msgs/Imu
구독:
  /aima/mc/locomotion/velocity                 aimdk_msgs/McLocomotionVelocity
"""
import io
import json
import math
import threading
import time

import mujoco
import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu
from std_msgs.msg import String
from std_srvs.srv import Trigger

from aimdk_msgs.msg import (
    JointState,
    JointStateArray,
    McLocomotionVelocity,
    PmuState,
)

from x2_sim.mc_services import McMock
from x2_sim.worlds import DEFAULT_WORLD, WORLDS

DEFAULT_MODEL = ('/sdk/aimdk-1.0.0/extra/x2_rl_deploy/x2_rl_deploy_mujoco/'
                 'configuration/robot/lx2501_3_t2d5/model_info/scene.xml')

# 관절 이름 → 실기 토픽 그룹 매핑
GROUP_KEYWORDS = {
    'arm': ('shoulder', 'elbow', 'wrist'),
    'leg': ('hip', 'knee', 'ankle'),
    'waist': ('waist',),
    'head': ('head',),
}


def group_of(joint_name: str):
    n = joint_name.lower()
    for group, keys in GROUP_KEYWORDS.items():
        if any(k in n for k in keys):
            return group
    return None


CAMERA_NAME = 'rgbd_head_front'


GEOM_TYPES = {
    'box': mujoco.mjtGeom.mjGEOM_BOX,
    'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER,
    'sphere': mujoco.mjtGeom.mjGEOM_SPHERE,
}


def load_model_with_camera_and_props(model_path: str,
                                     world_name: str = DEFAULT_WORLD):
    """공식 MJCF 로드 + 벤더 파일 무수정으로 카메라·월드 소품 주입.

    - 카메라: head_pitch_link 전면, 수직 FOV 55°(매뉴얼 RGB-D 스펙), 18° 하향
    - 소품/조명: worlds.py의 월드 정의에 따름
    반환: (compiled model, prop 이름 목록)
    """
    world = WORLDS[world_name]
    spec = mujoco.MjSpec.from_file(model_path)

    head = spec.body('head_pitch_link')
    cam = head.add_camera()
    cam.name = CAMERA_NAME
    cam.pos = [0.06, 0.0, 0.12]
    p = math.radians(18.0)
    s, c = math.sin(p), math.cos(p)
    rot = np.array([[0.0, s, -c],
                    [-1.0, 0.0, 0.0],
                    [0.0, c, s]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    cam.quat = quat
    cam.fovy = 55.0

    wb = spec.worldbody
    for pos, direction, diffuse in world['lights']:
        light = wb.add_light()
        light.pos = pos
        light.dir = direction
        light.diffuse = diffuse
        light.castshadow = False

    static_names, grasp_names = [], []
    for entry in world['props']:
        name, gtype, size, pos, rgba = entry[:5]
        opts = entry[5] if len(entry) > 5 else {}
        if opts.get('grasp'):
            # 집기 가능 물체 = 자유 물체(freejoint) — 밀리고 떨어지는
            # 실제 물리 오브젝트. 로봇이 집으면 키네마틱 캐리로 부착.
            b = wb.add_body()
            b.name = name
            b.pos = pos
            b.add_freejoint()
            g = b.add_geom()
            g.name = name
            g.type = GEOM_TYPES[gtype]
            g.size = size
            g.rgba = rgba
            g.density = 300.0     # 가벼운 소품 (물장난 방지)
            # 구름 마찰 부여 — MuJoCo 기본값(0.0001)은 사실상 0이라
            # 구형 물체가 테이블에 내려놓는 순간의 미세 충격만으로
            # 무한정 굴러 떨어짐 (사과 테이블 낙하 실측 사고).
            # ★rolling 마찰은 condim=6이어야 실제로 적용됨 (condim=3
            # 기본값에선 friction 3번째 값이 무시됨 — 1차 수정 실패 원인)
            g.condim = 6
            g.friction = [1.0, 0.01, 0.02]
            grasp_names.append(name)
        else:
            g = wb.add_geom()
            g.name = name
            g.type = GEOM_TYPES[gtype]
            g.size = size
            g.pos = pos
            g.rgba = rgba
            if opts.get('visual'):
                # 시각 전용(문짝/문틀 상단 등) — 충돌·이동 차단 없음
                g.contype = 0
                g.conaffinity = 0
            static_names.append(name)

    # 기능형 문 — 닫힘 포즈(통로 가로막음)로 생성. 런타임에 열면
    # 문설주 쪽으로 회전·이동 + 충돌/차단 해제 (sim_node._set_door)
    for d in world.get('doors', []):
        g = wb.add_geom()
        g.name = d['name']
        g.type = GEOM_TYPES['box']
        g.size = list(d['size'])
        g.pos = [d['closed_pos'][0], d['closed_pos'][1], d['size'][2]]
        g.rgba = [0.52, 0.38, 0.25, 1.0]
        static_names.append(d['name'])

    model = spec.compile()
    # CPU(OSMesa) 렌더 최적화 — 실측: 그림자맵 4096 + MSAA 4x가 프레임당
    # ~1.2s를 차지 (해상도 무관 고정비용). 끄면 640x480 기준 1.9s→0.5s.
    model.vis.quality.shadowsize = 256
    model.vis.quality.offsamples = 0
    for i in range(model.nlight):
        model.light_castshadow[i] = 0
    return model, static_names, grasp_names


class X2SimNode(Node):
    def __init__(self):
        super().__init__('x2_sim')
        self.declare_parameter('model_path', DEFAULT_MODEL)
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('kp', 120.0)
        self.declare_parameter('kd', 4.0)
        # 베이스 고정 모드: 보행 컨트롤러가 없으므로 기본 ON.
        # 이동 명령은 고정 베이스의 목표 위치/방향을 옮기는 방식으로 처리.
        self.declare_parameter('hold_base', True)

        # CPU(OSMesa) 렌더가 프레임당 ~0.7s라 실효 1~2fps.
        # 대화형 에이전트에는 충분. (Phase 3: look()은 필요 시 단발 렌더)
        # 0.2Hz 킵얼라이브만 — 고해상(640) 렌더는 ~0.5s간 llvmpipe
        # 스레드를 독점해 주기가 짧으면 프리뷰가 매초 멈칫(라이브 영상
        # 끊김의 원인)하고 전환 콜백도 굶는다(15s+ 지연 사고). 실제
        # 고해상 프레임은 look()이 /x2_sim/render_hq로 온디맨드 요청.
        self.declare_parameter('camera_rate', 0.2)
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('jpeg_quality', 80)

        self.declare_parameter('world', DEFAULT_WORLD)
        self.kp = float(self.get_parameter('kp').value)
        self.kd = float(self.get_parameter('kd').value)
        self.hold_base = bool(self.get_parameter('hold_base').value)
        w = int(self.get_parameter('camera_width').value)
        h = int(self.get_parameter('camera_height').value)
        self.cam_size = (w, h)
        self.declare_parameter('preview_width', 320)
        self.declare_parameter('preview_height', 240)
        self.prev_size = (int(self.get_parameter('preview_width').value),
                          int(self.get_parameter('preview_height').value))
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.renderer = None
        self.renderer_prev = None
        self.seq = 0

        # 월드 로드 (모델/매핑/베이스/소품/렌더러 전부 이 안에서 구성 —
        # 런타임 월드 전환 시에도 같은 메서드 재사용)
        # 마지막 월드 복원 — 재시작(수퍼바이저 등)마다 home으로 돌아가면
        # 탐색이 초기화된 것처럼 보인다 (학습 지도는 원래 영속).
        world_param = str(self.get_parameter('world').value)
        if world_param == DEFAULT_WORLD:
            try:
                last = open('/ws/log/last_world.txt').read().strip()
                if last in WORLDS:
                    world_param = last
            except Exception:
                pass
        self._build_world(world_param)

        # 퍼블리셔 (실기 토픽명)
        self.pub_joint = {
            g: self.create_publisher(
                JointStateArray, f'/aima/hal/joint/{g}/state',
                qos_profile_sensor_data)
            for g in GROUP_KEYWORDS}
        self.pub_imu = {
            name: self.create_publisher(
                Imu, f'/aima/hal/imu/{name}/state', qos_profile_sensor_data)
            for name in ('chest', 'torso')}
        # 위치(odometry) — 실기의 SLAM odom 토픽명 그대로
        self.pub_odom = self.create_publisher(
            Odometry, '/slam/lidar_odom', qos_profile_sensor_data)

        # PMU 목 — 감쇠 배터리 (만충 54.6V → 방전한계 39V, 기본 2시간)
        # "배터리 30% 미만 경고 / 15% 미만 세션 종료" 가드를 시뮬에서
        # 테스트할 수 있게 함. battery_minutes 파라미터로 감쇠 속도 조절.
        self.declare_parameter('battery_minutes', 120.0)
        self.batt_v_full, self.batt_v_empty = 54.6, 39.0
        self.batt_start_time = self.data.time
        self.pub_pmu = self.create_publisher(
            PmuState, '/aima/hal/pmu/state', qos_profile_sensor_data)
        self.create_timer(1.0, self.publish_pmu)
        self.create_subscription(
            McLocomotionVelocity, '/aima/mc/locomotion/velocity',
            self.on_cmd_vel, 10)

        # MC 목 서비스 (실기 프로토콜 강제: 모드/소스 게이트)
        self.mc = McMock(self)
        # 시뮬 전용 리셋 서비스 (웹 버튼/CLI에서 호출)
        self.create_service(Trigger, '/x2_sim/reset', self.on_reset)
        # 월드 전환: /x2_sim/set_world 구독, 현재 월드는 latched 발행
        from rclpy.qos import (DurabilityPolicy, QoSProfile)
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub_world = self.create_publisher(
            String, '/x2_sim/world', latched)
        self.create_subscription(
            String, '/x2_sim/set_world', self.on_set_world, 10)
        self.pub_world.publish(String(data=self.world_name))

        # 집기/놓기 인터페이스 (시뮬 전용 — 실기는 핸드 하드웨어+HandCommand)
        latched2 = QoSProfile(depth=1)
        latched2.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub_carrying = self.create_publisher(
            String, '/x2_sim/carrying', latched2)
        self.pub_grasp_result = self.create_publisher(
            String, '/x2_sim/grasp_result', 10)
        self.create_subscription(
            String, '/x2_sim/grasp', self.on_grasp, 10)
        self.create_subscription(
            String, '/x2_sim/release', self.on_release, 10)
        # 자세 직접 전환 (눕기 등 — MC 액션 체계 밖의 시뮬 전용 자세)
        self.create_subscription(
            String, '/x2_sim/posture',
            lambda m: self.set_posture(m.data.strip()), 10)
        # 문 조작 (open/close — 가장 가까운 문)
        self.create_subscription(
            String, '/x2_sim/door', self.on_door, 10)
        # 현재 자세 상태 latched 발행 — 에이전트가 재시작 무관하게 동기화
        # (에이전트측 플래그 추적은 시뮬 재시작 시 낡아서 '이미 누워
        #  있다' 오판을 만들었음)
        from rclpy.qos import DurabilityPolicy, QoSProfile
        _lat = QoSProfile(depth=1)
        _lat.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub_posture_state = self.create_publisher(
            String, '/x2_sim/posture_state', _lat)
        self.pub_posture_state.publish(String(data=self.posture))
        # 코스트맵 (경로 계획용 — 실기 SLAM 코스트맵 대응물)
        _lat2 = QoSProfile(depth=1)
        _lat2.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub_costmap = self.create_publisher(
            String, '/x2_sim/costmap', _lat2)
        self._rebuild_zones()   # 발행자 생성 후 1회 재구축·발행
        self.pub_carrying.publish(String(data=''))
        # 주변 물체 브로드캐스트 (에이전트 상태 주입용)
        self.pub_objects = self.create_publisher(
            String, '/x2_sim/objects', 10)
        self.create_timer(1.0, self.publish_objects)

        # ---------- 카메라 (Phase 2: 로봇이 보는 화면) ----------
        # 실기 토픽명 그대로. compressed가 주 소비 경로(실기 WiFi 대역폭 대비),
        # raw는 디버깅/후처리용.
        base = f'/aima/hal/sensor/{CAMERA_NAME}'
        self.pub_img_comp = self.create_publisher(
            CompressedImage, f'{base}/rgb_image/compressed',
            qos_profile_sensor_data)
        self.pub_img_raw = self.create_publisher(
            Image, f'{base}/rgb_image', qos_profile_sensor_data)
        self.pub_cam_info = self.create_publisher(
            CameraInfo, f'{base}/rgb_camera_info', qos_profile_sensor_data)
        self.cam_info = self.make_camera_info(
            self.cam_size[0], self.cam_size[1], fovy_deg=55.0)
        cam_rate = float(self.get_parameter('camera_rate').value)
        # 렌더(수십 ms)가 물리 틱을 막지 않도록 별도 콜백 그룹 + 상태 락
        # render_lock: 월드 전환 시 렌더러 교체와 렌더링의 경합 방지
        self.state_lock = threading.Lock()
        self.render_lock = threading.Lock()
        self.prev_lock = threading.Lock()
        self.cam_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / cam_rate, self.tick_camera,
                          callback_group=self.cam_group)
        # look() 온디맨드 고해상 렌더 요청 (같은 그룹 — 렌더 직렬화)
        self.create_subscription(
            String, '/x2_sim/render_hq',
            lambda m: self.tick_camera(), 1,
            callback_group=self.cam_group)
        # ---------- 라이브 프리뷰 스트림 ----------
        # UI용 저해상도 연속 렌더 — 별도 스레드가 쉬지 않고 렌더해서
        # CPU(OSMesa)가 허용하는 최대 fps로 발행. look()용 고해상 640x480
        # 스트림(위)과 분리해 '부드러운 영상'과 '선명한 스냅샷'을 양립.
        self.pub_preview = self.create_publisher(
            CompressedImage, '/x2_sim/preview/compressed',
            qos_profile_sensor_data)
        self._preview_thread = threading.Thread(
            target=self._preview_loop, daemon=True)
        self._preview_thread.start()
        # 전 월드 백그라운드 예열 — 전환 시 컴파일(~2s)+렌더러 생성이
        # 없게 미리 캐시. 부팅 후 수십 초에 걸쳐 조용히 진행.
        threading.Thread(target=self._warm_worlds, daemon=True).start()

        # 물리 스텝: publish_rate 주기마다 실시간 분량만큼 서브스텝
        rate = float(self.get_parameter('publish_rate').value)
        self.substeps = max(1, round(1.0 / rate / self.model.opt.timestep))
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f'시뮬 시작: {rate}Hz 퍼블리시, 스텝당 {self.substeps} 서브스텝, '
            f'카메라 {w}x{h}@{cam_rate}Hz → {base}/rgb_image[/compressed]')

    # ---------- 월드 구성/전환 ----------
    def _build_world(self, world_name: str):
        """월드(씬) 로드 — 초기화와 런타임 전환 양쪽에서 사용.

        모델/데이터/액추에이터 매핑/PD 목표/베이스 홀드/소품 충돌 목록/
        렌더러/배터리를 전부 새로 구성한다. 호출측에서 락을 잡는다
        (초기화 시엔 스레드가 없어 락 불필요).
        """
        model_path = self.get_parameter('model_path').value
        # 월드별 컴파일 캐시 — 재방문 시 MJCF/메시 재컴파일(~2s) 생략
        if not hasattr(self, '_world_cache'):
            self._world_cache = {}
        if world_name not in self._world_cache:
            self._world_cache[world_name] = load_model_with_camera_and_props(
                model_path, world_name)
        self.model, prop_names, grasp_names = self._world_cache[world_name]
        self.data = mujoco.MjData(self.model)
        self.world_name = world_name

        # 집기 시스템: 손(오른 손목 끝 링크) + 물체별 freejoint 주소
        self.hand_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_roll_link')
        self.graspables = {}
        self.grasp_bids = set()
        for name in grasp_names:
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jid = self.model.body_jntadr[bid]
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.graspables[name] = {
                'bid': bid,
                'gid': gid,
                'qpos_adr': self.model.jnt_qposadr[jid],
                'qvel_adr': self.model.jnt_dofadr[jid],
            }
            self.grasp_bids.add(bid)
            # 모델이 월드 캐시에 공유되므로, 이전 세션에서 운반 중이던
            # 물체의 접촉 비활성 상태가 남아있을 수 있음 — 항상 복원
            self.model.geom_contype[gid] = 1
            self.model.geom_conaffinity[gid] = 1
        self.carrying = None

        # 팔 계열 링크 — 이동 차단 판정에서 제외. 소품 옆에서 제자리
        # 회전할 때 팔이 카운터/아일랜드를 스치는 접촉이 회전 롤백을
        # 반복시켜 '옆에 서면 못 도는' 교착을 만들었음 (스튜디오행
        # 실패 사고). 베이스/다리 접촉은 계속 차단.
        self.arm_bids = set()
        for bid in range(self.model.nbody):
            bname = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ''
            # ankle/foot 포함: 가구 옆 회전·측면 보행 시 발끝 스침도 같은
            # 교착을 만들었음(island↔left_ankle_roll_link 실측). 가구
            # '진입'은 몸통/골반/무릎 접촉이 계속 차단하므로 안전 유지.
            if any(k in bname for k in
                   ('shoulder', 'arm', 'elbow', 'wrist', 'hand',
                    'ankle', 'foot')):
                self.arm_bids.add(bid)

        self.act_joint_ids = [
            self.model.actuator_trnid[i][0] for i in range(self.model.nu)]
        self.act_qpos_adr = [
            self.model.jnt_qposadr[j] for j in self.act_joint_ids]
        self.act_qvel_adr = [
            self.model.jnt_dofadr[j] for j in self.act_joint_ids]
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in self.act_joint_ids]
        self.groups = {g: [] for g in GROUP_KEYWORDS}
        for ai, name in enumerate(self.joint_names):
            g = group_of(name or '')
            if g:
                self.groups[g].append(ai)
        self.jname_to_ai = {n: i for i, n in enumerate(self.joint_names)}

        # 착석 강성 벡터 (허리/다리만 kp 5배, kd 2배 — apply_pd에서 사용)
        self._sit_gain = np.ones(self.model.nu)
        self._sit_kd_gain = np.ones(self.model.nu)
        # 회전 강성 벡터 (허리만 3배 — 커브 휘청 완화)
        self._turn_gain = np.ones(self.model.nu)
        self._turn_kd_gain = np.ones(self.model.nu)
        for jname, ai in self.jname_to_ai.items():
            if any(k in (jname or '') for k in
                   ('waist', 'hip', 'knee', 'ankle')):
                self._sit_gain[ai] = 5.0
                self._sit_kd_gain[ai] = 2.0
            if 'waist' in (jname or ''):
                self._turn_gain[ai] = 3.0
                self._turn_kd_gain[ai] = 1.7
        # 머리(카메라 마운트) 관절은 상시 강성 — 관성으로 머리가 흔들려
        # '넘어질 것 같은' 인상을 주던 것 억제 (실기의 헤드 스태빌라이저
        # 역할을 시뮬에서 대신)
        self._head_gain = np.ones(self.model.nu)
        self._head_kd_gain = np.ones(self.model.nu)
        for jname, ai in self.jname_to_ai.items():
            if 'head' in (jname or ''):
                self._head_gain[ai] = 4.0
                self._head_kd_gain[ai] = 2.0

        mujoco.mj_forward(self.model, self.data)
        self.q_ref = np.array(
            [self.data.qpos[a] for a in self.act_qpos_adr])
        self.cmd_vel = np.zeros(3)
        self.vel_smooth = np.zeros(3)   # 가감속 램프 적용된 실효 속도
        self.cmd_time = 0.0
        self.motion = None
        self.base_xyz = np.array(self.data.qpos[0:3])
        qw, qx, qy, qz = self.data.qpos[3:7]
        self.base_yaw = math.atan2(2 * (qw * qz + qx * qy),
                                   1 - 2 * (qy * qy + qz * qz))
        self.prev_base_linvel = np.zeros(3)
        self.batt_start_time = self.data.time

        self.posture = 'stand'
        # 일반화 자세 상태 — {z(절대), pitch(rad), joints{이름:오프셋}}를
        # 1.5s smoothstep으로 보간. 글라이드(수평 이동)는 전환에 실어
        # 침대 위로 오르내리기/소파에서 몸 빼기를 표현.
        self._pose_cur = {'z': float(self.data.qpos[2]), 'pitch': 0.0,
                          'joints': {}}
        self._pose_from = {'z': self._pose_cur['z'], 'pitch': 0.0,
                           'joints': {}}
        self._pose_to = {'z': self._pose_cur['z'], 'pitch': 0.0,
                         'joints': {}}
        self._pose_t = 1.0
        self._pose_glide = (0.0, 0.0)
        self._lie_entry_glide = (0.0, 0.0)
        self.base_z0 = float(self.data.qpos[2])
        self.prop_gids = set()
        # ★이동 차단 = 기하 판정 (물리 접촉과 분리).
        # 접촉 기반 롤백은 유연한 관절이 지오메트리에 박히면 관통·법선이
        # 물리 노이즈에 흔들려 전방향 교착이 반복 재발했음 (문틀 실측
        # 3차). 대신 '베이스 중심 vs 가구 2D AABB + 여유 반경'의 순수
        # 기하 규칙으로 게이팅: 결정론적, 노이즈 0, 팔다리 무관.
        # 낮은 가구(윗면 z<0.5 — 소파/침대/티테이블 등 앉는 면)는 여유를
        # 작게(0.12m) 둬 도킹 밀착이 가능하고, 벽/장롱류는 0.30m.
        self._zone_prop_names = list(prop_names)
        self._rebuild_zones()

        # 기능형 문 등록 + 초기 상태 (프로세스당 월드별 1회만 init 적용
        # — 이후엔 마지막 상태 유지, 실제 문처럼)
        self.doors = {}
        for dd in WORLDS[world_name].get('doors', []):
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, dd['name'])
            if gid < 0:
                continue
            self.doors[dd['name']] = {
                'gid': gid,
                'closed_pos': dd['closed_pos'],
                'open_pos': dd['open_pos'],
            }
        if not hasattr(self, '_door_init_done'):
            self._door_init_done = set()
        if world_name not in self._door_init_done:
            self._door_init_done.add(world_name)
            for dd in WORLDS[world_name].get('doors', []):
                if dd.get('init') == 'open' and dd['name'] in self.doors:
                    self._set_door(dd['name'], 'open', rebuild=False)
            self._rebuild_zones()
        self._contact_prev = False
        self._pen_last = 0.0
        self._blocked_logged = False

        # 렌더러도 월드별 캐시 — 전환마다 OSMesa 컨텍스트를 재생성하던
        # 것이 전환 지연의 주범(모델 컴파일과 함께). 캐시 스왑이면 즉시.
        if not hasattr(self, '_renderer_cache'):
            self._renderer_cache = {}
        rc = self._renderer_cache.get(world_name)
        if rc is None:
            rc = (mujoco.Renderer(self.model, height=self.cam_size[1],
                                  width=self.cam_size[0]),
                  mujoco.Renderer(self.model, height=self.prev_size[1],
                                  width=self.prev_size[0]))
            self._renderer_cache[world_name] = rc
        self.renderer, self.renderer_prev = rc
        self.get_logger().info(
            f'월드 로드: {world_name} — 소품 {len(prop_names)}개, '
            f'nq={self.model.nq}, nu={self.model.nu}')

    def on_set_world(self, msg: String):
        name = msg.data.strip()
        if name not in WORLDS:
            self.get_logger().warning(
                f'미등록 월드 {name!r} — 사용 가능: {list(WORLDS)}')
            return
        with self.render_lock, self.prev_lock:
            with self.state_lock:
                self._build_world(name)
        self.mc.reset()
        self.pub_world.publish(String(data=name))
        self.pub_carrying.publish(String(data=''))
        try:                                # 재시작 시 복원용
            with open('/ws/log/last_world.txt', 'w') as f:
                f.write(name)
        except Exception:
            pass
        # (즉시 렌더 특례 제거 — 프리뷰 스레드가 0.1s 내 새 씬을 발행)
        self.get_logger().info(f'월드 전환 완료: {name}')

    # ---------- 집기/놓기 (시뮬 전용) ----------
    GRASP_RANGE = 0.95   # 베이스 기준 도달 반경 (m)

    def on_grasp(self, msg: String):
        name = msg.data.strip()
        info = self.graspables.get(name)
        if info is None:
            self.pub_grasp_result.publish(String(
                data=f'fail:unknown:{name}'))
            return
        if self.carrying:
            self.pub_grasp_result.publish(String(
                data=f'fail:hands_full:{self.carrying}'))
            return
        with self.state_lock:
            obj = self.data.xpos[info['bid']]
            base = self.data.qpos[0:2]
            dist = float(np.hypot(obj[0] - base[0], obj[1] - base[1]))
            if dist > self.GRASP_RANGE:
                self.pub_grasp_result.publish(String(
                    data=f'fail:too_far:{name}:{dist:.2f}'))
                self.get_logger().info(
                    f'집기 거부: {name} 거리 {dist:.2f}m > {self.GRASP_RANGE}m')
                return
            self.carrying = name
            # 운반 중 물체의 물리 접촉 비활성 — 앉을 때 물체가 허벅지와
            # 겹치며 접촉력으로 상체를 밀어 자세가 기우는 문제 방지
            # (착석 화면 기울어짐의 원인). 놓으면 복원.
            self.model.geom_contype[info['gid']] = 0
            self.model.geom_conaffinity[info['gid']] = 0
        self.pub_carrying.publish(String(data=name))
        self.pub_grasp_result.publish(String(data=f'ok:{name}'))
        self.get_logger().info(f'집음: {name} (거리 {dist:.2f}m)')

    def on_release(self, msg: String):
        if self.carrying:
            self.get_logger().info(f'내려놓음: {self.carrying}')
            info = self.graspables.get(self.carrying)
            if info is not None:            # 접촉 복원 (다시 잡을 수 있게)
                self.model.geom_contype[info['gid']] = 1
                self.model.geom_conaffinity[info['gid']] = 1
                # 놓는 위치를 오른손이 아니라 몸 정면 중앙으로 보정 —
                # 손 기준으로 놓으면 우측(어깨 오프셋)으로 치우쳐
                # 테이블 모서리에 떨어지는 사고 (y=2.52 모서리 낙하 실측).
                # carrying 해제까지 같은 락 안에서 — 락 밖에서 해제하면
                # 물리 틱의 apply_carry가 물체를 손 위치로 되돌리는 레이스
                with self.state_lock:
                    c = math.cos(self.base_yaw)
                    s = math.sin(self.base_yaw)
                    adr = info['qpos_adr']
                    z = float(self.data.qpos[adr + 2])
                    self.data.qpos[adr] = self.base_xyz[0] + 0.45 * c
                    self.data.qpos[adr + 1] = self.base_xyz[1] + 0.45 * s
                    self.data.qpos[adr + 2] = z
                    vadr = info['qvel_adr']
                    self.data.qvel[vadr:vadr + 6] = 0.0
                    self.carrying = None
        self.carrying = None
        self.pub_carrying.publish(String(data=''))

    def apply_carry(self):
        """잡은 물체를 손 위치에 키네마틱 부착 (놓으면 물리로 낙하)."""
        if not self.carrying:
            return
        info = self.graspables.get(self.carrying)
        if info is None:
            self.carrying = None
            return
        hand = self.data.xpos[self.hand_bid]
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        # 손 약간 앞/아래 지점에 물체 고정 (몸 기준 전방 오프셋)
        adr = info['qpos_adr']
        self.data.qpos[adr:adr + 3] = (
            hand[0] + 0.10 * c, hand[1] + 0.10 * s, hand[2] - 0.03)
        vadr = info['qvel_adr']
        self.data.qvel[vadr:vadr + 6] = 0.0

    SIT_JOINTS = {
        'left_hip_pitch_joint': -1.35, 'right_hip_pitch_joint': -1.35,
        'left_knee_joint': 1.5, 'right_knee_joint': 1.5,
        'left_ankle_pitch_joint': -0.15, 'right_ankle_pitch_joint': -0.15,
    }

    def _platform_top(self, x, y):
        """(x,y) 아래의 낮은 가구(침대/소파 등) 윗면 높이. 없으면 0."""
        best = 0.0
        for (cx, cy, hx, hy, margin, top, name) in self.block_zones:
            if top < 0.6 and abs(x - cx) <= hx + 0.05 \
                    and abs(y - cy) <= hy + 0.05:
                best = max(best, top)
        return best

    def _find_platform(self, direction, dists):
        """direction(+1 전방/-1 후방) 원뿔 스캔으로 낮은 가구 탐색.

        좌우 ±0.28m 오프셋도 훑는다 — 도착 지점이 가구와 살짝 어긋나
        있어도(문 축 정렬 등) 잡아낸다. 반환: (글라이드 벡터 x, y,
        윗면 높이) 또는 None."""
        c, ss = math.cos(self.base_yaw), math.sin(self.base_yaw)
        px, py = -ss, c                 # 좌측 수직 방향
        for d in dists:
            for off in (0.0, 0.28, -0.28):
                x = (self.base_xyz[0] + direction * d * c + off * px)
                y = (self.base_xyz[1] + direction * d * ss + off * py)
                top = self._platform_top(x, y)
                if top > 0.05:
                    gx = direction * d * c + off * px
                    gy = direction * d * ss + off * py
                    return gx, gy, top
        return None

    def set_posture(self, posture: str):
        """자세 전환 (stand/sit/lie) — 1.5s 램프, ★가구 범용.

        sit: 주변 낮은 가구(침대/소파/의자 무엇이든) 자동 감지 —
          뒤에 있으면 걸터앉고, 앞에 있으면 반턴(180°)해서 앉는다.
          없으면 제자리 바닥 높이 앉기.
        lie: 전방의 낮은 가구를 찾아 그 위로 올라가 눕고, 없으면 바닥.
        특정 가구 이름/장소 설정에 의존하지 않는다 (범용성 원칙)."""
        if posture not in ('stand', 'sit', 'lie'):
            return
        yaw = self.base_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        glide = (0.0, 0.0)
        dyaw = 0.0
        if posture == 'sit':
            back = self._find_platform(-1, (0.25, 0.4, 0.55, 0.7))
            front = None if back else self._find_platform(
                +1, (0.25, 0.4, 0.55, 0.7))
            hit = back or front
            if hit:
                gx, gy, top = hit
                n = math.hypot(gx, gy) or 1.0
                # 감지 지점보다 0.10m 더 깊이 걸터앉기
                glide = (gx * (1 + 0.10 / n), gy * (1 + 0.10 / n))
                if front:               # 앞 가구 → 반턴 후 걸터앉기
                    dyaw = math.pi
                tgt = {'z': top + 0.30, 'pitch': 0.0,
                       'joints': dict(self.SIT_JOINTS)}
            else:
                tgt = {'z': self.base_z0 - 0.23, 'pitch': 0.0,
                       'joints': dict(self.SIT_JOINTS)}
            self._sit_entry_glide = glide
        elif posture == 'lie':
            # 전방 우선, 없으면 후방(걸터앉은 자세에서 뒤로 벌렁 눕기 —
            # 앉기가 가구를 등지므로 후방 스캔이 없으면 바닥에 누움)
            found = self._find_platform(
                +1, (0.35, 0.5, 0.65, 0.8, 0.95, 1.1))
            if found is None:
                found = self._find_platform(
                    -1, (0.25, 0.4, 0.55, 0.7, 0.85))
            if found:
                gx, gy, top = found     # 감지 지점보다 0.30m 안쪽까지
                n = math.hypot(gx, gy) or 1.0
                glide = (gx * (1 + 0.30 / n), gy * (1 + 0.30 / n))
            else:
                glide, top = (0.45 * c, 0.45 * s), 0.0
            # +0.22: 몸이 매트리스에 파묻히면 접촉 지터로 천장 시야가
            # 계속 흔들림 (사용자 제보) — 접촉을 최소화하는 높이.
            # 피치 -1.72(-98°): 카메라 자체가 18° 하향 장착이라 -80°로는
            # 시선이 윗벽에 걸림("천장 안 보고 있는데?" 제보) — 보정치
            tgt = {'z': top + 0.22, 'pitch': -1.72, 'joints': {}}
            self._lie_entry_glide = glide
        else:
            tgt = {'z': self.base_z0, 'pitch': 0.0, 'joints': {}}
            if self.posture == 'lie':      # 눕던 자리에서 역글라이드
                glide = (-self._lie_entry_glide[0],
                         -self._lie_entry_glide[1])
            elif self.posture == 'sit':    # 앉던 자리에서 앞으로 빠져나옴
                eg = getattr(self, '_sit_entry_glide', (0.0, 0.0))
                glide = (-eg[0] + 0.10 * c, -eg[1] + 0.10 * s)
        self._pose_from = {'z': self._pose_cur['z'],
                           'pitch': self._pose_cur['pitch'],
                           'joints': dict(self._pose_cur['joints'])}
        self._pose_to = tgt
        self._pose_glide = glide
        self._pose_dyaw = dyaw
        self._pose_t = 0.0
        self.posture = posture
        if hasattr(self, 'pub_posture_state'):
            self.pub_posture_state.publish(String(data=posture))
        self.get_logger().info(f'자세 전환 시작: {posture} (1.5s 램프'
                               + (', 반턴' if dyaw else '') + ')')

    def posture_offsets(self):
        j = self._pose_cur['joints']
        return dict(j) if j else None

    def carry_arm_offsets(self):
        """물건을 들고 있는 동안 오른팔 들기 자세."""
        if not self.carrying:
            return None
        return {'right_shoulder_pitch_joint': -0.9,
                'right_elbow_joint': -0.9}

    # 랜드마크로 노출할 정적 소품 → 친화적 이름 매핑.
    # 배경: "나무 쓰다듬어줘"에서 LLM이 나무까지의 거리를 알 방법이
    # 없어(집기 물체 거리만 주입) 3.5m 밖에서 팔만 흔들고 했다고
    # 거짓 보고한 사고 — confabulation은 데이터로 차단한다는 원칙.
    LANDMARKS = {
        'sofa_seat': 'sofa', 'tea_table': 'tea_table', 'tv': 'tv',
        'plant_pot': 'plant', 'island': 'island', 'counter': 'counter',
        'fridge': 'fridge', 'stove': 'stove', 'desk': 'desk',
        'studio_table': 'studio_table',
    }

    def publish_objects(self):
        objs = {}
        for name, info in self.graspables.items():
            p = self.data.xpos[info['bid']]
            objs[name] = {'x': round(float(p[0]), 2),
                          'y': round(float(p[1]), 2),
                          'z': round(float(p[2]), 2)}
        lands = {}
        for prop, friendly in self.LANDMARKS.items():
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, prop)
            if gid >= 0:
                p = self.data.geom_xpos[gid]
                lands[friendly] = {'x': round(float(p[0]), 2),
                                   'y': round(float(p[1]), 2)}
        doors = {}
        for name, d in self.doors.items():
            doors[name] = {
                'x': d['closed_pos'][0], 'y': d['closed_pos'][1],
                'state': ('open' if int(
                    self.model.geom_contype[d['gid']]) == 0 else 'closed')}
        payload = {'world': self.world_name, 'carrying': self.carrying,
                   'objects': objs, 'landmarks': lands, 'doors': doors}
        self.pub_objects.publish(String(
            data=json.dumps(payload, ensure_ascii=False)))

    # ---------- 콜백 ----------
    def on_reset(self, req, res):
        """시뮬을 최초 상태로 완전 초기화 (물리/베이스/모션/MC 상태)."""
        with self.state_lock:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.q_ref = np.array(
                [self.data.qpos[a] for a in self.act_qpos_adr])
            self.cmd_vel = np.zeros(3)
            self.vel_smooth = np.zeros(3)
            self.cmd_time = 0.0
            self.motion = None
            self.posture = 'stand'
            self._pose_cur = {'z': float(self.data.qpos[2]), 'pitch': 0.0,
                              'joints': {}}
            self._pose_to = dict(self._pose_cur, joints={})
            self._pose_t = 1.0
            self._pose_glide = (0.0, 0.0)
            self.base_xyz = np.array(self.data.qpos[0:3])
            qw, qx, qy, qz = self.data.qpos[3:7]
            self.base_yaw = math.atan2(2 * (qw * qz + qx * qy),
                                       1 - 2 * (qy * qy + qz * qz))
            self.prev_base_linvel = np.zeros(3)
            self.batt_start_time = self.data.time   # 배터리 만충 복원
            if self.carrying:               # 운반 중이었다면 접촉 복원
                info = self.graspables.get(self.carrying)
                if info is not None:
                    self.model.geom_contype[info['gid']] = 1
                    self.model.geom_conaffinity[info['gid']] = 1
            self.carrying = None
        self.mc.reset()
        self.pub_carrying.publish(String(data=''))
        self.get_logger().info('시뮬 초기화 완료 (초기 자세/위치/MC 상태)')
        res.success = True
        res.message = '시뮬 초기화 완료'
        return res

    def on_cmd_vel(self, msg: McLocomotionVelocity):
        # 실기 의미론 게이트: locomotion 모드 + 등록된 소스가 아니면 무시
        if not self.mc.gate_velocity(msg.source):
            return
        self.cmd_vel = np.array([
            msg.forward_velocity, msg.lateral_velocity, msg.angular_velocity])
        self.cmd_time = self.data.time
        self._blocked_logged = False   # 새 명령마다 차단 로그 1회 허용
        self.get_logger().info(
            f'이동 명령 수신: vx={msg.forward_velocity:.2f} '
            f'vy={msg.lateral_velocity:.2f} wz={msg.angular_velocity:.2f} '
            f'(source={msg.source!r})')

    def clear_locomotion(self):
        """비상(damping/zero-torque) 진입 시 이동 명령 즉시 소거."""
        self.cmd_vel = np.zeros(3)

    def play_motion(self, name: str):
        """프리셋 모션 시작 (McMock에서 검증 후 호출됨)."""
        self.motion = (name, self.data.time)

    def motion_offsets(self):
        """진행 중인 프리셋 모션의 관절 오프셋 계산 (q_ref에 가산)."""
        if self.motion is None:
            return None
        name, t0 = self.motion
        t = self.data.time - t0
        dur = 2.5
        if t >= dur:
            self.motion = None
            return None
        # 부드러운 시작/끝 (0→1→0 envelope)
        env = math.sin(math.pi * min(t / dur, 1.0))
        off = {}
        if name in ('POINT_HEAD',):
            off['head_pitch_joint'] = 0.3 * env * math.sin(2 * math.pi * 1.5 * t)
        elif name in ('SHAKE_HEAD',):
            off['head_yaw_joint'] = 0.35 * env * math.sin(2 * math.pi * 1.5 * t)
        else:
            # 손 계열 기본 동작: 오른팔 들어 흔들기
            off['right_shoulder_pitch_joint'] = -1.3 * env
            off['right_shoulder_roll_joint'] = -0.3 * env
            off['right_elbow_joint'] = (-0.6 + 0.4 * math.sin(
                2 * math.pi * 2.0 * t)) * env
        return off

    # ---------- 메인 루프 ----------
    def _robot_prop_contact(self) -> bool:
        """로봇 geom과 소품 geom 사이의 접촉 존재 여부.

        차단 대상 접촉의 관통 깊이 합(_last_contact_pen)도 계산한다 —
        탈출 판정은 '관통이 늘지 않으면 허용'(단조 규칙)."""
        self._last_contact_pen = 0.0
        found = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a, b = c.geom1, c.geom2
            a_prop, b_prop = a in self.prop_gids, b in self.prop_gids
            if a_prop == b_prop:
                continue              # 둘 다 소품이거나 둘 다 비소품
            other = b if a_prop else a
            other_body = self.model.geom_bodyid[other]
            if other_body == 0:
                continue              # 바닥 등 worldbody geom 제외
            if other_body in self.grasp_bids:
                continue              # 집기 물체가 소품 위에 놓인 것 — 로봇 아님
            if other_body in self.arm_bids:
                continue              # 팔 스침은 차단 사유 아님 (회전 교착 방지)
            self._last_contact_pen += max(0.0, -float(c.dist))
            if found:
                continue              # 이름 로깅은 첫 접촉만
            found = True
            self._last_contact_pair = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                  a if a_prop else b) or '?',
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                  other)
                or mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                     other_body) or '?')
        return found                  # 하나라도 로봇↔소품 접촉이면 True

    def tick(self):
        with self.state_lock:
            for _ in range(self.substeps):
                self.apply_pd()
                self.apply_locomotion()   # 이동 게이팅은 이 안(기하 판정)
                mujoco.mj_step(self.model, self.data)
                self.apply_carry()
        self.publish_joints()
        self.publish_imu()
        self.publish_odom()
        self.seq += 1

    def apply_pd(self):
        q = self.data.qpos[self.act_qpos_adr]
        v = self.data.qvel[self.act_qvel_adr]
        q_target = self.q_ref
        offs = self.motion_offsets() or {}
        carry = self.carry_arm_offsets()
        posture = self.posture_offsets()
        if posture:
            offs = {**posture, **offs}
        if carry:
            offs = {**carry, **offs}   # 모션이 있으면 모션 우선
        if offs:
            q_target = self.q_ref.copy()
            for jname, delta in offs.items():
                ai = self.jname_to_ai.get(jname)
                if ai is not None:
                    q_target[ai] += delta
        kp = self.kp * self._head_gain      # 머리 상시 안정화
        kd = self.kd * self._head_kd_gain
        # 회전 중 허리 강성 보강 — 유연한 허리가 원심/관성으로 휘어
        # 커브마다 카메라가 휘청이던 것(사용자 제보) 완화
        if abs(self.vel_smooth[2]) > 0.05:
            kp = kp * self._turn_gain
            kd = kd * self._turn_kd_gain
        if self.posture == 'lie':
            # 누운 자세: 전신 강성/감쇠 상향 — 침구 접촉 지터로 카메라가
            # 계속 떨리던 것(사용자 제보) 억제 (팔·목 포함 전 관절)
            kp = self.kp * 4.0
            kd = self.kd * 2.5
        elif self.posture != 'stand' or self._pose_t < 1.0:
            # 착석 중 허리/다리 강성 상향 — 소파 접촉력이 유연한(kp=120)
            # 관절을 밀어 카메라가 기울던 문제(허리 0.29rad 편차 실측).
            # 실기에서는 MC가 자체 자세 제어를 하므로 시뮬 전용 보정.
            kp = self.kp * self._sit_gain
            kd = self.kd * self._sit_kd_gain
        tau = kp * (q_target - q) - kd * v
        self.data.ctrl[:] = np.clip(
            tau, self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1]) \
            if self.model.actuator_ctrllimited.any() else tau

    def _rebuild_zones(self):
        """가구 차단 구역 재구축 — 현재 geom 포즈/contype 기준
        (문 열림/닫힘 토글 시에도 호출)."""
        self.block_zones = []
        for name in self._zone_prop_names:
            # 바닥/천장/러그는 통행면·머리 위 — 차단 대상 아님
            # (천장을 2D 차단 구역에 넣으면 전 면적이 막힌다!)
            if '_floor' in name or '_ceiling' in name or name == 'rug':
                continue
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            if int(self.model.geom_contype[gid]) == 0:
                continue              # 시각 전용/열린 문 — 차단 제외
            self.prop_gids.add(gid)
            p = self.model.geom_pos[gid]
            s = self.model.geom_size[gid]
            gtype = int(self.model.geom_type[gid])
            if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                hx, hy, top = float(s[0]), float(s[1]), \
                    float(p[2]) + float(s[2])
            else:                      # cylinder/sphere: 반경 기반
                hx = hy = float(s[0])
                top = float(p[2]) + float(s[1] if gtype == int(
                    mujoco.mjtGeom.mjGEOM_CYLINDER) else s[0])
            margin = 0.30 if top >= 0.5 else 0.12
            self.block_zones.append(
                (float(p[0]), float(p[1]), hx, hy, margin, top, name))
        # 코스트맵 발행 — 실기의 SLAM 코스트맵 대응물 (로봇이 라이다로
        # 실시간 구축하는 정보라 '지도 유출'이 아님). 에이전트의 경로
        # 계획(A*)이 사용. latched: 문 토글 등 갱신 시마다 재발행.
        if hasattr(self, 'pub_costmap'):
            self.pub_costmap.publish(String(data=json.dumps(
                [[z[0], z[1], z[2], z[3], z[4]]
                 for z in self.block_zones])))

    def _set_door(self, name, state, rebuild=True):
        """문 상태 전환 — open: 문설주로 접힘(시각만)/closed: 통로 차단."""
        d = self.doors.get(name)
        if d is None:
            return
        gid = d['gid']
        if state == 'open':
            self.model.geom_pos[gid][0] = d['open_pos'][0]
            self.model.geom_pos[gid][1] = d['open_pos'][1]
            self.model.geom_quat[gid][:] = (0.70710678, 0.0, 0.0,
                                            0.70710678)
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0
        else:
            self.model.geom_pos[gid][0] = d['closed_pos'][0]
            self.model.geom_pos[gid][1] = d['closed_pos'][1]
            self.model.geom_quat[gid][:] = (1.0, 0.0, 0.0, 0.0)
            self.model.geom_contype[gid] = 1
            self.model.geom_conaffinity[gid] = 1
        if rebuild:
            self._rebuild_zones()
        self.get_logger().info(f'문 {name}: {state}')

    def on_door(self, msg: String):
        """가장 가까운 문(1.35m 내)을 열거나 닫는다."""
        want = msg.data.strip()
        if want not in ('open', 'close'):
            return
        best = None
        for name, d in self.doors.items():
            dist = math.hypot(self.base_xyz[0] - d['closed_pos'][0],
                              self.base_xyz[1] - d['closed_pos'][1])
            if dist < 1.35 and (best is None or dist < best[0]):
                best = (dist, name)
        if best is None:
            self.get_logger().info('근처(1.35m)에 문 없음 — 무시')
            return
        self._set_door(best[1], 'open' if want == 'open' else 'closed')

    def _tall_pen(self, x, y):
        """키 큰 구역(벽/닫힌 문/장롱 — 윗면 0.5m 이상)만의 관통량."""
        worst = 0.0
        for (cx, cy, hx, hy, margin, top, name) in self.block_zones:
            if top < 0.5:
                continue
            dx = abs(x - cx) - hx
            dy = abs(y - cy) - hy
            pen = margin - math.hypot(max(0.0, dx), max(0.0, dy))
            if pen > worst:
                worst = pen
        return worst

    def _zone_pen(self, x, y):
        """(x,y)에서 가구 차단 구역 관통량(최댓값). 0이면 자유 공간."""
        worst = 0.0
        for (cx, cy, hx, hy, margin, top, name) in self.block_zones:
            dx = abs(x - cx) - hx
            dy = abs(y - cy) - hy
            d = math.hypot(max(0.0, dx), max(0.0, dy))
            pen = margin - d
            if pen > worst:
                worst = pen
                self._zone_hit_name = name
        return worst

    def apply_locomotion(self):
        """키네마틱 베이스 이동 (보행 컨트롤러 대체).

        hold_base 모드: 베이스 pose를 목표값으로 매 스텝 고정하고,
        이동 명령은 목표값(x, y, yaw)을 적분해 옮긴다. 로봇이 넘어질 수
        없어 안정적이며, 에이전트 입장에선 '명령 → 이동'이라는 관측
        결과가 실기와 동일하므로 Phase 2~4 개발에 충분하다.
        """
        dt = self.model.opt.timestep
        # 자세 램프: 파라미터(z/pitch/관절/글라이드)를 1.5s smoothstep 보간
        if self._pose_t < 1.0:
            t0 = self._pose_t
            s0 = t0 * t0 * (3.0 - 2.0 * t0)
            self._pose_t = min(1.0, self._pose_t + dt / 1.5)
            t1 = self._pose_t
            s1 = t1 * t1 * (3.0 - 2.0 * t1)
            f, tt = self._pose_from, self._pose_to
            self._pose_cur['z'] = f['z'] + (tt['z'] - f['z']) * s1
            self._pose_cur['pitch'] = (f['pitch']
                                       + (tt['pitch'] - f['pitch']) * s1)
            joints = {}
            for k in set(f['joints']) | set(tt['joints']):
                a = f['joints'].get(k, 0.0)
                b2 = tt['joints'].get(k, 0.0)
                joints[k] = a + (b2 - a) * s1
            self._pose_cur['joints'] = joints
            ds = s1 - s0
            # 글라이드 게이팅: 낮은 가구(침대/소파) 위 이동은 허용하되,
            # ★키 큰 구역(벽/닫힌 문/장롱)은 관통 금지 — 문 앞에서
            # 눕기/앉기가 실행되면 글라이드가 닫힌 문을 뚫던 사고 방지
            gnx = self.base_xyz[0] + self._pose_glide[0] * ds
            gny = self.base_xyz[1] + self._pose_glide[1] * ds
            if self._tall_pen(gnx, gny) <= self._tall_pen(
                    self.base_xyz[0], self.base_xyz[1]) + 1e-9:
                self.base_xyz[0] = gnx
                self.base_xyz[1] = gny
            self.base_xyz[2] = self._pose_cur['z']
            self.base_yaw += getattr(self, '_pose_dyaw', 0.0) * ds
        fresh = self.data.time - self.cmd_time <= 0.5
        target = self.cmd_vel if fresh else np.zeros(3)
        if self.posture == 'lie':
            target = np.zeros(3)   # 누운 채 보행 불가 (일어나야 이동)
        # 가감속 램프(τ=0.25s) — 명령 속도를 즉시 적용하면 방향 전환마다
        # 카메라가 덜컹거림. 단 ★정지 명령(전부 0)은 즉시 반영 —
        # 정지의 즉시성(안전 이중 경로)은 부드러움보다 우선.
        if not target.any():
            self.vel_smooth[:] = 0.0
        else:
            self.vel_smooth += (target - self.vel_smooth) * min(1.0, dt / 0.25)
        if self.vel_smooth.any():
            c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
            nx = self.base_xyz[0] + (c * self.vel_smooth[0]
                                     - s * self.vel_smooth[1]) * dt
            ny = self.base_xyz[1] + (s * self.vel_smooth[0]
                                     + c * self.vel_smooth[1]) * dt
            # 기하 게이팅: 가구 구역 관통이 늘지 않는 이동만 승인
            # (자유 이동·탈출·구역 경계 미끄러짐 허용, 파고들기 차단).
            # ★얕은 관통(≤6cm)은 항상 허용 — 여유 반경이 겹치는 구석
            # (옷장-벽 사이 등)에서 모든 방향이 '관통 증가'로 계산돼
            # 갇히는 국소 최소점 함정 방지 (복도 끝 못 나오던 실측 사고).
            # 6cm는 실제 가구 면까지 최소 6cm(낮은 가구)~24cm(벽) 여유.
            pen_new = self._zone_pen(nx, ny)
            pen_old = self._zone_pen(self.base_xyz[0], self.base_xyz[1])
            # 얕은 관통(≤6cm) 허용은 '이미 여유 구역 안'일 때만 —
            # 자유 공간→구역 진입은 여전히 경계(0)에서 차단해 가구/벽에
            # 다가붙는 '부딪힘' 인상을 만들지 않으면서, 구석 국소 최소점
            # 함정에선 빠져나올 수 있게 한다.
            if pen_new <= pen_old + 1e-9 or (
                    pen_old > 1e-9 and pen_new <= 0.06):
                self.base_xyz[0] = nx
                self.base_xyz[1] = ny
            elif not self._blocked_logged:
                self.get_logger().warning(
                    f'가구 구역 차단: {getattr(self, "_zone_hit_name", "?")}'
                    f' (이동 게이팅 — 회전은 허용)')
                self._blocked_logged = True
            self.base_yaw += self.vel_smooth[2] * dt

        moving = bool(self.vel_smooth.any())
        if self.hold_base:
            self.data.qpos[0:3] = self.base_xyz
            # 자세 피치(눕기) 합성: R = Rz(yaw) · Ry(pitch)
            hy = self.base_yaw / 2.0
            hp = self._pose_cur['pitch'] / 2.0
            cz, sz = math.cos(hy), math.sin(hy)
            cp, sp = math.cos(hp), math.sin(hp)
            self.data.qpos[3] = cz * cp
            self.data.qpos[4] = -sz * sp
            self.data.qpos[5] = cz * sp
            self.data.qpos[6] = sz * cp
            self.data.qvel[0:6] = 0.0
            if moving:
                # IMU 각속도/가속에 명령이 반영되게 수평 속도만 남긴다
                # (램프 적용된 실효 속도 기준 — IMU도 부드럽게)
                c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
                self.data.qvel[0] = (c * self.vel_smooth[0]
                                     - s * self.vel_smooth[1])
                self.data.qvel[1] = (s * self.vel_smooth[0]
                                     + c * self.vel_smooth[1])
                self.data.qvel[5] = self.vel_smooth[2]
        elif moving:
            c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
            self.data.qvel[0] = c * self.vel_smooth[0] - s * self.vel_smooth[1]
            self.data.qvel[1] = s * self.vel_smooth[0] + c * self.vel_smooth[1]
            self.data.qvel[5] = self.vel_smooth[2]

    # ---------- 카메라 ----------
    def make_camera_info(self, w, h, fovy_deg):
        info = CameraInfo()
        info.width = w
        info.height = h
        fy = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        cx, cy = w / 2.0, h / 2.0
        info.k = [fy, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.p = [fy, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.header.frame_id = CAMERA_NAME
        return info

    def tick_camera(self):
        with self.render_lock:
            with self.state_lock:
                self.renderer.update_scene(self.data, camera=CAMERA_NAME)
            rgb = self.renderer.render()      # (H, W, 3) uint8
        stamp = self.get_clock().now().to_msg()

        buf = io.BytesIO()
        PILImage.fromarray(rgb).save(buf, format='JPEG',
                                     quality=self.jpeg_quality)
        comp = CompressedImage()
        comp.header.stamp = stamp
        comp.header.frame_id = CAMERA_NAME
        comp.format = 'jpeg'
        comp.data = buf.getvalue()
        self.pub_img_comp.publish(comp)

        raw = Image()
        raw.header = comp.header
        raw.height, raw.width = rgb.shape[0], rgb.shape[1]
        raw.encoding = 'rgb8'
        raw.step = rgb.shape[1] * 3
        raw.data = rgb.tobytes()
        self.pub_img_raw.publish(raw)

        self.cam_info.header.stamp = stamp
        self.pub_cam_info.publish(self.cam_info)

    def _warm_worlds(self):
        """모든 월드 예열: 모델 컴파일(락 밖, CPU) + 렌더러 생성(GL 직렬화).

        전환 지연의 두 지연원을 부팅 직후 백그라운드로 선지불한다.
        렌더 품질/주기는 건드리지 않음 — 캐시일 뿐."""
        model_path = self.get_parameter('model_path').value
        time.sleep(3.0)                     # 부팅 안정화 후 시작
        for name in list(WORLDS):
            if not rclpy.ok():
                return
            try:
                if name not in self._world_cache:
                    compiled = load_model_with_camera_and_props(
                        model_path, name)
                    self._world_cache[name] = compiled
                if name not in self._renderer_cache:
                    m = self._world_cache[name][0]
                    with self.render_lock, self.prev_lock:
                        self._renderer_cache[name] = (
                            mujoco.Renderer(m, height=self.cam_size[1],
                                            width=self.cam_size[0]),
                            mujoco.Renderer(m, height=self.prev_size[1],
                                            width=self.prev_size[0]))
                    self.get_logger().info(f'월드 예열 완료: {name}')
            except Exception as e:
                self.get_logger().warning(f'월드 예열 실패({name}): {e}')

    def _preview_loop(self):
        """저해상도 프리뷰를 연속 렌더 → 라이브 영상.

        핵심 트릭: 로봇 자신의 비주얼 메시(geom group 1, 약 97만 정점 =
        렌더 비용의 ~96%)를 프리뷰에서만 제외 → 320x240 기준 18ms/frame.
        1인칭 카메라라 자기 몸은 화면에 거의 안 잡히므로 화질 손실 없음.
        10fps로 페이싱해 물리 틱/메인 카메라와 CPU를 나눠 쓴다."""
        opt = mujoco.MjvOption()
        opt.geomgroup[1] = 0        # 로봇 비주얼 메시 제외 (프리뷰 전용)
        PERIOD = 0.066              # ~15fps 상한 (고해상 렌더가 온디맨드로
                                    # 물러난 덕에 CPU 여유 확보)
        while rclpy.ok():
            t_start = time.monotonic()
            try:
                # render_lock 공유: 온디맨드 640 렌더와의 '동시 렌더'가
                # OSMesa 세그폴트(2회 실측)의 유력 원인 — GL은 직렬화
                with self.render_lock, self.prev_lock:
                    r = self.renderer_prev
                    if r is None:
                        time.sleep(0.1)
                        continue
                    with self.state_lock:
                        r.update_scene(self.data, camera=CAMERA_NAME,
                                       scene_option=opt)
                    rgb = r.render()
                buf = io.BytesIO()
                PILImage.fromarray(rgb).save(buf, format='JPEG', quality=70)
                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = CAMERA_NAME
                msg.format = 'jpeg'
                msg.data = buf.getvalue()
                self.pub_preview.publish(msg)
                time.sleep(max(0.005, PERIOD - (time.monotonic() - t_start)))
            except Exception as e:
                self.get_logger().warning(f'프리뷰 렌더 오류(계속): {e}')
                time.sleep(0.3)

    # ---------- 퍼블리시 ----------
    def stamp_header(self, header):
        t = self.get_clock().now().to_msg()
        header.stamp = t
        header.meas_stamp = t
        header.sequence = self.seq % (2 ** 32)

    def publish_joints(self):
        q = self.data.qpos
        v = self.data.qvel
        tau = self.data.actuator_force
        for g, idxs in self.groups.items():
            msg = JointStateArray()
            self.stamp_header(msg.header)
            for ai in idxs:
                js = JointState()
                js.name = self.joint_names[ai] or f'joint_{ai}'
                js.position = float(q[self.act_qpos_adr[ai]])
                js.velocity = float(v[self.act_qvel_adr[ai]])
                js.effort = float(tau[ai])
                msg.joints.append(js)
            self.pub_joint[g].publish(msg)

    def publish_imu(self):
        # freejoint 상태로 IMU 근사 (chest/torso 동일 값 사용)
        quat = self.data.qpos[3:7]        # w x y z
        angvel = self.data.qvel[3:6]
        linvel = np.array(self.data.qvel[0:3])
        dt = self.model.opt.timestep * self.substeps
        accel = (linvel - self.prev_base_linvel) / dt
        accel[2] += 9.81
        self.prev_base_linvel = linvel

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.orientation.w = float(quat[0])
        msg.orientation.x = float(quat[1])
        msg.orientation.y = float(quat[2])
        msg.orientation.z = float(quat[3])
        msg.angular_velocity.x = float(angvel[0])
        msg.angular_velocity.y = float(angvel[1])
        msg.angular_velocity.z = float(angvel[2])
        msg.linear_acceleration.x = float(accel[0])
        msg.linear_acceleration.y = float(accel[1])
        msg.linear_acceleration.z = float(accel[2])
        for pub in self.pub_imu.values():
            pub.publish(msg)

    def publish_odom(self):
        od = Odometry()
        od.header.stamp = self.get_clock().now().to_msg()
        od.header.frame_id = 'odom'
        od.child_frame_id = 'base_link'
        od.pose.pose.position.x = float(self.data.qpos[0])
        od.pose.pose.position.y = float(self.data.qpos[1])
        od.pose.pose.position.z = float(self.data.qpos[2])
        od.pose.pose.orientation.w = float(self.data.qpos[3])
        od.pose.pose.orientation.x = float(self.data.qpos[4])
        od.pose.pose.orientation.y = float(self.data.qpos[5])
        od.pose.pose.orientation.z = float(self.data.qpos[6])
        od.twist.twist.linear.x = float(self.data.qvel[0])
        od.twist.twist.linear.y = float(self.data.qvel[1])
        od.twist.twist.angular.z = float(self.data.qvel[5])
        self.pub_odom.publish(od)

    def publish_pmu(self):
        mins = float(self.get_parameter('battery_minutes').value)
        elapsed_min = (self.data.time - self.batt_start_time) / 60.0
        frac = max(0.0, 1.0 - elapsed_min / mins)
        msg = PmuState()
        msg.battery_voltage = self.batt_v_empty + frac * (
            self.batt_v_full - self.batt_v_empty)
        msg.battery_pack_voltage = msg.battery_voltage
        msg.battery_current = -3.0 if self.cmd_vel.any() else -1.0
        msg.battery_temperature = 32.0
        self.pub_pmu.publish(msg)


def main():
    from x2_sim.single_instance import claim
    _lock = claim('x2_sim')            # noqa: F841 — 수명 유지용
    # SIGTERM 즉시 종료 — 기본 동작은 rclpy만 반파된 채 프로세스가
    # 살아남아 토픽이 끊긴 반좀비가 됨 (agent_node와 동일 실측 패턴)
    import os as _os
    import signal as _signal
    _signal.signal(_signal.SIGTERM, lambda *_: _os._exit(0))
    rclpy.init()
    node = X2SimNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
