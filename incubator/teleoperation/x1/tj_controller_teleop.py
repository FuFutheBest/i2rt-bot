#!/usr/bin/env python3
"""Quest 3 controller -> TJ dual-arm teleop (move_line / TCP mode).

Adapted from R1_teleoperation/test/controller_to_flexiv_test.py, but:
  - arm control  : flexivrdk.SendCartesianMotionForce  ->  nmxrdk arm.move_line
  - gripper      : 食指扳机(trigger) 连续开合           ->  nmxrdk gripper.set_width
  - head (2 DOF) : 头显 pitch+yaw                        ->  head.move_joint([pitch, yaw])
  - video        : ZED WebRTC                           ->  TJ camera image-mode stream (ngrok)

Controller capture reuses the proven quest3_yam_teleop Vuer JS patch so the
Quest controllers' 6DOF grip pose + buttons/axes come back via HAND_MOVE.

Run INSIDE robot_test container (venv), with ngrok http 8012 on the host:
    source /root/tvenv/bin/activate
    cd /root/incubator/teleoperation/x1
    # dry-run, left arm only, verify axis alignment first:
    python3 tj_controller_teleop.py --arm-side left --dry-run --print-debug
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path
from multiprocessing import Array, Process, Value, shared_memory

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_scipy

# --- quest3 controller capture layer (Vuer JS patch lives here) ---
HERE = Path(__file__).resolve().parent
Q3_DIR = HERE / "Flexiv_VR" / "R1_teleoperation" / "quest3_yam_teleop"
if str(Q3_DIR) not in sys.path:
    sys.path.insert(0, str(Q3_DIR))

from vuer import Vuer                                          # noqa: E402
from vuer.schemas import (                                     # noqa: E402
    DefaultScene, Hands, ImageBackground, WebRTCStereoVideoPlane,
)
from vuer_controller_patch import patch_vuer                  # noqa: E402
from vuer_nocache import install_nocache_patch                # noqa: E402

# --- TJ robot SDK ---
import nmxrdk                                                  # noqa: E402
import nmxrt                                                   # noqa: E402
from nmxrdk.common.robot_side import RobotSide                # noqa: E402
import teleop_home                                            # noqa: E402


# ============================================================
# CONFIG ---- 调这里
# ============================================================
ROBOT_ID = "1e783be2dd870ef5"
HEAD_CAMERA_ID = "CAMID_0"
RESOLUTION = (540, 960)             # (h, w) 单眼（降分辨率减轻图传/编码负载）
MAX_BUTTONS = 16
MAX_AXES = 8

# 手柄坐标系 -> TJ 基坐标系，v_tj = R_QUEST_TO_TJ @ v_quest。
# 先用 R1 默认（操作者面对机器人）。用 --dry-run 对齐后按需改这里。
# WebXR: +X 右, +Y 上, -Z 前。TJ 基座(参考 TCP): +X 前, +Y 左, +Z 上。
# 根据实测端到端行为反推的修正映射(左臂, 与机械臂平行站位)。
# 旧映射现象: 手柄向右->TCP向上, 手柄向上->TCP向左 (轴被旋转了)。
# 期望: 手柄右->TCP右, 手柄上->TCP上, 手柄前->TCP前。
# WebXR 世界系: +X 右, +Y 上, -Z 前。TJ 基座: +X 前, +Y 左, +Z 上。
_R_QUEST_TO_TJ_BASE = np.array([
    [0.0, 0.0, -1.0],   # robot_x(前) = -quest_z
    [0.0, -1.0, 0.0],   # robot_y(左) = -quest_y
    [-1.0, 0.0, 0.0],   # robot_z(上) = -quest_x
], dtype=float)

# 基座前后/上下平面旋转补偿(度): 绕【基座 Z 轴(水平左右轴)】旋转。
# 几何: 仿人设备安装面垂直地面, 基座 Z 平行地面(水平左右), Y 竖直, X 水平前后,
#       但基座 X/Y 在竖直平面里整体转了一个角, 需绕水平的基座 Z 轴转回来。
# 实测标定: -60° 时 左臂"手柄前->TCP前, 手柄上->TCP上"。
# 调法: 若上下/前后还有偏, 在 -60 附近微调(如 -50/-70); 方向反了改正负号。
BASE_FWD_TILT_DEG = -60.0


def _Rz(deg):
    """绕基座 Z(水平左右)轴的旋转矩阵(在前/上竖直平面里旋转 X-Y)。"""
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


# 左臂映射: 先按轴向映射, 再绕基座 Z 把前后/上下平面转正。
R_QUEST_TO_TJ_LEFT = _Rz(BASE_FWD_TILT_DEG) @ _R_QUEST_TO_TJ_BASE

# 右臂与左臂"对装": 安装面平行, 基座 Z(左右轴)共线但相反。
# 实测: 右臂相对左臂"左右正确, 上下反、前后反" -> 即绕基座 X 轴 180°
# (Y、Z 取反, X 不变), 而非绕 Y。要让同一手柄动作在世界系下产生相同的
# TCP 运动, 右臂映射 = Rx(180) @ 左臂映射 = diag(1, -1, -1) @ R_left。
_RX_180 = np.diag([1.0, -1.0, -1.0]).astype(float)
R_QUEST_TO_TJ_RIGHT = _RX_180 @ R_QUEST_TO_TJ_LEFT

# 默认(向后兼容): 单值引用指向左臂。实际按 arm-side 选用(见 ArmController)。
R_QUEST_TO_TJ = R_QUEST_TO_TJ_LEFT

# TCP 自然朝向（手柄 identity 时 TCP 的姿态）。absolute 模式才用。
R_NATURAL = np.array([
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=float)

# move_line 参数
MOVE_LINE_VEL = 0.1                 # m/s
# 夹爪开口(米)：trigger=0 张开, trigger=1 闭合
GRIP_OPEN_M = 0.08
GRIP_CLOSE_M = 0.0

# 头部：两个电机 joint[0]=pitch(上下点头), joint[1]=yaw(左右转头)。
# 把 VR 头显的"抬头/低头"映射到 pitch，"左右转头"映射到 yaw（均为相对初始的增量）。
HEAD_PITCH_SIGN = 1.0              # 上下方向不对就改 -1.0
HEAD_PITCH_SCALE = 1.0
HEAD_YAW_SIGN = -1.0              # 左右方向不对就改回 1.0（实测左右反了，取 -1）
HEAD_YAW_SCALE = 1.0
# 关节【绝对】范围(rad)：最终下发的 j0/j1 会被夹到这里，防止打到机械限位。
HEAD_PITCH_RANGE = (-0.8, 0.95)    # rad，joint[0] pitch 绝对范围
HEAD_YAW_RANGE = (-1.5, 1.5)       # rad，joint[1] yaw 绝对范围
HEAD_VEL = [1.0, 1.0]              # rad/s [pitch, yaw]
HEAD_ACC = [2.0, 2.0]

STATE_WAIT = "WAIT"
STATE_FULL = "FULL"


# ============================================================
# Quaternion / matrix helpers (TJ uses wxyz, scipy uses xyzw)
# ============================================================
def mat_to_tj_quat(R3):
    qxyzw = R_scipy.from_matrix(R3).as_quat()
    return np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])


def tj_quat_to_mat(q_wxyz):
    qxyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return R_scipy.from_quat(qxyzw).as_matrix()


def pose7_to_T(pose7):
    T = np.eye(4)
    T[:3, 3] = pose7[:3]
    T[:3, :3] = tj_quat_to_mat(pose7[3:7])
    return T


def T_to_pose7(T):
    return np.concatenate([T[:3, 3], mat_to_tj_quat(T[:3, :3])])


def _orthonormalize(R3):
    U, _, Vt = np.linalg.svd(R3)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _is_identity_matrix(M, tol=1e-9):
    return np.allclose(M, 0.0, atol=tol) or np.allclose(M, np.eye(4), atol=tol)


def _slerp_step(R_from, R_to, max_step_rad):
    R_delta = R_to @ R_from.T
    ang = np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0))
    if ang <= max_step_rad or ang < 1e-9:
        return R_to.copy()
    rotvec = R_scipy.from_matrix(R_delta).as_rotvec()
    R_step = R_scipy.from_rotvec(rotvec * (max_step_rad / ang)).as_matrix()
    return R_step @ R_from


def view_map_from_head(head_mat):
    """由头显位姿构造"世界系 -> 机器人系"的旋转(视角相对)。

    取头显水平朝向(yaw)为机器人 +X(前),其右手方向取反为 +Y(左),
    世界竖直为 +Z(上)。这样操作方向始终相对"你看的方向",与站姿无关。
    返回 3x3；若头显无数据返回 None。
    """
    R = head_mat[:3, :3]
    f = -R[:, 2]                       # 头显前向(WebXR: -Z 为前)
    fh = np.array([f[0], 0.0, f[2]])   # 投影到水平面
    n = np.linalg.norm(fh)
    if n < 1e-6 or not np.isfinite(n):
        return None
    fh /= n
    up = np.array([0.0, 1.0, 0.0])     # WebXR 世界竖直
    rh = np.cross(fh, up)              # 头显右向(水平)
    rn = np.linalg.norm(rh)
    if rn < 1e-6:
        return None
    rh /= rn
    # robot = M @ d_world: 行0=前, 行1=左(=-右), 行2=上
    return np.array([fh, -rh, up])


def ensure_ok(result, action):
    if isinstance(result, nmxrt.Error):
        raise RuntimeError(f"{action} failed: {result.message}")
    return result


# ============================================================
# Vuer viewer: controller capture (from QuestVrWebRTCViewer) +
# TJ camera image streaming (from OpenTeleVision.main_image)
# ============================================================
class TJControllerViewer:
    def __init__(self, shm_name, img_shape, port=8012, stream_camera=True,
                 cert=None, key=None, webrtc_src=None):
        self.img_shape = img_shape          # (h, 2*w, 3)
        self.img_height = img_shape[0]
        self.img_width = img_shape[1] // 2
        self._shm_name = shm_name
        self._stream_camera = stream_camera  # False=不传机器人相机，保持 VR 透传
        self._webrtc_src = webrtc_src        # 非空=改用 WebRTC 立体拉流(低延迟)

        vuer_kwargs = dict(host="0.0.0.0", port=port,
                           queries=dict(grid=False), queue_len=3)
        if cert and key:
            # 局域网直连 HTTPS（绕过 ngrok，延迟更低）：vuer 自己跑 TLS。
            vuer_kwargs["cert"] = cert
            vuer_kwargs["key"] = key
        self.app = Vuer(**vuer_kwargs)

        self.head_matrix_shared = Array("d", 16, lock=True)
        self.left_controller_matrix_shared = Array("d", 16, lock=True)
        self.right_controller_matrix_shared = Array("d", 16, lock=True)
        self.left_buttons_shared = Array("d", MAX_BUTTONS, lock=True)
        self.right_buttons_shared = Array("d", MAX_BUTTONS, lock=True)
        self.left_axes_shared = Array("d", MAX_AXES, lock=True)
        self.right_axes_shared = Array("d", MAX_AXES, lock=True)
        self.left_buttons_count_shared = Value("i", 0, lock=True)
        self.right_buttons_count_shared = Value("i", 0, lock=True)
        self.left_axes_count_shared = Value("i", 0, lock=True)
        self.right_axes_count_shared = Value("i", 0, lock=True)
        self.left_controller_visible_shared = Value("i", 0, lock=True)
        self.right_controller_visible_shared = Value("i", 0, lock=True)
        self._hm_dbg = 0

        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.spawn(start=False)(self.main_image)
        self.process = Process(target=self.app.run)
        self.process.daemon = True

    # ---------- handlers ----------
    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
        except Exception:
            pass

    async def on_hand_move(self, event, session, fps=60):
        try:
            v = event.value if isinstance(event.value, dict) else {}
            if self._hm_dbg < 5:
                self._hm_dbg += 1
                keys = list(v.keys()) if isinstance(v, dict) else type(v)
                lc = v.get("leftController") if isinstance(v, dict) else None
                rc = v.get("rightController") if isinstance(v, dict) else None
                print(f"[HAND_MOVE #{self._hm_dbg}] keys={keys} "
                      f"hasL={lc is not None} hasR={rc is not None} "
                      f"lbtn={v.get('leftButtons')} rbtn={v.get('rightButtons')}",
                      flush=True)
            self._update_block(
                v, "leftController", "leftButtons", "leftAxes",
                self.left_controller_matrix_shared,
                self.left_buttons_shared, self.left_buttons_count_shared,
                self.left_axes_shared, self.left_axes_count_shared,
                self.left_controller_visible_shared)
            self._update_block(
                v, "rightController", "rightButtons", "rightAxes",
                self.right_controller_matrix_shared,
                self.right_buttons_shared, self.right_buttons_count_shared,
                self.right_axes_shared, self.right_axes_count_shared,
                self.right_controller_visible_shared)
        except Exception as exc:
            print(f"HAND_MOVE parse failed: {exc}", flush=True)

    def _update_block(self, payload, mkey, bkey, akey, m_sh, b_sh, bc_sh,
                      a_sh, ac_sh, vis_sh):
        matrix = payload.get(mkey)
        buttons = payload.get(bkey)
        axes = payload.get(akey)
        if matrix and len(matrix) == 16:
            m_sh[:] = matrix
            vis_sh.value = 1
        else:
            vis_sh.value = 0
        if buttons:
            n = min(len(buttons), MAX_BUTTONS)
            b_sh[:] = list(buttons[:n]) + [0.0] * (MAX_BUTTONS - n)
            bc_sh.value = n
        if axes:
            n = min(len(axes), MAX_AXES)
            a_sh[:] = list(axes[:n]) + [0.0] * (MAX_AXES - n)
            ac_sh.value = n

    # ---------- scene (image stream, dual eye) ----------
    async def main_image(self, session, fps=60):
        existing_shm = shared_memory.SharedMemory(name=self._shm_name)
        img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=existing_shm.buf)
        # 必须先 set DefaultScene(frameloop="always")：手柄采集补丁跑在 Hands 的
        # useFrame 逐帧回调里，只有 frameloop="always" 才会持续触发，否则 HAND_MOVE
        # 收不到任何手柄数据（与参考实现 test_vuer_webrtc_vr.py 对齐）。
        try:
            session.set @ DefaultScene(frameloop="always", frameRate=90)
        except Exception:
            pass
        # Hands 组件挂着才会触发打过补丁的手柄采集（手部网格隐藏）
        try:
            session.upsert @ Hands(fps=fps, stream=True, key="hands",
                                   showLeft=False, showRight=False)
        except Exception:
            pass
        # WebRTC 模式：画面由独立的 tj_camera_webrtc 服务推，这里只挂一次拉流面板。
        # 视频不再经 Vuer 的 JPEG-over-WebSocket，延迟显著降低。
        if self._webrtc_src:
            placed = False
            while True:
                if not placed:
                    try:
                        session.upsert @ WebRTCStereoVideoPlane(
                            src=self._webrtc_src,
                            key="tj-webrtc",
                            aspect=1.778,
                            height=8,
                            position=[0, -1, 3],
                        )
                        placed = True
                        print(f"[viewer] WebRTC plane upserted: {self._webrtc_src}",
                              flush=True)
                    except Exception:
                        # ws 还没连上，下个循环重试
                        pass
                await asyncio.sleep(0.5)
        # 不传机器人相机：只保持会话/手柄采集，画面留给 Quest 透传/默认场景
        if not self._stream_camera:
            while True:
                await asyncio.sleep(0.1)
        while True:
            display_image = img_array
            try:
                session.upsert(
                    [ImageBackground(
                        display_image[:, :self.img_width],
                        format="jpeg", quality=55, key="left-image",
                        interpolate=True, aspect=1.778, height=8,
                        position=[0, -1, 3], layers=1),
                     ImageBackground(
                        display_image[:, self.img_width:],
                        format="jpeg", quality=55, key="right-image",
                        interpolate=True, aspect=1.778, height=8,
                        position=[0, -1, 3], layers=2)],
                    to="bgChildren")
            except Exception:
                # ws 掉线时 upsert 会抛 "Websocket session is missing"，忽略等待重连
                pass
            await asyncio.sleep(0.05)

    def start(self):
        self.process.start()

    # ---------- properties ----------
    def _matrix(self, shared):
        return np.array(shared[:]).reshape(4, 4, order="F")

    @property
    def head_matrix(self):
        return self._matrix(self.head_matrix_shared)

    @property
    def left_controller_matrix(self):
        return self._matrix(self.left_controller_matrix_shared)

    @property
    def right_controller_matrix(self):
        return self._matrix(self.right_controller_matrix_shared)

    @property
    def left_controller_visible(self):
        return bool(self.left_controller_visible_shared.value)

    @property
    def right_controller_visible(self):
        return bool(self.right_controller_visible_shared.value)

    @property
    def left_buttons(self):
        n = int(self.left_buttons_count_shared.value)
        return list(self.left_buttons_shared[:n])

    @property
    def right_buttons(self):
        n = int(self.right_buttons_count_shared.value)
        return list(self.right_buttons_shared[:n])

    def controller(self, hand):
        if hand == "right":
            return (self.right_controller_matrix,
                    self.right_controller_visible, self.right_buttons)
        return (self.left_controller_matrix,
                self.left_controller_visible, self.left_buttons)


# ============================================================
# Button helpers (xr-standard mapping)
# ============================================================
def _btn(btns, idx, thresh=0.5):
    return idx < len(btns) and float(btns[idx]) > thresh


def _engage_pressed(args, btns):
    if args.engage_button == "trigger":
        return _btn(btns, 0)
    if args.engage_button == "a":
        return _btn(btns, 4)
    return _btn(btns, 5)


def _disengage_pressed(args, btns):
    if args.engage_button == "b":
        return False
    return _btn(btns, 5)


def _grip_held(btns):
    return _btn(btns, 1)


def _trigger_val(btns):
    return float(np.clip(btns[0], 0.0, 1.0)) if len(btns) >= 1 else 0.0


# ============================================================
# Per-arm controller
# ============================================================
class ArmController:
    def __init__(self, arm, hand, name, args):
        self.arm = arm
        self.hand = hand
        self.name = name
        self.args = args
        self.engaged = False           # grip 离合是否已捕捉基准
        self.T_ctrl_ref = None
        self.T_tcp_ref = None
        self.T_tcp_cmd = pose7_to_T(self._tcp_pose7())
        self.grip_prev = False
        # 按 arm-side 选用固定坐标系映射(左右臂对装, 矩阵不同)
        self.R_fixed = R_QUEST_TO_TJ_RIGHT if hand == "right" else R_QUEST_TO_TJ_LEFT
        self.R_map = self.R_fixed      # 当前 grip 周期使用的"世界->机器人"映射
        self._first_send = True        # 首次 move_line 打印返回状态(诊断用)
        self._last_send_t = 0.0        # 上次发 move_line 的时间(发送频率限制)
        self._send_interval = 1.0 / max(1.0, getattr(args, "send_hz", 15.0))
        self.p_filt = None             # EMA 平滑后的目标位置
        self.R_filt = None             # SLERP 限速后的目标姿态(与位置平滑对称)
        print(f"[{self.name}] init TCP pos = {self.T_tcp_cmd[:3,3].round(3)}")

    def _tcp_pose7(self):
        s = ensure_ok(self.arm.get_tcp_state(), f"{self.name} get_tcp_state")
        p = s.pose
        return np.array([p.pos.x, p.pos.y, p.pos.z,
                         p.quat.w, p.quat.x, p.quat.y, p.quat.z])

    def update(self, viewer):
        a = self.args
        ctrl_mat, ctrl_vis, btns = viewer.controller(self.hand)
        ctrl_ok = ctrl_vis and not _is_identity_matrix(ctrl_mat)

        # 握住侧握键(grip/squeeze, index 1)=离合按住跟随；松开=暂停。
        grip_active = _grip_held(btns) if not a.no_deadman else True
        grip_edge_on = grip_active and not self.grip_prev
        self.grip_prev = grip_active
        trigger = _trigger_val(btns)

        # 松开 grip -> 立即脱开，下次握住重新捕捉基准(可重新摆位)
        if not grip_active and self.engaged:
            self.engaged = False
            self.T_tcp_cmd = pose7_to_T(self._tcp_pose7())

        # 握住瞬间(且控制器可见) -> 捕捉基准(离合 engage)
        if grip_edge_on and ctrl_ok:
            self.T_ctrl_ref = ctrl_mat.copy()
            self.T_ctrl_ref[:3, :3] = _orthonormalize(self.T_ctrl_ref[:3, :3])
            self.T_tcp_ref = pose7_to_T(self._tcp_pose7())
            self.T_tcp_cmd = self.T_tcp_ref.copy()
            self.p_filt = self.T_tcp_ref[:3, 3].copy()
            self.R_filt = self.T_tcp_ref[:3, :3].copy()
            self.engaged = True
            # 视角相对:用握住瞬间的头显朝向锁定本次映射,与站姿无关
            if not a.fixed_frame:
                M = view_map_from_head(viewer.head_matrix)
                self.R_map = M if M is not None else self.R_fixed
                frame_tag = "view" if M is not None else "fixed(no-head)"
            else:
                self.R_map = self.R_fixed
                frame_tag = "fixed"
            print(f"[{self.name}] clutch engaged @ tcp="
                  f"{self.T_tcp_ref[:3,3].round(3)} frame={frame_tag}")

        # --- build & send ---
        tracking = (self.engaged and grip_active and ctrl_ok
                    and self.T_ctrl_ref is not None)
        pose7_target = None
        if tracking:
            R_now = _orthonormalize(ctrl_mat[:3, :3])
            p_now = ctrl_mat[:3, 3]
            R_ref = self.T_ctrl_ref[:3, :3]
            p_ref = self.T_ctrl_ref[:3, 3]

            Rmap = self.R_map
            if a.lock_orientation:
                R_tcp = self.T_tcp_ref[:3, :3]
            elif a.orientation_mode == "absolute":
                R_tcp = Rmap @ R_now @ Rmap.T @ R_NATURAL
            else:
                R_delta = Rmap @ (R_now @ R_ref.T) @ Rmap.T
                R_tcp = R_delta @ self.T_tcp_ref[:3, :3]

            p_delta = Rmap @ (p_now - p_ref)
            p_tcp = self.T_tcp_ref[:3, 3] + a.pos_scale * p_delta

            # workspace clamp（相对基准的最大偏移，防止走到限位）
            dp = p_tcp - self.T_tcp_ref[:3, 3]
            r = float(np.linalg.norm(dp))
            if r > a.workspace_radius:
                p_tcp = self.T_tcp_ref[:3, 3] + dp * (a.workspace_radius / r)

            # EMA 平滑：抑制手柄跳点。直接给绝对目标，让 move_line 自己规划
            # 平滑轨迹（不再用 6mm 逐步爬行——那会引入明显延迟）。
            if self.p_filt is None:
                self.p_filt = self.T_tcp_ref[:3, 3].copy()
            alpha = float(np.clip(a.smooth, 0.05, 1.0))
            self.p_filt = (1.0 - alpha) * self.p_filt + alpha * p_tcp

            # 姿态平滑：与位置对称，按 max-step-rad 限速 SLERP 到目标，
            # 抑制手柄抖动/突跳，避免机械臂猛转。lock 时 R_tcp=ref，结果仍是 ref。
            if self.R_filt is None:
                self.R_filt = self.T_tcp_ref[:3, :3].copy()
            self.R_filt = _slerp_step(self.R_filt, R_tcp, a.max_step_rad)

            self.T_tcp_cmd = np.eye(4)
            self.T_tcp_cmd[:3, 3] = self.p_filt
            self.T_tcp_cmd[:3, :3] = self.R_filt
            pose7_target = T_to_pose7(self.T_tcp_cmd)

            if a.dry_run:
                pass  # printed by caller via debug
            else:
                # 发送频率限制：太频繁会不断打断 move_line 规划 -> 抽搐
                now_send = time.time()
                if now_send - self._last_send_t >= self._send_interval:
                    self._last_send_t = now_send
                    res = self.arm.move_line(list(pose7_target), a.move_vel, False)
                    if isinstance(res, nmxrt.Error):
                        print(f"[{self.name}] move_line err: {res.message}")
                    elif self._first_send:
                        print(f"[{self.name}] move_line status(repr)={res!r}")
                        self._first_send = False

        return {
            "ctrl_ok": ctrl_ok, "grip": grip_active,
            "tracking": tracking, "trigger": trigger, "target": pose7_target,
        }


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arm-side", choices=("left", "right", "both"), default="left")
    p.add_argument("--dry-run", action="store_true",
                   help="不发 move_line，只打印目标（先验证轴对齐）")
    p.add_argument("--enable-grippers", action="store_true")
    p.add_argument("--enable-head", action="store_true")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--engage-button", choices=("a", "b", "trigger"), default="a")
    p.add_argument("--no-deadman", action="store_true")
    p.add_argument("--orientation-mode", choices=("absolute", "relative"),
                   default="relative")
    p.add_argument("--lock-orientation", action="store_true",
                   help="只平移、锁定 TCP 姿态(IK 更稳，推荐首测用)")
    p.add_argument("--fixed-frame", action="store_true",
                   help="用固定 R_QUEST_TO_TJ(需正对机器人站)；默认视角相对(随头显朝向)")
    p.add_argument("--pos-scale", type=float, default=1.0)
    p.add_argument("--send-hz", type=float, default=15.0,
                   help="move_line 发送频率(Hz)。太高会不断打断规划导致抽搐，默认15")
    p.add_argument("--move-vel", type=float, default=0.3,
                   help="move_line 线速度(m/s)，需足够跟上发送间隔的位移")
    p.add_argument("--smooth", type=float, default=0.4,
                   help="目标位置 EMA 平滑系数(0.05~1)。越小越平滑但越滞后，默认0.4")
    p.add_argument("--max-step-m", type=float, default=0.005)
    p.add_argument("--max-step-rad", type=float, default=0.05)
    p.add_argument("--workspace-radius", type=float, default=0.25)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--port", type=int, default=8012)
    p.add_argument("--skip-patch", action="store_true")
    p.add_argument("--allow-cache", action="store_true")
    p.add_argument("--print-debug", action="store_true")
    p.add_argument("--no-home", action="store_true",
                   help="跳过启动时回到初始位置（默认会先回初始位）")
    p.add_argument("--https", action="store_true",
                   help="局域网直连 HTTPS（绕过 ngrok，延迟更低）。默认用 certs/ 下自签证书。")
    p.add_argument("--cert", default=None, help="TLS 证书路径(默认 certs/cert.pem)")
    p.add_argument("--key", default=None, help="TLS 私钥路径(默认 certs/key.pem)")
    p.add_argument("--webrtc-src", default=None,
                   help="WebRTC 视频流 offer 端点，例如 "
                        "https://<主机IP>:8080/offer。给定后画面走 WebRTC(低延迟)，"
                        "本进程不再读相机/推 JPEG（相机由 tj_camera_webrtc.py 读）。")
    return p.parse_args()


def head_pitch_from_matrix(head_mat):
    """WebXR head 4x4 -> pitch(rad). forward=-R[:,2], +Y up. 抬头为正。"""
    fwd = -head_mat[:3, 2]
    return float(np.arcsin(np.clip(fwd[1], -1.0, 1.0)))


def head_yaw_from_matrix(head_mat):
    """WebXR head 4x4 -> yaw(rad). forward=-R[:,2]，水平面内左右转头。
    正前方(-Z)时 yaw=0；向右转(朝 +X)时 yaw 增大。"""
    fwd = -head_mat[:3, 2]
    return float(np.arctan2(fwd[0], -fwd[2]))


def main():
    args = parse_args()

    # Vuer 补丁必须在子进程启动前应用
    if not args.skip_patch:
        patch_vuer()
    if not args.allow_cache:
        install_nocache_patch()

    res = RESOLUTION
    img_shape = (res[0], 2 * res[1], 3)
    shm = shared_memory.SharedMemory(create=True,
                                     size=int(np.prod(img_shape)) * np.uint8().itemsize)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)

    cert = key = None
    if args.https:
        cert = args.cert or str(HERE / "certs" / "cert.pem")
        key = args.key or str(HERE / "certs" / "key.pem")
    # WebRTC 模式下视频由 tj_camera_webrtc.py 独占相机，本进程不再读相机/推 JPEG。
    use_webrtc = bool(args.webrtc_src)
    viewer = TJControllerViewer(shm.name, img_shape, port=args.port,
                                stream_camera=not args.no_camera,
                                cert=cert, key=key,
                                webrtc_src=args.webrtc_src)
    viewer.start()
    if use_webrtc:
        print(f"[init] 视频走 WebRTC: {args.webrtc_src} "
              f"(确保 tj_camera_webrtc.py 已在跑)")
    if cert:
        print(f"[init] Vuer HTTPS 直连。Quest 打开 https://<主机LAN_IP>:{args.port} "
              f"-> 接受证书 -> Enter VR")
    else:
        print(f"[init] Vuer 起来了。host 上 `ngrok http {args.port}`，"
              f"用 Quest 打开 ngrok 的 https 地址 -> Enter VR")

    robot = nmxrdk.RobotClient(ROBOT_ID)
    cam = robot.camera(HEAD_CAMERA_ID) if (not args.no_camera and not use_webrtc) else None

    # 相机抓取放到独立线程，绝不阻塞控制循环（延迟/丢帧的主因）
    cam_stop = threading.Event()

    def camera_loop():
        while not cam_stop.is_set():
            try:
                rgb_frame, _ = cam.get_rgbd(decode=True)
                if not isinstance(rgb_frame, nmxrt.Error):
                    rgb = rgb_frame.get("data")
                    if rgb is not None:
                        rgb = cv2.resize(rgb, (res[1], res[0]))
                        if rgb.shape[2] == 3:
                            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                        np.copyto(img_array, np.hstack((rgb, rgb)))
            except Exception:
                pass
            time.sleep(0.01)

    cam_thread = None
    if cam is not None:
        cam_thread = threading.Thread(target=camera_loop, daemon=True)
        cam_thread.start()

    # 启动时先回到统一的初始位置（双臂 Joint + 头部舵机），再进入遥操作。
    if not args.no_home:
        teleop_home.go_home(robot, arm_side="both", do_head=True, blocking=True)

    arms = []
    if args.arm_side in ("left", "both"):
        arms.append(ArmController(robot.arm(RobotSide.LEFT), "left", "left", args))
    if args.arm_side in ("right", "both"):
        arms.append(ArmController(robot.arm(RobotSide.RIGHT), "right", "right", args))

    grip_l = grip_r = head = None
    if args.enable_grippers:
        grip_l = robot.gripper(RobotSide.LEFT)
        grip_r = robot.gripper(RobotSide.RIGHT)
    head_pitch_ref = None
    head_yaw_ref = None
    head_joint_ref = None
    if args.enable_head:
        head = robot.head()
        ensure_ok(head.enable(), "head.enable")
        st = ensure_ok(head.get_joint_state(), "head.get_joint_state")
        head_joint_ref = [j.pos for j in st.joints]
        print(f"[head] enabled, joint ref = {np.round(head_joint_ref,3)}")

    last_grip = {"left": None, "right": None}
    last_print = 0.0
    last_head_dbg = 0.0
    dt = args.dt
    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"[{mode}] arm={args.arm_side} grip={args.enable_grippers} "
          f"head={args.enable_head} engage={args.engage_button} "
          f"deadman={not args.no_deadman}")

    try:
        while True:
            t0 = time.time()

            infos = {}
            for ctrl in arms:
                infos[ctrl.name] = ctrl.update(viewer)

            # ---- grippers：食指扳机(button 0)连续控制夹爪开合 ----
            # 扳机 0 -> 全开(GRIP_OPEN_M)，扳机 1 -> 全闭(GRIP_CLOSE_M)，中间按比例。
            # 只要手柄可见就响应(不依赖移动离合)，按多少夹多少。
            if args.enable_grippers and not args.dry_run:
                for ctrl in arms:
                    info = infos[ctrl.name]
                    if info["ctrl_ok"]:
                        width = GRIP_OPEN_M - info["trigger"] * (GRIP_OPEN_M - GRIP_CLOSE_M)
                        if last_grip[ctrl.name] is None or abs(width - last_grip[ctrl.name]) > 0.003:
                            g = grip_l if ctrl.name == "left" else grip_r
                            g.set_width(width)
                            last_grip[ctrl.name] = width

            # ---- head (pitch 上下 + yaw 左右) ----
            if args.enable_head:
                hm = viewer.head_matrix
                pitch = head_pitch_from_matrix(hm)
                yaw = head_yaw_from_matrix(hm)
                # 头显有数据后首帧捕捉参考姿态(此后都按相对增量控制)
                if head_pitch_ref is None and (pitch != 0.0 or yaw != 0.0):
                    head_pitch_ref = pitch
                    head_yaw_ref = yaw
                    print(f"[head] ref captured pitch={pitch:.3f} yaw={yaw:.3f}")
                if head_pitch_ref is not None:
                    dp = HEAD_PITCH_SIGN * HEAD_PITCH_SCALE * (pitch - head_pitch_ref)
                    dy = HEAD_YAW_SIGN * HEAD_YAW_SCALE * (yaw - head_yaw_ref)
                    # 绝对目标 = 初始关节 + 增量，再按关节绝对范围夹取
                    j0 = float(np.clip(head_joint_ref[0] + dp, *HEAD_PITCH_RANGE))  # pitch
                    j1 = float(np.clip(head_joint_ref[1] + dy, *HEAD_YAW_RANGE))    # yaw
                    if not args.dry_run:
                        try:
                            head.move_joint([j0, j1], HEAD_VEL, HEAD_ACC, False, 1000, {})
                        except Exception as exc:
                            print(f"[head] move_joint failed: {exc}", flush=True)

                    # 头部链路实时诊断：头显原始角 -> 增量 -> 下发 -> 舵机反馈
                    now_h = time.time()
                    if now_h - last_head_dbg > 0.5:
                        fb = "?"
                        try:
                            st = head.get_joint_state()
                            if not isinstance(st, nmxrt.Error):
                                fb = [round(j.pos, 3) for j in st.joints]
                        except Exception:
                            pass
                        print(
                            f"[head] raw pitch={np.degrees(pitch):6.1f} yaw={np.degrees(yaw):6.1f} (deg) | "
                            f"dp={dp:+.3f} dy={dy:+.3f} | cmd j0(pitch)={j0:+.3f} j1(yaw)={j1:+.3f} | "
                            f"fb={fb}",
                            flush=True)
                        last_head_dbg = now_h

            # ---- camera 抓取已移到独立线程(camera_loop)，此处不再阻塞 ----

            # ---- debug ----
            now = time.time()
            if args.print_debug and now - last_print > 1.0:
                for ctrl in arms:
                    info = infos[ctrl.name]
                    c = "C" if info["ctrl_ok"] else "-"
                    g = "G" if info["grip"] else "-"
                    t = "T" if info["tracking"] else "."
                    tgt = info["target"]
                    cmat, _vis, btns = viewer.controller(ctrl.hand)
                    pos = cmat[:3, 3]
                    # 只看真实按键(0~5)，过滤 7+ 的触摸电容噪声
                    active = ",".join(f"{i}:{v:.2f}"
                                      for i, v in enumerate(btns)
                                      if i < 6 and v > 0.5)
                    eng = "E" if ctrl.engaged else "-"
                    line = (f"[{ctrl.name}] {c}{g}{t}{eng} "
                            f"pos={pos.round(3)} btn[{active}]")
                    if tgt is not None:
                        dref = tgt[:3] - ctrl.T_tcp_ref[:3, 3]
                        line += (f" tgt={tgt[:3].round(3)} d_ref={dref.round(3)}")
                    print(line)
                last_print = now

            time.sleep(max(0.0, dt - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\n[exit] stopping")
    finally:
        cam_stop.set()
        if cam_thread is not None:
            cam_thread.join(timeout=1.0)
        for ctrl in arms:
            try:
                ctrl.arm.move_line  # no-op; arms stop when no new cmd
            except Exception:
                pass
        if head is not None:
            try:
                head.stop()
                head.disable()
            except Exception:
                pass
        shm.close()
        shm.unlink()
        print("[exit] done")


if __name__ == "__main__":
    main()

