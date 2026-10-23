#!/usr/bin/env python3
"""
X1 (TJ / nmxrdk) Quest 3 dual-arm VR teleoperator device.

Adapted from the incubator X1 controller script to fit
SuperInference's SHM device architecture. The proven X1 capture + coordinate
mapping + clutch / EMA / SLERP logic is kept verbatim; the only structural
change is that, instead of calling ``nmxrdk arm.move_line`` directly, this
device writes an absolute end-effector target to its data SHM, consumed by
``X1Robot`` (``devices/robot.py``) in ``eef_abs`` mode.

Control output (data SHM ``X1VuerTeleoperator_<id>_data``), 18D float64:
    [L_xyz(3), L_quat_xyzw(4), L_grip(1),
     R_xyz(3), R_quat_xyzw(4), R_grip(1),
     head_pitch(1), head_yaw(1)]
  - quaternion in scipy (x, y, z, w) order
  - grip in [0, 1]: 0=open, 1=closed
  - head angles are absolute targets in rad (clamped by X1Robot)

The TCP clutch anchor + head joint reference are read from the robot state SHM
(``robot_device_name`` / ``robot_device_class_id``, 30D layout published by X1Robot),
so X1Robot itself stays decoupled from this device.

Video is intentionally NOT handled here: run the existing
``incubator/teleoperation/x1/tj_camera_webrtc.py`` separately and point
``webrtc_src`` at it (low-latency FPV), or run with no camera plane.

Author: Zixi Ying
"""

from __future__ import annotations

import asyncio
import sys
import time
import multiprocessing as mp
from multiprocessing import Array, Value, shared_memory
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from devices.base import BaseDevice
from devices.config import StateFeedbackConfig
from utils.logger_config import logger
from utils.time_control import precise_loop_timing
from utils.shm_utils import (
    DEVICE_HEADER_SIZE,
    BUFFER_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    connect_to_device_shm,
    get_dtype,
    unpack_buffer_header,
    unpack_device_header,
    unpack_frame_header,
)

# Path to the vendored Vuer controller-capture patches (shared with the X1 scripts).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_Q3_DIR = (
    _PROJECT_ROOT
    / "scripts"
    / "VR_Teleoperation"
    / "Flexiv_VR"
    / "R1_teleoperation"
    / "quest3_yam_teleop"
)

MAX_BUTTONS = 16
MAX_AXES = 8


# ============================================================
# Coordinate frames (ported verbatim from tj_controller_teleop.py)
# ============================================================
# WebXR world: +X right, +Y up, -Z forward. TJ base: +X fwd, +Y left, +Z up.
_R_QUEST_TO_TJ_BASE = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=float,
)

# Compensation rotation about the base Z axis; -60 deg calibrated for the left arm.
BASE_FWD_TILT_DEG = -60.0


def _Rz(deg: float) -> np.ndarray:
    """Rotation about the base Z (horizontal left-right) axis."""
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


R_QUEST_TO_TJ_LEFT = _Rz(BASE_FWD_TILT_DEG) @ _R_QUEST_TO_TJ_BASE
# Right arm is mounted mirrored: Rx(180) @ left mapping.
_RX_180 = np.diag([1.0, -1.0, -1.0]).astype(float)
R_QUEST_TO_TJ_RIGHT = _RX_180 @ R_QUEST_TO_TJ_LEFT

# TCP natural orientation (used by absolute orientation mode only).
R_NATURAL = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)

# Head mapping signs/scales (ported from tj_controller_teleop.py).
HEAD_PITCH_SIGN = 1.0
HEAD_PITCH_SCALE = 1.0
HEAD_YAW_SIGN = -1.0
HEAD_YAW_SCALE = 1.0


# ============================================================
# Quaternion / matrix helpers (TJ uses wxyz; scipy uses xyzw)
# ============================================================
def mat_to_tj_quat(R3: np.ndarray) -> np.ndarray:
    """3x3 rotation -> [qw, qx, qy, qz] (wxyz)."""
    qxyzw = R_scipy.from_matrix(R3).as_quat()
    return np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])


def tj_quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    """[qw, qx, qy, qz] (wxyz) -> 3x3 rotation."""
    qxyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return R_scipy.from_quat(qxyzw).as_matrix()


def pose7_wxyz_to_T(pose7: np.ndarray) -> np.ndarray:
    """[x,y,z, qw,qx,qy,qz] -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, 3] = pose7[:3]
    T[:3, :3] = tj_quat_to_mat(pose7[3:7])
    return T


def T_to_pos_quat_xyzw(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 -> (pos[3], quat_xyzw[4]) for the scipy/data-SHM convention."""
    pos = T[:3, 3].copy()
    quat_xyzw = R_scipy.from_matrix(T[:3, :3]).as_quat()
    return pos, quat_xyzw


def _orthonormalize(R3: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R3)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def _is_identity_matrix(M: np.ndarray, tol: float = 1e-9) -> bool:
    return np.allclose(M, 0.0, atol=tol) or np.allclose(M, np.eye(4), atol=tol)


def _slerp_step(R_from: np.ndarray, R_to: np.ndarray, max_step_rad: float) -> np.ndarray:
    R_delta = R_to @ R_from.T
    ang = np.arccos(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0))
    if ang <= max_step_rad or ang < 1e-9:
        return R_to.copy()
    rotvec = R_scipy.from_matrix(R_delta).as_rotvec()
    R_step = R_scipy.from_rotvec(rotvec * (max_step_rad / ang)).as_matrix()
    return R_step @ R_from


def view_map_from_head(head_mat: np.ndarray) -> Optional[np.ndarray]:
    """Build a "world -> robot" rotation from the headset yaw (view-relative)."""
    R = head_mat[:3, :3]
    f = -R[:, 2]
    fh = np.array([f[0], 0.0, f[2]])
    n = np.linalg.norm(fh)
    if n < 1e-6 or not np.isfinite(n):
        return None
    fh /= n
    up = np.array([0.0, 1.0, 0.0])
    rh = np.cross(fh, up)
    rn = np.linalg.norm(rh)
    if rn < 1e-6:
        return None
    rh /= rn
    return np.array([fh, -rh, up])


def head_pitch_from_matrix(head_mat: np.ndarray) -> float:
    """WebXR head 4x4 -> pitch (rad). forward=-R[:,2], +Y up; look up is positive."""
    fwd = -head_mat[:3, 2]
    return float(np.arcsin(np.clip(fwd[1], -1.0, 1.0)))


def head_yaw_from_matrix(head_mat: np.ndarray) -> float:
    """WebXR head 4x4 -> yaw (rad). forward=-Z is 0; turning right increases yaw."""
    fwd = -head_mat[:3, 2]
    return float(np.arctan2(fwd[0], -fwd[2]))


# ============================================================
# Button helpers (xr-standard mapping)
# ============================================================
def _btn(btns, idx: int, thresh: float = 0.5) -> bool:
    return idx < len(btns) and float(btns[idx]) > thresh


def _grip_held(btns) -> bool:
    return _btn(btns, 1)


def _trigger_val(btns) -> float:
    return float(np.clip(btns[0], 0.0, 1.0)) if len(btns) >= 1 else 0.0


# ============================================================
# Vuer viewer: Quest controller capture (control-only; video via WebRTC/none)
# ============================================================
class _X1ControllerViewer:
    """Vuer WebXR viewer that captures Quest controller/head poses + buttons.

    Video is not streamed from here: pass ``webrtc_src`` to place a low-latency
    WebRTC plane (served by tj_camera_webrtc.py), otherwise only a session is
    kept alive so HAND_MOVE controller capture keeps firing.
    """

    def __init__(
        self,
        port: int = 8012,
        cert: Optional[str] = None,
        key: Optional[str] = None,
        webrtc_src: Optional[str] = None,
        skip_patch: bool = False,
        allow_cache: bool = False,
    ) -> None:
        self._port = port
        self._webrtc_src = webrtc_src

        # Vuer / vendored patches are only importable inside the docker container,
        # so import them lazily here (this runs in the device child process).
        if str(_Q3_DIR) not in sys.path:
            sys.path.insert(0, str(_Q3_DIR))
        from vuer import Vuer
        from vuer.schemas import DefaultScene, Hands, WebRTCStereoVideoPlane

        # The Vuer JS patch (controller HAND_MOVE capture) MUST be applied before
        # the viewer subprocess is forked in start(); apply it here in the device
        # process so the monkeypatched module state is inherited by the fork.
        from vuer_controller_patch import patch_vuer
        from vuer_nocache import install_nocache_patch

        if not skip_patch:
            patch_vuer()
        if not allow_cache:
            install_nocache_patch()

        self._DefaultScene = DefaultScene
        self._Hands = Hands
        self._WebRTCStereoVideoPlane = WebRTCStereoVideoPlane

        vuer_kwargs = dict(host="0.0.0.0", port=port, queries=dict(grid=False), queue_len=3)
        if cert and key:
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
        self.hand_move_count_shared = Value("i", 0, lock=True)

        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.spawn(start=False)(self.main_session)
        # Use a fork context so the Vuer app (which contains non-picklable
        # aiohttp FrozenList objects) is inherited via fork instead of pickled;
        # this works regardless of the parent process's start method.
        self._mp_ctx = mp.get_context("fork")
        self.process = self._mp_ctx.Process(target=self.app.run)
        self.process.daemon = True

    # ---------- handlers ----------
    async def on_cam_move(self, event, session, fps=60) -> None:
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
        except Exception:
            pass

    async def on_hand_move(self, event, session, fps=60) -> None:
        try:
            v = event.value if isinstance(event.value, dict) else {}
            self.hand_move_count_shared.value += 1
            self._update_block(
                v, "leftController", "leftButtons", "leftAxes",
                self.left_controller_matrix_shared,
                self.left_buttons_shared, self.left_buttons_count_shared,
                self.left_axes_shared, self.left_axes_count_shared,
                self.left_controller_visible_shared,
            )
            self._update_block(
                v, "rightController", "rightButtons", "rightAxes",
                self.right_controller_matrix_shared,
                self.right_buttons_shared, self.right_buttons_count_shared,
                self.right_axes_shared, self.right_axes_count_shared,
                self.right_controller_visible_shared,
            )
        except Exception as exc:
            logger.warning(f"HAND_MOVE parse failed: {exc}")

    def _update_block(self, payload, mkey, bkey, akey, m_sh, b_sh, bc_sh,
                      a_sh, ac_sh, vis_sh) -> None:
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

    # ---------- scene (control-only session) ----------
    async def main_session(self, session, fps=60) -> None:
        # frameloop="always" is required so the patched Hands useFrame callback
        # keeps firing HAND_MOVE; without it no controller data arrives.
        try:
            session.set @ self._DefaultScene(frameloop="always", frameRate=90)
        except Exception:
            pass
        try:
            session.upsert @ self._Hands(
                fps=fps, stream=True, key="hands", showLeft=False, showRight=False
            )
        except Exception:
            pass
        if self._webrtc_src:
            placed = False
            while True:
                if not placed:
                    try:
                        session.upsert @ self._WebRTCStereoVideoPlane(
                            src=self._webrtc_src, key="x1-webrtc",
                            aspect=1.778, height=8, position=[0, -1, 3],
                        )
                        placed = True
                        logger.info(f"[viewer] WebRTC plane upserted: {self._webrtc_src}")
                    except Exception:
                        pass
                await asyncio.sleep(0.5)
        else:
            while True:
                await asyncio.sleep(0.1)

    def start(self) -> None:
        self.process.start()

    def stop(self) -> None:
        try:
            if self.process.is_alive():
                self.process.terminate()
        except Exception:
            pass

    # ---------- properties ----------
    def _matrix(self, shared) -> np.ndarray:
        return np.array(shared[:]).reshape(4, 4, order="F")

    @property
    def head_matrix(self) -> np.ndarray:
        return self._matrix(self.head_matrix_shared)

    @property
    def left_controller_matrix(self) -> np.ndarray:
        return self._matrix(self.left_controller_matrix_shared)

    @property
    def right_controller_matrix(self) -> np.ndarray:
        return self._matrix(self.right_controller_matrix_shared)

    @property
    def left_controller_visible(self) -> bool:
        return bool(self.left_controller_visible_shared.value)

    @property
    def right_controller_visible(self) -> bool:
        return bool(self.right_controller_visible_shared.value)

    @property
    def left_buttons(self):
        n = int(self.left_buttons_count_shared.value)
        return list(self.left_buttons_shared[:n])

    @property
    def right_buttons(self):
        n = int(self.right_buttons_count_shared.value)
        return list(self.right_buttons_shared[:n])

    @property
    def hand_move_count(self) -> int:
        return int(self.hand_move_count_shared.value)

    def controller(self, hand: str):
        if hand == "right":
            return (self.right_controller_matrix,
                    self.right_controller_visible, self.right_buttons)
        return (self.left_controller_matrix,
                self.left_controller_visible, self.left_buttons)


# ============================================================
# Per-arm controller (clutch + mapping + smoothing; outputs target T, no SDK)
# ============================================================
class _X1ArmController:
    """Ported from tj_controller_teleop.ArmController but SDK-free.

    ``update(viewer, tcp7_wxyz)`` consumes the current measured TCP (read from
    the robot state SHM by the device) instead of querying nmxrdk directly, and
    returns the commanded TCP as a 4x4 transform plus the gripper target.
    """

    def __init__(self, hand: str, name: str, args: SimpleNamespace) -> None:
        self.hand = hand
        self.name = name
        self.args = args
        self.engaged = False
        self.T_ctrl_ref: Optional[np.ndarray] = None
        self.T_tcp_ref: Optional[np.ndarray] = None
        self.T_tcp_cmd: Optional[np.ndarray] = None
        self.grip_prev = False
        self.last_gripper = 0.0  # 0=open, 1=closed
        self.R_fixed = R_QUEST_TO_TJ_RIGHT if hand == "right" else R_QUEST_TO_TJ_LEFT
        self.R_map = self.R_fixed
        self.p_filt: Optional[np.ndarray] = None
        self.R_filt: Optional[np.ndarray] = None

    def update(self, viewer: _X1ControllerViewer, tcp7_wxyz: Optional[np.ndarray]) -> Dict[str, Any]:
        a = self.args
        ctrl_mat, ctrl_vis, btns = viewer.controller(self.hand)
        ctrl_ok = ctrl_vis and not _is_identity_matrix(ctrl_mat)

        # No robot state yet: cannot anchor; report nothing.
        if tcp7_wxyz is None:
            return {"ctrl_ok": ctrl_ok, "tracking": False,
                    "trigger": _trigger_val(btns), "target_T": None}

        grip_active = _grip_held(btns) if not a.no_deadman else True
        grip_edge_on = grip_active and not self.grip_prev
        self.grip_prev = grip_active
        trigger = _trigger_val(btns)

        # Release grip -> disengage and re-pin command to current TCP.
        if not grip_active and self.engaged:
            self.engaged = False
            self.T_tcp_cmd = pose7_wxyz_to_T(tcp7_wxyz)

        # Initialize the held command to current TCP until first engage.
        if self.T_tcp_cmd is None:
            self.T_tcp_cmd = pose7_wxyz_to_T(tcp7_wxyz)

        # Grip rising edge (and controller visible) -> capture clutch references.
        if grip_edge_on and ctrl_ok:
            self.T_ctrl_ref = ctrl_mat.copy()
            self.T_ctrl_ref[:3, :3] = _orthonormalize(self.T_ctrl_ref[:3, :3])
            self.T_tcp_ref = pose7_wxyz_to_T(tcp7_wxyz)
            self.T_tcp_cmd = self.T_tcp_ref.copy()
            self.p_filt = self.T_tcp_ref[:3, 3].copy()
            self.R_filt = self.T_tcp_ref[:3, :3].copy()
            self.engaged = True
            if not a.fixed_frame:
                M = view_map_from_head(viewer.head_matrix)
                self.R_map = M if M is not None else self.R_fixed
                frame_tag = "view" if M is not None else "fixed(no-head)"
            else:
                self.R_map = self.R_fixed
                frame_tag = "fixed"
            logger.info(f"[{self.name}] clutch engaged @ tcp="
                        f"{self.T_tcp_ref[:3, 3].round(3)} frame={frame_tag}")

        tracking = (self.engaged and grip_active and ctrl_ok
                    and self.T_ctrl_ref is not None)
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

            # Workspace clamp relative to the engage anchor.
            dp = p_tcp - self.T_tcp_ref[:3, 3]
            r = float(np.linalg.norm(dp))
            if r > a.workspace_radius:
                p_tcp = self.T_tcp_ref[:3, 3] + dp * (a.workspace_radius / r)

            # Position EMA smoothing.
            if self.p_filt is None:
                self.p_filt = self.T_tcp_ref[:3, 3].copy()
            alpha = float(np.clip(a.smooth, 0.05, 1.0))
            self.p_filt = (1.0 - alpha) * self.p_filt + alpha * p_tcp

            # Orientation SLERP rate limit.
            if self.R_filt is None:
                self.R_filt = self.T_tcp_ref[:3, :3].copy()
            self.R_filt = _slerp_step(self.R_filt, R_tcp, a.max_step_rad)

            self.T_tcp_cmd = np.eye(4)
            self.T_tcp_cmd[:3, 3] = self.p_filt
            self.T_tcp_cmd[:3, :3] = self.R_filt

            # Gripper: trigger pressed = close (1), released = open (0).
            self.last_gripper = trigger
        elif ctrl_ok:
            # Not tracking but controller visible: keep gripper responsive.
            self.last_gripper = trigger

        return {"ctrl_ok": ctrl_ok, "tracking": tracking,
                "trigger": trigger, "target_T": self.T_tcp_cmd}


# ============================================================
# Device
# ============================================================
class X1VuerTeleoperator(BaseDevice):
    """Quest 3 (Vuer / WebXR) dual-arm VR teleoperator for the X1 (nmxrdk) robot."""

    def __init__(
        self,
        device_class_id: int = 0,
        device_name: Optional[str] = None,
        *,
        state_feedback: StateFeedbackConfig | Mapping[str, Any],
        # --- robot state SHM (clutch TCP anchor + head joint ref) ---
        robot_device_name: str = "X1Robot",
        robot_device_class_id: int = 1,
        # --- Quest / Vuer viewer ---
        port: int = 8012,
        webrtc_src: str = "",
        https: bool = True,
        cert_file: str = "incubator/teleoperation/x1/certs/cert.pem",
        key_file: str = "incubator/teleoperation/x1/certs/key.pem",
        skip_patch: bool = False,
        allow_cache: bool = False,
        # --- arm selection + following params (tj defaults) ---
        arm_side: str = "both",
        orientation_mode: str = "relative",
        lock_orientation: bool = False,
        fixed_frame: bool = False,
        no_deadman: bool = False,
        pos_scale: float = 1.0,
        smooth: float = 0.4,
        max_step_rad: float = 0.05,
        workspace_radius: float = 0.25,
        # --- head following ---
        enable_head: bool = True,
        # --- optional gripper output ---
        enable_grippers: bool = True,
        # --- episode recording trigger (VR buttons -> signal files) ---
        enable_episode_trigger: bool = False,
        episode_signal_pattern: str = "/tmp/superinference_episode_{signal}",
        episode_discard_signal: str = "/tmp/superinference_episode_discard",
        episode_button_hand: str = "right",
        episode_toggle_button: int = 4,   # A (right) / X (left)
        episode_discard_button: int = 5,  # B (right) / Y (left)
        **kwargs,
    ) -> None:
        super().__init__(
            device_class_id=device_class_id,
            device_name=device_name or "X1VuerTeleoperator",
            state_feedback=state_feedback,
        )
        self.device_name = device_name or "X1VuerTeleoperator"
        self.shared_memory_name = f"{self.device_name}_{self.device_class_id}_data"

        self.robot_device_name = robot_device_name
        self.robot_device_class_id = robot_device_class_id
        self.robot_state_shm = None

        self.port = port
        self.webrtc_src = webrtc_src or None
        self.https = https
        self.cert_file = cert_file
        self.key_file = key_file
        self.skip_patch = skip_patch
        self.allow_cache = allow_cache

        self.arm_side = arm_side
        self.enable_left = arm_side in ("left", "both")
        self.enable_right = arm_side in ("right", "both")
        self.enable_head = enable_head
        self.enable_grippers = enable_grippers
        self._output_layout, self._head_output_slice = self._parse_output_format()

        # Following params bundled for the per-arm controllers.
        self._args = SimpleNamespace(
            orientation_mode=orientation_mode,
            lock_orientation=lock_orientation,
            fixed_frame=fixed_frame,
            no_deadman=no_deadman,
            pos_scale=pos_scale,
            smooth=smooth,
            max_step_rad=max_step_rad,
            workspace_radius=workspace_radius,
        )

        # Episode recording trigger (VR buttons write the saver's signal files).
        self.enable_episode_trigger = enable_episode_trigger
        self.episode_signal_pattern = episode_signal_pattern
        self.episode_discard_signal = episode_discard_signal
        self.episode_button_hand = episode_button_hand
        self.episode_toggle_button = int(episode_toggle_button)
        self.episode_discard_button = int(episode_discard_button)
        self._is_recording = False
        self._toggle_prev = False
        self._discard_prev = False

        self._viewer: Optional[_X1ControllerViewer] = None
        self._left = _X1ArmController("left", "left", self._args)
        self._right = _X1ArmController("right", "right", self._args)

        # Head reference state (captured on first valid head data).
        self._head_pitch_ref: Optional[float] = None
        self._head_yaw_ref: Optional[float] = None
        self._head_joint_ref: Optional[np.ndarray] = None

        self._diag_last_t = 0.0
        self._diag_last_hand_move = 0

        logger.info(
            f"X1VuerTeleoperator initialized: data SHM={self.shared_memory_name}, "
            f"arm_side={self.arm_side}, head={self.enable_head}, "
            f"grippers={self.enable_grippers}, "
            f"robot anchor={self.robot_device_name}_{self.robot_device_class_id}, port={self.port}"
        )

    def _parse_output_format(
        self,
    ) -> tuple[tuple[dict[str, Any], ...], slice | None]:
        """Build output slices from the structured state-feedback primitives."""
        primitives = self.state_feedback_config.format.primitives
        primitive_index = 0
        value_offset = 0
        arm_layouts: list[dict[str, Any]] = []

        def consume(expected_names: set[str], description: str):
            nonlocal primitive_index, value_offset
            if primitive_index >= len(primitives):
                raise ValueError(
                    f"X1VuerTeleoperator state_feedback.format is missing {description}"
                )
            primitive = primitives[primitive_index]
            if primitive.name not in expected_names:
                raise ValueError(
                    "X1VuerTeleoperator state_feedback.format expected "
                    f"{description}, got {primitive.name}"
                )
            primitive_slice = slice(value_offset, value_offset + primitive.width)
            primitive_index += 1
            value_offset += primitive.width
            return primitive, primitive_slice

        for side in ("left", "right"):
            _, position_slice = consume({"xyz"}, f"{side} xyz")
            quaternion, quaternion_slice = consume(
                {"qwqxqyqz", "qxqyqzqw"}, f"{side} quaternion"
            )
            gripper_slice = None
            if (
                primitive_index < len(primitives)
                and primitives[primitive_index].name == "gripper"
            ):
                _, gripper_slice = consume({"gripper"}, f"{side} gripper")
            arm_layouts.append(
                {
                    "position": position_slice,
                    "quaternion": quaternion_slice,
                    "quaternion_order": quaternion.quaternion_order,
                    "gripper": gripper_slice,
                }
            )

        head_slice = None
        if primitive_index < len(primitives):
            _, head_slice = consume({"j1-2"}, "head j1-2")
        if primitive_index != len(primitives):
            remaining = [primitive.name for primitive in primitives[primitive_index:]]
            raise ValueError(
                "X1VuerTeleoperator state_feedback.format contains unsupported "
                f"trailing primitives: {remaining}"
            )
        if value_offset != self.state_feedback_config.format.dimension:
            raise ValueError(
                "X1VuerTeleoperator state feedback layout is internally inconsistent"
            )
        return tuple(arm_layouts), head_slice

    # ------------------------------------------------------------------
    # Robot state SHM (clutch anchor + head joint reference)
    # ------------------------------------------------------------------
    def _connect_robot_state_shm(self) -> None:
        if self.robot_state_shm is not None:
            return
        try:
            self.robot_state_shm = connect_to_device_shm(
                device_name=self.robot_device_name,
                device_class_id=self.robot_device_class_id,
                read_only=True,
            )
        except FileNotFoundError:
            self.robot_state_shm = None
        except Exception as exc:
            logger.warning(f"Failed to connect robot state SHM: {exc}")
            self.robot_state_shm = None

    def _read_latest_robot_state(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (left_tcp7_wxyz, right_tcp7_wxyz, head_j2) from the 30D state."""
        self._connect_robot_state_shm()
        if self.robot_state_shm is None:
            return None, None, None

        buf = self.robot_state_shm.buf
        device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
        buffer_header = unpack_buffer_header(
            buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
        )
        if buffer_header["current_frames_count"] == 0:
            return None, None, None

        buffer_size = buffer_header["buffer_size"]
        write_index = buffer_header["write_index"]
        latest_index = (write_index - 1) % buffer_size
        frame_size = device_header["frame_size"]
        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + latest_index * frame_size

        frame_header = unpack_frame_header(buf[frame_offset : frame_offset + FRAME_HEADER_SIZE])
        if frame_header["timestamp_ns"] == 0:
            return None, None, None

        data_start = frame_offset + FRAME_HEADER_SIZE
        data_size = frame_size - FRAME_HEADER_SIZE
        dtype = get_dtype(device_header["data_dtype"], np.float64)
        flat = np.frombuffer(buf[data_start : data_start + data_size], dtype=dtype).reshape(-1)

        if flat.shape[0] < 30:
            return None, None, None
        left_tcp7 = flat[7:14].copy()
        right_tcp7 = flat[21:28].copy()
        head_j2 = flat[28:30].copy()
        return left_tcp7, right_tcp7, head_j2

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------
    def _make_viewer(self) -> _X1ControllerViewer:
        cert = key = None
        if self.https:
            cert = str(_PROJECT_ROOT / self.cert_file)
            key = str(_PROJECT_ROOT / self.key_file)
        return _X1ControllerViewer(
            port=self.port, cert=cert, key=key, webrtc_src=self.webrtc_src,
            skip_patch=self.skip_patch, allow_cache=self.allow_cache,
        )

    # ------------------------------------------------------------------
    # Head following
    # ------------------------------------------------------------------
    def _compute_head_target(
        self, head_mat: np.ndarray, head_ok: bool, head_j2: Optional[np.ndarray]
    ) -> Tuple[float, float]:
        """Return absolute (pitch, yaw) targets in rad (X1Robot clamps to range)."""
        if head_j2 is None:
            return 0.0, 0.0
        if not self.enable_head or not head_ok:
            return float(head_j2[0]), float(head_j2[1])

        pitch = head_pitch_from_matrix(head_mat)
        yaw = head_yaw_from_matrix(head_mat)
        if self._head_pitch_ref is None and (pitch != 0.0 or yaw != 0.0):
            self._head_pitch_ref = pitch
            self._head_yaw_ref = yaw
            self._head_joint_ref = np.asarray(head_j2, dtype=float).copy()
            logger.info(f"[head] ref captured pitch={pitch:.3f} yaw={yaw:.3f}")

        if self._head_pitch_ref is None or self._head_joint_ref is None:
            return float(head_j2[0]), float(head_j2[1])

        dp = HEAD_PITCH_SIGN * HEAD_PITCH_SCALE * (pitch - self._head_pitch_ref)
        dy = HEAD_YAW_SIGN * HEAD_YAW_SCALE * (yaw - self._head_yaw_ref)
        return float(self._head_joint_ref[0] + dp), float(self._head_joint_ref[1] + dy)

    # ------------------------------------------------------------------
    # Server loop
    # ------------------------------------------------------------------
    def _log_diag(self, head_ok: bool) -> None:
        now = time.time()
        if now - self._diag_last_t < 1.0:
            return
        hm = self._viewer.hand_move_count if self._viewer is not None else -1
        rate = hm - self._diag_last_hand_move if hm >= 0 else -1
        self._diag_last_hand_move = hm
        self._diag_last_t = now
        if rate == 0:
            logger.warning(
                f"[diag] NO HAND_MOVE in last 1s — enter VR / wake controllers / "
                f"reload :{self.port}. head_ok={head_ok}"
            )

    def start_server(self) -> None:
        if self.running:
            logger.info(f"X1VuerTeleoperator {self.device_class_id} already running")
            return

        logger.info(f"Starting X1VuerTeleoperator {self.device_class_id} server...")
        self.running = True

        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        self._viewer = self._make_viewer()
        self._viewer.start()
        scheme = "https" if self.https else "http"
        logger.info(f"Vuer viewer started on {scheme}://<host>:{self.port}")

        # Wait for the robot state SHM so the clutch has a valid TCP anchor.
        while self.running and self.robot_state_shm is None:
            self._connect_robot_state_shm()
            if self.robot_state_shm is None:
                time.sleep(0.1)
        logger.info("Robot state SHM connected")

        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        while self.running:
            try:
                out = self._step()
                if out is not None:
                    self._write_array_to_shm_with_timestamp(out, time.time_ns())
                wait_for_next_iteration()
            except Exception as exc:
                logger.error(f"Error in X1VuerTeleoperator loop: {exc}")
                break

    # ------------------------------------------------------------------
    # Episode recording trigger (VR buttons -> saver signal files)
    # ------------------------------------------------------------------
    def _write_episode_signal(self, signal_type: str) -> None:
        path = self.episode_signal_pattern.format(signal=signal_type)
        try:
            with open(path, "w") as f:
                f.write(f"{time.time():.6f}")
        except Exception as exc:
            logger.warning(f"Failed to write episode signal {path}: {exc}")

    def _touch_discard_signal(self) -> None:
        try:
            with open(self.episode_discard_signal, "w") as f:
                f.write(f"{time.time():.6f}")
        except Exception as exc:
            logger.warning(f"Failed to write discard signal: {exc}")

    def _handle_episode_buttons(self, viewer: _X1ControllerViewer) -> None:
        if not self.enable_episode_trigger:
            return
        btns = (viewer.right_buttons if self.episode_button_hand == "right"
                else viewer.left_buttons)

        toggle = _btn(btns, self.episode_toggle_button)
        if toggle and not self._toggle_prev:
            if not self._is_recording:
                self._is_recording = True
                self._write_episode_signal("start")
                logger.info("[episode] START recording (VR button)")
            else:
                self._is_recording = False
                self._write_episode_signal("end")
                logger.info("[episode] END recording (VR button)")
        self._toggle_prev = toggle

        discard = _btn(btns, self.episode_discard_button)
        if discard and not self._discard_prev:
            if self._is_recording:
                self._is_recording = False
                self._touch_discard_signal()
                self._write_episode_signal("end")
                logger.info("[episode] DISCARD current episode (VR button)")
            else:
                logger.info("[episode] discard pressed but not recording; ignored")
        self._discard_prev = discard

    def _step(self) -> Optional[np.ndarray]:
        viewer = self._viewer
        if viewer is None:
            return None

        head_mat = viewer.head_matrix
        head_ok = not _is_identity_matrix(head_mat)
        self._log_diag(head_ok)

        self._handle_episode_buttons(viewer)

        left_tcp7, right_tcp7, head_j2 = self._read_latest_robot_state()

        left_info = self._left.update(viewer, left_tcp7) if self.enable_left else None
        right_info = self._right.update(viewer, right_tcp7) if self.enable_right else None

        # Only enabled arms need a valid commanded TCP; disabled arms keep an
        # identity quaternion placeholder (X1Robot ignores disabled-arm slots).
        out = np.zeros(self.data_shape, dtype=get_dtype(self.data_dtype)).reshape(-1)
        arm_inputs = (
            (self.enable_left, left_info, self._left),
            (self.enable_right, right_info, self._right),
        )
        for layout, (enabled, arm_info, controller) in zip(
            self._output_layout, arm_inputs
        ):
            quaternion_slice = layout["quaternion"]
            if layout["quaternion_order"] == "wxyz":
                out[quaternion_slice] = [1.0, 0.0, 0.0, 0.0]
            else:
                out[quaternion_slice] = [0.0, 0.0, 0.0, 1.0]
            if not enabled:
                continue
            if arm_info is None or arm_info["target_T"] is None:
                return None
            position, quaternion = T_to_pos_quat_xyzw(arm_info["target_T"])
            out[layout["position"]] = position
            if layout["quaternion_order"] == "wxyz":
                quaternion = quaternion[[3, 0, 1, 2]]
            out[quaternion_slice] = quaternion
            gripper_slice = layout["gripper"]
            if self.enable_grippers and gripper_slice is not None:
                out[gripper_slice] = controller.last_gripper

        if self._head_output_slice is not None:
            head_pitch, head_yaw = self._compute_head_target(
                head_mat, head_ok, head_j2
            )
            out[self._head_output_slice] = [head_pitch, head_yaw]
        return out.reshape(self.data_shape)

    def stop_server(self) -> None:
        if not self.running:
            return
        logger.info(f"Stopping X1VuerTeleoperator {self.device_class_id} server...")
        self.running = False
        if self._viewer is not None:
            self._viewer.stop()
        if self.robot_state_shm is not None:
            try:
                self.robot_state_shm.close()
            except Exception as exc:
                logger.warning(f"Failed to close robot state SHM: {exc}")
            self.robot_state_shm = None
        self._cleanup_shared_memory()

    def get_device_info(self) -> Dict[str, Any]:
        info = super().get_device_info()
        info.update({
            "arm_side": self.arm_side,
            "enable_head": self.enable_head,
            "robot_device": f"{self.robot_device_name}_{self.robot_device_class_id}",
            "port": self.port,
            "webrtc_src": self.webrtc_src,
        })
        return info
