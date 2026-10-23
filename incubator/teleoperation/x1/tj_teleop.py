#!/usr/bin/env python3
"""Quest 3 -> TJ dual-arm teleoperation (TCP / move_line mode).

Run INSIDE the robot_test container (nmxrdk available, network_mode: host):
    cd /root/incubator/teleoperation/x1
    python3 tj_teleop.py

Then expose port 8012 with ngrok on the host and open the https URL in Quest 3.

Data flow:
    Quest 3 (Vuer/WebXR)
      -> OpenTeleVision: head pose + L/R wrist pose + finger landmarks
      -> Preprocessor: change-of-basis, wrist relative to head (robot Z-up)
      -> this script:
           wrist pose  -> arm.move_line(TCP target)      (incremental/engage based)
           pinch dist  -> gripper.set_width()
           head pose   -> robot.head().move_joint([pitch, yaw])
           camera rgbd -> shared image -> back to Quest 3
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations
from multiprocessing import shared_memory, Queue, Event

# --- TeleVision modules (reused as-is) ---
sys.path.append(str(Path(__file__).parent / "TeleVision" / "teleop"))
from TeleVision import OpenTeleVision          # noqa: E402
from Preprocessor import VuerPreprocessor      # noqa: E402
from constants_vuer import tip_indices         # noqa: E402

# --- TJ robot SDK ---
import nmxrdk                                   # noqa: E402
import nmxrt                                    # noqa: E402
from nmxrdk.common.robot_side import RobotSide  # noqa: E402
import teleop_home                              # noqa: E402


# ============================================================
# CONFIG  ----  调这里
# ============================================================
ROBOT_ID = "1e783be2dd870ef5"

# 头显里要显示的相机 id（单路 RGBD，左右眼同图）。按你的实际头部相机改。
HEAD_CAMERA_ID = "CAMID_0"

# 图像分辨率（与 OpenTeleVision 内部约定一致）
RESOLUTION = (720, 1280)              # (h, w) 单眼
CROP_W = 0
CROP_H = 0

# 手臂 TCP 控制
USE_TCP = True                       # True=move_line; 若固件不支持则改 joint 方案
LINEAR_VEL = 0.05                    # m/s, move_line 线速度，先给小值保安全
POS_SCALE = 1.0                     # VR 位移 -> 机器人位移 缩放
# 工作空间安全限位（机器人基坐标系，米），目标超出则忽略该帧
WS_MIN = np.array([0.15, -0.6, 0.0])
WS_MAX = np.array([0.8,  0.6, 0.7])

# 夹爪：拇指尖↔食指尖 距离(米) -> 开口宽度(米)
PINCH_MIN, PINCH_MAX = 0.02, 0.10   # 手指距离范围
GRIP_MIN, GRIP_MAX = 0.0, 0.08      # 夹爪开口范围
GRIP_DEADBAND = 0.005               # 变化小于此值不下发

# 头部舵机
HEAD_VEL = [1.0, 1.0]               # rad/s [pitch, yaw]
HEAD_ACC = [2.0, 2.0]
HEAD_PITCH_LIMIT = (-0.6, 0.6)      # rad
HEAD_YAW_LIMIT = (-1.2, 1.2)        # rad

CONTROL_HZ = 60
# ============================================================


def ensure_ok(result, action):
    if isinstance(result, nmxrt.Error):
        raise RuntimeError(f"{action} failed: {result.message}")
    return result


def tcp_state_to_xyzquat(tcp_state):
    p = tcp_state.pose
    return np.array([p.pos.x, p.pos.y, p.pos.z,
                     p.quat.w, p.quat.x, p.quat.y, p.quat.z])


def mat_to_xyzquat(mat):
    """4x4 -> [x,y,z,qw,qx,qy,qz]."""
    quat_wxyz = rotations.quaternion_from_matrix(mat[:3, :3])  # [w,x,y,z]
    return np.concatenate([mat[:3, 3], quat_wxyz])


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ArmController:
    """增量/对齐式 TCP 控制：engage 时记录基准，之后跟随 VR 手腕相对运动。"""

    def __init__(self, arm, name, dry_run=False):
        self.arm = arm
        self.name = name
        self.dry_run = dry_run
        self._tick = 0
        self.engaged = False
        self.robot_base_pose = None     # [x,y,z,qw,qx,qy,qz] 机器人基准
        self.vr_base_pos = None         # VR 手腕基准位置
        self.vr_base_rot = None         # VR 手腕基准旋转 (scipy Rotation)

    def engage(self, vr_wrist_mat):
        state = ensure_ok(self.arm.get_tcp_state(), f"{self.name} get_tcp_state")
        self.robot_base_pose = tcp_state_to_xyzquat(state)
        self.vr_base_pos = vr_wrist_mat[:3, 3].copy()
        self.vr_base_rot = R.from_matrix(vr_wrist_mat[:3, :3])
        self.engaged = True
        print(f"[{self.name}] engaged at {self.robot_base_pose[:3]}")

    def disengage(self):
        self.engaged = False

    def compute_target(self, vr_wrist_mat):
        """返回机器人 TCP 目标 [x,y,z,qw,qx,qy,qz] 或 None。"""
        if not self.engaged:
            return None
        # 平移增量
        dpos = (vr_wrist_mat[:3, 3] - self.vr_base_pos) * POS_SCALE
        target_pos = self.robot_base_pose[:3] + dpos

        if np.any(target_pos < WS_MIN) or np.any(target_pos > WS_MAX):
            return None  # 超出工作空间，丢弃该帧

        # 旋转增量： R_target = dR * R_base
        vr_rot = R.from_matrix(vr_wrist_mat[:3, :3])
        d_rot = vr_rot * self.vr_base_rot.inv()
        base_rot = R.from_quat(self.robot_base_pose[[4, 5, 6, 3]])  # x,y,z,w
        target_rot = d_rot * base_rot
        q_xyzw = target_rot.as_quat()  # [x,y,z,w]
        return np.array([target_pos[0], target_pos[1], target_pos[2],
                         q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    def send(self, target):
        if target is None:
            return
        if self.dry_run:
            self._tick += 1
            if self._tick % 10 == 0:  # 约每 10 帧打印一次，避免刷屏
                dp = target[:3] - self.robot_base_pose[:3]
                print(f"[{self.name}] target xyz={target[:3].round(3)} "
                      f"d_from_base={dp.round(3)} quat(wxyz)={target[3:].round(3)}")
            return
        res = self.arm.move_line(list(target), LINEAR_VEL, False)  # non-blocking
        if isinstance(res, nmxrt.Error):
            print(f"[{self.name}] move_line err: {res.message}")


def wait_for_hand_tracking(tv, sides, timeout=120):
    """阻塞直到头显真正传来手部追踪数据（原始手腕矩阵非退化）。"""
    print("等待头显手部追踪数据... 请进入 VR(Enter VR)并用【裸手】伸到视野内")
    start = time.time()
    last_log = 0.0
    while time.time() - start < timeout:
        det_l = np.linalg.det(tv.left_hand)
        det_r = np.linalg.det(tv.right_hand)
        det_head = np.linalg.det(tv.head_matrix)
        ok = True
        if "left" in sides and det_l == 0:
            ok = False
        if "right" in sides and det_r == 0:
            ok = False
        if ok:
            print("已检测到手部追踪数据。")
            return True
        now = time.time()
        if now - last_log > 2.0:
            print(f"  [诊断] det(left)={det_l:.3f} det(right)={det_r:.3f} "
                  f"det(head)={det_head:.3f}  (全为0=没数据；head非0=头显已连但手未追踪)")
            last_log = now
        time.sleep(0.2)
    print("等待超时，仍未检测到手部数据。")
    return False


def pinch_to_width(fingers):
    """fingers: (25,3) 单手关节点；返回夹爪宽度(米)。"""
    thumb_tip = fingers[tip_indices[0]]
    index_tip = fingers[tip_indices[1]]
    d = np.linalg.norm(thumb_tip - index_tip)
    t = clamp((d - PINCH_MIN) / (PINCH_MAX - PINCH_MIN), 0.0, 1.0)
    return GRIP_MIN + t * (GRIP_MAX - GRIP_MIN)


def head_targets_from_matrix(head_rmat):
    """head_rmat: 3x3 -> [pitch, yaw] (rad)。"""
    quat = rotations.quaternion_from_matrix(head_rmat)
    # yaw(2), pitch(1), roll(0)
    ypr = rotations.euler_from_quaternion(quat, 2, 1, 0, False)
    yaw, pitch = ypr[0], ypr[1]
    pitch = clamp(pitch, *HEAD_PITCH_LIMIT)
    yaw = clamp(yaw, *HEAD_YAW_LIMIT)
    return [pitch, yaw]


def parse_args():
    p = argparse.ArgumentParser(description="TJ dual-arm Quest 3 teleop (staged bring-up).")
    p.add_argument("--enable-arms", action="store_true", help="使能手臂 move_line 跟随")
    p.add_argument("--arm-side", choices=("left", "right", "both"), default="left",
                   help="使能哪条手臂，默认 left")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印计算目标，不真正下发 move_line（先验证映射方向）")
    p.add_argument("--enable-grippers", action="store_true", help="使能夹爪开合")
    p.add_argument("--enable-head", action="store_true", help="使能头部舵机跟随")
    p.add_argument("--no-camera", action="store_true", help="不推相机图像(只测控制)")
    p.add_argument("--no-home", action="store_true",
                   help="跳过启动时回到初始位置（默认会先回初始位）")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[cfg] arms={args.enable_arms} grippers={args.enable_grippers} "
          f"head={args.enable_head} camera={not args.no_camera}")
    res_cropped = (RESOLUTION[0] - CROP_H, RESOLUTION[1] - 2 * CROP_W)
    img_shape = (res_cropped[0], 2 * res_cropped[1], 3)

    shm = shared_memory.SharedMemory(create=True,
                                     size=int(np.prod(img_shape)) * np.uint8().itemsize)
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)

    tv = OpenTeleVision(res_cropped, shm.name, Queue(), Event(), ngrok=True)
    processor = VuerPreprocessor()

    robot = nmxrdk.RobotClient(ROBOT_ID)
    cam = robot.camera(HEAD_CAMERA_ID) if not args.no_camera else None

    # 启动时先回到统一的初始位置（双臂 Joint + 头部舵机），再进入遥操作。
    if not args.no_home:
        teleop_home.go_home(robot, arm_side="both", do_head=True, blocking=True)

    left = right = grip_l = grip_r = head = None
    arms = []  # [(controller, wrist_index)]，wrist_index: 1=left wrist, 2=right wrist
    if args.enable_arms:
        if args.arm_side in ("left", "both"):
            left = ArmController(robot.arm(RobotSide.LEFT), "left", dry_run=args.dry_run)
            arms.append((left, "left"))
        if args.arm_side in ("right", "both"):
            right = ArmController(robot.arm(RobotSide.RIGHT), "right", dry_run=args.dry_run)
            arms.append((right, "right"))
    if args.enable_grippers:
        grip_l = robot.gripper(RobotSide.LEFT)
        grip_r = robot.gripper(RobotSide.RIGHT)
    if args.enable_head:
        head = robot.head()
        ensure_ok(head.enable(), "head.enable")

    last_grip_l, last_grip_r = None, None
    dt = 1.0 / CONTROL_HZ

    if arms:
        mode = "DRY-RUN(不发指令)" if args.dry_run else "实际控制"
        sides = [side for _, side in arms]
        # 关键：必须等头显真正传来手部数据后再 engage，否则基准是错的。
        wait_for_hand_tracking(tv, sides)
        print(f"[{mode}] 3 秒后 engage，请把手放到舒适初始位并保持不动...")
        time.sleep(3)
        _, lw0, rw0, _, _ = processor.process(tv)
        for ctrl, side in arms:
            ctrl.engage(lw0 if side == "left" else rw0)

    try:
        while True:
            t0 = time.time()
            head_mat, lw, rw, lf, rf = processor.process(tv)

            # ---- arms ----
            for ctrl, side in arms:
                wrist = lw if side == "left" else rw
                ctrl.send(ctrl.compute_target(wrist))

            # ---- grippers ----
            if args.enable_grippers:
                wl = pinch_to_width(lf)
                wr = pinch_to_width(rf)
                if last_grip_l is None or abs(wl - last_grip_l) > GRIP_DEADBAND:
                    grip_l.set_width(wl)
                    last_grip_l = wl
                if last_grip_r is None or abs(wr - last_grip_r) > GRIP_DEADBAND:
                    grip_r.set_width(wr)
                    last_grip_r = wr

            # ---- head ----
            if args.enable_head:
                try:
                    pos = head_targets_from_matrix(head_mat[:3, :3])
                    head.move_joint(pos, HEAD_VEL, HEAD_ACC, False, 1000, {})
                except Exception:
                    pass

            # ---- camera -> headset (单路, 左右眼同图) ----
            if cam is not None:
                try:
                    rgb_frame, _ = cam.get_rgbd(decode=True)
                    if not isinstance(rgb_frame, nmxrt.Error):
                        rgb = rgb_frame.get("data")
                        if rgb is not None:
                            rgb = cv2.resize(rgb, (res_cropped[1], res_cropped[0]))
                            if rgb.shape[2] == 3:
                                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                            np.copyto(img_array, np.hstack((rgb, rgb)))
                except Exception:
                    pass

            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("stopping...")
    finally:
        if head is not None:
            head.stop()
            head.disable()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()

