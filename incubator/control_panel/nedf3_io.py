"""NEDF3 protobuf construction and decoding for inference rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from typing import Any

import numpy as np


PROCESSED_EXTRA_FIELDS = (
    "aligned_timestamp_ns",
    "alignment_mode",
    "downsample_kept",
    "aligned_pose",
    "aligned_hand_state",
    "aligned_raw_hand_state",
    "aligned_slam",
    "aligned_pose_tcp",
    "aligned_corrected_roi_marker_angle",
    "aligned_corrected_gripper_width",
    "aligned_arkit_diagnostics",
)


class NEDF3ContractError(ValueError):
    """A sample cannot be represented by the rollout NEDF3 contract."""


def _pose_values(values: Sequence[float]) -> np.ndarray:
    pose = np.asarray(values, dtype=np.float64).reshape(-1)
    if pose.size != 7 or not np.isfinite(pose).all():
        raise NEDF3ContractError("pose must contain seven finite values")
    return pose


def _fill_pose_xyzw(target: Any, values: Sequence[float]) -> None:
    pose = _pose_values(values)
    qx, qy, qz, qw = (float(value) for value in pose[3:7])
    if float(np.linalg.norm([qw, qx, qy, qz])) <= 1e-12:
        raise NEDF3ContractError("quaternion norm is zero")
    target.pos.x = float(pose[0])
    target.pos.y = float(pose[1])
    target.pos.z = float(pose[2])
    target.quat.w = qw
    target.quat.x = qx
    target.quat.y = qy
    target.quat.z = qz


def arm_tcp_cmd_xyzw_message(values: Sequence[float]) -> Any:
    """Build R1 ArmTcpCmd from ``[x,y,z,qx,qy,qz,qw]``."""
    from nmx_msg.Arm_pb2 import ArmTcpCmd

    message = ArmTcpCmd()
    _fill_pose_xyzw(message.pose, values)
    return message


def arm_tcp_state_xyzw_message(values: Sequence[float]) -> Any:
    """Build R1 ArmTcpState from ``[x,y,z,qx,qy,qz,qw]``."""
    from nmx_msg.Arm_pb2 import ArmTcpState

    message = ArmTcpState()
    _fill_pose_xyzw(message.pose, values)
    return message


def arm_tcp_state_wxyz_message(values: Sequence[float]) -> Any:
    """Build R1 ArmTcpState from ``[x,y,z,qw,qx,qy,qz]``."""
    pose = _pose_values(values)
    return arm_tcp_state_xyzw_message(
        [pose[0], pose[1], pose[2], pose[4], pose[5], pose[6], pose[3]]
    )


def arm_state_message(
    *,
    positions: Sequence[float] | None = None,
    velocities: Sequence[float] | None = None,
) -> Any:
    """Build R1 ArmState joint positions and/or velocities."""
    from nmx_msg.Arm_pb2 import ArmState

    pos = (
        None
        if positions is None
        else np.asarray(positions, dtype=np.float64).reshape(-1)
    )
    vel = (
        None
        if velocities is None
        else np.asarray(velocities, dtype=np.float64).reshape(-1)
    )
    size = max(0 if pos is None else pos.size, 0 if vel is None else vel.size)
    if (
        size == 0
        or (pos is not None and pos.size != size)
        or (vel is not None and vel.size != size)
    ):
        raise NEDF3ContractError(
            "joint positions and velocities must have one equal non-zero dimension"
        )
    if (pos is not None and not np.isfinite(pos).all()) or (
        vel is not None and not np.isfinite(vel).all()
    ):
        raise NEDF3ContractError("joint state must contain finite values")
    message = ArmState()
    for index in range(size):
        joint = message.joints.add()
        joint.name = f"joint{index}"
        if pos is not None:
            joint.pos = float(pos[index])
        if vel is not None:
            joint.vel = float(vel[index])
    return message


def arm_wrench_message(
    values: Sequence[float] = (0, 0, 0, 0, 0, 0),
) -> Any:
    from nmx_msg.Arm_pb2 import ArmEndForce

    wrench = np.asarray(values, dtype=np.float64).reshape(-1)
    if wrench.size != 6 or not np.isfinite(wrench).all():
        raise NEDF3ContractError("wrench must contain six finite values")
    message = ArmEndForce()
    message.force.x, message.force.y, message.force.z = map(float, wrench[:3])
    message.torque.x, message.torque.y, message.torque.z = map(float, wrench[3:])
    return message


def _hand_joint_message(message: Any, width_m: float) -> Any:
    width = float(width_m)
    if not math.isfinite(width):
        raise NEDF3ContractError("gripper width must be finite")
    joint = message.joints.add()
    joint.name = "gripper"
    joint.pos = width
    return message


def hand_cmd_message(width_m: float) -> Any:
    from nmx_msg.Hand_pb2 import HandCmd

    return _hand_joint_message(HandCmd(), width_m)


def hand_state_message(width_m: float) -> Any:
    from nmx_msg.Hand_pb2 import HandState

    return _hand_joint_message(HandState(), width_m)


def extra_message(control_panel_payload: Mapping[str, Any] | None = None) -> Any:
    from nmx_msg.Metadata_pb2 import Metadata

    payload: dict[str, Any] = {
        "processed": {field: None for field in PROCESSED_EXTRA_FIELDS}
    }
    if control_panel_payload:
        payload["control_panel"] = dict(control_panel_payload)
    message = Metadata()
    message.version = "nedf3"
    message.metadata = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return message


def rgbd_jpeg_message(
    data: bytes, rows: int, cols: int, channels: int
) -> Any:
    from nmx_msg.Image_pb2 import Image, ImageDataType, RGBD

    if rows <= 0 or cols <= 0 or channels <= 0 or not data:
        raise NEDF3ContractError("encoded RGB image must be non-empty")
    rgb = Image(
        data=bytes(data),
        type=ImageDataType.ENCODED,
        encoded_format="jpeg",
        rows=int(rows),
        cols=int(cols),
        channels=int(channels),
    )
    depth = Image(
        data=b"",
        type=ImageDataType.ENCODED,
        encoded_format="",
        rows=0,
        cols=0,
        channels=0,
    )
    message = RGBD()
    message.rgb.CopyFrom(rgb)
    message.depth.CopyFrom(depth)
    if hasattr(message, "depth_scale"):
        message.depth_scale = 1.0
    return message


def decoded_message(record: Any) -> Any:
    return getattr(record, "decoded_message", record)


def message_timestamp_ns(record: Any) -> int:
    message = getattr(record, "message", None)
    for owner, name in (
        (message, "log_time"),
        (record, "log_time"),
        (record, "log_timestamp_ns"),
    ):
        value = getattr(owner, name, None) if owner is not None else None
        if isinstance(value, int):
            return value
    raise NEDF3ContractError("decoded NEDF3 record has no log timestamp")


def extract_pose_wxyz(value: Any) -> np.ndarray:
    pose = getattr(decoded_message(value), "pose", None)
    if pose is None or not hasattr(pose, "pos") or not hasattr(pose, "quat"):
        raise NEDF3ContractError("message is not an UmiPose")
    result = np.asarray(
        [
            pose.pos.x,
            pose.pos.y,
            pose.pos.z,
            pose.quat.w,
            pose.quat.x,
            pose.quat.y,
            pose.quat.z,
        ],
        dtype=np.float32,
    )
    _pose_values(result)
    return result


def extract_gripper_width(value: Any) -> float | None:
    message = decoded_message(value)
    joints = getattr(message, "joints", ())
    fallback: float | None = None
    for joint in joints:
        position = float(getattr(joint, "pos", 0.0))
        if not math.isfinite(position):
            continue
        name = str(getattr(joint, "name", ""))
        if name == "gripper.width":
            return position
        if fallback is None:
            fallback = position
    return fallback


def extract_joint_state_positions(
    value: Any, *, expected_dimension: int | None = None
) -> np.ndarray:
    """Extract ordered positions from an R1 ArmState message."""
    joints = tuple(getattr(decoded_message(value), "joints", ()))
    if not joints:
        raise NEDF3ContractError("ArmState contains no joints")
    positions = np.asarray([float(joint.pos) for joint in joints], dtype=np.float32)
    if expected_dimension is not None and positions.size != expected_dimension:
        raise NEDF3ContractError(
            f"ArmState has dimension {positions.size}; expected {expected_dimension}"
        )
    if not np.isfinite(positions).all():
        raise NEDF3ContractError("ArmState contains non-finite positions")
    return positions


def extract_rgb_jpeg(value: Any) -> bytes:
    """Extract a non-empty encoded RGB payload from an NEDF3 RGBD message."""
    message = decoded_message(value)
    rgb = getattr(message, "rgb", None)
    data = bytes(getattr(rgb, "data", b"")) if rgb is not None else b""
    if not data:
        raise NEDF3ContractError("RGBD message has no encoded RGB payload")
    encoded_format = str(getattr(rgb, "encoded_format", "")).strip().lower()
    if encoded_format not in {"jpg", "jpeg"}:
        raise NEDF3ContractError(
            f"RGBD message uses unsupported RGB encoding {encoded_format!r}"
        )
    return data


def extract_extra_payload(value: Any) -> dict[str, Any]:
    message = decoded_message(value)
    raw = bytes(getattr(message, "metadata", b""))
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NEDF3ContractError("invalid NEDF3 extra JSON") from exc
    if not isinstance(payload, dict):
        raise NEDF3ContractError("NEDF3 extra payload must be an object")
    return payload


__all__ = [
    "NEDF3ContractError",
    "PROCESSED_EXTRA_FIELDS",
    "arm_state_message",
    "arm_tcp_cmd_xyzw_message",
    "arm_tcp_state_wxyz_message",
    "arm_tcp_state_xyzw_message",
    "arm_wrench_message",
    "decoded_message",
    "extra_message",
    "extract_extra_payload",
    "extract_gripper_width",
    "extract_joint_state_positions",
    "extract_pose_wxyz",
    "extract_rgb_jpeg",
    "hand_cmd_message",
    "hand_state_message",
    "message_timestamp_ns",
    "rgbd_jpeg_message",
]
