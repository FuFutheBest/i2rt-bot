"""Shared-memory layout constants for dual-arm Rizon robot state.

Author: SuperInference
"""

from typing import Final, Sequence, Tuple


class RizonSingleArmStateLayout:
    """Supported flat shared-memory layouts for one Rizon arm."""

    JOINT_SIZE: Final[int] = 14
    TCP_SIZE: Final[int] = 22
    SUPPORTED_SIZES: Final[Tuple[int, ...]] = (JOINT_SIZE, TCP_SIZE)


class RizonDualArmStateLayout:
    """Named offsets for the backward-compatible dual-arm state frame."""

    BASE_SIZE: Final[int] = 44
    LEFT_JOINT_POSITION: Final[slice] = slice(0, 7)
    LEFT_JOINT_VELOCITY: Final[slice] = slice(7, 14)
    LEFT_MEASURED_TCP: Final[slice] = slice(14, 21)
    LEFT_GRIPPER_VALUE: Final[int] = 21
    RIGHT_JOINT_POSITION: Final[slice] = slice(22, 29)
    RIGHT_JOINT_VELOCITY: Final[slice] = slice(29, 36)
    RIGHT_MEASURED_TCP: Final[slice] = slice(36, 43)
    RIGHT_GRIPPER_VALUE: Final[int] = 43

    GRIPPER_STATE_SIZE: Final[int] = 48
    LEFT_GRIPPER_WIDTH_M: Final[int] = 44
    LEFT_GRIPPER_RADIAN: Final[int] = 45
    RIGHT_GRIPPER_WIDTH_M: Final[int] = 46
    RIGHT_GRIPPER_RADIAN: Final[int] = 47

    ARM_DETAIL_SIZE: Final[int] = 74
    LEFT_COMMAND_TCP: Final[slice] = slice(48, 55)
    LEFT_EXTERNAL_WRENCH: Final[slice] = slice(55, 61)
    RIGHT_COMMAND_TCP: Final[slice] = slice(61, 68)
    RIGHT_EXTERNAL_WRENCH: Final[slice] = slice(68, 74)

    TCP_VELOCITY_SIZE: Final[int] = 86
    LEFT_TCP_VELOCITY: Final[slice] = slice(74, 80)
    RIGHT_TCP_VELOCITY: Final[slice] = slice(80, 86)

    COMMAND_GRIPPER_SIZE: Final[int] = 88
    LEFT_COMMAND_GRIPPER_WIDTH_M: Final[int] = 86
    RIGHT_COMMAND_GRIPPER_WIDTH_M: Final[int] = 87

    SUPPORTED_SIZES: Final[Tuple[int, ...]] = (
        BASE_SIZE,
        GRIPPER_STATE_SIZE,
        ARM_DETAIL_SIZE,
        TCP_VELOCITY_SIZE,
        COMMAND_GRIPPER_SIZE,
    )


def validate_rizon_state_shape(
    data_shape: Sequence[int],
    *,
    is_dual_arm: bool,
    using_tcp_pose: bool,
) -> int:
    """Validate a configured Rizon state schema and return its flat size."""
    normalized_shape = tuple(int(dimension) for dimension in data_shape)
    if len(normalized_shape) != 1:
        raise ValueError(
            f"Rizon data_shape must be one-dimensional, got {normalized_shape}"
        )

    state_size = normalized_shape[0]
    if is_dual_arm:
        if state_size not in RizonDualArmStateLayout.SUPPORTED_SIZES:
            raise ValueError(
                f"Unsupported dual-arm Rizon data_shape {normalized_shape}; "
                f"supported sizes are {RizonDualArmStateLayout.SUPPORTED_SIZES}"
            )
        return state_size

    expected_size = (
        RizonSingleArmStateLayout.TCP_SIZE
        if using_tcp_pose
        else RizonSingleArmStateLayout.JOINT_SIZE
    )
    if state_size != expected_size:
        mode = "TCP" if using_tcp_pose else "joint"
        raise ValueError(
            f"Single-arm Rizon {mode} state requires data_shape ({expected_size},), "
            f"got {normalized_shape}"
        )
    return state_size
