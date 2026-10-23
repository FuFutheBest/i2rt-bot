"""Seven-value leader and follower endpoints used by teleoperation workers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import FollowerConfig, GripperConfig
from .grippers import GripperBackend, LKTechGripper, NullGripper

ARM_DOF = 6
ENDPOINT_DOF = 7


def _vector(value: Any, size: int, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size != size:
        raise ValueError(f"{field} must contain exactly {size} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{field} must contain only finite values")
    return result


class LeaderEndpoint:
    """Expose six YAM leader joints and the teaching-handle opening value."""

    def __init__(self, arm: Any) -> None:
        self.arm = arm
        robot_info = arm.get_robot_info()
        self._hold_kp = _vector(robot_info.get("kp"), ARM_DOF, "leader hold kp").copy()
        self._hold_kd = _vector(robot_info.get("kd"), ARM_DOF, "leader hold kd").copy()

    def get_info(self) -> tuple[np.ndarray, np.ndarray]:
        joints = _vector(
            self.arm.get_observations()["joint_pos"], ARM_DOF, "leader joints"
        )
        encoder_states = self.arm.motor_chain.get_same_bus_device_states()
        if not encoder_states:
            raise RuntimeError("YAM teaching handle did not return an encoder state")
        gripper = float(np.clip(1.0 - encoder_states[0].position, 0.0, 1.0))
        buttons = np.asarray(encoder_states[0].io_inputs, dtype=np.float64).reshape(-1)
        if buttons.size == 0:
            buttons = np.zeros(1, dtype=np.float64)
        return np.concatenate([joints, [gripper]]), buttons

    def command_joint_pos(self, joint_pos: Any) -> None:
        self.arm.command_joint_pos(_vector(joint_pos, ARM_DOF, "leader command"))

    def enter_position_hold(self, joint_pos: Any) -> None:
        target = _vector(joint_pos, ARM_DOF, "leader hold target")
        self.arm.update_kp_kd(self._hold_kp.copy(), self._hold_kd.copy())
        self.arm.command_joint_pos(target)

    def update_kp_kd(self, kp: Any, kd: Any) -> None:
        self.arm.update_kp_kd(
            _vector(kp, ARM_DOF, "leader kp"),
            _vector(kd, ARM_DOF, "leader kd"),
        )
        current = _vector(
            self.arm.get_observations()["joint_pos"], ARM_DOF, "leader joints"
        )
        self.arm.command_joint_pos(current)

    def set_feedback_enabled(self, enabled: bool, kp: Any, kd: Any) -> None:
        zeros = np.zeros(ARM_DOF, dtype=np.float64)
        self.update_kp_kd(kp if enabled else zeros, kd if enabled else zeros)

    def enter_safe_state(self, state: Any) -> None:
        zeros = np.zeros(ARM_DOF, dtype=np.float64)
        self.arm.update_kp_kd(zeros, zeros)
        self.arm.command_joint_pos(_vector(state, ENDPOINT_DOF, "leader state")[:6])

    def close(self) -> None:
        self.arm.close()


class CompositeFollowerEndpoint:
    """Combine a YAM arm and configured gripper behind a seven-value API."""

    def __init__(
        self,
        arm: Any,
        gripper_config: GripperConfig,
        external_gripper: GripperBackend | None = None,
    ) -> None:
        self.arm = arm
        self.gripper_config = gripper_config
        self.external_gripper = external_gripper
        self._closed = False

        if gripper_config.backend == "i2rt_can" and external_gripper is not None:
            raise ValueError("Integrated I2RT grippers cannot use an external backend")
        if gripper_config.backend != "i2rt_can" and external_gripper is None:
            raise ValueError("External and no-gripper modes require a gripper backend")

    def start(self) -> None:
        if self.external_gripper is not None:
            self.external_gripper.start()

    def get_info(self) -> np.ndarray:
        observations = self.arm.get_observations()
        joints = _vector(observations["joint_pos"], ARM_DOF, "follower joints")
        if self.gripper_config.backend == "i2rt_can":
            gripper = _vector(
                observations.get("gripper_pos"), 1, "I2RT gripper feedback"
            )[0]
        else:
            gripper = float(self.external_gripper.read_normalized())
        return np.concatenate([joints, [float(np.clip(gripper, 0.0, 1.0))]])

    def command_joint_pos(self, joint_pos: Any) -> None:
        command = _vector(joint_pos, ENDPOINT_DOF, "follower command")
        if self.gripper_config.backend == "i2rt_can":
            self.arm.command_joint_pos(command.copy())
            return
        self.arm.command_joint_pos(command[:ARM_DOF].copy())
        self.external_gripper.command_normalized(float(command[-1]))

    def update_kp_kd(self, kp: Any, kd: Any) -> None:
        arm_dof = ENDPOINT_DOF if self.gripper_config.backend == "i2rt_can" else ARM_DOF
        self.arm.update_kp_kd(
            _vector(kp, arm_dof, "follower kp"),
            _vector(kd, arm_dof, "follower kd"),
        )

    def set_feedback_enabled(self, enabled: bool, kp: Any, kd: Any) -> None:
        del enabled, kp, kd

    def enter_safe_state(self, state: Any) -> None:
        self.command_joint_pos(_vector(state, ENDPOINT_DOF, "follower state"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.external_gripper is not None:
                self.external_gripper.close()
        finally:
            self.arm.close()


def create_leader_endpoint(can_channel: str) -> LeaderEndpoint:
    """Construct a physical YAM teaching-handle endpoint."""

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    arm = get_yam_robot(
        channel=can_channel,
        gripper_type=GripperType.YAM_TEACHING_HANDLE,
        use_coulomb_friction=True,
    )
    return LeaderEndpoint(arm)


def create_follower_endpoint(
    config: FollowerConfig, *, label: str
) -> CompositeFollowerEndpoint:
    """Construct a physical follower endpoint from its backend configuration."""

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    if config.gripper.backend == "i2rt_can":
        gripper_type = GripperType.from_string_name(config.gripper.i2rt_type)
        external_gripper = None
    else:
        gripper_type = GripperType.NO_GRIPPER
        external_gripper = (
            LKTechGripper(config.gripper, label=label)
            if config.gripper.backend == "lktech"
            else NullGripper()
        )

    arm = get_yam_robot(channel=config.can_channel, gripper_type=gripper_type)
    endpoint = CompositeFollowerEndpoint(
        arm=arm,
        gripper_config=config.gripper,
        external_gripper=external_gripper,
    )
    try:
        endpoint.start()
    except Exception:
        endpoint.close()
        raise
    return endpoint
