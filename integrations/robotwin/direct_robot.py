"""Minimal RoboTwin robot implementation for direct SAPIEN drive control.

RoboTwin's stock ``envs.robot`` imports MPLib and CuRobo at module import time,
even when no planner is created.  The async simulation bridge only needs the
articulation, its drives, cameras mounted on the robot, and measured joint
state.  This deliberately small compatibility implementation supplies that
surface without importing either planner stack.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np


class DirectDriveRobot:
    """RoboTwin task-compatible robot with no IK, TOPP, or planner process.

    It mirrors only the portions of RoboTwin's ``Robot`` used while creating a
    task and while the bridge drives a fixed-rate joint target.  Planning
    methods are intentionally absent: this object must never be passed to a
    task execution path that calls ``take_action()``.
    """

    def __init__(self, scene: Any, need_topp: bool = False, **kwargs: Any) -> None:
        del need_topp
        self._sapien = importlib.import_module("sapien.core")
        self._initialize(scene, **kwargs)

    def _initialize(self, scene: Any, **kwargs: Any) -> None:
        left_config = kwargs["left_embodiment_config"]
        right_config = kwargs["right_embodiment_config"]
        left_robot_file = str(kwargs["left_robot_file"])
        right_robot_file = str(kwargs["right_robot_file"])

        self.left_joint_stiffness = float(left_config.get("joint_stiffness", 1000))
        self.left_joint_damping = float(left_config.get("joint_damping", 200))
        self.left_gripper_stiffness = float(left_config.get("gripper_stiffness", 1000))
        self.left_gripper_damping = float(left_config.get("gripper_damping", 200))
        self.right_joint_stiffness = float(right_config.get("joint_stiffness", 1000))
        self.right_joint_damping = float(right_config.get("joint_damping", 200))
        self.right_gripper_stiffness = float(right_config.get("gripper_stiffness", 1000))
        self.right_gripper_damping = float(right_config.get("gripper_damping", 200))
        self._apply_drive_overrides(kwargs)

        self.left_arm_joints_name = self._side_config_value(left_config, "arm_joints_name", 0)
        self.right_arm_joints_name = self._side_config_value(right_config, "arm_joints_name", 1)
        self.left_ee_name = self._side_config_value(left_config, "ee_joints", 0)
        self.right_ee_name = self._side_config_value(right_config, "ee_joints", 1)
        self.left_gripper_name = self._side_config_value(left_config, "gripper_name", 0)
        self.right_gripper_name = self._side_config_value(right_config, "gripper_name", 1)
        self.left_gripper_bias = float(left_config["gripper_bias"])
        self.right_gripper_bias = float(right_config["gripper_bias"])
        self.left_gripper_scale = tuple(float(value) for value in left_config["gripper_scale"])
        self.right_gripper_scale = tuple(float(value) for value in right_config["gripper_scale"])
        left_homestate = left_config.get("homestate")
        right_homestate = right_config.get("homestate")
        self.left_homestate = list(
            self._side_config_value(left_config, "homestate", 0)
            if left_homestate is not None
            else [0.0] * len(self.left_arm_joints_name)
        )
        self.right_homestate = list(
            self._side_config_value(right_config, "homestate", 1)
            if right_homestate is not None
            else [0.0] * len(self.right_arm_joints_name)
        )
        self.left_fix_gripper_name = list(left_config.get("fix_gripper_name", []))
        self.right_fix_gripper_name = list(right_config.get("fix_gripper_name", []))
        self.left_delta_matrix = np.asarray(left_config.get("delta_matrix", np.eye(3)), dtype=np.float64)
        self.right_delta_matrix = np.asarray(right_config.get("delta_matrix", np.eye(3)), dtype=np.float64)
        self.left_global_trans_matrix = np.asarray(
            left_config.get("global_trans_matrix", np.eye(3)), dtype=np.float64
        )
        self.right_global_trans_matrix = np.asarray(
            right_config.get("global_trans_matrix", np.eye(3)), dtype=np.float64
        )
        self.left_entity_origion_pose = self._robot_pose(left_config, 0)
        self.right_entity_origion_pose = self._robot_pose(right_config, 1)

        self.left_urdf_path = os.path.join(left_robot_file, str(left_config["urdf_path"]))
        self.right_urdf_path = os.path.join(right_robot_file, str(right_config["urdf_path"]))
        self.is_dual_arm = bool(kwargs["dual_arm_embodied"])
        if self.is_dual_arm:
            loader = scene.create_urdf_loader()
            loader.fix_root_link = True
            entity = loader.load(self.left_urdf_path)
            if entity is None:
                raise RuntimeError(f"Failed to load RoboTwin URDF: {self.left_urdf_path}")
            self.left_entity = entity
            self.right_entity = entity
        else:
            embodiment_distance = float(kwargs["embodiment_dis"])
            self.left_entity_origion_pose.p += [-embodiment_distance / 2.0, 0.0, 0.0]
            self.right_entity_origion_pose.p += [embodiment_distance / 2.0, 0.0, 0.0]
            left_loader = scene.create_urdf_loader()
            right_loader = scene.create_urdf_loader()
            left_loader.fix_root_link = True
            right_loader.fix_root_link = True
            self.left_entity = left_loader.load(self.left_urdf_path)
            self.right_entity = right_loader.load(self.right_urdf_path)
            if self.left_entity is None or self.right_entity is None:
                raise RuntimeError("Failed to load one or both RoboTwin dual-arm URDFs")

        self.left_entity.set_root_pose(self.left_entity_origion_pose)
        self.right_entity.set_root_pose(self.right_entity_origion_pose)

    @staticmethod
    def _side_config_value(
        config: dict[str, Any],
        key: str,
        index: int,
        *,
        default: Any = None,
    ) -> Any:
        values = config.get(key, default)
        if values is None:
            raise KeyError(f"RoboTwin embodiment config is missing {key!r}")
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"RoboTwin embodiment config {key!r} must be a list")
        if len(values) == 1:
            return values[0]
        if index >= len(values):
            raise ValueError(f"RoboTwin embodiment config {key!r} has no entry for arm index {index}")
        return values[index]

    def _robot_pose(self, config: dict[str, Any], index: int) -> Any:
        pose_values = self._side_config_value(
            config,
            "robot_pose",
            index,
            default=[[0.0, -0.65, 0.0, 1.0, 0.0, 0.0, 0.0]],
        )
        if len(pose_values) != 7:
            raise ValueError("RoboTwin robot_pose entries must contain xyz + qw,qx,qy,qz")
        return self._sapien.Pose(pose_values[:3], pose_values[-4:])

    def init_joints(self) -> None:
        self.left_active_joints = self.left_entity.get_active_joints()
        self.right_active_joints = self.right_entity.get_active_joints()
        self.left_ee = self._require_joint(self.left_entity, self.left_ee_name)
        self.right_ee = self._require_joint(self.right_entity, self.right_ee_name)
        self.left_arm_joints = [self._require_joint(self.left_entity, name) for name in self.left_arm_joints_name]
        self.right_arm_joints = [self._require_joint(self.right_entity, name) for name in self.right_arm_joints_name]
        self.left_gripper = self._gripper_joints(self.left_entity, self.left_gripper_name)
        self.right_gripper = self._gripper_joints(self.right_entity, self.right_gripper_name)
        self.left_gripper_val = 0.0
        self.right_gripper_val = 0.0
        self.gripper_name = list(self.left_fix_gripper_name) + list(self.right_fix_gripper_name)
        self.gripper_name.extend(joint.child_link.get_name() for joint, _multiplier, _offset in self.left_gripper)
        self.gripper_name.extend(joint.child_link.get_name() for joint, _multiplier, _offset in self.right_gripper)

        self.left_camera = self._camera_link(self.left_entity, "left")
        self.right_camera = self._camera_link(self.right_entity, "right")
        self._configure_drive_properties()

    @staticmethod
    def _require_joint(entity: Any, name: str) -> Any:
        joint = entity.find_joint_by_name(name)
        if joint is None:
            raise RuntimeError(f"RoboTwin embodiment is missing required joint {name!r}")
        return joint

    def _gripper_joints(self, entity: Any, config: dict[str, Any]) -> list[tuple[Any, float, float]]:
        base_name = str(config["base"])
        result = [(self._require_joint(entity, base_name), 1.0, 0.0)]
        for mimic in config.get("mimic", []):
            if len(mimic) != 3:
                raise ValueError("RoboTwin gripper mimic entries must be [joint_name, multiplier, offset]")
            result.append((self._require_joint(entity, str(mimic[0])), float(mimic[1]), float(mimic[2])))
        return result

    @staticmethod
    def _camera_link(entity: Any, side: str) -> Any:
        for name in (f"{side}_camera", "camera"):
            link = entity.find_link_by_name(name)
            if link is not None:
                return link
        links = entity.get_links()
        if not links:
            raise RuntimeError(f"RoboTwin {side} arm has no links for wrist camera attachment")
        return links[0]

    def _configure_drive_properties(self) -> None:
        """Set arm and gripper PD terms in the same final order as RoboTwin.

        A dual-arm embodiment has one articulation, so applying each side's
        gripper properties inside its own active-joint pass can overwrite the
        other side's gripper with arm settings. Apply both grippers last.
        """
        for joint in self.left_arm_joints:
            joint.set_drive_property(stiffness=self.left_joint_stiffness, damping=self.left_joint_damping)
        for joint in self.right_arm_joints:
            joint.set_drive_property(stiffness=self.right_joint_stiffness, damping=self.right_joint_damping)
        for joint, _multiplier, _offset in self.left_gripper:
            joint.set_drive_property(stiffness=self.left_gripper_stiffness, damping=self.left_gripper_damping)
        for joint, _multiplier, _offset in self.right_gripper:
            joint.set_drive_property(stiffness=self.right_gripper_stiffness, damping=self.right_gripper_damping)

    def move_to_homestate(self) -> None:
        self._set_arm_drive_targets(self.left_arm_joints, self.left_homestate, "left")
        self._set_arm_drive_targets(self.right_arm_joints, self.right_homestate, "right")

    def set_origin_endpose(self) -> None:
        self.left_original_pose = self.get_left_ee_pose()
        self.right_original_pose = self.get_right_ee_pose()

    def set_arm_joints(self, target_position: Any, target_velocity: Any, arm_tag: str) -> None:
        joints = self.left_arm_joints if arm_tag == "left" else self.right_arm_joints
        target = np.asarray(target_position, dtype=np.float64).reshape(-1)
        velocity = np.asarray(target_velocity, dtype=np.float64).reshape(-1)
        if target.size != len(joints) or velocity.size != len(joints):
            raise ValueError(f"{arm_tag} arm target and velocity must each have {len(joints)} values")
        self._apply_passive_forces()
        for joint, position, speed in zip(joints, target, velocity, strict=True):
            joint.set_drive_target(float(position))
            joint.set_drive_velocity_target(float(speed))

    def _apply_drive_overrides(self, kwargs: dict[str, Any]) -> None:
        joint_stiffness = kwargs.get("joint_stiffness")
        joint_damping = kwargs.get("joint_damping")
        gripper_stiffness = kwargs.get("gripper_stiffness")
        gripper_damping = kwargs.get("gripper_damping")
        if joint_stiffness is not None:
            self.left_joint_stiffness = self.right_joint_stiffness = float(joint_stiffness)
        if joint_damping is not None:
            self.left_joint_damping = self.right_joint_damping = float(joint_damping)
        if gripper_stiffness is not None:
            self.left_gripper_stiffness = self.right_gripper_stiffness = float(gripper_stiffness)
        if gripper_damping is not None:
            self.left_gripper_damping = self.right_gripper_damping = float(gripper_damping)

    def set_gripper(
        self,
        gripper_val: float,
        arm_tag: str,
        gripper_eps: float = 0.1,
        gripper_velocity: float | None = None,
    ) -> None:
        gripper_value = float(np.clip(gripper_val, 0.0, 1.0))
        if arm_tag == "left":
            joints = self.left_gripper
            scale = self.left_gripper_scale
            current = self.get_normal_real_gripper_val()[0]
            self.left_gripper_val = gripper_value
        else:
            joints = self.right_gripper
            scale = self.right_gripper_scale
            current = self.get_normal_real_gripper_val()[1]
            self.right_gripper_val = gripper_value
        if not joints:
            return
        if (gripper_value - current > gripper_eps and gripper_eps > 0) or (
            gripper_value - current < gripper_eps and gripper_eps < 0
        ):
            gripper_value = current + gripper_eps
        physical_target = scale[0] + gripper_value * (scale[1] - scale[0])
        self._apply_passive_forces()
        for joint, multiplier, offset in joints:
            drive_target = physical_target * multiplier + offset
            current_target = float(joint.get_drive_target()[0])
            joint.set_drive_target(float(drive_target))
            if gripper_velocity is None:
                velocity_target = np.clip(drive_target - current_target, -1.0, 1.0) * 0.05
            else:
                velocity_target = float(gripper_velocity) * (scale[1] - scale[0]) * multiplier
            joint.set_drive_velocity_target(float(velocity_target))

    def get_left_gripper_val(self) -> float:
        return float(self.left_gripper_val)

    def get_right_gripper_val(self) -> float:
        return float(self.right_gripper_val)

    def get_normal_real_gripper_val(self) -> tuple[float, float]:
        return (
            self._normalized_drive_target(self.left_gripper, self.left_gripper_scale),
            self._normalized_drive_target(self.right_gripper, self.right_gripper_scale),
        )

    @staticmethod
    def _normalized_drive_target(joints: list[tuple[Any, float, float]], scale: tuple[float, float]) -> float:
        if not joints:
            return 0.0
        joint, multiplier, offset = joints[0]
        if multiplier == 0:
            return 0.0
        physical = (float(joint.get_drive_target()[0]) - offset) / multiplier
        span = scale[1] - scale[0]
        return float(np.clip((physical - scale[0]) / span, 0.0, 1.0)) if span else 0.0

    def get_left_arm_jointState(self) -> list[float]:
        return [float(joint.get_drive_target()[0]) for joint in self.left_arm_joints] + [self.get_left_gripper_val()]

    def get_right_arm_jointState(self) -> list[float]:
        return [float(joint.get_drive_target()[0]) for joint in self.right_arm_joints] + [self.get_right_gripper_val()]

    def get_left_ee_pose(self) -> list[float]:
        return self._trans_endpose("left", is_tcp=False)

    def get_right_ee_pose(self) -> list[float]:
        return self._trans_endpose("right", is_tcp=False)

    def get_left_tcp_pose(self) -> list[float]:
        return self._trans_endpose("left", is_tcp=True)

    def get_right_tcp_pose(self) -> list[float]:
        return self._trans_endpose("right", is_tcp=True)

    def _trans_endpose(self, arm_tag: str, *, is_tcp: bool) -> list[float]:
        t3d = importlib.import_module("transforms3d")
        if arm_tag == "left":
            gripper_bias = self.left_gripper_bias
            global_transform = self.left_global_trans_matrix
            delta_transform = self.left_delta_matrix
            ee_pose = self.left_ee.global_pose
        else:
            gripper_bias = self.right_gripper_bias
            global_transform = self.right_global_trans_matrix
            delta_transform = self.right_delta_matrix
            ee_pose = self.right_ee.global_pose
        endpose = np.eye(4)
        endpose[:3, :3] = t3d.quaternions.quat2mat(ee_pose.q) @ global_transform @ delta_transform
        offset = gripper_bias if is_tcp else gripper_bias - 0.12
        endpose[:3, 3] = ee_pose.p + endpose[:3, :3] @ np.asarray([offset, 0.0, 0.0])
        return endpose[:3, 3].tolist() + t3d.quaternions.mat2quat(endpose[:3, :3]).tolist()

    def _apply_passive_forces(self) -> None:
        seen: set[int] = set()
        for entity in (self.left_entity, self.right_entity):
            identity = id(entity)
            if identity in seen:
                continue
            seen.add(identity)
            entity.set_qf(entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))

    @staticmethod
    def _set_arm_drive_targets(joints: list[Any], targets: list[Any], arm_tag: str) -> None:
        if len(targets) != len(joints):
            raise ValueError(f"RoboTwin {arm_tag} homestate has {len(targets)} values for {len(joints)} joints")
        for joint, target in zip(joints, targets, strict=True):
            joint.set_drive_target(float(target))
