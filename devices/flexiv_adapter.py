#!/usr/bin/env python3
"""
Flexiv RDK adapter layer for robot device integration.

This module centralizes backend selection, version-aware method dispatch,
and lightweight robot/gripper/tool handles for Flexiv RDK integrations.

Author: Zixi Ying
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import argparse
import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from utils.flexiv_utils import (
    MethodCandidate,
    call_versioned_method,
    detect_pybind_rdk_version,
    detect_python_rdk_version,
    flexiv_pose_wxyz_to_matrix,
    format_version,
    infer_flange_frame_from_urdf,
)
from utils.logger_config import logger


class RobotPipelineBase:
    """Backend-specific robot call pipeline."""

    def __init__(self, sdk: Any, rdk_version: Tuple[int, int, int]) -> None:
        self.sdk = sdk
        self.rdk_version = rdk_version

    def switch_mode(self, robot: Any, mode: Any) -> Any:
        raise NotImplementedError

    def send_joint_position(
        self,
        robot: Any,
        positions: List[float],
        velocities: List[float],
        accelerations: Optional[List[float]],
        max_vel: List[float],
        max_acc: List[float],
    ) -> Any:
        raise NotImplementedError

    def send_cartesian_motion_force(
        self,
        robot: Any,
        pose: Any,
        wrench: Optional[Any],
        velocity: Optional[Any],
        max_linear_vel: float,
        max_angular_vel: float,
        max_linear_acc: float,
        max_angular_acc: float,
    ) -> Any:
        raise NotImplementedError

    def clear_fault(self, robot: Any) -> Any:
        raise NotImplementedError

    def enable(self, robot: Any) -> Any:
        raise NotImplementedError

    def fault(self, robot: Any) -> Any:
        raise NotImplementedError

    def is_operational(self, robot: Any) -> Any:
        raise NotImplementedError

    def states(self, robot: Any) -> Any:
        raise NotImplementedError


class PythonRobotPipeline(RobotPipelineBase):
    """Version-aware control path for the official flexivrdk Python SDK."""

    def switch_mode(self, robot: Any, mode: Any) -> Any:
        return call_versioned_method(
            robot,
            self.rdk_version,
            (MethodCandidate("SwitchMode"),),
            mode,
        )

    def send_joint_position(
        self,
        robot: Any,
        positions: List[float],
        velocities: List[float],
        accelerations: Optional[List[float]],
        max_vel: List[float],
        max_acc: List[float],
    ) -> Any:
        candidates = (MethodCandidate("SendJointPosition"),)
        if self.rdk_version >= (1, 8, 0):
            return call_versioned_method(
                robot, self.rdk_version, candidates, positions, velocities, max_vel, max_acc
            )
        normalized_accelerations = (
            [float(v) for v in accelerations]
            if accelerations is not None
            else [0.0] * len(positions)
        )
        return call_versioned_method(
            robot,
            self.rdk_version,
            candidates,
            positions,
            velocities,
            normalized_accelerations,
            max_vel,
            max_acc,
        )

    def send_cartesian_motion_force(
        self,
        robot: Any,
        pose: Any,
        wrench: Optional[Any],
        velocity: Optional[Any],
        max_linear_vel: float,
        max_angular_vel: float,
        max_linear_acc: float,
        max_angular_acc: float,
    ) -> Any:
        wrench = [0.0] * 6 if wrench is None else wrench
        velocity = [0.0] * 6 if velocity is None else velocity
        candidates = (
            MethodCandidate("SendCartesianMotionForce"),
        )
        if self.rdk_version >= (1, 8, 0):
            return call_versioned_method(
                robot,
                self.rdk_version,
                candidates,
                pose,
                wrench,
                velocity,
                max_linear_vel,
                max_angular_vel,
                max_linear_acc,
                max_angular_acc,
            )
        return call_versioned_method(
            robot,
            self.rdk_version,
            candidates,
            pose,
            wrench,
            max_linear_vel,
            max_angular_vel,
            max_linear_acc,
            max_angular_acc,
        )

    def clear_fault(self, robot: Any) -> Any:
        return call_versioned_method(
            robot, self.rdk_version, (MethodCandidate("ClearFault"),)
        )

    def enable(self, robot: Any) -> Any:
        return call_versioned_method(
            robot, self.rdk_version, (MethodCandidate("Enable"),)
        )

    def fault(self, robot: Any) -> Any:
        return call_versioned_method(
            robot, self.rdk_version, (MethodCandidate("fault"),)
        )

    def is_operational(self, robot: Any) -> Any:
        return call_versioned_method(
            robot,
            self.rdk_version,
            (MethodCandidate("operational"),),
        )

    def states(self, robot: Any) -> Any:
        return call_versioned_method(
            robot, self.rdk_version, (MethodCandidate("states"),)
        )


class PybindRobotPipeline(RobotPipelineBase):
    """Version-aware control path for flexiv_rdk_ext pybind backend."""

    def switch_mode(self, robot: Any, mode: Any) -> Any:
        return call_versioned_method(
            robot,
            self.rdk_version,
            (MethodCandidate("switch_mode"),),
            int(mode),
        )

    def send_joint_position(
        self,
        robot: Any,
        positions: List[float],
        velocities: List[float],
        accelerations: Optional[List[float]],
        max_vel: List[float],
        max_acc: List[float],
    ) -> Any:
        # pybind API currently does not expose accelerations in send_joint_position.
        _ = accelerations
        return call_versioned_method(
            robot,
            self.rdk_version,
            (MethodCandidate("send_joint_position"),),
            positions,
            velocities,
            max_vel,
            max_acc,
        )

    def send_cartesian_motion_force(
        self,
        robot: Any,
        pose: Any,
        wrench: Optional[Any],
        velocity: Optional[Any],
        max_linear_vel: float,
        max_angular_vel: float,
        max_linear_acc: float,
        max_angular_acc: float,
    ) -> Any:
        wrench = [0.0] * 6 if wrench is None else wrench
        velocity = [0.0] * 6 if velocity is None else velocity
        return call_versioned_method(
            robot,
            self.rdk_version,
            (MethodCandidate("send_cartesian_motion_force"),),
            pose,
            wrench,
            velocity,
            max_linear_vel,
            max_angular_vel,
            max_linear_acc,
            max_angular_acc,
        )

    def clear_fault(self, robot: Any) -> Any:
        return call_versioned_method(
            robot,
            self.rdk_version,
            (MethodCandidate("clear_fault"),),
            30,
        )

    def enable(self, robot: Any) -> Any:
        return call_versioned_method(
            robot, self.rdk_version, (MethodCandidate("enable"),)
        )

    def fault(self, robot: Any) -> Any:
        return call_versioned_method(
            robot, self.rdk_version, (MethodCandidate("fault"),)
        )

    def is_operational(self, robot: Any) -> Any:
        return call_versioned_method(
            robot,
            self.rdk_version,
            (MethodCandidate("operational"),),
        )

    def states(self, robot: Any) -> Any:
        class PybindStates:
            def __init__(self, robot_obj: Any) -> None:
                self.q = list(robot_obj.q())
                dq_fn = getattr(robot_obj, "dq", None)
                if not callable(dq_fn):
                    raise RuntimeError(
                        "dq is unavailable in current flexiv_rdk_ext build. Stop execution."
                    )
                self.dq = list(dq_fn())
                self.tcp_pose = list(robot_obj.tcp_pose())

        return PybindStates(robot)


class FlexivRobotHandle:
    """Robot wrapper to decouple call sites from raw flexivrdk object."""

    def __init__(
        self,
        robot: Any,
        rdk_version: Tuple[int, int, int],
        backend: str,
        sdk: Any,
        pipeline: RobotPipelineBase,
    ) -> None:
        self._robot = robot
        self._rdk_version = rdk_version
        self._backend = backend
        self._sdk = sdk
        self._pipeline = pipeline
        self._model: Optional[Any] = None

    @property
    def raw(self) -> Any:
        return self._robot

    def switch_mode(self, mode: Any) -> Any:
        return self._pipeline.switch_mode(self._robot, mode)

    def send_joint_position(
        self,
        positions: List[float],
        velocities: List[float],
        max_vel: List[float],
        max_acc: List[float],
        accelerations: Optional[List[float]] = None,
    ) -> Any:
        """Wrap Flexiv RDK SendJointPosition with backend/version dispatch.

        accelerations defaults to zeros when required by older RDK APIs.
        """
        return self._pipeline.send_joint_position(
            self._robot, positions, velocities, accelerations, max_vel, max_acc
        )

    def send_cartesian_motion_force(
        self,
        pose: Any,
        wrench: Optional[Any] = None,
        velocity: Optional[Any] = None,
        max_linear_vel: float = 0.5,
        max_angular_vel: float = 1.0,
        max_linear_acc: float = 2.0,
        max_angular_acc: float = 5.0,
    ) -> Any:
        """Wrap Flexiv RDK SendCartesianMotionForce with backend/version dispatch."""
        return self._pipeline.send_cartesian_motion_force(
            self._robot,
            pose,
            wrench,
            velocity,
            max_linear_vel,
            max_angular_vel,
            max_linear_acc,
            max_angular_acc,
        )

    def set_force_control_axis(self, axis: Any) -> Any:
        if self._backend == "pybind":
            return self._robot.set_force_control_axis([bool(v) for v in axis], [1.0, 1.0, 1.0])
        return call_versioned_method(
            self._robot,
            self._rdk_version,
            (MethodCandidate("SetForceControlAxis"), MethodCandidate("set_force_control_axis")),
            axis,
        )

    def set_cartesian_impedance(self, impedance: Any) -> Any:
        if self._backend == "pybind":
            return self._robot.set_cartesian_impedance(
                list(impedance), [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
            )
        return call_versioned_method(
            self._robot,
            self._rdk_version,
            (MethodCandidate("SetCartesianImpedance"), MethodCandidate("set_cartesian_impedance")),
            impedance,
        )

    def set_joint_impedance(
        self,
        k_q: Any,
        z_q: Optional[Any] = None,
    ) -> Any:
        if self._backend == "pybind":
            damping = [] if z_q is None else list(z_q)
            return self._robot.set_joint_impedance(list(k_q), damping)
        return call_versioned_method(
            self._robot,
            self._rdk_version,
            (MethodCandidate("SetJointImpedance"), MethodCandidate("set_joint_impedance")),
            list(k_q),
            [] if z_q is None else list(z_q),
        )

    def set_max_contact_wrench(self, wrench: List[float]) -> Any:
        """
        Wrap flexiv::rdk::Robot::SetMaxContactWrench().

        Args:
            wrench: 6-vector [Fx, Fy, Fz, Mx, My, Mz] in N / N*m. Each value
                must be positive; the SDK enforces this safety limit on the
                external contact wrench during NRT_CARTESIAN_MOTION_FORCE.

        Effect:
            When the measured contact wrench exceeds the limit, the robot
            automatically enters IN_REDUCED_STATE (slows/halts motion) and
            recovers automatically once the contact is released.
        """
        if len(wrench) != 6:
            raise ValueError(
                f"max_contact_wrench must be length 6, got {len(wrench)}."
            )
        return self._robot.SetMaxContactWrench(list(wrench))

    def operational_status(self) -> Any:
        """Return the current Flexiv OperationalStatus enum value."""
        return self._robot.operational_status()

    def clear_fault(self) -> Any:
        return self._pipeline.clear_fault(self._robot)

    def enable(self) -> Any:
        return self._pipeline.enable(self._robot)

    def fault(self) -> Any:
        return self._pipeline.fault(self._robot)

    def is_operational(self) -> Any:
        return self._pipeline.is_operational(self._robot)

    def states(self) -> Any:
        return self._pipeline.states(self._robot)

    def reachable(
        self,
        pose: Sequence[float],
        seed_positions: Sequence[float],
        free_orientation: bool = False,
    ) -> Tuple[bool, List[float]]:
        """Check whether target Cartesian pose is reachable and return one IK solution."""
        pose_list = np.asarray(pose, dtype=float).reshape(-1).tolist()
        seed_list = np.asarray(seed_positions, dtype=float).reshape(-1).tolist()
        if len(pose_list) != 7:
            raise ValueError(f"pose length must be 7, got {len(pose_list)}")
        if len(seed_list) == 0:
            raise ValueError("seed_positions must be non-empty")

        def normalize_result(raw: Any) -> Tuple[bool, List[float]]:
            if isinstance(raw, (tuple, list)) and len(raw) == 2:
                is_reach = bool(raw[0])
                ik_sol = np.asarray(raw[1], dtype=float).reshape(-1).tolist()
                return is_reach, ik_sol
            raise RuntimeError(f"Unexpected reachable() return type: {type(raw)}")

        if self._backend == "pybind":
            robot_reachable_fn = getattr(self._robot, "reachable", None)
            if callable(robot_reachable_fn):
                return normalize_result(
                    robot_reachable_fn(pose_list, seed_list, bool(free_orientation))
                )
            raise RuntimeError(
                "pybind backend does not expose reachable(). "
                "Please rebuild flexiv_rdk_ext with reachable binding."
            )

        if self._model is None:
            model_ctor = getattr(self._sdk, "Model", None)
            if not callable(model_ctor):
                raise RuntimeError("flexivrdk.Model is unavailable in current python backend")
            self._model = model_ctor(self._robot)
        model_reachable_fn = getattr(self._model, "reachable", None)
        if callable(model_reachable_fn):
            return normalize_result(
                model_reachable_fn(pose_list, seed_list, bool(free_orientation))
            )
        raise RuntimeError(
            "reachable() is unavailable in current backend/build. "
            "Please check your Flexiv RDK installation."
        )

    def sync_urdf(self, template_urdf_path: str) -> Any:
        """Sync the connected robot's actual kinematic parameters into a template URDF."""
        if not template_urdf_path:
            raise ValueError("template_urdf_path must be non-empty")
        if self._backend == "pybind":
            robot_sync_urdf_fn = getattr(self._robot, "sync_urdf", None)
            if callable(robot_sync_urdf_fn):
                return robot_sync_urdf_fn(template_urdf_path)
            raise RuntimeError(
                "pybind backend does not expose sync_urdf(). "
                "Please rebuild flexiv_rdk_ext with sync_urdf binding."
            )

        if self._model is None:
            model_ctor = getattr(self._sdk, "Model", None)
            if not callable(model_ctor):
                raise RuntimeError("flexivrdk.Model is unavailable in current python backend")
            self._model = model_ctor(self._robot)

        sync_urdf_fn = getattr(self._model, "SyncURDF", None)
        if callable(sync_urdf_fn):
            return sync_urdf_fn(template_urdf_path)
        sync_urdf_fn = getattr(self._model, "sync_urdf", None)
        if callable(sync_urdf_fn):
            return sync_urdf_fn(template_urdf_path)
        raise RuntimeError(
            "SyncURDF is unavailable in current flexivrdk.Model build. "
            "Please check your Flexiv RDK installation."
        )

    def start_joint_position_planner(
        self,
        control_dt: float,
        max_velocity: Optional[Sequence[float]] = None,
        max_acceleration: Optional[Sequence[float]] = None,
        max_jerk: Optional[Sequence[float]] = None,
        use_rt: bool = True,
    ) -> Any:
        """Start the C++ background joint planner worker."""
        self._require_pybind_planner("start_joint_position_planner")
        planner_fn = self._require_pybind_method("start_joint_position_planner")
        return planner_fn(
            float(control_dt),
            [] if max_velocity is None else np.asarray(max_velocity, dtype=float).reshape(-1).tolist(),
            []
            if max_acceleration is None
            else np.asarray(max_acceleration, dtype=float).reshape(-1).tolist(),
            [] if max_jerk is None else np.asarray(max_jerk, dtype=float).reshape(-1).tolist(),
            bool(use_rt),
        )

    def add_joint_position_plan(self, time_s: float, joints: Sequence[float]) -> Any:
        """Append one timed joint waypoint to the C++ planner worker.

        Args:
            time_s: Logical time of this waypoint on the planner segment timeline, in seconds.
                It is not wall-clock or Unix time. The first waypoint after an empty planner
                state must use ``time_s == 0``; later waypoints must have strictly increasing
                ``time_s``. The C++ worker maps this timeline to playback when execution starts.
            joints: Target joint positions (length equals arm DoF).
        """
        self._require_pybind_planner("add_joint_position_plan")
        planner_fn = self._require_pybind_method("add_joint_position_plan")
        return planner_fn(float(time_s), np.asarray(joints, dtype=float).reshape(-1).tolist())

    def start_cartesian_position_planner(
        self,
        seed_positions: Sequence[float],
        control_dt: float,
        urdf_path: str,
        max_velocity: Optional[Sequence[float]] = None,
        max_acceleration: Optional[Sequence[float]] = None,
        max_jerk: Optional[Sequence[float]] = None,
        free_orientation: bool = False,
        use_rt: bool = True,
        ik_dt: float = 0.1,
        ik_max_iters: int = 1000,
        ik_eps: float = 1e-5,
        ik_damping: float = 1e-6,
        world_frame_transform: Optional[Sequence[Sequence[float]]] = None,
        tool: Optional[Any] = None,
        use_traj_smooth: bool = False,
        traj_smooth_upsample_ratio: int = 4,
        traj_smooth_filter_window: int = 7,
        nrt_vel_ff: bool = False,
        nrt_direct_waypoint: bool = False,
        use_joint_impedance: bool = False,
        joint_impedance_k_q: Optional[Sequence[float]] = None,
        joint_impedance_z_q: Optional[Sequence[float]] = None,
    ) -> Any:
        """Start the C++ background Cartesian planner worker."""
        self._require_pybind_planner("start_cartesian_position_planner")
        planner_fn = self._require_pybind_method("start_cartesian_position_planner")
        tool_handle = tool if tool is not None else self._sdk.Tool(self._robot)
        raw_tool = tool_handle.raw if hasattr(tool_handle, "raw") else tool_handle
        tcp_frame_transform = flexiv_pose_wxyz_to_matrix(raw_tool.params().tcp_location)
        urdf_ee_frame = infer_flange_frame_from_urdf(urdf_path)
        return planner_fn(
            np.asarray(seed_positions, dtype=float).reshape(-1).tolist(),
            float(control_dt),
            str(urdf_path),
            str(urdf_ee_frame),
            [] if max_velocity is None else np.asarray(max_velocity, dtype=float).reshape(-1).tolist(),
            []
            if max_acceleration is None
            else np.asarray(max_acceleration, dtype=float).reshape(-1).tolist(),
            [] if max_jerk is None else np.asarray(max_jerk, dtype=float).reshape(-1).tolist(),
            bool(free_orientation),
            bool(use_rt),
            float(ik_dt),
            int(ik_max_iters),
            float(ik_eps),
            float(ik_damping),
            []
            if world_frame_transform is None
            else np.asarray(world_frame_transform, dtype=float).reshape(-1).tolist(),
            np.asarray(tcp_frame_transform, dtype=float).reshape(-1).tolist(),
            bool(use_traj_smooth),
            int(traj_smooth_upsample_ratio),
            int(traj_smooth_filter_window),
            bool(nrt_vel_ff),
            bool(nrt_direct_waypoint),
            bool(use_joint_impedance),
            []
            if joint_impedance_k_q is None
            else np.asarray(joint_impedance_k_q, dtype=float).reshape(-1).tolist(),
            []
            if joint_impedance_z_q is None
            else np.asarray(joint_impedance_z_q, dtype=float).reshape(-1).tolist(),
        )

    def add_cartesian_position_plan(self, time_s: float, pose: Sequence[float]) -> Any:
        """Append one timed Cartesian waypoint to the C++ planner worker.

        Args:
            time_s: Logical time of this waypoint on the planner segment timeline, in seconds.
                Same convention as :meth:`add_joint_position_plan`: not wall-clock or Unix time;
                first waypoint after an empty planner state must use ``time_s == 0``; later
                waypoints must have strictly increasing ``time_s``. Each pose is IK-solved into
                joint space and merged into the same timed joint queue as joint waypoints.
            pose: Flexiv pose ``[x, y, z, qw, qx, qy, qz]`` (7 values).
        """
        self._require_pybind_planner("add_cartesian_position_plan")
        planner_fn = self._require_pybind_method("add_cartesian_position_plan")
        return planner_fn(float(time_s), np.asarray(pose, dtype=float).reshape(-1).tolist())

    def stop_position_planner(self) -> Any:
        """Stop the C++ background planner worker."""
        self._require_pybind_planner("stop_position_planner")
        return self._require_pybind_method("stop_position_planner")()

    def clear_position_plan(self) -> Any:
        """Clear pending C++ planner waypoints and trajectory points."""
        self._require_pybind_planner("clear_position_plan")
        return self._require_pybind_method("clear_position_plan")()

    def position_planner_status(self) -> Any:
        """Return C++ planner worker status."""
        self._require_pybind_planner("position_planner_status")
        return self._require_pybind_method("position_planner_status")()

    def position_planner_command_trace(self) -> Any:
        """Return C++ planner commands as [[relative_time_s, joint_positions], ...]."""
        self._require_pybind_planner("position_planner_command_trace")
        trace_fn = self._require_pybind_method("position_planner_command_trace")
        return [
            [float(time_s), np.asarray(joints, dtype=float).reshape(-1).tolist()]
            for time_s, joints in trace_fn()
        ]

    def _require_pybind_planner(self, method_name: str) -> None:
        if self._backend != "pybind":
            raise RuntimeError(f"{method_name}() requires backend=pybind")

    def _require_pybind_method(self, method_name: str) -> Any:
        method = getattr(self._robot, method_name, None)
        if callable(method):
            return method
        raise RuntimeError(
            f"pybind backend does not expose {method_name}(). Please rebuild flexiv_rdk_ext."
        )


class FlexivGripperHandle:
    """Gripper wrapper to decouple call sites from raw flexivrdk object."""

    def __init__(self, gripper: Any, rdk_version: Tuple[int, int, int]) -> None:
        self._gripper = gripper
        self._rdk_version = rdk_version

    @property
    def raw(self) -> Any:
        return self._gripper

    def enable(self, gripper_name: str) -> Any:
        return call_versioned_method(
            self._gripper,
            self._rdk_version,
            (MethodCandidate("Enable"), MethodCandidate("enable")),
            gripper_name,
        )

    def move(self, width: float, velocity: float, force: float) -> Any:
        return call_versioned_method(
            self._gripper,
            self._rdk_version,
            (MethodCandidate("Move"), MethodCandidate("move")),
            width,
            velocity,
            force,
        )

    def params(self) -> Any:
        return call_versioned_method(
            self._gripper,
            self._rdk_version,
            (MethodCandidate("params"), MethodCandidate("Params")),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gripper, name)


class FlexivToolHandle:
    """Tool wrapper to decouple call sites from raw flexivrdk object."""

    def __init__(self, tool: Any, rdk_version: Tuple[int, int, int]) -> None:
        self._tool = tool
        self._rdk_version = rdk_version

    @property
    def raw(self) -> Any:
        return self._tool

    def switch(self, tool_name: str) -> Any:
        return call_versioned_method(
            self._tool,
            self._rdk_version,
            (MethodCandidate("Switch"), MethodCandidate("switch")),
            tool_name,
        )

    def name(self) -> str:
        return str(
            call_versioned_method(
                self._tool,
                self._rdk_version,
                (MethodCandidate("name"), MethodCandidate("Name")),
            )
        )

    def params(self) -> Any:
        return call_versioned_method(
            self._tool,
            self._rdk_version,
            (MethodCandidate("params"), MethodCandidate("Params")),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)


class FlexivAdapter:
    """
    Thin adapter around Flexiv RDK module symbols used by robot devices.

    Examples:
        adapter = FlexivAdapter()
        adapter = FlexivAdapter(backend="python")
        adapter = FlexivAdapter(backend="pybind")
        robot = adapter.create_robot("Rizon4R-062070")
    """

    def __init__(self, backend: str = "python") -> None:
        self._backend = self._parse_backend(backend)
        self._sdk = self._import_backend_sdk(self._backend)

        if self._backend == "python":
            self._rdk_version = detect_python_rdk_version(self._sdk)
            self._pipeline: RobotPipelineBase = PythonRobotPipeline(
                self._sdk, self._rdk_version
            )
        else:
            self._rdk_version = self._resolve_pybind_version(self._sdk)
            self._pipeline = PybindRobotPipeline(self._sdk, self._rdk_version)

        logger.info(
            "FlexivAdapter backend selected: {}, rdk_version={}",
            self._backend,
            format_version(self._rdk_version),
        )

    def create_robot(
        self,
        serial_number: str,
        network_interface_whitelist: Optional[Sequence[str]] = None,
    ) -> FlexivRobotHandle:
        whitelist = list(network_interface_whitelist or [])
        if self._backend == "python":
            robot = self._sdk.Robot(serial_number, whitelist)
        else:
            if whitelist:
                logger.warning(
                    "network_interface_whitelist is not supported by the pybind "
                    "backend and will be ignored: {}",
                    whitelist,
                )
            robot = self._sdk.FlexivRobot(serial_number)
        return FlexivRobotHandle(
            robot,
            self._rdk_version,
            self._backend,
            self._sdk,
            self._pipeline,
        )

    def create_gripper(self, robot: Any) -> FlexivGripperHandle:
        raw_robot = robot.raw if hasattr(robot, "raw") else robot
        return FlexivGripperHandle(self._sdk.Gripper(raw_robot), self._rdk_version)

    def create_tool(self, robot: Any) -> FlexivToolHandle:
        raw_robot = robot.raw if hasattr(robot, "raw") else robot
        return FlexivToolHandle(self._sdk.Tool(raw_robot), self._rdk_version)

    @property
    def mode_joint_position(self) -> Any:
        if self._backend == "pybind":
            return int(self._sdk.Mode.NRT_JOINT_POSITION)
        return self._sdk.Mode.NRT_JOINT_POSITION

    @property
    def mode_cartesian_motion_force(self) -> Any:
        if self._backend == "pybind":
            return int(self._sdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        return self._sdk.Mode.NRT_CARTESIAN_MOTION_FORCE

    @staticmethod
    def _parse_backend(backend: str) -> str:
        backend_normalized = backend.strip().lower()
        if backend_normalized in {"python", "pybind"}:
            return backend_normalized
        if backend_normalized.startswith("pybind_"):
            raise ValueError(
                "Manual pybind backend version suffix is no longer supported. "
                "Use backend='pybind' and rely on flexiv_rdk_ext.__rdk_version__ auto-detection."
            )
        raise ValueError("backend must be one of: python, pybind")

    @staticmethod
    def _import_backend_sdk(backend: str) -> Any:
        if backend == "python":
            try:
                import flexivrdk  # type: ignore[import-not-found]

                return flexivrdk
            except ImportError as exc:
                raise ImportError("backend=python but flexivrdk is not importable") from exc

        try:
            import flexiv_rdk_ext  # type: ignore[import-not-found]

            return flexiv_rdk_ext
        except ImportError as exc:
            raise ImportError(
                f"backend=pybind but flexiv_rdk_ext is not importable: {exc}"
            ) from exc

    @staticmethod
    def _resolve_pybind_version(sdk: Any) -> Tuple[int, int, int]:
        compiled_version = detect_pybind_rdk_version(sdk)
        if compiled_version is None:
            raise RuntimeError(
                "flexiv_rdk_ext does not export __rdk_version__. "
                "Rebuild it with the updated CMake and FLEXIV_RDK_API_VERSION."
            )
        logger.info(
            "Detected flexiv_rdk_ext compiled RDK version: {}",
            format_version(compiled_version),
        )
        return compiled_version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexiv adapter quick utility entry")
    parser.add_argument(
        "--task",
        type=str,
        default="read_states",
        help="One of: test_rdk, read_states, move_to_joints, move_to_pose",
    )
    parser.add_argument("--robot-sn", type=str, default="Rizon4R-062717")
    parser.add_argument(
        "--target",
        type=float,
        nargs=7,
        default=[1.336748, 1.111077, 0.987856, 2.054602, 0.719250, 2.156180, 1.131322],
        help=(
            "Target points (7 floats). "
            "For move_to_joints: [j1..j7]. "
            "For move_to_pose: [x, y, z, qw, qx, qy, qz]."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="python",
        help="Flexiv backend: python / pybind",
    )
    args = parser.parse_args()

    task = str(args.task).strip().lower()
    target_points: List[float] = np.asarray(args.target, dtype=float).reshape(-1).tolist()
    if task not in {"test_rdk", "read_states", "move_to_joints", "move_to_pose"}:
        raise ValueError(
            "task must be one of: test_rdk, read_states, move_to_joints, move_to_pose"
        )

    adapter = FlexivAdapter(backend=str(args.backend))
    robot = adapter.create_robot(str(args.robot_sn))

    if task == "test_rdk":
        print("test_rdk ok: adapter and robot created")
    elif task == "read_states":
        states = robot.states()
        joints = [round(float(v), 5) for v in np.asarray(states.q, dtype=float).reshape(-1)]
        pose = [
            round(float(v), 5)
            for v in np.asarray(states.tcp_pose, dtype=float).reshape(-1)
        ]
        tool = adapter.create_tool(robot)
        print("\n" + "=" * 80)
        print(f"joints: {joints}")
        print(f"pose_wxyz: {pose}")
        print(f"tool_name: {tool.name()}")
        print(f"tool_params: {tool.params()}")
    else:
        if robot.fault():
            logger.warning("Robot has faults, attempting to clear...")
            robot.clear_fault()

        robot.enable()
        while not robot.is_operational():
            time.sleep(0.5)

        if task == "move_to_pose":
            target_pose_wxyz = target_points
            print(f"target_pose_wxyz: {[round(float(v), 5) for v in target_pose_wxyz]}")
            robot.switch_mode(adapter.mode_cartesian_motion_force)
            time.sleep(1.0)
            robot.send_cartesian_motion_force(
                pose=np.asarray(target_pose_wxyz, dtype=float).reshape(-1).tolist(),
                max_linear_vel=0.5,
                max_angular_vel=1.0,
                max_linear_acc=2.0,
                max_angular_acc=5.0,
            )
        else:
            command_joints = target_points
            print(f"target_joints: {[round(float(v), 5) for v in command_joints]}")
            robot.switch_mode(adapter.mode_joint_position)
            time.sleep(1.0)
            robot.send_joint_position(
                positions=command_joints,
                velocities=[0.0] * len(command_joints),
                max_vel=[1.0] * len(command_joints),
                max_acc=[2.0] * len(command_joints),
            )

        time.sleep(2.0)
        states = robot.states()
        print("move done")
        print(f"joints: {[round(float(v), 5) for v in np.asarray(states.q, dtype=float).reshape(-1)]}")
        print(
            "pose_wxyz: "
            f"{[round(float(v), 5) for v in np.asarray(states.tcp_pose, dtype=float).reshape(-1)]}"
        )
