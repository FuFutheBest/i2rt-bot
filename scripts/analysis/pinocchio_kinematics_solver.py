#!/usr/bin/env python3
"""
C++ backed Pinocchio kinematics (FK/IK) solver.

Author: Zixi Ying
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Make project root importable for direct execution: python utils/xxx.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger_config import logger  # noqa: E402


def _load_cpp_solver_class() -> Any:
    """Load pinocchio_kinematics_ext.PinocchioKinematicsSolverCpp, trying the local build path first."""
    try:
        import pinocchio_kinematics_ext  # type: ignore[import-not-found]

        return pinocchio_kinematics_ext.PinocchioKinematicsSolverCpp
    except ImportError:
        build_dir = PROJECT_ROOT / "build" / "pinocchio_kinematics_ext"
        if str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))
        try:
            import pinocchio_kinematics_ext  # type: ignore[import-not-found]

            return pinocchio_kinematics_ext.PinocchioKinematicsSolverCpp
        except ImportError as second_error:
            raise ImportError(
                "Cannot import pinocchio_kinematics_ext.PinocchioKinematicsSolverCpp. Build the C++ extension first "
                "and make its runtime libraries visible via LD_LIBRARY_PATH."
            ) from second_error
        except AttributeError as attr_error:
            raise ImportError(
                "pinocchio_kinematics_ext was imported but does not expose PinocchioKinematicsSolverCpp. "
                "Please rebuild pinocchio_kinematics_ext."
            ) from attr_error
    except AttributeError as attr_error:
        raise ImportError(
            "pinocchio_kinematics_ext was imported but does not expose PinocchioKinematicsSolverCpp. "
            "Please rebuild pinocchio_kinematics_ext."
        ) from attr_error


def _flatten_optional_matrix(
    transform: Optional[Sequence[Sequence[float]]],
    name: str,
) -> list[float]:
    """Return [] for None or flattened row-major 4x4 matrix for the C++ solver."""
    if transform is None:
        return []
    arr = np.asarray(transform, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    return arr.reshape(-1).tolist()


def _flatten_pose(
    pose: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    name: str,
) -> list[float]:
    """Convert 4x4 matrix or Flexiv pose [x,y,z,qw,qx,qy,qz] to a flat list."""
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr.reshape(-1).tolist()
    flat = arr.reshape(-1)
    if int(flat.size) != 7:
        raise ValueError(f"{name} must be either 4x4 matrix or Flexiv pose [x,y,z,qw,qx,qy,qz]")
    return flat.tolist()


class PinocchioKinematicsSolver:
    """
    Pinocchio kinematics (FK/IK) backed by C++ `pinocchio_kinematics_ext`.

    Damped least-squares IK and frame-aligned FK run in C++.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        dt: float,
        max_iters: int,
        eps: float,
        ee_frame: str,
        world_frame_transform: Optional[Sequence[Sequence[float]]] = None,
        tcp_frame_transform: Optional[Sequence[Sequence[float]]] = None,
        damping: float = 1e-6,
    ) -> None:
        """Initialize the C++ solver, parameters, and frame alignment."""
        urdf = Path(urdf_path)
        if not urdf.is_absolute():
            urdf = PROJECT_ROOT / urdf
        if not urdf.exists():
            raise FileNotFoundError(f"URDF not found: {urdf}")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if damping < 0.0:
            raise ValueError("damping must be non-negative")

        self.urdf_path = urdf
        self.ee_frame = str(ee_frame)
        self.dt = float(dt)
        self.max_iters = int(max_iters)
        self.eps = float(eps)
        self.damping = float(damping)
        self.world_frame_transform = (
            None if world_frame_transform is None else np.asarray(world_frame_transform, dtype=np.float64)
        )
        self.tcp_frame_transform = (
            None if tcp_frame_transform is None else np.asarray(tcp_frame_transform, dtype=np.float64)
        )

        solver_cls = _load_cpp_solver_class()
        self._solver = solver_cls(
            str(urdf),
            self.dt,
            self.max_iters,
            self.eps,
            self.ee_frame,
            _flatten_optional_matrix(world_frame_transform, "world_frame_transform"),
            _flatten_optional_matrix(tcp_frame_transform, "tcp_frame_transform"),
            self.damping,
        )

    def solve_ik(
        self,
        q0: Sequence[float] | np.ndarray,
        tcp_pose: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """
        Solve IK from q0 toward tcp_pose.

        tcp_pose is first left-multiplied by world_frame_transform to align it
        with the URDF world frame.
        """
        result = self._solver.solve_ik(
            np.asarray(q0, dtype=np.float64).reshape(-1).tolist(),
            _flatten_pose(tcp_pose, "tcp_pose"),
            False,
        )
        return np.asarray(result, dtype=np.float64).copy()

    def compute_fk_pose_error(
        self,
        q: Sequence[float] | np.ndarray,
        tcp_pose: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    ) -> float:
        """Compute SE(3) pose error after running FK at q."""
        return float(
            self._solver.compute_fk_pose_error(
                np.asarray(q, dtype=np.float64).reshape(-1).tolist(),
                _flatten_pose(tcp_pose, "tcp_pose"),
            )
        )

    def compute_fk_to_fk_pose_error(
        self,
        q_ref: Sequence[float] | np.ndarray,
        q_est: Sequence[float] | np.ndarray,
    ) -> Tuple[float, float, float]:
        """Compute SE(3), position, and rotation error between FK(q_est) and FK(q_ref)."""
        result = self._solver.compute_fk_to_fk_pose_error(
            np.asarray(q_ref, dtype=np.float64).reshape(-1).tolist(),
            np.asarray(q_est, dtype=np.float64).reshape(-1).tolist(),
        )
        return float(result[0]), float(result[1]), float(result[2])

    def compute_raw_fk(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """
        Forward kinematics at ``ee_frame`` in the URDF world (Pinocchio ``oMf``).

        No ``world_frame_transform`` / ``tcp_frame_transform`` alignment. Returns
        Flexiv-style pose ``[x, y, z, qw, qx, qy, qz]`` for that placement.
        """
        result = self._solver.compute_raw_fk(np.asarray(q, dtype=np.float64).reshape(-1).tolist())
        return np.asarray(result, dtype=np.float64).copy()

    def compute_fk(self, q: Sequence[float] | np.ndarray) -> np.ndarray:
        """
        Forward kinematics in the robot TCP frame, consistent with ``solve_ik``.

        Applies ``inverse(world_frame_transform) * oMf_urdf * tcp_frame_transform``.
        Returns Flexiv-style pose ``[x, y, z, qw, qx, qy, qz]`` for that placement.
        """
        result = self._solver.compute_fk(np.asarray(q, dtype=np.float64).reshape(-1).tolist())
        return np.asarray(result, dtype=np.float64).copy()

    @property
    def last_success(self) -> bool:
        """Whether the last C++ IK solve converged."""
        return bool(self._solver.last_success)

    @property
    def last_iterations(self) -> int:
        """Iterations used by the last C++ IK solve."""
        return int(self._solver.last_iterations)

    @property
    def last_error_vector(self) -> np.ndarray:
        """SE(3) error vector from the last C++ IK solve."""
        return np.asarray(self._solver.last_error_vector, dtype=np.float64).copy()

    @property
    def last_error_norm(self) -> float:
        """Error norm from the last C++ IK solve."""
        return float(self._solver.last_error_norm)

    @property
    def nq(self) -> int:
        """Configuration space dimension from the loaded URDF model."""
        return int(self._solver.nq)

    @property
    def nv(self) -> int:
        """Velocity space dimension from the loaded URDF model."""
        return int(self._solver.nv)


def _angular_error_0_to_pi(q_ref: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    """Per-joint angular error mapped to [0, pi]."""
    diff = np.asarray(q_est, dtype=np.float64) - np.asarray(q_ref, dtype=np.float64)
    wrapped = (diff + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(wrapped)


def _matrix4_from_yaml_value(raw: Any, name: str) -> np.ndarray:
    """Parse a 4x4 from YAML (nested 4x4 list or flat 16 floats)."""
    if raw is None:
        raise ValueError(f"{name} is required (4x4 list)")
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 16:
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4 or 16 floats, got shape {arr.shape}")
    return arr


def _pose7_to_matrix4(pose7: Sequence[float]) -> np.ndarray:
    """Flexiv pose [x,y,z,qw,qx,qy,qz] to homogeneous 4x4."""
    from utils.flexiv_utils import flexiv_pose_wxyz_to_matrix

    flat = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if int(flat.size) != 7:
        raise ValueError(f"Expected 7 pose values, got {int(flat.size)}")
    return np.asarray(flexiv_pose_wxyz_to_matrix(flat.tolist()), dtype=np.float64)


def _average_transform_matrices(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average SE(3) transforms by translation mean and scipy rotation mean."""
    from scipy.spatial.transform import Rotation as R

    if len(transforms) == 0:
        raise ValueError("At least one transform is required")
    matrices = [np.asarray(transform, dtype=np.float64) for transform in transforms]
    for idx, matrix in enumerate(matrices):
        if matrix.shape != (4, 4):
            raise ValueError(f"Transform {idx} must be 4x4, got {matrix.shape}")

    avg_transform = np.eye(4, dtype=np.float64)
    avg_transform[:3, 3] = np.mean([matrix[:3, 3] for matrix in matrices], axis=0)
    avg_transform[:3, :3] = R.from_matrix([matrix[:3, :3] for matrix in matrices]).mean().as_matrix()
    return avg_transform


def _calibration_statistics(
    transforms: Sequence[np.ndarray],
    final_transform: np.ndarray,
) -> Dict[str, Any]:
    """Summarize calibration sample residuals against the final averaged transform."""
    from scipy.spatial.transform import Rotation as R

    matrices = [np.asarray(transform, dtype=np.float64) for transform in transforms]
    final = np.asarray(final_transform, dtype=np.float64)
    final_translation = final[:3, 3]
    final_rotation = R.from_matrix(final[:3, :3])

    translation_errors_mm: List[float] = []
    rotation_errors_deg: List[float] = []
    per_sample: List[Dict[str, Any]] = []
    for idx, matrix in enumerate(matrices):
        translation_error_mm = float(1000.0 * np.linalg.norm(matrix[:3, 3] - final_translation))
        rotation_error_deg = float(
            np.rad2deg((final_rotation.inv() * R.from_matrix(matrix[:3, :3])).magnitude())
        )
        translation_errors_mm.append(translation_error_mm)
        rotation_errors_deg.append(rotation_error_deg)
        per_sample.append(
            {
                "idx": idx,
                "translation_error_mm": translation_error_mm,
                "rotation_error_deg": rotation_error_deg,
            }
        )

    translation_errors = np.asarray(translation_errors_mm, dtype=np.float64)
    rotation_errors = np.asarray(rotation_errors_deg, dtype=np.float64)
    return {
        "sample_count": len(matrices),
        "translation_error_mm": {
            "mean": float(np.mean(translation_errors)),
            "std": float(np.std(translation_errors)),
            "max": float(np.max(translation_errors)),
        },
        "rotation_error_deg": {
            "mean": float(np.mean(rotation_errors)),
            "std": float(np.std(rotation_errors)),
            "max": float(np.max(rotation_errors)),
        },
        "per_sample": per_sample,
    }


def _joint_rows_from_yaml(raw: Any, nq: int) -> List[np.ndarray]:
    """Build joint waypoint rows from YAML: one row [nq] or multiple [[nq], ...]."""
    if raw is None:
        raise ValueError("source_joints must be set in configs/Pinocchio_KINE.YAML")
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("source_joints must be a non-empty list")
    if not isinstance(raw[0], (list, tuple)):
        matrices = [np.asarray(raw, dtype=np.float64).reshape(-1)]
    else:
        matrices = [np.asarray(row, dtype=np.float64).reshape(-1) for row in raw]
    for idx, q in enumerate(matrices):
        if int(q.size) != nq:
            raise ValueError(f"source_joints row {idx} length {int(q.size)} != model nq {nq}")
    return matrices


def _ensure_robot_ready(robot: Any) -> None:
    """Clear faults if needed, enable, and wait until operational."""
    if robot.fault():
        logger.warning("Robot reports fault; attempting clear_fault()")
        robot.clear_fault()
    robot.enable()
    while not robot.is_operational():
        time.sleep(0.5)


def _move_robot_to_joint(
    robot: Any,
    adapter: Any,
    joint: np.ndarray,
    *,
    settle_s: float,
    max_vel: Sequence[float],
    max_acc: Sequence[float],
) -> None:
    """Switch to joint position mode, command joints, then wait."""
    j = joint.reshape(-1).astype(float).tolist()
    robot.send_joint_position(
        positions=j,
        velocities=[0.0] * len(j),
        max_vel=list(max_vel),
        max_acc=list(max_acc),
    )
    time.sleep(float(settle_s))


def _resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path against ``PROJECT_ROOT`` when relative."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_pinocchio_kinematics_main_yaml(config_path: Path) -> Dict[str, Any]:
    """Load the Pinocchio kinematics hardware-test YAML."""
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return raw


def _load_flexiv_adapter_class() -> Any:
    """Load ``FlexivAdapter`` without importing the full ``devices`` package (avoids optional deps)."""
    module_path = PROJECT_ROOT / "devices" / "flexiv_adapter.py"
    spec = importlib.util.spec_from_file_location("flexiv_adapter_pinocchio_main", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "FlexivAdapter")


if __name__ == "__main__":
    # Edit configs/Pinocchio_KINE.YAML then run: python utils/pinocchio_kinematics_solver.py
    main_config_path = PROJECT_ROOT / "configs" / "pinocchio_kinematics.yaml"
    cfg = _load_pinocchio_kinematics_main_yaml(main_config_path)

    task = str(cfg.get("task", "calibrate_world_frame_transform")).strip().lower()
    urdf_path = _resolve_project_path(str(cfg["urdf_path"]))
    dt = float(cfg["dt"])
    max_iters = int(cfg["max_iters"])
    eps = float(cfg["eps"])
    ee_frame = str(cfg["ee_frame"])
    damping = float(cfg["damping"])
    robot_sn = str(cfg["robot_sn"])
    backend = str(cfg.get("backend", "pybind"))
    settle_seconds = float(cfg.get("settle_seconds", 4.0))
    max_joint_vel = float(cfg.get("max_joint_vel", 1.0))
    max_joint_acc = float(cfg.get("max_joint_acc", 2.0))

    world_frame_yaml = cfg.get("world_frame_transform")

    from utils.flexiv_utils import flexiv_pose_wxyz_to_matrix

    FlexivAdapter = _load_flexiv_adapter_class()
    adapter = FlexivAdapter(backend=str(backend))
    robot = adapter.create_robot(str(robot_sn))
    tool = adapter.create_tool(robot)
    raw_tool = tool.raw if hasattr(tool, "raw") else tool
    tcp_frame_transform = np.asarray(
        flexiv_pose_wxyz_to_matrix(list(raw_tool.params().tcp_location)),
        dtype=np.float64,
    )

    if task == "test_ik":
        world_frame_transform = _matrix4_from_yaml_value(world_frame_yaml, "world_frame_transform")
    elif task == "calibrate_world_frame_transform":
        world_frame_transform = np.eye(4, dtype=np.float64)
    else:
        raise ValueError(f"Unknown task: {task!r}")

    solver = PinocchioKinematicsSolver(
        urdf_path=urdf_path,
        dt=dt,
        max_iters=max_iters,
        eps=eps,
        ee_frame=ee_frame,
        world_frame_transform=world_frame_transform,
        tcp_frame_transform=tcp_frame_transform,
        damping=damping,
    )
    nq = solver.nq
    joint_rows = _joint_rows_from_yaml(cfg.get("source_joints"), nq)

    max_vel = [float(max_joint_vel)] * nq
    max_acc = [float(max_joint_acc)] * nq
    calibration_estimates: List[np.ndarray] = []

    if robot.fault():
        robot.clear_fault()
    _ensure_robot_ready(robot)
    robot.switch_mode(adapter.mode_joint_position)

    for idx, q_tgt in enumerate(joint_rows):
        logger.info("Waypoint {} / {}: moving to joint target", idx + 1, len(joint_rows))
        _move_robot_to_joint(
            robot,
            adapter,
            q_tgt,
            settle_s=float(settle_seconds),
            max_vel=max_vel,
            max_acc=max_acc,
        )
        states = robot.states()
        tcp_pose_meas = np.asarray(states.tcp_pose, dtype=np.float64).reshape(-1)
        if int(tcp_pose_meas.size) != 7:
            raise RuntimeError(f"Expected tcp_pose length 7, got {int(tcp_pose_meas.size)}")

        if task == "test_ik":
            t_solve = time.perf_counter()
            q_sol = solver.solve_ik(q0=q_tgt, tcp_pose=tcp_pose_meas)
            solve_ms = (time.perf_counter() - t_solve) * 1000.0
            pose_err = solver.compute_fk_pose_error(q=q_sol, tcp_pose=tcp_pose_meas)
            q_err = _angular_error_0_to_pi(q_ref=q_tgt, q_est=q_sol)
            q_err_max_deg = float(np.rad2deg(np.max(q_err)))
            logger.info(
                "test_ik idx={} ok={} iters={} solve_ms={:.3f} pose_err={:.6e} q_err_max_deg={:.6f}",
                idx,
                solver.last_success,
                solver.last_iterations,
                solve_ms,
                pose_err,
                q_err_max_deg,
            )
        else:
            fk_pose7 = solver.compute_fk(q_tgt)
            t_fk = _pose7_to_matrix4(fk_pose7)
            t_tcp = _pose7_to_matrix4(tcp_pose_meas)
            world_est = t_fk @ np.linalg.inv(t_tcp)
            calibration_estimates.append(world_est)
            logger.info("calibrate idx={} collected one world-frame estimate", idx)
            print(
                "urdfworld_T_robotworld:")
            print(
                f"{np.array2string(world_est, precision=8, suppress_small=True)}"
            )

    if task == "calibrate_world_frame_transform":
        final_world_transform = _average_transform_matrices(calibration_estimates)
        calibration_stats = _calibration_statistics(calibration_estimates, final_world_transform)
        logger.info(
            "final calibration finished from {} samples",
            len(calibration_estimates),
        )

        print("\n" + "=" * 60)
        print("Final urdfworld_T_robotworld (4x4 matrix):")
        print("=" * 60)

        matrix = final_world_transform
        print("np.array([")
        for i in range(4):
            row_str = "    [" + ", ".join(f"{matrix[i, j]:.8f}" for j in range(4)) + "]"
            if i < 3:
                print(row_str + ",")
            else:
                print(row_str)
        print("])")
        print("=" * 60)

    print("\n# Full calibration statistics (JSON):")
    json.dump(calibration_stats, sys.stdout, indent=2)
    print()
