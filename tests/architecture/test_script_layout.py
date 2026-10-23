"""Path contracts for categorized project-owned scripts.

Author: SuperInference Contributors
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


EXPECTED_SCRIPTS: dict[str, tuple[str, ...]] = {
    "analysis": (
        "async_episode_visualizer.py",
        "data_collection_gui.py",
        "dump_camera_shm.py",
        "evaluation_camera/camera_recorder.py",
        "hdf5_anomaly_detector.py",
        "hdf5_to_zarr_data_converter.py",
        "inspect_episode_data.py",
        "merge_zarr_datasets.py",
        "pinocchio_kinematics_solver.py",
        "transform_visualizer_gui.py",
        "visualize_serial_rotary_encoder.py",
        "visualize_zarr_cameras.py",
    ),
    "benchmark": (
        "test_hik_camera_latency_async.py",
        "test_iphone_camera_latency_async.py",
        "test_trajectory_planner.py",
        "test_usb_camera_latency_async.py",
        "vive_tracker_latency.py",
        "waypoint_jerk_analyzer.py",
        "waypoint_trajectory_analyzer.py",
        "waypoint_velocity_analyzer.py",
    ),
    "calibration": (
        "auto_calibrate_grippers.py",
        "calib_cam_deviation.py",
        "calibrate_lk_gripper.py",
        "calibrate_safety_bounds.py",
        "mask_mapper.py",
        "mask_offset_analyzer.py",
        "mask_offset_visualizer.py",
        "plot_mask.py",
        "zero_ft_sensor.py",
    ),
    "diagnostics": (
        "rollout_device_preflight.py",
        "runtime_profiles.py",
    ),
    "legacy": (
        "check_robot_arm.py",
        "move_grippers_check.py",
        "move_lk_gripper.py",
        "preview_video_devices.py",
        "scan_hik_cameras.py",
        "scan_opencv_cameras.py",
        "test_camera_ids.py",
        "test_max_contact_wrench1.py",
        "test_policy_gripper_control.py",
    ),
    "maintenance": ("update_submodules.py",),
    "replay": (
        "replay_left_gripper_parquet.py",
        "replay_right_gripper_parquet.py",
        "robot_record_replay.py",
        "robot_record_replay_absolute.py",
    ),
    "services": (),
}


def _expected_paths(repo_root: Path) -> Iterable[Path]:
    for category, names in EXPECTED_SCRIPTS.items():
        for name in names:
            yield repo_root / "scripts" / category / name


def test_project_owned_root_scripts_are_categorized(repo_root: Path) -> None:
    root_scripts = sorted(path.name for path in (repo_root / "scripts").glob("*.py"))

    assert root_scripts == []
    assert all(path.is_file() for path in _expected_paths(repo_root))


def test_active_domain_script_directories_remain_explicit(repo_root: Path) -> None:
    scripts_root = repo_root / "scripts"

    assert (scripts_root / "openpi" / "dump_model_input.py").is_file()
    assert (scripts_root / "lingbotva" / "run_infer_split.sh").is_file()
    assert (
        scripts_root / "yam_head_hacking" / "compute_yamhead_focus_trajectory.py"
    ).is_file()


def test_human_script_docs_are_centralized(repo_root: Path) -> None:
    scripts_root = repo_root / "scripts"

    assert (repo_root / "useful_tools_commands.md").is_file()
    for name in (
        "openpi_inference_commands.md",
        "fastwam_inference_commands.md",
        "fastwam_memory_inference_commands.md",
        "fasterwam_inference_commands.md",
        "robotwin_inference_commands.md",
    ):
        assert (repo_root / name).is_file()
    assert not (scripts_root / "README.md").exists()
    assert not any((scripts_root / category / "README.md").exists() for category in EXPECTED_SCRIPTS)


def test_gripper_service_process_is_removed(repo_root: Path) -> None:
    assert not (repo_root / "scripts" / "services" / "lktech_gripper_server.py").exists()
    assert not (
        repo_root
        / "incubator"
        / "control_panel"
        / "services"
        / "lktech_gripper_server.py"
    ).exists()
    direct = (
        repo_root / "incubator" / "control_panel" / "gripper_control.py"
    ).read_text(encoding="utf-8")
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in direct
    assert "direct_serial" in direct
