"""Architecture contracts for the optional control-panel application."""

from pathlib import Path

from incubator.control_panel.runtime_config import (
    DEFAULT_CONFIG_DIR,
    compose_runtime_config,
    list_runtime_configs,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PANEL_ROOT = REPO_ROOT / "incubator" / "control_panel"


def test_control_panel_keeps_application_specific_assets() -> None:
    expected = (
        "incubator/control_panel/release/build_release.py",
        "incubator/control_panel/release/doctor.py",
        "incubator/control_panel/release/runtime_bootstrap.py",
        "incubator/control_panel/release/requirements/build_constraints.txt",
        "incubator/control_panel/recording_control.py",
        "incubator/control_panel/static/assets/tools/umi3_camera/soft/umi3_gripper.urdf",
        "incubator/control_panel/static/assets/tools/umi3_camera/hard/umi3_gripper.urdf",
        "incubator/control_panel/static/assets/yam/yam_no_gripper.urdf",
    )

    missing = [path for path in expected if not (REPO_ROOT / path).is_file()]

    assert missing == []


def test_shared_modules_have_one_canonical_implementation() -> None:
    expected = (
        "utils/yam_disconnect_safety.py",
        "utils/yam_head_tracking.py",
        "utils/episode_anomaly_algorithms.py",
        "utils/home_transition.py",
        "utils/inference_control.py",
        "utils/recording_control.py",
        "incubator/control_panel/gripper_control.py",
    )
    obsolete = (
        "incubator/control_panel/yam_disconnect_safety.py",
    )

    missing = [path for path in expected if not (REPO_ROOT / path).is_file()]
    present = [path for path in obsolete if (REPO_ROOT / path).exists()]

    assert missing == []
    assert present == []

    direct_source = (REPO_ROOT / "incubator/control_panel/gripper_control.py").read_text(
        encoding="utf-8"
    )
    assert "direct_serial" in direct_source
    assert "LOCK_EX | fcntl.LOCK_NB" in direct_source


def test_root_configs_are_the_only_runtime_config_source() -> None:
    assert DEFAULT_CONFIG_DIR == REPO_ROOT / "configs"
    assert not (PANEL_ROOT / "configs").exists()
    assert not (REPO_ROOT / "configs" / "Untitled").exists()


def test_control_panel_uses_shared_device_starter_config_source() -> None:
    names = list_runtime_configs()

    native_name = "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
    rtc_name = "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc"
    assert native_name in names
    assert rtc_name in names
    assert "control_panel/openpi" not in names
    assert compose_runtime_config(native_name)["policy"]["params"]["enable_rtc"] is False
    assert compose_runtime_config(rtc_name)["policy"]["params"]["enable_rtc"] is True


def test_fastwam_control_panel_has_only_the_lifecycle_adapter() -> None:
    offending = [
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "incubator").rglob("*")
        if path.is_file()
        and "fastwam" in path.name.casefold()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and "tests" not in path.parts
        and path.relative_to(REPO_ROOT).as_posix()
        != "incubator/control_panel/fastwam_workflow.py"
    ]

    assert offending == []
    assert (PANEL_ROOT / "fastwam_workflow.py").is_file()
    assert any(path.is_file() for path in (REPO_ROOT / "configs").rglob("*fastwam*"))
