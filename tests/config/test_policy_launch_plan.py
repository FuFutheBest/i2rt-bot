from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.config import policy_launch_plan as launch_plan
from tests.config.policy_launch_plan import (
    LaunchPlanError,
    build_launch_plan,
    build_supported_launch_contracts,
    compose_launch_config,
    discover_supported_config_names,
    render_text,
)


def _write_config(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    root = tmp_path / "configs"
    _write_config(
        root / "policy_infer" / "openpi" / "demo.yaml",
        """
# @package _global_
defaults:
  - /devices/infer@devices: demo
  - /device_manager: demo
  - /action_executor: demo
  - _self_

camera:
  left_capture_id: 3
policy:
  class: OpenPiDemoPolicy
  params:
    pi_config_name: pi0_demo
    ckpt_dir: /models/demo
    obs_horizon: 2
    num_robots: 2
  fps: 4.0
  control_fps: 30.0
  chunk_length: 16
  obs:
    devices: ["WebCameraOpenPi_${camera.left_capture_id}", "RizonRobot_2"]
  controls:
    - endpoint:
        device: RizonRobot
        device_class_id: 2
        port: command
      shape: [16]
      dtype: float64
summary_shm: fixture_summary
hydra:
  run:
    dir: .
  output_subdir: null
""",
    )
    _write_config(
        root / "devices" / "infer" / "demo.yaml",
        """
devices:
  - class: WebCameraOpenPi
    device_class_id: ${camera.left_capture_id}
    fps: 30.0
    data_shape: [360, 640, 3]
    data_dtype: uint8
  - class: RizonRobot
    device_class_id: 2
    robot_sn: [left, right]
    state_feedback:
      fps: 120.0
      format:
        shape: [44]
        dtype: float64
        type: [j1-7, dq1-7, xyz, qwqxqyqz, gripper, j1-7, dq1-7, xyz, qwqxqyqz, gripper]
    control:
      type: eef_abs
      format:
        shape: [16]
        dtype: float64
        type: [xyz, qxqyqzqw, gripper, xyz, qxqyqzqw, gripper]
    gripper:
      backend: flexiv
      name: fixture_gripper
  - class: DisabledCamera
    device_class_id: 9
    enabled: false
""",
    )
    _write_config(
        root / "device_manager" / "demo.yaml",
        """
device_manager:
  type: synchronized
  master_device_class_id: 2
  target_fps: 30.0
""",
    )
    _write_config(
        root / "action_executor" / "demo.yaml",
        """
action_executor:
  type: base_policy_runner
  robot_control_shms: ${policy.controls}
  execution_fps: ${policy.control_fps}
  chunk_length: ${policy.chunk_length}
  chunk_manager:
    type: base_manager
""",
    )
    return root


def test_native_openpi_name_composes_and_generates_exact_commands(
    config_dir: Path,
) -> None:
    plan = build_launch_plan(
        "policy_infer/openpi/demo",
        config_dir=config_dir,
        device_env="device-env",
        policy_env="policy-env",
    )

    assert plan["mode"] == "read-only-launch-plan"
    assert plan["started_processes"] == 0
    assert plan["config"] == {
        "requested_name": "policy_infer/openpi/demo",
        "hydra_config_name": "policy_infer/openpi/demo",
        "source_path": str(config_dir / "policy_infer" / "openpi" / "demo.yaml"),
    }
    assert plan["policy"]["runner_module"] == "consumer.inference.openpi_policy_infer"
    assert plan["policy"]["observation_devices"] == [
        "WebCameraOpenPi_3",
        "RizonRobot_2",
    ]
    assert plan["devices"]["enabled"][1]["gripper_configured"] is True
    assert plan["devices"]["disabled_count"] == 1
    assert plan["action_executor"]["execution_fps"] == 30.0
    assert plan["action_executor"]["control_channels"] == plan["policy"]["controls"]
    assert plan["commands"]["device_starter"]["argv"] == [
        "conda",
        "run",
        "-n",
        "device-env",
        "python",
        "device_starter.py",
        "--config-name",
        "policy_infer/openpi/demo",
    ]
    assert plan["commands"]["consumer"]["argv"][-3:] == [
        "consumer.inference.openpi_policy_infer",
        "--config-name",
        "policy_infer/openpi/demo",
    ]


def test_bare_openpi_shorthand_resolves_to_native_hydra_name(config_dir: Path) -> None:
    native_name, source, config = compose_launch_config(
        "demo.yaml", config_dir=config_dir
    )

    assert native_name == "policy_infer/openpi/demo"
    assert source == (config_dir / "policy_infer" / "openpi" / "demo.yaml")
    assert config["devices"][0]["device_class_id"] == 3


def test_rejects_traversal_and_text_explicitly_states_no_hardware(
    config_dir: Path,
) -> None:
    with pytest.raises(LaunchPlanError, match="relative Hydra name"):
        build_launch_plan("../policy_infer/openpi/demo", config_dir=config_dir)

    text = render_text(build_launch_plan("policy_infer/openpi/demo", config_dir=config_dir))
    assert "No process was started and no hardware was opened." in text
    assert "policy_infer/openpi/demo" in text


def test_cli_json_output_is_a_read_only_plan(
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(launch_plan, "DEFAULT_CONFIG_DIR", config_dir)

    assert launch_plan.main(["--config-name", "policy_infer/openpi/demo", "--json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["started_processes"] == 0
    assert result["config"]["hydra_config_name"] == "policy_infer/openpi/demo"
    assert result["environment_resolution_required"] is True
    assert (
        result["commands"]["consumer"]["argv"][2]
        == "consumer.inference.openpi_policy_infer"
    )


def test_repository_control_panel_presets_are_valid() -> None:
    report = build_supported_launch_contracts()
    assert report["scope"] == "configs/policy_infer/openpi + configs/policy_infer/fastwam (native launch configs)"
    assert report["started_processes"] == 0
    assert report["summary"]["total"] >= 1
    assert report["summary"]["failed"] == 0


def test_fastwam_plan_declares_independent_server_and_client() -> None:
    plan = build_launch_plan(
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
        device_env="device-env",
        policy_env="fastwam-env",
    )

    assert plan["policy"]["server_required"] is True
    assert plan["policy"]["runner_module"] == "consumer.inference.fastwam_policy_infer"
    assert plan["commands"]["server"]["argv"] == [
        "conda",
        "run",
        "-n",
        "fastwam-env",
        "python",
        "consumer/fastwam_server.py",
        "--config-name",
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
    ]
    assert plan["commands"]["consumer"]["argv"][-3:] == [
        "consumer.inference.fastwam_policy_infer",
        "--config-name",
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
    ]


def test_batch_supported_contracts_check_only_native_openpi_presets(
    config_dir: Path,
) -> None:
    _write_config(
        config_dir / "policy_infer" / "openpi" / "generated_should_skip.yaml",
        "this_is_not_a_launch_preset: true" + chr(10),
    )

    assert discover_supported_config_names(config_dir) == ["policy_infer/openpi/demo"]
    report = build_supported_launch_contracts(config_dir=config_dir)

    assert report["started_processes"] == 0
    assert report["summary"] == {"total": 1, "passed": 1, "failed": 0}
    assert report["configs"][0]["status"] == "passed"
    assert report["configs"][0]["plan"]["contract"]["valid"] is True


def test_batch_contract_command_returns_nonzero_with_machine_readable_failures(
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_config(
        config_dir / "policy_infer" / "openpi" / "broken.yaml",
        "# @package _global_" + chr(10) + "policy: {}" + chr(10),
    )
    monkeypatch.setattr(launch_plan, "DEFAULT_CONFIG_DIR", config_dir)

    assert launch_plan.main(["--all-supported", "--json"]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["summary"] == {"total": 2, "passed": 1, "failed": 1}
    failed = next(
        item
        for item in report["configs"]
        if item["config_name"] == "policy_infer/openpi/broken"
    )
    assert failed["status"] == "failed"
    assert failed["plan"]["contract"]["valid"] is False
    assert failed["errors"]


def test_compose_launch_config_delegates_to_shared_normalizer(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    original = launch_plan.flatten_hydra_config

    def record_normalization(config):
        calls.append(config)
        return original(config)

    monkeypatch.setattr(launch_plan, "flatten_hydra_config", record_normalization)

    native_name, _, config = compose_launch_config("policy_infer/openpi/demo", config_dir=config_dir)

    assert native_name == "policy_infer/openpi/demo"
    assert config["camera"]["left_capture_id"] == 3
    assert len(calls) == 1



def test_direct_cli_entrypoint_can_import_repository_modules(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(launch_plan.REPO_ROOT / "tests" / "config" / "policy_launch_plan.py"),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--config-name" in result.stdout



def test_default_plan_keeps_device_and_policy_interpreters_unresolved(
    config_dir: Path,
) -> None:
    plan = build_launch_plan("policy_infer/openpi/demo", config_dir=config_dir)

    assert plan["environment_resolution_required"] is True
    assert plan["commands"]["device_starter"] == {
        "argv": ["python", "device_starter.py", "--config-name", "policy_infer/openpi/demo"],
        "shell": "python device_starter.py --config-name policy_infer/openpi/demo",
        "execution_prefix_resolution_required": True,
    }
    assert plan["commands"]["consumer"] == {
        "argv": [
            "python",
            "-m",
            "consumer.inference.openpi_policy_infer",
            "--config-name",
            "policy_infer/openpi/demo",
        ],
        "shell": "python -m consumer.inference.openpi_policy_infer --config-name policy_infer/openpi/demo",
        "execution_prefix_resolution_required": True,
    }

    text = render_text(plan)
    assert "<device-prefix> device_starter.py --config-name policy_infer/openpi/demo" in text
    assert (
        "<policy-prefix> -m consumer.inference.openpi_policy_infer --config-name policy_infer/openpi/demo"
        in text
    )
    assert "conda run" not in text


@pytest.mark.parametrize("field", ["device_env", "policy_env"])
def test_rejects_empty_explicit_conda_environment(
    config_dir: Path, field: str
) -> None:
    kwargs = {field: " "}
    with pytest.raises(LaunchPlanError, match=field):
        build_launch_plan("policy_infer/openpi/demo", config_dir=config_dir, **kwargs)
