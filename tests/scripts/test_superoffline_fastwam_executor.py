"""Read-only execution contracts for the built-in FastWAM comparison runner."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.analysis.superoffline_fastwam_executor import (
    FastWAMOfflineComparisonExecutor,
    RealRobotOfflineComparisonExecutor,
)
from utils.observation_recorder import TimestampedObservationRecorder
from utils.superoffline_observation import (
    RECORDER_OBSERVATION_CONTRACT,
    device_roles_from_arrays,
    get_source_profile_contract,
    reviewed_calibration_files,
    reviewed_capture_device_descriptors,
    source_profile_provenance_from_files,
)
from utils.superoffline_registry import (
    DUAL_ARM_TCP_ACTION_CONTRACT,
    DUAL_ARM_TCP_ACTION_FRAME,
    DUAL_ARM_TCP_ACTION_LAYOUT,
    DUAL_ARM_TCP_ACTION_UNITS,
    SUPER_OFFLINE_ADAPTERS,
    RuntimeProfileComparisonRunner,
    SuperOfflineAdapterError,
    SuperOfflineRuntimeContext,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_reviewed_calibration_bundle(repo_root: Path) -> dict[str, Path]:
    source_files = reviewed_calibration_files(_REPO_ROOT)
    for source in source_files.values():
        destination = repo_root / source.relative_to(_REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return reviewed_calibration_files(repo_root)


def _request(observation_path: Path):
    request = SUPER_OFFLINE_ADAPTERS.parse_request(
        {
            "observation_path": str(observation_path),
            "models": [
                {
                    "adapter_id": "fastwam",
                    "variant_id": "fastwam",
                    "checkpoint": "checkpoints/fastwam",
                },
                {
                    "adapter_id": "fastwam",
                    "variant_id": "fasterwam",
                    "checkpoint": "checkpoints/fasterwam",
                },
            ],
        }
    )
    return replace(request, job_id="a" * 32)


def _contexts(request: Any) -> tuple[SuperOfflineRuntimeContext, ...]:
    return tuple(
        SuperOfflineRuntimeContext(
            model=model,
            python=Path(f"/runtime/{model.adapter.id}/bin/python"),
            profile_name=model.adapter.id,
            environment={"PATH": f"/runtime/{model.adapter.id}/bin"},
        )
        for model in request.models
    )


def _openpi_checkpoint(repo_root: Path) -> Path:
    checkpoint = repo_root / "checkpoints" / "openpi"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params" / "manifest").write_text("fixture", encoding="utf-8")
    asset = checkpoint / "assets" / "gift"
    asset.mkdir(parents=True)
    (asset / "norm_stats.json").write_text("{}", encoding="utf-8")
    return checkpoint


def _recorder_run(
    tmp_path: Path,
    *,
    simulation: bool = False,
    source_profile: str = "rizon-umi3-fisheye-openpi-rgb.v1",
) -> Path:
    root = tmp_path / "recording"
    root.mkdir(parents=True)
    frames_dir = root / "frames"
    inference_dir = root / "inference"
    frames_dir.mkdir()
    inference_dir.mkdir()
    source_contract = get_source_profile_contract(source_profile)
    calibration_files = _install_reviewed_calibration_bundle(tmp_path)
    left = source_contract.capture_device_keys["left_wrist_rgb"]
    right = source_contract.capture_device_keys["right_wrist_rgb"]
    robot = source_contract.capture_device_keys["rizon_dual_arm_state"]
    arrays = {
        left: np.zeros(
            source_contract.camera_shapes["left_wrist_rgb"], dtype=np.uint8
        ),
        right: np.ones(
            source_contract.camera_shapes["right_wrist_rgb"], dtype=np.uint8
        ),
        robot: np.zeros(44, dtype=np.float64),
    }
    np.savez_compressed(frames_dir / "frame_000000.npz", **arrays)
    np.savez_compressed(inference_dir / "inference_000000.npz", **arrays)
    (root / "frames.jsonl").write_text(
        json.dumps(
            {
                "kind": "frame",
                "index": 0,
                "timestamp_ns": 10,
                "file": "frames/frame_000000.npz",
                "device_array_keys": {left: left, right: right, robot: robot},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "inference.jsonl").write_text(
        json.dumps(
            {
                "kind": "inference",
                "index": 0,
                "timestamp_ns": 11,
                "file": "inference/inference_000000.npz",
                "keys": [
                    {"name": left, "array_key": left},
                    {"name": right, "array_key": right},
                    {"name": robot, "array_key": robot},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "superinference.timestamped_observation",
        "stopped_at_ns": 1,
        "frame_count": 1,
        "inference_count": 1,
        "camera_devices": [left, right],
        "robot_device": robot,
        "observation_contract": RECORDER_OBSERVATION_CONTRACT,
        "source_profile": source_profile,
        "source_provenance": source_profile_provenance_from_files(
            source_profile,
            calibration_files=calibration_files,
            capture_device_descriptors=reviewed_capture_device_descriptors(
                source_profile
            ),
        ),
        "device_roles": device_roles_from_arrays(
            camera_devices=[left, right],
            robot_device=robot,
            arrays=arrays,
        ),
    }
    if simulation:
        manifest["source_type"] = "simulation"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _write_worker_output(command: list[str]) -> None:
    if "--inference-indices" in command:
        root = Path(command[command.index("--output-dir") + 1])
        indices = command[command.index("--inference-indices") + 1].split(",")
        for raw_index in indices:
            child = list(command)
            flag_index = child.index("--inference-indices")
            del child[flag_index : flag_index + 2]
            child[child.index("--output-dir") + 1] = str(
                root / "observations" / f"{int(raw_index):06d}"
            )
            _write_worker_output(child)
        return
    output_dir = Path(command[command.index("--output-dir") + 1])
    output_dir.mkdir(parents=True)
    action = np.zeros((48, 16), dtype=np.float64)
    action[:, 3] = 1.0
    action[:, 11] = 1.0
    evaluator = command[command.index("--") + 1]
    if evaluator.endswith("evaluate_fastwam_reverse_offline.py"):
        action_key = "absolute_runtime"
    else:
        action_key = "action"
    action_step_interval_ns = (
        20_000_000 if evaluator.endswith("evaluate_diffusion_offline.py") else 33_333_333
    )
    binding_path = Path(command[command.index("--observation-binding") + 1])
    binding_index = int(command[command.index("--binding-index") + 1])
    binding = json.loads(binding_path.read_text(encoding="utf-8"))["bindings"][
        binding_index
    ]
    np.savez_compressed(output_dir / "offline_result.npz", **{action_key: action})
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "action_shape": [48, 16],
                "action_contract": DUAL_ARM_TCP_ACTION_CONTRACT,
                "action_layout": DUAL_ARM_TCP_ACTION_LAYOUT,
                "action_frame_id": DUAL_ARM_TCP_ACTION_FRAME,
                "action_units": DUAL_ARM_TCP_ACTION_UNITS,
                "action_calibration_id": binding["action_calibration_id"],
                "action_calibration_digest": binding["action_calibration_digest"],
                "observation_binding_digest": binding["digest"],
                "source_profile": binding["source_profile"],
                "source_provenance_digest": binding["source_provenance_digest"],
                "recording_artifact_digest": binding["recording_artifact_digest"],
                "action_step_interval_ns": action_step_interval_ns,
                "finite": True,
                "observation_timestamp_ns": 123,
                "artifacts": {
                    "npz": "offline_result.npz",
                    "canonical_action_key": action_key,
                    "action_contract": DUAL_ARM_TCP_ACTION_CONTRACT,
                    "action_layout": DUAL_ARM_TCP_ACTION_LAYOUT,
                    "action_frame_id": DUAL_ARM_TCP_ACTION_FRAME,
                    "action_units": DUAL_ARM_TCP_ACTION_UNITS,
                    "action_calibration_id": binding["action_calibration_id"],
                    "action_calibration_digest": binding[
                        "action_calibration_digest"
                    ],
                    "observation_binding_digest": binding["digest"],
                    "source_provenance_digest": binding["source_provenance_digest"],
                    "recording_artifact_digest": binding["recording_artifact_digest"],
                    "action_step_interval_ns": action_step_interval_ns,
                },
            }
        ),
        encoding="utf-8",
    )


def test_builtin_executor_runs_two_fastwam_models_through_profile_tool(
    tmp_path: Path,
) -> None:
    observation = _recorder_run(tmp_path)
    request = _request(observation)
    calls: list[list[str]] = []

    def command_runner(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if "run" in command:
            _write_worker_output(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    executor = FastWAMOfflineComparisonExecutor(
        repo_root=tmp_path,
        command_runner=command_runner,
    )
    result = executor.run(request, _contexts(request))

    assert result["executor"] == "builtin-real-robot-offline"
    assert len(result["models"]) == 2
    assert [command[2] for command in calls] == ["doctor", "run", "doctor", "run"]
    run_commands = [command for command in calls if command[2] == "run"]
    assert all("--role" in command for command in run_commands)
    assert all(command[command.index("--role") + 1] == "fastwam-server" for command in run_commands)
    assert [
        command[command.index("--config-name") + 1] for command in run_commands
    ] == [
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
    ]
    assert [command[command.index("--checkpoint") + 1] for command in run_commands] == [
        str((tmp_path / "checkpoints" / "fastwam").resolve()),
        str((tmp_path / "checkpoints" / "fasterwam").resolve()),
    ]
    assert (tmp_path / "logs" / "superoffline" / ("a" * 32) / "comparison.json").is_file()
    assert result["comparison"]["steps_compared"] == 48
    assert result["comparison"]["arms"]["left"]["position_error_m"]["max"] == 0.0


def test_builtin_executor_runs_mixed_fastwam_openpi_pair_through_each_profile(
    tmp_path: Path,
) -> None:
    observation = _recorder_run(tmp_path)
    _openpi_checkpoint(tmp_path)
    request = SUPER_OFFLINE_ADAPTERS.parse_request(
        {
            "observation_path": str(observation),
            "metadata": {"prompt": "place the object safely"},
            "models": [
                {
                    "adapter_id": "fastwam",
                    "variant_id": "fastwam-memory",
                    "checkpoint": "checkpoints/fastwam-memory",
                },
                {
                    "adapter_id": "openpi",
                    "checkpoint": "checkpoints/openpi",
                },
            ],
        }
    )
    request = replace(request, job_id="b" * 32)
    calls: list[list[str]] = []

    def command_runner(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if "run" in command:
            _write_worker_output(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    executor = RealRobotOfflineComparisonExecutor(
        repo_root=tmp_path,
        command_runner=command_runner,
    )
    result = executor.run(request, _contexts(request))

    assert result["executor"] == "builtin-real-robot-offline"
    assert executor.public_metadata() == {
        "mode": "builtin-real-robot-offline",
        "requires_same_adapter": False,
        "supported_adapter_ids": ["fastwam", "lingbotva", "openpi"],
    }
    run_commands = [command for command in calls if command[2] == "run"]
    assert [command[command.index("--role") + 1] for command in run_commands] == [
        "fastwam-server",
        "policy",
    ]
    assert run_commands[0][run_commands[0].index("--") + 1].endswith(
        "evaluate_fastwam_reverse_offline.py"
    )
    assert run_commands[1][run_commands[1].index("--") + 1].endswith(
        "evaluate_openpi_offline.py"
    )
    assert "--num-inference-steps" in run_commands[0]
    assert "--skip-diagnostic-plots" in run_commands[0]
    assert "--num-inference-steps" not in run_commands[1]
    assert [command[command.index("--prompt") + 1] for command in run_commands] == [
        "place the object safely",
        "place the object safely",
    ]
    assert [command[command.index("--checkpoint") + 1] for command in run_commands] == [
        str((tmp_path / "checkpoints" / "fastwam-memory").resolve()),
        str((tmp_path / "checkpoints" / "openpi").resolve()),
    ]


def test_builtin_executor_rejects_openpi_checkpoint_without_norm_stats(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoints" / "openpi"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params" / "manifest").write_text("fixture", encoding="utf-8")
    request = SUPER_OFFLINE_ADAPTERS.parse_request(
        {
            "observation_path": str(tmp_path / "recording"),
            "models": [
                {
                    "adapter_id": "fastwam",
                    "variant_id": "fastwam-memory",
                    "checkpoint": "checkpoints/fastwam-memory",
                },
                {
                    "adapter_id": "openpi",
                    "checkpoint": "checkpoints/openpi",
                },
            ],
        }
    )
    executor = RealRobotOfflineComparisonExecutor(repo_root=tmp_path)

    with pytest.raises(
        SuperOfflineAdapterError,
        match=r"no readable assets/\*/norm_stats.json",
    ):
        executor.validate_request(request)


def test_fastwam_task_mismatch_is_a_non_blocking_warning(tmp_path: Path) -> None:
    for name, task in (
        ("fastwam", "Pick up the phone"),
        ("fasterwam", "Pack the gift"),
    ):
        checkpoint = tmp_path / "checkpoints" / name
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.yaml").write_text(
            "datasets:\n"
            "  vla_data:\n"
            f"    override_instruction: {task!r}\n"
            "framework:\n"
            "  fastwam:\n"
            "    task: ignored lower-priority task\n",
            encoding="utf-8",
        )
    request = replace(
        _request(tmp_path / "recording"),
        metadata={"prompt": "Pack the gift"},
    )
    executor = RealRobotOfflineComparisonExecutor(repo_root=tmp_path)

    warnings = executor.request_warnings(request)

    assert warnings == [
        {
            "code": "task_mismatch",
            "severity": "warning",
            "model_index": 0,
            "checkpoint_label": "FastWAM",
            "requested_task": "Pack the gift",
            "training_task": "Pick up the phone",
            "message": (
                "FastWAM was trained for 'Pick up the phone', but this run requests "
                "'Pack the gift'; inference will continue."
            ),
        }
    ]
    assert executor._task_validation(
        request.models[0], runtime_task="Pack the gift"
    )["status"] == "mismatch_warning"
    assert executor._task_validation(
        request.models[1], runtime_task="  pack   the GIFT "
    )["status"] == "match"


def test_streaming_invoke_forwards_output_and_honors_cancellation(
    tmp_path: Path,
) -> None:
    executor = RealRobotOfflineComparisonExecutor(repo_root=tmp_path)
    events: list[dict[str, Any]] = []
    cancel_event = threading.Event()
    timer = threading.Timer(0.2, cancel_event.set)
    timer.start()
    try:
        with pytest.raises(Exception, match="cancelled by operator"):
            executor._invoke(
                [
                    sys.executable,
                    "-c",
                    "import time; print('model loading', flush=True); time.sleep(10)",
                ],
                environment=os.environ,
                log_path=tmp_path / "worker.log",
                stage="test evaluator",
                event_callback=lambda event: events.append(dict(event)),
                cancel_event=cancel_event,
                progress=25,
            )
    finally:
        timer.cancel()

    assert any(event["message"] == "model loading" for event in events)
    assert any(event["stage"] == "load_checkpoint" for event in events)
    assert "model loading" in (tmp_path / "worker.log").read_text(encoding="utf-8")


def test_builtin_executor_dispatches_two_lingbotva_configs(
    tmp_path: Path,
) -> None:
    observation = _recorder_run(
        tmp_path,
        source_profile="rizon-umi3-fisheye-lingbotva-rgb.v1",
    )
    request = SUPER_OFFLINE_ADAPTERS.parse_request(
        {
            "observation_path": str(observation),
            "models": [
                {
                    "adapter_id": "lingbotva",
                    "checkpoint": "checkpoints/lingbotva-sync",
                    "config_name": "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva",
                },
                {
                    "adapter_id": "lingbotva",
                    "checkpoint": "checkpoints/lingbotva-async",
                    "config_name": "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva_async",
                },
            ],
        }
    )
    request = replace(request, job_id="c" * 32)
    calls: list[list[str]] = []

    def command_runner(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if "run" in command:
            _write_worker_output(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    executor = RealRobotOfflineComparisonExecutor(
        repo_root=tmp_path,
        command_runner=command_runner,
    )
    result = executor.run(request, _contexts(request))

    run_commands = [command for command in calls if command[2] == "run"]
    assert [command[command.index("--role") + 1] for command in run_commands] == [
        "policy-server",
        "policy-server",
    ]
    assert run_commands[0][run_commands[0].index("--") + 1].endswith(
        "evaluate_lingbotva_offline.py"
    )
    assert run_commands[1][run_commands[1].index("--") + 1].endswith(
        "evaluate_lingbotva_offline.py"
    )
    assert result["comparison"]["resampled"] is False
    assert result["comparison"]["comparison_step_interval_ns"] == 33_333_333


def test_builtin_executor_resamples_mismatched_action_timebases() -> None:
    first = np.zeros((4, 16), dtype=np.float64)
    second = np.zeros((3, 16), dtype=np.float64)
    first[:, 3] = 1.0
    first[:, 11] = 1.0
    second[:, 3] = 1.0
    second[:, 11] = 1.0
    first[:, 0] = [0.0, 0.02, 0.04, 0.06]
    second[:, 0] = [0.0, 0.03, 0.06]

    comparison = RealRobotOfflineComparisonExecutor._comparison_metrics(
        first,
        second,
        first_action_step_interval_ns=20_000_000,
        second_action_step_interval_ns=30_000_000,
    )

    assert comparison["resampled"] is True
    assert comparison["comparison_step_interval_ns"] == 30_000_000
    assert comparison["steps_compared"] == 3
    assert comparison["arms"]["left"]["position_error_m"]["max"] == 0.0


def test_builtin_executor_selects_all_current_or_strided_samples(
    tmp_path: Path,
) -> None:
    observation = tmp_path / "recording"
    observation.mkdir()
    (observation / "manifest.json").write_text(
        json.dumps({"inference_count": 7}),
        encoding="utf-8",
    )

    select = RealRobotOfflineComparisonExecutor._selected_inference_indices
    assert select(observation, {"run_scope": "all"}) == list(range(7))
    assert select(
        observation,
        {"run_scope": "sample", "inference_index": 4},
    ) == [4]
    assert select(
        observation,
        {"run_scope": "stride", "sample_stride": 3},
    ) == [0, 3, 6]


def test_builtin_executor_rejects_simulation_inputs(tmp_path: Path) -> None:
    observation = _recorder_run(tmp_path, simulation=True)
    executor = FastWAMOfflineComparisonExecutor(repo_root=tmp_path)

    with pytest.raises(SuperOfflineAdapterError, match="simulation observation"):
        executor.validate_request(_request(observation))


def test_runtime_runner_preflights_before_an_external_executor(tmp_path: Path) -> None:
    observation = _recorder_run(tmp_path, simulation=True)
    calls: list[str] = []

    class NoOpExecutor:
        def validate_request(self, _request_arg: object) -> None:
            calls.append("executor-validate")

        def run(self, _request_arg: object, _contexts: object) -> dict[str, object]:
            calls.append("executor-run")
            return {"artifact": "comparison.json"}

    def selector(_role: str, **_kwargs: object):
        raise AssertionError("runtime selection must follow common input preflight")

    runner = RuntimeProfileComparisonRunner(
        NoOpExecutor(),  # type: ignore[arg-type]
        repo_root=tmp_path,
        selector=selector,
    )

    with pytest.raises(SuperOfflineAdapterError, match="simulation observation"):
        runner.validate_request(_request(observation))
    assert calls == []


def test_builtin_executor_rejects_recorder_source_profile_before_doctor(
    tmp_path: Path,
) -> None:
    observation = _recorder_run(
        tmp_path,
        source_profile="rizon-umi3-iphone-rgb.v1",
    )
    executor = FastWAMOfflineComparisonExecutor(repo_root=tmp_path)

    with pytest.raises(SuperOfflineAdapterError, match="source_profile"):
        executor.validate_request(_request(observation))


def test_builtin_executor_rejects_missing_role_array_before_doctor(
    tmp_path: Path,
) -> None:
    observation = _recorder_run(tmp_path)
    frame_path = observation / "frames" / "frame_000000.npz"
    np.savez_compressed(
        frame_path,
        WebCameraOpenPi_1=np.zeros((360, 640, 3), dtype=np.uint8),
        RizonRobot_2=np.zeros(44, dtype=np.float64),
    )
    executor = FastWAMOfflineComparisonExecutor(repo_root=tmp_path)

    with pytest.raises(SuperOfflineAdapterError, match="missing"):
        executor.validate_request(_request(observation))


def test_builtin_executor_writes_binding_and_passes_it_to_each_evaluator(
    tmp_path: Path,
) -> None:
    observation = _recorder_run(tmp_path)
    request = _request(observation)
    calls: list[list[str]] = []

    def command_runner(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if "run" in command:
            _write_worker_output(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    executor = FastWAMOfflineComparisonExecutor(
        repo_root=tmp_path,
        command_runner=command_runner,
    )
    result = executor.run(request, _contexts(request))

    binding_path = Path(result["observation_bindings"])
    bindings = json.loads(binding_path.read_text(encoding="utf-8"))["bindings"]
    assert [binding["target_to_source"] for binding in bindings] == [
        {
            "WebCameraOpenPi_0": "WebCameraOpenPi_1",
            "WebCameraOpenPi_1": "WebCameraOpenPi_3",
            "RizonRobot_2": "RizonRobot_2",
        },
        {
            "WebCameraOpenPi_0": "WebCameraOpenPi_1",
            "WebCameraOpenPi_1": "WebCameraOpenPi_3",
            "RizonRobot_2": "RizonRobot_2",
        },
    ]
    run_commands = [command for command in calls if command[2] == "run"]
    assert all("--observation-binding" in command for command in run_commands)
    assert [command[command.index("--binding-index") + 1] for command in run_commands] == [
        "0",
        "1",
    ]


def test_builtin_executor_reuses_an_exact_completed_result(tmp_path: Path) -> None:
    observation = _recorder_run(tmp_path)
    request = _request(observation)
    calls: list[list[str]] = []

    def command_runner(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if "run" in command:
            _write_worker_output(command)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    executor = FastWAMOfflineComparisonExecutor(
        repo_root=tmp_path,
        command_runner=command_runner,
    )
    first = executor.run(request, _contexts(request))
    second_request = replace(request, job_id="b" * 32)
    second = executor.run(second_request, _contexts(second_request))

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["cached_from_job_id"] == "a" * 32
    assert second["output_dir"] == first["output_dir"]
    assert [command[2] for command in calls] == ["doctor", "run", "doctor", "run"]
    assert not (tmp_path / "logs" / "superoffline" / ("b" * 32)).exists()

    changed_request = replace(
        request,
        job_id="c" * 32,
        metadata={"prompt": "A different instruction"},
    )
    changed = executor.run(changed_request, _contexts(changed_request))

    assert changed["cache_hit"] is False
    assert [command[2] for command in calls] == [
        "doctor",
        "run",
        "doctor",
        "run",
        "doctor",
        "run",
        "doctor",
        "run",
    ]


def test_builtin_executor_rejects_an_unfinalized_recorder_run(tmp_path: Path) -> None:
    observation = _recorder_run(tmp_path)
    manifest_path = observation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["stopped_at_ns"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    executor = FastWAMOfflineComparisonExecutor(repo_root=tmp_path)

    with pytest.raises(SuperOfflineAdapterError, match="completed recorder"):
        executor.validate_request(_request(observation))


def test_builtin_executor_accepts_a_finalized_timestamped_recorder_run(
    tmp_path: Path,
) -> None:
    _install_reviewed_calibration_bundle(tmp_path)
    recorder = TimestampedObservationRecorder(
        {
            "enabled": True,
            "output_dir": str(tmp_path / "recordings"),
            "source_profile": "rizon-umi3-fisheye-openpi-rgb.v1",
            "capture_device_descriptors": reviewed_capture_device_descriptors(
                "rizon-umi3-fisheye-openpi-rgb.v1"
            ),
            "calibration_files": {
                "tcp_to_umi": "calibration/tcp_to_umi_transform.json",
                "left_base_to_world": "calibration/left_base_to_world.json",
                "right_base_to_world": "calibration/right_base_to_world.json",
                "left_gripper": "configs/devices/calibration/lktech_gripper/LKTech-Gripper-1_2_id1_calibration.json",
                "right_gripper": "configs/devices/calibration/lktech_gripper/LKTech-Gripper-2_2_id1_calibration.json",
            },
        },
        run_started_at_ns=1,
        camera_devices=["WebCameraOpenPi_1", "WebCameraOpenPi_3"],
        robot_device="RizonRobot_2",
    )
    recorder.start(lambda: None)
    recorder.record_frame(
        10,
        {
            "WebCameraOpenPi_1": np.zeros((360, 640, 3), dtype=np.uint8),
            "WebCameraOpenPi_3": np.zeros((360, 640, 3), dtype=np.uint8),
            "RizonRobot_2": np.zeros(44, dtype=np.float64),
        },
    )
    recorder.record_inference(
        11,
        {
            "WebCameraOpenPi_1": np.zeros((360, 640, 3), dtype=np.uint8),
            "WebCameraOpenPi_3": np.zeros((360, 640, 3), dtype=np.uint8),
            "RizonRobot_2": np.zeros(44, dtype=np.float64),
        },
    )
    recorder.stop()

    executor = FastWAMOfflineComparisonExecutor(repo_root=tmp_path)

    executor.validate_request(_request(recorder.run_dir))


@pytest.mark.parametrize(
    ("index_name", "row"),
    [
        (
            "frames.jsonl",
            '{"kind":"frame","index":0,"timestamp_ns":10,'
            '"file":"frames/missing.npz"}\n',
        ),
        (
            "inference.jsonl",
            '{"kind":"inference","index":0,"timestamp_ns":11,'
            '"file":"inference/../../outside.npz"}\n',
        ),
    ],
)
def test_builtin_executor_rejects_missing_or_escaping_recorder_artifacts(
    tmp_path: Path,
    index_name: str,
    row: str,
) -> None:
    observation = _recorder_run(tmp_path)
    (observation / index_name).write_text(row, encoding="utf-8")
    executor = FastWAMOfflineComparisonExecutor(repo_root=tmp_path)

    with pytest.raises(SuperOfflineAdapterError, match="no valid"):
        executor.validate_request(_request(observation))
