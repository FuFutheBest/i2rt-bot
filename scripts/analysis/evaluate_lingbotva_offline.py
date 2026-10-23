#!/usr/bin/env python3
"""Evaluate one dual-arm LingBotVA checkpoint from a completed recorder run.

The evaluator is deliberately read-only: it starts a temporary loopback
LingBotVA server in the already selected runtime-profile interpreter, reads one
exact recorded observation, and writes an action artifact.  It never creates a
policy connector, attaches to shared memory, or opens a robot control output.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, TextIO
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.fastwam_offline_utils import (  # noqa: E402
    ObservationDataset,
    batch_child_argv,
    observation_image_preview,
    parse_inference_indices,
    write_image,
)
from utils.superoffline_action_calibration import prepare_action_calibration  # noqa: E402
from utils.superoffline_registry import (  # noqa: E402
    DUAL_ARM_TCP_ACTION_CONTRACT,
    DUAL_ARM_TCP_ACTION_FRAME,
    DUAL_ARM_TCP_ACTION_LAYOUT,
    DUAL_ARM_TCP_ACTION_UNITS,
    get_superoffline_model_contract,
)
from utils.superoffline_observation import (  # noqa: E402
    read_observation_binding,
    validate_binding_policy_devices,
)


SUPPORTED_CONFIG_NAMES = frozenset(
    {
        "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva",
        "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva_async",
        "policy_infer/lingbotva/config_infer_iphone_umi3_dual_lingbotva",
    }
)
DEFAULT_CONFIG_NAME = "policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva"
RECORDER_SCHEMA = "superinference.timestamped_observation"
ACTION_DIM = 16
DEFAULT_SERVER_START_TIMEOUT_S = 300.0
SERVER_STOP_TIMEOUT_S = 15.0


@dataclass
class _LingBotServer:
    process: Any
    port: int
    log_path: Path
    log_handle: TextIO


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping in the selected config")
    return value


def _compose_runtime_config(config_name: str) -> Mapping[str, Any]:
    from incubator.control_panel.runtime_config import compose_runtime_config

    return compose_runtime_config(config_name)


def _config_for_offline_evaluation(
    config_name: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Compose only an explicitly supported, dual-arm LingBotVA preset."""
    if config_name not in SUPPORTED_CONFIG_NAMES:
        supported = ", ".join(sorted(SUPPORTED_CONFIG_NAMES))
        raise ValueError(
            "LingBotVA offline evaluation supports only explicit dual-arm configs: "
            f"{supported}"
        )
    root_config = _compose_runtime_config(config_name)
    policy_config = _mapping(root_config.get("policy"), field="policy")
    if policy_config.get("class") != "LingBotVAPolicy":
        raise ValueError(
            f"{config_name!r} configures {policy_config.get('class')!r}, "
            "not LingBotVAPolicy"
        )
    _validate_dual_arm_control_contract(policy_config)
    return root_config, policy_config


def _control_shape_size(value: Any) -> int | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    size = 1
    for dimension in value:
        if isinstance(dimension, bool):
            return None
        try:
            parsed = int(dimension)
        except (TypeError, ValueError):
            return None
        if parsed < 1:
            return None
        size *= parsed
    return size


def _validate_dual_arm_control_contract(policy_config: Mapping[str, Any]) -> None:
    controls = policy_config.get("controls")
    if not isinstance(controls, list):
        raise ValueError("LingBotVA offline evaluation requires policy.controls")
    dimensions = [
        _control_shape_size(control.get("shape"))
        for control in controls
        if isinstance(control, Mapping)
    ]
    if ACTION_DIM not in dimensions:
        raise ValueError(
            "LingBotVA offline evaluation requires a dual-arm 16-dimensional "
            f"policy control, got shapes={dimensions}"
        )


def _string_list(value: Any, *, field: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = [str(item).strip() for item in value if str(item).strip()]
    if len(result) != len(value):
        raise ValueError(f"{field} must not contain empty values")
    return result


def _selected_devices(policy_config: Mapping[str, Any]) -> tuple[list[str], str]:
    """Select recorder keys from the policy's resolved live configuration."""
    params = _mapping(policy_config.get("params"), field="policy.params")
    camera_devices = _string_list(
        params.get("camera_devices"), field="policy.params.camera_devices"
    )
    if len(camera_devices) != 2:
        raise ValueError(
            "LingBotVA offline evaluation requires exactly two configured cameras, "
            f"got {camera_devices}"
        )
    robot_device = params.get("robot_device")
    if not isinstance(robot_device, str) or not robot_device.strip():
        raise ValueError("policy.params.robot_device must be a non-empty string")
    robot_device = robot_device.strip()

    observation = _mapping(policy_config.get("obs"), field="policy.obs")
    configured_devices = _string_list(
        observation.get("devices"), field="policy.obs.devices"
    )
    expected_devices = [*camera_devices, robot_device]
    if configured_devices != expected_devices:
        raise ValueError(
            "LingBotVA policy camera/robot parameters must match policy.obs.devices: "
            f"expected {expected_devices}, got {configured_devices}"
        )
    model_camera_keys = _string_list(
        params.get("model_camera_keys"), field="policy.params.model_camera_keys"
    )
    if len(model_camera_keys) != len(camera_devices):
        raise ValueError(
            "policy.params.model_camera_keys must have one entry per camera device"
        )
    return camera_devices, robot_device


def _action_step_interval_ns(policy_config: Mapping[str, Any]) -> int:
    raw_fps = policy_config.get("control_fps")
    if isinstance(raw_fps, bool):
        raise ValueError("policy.control_fps must be a positive finite number")
    try:
        control_fps = float(raw_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("policy.control_fps must be a positive finite number") from exc
    if not math.isfinite(control_fps) or control_fps <= 0:
        raise ValueError("policy.control_fps must be a positive finite number")
    interval_ns = int(round(1_000_000_000 / control_fps))
    if interval_ns < 1:
        raise ValueError("policy.control_fps produces an invalid action interval")
    return interval_ns


def _completed_observation_dataset(
    path: Path | str,
    *,
    device_aliases: Mapping[str, str] | None = None,
) -> ObservationDataset:
    """Load a finalized real-robot recorder run before starting any model work."""
    root = Path(path).expanduser().resolve()
    if root.is_file() and root.name == "manifest.json":
        root = root.parent
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise ValueError(
            "observation-dir must be a completed timestamped recorder run with manifest.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read recorder manifest: {manifest_path}") from exc
    if manifest.get("schema") != RECORDER_SCHEMA:
        raise ValueError("observation-dir does not use the timestamped recorder schema")
    source = str(manifest.get("source", "")).lower()
    source_type = str(manifest.get("source_type", "")).lower()
    if "sim" in source or "sim" in source_type:
        raise ValueError("simulation recorder inputs are not accepted by LingBotVA offline evaluation")
    try:
        stopped_at_ns = int(manifest.get("stopped_at_ns"))
    except (TypeError, ValueError) as exc:
        raise ValueError("observation-dir recorder run is not finalized") from exc
    if stopped_at_ns <= 0:
        raise ValueError("observation-dir recorder run is not finalized")

    dataset = ObservationDataset(root, device_aliases=device_aliases)
    for field, actual_count in (
        ("frame_count", len(dataset.frames)),
        ("inference_count", len(dataset.inferences)),
    ):
        try:
            declared_count = int(manifest.get(field))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"recorder manifest is missing {field}") from exc
        if declared_count < 1 or declared_count != actual_count:
            raise ValueError(
                f"recorder {field} does not match the finalized recorder contents"
            )
    return dataset


def _resolve_inference_index(dataset: ObservationDataset, requested: int) -> int:
    index = requested
    if index < 0:
        index += len(dataset.inferences)
    if index < 0 or index >= len(dataset.inferences):
        raise IndexError(f"inference-index {requested} is outside the recorded run")
    return index


def _server_config_path(config_name: str) -> Path:
    if config_name not in SUPPORTED_CONFIG_NAMES:
        raise ValueError(f"unsupported LingBotVA config: {config_name!r}")
    path = (REPO_ROOT / "configs" / f"{config_name}.yaml").resolve()
    config_root = (REPO_ROOT / "configs").resolve()
    if not path.is_relative_to(config_root) or not path.is_file():
        raise ValueError(f"LingBotVA config file does not exist: {path}")
    return path


def _allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _server_command(
    *,
    config_name: str,
    checkpoint: Path,
    port: int,
    device: str,
    save_root: Path,
) -> list[str]:
    """Use the selected evaluator interpreter for the temporary model server."""
    return [
        sys.executable,
        str(REPO_ROOT / "consumer" / "inference" / "lingbot_va_server.py"),
        "--config",
        str(_server_config_path(config_name)),
        "--ckpt-path",
        str(checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        device,
        "--save-root",
        str(save_root),
    ]


def _server_environment() -> dict[str, str]:
    """Preserve the profile environment while making the script-path server importable."""
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    source_paths = [str(REPO_ROOT), str(REPO_ROOT / "third_party" / "nmx_vla")]
    existing = [
        value
        for value in environment.get("PYTHONPATH", "").split(os.pathsep)
        if value and value not in source_paths
    ]
    environment["PYTHONPATH"] = os.pathsep.join([*source_paths, *existing])
    return environment


def _start_server(
    *,
    config_name: str,
    checkpoint: Path,
    device: str,
    output_dir: Path,
) -> _LingBotServer:
    """Start the heavy server in its own process group for reliable cleanup."""
    server_dir = output_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=False)
    log_path = server_dir / "server.log"
    log_handle = log_path.open("w", encoding="utf-8")
    port = _allocate_loopback_port()
    command = _server_command(
        config_name=config_name,
        checkpoint=checkpoint,
        port=port,
        device=device,
        save_root=server_dir,
    )
    popen_kwargs: dict[str, Any] = {
        "cwd": REPO_ROOT,
        "env": _server_environment(),
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **popen_kwargs)
    except BaseException:
        log_handle.close()
        raise
    return _LingBotServer(
        process=process,
        port=port,
        log_path=log_path,
        log_handle=log_handle,
    )


def _server_log_tail(log_path: Path, *, max_chars: int = 2000) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:].strip()


def _server_is_healthy(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1.0) as response:
            return int(getattr(response, "status", response.getcode())) == 200
    except (OSError, URLError, ValueError):
        return False


def _wait_for_server(server: _LingBotServer, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        exit_code = server.process.poll()
        if exit_code is not None:
            detail = _server_log_tail(server.log_path)
            raise RuntimeError(
                "LingBotVA server exited before it became healthy "
                f"(exit code {exit_code}): {detail or 'no server log output'}"
            )
        if _server_is_healthy(server.port):
            return
        if time.monotonic() >= deadline:
            detail = _server_log_tail(server.log_path)
            raise TimeoutError(
                "LingBotVA server did not become healthy before the startup timeout: "
                f"{detail or 'no server log output'}"
            )
        time.sleep(0.25)


def _stop_server(server: _LingBotServer, *, timeout_s: float = SERVER_STOP_TIMEOUT_S) -> None:
    """Terminate the server's complete POSIX process group, even on failures."""
    process = server.process
    try:
        if process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(int(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif hasattr(signal, "CTRL_BREAK_EVENT"):
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                except (OSError, AttributeError):
                    process.terminate()
            else:
                process.terminate()
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(int(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait(timeout=timeout_s)
    finally:
        server.log_handle.close()


def _build_policy(
    policy_config: Mapping[str, Any],
    *,
    port: int,
    prompt: str,
    camera_devices: Sequence[str],
    robot_device: str,
    connect_timeout_s: float,
) -> Any:
    """Import the explicitly configured LingBotVA client only after server startup."""
    from policy.lingbot_va_policy import LingBotVAPolicy

    params = dict(_mapping(policy_config.get("params"), field="policy.params"))
    params.update(
        {
            "host": "127.0.0.1",
            "port": int(port),
            "task": prompt,
            "camera_devices": list(camera_devices),
            "robot_device": robot_device,
            "connect_timeout": float(connect_timeout_s),
            # A one-shot offline evaluation has no action-history continuity.
            "enable_rtc": False,
            "reset_on_load": True,
            "stateless_reset_each_predict": False,
        }
    )
    return LingBotVAPolicy(**params)


def _predict_like_live(
    policy: Any,
    observation: Mapping[str, np.ndarray],
    *,
    robot_device: str,
    chunk_length: int,
) -> np.ndarray:
    """Use LingBotVA's public client conversion to canonical TCP actions."""
    result = policy.predict(
        dict(observation),
        action_configs=[{"device_name": robot_device, "action_dim": ACTION_DIM}],
        chunk_length=chunk_length,
        action_index=None,
    )
    if not isinstance(result, Mapping) or robot_device not in result:
        raise ValueError(
            "LingBotVA policy did not return an action for configured robot "
            f"{robot_device!r}"
        )
    action = np.asarray(result[robot_device], dtype=np.float64)
    if action.ndim != 2 or action.shape[0] < 1 or action.shape[1] != ACTION_DIM:
        raise ValueError(
            "LingBotVA policy must return canonical dual-arm action [T,16], "
            f"got {action.shape}"
        )
    return np.ascontiguousarray(action)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--inference-index", type=int, default=-1)
    parser.add_argument("--inference-indices", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--server-start-timeout", type=float, default=DEFAULT_SERVER_START_TIMEOUT_S)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observation-binding", type=Path, default=None)
    parser.add_argument("--binding-index", type=int, default=0)
    parser.add_argument("--action-calibration-snapshot", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw_argv)
    if args.inference_indices is not None:
        indices = parse_inference_indices(args.inference_indices)
        batch_root = args.output_dir.expanduser().resolve()
        batch_root.mkdir(parents=True, exist_ok=False)
        summaries: list[dict[str, Any]] = []
        for inference_index in indices:
            observation_dir = batch_root / "observations" / f"{inference_index:06d}"
            main(
                batch_child_argv(
                    raw_argv,
                    inference_index=inference_index,
                    output_dir=observation_dir,
                )
            )
            summaries.append(
                json.loads((observation_dir / "summary.json").read_text(encoding="utf-8"))
            )
        (batch_root / "batch_summary.json").write_text(
            json.dumps(
                {"inference_indices": indices, "observations": summaries},
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    if not math.isfinite(args.server_start_timeout) or args.server_start_timeout <= 0:
        raise ValueError("server-start-timeout must be a positive finite number")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    _root_config, policy_config = _config_for_offline_evaluation(args.config_name)
    camera_devices, robot_device = _selected_devices(policy_config)
    action_step_interval_ns = _action_step_interval_ns(policy_config)
    binding = (
        None
        if args.observation_binding is None
        else read_observation_binding(
            args.observation_binding,
            binding_index=int(args.binding_index),
            config_name=str(args.config_name),
            observation_dir=args.observation_dir,
        )
    )
    if binding is not None:
        camera_devices, robot_device = validate_binding_policy_devices(
            binding,
            camera_devices=camera_devices,
            robot_device=robot_device,
            config_name=str(args.config_name),
        )
    if binding is None:
        dataset = _completed_observation_dataset(args.observation_dir)
    else:
        dataset = _completed_observation_dataset(
            args.observation_dir,
            device_aliases=binding.target_to_source,
        )
    inference_index = _resolve_inference_index(dataset, args.inference_index)
    observation, observation_timestamp_ns = dataset.exact_policy_observation_at(
        inference_index,
        camera_devices=camera_devices,
        robot_device=robot_device,
    )
    params = _mapping(policy_config.get("params"), field="policy.params")
    if binding is not None:
        action_calibration, calibration_overrides = prepare_action_calibration(
            get_superoffline_model_contract(args.config_name).action_calibration,
            params,
            binding_id=binding.action_calibration_id,
            binding_file_digests=binding.action_calibration_file_digests,
            binding_digest=binding.action_calibration_digest,
            snapshot_root=args.action_calibration_snapshot,
            repo_root=REPO_ROOT,
        )
        if calibration_overrides:
            params = {**params, **calibration_overrides}
            policy_config = {**policy_config, "params": params}
    prompt = str(args.prompt if args.prompt is not None else params.get("task") or "")
    checkpoint = args.checkpoint.expanduser()
    server = _start_server(
        config_name=args.config_name,
        checkpoint=checkpoint,
        device=str(args.device),
        output_dir=output_dir,
    )
    policy: Any | None = None
    try:
        _wait_for_server(server, timeout_s=float(args.server_start_timeout))
        policy = _build_policy(
            policy_config,
            port=server.port,
            prompt=prompt,
            camera_devices=camera_devices,
            robot_device=robot_device,
            connect_timeout_s=float(args.server_start_timeout),
        )
        policy.load()
        started = time.perf_counter()
        action = _predict_like_live(
            policy,
            observation,
            robot_device=robot_device,
            chunk_length=int(policy_config.get("chunk_length", 1)),
        )
        inference_seconds = time.perf_counter() - started
    finally:
        try:
            if policy is not None:
                policy.close()
        finally:
            _stop_server(server)

    finite = bool(np.isfinite(action).all())
    np.savez_compressed(
        output_dir / "offline_result.npz",
        action=action,
        observation_timestamp_ns=np.asarray(observation_timestamp_ns, dtype=np.int64),
    )
    write_image(
        output_dir / "input_grid.jpg",
        observation_image_preview(observation, camera_devices),
    )
    summary = {
        "schema_version": 1,
        "config_name": args.config_name,
        "checkpoint": str(checkpoint.resolve()),
        "observation_dir": str(args.observation_dir.expanduser().resolve()),
        "inference_index": inference_index,
        "observation_timestamp_ns": observation_timestamp_ns,
        "camera_devices": camera_devices,
        "robot_device": robot_device,
        "prompt": prompt,
        "seed_requested": int(args.seed),
        "device": str(args.device),
        "action_shape": list(action.shape),
        "action_contract": DUAL_ARM_TCP_ACTION_CONTRACT,
        "action_layout": DUAL_ARM_TCP_ACTION_LAYOUT,
        "action_frame_id": DUAL_ARM_TCP_ACTION_FRAME,
        "action_units": DUAL_ARM_TCP_ACTION_UNITS,
        "action_step_interval_ns": action_step_interval_ns,
        "action_step_interval_source": "policy.control_fps",
        "finite": finite,
        "inference_seconds": inference_seconds,
        "artifacts": {
            "npz": "offline_result.npz",
            "canonical_action_key": "action",
            "action_contract": DUAL_ARM_TCP_ACTION_CONTRACT,
            "action_layout": DUAL_ARM_TCP_ACTION_LAYOUT,
            "action_frame_id": DUAL_ARM_TCP_ACTION_FRAME,
            "action_units": DUAL_ARM_TCP_ACTION_UNITS,
            "action_step_interval_ns": action_step_interval_ns,
            "input_preview": "input_grid.jpg",
        },
        "server_log": "server/server.log",
    }
    if binding is not None:
        summary["action_calibration_id"] = action_calibration.calibration_id
        summary["action_calibration_digest"] = action_calibration.digest
        summary["observation_binding_digest"] = binding.digest
        summary["source_profile"] = binding.source_profile
        summary["source_provenance_digest"] = binding.source_provenance_digest
        summary["recording_artifact_digest"] = binding.recording_artifact_digest
        summary["artifacts"]["observation_binding_digest"] = binding.digest
        summary["artifacts"]["action_calibration_id"] = action_calibration.calibration_id
        summary["artifacts"]["action_calibration_digest"] = action_calibration.digest
        summary["artifacts"]["source_provenance_digest"] = binding.source_provenance_digest
        summary["artifacts"]["recording_artifact_digest"] = binding.recording_artifact_digest
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not finite:
        raise RuntimeError("LingBotVA policy returned non-finite actions")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
