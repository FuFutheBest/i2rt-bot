"""Hardware-free contracts for the LingBotVA SuperOffline evaluator."""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts.analysis import evaluate_lingbotva_offline as evaluator
from scripts.analysis.capture_fastwam_offline_observation import write_observation_run
from utils.superoffline_observation import reviewed_capture_device_descriptors


def _policy_config(policy_class: str = "LingBotVAPolicy") -> dict[str, object]:
    return {
        "class": policy_class,
        "params": {
            "camera_devices": ["WebCameraDevice_1", "WebCameraDevice_0"],
            "model_camera_keys": [
                "observation.images.wrist_image_1",
                "observation.images.wrist_image_2",
            ],
            "robot_device": "RizonRobot_2",
            "task": "place parts",
        },
        "control_fps": 30.0,
        "chunk_length": 4,
        "obs": {
            "devices": [
                "WebCameraDevice_1",
                "WebCameraDevice_0",
                "RizonRobot_2",
            ]
        },
        "controls": [{"shape": [16], "dtype": "float64"}],
    }


@pytest.mark.parametrize("config_name", sorted(evaluator.SUPPORTED_CONFIG_NAMES))
def test_explicit_native_configs_compose_dual_lingbot_contract(config_name: str) -> None:
    _root, policy_config = evaluator._config_for_offline_evaluation(config_name)

    assert policy_config["class"] == "LingBotVAPolicy"
    camera_devices, robot_device = evaluator._selected_devices(policy_config)
    assert len(camera_devices) == 2
    assert robot_device == "RizonRobot_2"
    assert evaluator._action_step_interval_ns(policy_config) == 33_333_333


def test_config_allowlist_and_policy_class_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="explicit dual-arm configs"):
        evaluator._config_for_offline_evaluation("policy_infer/lingbotva/config_infer_iphone_umi_single_lingbotva")

    monkeypatch.setattr(
        evaluator,
        "_compose_runtime_config",
        lambda _name: {"policy": _policy_config("OtherPolicy")},
    )
    with pytest.raises(ValueError, match="not LingBotVAPolicy"):
        evaluator._config_for_offline_evaluation(evaluator.DEFAULT_CONFIG_NAME)


def test_selected_devices_and_control_interval_require_a_dual_contract() -> None:
    policy_config = _policy_config()
    assert evaluator._selected_devices(policy_config) == (
        ["WebCameraDevice_1", "WebCameraDevice_0"],
        "RizonRobot_2",
    )
    assert evaluator._action_step_interval_ns(policy_config) == 33_333_333

    single_arm = dict(policy_config)
    single_arm["controls"] = [{"shape": [8], "dtype": "float64"}]
    with pytest.raises(ValueError, match="16-dimensional"):
        evaluator._validate_dual_arm_control_contract(single_arm)

    invalid_fps = dict(policy_config)
    invalid_fps["control_fps"] = 0
    with pytest.raises(ValueError, match="positive finite"):
        evaluator._action_step_interval_ns(invalid_fps)


def test_completed_recorder_uses_exact_current_observation_utilities(tmp_path: Path) -> None:
    observation_dir = tmp_path / "observation"
    robot = np.arange(44, dtype=np.float64)
    write_observation_run(
        observation_dir,
        arrays={
            "WebCameraDevice_1": np.zeros((360, 640, 3), dtype=np.uint8),
            "WebCameraDevice_0": np.ones((360, 640, 3), dtype=np.uint8),
            "RizonRobot_2": robot,
        },
        device_timestamps_ns={
            "WebCameraDevice_1": 100,
            "WebCameraDevice_0": 101,
            "RizonRobot_2": 99,
        },
        camera_devices=["WebCameraDevice_1", "WebCameraDevice_0"],
        robot_device="RizonRobot_2",
        source_profile="rizon-umi3-fisheye-lingbotva-rgb.v1",
        capture_device_descriptors=reviewed_capture_device_descriptors(
            "rizon-umi3-fisheye-lingbotva-rgb.v1"
        ),
    )

    dataset = evaluator._completed_observation_dataset(observation_dir)
    observation, timestamp_ns = dataset.exact_policy_observation_at(
        0,
        camera_devices=["WebCameraDevice_1", "WebCameraDevice_0"],
        robot_device="RizonRobot_2",
    )

    assert timestamp_ns == 101
    assert np.array_equal(observation["RizonRobot_2"], robot)
    manifest_path = observation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("stopped_at_ns")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not finalized"):
        evaluator._completed_observation_dataset(observation_dir)


def test_server_command_stays_in_the_selected_evaluator_interpreter(tmp_path: Path) -> None:
    command = evaluator._server_command(
        config_name=evaluator.DEFAULT_CONFIG_NAME,
        checkpoint=tmp_path / "checkpoint",
        port=18123,
        device="cuda:0",
        save_root=tmp_path / "server",
    )

    assert command[:2] == [
        sys.executable,
        str(evaluator.REPO_ROOT / "consumer" / "inference" / "lingbot_va_server.py"),
    ]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "18123"
    assert command[command.index("--config") + 1].endswith(
        "configs/policy_infer/lingbotva/config_infer_fisheye_umi3_dual_lingbotva.yaml"
    )


def test_server_environment_preserves_the_profile_and_adds_repo_import_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/runtime/profile/site-packages")
    environment = evaluator._server_environment()

    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["PYTHONPATH"].split(evaluator.os.pathsep)[:2] == [
        str(evaluator.REPO_ROOT),
        str(evaluator.REPO_ROOT / "third_party" / "nmx_vla"),
    ]
    assert environment["PYTHONPATH"].endswith("/runtime/profile/site-packages")


def test_stop_server_terminates_the_posix_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, int, object]] = []

    class Process:
        pid = 773

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            events.append(("wait", 0, timeout))
            return 0

    monkeypatch.setattr(
        evaluator.os,
        "killpg",
        lambda pid, signal_number: events.append(("killpg", pid, signal_number)),
    )
    log_handle = StringIO()
    server = evaluator._LingBotServer(
        process=Process(),
        port=0,
        log_path=tmp_path / "server.log",
        log_handle=log_handle,
    )

    evaluator._stop_server(server, timeout_s=2.0)

    assert events[0] == ("killpg", 773, evaluator.signal.SIGTERM)
    assert events[1] == ("wait", 0, 2.0)
    assert log_handle.closed


def test_main_writes_a_canonical_action_without_importing_model_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy_config = _policy_config()
    events: list[str] = []

    class Dataset:
        inferences = [object()]

        def exact_policy_observation_at(
            self,
            inference_index: int,
            *,
            camera_devices: list[str],
            robot_device: str,
        ) -> tuple[dict[str, np.ndarray], int]:
            assert inference_index == 0
            return {
                camera_devices[0]: np.zeros((4, 5, 3), dtype=np.uint8),
                camera_devices[1]: np.ones((4, 5, 3), dtype=np.uint8),
                robot_device: np.zeros(44, dtype=np.float64),
            }, 123

    class Policy:
        def load(self) -> None:
            events.append("load")

        def close(self) -> None:
            events.append("close")

        def predict(self, _observation: object, **kwargs: object) -> dict[str, np.ndarray]:
            action_configs = kwargs["action_configs"]
            assert isinstance(action_configs, list)
            assert action_configs == [{"device_name": "RizonRobot_2", "action_dim": 16}]
            assert kwargs["chunk_length"] == 4
            return {"RizonRobot_2": np.full((4, 16), 1.25, dtype=np.float64)}

    class Server:
        port = 19300

    monkeypatch.setattr(
        evaluator,
        "_config_for_offline_evaluation",
        lambda _name: ({}, policy_config),
    )
    monkeypatch.setattr(evaluator, "_completed_observation_dataset", lambda _path: Dataset())
    monkeypatch.setattr(evaluator, "_start_server", lambda **_kwargs: Server())
    monkeypatch.setattr(evaluator, "_wait_for_server", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluator, "_stop_server", lambda _server: events.append("stop"))
    monkeypatch.setattr(evaluator, "_build_policy", lambda *_args, **_kwargs: Policy())

    output_dir = tmp_path / "result"
    evaluator.main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--observation-dir",
            str(tmp_path / "observation"),
            "--prompt",
            "stack blocks",
            "--output-dir",
            str(output_dir),
        ]
    )

    with np.load(output_dir / "offline_result.npz", allow_pickle=False) as artifact:
        assert artifact["action"].shape == (4, 16)
        assert np.all(artifact["action"] == 1.25)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["observation_timestamp_ns"] == 123
    assert summary["action_contract"] == evaluator.DUAL_ARM_TCP_ACTION_CONTRACT
    assert summary["action_layout"] == evaluator.DUAL_ARM_TCP_ACTION_LAYOUT
    assert summary["action_step_interval_ns"] == 33_333_333
    assert summary["artifacts"] == {
        "npz": "offline_result.npz",
        "input_preview": "input_grid.jpg",
        "canonical_action_key": "action",
        "action_contract": evaluator.DUAL_ARM_TCP_ACTION_CONTRACT,
        "action_layout": evaluator.DUAL_ARM_TCP_ACTION_LAYOUT,
        "action_frame_id": evaluator.DUAL_ARM_TCP_ACTION_FRAME,
        "action_units": evaluator.DUAL_ARM_TCP_ACTION_UNITS,
        "action_step_interval_ns": 33_333_333,
    }
    assert events == ["load", "close", "stop"]


def test_main_stops_server_when_client_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy_config = _policy_config()
    events: list[str] = []

    class Dataset:
        inferences = [object()]

        def exact_policy_observation_at(
            self,
            _inference_index: int,
            *,
            camera_devices: list[str],
            robot_device: str,
        ) -> tuple[dict[str, np.ndarray], int]:
            return {
                camera_devices[0]: np.zeros((4, 5, 3), dtype=np.uint8),
                camera_devices[1]: np.zeros((4, 5, 3), dtype=np.uint8),
                robot_device: np.zeros(44, dtype=np.float64),
            }, 123

    class Policy:
        def load(self) -> None:
            pass

        def close(self) -> None:
            events.append("close")
            raise RuntimeError("client close failed")

        def predict(self, _observation: object, **kwargs: object) -> dict[str, np.ndarray]:
            action_configs = kwargs["action_configs"]
            assert isinstance(action_configs, list)
            return {action_configs[0]["device_name"]: np.zeros((4, 16))}

    class Server:
        port = 19300

    monkeypatch.setattr(
        evaluator,
        "_config_for_offline_evaluation",
        lambda _name: ({}, policy_config),
    )
    monkeypatch.setattr(evaluator, "_completed_observation_dataset", lambda _path: Dataset())
    monkeypatch.setattr(evaluator, "_start_server", lambda **_kwargs: Server())
    monkeypatch.setattr(evaluator, "_wait_for_server", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluator, "_stop_server", lambda _server: events.append("stop"))
    monkeypatch.setattr(evaluator, "_build_policy", lambda *_args, **_kwargs: Policy())

    with pytest.raises(RuntimeError, match="client close failed"):
        evaluator.main(
            [
                "--checkpoint",
                str(tmp_path / "checkpoint"),
                "--observation-dir",
                str(tmp_path / "observation"),
                "--output-dir",
                str(tmp_path / "result"),
            ]
        )
    assert events == ["close", "stop"]


def test_predict_rejects_single_arm_action() -> None:
    class Policy:
        def predict(self, _observation: object, **kwargs: object) -> dict[str, np.ndarray]:
            action_configs = kwargs["action_configs"]
            assert isinstance(action_configs, list)
            return {action_configs[0]["device_name"]: np.zeros((2, 8))}

    with pytest.raises(ValueError, match=r"\[T,16\]"):
        evaluator._predict_like_live(
            Policy(),
            {"RizonRobot_2": np.zeros(44)},
            robot_device="RizonRobot_2",
            chunk_length=2,
        )
