from __future__ import annotations

import json
from pathlib import Path

from incubator.control_panel.inference_options import (
    as_bool,
    build_control_panel_recording_config,
    load_rollout_output_root,
    mock_inference_config_error,
    save_rollout_output_root,
)
from utils.observation_recorder import TimestampedObservationRecorder


def supported_config(tmp_path: Path) -> dict[str, object]:
    return {
        "summary_shm": "device_summary_data",
        "devices": [
            {
                "class": "WebCameraOpenPi",
                "device_class_id": 0,
                "data_shape": [480, 640, 3],
            },
            {
                "class": "WebCameraOpenPi",
                "device_class_id": 2,
                "data_shape": [480, 640, 3],
            },
            {
                "class": "RizonRobot",
                "device_class_id": 2,
                "state_feedback": {
                    "fps": 120.0,
                    "format": {
                        "shape": [44],
                        "dtype": "float64",
                        "type": ["linear:44"],
                    },
                },
                "control": {
                    "type": "eef_abs",
                    "format": {
                        "shape": [16],
                        "dtype": "float64",
                        "type": ["linear:16"],
                    },
                },
                "gripper": {"value_mode": "width_m"},
            },
        ],
        "policy": {
            "params": {
                "task": "test recording",
                "ckpt_dir": str(tmp_path / "missing-checkpoint"),
            },
            "obs": {
                "devices": [
                    "WebCameraOpenPi_0",
                    "WebCameraOpenPi_2",
                    "RizonRobot_2",
                ]
            },
            "controls": [
                {
                    "endpoint": {
                        "device": "RizonRobot",
                        "device_class_id": 2,
                        "port": "command",
                    },
                    "shape": [16],
                    "dtype": "float64",
                }
            ],
            "infer_rollout": {
                "enabled": True,
                "output_dir": "/yaml/must/not/win",
                "save_interval": 99,
                "queue_size": 1,
            },
        },
    }


def test_panel_recording_profile_replaces_yaml_without_mutating_source(
    tmp_path: Path,
) -> None:
    config = supported_config(tmp_path)
    output = tmp_path / "rollouts"
    output.mkdir()
    resolved = build_control_panel_recording_config(
        config, enabled=True, output_root=output
    )
    rollout = resolved["policy"]["infer_rollout"]

    assert config["policy"]["infer_rollout"]["save_interval"] == 99
    assert rollout["enabled"] is True
    assert rollout["required"] is True
    assert rollout["output_dir"] == str(output)
    assert rollout["save_interval"] == 0.001
    assert rollout["queue_size"] == 50000
    assert rollout["auto_start"] is False
    assert rollout["data_format"] == "NEDF3"
    assert rollout["task_name"] == "test recording"
    assert rollout["left_camera"] == "WebCameraOpenPi_0"
    assert rollout["right_camera"] == "WebCameraOpenPi_2"
    assert rollout["head_camera"] is None
    assert rollout["control_shms"] == config["policy"]["controls"]
    assert rollout["control_shms"] is not config["policy"]["controls"]


def test_mock_support_does_not_depend_on_checkpoint(tmp_path: Path) -> None:
    config = supported_config(tmp_path)
    assert not Path(config["policy"]["params"]["ckpt_dir"]).exists()
    assert mock_inference_config_error(config) is None

    config["devices"][2]["control"]["type"] = "joints"
    assert "eef_abs" in str(mock_inference_config_error(config))

    config = supported_config(tmp_path)
    config["policy"]["obs"]["devices"].remove("RizonRobot_2")
    assert "policy.obs.devices" in str(mock_inference_config_error(config))


def test_shared_boolean_parser_handles_string_values() -> None:
    assert as_bool("true") is True
    assert as_bool("false") is False
    assert as_bool("0") is False
    assert as_bool("yes") is True


def test_mock_support_ignores_string_disabled_devices(tmp_path: Path) -> None:
    config = supported_config(tmp_path)
    config["devices"][2]["enabled"] = "false"
    assert "exactly one configured" in str(mock_inference_config_error(config))


def test_mock_support_accepts_rizon_without_arm_switch(tmp_path: Path) -> None:
    config = supported_config(tmp_path)
    assert mock_inference_config_error(config) is None


def test_observation_recorder_parses_string_flags(tmp_path: Path) -> None:
    recorder = TimestampedObservationRecorder(
        {"enabled": "false", "compress": "false"},
        base_dir=tmp_path,
    )
    assert recorder.enabled is False
    assert recorder.compress is False


def test_rollout_output_root_preferences_round_trip(tmp_path: Path) -> None:
    preferences = tmp_path / "state" / "inference-options.json"
    output = tmp_path / "rollouts"
    output.mkdir()
    save_rollout_output_root(preferences, output)

    assert load_rollout_output_root(preferences) == output.resolve()
    payload = json.loads(preferences.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "rollout_output_root": str(output.resolve()),
    }


def test_absolute_rollout_preference_is_not_workspace_specific(
    tmp_path: Path,
) -> None:
    preferences = tmp_path / "state" / "inference-options.json"
    preferences.parent.mkdir()
    rollout_root = tmp_path / "external-workspace" / "rollouts"
    preferences.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rollout_output_root": str(rollout_root),
            }
        ),
        encoding="utf-8",
    )

    assert load_rollout_output_root(preferences) == rollout_root.resolve()
