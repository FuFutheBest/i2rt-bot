"""Lightweight config schema contract tests.

Author: OpenAI Assistant
"""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.config_schema import (
    REQUIRED_RUNTIME_SIGNAL_FILES,
    validate_superinference_config,
    validate_top_level_configs,
)
from utils.errors import ConfigError


@pytest.mark.config
def test_top_level_configs_follow_basic_schema(repo_root: Path) -> None:
    """Top-level runtime configs should keep common Hydra/policy fields well-formed."""
    issues = validate_top_level_configs((repo_root / "configs").glob("*.yaml"))

    assert [issue.format(repo_root) for issue in issues] == []


def _runtime_signal_files() -> dict[str, str]:
    return {key: f"/tmp/{key}" for key in REQUIRED_RUNTIME_SIGNAL_FILES}


def _valid_config() -> dict:
    return {
        "devices": {
            "devices": [
                {
                    "class": "BaseRobot",
                    "device_class_id": 0,
                    "state_feedback": {
                        "fps": 10.0,
                        "buffer_size": 3,
                        "hardware_latency_ms": 0.0,
                        "format": {
                            "shape": [2],
                            "dtype": "float64",
                            "type": ["linear:2"],
                        },
                    },
                    "control": {
                        "type": "joints",
                        "format": {
                            "shape": [2],
                            "dtype": "float64",
                            "type": ["linear:2"],
                        },
                    },
                }
            ]
        },
        "device_manager": {
            "type": "base",
            "params": {"buffer_size": 10},
        },
        "action_executor": {
            "type": None,
            "chunk_manager": {"type": "base_manager"},
        },
        "policy": {
            "class": "TinyMLPPolicy",
            "fps": 5.0,
            "chunk_length": 4,
            "obs": {"devices": ["BaseRobot_0"]},
            "controls": [
                {
                    "endpoint": {
                        "device": "BaseRobot",
                        "device_class_id": 0,
                        "port": "command",
                    },
                    "shape": [2],
                    "dtype": "float64",
                }
            ],
        },
        "runtime": {
            "signal_files": _runtime_signal_files(),
        },
    }


@pytest.mark.config
def test_validate_superinference_config_accepts_valid_minimal_config() -> None:
    """A minimal hardware-free config should satisfy the schema contract."""
    validate_superinference_config(
        _valid_config(),
        device_registry={"BaseRobot"},
        policy_registry={"TinyMLPPolicy"},
        manager_registry={"base", "base_policy_runner"},
        chunk_manager_registry={"base_manager"},
    )


@pytest.mark.config
@pytest.mark.parametrize("requirement", ["any", "usb_2.0", "USB_3.0"])
def test_validate_superinference_config_accepts_usb_port_requirement(
    requirement: str,
) -> None:
    config = _valid_config()
    config["devices"]["devices"][0]["usb_port_requirement"] = requirement

    validate_superinference_config(config)


@pytest.mark.config
def test_validate_superinference_config_rejects_unknown_usb_port_requirement() -> None:
    config = _valid_config()
    config["devices"]["devices"][0]["usb_port_requirement"] = "usb_1.1"

    with pytest.raises(ConfigError, match="usb_port_requirement"):
        validate_superinference_config(config)


@pytest.mark.config
def test_validate_superinference_config_rejects_flat_data_format() -> None:
    config = _valid_config()
    device = config["devices"]["devices"][0]
    device.pop("state_feedback")
    device["data_format"] = {"shape": [2], "type": ["linear:2"]}

    with pytest.raises(ConfigError, match="state_feedback"):
        validate_superinference_config(config)


@pytest.mark.config
def test_validate_superinference_config_accepts_grouped_device_contract() -> None:
    config = _valid_config()
    validate_superinference_config(config)


@pytest.mark.config
def test_validate_superinference_config_rejects_grouped_format_without_dtype() -> None:
    config = _valid_config()
    device = config["devices"]["devices"][0]
    device["state_feedback"] = {
        "fps": 10.0,
        "buffer_size": 3,
        "hardware_latency_ms": 0.0,
        "format": {"shape": [2], "type": ["linear:2"]},
    }

    with pytest.raises(ConfigError, match="dtype is required"):
        validate_superinference_config(config)


@pytest.mark.config
def test_validate_superinference_config_rejects_invalid_data_format() -> None:
    config = _valid_config()
    config["devices"]["devices"][0]["state_feedback"]["format"] = {
        "shape": [3],
        "type": ["xyz", "scalar"],
        "dtype": "float64",
    }

    with pytest.raises(ConfigError, match="primitive width"):
        validate_superinference_config(config)


@pytest.mark.config
def test_validate_superinference_config_rejects_unknown_policy() -> None:
    """Unknown policy classes should fail before a model process starts."""
    config = _valid_config()
    config["policy"]["class"] = "MissingPolicy"

    with pytest.raises(ConfigError, match="unknown registry value"):
        validate_superinference_config(config, policy_registry={"TinyMLPPolicy"})


@pytest.mark.config
def test_validate_superinference_config_rejects_invalid_control_shape() -> None:
    """Policy control shape is part of the action output contract."""
    config = _valid_config()
    config["policy"]["controls"][0]["shape"] = [0]

    with pytest.raises(ConfigError):
        validate_superinference_config(config)


@pytest.mark.config
def test_validate_superinference_config_rejects_raw_control_shm_name() -> None:
    """Policy targets must use a logical command endpoint, never a raw name."""
    config = _valid_config()
    config["policy"]["controls"][0] = {
        "shm": "BaseDevice_0_control",
        "shape": [2],
        "dtype": "float64",
    }

    with pytest.raises(ConfigError, match="shm was removed"):
        validate_superinference_config(config)


@pytest.mark.config
def test_validate_superinference_config_rejects_raw_sim_clock_name() -> None:
    config = _valid_config()
    config["action_executor"]["sim_clock_shm"] = "RoboTwinSimRobot_0_sim_clock"

    with pytest.raises(ConfigError, match="sim_clock_shm was removed"):
        validate_superinference_config(config)


@pytest.mark.config
def test_validate_superinference_config_rejects_missing_signal_file() -> None:
    """Runtime signal files should fail fast when required keys are absent."""
    config = _valid_config()
    del config["runtime"]["signal_files"]["policy_runner_reset_command"]

    with pytest.raises(ConfigError, match="missing signal_files keys"):
        validate_superinference_config(config)


@pytest.mark.config
def test_validate_superinference_config_requires_master_for_synchronized_manager() -> None:
    """Synchronized manager configs must declare the master device-list index."""
    config = _valid_config()
    config["device_manager"] = {"type": "synchronized", "params": {"buffer_size": 10}}

    with pytest.raises(ConfigError, match="master_device_class_id"):
        validate_superinference_config(config, manager_registry={"synchronized"})


@pytest.mark.config
def test_synchronized_manager_accepts_top_level_master_device_class_id() -> None:
    config = _valid_config()
    config["device_manager"] = {
        "type": "synchronized",
        "master_device_class_id": 0,
        "params": {"buffer_size": 10},
    }

    validate_superinference_config(config, manager_registry={"synchronized"})


@pytest.mark.config
def test_synchronized_manager_rejects_conflicting_master_device_class_ids() -> None:
    config = _valid_config()
    config["device_manager"] = {
        "type": "synchronized",
        "master_device_class_id": 0,
        "params": {"buffer_size": 10, "master_device_class_id": 1},
    }

    with pytest.raises(ConfigError, match="conflicts"):
        validate_superinference_config(config, manager_registry={"synchronized"})


@pytest.mark.config
def test_synchronized_manager_rejects_unknown_sync_mode() -> None:
    config = _valid_config()
    config["device_manager"] = {
        "type": "synchronized",
        "sync_mode": "nearest-neighbor",
        "master_device_class_id": 0,
        "params": {"buffer_size": 10},
    }

    with pytest.raises(ConfigError, match="sync_mode"):
        validate_superinference_config(config, manager_registry={"synchronized"})
