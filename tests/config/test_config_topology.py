"""Focused static validation tests for structured endpoint topology."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from device_starter import DeviceStarter
from utils.config_topology import ConfigTopologyError, validate_config_topology
from utils.hydra_config import normalize_hydra_config


def _state(shape: list[int], dtype: str = "float64") -> dict[str, object]:
    return {
        "fps": 30.0,
        "buffer_size": 4,
        "hardware_latency_ms": 0.0,
        "format": {
            "shape": shape,
            "dtype": dtype,
            "type": [f"linear:{shape[0]}"],
        },
    }


def _control(
    shape: list[int],
    *,
    dtype: str = "float64",
    source_selection: dict[str, object] | None = None,
) -> dict[str, object]:
    control: dict[str, object] = {
        "type": "joints",
        "format": {
            "shape": shape,
            "dtype": dtype,
            "type": [f"linear:{shape[0]}"],
        },
    }
    if source_selection is not None:
        control["source_selection"] = source_selection
    return control


def _robot(
    device_class_id: int,
    shape: list[int],
    *,
    source_selection: dict[str, object] | None = None,
    enabled: bool = True,
) -> dict[str, object]:
    return {
        "class": "BaseRobot",
        "device_class_id": device_class_id,
        "enabled": enabled,
        "state_feedback": _state(shape),
        "control": _control(shape, source_selection=source_selection),
    }


def _reader(
    device_class: str,
    device_class_id: int,
    shape: list[int],
    *,
    enabled: bool = True,
) -> dict[str, object]:
    return {
        "class": device_class,
        "device_class_id": device_class_id,
        "enabled": enabled,
        "state_feedback": _state(shape),
    }


def _policy_target(
    device: str = "BaseRobot", device_class_id: int = 0, shape: list[int] | None = None
) -> dict[str, object]:
    return {
        "endpoint": {"device": device, "device_class_id": device_class_id, "port": "command"},
        "shape": shape or [2],
        "dtype": "float64",
    }


def _base_config() -> dict[str, object]:
    return {
        "devices": [_robot(0, [2])],
        "policy": {"controls": [_policy_target()]},
        "action_executor": {"robot_control_shms": [_policy_target()]},
    }


def test_policy_and_action_targets_require_enabled_writable_exact_control() -> None:
    config = _base_config()
    validate_config_topology(config, validate_all_enabled_devices=True)

    disabled = deepcopy(config)
    disabled["devices"][0]["enabled"] = False
    with pytest.raises(
        ConfigTopologyError, match="refers to disabled device BaseRobot_0"
    ):
        validate_config_topology(disabled, validate_all_enabled_devices=True)

    wrong_shape = deepcopy(config)
    wrong_shape["policy"]["controls"][0]["shape"] = [3]
    with pytest.raises(
        ConfigTopologyError, match=r"policy\.controls\[0\] command payload"
    ):
        validate_config_topology(wrong_shape, validate_all_enabled_devices=True)

    wrong_dtype = deepcopy(config)
    wrong_dtype["action_executor"]["robot_control_shms"][0]["dtype"] = "float32"
    with pytest.raises(
        ConfigTopologyError,
        match=r"action_executor\.robot_control_shms\[0\] command payload",
    ):
        validate_config_topology(wrong_dtype, validate_all_enabled_devices=True)

    read_only = deepcopy(config)
    read_only["devices"] = [_reader("BaseDevice", 0, [2])]
    read_only["policy"]["controls"][0]["endpoint"]["device"] = "BaseDevice"
    read_only["action_executor"]["robot_control_shms"][0]["endpoint"]["device"] = (
        "BaseDevice"
    )
    with pytest.raises(
        ConfigTopologyError, match="must refer to a writable device with control"
    ):
        validate_config_topology(read_only, validate_all_enabled_devices=True)


def test_source_state_and_command_endpoints_match_consumer_control() -> None:
    source_selection = {
        "mode": "single_source",
        "source": {"device": "BaseDevice", "device_class_id": 2, "port": "state"},
    }
    config = {
        "devices": [
            _robot(0, [2], source_selection=source_selection),
            _reader("BaseDevice", 2, [3]),
        ]
    }
    with pytest.raises(
        ConfigTopologyError, match="state payload from BaseDevice_2 must match"
    ):
        validate_config_topology(config, validate_all_enabled_devices=True)

    wrong_dtype = deepcopy(config)
    wrong_dtype["devices"][1]["state_feedback"]["format"]["shape"] = [2]
    wrong_dtype["devices"][1]["state_feedback"]["format"]["dtype"] = "float32"
    wrong_dtype["devices"][1]["state_feedback"]["format"]["type"] = ["linear:2"]
    with pytest.raises(
        ConfigTopologyError, match="state payload from BaseDevice_2 must match"
    ):
        validate_config_topology(wrong_dtype, validate_all_enabled_devices=True)

    disabled = deepcopy(config)
    disabled["devices"][1]["enabled"] = False
    with pytest.raises(
        ConfigTopologyError, match="refers to disabled device BaseDevice_2"
    ):
        validate_config_topology(disabled, validate_all_enabled_devices=True)

    command_source = deepcopy(config)
    command_source["devices"][0]["control"]["source_selection"]["source"] = {
        "device": "BaseDevice",
        "device_class_id": 2,
        "port": "command",
    }
    command_source["devices"][1]["state_feedback"] = _state([2])
    with pytest.raises(
        ConfigTopologyError, match="must refer to a writable device with control"
    ):
        validate_config_topology(command_source, validate_all_enabled_devices=True)


def test_self_resolves_to_the_enclosing_device_instance() -> None:
    config = {
        "devices": [
            _robot(
                0,
                [2],
                source_selection={
                    "mode": "single_source",
                    "source": {"device": "self", "port": "command"},
                },
            ),
            _robot(1, [3]),
        ]
    }

    validate_config_topology(config, validate_all_enabled_devices=True)


def test_dual_source_selector_requires_a_known_flag_producer() -> None:
    selection = {
        "mode": "dual_source",
        "sources": {
            "policy": {"device": "self", "port": "command"},
            "teleoperation": {"device": "BaseDevice", "device_class_id": 2, "port": "state"},
        },
        "dual_source_selector": {
            "endpoint": {
                "device": "BaseDevice",
                "device_class_id": 2,
                "port": "control_flag",
            },
        },
    }
    config = {
        "devices": [
            _robot(0, [2], source_selection=selection),
            _reader("BaseDevice", 2, [2]),
        ]
    }
    with pytest.raises(ConfigTopologyError, match="does not produce control_flag"):
        validate_config_topology(config, validate_all_enabled_devices=True)

    valid_selector = deepcopy(config)
    valid_selector["devices"][1] = _reader("KeyboardTeleoperator", 2, [2])
    valid_selector["devices"][0]["control"]["source_selection"]["sources"][
        "teleoperation"
    ]["device"] = "KeyboardTeleoperator"
    valid_selector["devices"][0]["control"]["source_selection"]["dual_source_selector"][
        "endpoint"
    ]["device"] = "KeyboardTeleoperator"
    validate_config_topology(valid_selector, validate_all_enabled_devices=True)


def test_device_starter_preflight_blocks_invalid_topology_before_parsing_devices() -> (
    None
):
    config = _base_config()
    config["devices"][0]["enabled"] = False
    starter = DeviceStarter.__new__(DeviceStarter)
    starter.config = config
    starter.running = False
    starter._parse_device_configs = lambda: pytest.fail("device parsing must not run")

    assert not starter.start_all()


@pytest.mark.config
def test_all_composed_launch_configs_have_a_valid_endpoint_topology(
    repo_root: Path,
) -> None:
    """Every native launch config must resolve its enabled endpoint graph."""
    config_root = repo_root / "configs"
    config_names: list[str] = []
    for path in sorted(config_root.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "defaults" in raw:
            config_names.append(
                path.relative_to(config_root).with_suffix("").as_posix()
            )

    for config_name in config_names:
        hydra_state = HydraConfig.instance()
        previous_hydra_config = hydra_state.cfg
        try:
            with initialize_config_dir(
                config_dir=str(config_root),
                version_base="1.3",
                job_name="config_topology",
            ):
                composed = compose(config_name=config_name, return_hydra_config=True)
                hydra_state.set_config(composed)
                application_keys = [str(key) for key in composed if str(key) != "hydra"]
                config = normalize_hydra_config(
                    OmegaConf.masked_copy(composed, application_keys)
                )
        finally:
            hydra_state.cfg = previous_hydra_config

        validate_config_topology(config, validate_all_enabled_devices=True)
