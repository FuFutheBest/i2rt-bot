"""Unit tests for lightweight config parsing contracts.

Author: OpenAI Assistant
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils.config_parser import (
    ensure_config_dict,
    flatten_hydra_config,
    normalize_hydra_config,
    parse_device_configs,
)


@pytest.mark.unit
def test_parse_device_configs_accepts_nested_devices(minimal_device_config: dict) -> None:
    """Hydra-style nested device configs should parse into standard device info."""
    devices = parse_device_configs(minimal_device_config)

    assert len(devices) == 1
    assert devices[0]["device_class"] == "BaseDevice"
    assert devices[0]["device_class_id"] == 0
    assert devices[0]["config"] == minimal_device_config["devices"]["devices"][0]
    assert devices[0]["parsed_config"].state_feedback.fps == 10.0


@pytest.mark.unit
def test_parse_device_configs_rejects_flat_devices() -> None:
    """Flat stream fields are no longer a device construction contract."""
    config = {
        "devices": [
            {
                "class": "BaseDevice",
                "device_class_id": 0,
                "data_shape": [1],
            }
        ]
    }

    with pytest.raises(ValueError, match="state_feedback"):
        parse_device_configs(config)


@pytest.mark.unit
def test_parse_device_configs_does_not_materialize_legacy_shape() -> None:
    with pytest.raises(ValueError, match="state_feedback"):
        parse_device_configs(
            {
                "devices": [
                    {
                        "class": "BaseDevice",
                        "device_class_id": 0,
                        "data_format": {
                            "shape": [7],
                            "type": ["xyz", "qwqxqyqz"],
                        },
                    },
                ]
            }
        )


@pytest.mark.unit
def test_parse_device_configs_preserves_grouped_robot_contract() -> None:
    devices = parse_device_configs(
        {
            "devices": [
                {
                    "class": "RizonRobot",
                    "device_class_id": 0,
                    "state_feedback": {
                        "fps": 120.0,
                        "buffer_size": 1000,
                        "hardware_latency_ms": 5.0,
                        "format": {
                            "shape": [22],
                            "type": [
                                "j1-7",
                                "dq1-7",
                                "xyz",
                                "qwqxqyqz",
                                "gripper",
                            ],
                            "dtype": "float64",
                        },
                    },
                    "control": {
                        "type": "eef_abs",
                        "source_selection": {
                            "mode": "dual_source",
                            "sources": {
                                "policy": {
                                    "device": "self",
                                    "port": "command",
                                },
                                "teleoperation": {
                                    "device": "Teleoperator",
                                    "device_class_id": 0,
                                    "port": "state",
                                },
                            },
                            "dual_source_selector": {
                                "endpoint": {
                                    "device": "Teleoperator",
                                    "device_class_id": 0,
                                    "port": "control_flag",
                                },
                            },
                        },
                        "buffer_size": 100,
                        "format": {
                            "shape": [8],
                            "type": ["xyz", "qxqyqzqw", "gripper"],
                            "dtype": "float64",
                        },
                        "cartesian_impedance": {
                            "translation_n_m": [10000.0, 10000.0, 10000.0],
                            "rotation_nm_rad": [1000.0, 1000.0, 1000.0],
                        },
                    },
                    "motion_limits": {
                        "velocity": {
                            "joints_rad_s": [0.3] * 7,
                            "tcp_linear_m_s": 0.5,
                            "tcp_angular_rad_s": 0.8,
                        },
                        "acceleration": {
                            "joints_rad_s2": [2.0] * 7,
                            "tcp_linear_m_s2": 5.0,
                            "tcp_angular_rad_s2": 2.0,
                        },
                        "max_tcp_contact_wrench": None,
                    },
                    "gripper": {
                        "backend": "flexiv",
                        "name": "test_gripper",
                    },
                }
            ]
        }
    )

    config = devices[0]["config"]
    assert config["state_feedback"]["format"]["shape"] == [22]
    assert config["state_feedback"]["format"]["dtype"] == "float64"
    assert config["state_feedback"]["fps"] == 120.0
    assert config["control"]["type"] == "eef_abs"
    assert config["control"]["format"]["shape"] == [8]
    assert config["control"]["format"]["dtype"] == "float64"
    source_selection = config["control"]["source_selection"]
    assert source_selection["mode"] == "dual_source"
    assert source_selection["sources"]["teleoperation"]["device"] == "Teleoperator"
    assert source_selection["dual_source_selector"]["endpoint"]["port"] == "control_flag"
    assert not {
        "data_shape",
        "data_dtype",
        "fps",
        "control_type",
        "control_shape",
        "control_dtype",
        "enable",
        "control_shm_name",
        "max_velocity",
        "impedance_parameter",
    } & set(config)


@pytest.mark.unit
def test_parse_device_configs_rejects_unknown_grouped_field() -> None:
    with pytest.raises(ValueError, match="state_feedback contains unknown fields"):
        parse_device_configs(
            {
                "devices": [
                    {
                        "class": "BaseDevice",
                        "device_class_id": 0,
                        "state_feedback": {
                            "fps": 10.0,
                            "typo": 10,
                            "format": {
                                "shape": [1],
                                "type": ["scalar"],
                                "dtype": "float64",
                            },
                        },
                    }
                ]
            }
        )


@pytest.mark.unit
def test_grouped_x1_control_derives_components_from_format() -> None:
    devices = parse_device_configs(
        {
            "devices": [
                {
                    "class": "X1Robot",
                    "device_class_id": 0,
                    "state_feedback": {
                        "fps": 60.0,
                        "buffer_size": 100,
                        "hardware_latency_ms": 0.0,
                        "format": {
                            "shape": [30],
                            "type": [
                                "j1-7",
                                "xyz",
                                "qwqxqyqz",
                                "j1-7",
                                "xyz",
                                "qwqxqyqz",
                                "j1-2",
                            ],
                            "dtype": "float64",
                        },
                    },
                    "control": {
                        "type": "eef_abs",
                        "format": {
                            "shape": [18],
                            "type": [
                                "xyz",
                                "qxqyqzqw",
                                "gripper",
                                "xyz",
                                "qxqyqzqw",
                                "gripper",
                                "j1-2",
                            ],
                            "dtype": "float64",
                        },
                    },
                    "motion_limits": {
                        "velocity": {"tcp_linear_m_s": 0.3},
                    },
                }
            ]
        }
    )

    config = devices[0]["config"]
    assert "utilize_gripper" not in config
    assert "enable_head" not in config
    assert config["motion_limits"]["velocity"]["tcp_linear_m_s"] == 0.3


@pytest.mark.unit
@pytest.mark.parametrize("legacy_field", ["grippers_enabled", "head_enabled"])
def test_grouped_control_rejects_redundant_component_switches(
    legacy_field: str,
) -> None:
    with pytest.raises(ValueError, match=legacy_field):
        parse_device_configs(
            {
                "devices": [
                    {
                        "class": "X1Robot",
                        "device_class_id": 0,
                        "state_feedback": {
                            "fps": 60.0,
                            "buffer_size": 100,
                            "hardware_latency_ms": 0.0,
                            "format": {
                                "shape": [30],
                                "type": [
                                    "j1-7",
                                    "xyz",
                                    "qwqxqyqz",
                                    "j1-7",
                                    "xyz",
                                    "qwqxqyqz",
                                    "j1-2",
                                ],
                                "dtype": "float64",
                            },
                        },
                        "control": {
                            "type": "eef_abs",
                            legacy_field: True,
                            "format": {
                                "shape": [14],
                                "type": [
                                    "xyz",
                                    "qxqyqzqw",
                                    "xyz",
                                    "qxqyqzqw",
                                ],
                                "dtype": "float64",
                            },
                        },
                    }
                ]
            }
        )


@pytest.mark.unit
def test_parse_device_configs_rejects_flat_robot_stream_contract() -> None:
    with pytest.raises(ValueError, match="requires a state_feedback mapping"):
        parse_device_configs(
            {
                "devices": [
                    {
                        "class": "SimRobot",
                        "device_class_id": 0,
                        "data_shape": [14],
                        "control_shape": [7],
                    }
                ]
            }
        )


@pytest.mark.unit
def test_parse_device_configs_returns_empty_for_invalid_devices_section() -> None:
    """Invalid devices sections should fail closed instead of producing partial configs."""
    assert parse_device_configs({"devices": "BaseDevice"}) == []


@pytest.mark.unit
def test_parse_device_configs_treats_string_false_as_disabled() -> None:
    config = {
        "devices": [
            {"class": "BaseDevice", "device_class_id": 0, "enabled": "false"},
            {
                "class": "BaseDevice",
                "device_class_id": 1,
                "enabled": ["false", "false"],
            },
            {
                "class": "BaseDevice",
                "device_class_id": 2,
                "enabled": "true",
                "state_feedback": {
                    "fps": 10.0,
                    "buffer_size": 3,
                    "hardware_latency_ms": 0.0,
                    "format": {
                        "shape": [1],
                        "dtype": "float64",
                        "type": ["linear:1"],
                    },
                },
            },
        ]
    }

    devices = parse_device_configs(config)

    assert [device["device_class_id"] for device in devices] == [2]


@pytest.mark.unit
def test_robot_without_structured_streams_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires a state_feedback mapping"):
        parse_device_configs(
            {
                "devices": [
                    {
                        "class": "RizonRobot",
                        "device_class_id": 0,
                        "enable": [False, False],
                    }
                ]
            }
        )


@pytest.mark.unit
def test_ensure_config_dict_rejects_unsupported_type() -> None:
    """Unsupported config inputs should not be silently accepted."""
    assert ensure_config_dict({"a": 1}) == {"a": 1}
    assert ensure_config_dict([("a", 1)]) is None


@pytest.mark.unit
def test_normalize_hydra_config_is_the_shared_runtime_shape() -> None:
    """Both launchers receive flat devices and manager/runner sections."""
    config = {
        "devices": {"devices": [{"class": "BaseDevice", "device_class_id": 0}]},
        "device_manager": {"device_manager": {"type": "synchronized"}},
        "action_executor": {"action_executor": {"type": "base_policy_runner"}},
    }

    normalized = normalize_hydra_config(config)

    assert normalized["devices"] == [{"class": "BaseDevice", "device_class_id": 0}]
    assert normalized["device_manager"] == {"type": "synchronized"}
    assert normalized["action_executor"] == {"type": "base_policy_runner"}


def _write_config(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _resolve_composed_config(
    config_dir: Path, config_name: str, *, include_hydra: bool = False
) -> tuple[dict, str]:
    hydra_state = HydraConfig.instance()
    previous_hydra_config = hydra_state.cfg
    try:
        with initialize_config_dir(
            config_dir=str(config_dir),
            version_base="1.3",
            job_name="config_parser_test",
        ):
            config = compose(config_name=config_name, return_hydra_config=True)
            runtime_output_dir = str(config_dir / "hydra-output")
            config.hydra.runtime.output_dir = runtime_output_dir
            hydra_state.set_config(config)
            application_keys = [str(key) for key in config.keys() if str(key) != "hydra"]
            application_config = OmegaConf.masked_copy(config, application_keys)
            resolved = flatten_hydra_config(config if include_hydra else application_config)
    finally:
        hydra_state.cfg = previous_hydra_config
    return resolved, runtime_output_dir


@pytest.mark.unit
def test_flatten_hydra_config_unwraps_historical_openpi_config_before_resolution(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    _write_config(
        config_dir / "policy_infer" / "openpi" / "historical.yaml",
        "camera:\n"
        "  left_capture_id: 7\n"
        "camera_id: ${camera.left_capture_id}\n"
        "runtime_output_dir: ${hydra:runtime.output_dir}\n",
    )

    resolved, runtime_output_dir = _resolve_composed_config(
        config_dir, "policy_infer/openpi/historical", include_hydra=True
    )

    assert resolved == {
        "camera": {"left_capture_id": 7},
        "camera_id": 7,
        "runtime_output_dir": runtime_output_dir,
    }


@pytest.mark.unit
def test_flatten_hydra_config_preserves_global_top_level_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    _write_config(
        config_dir / "policy_infer" / "openpi" / "global.yaml",
        "# @package _global_\nglobal_value: preserve\n",
    )

    resolved, _ = _resolve_composed_config(config_dir, "policy_infer/openpi/global")

    assert resolved == {"global_value": "preserve"}


@pytest.mark.unit
def test_flatten_hydra_config_preserves_top_level_openpi_mapping(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    _write_config(config_dir / "single.yaml", "openpi:\n  mode: preserve\n")

    resolved, _ = _resolve_composed_config(config_dir, "single")

    assert resolved["openpi"] == {"mode": "preserve"}


@pytest.mark.unit
def test_flatten_hydra_config_preserves_non_openpi_single_key_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    _write_config(
        config_dir / "other" / "single.yaml",
        "# @package _global_\nopenpi:\n  mode: preserve\n",
    )

    resolved, _ = _resolve_composed_config(config_dir, "other/single")

    assert resolved["openpi"] == {"mode": "preserve"}


@pytest.mark.unit
def test_flatten_hydra_config_requires_initialized_hydra_context() -> None:
    hydra_state = HydraConfig.instance()
    previous_hydra_config = hydra_state.cfg
    hydra_state.cfg = None
    try:
        config = OmegaConf.create({"openpi": {"mode": "preserve"}})
        assert isinstance(config, DictConfig)
        assert flatten_hydra_config(config) == {"openpi": {"mode": "preserve"}}
    finally:
        hydra_state.cfg = previous_hydra_config
