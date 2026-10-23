#!/usr/bin/env python3
"""
Configuration Parser - Generic device configuration parsing utilities.

This module provides centralized configuration parsing logic that can be used
by both device_starter.py and device_manager.py to ensure consistency.

Author: Jun Lv
"""

from typing import Any, Dict, List, Mapping, Optional, cast

import yaml
from omegaconf import DictConfig, OmegaConf

from devices import DEVICE_CLASSES
from utils.device_config import DeviceConfigError, parse_device_config
from utils.device_identity import validate_device_class_ids
from utils.inference_options import as_bool
from utils.logger_config import logger
from utils.hydra_config import (
    ConfigInput,
    DEFAULT_HYDRA_CONFIG_NAME,
    flatten_hydra_config,
    normalize_hydra_config,
)

__all__ = [
    "ConfigInput",
    "DEFAULT_HYDRA_CONFIG_NAME",
    "ensure_config_dict",
    "flatten_hydra_config",
    "load_config",
    "normalize_hydra_config",
    "parse_device_configs",
    "parse_device_configs_with_fields",
]


def ensure_config_dict(config: ConfigInput) -> Dict[str, Any]:
    """Convert Hydra DictConfig or plain mapping into a standard dictionary."""
    if isinstance(config, DictConfig):
        return cast(Dict[str, Any], OmegaConf.to_container(config, resolve=True))
    if isinstance(config, Mapping):
        return dict(config)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def parse_device_configs(config: ConfigInput) -> List[Dict[str, Any]]:
    """
    Parse device configurations and create device dictionaries.

    This is a generic parser that works with any device class defined in the config.
    The device class is specified directly in the config (e.g., 'BaseDevice', 'SimRobot').

    Args:
        config: Configuration dictionary loaded from YAML

    Returns:
        List of device dictionaries with standard format
    """
    config_dict = ensure_config_dict(config)

    devices: List[Dict[str, Any]] = []
    device_section = config_dict.get('devices', [])

    if isinstance(device_section, dict):
        device_list = device_section.get('devices', [])
    else:
        device_list = device_section

    if not isinstance(device_list, list):
        logger.error("Device configuration is not a list; received type %s", type(device_list))
        return devices

    validate_device_class_ids(device_list)
    logger.info(f"Found {len(device_list)} devices in configuration")

    # Parse all devices - each device has its class specified
    for i, device_config in enumerate(device_list):
        logger.info(f"Processing device {i}: {device_config}")
        enabled = device_config.get('enabled', True)
        if not as_bool(enabled) or (
            isinstance(enabled, (list, tuple))
            and not any(as_bool(item) for item in enabled)
        ):
            logger.info(
                f"Skipping disabled device: {device_config['class']}:{device_config['device_class_id']}"
            )
            continue

        device_class = str(device_config.get("class", ""))
        try:
            parsed_device_config = parse_device_config(
                device_config,
                device_kind=DEVICE_CLASSES.kind(device_class),
            )
        except (DeviceConfigError, KeyError, ValueError) as exc:
            logger.error(
                "Invalid device config for %s:%s: %s",
                device_config.get("class"),
                device_config.get("device_class_id"),
                exc,
            )
            raise

        # Create standard device info dictionary
        device_info = {
            "device_class": parsed_device_config.device_class,
            "device_class_id": parsed_device_config.device_class_id,
            "config": parsed_device_config.raw,
            "parsed_config": parsed_device_config,
            # Additional fields can be added by the caller as needed
        }

        devices.append(device_info)
        logger.info(
            "Added device: %s:%s",
            parsed_device_config.device_class,
            parsed_device_config.device_class_id,
        )

    logger.info(f"Total devices parsed: {len(devices)}")
    return devices


def parse_device_configs_with_fields(config: ConfigInput, additional_fields: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Parse device configurations with additional fields.

    Args:
        config: Configuration dictionary loaded from YAML
        additional_fields: Dictionary of additional fields to add to each device info

    Returns:
        List of device dictionaries with standard format plus additional fields
    """
    devices = parse_device_configs(config)

    if additional_fields:
        for device in devices:
            device.update(additional_fields)

    return devices
