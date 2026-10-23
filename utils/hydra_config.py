"""Side-effect-free Hydra composition normalization shared by launchers.

This module intentionally does not import the repository logger or hardware
registries.  It is safe for read-only config inspection from the Control Panel.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import threading
from typing import Any, Dict, Mapping, Union, cast

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import get_gh_backup, restore_gh_from_backup
from omegaconf import DictConfig, OmegaConf

ConfigInput = Union[Dict[str, Any], DictConfig]
DEFAULT_HYDRA_CONFIG_NAME = "config"
_COMPOSE_LOCK = threading.RLock()


def compose_hydra_application_config(
    config_name: str,
    *,
    config_dir: Path | str,
    job_name: str = "superinference",
) -> Dict[str, Any]:
    """Compose one native Hydra config without a launcher or UI dependency.

    Callers own config-name authorization and config discovery.  This helper
    only provides the shared, process-safe composition semantics needed by
    runtime contracts that must inspect the resolved application mapping.
    """
    if not isinstance(config_name, str) or not config_name.strip():
        raise ValueError("config_name must be a non-empty string")
    name = config_name.strip()
    parts = Path(name).parts
    if Path(name).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("config_name must be a relative Hydra config name")
    if not isinstance(job_name, str) or not job_name.strip():
        raise ValueError("job_name must be a non-empty string")

    root = Path(config_dir).resolve()
    # HydraConfig and GlobalHydra are process-global.  Serialize callers of
    # this helper, then install an isolated GlobalHydra while composition runs
    # so an enclosing @hydra.main context remains intact for its owner.
    with _COMPOSE_LOCK:
        hydra_state = HydraConfig.instance()
        previous_hydra_config = hydra_state.cfg
        global_hydra_backup = get_gh_backup()
        caller_global_hydra = GlobalHydra.instance()
        isolated_global_hydra = deepcopy(caller_global_hydra)
        isolated_global_hydra.clear()
        GlobalHydra.set_instance(isolated_global_hydra)
        try:
            with initialize_config_dir(
                config_dir=str(root),
                version_base="1.3",
                job_name=job_name.strip(),
            ):
                # The typed Hydra node makes `${hydra:runtime.*}` resolvable,
                # while the returned mapping intentionally excludes Hydra's
                # incomplete sweep-only internal configuration.
                cfg = compose(config_name=name, return_hydra_config=True)
                hydra_state.set_config(cfg)
                application_keys = [str(key) for key in cfg.keys() if str(key) != "hydra"]
                application_cfg = OmegaConf.masked_copy(cfg, application_keys)
                return normalize_hydra_config(application_cfg)
        finally:
            hydra_state.cfg = previous_hydra_config
            if global_hydra_backup is None:
                restore_gh_from_backup(None)
            else:
                GlobalHydra.set_instance(caller_global_hydra)


def flatten_hydra_config(config: DictConfig) -> Dict[str, Any]:
    """Resolve a Hydra config, unwrapping historical OpenPI packages."""
    config_to_resolve: DictConfig = config
    application_keys = [str(key) for key in config.keys() if str(key) != "hydra"]
    config_name: object | None = None
    if HydraConfig.initialized():
        config_name = HydraConfig.get().job.config_name

    if isinstance(config_name, str) and config_name.startswith("policy_infer/openpi/"):
        openpi_package: Mapping[str, Any] | None = None
        if application_keys == ["openpi"] and isinstance(config.get("openpi"), Mapping):
            openpi_package = config["openpi"]
        elif (
            application_keys == ["policy_infer"]
            and isinstance(config.get("policy_infer"), Mapping)
            and isinstance(config["policy_infer"].get("openpi"), Mapping)
        ):
            openpi_package = config["policy_infer"]["openpi"]

        if openpi_package is not None:
            # Recreate the nested config node at the application root before
            # resolving Hydra interpolations.
            config_to_resolve = OmegaConf.create(
                OmegaConf.to_container(openpi_package, resolve=False)
            )

    return cast(Dict[str, Any], OmegaConf.to_container(config_to_resolve, resolve=True))


def normalize_hydra_config(config: ConfigInput) -> Dict[str, Any]:
    """Return the application mapping shared by ``device_starter`` and panel."""
    if isinstance(config, DictConfig):
        application_keys = [str(key) for key in config.keys() if str(key) != "hydra"]
        application_config = OmegaConf.masked_copy(config, application_keys)
        normalized = flatten_hydra_config(application_config)
    elif isinstance(config, Mapping):
        normalized = cast(
            Dict[str, Any],
            OmegaConf.to_container(OmegaConf.create(dict(config)), resolve=True),
        )
    else:
        raise TypeError(f"configuration must be a mapping, got {type(config).__name__}")

    devices = normalized.get("devices")
    if isinstance(devices, dict):
        normalized["devices"] = devices.get("devices", [])

    for section_name in ("device_manager", "action_executor"):
        section = normalized.get(section_name)
        if (
            isinstance(section, dict)
            and "type" not in section
            and isinstance(section.get(section_name), dict)
        ):
            normalized[section_name] = section[section_name]

    return normalized
