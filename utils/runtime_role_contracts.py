"""Code-owned required runtime roles for native workflow configurations."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

FASTWAM_REQUIRED_RUNTIME_ROLES = (
    "launcher",
    "planner",
    "device",
    "policy",
    "fastwam-server",
    "recording",
)

_FASTWAM_CONFIG_NAMES = (
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_memory",
    "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam",
)

REQUIRED_RUNTIME_ROLES_BY_CONFIG: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        config_name: FASTWAM_REQUIRED_RUNTIME_ROLES
        for config_name in _FASTWAM_CONFIG_NAMES
    }
)


def required_runtime_roles(config_name: str | None) -> tuple[str, ...]:
    """Return roles that must be calibrated before a config is rollout-ready."""
    if config_name is None:
        return ()
    return REQUIRED_RUNTIME_ROLES_BY_CONFIG.get(str(config_name), ())


__all__ = [
    "FASTWAM_REQUIRED_RUNTIME_ROLES",
    "REQUIRED_RUNTIME_ROLES_BY_CONFIG",
    "required_runtime_roles",
]
