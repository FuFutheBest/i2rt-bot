"""Validated server-owned home targets for whole-episode transitions."""

from __future__ import annotations

import hashlib
import math
import re
import secrets
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
# Device motion is bounded to 30 seconds; reserve margin for command pickup/result IO.
MIN_CARTESIAN_HOME_ACK_TIMEOUT_S = 35.0
_SIDES = ("left", "right")
# Hydra config names commonly contain a namespace separator (``policy_infer/openpi/foo``).
# Keep the value single-rooted and token-like so it cannot become a path.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_SAFE_CONFIG_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,159}$")
_CONFIG_HASH = re.compile(r"^[a-f0-9]{64}$")


class HomeTransitionError(ValueError):
    """A generated home target is incomplete or unsafe to serialize."""


def _finite_joints(value: Any, *, side: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise HomeTransitionError(f"{side} home joints must be a 7D sequence")
    joints = [float(item) for item in value]
    if len(joints) != 7 or not all(math.isfinite(item) for item in joints):
        raise HomeTransitionError(f"{side} home joints must contain 7 finite values")
    return joints


def normalize_home_target(value: Any) -> Optional[dict[str, Any]]:
    """Return a bounded canonical candidate target or ``None``."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise HomeTransitionError("home_target must be an object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise HomeTransitionError("Unsupported home_target schema_version")
    if value.get("source") != "home_pose_candidates":
        raise HomeTransitionError("Unsupported home_target source")

    config_name = value.get("config_name")
    config_hash = value.get("config_hash")
    if (
        not isinstance(config_name, str)
        or _SAFE_CONFIG_NAME.fullmatch(config_name) is None
        or any(part in {"", ".", ".."} for part in config_name.split("/"))
    ):
        raise HomeTransitionError("home_target config_name is invalid")
    if not isinstance(config_hash, str) or _CONFIG_HASH.fullmatch(config_hash) is None:
        raise HomeTransitionError("home_target config_hash is invalid")

    arms = value.get("arms")
    if not isinstance(arms, Mapping):
        raise HomeTransitionError("home_target arms are missing")
    normalized_arms: dict[str, dict[str, Any]] = {}
    for side in _SIDES:
        arm = arms.get(side)
        if not isinstance(arm, Mapping):
            raise HomeTransitionError(f"home_target {side} arm is missing")
        candidate_id = arm.get("candidate_id", arm.get("id"))
        if (
            not isinstance(candidate_id, str)
            or _SAFE_ID.fullmatch(candidate_id) is None
        ):
            raise HomeTransitionError(f"home_target {side} candidate_id is invalid")
        normalized_arms[side] = {
            "candidate_id": candidate_id,
            "joints": _finite_joints(arm.get("joints"), side=side),
            "collision_checked": arm.get("collision_checked") is True,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "home_pose_candidates",
        "config_name": config_name,
        "config_hash": config_hash,
        "arms": normalized_arms,
    }


def select_candidate_home_target(
    state: Any,
    *,
    expected_config: Optional[str],
    selection_key: Optional[str] = None,
    chooser: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]] = secrets.choice,
) -> Optional[dict[str, Any]]:
    """Choose one independently sampled candidate per arm from current state."""
    if not isinstance(state, Mapping):
        return None
    if state.get("status") not in {"ready", "partial"}:
        return None
    config_name = state.get("config_name")
    if not isinstance(config_name, str) or config_name != expected_config:
        return None
    config_hash = state.get("config_hash")
    arms = state.get("arms")
    if not isinstance(config_hash, str) or not isinstance(arms, Mapping):
        return None

    selected: dict[str, Mapping[str, Any]] = {}
    for side in _SIDES:
        arm = arms.get(side)
        candidates = arm.get("candidates") if isinstance(arm, Mapping) else None
        usable = [
            candidate
            for candidate in (candidates or [])
            if isinstance(candidate, Mapping)
        ]
        if not usable:
            return None
        if selection_key is None:
            selected[side] = chooser(usable)
        else:
            digest = hashlib.sha256(
                f"{config_hash}\0{selection_key}\0{side}".encode("utf-8")
            ).digest()
            selected[side] = usable[int.from_bytes(digest, "big") % len(usable)]

    return normalize_home_target(
        {
            "schema_version": SCHEMA_VERSION,
            "source": "home_pose_candidates",
            "config_name": config_name,
            "config_hash": config_hash,
            "arms": {
                side: {
                    "candidate_id": selected[side].get("id"),
                    "joints": selected[side].get("joints"),
                    "collision_checked": selected[side].get("collision_checked") is True,
                }
                for side in _SIDES
            },
        }
    )


def home_result_path(command_path: Path) -> Path:
    """Return the fixed sidecar used to acknowledge one serialized home move."""
    path = Path(command_path)
    return path.with_name(f"{path.name}.result.json")


__all__ = [
    "HomeTransitionError",
    "MIN_CARTESIAN_HOME_ACK_TIMEOUT_S",
    "home_result_path",
    "normalize_home_target",
    "select_candidate_home_target",
]
