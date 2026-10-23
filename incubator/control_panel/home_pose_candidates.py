"""Persistent, hardware-free orchestration for sampled robot home poses."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .runtime_config import extract_robot_home, extract_robot_tools, normalize_config_name


SCHEMA_VERSION = 1
DEFAULT_PRESET = "conservative"
DEFAULT_COUNT = 6
DEFAULT_SEED = 0
ALLOWED_PRESETS = frozenset({"conservative", "default", "wide"})
MAX_CANDIDATES_PER_ARM = 12
MAX_SEED = (1 << 32) - 1
MAX_CACHE_BYTES = 256 * 1024
_CACHE_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_SIDES = ("left", "right")
_MODELS = {"left": "Rizon4", "right": "Rizon4R"}


class HomePoseCandidateError(RuntimeError):
    """Base exception for candidate orchestration failures."""


class HomePoseCandidateRequestError(ValueError):
    """Raised when a regeneration request contains unsafe parameters."""


class HomePoseCandidateStorageError(HomePoseCandidateError):
    """Raised when a cache record cannot be written safely."""


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _finite_vector(value: Any, size: int) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    result = [_finite_number(item) for item in value]
    if any(item is None for item in result):
        return None
    return [float(item) for item in result if item is not None]


def _safe_error(error: Any) -> str:
    message = " ".join(str(error).split())
    return message[:500] or type(error).__name__


def _default_cache_root() -> Path:
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        base = Path(xdg_cache).expanduser()
    else:
        base = Path(tempfile.gettempdir())
    return base / "superinference-control-panel" / "home_pose_candidates"


def normalize_generation_options(payload: Any = None) -> Dict[str, Any]:
    """Validate the complete public regeneration parameter allowlist."""
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise HomePoseCandidateRequestError("JSON body must be an object")
    unknown = sorted(set(str(key) for key in payload) - {"preset", "count", "seed"})
    if unknown:
        raise HomePoseCandidateRequestError(
            "Unknown regeneration fields: " + ", ".join(unknown)
        )

    preset = payload.get("preset", DEFAULT_PRESET)
    if not isinstance(preset, str) or preset not in ALLOWED_PRESETS:
        raise HomePoseCandidateRequestError(
            "preset must be one of: " + ", ".join(sorted(ALLOWED_PRESETS))
        )
    count = payload.get("count", DEFAULT_COUNT)
    if isinstance(count, bool) or not isinstance(count, int):
        raise HomePoseCandidateRequestError("count must be an integer")
    if not 1 <= count <= MAX_CANDIDATES_PER_ARM:
        raise HomePoseCandidateRequestError(
            f"count must be between 1 and {MAX_CANDIDATES_PER_ARM}"
        )
    seed = payload.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise HomePoseCandidateRequestError("seed must be an integer")
    if not 0 <= seed <= MAX_SEED:
        raise HomePoseCandidateRequestError(f"seed must be between 0 and {MAX_SEED}")
    return {"preset": preset, "count": count, "seed": seed}


class HomePoseCandidateStore:
    """A bounded, content-addressed JSON cache with atomic replacement."""

    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        max_bytes: int = MAX_CACHE_BYTES,
    ) -> None:
        self.root = (root or _default_cache_root()).expanduser().resolve()
        self.max_bytes = int(max_bytes)
        self._lock = threading.RLock()

    def _path_for(self, key: str) -> Path:
        if not isinstance(key, str) or not _CACHE_KEY_RE.fullmatch(key):
            raise HomePoseCandidateStorageError("Invalid candidate cache key")
        path = self.root / f"{key}.json"
        if path.parent.resolve() != self.root:
            raise HomePoseCandidateStorageError("Candidate cache path escaped root")
        return path

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path_for(key)
        with self._lock:
            try:
                size = path.stat().st_size
                if size <= 0 or size > self.max_bytes:
                    return None
                raw = path.read_bytes()
            except (FileNotFoundError, OSError):
                return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != SCHEMA_VERSION:
            return None
        if payload.get("key") != key:
            return None
        if payload.get("status") not in {"ready", "partial"}:
            return None
        if not _cached_arms_are_safe(payload.get("arms")):
            return None
        return payload

    def write(self, key: str, payload: Mapping[str, Any]) -> Path:
        path = self._path_for(key)
        record = copy.deepcopy(dict(payload))
        record["schema_version"] = SCHEMA_VERSION
        record["key"] = key
        try:
            encoded = _stable_json_bytes(record)
        except (TypeError, ValueError) as exc:
            raise HomePoseCandidateStorageError(
                "Candidate cache record is not JSON-safe"
            ) from exc
        if len(encoded) > self.max_bytes:
            raise HomePoseCandidateStorageError(
                f"Candidate cache record exceeds {self.max_bytes} bytes"
            )

        with self._lock:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                os.chmod(self.root, 0o700)
            except OSError:
                pass
            temporary = self.root / f".{key}.{uuid.uuid4().hex}.tmp"
            descriptor: Optional[int] = None
            try:
                descriptor = os.open(
                    temporary,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = None
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                os.chmod(path, 0o600)
                try:
                    directory_fd = os.open(self.root, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError:
                    pass
            except OSError as exc:
                if descriptor is not None:
                    os.close(descriptor)
                try:
                    temporary.unlink()
                except OSError:
                    pass
                raise HomePoseCandidateStorageError(
                    "Failed to atomically write candidate cache"
                ) from exc
        return path


def _cached_arms_are_safe(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for side in _SIDES:
        arm = value.get(side)
        if not isinstance(arm, Mapping) or arm.get("model") != _MODELS[side]:
            return False
        candidates = arm.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) > MAX_CANDIDATES_PER_ARM:
            return False
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                return False
            if _finite_vector(candidate.get("joints"), 7) is None:
                return False
    return True


def _tool_matrix(tool: Mapping[str, Any]) -> Optional[list[list[float]]]:
    transform = tool.get("transform")
    if not isinstance(transform, Mapping):
        return None
    translation = _finite_vector(transform.get("translation"), 3)
    quaternion = _finite_vector(transform.get("quaternion"), 4)
    if translation is None or quaternion is None:
        return None
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        return None
    w, x, y, z = (component / norm for component in (w, x, y, z))
    rotation = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], translation[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], translation[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _default_urdf_paths() -> Dict[str, Path]:
    root = Path(__file__).resolve().parent / "static" / "assets" / "flexiv"
    return {
        "left": root / "flexiv_Rizon4_kinematics.urdf",
        "right": root / "flexiv_Rizon4R_kinematics.urdf",
    }


def _sampler_source_hash() -> Optional[str]:
    return _sha256_file(Path(__file__).resolve().with_name("home_pose_sampler.py"))


def build_generation_context(
    config_name: str,
    effective_config: Mapping[str, Any],
    options: Mapping[str, Any],
    *,
    urdf_paths: Optional[Mapping[str, Path]] = None,
    sampler_version: str = "home-pose-sampler-v1",
    sampler_source_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a redacted, kinematics-only cache context."""
    name = normalize_config_name(config_name)
    normalized_options = normalize_generation_options(options)
    homes = extract_robot_home(effective_config)
    tools = extract_robot_tools(effective_config)
    paths = {side: Path(path).resolve() for side, path in (urdf_paths or _default_urdf_paths()).items()}
    arms: Dict[str, Dict[str, Any]] = {}
    for side in _SIDES:
        home = _finite_vector(homes.get(f"joints_{side}"), 7)
        tool = tools.get(side) if isinstance(tools.get(side), Mapping) else {}
        path = paths.get(side)
        urdf_hash = _sha256_file(path) if path is not None else None
        arms[side] = {
            "model": _MODELS[side],
            "home_joints": home,
            "tool": {
                "name": tool.get("name"),
                "status": tool.get("status"),
                "transform": copy.deepcopy(tool.get("transform")),
            },
            "tool_transform": _tool_matrix(tool),
            "urdf_path": path,
            "urdf_sha256": urdf_hash,
        }

    fingerprint = {
        "schema_version": SCHEMA_VERSION,
        "config_name": name,
        "sampler": {
            "version": str(sampler_version),
            "source_sha256": sampler_source_hash,
        },
        "arms": {
            side: {
                "model": arms[side]["model"],
                "home_joints": arms[side]["home_joints"],
                "tool": arms[side]["tool"],
                "urdf_sha256": arms[side]["urdf_sha256"],
            }
            for side in _SIDES
        },
    }
    config_hash = _sha256_bytes(_stable_json_bytes(fingerprint))
    cache_key = _sha256_bytes(
        _stable_json_bytes(
            {
                "config_hash": config_hash,
                "params": normalized_options,
            }
        )
    )
    return {
        "config_name": name,
        "config_hash": config_hash,
        "key": cache_key,
        "params": normalized_options,
        "arms": arms,
        "sampler": fingerprint["sampler"],
    }


def _default_sampler(**kwargs: Any) -> Any:
    from .home_pose_sampler import sample_home_poses

    return sample_home_poses(**kwargs)


def _default_parameter_factory(preset: str, count: int) -> Any:
    from .home_pose_sampler import sampling_preset

    parameters = sampling_preset(preset)
    for field in ("candidate_count", "count", "num_candidates"):
        if hasattr(parameters, field):
            return dataclasses.replace(parameters, **{field: count})
    raise HomePoseCandidateError(
        "SamplingParameters does not expose a candidate count field"
    )


def _mapping_result(result: Any) -> Dict[str, Any]:
    if hasattr(result, "to_dict") and callable(result.to_dict):
        result = result.to_dict()
    if not isinstance(result, Mapping):
        raise HomePoseCandidateError("Sampler returned a non-object result")
    return dict(result)


def _pose_payload(value: Any) -> Optional[Dict[str, list[float]]]:
    if isinstance(value, (list, tuple)):
        pose = _finite_vector(value, 7)
        if pose is None:
            return None
        return {"translation": pose[:3], "quaternion_wxyz": pose[3:]}
    if not isinstance(value, Mapping):
        return None
    translation = None
    for key in ("translation", "position", "position_m", "xyz"):
        translation = _finite_vector(value.get(key), 3)
        if translation is not None:
            break
    quaternion = _finite_vector(value.get("quaternion_wxyz"), 4)
    if quaternion is None:
        xyzw = _finite_vector(value.get("quaternion_xyzw"), 4)
        if xyzw is not None:
            quaternion = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    if translation is None or quaternion is None:
        return None
    return {"translation": translation, "quaternion_wxyz": quaternion}


def _normalize_candidate(
    side: str,
    index: int,
    value: Any,
    *,
    config_hash: str,
    generated_at: str,
    collision_checked: bool,
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    joints = _finite_vector(value.get("joints", value.get("joint_positions")), 7)
    if joints is None:
        return None
    target_pose = _pose_payload(
        value.get("target_pose", value.get("requested_pose", value.get("sampled_pose")))
    )
    fk_pose = _pose_payload(
        value.get(
            "fk_pose",
            value.get("target_frame_pose", value.get("tcp_pose")),
        )
    )
    residual_value = value.get("residual")
    residual = residual_value if isinstance(residual_value, Mapping) else {}
    position_residual = _finite_number(
        residual.get("position_m", value.get("position_residual_m"))
    )
    orientation_residual = _finite_number(
        residual.get("orientation_rad", value.get("orientation_residual_rad"))
    )
    joint_distance = _finite_number(
        value.get("joint_distance", value.get("joint_distance_from_home_rad"))
    )
    return {
        "id": f"{side}-{index + 1:02d}",
        "joints": joints,
        "target_pose": target_pose,
        "fk_pose": fk_pose,
        "residual": {
            "position_m": position_residual,
            "orientation_rad": orientation_residual,
        },
        "joint_distance": joint_distance,
        "collision_checked": bool(value.get("collision_checked", collision_checked)),
        "source": "offline_sampler",
        "config_hash": config_hash,
        "generated_at": generated_at,
    }


class HomePoseCandidateCoordinator:
    """Schedule sampled home poses without blocking Flask request threads."""

    def __init__(
        self,
        *,
        store: Optional[HomePoseCandidateStore] = None,
        sampler: Optional[Callable[..., Any]] = None,
        parameter_factory: Optional[Callable[[str, int], Any]] = None,
        urdf_paths: Optional[Mapping[str, Path]] = None,
        sampler_version: str = "home-pose-sampler-v1",
        sampler_source_hash: Optional[str] = None,
    ) -> None:
        self.store = store or HomePoseCandidateStore()
        self._sampler = sampler or _default_sampler
        self._parameter_factory = parameter_factory or _default_parameter_factory
        self._urdf_paths = dict(urdf_paths or _default_urdf_paths())
        self._sampler_version = str(sampler_version)
        self._sampler_source_hash = (
            sampler_source_hash
            if sampler_source_hash is not None
            else (_sampler_source_hash() if sampler is None else None)
        )
        self._lock = threading.RLock()
        self._generation = 0
        self._cancel_event: Optional[threading.Event] = None
        self._threads: set[threading.Thread] = set()
        self._closed = False
        self._state: Dict[str, Any] = self._empty_state("no_config")

    @staticmethod
    def _empty_state(status: str) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "key": None,
            "config_name": None,
            "config_hash": None,
            "params": normalize_generation_options(),
            "generated_at": None,
            "error": None,
            "arms": {
                side: {
                    "model": _MODELS[side],
                    "status": "unavailable",
                    "target_frame": "flange",
                    "candidates": [],
                    "error": None,
                }
                for side in _SIDES
            },
        }

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def clear(self) -> Dict[str, Any]:
        with self._lock:
            self._cancel_locked()
            self._state = self._empty_state("no_config")
            return copy.deepcopy(self._state)

    def schedule(
        self,
        config_name: str,
        effective_config: Mapping[str, Any],
        *,
        options: Any = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        context = build_generation_context(
            config_name,
            effective_config,
            normalize_generation_options(options),
            urdf_paths=self._urdf_paths,
            sampler_version=self._sampler_version,
            sampler_source_hash=self._sampler_source_hash,
        )
        available_sides = [
            side
            for side in _SIDES
            if context["arms"][side]["home_joints"] is not None
            and context["arms"][side]["urdf_sha256"] is not None
        ]
        with self._lock:
            if self._closed:
                raise HomePoseCandidateError("Candidate coordinator is shut down")
            if (
                not force
                and self._state.get("key") == context["key"]
                and self._state.get("status")
                in {"generating", "ready", "partial"}
            ):
                return copy.deepcopy(self._state)
            self._cancel_locked()
            self._generation += 1
            generation = self._generation

            if not available_sides:
                state = self._empty_state("unavailable")
                state.update(
                    {
                        "key": context["key"],
                        "config_name": context["config_name"],
                        "config_hash": context["config_hash"],
                        "params": context["params"],
                        "error": "No valid config home pose and URDF pair is available",
                    }
                )
                self._state = state
                return copy.deepcopy(state)

            if not force:
                cached = self.store.load(context["key"])
                if cached is not None:
                    self._state = cached
                    return copy.deepcopy(cached)

            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            state = self._empty_state("generating")
            state.update(
                {
                    "key": context["key"],
                    "config_name": context["config_name"],
                    "config_hash": context["config_hash"],
                    "params": context["params"],
                }
            )
            for side in _SIDES:
                state["arms"][side]["status"] = (
                    "generating" if side in available_sides else "unavailable"
                )
                if side not in available_sides:
                    state["arms"][side]["error"] = (
                        "Config home pose or vendored URDF is unavailable"
                    )
            self._state = state
            thread = threading.Thread(
                target=self._run_generation,
                args=(generation, context, cancel_event, available_sides),
                name=f"HomePoseCandidates-{generation}",
                daemon=True,
            )
            self._threads.add(thread)
            thread.start()
            return copy.deepcopy(state)

    def _cancel_locked(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cancel_event = None

    def _run_generation(
        self,
        generation: int,
        context: Mapping[str, Any],
        cancel_event: threading.Event,
        available_sides: list[str],
    ) -> None:
        generated_at = _utc_timestamp()
        arms: Dict[str, Dict[str, Any]] = {}
        for side in _SIDES:
            arm_context = context["arms"][side]
            target_frame = (
                "tool" if arm_context["tool_transform"] is not None else "flange"
            )
            if side not in available_sides:
                arms[side] = {
                    "model": _MODELS[side],
                    "status": "unavailable",
                    "target_frame": target_frame,
                    "candidates": [],
                    "error": "Config home pose or vendored URDF is unavailable",
                }
                continue
            try:
                parameters = self._parameter_factory(
                    context["params"]["preset"], context["params"]["count"]
                )
                result = _mapping_result(
                    self._sampler(
                        side=side,
                        urdf_path=arm_context["urdf_path"],
                        home_joints=list(arm_context["home_joints"]),
                        parameters=parameters,
                        seed=context["params"]["seed"],
                        tool_transform=copy.deepcopy(arm_context["tool_transform"]),
                        should_cancel=cancel_event.is_set,
                    )
                )
                raw_candidates = result.get("candidates", [])
                if not isinstance(raw_candidates, list):
                    raw_candidates = []
                collision = result.get("collision_check")
                collision_payload = (
                    copy.deepcopy(dict(collision))
                    if isinstance(collision, Mapping)
                    else {"performed": bool(result.get("collision_checked", False))}
                )
                collision_checked = collision_payload.get("performed") is True
                candidates = []
                for item in raw_candidates[: context["params"]["count"]]:
                    normalized = _normalize_candidate(
                        side,
                        len(candidates),
                        item,
                        config_hash=context["config_hash"],
                        generated_at=generated_at,
                        collision_checked=collision_checked,
                    )
                    if normalized is not None:
                        candidates.append(normalized)
                if candidates:
                    arm_status = "ready"
                    error = None
                else:
                    arm_status = "error"
                    error = _safe_error(result.get("error") or "Sampler returned no valid candidates")
                arms[side] = {
                    "model": _MODELS[side],
                    "status": arm_status,
                    "target_frame": str(result.get("target_frame") or target_frame),
                    "tool_status": result.get("tool_status"),
                    "sampling_status": result.get("status"),
                    "attempts": result.get("attempts"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "failure_counts": copy.deepcopy(result.get("failure_counts", {})),
                    "collision_check": collision_payload,
                    "candidates": candidates,
                    "error": error,
                }
            except Exception as exc:  # Background errors are reported through state.
                arms[side] = {
                    "model": _MODELS[side],
                    "status": "error",
                    "target_frame": target_frame,
                    "candidates": [],
                    "error": _safe_error(exc),
                }

        ready_count = sum(arm["status"] == "ready" for arm in arms.values())
        if ready_count == len(_SIDES):
            status = "ready"
        elif ready_count:
            status = "partial"
        else:
            status = "error"
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "key": context["key"],
            "config_name": context["config_name"],
            "config_hash": context["config_hash"],
            "params": copy.deepcopy(context["params"]),
            "generated_at": generated_at,
            "error": None if ready_count else "No arm produced a valid candidate",
            "arms": arms,
        }
        if status in {"ready", "partial"}:
            try:
                self.store.write(context["key"], payload)
            except HomePoseCandidateStorageError as exc:
                payload["status"] = "error"
                payload["error"] = _safe_error(exc)

        with self._lock:
            current = threading.current_thread()
            self._threads.discard(current)
            if (
                self._closed
                or cancel_event.is_set()
                or generation != self._generation
                or self._state.get("key") != context["key"]
            ):
                return
            self._cancel_event = None
            self._state = payload

    def shutdown(self, timeout: float = 1.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancel_locked()
            threads = list(self._threads)
        deadline = max(0.0, float(timeout))
        for thread in threads:
            thread.join(timeout=deadline)


__all__ = [
    "ALLOWED_PRESETS",
    "DEFAULT_COUNT",
    "DEFAULT_PRESET",
    "DEFAULT_SEED",
    "HomePoseCandidateCoordinator",
    "HomePoseCandidateError",
    "HomePoseCandidateRequestError",
    "HomePoseCandidateStorageError",
    "HomePoseCandidateStore",
    "MAX_CACHE_BYTES",
    "MAX_CANDIDATES_PER_ARM",
    "MAX_SEED",
    "SCHEMA_VERSION",
    "build_generation_context",
    "normalize_generation_options",
]
