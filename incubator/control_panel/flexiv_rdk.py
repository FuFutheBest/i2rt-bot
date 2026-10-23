"""Firmware-compatible Flexiv RDK discovery and isolated connection probes."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence


RDK_VERSION_ENV = "SUPERINFERENCE_FLEXIV_RDK_VERSION"
RDK_ROOT_ENV = "SUPERINFERENCE_FLEXIV_RDK_ROOT"
RDK_SITE_ENV = "SUPERINFERENCE_FLEXIV_RDK_SITE"
RDK_AVAILABLE_ENV = "SUPERINFERENCE_FLEXIV_RDK_VERSIONS"
RDK_PROBE_TIMEOUT_ENV = "SUPERINFERENCE_FLEXIV_RDK_PROBE_TIMEOUT"

# The adapter is reviewed for the 1.7-1.9 API family. RDK 1.6 lacks required
# calls and 2.x must not be tried against real hardware before an API audit.
SUPPORTED_RDK_VERSIONS = (
    "1.9.2",
    "1.9.1",
    "1.9.0",
    "1.8.1",
    "1.8.0",
    "1.7.0",
)

WORKER_RDK_VERSION_KEY = "__superinference_flexiv_rdk_version"
WORKER_RDK_SITE_KEY = "__superinference_flexiv_rdk_site"

_RDK_VERSION_RE = re.compile(r"^1\.(?:7|8|9)\.\d+$")
_LEGACY_RELEASE_DIRECTORIES = {
    "1.7.0": "elements-3.9",
    "1.8.1": "elements-3.10",
    "1.9.2": "elements-3.11",
}


class FlexivRDKError(RuntimeError):
    """Base error for RDK inventory or probe failures."""


class FlexivRDKProbeError(FlexivRDKError):
    """No packaged RDK could establish a read-only robot connection."""

    def __init__(self, message: str, attempts: Sequence["FlexivRDKProbeAttempt"]):
        super().__init__(message)
        self.attempts = tuple(attempts)


@dataclass(frozen=True)
class FlexivRDKCandidate:
    version: str
    site_packages: Optional[Path]
    source: str


@dataclass(frozen=True)
class FlexivRDKProbeAttempt:
    version: str
    ok: bool
    reason: Optional[str]
    duration_seconds: float
    timed_out: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "ok": self.ok,
            "reason": self.reason,
            "duration_seconds": round(self.duration_seconds, 3),
            "timed_out": self.timed_out,
        }


@dataclass(frozen=True)
class FlexivRDKSelection:
    candidate: FlexivRDKCandidate
    attempts: tuple[FlexivRDKProbeAttempt, ...]
    cache_hit: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "version": self.candidate.version,
            "source": self.candidate.source,
            "cache_hit": self.cache_hit,
            "attempts": [attempt.public_dict() for attempt in self.attempts],
        }


def normalize_rdk_version(value: object) -> str:
    version = str(value or "").strip()
    if not _RDK_VERSION_RE.fullmatch(version) or version not in SUPPORTED_RDK_VERSIONS:
        raise FlexivRDKError(
            "Flexiv RDK version must be one of: "
            + ", ".join(SUPPORTED_RDK_VERSIONS)
        )
    return version


def _candidate_site(root: Path, version: str) -> Optional[Path]:
    current = root / f"rdk-{version}" / "site-packages"
    if current.is_dir():
        return current.resolve()
    legacy_name = _LEGACY_RELEASE_DIRECTORIES.get(version)
    if legacy_name:
        legacy = root / legacy_name / "site-packages"
        if legacy.is_dir():
            return legacy.resolve()
    return None


def discover_flexiv_rdk_candidates(
    *,
    explicit_version: Optional[str] = None,
    environment: Optional[Mapping[str, str]] = None,
) -> list[FlexivRDKCandidate]:
    """Return packaged candidates without importing any binary RDK module."""
    env = os.environ if environment is None else environment
    requested = explicit_version or env.get(RDK_VERSION_ENV)
    normalized_requested = normalize_rdk_version(requested) if requested else None
    versions = (
        (normalized_requested,) if normalized_requested else SUPPORTED_RDK_VERSIONS
    )

    root_value = str(env.get(RDK_ROOT_ENV, "")).strip()
    if root_value:
        root = Path(root_value).expanduser().resolve(strict=False)
        candidates = [
            FlexivRDKCandidate(version, site, "packaged-release")
            for version in versions
            if (site := _candidate_site(root, version)) is not None
        ]
        if candidates:
            return candidates
        if normalized_requested:
            raise FlexivRDKError(
                f"Requested Flexiv RDK {normalized_requested} is not bundled under {root}"
            )
        raise FlexivRDKError(f"No supported Flexiv RDK profiles found under {root}")

    try:
        installed = metadata.version("flexivrdk")
    except metadata.PackageNotFoundError as exc:
        raise FlexivRDKError(
            "No packaged Flexiv RDK inventory or source-installed flexivrdk was found"
        ) from exc
    installed = normalize_rdk_version(installed)
    if normalized_requested and installed != normalized_requested:
        raise FlexivRDKError(
            f"Requested Flexiv RDK {normalized_requested}, but source environment has {installed}"
        )
    return [FlexivRDKCandidate(installed, None, "source-environment")]


def _is_packaged_rdk_path(value: str) -> bool:
    normalized = value.replace("\\", "/").rstrip("/")
    return "/runtime/flexivrdk/" in normalized and normalized.endswith(
        "/site-packages"
    )


def _candidate_environment(
    candidate: FlexivRDKCandidate,
    base_environment: Mapping[str, str],
) -> dict[str, str]:
    environment = dict(base_environment)
    environment["PYTHONNOUSERSITE"] = "1"
    environment[RDK_VERSION_ENV] = candidate.version
    existing = [
        item
        for item in environment.get("PYTHONPATH", "").split(os.pathsep)
        if item and not _is_packaged_rdk_path(item)
    ]
    if candidate.site_packages is not None:
        site = str(candidate.site_packages)
        existing.insert(0, site)
        environment[RDK_SITE_ENV] = site
    else:
        environment.pop(RDK_SITE_ENV, None)
    environment["PYTHONPATH"] = os.pathsep.join(existing)
    return environment


def activate_flexiv_rdk_worker(device_config: Mapping[str, Any]) -> None:
    """Select one RDK path before the spawned worker imports ``devices``."""
    raw_version = device_config.get(WORKER_RDK_VERSION_KEY)
    if raw_version is None:
        return
    if "flexivrdk" in sys.modules:
        raise FlexivRDKError("flexivrdk was imported before worker RDK activation")
    version = normalize_rdk_version(raw_version)
    raw_site = device_config.get(WORKER_RDK_SITE_KEY)
    site = Path(str(raw_site)).expanduser().resolve() if raw_site else None
    if site is not None and not site.is_dir():
        raise FlexivRDKError(f"Selected Flexiv RDK site-packages is missing: {site}")

    sys.path[:] = [item for item in sys.path if not _is_packaged_rdk_path(item)]
    if site is not None:
        sys.path.insert(0, str(site))
    environment = _candidate_environment(
        FlexivRDKCandidate(version, site, "worker-selection"), os.environ
    )
    os.environ.clear()
    os.environ.update(environment)


def loaded_flexiv_rdk_version(
    sdk: Any,
    *,
    environment: Optional[Mapping[str, str]] = None,
) -> str:
    """Return the selected distribution version for an imported RDK module.

    Flexiv's 1.8.1 wheel reports ``flexivrdk.__version__ == "1.8.0"`` even
    though its distribution metadata and binary payload are version 1.8.1.
    A packaged worker therefore uses the selected distribution as the source
    of truth, after verifying that the imported module came from that exact
    site-packages directory.
    """
    env = os.environ if environment is None else environment
    raw_site = str(env.get(RDK_SITE_ENV, "")).strip()
    if raw_site:
        site = Path(raw_site).expanduser().resolve(strict=False)
        if not site.is_dir():
            raise FlexivRDKError(
                f"Selected Flexiv RDK site-packages is missing: {site}"
            )

        raw_module_file = getattr(sdk, "__file__", None)
        if not raw_module_file:
            raise FlexivRDKError("Loaded flexivrdk module has no file path")
        module_file = Path(str(raw_module_file)).expanduser().resolve(strict=False)
        try:
            module_file.relative_to(site)
        except ValueError as exc:
            raise FlexivRDKError(
                "Loaded flexivrdk module is outside the selected RDK profile: "
                f"module={module_file}, selected={site}"
            ) from exc

        distributions = [
            distribution
            for distribution in metadata.distributions(path=[str(site)])
            if re.sub(
                r"[-_.]+",
                "-",
                str(distribution.metadata.get("Name", "")).strip().lower(),
            )
            == "flexivrdk"
        ]
        if len(distributions) != 1:
            raise FlexivRDKError(
                "Selected Flexiv RDK profile must contain exactly one flexivrdk "
                f"distribution, found {len(distributions)} under {site}"
            )
        return normalize_rdk_version(distributions[0].version)

    for attr_name in ("__version__", "VERSION", "version"):
        attr_value = getattr(sdk, attr_name, None)
        if isinstance(attr_value, str) and attr_value.strip():
            return normalize_rdk_version(attr_value.strip())
    try:
        return normalize_rdk_version(metadata.version("flexivrdk"))
    except metadata.PackageNotFoundError:
        pass
    raise FlexivRDKError(
        "Cannot determine flexivrdk version from distribution or module metadata"
    )


def _robot_serials(device_config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = device_config.get("robot_sn")
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    serials = tuple(str(item).strip() for item in values if str(item or "").strip())
    if not serials:
        raise FlexivRDKError("Rizon config has no robot_sn to probe")
    return serials


def _network_whitelist(device_config: Mapping[str, Any]) -> list[str]:
    raw = device_config.get("network_interface_whitelist")
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise FlexivRDKError("network_interface_whitelist must be a list")
    return [str(item).strip() for item in raw if str(item).strip()]


def _last_json_object(output: str) -> Optional[dict[str, Any]]:
    for line in reversed(str(output or "").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _bounded_reason(value: object, limit: int = 500) -> str:
    text = " ".join(str(value or "unknown failure").split())
    return text[:limit]


class FlexivRDKResolver:
    """Probe candidates sequentially and cache the first successful version."""

    def __init__(
        self,
        *,
        python: Optional[Path] = None,
        environment: Optional[Mapping[str, str]] = None,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.python = Path(
            python
            or os.environ.get("SUPERINFERENCE_DEVICE_PYTHON")
            or sys.executable
        )
        self._environment = environment
        self._runner = runner
        self._cache: dict[tuple[tuple[str, ...], tuple[str, ...]], str] = {}
        self._lock = threading.Lock()

    def _probe_timeout(self, serial_count: int, environment: Mapping[str, str]) -> float:
        raw = environment.get(RDK_PROBE_TIMEOUT_ENV, "15")
        try:
            per_robot = float(raw)
        except (TypeError, ValueError):
            per_robot = 15.0
        per_robot = min(60.0, max(3.0, per_robot))
        return per_robot * max(1, serial_count)

    def _probe_candidate(
        self,
        candidate: FlexivRDKCandidate,
        serials: tuple[str, ...],
        whitelist: list[str],
        environment: Mapping[str, str],
    ) -> FlexivRDKProbeAttempt:
        started = time.monotonic()
        request = {
            "expected_version": candidate.version,
            "robot_serials": list(serials),
            "network_interface_whitelist": whitelist,
        }
        try:
            completed = self._runner(
                [str(self.python), "-m", "incubator.control_panel.flexiv_rdk"],
                input=json.dumps(request),
                text=True,
                capture_output=True,
                check=False,
                timeout=self._probe_timeout(len(serials), environment),
                env=_candidate_environment(candidate, environment),
            )
        except subprocess.TimeoutExpired:
            return FlexivRDKProbeAttempt(
                candidate.version,
                False,
                "connection probe timed out",
                time.monotonic() - started,
                timed_out=True,
            )
        except OSError as exc:
            return FlexivRDKProbeAttempt(
                candidate.version,
                False,
                _bounded_reason(exc),
                time.monotonic() - started,
            )

        payload = _last_json_object(completed.stdout)
        ok = bool(
            completed.returncode == 0
            and payload
            and payload.get("ok") is True
            and payload.get("loaded_version") == candidate.version
        )
        reason = None
        if not ok:
            reason = _bounded_reason(
                (payload or {}).get("error")
                or completed.stderr
                or completed.stdout
                or f"probe exited with code {completed.returncode}"
            )
        return FlexivRDKProbeAttempt(
            candidate.version,
            ok,
            reason,
            time.monotonic() - started,
        )

    def select(self, device_config: Mapping[str, Any]) -> FlexivRDKSelection:
        serials = _robot_serials(device_config)
        whitelist = _network_whitelist(device_config)
        explicit = device_config.get("flexiv_rdk_version")
        environment = dict(os.environ if self._environment is None else self._environment)
        candidates = discover_flexiv_rdk_candidates(
            explicit_version=str(explicit).strip() if explicit else None,
            environment=environment,
        )
        cache_key = (serials, tuple(whitelist))

        with self._lock:
            cached_version = self._cache.get(cache_key)
            if cached_version and not explicit:
                candidates.sort(key=lambda item: item.version != cached_version)

            attempts: list[FlexivRDKProbeAttempt] = []
            for candidate in candidates:
                attempt = self._probe_candidate(
                    candidate, serials, whitelist, environment
                )
                attempts.append(attempt)
                if attempt.ok:
                    cache_hit = cached_version == candidate.version
                    self._cache[cache_key] = candidate.version
                    return FlexivRDKSelection(
                        candidate, tuple(attempts), cache_hit=cache_hit
                    )

        detail = "; ".join(
            f"{attempt.version}: {attempt.reason or 'failed'}" for attempt in attempts
        )
        raise FlexivRDKProbeError(
            "No bundled Flexiv RDK could connect to every configured robot"
            + (f" ({detail})" if detail else ""),
            attempts,
        )


def _probe_request(request: Mapping[str, Any], sdk: Any = None) -> dict[str, Any]:
    """Execute the read-only probe in its isolated process."""
    if sdk is None:
        import flexivrdk as sdk  # type: ignore[import-not-found,no-redef]

    expected = normalize_rdk_version(request.get("expected_version"))
    loaded = normalize_rdk_version(metadata.version("flexivrdk"))
    if loaded != expected:
        raise FlexivRDKError(f"Expected Flexiv RDK {expected}, loaded {loaded}")
    serials = _robot_serials({"robot_sn": request.get("robot_serials")})
    whitelist = _network_whitelist(
        {"network_interface_whitelist": request.get("network_interface_whitelist")}
    )

    statuses = []
    for serial in serials:
        robot = sdk.Robot(serial, whitelist, False)
        status_method = getattr(robot, "operational_status", None)
        status = status_method() if callable(status_method) else None
        statuses.append(str(getattr(status, "name", status)))
        del robot
        gc.collect()
    return {
        "ok": True,
        "loaded_version": loaded,
        "robot_count": len(serials),
        "operational_status": statuses,
    }


def _main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise FlexivRDKError("probe request must be a JSON object")
        payload = _probe_request(request)
        print(json.dumps(payload, sort_keys=True))
        return 0
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": _bounded_reason(exc),
                },
                sort_keys=True,
            )
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "RDK_AVAILABLE_ENV",
    "RDK_PROBE_TIMEOUT_ENV",
    "RDK_ROOT_ENV",
    "RDK_SITE_ENV",
    "RDK_VERSION_ENV",
    "SUPPORTED_RDK_VERSIONS",
    "WORKER_RDK_SITE_KEY",
    "WORKER_RDK_VERSION_KEY",
    "FlexivRDKCandidate",
    "FlexivRDKError",
    "FlexivRDKProbeAttempt",
    "FlexivRDKProbeError",
    "FlexivRDKResolver",
    "FlexivRDKSelection",
    "activate_flexiv_rdk_worker",
    "discover_flexiv_rdk_candidates",
    "loaded_flexiv_rdk_version",
    "normalize_rdk_version",
]
