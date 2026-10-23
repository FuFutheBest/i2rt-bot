"""Machine-local Python runtime profiles for multi-process inference workflows.

The registry is intentionally dependency-free.  It can be read before Hydra,
vendor SDKs, or a model framework are available, and it stores interpreter
entry paths without resolving virtual-environment symlinks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REGISTRY_ENV = "SUPERINFERENCE_RUNTIME_REGISTRY"
ACTIVE_PROFILE_ENV = "SUPERINFERENCE_RUNTIME_PROFILE"
CONFIG_NAME_ENV = "SUPERINFERENCE_CONFIG_NAME"
DEFAULT_REGISTRY_RELATIVE_PATH = Path(".superinference/local/runtime_profiles.json")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_ROLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_CONFIG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IMPORT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_PROBE_MARKER = "__SUPERINFERENCE_RUNTIME_PROFILE__="
_MAX_REGISTRY_BYTES = 1024 * 1024

_PROBE_CODE = r"""
import importlib
import json
import platform
import sys

requested = json.loads(sys.argv[1])
modules = {}
ok = True
for name in requested:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        modules[name] = {
            "ok": True,
            "version": None if version is None else str(version),
        }
    except BaseException as exc:
        ok = False
        modules[name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

payload = {
    "ok": ok,
    "python_version": platform.python_version(),
    "implementation": platform.python_implementation(),
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "modules": modules,
}
print("__SUPERINFERENCE_RUNTIME_PROFILE__=" + json.dumps(payload, sort_keys=True))
raise SystemExit(0 if ok else 17)
"""


class RuntimeProfileError(RuntimeError):
    """Base error for a local runtime registry or profile."""


class RuntimeRegistryError(RuntimeProfileError):
    """The local registry is malformed or cannot be updated."""


class RuntimeProfileNotFoundError(RuntimeProfileError):
    """A requested role or profile has no local calibration."""


class RuntimeProfileVerificationError(RuntimeProfileError):
    """A configured interpreter failed its hardware-free import probe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute(path: Path, *, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def repository_root(path: Path | None = None) -> Path:
    """Return a lexical repository root without collapsing environment symlinks."""
    if path is not None:
        return _absolute(Path(path), base=Path.cwd())
    return _absolute(Path(__file__).parents[1], base=Path.cwd())


def default_registry_path(
    *,
    repo_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the selected machine-local registry path."""
    root = repository_root(repo_root)
    environment = os.environ if environ is None else environ
    configured = environment.get(REGISTRY_ENV, "").strip()
    if configured:
        return _absolute(Path(configured), base=root)
    return root / DEFAULT_REGISTRY_RELATIVE_PATH


def normalize_profile_name(value: str) -> str:
    name = str(value).strip()
    if _NAME_RE.fullmatch(name) is None:
        raise RuntimeRegistryError(
            "profile name must contain only letters, digits, dot, underscore, or dash"
        )
    return name


def normalize_role(value: str) -> str:
    role = str(value).strip().replace("_", "-")
    if _ROLE_RE.fullmatch(role) is None:
        raise RuntimeRegistryError(
            "runtime role must contain only letters, digits, dot, underscore, or dash"
        )
    return role


def normalize_config_name(value: str | None) -> str | None:
    if value is None:
        return None
    name = str(value).strip()
    for suffix in (".yaml", ".yml"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if (
        _CONFIG_RE.fullmatch(name) is None
        or name.startswith("/")
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise RuntimeRegistryError(
            "config name must be a safe repository-relative name"
        )
    return name


def _path_payload(path: Path, *, repo_root: Path) -> dict[str, str]:
    absolute = _absolute(path, base=repo_root)
    try:
        relative = absolute.relative_to(repo_root)
    except ValueError:
        return {"base": "absolute", "path": str(absolute)}
    return {"base": "repo", "path": relative.as_posix() or "."}


def _path_from_payload(
    value: Any,
    *,
    repo_root: Path,
    field: str,
) -> Path:
    if isinstance(value, str):
        return _absolute(Path(value), base=repo_root)
    if not isinstance(value, Mapping):
        raise RuntimeRegistryError(f"{field} must be a path mapping")
    base = value.get("base")
    raw_path = value.get("path")
    if (
        base not in {"repo", "absolute"}
        or not isinstance(raw_path, str)
        or not raw_path
    ):
        raise RuntimeRegistryError(
            f"{field} must contain base=repo|absolute and a non-empty path"
        )
    if base == "absolute":
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            raise RuntimeRegistryError(f"{field} absolute path is not absolute")
        return _absolute(candidate, base=repo_root)
    candidate = _absolute(Path(raw_path), base=repo_root)
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeRegistryError(f"{field} escapes the repository root") from exc
    return candidate


def _normalized_environment(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeRegistryError("profile environment must be a mapping")
    result: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name)
        if _ENVIRONMENT_NAME_RE.fullmatch(name) is None:
            raise RuntimeRegistryError(f"invalid environment variable name: {name!r}")
        if not isinstance(raw_value, str):
            raise RuntimeRegistryError(f"environment value for {name} must be a string")
        if "\0" in raw_value:
            raise RuntimeRegistryError(
                f"environment value for {name} contains a NUL byte"
            )
        result[name] = raw_value
    return result


def _normalized_imports(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RuntimeRegistryError("profile imports must be a list of module names")
    imports: list[str] = []
    for raw_name in value:
        name = str(raw_name).strip()
        if _IMPORT_RE.fullmatch(name) is None:
            raise RuntimeRegistryError(f"invalid import module name: {raw_name!r}")
        if name not in imports:
            imports.append(name)
    return tuple(imports)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _profile_spec_hash(profile: Mapping[str, Any]) -> str:
    return _canonical_hash(
        {
            "python": profile.get("python"),
            "cwd": profile.get("cwd"),
            "environment": profile.get("environment", {}),
            "imports": profile.get("imports", []),
        }
    )


@dataclass(frozen=True)
class RuntimeProfile:
    """A resolved local profile with an exact interpreter entry path."""

    name: str
    python: Path
    cwd: Path
    environment: dict[str, str]
    imports: tuple[str, ...]
    verification: dict[str, Any] | None
    registry_path: Path
    spec_hash: str

    @property
    def verified(self) -> bool:
        if not (
            isinstance(self.verification, Mapping)
            and self.verification.get("spec_hash") == self.spec_hash
        ):
            return False
        try:
            return (
                self.python.is_file()
                and os.access(self.python, os.X_OK)
                and self.cwd.is_dir()
            )
        except OSError:
            return False

    def command(self, arguments: Sequence[str]) -> list[str]:
        return [str(self.python), *(str(argument) for argument in arguments)]

    def process_environment(
        self,
        base: Mapping[str, str] | None = None,
        *,
        config_name: str | None = None,
    ) -> dict[str, str]:
        environment = dict(os.environ if base is None else base)
        environment.update(self.environment)
        existing_path = environment.get("PATH", "")
        profile_bin = str(self.python.parent)
        path_entries = [
            entry
            for entry in existing_path.split(os.pathsep)
            if entry and entry != profile_bin
        ]
        environment["PATH"] = os.pathsep.join((profile_bin, *path_entries))
        environment[REGISTRY_ENV] = str(self.registry_path)
        environment[ACTIVE_PROFILE_ENV] = self.name
        normalized_config = normalize_config_name(config_name)
        if normalized_config is not None:
            environment[CONFIG_NAME_ENV] = normalized_config
        return environment


@dataclass(frozen=True)
class RuntimeSelection:
    """The selected interpreter plus the source that won precedence."""

    python: Path
    source: str
    profile: RuntimeProfile | None = None


class RuntimeRegistry:
    """Validated in-memory view of one machine-local runtime registry."""

    def __init__(
        self,
        *,
        repo_root: Path,
        path: Path,
        profiles: Mapping[str, Mapping[str, Any]] | None = None,
        defaults: Mapping[str, str] | None = None,
        bindings: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.repo_root = repository_root(repo_root)
        self.path = _absolute(path, base=self.repo_root)
        self.profiles = {
            normalize_profile_name(name): dict(profile)
            for name, profile in (profiles or {}).items()
        }
        self.defaults = {
            normalize_role(role): normalize_profile_name(profile)
            for role, profile in (defaults or {}).items()
        }
        self.bindings = {
            normalize_config_name(config_name) or "": {
                normalize_role(role): normalize_profile_name(profile)
                for role, profile in roles.items()
            }
            for config_name, roles in (bindings or {}).items()
        }
        for name in self.profiles:
            self._resolved_profile(name)
        self._validate_bindings()

    @classmethod
    def load(
        cls,
        *,
        repo_root: Path | None = None,
        path: Path | None = None,
        environ: Mapping[str, str] | None = None,
        missing_ok: bool = True,
    ) -> RuntimeRegistry:
        root = repository_root(repo_root)
        registry_path = (
            _absolute(path, base=root)
            if path is not None
            else default_registry_path(repo_root=root, environ=environ)
        )
        if not registry_path.exists():
            if not missing_ok:
                raise RuntimeRegistryError(
                    f"runtime registry does not exist: {registry_path}"
                )
            return cls(repo_root=root, path=registry_path)
        try:
            registry_info = registry_path.lstat()
        except OSError as exc:
            raise RuntimeRegistryError(
                f"cannot inspect runtime registry {registry_path}: {exc}"
            ) from exc
        if registry_path.is_symlink() or not registry_path.is_file():
            raise RuntimeRegistryError(
                f"runtime registry must be a regular non-symlink file: {registry_path}"
            )
        if registry_info.st_size <= 0 or registry_info.st_size > _MAX_REGISTRY_BYTES:
            raise RuntimeRegistryError(
                f"runtime registry has an invalid size: {registry_path}"
            )
        try:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeRegistryError(
                f"cannot read runtime registry {registry_path}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RuntimeRegistryError("runtime registry root must be a JSON object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeRegistryError(
                f"unsupported runtime registry schema: {payload.get('schema_version')!r}"
            )
        payload_root = payload.get("repo_root")
        if not isinstance(payload_root, str) or not Path(payload_root).is_absolute():
            raise RuntimeRegistryError(
                "runtime registry repo_root must be an absolute path"
            )
        stored_root = _absolute(Path(payload_root), base=root)
        if stored_root != root:
            raise RuntimeRegistryError(
                "runtime registry belongs to a different repository root: "
                f"{stored_root} (current: {root})"
            )
        profiles = payload.get("profiles", {})
        defaults = payload.get("defaults", {})
        bindings = payload.get("bindings", {})
        if not isinstance(profiles, Mapping):
            raise RuntimeRegistryError("runtime registry profiles must be an object")
        if not isinstance(defaults, Mapping):
            raise RuntimeRegistryError("runtime registry defaults must be an object")
        if not isinstance(bindings, Mapping):
            raise RuntimeRegistryError("runtime registry bindings must be an object")
        if any(not isinstance(value, Mapping) for value in profiles.values()):
            raise RuntimeRegistryError("every runtime profile must be an object")
        if any(not isinstance(value, str) for value in defaults.values()):
            raise RuntimeRegistryError(
                "every default runtime binding must be a profile name"
            )
        if any(not isinstance(value, Mapping) for value in bindings.values()):
            raise RuntimeRegistryError("every config binding must be an object")
        if any(
            not isinstance(profile, str)
            for roles in bindings.values()
            for profile in roles.values()
        ):
            raise RuntimeRegistryError(
                "every config runtime binding must be a profile name"
            )
        return cls(
            repo_root=root,
            path=registry_path,
            profiles=profiles,
            defaults=defaults,
            bindings=bindings,
        )

    def _validate_bindings(self) -> None:
        for role, profile in self.defaults.items():
            if profile not in self.profiles:
                raise RuntimeRegistryError(
                    f"default role {role!r} references unknown profile {profile!r}"
                )
        for config_name, roles in self.bindings.items():
            for role, profile in roles.items():
                if profile not in self.profiles:
                    raise RuntimeRegistryError(
                        f"binding {config_name}:{role} references unknown profile {profile!r}"
                    )

    def _resolved_profile(self, name: str) -> RuntimeProfile:
        normalized_name = normalize_profile_name(name)
        try:
            raw = self.profiles[normalized_name]
        except KeyError as exc:
            raise RuntimeProfileNotFoundError(
                f"runtime profile is not registered: {normalized_name}"
            ) from exc
        python = _path_from_payload(
            raw.get("python"),
            repo_root=self.repo_root,
            field=f"profiles.{normalized_name}.python",
        )
        cwd = _path_from_payload(
            raw.get("cwd", {"base": "repo", "path": "."}),
            repo_root=self.repo_root,
            field=f"profiles.{normalized_name}.cwd",
        )
        environment = _normalized_environment(raw.get("environment"))
        imports = _normalized_imports(raw.get("imports"))
        verification = raw.get("verification")
        if verification is not None and not isinstance(verification, Mapping):
            raise RuntimeRegistryError(
                f"profiles.{normalized_name}.verification must be an object or null"
            )
        return RuntimeProfile(
            name=normalized_name,
            python=python,
            cwd=cwd,
            environment=environment,
            imports=imports,
            verification=dict(verification)
            if isinstance(verification, Mapping)
            else None,
            registry_path=self.path,
            spec_hash=_profile_spec_hash(raw),
        )

    def profile(self, name: str) -> RuntimeProfile:
        return self._resolved_profile(name)

    def bound_profile_name(
        self,
        role: str,
        *,
        config_name: str | None = None,
    ) -> str | None:
        normalized_role = normalize_role(role)
        normalized_config = normalize_config_name(config_name)
        if normalized_config is not None:
            configured = self.bindings.get(normalized_config, {}).get(normalized_role)
            if configured is not None:
                return configured
        return self.defaults.get(normalized_role)

    def resolve(
        self,
        role: str,
        *,
        config_name: str | None = None,
        required: bool = False,
    ) -> RuntimeProfile | None:
        profile_name = self.bound_profile_name(role, config_name=config_name)
        if profile_name is None:
            if required:
                detail = f" for config {config_name!r}" if config_name else ""
                raise RuntimeProfileNotFoundError(
                    f"runtime role {normalize_role(role)!r} is not calibrated{detail}"
                )
            return None
        return self.profile(profile_name)

    def register(
        self,
        name: str,
        *,
        python: Path,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        imports: Sequence[str] = (),
    ) -> RuntimeProfile:
        normalized_name = normalize_profile_name(name)
        existing = self.profiles.get(normalized_name, {})
        now = _utc_now()
        self.profiles[normalized_name] = {
            "python": _path_payload(Path(python), repo_root=self.repo_root),
            "cwd": _path_payload(Path(cwd or self.repo_root), repo_root=self.repo_root),
            "environment": _normalized_environment(environment),
            "imports": list(_normalized_imports(imports)),
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            "verification": None,
        }
        return self.profile(normalized_name)

    def bind(
        self,
        role: str,
        profile_name: str,
        *,
        config_name: str | None = None,
    ) -> None:
        normalized_role = normalize_role(role)
        normalized_profile = normalize_profile_name(profile_name)
        if normalized_profile not in self.profiles:
            raise RuntimeProfileNotFoundError(
                f"cannot bind unknown runtime profile: {normalized_profile}"
            )
        normalized_config = normalize_config_name(config_name)
        if normalized_config is None:
            self.defaults[normalized_role] = normalized_profile
            return
        self.bindings.setdefault(normalized_config, {})[normalized_role] = (
            normalized_profile
        )

    def unbind(self, role: str, *, config_name: str | None = None) -> bool:
        normalized_role = normalize_role(role)
        normalized_config = normalize_config_name(config_name)
        if normalized_config is None:
            return self.defaults.pop(normalized_role, None) is not None
        roles = self.bindings.get(normalized_config)
        if roles is None or roles.pop(normalized_role, None) is None:
            return False
        if not roles:
            self.bindings.pop(normalized_config, None)
        return True

    def _verification_payload(
        self,
        profile: RuntimeProfile,
        probe: Mapping[str, Any],
    ) -> dict[str, Any]:
        fingerprint_source = {
            "spec_hash": profile.spec_hash,
            "python_version": probe.get("python_version"),
            "implementation": probe.get("implementation"),
            "executable": probe.get("executable"),
            "prefix": probe.get("prefix"),
            "base_prefix": probe.get("base_prefix"),
            "modules": probe.get("modules", {}),
        }
        return {
            "verified_at": _utc_now(),
            "spec_hash": profile.spec_hash,
            "fingerprint": _canonical_hash(fingerprint_source),
            **fingerprint_source,
        }

    def probe(self, name: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
        profile = self.profile(name)
        if not profile.python.is_file() or not os.access(profile.python, os.X_OK):
            raise RuntimeProfileVerificationError(
                f"runtime profile {profile.name!r} Python is missing or not executable: {profile.python}"
            )
        if not profile.cwd.is_dir():
            raise RuntimeProfileVerificationError(
                f"runtime profile {profile.name!r} working directory does not exist: {profile.cwd}"
            )
        try:
            completed = subprocess.run(
                [str(profile.python), "-c", _PROBE_CODE, json.dumps(profile.imports)],
                cwd=profile.cwd,
                env=profile.process_environment(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=max(0.1, float(timeout_s)),
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeProfileVerificationError(
                f"runtime profile {profile.name!r} probe failed: {type(exc).__name__}: {exc}"
            ) from exc
        marker_line = next(
            (
                line[len(_PROBE_MARKER) :]
                for line in reversed(completed.stdout.splitlines())
                if line.startswith(_PROBE_MARKER)
            ),
            None,
        )
        if marker_line is None:
            detail = (completed.stderr or completed.stdout or "no probe output").strip()
            raise RuntimeProfileVerificationError(
                f"runtime profile {profile.name!r} produced no valid probe result: {detail}"
            )
        try:
            probe = json.loads(marker_line)
        except json.JSONDecodeError as exc:
            raise RuntimeProfileVerificationError(
                f"runtime profile {profile.name!r} returned invalid probe JSON"
            ) from exc
        if completed.returncode != 0 or probe.get("ok") is not True:
            failures = [
                f"{module}: {result.get('error', 'import failed')}"
                for module, result in probe.get("modules", {}).items()
                if isinstance(result, Mapping) and result.get("ok") is not True
            ]
            detail = "; ".join(failures) or (completed.stderr.strip() or "probe failed")
            raise RuntimeProfileVerificationError(
                f"runtime profile {profile.name!r} failed verification: {detail}"
            )
        return self._verification_payload(profile, probe)

    def verify(self, name: str, *, timeout_s: float = 30.0) -> RuntimeProfile:
        normalized_name = normalize_profile_name(name)
        verification = self.probe(normalized_name, timeout_s=timeout_s)
        self.profiles[normalized_name]["verification"] = verification
        self.profiles[normalized_name]["updated_at"] = _utc_now()
        return self.profile(normalized_name)

    def status(self, name: str) -> dict[str, Any]:
        profile = self.profile(name)
        return {
            "name": profile.name,
            "python": str(profile.python),
            "cwd": str(profile.cwd),
            "imports": list(profile.imports),
            "python_executable": profile.python.is_file()
            and os.access(profile.python, os.X_OK),
            "cwd_exists": profile.cwd.is_dir(),
            "verified": profile.verified,
            "verified_at": (
                profile.verification.get("verified_at")
                if isinstance(profile.verification, Mapping)
                else None
            ),
            "fingerprint": (
                profile.verification.get("fingerprint")
                if isinstance(profile.verification, Mapping)
                else None
            ),
        }

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "repo_root": str(self.repo_root),
            "updated_at": _utc_now(),
            "profiles": self.profiles,
            "defaults": self.defaults,
            "bindings": self.bindings,
        }

    def save(self) -> None:
        self._validate_bindings()
        serialized = (
            json.dumps(
                self.payload(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.path.parent.chmod(0o700)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.chmod(0o600)
            os.replace(temporary_path, self.path)
            self.path.chmod(0o600)
        except OSError as exc:
            raise RuntimeRegistryError(
                f"cannot write runtime registry {self.path}: {exc}"
            ) from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)


def resolve_runtime_profile(
    role: str,
    *,
    config_name: str | None = None,
    repo_root: Path | None = None,
    registry_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    required: bool = False,
) -> RuntimeProfile | None:
    registry = RuntimeRegistry.load(
        repo_root=repo_root,
        path=registry_path,
        environ=environ,
    )
    return registry.resolve(role, config_name=config_name, required=required)


def select_runtime_python(
    role: str,
    *,
    explicit: Path | str | None = None,
    environment_names: Sequence[str] = (),
    config_name: str | None = None,
    fallback: Path | str | None = None,
    repo_root: Path | None = None,
    registry_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    require_verified_profile: bool = False,
    required: bool = False,
) -> RuntimeSelection | None:
    """Select explicit, environment, registry, then fallback interpreter."""
    root = repository_root(repo_root)
    environment = os.environ if environ is None else environ
    if explicit is not None:
        return RuntimeSelection(_absolute(Path(explicit), base=root), "explicit")
    for name in environment_names:
        value = environment.get(name, "").strip()
        if value:
            return RuntimeSelection(
                _absolute(Path(value), base=root), f"environment:{name}"
            )
    profile = resolve_runtime_profile(
        role,
        config_name=config_name,
        repo_root=root,
        registry_path=registry_path,
        environ=environment,
    )
    if profile is not None:
        if require_verified_profile and not profile.verified:
            raise RuntimeProfileVerificationError(
                f"runtime profile {profile.name!r} is unverified or stale; run verify first"
            )
        return RuntimeSelection(profile.python, f"profile:{profile.name}", profile)
    if fallback is not None:
        return RuntimeSelection(_absolute(Path(fallback), base=root), "fallback")
    if required:
        detail = f" for config {config_name!r}" if config_name else ""
        raise RuntimeProfileNotFoundError(
            f"no interpreter selected for runtime role {normalize_role(role)!r}{detail}"
        )
    return None


__all__ = [
    "ACTIVE_PROFILE_ENV",
    "CONFIG_NAME_ENV",
    "DEFAULT_REGISTRY_RELATIVE_PATH",
    "REGISTRY_ENV",
    "RuntimeProfile",
    "RuntimeProfileError",
    "RuntimeProfileNotFoundError",
    "RuntimeProfileVerificationError",
    "RuntimeRegistry",
    "RuntimeRegistryError",
    "RuntimeSelection",
    "default_registry_path",
    "normalize_config_name",
    "normalize_profile_name",
    "normalize_role",
    "repository_root",
    "resolve_runtime_profile",
    "select_runtime_python",
]
