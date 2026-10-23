#!/usr/bin/env python3
"""Execute one profile from a self-contained SuperInference release."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence


PROFILE_NAMES = ("control-panel", "device", "policy", "recording", "converter")
MAIN_PYTHON_VERSION = "3.12"
# The release builder exports one relocatable CPython runtime and one shared
# site-packages tree for every profile, including NEDF3 recording/conversion.
# Keep this map aligned with build_release.py's manifest layout.
PROFILE_PYTHON_VERSIONS = {
    profile: MAIN_PYTHON_VERSION for profile in PROFILE_NAMES
}
PROFILE_SITE_GROUPS = {
    profile: "main" for profile in PROFILE_NAMES
}
FASTWAM_SERVER_PYTHON_ENV = "SUPERINFERENCE_FASTWAM_SERVER_PYTHON"
# A FastWAM model environment may be provisioned separately because the
# nmx_vla stack has a different Transformers/CUDA dependency contract than the
# bundled OpenPI policy profile.  Requiring an explicit opt-in prevents a
# stale environment variable from silently changing a packaged launch.
FASTWAM_SERVER_PYTHON_EXTERNAL_ENV = "SUPERINFERENCE_FASTWAM_SERVER_PYTHON_EXTERNAL"


class ReleaseRuntimeError(RuntimeError):
    pass


def _release_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = os.environ.get("SUPERINFERENCE_RELEASE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ReleaseRuntimeError(f"{label} is missing from the release: {path}")
    return path


def _profile_site(root: Path, profile: str) -> Path:
    profiles_root = root / "runtime" / "profiles"
    shared_name = PROFILE_SITE_GROUPS.get(profile)
    if shared_name is not None:
        shared = profiles_root / shared_name / "site-packages"
        if shared.is_dir():
            return shared
    path = profiles_root / profile / "site-packages"
    if not path.is_dir():
        raise ReleaseRuntimeError(f"Runtime profile is missing: {path}")
    return path


def _python_path(root: Path, profile: str) -> Path:
    version = PROFILE_PYTHON_VERSIONS.get(profile)
    if version is None:
        raise ReleaseRuntimeError(f"Unknown runtime profile: {profile}")
    compact_version = version.replace(".", "")
    return _required_file(
        root / "runtime" / f"python{compact_version}" / "bin" / f"python{version}",
        f"Python {version}",
    )


def _prepend_paths(environment: dict[str, str], paths: Sequence[Path]) -> None:
    resolved = [str(path) for path in paths if path.exists()]
    if environment.get("SUPERINFERENCE_RELEASE_ALLOW_HOST_PYTHONPATH") == "1":
        existing = environment.get("PYTHONPATH")
        if existing:
            resolved.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(resolved)


def _configure_fastwam_server_python(environment: dict[str, str], root: Path) -> None:
    """Select the model-server interpreter for a packaged control panel.

    The bundled policy runtime is the default.  An external interpreter is
    accepted only when the caller explicitly sets the opt-in flag and a
    non-empty absolute path; dependency validation then remains the server
    process's responsibility so the panel can report its actionable error.
    """
    bundled = root / "bin" / "python-policy"
    external_enabled = environment.get(FASTWAM_SERVER_PYTHON_EXTERNAL_ENV, "")
    if str(external_enabled).strip().lower() in {"1", "true", "yes", "on"}:
        configured = environment.get(FASTWAM_SERVER_PYTHON_ENV, "").strip()
        if not configured:
            raise ReleaseRuntimeError(
                f"{FASTWAM_SERVER_PYTHON_ENV} must be set when "
                f"{FASTWAM_SERVER_PYTHON_EXTERNAL_ENV}=1"
            )
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            raise ReleaseRuntimeError(
                f"{FASTWAM_SERVER_PYTHON_ENV} must be an absolute path for an "
                "external FastWAM model environment"
            )
        environment[FASTWAM_SERVER_PYTHON_ENV] = str(candidate.resolve(strict=False))
        return

    # Ignore inherited values unless the explicit opt-in above is present.
    environment[FASTWAM_SERVER_PYTHON_ENV] = str(bundled)


def _available_flexiv_rdks(root: Path) -> tuple[str, ...]:
    app_root = root / "app"
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    from incubator.control_panel.flexiv_rdk import SUPPORTED_RDK_VERSIONS

    rdk_root = root / "runtime/flexivrdk"
    return tuple(
        version
        for version in SUPPORTED_RDK_VERSIONS
        if (rdk_root / f"rdk-{version}/site-packages").is_dir()
    )


def _select_flexiv_rdk(root: Path, explicit: str) -> tuple[str, Path]:
    app_root = root / "app"
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    from incubator.control_panel.flexiv_rdk import normalize_rdk_version

    version = normalize_rdk_version(explicit)
    site = root / "runtime/flexivrdk" / f"rdk-{version}" / "site-packages"
    if not site.is_dir():
        raise ReleaseRuntimeError(f"Flexiv RDK {version} is missing: {site}")
    return version, site


def _detect_elements_for_display(environment: dict[str, str], root: Path) -> None:
    app_root = root / "app"
    if str(app_root) not in sys.path:
        sys.path.insert(0, str(app_root))
    try:
        from incubator.control_panel.flexiv_compat import (
            ELEMENTS_VERSION_ENV,
            select_flexiv_elements,
        )

        selection = select_flexiv_elements(environment=environment)
    except Exception:
        return
    environment.setdefault(ELEMENTS_VERSION_ENV, selection.detected_version)


def build_profile_environment(
    root: Path,
    profile: str,
    *,
    rdk_version: Optional[str] = None,
    elements_version: Optional[str] = None,
    base_environment: Optional[Mapping[str, str]] = None,
) -> tuple[dict[str, str], Path]:
    """Build an isolated environment without importing profile dependencies."""
    if profile not in PROFILE_NAMES:
        raise ReleaseRuntimeError(f"Unknown runtime profile: {profile}")
    if elements_version is not None:
        raise ReleaseRuntimeError(
            "Elements version does not determine robot firmware compatibility; "
            "use --rdk-version only for a manual RDK override"
        )
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["SUPERINFERENCE_RELEASE_ROOT"] = str(root)
    environment.setdefault("SUPERINFERENCE_DEVICE_PROFILE_NAME", "device312")
    environment.setdefault("SUPERINFERENCE_INFERENCE_PROFILE_NAME", "openpi312")

    app_root = root / "app"
    if not app_root.is_dir():
        raise ReleaseRuntimeError(f"Application tree is missing: {app_root}")

    selected_profile = "device" if profile == "control-panel" else profile
    profile_site = _profile_site(root, selected_profile)
    python = _python_path(root, selected_profile)
    paths = [app_root, profile_site, profile_site / "rerun_sdk"]

    if selected_profile == "policy":
        paths.extend(
            [
                app_root / "third_party" / "openpi" / "src",
                app_root
                / "third_party"
                / "openpi"
                / "packages"
                / "openpi-client"
                / "src",
            ]
        )
    elif selected_profile == "device":
        paths.append(app_root / "third_party" / "i2rt")
        rdk_root = root / "runtime/flexivrdk"
        available = _available_flexiv_rdks(root)
        from incubator.control_panel.flexiv_rdk import (
            RDK_AVAILABLE_ENV,
            RDK_ROOT_ENV,
            RDK_SITE_ENV,
            RDK_VERSION_ENV,
        )

        environment[RDK_ROOT_ENV] = str(rdk_root)
        environment[RDK_AVAILABLE_ENV] = ",".join(available)
        if profile == "control-panel":
            _detect_elements_for_display(environment, root)
        else:
            requested_rdk = rdk_version or environment.get(RDK_VERSION_ENV)
            if requested_rdk:
                selected_version, flexiv_site = _select_flexiv_rdk(
                    root, requested_rdk
                )
                paths.append(flexiv_site)
                environment[RDK_VERSION_ENV] = selected_version
                environment[RDK_SITE_ENV] = str(flexiv_site)
            else:
                environment.pop(RDK_VERSION_ENV, None)
                environment.pop(RDK_SITE_ENV, None)

    _prepend_paths(environment, paths)
    environment["PATH"] = os.pathsep.join(
        [
            str(root / "bin"),
            str(python.parent),
            environment.get("PATH", ""),
        ]
    )

    data_root = Path(
        environment.get(
            "SUPERINFERENCE_DATA_ROOT",
            str(Path.home() / ".local/share/superinference-control-panel"),
        )
    ).expanduser()
    environment.setdefault(
        "SUPERINFERENCE_DEVICE_PYTHON", str(_python_path(root, "device"))
    )
    # Packaged launches own the lightweight client interpreter too.  This keeps
    # a path left by an older release from escaping into the current archive.
    environment["SUPERINFERENCE_INFERENCE_PYTHON"] = str(root / "bin/python-policy")
    _configure_fastwam_server_python(environment, root)
    environment.setdefault(
        "SUPERINFERENCE_RECORDING_PYTHON", str(root / "bin/python-recording")
    )
    environment.setdefault(
        "SUPERINFERENCE_CONVERTER_RUNTIME_ROOT", str(root / "runtime/converter")
    )
    environment.setdefault(
        "SUPERINFERENCE_CONVERTER_PYTHON", str(root / "bin/python-converter")
    )
    environment.setdefault(
        "SUPERINFERENCE_CONVERTER_FFMPEG",
        str(root / "runtime/converter/ffmpeg-av1/bin/ffmpeg"),
    )
    environment.setdefault(
        "SUPERINFERENCE_CONVERTER_FFPROBE",
        str(root / "runtime/converter/ffmpeg-av1/bin/ffprobe"),
    )
    environment.setdefault(
        "SUPERINFERENCE_CONVERTER_FULL_DECODE_FFMPEG",
        str(root / "runtime/converter/full-decode-ffmpeg/bin/ffmpeg"),
    )
    environment.setdefault(
        "SUPERINFERENCE_CONVERSION_OUTPUT_ROOT",
        str(data_root / "conversion_jobs"),
    )
    environment.setdefault(
        "SUPERINFERENCE_CONVERTER_ACTIVATION_RECEIPT",
        str(data_root / "converter/activation-smoke.json"),
    )
    return environment, python


def execute_profile(
    root: Path,
    profile: str,
    command: Sequence[str],
    *,
    rdk_version: Optional[str] = None,
    elements_version: Optional[str] = None,
) -> None:
    environment, python = build_profile_environment(
        root,
        profile,
        rdk_version=rdk_version,
        elements_version=elements_version,
    )
    arguments = list(command)
    if profile == "control-panel":
        arguments = ["-m", "control_panel", *arguments]
    if not arguments:
        raise ReleaseRuntimeError(f"No command supplied for profile {profile}")
    os.chdir(root / "app")
    os.execve(str(python), [str(python), *arguments], environment)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILE_NAMES)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--rdk-version")
    parser.add_argument("--elements-version")
    args, command = parser.parse_known_args(argv)
    execute_profile(
        _release_root(args.release_root),
        args.profile,
        command,
        rdk_version=args.rdk_version,
        elements_version=args.elements_version,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except ReleaseRuntimeError as exc:
        print(f"release runtime error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
