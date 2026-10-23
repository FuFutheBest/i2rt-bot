#!/usr/bin/env python3
"""Build a self-contained Linux x86_64 SuperInference release archive."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
from typing import Any, Iterable, Mapping, Optional, Sequence
import zipfile


SCHEMA_VERSION = 4
MAIN_PYTHON_VERSION = "3.12"
PROFILE_PYTHON_VERSIONS = {
    profile: MAIN_PYTHON_VERSION
    for profile in ("control-panel", "device", "policy", "recording", "converter")
}
FLEXIV_RDK_VERSIONS: tuple[str, ...] = (
    "1.7.0",
    "1.8.0",
    "1.8.1",
    "1.9.0",
    "1.9.1",
    "1.9.2",
)
NEDF_SDK_PACKAGES: tuple[tuple[str, str], ...] = (
    ("dm3-core", "0.2.18"),
    ("nmx-msg", "2.2.0"),
)
REQUIRED_SUBMODULES = frozenset(
    {"MvImport", "i2rt", "lktech_motor_sdk", "nmx_vla", "openpi"}
)
PROFILE_EXCLUDES = (
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "flexivrdk/",
    "flexivrdk.libs/",
    "flexivrdk-*.dist-info/",
    "__editable__*",
    "_editable_impl_*.pth",
    "_virtualenv.pth",
    "_virtualenv.py",
)
SOURCE_EXCLUDES = (
    ".git/",
    ".venv/",
    ".cache/",
    "__pycache__/",
    "*.pyc",
    "logs/",
    "wandb/",
    "openpi_ckpt/",
    "checkpoints/",
)


class ReleaseBuildError(RuntimeError):
    pass


def _run(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    capture: bool = False,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    arguments = [str(item) for item in command]
    result = subprocess.run(
        arguments,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ReleaseBuildError(
            f"Command failed ({result.returncode}): {' '.join(arguments)}\n{detail}"
        )
    return result


def _capture(command: Sequence[str], *, cwd: Optional[Path] = None) -> str:
    return _run(command, cwd=cwd, capture=True).stdout.strip()


def _absolute_entry(path: Path) -> Path:
    """Make an interpreter path absolute without dereferencing its venv link."""
    return Path(os.path.abspath(path.expanduser()))


def _ensure_linux_host() -> None:
    if sys.platform != "linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise ReleaseBuildError("Release builds currently target Linux x86_64 only")
    for command in ("git", "rsync", "tar", "zstd"):
        if shutil.which(command) is None:
            raise ReleaseBuildError(f"Required build command is missing: {command}")


def _safe_reset_directory(path: Path, parent: Path) -> None:
    resolved_parent = parent.resolve()
    resolved = path.resolve(strict=False)
    if resolved.parent != resolved_parent or resolved == resolved_parent:
        raise ReleaseBuildError(f"Unsafe staging directory: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _safe_tar_members(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive:
        target = (destination / member.name).resolve(strict=False)
        if target != root and root not in target.parents:
            raise ReleaseBuildError(f"Unsafe git archive member: {member.name}")
        archive.extract(member, destination)


def _export_git_tree(repo_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            _safe_tar_members(archive, destination)
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise ReleaseBuildError(f"git archive failed: {stderr.strip()}")


def _rsync_tree(
    source: Path,
    destination: Path,
    *,
    excludes: Iterable[str] = (),
) -> None:
    if not source.is_dir():
        raise ReleaseBuildError(f"Source directory is missing: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-a", "--delete"]
    for pattern in excludes:
        command.extend(["--exclude", pattern])
    command.extend([f"{source.resolve()}/", f"{destination.resolve()}/"])
    _run(command)


def _git_snapshot(source: Path) -> dict[str, Any]:
    try:
        top_level = Path(
            _capture(["git", "rev-parse", "--show-toplevel"], cwd=source)
        ).resolve()
        if top_level != source.resolve():
            return {"git": False}
        commit = _capture(["git", "rev-parse", "HEAD"], cwd=source)
        status = _capture(
            ["git", "status", "--short", "--untracked-files=all"], cwd=source
        )
        diff = _run(
            ["git", "diff", "--binary", "HEAD"], cwd=source, capture=True
        ).stdout
    except ReleaseBuildError:
        return {"git": False}
    return {
        "git": True,
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines(),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "diff": diff,
    }


def _copy_source_snapshot(
    name: str,
    source: Path,
    destination: Path,
    manifest_dir: Path,
) -> dict[str, Any]:
    snapshot = _git_snapshot(source)
    _rsync_tree(source, destination, excludes=SOURCE_EXCLUDES)
    diff = snapshot.pop("diff", "")
    if diff:
        patch_path = manifest_dir / f"{name}-working-tree.patch"
        patch_path.write_text(diff, encoding="utf-8")
        snapshot["working_tree_patch"] = str(
            patch_path.relative_to(manifest_dir.parent)
        )
    snapshot["source"] = str(source.resolve())
    return snapshot


def _python_info(executable: Path) -> dict[str, Any]:
    if not executable.is_file():
        raise ReleaseBuildError(f"Python interpreter is missing: {executable}")
    code = (
        "import json,sys,sysconfig; "
        "print(json.dumps({'version':sys.version.split()[0],"
        "'base_prefix':sys.base_prefix,'purelib':sysconfig.get_paths()['purelib']}))"
    )
    output = _capture([str(executable), "-c", code])
    try:
        info = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ReleaseBuildError(f"Cannot inspect Python profile {executable}") from exc
    info["executable"] = str(executable.absolute())
    info["resolved_executable"] = str(executable.resolve())
    return info


def _copy_python_runtime(info: Mapping[str, Any], destination: Path) -> None:
    source = Path(str(info["base_prefix"]))
    _rsync_tree(
        source,
        destination,
        excludes=("__pycache__/", "*.pyc", "lib/python*/test/"),
    )


def _sanitize_profile_site(site: Path) -> None:
    for path in site.glob("*.pth"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "/home/" in text or "\\Users\\" in text:
            path.unlink()
    for pattern in ("__editable__*", "_editable_impl_*.pth", "_virtualenv.pth"):
        for path in site.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _copy_profile(info: Mapping[str, Any], destination: Path) -> None:
    source = Path(str(info["purelib"]))
    _rsync_tree(source, destination, excludes=PROFILE_EXCLUDES)
    _sanitize_profile_site(destination)


def _same_python_profile(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    """Return whether two interpreter entries expose the same site-packages."""
    first_site = os.path.normcase(os.path.abspath(str(first["purelib"])))
    second_site = os.path.normcase(os.path.abspath(str(second["purelib"])))
    return first_site == second_site


def _package_inventory(executable: Path) -> list[dict[str, str]]:
    code = """
import importlib.metadata as metadata
import json
items = sorted(
    ({'name': dist.metadata.get('Name', ''), 'version': dist.version}
     for dist in metadata.distributions()),
    key=lambda item: (item['name'].lower(), item['version']),
)
print(json.dumps(items))
"""
    output = _capture([str(executable), "-c", code])
    return json.loads(output.splitlines()[-1])


def _download_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "SuperInference-release"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "SuperInference-release"}
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        destination.open("wb") as output,
    ):
        shutil.copyfileobj(response, output, length=4 * 1024 * 1024)


def _copy_executable(source: Path, destination: Path, label: str) -> dict[str, str]:
    source = source.expanduser().resolve()
    if not source.is_file() or not os.access(source, os.X_OK):
        raise ReleaseBuildError(f"{label} is missing or not executable: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(
        destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    version = _capture([str(destination), "-hide_banner", "-version"])
    return {
        "source": str(source),
        "path": str(destination),
        "sha256": _sha256_file(destination),
        "version": version.splitlines()[0],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_extract_wheel(wheel: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve(strict=False)
            if target != root and root not in target.parents:
                raise ReleaseBuildError(f"Unsafe wheel member: {member.filename}")
        archive.extractall(destination)


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", str(value).strip()).casefold()


def _wheel_identity(wheel: Path) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(wheel) as archive:
            metadata_members = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_members) != 1:
                raise ReleaseBuildError(
                    f"Wheel must contain exactly one dist-info METADATA: {wheel.name}"
                )
            metadata = Parser().parsestr(
                archive.read(metadata_members[0]).decode("utf-8")
            )
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise ReleaseBuildError(f"Cannot inspect wheel {wheel.name}: {exc}") from exc
    name = _canonical_distribution_name(str(metadata.get("Name") or ""))
    version = str(metadata.get("Version") or "").strip()
    if not name or not version:
        raise ReleaseBuildError(f"Wheel has incomplete package metadata: {wheel.name}")
    return name, version


def _wheel_tags(wheel: Path) -> tuple[str, ...]:
    try:
        with zipfile.ZipFile(wheel) as archive:
            wheel_members = [
                name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
            ]
            if len(wheel_members) != 1:
                raise ReleaseBuildError(
                    f"Wheel must contain exactly one dist-info WHEEL: {wheel.name}"
                )
            metadata = Parser().parsestr(
                archive.read(wheel_members[0]).decode("utf-8")
            )
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise ReleaseBuildError(f"Cannot inspect wheel tags {wheel.name}: {exc}") from exc
    tags = tuple(str(value).strip() for value in metadata.get_all("Tag", []) if value)
    if not tags:
        raise ReleaseBuildError(f"Wheel declares no compatibility tags: {wheel.name}")
    return tags


def _wheel_filename_tag(wheel: Path) -> str:
    if wheel.suffix != ".whl":
        raise ReleaseBuildError(f"Invalid wheel filename: {wheel.name}")
    parts = wheel.name[:-4].rsplit("-", 3)
    if len(parts) != 4 or not all(parts[-3:]):
        raise ReleaseBuildError(f"Invalid wheel filename: {wheel.name}")
    return "-".join(parts[-3:])


def _is_cp312_linux_x86_64_wheel(wheel: Path) -> bool:
    interpreter, abi, platform_tag = _wheel_filename_tag(wheel).split("-", 2)
    if interpreter == "py3" and abi == "none" and platform_tag == "any":
        return True
    return (
        interpreter == "cp312"
        and abi in {"cp312", "abi3"}
        and platform_tag.startswith("manylinux")
        and platform_tag.endswith("_x86_64")
    )


def _install_nedf_sdk_wheels(
    wheelhouse: Path,
    release_root: Path,
) -> list[dict[str, Any]]:
    candidate = wheelhouse.expanduser()
    try:
        candidate.lstat()
        source = candidate.resolve(strict=True)
    except OSError as exc:
        raise ReleaseBuildError(
            f"NEDF SDK wheelhouse is unavailable: {candidate}"
        ) from exc
    if candidate.is_symlink() or not source.is_dir():
        raise ReleaseBuildError(f"NEDF SDK wheelhouse is unsafe: {source}")
    discovered: dict[tuple[str, str], list[Path]] = {}
    for wheel in sorted(source.glob("*.whl")):
        if wheel.is_symlink() or not wheel.is_file():
            continue
        identity = _wheel_identity(wheel)
        if identity in {
            (_canonical_distribution_name(package), version)
            for package, version in NEDF_SDK_PACKAGES
        } and not _is_cp312_linux_x86_64_wheel(wheel):
            raise ReleaseBuildError(
                "NEDF SDK wheel is not compatible with CPython 3.12 Linux x86_64: "
                f"{wheel.name} ({', '.join(_wheel_tags(wheel))})"
            )
        discovered.setdefault(identity, []).append(wheel)

    destination = release_root / "vendor/wheels/nedf3"
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for package, version in NEDF_SDK_PACKAGES:
        identity = (_canonical_distribution_name(package), version)
        candidates = discovered.get(identity, [])
        if len(candidates) != 1:
            raise ReleaseBuildError(
                f"Expected one {package}=={version} wheel, found {len(candidates)}"
            )
        source_wheel = candidates[0]
        target = destination / source_wheel.name
        shutil.copy2(source_wheel, target)
        records.append(
            {
                "package": package,
                "version": version,
                "wheel": target.relative_to(release_root).as_posix(),
                "filename_tag": _wheel_filename_tag(target),
                "declared_tags": list(_wheel_tags(target)),
                "sha256": _sha256_file(target),
            }
        )
    return records


def _install_flexiv_profiles(
    release_root: Path, vendor_root: Path
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    wheel_root = vendor_root / "flexivrdk"
    wheel_root.mkdir(parents=True, exist_ok=True)
    for version in FLEXIV_RDK_VERSIONS:
        metadata = _download_json(f"https://pypi.org/pypi/flexivrdk/{version}/json")
        candidates = [
            item
            for item in metadata.get("urls", [])
            if item.get("packagetype") == "bdist_wheel"
            and "cp312-cp312" in str(item.get("filename", ""))
            and "x86_64" in str(item.get("filename", ""))
            and "manylinux" in str(item.get("filename", ""))
        ]
        if len(candidates) != 1:
            raise ReleaseBuildError(
                f"Expected one CPython 3.12 Linux x86_64 wheel for flexivrdk {version}, "
                f"found {len(candidates)}"
            )
        candidate = candidates[0]
        wheel = wheel_root / str(candidate["filename"])
        if not wheel.is_file():
            _download_file(str(candidate["url"]), wheel)
        expected_hash = str(candidate.get("digests", {}).get("sha256", ""))
        actual_hash = _sha256_file(wheel)
        if not expected_hash or actual_hash != expected_hash:
            raise ReleaseBuildError(
                f"Flexiv wheel hash mismatch for {wheel.name}: {actual_hash}"
            )
        site = (
            release_root
            / "runtime"
            / "flexivrdk"
            / f"rdk-{version}"
            / "site-packages"
        )
        site.mkdir(parents=True, exist_ok=True)
        _safe_extract_wheel(wheel, site)
        records.append(
            {
                "rdk_version": version,
                "wheel": str(wheel.relative_to(release_root)),
                "sha256": actual_hash,
            }
        )
    return records


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _wrapper(profile: str) -> str:
    python_version = PROFILE_PYTHON_VERSIONS[profile]
    return f"""#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")/.." && pwd)"
exec "$ROOT/runtime/python{python_version.replace(".", "")}/bin/python{python_version}" \\
  "$ROOT/app/incubator/control_panel/release/runtime_bootstrap.py" {profile} "$@"
"""


def _write_wrappers(release_root: Path) -> None:
    bin_root = release_root / "bin"
    bin_root.mkdir(parents=True, exist_ok=True)
    _write_executable(
        bin_root / "superinference-control-panel", _wrapper("control-panel")
    )
    _write_executable(bin_root / "python-device", _wrapper("device"))
    _write_executable(bin_root / "python-policy", _wrapper("policy"))
    _write_executable(bin_root / "python-recording", _wrapper("recording"))
    _write_executable(bin_root / "python-converter", _wrapper("converter"))
    _write_executable(
        bin_root / "superinference-doctor",
        """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export SUPERINFERENCE_READ_ONLY_DOCTOR=1
exec "$ROOT/runtime/python312/bin/python3.12" \\
  "$ROOT/app/incubator/control_panel/release/doctor.py" --release-root "$ROOT" "$@"
""",
    )
    _write_executable(
        release_root / "start_server.sh",
        """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/app/incubator/control_panel/start_server.sh" "$@"
""",
    )


def _deduplicate_tree(root: Path) -> tuple[int, int, dict[Path, str]]:
    groups: dict[tuple[int, int], list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        info = path.stat()
        if info.st_size < 16 * 1024:
            continue
        groups.setdefault((info.st_size, stat.S_IMODE(info.st_mode)), []).append(path)

    linked_files = 0
    saved_bytes = 0
    hashes: dict[Path, str] = {}
    for (size, _mode), paths in groups.items():
        if len(paths) < 2:
            continue
        by_hash: dict[str, Path] = {}
        for path in paths:
            digest = _sha256_file(path)
            hashes[path] = digest
            existing = by_hash.get(digest)
            if existing is None:
                by_hash[digest] = path
                continue
            path.unlink()
            os.link(existing, path)
            linked_files += 1
            saved_bytes += size
    return linked_files, saved_bytes, hashes


def _write_checksums(
    root: Path, cached_hashes: Optional[Mapping[Path, str]] = None
) -> Path:
    checksum_path = root / "manifest/SHA256SUMS"
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != checksum_path
    ]
    cache = dict(cached_hashes or {})

    def digest(path: Path) -> tuple[Path, str]:
        return path, cache.get(path) or _sha256_file(path)

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        rows = list(pool.map(digest, paths))
    rows.sort(key=lambda item: str(item[0].relative_to(root)))
    checksum_path.write_text(
        "".join(
            f"{value}  {path.relative_to(root).as_posix()}\n" for path, value in rows
        ),
        encoding="utf-8",
    )
    return checksum_path


def _create_archive(root: Path, output: Path, epoch: int, level: int) -> None:
    tar_command = [
        "tar",
        "--sort=name",
        f"--mtime=@{epoch}",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "-C",
        str(root.parent),
        "-cf",
        "-",
        root.name,
    ]
    tar_process = subprocess.Popen(tar_command, stdout=subprocess.PIPE)
    assert tar_process.stdout is not None
    zstd_process = subprocess.Popen(
        ["zstd", "-T0", f"-{level}", "-f", "-o", str(output)],
        stdin=tar_process.stdout,
    )
    tar_process.stdout.close()
    zstd_return = zstd_process.wait()
    tar_return = tar_process.wait()
    if tar_return != 0 or zstd_return != 0:
        raise ReleaseBuildError(
            f"Archive creation failed: tar={tar_return}, zstd={zstd_return}"
        )
    output.chmod(0o644)


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _read_http_json(url: str, *, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise ReleaseBuildError(f"HTTP smoke returned {response.status}: {url}")
        return json.loads(response.read().decode("utf-8"))


def _validate_built_archive(
    archive: Path,
    output_dir: Path,
    release_name: str,
) -> dict[str, Any]:
    """Extract and start the final archive exactly as an operator will."""

    smoke_root = output_dir / f".{release_name}.archive-smoke"
    _safe_reset_directory(smoke_root, output_dir)
    extract_root = smoke_root / "extract"
    data_root = smoke_root / "data"
    extract_root.mkdir()
    data_root.mkdir()
    release_root = extract_root / release_name
    log_path = smoke_root / "server.log"
    server: Optional[subprocess.Popen[bytes]] = None
    try:
        _run(
            ["tar", "--zstd", "-xf", str(archive), "-C", str(extract_root)],
            timeout=1800,
        )
        if not release_root.is_dir():
            raise ReleaseBuildError(
                f"Archive did not contain expected root: {release_name}"
            )
        _run(
            ["sha256sum", "-c", "manifest/SHA256SUMS"],
            cwd=release_root,
            timeout=1800,
        )
        _run(
            [str(release_root / "bin/superinference-doctor"), "--json"],
            cwd=release_root,
            timeout=300,
        )

        port = _reserve_local_port()
        environment = dict(os.environ)
        environment.update(
            {
                "SUPERINFERENCE_DATA_ROOT": str(data_root),
                "SUPERINFERENCE_CONTROL_PANEL_HOST": "127.0.0.1",
                "SUPERINFERENCE_CONTROL_PANEL_PORT": str(port),
            }
        )
        with log_path.open("wb") as log_file:
            server = subprocess.Popen(
                [
                    str(release_root / "start_server.sh"),
                    "--lock-file",
                    str(smoke_root / "server.lock"),
                ],
                cwd=release_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 60.0
            last_error: Optional[BaseException] = None
            state: Optional[Mapping[str, Any]] = None
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/", timeout=2
                    ) as response:
                        if response.status != 200:
                            raise ReleaseBuildError("Control panel root is not HTTP 200")
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/recordings", timeout=2
                    ) as response:
                        if response.status != 200:
                            raise ReleaseBuildError("Recordings route is not HTTP 200")
                    payload = _read_http_json(
                        f"http://127.0.0.1:{port}/api/state", timeout=2
                    )
                    if not isinstance(payload, Mapping):
                        raise ReleaseBuildError("State route did not return an object")
                    state = payload
                    break
                except (OSError, ValueError, ReleaseBuildError) as error:
                    last_error = error
                    time.sleep(0.25)
            if state is None:
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                raise ReleaseBuildError(
                    f"Extracted release failed HTTP smoke: {last_error}\n{detail}"
                )
            active_devices = [
                str(device.get("key"))
                for device in state.get("devices", [])
                if isinstance(device, Mapping)
                and str(device.get("status", "")).lower()
                in {"running", "starting", "external", "degraded"}
            ]
            workflow = state.get("workflow", {})
            runtime_active = bool(
                isinstance(workflow, Mapping)
                and isinstance(workflow.get("runtime"), Mapping)
                and workflow["runtime"].get("active")
            )
            inference_running = bool(
                isinstance(workflow, Mapping)
                and isinstance(workflow.get("inference"), Mapping)
                and workflow["inference"].get("running")
            )
            if active_devices or runtime_active or inference_running:
                raise ReleaseBuildError(
                    "Archive smoke unexpectedly started hardware or workflow: "
                    + ", ".join(active_devices)
                )
        return {
            "doctor": "passed",
            "http": ["/", "/recordings", "/api/state"],
            "hardware_started": False,
        }
    finally:
        if server is not None and server.poll() is None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except OSError:
                    pass
                server.wait(timeout=10)
        if smoke_root.exists():
            shutil.rmtree(smoke_root)


def _parse_submodule_sources(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ReleaseBuildError(
                f"Invalid --submodule-source value {value!r}; expected name=/path"
            )
        if "/" in name or "\\" in name:
            raise ReleaseBuildError(f"Unsafe submodule name: {name}")
        result[name.strip()] = Path(raw_path.strip()).expanduser().resolve()
    return result


def _require_python_version(info: Mapping[str, Any], label: str, expected: str) -> None:
    detected = str(info.get("version", ""))
    if not detected.startswith(f"{expected}."):
        raise ReleaseBuildError(
            f"{label} profile must use Python {expected}, detected {detected or 'unknown'}"
        )


def build_release(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_linux_host()
    repo_root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = _capture(["git", "rev-parse", "HEAD"], cwd=repo_root)
    short_commit = commit[:8]
    release_version = args.version or datetime.now().strftime("%Y%m%d")
    release_name = (
        f"superinference-openpi-{release_version}-{short_commit}-linux-x86_64"
    )
    stage = output_dir / release_name
    _safe_reset_directory(stage, output_dir)

    app_root = stage / "app"
    manifest_root = stage / "manifest"
    manifest_root.mkdir(parents=True)
    _export_git_tree(repo_root, app_root)

    source_records: dict[str, Any] = {}
    submodule_sources = _parse_submodule_sources(args.submodule_source)
    missing_submodules = sorted(REQUIRED_SUBMODULES - set(submodule_sources))
    if missing_submodules:
        raise ReleaseBuildError(
            "Missing required --submodule-source entries: "
            + ", ".join(missing_submodules)
        )
    for name, source in sorted(submodule_sources.items()):
        source_records[name] = _copy_source_snapshot(
            name,
            source,
            app_root / "third_party" / name,
            manifest_root,
        )

    device_python = _absolute_entry(args.device_python)
    recording_python = _absolute_entry(args.recording_python)
    converter_python = _absolute_entry(args.converter_python)
    policy_python = (
        _absolute_entry(args.policy_python) if args.policy_python is not None else None
    )
    device_info = _python_info(device_python)
    recording_info = _python_info(recording_python)
    converter_info = _python_info(converter_python)
    policy_info = _python_info(policy_python) if policy_python is not None else None
    _require_python_version(device_info, "Device", MAIN_PYTHON_VERSION)
    _require_python_version(recording_info, "Recording", MAIN_PYTHON_VERSION)
    _require_python_version(converter_info, "Converter", MAIN_PYTHON_VERSION)
    if policy_info is not None:
        _require_python_version(policy_info, "Policy", MAIN_PYTHON_VERSION)

    profile_inputs = [recording_info, converter_info]
    if policy_info is not None:
        profile_inputs.append(policy_info)
    if not all(_same_python_profile(device_info, info) for info in profile_inputs):
        raise ReleaseBuildError(
            "All release profiles must use the same Python 3.12 environment"
        )

    _copy_python_runtime(device_info, stage / "runtime/python312")
    _copy_profile(device_info, stage / "runtime/profiles/main/site-packages")

    converter_root = stage / "runtime/converter"
    _rsync_tree(
        args.ffmpeg_root.expanduser().resolve(),
        converter_root / "ffmpeg-av1",
    )
    full_decode_record = _copy_executable(
        args.full_decode_ffmpeg,
        converter_root / "full-decode-ffmpeg/bin/ffmpeg",
        "full-decode FFmpeg",
    )
    full_decode_record["path"] = str(
        Path(full_decode_record["path"]).relative_to(stage)
    )
    vendor_root = stage / "vendor/wheels"
    nedf_sdk_records = _install_nedf_sdk_wheels(
        args.vendor_wheelhouse,
        stage,
    )
    flexiv_records = _install_flexiv_profiles(stage, vendor_root)
    _write_wrappers(stage)

    profiles: dict[str, Any] = {
        "device": {
            "included": True,
            "python": device_info,
            "site_packages": "runtime/profiles/main/site-packages",
            "packages": _package_inventory(device_python),
        },
        "recording": {
            "included": True,
            "python": recording_info,
            "site_packages": "runtime/profiles/main/site-packages",
            "packages": _package_inventory(recording_python),
        },
        "converter": {
            "included": True,
            "python": converter_info,
            "site_packages": "runtime/profiles/main/site-packages",
            "packages": _package_inventory(converter_python),
        },
        "policy": {
            "included": policy_info is not None,
            "python": policy_info,
            "site_packages": "runtime/profiles/main/site-packages",
            "packages": (
                _package_inventory(policy_python) if policy_python is not None else []
            ),
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release": release_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_status": _capture(
            ["git", "status", "--short", "--untracked-files=no"], cwd=repo_root
        ).splitlines(),
        "target": "linux-x86_64",
        "minimum_glibc": "2.35",
        "minimum_ubuntu": "22.04",
        "checkpoint_included": False,
        "runtime_layout": {
            "main": {
                "python": MAIN_PYTHON_VERSION,
                "path": "runtime/python312",
                "profiles": [
                    "control-panel",
                    "device",
                    "recording",
                    "converter",
                    *(["policy"] if policy_info is not None else []),
                ],
                "site_packages": "runtime/profiles/main/site-packages",
                "unified": True,
            },
        },
        "profiles": profiles,
        "nedf_sdk_wheels": nedf_sdk_records,
        "flexivrdk": flexiv_records,
        "converter_runtime": {
            "full_decode_ffmpeg": full_decode_record,
        },
        "submodule_snapshots": source_records,
    }
    (manifest_root / "release.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not args.skip_doctor:
        command = [str(stage / "bin/superinference-doctor"), "--json"]
        if policy_info is None:
            command.append("--quick")
        _run(command, timeout=300)

    linked_files = 0
    saved_bytes = 0
    cached_hashes: dict[Path, str] = {}
    if not args.no_deduplicate:
        linked_files, saved_bytes, cached_hashes = _deduplicate_tree(stage)
    manifest["deduplication"] = {
        "linked_files": linked_files,
        "saved_bytes": saved_bytes,
    }
    (manifest_root / "release.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cached_hashes.pop(manifest_root / "release.json", None)
    _write_checksums(stage, cached_hashes)

    epoch = int(_capture(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=repo_root))
    archive = output_dir / f"{release_name}.tar.zst"
    _create_archive(stage, archive, epoch, args.zstd_level)
    archive_hash = _sha256_file(archive)
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    sidecar.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")
    sidecar.chmod(0o644)
    archive_smoke = _validate_built_archive(archive, output_dir, release_name)
    result = {
        "release": release_name,
        "archive": str(archive),
        "archive_sha256": archive_hash,
        "archive_bytes": archive.stat().st_size,
        "stage": str(stage) if args.keep_stage else None,
        "archive_smoke": archive_smoke,
    }
    if not args.keep_stage:
        shutil.rmtree(stage)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version")
    parser.add_argument("--device-python", type=Path, required=True)
    parser.add_argument("--recording-python", type=Path, required=True)
    parser.add_argument("--converter-python", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path)
    parser.add_argument("--ffmpeg-root", type=Path, required=True)
    parser.add_argument("--full-decode-ffmpeg", type=Path, required=True)
    parser.add_argument("--vendor-wheelhouse", type=Path, required=True)
    parser.add_argument(
        "--submodule-source",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument("--skip-doctor", action="store_true")
    parser.add_argument("--keep-stage", action="store_true")
    parser.add_argument("--zstd-level", type=int, default=10)
    return parser


def _main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = build_release(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except (ReleaseBuildError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"release build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
