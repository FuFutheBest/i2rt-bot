#!/usr/bin/env python3
"""Read-only validation for an extracted SuperInference release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence


class DoctorFailure(RuntimeError):
    pass


PROBE_JSON_MARKER = "__SUPERINFERENCE_DOCTOR_JSON__="
PROBE_PARSE_ATTEMPTS = 2
READ_ONLY_DOCTOR_ENV = "SUPERINFERENCE_READ_ONLY_DOCTOR"


def _run(
    command: Sequence[str], *, environment: Optional[dict[str, str]] = None
) -> str:
    probe_environment = dict(os.environ if environment is None else environment)
    probe_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    probe_environment[READ_ONLY_DOCTOR_ENV] = "1"
    result = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        env=probe_environment,
        timeout=120,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise DoctorFailure(f"Command failed ({result.returncode}): {detail}")
    return result.stdout.strip()


def _parse_probe_json(output: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for line in reversed(output.splitlines()):
        marker_index = line.rfind(PROBE_JSON_MARKER)
        if marker_index < 0:
            continue
        candidate = line[marker_index + len(PROBE_JSON_MARKER) :].lstrip()
        try:
            payload, _end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise DoctorFailure("Profile probe returned no marked JSON object")


def _run_json_probe(
    command: Sequence[str], *, environment: Optional[dict[str, str]] = None
) -> dict[str, Any]:
    parse_error: Optional[DoctorFailure] = None
    for _attempt in range(PROBE_PARSE_ATTEMPTS):
        output = _run(command, environment=environment)
        try:
            return _parse_probe_json(output)
        except DoctorFailure as exc:
            parse_error = exc
    assert parse_error is not None
    raise parse_error


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for component in value.split("."):
        digits = "".join(character for character in component if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _check_system(manifest: dict[str, Any]) -> dict[str, Any]:
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        raise DoctorFailure(f"Unsupported architecture: {machine}")
    libc_name, libc_version = platform.libc_ver()
    required = str(manifest.get("minimum_glibc", "2.35"))
    if libc_name != "glibc" or _version_tuple(libc_version) < _version_tuple(required):
        raise DoctorFailure(
            f"glibc {required}+ is required, detected {libc_name} {libc_version}"
        )
    return {
        "architecture": machine,
        "glibc": libc_version,
        "minimum_glibc": required,
    }


def _check_flexiv_profiles(
    root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    code = """
import json
from importlib import metadata
import flexivrdk
from devices.flexiv_adapter import FlexivAdapter

adapter = FlexivAdapter()
print("__SUPERINFERENCE_DOCTOR_JSON__=" + json.dumps({
    'version': metadata.version('flexivrdk'),
    'adapter_version': '.'.join(str(item) for item in adapter._rdk_version),
    'module_version': getattr(flexivrdk, '__version__', None),
    'module': flexivrdk.__file__,
}))
"""
    results: list[dict[str, Any]] = []
    records = manifest.get("flexivrdk")
    if not isinstance(records, list) or not records:
        raise DoctorFailure("Release manifest contains no Flexiv RDK inventory")
    expected_versions = [
        str(item.get("rdk_version"))
        for item in records
        if isinstance(item, Mapping) and item.get("rdk_version")
    ]
    if len(expected_versions) != len(set(expected_versions)):
        raise DoctorFailure("Release manifest contains duplicate Flexiv RDK versions")
    for expected in expected_versions:
        payload = _run_json_probe(
            [
                str(root / "bin/python-device"),
                "--rdk-version",
                expected,
                "-c",
                code,
            ]
        )
        if payload.get("version") != expected:
            raise DoctorFailure(
                f"RDK profile {expected} loaded {payload.get('version')}"
            )
        if payload.get("adapter_version") != expected:
            raise DoctorFailure(
                f"RDK profile {expected} was identified by the device adapter as "
                f"{payload.get('adapter_version')}"
            )
        results.append(payload)
    return results


def _check_device_profile(root: Path) -> dict[str, Any]:
    code = """
import json
import os
import sys
import importlib.util
import can
import cv2
import flask
import hydra
import i2rt
import i2rt.motor_drivers.dm_driver as dm_driver
import mujoco
import numpy
import pyarrow
from importlib import metadata
assert importlib.util.find_spec('flexivrdk') is None, 'device profile leaked an unselected flexivrdk'
os.environ.pop('MVCAM_COMMON_RUNENV', None)
from devices import WebCameraOpenPi
from incubator.control_panel.runtime_config import compose_runtime_config, extract_robot_slots
from incubator.control_panel.gripper_control import (
    SDK_MODULE,
    GripperControlManager,
)
from third_party.lktech_motor_sdk.example.lktech_gripper import Gripper as LKTechGripper
assert sys.version_info[:2] == (3, 12), sys.version
assert SDK_MODULE.is_file(), SDK_MODULE
assert hasattr(LKTechGripper, '_has_position_calibration')
assert GripperControlManager.__name__ == 'GripperControlManager'
config = compose_runtime_config(
    'policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi'
)
slots = extract_robot_slots(config)
print("__SUPERINFERENCE_DOCTOR_JSON__=" + json.dumps({
    'python': sys.version.split()[0],
    'numpy': numpy.__version__,
    'opencv': cv2.__version__,
    'python_can': metadata.version('python-can'),
    'flask': flask.__version__ if hasattr(flask, '__version__') else 'installed',
    'i2rt': metadata.version('i2rt'),
    'i2rt_dm_driver': dm_driver.__file__,
    'mujoco': mujoco.__version__,
    'pyarrow': pyarrow.__version__,
    'gripper_transport': 'direct_serial_worker',
    'gripper_sdk_module': str(SDK_MODULE),
    'gripper_control_manager': GripperControlManager.__name__,
    'flexivrdk_unselected': True,
    'webcamera_without_hik_sdk': WebCameraOpenPi.__name__,
    'robot_slots': slots,
}))
"""
    environment = dict(os.environ)
    environment.pop("SUPERINFERENCE_FLEXIV_RDK_VERSION", None)
    environment.pop("SUPERINFERENCE_FLEXIV_RDK_SITE", None)
    return _run_json_probe(
        [
            str(root / "bin/python-device"),
            "-c",
            code,
        ],
        environment=environment,
    )


def _check_recording_profile(root: Path) -> dict[str, Any]:
    code = """
import json
import sys
from importlib import metadata
import nmx_msg
from nmx_msg.Image_pb2 import RGBD
from nmx_msg.Metadata_pb2 import Metadata
from nmx_msg.Umi_pb2 import UmiPose, UmiState
from nmx_nedf_api import (
    NEDFFactory,
    NEDFReaderBase,
    NEDFType,
    NEDFWriterBase,
)
assert sys.version_info[:2] == (3, 10), sys.version
assert callable(NEDFFactory.get_reader)
assert callable(NEDFFactory.get_writer)
assert callable(NEDFReaderBase.iter_decoded_messages)
assert callable(NEDFWriterBase.write_message)
assert hasattr(NEDFType, 'NEDF3')
assert metadata.version('nm-storage-api') == '0.3.2'
assert metadata.version('nmx-msg') == '2.2.0'
assert metadata.version('nmx-nedf-api') == '0.3.1'
print("__SUPERINFERENCE_DOCTOR_JSON__=" + json.dumps({
    'python': sys.version.split()[0],
    'nm_storage_api': metadata.version('nm-storage-api'),
    'nmx_msg': metadata.version('nmx-msg'),
    'nmx_nedf_api': metadata.version('nmx-nedf-api'),
    'unified_nedf_reader': True,
    'nedf3_writer': True,
    'nedf3_messages': [RGBD.__name__, UmiPose.__name__, UmiState.__name__, Metadata.__name__],
}))
"""
    return _run_json_probe([str(root / "bin/python-recording"), "-c", code])


def _check_converter_profile(root: Path) -> dict[str, Any]:
    code = """
import hashlib
import json
import sys
from importlib import metadata
import pyarrow
from incubator.control_panel.converter_profile import default_profile
from nmx_nedf_api import NEDFFactory, NEDFReaderBase, NEDFType

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

assert sys.version_info[:2] == (3, 10), sys.version
assert callable(NEDFFactory.get_reader)
assert callable(NEDFReaderBase.iter_decoded_messages)
assert hasattr(NEDFType, 'NEDF3')
assert metadata.version('nm-storage-api') == '0.3.2'
assert metadata.version('nmx-msg') == '2.2.0'
assert metadata.version('nmx-nedf-api') == '0.3.1'
profile = default_profile()
conversion_scripts = {
    'wrapper': sha256_file(profile.wrapper_script),
    'converter': sha256_file(profile.converter_source),
    'merger': sha256_file(profile.merger_source),
}
assert conversion_scripts == {
    'wrapper': profile.wrapper_script_sha256,
    'converter': profile.converter_source_sha256,
    'merger': profile.merger_source_sha256,
}
print("__SUPERINFERENCE_DOCTOR_JSON__=" + json.dumps({
    'python': sys.version.split()[0],
    'nm_storage_api': metadata.version('nm-storage-api'),
    'nmx_msg': metadata.version('nmx-msg'),
    'nmx_nedf_api': metadata.version('nmx-nedf-api'),
    'unified_nedf_reader': True,
    'pyarrow': pyarrow.__version__,
    'conversion_scripts': conversion_scripts,
}))
"""
    payload = _run_json_probe([str(root / "bin/python-converter"), "-c", code])
    ffmpeg = root / "runtime/converter/ffmpeg-av1/bin/ffmpeg"
    version = _run([str(ffmpeg), "-hide_banner", "-version"]).splitlines()[0]
    full_decode_ffmpeg = root / "runtime/converter/full-decode-ffmpeg/bin/ffmpeg"
    full_decode_version = _run(
        [str(full_decode_ffmpeg), "-hide_banner", "-version"]
    ).splitlines()[0]
    decoders = _run([str(full_decode_ffmpeg), "-hide_banner", "-decoders"])
    if "libdav1d" not in decoders and "libaom-av1" not in decoders:
        raise DoctorFailure("Bundled full-decode FFmpeg has no software AV1 decoder")
    payload["ffmpeg"] = version
    payload["full_decode_ffmpeg"] = full_decode_version
    payload["software_av1_decoder"] = (
        "libdav1d" if "libdav1d" in decoders else "libaom-av1"
    )
    return payload


def _check_policy_profile(root: Path) -> dict[str, Any]:
    code = """
import json
import sys
import hydra
import jax
import loguru
import matplotlib
import openpi
import openpi_client
import skimage
from consumer import openpi_policy_infer
from policy.openpi_umi_policy import OpenPiUmiPolicy
assert sys.version_info[:2] == (3, 12), sys.version
print("__SUPERINFERENCE_DOCTOR_JSON__=" + json.dumps({
    'python': sys.version.split()[0],
    'jax': jax.__version__,
    'hydra': hydra.__version__,
    'loguru': loguru.__version__,
    'matplotlib': matplotlib.__version__,
    'scikit_image': skimage.__version__,
    'openpi': openpi.__file__,
    'openpi_client': openpi_client.__file__,
    'policy_adapter': OpenPiUmiPolicy.__name__,
    'policy_entrypoint': openpi_policy_infer.__file__,
    'devices': [str(item) for item in jax.devices()],
}))
"""
    environment = dict(os.environ)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return _run_json_probe(
        [str(root / "bin/python-policy"), "-c", code],
        environment=environment,
    )


def run_doctor(root: Path, *, quick: bool = False) -> dict[str, Any]:
    manifest_path = root / "manifest/release.json"
    if not manifest_path.is_file():
        raise DoctorFailure(f"Release manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "ok": True,
        "release": manifest.get("release"),
        "system": _check_system(manifest),
        "flexivrdk": _check_flexiv_profiles(root, manifest),
        "device": _check_device_profile(root),
        "recording": _check_recording_profile(root),
        "converter": _check_converter_profile(root),
    }
    if not quick and manifest.get("profiles", {}).get("policy", {}).get("included"):
        result["policy"] = _check_policy_profile(root)
    return result


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = run_doctor(args.release_root.resolve(), quick=args.quick)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"release doctor: OK ({result.get('release')})")
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except (DoctorFailure, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"release doctor: FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
