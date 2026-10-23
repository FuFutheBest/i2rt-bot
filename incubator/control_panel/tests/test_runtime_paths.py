import sys
from pathlib import Path

import pytest

from incubator.control_panel import converter_profile, server
from incubator.control_panel.runtime_paths import (
    DEVICE_PROFILE_NAME_ENV,
    RELEASE_DEVICE_PYTHON,
    resolve_device_python,
    resolve_inference_python,
    resolve_python_profile_name,
)
from utils.runtime_profiles import REGISTRY_ENV, RuntimeRegistry


@pytest.fixture(autouse=True)
def _isolated_runtime_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(REGISTRY_ENV, str(tmp_path / "missing-runtime-profiles.json"))


def test_runtime_python_resolution_prefers_explicit_then_environment_and_active(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SUPERINFERENCE_RELEASE_ROOT", raising=False)
    monkeypatch.setenv("SUPERINFERENCE_DEVICE_PYTHON", "/release/python-device")
    monkeypatch.setenv("SUPERINFERENCE_INFERENCE_PYTHON", "/release/python-policy")

    assert resolve_device_python() == Path("/release/python-device")
    assert resolve_inference_python() == Path("/release/python-policy")
    assert resolve_device_python(Path("/cli/device")) == Path("/cli/device")
    assert resolve_inference_python(Path("/cli/policy")) == Path("/cli/policy")

    monkeypatch.delenv("SUPERINFERENCE_DEVICE_PYTHON")
    monkeypatch.delenv("SUPERINFERENCE_INFERENCE_PYTHON")
    assert resolve_device_python(active_python=Path(sys.executable)) == Path(
        sys.executable
    )
    assert resolve_inference_python(active_python=Path(sys.executable)) == Path(
        sys.executable
    )


def test_source_device_python_fails_when_active_is_invalid(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SUPERINFERENCE_RELEASE_ROOT", raising=False)
    monkeypatch.delenv("SUPERINFERENCE_DEVICE_PYTHON", raising=False)

    with pytest.raises(RuntimeError, match='Device Python is not configured'):
        resolve_device_python(active_python=tmp_path / "missing-python")


def test_source_inference_python_fails_when_active_is_invalid(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SUPERINFERENCE_INFERENCE_PYTHON", raising=False)

    with pytest.raises(RuntimeError, match='Inference Python is not configured'):
        resolve_inference_python(active_python=tmp_path / "missing-python")


def test_release_device_python_is_pinned_to_bundled_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    release_root = tmp_path / "release"
    monkeypatch.setenv("SUPERINFERENCE_RELEASE_ROOT", str(release_root))
    monkeypatch.setenv("SUPERINFERENCE_DEVICE_PYTHON", "/host/python")

    assert resolve_device_python(active_python=Path(sys.executable)) == (
        release_root / RELEASE_DEVICE_PYTHON
    )


def test_source_runtime_uses_verified_config_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "runtime_profiles.json"
    registry = RuntimeRegistry.load(repo_root=tmp_path, path=registry_path)
    registry.register("robotwin", python=Path(sys.executable), imports=["json"])
    registry.verify("robotwin")
    registry.bind(
        "device",
        "robotwin",
        config_name="policy_infer/robotwin/config_infer_robotwin_pi05_joint14",
    )
    registry.bind(
        "policy",
        "robotwin",
        config_name="policy_infer/robotwin/config_infer_robotwin_pi05_joint14",
    )
    registry.save()
    monkeypatch.setenv(REGISTRY_ENV, str(registry_path))
    monkeypatch.delenv("SUPERINFERENCE_DEVICE_PYTHON", raising=False)
    monkeypatch.delenv("SUPERINFERENCE_INFERENCE_PYTHON", raising=False)

    assert resolve_device_python(
        active_python=tmp_path / "missing",
        config_name="policy_infer/robotwin/config_infer_robotwin_pi05_joint14",
        repo_root=tmp_path,
    ) == Path(sys.executable)
    assert resolve_inference_python(
        active_python=tmp_path / "missing",
        config_name="policy_infer/robotwin/config_infer_robotwin_pi05_joint14",
        repo_root=tmp_path,
    ) == Path(sys.executable)


def test_profile_name_uses_override_then_interpreter_path(monkeypatch) -> None:
    monkeypatch.setenv(DEVICE_PROFILE_NAME_ENV, "device312-release")
    assert (
        resolve_python_profile_name(
            "device",
            Path("/runtime/python310/bin/python3.10"),
            environment_name=DEVICE_PROFILE_NAME_ENV,
        )
        == "device312-release"
    )

    monkeypatch.delenv(DEVICE_PROFILE_NAME_ENV)
    assert (
        resolve_python_profile_name(
            "device",
            Path("/runtime/python312/bin/python3.12"),
            environment_name=DEVICE_PROFILE_NAME_ENV,
        )
        == "device312"
    )


def test_controller_passes_resolved_device_python_to_gripper_manager(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "device-python"
    controller = server.DeviceController(
        config_dir=tmp_path,
        device_python=explicit,
    )
    try:
        assert controller.device_python == explicit
        assert controller.grippers.device_python == explicit
    finally:
        controller.shutdown()


def test_converter_profile_accepts_release_owned_paths(monkeypatch, tmp_path) -> None:
    runtime = tmp_path / "runtime"
    data = tmp_path / "data"
    monkeypatch.setenv(converter_profile.CONVERTER_RUNTIME_ROOT_ENV, str(runtime))
    monkeypatch.setenv(
        converter_profile.CONVERSION_OUTPUT_ROOT_ENV,
        str(data / "conversion_jobs"),
    )
    monkeypatch.setenv(
        converter_profile.CONVERTER_ACTIVATION_RECEIPT_ENV,
        str(data / "activation.json"),
    )

    profile = converter_profile.default_profile(tmp_path)

    assert profile.python == runtime / "converter_py310/bin/python"
    assert profile.ffmpeg == runtime / "ffmpeg-av1/bin/ffmpeg"
    assert profile.ffprobe == runtime / "ffmpeg-av1/bin/ffprobe"
    assert profile.output_root == data / "conversion_jobs"
    assert profile.activation_receipt == data / "activation.json"
