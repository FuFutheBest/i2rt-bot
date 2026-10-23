"""Identity migrations must keep physical binding independent of SHM topology."""
from pathlib import Path

import pytest
import yaml

from utils.device_identity import require_capture_id, validate_device_class_ids
from utils.hydra_config import compose_hydra_application_config


@pytest.mark.parametrize("ids", [[1], [0, 0], [0, 2], [True], [0.0], ["0"]])
def test_invalid_identity_sequences(ids):
    with pytest.raises(ValueError):
        validate_device_class_ids([{"device_class_id": value} for value in ids])


def test_old_identity_field_is_rejected():
    with pytest.raises(ValueError, match="removed"):
        validate_device_class_ids([{"device_id": 0, "device_class_id": 0}])


def test_all_device_presets_have_contiguous_ids(repo_root: Path):
    for path in (repo_root / "configs/devices").rglob("*.yaml"):
        document = yaml.safe_load(path.read_text())
        if isinstance(document, dict) and isinstance(document.get("devices"), list):
            try:
                validate_device_class_ids(document["devices"])
            except ValueError as exc:
                pytest.fail(f"{path}: {exc}")


@pytest.mark.parametrize(
    "name",
    [
        "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc",
        "policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc",
    ],
)
def test_capture_override_does_not_change_policy_or_device_identity(
    repo_root: Path,
    monkeypatch,
    name: str,
):
    monkeypatch.setenv("SUPERINFERENCE_CAMERA_LEFT_CAPTURE_ID", "/dev/video98")
    monkeypatch.setenv("SUPERINFERENCE_CAMERA_RIGHT_CAPTURE_ID", "/dev/video99")
    config = compose_hydra_application_config(name, config_dir=repo_root / "configs")
    devices = config["devices"]
    validate_device_class_ids(devices)
    assert [d["device_class_id"] for d in devices] == [0, 1, 2]
    assert [d["capture_id"] for d in devices[:2]] == ["/dev/video98", "/dev/video99"]
    assert config["policy"]["obs"]["devices"] == [
        "WebCameraOpenPi_0", "WebCameraOpenPi_1", "RizonRobot_2"
    ]


def test_capture_is_explicit_and_environment_numeric_strings_are_indices():
    assert require_capture_id("4") == 4
    assert require_capture_id("/dev/v4l/by-id/example") == "/dev/v4l/by-id/example"
    for value in (None, "", -1, True):
        with pytest.raises(ValueError):
            require_capture_id(value)
