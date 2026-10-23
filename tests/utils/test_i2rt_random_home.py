"""Hardware-free tests for I2RT randomized-home file coordination."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils.i2rt_random_home import (
    RANDOM_HOME_REQUEST_SCHEMA,
    RANDOM_HOME_RESULT_SCHEMA,
    RANDOM_HOME_TARGET_SCHEMA,
    RandomHomeSyncError,
    new_random_home_request_id,
    publish_random_home_request,
    publish_random_home_result,
    publish_random_home_target,
    random_home_result_path,
    read_random_home_request,
    read_random_home_result,
    read_random_home_target,
)


REQUEST_A = "a" * 32
REQUEST_B = "b" * 32


def _home(start: float = 0.0) -> list[float]:
    return [start + index / 10.0 for index in range(14)]


def test_request_and_target_round_trip_preserve_only_matching_identity(
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "random-home-request.json"
    target_path = tmp_path / "random-home-target.json"

    request_id = publish_random_home_request(
        request_path,
        reason="startup",
        request_id=REQUEST_A.upper(),
    )
    publish_random_home_target(
        target_path,
        request_id=request_id,
        home_joint_pos=_home(),
        source_device_id=1,
    )

    assert request_id == REQUEST_A
    assert read_random_home_request(request_path, reason="startup") == REQUEST_A
    assert (
        read_random_home_target(
            target_path,
            request_id=REQUEST_B,
            expected_source_device_id=1,
        )
        is None
    )
    np.testing.assert_allclose(
        read_random_home_target(
            target_path,
            request_id=REQUEST_A,
            expected_source_device_id=1,
        ),
        _home(),
    )


def test_new_request_rejects_previous_episode_target(tmp_path: Path) -> None:
    request_path = tmp_path / "random-home-request.json"
    target_path = tmp_path / "random-home-target.json"
    publish_random_home_request(
        request_path, reason="startup", request_id=REQUEST_A
    )
    publish_random_home_target(
        target_path,
        request_id=REQUEST_A,
        home_joint_pos=_home(),
        source_device_id=1,
    )

    publish_random_home_request(
        request_path, reason="episode_reset", request_id=REQUEST_B
    )

    assert read_random_home_request(request_path, reason="startup") is None
    assert read_random_home_request(request_path, reason="episode_reset") == REQUEST_B
    assert (
        read_random_home_target(
            target_path,
            request_id=REQUEST_B,
            expected_source_device_id=1,
        )
        is None
    )


def test_device_results_require_matching_request_and_device_identity(
    tmp_path: Path,
) -> None:
    pattern = str(tmp_path / "random-home-result-{device_id}.json")
    follower_path = publish_random_home_result(
        pattern,
        request_id=REQUEST_A,
        source_device_id=1,
        success=True,
    )
    publish_random_home_result(
        pattern,
        request_id=REQUEST_A,
        source_device_id=0,
        success=False,
    )

    assert follower_path == random_home_result_path(pattern, source_device_id=1)
    payload = json.loads(follower_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == RANDOM_HOME_RESULT_SCHEMA
    assert read_random_home_result(
        pattern,
        request_id=REQUEST_A,
        source_device_id=1,
    ) is True
    assert read_random_home_result(
        pattern,
        request_id=REQUEST_A,
        source_device_id=0,
    ) is False
    assert read_random_home_result(
        pattern,
        request_id=REQUEST_B,
        source_device_id=1,
    ) is None


def test_result_pattern_requires_one_device_placeholder(tmp_path: Path) -> None:
    with pytest.raises(RandomHomeSyncError, match="device_id"):
        publish_random_home_result(
            tmp_path / "shared-result.json",
            request_id=REQUEST_A,
            source_device_id=1,
            success=True,
        )


def test_result_reader_rejects_boolean_device_identity(tmp_path: Path) -> None:
    pattern = str(tmp_path / "random-home-result-{device_id}.json")
    result_path = random_home_result_path(pattern, source_device_id=1)
    result_path.write_text(
        json.dumps(
            {
                "schema_version": RANDOM_HOME_RESULT_SCHEMA,
                "request_id": REQUEST_A,
                "source_device_id": True,
                "success": True,
                "completed_at_ns": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RandomHomeSyncError, match="source_device_id"):
        read_random_home_result(
            pattern,
            request_id=REQUEST_A,
            source_device_id=1,
        )


def test_request_and_target_payloads_have_versioned_atomic_json(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    target_path = tmp_path / "target.json"

    publish_random_home_request(
        request_path, reason="episode_reset", request_id=REQUEST_A
    )
    publish_random_home_target(
        target_path,
        request_id=REQUEST_A,
        home_joint_pos=_home(),
        source_device_id=1,
    )

    request = json.loads(request_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    assert request["schema_version"] == RANDOM_HOME_REQUEST_SCHEMA
    assert request["request_id"] == REQUEST_A
    assert target["schema_version"] == RANDOM_HOME_TARGET_SCHEMA
    assert target["request_id"] == REQUEST_A
    assert target["source_device_id"] == 1
    assert target["home_joint_pos"] == _home()
    assert not list(tmp_path.glob(".request.json.*"))
    assert not list(tmp_path.glob(".target.json.*"))


def test_atomic_target_publish_keeps_existing_target_and_cleans_temp_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_path = tmp_path / "target.json"
    publish_random_home_target(
        target_path,
        request_id=REQUEST_A,
        home_joint_pos=_home(),
        source_device_id=1,
    )
    before = target_path.read_bytes()

    def fail_replace(*_: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("utils.i2rt_random_home.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        publish_random_home_target(
            target_path,
            request_id=REQUEST_B,
            home_joint_pos=_home(1.0),
            source_device_id=1,
        )

    assert target_path.read_bytes() == before
    assert not list(tmp_path.glob(".target.json.*"))


@pytest.mark.parametrize(
    ("request_id", "home_joint_pos", "source_device_id", "message"),
    [
        ("not-a-valid-request-id", _home(), 1, "request_id"),
        (REQUEST_A, _home()[:-1], 1, "14 finite"),
        (REQUEST_A, _home()[:-1] + [float("nan")], 1, "finite"),
        (REQUEST_A, _home(), True, "source_device_id"),
    ],
)
def test_target_publication_rejects_unsafe_payloads(
    tmp_path: Path,
    request_id: str,
    home_joint_pos: list[float],
    source_device_id: int | bool,
    message: str,
) -> None:
    with pytest.raises(RandomHomeSyncError, match=message):
        publish_random_home_target(
            tmp_path / "target.json",
            request_id=request_id,
            home_joint_pos=home_joint_pos,
            source_device_id=source_device_id,
        )


def test_target_reader_rejects_same_request_from_the_wrong_device(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    publish_random_home_target(
        target_path,
        request_id=REQUEST_A,
        home_joint_pos=_home(),
        source_device_id=0,
    )

    with pytest.raises(RandomHomeSyncError, match="source_device_id"):
        read_random_home_target(
            target_path,
            request_id=REQUEST_A,
            expected_source_device_id=1,
        )


def test_read_rejects_malformed_or_symlink_target(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    target_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RandomHomeSyncError, match="unreadable"):
        read_random_home_target(
            target_path,
            request_id=REQUEST_A,
            expected_source_device_id=1,
        )

    target_path.unlink()
    actual_path = tmp_path / "actual-target.json"
    actual_path.write_text("{}", encoding="utf-8")
    target_path.symlink_to(actual_path)
    with pytest.raises(RandomHomeSyncError, match="safe regular file"):
        read_random_home_target(
            target_path,
            request_id=REQUEST_A,
            expected_source_device_id=1,
        )


def test_request_validation_rejects_bad_schema_and_unknown_reason(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "wrong/schema",
                "request_id": REQUEST_A,
                "reason": "startup",
                "issued_at_ns": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RandomHomeSyncError, match="schema"):
        read_random_home_request(request_path, reason="startup")
    with pytest.raises(RandomHomeSyncError, match="unsupported"):
        publish_random_home_request(request_path, reason="recovery")


def test_generated_request_id_is_canonical_hex() -> None:
    request_id = new_random_home_request_id()

    assert len(request_id) == 32
    assert request_id == request_id.lower()
    assert int(request_id, 16) >= 0
