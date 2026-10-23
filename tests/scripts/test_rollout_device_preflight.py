from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.diagnostics import rollout_device_preflight as preflight
from scripts.diagnostics.rollout_device_preflight import (
    RolloutPreflightError,
    _ask_side,
    _specs_from_config,
)


def _config(max_width: object) -> dict[str, object]:
    return {
        "devices": [
            {
                "gripper": {
                    "port": ["/dev/ttyUSB0", "/dev/ttyUSB1"],
                    "motor_id": [1, 1],
                    "gripper_type": "soft",
                    "max_width": max_width,
                }
            }
        ]
    }


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, [0.0925, 0.0925]),
        (0.08, [0.08, 0.08]),
        ([0.07, None], [0.07, 0.0925]),
    ],
)
def test_specs_accept_null_scalar_and_per_gripper_max_width(
    configured: object, expected: list[float], tmp_path: Path
) -> None:
    specs = _specs_from_config(_config(configured), tmp_path)

    assert [spec.max_width for spec in specs] == expected


@pytest.mark.parametrize(("answer", "expected"), [("l", "left"), ("r", "right")])
def test_ask_side_accepts_short_keys(answer: str, expected: str) -> None:
    assert _ask_side("side", lambda _: answer) == expected


def test_relative_calibration_paths_are_repository_relative(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    assert preflight._resolve_calibration_path("calibration/gripper.json", tmp_path / "configs/devices/infer/rig.yaml") == (
        tmp_path / "calibration/gripper.json"
    )


def test_leaf_device_config_is_rejected_before_hardware_access(monkeypatch: pytest.MonkeyPatch) -> None:
    leaf = preflight.REPO_ROOT / "configs/devices/infer/infer_rizon_fisheye_umi3_dual_openpi.yaml"
    monkeypatch.setattr(preflight, "discover_serial_ports", lambda: pytest.fail("hardware accessed"))

    with pytest.raises(RolloutPreflightError, match="顶层 rollout 配置"):
        preflight.bind_devices(leaf)


def test_gripper_keeps_moving_until_side_is_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    stopped = threading.Event()

    def fake_exercise(
        port: str,
        spec: preflight.GripperSpec,
        *,
        settle_s: float = 0.5,
        stop_event: threading.Event | None = None,
    ) -> None:
        assert port == "/dev/ttyUSB3"
        assert spec.motor_id == 1
        assert settle_s == 0.5
        assert stop_event is not None
        started.set()
        assert stop_event.wait(1.0)
        stopped.set()

    monkeypatch.setattr(preflight, "exercise_gripper", fake_exercise)
    spec = preflight.GripperSpec("/dev/ttyUSB3", 1, None, 0.0925, "soft")

    side = preflight.identify_gripper(
        "/dev/ttyUSB3",
        spec,
        lambda prompt: "r" if started.wait(1.0) and "[l/r]" in prompt else pytest.fail("motion not started"),
    )

    assert side == "right"
    assert stopped.is_set()


def test_identification_uses_calibration_port_when_config_binding_is_stale() -> None:
    left = preflight.GripperSpec("/dev/ttyUSB2", 1, None, 0.0925, "soft", {"port": "/dev/ttyUSB1"})
    right = preflight.GripperSpec("/dev/ttyUSB3", 1, None, 0.0925, "soft", {"port": "/dev/ttyUSB0"})
    assert preflight._identification_spec("/dev/ttyUSB0", [left, right]) is right
    assert preflight._identification_spec("/dev/ttyUSB1", [left, right]) is left
    with pytest.raises(RolloutPreflightError, match="唯一匹配"):
        preflight._identification_spec("/dev/ttyUSB4", [left, right])
    with pytest.raises(RolloutPreflightError, match="唯一匹配"):
        preflight._identification_spec("/dev/ttyUSB1", [left, left])


def test_gripper_targets_repeat_at_requested_interval_until_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    moves = []
    waits = []
    disabled = []
    gripper = SimpleNamespace(
        Enable=lambda _: None,
        Move=lambda width, velocity: moves.append((width, velocity)),
        Disable=lambda: disabled.append(True),
    )
    monkeypatch.setitem(
        sys.modules,
        "third_party.lktech_motor_sdk.example.lktech_gripper",
        SimpleNamespace(Gripper=lambda **_: gripper),
    )

    def wait(timeout: float) -> bool:
        waits.append(timeout)
        return len(waits) == 4

    stop = SimpleNamespace(is_set=lambda: False, wait=wait)
    spec = preflight.GripperSpec("/dev/ttyUSB0", 1, None, 0.0925, "soft")
    preflight.exercise_gripper(spec.port, spec, stop_event=stop)

    assert [width for width, _ in moves] == pytest.approx([0.02775, 0.06475] * 2)
    assert waits == [0.5] * 4
    assert disabled == [True]


@pytest.mark.parametrize("key", [ord("s"), ord("S"), ord("q"), ord("Q"), 27])
def test_camera_skip_and_abort_release_preview(monkeypatch: pytest.MonkeyPatch, key: int) -> None:
    released = []
    closed = []
    capture = SimpleNamespace(
        isOpened=lambda: True,
        read=lambda: (True, object()),
        release=lambda: released.append(True),
    )
    cv2 = SimpleNamespace(
        VideoCapture=lambda *_: capture,
        CAP_V4L2=0,
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=0,
        putText=lambda *_: None,
        imshow=lambda *_: None,
        waitKey=lambda _: key,
        destroyWindow=closed.append,
        error=type("OpenCVError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    if key in (ord("s"), ord("S")):
        assert preflight.identify_camera("/dev/video9") is None
    else:
        with pytest.raises(RolloutPreflightError, match="操作员已终止"):
            preflight.identify_camera("/dev/video9")

    assert released == [True]
    assert closed == ["/dev/video9"]


@pytest.mark.parametrize("mutate_identity", [False, True])
def test_top_level_config_updates_camera_ids_and_referenced_device_ports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutate_identity: bool
) -> None:
    config_root = tmp_path / "configs"
    device_root = config_root / "devices" / "infer"
    top_level = config_root / "policy_infer" / "openpi" / "rollout.yaml"
    device = device_root / "rig.yaml"
    top_level.parent.mkdir(parents=True)
    device_root.mkdir(parents=True)
    top_level.write_text(
        "defaults:\n  - /devices/infer@devices: rig\n  - _self_\ncamera:\n  left_capture_id: 0\n  right_capture_id: 2\n",
        encoding="utf-8",
    )
    device.write_text(
        "devices:\n"
        "  - class: WebCameraOpenPi\n    device_class_id: 0\n    capture_id: ${camera.left_capture_id}\n"
        "  - class: WebCameraOpenPi\n    device_class_id: 1\n    capture_id: ${camera.right_capture_id}\n"
        "  - class: RizonRobot\n    device_class_id: 2\n    robot_sn: [old-left, old-right]\n"
        "    gripper:\n      port: [/dev/ttyUSB2, /dev/ttyUSB3]\n"
        "      motor_id: [1, 1]\n      gripper_type: soft\n      max_width: null\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(preflight, "exercise_gripper", lambda *_, **__: None)
    answers = iter(("b,a", "r", "l"))
    camera_sides = iter((None, "right", "left"))

    original_top = top_level.read_bytes()
    original_devices = device.read_bytes()
    if mutate_identity:
        original_bind = preflight.update_robot_binding_from_elements

        def invalid_bind(document, *args, **kwargs):
            document["devices"][0]["device_class_id"] = 99
            return original_bind(document, *args, **kwargs)

        monkeypatch.setattr(preflight, "update_robot_binding_from_elements", invalid_bind)

    def bind():
        return preflight.bind_devices(
            top_level,
            serial_ports=["/dev/ttyUSB2", "/dev/ttyUSB3"],
            video_devices=["/dev/video1", "/dev/video2", "/dev/video4"],
            input_fn=lambda _: next(answers),
            camera_side_fn=lambda _: next(camera_sides),
            robot_discovery_fn=lambda: SimpleNamespace(
                robots=[{"rdk_sn": "right-sn"}, {"rdk_sn": "left-sn"}, {"rdk_sn": "unused-sn"}],
                interfaces=[],
            ),
        )

    if mutate_identity:
        with pytest.raises(RolloutPreflightError, match="不能修改 device_class_id"):
            bind()
        assert top_level.read_bytes() == original_top
        assert device.read_bytes() == original_devices
        return

    result = bind()

    yaml = pytest.importorskip(
        "ruamel.yaml",
        reason="round-trip YAML assertions require the optional ruamel dependency",
    ).YAML()
    with result.open(encoding="utf-8") as stream:
        top_doc = yaml.load(stream)
    with device.open(encoding="utf-8") as stream:
        device_doc = yaml.load(stream)
    assert top_doc["camera"] == {"left_capture_id": 4, "right_capture_id": 2}
    assert device_doc["devices"][2]["gripper"]["port"] == ["/dev/ttyUSB3", "/dev/ttyUSB2"]
    assert device_doc["devices"][2]["robot_sn"] == ["left-sn", "right-sn"]
    assert [d["device_class_id"] for d in device_doc["devices"]] == [0, 1, 2]
    assert device_doc["devices"][0]["capture_id"] == "${camera.left_capture_id}"
    assert device_doc["devices"][1]["capture_id"] == "${camera.right_capture_id}"


def test_elements_cache_updates_robot_serials_and_interface() -> None:
    doc = {
        "devices": [
            {
                "class": "RizonRobot",
                "robot_sn": ["old-left", "old-right"],
                "network_interface_whitelist": ["10.0.0.2"],
            }
        ]
    }
    snapshot = SimpleNamespace(
        robots=[
            {"model": "Rizon4", "rdk_sn": "Rizon4-063256", "ip": "192.168.1.101"},
            {"model": "Rizon4R", "rdk_sn": "Rizon4R-062088", "ip": "192.168.1.100"},
        ],
        interfaces=[
            {"name": "eth0", "ipv4": "192.168.1.16", "network": "192.168.1.0/24"},
            {"name": "wlan0", "ipv4": "10.0.0.2", "network": "10.0.0.0/24"},
        ],
    )

    messages = preflight.update_robot_binding_from_elements(doc, snapshot)

    robot = doc["devices"][0]
    assert robot["robot_sn"] == ["Rizon4-063256", "Rizon4R-062088"]
    assert robot["network_interface_whitelist"] == ["192.168.1.16"]
    assert any("已更新设备配置的 robot_sn" in message for message in messages)
    assert any("已更新设备配置的 network_interface_whitelist" in message for message in messages)


def test_ambiguous_elements_cache_does_not_change_robot_config() -> None:
    robot = {
        "class": "RizonRobot",
        "robot_sn": ["configured-left", "configured-right"],
        "network_interface_whitelist": ["192.168.1.16"],
    }
    doc = {"devices": [robot]}
    snapshot = SimpleNamespace(
        robots=[
            {"model": "Rizon4", "rdk_sn": "Rizon4-000001", "ip": "192.168.1.101"},
            {"model": "Rizon4", "rdk_sn": "Rizon4-000002", "ip": "192.168.1.102"},
            {"model": "Rizon4R", "rdk_sn": "Rizon4R-000003", "ip": "192.168.1.100"},
        ],
        interfaces=[],
    )

    messages = preflight.update_robot_binding_from_elements(doc, snapshot)

    assert robot["robot_sn"] == ["configured-left", "configured-right"]
    assert robot["network_interface_whitelist"] == ["192.168.1.16"]
    assert any("无法唯一确定" in message for message in messages)


@pytest.mark.parametrize("configured", ["old", ["old"], ["left", "right"], ["one", "two", "three"]])
def test_robot_selection_respects_config_count_and_order(configured: object) -> None:
    count = 1 if isinstance(configured, str) else len(configured)
    robot = {"class": "RizonRobot", "robot_sn": configured}
    records = [{"rdk_sn": f"robot-{i}", "ip": f"192.168.1.{100 + i}"} for i in range(4)]
    snapshot = SimpleNamespace(
        robots=[*records, records[0], {"model": "no-serial"}],
        interfaces=[{"name": "eth0", "ipv4": "192.168.1.16", "network": "192.168.1.0/24"}],
    )
    answers = iter(["", "z", "a,a", "a,b,c,d", ",".join(["D", "B", "A"][:count])])
    preflight.update_robot_binding_from_elements(
        {"devices": [robot]}, snapshot, input_fn=lambda _: next(answers)
    )
    expected = ["robot-3", "robot-1", "robot-0"][:count]
    assert robot["robot_sn"] == (expected[0] if isinstance(configured, str) else expected)
    assert robot["network_interface_whitelist"] == ["192.168.1.16"]


@pytest.mark.parametrize("answer", ["abort", None])
def test_robot_selection_abort_preserves_config(answer: str | None) -> None:
    robot = {"class": "RizonRobot", "robot_sn": ["left", "right"]}
    snapshot = SimpleNamespace(robots=[{"rdk_sn": "a-sn"}, {"rdk_sn": "b-sn"}], interfaces=[])

    def respond(_: str) -> str:
        if answer is None:
            raise EOFError
        return answer

    with pytest.raises(RolloutPreflightError):
        preflight.update_robot_binding_from_elements({"devices": [robot]}, snapshot, input_fn=respond)
    assert robot["robot_sn"] == ["left", "right"]


@pytest.mark.parametrize("answer", ["s", "S", "skip", "SKIP"])
def test_robot_selection_skip_preserves_current_binding(answer: str) -> None:
    robot = {
        "class": "RizonRobot",
        "robot_sn": ["configured-left", "configured-right"],
        "network_interface_whitelist": ["10.0.0.2"],
    }
    snapshot = SimpleNamespace(
        robots=[
            {"model": "Rizon4", "rdk_sn": "new-left", "ip": "192.168.1.101"},
            {"model": "Rizon4R", "rdk_sn": "new-right", "ip": "192.168.1.102"},
            {"model": "Rizon4", "rdk_sn": "other-left", "ip": "192.168.1.103"},
        ],
        interfaces=[{"name": "eth0", "ipv4": "192.168.1.16", "network": "192.168.1.0/24"}],
    )
    prompts: list[str] = []

    messages = preflight.update_robot_binding_from_elements(
        {"devices": [robot]},
        snapshot,
        input_fn=lambda prompt: prompts.append(prompt) or answer,
    )

    assert "s/skip 保留当前配置" in prompts[0]
    assert robot["robot_sn"] == ["configured-left", "configured-right"]
    assert robot["network_interface_whitelist"] == ["10.0.0.2"]
    assert any("已跳过机器人绑定" in message for message in messages)


def test_robot_selection_labels_reserve_skip_shortcut() -> None:
    labels = [preflight._robot_selection_label(index) for index in range(700)]

    assert len(labels) == len(set(labels))
    assert "s" not in labels
    assert "skip" not in labels


def test_robot_selection_skip_allows_insufficient_candidates() -> None:
    robot = {
        "class": "RizonRobot",
        "robot_sn": ["configured-left", "configured-middle", "configured-right"],
        "network_interface_whitelist": ["10.0.0.2"],
    }
    snapshot = SimpleNamespace(
        robots=[{"rdk_sn": "one-sn"}, {"rdk_sn": "two-sn"}],
        interfaces=[],
    )

    messages = preflight.update_robot_binding_from_elements(
        {"devices": [robot]}, snapshot, input_fn=lambda _: "skip"
    )

    assert robot["robot_sn"] == ["configured-left", "configured-middle", "configured-right"]
    assert robot["network_interface_whitelist"] == ["10.0.0.2"]
    assert any("已跳过机器人绑定" in message for message in messages)


def test_robot_selection_rejects_insufficient_candidates() -> None:
    robot = {"class": "RizonRobot", "robot_sn": ["one", "two", "three"]}
    snapshot = SimpleNamespace(robots=[{"rdk_sn": "a-sn"}, {"rdk_sn": "b-sn"}], interfaces=[])
    with pytest.raises(RolloutPreflightError, match="只发现 2"):
        preflight.update_robot_binding_from_elements(
            {"devices": [robot]}, snapshot, input_fn=lambda _: "a,b"
        )
    assert robot["robot_sn"] == ["one", "two", "three"]


def test_main_prints_ready_message_after_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    monkeypatch.setattr(preflight, "force_cleanup_requested", lambda: False)
    monkeypatch.setattr(
        preflight,
        "cleanup_before_startup",
        lambda *_args, **_kwargs: SimpleNamespace(candidates=[]),
    )
    monkeypatch.setattr(preflight, "bind_devices", lambda *_args, **_kwargs: config_path)

    assert preflight.main([str(config_path)]) == 0

    captured = capsys.readouterr()
    assert "可以直接启动推理，无须修改配置" in captured.out


def test_main_does_not_print_ready_message_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    monkeypatch.setattr(preflight, "force_cleanup_requested", lambda: False)
    monkeypatch.setattr(
        preflight,
        "cleanup_before_startup",
        lambda *_args, **_kwargs: SimpleNamespace(candidates=[]),
    )

    def fail_bind(*_args: object, **_kwargs: object) -> None:
        raise preflight.RolloutPreflightError("device discovery failed")

    monkeypatch.setattr(preflight, "bind_devices", fail_bind)

    assert preflight.main([str(config_path)]) == 2

    captured = capsys.readouterr()
    assert "可以直接启动推理，无须修改配置" not in captured.out
