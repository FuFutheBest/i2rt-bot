"""Controller and API contracts for unified ttyUSB/Yam CAN authorization."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from incubator.control_panel import gripper_control, server
from incubator.control_panel.can_permissions import CanPermissionRequiredError
from incubator.control_panel.gripper_permissions import GripperPermissionError


SECRET = b"not-a-real-password-7Yz!"


class _CanAuthorizer:
    bitrate = 1_000_000

    def __init__(self) -> None:
        self.ready = False
        self.present = True
        self.password_calls = 0
        self.ensure_calls = 0
        self.last_secret: Optional[bytearray] = None
        self.last_interfaces: list[str] = []

    def status(self, interfaces: Any = (), **_kwargs: Any) -> dict[str, Any]:
        targets = sorted(set(map(str, interfaces)))
        unavailable = [] if self.present else list(targets)
        affected = list(targets) if self.present and not self.ready else []
        return {
            "required": bool(affected),
            "target_interfaces": targets,
            "affected_interfaces": affected,
            "unavailable_interfaces": unavailable,
            "expected_bitrate": self.bitrate,
            "context_id": f"can-context-{self.present}-{self.ready}",
            "interfaces": {
                interface: {
                    "present": self.present,
                    "up": self.ready,
                    "bitrate": self.bitrate if self.ready else None,
                    "expected_bitrate": self.bitrate,
                    "ready": self.ready,
                }
                for interface in targets
            },
        }

    def ensure_ready(self, interfaces: Any) -> dict[str, Any]:
        self.ensure_calls += 1
        status = self.status(interfaces)
        if status["unavailable_interfaces"]:
            raise GripperPermissionError(
                "can_interface_unavailable",
                "missing CAN",
                status=409,
                details={
                    "unavailable_can_interfaces": status["unavailable_interfaces"]
                },
            )
        if status["affected_interfaces"]:
            raise CanPermissionRequiredError(status["affected_interfaces"])
        return status

    def authorize_with_password(
        self, secret: bytearray, _client: str, interfaces: Any
    ) -> dict[str, Any]:
        self.password_calls += 1
        self.last_secret = secret
        self.last_interfaces = list(interfaces)
        assert bytes(secret) == SECRET
        self.ready = True
        return {
            "ok": True,
            "authorized": True,
            "method": "sudo_ip_link",
            "interfaces": list(interfaces),
            "permission": self.status(interfaces),
        }


@pytest.fixture
def can_permission_app(
    tmp_path: Path,
) -> tuple[Any, server.DeviceController, _CanAuthorizer]:
    can_authorizer = _CanAuthorizer()
    manager = gripper_control.GripperControlManager(auto_monitor=False)
    manager.configured_ports = lambda: {}  # type: ignore[method-assign]
    conversion_jobs = server.ConversionJobManager(
        queue_root=tmp_path / "conversion_jobs", auto_start=False
    )
    app = server.create_app(
        config_dir=tmp_path,
        gripper_control=manager,
        can_permissions=can_authorizer,
        conversion_jobs=conversion_jobs,
    )
    app.config.update(TESTING=True, GRIPPER_PERMISSION_CSRF_TOKEN="csrf-test")
    controller = app.config["DEVICE_CONTROLLER"]
    config = {
        "devices": [
            {
                "class": "YamHeadRobot",
                "device_class_id": 0,
                "enabled": True,
                "yam_channel": "can0",
                "yam_sim": False,
            }
        ]
    }
    controller.selected_config = "yam-config"
    controller._base_config = config
    controller._effective_config = config
    controller.revision = 7
    yield app, controller, can_authorizer
    controller.starter = None
    controller.workflow = None
    controller.shutdown()
    conversion_jobs.close()


def _headers(origin: str = "http://panel.local") -> dict[str, str]:
    return {
        "Origin": origin,
        "Sec-Fetch-Site": "same-origin",
        "X-CSRF-Token": "csrf-test",
    }


def test_state_exposes_can_only_permission_prompt_context(
    can_permission_app: tuple[Any, server.DeviceController, _CanAuthorizer],
) -> None:
    _app, controller, _can_authorizer = can_permission_app

    permission = controller.gripper_state()["grippers"]["permission"]

    assert permission["required"] is True
    assert permission["tty_required"] is False
    assert permission["can_required"] is True
    assert permission["target_ports"] == []
    assert permission["target_can_interfaces"] == ["can0"]
    assert permission["affected_can_interfaces"] == ["can0"]
    assert permission["expected_can_bitrate"] == 1_000_000


def test_hardware_permission_route_repairs_can_and_zeroes_secret(
    can_permission_app: tuple[Any, server.DeviceController, _CanAuthorizer],
) -> None:
    app, controller, can_authorizer = can_permission_app

    response = app.test_client().post(
        "/api/hardware/permissions/sudo",
        base_url="http://panel.local",
        headers=_headers(),
        json={
            "revision": 7,
            "ports": [],
            "can_interfaces": ["can0"],
            "password": SECRET.decode(),
        },
    )

    assert response.status_code == 200
    assert response.get_json()["permission"]["required"] is False
    assert can_authorizer.password_calls == 1
    assert can_authorizer.last_interfaces == ["can0"]
    assert can_authorizer.last_secret == bytearray(len(SECRET))
    assert controller.gripper_state()["grippers"]["permission"]["required"] is False
    assert SECRET not in response.data


@pytest.mark.parametrize(
    ("revision", "interfaces"),
    [(6, ["can0"]), (7, ["can1"])],
)
def test_hardware_permission_route_rejects_stale_context(
    can_permission_app: tuple[Any, server.DeviceController, _CanAuthorizer],
    revision: int,
    interfaces: list[str],
) -> None:
    app, _controller, can_authorizer = can_permission_app

    response = app.test_client().post(
        "/api/hardware/permissions/sudo",
        base_url="http://panel.local",
        headers=_headers(),
        json={
            "revision": revision,
            "ports": [],
            "can_interfaces": interfaces,
            "password": SECRET.decode(),
        },
    )

    assert response.status_code == 409
    assert response.get_json()["code"] == "permission_context_changed"
    assert can_authorizer.password_calls == 0


def test_yam_start_is_rejected_before_spawning_when_can_needs_setup(
    can_permission_app: tuple[Any, server.DeviceController, _CanAuthorizer],
) -> None:
    _app, controller, can_authorizer = can_permission_app
    controller.starter = SimpleNamespace()

    with pytest.raises(CanPermissionRequiredError) as captured:
        controller.control_device("start", "YamHeadRobot", 0)

    assert captured.value.code == "can_interface_permission_required"
    assert can_authorizer.ensure_calls == 1
    assert controller._device_operations == {}


def test_missing_can_is_reported_without_requesting_password(
    can_permission_app: tuple[Any, server.DeviceController, _CanAuthorizer],
) -> None:
    _app, controller, can_authorizer = can_permission_app
    can_authorizer.present = False

    permission = controller.gripper_state()["grippers"]["permission"]
    assert permission["required"] is False
    assert permission["unavailable_can_interfaces"] == ["can0"]
    controller.starter = SimpleNamespace()
    with pytest.raises(GripperPermissionError) as captured:
        controller.control_device("start", "YamHeadRobot", 0)
    assert captured.value.code == "can_interface_unavailable"
    assert can_authorizer.password_calls == 0


def test_simulated_yam_does_not_require_host_can(
    can_permission_app: tuple[Any, server.DeviceController, _CanAuthorizer],
) -> None:
    _app, controller, _can_authorizer = can_permission_app
    controller._effective_config["devices"][0]["yam_sim"] = True

    permission = controller.gripper_state()["grippers"]["permission"]

    assert permission["target_can_interfaces"] == []
    assert permission["required"] is False
