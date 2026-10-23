"""Hardware-free regressions for Yam SocketCAN sudo recovery."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional

import pytest

from incubator.control_panel import can_permissions
from incubator.control_panel.gripper_permissions import GripperPermissionError


SECRET = b"not-a-real-password-7Yz!"


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, value: Any) -> int:
        data = bytes(value)
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int = 0,
        callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.returncode = returncode
        self.callback = callback
        self.stdin = _FakeStdin()
        self.stdin_stream = self.stdin
        self.killed = False

    def communicate(self, timeout: Optional[float] = None) -> tuple[None, bytes]:
        del timeout
        if self.callback is not None:
            self.callback()
        return None, b""

    def kill(self) -> None:
        self.killed = True


class _CanWorld:
    def __init__(self) -> None:
        self.snapshots = {
            "can0": can_permissions.CanLinkSnapshot(
                interface="can0",
                present=True,
                identity="4:/sys/devices/fake-can0:usb:1-1",
                ifindex=4,
                up=False,
                bitrate=None,
                state="STOPPED",
            )
        }
        self.reads = 0
        self.calls: list[tuple[list[str], dict[str, Any], _FakeProcess]] = []
        self.invalidations = 0

    def read(self, interface: str) -> can_permissions.CanLinkSnapshot:
        self.reads += 1
        return self.snapshots.get(
            interface,
            can_permissions.CanLinkSnapshot(interface=interface, present=False),
        )

    def process_factory(self, argv: list[str], **kwargs: Any) -> _FakeProcess:
        if argv == [can_permissions.SUDO_PATH, "-k"]:
            self.invalidations += 1
            return _FakeProcess()

        def apply() -> None:
            interface = argv[9]
            current = self.snapshots[interface]
            if argv[10:] == ["down"]:
                self.snapshots[interface] = replace(current, up=False, state="STOPPED")
            else:
                self.snapshots[interface] = replace(
                    current,
                    up=True,
                    bitrate=can_permissions.DEFAULT_BITRATE,
                    state="ERROR-ACTIVE",
                )

        process = _FakeProcess(callback=apply)
        self.calls.append((list(argv), dict(kwargs), process))
        return process

    def authorizer(
        self, **kwargs: Any
    ) -> can_permissions.CanInterfacePermissionAuthorizer:
        return can_permissions.CanInterfacePermissionAuthorizer(
            link_reader=self.read,
            process_factory=self.process_factory,
            cache_ttl_s=10.0,
            **kwargs,
        )


def test_status_is_cached_and_requires_down_or_wrong_bitrate_link() -> None:
    world = _CanWorld()
    authorizer = world.authorizer()

    first = authorizer.status(["can0"])
    second = authorizer.status(["can0"])

    assert first["required"] is True
    assert first["affected_interfaces"] == ["can0"]
    assert first["unavailable_interfaces"] == []
    assert first["interfaces"]["can0"]["ready"] is False
    assert second == first
    assert world.reads == 1

    world.snapshots["can0"] = replace(
        world.snapshots["can0"],
        up=True,
        bitrate=500_000,
        state="ERROR-ACTIVE",
    )
    refreshed = authorizer.status(["can0"], refresh=True)
    assert refreshed["required"] is True
    assert refreshed["interfaces"]["can0"]["bitrate"] == 500_000
    assert world.reads == 2


def test_password_runs_only_fixed_ip_link_commands_and_verifies_ready() -> None:
    world = _CanWorld()
    authorizer = world.authorizer()
    secret = bytearray(SECRET)

    result = authorizer.authorize_with_password(secret, "127.0.0.1", ["can0"])

    assert secret == bytearray(len(SECRET))
    assert result["authorized"] is True
    assert result["method"] == "sudo_ip_link"
    assert result["permission"]["required"] is False
    assert len(world.calls) == 2
    assert world.calls[0][0] == [
        can_permissions.SUDO_PATH,
        "-S",
        "-k",
        "-p",
        "",
        "--",
        can_permissions.IP_PATH,
        "link",
        "set",
        "can0",
        "down",
    ]
    assert world.calls[1][0] == [
        can_permissions.SUDO_PATH,
        "-S",
        "-k",
        "-p",
        "",
        "--",
        can_permissions.IP_PATH,
        "link",
        "set",
        "can0",
        "up",
        "type",
        "can",
        "bitrate",
        "1000000",
    ]
    for argv, kwargs, process in world.calls:
        assert SECRET.decode() not in argv
        assert SECRET.decode() not in repr(kwargs["env"])
        assert kwargs["shell"] is False
        assert process.stdin is None
        assert process.stdin_stream.closed is True
        assert b"".join(process.stdin_stream.writes) == SECRET + b"\n"
    assert world.invalidations == 1


def test_unavailable_interface_is_not_misreported_as_password_repair() -> None:
    world = _CanWorld()
    world.snapshots["can0"] = can_permissions.CanLinkSnapshot(
        interface="can0", present=False
    )
    authorizer = world.authorizer()

    status = authorizer.status(["can0"], refresh=True)
    assert status["required"] is False
    assert status["unavailable_interfaces"] == ["can0"]
    with pytest.raises(GripperPermissionError) as captured:
        authorizer.ensure_ready(["can0"])
    assert captured.value.code == "can_interface_unavailable"
    assert world.calls == []


@pytest.mark.parametrize("interface", ["vcan0", "can-0", "can0;id", "../can0"])
def test_unsafe_interface_names_never_reach_ip_or_sudo(interface: str) -> None:
    world = _CanWorld()
    authorizer = world.authorizer()

    with pytest.raises(GripperPermissionError) as captured:
        authorizer.status([interface], refresh=True)
    assert captured.value.code == "unsafe_can_interface"
    assert world.reads == 0
    assert world.calls == []


def test_failed_sudo_is_sanitized_and_zeroes_password() -> None:
    world = _CanWorld()

    def failed_process(argv: list[str], **kwargs: Any) -> _FakeProcess:
        if argv == [can_permissions.SUDO_PATH, "-k"]:
            return _FakeProcess()
        process = _FakeProcess(returncode=1)
        world.calls.append((list(argv), dict(kwargs), process))
        return process

    authorizer = can_permissions.CanInterfacePermissionAuthorizer(
        link_reader=world.read,
        process_factory=failed_process,
    )
    secret = bytearray(SECRET)
    with pytest.raises(GripperPermissionError) as captured:
        authorizer.authorize_with_password(secret, "client", ["can0"])

    assert captured.value.code == "authorization_failed"
    assert SECRET.decode() not in str(captured.value)
    assert secret == bytearray(len(SECRET))
    assert len(world.calls) == 1
    assert world.snapshots["can0"].up is False


def test_identity_change_between_down_and_up_aborts_second_command() -> None:
    world = _CanWorld()

    def replacing_process(argv: list[str], **kwargs: Any) -> _FakeProcess:
        if argv == [can_permissions.SUDO_PATH, "-k"]:
            return _FakeProcess()

        def replace_interface() -> None:
            current = world.snapshots["can0"]
            world.snapshots["can0"] = replace(
                current,
                identity="5:/sys/devices/replugged-can0:usb:1-2",
                ifindex=5,
            )

        process = _FakeProcess(callback=replace_interface)
        world.calls.append((list(argv), dict(kwargs), process))
        return process

    authorizer = can_permissions.CanInterfacePermissionAuthorizer(
        link_reader=world.read,
        process_factory=replacing_process,
        cache_ttl_s=0,
    )
    with pytest.raises(GripperPermissionError) as captured:
        authorizer.authorize_with_password(bytearray(SECRET), "client", ["can0"])

    assert captured.value.code == "can_interface_changed"
    assert len(world.calls) == 1
