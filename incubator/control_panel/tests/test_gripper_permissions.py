"""Security regressions for ttyUSB sudo permission recovery.

No test invokes a real sudo/chmod command or opens serial hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from types import SimpleNamespace
import logging
import stat
import subprocess
import threading
import time
from typing import Any, Callable, Optional

import pytest
from werkzeug.test import EnvironBuilder

from incubator.control_panel import gripper_control, server
from incubator.control_panel import gripper_permissions as permissions


SECRET = b'not-a-real-password-7Yz!'


@dataclass
class _FakeDevice:
    path: str
    st_dev: int
    st_ino: int
    st_rdev: int
    mode: int = 0o600
    present: bool = True
    symlink: bool = False

    def metadata(self) -> Any:
        kind = stat.S_IFLNK if self.symlink else stat.S_IFCHR
        return SimpleNamespace(
            st_mode=kind | self.mode,
            st_dev=self.st_dev,
            st_ino=self.st_ino,
            st_rdev=self.st_rdev,
            st_uid=0,
            st_gid=0,
        )


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
        stderr: bytes = b'',
        on_communicate: Optional[Callable[[], None]] = None,
        timeout: bool = False,
        wait_event: Optional[threading.Event] = None,
    ) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdin = _FakeStdin()
        self.stdin_stream = self.stdin
        self.on_communicate = on_communicate
        self.timeout = timeout
        self.wait_event = wait_event
        self.killed = False

    def communicate(self, timeout: Optional[float] = None) -> tuple[None, bytes]:
        del timeout
        if self.timeout and not self.killed:
            raise subprocess.TimeoutExpired('sudo', 0.01)
        if self.wait_event is not None:
            self.wait_event.wait(2.0)
        if self.on_communicate is not None:
            self.on_communicate()
        return None, self.stderr

    def kill(self) -> None:
        self.killed = True


class _FakeWorld:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, count: int = 2) -> None:
        self.devices = {
            f'/dev/ttyUSB{index}': _FakeDevice(
                f'/dev/ttyUSB{index}',
                st_dev=10,
                st_ino=100 + index,
                st_rdev=200 + index,
            )
            for index in range(count)
        }
        self.descriptors: dict[int, _FakeDevice] = {}
        self.next_descriptor = 30
        self.accessible: set[str] = set()
        self.closed: list[int] = []
        self.process_calls: list[tuple[list[str], dict[str, Any], _FakeProcess]] = []
        self.invalidation_calls = 0
        self.process_builder: Callable[[], _FakeProcess] = self.success_process

        monkeypatch.setattr(permissions.os, 'O_PATH', 0x200000, raising=False)
        monkeypatch.setattr(permissions.os, 'O_NOFOLLOW', 0x20000, raising=False)
        monkeypatch.setattr(permissions.os, 'O_CLOEXEC', 0x80000, raising=False)
        monkeypatch.setattr(permissions.os, 'lstat', self.lstat)
        monkeypatch.setattr(permissions.os, 'open', self.open)
        monkeypatch.setattr(permissions.os, 'fstat', self.fstat)
        monkeypatch.setattr(permissions.os, 'close', self.close)
        monkeypatch.setattr(permissions.os, 'getpid', lambda: 4321)

    def paths(self) -> list[str]:
        return [path for path, device in sorted(self.devices.items()) if device.present]

    def lstat(self, path: Any) -> Any:
        device = self.devices.get(str(path))
        if device is None or not device.present:
            raise FileNotFoundError(str(path))
        return device.metadata()

    def open(self, path: Any, flags: int) -> int:
        assert flags & permissions.os.O_PATH
        device = self.devices[str(path)]
        descriptor = self.next_descriptor
        self.next_descriptor += 1
        self.descriptors[descriptor] = device
        return descriptor

    def fstat(self, descriptor: int) -> Any:
        return self.descriptors[descriptor].metadata()

    def close(self, descriptor: int) -> None:
        self.closed.append(descriptor)
        self.descriptors.pop(descriptor, None)

    def access(self, path: str) -> bool:
        return path in self.accessible

    def apply_permissions(self) -> None:
        for device in self.devices.values():
            if device.present:
                device.mode = 0o666
                self.accessible.add(device.path)

    def success_process(self) -> _FakeProcess:
        return _FakeProcess(on_communicate=self.apply_permissions)

    def process_factory(self, argv: list[str], **kwargs: Any) -> _FakeProcess:
        if argv == [permissions.SUDO_PATH, '-k']:
            self.invalidation_calls += 1
            return _FakeProcess()
        process = self.process_builder()
        self.process_calls.append((list(argv), dict(kwargs), process))
        return process

    def authorizer(self, **kwargs: Any) -> permissions.TtyUsbPermissionAuthorizer:
        return permissions.TtyUsbPermissionAuthorizer(
            port_discovery=self.paths,
            access_check=self.access,
            process_factory=self.process_factory,
            ownership_check=lambda _metadata: False,
            **kwargs,
        )


def test_password_uses_stdin_fixed_argv_and_chmods_all_opened_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch, count=3)
    authorizer = world.authorizer()
    secret = bytearray(SECRET)

    result = authorizer.authorize_with_password(
        secret, '127.0.0.1', world.paths()
    )

    assert result['authorized'] is True
    assert result['ports'] == ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2']
    assert secret == bytearray(len(SECRET))
    assert len(world.process_calls) == 1
    argv, kwargs, process = world.process_calls[0]
    assert argv[:8] == [
        permissions.SUDO_PATH,
        '-S',
        '-k',
        '-p',
        '',
        '--',
        permissions.CHMOD_PATH,
        '0666',
    ]
    assert argv[8] == '--'
    assert argv[9:] == [
        '/proc/4321/fd/30',
        '/proc/4321/fd/31',
        '/proc/4321/fd/32',
    ]
    assert not any('/dev/ttyUSB' in argument for argument in argv)
    assert not any('helper' in argument.casefold() for argument in argv)
    assert not any(SECRET.decode() in argument for argument in argv)
    assert kwargs['shell'] is False
    assert kwargs['close_fds'] is True
    assert kwargs['stderr'] == subprocess.DEVNULL
    assert SECRET.decode() not in repr(kwargs['env'])
    assert process.stdin is None
    assert process.stdin_stream.closed is True
    assert b''.join(process.stdin_stream.writes) == SECRET + b'\n'
    assert world.invalidation_calls == 1
    assert all(device.mode == 0o666 for device in world.devices.values())
    assert sorted(world.closed) == [30, 31, 32]


def test_only_current_configured_gripper_ports_are_chmod_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch, count=3)

    def apply_selected() -> None:
        device = world.devices['/dev/ttyUSB1']
        device.mode = 0o666
        world.accessible.add(device.path)

    world.process_builder = lambda: _FakeProcess(on_communicate=apply_selected)
    authorizer = world.authorizer()
    result = authorizer.authorize_with_password(
        bytearray(SECRET), 'client-a', ['/dev/ttyUSB1']
    )
    assert result['ports'] == ['/dev/ttyUSB1']
    argv, _kwargs, _process = world.process_calls[0]
    assert argv[9:] == ['/proc/4321/fd/30']
    assert world.devices['/dev/ttyUSB1'].mode == 0o666
    assert world.devices['/dev/ttyUSB0'].mode == 0o600
    assert world.devices['/dev/ttyUSB2'].mode == 0o600


def test_secret_is_never_in_argv_environment_error_or_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    world = _FakeWorld(monkeypatch)
    world.process_builder = lambda: _FakeProcess(
        returncode=1,
        stderr=b'Sorry, try again.\nsudo: 1 incorrect password attempt\n',
    )
    authorizer = world.authorizer()
    secret = bytearray(SECRET)
    caplog.set_level(logging.DEBUG)

    with pytest.raises(permissions.GripperPermissionError) as raised:
        authorizer.authorize_with_password(secret, 'client-a', world.paths())

    assert raised.value.code == 'authorization_failed'
    assert raised.value.status == 403
    assert SECRET.decode() not in str(raised.value)
    assert SECRET.decode() not in repr(raised.value.details)
    assert SECRET.decode() not in caplog.text
    argv, kwargs, _process = world.process_calls[0]
    assert SECRET.decode() not in repr(argv)
    assert SECRET.decode() not in repr(kwargs)
    assert secret == bytearray(len(SECRET))


def test_missing_sudo_is_distinct_and_zeroes_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch)

    def missing(_argv: list[str], **_kwargs: Any) -> Any:
        raise FileNotFoundError('sudo')

    authorizer = permissions.TtyUsbPermissionAuthorizer(
        port_discovery=world.paths,
        access_check=world.access,
        process_factory=missing,
    )
    secret = bytearray(SECRET)
    with pytest.raises(permissions.GripperPermissionError) as raised:
        authorizer.authorize_with_password(secret, 'client-a', world.paths())
    assert raised.value.code == 'authorization_failed'
    assert secret == bytearray(len(SECRET))


def test_timeout_kills_process_closes_stdin_and_zeroes_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch)
    process = _FakeProcess(timeout=True)
    world.process_builder = lambda: process
    authorizer = world.authorizer(timeout_s=0.1)
    secret = bytearray(SECRET)

    with pytest.raises(permissions.GripperPermissionError) as raised:
        authorizer.authorize_with_password(secret, 'client-a', world.paths())

    assert raised.value.code == 'sudo_timeout'
    assert process.killed is True
    assert process.stdin is None
    assert secret == bytearray(len(SECRET))


def test_single_concurrency_guard_does_not_wait_or_consume_second_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch)
    release = threading.Event()
    entered = threading.Event()

    class _BlockingProcess(_FakeProcess):
        def communicate(self, timeout: Optional[float] = None) -> tuple[None, bytes]:
            del timeout
            entered.set()
            release.wait(2.0)
            world.apply_permissions()
            return None, b''

    world.process_builder = _BlockingProcess
    authorizer = world.authorizer()
    first_secret = bytearray(SECRET)
    result: dict[str, Any] = {}

    def first() -> None:
        result.update(authorizer.authorize_with_password(
            first_secret, 'client-a', world.paths()
        ))

    thread = threading.Thread(target=first)
    thread.start()
    assert entered.wait(1.0)
    second_secret = bytearray(b'second-placeholder')
    with pytest.raises(permissions.GripperPermissionError) as raised:
        authorizer.authorize_with_password(
            second_secret, 'client-b', world.paths()
        )
    assert raised.value.code == 'permission_repair_busy'
    assert second_secret == bytearray(len(b'second-placeholder'))
    release.set()
    thread.join(2.0)
    assert not thread.is_alive()
    assert result['authorized'] is True


def test_rate_limit_is_shared_by_password_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch)
    world.process_builder = lambda: _FakeProcess(
        returncode=1,
        stderr=b'incorrect password',
    )
    now = [10.0]
    authorizer = world.authorizer(
        monotonic=lambda: now[0],
        rate_attempts=2,
        global_rate_attempts=10,
    )
    for _index in range(2):
        with pytest.raises(permissions.GripperPermissionError) as raised:
            authorizer.authorize_with_password(
                bytearray(SECRET), 'client-a', world.paths()
            )
        assert raised.value.code == 'authorization_failed'
    with pytest.raises(permissions.GripperPermissionError) as raised:
        authorizer.authorize_with_password(
            bytearray(SECRET), 'client-a', world.paths()
        )
    assert raised.value.code == 'permission_repair_rate_limited'
    assert raised.value.retry_after == 60
    assert len(world.process_calls) == 2


def test_rate_identity_table_has_ttl_cleanup_and_lru_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch, count=1)
    now = [1.0]
    authorizer = world.authorizer(
        monotonic=lambda: now[0],
        rate_attempts=1000,
        global_rate_attempts=1000,
    )
    for index in range(permissions.MAX_RATE_CLIENTS + 20):
        authorizer._consume_rate_limit(f'client-{index}')
    assert len(authorizer._attempts) == permissions.MAX_RATE_CLIENTS
    now[0] += 61.0
    authorizer._consume_rate_limit('fresh-client')
    assert list(authorizer._attempts) == ['fresh-client']


def test_symlink_and_injected_discovery_values_fail_before_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch, count=1)
    world.devices['/dev/ttyUSB0'].symlink = True
    authorizer = world.authorizer()
    with pytest.raises(permissions.GripperPermissionError) as raised:
        authorizer.authorize_with_password(
            bytearray(SECRET), 'client-a', world.paths()
        )
    assert raised.value.code == 'unsafe_ttyusb_device'
    assert world.process_calls == []

    injected = permissions.TtyUsbPermissionAuthorizer(
        port_discovery=lambda: ['/dev/ttyUSB0;touch /tmp/pwned'],
        access_check=lambda _path: False,
        process_factory=world.process_factory,
    )
    with pytest.raises(permissions.GripperPermissionError) as raised:
        injected.authorize_with_password(
            bytearray(SECRET),
            'client-b',
            ['/dev/ttyUSB0;touch /tmp/pwned'],
        )
    assert raised.value.code == 'unsafe_ttyusb_discovery'
    assert world.process_calls == []


def test_device_replacement_after_chmod_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch, count=1)

    def replace_after_chmod() -> None:
        device = world.devices['/dev/ttyUSB0']
        device.mode = 0o666
        world.accessible.add(device.path)
        # The open descriptor keeps the old object while the path now resolves
        # to a replacement inode.
        world.devices['/dev/ttyUSB0'] = _FakeDevice(
            '/dev/ttyUSB0', st_dev=10, st_ino=999, st_rdev=299, mode=0o666
        )

    world.process_builder = lambda: _FakeProcess(on_communicate=replace_after_chmod)
    authorizer = world.authorizer()
    with pytest.raises(permissions.GripperPermissionError) as raised:
        authorizer.authorize_with_password(
            bytearray(SECRET), 'client-a', world.paths()
        )
    assert raised.value.code == 'ttyusb_changed'


def test_authorizer_has_no_helper_or_passwordless_repair_entrypoint() -> None:
    assert not hasattr(permissions, 'ROOT_HELPER_PATH')
    assert not hasattr(
        permissions.TtyUsbPermissionAuthorizer,
        'repair_without_password',
    )


def test_explicit_permission_rehydrates_across_revision_ttl_and_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    revision = [7]
    world = _FakeWorld(monkeypatch, count=1)
    authorizer = world.authorizer(
        monotonic=lambda: clock[0],
        lease_ttl_s=2.0,
    )
    authorizer.set_revision_provider(lambda: revision[0])
    result = authorizer.authorize_with_password(
        bytearray(SECRET), 'client-a', world.paths()
    )
    authorizer.commit_authorization(
        result.pop('_lease_token'), result['ports'], controller_revision=7
    )
    privileged_calls = len(world.process_calls)
    original_context = authorizer.status(world.paths())['context_id']

    revision[0] = 8
    assert authorizer.lease_allows('/dev/ttyUSB0') is True
    assert authorizer._leases['/dev/ttyUSB0'].controller_revision == 8
    assert authorizer.status(world.paths())['context_id'] == original_context

    clock[0] += 3.0
    assert authorizer.lease_allows('/dev/ttyUSB0') is True
    assert authorizer._leases['/dev/ttyUSB0'].expires_at > clock[0]

    restarted = world.authorizer(
        monotonic=lambda: clock[0],
        lease_ttl_s=2.0,
    )
    restarted.set_revision_provider(lambda: revision[0])
    assert restarted.verify_lease_for_connection('/dev/ttyUSB0') == (
        '/dev/ttyUSB0'
    )
    assert restarted.status(world.paths())['context_id'] == original_context
    assert len(world.process_calls) == privileged_calls


def test_replacement_identity_is_fail_closed_until_explicit_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch, count=1)
    world.apply_permissions()
    revision = [3]
    authorizer = world.authorizer()
    authorizer.set_revision_provider(lambda: revision[0])
    assert authorizer.lease_allows('/dev/ttyUSB0') is True
    original_identity = authorizer._trusted_identities['/dev/ttyUSB0']
    original_context = authorizer.status(world.paths())['context_id']

    world.devices['/dev/ttyUSB0'] = _FakeDevice(
        '/dev/ttyUSB0', 10, 999, 299, mode=0o666
    )
    assert authorizer.lease_allows('/dev/ttyUSB0') is False
    first_status = authorizer.status(world.paths())
    second_status = authorizer.status(world.paths())
    assert first_status['required'] is True
    assert second_status['required'] is True
    assert first_status['context_id'] != original_context
    assert second_status['context_id'] == first_status['context_id']
    assert authorizer._trusted_identities['/dev/ttyUSB0'] == original_identity

    # A fresh process may bootstrap the already-world-RW replacement.
    restarted = world.authorizer()
    restarted.set_revision_provider(lambda: revision[0])
    assert restarted.lease_allows('/dev/ttyUSB0') is True

    # The original process requires an explicit two-phase commit. Since the
    # node is already accessible this path does not invoke sudo/chmod.
    privileged_calls = len(world.process_calls)
    result = authorizer.authorize_with_password(
        bytearray(SECRET), 'client-b', world.paths()
    )
    assert result['method'] == 'existing_explicit_permission'
    assert result['permission']['required'] is True
    token = result.pop('_lease_token')
    assert authorizer.lease_allows('/dev/ttyUSB0') is False
    authorizer.commit_authorization(
        token, result['ports'], controller_revision=revision[0]
    )
    assert authorizer.lease_allows('/dev/ttyUSB0') is True
    assert authorizer._trusted_identities['/dev/ttyUSB0'].st_ino == 999
    assert len(world.process_calls) == privileged_calls

    # Permission loss is fail-closed, but restoring 0666 on the same trusted
    # identity can rehydrate without another password.
    world.devices['/dev/ttyUSB0'].mode = 0o660
    assert authorizer.lease_allows('/dev/ttyUSB0') is False
    assert authorizer.permission_required('/dev/ttyUSB0') is True
    world.devices['/dev/ttyUSB0'].mode = 0o666
    assert authorizer.lease_allows('/dev/ttyUSB0') is True
    world.accessible.discard('/dev/ttyUSB0')
    assert authorizer.lease_allows('/dev/ttyUSB0') is False


def test_world_rw_adoption_requires_access_direct_character_and_no_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch, count=1)
    device = world.devices['/dev/ttyUSB0']
    authorizer = world.authorizer()

    world.accessible.add(device.path)
    device.mode = 0o660
    assert authorizer.lease_allows(device.path) is False
    device.mode = 0o666
    world.accessible.clear()
    assert authorizer.lease_allows(device.path) is False
    world.accessible.add(device.path)
    device.symlink = True
    assert authorizer.lease_allows(device.path) is False
    device.symlink = False
    assert authorizer.lease_allows('/dev//ttyUSB0') is False

    token = authorizer._stage_authorization(
        authorizer._current_identities([device.path])
    )
    assert authorizer.lease_allows(device.path) is False
    assert device.path not in authorizer._trusted_identities
    authorizer.discard_authorization(token)
    assert authorizer.lease_allows(device.path) is False
    assert device.path in authorizer._uncommitted_identities

    committed_token = authorizer._stage_authorization(
        authorizer._current_identities([device.path])
    )
    authorizer.commit_authorization(
        committed_token, [device.path], controller_revision=0
    )
    assert authorizer.verify_lease_for_connection(device.path) == device.path
    assert device.path in authorizer._trusted_identities
    assert device.path not in authorizer._uncommitted_identities
    assert world.process_calls == []

    existing_token = authorizer._stage_authorization(
        authorizer._current_identities([device.path])
    )
    assert authorizer.lease_allows(device.path) is True
    authorizer.discard_authorization(existing_token)


def test_failed_controller_commit_cannot_bootstrap_chmod_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = [1]
    world = _FakeWorld(monkeypatch, count=1)
    authorizer = world.authorizer()
    authorizer.set_revision_provider(lambda: revision[0])
    result = authorizer.authorize_with_password(
        bytearray(SECRET), 'client-a', world.paths()
    )
    token = result.pop('_lease_token')
    revision[0] = 2
    with pytest.raises(permissions.GripperPermissionError) as raised:
        authorizer.commit_authorization(
            token, result['ports'], controller_revision=1
        )
    assert raised.value.code == 'permission_context_changed'
    assert authorizer.lease_allows('/dev/ttyUSB0') is False
    assert authorizer.status(world.paths())['required'] is True
    assert '/dev/ttyUSB0' in authorizer._uncommitted_identities

    retry = authorizer.authorize_with_password(
        bytearray(SECRET), 'client-b', world.paths()
    )
    assert retry['method'] == 'existing_explicit_permission'
    authorizer.commit_authorization(
        retry.pop('_lease_token'), retry['ports'], controller_revision=2
    )
    assert authorizer.lease_allows('/dev/ttyUSB0') is True
    assert '/dev/ttyUSB0' not in authorizer._uncommitted_identities


def test_permission_context_id_is_opaque_stable_and_identity_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch, count=2)
    authorizer = world.authorizer()
    first = authorizer.status(['/dev/ttyUSB1', '/dev/ttyUSB0'])
    reordered = authorizer.status(['/dev/ttyUSB0', '/dev/ttyUSB1'])
    context_id = first['context_id']
    assert first['required'] is True
    assert first['target_ports'] == ['/dev/ttyUSB0', '/dev/ttyUSB1']
    assert context_id == reordered['context_id']
    assert isinstance(context_id, str) and len(context_id) == 64
    int(context_id, 16)
    assert '/dev/' not in context_id

    world.apply_permissions()
    accessible = authorizer.status(world.paths())
    assert accessible['required'] is False
    assert accessible['context_id'] == context_id
    assert world.authorizer().status(world.paths())['context_id'] == context_id

    world.devices['/dev/ttyUSB1'] = _FakeDevice(
        '/dev/ttyUSB1', 10, 777, 277, mode=0o666
    )
    changed = authorizer.status(world.paths())
    assert changed['required'] is True
    assert changed['affected_ports'] == ['/dev/ttyUSB1']
    assert changed['context_id'] != context_id
    assert authorizer.status([])['context_id'] is None

def test_lease_toctou_never_reapplies_chmod_or_sudo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _FakeWorld(monkeypatch, count=1)
    authorizer = world.authorizer()
    result = authorizer.authorize_with_password(
        bytearray(SECRET), 'client-a', world.paths()
    )
    authorizer.commit_authorization(
        result.pop('_lease_token'), result['ports'], controller_revision=0
    )
    privileged_calls = len(world.process_calls)

    def race_check(port: str, *, renew: bool = False) -> bool:
        allowed = authorizer.lease_allows(port, renew=renew)
        if allowed:
            world.devices[port].mode = 0o600
            world.accessible.discard(port)
        return allowed

    manager = gripper_control.GripperControlManager(
        auto_monitor=False,
        port_discovery=world.paths,
        port_validator=lambda port: str(port),
        permission_granter=lambda _port: pytest.fail(
            'legacy granter must not run for an explicit lease'
        ),
        permission_lease_check=race_check,
        permission_lease_verify=authorizer.verify_lease_for_connection,
    )
    manager._profiles['left'] = SimpleNamespace(port='/dev/ttyUSB0')

    with manager._lock, pytest.raises(gripper_control.GripperControlError):
        manager._prepare_ports_locked(['left'])
    assert len(world.process_calls) == privileged_calls


class _RouteAuthorizer:
    def __init__(self) -> None:
        self.password_calls = 0
        self.commit_calls = 0
        self.discard_calls = 0
        self.revoke_calls = 0
        self.last_secret: Optional[bytearray] = None
        self.last_ports: list[str] = []
        self.revision_provider: Optional[Callable[[], int]] = None

    def set_revision_provider(
        self, provider: Optional[Callable[[], int]]
    ) -> None:
        self.revision_provider = provider

    def lease_allows(self, _port: Any, *, renew: bool = False) -> bool:
        del renew
        return False

    def verify_lease_for_connection(self, port: Any) -> str:
        return str(port)

    def revoke_leases(self, _ports: Any = ()) -> None:
        self.revoke_calls += 1

    def discard_authorization(self, token: Optional[str]) -> None:
        if token:
            self.discard_calls += 1

    def commit_authorization(
        self,
        token: str,
        expected_ports: Any,
        *,
        controller_revision: int,
    ) -> dict[str, Any]:
        assert token == 'route-lease-token'
        assert controller_revision == 0
        assert self.revision_provider is not None
        assert self.revision_provider() == controller_revision
        self.commit_calls += 1
        self.last_ports = list(expected_ports)
        payload = self.status(expected_ports)
        payload['required'] = False
        payload['affected_ports'] = []
        return payload

    def status(self, _allowed_ports: Any = ()) -> dict[str, Any]:
        return {
            'required': True,
            'target_ports': ['/dev/ttyUSB0'],
            'affected_ports': ['/dev/ttyUSB0'],
        }

    def authorize_with_password(
        self, secret: bytearray, _client: str, _allowed_ports: Any
    ) -> dict[str, Any]:
        self.password_calls += 1
        self.last_ports = list(_allowed_ports)
        self.last_secret = secret
        assert bytes(secret) == SECRET
        return {
            'ok': True,
            'authorized': True,
            'ports': ['/dev/ttyUSB0'],
            '_lease_token': 'route-lease-token',
        }


@pytest.fixture
def permission_app(tmp_path: Path) -> tuple[Any, _RouteAuthorizer]:
    authorizer = _RouteAuthorizer()
    manager = gripper_control.GripperControlManager(auto_monitor=False)
    manager.configured_ports = lambda: {'left': '/dev/ttyUSB0'}  # type: ignore[method-assign]
    conversion_jobs = server.ConversionJobManager(
        queue_root=tmp_path / 'conversion_jobs', auto_start=False
    )
    app = server.create_app(
        config_dir=tmp_path,
        gripper_control=manager,
        gripper_permissions=authorizer,
        conversion_jobs=conversion_jobs,
    )
    app.config.update(TESTING=True, GRIPPER_PERMISSION_CSRF_TOKEN='csrf-test')
    yield app, authorizer
    app.config['DEVICE_CONTROLLER'].shutdown()
    conversion_jobs.close()


def _headers(origin: str) -> dict[str, str]:
    return {
        'Origin': origin,
        'Sec-Fetch-Site': 'same-origin',
        'X-CSRF-Token': 'csrf-test',
    }


def test_plain_http_accepts_stdin_password_and_never_echoes_it(
    permission_app: tuple[Any, _RouteAuthorizer],
    caplog: pytest.LogCaptureFixture,
) -> None:
    app, authorizer = permission_app
    caplog.set_level(logging.DEBUG)
    response = app.test_client().post(
        '/api/grippers/permissions/sudo',
        base_url='http://10.128.1.138:18088',
        headers=_headers('http://10.128.1.138:18088'),
        data=(
            b'{"revision":0,"ports":["/dev/ttyUSB0"],"password":"'
            + SECRET
            + b'"}'
        ),
        content_type='application/json',
    )
    assert response.status_code == 200
    assert SECRET not in response.data
    assert SECRET.decode() not in caplog.text
    assert authorizer.password_calls == 1
    assert authorizer.commit_calls == 1
    assert authorizer.last_secret == bytearray(len(SECRET))
    assert response.headers['Cache-Control'].startswith('no-store')
    assert response.headers['Pragma'] == 'no-cache'


@pytest.mark.parametrize(
    ('base_url', 'origin'),
    [
        ('http://10.128.1.138:18088', 'http://10.128.1.138:18088'),
        ('https://panel.example', 'https://panel.example'),
        ('http://127.0.0.1:18088', 'http://127.0.0.1:18088'),
    ],
)
def test_tls_or_loopback_accepts_stdin_password_and_route_zeroes_buffer(
    permission_app: tuple[Any, _RouteAuthorizer],
    base_url: str,
    origin: str,
) -> None:
    app, authorizer = permission_app
    response = app.test_client().post(
        '/api/grippers/permissions/sudo',
        base_url=base_url,
        headers=_headers(origin),
        json={
            'revision': 0,
            'ports': ['/dev/ttyUSB0'],
            'password': SECRET.decode(),
        },
    )
    assert response.status_code == 200
    assert authorizer.password_calls == 1
    assert authorizer.commit_calls == 1
    assert authorizer.last_secret == bytearray(len(SECRET))
    assert SECRET not in response.data
    assert '_lease_token' not in response.get_json()
    assert response.headers['Referrer-Policy'] == 'no-referrer'


def test_permission_endpoints_require_origin_and_csrf(
    permission_app: tuple[Any, _RouteAuthorizer],
) -> None:
    app, authorizer = permission_app
    client = app.test_client()
    cross_origin = client.post(
        '/api/grippers/permissions/sudo',
        base_url='http://panel.local',
        headers=_headers('http://evil.example'),
        json={
            'revision': 0,
            'ports': ['/dev/ttyUSB0'],
            'password': SECRET.decode(),
        },
    )
    assert cross_origin.status_code == 403
    assert cross_origin.get_json()['code'] == 'permission_origin_rejected'
    missing_csrf = client.post(
        '/api/grippers/permissions/sudo',
        base_url='http://panel.local',
        headers={'Origin': 'http://panel.local'},
        json={
            'revision': 0,
            'ports': ['/dev/ttyUSB0'],
            'password': SECRET.decode(),
        },
    )
    assert missing_csrf.status_code == 403
    assert missing_csrf.get_json()['code'] == 'permission_csrf_rejected'
    assert authorizer.password_calls == 0


def test_passwordless_repair_route_is_not_exposed(
    permission_app: tuple[Any, _RouteAuthorizer],
) -> None:
    app, authorizer = permission_app
    response = app.test_client().post(
        '/api/grippers/permissions/repair',
        base_url='http://panel.local',
        headers=_headers('http://panel.local'),
        json={'revision': 0, 'ports': ['/dev/ttyUSB0']},
    )
    assert response.status_code == 404
    assert authorizer.password_calls == 0
    assert authorizer.commit_calls == 0


@pytest.mark.parametrize('change', ['revision', 'ports'])
def test_permission_post_rejects_binding_context_changed_after_modal_open(
    permission_app: tuple[Any, _RouteAuthorizer],
    change: str,
) -> None:
    app, authorizer = permission_app
    controller = app.config['DEVICE_CONTROLLER']
    if change == 'revision':
        controller.revision += 1
    else:
        controller.grippers.configured_ports = (  # type: ignore[method-assign]
            lambda: {'left': '/dev/ttyUSB1'}
        )
    response = app.test_client().post(
        '/api/grippers/permissions/sudo',
        base_url='http://panel.local',
        headers=_headers('http://panel.local'),
        json={
            'revision': 0,
            'ports': ['/dev/ttyUSB0'],
            'password': SECRET.decode(),
        },
    )
    assert response.status_code == 409
    assert response.get_json()['code'] == 'permission_context_changed'
    assert authorizer.password_calls == 0
    assert b'/dev/ttyUSB1' not in response.data


def test_permission_reservation_keeps_state_and_safety_stop_responsive(
    permission_app: tuple[Any, _RouteAuthorizer],
) -> None:
    app, authorizer = permission_app
    controller = app.config['DEVICE_CONTROLLER']
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def blocking_authorize(
        secret: bytearray,
        _client: str,
        allowed_ports: Any,
    ) -> dict[str, Any]:
        assert bytes(secret) == SECRET
        authorizer.password_calls += 1
        authorizer.last_ports = list(allowed_ports)
        authorizer.last_secret = secret
        entered.set()
        release.wait(2.0)
        for index in range(len(secret)):
            secret[index] = 0
        return {
            'ok': True,
            'authorized': True,
            'ports': list(allowed_ports),
            '_lease_token': 'route-lease-token',
        }

    authorizer.authorize_with_password = blocking_authorize  # type: ignore[method-assign]

    def authorize() -> None:
        try:
            controller.repair_gripper_permissions(
                expected_revision=0,
                expected_ports=['/dev/ttyUSB0'],
                client_key='client-a',
                secret=bytearray(SECRET),
            )
        except BaseException as error:
            failures.append(error)

    authorize_thread = threading.Thread(target=authorize)
    authorize_thread.start()
    assert entered.wait(1.0)

    state_started = time.monotonic()
    controller.gripper_state()
    assert time.monotonic() - state_started < 0.2

    mutation_started = time.monotonic()
    with pytest.raises(server.DeviceConflictError):
        controller.update_gripper_ports(
            {'left': '/dev/ttyUSB0'}, expected_revision=0
        )
    assert time.monotonic() - mutation_started < 0.2

    stop_started = time.monotonic()
    stop_result = controller.control_gripper('stop', expected_revision=0)
    assert time.monotonic() - stop_started < 0.2
    assert stop_result['ok'] is True

    release.set()
    authorize_thread.join(2.0)
    assert not authorize_thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], permissions.GripperPermissionError)
    assert failures[0].code == 'permission_context_changed'
    assert authorizer.discard_calls == 1
    assert authorizer.revoke_calls >= 1


def test_main_page_allows_plain_http_password_transport(
    permission_app: tuple[Any, _RouteAuthorizer],
) -> None:
    app, _authorizer = permission_app
    response = app.test_client().get(
        '/', base_url='http://10.128.1.138:18088'
    )
    assert response.status_code == 200
    assert b'name="gripper-password-transport" content="allowed"' in response.data
    assert b'name="gripper-permission-csrf" content="csrf-test"' in response.data
    assert response.headers['Cache-Control'].startswith('no-store')


def test_known_and_chunked_oversized_password_bodies_are_bounded(
    permission_app: tuple[Any, _RouteAuthorizer],
) -> None:
    app, authorizer = permission_app
    client = app.test_client()
    oversized = (
        b'{"revision":0,"ports":["/dev/ttyUSB0"],"password":"'
        + (b'x' * 3000)
        + b'"}'
    )
    known = client.post(
        '/api/grippers/permissions/sudo',
        base_url='https://panel.example',
        headers=_headers('https://panel.example'),
        data=oversized,
        content_type='application/json',
    )
    assert known.status_code == 422
    assert authorizer.password_calls == 0

    builder = EnvironBuilder(
        path='/api/grippers/permissions/sudo',
        base_url='https://panel.example',
        method='POST',
        headers=_headers('https://panel.example'),
        input_stream=io.BytesIO(oversized),
        content_type='application/json',
    )
    environ = builder.get_environ()
    environ.pop('CONTENT_LENGTH', None)
    environ['wsgi.input_terminated'] = True
    chunked = client.open(environ)
    assert chunked.status_code == 422
    assert authorizer.password_calls == 0
    assert SECRET not in chunked.data
