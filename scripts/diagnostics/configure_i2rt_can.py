"""Interactively bind I2RT CAN adapter serials to fixed teleoperation roles."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devices.robots.i2rt_teleop.config import load_teleoperation_config
from utils.runtime_profiles import (
    RuntimeProfile,
    RuntimeProfileError,
    RuntimeProfileNotFoundError,
    RuntimeRegistry,
)

DEFAULT_TELEOP_CONFIG = REPO_ROOT / "configs/i2rt_d405_yam_teleop.yaml"
DEFAULT_PROFILE = "i2rt-d405-yam"
DEFAULT_CONFIG_NAME = "i2rt_d405_yam_collect_async"
DEFAULT_IMPORTS = (
    "hydra",
    "omegaconf",
    "pyrealsense2",
    "i2rt",
    "flask",
    "cv2",
)
DEFAULT_ENVIRONMENT = {
    "PYTHONNOUSERSITE": "1",
    "SI_I2RT_YAM_GRIPPER_TYPE": "crank_4310",
}
RUNTIME_ROLES = ("launcher", "device")
CAN_SERIAL_ENVIRONMENT_NAMES = frozenset(
    {
        "SI_I2RT_CAN_LEADER_LEFT_SERIAL",
        "SI_I2RT_CAN_FOLLOWER_LEFT_SERIAL",
        "SI_I2RT_CAN_LEADER_RIGHT_SERIAL",
        "SI_I2RT_CAN_FOLLOWER_RIGHT_SERIAL",
    }
)
_CAN_CHANNEL_RE = re.compile(r"^can(\d+)$")


class CanProvisioningError(RuntimeError):
    """The observed CAN topology cannot be registered safely."""


@dataclass(frozen=True)
class ExpectedCanAdapter:
    channel: str
    side: str
    endpoint: str

    @property
    def environment_name(self) -> str:
        return f"SI_I2RT_CAN_{self.endpoint.upper()}_{self.side.upper()}_SERIAL"

    @property
    def confirmation_token(self) -> str:
        return f"{self.side.upper()}-{self.endpoint.upper()}"


@dataclass(frozen=True)
class CanIdentity:
    channel: str
    serial: str
    driver: str = ""


DiscoverCan = Callable[[], Mapping[str, CanIdentity]]


def expected_adapters(config_path: Path) -> tuple[ExpectedCanAdapter, ...]:
    """Return the fixed role order declared by the teleoperation YAML."""
    config = load_teleoperation_config(config_path)
    adapters = [
        ExpectedCanAdapter(
            channel=getattr(arm, endpoint).can_channel,
            side=side,
            endpoint=endpoint,
        )
        for side, arm in config.arms.items()
        for endpoint in ("leader", "follower")
    ]
    indexed: list[tuple[int, ExpectedCanAdapter]] = []
    for adapter in adapters:
        match = _CAN_CHANNEL_RE.fullmatch(adapter.channel)
        if match is None:
            raise CanProvisioningError(
                "interactive provisioning requires canN channel names; "
                f"found {adapter.channel!r}"
            )
        indexed.append((int(match.group(1)), adapter))
    indexed.sort(key=lambda item: item[0])
    indices = [index for index, _adapter in indexed]
    if indices != list(range(len(indexed))):
        raise CanProvisioningError(
            "interactive provisioning requires contiguous channels starting at can0; "
            f"found {', '.join(adapter.channel for _, adapter in indexed)}"
        )
    return tuple(adapter for _index, adapter in indexed)


def _udev_properties(sysfs_path: Path) -> dict[str, str]:
    try:
        completed = subprocess.run(
            ["udevadm", "info", "-q", "property", "-p", str(sysfs_path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CanProvisioningError(f"cannot inspect {sysfs_path}: {exc}") from exc
    if completed.returncode != 0:
        return {}
    return {
        name: value
        for line in completed.stdout.splitlines()
        for name, separator, value in (line.partition("="),)
        if separator and name
    }


def discover_can_interfaces(
    sys_class_net: Path = Path("/sys/class/net"),
) -> dict[str, CanIdentity]:
    """Discover SocketCAN interfaces without changing link state or sending frames."""
    discovered: dict[str, CanIdentity] = {}
    try:
        interfaces = tuple(sys_class_net.iterdir())
    except OSError as exc:
        raise CanProvisioningError(f"cannot read {sys_class_net}: {exc}") from exc
    for interface in interfaces:
        if _CAN_CHANNEL_RE.fullmatch(interface.name) is None:
            continue
        try:
            if (interface / "type").read_text(encoding="ascii").strip() != "280":
                continue
        except OSError:
            continue
        properties = _udev_properties(interface)
        discovered[interface.name] = CanIdentity(
            channel=interface.name,
            serial=properties.get("ID_SERIAL", "").strip(),
            driver=properties.get("ID_USB_DRIVER", "").strip(),
        )
    return discovered


def wait_until_all_unplugged(
    discover: DiscoverCan,
    *,
    poll_interval_s: float,
    stable_samples: int,
    timeout_s: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    report: Callable[[str], None] = print,
) -> None:
    deadline = monotonic() + timeout_s
    empty_samples = 0
    previous_channels: tuple[str, ...] | None = None
    while monotonic() < deadline:
        channels = tuple(sorted(discover()))
        if channels != previous_channels:
            report(
                "Waiting for all CAN adapters to be unplugged"
                + (f"; present: {', '.join(channels)}" if channels else "")
            )
            previous_channels = channels
        empty_samples = empty_samples + 1 if not channels else 0
        if empty_samples >= stable_samples:
            return
        sleep(poll_interval_s)
    raise CanProvisioningError("timed out waiting for all CAN adapters to be unplugged")


def wait_for_adapter(
    expected: ExpectedCanAdapter,
    accepted: Mapping[str, CanIdentity],
    discover: DiscoverCan,
    *,
    poll_interval_s: float,
    stable_samples: int,
    timeout_s: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> CanIdentity:
    """Wait until exactly the expected next adapter is stable and identifiable."""
    deadline = monotonic() + timeout_s
    candidate: CanIdentity | None = None
    candidate_samples = 0
    while monotonic() < deadline:
        current = dict(discover())
        for channel, identity in accepted.items():
            observed = current.get(channel)
            if observed is None:
                raise CanProvisioningError(
                    f"accepted adapter {channel} was unplugged; no changes were saved"
                )
            if observed.serial and observed.serial != identity.serial:
                raise CanProvisioningError(
                    f"adapter identity changed on {channel}; no changes were saved"
                )

        new_channels = sorted(set(current) - set(accepted))
        if len(new_channels) > 1:
            raise CanProvisioningError(
                "more than one new CAN adapter appeared: "
                f"{', '.join(new_channels)}; no changes were saved"
            )
        if not new_channels:
            candidate = None
            candidate_samples = 0
            sleep(poll_interval_s)
            continue

        channel = new_channels[0]
        if channel != expected.channel:
            raise CanProvisioningError(
                f"expected {expected.channel}, but the new interface is {channel}; "
                "no changes were saved"
            )
        observed = current[channel]
        if not observed.serial:
            candidate = None
            candidate_samples = 0
            sleep(poll_interval_s)
            continue
        if observed.serial in {identity.serial for identity in accepted.values()}:
            raise CanProvisioningError(
                f"duplicate ID_SERIAL {observed.serial!r}; no changes were saved"
            )
        if observed == candidate:
            candidate_samples += 1
        else:
            candidate = observed
            candidate_samples = 1
        if candidate_samples >= stable_samples:
            return observed
        sleep(poll_interval_s)
    raise CanProvisioningError(
        f"timed out waiting for a stable {expected.channel} with a complete ID_SERIAL"
    )


def provision_can_serials(
    adapters: Sequence[ExpectedCanAdapter],
    discover: DiscoverCan,
    confirm: Callable[[ExpectedCanAdapter, CanIdentity], bool],
    *,
    announce: Callable[[ExpectedCanAdapter], None] | None = None,
    poll_interval_s: float = 0.5,
    stable_samples: int = 3,
    timeout_s: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, str]:
    accepted: dict[str, CanIdentity] = {}
    for adapter in adapters:
        if announce is not None:
            announce(adapter)
        identity = wait_for_adapter(
            adapter,
            accepted,
            discover,
            poll_interval_s=poll_interval_s,
            stable_samples=stable_samples,
            timeout_s=timeout_s,
            sleep=sleep,
            monotonic=monotonic,
        )
        if not confirm(adapter, identity):
            raise CanProvisioningError("operator rejected the role assignment")
        accepted[identity.channel] = identity

    current = dict(discover())
    if set(current) != set(accepted):
        raise CanProvisioningError(
            "CAN topology changed before registration; no changes were saved"
        )
    for channel, identity in accepted.items():
        if current[channel].serial != identity.serial:
            raise CanProvisioningError(
                f"adapter identity changed on {channel}; no changes were saved"
            )
    return {
        adapter.environment_name: accepted[adapter.channel].serial
        for adapter in adapters
    }


def save_runtime_profile(
    registry: RuntimeRegistry,
    serial_environment: Mapping[str, str],
    *,
    profile_name: str,
    config_name: str,
    python: Path | None = None,
    cwd: Path | None = None,
    required_imports: Sequence[str] = DEFAULT_IMPORTS,
    timeout_s: float = 30.0,
) -> RuntimeProfile:
    """Verify and atomically persist the complete CAN binding set."""
    serials = tuple(serial.strip() for serial in serial_environment.values())
    if (
        set(serial_environment) != CAN_SERIAL_ENVIRONMENT_NAMES
        or any(not serial for serial in serials)
        or len(set(serials)) != len(CAN_SERIAL_ENVIRONMENT_NAMES)
    ):
        raise CanProvisioningError(
            "all four CAN role serials must be present, non-empty, and unique"
        )
    try:
        existing = registry.profile(profile_name)
    except RuntimeProfileNotFoundError:
        existing = None

    environment = dict(existing.environment) if existing is not None else {}
    environment.update(DEFAULT_ENVIRONMENT)
    environment.update(
        {name: serial.strip() for name, serial in serial_environment.items()}
    )
    imports = tuple(
        dict.fromkeys(
            (*(existing.imports if existing is not None else ()), *required_imports)
        )
    )
    profile = registry.register(
        profile_name,
        python=python
        or (existing.python if existing is not None else Path(sys.executable)),
        cwd=cwd or (existing.cwd if existing is not None else registry.repo_root),
        environment=environment,
        imports=imports,
    )
    for role in RUNTIME_ROLES:
        registry.bind(role, profile.name, config_name=config_name)
    profile = registry.verify(profile.name, timeout_s=timeout_s)
    registry.save()
    return profile


def _confirm_adapter(adapter: ExpectedCanAdapter, identity: CanIdentity) -> bool:
    print(f"Detected {identity.channel}: ID_SERIAL={identity.serial}")
    if identity.driver:
        print(f"Driver: {identity.driver}")
    response = input(
        f"Type {adapter.confirmation_token} to bind this adapter: "
    ).strip()
    return response.upper() == adapter.confirmation_token


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--registry", default=None)
    parser.add_argument("--teleop-config", default=str(DEFAULT_TELEOP_CONFIG))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--python", default=None)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--adapter-timeout", type=float, default=120.0)
    parser.add_argument("--verification-timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.poll_interval <= 0
        or args.stable_samples < 1
        or args.adapter_timeout <= 0
        or args.verification_timeout <= 0
    ):
        print("error: polling values and timeouts must be positive", file=sys.stderr)
        return 2
    try:
        repo_root = Path(args.repo_root).expanduser().absolute()
        adapters = expected_adapters(Path(args.teleop_config))
        print(
            "This tool only reads CAN/udev identity; "
            "it never configures links or sends frames."
        )
        print("Role order from the teleoperation YAML:")
        for adapter in adapters:
            print(
                f"  {adapter.channel}: {adapter.side} {adapter.endpoint} "
                f"({adapter.environment_name})"
            )
        input("Unplug all four CAN adapters, then press Enter to continue: ")
        wait_until_all_unplugged(
            discover_can_interfaces,
            poll_interval_s=args.poll_interval,
            stable_samples=args.stable_samples,
            timeout_s=args.adapter_timeout,
        )

        def announce(adapter: ExpectedCanAdapter) -> None:
            print(
                f"\nInsert only the {adapter.side} {adapter.endpoint} adapter; "
                f"it must appear as {adapter.channel}."
            )

        def confirm(adapter: ExpectedCanAdapter, identity: CanIdentity) -> bool:
            return _confirm_adapter(adapter, identity)

        serial_environment = provision_can_serials(
            adapters,
            discover_can_interfaces,
            confirm,
            announce=announce,
            poll_interval_s=args.poll_interval,
            stable_samples=args.stable_samples,
            timeout_s=args.adapter_timeout,
        )
        registry = RuntimeRegistry.load(
            repo_root=repo_root,
            path=Path(args.registry) if args.registry else None,
        )
        profile = save_runtime_profile(
            registry,
            serial_environment,
            profile_name=args.profile,
            config_name=args.config_name,
            python=Path(args.python) if args.python else None,
            cwd=Path(args.cwd) if args.cwd else None,
            timeout_s=args.verification_timeout,
        )
    except (CanProvisioningError, RuntimeProfileError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled; no changes were saved.", file=sys.stderr)
        return 130

    print(f"\nSaved and verified runtime profile {profile.name!r}.")
    print(f"Registry: {profile.registry_path}")
    print(f"Bound {args.config_name}: {', '.join(RUNTIME_ROLES)}")
    print("Before hardware startup, configure every CAN link UP at 1 Mbit/s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
