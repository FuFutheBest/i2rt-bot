#!/usr/bin/env python3
"""
check_robot_arm.py -- Flexiv Rizon connectivity / health diagnostic.

Pre-flight tool that verifies each configured Rizon arm can actually be reached
by the Flexiv RDK *before* launching the full device stack. It surfaces the two
most common startup failures in a legible way:

  1. Malformed serial numbers. The RDK derives the robot MODEL from the part of
     the serial number before the first '-' (whitespace removed). A typo such as
     ``"Rizon6 4-03087"`` collapses to model ``Rizon64`` and the RDK aborts with
     ``Robot model [Rizon64] is not supported``. This script performs the same
     static parse and warns *before* connecting, then still attempts the
     connection so the exact RDK error is captured too.
  2. Network / NIC-whitelist problems. On multi-NIC hosts the serial-number
     discovery must be pinned to the robot-facing NIC's local IPv4 or it scans
     the wrong interface and times out.

For every serial number it reports: parsed model sanity, connection result, the
RDK ``RobotInfo`` (model, DoF, F/T sensor, software version), health flags
(fault / operational / e-stop / reduced), and a snapshot of the live states
(joint positions, TCP pose, external wrench, joint temperatures).

By default the script is READ-ONLY: it connects and reads. Pass ``--enable`` to
additionally clear faults and enable the arm (moves nothing, but energizes it).

Usage (run through the verified device runtime profile, or another explicitly
verified environment that imports Flexiv RDK):

    # Read serial numbers + NIC whitelist straight from the device config:
    python scripts/legacy/check_robot_arm.py

    # Check explicit serial numbers:
    python scripts/legacy/check_robot_arm.py --robot-sn "Rizon 4-063087" "Rizon 4-062071"

    # Use a different device config and also enable the arms:
    python scripts/legacy/check_robot_arm.py \
        --config configs/devices/infer/infer_rizon_fisheye_umi3_dual_openpi.yaml --enable

Author: Zixi Ying
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.logger_config import logger  # noqa: E402

try:
    import flexivrdk  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - runtime dependency
    logger.error(f"flexivrdk import failed: {exc}")
    logger.error(
        "Resolve and verify the selected config's device runtime profile before retrying."
    )
    raise

_DEFAULT_DEVICE_CONFIG = (
    _REPO_ROOT / "configs/devices/infer/infer_rizon_fisheye_umi3_dual_openpi.yaml"
)

# Known-good Flexiv model names accepted by RDK. Informational only; the
# authoritative check is the connection attempt itself (e.g. "Rizon4R" is a real
# model that is not in this set).
_KNOWN_MODELS = {"Rizon4", "Rizon4s", "Rizon4R", "Rizon10", "Rizon10s"}
# A valid serial-number head is "Rizon" + at most one space + a SINGLE digit group
# + at most one trailing letter (s / R variants), e.g. "Rizon 4", "Rizon4s",
# "Rizon 4R", "Rizon10". The genuine failure signature is instead two number
# groups separated by whitespace (e.g. "Rizon6 4"), which RDK collapses into the
# bogus model "Rizon64"; that is checked separately.
_HEAD_PATTERN = re.compile(r"^Rizon ?\d+[A-Za-z]?$")
# Serial-number head that collapses two number groups into one bogus model.
_COLLAPSE_PATTERN = re.compile(r"\d\s+\d")


def _local_ipv4_addresses() -> List[str]:
    """Return non-loopback local IPv4 addresses via ``ip -4 addr`` (best effort)."""
    ip_cmd = shutil.which("ip")
    if not ip_cmd:
        return []
    try:
        result = subprocess.run(
            [ip_cmd, "-4", "-o", "addr", "show", "scope", "global"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    out: List[str] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if "inet" not in parts:
            continue
        ip = parts[parts.index("inet") + 1].split("/")[0]
        if not ip.startswith("127.") and ip not in out:
            out.append(ip)
    return out


def _derive_model(serial_number: str) -> str:
    """Reproduce how Flexiv RDK derives the model name from a serial number.

    The model is the substring before the first '-', with all whitespace
    removed. e.g. ``"Rizon 4-062071" -> "Rizon4"`` and the buggy
    ``"Rizon6 4-03087" -> "Rizon64"``.
    """
    head = serial_number.split("-", 1)[0]
    return re.sub(r"\s+", "", head)


def _validate_serial_number(serial_number: str) -> Tuple[bool, str, str]:
    """Statically sanity-check a serial number.

    Returns ``(looks_ok, derived_model, message)``. ``looks_ok`` is False when the
    derived model is clearly malformed (e.g. the ``Rizon64`` collapse), which lets
    the caller warn loudly before wasting time on the connection attempt.
    """
    head = serial_number.split("-", 1)[0].strip()
    model = _derive_model(serial_number)
    # Genuine failure signature first: two number groups (e.g. "Rizon6 4").
    if _COLLAPSE_PATTERN.search(head):
        return (
            False,
            model,
            f"parsed model '{model}' does NOT look valid -- head '{head}' has two "
            f"number groups that RDK collapses into '{model}'; a serial like "
            "'Rizon6 4-03087' should read 'Rizon 4-...'. Check the sticker on the "
            "robot base.",
        )
    if model in _KNOWN_MODELS:
        return True, model, f"parsed model '{model}' looks valid"
    if _HEAD_PATTERN.match(head):
        return (
            True,
            model,
            f"parsed model '{model}' matches the Rizon<N>[variant] pattern but is "
            f"not in the known set {sorted(_KNOWN_MODELS)}; continuing anyway",
        )
    return (
        False,
        model,
        f"parsed model '{model}' does NOT look like a valid Flexiv model -- head "
        f"'{head}' is not of the form 'Rizon <N>[variant]'.",
    )


def _load_robot_config(
    config_path: Path,
) -> Tuple[List[str], List[str]]:
    """Extract ``robot_sn`` and ``network_interface_whitelist`` from a device YAML.

    Uses ``yaml.safe_load`` (not OmegaConf) so unresolved Hydra interpolations such
    as ``${runtime.signal_files}`` on unrelated keys do not break parsing.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Device config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    devices = (cfg or {}).get("devices", [])
    if not isinstance(devices, list):
        raise ValueError(f"'devices' is not a list in {config_path}")

    for device in devices:
        if not isinstance(device, dict):
            continue
        if device.get("class") != "RizonRobot":
            continue
        raw_sn: Union[str, List[str], None] = device.get("robot_sn")
        if raw_sn is None:
            raise ValueError(f"RizonRobot entry has no 'robot_sn' in {config_path}")
        sn_list = [raw_sn] if isinstance(raw_sn, str) else list(raw_sn)
        whitelist = device.get("network_interface_whitelist") or []
        if isinstance(whitelist, str):
            whitelist = [whitelist]
        return sn_list, list(whitelist)

    raise ValueError(f"No 'RizonRobot' device found in {config_path}")


def _create_robot(serial_number: str, whitelist: Sequence[str]) -> Any:
    """Construct a ``flexivrdk.Robot`` handling both 2-arg and 4-arg signatures."""
    wl = list(whitelist)
    try:
        return flexivrdk.Robot(serial_number, wl)
    except TypeError:
        # Older/newer bindings expose extra boolean flags with defaults.
        return flexivrdk.Robot(serial_number, wl, False, False)


def _fmt_vector(values: Any, digits: int = 4) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{v:.{digits}f}" for v in arr) + "]"


def _log_robot_info(robot: Any) -> None:
    """Log the RDK ``RobotInfo`` struct."""
    try:
        info = robot.info()
    except Exception as exc:  # pragma: no cover - hardware dependent
        logger.warning(f"  info() unavailable: {exc}")
        return
    logger.info(f"  model_name  : {getattr(info, 'model_name', 'n/a')}")
    logger.info(f"  serial_num  : {getattr(info, 'serial_num', 'n/a')}")
    logger.info(f"  software_ver: {getattr(info, 'software_ver', 'n/a')}")
    logger.info(f"  license_type: {getattr(info, 'license_type', 'n/a')}")
    logger.info(
        f"  DoF={getattr(info, 'DoF', '?')} "
        f"(motion={getattr(info, 'DoF_m', '?')}, external={getattr(info, 'DoF_e', '?')})"
    )
    logger.info(f"  has_FT_sensor: {getattr(info, 'has_FT_sensor', 'n/a')}")


def _log_health_flags(robot: Any) -> None:
    """Log the boolean/enum health flags that explain most enable failures."""
    def _safe(name: str) -> Any:
        try:
            return getattr(robot, name)()
        except Exception as exc:  # pragma: no cover - hardware dependent
            return f"<{exc}>"

    logger.info(f"  connected            : {_safe('connected')}")
    logger.info(f"  fault                : {_safe('fault')}")
    logger.info(f"  operational          : {_safe('operational')}")
    logger.info(f"  operational_status   : {_safe('operational_status')}")
    logger.info(f"  busy                 : {_safe('busy')}")
    logger.info(f"  reduced (speed-limit): {_safe('reduced')}")
    logger.info(f"  estop_released       : {_safe('estop_released')}")
    logger.info(f"  enabling_btn_pressed : {_safe('enabling_button_pressed')}")


def _log_states(robot: Any) -> None:
    """Log a snapshot of the live robot states (joints, TCP, wrench, temps)."""
    try:
        states = robot.states()
    except Exception as exc:  # pragma: no cover - hardware dependent
        logger.warning(f"  states() unavailable: {exc}")
        return

    q = getattr(states, "q", None)
    if q is not None:
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        logger.info(f"  joint q (rad): {_fmt_vector(q_arr)}")
        logger.info(f"  joint q (deg): {_fmt_vector(np.degrees(q_arr), digits=2)}")

    tcp = getattr(states, "tcp_pose", None)
    if tcp is not None:
        tcp_arr = np.asarray(tcp, dtype=float).reshape(-1)
        # Flexiv TCP pose is [x, y, z, qw, qx, qy, qz] (quaternion is w-first).
        if tcp_arr.size >= 7:
            logger.info(
                f"  tcp_pose xyz (m)  : {_fmt_vector(tcp_arr[:3])}"
            )
            logger.info(
                f"  tcp_pose quat wxyz: {_fmt_vector(tcp_arr[3:7])}"
            )
        else:
            logger.info(f"  tcp_pose: {_fmt_vector(tcp_arr)}")

    wrench = getattr(states, "ext_wrench_in_tcp", None)
    if wrench is not None:
        w_arr = np.asarray(wrench, dtype=float).reshape(-1)
        if w_arr.size >= 6:
            f_mag = float(np.linalg.norm(w_arr[:3]))
            m_mag = float(np.linalg.norm(w_arr[3:6]))
            logger.info(
                f"  ext_wrench_in_tcp : |F|={f_mag:.2f} N, |M|={m_mag:.2f} Nm"
            )

    temps = getattr(states, "temperature", None)
    if temps is not None:
        t_arr = np.asarray(temps, dtype=float).reshape(-1)
        if t_arr.size:
            logger.info(
                f"  joint temperature : max={t_arr.max():.1f}C {_fmt_vector(t_arr, 1)}"
            )


def _enable_robot(robot: Any, enable_timeout_s: float) -> bool:
    """Clear faults and enable the robot, waiting for operational state."""
    try:
        if robot.fault():
            logger.warning("  robot reports FAULT; calling ClearFault()...")
            robot.ClearFault()
            deadline = time.time() + max(enable_timeout_s, 5.0)
            while robot.fault() and time.time() < deadline:
                time.sleep(0.2)
            if robot.fault():
                logger.error("  fault did NOT clear; cannot enable.")
                return False
            logger.info("  fault cleared.")

        if robot.operational():
            logger.info("  already operational.")
            return True

        logger.info("  calling Enable() and waiting for operational state...")
        robot.Enable()
        deadline = time.time() + enable_timeout_s
        while time.time() < deadline:
            if robot.operational():
                logger.info("  robot is now operational.")
                return True
            time.sleep(0.2)
        logger.error(
            f"  robot did NOT reach operational state within {enable_timeout_s:.0f}s. "
            "Check the physical E-stop and the enabling button."
        )
        return False
    except Exception as exc:  # pragma: no cover - hardware dependent
        logger.error(f"  enable failed: {exc}")
        return False


def _check_one_robot(
    serial_number: str,
    whitelist: Sequence[str],
    enable: bool,
    enable_timeout_s: float,
    settle_s: float,
) -> bool:
    """Run the full check for a single robot. Returns True on success."""
    logger.info("=" * 72)
    logger.info(f"Checking robot: '{serial_number}'  (NIC whitelist={list(whitelist) or 'all'})")

    sn_ok, model, message = _validate_serial_number(serial_number)
    if sn_ok:
        logger.info(f"  serial-number sanity: {message}")
    else:
        logger.warning(f"  serial-number sanity: {message}")
        logger.warning(
            "  This will almost certainly fail with "
            f"'Robot model [{model}] is not supported'. Attempting anyway..."
        )

    robot: Optional[Any] = None
    try:
        robot = _create_robot(serial_number, whitelist)
    except Exception as exc:
        logger.error(f"  CONNECT FAILED: {exc}")
        if not sn_ok:
            logger.error(
                f"  -> Fix the serial number (parsed model was '{model}'). "
                "Compare it against the sticker on the robot base."
            )
        else:
            logger.error(
                "  -> Robot object could not be created. Check power, network, and "
                "that the NIC whitelist points at the robot-facing interface."
            )
        return False

    logger.info("  connect: OK")
    if settle_s > 0:
        time.sleep(settle_s)

    _log_robot_info(robot)
    _log_health_flags(robot)

    success = True
    if enable:
        success = _enable_robot(robot, enable_timeout_s)

    _log_states(robot)

    if success:
        logger.info(f"  RESULT: '{serial_number}' OK")
    else:
        logger.error(f"  RESULT: '{serial_number}' reachable but NOT operational")
    return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Flexiv Rizon arm connectivity/health before launching the "
            "device stack. Read-only by default."
        )
    )
    parser.add_argument(
        "--robot-sn",
        nargs="+",
        default=None,
        help=(
            "One or more robot serial numbers to check. If omitted, they are read "
            "from the device config (--config)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_DEVICE_CONFIG,
        help=f"Device YAML to read robot_sn from (default: {_DEFAULT_DEVICE_CONFIG}).",
    )
    parser.add_argument(
        "--nic",
        nargs="*",
        default=None,
        help=(
            "Network-interface IPv4 whitelist override for RDK discovery. If omitted, "
            "the config value is used; if that is empty, local IPv4s are auto-detected."
        ),
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Also clear faults and enable the arm (energizes it; moves nothing).",
    )
    parser.add_argument(
        "--enable-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for the robot to reach operational state (with --enable).",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.5,
        help="Seconds to wait after connecting before reading states.",
    )
    args = parser.parse_args()

    if args.enable_timeout <= 0.0:
        parser.error("--enable-timeout must be > 0.")
    if args.settle < 0.0:
        parser.error("--settle must be >= 0.")

    config_whitelist: List[str] = []
    if args.robot_sn:
        serial_numbers = list(args.robot_sn)
        logger.info(f"Using {len(serial_numbers)} serial number(s) from --robot-sn")
    else:
        serial_numbers, config_whitelist = _load_robot_config(args.config)
        logger.info(
            f"Loaded {len(serial_numbers)} serial number(s) from {args.config}: "
            f"{serial_numbers}"
        )

    if args.nic is not None:
        whitelist = list(args.nic)
        logger.info(f"NIC whitelist from --nic: {whitelist or 'all'}")
    elif config_whitelist:
        whitelist = config_whitelist
        logger.info(f"NIC whitelist from config: {whitelist}")
    else:
        whitelist = _local_ipv4_addresses()
        logger.info(
            f"NIC whitelist auto-detected (no config value): {whitelist or 'all'}"
        )

    results: List[Tuple[str, bool]] = []
    for serial_number in serial_numbers:
        try:
            ok = _check_one_robot(
                serial_number,
                whitelist,
                enable=args.enable,
                enable_timeout_s=args.enable_timeout,
                settle_s=args.settle,
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
            raise
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.error(f"  UNEXPECTED ERROR for '{serial_number}': {exc}")
            ok = False
        results.append((serial_number, ok))

    logger.info("=" * 72)
    logger.info("SUMMARY")
    all_ok = True
    for serial_number, ok in results:
        status = "OK" if ok else "FAIL"
        log_fn = logger.info if ok else logger.error
        log_fn(f"  [{status}] {serial_number}")
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
