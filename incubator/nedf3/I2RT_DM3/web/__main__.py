"""Launch the local I2RT DM3 browser interface."""

from __future__ import annotations

import argparse
import ipaddress
import signal

from incubator.nedf3.I2RT_DM3.controller import I2RTDM3Controller
from incubator.nedf3.I2RT_DM3.process_manager import (
    DEFAULT_DEVICE_CONFIG,
    DEFAULT_SAVER_CONFIG,
)

from .app import create_app


def _is_loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device-config", default=DEFAULT_DEVICE_CONFIG)
    parser.add_argument("--saver-config", default=DEFAULT_SAVER_CONFIG)
    parser.add_argument("--no-gripper-calibration", action="store_true")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow binding to a non-loopback interface",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not _is_loopback(args.host) and not args.allow_remote:
        parser.error("non-loopback binding requires --allow-remote")

    app = create_app(
        I2RTDM3Controller(
            device_config_name=args.device_config,
            saver_config_path=args.saver_config,
            gripper_calibration_supported=not args.no_gripper_calibration,
        )
    )
    service = app.config["I2RT_WEB_SERVICE"]

    def terminate(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
