"""RPC worker for one I2RT leader or composite follower endpoint."""

from __future__ import annotations

import argparse
import logging
import signal
import threading
from pathlib import Path
from typing import Any

import portal

from utils.logger_config import logger as app_logger

from .config import SIDES, load_teleoperation_config
from .endpoints import create_follower_endpoint, create_leader_endpoint
from .runtime import WorkerControlRuntime


LOGGER = logging.getLogger(__name__)


def serve_worker(config_path: Path, side: str, role: str) -> None:
    config = load_teleoperation_config(config_path)
    arm_config = config.arms[side]
    endpoint = None
    server = None
    runtime = None
    stage = "constructing the hardware endpoint"
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        LOGGER.info("Stopping %s %s worker", side, role)
        stop_requested.set()
        if runtime is not None and (server is None or not server.running):
            runtime.request_stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        app_logger.info(f"I2RT worker {side}/{role}: {stage}")
        if role == "leader":
            endpoint = create_leader_endpoint(arm_config.leader.can_channel)
            port = arm_config.leader.rpc_port
        else:
            endpoint = create_follower_endpoint(
                arm_config.follower, label=f"{side}_lktech"
            )
            port = arm_config.follower.rpc_port
        if stop_requested.is_set():
            return

        runtime = WorkerControlRuntime(
            endpoint,
            role=role,
            side=side,
            control=config.control,
        )
        if stop_requested.is_set():
            return
        stage = "initializing the safe-hold control loop"
        app_logger.info(f"I2RT worker {side}/{role}: {stage}")
        if not runtime.start(
            config.control.startup_timeout_s,
            cancel_event=stop_requested,
        ):
            return
        if stop_requested.is_set():
            return
        stage = f"starting RPC server on port {port}"
        server = portal.Server(port)
        if stop_requested.is_set():
            return
        server.bind("ping", runtime.ping)
        server.bind("get_sample", runtime.get_sample)
        server.bind("get_status", runtime.get_status)
        server.bind("begin_control_session", runtime.begin_control_session)
        server.bind("set_phase", runtime.set_phase)
        server.bind("submit_control_frame", runtime.submit_control_frame)
        server.bind("end_control_session", runtime.end_control_session)
        server.bind("reset_fault", runtime.reset_fault)
        if stop_requested.is_set():
            return
        LOGGER.info("Serving %s %s endpoint on 127.0.0.1:%s", side, role, port)
        app_logger.info(f"I2RT worker {side}/{role}: ready on port {port}")
        stage = "serving RPC requests"
        server.start(block=False)
        while server.running:
            if stop_requested.wait(0.1):
                break
    except KeyboardInterrupt:
        LOGGER.info("I2RT worker %s/%s interrupted", side, role)
    except Exception as exc:
        app_logger.exception(
            f"I2RT worker {side}/{role} failed while {stage}: "
            f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        if server is not None and server.running:
            server.close()
        if runtime is not None:
            if runtime.close() and endpoint is not None:
                endpoint.close()
            else:
                LOGGER.error(
                    "Skipping concurrent endpoint close for blocked %s/%s worker",
                    side,
                    role,
                )
        elif endpoint is not None:
            endpoint.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--side", required=True, choices=SIDES)
    parser.add_argument("--role", required=True, choices=("leader", "follower"))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    serve_worker(args.config.resolve(), args.side, args.role)


if __name__ == "__main__":
    main()
