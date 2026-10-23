#!/usr/bin/env python3
"""
Shared helpers for lightweight rerun timestamp logging.

These utilities make it easy for multiple modules to log into the same rerun
viewer session without duplicating initialization logic.
"""

import os
import threading
from typing import Dict, Any, Optional

import rerun as rr

from utils.logger_config import logger
from utils.rerun_visualization import set_time_context

_SHARED_RERUN_LOCK = threading.Lock()
_SHARED_RERUN_INITIALIZED = False


def ensure_shared_rerun_logging(app_name: str = "si_timestamp_logger") -> bool:
    """
    Lazily initialize rerun so multiple modules can share the same viewer session.

    Returns:
        bool: True when rerun is configured for visualization, False otherwise.
    """
    global _SHARED_RERUN_INITIALIZED

    if _SHARED_RERUN_INITIALIZED:
        return True

    with _SHARED_RERUN_LOCK:
        if _SHARED_RERUN_INITIALIZED:
            return True

        env_mode = os.getenv("RERUN_MODE", "spawn")
        env_connect = os.getenv("RERUN_CONNECT")
        env_save_path = os.getenv("RERUN_SAVE_PATH")
        env_serve_web = os.getenv("RERUN_SERVE_WEB", "0").lower() in ("1", "true", "yes")

        try:
            rr.init(app_name, recording_id="1")
            if env_mode == "spawn":
                rr.spawn()
                logger.info("Shared rerun logging: spawned external viewer")
            elif env_mode == "connect_grpc":
                if not env_connect:
                    raise ValueError("connect_grpc mode requires RERUN_CONNECT (e.g. grpc://HOST:PORT)")
                rr.connect_grpc(env_connect)
                logger.info(f"Shared rerun logging: connected to {env_connect}")
            elif env_mode == "serve_grpc":
                server_uri = rr.serve_grpc()
                logger.info(f"Shared rerun logging: serving at {server_uri}")
                if env_serve_web:
                    try:
                        web_url = rr.serve_web_viewer(connect_to=server_uri)
                        logger.info(f"Shared rerun logging: web viewer at {web_url}")
                    except Exception as exc:
                        logger.error(f"Failed to start rerun web viewer: {exc}")
            elif env_mode == "save":
                path = env_save_path or f"{app_name}.rrd"
                rr.save(path)
                logger.info(f"Shared rerun logging: saving to {path}")
            elif env_mode == "stdout":
                rr.stdout()
                logger.info("Shared rerun logging: streaming to STDOUT")
            elif env_mode == "none":
                logger.info("Shared rerun logging: buffering in-memory (no sink configured)")
            else:
                # Unknown mode: fall back to spawn
                rr.spawn()
                logger.warning(f"Unknown RERUN_MODE '{env_mode}', defaulting to spawn")
        except Exception as exc:
            logger.error(f"Failed to initialize shared rerun logging: {exc}")
            return False

        _SHARED_RERUN_INITIALIZED = True
        return True


def log_shared_rerun_timestamp(
    event_name: str,
    timestamp_ns: Optional[int],
    metadata: Optional[Dict[str, Any]] = None,
    time_type: str = "timestamp"
) -> None:
    """
    Log a timestamp (and optional metadata) into the shared rerun file.

    Args:
        event_name: Hierarchical name, e.g. "policy_connector/observation"
        timestamp_ns: Timestamp in nanoseconds. No-op if None.
        metadata: Optional dictionary with numeric or textual metadata.
        time_type: rerun timeline name.
    """
    if timestamp_ns is None:
        return

    if not ensure_shared_rerun_logging():
        return

    base_path = f"shared_timestamps/{event_name}"
    try:
        set_time_context(timestamp_ns, time_type=time_type)
        rr.log(f"{base_path}/timestamp_ns", rr.Scalars(float(timestamp_ns)))

        if metadata:
            for key, value in metadata.items():
                if value is None:
                    continue
                entity_path = f"{base_path}/{key}"
                if isinstance(value, (int, float)):
                    rr.log(entity_path, rr.Scalars(float(value)))
                else:
                    rr.log(entity_path, rr.TextLog(str(value)))
    except Exception as exc:
        logger.error(f"Failed to log rerun timestamp for '{event_name}': {exc}")
