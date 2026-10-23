#!/usr/bin/env python3
"""Launch the RoboTwin bridge and its SuperInference policy connector as one trace run."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # A path-executed script otherwise exposes scripts/robotwin/, not the repo.
    sys.path.insert(0, str(REPO_ROOT))

from utils.runtime_profiles import (
    RuntimeProfile,
    RuntimeProfileError,
    resolve_runtime_profile,
    select_runtime_python,
)

DEFAULT_CONFIG_NAME = "policy_infer/robotwin/config_infer_robotwin_replay_joint14"
DEFAULT_SUMMARY_SHM = "robotwin_summary_data"


def _ensure_no_user_site_packages() -> None:
    """Relaunch with user/system site packages hidden from this Conda prefix."""
    if os.environ.get("PYTHONNOUSERSITE") == "1":
        return
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


def _make_run_id() -> str:
    return f"robotwin_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _compose_config(config_name: str) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base="1.3", config_dir=str(REPO_ROOT / "configs")
    ):
        return compose(config_name=config_name)


def _resolve_preflight_settings(config_name: str) -> dict[str, str | None]:
    """Read only the values that make preflight specific to this preset."""
    from omegaconf import OmegaConf

    config = _compose_config(config_name)
    robotwin = OmegaConf.to_container(
        OmegaConf.select(config, "robotwin", default={}), resolve=True
    )
    if not isinstance(robotwin, dict):
        raise ValueError(f"RoboTwin preset {config_name!r} has no robotwin mapping")
    policy = OmegaConf.to_container(
        OmegaConf.select(config, "policy", default={}), resolve=True
    )
    policy = policy if isinstance(policy, dict) else {}
    params = policy.get("params", {})
    params = params if isinstance(params, dict) else {}
    settings: dict[str, str | None] = {
        "robotwin_root": robotwin.get("root"),
        "task": robotwin.get("task_name"),
        "task_config": robotwin.get("task_config"),
        "act_source": None,
        "act_ckpt": None,
        "xpolicylab_root": None,
        "xpolicylab_policy": None,
    }
    if policy.get("class") == "RoboTwinACTPolicy":
        settings["act_source"] = params.get("act_source")
        settings["act_ckpt"] = params.get("ckpt_dir")
    if policy.get("class") == "RoboTwinXPolicyLabPolicy":
        settings["xpolicylab_root"] = params.get("xpolicylab_root")
        settings["xpolicylab_policy"] = params.get("policy_name")
    missing = [
        key for key in ("robotwin_root", "task", "task_config") if not settings[key]
    ]
    if missing:
        raise ValueError(
            f"RoboTwin preset {config_name!r} is missing {', '.join(missing)}"
        )
    return {
        key: None if value is None else str(value) for key, value in settings.items()
    }


def _preflight_command(
    settings: dict[str, str | None],
    *,
    python_path: Path | str | None = None,
) -> list[str]:
    command = [
        str(python_path or sys.executable),
        str(REPO_ROOT / "scripts" / "robotwin" / "preflight.py"),
        "--robotwin-root",
        str(settings["robotwin_root"]),
        "--task",
        str(settings["task"]),
        "--task-config",
        str(settings["task_config"]),
    ]
    if settings["act_source"] or settings["act_ckpt"]:
        if not settings["act_source"] or not settings["act_ckpt"]:
            raise ValueError(
                "RoboTwin ACT policy requires both act_source and ckpt_dir"
            )
        command.extend(
            ("--act-source", settings["act_source"], "--act-ckpt", settings["act_ckpt"])
        )
    if settings.get("xpolicylab_root") or settings.get("xpolicylab_policy"):
        if not settings.get("xpolicylab_root") or not settings.get("xpolicylab_policy"):
            raise ValueError(
                "RoboTwin XPolicyLab policy requires both xpolicylab_root and policy_name"
            )
        command.extend(
            (
                "--xpolicylab-root",
                str(settings["xpolicylab_root"]),
                "--xpolicylab-policy",
                str(settings["xpolicylab_policy"]),
            )
        )
    return command


def _resolve_xpolicylab_server_settings(
    config_name: str,
    *,
    force_enabled: bool = False,
    ckpt_name_override: Optional[str] = None,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    config = _compose_config(config_name)
    robotwin = OmegaConf.to_container(
        OmegaConf.select(config, "robotwin", default={}), resolve=True
    )
    robotwin = robotwin if isinstance(robotwin, dict) else {}
    server = robotwin.get("xpolicylab_server", {})
    if not isinstance(server, dict) or not (
        _as_bool(server.get("enabled", False)) or force_enabled
    ):
        return {"enabled": False}
    settings = {
        "enabled": True,
        "xpolicylab_root": robotwin.get("xpolicylab_root") or "third_party/RoboTwin/XPolicyLab",
        "policy_name": server.get("policy_name"),
        "bench_name": server.get("bench_name", "RoboTwin"),
        "task_name": server.get("task_name") or robotwin.get("task_name"),
        "ckpt_name": ckpt_name_override or server.get("ckpt_name"),
        "env_cfg_type": server.get("env_cfg_type", "aloha_agilex"),
        "action_type": server.get("action_type", "joint"),
        "seed": server.get("seed", robotwin.get("seed", 0)),
        "policy_gpu_id": server.get("policy_gpu_id", 0),
        "runtime_env": server.get("runtime_env", "uv"),
        "port": server.get("port"),
        "host": server.get("host", "127.0.0.1"),
    }
    for key in (
        "policy_name",
        "task_name",
        "ckpt_name",
        "env_cfg_type",
        "action_type",
        "port",
        "host",
    ):
        value = settings.get(key)
        if value is None or str(value).strip().lower() in {"", "none", "null", "not_configured"}:
            raise ValueError(
                f"robotwin.xpolicylab_server.enabled=true requires {key}"
            )
    return settings


def _validate_xpolicylab_server_runtime(settings: dict[str, Any]) -> None:
    runtime_env = settings.get("runtime_env")
    if runtime_env is None or str(runtime_env).strip().lower() in {
        "",
        "none",
        "null",
        "not_configured",
    }:
        raise RuntimeProfileError(
            "XPolicyLab policy server runtime is not calibrated; bind a verified "
            "policy-server profile for this config or set the legacy runtime "
            "environment override explicitly"
        )


def _xpolicylab_server_command(settings: dict[str, Any]) -> list[str]:
    root = Path(str(settings["xpolicylab_root"])).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    script = root / "policy" / str(settings["policy_name"]) / "setup_eval_policy_server.sh"
    if not script.is_file():
        raise FileNotFoundError(f"XPolicyLab server launcher not found: {script}")
    return [
        "bash",
        str(script),
        str(settings["bench_name"]),
        str(settings["task_name"]),
        str(settings["ckpt_name"]),
        str(settings["env_cfg_type"]),
        str(settings["action_type"]),
        str(settings["seed"]),
        str(settings["policy_gpu_id"]),
        str(settings["runtime_env"]),
        str(settings["port"]),
        str(settings["host"]),
    ]


def _runtime_selection(
    role: str,
    config_name: str,
    *,
    environment_names: tuple[str, ...] = (),
) -> tuple[Path, Optional[RuntimeProfile]]:
    selection = select_runtime_python(
        role,
        environment_names=environment_names,
        config_name=config_name,
        fallback=sys.executable,
        repo_root=REPO_ROOT,
        require_verified_profile=True,
    )
    assert selection is not None
    return selection.python, selection.profile


def _policy_server_profile(config_name: str) -> Optional[RuntimeProfile]:
    profile = resolve_runtime_profile(
        "policy-server",
        config_name=config_name,
        repo_root=REPO_ROOT,
    )
    if profile is not None and not profile.verified:
        raise RuntimeProfileError(
            f"runtime profile {profile.name!r} is unverified or stale; run verify first"
        )
    return profile


def _child_environment(
    profile: Optional[RuntimeProfile],
    *,
    config_name: str,
    run_id: str,
) -> dict[str, str]:
    environment = (
        profile.process_environment(os.environ, config_name=config_name)
        if profile is not None
        else dict(os.environ)
    )
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "SI_TIMESTAMP_RUN_ID": run_id,
            "SI_ACTION_STATE_XYZ_RUN_ID": run_id,
        }
    )
    return environment


def _apply_xpolicylab_server_profile(
    settings: dict[str, Any],
    profile: Optional[RuntimeProfile],
) -> None:
    if profile is None:
        return
    if str(settings.get("policy_name")) == "Pi_05":
        expected_python = profile.cwd / ".venv/bin/python"
        try:
            matches_uv_project = expected_python.absolute() == profile.python.absolute()
        except OSError:
            matches_uv_project = False
        if not matches_uv_project:
            raise RuntimeProfileError(
                "Pi_05 policy-server profile cwd must be the uv project containing .venv/bin/python"
            )
        settings["runtime_env"] = str(profile.cwd)
        return
    if str(settings.get("policy_name")) == "FastWAM":
        verification = profile.verification or {}
        raw_prefix = verification.get("prefix")
        if not isinstance(raw_prefix, str) or not raw_prefix.strip():
            raise RuntimeProfileError(
                "FastWAM policy-server profile has no verified environment prefix"
            )
        prefix = Path(os.path.abspath(os.path.expanduser(raw_prefix)))
        try:
            profile.python.absolute().relative_to(prefix / "bin")
        except ValueError as exc:
            raise RuntimeProfileError(
                "FastWAM policy-server Python must be inside its verified prefix/bin"
            ) from exc
        settings["runtime_env"] = str(prefix)


def _wait_for_summary_shm(
    name: str, device_process: subprocess.Popen[object], timeout_s: float
) -> None:
    from utils.shm_utils import connect_to_shared_memory

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if device_process.poll() is not None:
            raise RuntimeError(
                f"DeviceStarter exited before creating summary SHM (code {device_process.returncode})"
            )
        try:
            channel = connect_to_shared_memory(name, read_only=True, log_missing=False)
        except FileNotFoundError:
            time.sleep(0.25)
            continue
        channel.close()
        return
    raise TimeoutError(
        f"Timed out waiting for summary SHM {name!r} after {timeout_s:.1f}s"
    )


def _wait_for_tcp_port(
    host: str,
    port: int,
    process: subprocess.Popen[object],
    timeout_s: float,
    *,
    label: str,
) -> None:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited before opening {host}:{port} (code {process.returncode})")
        try:
            with socket.create_connection((connect_host, int(port)), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {label} at {host}:{port} after {timeout_s:.1f}s")


def _terminate(
    process: Optional[subprocess.Popen[object]],
    label: str,
    *,
    graceful_timeout_s: float = 12.0,
) -> None:
    try:
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=graceful_timeout_s)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
        print(f"Stopped {label}")
    except KeyboardInterrupt:
        # A Ctrl+C can reach the launcher again while it is already relaying
        # SIGINT to children. Their process group has the same stop request;
        # avoid turning a normal user shutdown into a traceback.
        return


def main() -> int:
    _ensure_no_user_site_packages()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--summary-shm", default=DEFAULT_SUMMARY_SHM)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument(
        "--gui", action="store_true", help="Enable the RoboTwin SAPIEN Viewer"
    )
    parser.add_argument(
        "--open-trace",
        action="store_true",
        help="Open the generated trace page after clean shutdown",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--start-xpolicylab-server",
        action="store_true",
        help="Start the XPolicyLab policy server described by the selected RoboTwin config",
    )
    parser.add_argument(
        "--xpolicylab-ckpt-name",
        default=None,
        help="Override robotwin.xpolicylab_server.ckpt_name when starting an XPolicyLab server",
    )
    args = parser.parse_args()

    run_id = args.run_id or os.environ.get("SI_TIMESTAMP_RUN_ID") or _make_run_id()
    device_python, device_profile = _runtime_selection(
        "device",
        args.config_name,
        environment_names=("SUPERINFERENCE_DEVICE_PYTHON",),
    )
    policy_python, policy_profile = _runtime_selection(
        "policy",
        args.config_name,
        environment_names=("SUPERINFERENCE_INFERENCE_PYTHON",),
    )
    device_environment = _child_environment(
        device_profile,
        config_name=args.config_name,
        run_id=run_id,
    )
    policy_environment = _child_environment(
        policy_profile,
        config_name=args.config_name,
        run_id=run_id,
    )

    if not args.skip_preflight:
        preflight = _preflight_command(
            _resolve_preflight_settings(args.config_name),
            python_path=device_python,
        )
        preflight_result = subprocess.run(
            preflight,
            cwd=REPO_ROOT,
            env=device_environment,
            check=False,
        )
        if preflight_result.returncode:
            return int(preflight_result.returncode)
    xpolicylab_server = _resolve_xpolicylab_server_settings(
        args.config_name,
        force_enabled=args.start_xpolicylab_server,
        ckpt_name_override=args.xpolicylab_ckpt_name,
    )
    policy_server_profile = (
        _policy_server_profile(args.config_name)
        if xpolicylab_server.get("enabled")
        else None
    )
    _apply_xpolicylab_server_profile(xpolicylab_server, policy_server_profile)
    if xpolicylab_server.get("enabled"):
        _validate_xpolicylab_server_runtime(xpolicylab_server)

    # Both processes compose the same config independently. Keep their summary
    # SHM name synchronized when the caller overrides it.
    overrides: list[str] = [f"summary_shm={args.summary_shm}"]
    if args.gui:
        overrides.append("robotwin.enable_gui=true")
    if args.open_trace:
        overrides.append("robotwin.trace_visualizer.open_browser=true")
    device_command = [
        str(device_python),
        "device_starter.py",
        "--config-name",
        args.config_name,
        *overrides,
    ]
    policy_command = [
        str(policy_python),
        "consumer/inference/robotwin_policy_infer.py",
        "--config-name",
        args.config_name,
        "--summary-shm",
        args.summary_shm,
    ]

    device_process: Optional[subprocess.Popen[object]] = None
    policy_process: Optional[subprocess.Popen[object]] = None
    policy_server_process: Optional[subprocess.Popen[object]] = None
    try:
        if xpolicylab_server.get("enabled"):
            server_env = os.environ.copy()
            server_env.pop("PYTHONNOUSERSITE", None)
            if policy_server_profile is not None:
                server_env = policy_server_profile.process_environment(
                    server_env,
                    config_name=args.config_name,
                )
            policy_server_process = subprocess.Popen(
                _xpolicylab_server_command(xpolicylab_server),
                cwd=REPO_ROOT,
                env=server_env,
            )
            _wait_for_tcp_port(
                str(xpolicylab_server["host"]),
                int(xpolicylab_server["port"]),
                policy_server_process,
                args.startup_timeout,
                label="RoboTwin XPolicyLab policy server",
            )
        device_process = subprocess.Popen(
            device_command,
            cwd=REPO_ROOT,
            env=device_environment,
        )
        _wait_for_summary_shm(args.summary_shm, device_process, args.startup_timeout)
        policy_process = subprocess.Popen(
            policy_command,
            cwd=REPO_ROOT,
            env=policy_environment,
        )
        print(f"RoboTwin SuperInference run id: {run_id}")
        while True:
            if device_process.poll() is not None:
                raise RuntimeError(
                    "RoboTwin DeviceStarter exited unexpectedly "
                    f"(code {device_process.returncode})"
                )
            if policy_process.poll() is not None:
                raise RuntimeError(
                    "RoboTwin policy connector exited unexpectedly "
                    f"(code {policy_process.returncode})"
                )
            if (
                policy_server_process is not None
                and policy_server_process.poll() is not None
            ):
                raise RuntimeError(
                    "RoboTwin XPolicyLab policy server exited unexpectedly "
                    f"(code {policy_server_process.returncode})"
                )
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        _terminate(policy_process, "RoboTwin policy connector")
        _terminate(device_process, "RoboTwin DeviceStarter", graceful_timeout_s=75.0)
        _terminate(policy_server_process, "RoboTwin XPolicyLab policy server", graceful_timeout_s=30.0)


if __name__ == "__main__":
    raise SystemExit(main())
