#!/usr/bin/env python3
"""Inspect RoboTwin bridge prerequisites without creating a simulator or downloading files."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _check(path: Path, label: str) -> dict[str, Any]:
    return {"kind": "path", "label": label, "value": str(path), "ok": path.exists()}


def _module(module_name: str) -> dict[str, Any]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        check: dict[str, Any] = {
            "kind": "python_module",
            "label": module_name,
            "value": module_name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if module_name == "sapien" and "pkg_resources" in str(exc):
            check["hint"] = (
                "Run: python -m pip install -r requirements/requirements-robotwin.txt"
            )
        return check
    return {
        "kind": "python_module",
        "label": module_name,
        "value": module_name,
        "ok": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robotwin-root", type=Path, default=REPO_ROOT / "third_party" / "RoboTwin"
    )
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--act-source", type=Path)
    parser.add_argument("--act-ckpt", type=Path)
    parser.add_argument("--xpolicylab-root", type=Path)
    parser.add_argument("--xpolicylab-policy")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = args.robotwin_root.expanduser().resolve()
    checks = [
        _check(root / "envs", "RoboTwin envs"),
        _check(root / "envs" / f"{args.task}.py", f"task {args.task}"),
        _check(
            root / "env_cfg" / "task_config" / f"{args.task_config}.yml",
            f"task config {args.task_config}",
        ),
        _check(
            root / "assets" / "embodiments" / "aloha-agilex" / "config.yml",
            "Aloha embodiment assets",
        ),
    ]
    # The direct-drive bridge shims RoboTwin's planner-owning envs.robot
    # module, so MPLib and CuRobo are intentionally not runtime prerequisites.
    for module_name in (
        "sapien",
        "gymnasium",
        "h5py",
        "toppra",
        "scipy",
        "transforms3d",
        "yaml",
        "torch",
        "cv2",
        "open3d",
        "PIL",
        "trimesh",
        "imageio",
    ):
        checks.append(_module(module_name))
    if args.act_source is not None:
        checks.append(
            _check(
                args.act_source.expanduser().resolve() / "ACT", "standalone ACT source"
            )
        )
        checks.extend(
            _module(module_name) for module_name in ("torchvision", "IPython")
        )
    if args.act_ckpt is not None:
        checkpoint_dir = args.act_ckpt.expanduser().resolve()
        checks.extend(
            (
                _check(checkpoint_dir / "policy_last.ckpt", "ACT policy checkpoint"),
                _check(checkpoint_dir / "dataset_stats.pkl", "ACT normalization stats"),
            )
        )
    if args.xpolicylab_root is not None:
        xpolicylab_root = args.xpolicylab_root.expanduser().resolve()
        checks.extend(
            (
                _check(xpolicylab_root, "XPolicyLab checkout"),
                _check(xpolicylab_root / "client_server" / "ws", "XPolicyLab websocket client"),
            )
        )
        if args.xpolicylab_policy:
            checks.append(
                _check(
                    xpolicylab_root / "policy" / str(args.xpolicylab_policy),
                    f"XPolicyLab policy {args.xpolicylab_policy}",
                )
            )

    ok = all(check["ok"] for check in checks)
    result = {"robotwin_root": str(root), "ok": ok, "checks": checks}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for check in checks:
            status = (
                "OK"
                if check["ok"]
                else "MISSING"
                if check["kind"] == "path"
                else "FAILED"
            )
            details = ""
            if error := check.get("error"):
                details += f" ({error})"
            if hint := check.get("hint"):
                details += f"; {hint}"
            print(f"{status:7} {check['label']}: {check['value']}{details}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
