"""Node behavior tests for ttyUSB permission prompt identity and stale bindings."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NODE_TEST = ROOT / "tests" / "gripper_permission_context.test.mjs"


def test_gripper_permission_context_frontend_contract() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    subprocess.run(
        [node, "--test", str(NODE_TEST)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )
