"""Node and static contracts for the temporary checkpoint override UI."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NODE_TEST = ROOT / "tests" / "checkpoint_override_context.test.mjs"


def test_checkpoint_override_frontend_contract() -> None:
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
