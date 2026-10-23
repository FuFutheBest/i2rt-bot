"""Tests for the standalone RoboTwin joint trace HTML generator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "visualization" / "robotwin_trace_visualizer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("robotwin_trace_visualizer_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visualizer_generates_joint14_trace_html(tmp_path: Path, monkeypatch) -> None:
    trace = tmp_path / "trace.jsonl"
    vector = list(range(14))
    trace.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "event_type": "sim_action_applied",
                    "timestamp_ns": 2_000_000_000,
                    "control_write_timestamp_ns": 1_995_000_000,
                    "executed_action_timestamp_ns": 2_000_000_000,
                    "target_qpos": vector,
                },
                {
                    "event_type": "sim_measured_state",
                    "timestamp_ns": 2_010_000_000,
                    "measured_qpos": vector,
                    "target_qpos": vector,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "trace.html"
    module = _load_module()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--trace", str(trace), "--output", str(output)])

    assert module.main() == 0
    html = output.read_text(encoding="utf-8")
    assert "RoboTwin Joint Trace" in html
    assert '"applied":1' in html
    assert '"measured":1' in html
