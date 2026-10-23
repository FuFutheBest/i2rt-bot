"""Regression contract for synchronizing sliders after gripper calibration."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _frontend_source() -> str:
    return (ROOT / "static" / "device_control.js").read_text(encoding="utf-8")


def test_backend_marks_only_a_successfully_completed_calibration() -> None:
    from incubator.control_panel import gripper_control

    manager = gripper_control.GripperControlManager(auto_monitor=False)
    slot = SimpleNamespace(
        lane="calibration",
        process=None,
        connection=None,
        owner_token="owner-calibration",
        ready=True,
        pid=123,
        pgid=123,
        inflight={
            "command_id": "calibration-success",
            "action": "calibrate",
            "side": "both",
        },
    )
    manager._workers = {"calibration": slot}
    for side in ("left", "right"):
        manager._sides[side]["configured"] = True
        manager._sides[side]["status"] = "calibrating"
        assert manager._sides[side]["last_completed_command_id"] is None
        assert manager._sides[side]["last_completed_action"] is None

    manager._apply_event_locked(
        {
            "event": "completed",
            "owner_token": "owner-calibration",
            "command_id": "calibration-success",
            "results": [
                {
                    "side": "left",
                    "target_percent": None,
                    "actual_percent": 17.2,
                    "actual_width_m": 0.0172,
                },
                {
                    "side": "right",
                    "target_percent": None,
                    "actual_percent": 83.8,
                    "actual_width_m": 0.0838,
                },
            ],
        }
    )

    assert manager._sides["left"]["last_completed_command_id"] == "calibration-success"
    assert manager._sides["right"]["last_completed_command_id"] == "calibration-success"
    assert manager._sides["left"]["last_completed_action"] == "calibrate"
    assert manager._sides["right"]["last_completed_action"] == "calibrate"
    assert manager._sides["left"]["actual_percent"] == pytest.approx(17.2)
    assert manager._sides["right"]["actual_percent"] == pytest.approx(83.8)

    failed_slot = SimpleNamespace(
        lane="calibration",
        process=None,
        connection=None,
        owner_token="owner-failed-calibration",
        ready=True,
        pid=124,
        pgid=124,
        inflight={
            "command_id": "calibration-failed",
            "action": "calibrate",
            "side": "both",
        },
    )
    manager._workers = {"calibration": failed_slot}
    for side in ("left", "right"):
        manager._sides[side]["status"] = "calibrating"
        manager._sides[side]["last_command_id"] = "calibration-failed"
    manager._apply_event_locked(
        {
            "event": "failed",
            "owner_token": "owner-failed-calibration",
            "command_id": "calibration-failed",
            "error": "injected calibration failure",
        }
    )
    for side in ("left", "right"):
        assert (
            manager._sides[side]["last_completed_command_id"]
            == "calibration-success"
        )
        assert manager._sides[side]["last_completed_action"] == "calibrate"


def test_calibration_slider_sync_uses_only_successful_actual_positions() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the calibration slider contract")

    source = _frontend_source()
    marker = "  initialize();\n})();"
    assert marker in source
    injected = r'''  globalThis.__calibrationSliderContract = {
    run() {
      const fixture = ({
        commandId,
        status = "idle",
        leftActual = null,
        rightActual = null,
        completed = false,
        lastError = null,
      }) => ({
        status,
        busy: status === "calibrating",
        last_error: lastError,
        capabilities: {
          calibrate: { can: status === "idle", blockers: [] },
          move: { can: status === "idle", blockers: [] },
        },
        sides: Object.fromEntries([
          ["left", leftActual],
          ["right", rightActual],
        ].map(([side, actual]) => [side, {
          status,
          configured: true,
          calibrated: true,
          actual_percent: actual,
          target_percent: null,
          last_command_id: commandId,
          last_completed_command_id: completed ? commandId : null,
          last_completed_action: completed ? "calibrate" : null,
          can_calibrate: status === "idle",
          can_position: status === "idle",
        }])),
      });
      const reset = (commandId, leftValue = "45", rightValue = "55") => {
        app.state = { revision: 1, grippers: fixture({ commandId, status: "calibrating" }) };
        app.gripperCalibrationSync = null;
        app.gripperCalibrationSliderValues = new Map();
        app.gripperDrafts = new Map();
        app.gripperEditing = new Set();
        app.gripperMovePending = new Set();
        app.gripperQueuedTargets = new Map();
        app.gripperErrors = new Map();
        elements.grippers.left.position.value = leftValue;
        elements.grippers.right.position.value = rightValue;
        beginGripperCalibrationSync({
          command: {
            action: "calibrate",
            side: "both",
            command_id: commandId,
            accepted: true,
            status: "pending",
          },
        });
      };

      reset("cal-success");
      app.state.grippers = fixture({
        commandId: "cal-success",
        leftActual: 17.2,
        rightActual: 83.8,
        completed: true,
      });
      const successObserved = observeGripperCalibrationSync();
      renderGrippers();
      const success = {
        observed: successObserved,
        left: elements.grippers.left.position.value,
        right: elements.grippers.right.position.value,
      };

      reset("cal-delayed");
      app.state.grippers = fixture({
        commandId: "cal-delayed",
        status: "calibrating",
        leftActual: 9,
        rightActual: 91,
        completed: false,
      });
      const delayedObserved = observeGripperCalibrationSync();
      const delayed = {
        observed: delayedObserved,
        tracking: app.gripperCalibrationSync?.commandId || null,
        snapshots: app.gripperCalibrationSliderValues.size,
      };

      reset("cal-failed");
      app.state.grippers = fixture({
        commandId: "cal-failed",
        status: "error",
        leftActual: null,
        rightActual: null,
        completed: false,
        lastError: "calibration failed",
      });
      const failedObserved = observeGripperCalibrationSync();
      renderGrippers();
      const failed = {
        observed: failedObserved,
        tracking: app.gripperCalibrationSync?.commandId || null,
        snapshots: app.gripperCalibrationSliderValues.size,
        left: elements.grippers.left.position.value,
        right: elements.grippers.right.position.value,
      };

      reset("cal-timeout", "42", "58");
      const timeoutStartedAt = app.gripperCalibrationSync.startedAtMs;
      const timeoutObserved = observeGripperCalibrationSync(
        timeoutStartedAt + GRIPPER_CALIBRATION_SYNC_TIMEOUT_MS
      );
      renderGrippers();
      const timedOut = {
        observed: timeoutObserved,
        tracking: app.gripperCalibrationSync?.commandId || null,
        snapshots: app.gripperCalibrationSliderValues.size,
        left: elements.grippers.left.position.value,
        right: elements.grippers.right.position.value,
      };

      reset("cal-drag", "31", "69");
      app.gripperEditing.add("left");
      app.gripperDrafts.set("left", 31);
      app.state.grippers = fixture({
        commandId: "cal-drag",
        leftActual: 12,
        rightActual: 88,
        completed: true,
      });
      observeGripperCalibrationSync();
      renderGrippers();
      const dragging = {
        left: elements.grippers.left.position.value,
        right: elements.grippers.right.position.value,
        leftSnapshot: app.gripperCalibrationSliderValues.get("left")?.actualPercent ?? null,
      };
      app.gripperEditing.delete("left");
      app.gripperDrafts.delete("left");
      renderGrippers();
      dragging.leftAfterCancel = elements.grippers.left.position.value;

      return { success, delayed, failed, timedOut, dragging };
    },
  };
})();'''
    test_source = source.replace(marker, injected, 1)
    harness = f'''
const elementStore = new Map();
function makeElement() {{
  return {{
    value: "0", disabled: false, hidden: false, textContent: "", title: "",
    dataset: {{}},
    classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
    setAttribute() {{}}, addEventListener() {{}}, querySelector() {{ return null; }},
    appendChild() {{}}, replaceChildren() {{}},
  }};
}}
globalThis.document = {{
  hidden: false,
  getElementById(id) {{
    if (!elementStore.has(id)) elementStore.set(id, makeElement());
    return elementStore.get(id);
  }},
  querySelector() {{ return makeElement(); }},
  querySelectorAll() {{ return []; }},
}};
globalThis.window = {{ setTimeout() {{}}, clearTimeout() {{}}, confirm() {{ return false; }} }};
eval({json.dumps(test_source)});
process.stdout.write(JSON.stringify(globalThis.__calibrationSliderContract.run()));
'''
    completed = subprocess.run(
        [node],
        input=harness,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=10,
    )
    result = json.loads(completed.stdout)

    assert result["success"] == {"observed": True, "left": "17", "right": "84"}
    assert result["delayed"] == {
        "observed": False,
        "tracking": "cal-delayed",
        "snapshots": 0,
    }
    assert result["failed"] == {
        "observed": False,
        "tracking": None,
        "snapshots": 0,
        "left": "45",
        "right": "55",
    }, result["failed"]
    assert result["timedOut"] == {
        "observed": False,
        "tracking": None,
        "snapshots": 0,
        "left": "42",
        "right": "58",
    }
    assert result["dragging"] == {
        "left": "31",
        "right": "88",
        "leftSnapshot": 12,
        "leftAfterCancel": "12",
    }
