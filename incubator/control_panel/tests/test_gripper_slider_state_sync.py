"""Frontend contracts for real gripper position synchronization."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _frontend_source() -> str:
    return (ROOT / "static" / "device_control.js").read_text(encoding="utf-8")


def test_closed_loop_bundle_has_a_fresh_cache_key() -> None:
    template = (ROOT / "templates" / "device_control.html").read_text(
        encoding="utf-8"
    )
    assert '/static/device_control.css?v=20260902-control-panel-save-v2' in template
    assert '/static/device_control.js?v=20260902-control-panel-save-v2' in template


def test_real_position_probes_are_event_triggered_only() -> None:
    source = _frontend_source()

    assert 'requestGripperActualAlignment(GRIPPER_SIDES, "startup")' in source
    assert 'requestGripperActualAlignment(GRIPPER_SIDES, "config_changed")' in source
    assert 'requestGripperActualAlignment(GRIPPER_SIDES, "config_auto_applied")' in source
    assert 'requestGripperActualAlignment(affectedSides, "tty_binding_changed")' in source
    assert 'requestGripperActualAlignment(pending.sides, "calibration_completed")' in source
    assert 'queueGripperActualProbe([side], `position_${outcome}`)' not in source
    assert 'queueGripperActualProbe([side], "position_request_failed")' not in source
    assert 'api("/api/grippers/probe"' in source

    queue_start = source.index("  function queueGripperActualProbe(")
    queue_end = source.index("  async function flushGripperActualProbes(", queue_start)
    queue_body = source[queue_start:queue_end]
    assert ".configured" not in queue_body
    assert "app.gripperProbeQueued.add(side);" in queue_body

    blocked_start = source.index("  function gripperProbeBlocked(")
    blocked_end = source.index("  function queueGripperActualProbe(", blocked_start)
    blocked_body = source[blocked_start:blocked_end]
    assert "!state?.configured" in blocked_body

    poll_start = source.index("  async function refreshState(")
    poll_end = source.index("  function initialStateAllowsBootstrap(", poll_start)
    poll_body = source[poll_start:poll_end]
    assert 'api("/api/grippers/probe"' not in poll_body

    config_switch = source[
        source.index("  async function selectConfigWithConfirmation("):
        source.index("  function stateFromPayload(")
    ]
    bootstrap = source[
        source.index("  async function maybeAutoApplyDefaultConfig("):
        source.index("  function shapeText(")
    ]
    tty_binding = source[
        source.index("  async function submitGripperPortBinding("):
        source.index("  async function runGripperCalibration(")
    ]
    assert "app.configSwitching = false;" in config_switch
    assert "void flushGripperActualProbes();" in config_switch
    assert "app.configSwitching = false;" in bootstrap
    assert "void flushGripperActualProbes();" in bootstrap
    assert "app.gripperPortPending.delete(bindingSide);" in tty_binding
    assert "void flushGripperActualProbes();" in tty_binding


def test_slider_sync_freezes_per_side_and_rejects_stale_snapshots() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the slider state-machine contract")

    source = _frontend_source()
    marker = "  initialize();\n})();"
    assert marker in source
    injected = r'''  globalThis.__gripperSliderSyncContract = {
    async run() {
      const probeEvents = [];
      const probeCalls = [];
      const fixture = ({
        revision = 10,
        leftRevision = 2,
        rightRevision = 2,
        leftActual = 20,
        rightActual = 80,
        leftStatus = "idle",
        rightStatus = "idle",
        leftCommand = "",
        rightCommand = "",
        leftCompleted = "",
        rightCompleted = "",
        leftAction = "",
        rightAction = "",
        leftError = "",
        rightError = "",
      } = {}) => ({
        revision,
        config_name: "cfg",
        status: [leftStatus, rightStatus].some((value) => value === "moving") ? "moving" : "idle",
        global_busy: false,
        busy_sides: [
          ...(leftStatus === "moving" ? ["left"] : []),
          ...(rightStatus === "moving" ? ["right"] : []),
        ],
        available_ports: ["/dev/ttyUSB0", "/dev/ttyUSB1"],
        can_bind_ports: true,
        capabilities: {
          calibrate: { can: true, blockers: [] },
          move: { can: true, blockers: [] },
          probe: { can: true, blockers: [] },
        },
        sides: {
          left: {
            configured: true,
            port: "/dev/ttyUSB1",
            port_present: true,
            revision: leftRevision,
            status: leftStatus,
            busy: leftStatus === "moving",
            actual_percent: leftActual,
            target_percent: null,
            last_command_id: leftCommand,
            last_completed_command_id: leftCompleted,
            last_completed_action: leftAction,
            last_error: leftError,
            can_position: leftStatus === "idle",
            can_calibrate: leftStatus === "idle",
            can_probe: !["moving", "calibrating"].includes(leftStatus),
          },
          right: {
            configured: true,
            port: "/dev/ttyUSB0",
            port_present: true,
            revision: rightRevision,
            status: rightStatus,
            busy: rightStatus === "moving",
            actual_percent: rightActual,
            target_percent: null,
            last_command_id: rightCommand,
            last_completed_command_id: rightCompleted,
            last_completed_action: rightAction,
            last_error: rightError,
            can_position: rightStatus === "idle",
            can_calibrate: rightStatus === "idle",
            can_probe: !["moving", "calibrating"].includes(rightStatus),
          },
        },
      });
      const reset = () => {
        app.state = { revision: 10, selected_config: "cfg", grippers: fixture() };
        app.selectedConfig = "cfg";
        app.configSwitching = false;
        app.gripperCalibratePending = false;
        app.gripperMovePending = new Set();
        app.gripperQueuedTargets = new Map();
        app.gripperDrafts = new Map();
        app.gripperDesiredTargets = new Map();
        app.gripperTargetSessions = new Map();
        app.gripperTargetTimers = new Map();
        app.gripperEditing = new Set();
        app.gripperCalibrationSync = null;
        app.gripperCalibrationSliderValues = new Map();
        app.gripperActualValues = new Map();
        app.gripperPositionSync = new Map();
        app.gripperProbeSync = new Map();
        app.gripperProbeQueued = new Set();
        app.gripperProbePending = new Set();
        app.gripperProbePromise = null;
        app.gripperSyncEpoch = 0;
        app.gripperSnapshotRequest = 0;
        app.gripperSnapshotRevision = null;
        app.gripperSideSnapshotRevisions = new Map();
        app.gripperSideSnapshotRequests = new Map();
        app.gripperTopSnapshotRequest = 0;
        app.gripperErrors = new Map();
        elements.grippers.left.position.value = "50";
        elements.grippers.right.position.value = "50";
      };
      const values = () => ({
        left: elements.grippers.left.position.value,
        right: elements.grippers.right.position.value,
      });
      scheduleStatePoll = () => {};
      showToast = () => {};
      reset();

      syncGripperActualValues();
      renderGrippers();
      const startup = values();

      app.gripperEditing.add("left");
      app.gripperDrafts.set("left", 44);
      elements.grippers.left.position.value = "44";
      app.state.grippers = fixture({ leftActual: 25, rightActual: 75 });
      syncGripperActualValues();
      renderGrippers();
      const editing = values();
      app.gripperEditing.delete("left");
      app.gripperDrafts.delete("left");
      syncGripperActualValues();
      renderGrippers();
      const editingReleased = values();

      const realQueueProbe = queueGripperActualProbe;
      queueGripperActualProbe = (sides, reason) => probeEvents.push({ sides: [...sides], reason });
      app.gripperDrafts.set("left", 60);
      app.gripperDesiredTargets.set("left", 60);
      app.gripperTargetSessions.set("left", {
        targetPercent: 60, startedAtMs: Date.now(), attempts: 1, lastSubmittedAtMs: Date.now(),
      });
      const accepted = beginGripperPositionSync({
        revision: 10,
        command: {
          accepted: true,
          action: "position",
          side: "left",
          command_id: "move-left-1",
          side_revision: 3,
        },
      }, "left", 60);
      app.state.grippers = fixture({
        leftRevision: 3,
        leftActual: 26,
        rightActual: 70,
        leftStatus: "moving",
        leftCommand: "move-left-1",
      });
      observeGripperPositionSync();
      syncGripperActualValues();
      renderGrippers();
      const pending = values();

      app.state.grippers = fixture({
        leftRevision: 3,
        leftActual: 61,
        rightActual: 69,
        leftCommand: "move-left-1",
        leftCompleted: "move-left-1",
        leftAction: "position",
      });
      const completed = observeGripperPositionSync();
      syncGripperActualValues();
      renderGrippers();
      const completedValues = values();
      const completedControl = {
        session: app.gripperTargetSessions.has("left"),
        desired: app.gripperDesiredTargets.get("left"),
        probeQueued: app.gripperProbeQueued.has("left"),
      };

      app.gripperDrafts.set("left", 0);
      app.gripperDesiredTargets.set("left", 0);
      app.gripperTargetSessions.set("left", {
        targetPercent: 0, startedAtMs: Date.now(), attempts: 1, lastSubmittedAtMs: Date.now(),
      });
      beginGripperPositionSync({
        revision: 10,
        command: {
          accepted: true,
          action: "position",
          side: "left",
          command_id: "move-left-off-target",
          side_revision: 4,
        },
      }, "left", 0);
      app.state.grippers = fixture({
        leftRevision: 4,
        leftActual: 50,
        rightActual: 69,
        leftCommand: "move-left-off-target",
        leftCompleted: "move-left-off-target",
        leftAction: "position",
      });
      const offTargetCompleted = observeGripperPositionSync();
      renderGrippers();
      const offTargetControl = {
        values: values(),
        session: app.gripperTargetSessions.has("left"),
        retryTimer: app.gripperTargetTimers.has("left"),
        desired: app.gripperDesiredTargets.get("left"),
        probeQueued: app.gripperProbeQueued.has("left"),
      };

      clearGripperTargetTimer("left");
      app.gripperDrafts.set("left", 35);
      app.gripperDesiredTargets.set("left", 35);
      app.gripperTargetSessions.set("left", {
        targetPercent: 35, startedAtMs: Date.now(), attempts: 1, lastSubmittedAtMs: Date.now(),
      });
      beginGripperPositionSync({
        revision: 10,
        command: {
          accepted: true,
          action: "position",
          side: "left",
          command_id: "move-left-failed",
          side_revision: 4,
        },
      }, "left", 35);
      app.state.grippers = fixture({
        leftRevision: 4,
        leftActual: null,
        leftStatus: "error",
        leftCommand: "move-left-failed",
        leftError: "injected failure",
      });
      const failed = observeGripperPositionSync();
      renderGrippers();
      const failedValues = values();
      const failedControl = {
        session: app.gripperTargetSessions.has("left"),
        retryTimer: app.gripperTargetTimers.has("left"),
        desired: app.gripperDesiredTargets.get("left"),
      };

      clearGripperTargetTimer("left");
      app.gripperDrafts.set("left", 40);
      app.gripperDesiredTargets.set("left", 40);
      app.gripperTargetSessions.set("left", {
        targetPercent: 40, startedAtMs: Date.now(), attempts: 1, lastSubmittedAtMs: Date.now(),
      });
      beginGripperPositionSync({
        revision: 10,
        command: {
          accepted: true,
          action: "position",
          side: "left",
          command_id: "move-left-timeout",
          side_revision: 5,
        },
      }, "left", 40);
      const timeoutStart = app.gripperPositionSync.get("left").startedAtMs;
      const timedOut = observeGripperPositionSync(timeoutStart + GRIPPER_POSITION_SYNC_TIMEOUT_MS);
      renderGrippers();
      const timedOutValues = values();
      const timedOutControl = {
        session: app.gripperTargetSessions.has("left"),
        retryTimer: app.gripperTargetTimers.has("left"),
        desired: app.gripperDesiredTargets.get("left"),
      };

      app.state.grippers = fixture({ leftRevision: 5, leftActual: null, rightActual: 88 });
      syncGripperActualValues();
      renderGrippers();
      const unknownPreserved = values();

      clearGripperTargetTimer("left");
      app.gripperDrafts.set("left", 52);
      app.gripperDesiredTargets.set("left", 52);
      app.gripperTargetSessions.set("left", {
        targetPercent: 52, startedAtMs: Date.now(), attempts: 1, lastSubmittedAtMs: Date.now(),
      });
      beginGripperPositionSync({
        revision: 10,
        command: {
          accepted: true,
          action: "position",
          side: "left",
          command_id: "move-left-independent",
          side_revision: 6,
        },
      }, "left", 52);
      app.state.grippers = fixture({
        leftRevision: 6,
        rightRevision: 3,
        leftActual: 10,
        rightActual: 91,
        leftStatus: "moving",
        leftCommand: "move-left-independent",
      });
      syncGripperActualValues();
      renderGrippers();
      const independent = values();

      reset();
      const freshAccepted = applyGripperSnapshot(
        fixture({ leftRevision: 7, leftActual: 33 }),
        { epoch: 0, requestId: 20, controllerRevision: 10 }
      );
      syncGripperActualValues();
      const staleAccepted = applyGripperSnapshot(
        fixture({ leftRevision: 7, leftActual: 99 }),
        { epoch: 0, requestId: 19, controllerRevision: 10 }
      );
      syncGripperActualValues();
      renderGrippers();
      const outOfOrder = { freshAccepted, staleAccepted, ...values() };

      const staleEpoch = app.gripperSyncEpoch;
      invalidateGripperSyncContext(["left"]);
      const oldContextAccepted = applyGripperSnapshot(
        fixture({ revision: 10, leftRevision: 8, leftActual: 97 }),
        { epoch: staleEpoch, requestId: 21, controllerRevision: 10 }
      );
      const currentContextAccepted = applyGripperSnapshot(
        fixture({ revision: 11, leftRevision: 8, leftActual: 47 }),
        { epoch: app.gripperSyncEpoch, requestId: 22, controllerRevision: 11 }
      );
      syncGripperActualValues();
      renderGrippers();
      const topologyRace = { oldContextAccepted, currentContextAccepted, ...values() };

      reset();
      queueGripperActualProbe = realQueueProbe;
      api = async (path, request = {}) => {
        probeCalls.push({ path, payload: JSON.parse(request.body) });
        const payload = JSON.parse(request.body);
        return {
          revision: 10,
          command: {
            accepted: true,
            action: "probe",
            side: payload.side,
            command_id: `probe-${payload.side}`,
            side_revision: 3,
          },
          grippers: fixture({
            leftRevision: payload.side === "left" ? 3 : 2,
            rightRevision: payload.side === "right" ? 3 : 2,
            leftStatus: payload.side === "left" ? "probing" : "idle",
            rightStatus: payload.side === "right" ? "probing" : "idle",
            leftCommand: payload.side === "left" ? "probe-left" : "",
            rightCommand: payload.side === "right" ? "probe-right" : "",
          }),
        };
      };
      app.gripperEditing.add("left");
      realQueueProbe(["left"], "editing");
      await Promise.resolve();
      const delayedProbe = { queued: app.gripperProbeQueued.has("left"), calls: probeCalls.length };
      app.gripperEditing.delete("left");
      await flushGripperActualProbes();
      const startedProbe = {
        queued: app.gripperProbeQueued.has("left"),
        calls: probeCalls.length,
        tracking: app.gripperProbeSync.get("left")?.commandId || null,
      };
      realQueueProbe(["left"], "merged");
      await flushGripperActualProbes();
      const mergedProbe = { queued: app.gripperProbeQueued.has("left"), calls: probeCalls.length };

      reset();
      const staleToasts = [];
      const stalePolls = [];
      const positionCalls = [];
      showToast = (message) => staleToasts.push(message);
      scheduleStatePoll = (delay) => stalePolls.push(delay);
      app.state.state_sync = {
        epoch: "epoch-slider",
        stale: false,
        authoritative: true,
        stale_after_ms: 3000,
      };
      app.stateEpoch = "epoch-slider";
      app.stateTransportFailed = true;
      app.stateTransportDeadlineAtMs = 0;
      api = async (path, request = {}) => {
        const payload = JSON.parse(request.body);
        positionCalls.push({ path, payload });
        return {
          revision: 10,
          command: {
            accepted: true,
            action: "position",
            side: payload.side,
            command_id: "move-left-after-fresh-state",
            side_revision: 3,
          },
          grippers: fixture({
            leftRevision: 3,
            leftActual: 20,
            leftStatus: "moving",
            leftCommand: "move-left-after-fresh-state",
          }),
        };
      };
      latchGripperTarget("left", 72);
      const staleSubmitted = await submitGripperTarget("left");
      const transientStale = {
        submitted: staleSubmitted,
        calls: positionCalls.length,
        toasts: staleToasts.length,
        polls: [...stalePolls],
        session: app.gripperTargetSessions.has("left"),
        retryTimer: app.gripperTargetTimers.has("left"),
        desired: app.gripperDesiredTargets.get("left"),
      };

      clearGripperTargetTimer("left");
      app.stateTransportFailed = false;
      app.stateTransportDeadlineAtMs = Date.now() + 3000;
      const recoveredSubmitted = await submitGripperTarget("left");
      const recoveredState = {
        submitted: recoveredSubmitted,
        calls: positionCalls.length,
        toasts: staleToasts.length,
        tracking: app.gripperPositionSync.get("left")?.commandId || null,
      };

      return {
        startup,
        editing,
        editingReleased,
        accepted,
        pending,
        completed,
        completedValues,
        completedControl,
        offTargetCompleted,
        offTargetControl,
        failed,
        failedValues,
        failedControl,
        timedOut,
        timedOutValues,
        timedOutControl,
        unknownPreserved,
        independent,
        probeEvents,
        outOfOrder,
        topologyRace,
        delayedProbe,
        startedProbe,
        mergedProbe,
        transientStale,
        recoveredState,
      };
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
  activeElement: null,
  getElementById(id) {{
    if (!elementStore.has(id)) elementStore.set(id, makeElement());
    return elementStore.get(id);
  }},
  querySelector() {{ return makeElement(); }},
  querySelectorAll() {{ return []; }},
  createElement() {{ return makeElement(); }},
}};
globalThis.window = {{ setTimeout() {{}}, clearTimeout() {{}}, confirm() {{ return false; }} }};
eval({json.dumps(test_source)});
globalThis.__gripperSliderSyncContract.run().then((result) => {{
  process.stdout.write(JSON.stringify(result));
}}).catch((error) => {{
  process.stderr.write(error.stack || String(error));
  process.exitCode = 1;
}});
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

    assert result["startup"] == {"left": "20", "right": "80"}
    assert result["editing"] == {"left": "44", "right": "75"}
    assert result["editingReleased"] == {"left": "25", "right": "75"}
    assert result["accepted"] is True
    assert result["pending"] == {"left": "60", "right": "70"}
    assert result["completed"] is True
    assert result["completedValues"] == {"left": "60", "right": "69"}
    assert result["completedControl"] == {
        "session": False,
        "desired": 60,
        "probeQueued": False,
    }
    assert result["offTargetCompleted"] is True
    assert result["offTargetControl"] == {
        "values": {"left": "0", "right": "69"},
        "session": True,
        "retryTimer": True,
        "desired": 0,
        "probeQueued": False,
    }
    assert result["failed"] is True
    assert result["failedValues"] == {"left": "35", "right": "69"}
    assert result["failedControl"] == {
        "session": True,
        "retryTimer": True,
        "desired": 35,
    }
    assert result["timedOut"] is True
    assert result["timedOutValues"] == {"left": "40", "right": "69"}
    assert result["timedOutControl"] == {
        "session": True,
        "retryTimer": True,
        "desired": 40,
    }
    assert result["unknownPreserved"] == {"left": "40", "right": "88"}
    assert result["independent"] == {"left": "52", "right": "91"}
    assert result["probeEvents"] == []
    assert result["outOfOrder"] == {
        "freshAccepted": True,
        "staleAccepted": True,
        "left": "33",
        "right": "80",
    }
    assert result["topologyRace"] == {
        "oldContextAccepted": False,
        "currentContextAccepted": True,
        "left": "47",
        "right": "80",
    }
    assert result["delayedProbe"] == {"queued": True, "calls": 0}
    assert result["startedProbe"] == {
        "queued": False,
        "calls": 1,
        "tracking": "probe-left",
    }
    assert result["mergedProbe"] == {"queued": False, "calls": 1}
    assert result["transientStale"] == {
        "submitted": False,
        "calls": 0,
        "toasts": 0,
        "polls": [0],
        "session": True,
        "retryTimer": True,
        "desired": 72,
    }
    assert result["recoveredState"] == {
        "submitted": True,
        "calls": 1,
        "toasts": 0,
        "tracking": "move-left-after-fresh-state",
    }
