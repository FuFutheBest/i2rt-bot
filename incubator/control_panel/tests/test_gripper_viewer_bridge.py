"""Node contracts for the accepted idle gripper-to-viewer fallback bridge."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "static" / "device_control.js"


def _source() -> str:
    return SOURCE_PATH.read_text(encoding="utf-8")


def test_bridge_is_post_arbitration_actual_only_and_adds_no_polling() -> None:
    source = _source()
    bridge_start = source.index("  function publishAcceptedGripperSnapshotToViewer(")
    bridge_end = source.index("  function applyGripperSnapshot(", bridge_start)
    bridge = source[bridge_start:bridge_end]
    apply_start = bridge_end
    apply_end = source.index("  function updateGrippersFromPayload(", apply_start)
    apply = source[apply_start:apply_end]

    assert "actual_percent" in bridge
    assert "target_percent" not in bridge
    assert "hardware_unknown" in bridge
    assert 'status === "idle"' in bridge
    assert "sync.stale === true" in bridge
    assert "sync.authoritative === false" in bridge
    assert "window.__dualArmViewer.updateGripperStates(payload, viewerOptions)" in bridge
    assert 'const viewerOptions = { source: "control-panel" }' in bridge
    assert "window.__pendingGripperViewerStates" in bridge
    assert "fetch(" not in bridge
    assert "setTimeout" not in bridge
    assert "setInterval" not in bridge

    assert apply.index("if (staleSide) continue;") < apply.index(
        "acceptedSides.push(side);"
    )
    assert apply.index("app.state.grippers = merged;") < apply.index(
        "publishAcceptedGripperSnapshotToViewer(acceptedSides);"
    )


def test_accepted_side_snapshots_drive_only_trusted_idle_actual_values_in_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    source = _source()
    marker = "  initialize();\n})();"
    assert marker in source
    injected = r'''  globalThis.__gripperViewerBridgeContract = {
    run() {
      const reset = () => {
        calls.length = 0;
        app.state = {
          revision: 10,
          state_sync: {
            stale: false,
            authoritative: true,
            stale_after_ms: 10000,
            age_ms: 0,
          },
          grippers: {
            revision: 10,
            config_name: "cfg",
            status: "idle",
            hardware_unknown: false,
            side_revisions: { left: 4, right: 4 },
            sides: {
              left: { revision: 4, status: "idle", actual_percent: 20 },
              right: { revision: 4, status: "idle", actual_percent: 80 },
            },
          },
        };
        app.selectedConfig = "cfg";
        app.gripperSyncEpoch = 0;
        app.gripperSnapshotRevision = 10;
        app.gripperTopSnapshotRequest = 10;
        app.gripperSideSnapshotRevisions = new Map([["left", 4], ["right", 4]]);
        app.gripperSideSnapshotRequests = new Map([["left", 10], ["right", 10]]);
        app.stateTransportFailed = false;
        app.stateTransportDeadlineAtMs = Date.now() + 10000;
      };
      const snapshot = (left, right, extras = {}) => ({
        revision: extras.revision ?? 10,
        config_name: "cfg",
        status: extras.status ?? "idle",
        hardware_unknown: extras.hardware_unknown ?? false,
        side_revisions: {
          left: left?.revision ?? 4,
          right: right?.revision ?? 4,
        },
        sides: { ...(left ? { left } : {}), ...(right ? { right } : {}) },
      });

      reset();
      const accepted = applyGripperSnapshot(snapshot(
        { revision: 5, status: "idle", actual_percent: 25, target_percent: 99 },
        { revision: 5, status: "idle", actual_percent: 75, target_percent: 1 }
      ), { epoch: 0, requestId: 11, controllerRevision: 10 });
      const idleCall = calls.at(-1);

      const staleSideAccepted = applyGripperSnapshot(snapshot(
        { revision: 4, status: "idle", actual_percent: 91 },
        { revision: 6, status: "idle", actual_percent: 70 }
      ), { epoch: 0, requestId: 10, controllerRevision: 10 });
      const partialCall = calls.at(-1);

      app.state.state_sync = { stale: true, authoritative: false };
      applyGripperSnapshot(snapshot(
        { revision: 7, status: "idle", actual_percent: 30 },
        null,
        { revision: 11 }
      ), { epoch: 0, requestId: 12, controllerRevision: 11 });
      const transportStaleCall = calls.at(-1);

      app.state.state_sync = { stale: false, authoritative: true };
      app.stateTransportFailed = false;
      app.stateTransportDeadlineAtMs = Date.now() + 10000;
      applyGripperSnapshot(snapshot(
        { revision: 8, status: "moving", actual_percent: 31 },
        { revision: 8, status: "idle", actual_percent: 69, hardware_unknown: true },
        { revision: 12 }
      ), { epoch: 0, requestId: 13, controllerRevision: 12 });
      const unsafeSideCall = calls.at(-1);

      applyGripperSnapshot(snapshot(
        { revision: 9, status: "idle", actual_percent: 32 },
        { revision: 9, status: "idle", actual_percent: 68 },
        { revision: 13, hardware_unknown: true }
      ), { epoch: 0, requestId: 14, controllerRevision: 13 });
      const globalUnknownCall = calls.at(-1);

      const beforeRejected = calls.length;
      const rejectedEpoch = applyGripperSnapshot(snapshot(
        { revision: 10, status: "idle", actual_percent: 40 }, null,
        { revision: 14 }
      ), { epoch: 99, requestId: 15, controllerRevision: 14 });
      const rejectedOldRevision = applyGripperSnapshot(snapshot(
        { revision: 11, status: "idle", actual_percent: 41 }, null,
        { revision: 12 }
      ), { epoch: 0, requestId: 16, controllerRevision: 12 });

      return {
        accepted,
        staleSideAccepted,
        idleCall,
        partialCall,
        transportStaleCall,
        unsafeSideCall,
        globalUnknownCall,
        rejectedEpoch,
        rejectedOldRevision,
        rejectedCallDelta: calls.length - beforeRejected,
        callCount: calls.length,
      };
    },
  };
})();'''
    test_source = source.replace(marker, injected, 1)
    harness = f'''
const elements = new Map();
function element() {{
  return {{
    value: "", disabled: false, hidden: false, textContent: "", dataset: {{}},
    classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
    setAttribute() {{}}, addEventListener() {{}}, querySelector() {{ return null; }},
    appendChild() {{}}, replaceChildren() {{}},
  }};
}}
const calls = [];
globalThis.document = {{
  hidden: false,
  activeElement: null,
  getElementById(id) {{ if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }},
  querySelector() {{ return element(); }},
  querySelectorAll() {{ return []; }},
  createElement() {{ return element(); }},
}};
globalThis.window = {{
  setTimeout() {{ return 1; }}, clearTimeout() {{}}, confirm() {{ return false; }},
  __dualArmViewer: {{ updateGripperStates(raw, options) {{ calls.push({{ raw, options }}); }} }},
}};
eval({json.dumps(test_source)});
process.stdout.write(JSON.stringify(globalThis.__gripperViewerBridgeContract.run()));
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

    assert result["accepted"] is True
    assert result["idleCall"] == {
        "raw": {
            "left": {"known": True, "actual_percent": 25},
            "right": {"known": True, "actual_percent": 75},
        },
        "options": {"source": "control-panel"},
    }
    assert result["staleSideAccepted"] is True
    assert result["partialCall"]["raw"] == {
        "right": {"known": True, "actual_percent": 70}
    }
    assert result["transportStaleCall"]["raw"] == {"left": {"known": False}}
    assert result["unsafeSideCall"]["raw"] == {
        "left": {"known": False},
        "right": {"known": False},
    }
    assert result["globalUnknownCall"]["raw"] == {
        "left": {"known": False},
        "right": {"known": False},
    }
    assert result["rejectedEpoch"] is False
    assert result["rejectedOldRevision"] is False
    assert result["rejectedCallDelta"] == 0
    assert result["callCount"] == 5


def test_device_template_does_not_embed_the_retired_live_viewer() -> None:
    template = (ROOT / "templates" / "device_control.html").read_text(
        encoding="utf-8"
    )
    assert "/static/arm_viewer.js" not in template
