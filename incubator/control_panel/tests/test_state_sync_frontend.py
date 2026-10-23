"""Node contracts for stale-response rejection and observed-only controls."""

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


def test_actions_do_not_optimistically_claim_device_or_workflow_state() -> None:
    source = _source()
    device_start = source.index("  async function runDeviceAction(")
    device_end = source.index("  function renderLogDeviceOptions(", device_start)
    workflow_start = source.index("  async function runWorkflowAction(")
    workflow_end = source.index("  function cameraEndpoint(", workflow_start)

    device = source[device_start:device_end]
    workflow = source[workflow_start:workflow_end]
    assert "device.status =" not in device
    assert "previousStatus" not in device
    assert '"X-Control-Transition": transition.token' in device
    assert 'Prefer: "respond-async"' in device
    assert "yamCanPermissionRequired" in device
    assert "permission.affectedCanInterfaces.length" in device
    assert "requestGripperPermission({ force: true })" in device
    assert "app.state.workflow = payload.workflow" not in workflow
    assert '"X-Control-Transition": transition.token' in workflow
    assert 'Prefer: "respond-async"' in workflow
    assert "runtimeCanPermissionRequired" in workflow
    assert "permission.affectedCanInterfaces.length" in workflow
    assert "requestGripperPermission({ force: true })" in workflow


def test_visibility_and_fast_transition_polling_contract() -> None:
    source = _source()
    assert "statePending: 200" in source
    assert "stateRecovery: 250" in source
    assert "stateActiveFast: 500" in source
    assert "stateIdle: 2500" in source
    assert "if (document.hidden) return;" in source
    assert 'document.addEventListener("visibilitychange"' in source
    assert "app.stateRequestController?.abort();" in source
    assert 'new globalThis.EventSource("/api/events")' in source
    assert 'source.addEventListener("state", handleSnapshot)' in source
    assert 'source.addEventListener("heartbeat", handleSnapshot)' in source
    assert 'markStateTransportStale("页面已恢复' in source
    assert 'cache: "no-store"' in source


def test_transport_failure_and_handlers_recheck_fail_closed_before_post() -> None:
    source = _source()
    refresh_start = source.index("  async function refreshState(")
    refresh_end = source.index("  function initialStateAllowsBootstrap(", refresh_start)
    refresh = source[refresh_start:refresh_end]
    helper_start = source.index("  async function ensureControlActionAuthorized(")
    helper_end = source.index("  function transitionToken()", helper_start)
    helper = source[helper_start:helper_end]
    assert "await refreshState(false)" in helper
    assert "while (Date.now() < recoveryDeadline)" in helper
    assert "controlSafetyFailClosed()" in helper
    assert "controlStateTransportStale()" in helper
    assert "POLL_INTERVALS.controlRecoveryRetry" in helper
    assert (
        "controlActionStillAuthorized(action, expectedEpoch, { notify: false })"
        in helper
    )
    assert "markStateTransportStale(errorMessage(error))" in refresh
    assert "markStateTransportStale(\"真实设备状态请求超时\")" in refresh

    boundaries = [
        ("runGripperCalibration", "latchGripperTarget", "calibrate"),
        ("submitGripperTarget", "runGripperMove", "position"),
        ("runRecordingAction", "runWorkflowAction", "recording"),
        ("runWorkflowAction", "cameraEndpoint", "action"),
        ("runDeviceAction", "renderLogDeviceOptions", "action"),
    ]
    for name, next_name, action in boundaries:
        start = source.index(f"function {name}(")
        end = source.index(f"function {next_name}(", start)
        block = source[start:end]
        assert (
            "controlActionStillAuthorized" in block
            or "ensureControlActionAuthorized" in block
        )
        if name in {
            "runGripperCalibration",
            "runRecordingAction",
            "runWorkflowAction",
            "runDeviceAction",
        }:
            assert "await ensureControlActionAuthorized" in block
        if name in {"runGripperCalibration", "submitGripperTarget"}:
            retry = block.index("await withGripperPermissionRetry(")
            post = block.index("() => api(", retry)
            callback_authorization = (
                block.index("authorizeTargetSubmission", post)
                if name == "submitGripperTarget"
                else block.index("() => controlActionStillAuthorized", post)
            )
            preflight = (
                "controlActionStillAuthorized"
                if name == "submitGripperTarget"
                else "ensureControlActionAuthorized"
            )
            assert block.rfind(preflight, 0, retry) >= 0
            assert retry < post < callback_authorization
        elif name in {"runWorkflowAction", "runDeviceAction"}:
            post = block.index("const operation = () => api(")
            retry = block.index("await withGripperPermissionRetry(", post)
            callback_authorization = block.index(
                "() => controlActionStillAuthorized", retry
            )
            assert block.rfind("ensureControlActionAuthorized", 0, post) >= 0
            assert post < retry < callback_authorization
            assert block.index("await operation()", retry) > callback_authorization
        else:
            post = block.index("await api(")
            assert block.rfind("ensureControlActionAuthorized", 0, post) >= 0

    retry_start = source.index("  async function withGripperPermissionRetry(")
    retry_end = source.index("\n  function firstDefined", retry_start)
    retry = source[retry_start:retry_end]
    assert retry.index("if (!authorize()) return null") < retry.index(
        "return await operation()"
    )
    assert retry.index("!authorized || !authorize()") < retry.index(
        "return operation()"
    )


def test_control_authorization_recovers_transport_staleness_only_in_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")

    source = _source()
    helper_start = source.index("  function controlStateTransportStale()")
    helper_end = source.index("  function transitionToken()", helper_start)
    helpers = source[helper_start:helper_end]
    harness = f'''
const POLL_INTERVALS = {{
  stateWatchdog: 6000,
  controlRecoveryTimeout: 3000,
  controlRecoveryRetry: 150,
}};
const app = {{
  state: {{
    state_sync: {{ stale: true, authoritative: false }},
    safety: {{ fail_closed: false }},
  }},
  stateEpoch: "epoch-a",
  stateTransportFailed: true,
  stateTransportDeadlineAtMs: 0,
  stateTransportReason: "",
  devices: [],
}};
const notices = [];
let refreshCalls = 0;
function stateSyncMetadata(state = app.state) {{ return state.state_sync || {{}}; }}
function asNumber(value) {{ return Number.isFinite(Number(value)) ? Number(value) : null; }}
function deviceProgressPhase() {{ return ""; }}
function setServerStatus() {{}}
function setGlobalNotice(message) {{ notices.push(["notice", message]); }}
function showToast(message, kind) {{ notices.push([kind, message]); }}
function scheduleStatePoll() {{}}
function renderBindingControls() {{}}
function renderWorkflow() {{}}
function renderGrippers() {{}}
function renderRecording() {{}}
async function refreshState() {{
  refreshCalls += 1;
  if (refreshCalls === 2) {{
    app.state.state_sync = {{ stale: false, authoritative: true }};
    app.stateTransportFailed = false;
    app.stateTransportDeadlineAtMs = Date.now() + 10000;
  }}
}}
const window = {{ setTimeout(resolve) {{ resolve(); return 1; }} }};
{helpers}
(async () => {{
  const recovered = await ensureControlActionAuthorized("start", "epoch-a");
  const recoveryCalls = refreshCalls;

  app.state.safety = {{ fail_closed: true }};
  const beforeSafety = refreshCalls;
  const safetyRejected = await ensureControlActionAuthorized("start", "epoch-a");
  const safetyRefreshes = refreshCalls - beforeSafety;

  app.state.safety = {{ fail_closed: false }};
  app.stateEpoch = "epoch-b";
  const beforeEpoch = refreshCalls;
  const epochRejected = await ensureControlActionAuthorized("start", "epoch-a");
  const epochRefreshes = refreshCalls - beforeEpoch;

  process.stdout.write(JSON.stringify({{
    recovered,
    recoveryCalls,
    safetyRejected,
    safetyRefreshes,
    epochRejected,
    epochRefreshes,
  }}));
}})().catch((error) => {{
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
    assert json.loads(completed.stdout) == {
        "recovered": True,
        "recoveryCalls": 2,
        "safetyRejected": False,
        "safetyRefreshes": 0,
        "epochRejected": False,
        "epochRefreshes": 0,
    }


def test_state_and_sse_endpoints_explicitly_disable_caching() -> None:
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    state_start = server.index("    @app.get('/api/state')")
    state_end = server.index("    @app.get('/api/recording')", state_start)
    state_block = server[state_start:state_end]
    events_start = server.index("    @app.get('/api/events')")
    events_end = server.index("    @app.errorhandler", events_start)
    events_block = server[events_start:events_end]

    assert "no-store, max-age=0" in state_block
    assert "Pragma" in state_block
    assert "no-store, max-age=0" in events_block
    assert "X-Accel-Buffering" in events_block
    assert "if client_id in event_stream_clients" in events_block
    assert "Response(status=204)" in events_block
    assert "event_stream_clients.discard(client_id)" in events_block
    assert "response.call_on_close(release_client)" in events_block

    index_start = server.index("    @app.get('/')")
    index_end = server.index("    @app.get('/recordings')", index_start)
    index_block = server[index_start:index_end]
    assert "EVENT_STREAM_CLIENT_COOKIE" in index_block
    assert "response.set_cookie(" in index_block
    assert "httponly=True" in index_block


def test_snapshot_order_epoch_and_fail_closed_rules_in_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    source = _source()
    marker = "  initialize();\n})();"
    assert marker in source
    injected = r'''  globalThis.__stateSyncContract = {
    run() {
      app.state = {};
      app.stateEpoch = "";
      app.stateSequence = -1;
      app.stateEventRevision = -1;
      app.stateObservedRevision = -1;
      app.stateAppliedRequestSerial = 0;
      app.stateStablePolls = 0;
      const firstState = {
        state_sync: {
          epoch: "epoch-a", sequence: 2, event_revision: 2,
          observed_revision: 1, stale: false, authoritative: true,
        },
      };
      const first = acceptStateSnapshot(firstState, 1);
      app.state = firstState;
      const olderSequence = acceptStateSnapshot({
        state_sync: {
          epoch: "epoch-a", sequence: 1, event_revision: 2,
          observed_revision: 1, stale: false, authoritative: true,
        },
      }, 2);
      const olderEvent = acceptStateSnapshot({
        state_sync: {
          epoch: "epoch-a", sequence: 99, event_revision: 1,
          observed_revision: 1, stale: false, authoritative: true,
        },
      }, 3);
      const freshnessChange = acceptStateSnapshot({
        state_sync: {
          epoch: "epoch-a", sequence: 2, event_revision: 2,
          observed_revision: 1, stale: true, authoritative: false,
        },
      }, 4);
      app.controlTransitions.set("device:RizonRobot:2", { token: "old" });
      app.pendingActions.add("RizonRobot:2");
      app.workflowPending.add("runtime");
      const newEpoch = acceptStateSnapshot({
        state_sync: {
          epoch: "epoch-b", sequence: 1, event_revision: 1,
          observed_revision: 1, stale: false, authoritative: true,
        },
      }, 5);
      const pendingCountAfterEpoch = app.pendingActions.size + app.workflowPending.size;
      const transitionCountAfterEpoch = app.controlTransitions.size;
      const olderStart = beginClientTransition(
        "device:RizonRobot:1", "start", "running", 45000
      );
      const newerStop = beginClientTransition(
        "device:RizonRobot:1", "stop", "stopped", 75000
      );
      clearClientTransition("device:RizonRobot:1", olderStart.token);
      const newerStopPreserved =
        app.controlTransitions.get("device:RizonRobot:1")?.token === newerStop.token;
      app.controlTransitions.delete("device:RizonRobot:1");

      app.state = {
        state_sync: { stale: true, authoritative: false },
        safety: { fail_closed: true },
      };
      const staleStartBlocked = controlsFailClosed("start");
      const staleStopBlocked = controlsFailClosed("stop");
      app.state = {
        state_sync: {
          stale: false,
          authoritative: true,
          transitions: {
            "device:RizonRobot:1": {
              status: "acknowledged",
              desired: "running",
            },
          },
        },
        safety: { fail_closed: false },
      };
      app.stateTransportFailed = false;
      app.stateTransportDeadlineAtMs = Date.now() + 10000;
      app.state.safety = { fail_closed: true };
      app.devices = [normalizeDevice({
        class: "RizonRobot", id: 9, key: "RizonRobot:9",
        status: "degraded", running: true, ready: false,
        desired_state: "running",
        progress: { phase: "rizon_resolving_home" },
        safety: { fail_closed: true, initializing: true },
      }, 0)];
      const initializingStartBlocked = controlsFailClosed("start");
      const initializingOutcome = observedOutcome({
        scope: "device:RizonRobot:9", desired: "running",
      }).status;
      const initializingNotice = currentDeviceFailureNotice();
      app.state.safety = {
        fail_closed: true,
        estop: { active: true, known: true },
      };
      app.devices = [normalizeDevice({
        class: "RizonRobot", id: 9, key: "RizonRobot:9",
        status: "starting", running: true, ready: false,
        desired_state: "running",
        progress: { phase: "rizon_moving_home" },
        safety: {
          fail_closed: true, initializing: true,
          estop: { active: true, known: true },
        },
      }, 0)];
      const initializingEstopBlocked = controlsFailClosed("start");
      const initializingEstopOutcome = observedOutcome({
        scope: "device:RizonRobot:9", desired: "running",
      }).status;
      app.state.safety = { fail_closed: true };
      app.devices = [normalizeDevice({
        class: "RizonRobot", id: 9, key: "RizonRobot:9",
        status: "degraded", running: true, ready: false,
        desired_state: "running", progress: { phase: "ready" },
        safety: { fail_closed: true, initializing: false },
      }, 0)];
      const degradedStartBlocked = controlsFailClosed("start");
      const degradedOutcome = observedOutcome({
        scope: "device:RizonRobot:9", desired: "running",
      }).status;
      const degradedNotice = Boolean(currentDeviceFailureNotice());
      app.state.safety = { fail_closed: false };
      app.devices = [];
      app.selectedConfig = "review-config";
      app.configSwitching = false;
      app.pendingActions.add("RizonRobot:1");
      elements.configSelect.value = "review-config";
      const pendingButton = {
        dataset: {}, disabled: false, textContent: "",
        classList: { toggle() {} }, setAttribute() {},
      };
      updateDeviceToggle(pendingButton, {
        key: "RizonRobot:1",
        status: "stopped",
        running: false,
        ready: false,
        progress: { phase: "stopped" },
      }, "Rizon", true);
      const pendingStopButton = {
        dataset: {}, disabled: false, textContent: "",
        classList: { toggle() {} }, setAttribute() {},
      };
      app.state.state_sync.transitions["device:RizonRobot:2"] = {
        status: "acknowledged",
        desired: "stopped",
      };
      updateDeviceToggle(pendingStopButton, {
        key: "RizonRobot:2",
        status: "running",
        running: true,
        ready: true,
        progress: { phase: "ready" },
      }, "Rizon", true);

      const workflowStartButton = {
        dataset: {}, disabled: false, textContent: "",
        classList: { toggle() {} },
      };
      app.state.state_sync.transitions["workflow:runtime"] = {
        status: "acknowledged",
        desired: "running",
      };
      renderWorkflowButton(workflowStartButton, "runtime", {
        status: "starting", active: true, can_stop: true,
      }, {
        start: "启动运行时", starting: "运行时启动中",
        stop: "停止运行时", stopping: "运行时停止中",
      });
      const workflowStopButton = {
        dataset: {}, disabled: false, textContent: "",
        classList: { toggle() {} },
      };
      app.state.state_sync.transitions["workflow:inference"] = {
        status: "acknowledged",
        desired: "stopped",
      };
      renderWorkflowButton(workflowStopButton, "inference", {
        status: "stopping", running: true, can_stop: true,
      }, {
        start: "开始推理", starting: "推理启动中",
        stop: "停止推理", stopping: "推理停止中",
      });
      return {
        first,
        olderSequence,
        olderEvent,
        freshnessChange,
        newEpoch,
        epoch: app.stateEpoch,
        transitionCount: transitionCountAfterEpoch,
        pendingCount: pendingCountAfterEpoch,
        staleStartBlocked,
        staleStopBlocked,
        initializingStartBlocked,
        initializingOutcome,
        initializingNotice,
        initializingEstopBlocked,
        initializingEstopOutcome,
        degradedStartBlocked,
        degradedOutcome,
        degradedNotice,
        pendingStartAction: pendingButton.dataset.action,
        pendingStartStopDisabled: pendingButton.disabled,
        pendingStartLabel: pendingButton.textContent,
        pendingStopLabel: pendingStopButton.textContent,
        workflowStartLabel: workflowStartButton.textContent,
        workflowStopLabel: workflowStopButton.textContent,
        newerStopPreserved,
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
  }};
}}
globalThis.document = {{
  hidden: false,
  activeElement: null,
  getElementById(id) {{ if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }},
  querySelector() {{ return null; }},
  querySelectorAll() {{ return []; }},
  createElement() {{ return element(); }},
}};
globalThis.window = {{ setTimeout() {{ return 1; }}, clearTimeout() {{}}, confirm() {{ return false; }} }};
globalThis.Option = class Option {{ constructor(label, value) {{ this.label = label; this.value = value; }} }};
eval({json.dumps(test_source)});
process.stdout.write(JSON.stringify(globalThis.__stateSyncContract.run()));
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

    assert json.loads(completed.stdout) == {
        "first": True,
        "olderSequence": False,
        "olderEvent": False,
        "freshnessChange": True,
        "newEpoch": True,
        "epoch": "epoch-b",
        "transitionCount": 0,
        "pendingCount": 0,
        "staleStartBlocked": True,
        "staleStopBlocked": False,
        "initializingStartBlocked": False,
        "initializingOutcome": "pending",
        "initializingNotice": "",
        "initializingEstopBlocked": True,
        "initializingEstopOutcome": "failed",
        "degradedStartBlocked": True,
        "degradedOutcome": "failed",
        "degradedNotice": True,
        "pendingStartAction": "stop",
        "pendingStartStopDisabled": False,
        "pendingStartLabel": "启动中 · 紧急停止",
        "pendingStopLabel": "停止中",
        "workflowStartLabel": "运行时启动中",
        "workflowStopLabel": "推理停止中",
        "newerStopPreserved": True,
    }
