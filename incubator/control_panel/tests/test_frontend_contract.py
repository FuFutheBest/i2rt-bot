"""Static contracts for the dependency-free device control frontend."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _frontend_source() -> str:
    candidates = (
        ROOT / "control_panel" / "static" / "device_control.js",
        ROOT / "static" / "device_control.js",
    )
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise AssertionError(
        "device_control.js was not found under incubator/control_panel/static or static"
    )


def _arm_viewer_source() -> str:
    return (ROOT / "static" / "arm_viewer.js").read_text(encoding="utf-8")


def _template_source() -> str:
    return (ROOT / "templates" / "device_control.html").read_text(encoding="utf-8")


def _css_source() -> str:
    return (ROOT / "static" / "device_control.css").read_text(encoding="utf-8")


def _function_body(source: str, name: str, next_name: str) -> str:
    start_match = re.search(rf"  (?:async )?function {re.escape(name)}\(", source)
    end_match = re.search(rf"  (?:async )?function {re.escape(next_name)}\(", source[start_match.end():] if start_match else "")
    if start_match is None or end_match is None:
        raise AssertionError(f"Unable to locate function boundary: {name} -> {next_name}")
    start = start_match.start()
    end = start_match.end() + end_match.start()
    return source[start:end]


def test_desktop_control_rail_scrolls_instead_of_clipping_controls() -> None:
    css = _css_source()
    match = re.search(r"\.control-rail\s*\{(?P<body>.*?)\}", css, re.S)

    assert match is not None
    rule = match.group("body")
    assert "overflow-x: hidden;" in rule
    assert "overflow-y: auto;" in rule
    assert "overscroll-behavior-y: contain;" in rule
    assert "scrollbar-gutter: stable;" in rule
    assert "overflow: hidden;" not in rule


def test_stop_timeout_is_a_visible_safety_error() -> None:
    source = _frontend_source()

    assert 'stop_timeout: "安全停机超时"' in source
    assert re.search(
        r"ERROR_STATUSES\s*=\s*new Set\([^;]*\"stop_timeout\"", source, re.S
    )
    assert re.search(
        r"ACTIVE_STATUSES\s*=\s*new Set\([^;]*\"stop_timeout\"", source, re.S
    )
    assert 'return status === "stop_timeout" ? "failed" : status;' in source
    assert "安全停机超时 · 最后错误：" in source

    normalize = _function_body(source, "normalizeStatus", "timestampAgeMs")
    assert 'if (lifecycleStatus === "stop_timeout") return "stop_timeout";' in normalize
    assert normalize.index("if (STATUS_LABELS[status])") < normalize.index(
        "if (running === true)"
    )

    actions = _function_body(source, "actionCapability", "presentationStatus")
    assert 'action === "stop"' in actions
    assert '"stop_timeout"' in actions
    assert 'action === "restart"' in actions


def test_preview_polling_keeps_top_level_and_nested_metadata() -> None:
    source = _frontend_source()
    normalize = _function_body(
        source, "normalizePreviewMetadata", "createCameraCard"
    )

    assert "payload?.status || payload" not in source
    assert "payload.metadata" in normalize
    assert "payload.data?.metadata" in normalize
    assert 'typeof payload.status === "object"' in normalize
    assert "return nested ? { ...payload, ...nested } : payload;" in normalize
    assert "normalizePreviewMetadata(payload)" in source


def test_camera_routes_are_class_and_id_based_not_shm_based() -> None:
    source = _frontend_source()
    endpoint = _function_body(source, "cameraEndpoint", "normalizePreviewMetadata")

    assert "encodeURIComponent(device.className)" in endpoint
    assert "encodeURIComponent(String(device.id))" in endpoint
    assert "device.shm" not in endpoint


def test_process_controls_live_in_visual_panels_not_left_device_list() -> None:
    source = _frontend_source()
    template = _template_source()

    assert 'id="device-list"' not in template
    assert 'id="apply-config-button"' not in template
    assert 'id="refresh-button"' not in template
    assert "elements.deviceList" not in source
    assert "renderDevices" not in source
    assert 'id="robot-left-binding-select"' in template
    assert 'id="robot-right-binding-select"' in template
    assert 'id="robot-device-toggle"' in template
    assert 'id="rizon-device-requirement"' in template
    assert 'id="i2rt-head-device-toggle"' in template
    assert 'id="i2rt-head-device-requirement"' in template
    assert 'data-role="camera-binding-select"' in source
    assert 'data-role="camera-device-toggle"' in source
    assert 'role="switch"' in template


def test_force_stop_all_control_is_explicit_async_and_double_confirmed() -> None:
    source = _frontend_source()
    template = _template_source()
    css = _css_source()
    body = _function_body(source, 'runForceStopAll', 'runDeviceAction')

    assert 'id="force-stop-all"' in template
    assert 'id="force-stop-hint"' in template
    assert 'force-stop-button topbar-force-stop-button' in template
    assert template.index('class="topbar-actions"') < template.index(
        'id="force-stop-all"'
    ) < template.index('</header>')
    assert '/api/processes/force-stop-all' in body
    assert 'JSON.stringify({ confirmation: "FORCE_STOP_ALL" })' in body
    assert body.count('await confirmOperation(') == 2
    assert 'await yieldForPendingPaint()' in body
    assert 'scheduleStatePoll(0)' in body
    assert 'elements.forceStopAll?.addEventListener' in source
    assert '.topbar-force-stop-button' in css
    assert '.force-stop-button.is-busy' in css


def test_gripper_visual_variant_control_is_available_in_visual_qa() -> None:
    visual_qa = (ROOT / "tests" / "umi3_visual_qa.html").read_text(
        encoding="utf-8"
    )
    css = _css_source()
    viewer = _arm_viewer_source()

    assert 'id="tool-variant-control"' in visual_qa
    assert 'data-tool-visual-variant="soft"' in visual_qa
    assert 'data-tool-visual-variant="hard"' in visual_qa
    assert 'class="tool-variant-control tool-variant-control--viewer"' in visual_qa
    assert visual_qa.index('id="qa-viewer-stage"') < visual_qa.index(
        'id="tool-variant-control"'
    )
    assert "软指夹爪" in visual_qa
    assert "硬指夹爪" in visual_qa
    assert 'role="group"' in visual_qa
    assert ".tool-variant-segmented" in css
    assert ".tool-variant-control--viewer" in css
    assert '.tool-variant-segmented button[aria-pressed="true"]' in css
    assert "hasSelectableVariants" in viewer
    assert "setToolVisualVariant(value)" in viewer
    assert "TOOL_VISUAL_VARIANT_STORAGE_KEY" in viewer
    assert 'if (tool.visual?.mirrorAxis === "y") mounted.scale.y *= -1;' in viewer
    assert "/api/" not in viewer[
        viewer.index("setToolVisualVariant(value)"):
        viewer.index("_setupScene()")
    ]


def test_two_stage_runtime_and_inference_controls_are_fail_closed() -> None:
    source = _frontend_source()
    template = _template_source()
    css = _css_source()

    assert 'id="runtime-toggle"' in template
    assert 'id="inference-toggle"' in template
    assert 'id="workflow-blockers"' in template
    assert '启动运行时' in template
    assert '开始推理' in template
    assert '/api/workflow/${stage}/${action}' in source
    assert 'workflow.inference?.running' in source
    assert 'workflow.runtime?.active' in source
    assert '推理、ActionExecutor、DeviceManager 和设备' in source
    assert 'stage.can_stop ? "stop" : stage.can_start ? "start" : ""' in source
    assert 'runtime.blockers || []' in source
    assert 'inference.blockers || []' in source
    assert '.workflow-button.is-running' in css
    assert 'white-space: pre-line' in css


def test_dark_theme_and_next_stage_guidance_are_explicit() -> None:
    source = _frontend_source()
    template = _template_source()
    css = _css_source()

    assert '<meta name="color-scheme" content="dark">' in template
    assert "color-scheme: dark" in css
    assert "--bg: #000000" in css
    assert '[data-guidance="next"]:not(:disabled)' in css
    guidance_rule = re.search(
        r'\[data-guidance="next"\]:not\(:disabled\)\s*\{(?P<body>.*?)\}',
        css,
        re.S,
    )
    assert guidance_rule is not None
    assert "outline: 2px solid rgb(255 255 255 / 58%)" in guidance_rule.group("body")
    assert "outline-offset: 2px" in guidance_rule.group("body")
    assert (
        "animation: next-stage-guidance-pulse 1.15s ease-in-out infinite"
        in guidance_rule.group("body")
    )
    assert "will-change: outline-color, box-shadow" in guidance_rule.group("body")
    assert '[data-guidance="next"]:not(:disabled)::before' not in css
    assert "@keyframes next-stage-guidance-pulse" in css
    assert "@keyframes next-stage-guidance-breathe" not in css
    assert "@keyframes next-stage-guidance-flash" not in css
    assert "@keyframes busy-indicator-pulse" in css
    busy_toggle_rule = re.search(
        r"\.panel-device-toggle\.is-busy\s*\{(?P<body>.*?)\}", css, re.S
    )
    assert busy_toggle_rule is not None
    assert "cursor: wait" not in busy_toggle_rule.group("body")
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert (
        'animation: next-stage-guidance-pulse 1.4s ease-in-out infinite !important'
        in css
    )
    reduced_motion = css.index("@media (prefers-reduced-motion: reduce)")
    global_reduction = css.index("animation-duration: 0.01ms !important", reduced_motion)
    guidance_override = css.index(
        "animation: next-stage-guidance-pulse 1.4s ease-in-out infinite !important",
        global_reduction,
    )
    busy_override = css.index(
        "animation: busy-indicator-pulse 0.9s ease-in-out infinite !important",
        global_reduction,
    )
    assert global_reduction < guidance_override < busy_override
    assert "function renderStageGuidance()" in source
    assert "entry.device?.ready !== true" in source
    assert 'deviceToggleAction(entry.device) === "start"' in source
    assert "runtime.can_start === true" in source
    assert "inference.can_start === true" in source
    assert 'controlsFailClosed("start")' in source


def test_device_action_paints_pending_state_before_async_request() -> None:
    source = _frontend_source()
    action = _function_body(source, "runDeviceAction", "renderLogDeviceOptions")

    assert "function yieldForPendingPaint()" in source
    assert action.index("renderBindingControls();") < action.index(
        "await yieldForPendingPaint();"
    )
    assert action.index("await yieldForPendingPaint();") < action.index(
        "scheduleStatePoll(0);"
    )
    assert action.index("scheduleStatePoll(0);") < action.index(
        "const operation = () => api("
    )


def test_start_transition_exposes_exactly_one_emergency_stop() -> None:
    source = _frontend_source()
    action = _function_body(source, "deviceToggleAction", "controlsMatchActiveConfig")
    render = _function_body(source, "updateDeviceToggle", "robotSlotRequirement")

    stop_precedence = 'if (transition?.desired === "stopped") return "";'
    starting_action = 'device.status === "starting"'
    assert stop_precedence in action
    assert action.index(stop_precedence) < action.index(starting_action)
    assert '`${progressLabel} · 紧急停止`' in render
    assert '点击立即发出停止请求' in render


def test_next_stage_guidance_follows_required_devices_runtime_and_inference() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; static frontend contracts still run")

    source = _frontend_source()
    marker = "  initialize();\n})();"
    assert marker in source
    injected = r'''  globalThis.__stageGuidanceContract = {
    run() {
      function button(action = "start", disabled = false) {
        return {
          disabled,
          dataset: { action },
          classList: { toggle() {}, add() {}, remove() {} },
          setAttribute() {},
          removeAttribute(name) {
            if (name === "data-guidance") delete this.dataset.guidance;
          },
        };
      }
      function guided(buttons) {
        return Object.fromEntries(
          Object.entries(buttons).map(([name, item]) => [
            name, item.dataset.guidance === "next",
          ])
        );
      }

      const buttons = {
        rizon: button(),
        i2rt: button(),
        head: button(),
        left: button(),
        right: button(),
        runtime: button(),
        inference: button(),
      };
      elements.robotDeviceToggle = buttons.rizon;
      elements.i2rtHeadDeviceToggle = buttons.i2rt;
      elements.runtimeToggle = buttons.runtime;
      elements.inferenceToggle = buttons.inference;
      elements.configSelect.value = "fixture";
      app.selectedConfig = "fixture";
      app.configSwitching = false;
      app.stateTransportFailed = false;
      app.stateTransportDeadlineAtMs = Date.now() + 10000;
      app.robotSlots = {
        rizon: {
          configured: true, inferenceRequired: true, deviceKey: "RizonRobot:2",
        },
        i2rt_head: {
          configured: true, inferenceRequired: false, deviceKey: "YamHeadRobot:0",
        },
      };
      app.cameraSlots = {
        head: {
          configured: true, inferenceRequired: false, deviceKey: "Camera:head",
        },
        left: {
          configured: true, inferenceRequired: true, deviceKey: "Camera:left",
        },
        right: {
          configured: true, inferenceRequired: true, deviceKey: "Camera:right",
        },
      };
      const cards = {
        head: { querySelector() { return buttons.head; } },
        left: { querySelector() { return buttons.left; } },
        right: { querySelector() { return buttons.right; } },
      };
      app.cameraCards = new Map(Object.entries(cards));
      app.devices = [
        { key: "RizonRobot:2", status: "stopped", ready: false },
        { key: "YamHeadRobot:0", status: "stopped", ready: false },
        { key: "Camera:head", status: "stopped", ready: false },
        { key: "Camera:left", status: "stopped", ready: false },
        { key: "Camera:right", status: "stopped", ready: false },
      ];
      app.state = {
        effective_config: { name: "fixture" },
        state_sync: { stale: false, authoritative: true, transitions: {} },
        safety: { fail_closed: false },
        workflow: {
          runtime: { ready: false, active: false, can_start: true },
          inference: { running: false, can_start: false },
        },
      };

      renderStageGuidance();
      const devices = guided(buttons);

      for (const device of app.devices) {
        if (["RizonRobot:2", "Camera:left", "Camera:right"].includes(device.key)) {
          device.status = "running";
          device.ready = true;
        }
      }
      buttons.rizon.dataset.action = "stop";
      buttons.left.dataset.action = "stop";
      buttons.right.dataset.action = "stop";
      renderStageGuidance();
      const runtime = guided(buttons);

      app.state.workflow.runtime = {
        ready: true, active: true, can_start: false, can_stop: true,
      };
      app.state.workflow.inference = { running: false, can_start: true };
      buttons.runtime.dataset.action = "stop";
      renderStageGuidance();
      const inference = guided(buttons);

      app.state.workflow.inference = { running: true, can_start: false };
      buttons.inference.dataset.action = "stop";
      renderStageGuidance();
      const complete = guided(buttons);

      const left = app.devices.find((device) => device.key === "Camera:left");
      left.status = "stopped";
      left.ready = false;
      buttons.left.dataset.action = "start";
      buttons.left.disabled = true;
      renderStageGuidance();
      const disabled = guided(buttons);

      buttons.left.disabled = false;
      app.state.state_sync.stale = true;
      renderStageGuidance();
      const stale = guided(buttons);

      return { devices, runtime, inference, complete, disabled, stale };
    },
  };
})();'''
    test_source = source.replace(marker, injected, 1)
    harness = f'''
const elementStore = new Map();
function makeElement() {{
  return {{
    value: "", disabled: false, hidden: false, textContent: "", title: "",
    dataset: {{}},
    classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
    setAttribute() {{}}, removeAttribute(name) {{
      if (name === "data-guidance") delete this.dataset.guidance;
    }},
    addEventListener() {{}}, querySelector() {{ return null; }},
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
process.stdout.write(JSON.stringify(globalThis.__stageGuidanceContract.run()));
'''
    completed = subprocess.run(
        [node], input=harness, text=True, encoding="utf-8",
        capture_output=True, check=True, timeout=10
    )
    result = json.loads(completed.stdout)

    assert result["devices"] == {
        "rizon": True,
        "i2rt": False,
        "head": False,
        "left": True,
        "right": True,
        "runtime": False,
        "inference": False,
    }
    assert result["runtime"] == {
        "rizon": False,
        "i2rt": False,
        "head": False,
        "left": False,
        "right": False,
        "runtime": True,
        "inference": False,
    }
    assert result["inference"] == {
        "rizon": False,
        "i2rt": False,
        "head": False,
        "left": False,
        "right": False,
        "runtime": False,
        "inference": True,
    }
    assert not any(result["complete"].values())
    assert not any(result["disabled"].values())
    assert not any(result["stale"].values())


def test_recording_controls_and_dense_timeline_are_fail_closed() -> None:
    source = _frontend_source()
    template = _template_source()
    css = _css_source()

    for action, label in (
        ("episode_accept", "接受并复位"),
        ("episode_discard", "拒绝并复位"),
        ("segment_accept", "接受片段"),
        ("segment_reject", "拒绝片段"),
    ):
        assert f'data-recording-action="{action}"' in template
        assert label in template
        assert f'"{action}"' in source

    assert 'id="recording-timeline-content"' in template
    assert 'id="recording-track-segments"' in template
    assert 'id="recording-track-checkpoints"' in template
    assert 'id="recording-camera-roles"' in template
    for role in ("head", "left", "right"):
        assert f'data-camera-role="{role}"' in template
    assert "data-camera-role-device" in template
    assert "data-camera-role-topic" in template
    assert "data-camera-role-shm" in template
    assert '拒绝（保留）' in template
    assert 'data-lucide' not in template
    assert '/api/recording/actions/${action}' in source
    assert "episode_id: episode.id" in source
    assert "expected_checkpoint_revision: episode.checkpoint_revision" in source
    assert "idempotency_key: recordingIdempotencyKey()" in source
    assert "capability.can" in source
    assert "recording.pending_command" in source
    assert "normalizeRecordingCameraRoles(source.camera_roles)" in source
    assert "RECORDING_CAMERA_STATUS_LABELS" in source
    assert 'camera.shm_name || "—"' in source
    assert '"未配置映射"' in source
    assert "const warning = recordingCameraWarning(camera.reason);" in source
    assert "RECORDING_CAMERA_STATUS_LABELS[camera.status]" in source
    assert "error.status === 409" in source
    recording_action_body = _function_body(
        source, "runRecordingAction", "runWorkflowAction"
    )
    assert "confirmOperation(" not in recording_action_body
    for action, code, key in (
        ("episode_accept", "KeyQ", "Q"),
        ("episode_discard", "KeyW", "W"),
        ("segment_accept", "KeyA", "A"),
        ("segment_reject", "KeyS", "S"),
    ):
        assert f'{action}: Object.freeze({{ code: "{code}", key: "{key}" }})' in source
    shortcut_body = _function_body(
        source, "handleRecordingShortcut", "runRecordingAction"
    )
    for guard in (
        "event.defaultPrevented",
        "event.repeat",
        "event.isComposing",
        "event.ctrlKey",
        "event.altKey",
        "event.metaKey",
        "event.shiftKey",
        "recordingShortcutTargetIsEditable(event.target)",
        "recordingShortcutDialogIsOpen()",
        "button.disabled",
    ):
        assert guard in shortcut_body
    assert 'document.addEventListener("keydown", handleRecordingShortcut)' in source
    assert 'button.setAttribute("aria-keyshortcuts", shortcut.key)' in source
    assert 'setText(button.querySelector("[data-recording-shortcut-label]")' in source
    assert "快捷键 ${shortcut.key}" in source
    assert template.count("data-recording-shortcut-label") == 4
    assert template.count('class="recording-action-key"') == 4
    assert ".recording-action-key" in css
    assert 'id="recording-action-grid"' in template
    assert 'id="standalone-reset-control"' in template
    assert 'id="standalone-pause-button"' in template
    assert 'id="standalone-reset-button"' in template
    assert re.search(
        r'<button[^>]+id="standalone-reset-button"[^>]*>\s*复位\s*</button>',
        template,
    )
    assert ".standalone-reset-button" in css
    assert ".standalone-pause-button" in css
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in css
    assert "background: #c65f08" in css
    assert 'api("/api/inference/reset"' in source
    assert 'api(`/api/inference/${action}`' in source
    assert 'action === "pause" ? "暂停已提交" : "继续已提交"' in source
    assert 'elements.standalonePauseButton.textContent = reset.paused' in source
    assert '"继续推理"' in source
    assert '"暂停推理"' in source
    assert "const showRecordingActions = recording.enabled === true" in source
    assert "elements.recordingActionGrid.hidden = !showRecordingActions" in source
    assert "elements.standaloneResetControl.hidden = showRecordingActions" in source
    assert "expected_selected_config: selectedConfig" in source
    assert "expected_revision: expectedRevision" in source
    assert template.count("aria-valuenow") == 1
    assert 'id="workspace-splitter"' in template
    assert ".recording-track-segment--accept" in css
    assert ".recording-track-segment--reject" in css
    assert ".recording-track-checkpoint" in css
    assert '.recording-camera-role[data-status="missing"]' in css
    assert ".workflow-panel { order: 4; }" in css


def test_rollout_storage_budget_is_visible_and_escalates() -> None:
    source = _frontend_source()
    template = _template_source()
    css = _css_source()

    for element_id in (
        'rollout-storage-budget',
        'rollout-storage-free',
        'rollout-storage-remaining',
        'rollout-storage-detail',
    ):
        assert f'id="{element_id}"' in template
    assert 'normalizeRolloutStorage(source?.rollout_storage)' in source
    assert 'renderRolloutStorageBudget(options)' in source
    assert 'Rollout 存储空间严重不足' in source
    assert '.rollout-storage-budget[data-status="warning"]' in css
    assert '.rollout-storage-budget[data-status="critical"]' in css
    assert '@keyframes rollout-storage-alert-pulse' in css
    assert '.toast--warning' in css


def test_server_path_browser_separates_current_path_from_root_shortcuts() -> None:
    source = _frontend_source()
    template = _template_source()
    css = _css_source()

    assert 'id="checkpoint-browser-path"' in template
    assert '>路径</span>' in template
    assert 'id="checkpoint-browser-roots"' in template
    assert 'aria-label="快速切换存储位置"' in template
    assert 'id="checkpoint-browser-root"' not in template
    assert 'checkpointBrowserRoots: document.getElementById(' in source
    assert 'button.className = "checkpoint-browser-root-button"' in source
    assert '"用户目录 /home"' in source
    assert '"挂载盘 /mnt"' in source
    assert '.checkpoint-browser-current-location' in css
    assert '.checkpoint-browser-root-button.is-active' in css


def test_rollout_path_selection_is_independent_from_recording_enablement() -> None:
    source = _frontend_source()

    browser = _function_body(source, "openServerPathBrowser", "openCheckpointBrowser")
    assert "enableRolloutOnSelect = false" in browser
    assert "app.rolloutBrowserEnableOnSelect = isRollout && enableRolloutOnSelect" in browser
    assert "选择为保存及回放目录" in browser
    assert 'openRolloutBrowser({ enableOnSelect: true })' in source
    assert 'openRolloutBrowser({ enableOnSelect: false })' in source
    assert "options.rolloutEnabled || app.rolloutBrowserEnableOnSelect" in source
    assert "Rollout 保存及回放目录已更新" in source


def test_active_devices_allow_source_save_but_not_generated_config() -> None:
    source = _frontend_source()
    availability = _function_body(
        source,
        "renderGeneratedConfigAvailability",
        "runtimeTuningOverrides",
    )

    assert "const canSave = Boolean(" in availability
    assert "!app.pendingActions.size" in availability
    assert "!app.devices.some((device) => ACTIVE_STATUSES.has(device.status))" in availability
    assert "elements.generatedConfigExport.disabled = !canExport" in availability
    assert "elements.currentConfigSave.disabled = !canSave" in availability


def test_server_path_browser_accepts_pasted_paths_without_stale_selection() -> None:
    source = _frontend_source()
    template = _template_source()

    assert 'id="checkpoint-browser-path" type="text"' in template
    assert 'placeholder="粘贴服务器绝对路径"' in template
    assert 'id="checkpoint-browser-go"' in template
    assert 'validateCheckpointBrowserInput' in source
    assert 'checkpointBrowserFriendlyError' in source
    assert 'checkpointBrowserInputPath() !== String(payload.path)' in source
    assert 'elements.checkpointBrowserSelect.disabled = !(' in source
    assert 'elements.checkpointBrowserPath?.addEventListener("input"' in source
    assert 'elements.checkpointBrowserPath?.addEventListener("keydown"' in source
    assert 'event.key !== "Enter" || event.isComposing' in source
    assert '路径尚未加载，按 Enter 或点击“打开”进行验证。' in source
    assert '该路径不在允许访问的位置内' in source


def test_gripper_check_controls_are_motion_safe_and_fail_closed() -> None:
    source = _frontend_source()
    template = _template_source()
    css = _css_source()

    for element_id in (
        "gripper-calibrate-all",
        "gripper-left-position",
        "gripper-right-position",
        "gripper-left-value",
        "gripper-right-value",
        "gripper-check-reason",
    ):
        assert f'id="{element_id}"' in template
    assert 'type="range" min="0" max="100" step="1"' in template
    assert "0% 闭合" in template
    assert "100% 张开" in template
    assert "夹爪将依次执行两轮全开、全合并触碰机械限位" in source
    assert "最久约 2.5 分钟" in source
    assert 'api("/api/grippers/calibrate"' in source
    assert 'api("/api/grippers/position"' in source
    assert "target_percent: target" in source
    assert "idempotency_key: gripperIdempotencyKey()" in source
    assert "gripperCapabilityValue" in source
    assert "source.can_position" in source
    assert "app.gripperQueuedTargets.set(side, target)" in source
    assert "GRIPPER_TARGET_COMMAND_INTERVAL_MS = 200" in source
    assert "gripperDesiredTargets" in source
    assert "scheduleGripperTargetSubmission(side, 0)" in source
    assert 'slider.addEventListener("input"' in source
    assert 'slider.addEventListener("change"' in source
    input_handler = source[source.index('slider.addEventListener("input"'):source.index(
        'slider.addEventListener("change"'
    )]
    assert "latchGripperTarget" in input_handler
    assert "scheduleGripperTargetSubmission" in input_handler
    assert ".gripper-control-row" in css
    assert ".gripper-slider-control" in css
    assert '@media (max-width: 620px)' in css


def test_gripper_permission_button_respects_hidden_attribute() -> None:
    template = _template_source()
    css = _css_source()

    button = re.search(
        r'<button\b[^>]*\bid="gripper-authorize-permissions"[^>]*>',
        template,
    )
    assert button is not None
    assert re.search(r"\shidden(?:\s|=|>)", button.group(0))

    hidden_rule = re.search(
        r"\.gripper-permission-button\[hidden\]\s*\{(?P<body>[^}]*)\}",
        css,
        re.S,
    )
    assert hidden_rule is not None
    assert re.search(
        r"display\s*:\s*none\s*!important\s*;", hidden_rule.group("body")
    )


def test_gripper_capability_pending_coalesce_confirm_and_payload_in_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; static frontend contracts still run")

    source = _frontend_source()
    marker = "  initialize();\n})();"
    assert marker in source
    injected = r'''  globalThis.__gripperContract = {
    async run() {
      const calls = [];
      const confirmations = [];
      const fixture = {
        revision: 17,
        status: "idle",
        position_labels: { zero: "closed", hundred: "open" },
        capabilities: {
          calibrate: { can: true, blockers: [] },
          move: { can: true, blockers: [] },
        },
        sides: {
          left: { status: "idle", actual_percent: 20, revision: 3 },
          right: { status: "idle", actual_percent: 80, revision: 4 },
        },
      };
      app.state = { revision: 17, grippers: fixture };
      app.stateTransportFailed = false;
      app.stateTransportDeadlineAtMs = Date.now() + 10000;
      app.configSwitching = false;
      app.gripperCalibratePending = false;
      app.gripperMovePending = new Set();
      app.gripperQueuedTargets = new Map();
      app.gripperDrafts = new Map();
      app.gripperDesiredTargets = new Map();
      app.gripperTargetSessions = new Map();
      app.gripperTargetTimers = new Map();
      app.gripperEditing = new Set();
      app.gripperErrors = new Map();
      api = async (path, request = {}) => {
        calls.push({ path, method: request.method, payload: JSON.parse(request.body) });
        return { grippers: fixture };
      };
      showToast = () => {};
      scheduleStatePoll = () => {};
      refreshState = async () => {};
      confirmOperation = async (title, message, confirmText) => {
        confirmations.push({ title, message, confirmText });
        return confirmations.length > 1;
      };

      const normalized = normalizeGrippers();
      renderGrippers();
      const enabled = {
        calibrate: !elements.gripperCalibrateAll.disabled,
        left: !elements.grippers.left.position.disabled,
        right: !elements.grippers.right.position.disabled,
        minLabel: elements.grippers.left.minLabel.textContent,
        maxLabel: elements.grippers.left.maxLabel.textContent,
      };

      app.gripperCalibratePending = true;
      renderGrippers();
      const pendingDisabled = elements.gripperCalibrateAll.disabled;
      app.gripperCalibratePending = false;
      await runGripperCalibration();
      await runGripperCalibration();
      await runGripperMove("left", 41);

      app.gripperMovePending.add("right");
      await runGripperMove("right", 22);
      await runGripperMove("right", 77);
      const coalesced = app.gripperQueuedTargets.get("right");
      app.gripperMovePending.delete("right");
      app.gripperQueuedTargets.clear();

      app.state.grippers = {
        revision: 17,
        busy: true,
        global_busy: false,
        busy_sides: ["left"],
        capabilities: {
          calibrate: { can: false, blockers: ["夹爪动作中"] },
          move: { can: true, blockers: [] },
        },
        sides: {
          left: {
            status: "moving", busy: true, revision: 4,
            can_position: false, blockers: ["left gripper command is active"],
            can_queue_position: true, queue_position_blockers: [],
          },
          right: {
            status: "idle", busy: false, revision: 5,
            can_position: true, blockers: [],
            can_queue_position: true, queue_position_blockers: [],
          },
        },
      };
      renderGrippers();
      const independent = {
        leftDisabled: elements.grippers.left.position.disabled,
        rightEnabled: !elements.grippers.right.position.disabled,
      };
      await runGripperMove("right", 66);

      app.state.grippers = {
        revision: 18,
        capabilities: {
          calibrate: { can: false, blockers: ["推理运行中"] },
          move: { can: false, blockers: ["推理运行中"] },
        },
        sides: { left: { status: "idle" }, right: { status: "idle" } },
      };
      renderGrippers();
      const blocked = {
        calibrate: elements.gripperCalibrateAll.disabled,
        left: elements.grippers.left.position.disabled,
        reason: elements.gripperCheckReason.textContent,
      };
      return {
        normalized: {
          revision: normalized.revision,
          leftActual: normalized.sides.left.actualPercent,
          canCalibrate: normalized.capabilities.calibrate.can,
        },
        enabled,
        pendingDisabled,
        confirmations,
        calls,
        coalesced,
        independent,
        blocked,
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
  getElementById(id) {{
    if (!elementStore.has(id)) elementStore.set(id, makeElement());
    return elementStore.get(id);
  }},
  querySelector() {{ return makeElement(); }},
  querySelectorAll() {{ return []; }},
}};
globalThis.window = {{ setTimeout() {{}}, clearTimeout() {{}}, confirm() {{ return false; }} }};
eval({json.dumps(test_source)});
globalThis.__gripperContract.run().then((result) => process.stdout.write(JSON.stringify(result))).catch((error) => {{
  process.stderr.write(error.stack || String(error));
  process.exitCode = 1;
}});
'''
    completed = subprocess.run(
        [node], input=harness, text=True, encoding="utf-8",
        capture_output=True, check=True, timeout=10
    )
    result = json.loads(completed.stdout)
    assert result["normalized"] == {
        "revision": 17,
        "leftActual": 20,
        "canCalibrate": True,
    }
    assert result["enabled"] == {
        "calibrate": True,
        "left": True,
        "right": True,
        "minLabel": "0% 闭合",
        "maxLabel": "100% 张开",
    }
    assert result["pendingDisabled"] is True
    assert len(result["confirmations"]) == 2
    assert "物体和手指" in result["confirmations"][0]["message"]
    assert len(result["calls"]) == 3
    assert result["calls"][0]["path"] == "/api/grippers/calibrate"
    assert result["calls"][0]["method"] == "POST"
    assert result["calls"][0]["payload"]["side"] == "both"
    assert result["calls"][0]["payload"]["revision"] == 17
    assert set(result["calls"][0]["payload"]) == {"side", "revision", "idempotency_key"}
    assert result["calls"][1]["path"] == "/api/grippers/position"
    assert result["calls"][1]["payload"]["side"] == "left"
    assert result["calls"][1]["payload"]["target_percent"] == 41
    assert result["calls"][1]["payload"]["side_revision"] == 3
    assert set(result["calls"][1]["payload"]) == {
        "side", "target_percent", "revision", "side_revision", "idempotency_key"
    }
    assert result["independent"] == {"leftDisabled": False, "rightEnabled": True}
    assert result["calls"][2]["path"] == "/api/grippers/position"
    assert result["calls"][2]["payload"]["side"] == "right"
    assert result["calls"][2]["payload"]["side_revision"] == 5
    assert result["coalesced"] == 77
    assert result["blocked"]["calibrate"] is True
    assert result["blocked"]["left"] is True
    assert "推理运行中" in result["blocked"]["reason"]


def test_discovery_bindings_are_opaque_recommended_and_session_only() -> None:
    source = _frontend_source()
    defaults = _function_body(source, "resetBindingsForSelectedConfig", "loadHardwareDiscovery")

    assert 'HARDWARE_DISCOVERY_URL = "/api/hardware/discovery"' in source
    assert '?config_name=${encodeURIComponent(configName)}' in source
    assert "source.recommended?.overrides" in source
    assert "availableRobotBinding" in source
    assert "availableCameraBinding" in source
    assert "discovery_revision: discoveryRevision" in source
    assert "overrides: buildBindingOverrides()" in source
    assert "robotOverride.robot_left" in source
    assert "robotOverride.robot_right" in source
    assert "capture_device:" in source
    assert "localStorage" not in defaults
    assert defaults.index('effectiveBindingId(device.key, "capture_device")') < defaults.index(
        "cameraBindingFromLogicalId(device, used)"
    )
    assert defaults.index("cameraBindingFromLogicalId(device, used)") < defaults.index(
        "recommendation.capture_device"
    )


def test_discovery_schema_and_sparse_override_contracts() -> None:
    source = _frontend_source()
    normalize = _function_body(source, "normalizeHardwareDiscovery", "isRizonDevice")
    capture_capability = _function_body(
        source, "hasCaptureBindingCapability", "normalizeDevice"
    )
    overrides = _function_body(source, "buildBindingOverrides", "activeConfigSwitchDevices")
    camera_card = _function_body(source, "createCameraCard", "updateCameraBindingControls")

    assert "item?.rdk_sn, item?.serial_number, item?.serial, item?.name" in normalize
    assert "const explicitVariant" in normalize
    assert "/4r/i.test(rdkSn)" in normalize
    assert "/^Rizon4R$/i.test" in normalize
    assert "item?.devnode" in normalize

    assert 'capabilities.capture_binding === false' in capture_capability
    assert 'capabilities.capture_binding === true' in capture_capability
    assert '"webcameradevice", "webcameraopenpi"' in source
    assert "opencvcameradevice" not in source.lower()
    assert "device?.supportsCaptureBinding" in camera_card

    assert "if (left) robotOverride.robot_left = left;" in overrides
    assert "if (right) robotOverride.robot_right = right;" in overrides
    assert "capture_device: captureDevice || null" in overrides
    assert "Object.keys(robotOverride).length" in overrides
    assert "item.isCamera && item.supportsCaptureBinding" in source
    assert "const bindingsReady = Boolean(configuredRobot);" in source


def test_actual_backend_discovery_fixture_parks_camera_without_hardware() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; static frontend contracts still run")

    source = _frontend_source()
    marker = "  initialize();\n})();"
    assert marker in source
    injected = """  globalThis.__controlPanelContract = {
    run(discoveryPayload, rawDevices) {
      app.discovery = normalizeHardwareDiscovery(discoveryPayload);
      app.configDevices = rawDevices.map(normalizeDevice);
      app.devices = [];
      app.robotBindings = { left: "", right: "" };
      app.cameraBindings = new Map();
      elements.configSelect.value = "fixture";
      renderBindingControls = () => {};

      resetBindingsForSelectedConfig({ preserve: false });
      const complete = {
        bindings: { ...app.robotBindings },
        overrides: buildBindingOverrides(),
        bindable: Object.fromEntries(
          app.configDevices.map((device) => [device.key, device.supportsCaptureBinding])
        ),
      };

      app.discovery = {
        ...app.discovery,
        robots: app.discovery.robots.filter((robot) => robot.variant !== "4r"),
      };
      app.robotBindings = { left: "", right: "" };
      resetBindingsForSelectedConfig({ preserve: false });
      const sparse = {
        bindings: { ...app.robotBindings },
        overrides: buildBindingOverrides(),
      };
      return { complete, sparse };
    },
  };
})();"""
    test_source = source.replace(marker, injected, 1)

    discovery_fixture = {
        "revision": "fixture-hardware-revision",
        "generated_at": "2026-07-12T00:00:00+00:00",
        "interfaces": [],
        "cameras": [
            {
                "id": "camera_opaque_0",
                "devnode": "/dev/video0",
                "stable_path": "/dev/v4l/by-id/camera-a-video-index0",
                "name": "Camera A",
                "capture_capable": True,
                "source": "sysfs_v4l2",
            },
            {
                "id": "camera_opaque_2",
                "devnode": "/dev/video2",
                "stable_path": "/dev/v4l/by-id/camera-b-video-index0",
                "name": "Camera B",
                "capture_capable": True,
                "source": "sysfs_v4l2",
            }
        ],
        "robots": [
            {
                "id": "robot_opaque_left",
                "model": "Rizon4",
                "serial_number": "Rizon4-062717",
                "ip": "192.168.2.101",
                "source": "elements_cache",
                "confidence": "cached",
                "peer_active": True,
            },
            {
                "id": "robot_opaque_right",
                "model": "Rizon4R",
                "serial_number": "Rizon4R-062069",
                "ip": "192.168.2.102",
                "source": "elements_cache",
                "confidence": "cached",
                "peer_active": True,
            },
        ],
        "providers": [],
        "warnings": [],
        "recommended_overrides": {
            "RizonRobot:2": {
                "robot_left": "robot_opaque_left",
                "robot_right": "robot_opaque_right",
            },
            "WebCameraOpenPi:0": {"capture_device": "camera_opaque_0"},
            "WebCameraOpenPi:2": {"capture_device": "camera_opaque_2"},
        },
    }
    device_fixture = [
        {
            "key": "RizonRobot:2",
            "class": "RizonRobot",
            "device_class": "RizonRobot",
            "type": "RizonRobot",
            "id": 2,
            "device_class_id": 2,
            "enabled": True,
            "capabilities": {"preview": False, "motion": True},
        },
        {
            "key": "WebCameraOpenPi:0",
            "class": "WebCameraOpenPi",
            "device_class": "WebCameraOpenPi",
            "type": "WebCameraOpenPi",
            "id": 0,
            "device_class_id": 0,
            "enabled": True,
            "capabilities": {"preview": True, "capture_binding": True},
        },
        {
            "key": "WebCameraOpenPi:4",
            "class": "WebCameraOpenPi",
            "device_class": "WebCameraOpenPi",
            "type": "WebCameraOpenPi",
            "id": 4,
            "device_class_id": 4,
            "enabled": True,
            "capabilities": {"preview": True, "capture_binding": True},
        },
        {
            "key": "OpenCVCameraDevice:9",
            "class": "OpenCVCameraDevice",
            "device_class": "OpenCVCameraDevice",
            "type": "OpenCVCameraDevice",
            "id": 9,
            "device_class_id": 9,
            "enabled": True,
            "capabilities": {"preview": True, "capture_binding": False},
        },
        {
            "key": "WebCameraOpenPi:2",
            "class": "WebCameraOpenPi",
            "device_class": "WebCameraOpenPi",
            "type": "WebCameraOpenPi",
            "id": 2,
            "device_class_id": 2,
            "enabled": True,
            "capabilities": {"preview": True, "capture_binding": True},
        },
    ]
    harness = f"""
globalThis.document = {{
  hidden: false,
  activeElement: null,
  getElementById() {{ return {{ value: "" }}; }},
  querySelector() {{ return null; }},
}};
globalThis.window = {{}};
eval({json.dumps(test_source)});
const result = globalThis.__controlPanelContract.run(
  {json.dumps(discovery_fixture)},
  {json.dumps(device_fixture)}
);
process.stdout.write(JSON.stringify(result));
"""
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

    assert result["complete"]["bindings"] == {
        "left": "robot_opaque_left",
        "right": "robot_opaque_right",
    }
    assert result["complete"]["overrides"]["RizonRobot:2"] == {
        "robot_left": "robot_opaque_left",
        "robot_right": "robot_opaque_right",
    }
    assert result["complete"]["overrides"]["WebCameraOpenPi:0"] == {
        "capture_device": "camera_opaque_0"
    }
    assert result["complete"]["overrides"]["WebCameraOpenPi:4"] == {
        "capture_device": None
    }
    assert result["complete"]["overrides"]["WebCameraOpenPi:2"] == {
        "capture_device": "camera_opaque_2"
    }
    assert "OpenCVCameraDevice:9" not in result["complete"]["overrides"]
    assert result["complete"]["bindable"] == {
        "RizonRobot:2": False,
        "WebCameraOpenPi:0": True,
        "WebCameraOpenPi:4": True,
        "OpenCVCameraDevice:9": False,
        "WebCameraOpenPi:2": True,
    }

    sparse_robot = result["sparse"]["overrides"]["RizonRobot:2"]
    assert result["sparse"]["bindings"]["right"] == ""
    assert sparse_robot == {"robot_left": "robot_opaque_left"}
    assert result["sparse"]["overrides"]["WebCameraOpenPi:4"] == {
        "capture_device": None
    }


def test_config_focus_refreshes_and_change_requires_safe_stop_confirmation() -> None:
    source = _frontend_source()
    switch = _function_body(source, "selectConfigWithConfirmation", "stateFromPayload")

    assert 'addEventListener("pointerdown", refreshConfigSourcesForInteraction)' in source
    assert 'addEventListener("focus", refreshConfigSourcesForInteraction)' in source
    assert 'loadConfigs({ render: false })' in source
    assert "document.activeElement !== elements.configSelect" in source
    assert '"切换运行配置"' in switch
    assert "activeConfigSwitchDevices()" in switch
    assert "statusCounts" in switch
    parallel_start = switch.index("const [detailResult] = await Promise.allSettled([")
    detail_start = switch.index(
        "loadConfigDetail(name, { throwOnError: true })", parallel_start
    )
    discovery_start = switch.index("loadHardwareDiscovery({", parallel_start)
    confirm_start = switch.index("confirmOperation(")
    assert parallel_start < detail_start < confirm_start
    assert parallel_start < discovery_start < confirm_start
    assert "throwOnError: false" in switch
    assert 'if (detailResult.status === "rejected")' in switch
    assert "Hardware inventory is advisory while selecting a config" in switch
    assert "stop_active: true" in switch
    cancel_start = switch.index("if (!confirmed)")
    post_start = switch.index('await api("/api/config/select"')
    assert cancel_start < post_start
    assert 'await restoreConfigChoice(previousValue);' in switch[cancel_start:post_start]
    assert 'api("/api/config/select"' not in switch[cancel_start:post_start]


def test_camera_selection_atomically_swaps_two_slots_without_touching_left() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; static frontend contracts still run")

    source = _frontend_source()
    marker = "  initialize();\n})();"
    injected = """  globalThis.__cameraSwapContract = {
    run() {
      app.discovery = normalizeHardwareDiscovery({
        revision: "camera-swap",
        cameras: [
          { id: "left-camera", devnode: "/dev/video0", capture_capable: true },
          { id: "right-camera", devnode: "/dev/video2", capture_capable: true },
          { id: "head-camera", devnode: "/dev/video4", capture_capable: true },
        ],
      });
      app.configDevices = [0, 2, 4].map((id) => normalizeDevice({
        key: `WebCameraOpenPi:${id}`,
        class: "WebCameraOpenPi",
        device_class: "WebCameraOpenPi",
        type: "WebCameraOpenPi",
        id,
        device_class_id: id,
        enabled: true,
        capabilities: { preview: true, capture_binding: true },
      }));
      app.devices = app.configDevices.map((device) => ({
        ...device, status: "stopped", running: false,
      }));
      app.cameraBindings = new Map([
        ["WebCameraOpenPi:0", "left-camera"],
        ["WebCameraOpenPi:2", "right-camera"],
        ["WebCameraOpenPi:4", "head-camera"],
      ]);
      const affected = applyCameraBindingSelection(
        "WebCameraOpenPi:4", "right-camera"
      );
      const swapped = {
        affected,
        bindings: Object.fromEntries(app.cameraBindings),
        overrides: buildBindingUpdateOverrides(affected),
      };
      const parkedAffected = applyCameraBindingSelection(
        "WebCameraOpenPi:4", ""
      );
      return {
        swapped,
        parked: {
          affected: parkedAffected,
          bindings: Object.fromEntries(app.cameraBindings),
          overrides: buildBindingUpdateOverrides(parkedAffected),
        },
      };
    },
  };
})();"""
    test_source = source.replace(marker, injected, 1)
    harness = f"""
globalThis.document = {{
  hidden: false,
  activeElement: null,
  getElementById() {{ return {{ value: "" }}; }},
  querySelector() {{ return null; }},
}};
globalThis.window = {{}};
eval({json.dumps(test_source)});
process.stdout.write(JSON.stringify(globalThis.__cameraSwapContract.run()));
"""
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

    assert result["swapped"]["affected"] == [
        "WebCameraOpenPi:4", "WebCameraOpenPi:2"
    ]
    assert result["swapped"]["bindings"] == {
        "WebCameraOpenPi:0": "left-camera",
        "WebCameraOpenPi:2": "head-camera",
        "WebCameraOpenPi:4": "right-camera",
    }
    assert result["swapped"]["overrides"] == {
        "WebCameraOpenPi:4": {"capture_device": "right-camera"},
        "WebCameraOpenPi:2": {"capture_device": "head-camera"},
    }
    assert result["parked"] == {
        "affected": ["WebCameraOpenPi:4"],
        "bindings": {
            "WebCameraOpenPi:0": "left-camera",
            "WebCameraOpenPi:2": "head-camera",
            "WebCameraOpenPi:4": "",
        },
        "overrides": {
            "WebCameraOpenPi:4": {"capture_device": None},
        },
    }


def test_initial_default_config_is_one_shot_cas_without_confirmation() -> None:
    source = _frontend_source()
    chooser = _function_body(source, "chooseBootstrapConfig", "normalizeStatus")
    bootstrap = _function_body(
        source, "maybeAutoApplyDefaultConfig", "shapeText"
    )

    assert "recommended && byName.has(recommended)" in chooser
    assert 'name.includes("dual") && name.includes("pi05")' in chooser
    assert chooser.index("recommended && byName.has(recommended)") < chooser.index(
        'name.includes("dual") && name.includes("pi05")'
    )
    assert "config.default === true" in chooser
    assert "eligible[0].name" in chooser
    assert "app.bootstrapAttempted = true" in bootstrap
    assert "initialStateAllowsBootstrap()" in bootstrap
    assert bootstrap.index("loadConfigDetail(candidate") < bootstrap.index(
        "loadHardwareDiscovery({"
    ) < bootstrap.index('api("/api/config/select"')
    assert "throwOnError: false" in bootstrap
    assert "overrides: buildBindingOverrides()" in bootstrap
    assert "stop_active: false" in bootstrap
    assert "auto_apply: true" in bootstrap
    assert "expected_revision: expectedRevision" in bootstrap
    assert "expected_selected_config: null" in bootstrap
    assert "confirmOperation(" not in bootstrap
    assert "app.bootstrapFailureNotice = message" in bootstrap
    assert 'elements.configSelect.value = ""' in bootstrap
    assert "elements.configSelect.selectedIndex = -1" in bootstrap
    assert "await refreshState(false)" in bootstrap
    assert source.count("await maybeAutoApplyDefaultConfig()") == 1
    assert "setInterval" not in source
    assert "scheduleStatePoll();" in source


def test_default_config_selection_and_bootstrap_flow_in_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; static frontend contracts still run")

    source = _frontend_source()
    marker = "  initialize();\n})();"
    assert marker in source
    injected = r'''  globalThis.__defaultConfigContract = {
    choose(payload) {
      return chooseBootstrapConfig(
        extractConfigs(payload),
        normalizeConfigName(payload?.recommended_config)
      );
    },
    async runScenario(options = {}) {
      const calls = { detail: [], discovery: [], posts: [], refresh: 0, toasts: [] };
      app.configs = extractConfigs({ items: options.items || [] });
      app.recommendedConfig = normalizeConfigName(options.recommended_config || "");
      app.state = {
        revision: options.revision ?? 7,
        selected_config: options.selected_config ?? null,
        workflow: options.workflow || {
          runtime: { active: false, status: "stopped" },
          inference: { running: false, status: "stopped" },
        },
      };
      app.devices = (options.deviceStatuses || []).map((status) => ({ status }));
      app.initialStateObserved = options.stateObserved !== false;
      app.bootstrapAttempted = false;
      app.bootstrapFailureNotice = "";
      app.bootstrapManualChoiceRequired = false;
      app.selectedConfig = "";
      app.activeConfig = "";
      app.configSelectionBeforeOpen = "";
      app.configSwitching = false;
      app.bindingDirty = false;
      app.bindingDirtyKeys = new Set();
      app.bindingErrors = new Map();
      app.discovery = {
        revision: "hardware-r1",
        robots: [],
        cameras: [],
        recommendedOverrides: {},
      };
      elements.configSelect.value = app.configs[0]?.name || "";
      elements.configSelect.selectedIndex = app.configs.length ? 0 : -1;
      elements.globalNotice.textContent = "";
      elements.globalNotice.hidden = true;

      renderBindingControls = () => {};
      showToast = (message, type) => calls.toasts.push({ message, type });
      buildBindingOverrides = () => options.overrides || {
        "RizonRobot:2": { robot_left: "left-id", robot_right: "right-id" },
      };
      loadConfigDetail = async (name, detailOptions = {}) => {
        calls.detail.push({ name, throwOnError: detailOptions.throwOnError === true });
      };
      loadHardwareDiscovery = async (discoveryOptions = {}) => {
        calls.discovery.push({ ...discoveryOptions });
        app.discovery.revision = "hardware-r1";
      };
      refreshState = async () => {
        calls.refresh += 1;
        if (options.refreshedSelectedConfig) {
          app.state.selected_config = options.refreshedSelectedConfig;
          app.selectedConfig = options.refreshedSelectedConfig;
          app.activeConfig = options.refreshedSelectedConfig;
        }
      };
      api = async (path, request = {}) => {
        if (path !== "/api/config/select") throw new Error(`unexpected API ${path}`);
        const payload = JSON.parse(request.body);
        calls.posts.push({ path, method: request.method, payload });
        if (options.apiErrorStatus) {
          throw new ApiError(`fixture HTTP ${options.apiErrorStatus}`, options.apiErrorStatus, {
            selected_config: options.conflictSelectedConfig || null,
            revision: payload.expected_revision + 1,
          });
        }
        return { selected_config: payload.name, revision: payload.expected_revision + 1 };
      };

      const first = await maybeAutoApplyDefaultConfig();
      const second = options.callTwice ? await maybeAutoApplyDefaultConfig() : null;
      return {
        firstStatus: first.status,
        secondStatus: second?.status || null,
        calls,
        bootstrapAttempted: app.bootstrapAttempted,
        failureNotice: app.bootstrapFailureNotice,
        manualChoiceRequired: app.bootstrapManualChoiceRequired,
        selectValue: elements.configSelect.value,
        selectedIndex: elements.configSelect.selectedIndex,
      };
    },
  };
})();'''
    test_source = source.replace(marker, injected, 1)
    harness = f'''
const elementStore = new Map();
function makeElement() {{
  return {{
    value: "",
    selectedIndex: -1,
    disabled: false,
    hidden: false,
    textContent: "",
    className: "",
    dataset: {{}},
    classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
    setAttribute() {{}},
    appendChild() {{}},
    replaceChildren() {{}},
    addEventListener() {{}},
    querySelector() {{ return null; }},
  }};
}}
globalThis.document = {{
  hidden: false,
  activeElement: null,
  getElementById(id) {{
    if (!elementStore.has(id)) elementStore.set(id, makeElement());
    return elementStore.get(id);
  }},
  querySelector() {{ return null; }},
  createElement() {{ return makeElement(); }},
}};
globalThis.window = {{ setTimeout() {{}}, clearTimeout() {{}}, setInterval() {{}} }};
eval({json.dumps(test_source)});
(async () => {{
  const contract = globalThis.__defaultConfigContract;
  const exact = "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi";
  const choices = {{
    preferred: contract.choose({{
      recommended_config: "operator-recommended",
      items: [{{ name: exact }}, {{ name: "operator-recommended" }}],
    }}),
    recommendedFallback: contract.choose({{
      recommended_config: "operator-recommended",
      items: [{{ name: "z-config" }}, {{ name: "operator-recommended" }}],
    }}),
    exact: contract.choose({{
      items: [{{ name: "a_dual_pi05" }}, {{ name: exact }}],
    }}),
    dualStable: contract.choose({{
      items: [{{ name: "z_dual_pi05" }}, {{ name: "a_dual_pi05" }}],
    }}),
    markedDefault: contract.choose({{
      items: [{{ name: "z-config", default: true }}, {{ name: "a-config" }}],
    }}),
    stableFirst: contract.choose({{
      items: [{{ name: "z-config" }}, {{ name: "a-config" }}],
    }}),
  }};
  const success = await contract.runScenario({{
    items: [{{ name: "candidate" }}],
    recommended_config: "candidate",
    revision: 41,
    callTwice: true,
  }});
  const activeDevice = await contract.runScenario({{
    items: [{{ name: "candidate" }}],
    recommended_config: "candidate",
    deviceStatuses: ["running"],
    callTwice: true,
  }});
  const activeRuntime = await contract.runScenario({{
    items: [{{ name: "candidate" }}],
    recommended_config: "candidate",
    workflow: {{
      runtime: {{ active: true, status: "running" }},
      inference: {{ running: false, status: "stopped" }},
    }},
  }});
  const failure422 = await contract.runScenario({{
    items: [{{ name: "candidate" }}],
    recommended_config: "candidate",
    apiErrorStatus: 422,
    callTwice: true,
  }});
  const conflict409 = await contract.runScenario({{
    items: [{{ name: "candidate" }}],
    recommended_config: "candidate",
    apiErrorStatus: 409,
    conflictSelectedConfig: "other-manual-selection",
    refreshedSelectedConfig: "other-manual-selection",
    callTwice: true,
  }});
  process.stdout.write(JSON.stringify({{
    choices, success, activeDevice, activeRuntime, failure422, conflict409,
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
    result = json.loads(completed.stdout)

    exact = "policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi"
    assert result["choices"] == {
        "preferred": "operator-recommended",
        "recommendedFallback": "operator-recommended",
        "exact": "a_dual_pi05",
        "dualStable": "a_dual_pi05",
        "markedDefault": "z-config",
        "stableFirst": "a-config",
    }

    success = result["success"]
    assert success["firstStatus"] == "applied"
    assert success["secondStatus"] == "already_attempted"
    assert len(success["calls"]["posts"]) == 1
    assert success["calls"]["detail"][0] == {
        "name": "candidate",
        "throwOnError": True,
    }
    assert success["calls"]["discovery"][0] == {
        "configName": "candidate",
        "preserve": False,
        "throwOnError": False,
    }
    posted = success["calls"]["posts"][0]
    assert posted["method"] == "POST"
    assert posted["payload"] == {
        "name": "candidate",
        "discovery_revision": "hardware-r1",
        "overrides": {
            "RizonRobot:2": {
                "robot_left": "left-id",
                "robot_right": "right-id",
            }
        },
        "stop_active": False,
        "auto_apply": True,
        "expected_revision": 41,
        "expected_selected_config": None,
    }

    for key in ("activeDevice", "activeRuntime"):
        skipped = result[key]
        assert skipped["firstStatus"] == "skipped"
        assert skipped["calls"]["detail"] == []
        assert skipped["calls"]["discovery"] == []
        assert skipped["calls"]["posts"] == []
    assert result["activeDevice"]["secondStatus"] == "already_attempted"

    failure = result["failure422"]
    assert failure["firstStatus"] == "failed"
    assert failure["secondStatus"] == "already_attempted"
    assert len(failure["calls"]["posts"]) == 1
    assert failure["calls"]["refresh"] == 0
    assert failure["manualChoiceRequired"] is True
    assert failure["selectValue"] == ""
    assert failure["selectedIndex"] == -1
    assert "请从下拉框手动选择" in failure["failureNotice"]

    conflict = result["conflict409"]
    assert conflict["firstStatus"] == "converged"
    assert conflict["secondStatus"] == "already_attempted"
    assert len(conflict["calls"]["posts"]) == 1
    assert conflict["calls"]["refresh"] == 1
    assert conflict["manualChoiceRequired"] is False
    assert conflict["failureNotice"] == ""
    assert conflict["selectValue"] == "other-manual-selection"
    assert conflict["calls"]["detail"][-1]["name"] == "other-manual-selection"


def test_binding_controls_lock_while_running_and_fit_mobile_layout() -> None:
    source = _frontend_source()
    template = _template_source()
    css = _css_source()

    assert "ACTIVE_STATUSES.has(device.status)" in source
    assert "select.disabled = locked" in source
    assert "deviceToggleAction" in source
    assert 'device.status === "failed"' in source
    assert 'device.running ? "stop" : "start"' in source
    assert 'device.status === "stopped"' in source
    assert '["running", "degraded", "stop_timeout"]' in source
    assert "@media (max-width: 620px)" in css
    assert ".viewer-toolbar-mount" in css
    assert 'class="viewer-toolbar-bindings"' in template
    assert 'class="viewer-toolbar-devices"' in template
    assert ".viewer-toolbar-bindings" in css
    assert "grid-template-columns: 30px repeat(2, minmax(0, 1fr))" in css
    assert ".viewer-toolbar-devices > .panel-device-control" in css
    assert "@container arm-viewer-panel (max-width: 380px)" in css
    assert ".camera-card__controls" in css


def test_workspace_splitter_and_semantic_camera_slots_are_stable() -> None:
    source = _frontend_source()
    template = _template_source()
    css = _css_source()
    render = _function_body(source, "renderCameras", "refreshPreviewStatuses")
    requirement = _function_body(
        source, "cameraSlotRequirement", "cameraSlotForDevice"
    )
    splitter = _function_body(
        source, "workspaceSplitBounds", "defaultWorkspaceSplitWidth"
    ) + _function_body(
        source, "applyWorkspaceSplitWidth", "restoreWorkspaceSplitWidth"
    )

    assert 'id="workspace-splitter"' in template
    assert 'role="separator"' in template
    assert 'aria-orientation="vertical"' in template
    assert 'CAMERA_SLOT_ORDER = ["head", "left", "right"]' in source
    assert 'id="camera-empty"' not in template
    assert "cameraEmpty" not in source
    assert "for (const slot of slots)" in render
    assert "app.cameraCards.get(slot.role)" in render
    assert "elements.cameraGrid.appendChild(app.cameraCards.get(slot.role))" in render
    assert 'label: "推理必需"' in requirement
    assert 'label: "录制可选"' in requirement
    assert 'label: "未使用"' in requirement
    assert "submitBindingChange(deviceKey, { affectedKeys })" in source

    assert 'grid-template-columns: repeat(2, minmax(0, 1fr))' in css
    assert ".camera-card--head" in css
    assert "grid-column: 1 / -1" in css
    assert ".camera-card--left" in css
    assert ".camera-card--right" in css
    assert "@media (max-width: 980px)" in css
    assert ".workspace-splitter" in css
    assert "display: none" in css[css.index("@media (max-width: 980px)") :]

    assert "WORKSPACE_VISUAL_MIN_PX" in splitter
    assert "Math.min(620" in splitter
    assert "localStorage?.setItem(WORKSPACE_SPLIT_STORAGE_KEY" in source
    assert 'event.key === "ArrowLeft"' in source
    assert 'event.key === "ArrowRight"' in source
    assert 'event.key === "Home"' in source
    assert 'event.key === "End"' in source


def test_rizon_and_yam_head_controls_follow_config_requirements() -> None:
    source = _frontend_source()
    template = _template_source()
    css = _css_source()
    requirement = _function_body(
        source, "robotSlotRequirement", "updateRobotSlotRequirement"
    )
    render = _function_body(
        source, "renderRobotBindingControls", "renderBindingControls"
    )

    assert 'ROBOT_SLOT_ORDER = ["rizon", "i2rt_head"]' in source
    assert 'id="robot-device-toggle"' in template
    assert 'id="i2rt-head-device-toggle"' in template
    assert '启动或停止 Yam 头部机械臂' in template
    assert 'label: "推理必需"' in requirement
    assert 'label: "可选"' in requirement
    assert 'label: "未使用"' in requirement
    assert "app.robotSlots.rizon" in render
    assert "app.robotSlots.i2rt_head" in render
    assert '"Yam 头臂"' in render
    assert "elements.i2rtHeadDeviceToggle" in render
    assert 'elements.i2rtHeadDeviceToggle?.addEventListener("click"' in source
    assert ".panel-device-control" in css
    assert '.panel-device-requirement[data-kind="required"]' in css
    assert '.panel-device-requirement[data-kind="optional"]' in css


def test_yam_stop_is_presented_as_verified_zero_disconnect() -> None:
    source = _frontend_source()
    render = _function_body(source, "updateDeviceToggle", "robotSlotRequirement")
    action = _function_body(source, "runDeviceAction", "renderLogDeviceOptions")

    assert '"归零并断连 Yam 头臂"' in render
    assert '"重试归零并断连 Yam 头臂"' in render
    assert '"归零确认中"' in render
    assert 'action === "stop" && isI2rtHeadDevice(device)' in action
    assert "断连前将先回到 joint 全 0" in action
    assert "此断连零位不同于启动后的工作 homepose" in action
    assert "如果无法确认到位，系统会保持连接" in action


def test_binding_controls_are_per_device_sparse_and_focus_stable() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; static frontend contracts still run")

    source = _frontend_source()
    assert 'api("/api/bindings"' in source
    assert 'buildBindingUpdateOverrides(requestKeys)' in source
    assert 'completeCameraBindingRequestKeys(deviceKey, affectedKeys)' in source
    assert 'Freeze only the edited slot and any slot displaced by an atomic swap' in source
    assert '...bindingDevices()' not in _function_body(
        source,
        "completeCameraBindingRequestKeys",
        "configuredCameraBindingId",
    )
    assert 'postBindingChangeWithFreshDiscovery(bindingOverrides)' in source
    assert 'refreshHardwareDiscoveryForBindingSubmit()' in source
    assert 'void submitBindingChange(deviceKey, { affectedKeys })' in source
    assert 'const previousPhysical = resolvedCameraBindingId(targetDevice);' in source
    assert 'if (resolvedCameraBindingId(other) !== selectedPhysical) continue;' in source
    assert 'affectedKeys.push(other.key);' in source
    assert 'disabled: !camera.detected || !camera.capture || usedByLockedOthers.has(camera.id)' in source
    assert 'if (!select || bindingEditorActive(select)) return;' in source
    assert 'app.bindingErrors.set(key, message)' in source
    run_toggle = _function_body(source, "runToggleButton", "stopStateEvents")
    assert '/api/config/select' not in run_toggle
    assert "!bindingSelectionMatchesEffective(device)" in run_toggle
    assert 'const deviceKey = String(button?.dataset?.deviceKey || "")' in run_toggle
    assert "const selectedConfig = app.selectedConfig" in run_toggle
    assert "await refreshState(false)" not in run_toggle
    assert "app.selectedConfig !== selectedConfig" in run_toggle
    assert "!controlsMatchActiveConfig()" in run_toggle
    assert "item.key === deviceKey" in run_toggle
    assert "return runDeviceAction(device, action)" in run_toggle

    binding_match = _function_body(
        source,
        "bindingSelectionMatchesEffective",
        "buildBindingUpdateOverrides",
    )
    assert '["robot_left", "robot_right", "network_interface"]' in binding_match
    assert '["capture_device"]' in binding_match
    assert "bindingOverrideForDevice(device.key)" in binding_match
    assert "effectiveBindingId(device.key, field)" in binding_match

    camera_controls = _function_body(
        source,
        "updateCameraBindingControls",
        "stopCameraPreview",
    )
    assert "const bindingReady = Boolean(availableCameraBinding(selected))" in camera_controls
    assert "device?.capabilities?.start" not in camera_controls

    toggle_harness = f"""
const calls = [];
const originalDevice = {{ key: "RizonRobot:2", status: "stopped" }};
const replacementDevice = {{ key: "RizonRobot:2", status: "stopped", replacement: true }};
const app = {{
  devices: [originalDevice],
  selectedConfig: "fixture",
  bindingDirtyKeys: new Set(),
  bindingRequests: new Map(),
}};
function bindingSelectionMatchesEffective() {{ return false; }}
function controlsMatchActiveConfig() {{ return true; }}
async function submitBindingChange(deviceKey) {{
  calls.push(["binding", deviceKey]);
  app.devices = [replacementDevice];
  // Reproduce an async render changing the live button after the click.
  button.dataset.deviceKey = "";
  return true;
}}
async function runDeviceAction(device, action) {{
  calls.push(["device", device.key, action, device.replacement === true]);
  return "started";
}}
{run_toggle}
const button = {{
  dataset: {{ deviceKey: "RizonRobot:2", action: "start" }},
}};
(async () => {{
  const result = await runToggleButton(button);
  process.stdout.write(JSON.stringify({{ result, calls }}));
}})().catch((error) => {{
  process.stderr.write(error.stack || String(error));
  process.exitCode = 1;
}});
"""
    toggle_completed = subprocess.run(
        [node],
        input=toggle_harness,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert json.loads(toggle_completed.stdout) == {
        "result": "started",
        "calls": [
            ["binding", "RizonRobot:2"],
            ["device", "RizonRobot:2", "start", True],
        ],
    }

    marker = "  initialize();\n})();"
    assert marker in source
    injected = """  globalThis.__bindingLockContract = {
    run() {
      const cameraA = { key: "WebCameraOpenPi:0", status: "running" };
      const cameraB = { key: "WebCameraOpenPi:4", status: "stopped" };
      const robot = { key: "RizonRobot:2", status: "stopped" };
      const i2rtHead = { key: "YamHeadRobot:0", status: "stopped" };
      app.devices = [cameraA, cameraB, robot, i2rtHead];
      app.activeConfig = "fixture";
      app.selectedConfig = "fixture";
      elements.configSelect.value = "fixture";
      app.configSwitching = false;
      app.hardwareLoading = false;
      app.pendingActions = new Set();
      app.bindingPending = new Set();
      app.workflowPending = new Set();
      app.state = { workflow: { runtime: { active: false }, inference: { running: false } } };
      app.state.state_sync = { stale: false, authoritative: true, transitions: {} };
      app.state.safety = { fail_closed: false };
      app.stateTransportFailed = false;
      app.stateTransportDeadlineAtMs = Date.now() + 10000;
      const focused = {};
      document.activeElement = focused;
      const normal = {
        cameraA: bindingTargetLocked(cameraA),
        cameraB: bindingTargetLocked(cameraB),
        robot: bindingTargetLocked(robot),
        i2rtHead: bindingTargetLocked(i2rtHead),
        absent: bindingTargetLocked(undefined),
        focused: bindingEditorActive(focused),
        other: bindingEditorActive({}),
      };
      app.hardwareLoading = true;
      const discoveryLoading = {
        cameraB: bindingTargetLocked(cameraB),
        absent: bindingTargetLocked(undefined),
      };
      app.hardwareLoading = false;
      app.pendingActions.add(robot.key);
      const cameraButton = {
        dataset: {}, disabled: false, textContent: "",
        classList: { toggle() {} }, setAttribute() {},
      };
      const i2rtButton = {
        dataset: {}, disabled: false, textContent: "",
        classList: { toggle() {} }, setAttribute() {},
      };
      updateDeviceToggle(cameraButton, cameraB, "相机", true);
      updateDeviceToggle(i2rtButton, i2rtHead, "Yam 头臂", true);
      const parallelStart = {
        camera: { action: cameraButton.dataset.action, disabled: cameraButton.disabled },
        i2rt: { action: i2rtButton.dataset.action, disabled: i2rtButton.disabled },
      };
      app.pendingActions.delete(robot.key);
      app.state.workflow.runtime.active = true;
      const workflow = { cameraB: bindingTargetLocked(cameraB), robot: bindingTargetLocked(robot) };
      app.state.workflow.runtime.active = false;
      app.bindingPending.add(cameraB.key);
      const pending = bindingTargetLocked(cameraB);
      app.bindingPending.delete(cameraB.key);
      app.discovery = { revision: null, robots: [], cameras: [], recommendedOverrides: {} };
      app.cameraBindings = new Map([[cameraB.key, ""]]);
      const emptySelect = {
        dataset: {},
        disabled: true,
        title: "",
        value: "",
        options: [],
        replaceChildren(...options) { this.options = options; },
      };
      const emptyButton = {
        dataset: {}, disabled: false, textContent: "",
        classList: { toggle() {} }, setAttribute() {},
      };
      const emptyCard = {
        classList: { toggle() {} },
        querySelector(selector) {
          if (selector.includes("camera-binding-select")) return emptySelect;
          if (selector.includes("camera-device-toggle")) return emptyButton;
          return null;
        },
      };
      updateCameraBindingControls(cameraB, emptyCard);
      const emptyDiscovery = {
        disabled: emptySelect.disabled,
        label: emptySelect.options[0]?.text,
        action: emptyButton.dataset.action,
        startDisabled: emptyButton.disabled,
      };
      const failedAlive = deviceToggleAction({
        key: "RizonRobot:2", status: "failed", running: true,
      });
      const failedExited = deviceToggleAction({
        key: "RizonRobot:2", status: "failed", running: false,
      });
      const transitionButton = {
        dataset: {}, disabled: false, textContent: "", title: "",
        classList: { toggle() {} }, setAttribute() {},
      };
      const startingRobot = {
        key: "RizonRobot:2",
        status: "starting",
        progress: { phase: "rizon_selecting_rdk" },
      };
      app.controlTransitions.set("device:RizonRobot:2", {
        token: "start-token", desired: "running", action: "start",
      });
      updateDeviceToggle(transitionButton, startingRobot, "Rizon", true);
      const startResponse = {
        action: transitionButton.dataset.action,
        disabled: transitionButton.disabled,
        emergency: transitionButton.textContent.includes("紧急停止"),
      };
      app.controlTransitions.set("device:RizonRobot:2", {
        token: "stop-token", desired: "stopped", action: "stop",
      });
      updateDeviceToggle(transitionButton, startingRobot, "Rizon", true);
      const stopResponse = {
        action: transitionButton.dataset.action,
        disabled: transitionButton.disabled,
        emergency: transitionButton.textContent.includes("紧急停止"),
      };
      app.controlTransitions.clear();
      return {
        normal, discoveryLoading, parallelStart, workflow, pending,
        emptyDiscovery, failedAlive, failedExited, startResponse, stopResponse,
      };
    },
  };
})();"""
    test_source = source.replace(marker, injected, 1)
    harness = f"""
globalThis.document = {{
  hidden: false,
  activeElement: null,
  getElementById() {{ return {{ value: "" }}; }},
  querySelector() {{ return null; }},
}};
globalThis.window = {{}};
globalThis.Option = class Option {{
  constructor(text, value) {{
    this.text = text;
    this.value = value;
    this.disabled = false;
    this.title = "";
  }}
}};
eval({json.dumps(test_source)});
process.stdout.write(JSON.stringify(globalThis.__bindingLockContract.run()));
"""
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
    assert result == {
        "normal": {
            "cameraA": True,
            "cameraB": False,
            "robot": False,
            "i2rtHead": False,
            "absent": False,
            "focused": True,
            "other": False,
        },
        "discoveryLoading": {"cameraB": False, "absent": False},
        "parallelStart": {
            "camera": {"action": "start", "disabled": False},
            "i2rt": {"action": "start", "disabled": False},
        },
        "workflow": {"cameraB": True, "robot": True},
        "pending": True,
        "emptyDiscovery": {
            "disabled": False,
            "label": "缺失 / 未绑定（不可启动）",
            "action": "start",
            "startDisabled": True,
        },
        "failedAlive": "stop",
        "failedExited": "start",
        "startResponse": {
            "action": "stop",
            "disabled": False,
            "emergency": True,
        },
        "stopResponse": {
            "action": "",
            "disabled": True,
            "emergency": False,
        },
    }


def test_binding_submit_refreshes_discovery_and_retries_only_revision_conflicts() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; static frontend contracts still run")

    source = _frontend_source()
    helpers = _function_body(
        source,
        "isHardwareDiscoveryRevisionConflict",
        "submitBindingChange",
    )
    assert helpers.index("refreshHardwareDiscoveryForBindingSubmit()") < helpers.index(
        'api("/api/bindings"'
    )
    assert "preserve: true" in helpers
    assert "reconcileBindings: false" in helpers
    assert "attempt < 2" in helpers
    assert 'code === "hardware_discovery_changed"' in helpers
    submit = _function_body(source, "submitBindingChange", "runToggleButton")
    snapshot_start = submit.index(
        "const bindingOverrides = buildBindingUpdateOverrides(requestKeys);"
    )
    request_start = submit.index("const request = (async () =>")
    post_start = submit.index(
        "postBindingChangeWithFreshDiscovery(bindingOverrides)"
    )
    assert snapshot_start < request_start < post_start

    harness = f'''
class ApiError extends Error {{
  constructor(message, status, payload) {{
    super(message);
    this.status = status;
    this.payload = payload;
  }}
}}
const app = {{
  selectedConfig: "fixture",
  discovery: {{ revision: "stale" }},
  bindingDiscoveryRefreshPromise: null,
}};
const elements = {{ configSelect: {{ value: "fixture" }} }};
let discoveryCalls = [];
let posts = [];
let mode = "revision-conflict";
let mutableDraft = {{
  "WebCameraOpenPi:4": {{ capture_device: "camera-right" }},
  "WebCameraOpenPi:2": {{ capture_device: "camera-head" }},
}};
async function loadHardwareDiscovery(options) {{
  discoveryCalls.push({{ ...options }});
  const revision = `fresh-${{discoveryCalls.length}}`;
  app.discovery = {{ revision }};
  // Reproduce the old race: discovery reconciliation polluted the mutable UI
  // draft after the user had already produced a valid atomic swap.
  mutableDraft = {{
    "WebCameraOpenPi:4": {{ capture_device: "camera-head" }},
    "WebCameraOpenPi:2": {{ capture_device: "camera-head" }},
  }};
}}
function buildBindingUpdateOverrides(keys) {{
  return Object.fromEntries(
    keys.map((key) => [key, {{ ...mutableDraft[key] }}])
  );
}}
async function api(path, request) {{
  const payload = JSON.parse(request.body);
  posts.push({{ path, payload }});
  if (mode === "revision-conflict" && posts.length === 1) {{
    throw new ApiError(
      "Hardware discovery changed; refresh discovery before applying bindings",
      409,
      {{ error: "Hardware discovery changed; refresh discovery before applying bindings" }}
    );
  }}
  if (mode === "other-conflict") {{
    throw new ApiError(
      "Stop the target device first",
      409,
      {{ error: "Stop the target device first" }}
    );
  }}
  return {{ revision: payload.discovery_revision }};
}}
{helpers}
(async () => {{
  const requestKeys = ["WebCameraOpenPi:4", "WebCameraOpenPi:2"];
  const frozenIntent = buildBindingUpdateOverrides(requestKeys);
  const success = await postBindingChangeWithFreshDiscovery(frozenIntent);
  const revisionConflict = {{
    success,
    discoveryCalls: [...discoveryCalls],
    posts: [...posts],
  }};

  discoveryCalls = [];
  posts = [];
  mode = "other-conflict";
  let otherError = null;
  try {{
    await postBindingChangeWithFreshDiscovery(frozenIntent);
  }} catch (error) {{
    otherError = {{ status: error.status, message: error.message }};
  }}
  process.stdout.write(JSON.stringify({{
    revisionConflict,
    otherConflict: {{ error: otherError, discoveryCalls, posts }},
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
    result = json.loads(completed.stdout)

    revision_conflict = result["revisionConflict"]
    assert [call["configName"] for call in revision_conflict["discoveryCalls"]] == [
        "fixture",
        "fixture",
    ]
    assert all(call["preserve"] is True for call in revision_conflict["discoveryCalls"])
    assert [
        post["payload"]["discovery_revision"]
        for post in revision_conflict["posts"]
    ] == ["fresh-1", "fresh-2"]
    assert revision_conflict["posts"][-1]["payload"]["overrides"] == {
        "WebCameraOpenPi:4": {"capture_device": "camera-right"},
        "WebCameraOpenPi:2": {"capture_device": "camera-head"},
    }
    assert revision_conflict["success"] == {"revision": "fresh-2"}

    other_conflict = result["otherConflict"]
    assert other_conflict["error"] == {
        "status": 409,
        "message": "Stop the target device first",
    }
    assert len(other_conflict["discoveryCalls"]) == 1
    assert len(other_conflict["posts"]) == 1


def test_arm_viewer_labels_world_and_fresh_tcp_coordinate_axes() -> None:
    source = _arm_viewer_source()

    assert 'X: "#ff4d4d"' in source
    assert 'Y: "#42d66b"' in source
    assert 'Z: "#4d85ff"' in source
    assert 'createLabeledAxes("world-coordinate-axes"' in source
    assert "this.worldFrame.quaternion.copy(FLEXIV_TO_THREE_QUATERNION);" in source
    assert 'aria-label", "World axes: X red, Y green, Z blue"' in source
    assert 'new THREE.AxesHelper(size)' in source
    assert "const tcpAxes = createLabeledAxes(" in source
    assert '`${side}-active-tcp-axes`' in source
    assert "new THREE.AxesHelper(TCP_AXIS_SIZE)" in source
    assert "payload.metadata?.fresh !== false" in source
    assert "frame.userData.fresh = false;" in source
    assert "frame.userData.fresh = true;" in source


def test_arm_viewer_prefers_fresh_joints_then_top_level_home_per_arm() -> None:
    source = _arm_viewer_source()
    state_start = source.index("function robotStateFromPayload(")
    state_end = source.index("function createAxisLabel(", state_start)
    state_parser = source[state_start:state_end]
    neutral_start = source.index("  _useNeutralRobotState()")
    neutral_end = source.index("  _showRobotStateStatus()", neutral_start)
    neutral = source[neutral_start:neutral_end]

    assert 'payload.status === "ok"' in state_parser
    assert "payload.metadata?.fresh !== false" in state_parser
    assert 'payload.home && typeof payload.home === "object"' in state_parser
    assert "home.joints_left" in state_parser
    assert "home.joints_right" in state_parser
    assert "left: liveLeft || homeLeft || [...DEFAULT_POSE]" in state_parser
    assert "right: liveRight || homeRight || [...DEFAULT_POSE]" in state_parser
    assert 'left: liveLeft ? "live" : homeLeft ? "home" : "default"' in state_parser
    assert "left: liveLeft ? normalizedTcpPose" in state_parser
    assert "this.homeJointStates = state.home;" in source
    assert "this._liveFlexivStateAvailable = state.fresh.left || state.fresh.right;" in source
    assert "this._liveStateAvailable = this._liveFlexivStateAvailable || state.yam.fresh;" in source
    assert "yam: normalizedYamState(payload.yam_head ?? payload.yam)" in state_parser
    assert "this.updateYamState(state.yam);" in source
    assert "this.homeJointStates.left || DEFAULT_POSE" in neutral
    assert "this.homeJointStates.right || DEFAULT_POSE" in neutral
    assert "this.updateJointStates(left, right);" in neutral


def test_device_control_polling_uses_canonical_state_and_visibility_gates() -> None:
    source = _frontend_source()
    refresh = _function_body(source, "refreshState", "initialStateAllowsBootstrap")
    preview = _function_body(source, "refreshPreviewStatuses", "isMotionRisk")
    preview_request = _function_body(
        source, "refreshPreviewStatus", "refreshPreviewStatuses"
    )
    timeline = _function_body(
        source, "renderRecordingTimeline", "renderRecording"
    )
    robot_options = _function_body(
        source, "replaceRobotBindingOptions", "deviceToggleAction"
    )
    camera_options = _function_body(
        source, "updateCameraBindingControls", "stopCameraPreview"
    )

    assert refresh.index('api("/api/state"') < refresh.index('api("/api/devices"')
    assert "if (!Array.isArray(devicePayload))" in refresh
    assert "Promise.allSettled([" not in refresh
    assert "signal: controller.signal" in refresh
    assert "statePending: 200" in source
    assert "stateRecovery: 250" in source
    assert "stateActiveFast: 500" in source
    assert "stateActive: 1000" in source
    assert "stateIdle: 2500" in source
    assert "setInterval" not in source
    assert 'document.addEventListener("visibilitychange"' in source
    assert "if (document.hidden) suspendPolling();" in source
    assert "scheduleStatePoll(0)" in source

    assert "cameraCardCanStream" in preview
    assert "signal: controller.signal" in preview_request
    assert "IntersectionObserver" in source
    assert "releaseCameraImage(image)" in source
    assert "URL?.revokeObjectURL" in source

    assert "replaceChildren" not in timeline
    assert "syncRecordingTrackNodes" in timeline
    assert "replaceChildren" not in robot_options
    assert "syncSelectOptions" in robot_options
    assert "replaceChildren" not in camera_options
    assert "syncSelectOptions" in camera_options
    assert "app.renderKeys.logDevices === key" in source
    assert "globalThis.__controlPanelPerf" in source


def test_camera_operations_keep_previews_and_status_requests_independent() -> None:
    source = _frontend_source()
    css = _css_source()
    stream_gate = _function_body(
        source, "cameraCardCanStream", "releaseCameraImage"
    )
    preview = _function_body(
        source, "refreshPreviewStatuses", "isMotionRisk"
    )
    per_camera = _function_body(
        source, "refreshPreviewStatus", "refreshPreviewStatuses"
    )
    frame_loop = _function_body(
        source, "requestCameraPreviewFrame", "scheduleCameraPreviewFrame"
    )
    frame_schedule = _function_body(
        source, "scheduleCameraPreviewFrame", "updateCameraBindingControls"
    )

    assert 'controlsFailClosed("preview")' not in stream_gate
    assert "previewStatusRequests: new Map()" in source
    assert (
        "for (const device of cameras) void refreshPreviewStatus(device)"
        in preview
    )
    assert "Promise.allSettled" not in preview
    assert "POLL_INTERVALS.previewRequestTimeout" in per_camera
    assert "app.previewStatusRequests.get(device.key)" in per_camera
    assert "updateCameraCard(current, card, slot)" in per_camera
    assert "previewFrameTimers: new Map()" in source
    assert "previewFrameInFlight: new Set()" in source
    assert "MAX_CONCURRENT_PREVIEW_FRAMES = 2" in source
    assert "app.refreshing ||" in frame_loop
    assert "app.previewFrameInFlight.size >= MAX_CONCURRENT_PREVIEW_FRAMES" in frame_loop
    assert "app.previewFrameInFlight.add(device.key)" in frame_loop
    assert "finishCameraPreviewFrame(device.key, image)" in source
    assert "app.previewFrameInFlight.delete(deviceKey)" in source
    assert "PREVIEW_BROWSER_CLIENT_VERSION" in frame_loop
    assert "client_version=" in frame_loop
    assert 'cameraEndpoint(device, "preview.jpg")' in frame_loop
    assert 'cameraEndpoint(device, "preview.mjpg")' not in source
    assert "app.previewFrameTimers.set(device.key, timer)" in frame_schedule
    assert "cancelCameraPreviewFrame(device.key)" in frame_schedule
    assert 'card.classList.toggle("is-busy", transitioning)' in source
    assert 'button.setAttribute("aria-busy"' in source
    assert ".panel-device-toggle.is-busy::before" in css
    assert "animation: camera-operation-progress" in css

    assert 'typeof document.hasFocus === "function"' in stream_gate
    assert 'globalThis.addEventListener?.("blur"' in source
    assert 'globalThis.addEventListener?.("focus"' in source
    blur_start = source.index('globalThis.addEventListener?.("blur"')
    blur_end = source.index('globalThis.addEventListener?.("focus"', blur_start)
    blur_handler = source[blur_start:blur_end]
    assert "for (const card of app.cameraCards.values())" in blur_handler
    assert "stopCameraPreview(card)" in blur_handler
    assert "cancelCameraPreviewFrame(deviceKey)" not in blur_handler


def test_gripper_control_uses_direct_transport_without_service_toggle() -> None:
    source = _frontend_source()
    assert "runGripperServiceAction" not in source
    assert "/api/gripper-service/" not in source
    assert "gripperServicePending" not in source
    assert "gripper-calibrate-all" in source


@pytest.mark.skip(reason="YAM progress labels require the separate R1/YAM adapter")
def test_device_progress_labels_follow_real_robot_code_paths() -> None:
    frontend = _frontend_source()
    lifecycle = (ROOT / "lifecycle.py").read_text(encoding="utf-8")
    robot = (ROOT.parents[1] / "devices" / "robots" / "rizon.py").read_text(
        encoding="utf-8"
    )

    phases = {
        "rizon_connecting",
        "rizon_configuring_control",
        "rizon_resolving_home",
        "rizon_moving_home",
        "rizon_starting_stream",
        "yam_connecting",
        "yam_moving_init_pose",
        "yam_waiting_tracking",
        "yam_waiting_head_shm",
        "yam_starting_publisher",
        "yam_moving_zero",
        "yam_confirming_zero",
        "yam_disconnecting",
        "yam_zero_hold",
    }
    for phase in phases:
        assert f'"{phase}"' in robot
        assert f"{phase}:" in frontend

    assert "'phase': 'rizon_selecting_rdk'" in lifecycle
    assert "rizon_selecting_rdk:" in frontend

    assert "DEVICE_START_PROGRESS_PHASES" in frontend
    assert "DEVICE_STOP_PROGRESS_PHASES" in frontend
    assert 'desired === "running" && !DEVICE_START_PROGRESS_PHASES.has(phase)' in frontend
    assert 'desired === "stopped" && !DEVICE_STOP_PROGRESS_PHASES.has(phase)' in frontend


def test_device_control_adaptive_poll_and_select_diff_contract_in_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; static frontend contracts still run")

    source = _frontend_source()
    marker = "  initialize();\n})();"
    assert marker in source
    injected = r'''  globalThis.__performanceContract = {
    run() {
      const select = {
        value: "",
        dataset: {},
        replacements: 0,
        replaceChildren(...options) {
          this.options = options;
          this.replacements += 1;
        },
      };
      const first = [
        { label: "Use config", value: "", disabled: false },
        { label: "Left", value: "left", disabled: false },
      ];
      syncSelectOptions(select, first, "left");
      syncSelectOptions(select, first, "left");
      document.activeElement = select;
      syncSelectOptions(select, [...first, { label: "Other", value: "other", disabled: false }], "other");
      const focusedReplacements = select.replacements;
      document.activeElement = null;
      syncSelectOptions(select, [...first, { label: "Other", value: "other", disabled: false }], "other");

      app.pendingActions = new Set();
      app.bindingPending = new Set();
      app.workflowPending = new Set();
      app.recordingActionPending = "";
      app.state = { workflow: {}, recording: {} };
      app.devices = [];
      const idle = statePollDelay();
      app.devices = [{ status: "running" }];
      const active = statePollDelay();
      app.pendingActions.add("device");
      const pending = statePollDelay();
      return {
        replacements: select.replacements,
        focusedReplacements,
        selected: select.value,
        idle,
        active,
        pending,
      };
    },
  };
})();'''
    test_source = source.replace(marker, injected, 1)
    harness = f'''
const elementStore = new Map();
function makeElement() {{
  return {{
    value: "",
    disabled: false,
    hidden: false,
    textContent: "",
    className: "",
    dataset: {{}},
    classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
    setAttribute() {{}},
    addEventListener() {{}},
    querySelector() {{ return null; }},
  }};
}}
globalThis.document = {{
  hidden: false,
  activeElement: null,
  getElementById(id) {{
    if (!elementStore.has(id)) elementStore.set(id, makeElement());
    return elementStore.get(id);
  }},
  querySelector() {{ return null; }},
  querySelectorAll() {{ return []; }},
  createElement() {{ return makeElement(); }},
}};
globalThis.window = {{ setTimeout() {{ return 1; }}, clearTimeout() {{}} }};
globalThis.Option = class Option {{
  constructor(label, value) {{ this.label = label; this.value = value; }}
}};
eval({json.dumps(test_source)});
process.stdout.write(JSON.stringify(globalThis.__performanceContract.run()));
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
        "replacements": 2,
        "focusedReplacements": 1,
        "selected": "other",
        "idle": 1000,
        "active": 500,
        "pending": 200,
    }


if __name__ == "__main__":
    test_stop_timeout_is_a_visible_safety_error()
    test_preview_polling_keeps_top_level_and_nested_metadata()
    test_camera_routes_are_class_and_id_based_not_shm_based()
    print("3 frontend contract tests passed")


def test_runtime_tuning_is_bound_to_selected_config_revision() -> None:
    source = _frontend_source()

    assert "function selectedRuntimeTuningContext()" in source
    assert "function runtimeTuningMatchesContext(" in source
    assert "function invalidateRuntimeTuning({ preserveDraft = false } = {})" in source
    assert "if (!preserveDraft) {" in source
    assert 'app.generatedConfigSource = "";' in source
    assert "function setSelectedConfig(nextValue)" in source
    assert 'payloadRevision === context.revision' in source
    assert "const tuningContext = selectedRuntimeTuningContext();" in source
    assert "runtimeTuningMatchesContext(schema, tuningContext)" in source
    assert 'error: "调参配置已过期，请等待当前配置的参数重新加载。"' in source
    assert "preserveDraft: app.currentConfigSavePending" in source


def test_config_save_preserves_committed_draft_until_deferred_reload() -> None:
    source = _frontend_source()
    save = _function_body(source, "saveCurrentConfig", "renderRuntimeTuning")
    diagnostics = _function_body(
        source,
        "loadRuntimeDiagnostics",
        "ensureDiagnosticsPolling",
    )

    assert "if (!reloadDeferred) app.runtimeTuningDraft.clear();" in save
    assert "loadRuntimeDiagnostics({ includeTuning: true, force: true })" in save
    assert "app.stateRequestController?.abort();" in save
    assert "if (app.stateRefreshPromise) await app.stateRefreshPromise;" in save
    assert 'const allowedSections = new Set(["policy"]);' in source
    assert "if (!force) return null;" in diagnostics
    assert "app.runtimeMetricsController?.abort();" in diagnostics
    assert "app.runtimeMetricsController === controller" in diagnostics
