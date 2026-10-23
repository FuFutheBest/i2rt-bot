"""Behavioral contracts for keeping the config chooser on server truth."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_config_dropdown_tracks_server_selection_across_async_paths() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    source = (ROOT / "static" / "device_control.js").read_text(encoding="utf-8")
    marker = "  initialize();\n})();"
    assert marker in source
    injected = r'''  globalThis.__configSelectionContract = {
    reset(items, selected) {
      app.configs = extractConfigs({ items });
      app.recommendedConfig = "";
      app.selectedConfig = selected || "";
      app.activeConfig = "";
      app.configSelectionBeforeOpen = "";
      app.configSwitching = false;
      app.configSelectionDeferred = false;
      app.configOptionsDeferred = false;
      app.bootstrapManualChoiceRequired = false;
      app.bootstrapFailureNotice = "";
      app.bindingDirty = false;
      app.bindingDirtyKeys = new Set();
      app.bindingErrors = new Map();
      app.devices = [];
      app.state = { workflow: {} };
      elements.configSelect.value = "";
      elements.configSelect.selectedIndex = -1;
      renderConfigOptions({ preserveValue: false });
    },
    initialSelection() {
      this.reset([{ name: "a" }, { name: "server-selected" }], "server-selected");
      return elements.configSelect.value;
    },
    async autoSelection() {
      this.reset([{ name: "a" }, { name: "b" }], "");
      app.recommendedConfig = "b";
      app.state = {
        revision: 11,
        selected_config: null,
        workflow: {
          runtime: { active: false, status: "stopped" },
          inference: { running: false, status: "stopped" },
        },
      };
      app.initialStateObserved = true;
      app.bootstrapAttempted = false;
      app.discovery = { revision: "hardware-r1", robots: [], cameras: [], recommendedOverrides: {} };
      let resolvePost;
      let postStarted = false;
      const response = new Promise((resolve) => { resolvePost = resolve; });
      loadConfigDetail = async () => {};
      loadHardwareDiscovery = async () => {};
      buildBindingOverrides = () => ({});
      renderBindingControls = () => {};
      renderGrippers = () => {};
      showToast = () => {};
      refreshState = async () => {};
      api = async (path) => {
        if (path !== "/api/config/select") throw new Error(`unexpected API ${path}`);
        postStarted = true;
        return response;
      };
      const pending = maybeAutoApplyDefaultConfig();
      while (!postStarted) await Promise.resolve();
      const during = elements.configSelect.value;
      resolvePost({ selected_config: "b", active_config: null, revision: 12 });
      const result = await pending;
      return { during, after: elements.configSelect.value, status: result.status };
    },
    async stateFailureRecovery() {
      this.reset([{ name: "old" }, { name: "recovered" }], "old");
      app.state = { selected_config: "old", devices: [], workflow: {} };
      app.refreshing = false;
      app.stateRefreshPromise = null;
      app.stateRequestController = null;
      renderBindingControls = () => {};
      renderMetrics = () => {};
      renderWorkflow = () => {};
      renderGrippers = () => {};
      renderRecording = () => {};
      renderLogDeviceOptions = () => {};
      setServerStatus = () => {};
      setGlobalNotice = () => {};
      let stateFails = true;
      api = async (path) => {
        if (path === "/api/state") {
          if (stateFails) throw new ApiError("temporary", 503, {});
          return {
            revision: 22,
            selected_config: "recovered",
            active_config: null,
            devices: [],
            workflow: {},
          };
        }
        if (path === "/api/devices") return [];
        throw new Error(`unexpected API ${path}`);
      };
      await refreshState(false);
      const afterFailure = elements.configSelect.value;
      stateFails = false;
      await refreshState(false);
      return { afterFailure, afterRecovery: elements.configSelect.value };
    },
    async switchResult({ fail }) {
      this.reset([{ name: "applied" }, { name: "candidate" }], "applied");
      app.discovery = { revision: "hardware-r2", robots: [], cameras: [], recommendedOverrides: {} };
      app.configDevices = [];
      app.bindingConfigName = "applied";
      app.robotBindings = { left: "", right: "" };
      app.cameraBindings = new Map();
      elements.configSelect.value = "candidate";
      renderBindingControls = () => {};
      renderGrippers = () => {};
      showToast = () => {};
      loadConfigDetail = async () => {};
      loadHardwareDiscovery = async () => {};
      buildBindingOverrides = () => ({});
      confirmOperation = async () => true;
      refreshState = async () => {};
      api = async (path) => {
        if (path !== "/api/config/select") throw new Error(`unexpected API ${path}`);
        if (fail) throw new ApiError("switch failed", 500, {});
        return { selected_config: "candidate", active_config: null, revision: 31 };
      };
      await selectConfigWithConfirmation("candidate", "applied");
      return { value: elements.configSelect.value, selected: app.selectedConfig };
    },
  };
})();'''
    test_source = source.replace(marker, injected, 1)
    harness = f'''
class MockOption {{
  constructor(label, value) {{
    this.label = label;
    this.textContent = label;
    this.value = String(value);
    this.disabled = false;
    this.title = "";
  }}
}}
globalThis.Option = MockOption;
function makeElement() {{
  return {{
    value: "", selectedIndex: -1, disabled: false, hidden: false,
    textContent: "", title: "", dataset: {{}}, children: [],
    classList: {{ toggle() {{}}, add() {{}}, remove() {{}} }},
    setAttribute() {{}}, addEventListener() {{}}, querySelector() {{ return null; }},
    appendChild(child) {{ this.children.push(child); }},
    replaceChildren(...children) {{ this.children = children; this.value = ""; this.selectedIndex = children.length ? 0 : -1; }},
    add(option) {{ this.children.push(option); if (this.selectedIndex < 0) {{ this.selectedIndex = 0; this.value = option.value; }} }},
  }};
}}
const elementStore = new Map();
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
  createDocumentFragment() {{ return makeElement(); }},
}};
globalThis.window = {{ setTimeout() {{ return 1; }}, clearTimeout() {{}}, confirm() {{ return true; }} }};
eval({json.dumps(test_source)});
(async () => {{
  const contract = globalThis.__configSelectionContract;
  const result = {{
    initial: contract.initialSelection(),
    recovery: await contract.stateFailureRecovery(),
    automatic: await contract.autoSelection(),
    failedSwitch: await contract.switchResult({{ fail: true }}),
    successfulSwitch: await contract.switchResult({{ fail: false }}),
  }};
  process.stdout.write(JSON.stringify(result));
}})().catch((error) => {{
  process.stderr.write(error.stack || String(error));
  process.exitCode = 1;
}});
'''
    completed = subprocess.run(
        [node],
        input=harness,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert json.loads(completed.stdout) == {
        "initial": "server-selected",
        "automatic": {"during": "b", "after": "b", "status": "applied"},
        "recovery": {"afterFailure": "old", "afterRecovery": "recovered"},
        "failedSwitch": {"value": "applied", "selected": "applied"},
        "successfulSwitch": {"value": "candidate", "selected": "candidate"},
    }
