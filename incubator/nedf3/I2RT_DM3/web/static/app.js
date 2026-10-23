"use strict";

const csrf = document.querySelector('meta[name="i2rt-csrf"]').content;
const titles = {
  "login-view": ["SESSION", "DM3 登录"],
  "collection-view": ["COLLECTION", "数据采集"],
  "upload-view": ["UPLOAD", "上传与性能监控"],
};
const state = {
  runtime: null,
  selectedTaskIndex: null,
  logOffsets: { device: 0, saver: 0, calibration: 0, upload: 0 },
  stateBusy: false,
  uploadBusy: false,
  settingsInitialized: false,
  cameraStreaming: false,
  cameraRevision: 0,
  systemProfiles: [],
  hardwareInventory: null,
  hardwareProfileLoaded: null,
  hardwareReady: false,
  hardwareBusy: false,
  visualizerEpisodes: [],
};

const byId = (id) => document.getElementById(id);
const ANSI_ESCAPE_PATTERN = /(?:\x1B\][^\x07]*(?:\x07|\x1B\\))|(?:\x1B\[[0-?]*[ -/]*[@-~])/g;

function stripAnsi(text) {
  return text.replace(ANSI_ESCAPE_PATTERN, "");
}

async function api(path, options = {}) {
  const requestOptions = { method: "GET", ...options, headers: { ...(options.headers || {}) } };
  if (requestOptions.method !== "GET") {
    requestOptions.headers["Content-Type"] = "application/json";
    requestOptions.headers["X-I2RT-CSRF"] = csrf;
    if (requestOptions.body === undefined) requestOptions.body = "{}";
  }
  const response = await fetch(path, requestOptions);
  let payload;
  try { payload = await response.json(); }
  catch (_error) { throw new Error(`服务器返回了无效响应 (${response.status})`); }
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `请求失败 (${response.status})`);
  return payload.data;
}

function notify(message, error = false) {
  const toast = document.createElement("div");
  toast.className = `toast${error ? " error" : ""}`;
  toast.textContent = message;
  byId("alert-region").appendChild(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function setView(viewId) {
  if (!titles[viewId]) return;
  if (viewId !== "login-view" && !state.runtime?.session?.logged_in) viewId = "login-view";
  if (viewId === "collection-view" && !state.runtime?.session?.task_selected) viewId = "login-view";
  document.querySelectorAll(".view").forEach((node) => node.classList.toggle("active", node.id === viewId));
  document.querySelectorAll(".nav-item").forEach((node) => node.classList.toggle("active", node.dataset.view === viewId));
  byId("page-eyebrow").textContent = titles[viewId][0];
  byId("page-title").textContent = titles[viewId][1];
  window.location.hash = viewId.replace("-view", "");
  if (viewId === "upload-view") refreshUpload();
  syncCameraPreviews();
}

function statusBadge(node, text, tone) {
  node.textContent = text;
  node.className = `badge ${tone}`;
}

function processLabels(runtime) {
  const device = runtime.processes.device;
  const saver = runtime.processes.saver;
  const teleoperation = {
    starting: ["启动中", "warn"],
    safe_hold: ["安全保持", "good"],
    aligning: ["校准中", "warn"],
    active: ["遥操中", "good"],
    fault: ["故障", "bad"],
    stopped: ["停止", "neutral"],
  }[device.teleoperation_phase] || ["未知", "neutral"];
  const deviceText = device.stopping ? "停止中" : device.teleoperation_phase === "fault" ? "遥操作故障" : device.running ? (device.ready_for_episode ? "运行" : "启动中") : device.exit_code && device.exit_code !== 0 ? "异常退出" : "停止";
  const deviceTone = device.teleoperation_phase === "fault" ? "bad" : device.running && device.ready_for_episode ? "good" : device.running || device.stopping ? "warn" : device.exit_code ? "bad" : "neutral";
  const saverText = saver.stopping ? "停止中" : saver.finalizing_episode ? "完成中" : saver.recording ? "采集中" : saver.running ? "待机" : saver.exit_code && saver.exit_code !== 0 ? "异常退出" : "停止";
  const saverTone = saver.recording ? "bad" : saver.running ? "good" : saver.stopping || saver.finalizing_episode ? "warn" : saver.exit_code ? "bad" : "neutral";
  return { deviceText, deviceTone, saverText, saverTone, teleoperationText: teleoperation[0], teleoperationTone: teleoperation[1] };
}

function operationStatus(runtime) {
  if (!runtime) return null;
  const { device, saver, calibration } = runtime.processes;
  if (calibration.running || calibration.stopping) {
    return { label: "校准中", tone: "calibrating" };
  }
  if (device.teleoperation_phase === "aligning") {
    return { label: "校准中", tone: "calibrating" };
  }
  if (saver.recording) return { label: "采集中", tone: "active" };
  if (device.teleoperation_phase === "active") return { label: "遥操中", tone: "active" };
  return null;
}

function renderOperationStatus(runtime) {
  const node = byId("operation-status");
  const status = operationStatus(runtime);
  node.hidden = !status;
  node.className = `operation-status${status ? ` ${status.tone}` : ""}`;
  byId("operation-status-label").textContent = status?.label || "";
}

function renderState(runtime) {
  state.runtime = runtime;
  const session = runtime.session;
  const controls = runtime.controls;
  const calibration = runtime.processes.calibration;
  const { deviceText, deviceTone, saverText, saverTone, teleoperationText, teleoperationTone } = processLabels(runtime);
  const identity = session.logged_in ? `${session.user_name || session.login_phone} (${session.user_id || "—"})` : "未登录";

  byId("operator-label").textContent = identity;
  byId("logout-button").disabled = !session.logged_in;
  byId("login-button").disabled = session.logged_in;
  byId("fetch-task-button").disabled = !session.logged_in;
  statusBadge(byId("login-badge"), session.logged_in ? "已登录" : "未登录", session.logged_in ? "good" : "neutral");
  byId("dm3-endpoint").textContent = runtime.config.dm3_base_url;
  byId("selected-task-label").textContent = session.task_selected ? `${session.task_execution_id} / ${session.task_name}` : "未选择任务";
  document.querySelectorAll("[data-auth-required]").forEach((node) => { node.disabled = !session.logged_in; });
  document.querySelectorAll("[data-task-required]").forEach((node) => {
    node.disabled = !session.logged_in || !session.task_selected;
    node.title = session.task_selected ? "" : "请先选择 DM3 任务";
  });

  byId("device-metric").textContent = deviceText;
  byId("teleoperation-metric").textContent = teleoperationText;
  byId("saver-metric").textContent = saverText;
  byId("episode-count").textContent = runtime.processes.saver.saved_episode_count;
  byId("last-episode").textContent = runtime.processes.saver.last_episode_name;
  byId("collection-task").textContent = session.task_selected ? session.task_name : "未选择";
  byId("output-dir").textContent = runtime.config.output_dir;
  byId("device-config").textContent = runtime.config.device;
  byId("saver-config").textContent = runtime.config.saver;
  statusBadge(byId("device-badge"), deviceText, deviceTone);
  statusBadge(byId("teleoperation-badge"), teleoperationText, teleoperationTone);
  statusBadge(byId("saver-badge"), saverText, saverTone);
  const calibrationRunning = calibration.running || calibration.stopping;
  statusBadge(byId("calibration-badge"), calibration.stopping ? "停止中" : calibration.running ? "标定中" : calibration.exit_code && calibration.exit_code !== 0 ? "失败" : "停止", calibration.stopping ? "warn" : calibration.running ? "warn" : calibration.exit_code && calibration.exit_code !== 0 ? "bad" : "neutral");

  const teleoperationFault = byId("teleoperation-fault");
  teleoperationFault.hidden = !runtime.processes.device.teleoperation_fault;
  byId("teleoperation-fault-reason").textContent = runtime.processes.device.teleoperation_fault || "";

  byId("start-device-button").disabled = !controls.can_start_device || (state.systemProfiles.length > 0 && !state.hardwareReady);
  byId("stop-device-button").disabled = !controls.can_stop_device;
  byId("start-saver-button").disabled = !controls.can_start_saver;
  byId("stop-saver-button").disabled = !controls.can_stop_saver;
  byId("toggle-episode-button").disabled = !controls.can_toggle_episode;
  byId("discard-episode-button").disabled = !controls.can_discard_episode;
  byId("visualize-episode-button").disabled = !controls.can_visualize_episodes;
  byId("calibrate-grippers-button").disabled = !controls.can_calibrate_grippers || (state.systemProfiles.length > 0 && !state.hardwareReady);
  byId("calibration-control").hidden = !controls.calibration_supported;
  byId("calibration-log-section").hidden = !controls.calibration_supported;
  byId("calibrate-grippers-button").textContent = calibrationRunning ? "夹爪标定中" : "标定左右夹爪";

  const saver = runtime.processes.saver;
  byId("episode-status").textContent = saver.finalizing_episode ? "正在完成 Episode" : saver.recording ? "正在采集" : saver.running ? "等待开始" : "保存进程未运行";
  byId("toggle-episode-button").textContent = saver.recording ? "网页完成 Episode" : "网页开始 Episode";
  byId("auto-upload-toggle").checked = Boolean(runtime.upload.auto_enabled);
  const hardwareLocked = [runtime.processes.device, runtime.processes.saver, runtime.processes.calibration]
    .some((process) => process.running || process.stopping);
  byId("system-profile-select").disabled = !session.logged_in || hardwareLocked || state.hardwareBusy;
  byId("refresh-hardware-button").disabled = !session.logged_in || hardwareLocked || state.hardwareBusy || !state.systemProfiles.length;
  byId("save-hardware-button").disabled = !session.logged_in || hardwareLocked || state.hardwareBusy || !state.hardwareInventory;
  document.querySelectorAll("#hardware-fields select").forEach((node) => { node.disabled = hardwareLocked || state.hardwareBusy; });
  if (session.logged_in && state.systemProfiles.length && state.hardwareProfileLoaded !== runtime.config.system_profile && !state.hardwareBusy) {
    void loadHardware(runtime.config.system_profile || byId("system-profile-select").value, false);
  }
  renderOperationStatus(runtime);
  syncCameraPreviews();
}

function profileById(profileId) {
  return state.systemProfiles.find((profile) => profile.id === profileId);
}

function hardwareSelect(labelText, group, role, backend, candidates, saved) {
  const label = document.createElement("label");
  label.textContent = labelText;
  const select = document.createElement("select");
  select.dataset.group = group;
  select.dataset.role = role;
  select.dataset.backend = backend;
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "请选择";
  select.appendChild(placeholder);
  candidates.forEach((candidate) => {
    const option = document.createElement("option");
    option.value = candidate.identity;
    option.textContent = candidate.label;
    select.appendChild(option);
  });
  const savedIdentity = saved?.identity || "";
  if (savedIdentity && !candidates.some((candidate) => candidate.identity === savedIdentity)) {
    const missing = document.createElement("option");
    missing.value = savedIdentity;
    missing.textContent = `${savedIdentity}（当前未发现）`;
    missing.disabled = true;
    select.appendChild(missing);
  }
  select.value = savedIdentity;
  select.addEventListener("change", () => {
    state.hardwareReady = false;
    byId("hardware-binding-status").textContent = "配置已修改，尚未保存";
    if (state.runtime) renderState(state.runtime);
  });
  label.appendChild(select);
  return label;
}

function renderHardwareInventory(inventory) {
  const profile = inventory.profile;
  const binding = inventory.binding;
  const fields = byId("hardware-fields");
  fields.replaceChildren();
  const cameraLabels = { left: "左腕相机", right: "右腕相机", head: "头部相机" };
  for (const [role, camera] of Object.entries(profile.cameras)) {
    fields.appendChild(hardwareSelect(
      cameraLabels[role] || `${role}相机`,
      "cameras",
      role,
      camera.backend,
      inventory.candidates[camera.backend] || [],
      binding?.cameras?.[role],
    ));
  }
  for (const role of profile.gripper.roles) {
    fields.appendChild(hardwareSelect(
      `${role === "left" ? "左" : "右"}夹爪串口`,
      "grippers",
      role,
      "serial",
      inventory.candidates.serial || [],
      binding?.grippers?.[role],
    ));
  }
  state.hardwareReady = Boolean(binding && binding.profile_revision === profile.revision &&
    [...fields.querySelectorAll("select")].every((select) => select.value && !select.selectedOptions[0]?.disabled));
  const warning = inventory.warnings?.length ? `；${inventory.warnings.join("；")}` : "";
  byId("hardware-binding-status").textContent = `${state.hardwareReady ? "已加载本机配置" : "请选择并保存全部硬件"}${warning}`;
}

async function loadProfiles() {
  try {
    const result = await api("/api/device/profiles");
    state.systemProfiles = result.profiles || [];
    const select = byId("system-profile-select");
    select.replaceChildren();
    state.systemProfiles.forEach((profile) => {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = profile.display_name;
      select.appendChild(option);
    });
    if (result.selected_profile_id) select.value = result.selected_profile_id;
    if (state.runtime) renderState(state.runtime);
  } catch (error) {
    byId("hardware-binding-status").textContent = error.message;
  }
}

async function loadHardware(profileId, selectProfile) {
  if (!profileId || state.hardwareBusy || !state.runtime?.session?.logged_in) return;
  state.hardwareBusy = true;
  state.hardwareReady = false;
  byId("hardware-binding-status").textContent = "正在枚举本机硬件…";
  try {
    if (selectProfile) {
      const runtime = await api("/api/device/profile", { method: "POST", body: JSON.stringify({ profile_id: profileId }) });
      renderState(runtime);
    }
    const inventory = await api(`/api/device/inventory?profile_id=${encodeURIComponent(profileId)}`);
    state.hardwareInventory = inventory;
    state.hardwareProfileLoaded = profileId;
    byId("system-profile-select").value = profileId;
    renderHardwareInventory(inventory);
  } catch (error) {
    state.hardwareInventory = null;
    state.hardwareProfileLoaded = null;
    byId("hardware-fields").replaceChildren();
    byId("hardware-binding-status").textContent = error.message;
    notify(error.message, true);
  } finally {
    state.hardwareBusy = false;
    if (state.runtime) renderState(state.runtime);
  }
}

async function saveHardware() {
  const profile = profileById(byId("system-profile-select").value);
  if (!profile) return;
  const cameras = {};
  const grippers = {};
  for (const select of byId("hardware-fields").querySelectorAll("select")) {
    if (!select.value || select.selectedOptions[0]?.disabled) {
      notify("请为所有角色选择当前可用的硬件。", true);
      return;
    }
    (select.dataset.group === "cameras" ? cameras : grippers)[select.dataset.role] = select.value;
  }
  state.hardwareBusy = true;
  try {
    const result = await api("/api/device/bindings", {
      method: "POST",
      body: JSON.stringify({ profile_id: profile.id, profile_revision: profile.revision, cameras, grippers }),
    });
    renderState(result.state);
    state.hardwareBusy = false;
    await loadHardware(profile.id, false);
    notify("本机硬件配置已保存");
  } catch (error) {
    notify(error.message, true);
  } finally {
    state.hardwareBusy = false;
    if (state.runtime) renderState(state.runtime);
  }
}

const cameraRoles = ["right", "left", "head"];
const isSyntheticHead = (role) => role === "head" && !state.runtime?.config?.head_camera_configured;

function setCameraStatus(role, text, live) {
  const feed = document.querySelector(`[data-camera-role="${role}"]`);
  feed.classList.toggle("live", live);
  byId(`camera-${role}-status`).textContent = text;
}

function refreshCameraPreviews() {
  if (!state.runtime?.processes?.device?.running || !document.querySelector("#collection-view.active")) return;
  state.cameraRevision += 1;
  for (const role of cameraRoles) {
    byId(`camera-${role}-image`).src = `/api/cameras/${role}/stream.mjpg?v=${state.cameraRevision}`;
  }
  state.cameraStreaming = true;
}

function syncCameraPreviews() {
  const active = Boolean(state.runtime?.processes?.device?.running && document.querySelector("#collection-view.active"));
  statusBadge(byId("camera-monitor-badge"), active ? "实时" : "离线", active ? "good" : "neutral");
  if (active && !state.cameraStreaming) {
    refreshCameraPreviews();
  }
  if (!active) {
    state.cameraStreaming = false;
    for (const role of cameraRoles) {
      byId(`camera-${role}-image`).removeAttribute("src");
      setCameraStatus(role, isSyntheticHead(role) ? "占位" : "离线", false);
    }
  }
}

for (const role of cameraRoles) {
  const image = byId(`camera-${role}-image`);
  image.addEventListener("load", () => setCameraStatus(role, isSyntheticHead(role) ? "占位" : "实时", true));
  image.addEventListener("error", () => {
    setCameraStatus(role, isSyntheticHead(role) ? "占位" : "无画面", false);
    state.cameraStreaming = false;
    window.setTimeout(syncCameraPreviews, 1000);
  });
}

async function pollState() {
  if (state.stateBusy) return;
  state.stateBusy = true;
  try {
    const runtime = await api("/api/state");
    renderState(runtime);
    byId("server-status").textContent = "已连接";
    byId("server-dot").className = "status-dot good";
  } catch (error) {
    byId("server-status").textContent = "连接中断";
    byId("server-dot").className = "status-dot bad";
    renderOperationStatus(null);
  } finally { state.stateBusy = false; }
}

function appendLog(id, lines) {
  if (!lines.length) return;
  const node = byId(id);
  const atBottom = node.scrollTop + node.clientHeight >= node.scrollHeight - 20;
  node.textContent += stripAnsi(lines.join(""));
  if (node.textContent.length > 120000) node.textContent = node.textContent.slice(-100000);
  if (atBottom) node.scrollTop = node.scrollHeight;
}

async function pollLogs() {
  const query = new URLSearchParams(state.logOffsets).toString();
  try {
    const logs = await api(`/api/logs?${query}`);
    for (const name of ["device", "saver", "calibration", "upload"]) {
      appendLog(`${name}-log`, logs[name].lines);
      state.logOffsets[name] = logs[name].offset;
    }
  } catch (_error) { }
}

async function runAction(button, operation, successMessage) {
  const previous = button.disabled;
  button.disabled = true;
  try {
    const result = await operation();
    if (result?.session) renderState(result);
    if (successMessage) notify(successMessage);
    await pollState();
    return result;
  } catch (error) {
    notify(error.message, true);
    return null;
  } finally { button.disabled = previous; }
}

function renderTasks(tasks) {
  const body = byId("task-table-body");
  body.replaceChildren();
  state.selectedTaskIndex = null;
  byId("select-task-button").disabled = true;
  if (!tasks.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 3; cell.className = "empty"; cell.textContent = "没有匹配任务";
    return;
  }
  tasks.forEach((task) => {
    const row = body.insertRow();
    row.dataset.selectable = "true";
    [task.id ?? "—", task.status || "—", task.name].forEach((value) => { const cell = row.insertCell(); cell.textContent = value; });
    row.addEventListener("click", () => {
      body.querySelectorAll("tr").forEach((item) => item.classList.remove("selected"));
      row.classList.add("selected");
      state.selectedTaskIndex = task.index;
      byId("select-task-button").disabled = false;
    });
  });
}

function formatBytes(value) {
  let bytes = Number(value || 0);
  const units = ["B", "KB", "MB", "GB", "TB"];
  let unit = 0;
  while (bytes >= 1024 && unit < units.length - 1) { bytes /= 1024; unit += 1; }
  return `${bytes.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function progressPercent(progress) {
  if (!progress) return 0;
  const total = Number(progress.total_bytes || progress.total_files || 0);
  const complete = Number(progress.completed_bytes || progress.completed_files || 0);
  return total > 0 ? Math.min(100, Math.round(complete * 100 / total)) : 0;
}

function renderUpload(snapshot) {
  const stats = snapshot.stats || {};
  const performance = snapshot.performance || {};
  byId("upload-queued").textContent = stats.queued || 0;
  byId("upload-success").textContent = stats.success || 0;
  byId("upload-failed").textContent = stats.failed || 0;
  byId("performance-metric").textContent = `${Math.round(performance.cpu_percent || 0)}% / ${Math.round(performance.memory_percent || 0)}%`;
  byId("upload-data-dir").textContent = snapshot.data_dir || "—";
  byId("auto-upload-toggle").checked = Boolean(snapshot.auto_enabled);
  byId("upload-summary").textContent = `${stats.total || 0} 条记录 · ${stats.complete || 0} 条完整`;

  const policy = snapshot.policy || {};
  if (!state.settingsInitialized) {
    byId("protect-minutes-input").value = policy.protect_minutes ?? 10;
    byId("keep-latest-input").value = policy.keep_latest_count ?? 5;
    byId("max-retries-input").value = policy.max_retries ?? 5;
    byId("upload-workers-input").value = policy.upload_workers ?? 2;
    byId("multipart-threads-input").value = policy.multipart_threads ?? 4;
    state.settingsInitialized = true;
  }

  const current = snapshot.current_episode;
  const progress = snapshot.current_progress;
  const progressWrap = byId("upload-progress-wrap");
  progressWrap.hidden = !current;
  if (current) {
    const percent = progressPercent(progress);
    byId("upload-progress-name").textContent = current.split("/").pop();
    byId("upload-progress-label").textContent = `${percent}%`;
    byId("upload-progress").value = percent;
  }

  const body = byId("upload-table-body");
  body.replaceChildren();
  const episodes = [...(snapshot.episodes || [])].reverse();
  if (!episodes.length) {
    const row = body.insertRow(); const cell = row.insertCell();
    cell.colSpan = 5; cell.className = "empty"; cell.textContent = "暂无 Episode";
  } else {
    episodes.forEach((episode) => {
      const row = body.insertRow();
      const name = row.insertCell(); name.textContent = episode.name;
      const task = row.insertCell(); task.textContent = episode.task_name || "—";
      const status = row.insertCell();
      const statusText = document.createElement("span"); statusText.className = `status-text ${episode.status}`; statusText.textContent = episode.status; status.appendChild(statusText);
      const size = row.insertCell(); size.textContent = formatBytes(episode.size_bytes);
      const detail = row.insertCell();
      const percent = progressPercent(episode.progress);
      detail.textContent = episode.reason || (percent ? `${percent}%` : "—");
    });
  }
}

async function refreshUpload(force = false) {
  if (state.uploadBusy || !state.runtime?.session?.logged_in) return;
  state.uploadBusy = true;
  try {
    const snapshot = force
      ? await api("/api/upload/refresh", { method: "POST" })
      : await api("/api/upload");
    renderUpload(snapshot);
  } catch (error) { if (force) notify(error.message, true); }
  finally { state.uploadBusy = false; }
}

document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
document.querySelectorAll("[data-clear-log]").forEach((button) => button.addEventListener("click", () => { byId(button.dataset.clearLog).textContent = ""; }));
byId("system-profile-select").addEventListener("change", (event) => loadHardware(event.currentTarget.value, true));
byId("refresh-hardware-button").addEventListener("click", () => loadHardware(byId("system-profile-select").value, false));
byId("save-hardware-button").addEventListener("click", saveHardware);

byId("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = byId("login-button");
  const result = await runAction(button, () => api("/api/session/login", { method: "POST", body: JSON.stringify({ phone: byId("phone-input").value.trim(), password: byId("password-input").value }) }), "DM3 登录成功");
  if (result) { byId("password-input").value = ""; }
});

byId("logout-button").addEventListener("click", async (event) => {
  const result = await runAction(event.currentTarget, () => api("/api/session/logout", { method: "POST" }), "已退出 DM3");
  if (result) { renderTasks([]); setView("login-view"); state.settingsInitialized = false; }
});

byId("task-filter-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = byId("fetch-task-button");
  await runAction(button, async () => {
    const query = new URLSearchParams({ page: byId("page-input").value, page_size: byId("page-size-input").value });
    if (byId("execute-day-input").value) query.set("execute_day", byId("execute-day-input").value);
    const tasks = await api(`/api/tasks?${query}`); renderTasks(tasks); return null;
  }, "任务列表已更新");
});

byId("select-task-button").addEventListener("click", async (event) => {
  if (state.selectedTaskIndex === null) return;
  const result = await runAction(event.currentTarget, () => api("/api/tasks/select", { method: "POST", body: JSON.stringify({ index: state.selectedTaskIndex }) }), "任务已选择");
  if (result) setView("collection-view");
});

const simpleAction = (id, path, message, confirmText = "") => byId(id).addEventListener("click", async (event) => {
  if (confirmText && !window.confirm(confirmText)) return;
  await runAction(event.currentTarget, () => api(path, { method: "POST" }), message);
});
byId("start-device-button").addEventListener("click", async (event) => {
  const profile = profileById(byId("system-profile-select").value);
  if (!profile || !state.hardwareReady) {
    notify("请先确认并保存当前系统方案的本机硬件。", true);
    return;
  }
  await runAction(
    event.currentTarget,
    () => api("/api/device/start", { method: "POST", body: JSON.stringify({ profile_id: profile.id, profile_revision: profile.revision }) }),
    "设备进程已启动",
  );
});
simpleAction("stop-device-button", "/api/device/stop", "设备进程已停止", "确定停止设备进程？");
simpleAction("start-saver-button", "/api/saver/start", "NEDF3 保存进程已启动");
simpleAction("stop-saver-button", "/api/saver/stop", "NEDF3 保存进程已停止");
simpleAction("toggle-episode-button", "/api/episode/toggle", "Episode 命令已发送");
simpleAction("discard-episode-button", "/api/episode/discard", "丢弃命令已发送", "确定丢弃当前 Episode？");
function episodesForTask(task) {
  return state.visualizerEpisodes.filter((episode) => episode.task === task);
}

function renderEpisodeOptions() {
  const task = byId("rerun-task-select").value;
  const episodeSelect = byId("rerun-episode-select");
  episodeSelect.replaceChildren();
  episodesForTask(task).forEach((episode) => {
    const option = document.createElement("option");
    option.value = episode.path;
    option.textContent = `${episode.name}  ·  ${new Date(episode.modified_at_ms).toLocaleString()}`;
    episodeSelect.appendChild(option);
  });
  if (episodeSelect.options.length) episodeSelect.selectedIndex = 0;
  updateEpisodeSelection();
}

function updateEpisodeSelection() {
  const episodeSelect = byId("rerun-episode-select");
  const selected = episodeSelect.value;
  byId("rerun-episode-path").textContent = selected || "该任务文件夹没有可用 Episode";
  byId("open-rerun-button").disabled = !selected;
}

async function openEpisodePicker() {
  const button = byId("visualize-episode-button");
  button.disabled = true;
  try {
    const result = await api("/api/visualizer/episodes");
    state.visualizerEpisodes = result.episodes || [];
    if (!state.visualizerEpisodes.length) {
      notify("数据目录中没有已完成的 NEDF3 Episode。", true);
      return;
    }
    const taskSelect = byId("rerun-task-select");
    taskSelect.replaceChildren();
    [...new Set(state.visualizerEpisodes.map((episode) => episode.task))].forEach((task) => {
      const option = document.createElement("option");
      option.value = task;
      option.textContent = task;
      taskSelect.appendChild(option);
    });
    renderEpisodeOptions();
    byId("episode-picker-dialog").showModal();
  } catch (error) {
    notify(error.message, true);
  } finally {
    button.disabled = !state.runtime?.controls?.can_visualize_episodes;
  }
}

byId("visualize-episode-button").addEventListener("click", openEpisodePicker);
byId("rerun-task-select").addEventListener("change", renderEpisodeOptions);
byId("rerun-episode-select").addEventListener("change", updateEpisodeSelection);
byId("rerun-episode-select").addEventListener("dblclick", () => byId("open-rerun-button").click());
byId("open-rerun-button").addEventListener("click", async (event) => {
  const episode = byId("rerun-episode-select").value;
  if (!episode) return;
  const result = await runAction(
    event.currentTarget,
    () => api("/api/visualizer/start", { method: "POST", body: JSON.stringify({ episode }) }),
    "正在 Rerun 中打开所选 Episode",
  );
  if (result) byId("episode-picker-dialog").close();
});
byId("calibrate-grippers-button").addEventListener("click", async (event) => {
  const profile = profileById(byId("system-profile-select").value);
  if (!profile || !state.hardwareReady) {
    notify("请先确认并保存当前系统方案的本机硬件。", true);
    return;
  }
  if (!window.confirm("请确认两侧夹爪周围无人和无障碍物，并确认设备与保存进程均已停止。开始标定？")) return;
  await runAction(
    event.currentTarget,
    () => api("/api/grippers/calibrate", { method: "POST", body: JSON.stringify({ profile_id: profile.id, profile_revision: profile.revision }) }),
    "夹爪标定进程已启动",
  );
});

byId("auto-upload-toggle").addEventListener("change", async (event) => {
  const enabled = event.currentTarget.checked;
  try { await api("/api/upload/auto", { method: "POST", body: JSON.stringify({ enabled }) }); notify(enabled ? "自动上传已启动" : "自动上传已停止"); }
  catch (error) { event.currentTarget.checked = !enabled; notify(error.message, true); }
});
byId("refresh-upload-button").addEventListener("click", () => refreshUpload(true));

async function enqueue(mode, confirmation = "", message = "已完成入队") {
  try {
    const result = await api("/api/upload/enqueue", { method: "POST", body: JSON.stringify({ mode, confirmation }) });
    notify(`${message}：${result.queued} 条`); await refreshUpload();
  } catch (error) { notify(error.message, true); }
}
byId("enqueue-eligible-button").addEventListener("click", () => enqueue("eligible"));
byId("retry-failed-button").addEventListener("click", () => enqueue("failed", "", "失败项重新入队"));
byId("enqueue-all-button").addEventListener("click", () => {
  if (window.confirm("确定绕过保护规则，将所有未成功上传的完整 Episode 入队？")) enqueue("all", "UPLOAD_ALL", "全部 Episode 已入队");
});

byId("upload-settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  const payload = {
    protect_minutes: Number(byId("protect-minutes-input").value),
    keep_latest_count: Number(byId("keep-latest-input").value),
    max_retries: Number(byId("max-retries-input").value),
    upload_workers: Number(byId("upload-workers-input").value),
    multipart_threads: Number(byId("multipart-threads-input").value),
  };
  await runAction(button, async () => { const snapshot = await api("/api/upload/settings", { method: "POST", body: JSON.stringify(payload) }); renderUpload(snapshot); return null; }, "上传设置已保存");
});

const requestedView = `${window.location.hash.slice(1)}-view`;
setView(titles[requestedView] ? requestedView : "login-view");
loadProfiles(); pollState(); pollLogs();
window.setInterval(pollState, 700);
window.setInterval(pollLogs, 800);
window.setInterval(() => { if (document.querySelector("#upload-view.active")) refreshUpload(); }, 3000);
