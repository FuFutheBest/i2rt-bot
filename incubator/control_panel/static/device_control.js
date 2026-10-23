(() => {
  "use strict";

  const STATUS_LABELS = {
    stopped: "已停止",
    starting: "启动中",
    running: "运行中",
    degraded: "数据异常",
    stopping: "停止中",
    stop_timeout: "安全停机超时",
    failed: "启动失败",
    orphaned: "外部占用",
    unknown: "未知",
  };

  const ACTIVE_STATUSES = new Set(["starting", "running", "degraded", "stopping", "stop_timeout"]);
  const TRANSITION_STATUSES = new Set(["starting", "stopping"]);
  const ERROR_STATUSES = new Set(["failed", "degraded", "orphaned", "stop_timeout"]);
  const MAX_LOG_LINES = 600;
  const CHECKPOINT_BROWSER_PATH_MAX_LENGTH = 4096;
  const HARDWARE_DISCOVERY_URL = "/api/hardware/discovery";
  const CAMERA_BINDING_CLASSES = new Set(["webcameradevice", "webcameraopenpi"]);
  const CAMERA_SLOT_ORDER = ["head", "left", "right"];
  const ROBOT_SLOT_ORDER = ["rizon", "i2rt_head"];
  const DEVICE_PROGRESS_LABELS = Object.freeze({
    queued: "等待执行",
    spawning_worker: "创建进程",
    worker_bootstrap: "启动进程",
    constructing_device: "加载设备",
    initializing_device: "初始化设备",
    creating_robot_shm: "创建数据通道",
    checking_shm: "检查数据通道",
    claiming_shm: "建立数据通道",
    rizon_selecting_rdk: "匹配 RDK",
    rizon_connecting: "连接 Rizon",
    rizon_configuring_control: "配置控制模式",
    rizon_resolving_home: "读取 Home",
    rizon_moving_home: "前往 Home",
    rizon_starting_stream: "启动数据流",
    rizon_stopping: "断连 Rizon",
    stopping_robot: "停止数据流",
    yam_creating_shm: "创建数据通道",
    yam_connecting: "连接 Yam",
    yam_moving_init_pose: "前往 Init Pose",
    yam_waiting_tracking: "等待跟随输入",
    yam_waiting_head_shm: "等待头部数据",
    yam_starting_publisher: "启动控制发布",
    yam_moving_zero: "回到 Joint Zero",
    yam_confirming_zero: "确认 Joint Zero",
    yam_disconnecting: "断连 Yam",
    yam_zero_hold: "保持零位",
    stop_requested: "停止请求已接收",
    force_stop_requested: "强制关闭请求已接收",
    start_failed: "启动失败",
    worker_failed: "进程异常",
    ready: "已就绪",
    stopped: "已停止",
    failed: "操作失败",
  });
  const DEVICE_START_PROGRESS_PHASES = new Set([
    "queued",
    "spawning_worker",
    "worker_bootstrap",
    "constructing_device",
    "initializing_device",
    "creating_robot_shm",
    "checking_shm",
    "claiming_shm",
    "rizon_selecting_rdk",
    "rizon_connecting",
    "rizon_configuring_control",
    "rizon_resolving_home",
    "rizon_moving_home",
    "rizon_starting_stream",
    "yam_creating_shm",
    "yam_connecting",
    "yam_moving_init_pose",
    "yam_waiting_tracking",
    "yam_waiting_head_shm",
    "yam_starting_publisher",
    "ready",
  ]);
  const DEVICE_STOP_PROGRESS_PHASES = new Set([
    "stop_requested",
    "force_stop_requested",
    "rizon_stopping",
    "stopping_robot",
    "yam_moving_zero",
    "yam_confirming_zero",
    "yam_disconnecting",
    "yam_zero_hold",
    "stopped",
  ]);
  const CAMERA_SLOT_LABELS = {
    head: "头部相机",
    left: "左腕相机",
    right: "右腕相机",
  };
  const WORKSPACE_SPLIT_STORAGE_KEY = "superinference.controlRailWidth";
  const WORKSPACE_SPLIT_MIN_PX = 280;
  const WORKSPACE_VISUAL_MIN_PX = 560;
  const POLL_INTERVALS = Object.freeze({
    statePending: 200,
    stateRecovery: 250,
    stateActiveFast: 500,
    stateActive: 1000,
    stateIdleFast: 1000,
    stateIdle: 2500,
    stateWatchdog: 6000,
    stateRequestTimeout: 2000,
    controlRecoveryTimeout: 3000,
    controlRecoveryRetry: 150,
    preview: 2500,
    previewFrame: 200,
    previewRequestTimeout: 1500,
    logs: 1000,
    recordingClock: 250,
    diagnostics: 2000,
  });
  const MAX_CONCURRENT_PREVIEW_FRAMES = 2;
  const PREVIEW_FRAME_BACKPRESSURE_MS = 60;
  const PREVIEW_BROWSER_CLIENT_VERSION = "20260722-short-preview5";
  const WORKFLOW_STATUS_LABELS = {
    stopped: "已停止",
    starting: "启动中",
    running: "运行中",
    stopping: "停止中",
    failed: "异常",
  };
  const FASTWAM_SERVER_PROGRESS_LABELS = Object.freeze({
    spawning_process: "启动模型进程中",
    validating_config: "校验配置中",
    resolving_checkpoint: "解析 checkpoint 中",
    validating_contract: "校验模型契约中",
    building_model: "构建 FastWAM 中",
    loading_checkpoint: "加载模型权重中",
    loading_text_encoder: "加载文本编码器中",
    finalizing_model: "整理 GPU 内存中",
    compiling_model: "编译推理热路径中",
    warming_model: "预热 FastWAM 中",
    binding_server: "启动服务监听中",
    ready: "模型 Server 已就绪",
    stopping: "停止模型 Server 中",
    stopped: "FastWAM Server 已停止",
    failed: "模型 Server 启动失败",
  });
  const RECORDING_STATUS_LABELS = {
    disabled: "未配置",
    stopped: "未启动",
    starting: "准备中",
    ready: "已就绪",
    waiting: "等待首条 action",
    recording: "录制中",
    finalizing: "正在保存",
    transitioning: "正在回 Home",
    error: "录制异常",
  };
  const RECORDING_ACTIONS = [
    "episode_accept",
    "episode_discard",
    "segment_accept",
    "segment_reject",
  ];
  const RECORDING_ACTION_SHORTCUTS = Object.freeze({
    episode_accept: Object.freeze({ code: "KeyQ", key: "Q" }),
    episode_discard: Object.freeze({ code: "KeyW", key: "W" }),
    segment_accept: Object.freeze({ code: "KeyA", key: "A" }),
    segment_reject: Object.freeze({ code: "KeyS", key: "S" }),
  });
  const RECORDING_CAMERA_ROLES = ["head", "left", "right"];
  const RECORDING_CAMERA_STATUS_LABELS = {
    healthy: "正常",
    configured: "已配置",
    missing: "缺失",
    black_placeholder: "黑帧占位",
    unknown: "未上报",
  };
  const GRIPPER_SIDES = ["left", "right"];
  const GRIPPER_CALIBRATION_SYNC_TIMEOUT_MS = 4 * 60 * 1000;
  const GRIPPER_POSITION_SYNC_TIMEOUT_MS = 30 * 1000;
  const GRIPPER_PROBE_SYNC_TIMEOUT_MS = 20 * 1000;
  const GRIPPER_TARGET_COMMAND_INTERVAL_MS = 200;
  const GRIPPER_TARGET_RETRY_DELAY_MS = 300;
  const GRIPPER_TARGET_SESSION_TIMEOUT_MS = 60 * 1000;
  const GRIPPER_TARGET_MAX_ATTEMPTS = 8;
  const GRIPPER_TARGET_TOLERANCE_PERCENT = 1;
  const GRIPPER_STATUS_LABELS = {
    unavailable: "不可用",
    idle: "就绪",
    calibrating: "标定中",
    moving: "移动中",
    queued: "等待中",
    probing: "读取中",
    error: "异常",
  };
  const GRIPPER_ARTIFACT_STATUS_LABELS = {
    valid: "标定有效",
    missing: "标定文件缺失",
    invalid: "标定文件无效",
    stale: "标定身份过期",
    unverified: "标定身份未核验",
  };

  const elements = {
    mainContent: document.getElementById("main-content"),
    workspaceSplitter: document.getElementById("workspace-splitter"),
    serverStateDot: document.getElementById("server-state-dot"),
    serverStateText: document.getElementById("server-state-text"),
    configSelect: document.getElementById("config-select"),
    configDetail: document.getElementById("config-detail"),
    checkpointSourceBadge: document.getElementById("checkpoint-source-badge"),
    checkpointDefaultPath: document.getElementById("checkpoint-default-path"),
    checkpointOverridePath: document.getElementById("checkpoint-override-path"),
    checkpointEffectivePath: document.getElementById("checkpoint-effective-path"),
    checkpointBrowseButton: document.getElementById("checkpoint-browse-button"),
    checkpointClearButton: document.getElementById("checkpoint-clear-button"),
    checkpointStatus: document.getElementById("checkpoint-status"),
    testInferenceToggle: document.getElementById("test-inference-toggle"),
    testInferenceStatus: document.getElementById("test-inference-status"),
    checkpointBrowserDialog: document.getElementById("checkpoint-browser-dialog"),
    checkpointBrowserForm: document.getElementById("checkpoint-browser-form"),
    checkpointBrowserEyebrow: document.getElementById("checkpoint-browser-eyebrow"),
    checkpointBrowserTitle: document.getElementById("checkpoint-browser-title"),
    checkpointBrowserHint: document.getElementById("checkpoint-browser-hint"),
    checkpointBrowserRefresh: document.getElementById("checkpoint-browser-refresh"),
    checkpointBrowserUp: document.getElementById("checkpoint-browser-up"),
    checkpointBrowserGo: document.getElementById("checkpoint-browser-go"),
    checkpointBrowserRoots: document.getElementById("checkpoint-browser-roots"),
    checkpointBrowserPath: document.getElementById("checkpoint-browser-path"),
    checkpointBrowserList: document.getElementById("checkpoint-browser-list"),
    checkpointBrowserStatus: document.getElementById("checkpoint-browser-status"),
    checkpointBrowserError: document.getElementById("checkpoint-browser-error"),
    checkpointBrowserCancel: document.getElementById("checkpoint-browser-cancel"),
    checkpointBrowserSelect: document.getElementById("checkpoint-browser-select"),
    globalNotice: document.getElementById("global-notice"),
    metricConfigured: document.getElementById("metric-configured"),
    metricRunning: document.getElementById("metric-running"),
    metricHealthy: document.getElementById("metric-healthy"),
    metricErrors: document.getElementById("metric-errors"),
    runtimeToggle: document.getElementById("runtime-toggle"),
    runtimeStatus: document.getElementById("runtime-status"),
    serverControl: document.getElementById("server-control"),
    serverToggle: document.getElementById("server-toggle"),
    serverStatus: document.getElementById("server-status"),
    inferenceToggle: document.getElementById("inference-toggle"),
    inferenceStatus: document.getElementById("inference-status"),
    workflowBlockers: document.getElementById("workflow-blockers"),
    workflowProfile: document.getElementById("workflow-profile"),
    preflightRefresh: document.getElementById("preflight-refresh"),
    runtimeMetricsRefresh: document.getElementById("runtime-metrics-refresh"),
    preflightStatus: document.getElementById("preflight-status"),
    preflightSummary: document.getElementById("preflight-summary"),
    preflightChecks: document.getElementById("preflight-checks"),
    debugPolicyFps: document.getElementById("debug-policy-fps"),
    debugControlFps: document.getElementById("debug-control-fps"),
    debugActionManager: document.getElementById("debug-action-manager"),
    debugActionFps: document.getElementById("debug-action-fps"),
    debugDeviceFps: document.getElementById("debug-device-fps"),
    debugActionQueue: document.getElementById("debug-action-queue"),
    runtimeTuningControls: document.getElementById("runtime-tuning-controls"),
    generatedConfigId: document.getElementById("generated-config-id"),
    generatedConfigExport: document.getElementById("generated-config-export"),
    currentConfigSave: document.getElementById("current-config-save"),
    generatedConfigStatus: document.getElementById("generated-config-status"),
    forceStopAll: document.getElementById("force-stop-all"),
    forceStopHint: document.getElementById("force-stop-hint"),
    recordingActionStatus: document.getElementById("recording-action-status"),
    recordingActionHint: document.getElementById("recording-action-hint"),
    recordingActionGrid: document.getElementById("recording-action-grid"),
    standaloneResetControl: document.getElementById("standalone-reset-control"),
    standalonePauseButton: document.getElementById("standalone-pause-button"),
    standaloneResetButton: document.getElementById("standalone-reset-button"),
    rolloutToggle: document.getElementById("rollout-toggle"),
    rolloutOutputPath: document.getElementById("rollout-output-path"),
    rolloutPathButton: document.getElementById("rollout-path-button"),
    rolloutStorageBudget: document.getElementById("rollout-storage-budget"),
    rolloutStorageTitle: document.getElementById("rollout-storage-title"),
    rolloutStorageBadge: document.getElementById("rollout-storage-badge"),
    rolloutStorageFree: document.getElementById("rollout-storage-free"),
    rolloutStorageRemaining: document.getElementById("rollout-storage-remaining"),
    rolloutStorageBar: document.getElementById("rollout-storage-bar"),
    rolloutStorageDetail: document.getElementById("rollout-storage-detail"),
    recordingActionButtons: typeof document.querySelectorAll === "function"
      ? [...document.querySelectorAll("[data-recording-action]")]
      : [],
    recordingStateBadge: document.getElementById("recording-state-badge"),
    recordingEpisodeLabel: document.getElementById("recording-episode-label"),
    recordingFrameCount: document.getElementById("recording-frame-count"),
    recordingElapsed: document.getElementById("recording-elapsed"),
    recordingTimelineEmpty: document.getElementById("recording-timeline-empty"),
    recordingTimelineContent: document.getElementById("recording-timeline-content"),
    recordingScaleEnd: document.getElementById("recording-scale-end"),
    recordingTrack: document.getElementById("recording-track"),
    recordingTrackSegments: document.getElementById("recording-track-segments"),
    recordingTrackCheckpoints: document.getElementById("recording-track-checkpoints"),
    recordingTrackCursor: document.getElementById("recording-track-cursor"),
    recordingReference: document.getElementById("recording-reference"),
    recordingTimelineError: document.getElementById("recording-timeline-error"),
    recordingCameraRoles: typeof document.querySelectorAll === "function"
      ? [...document.querySelectorAll("[data-camera-role]")]
      : [],
    robotLeftBindingSelect: document.getElementById("robot-left-binding-select"),
    robotRightBindingSelect: document.getElementById("robot-right-binding-select"),
    hardwareDiscoveryRefresh: document.getElementById("hardware-discovery-refresh"),
    robotDeviceToggle: document.getElementById("robot-device-toggle"),
    rizonDeviceRequirement: document.getElementById("rizon-device-requirement"),
    i2rtHeadDeviceToggle: document.getElementById("i2rt-head-device-toggle"),
    i2rtHeadDeviceRequirement: document.getElementById("i2rt-head-device-requirement"),
    gripperCalibrateAll: document.getElementById("gripper-calibrate-all"),
    gripperAuthorizePermissions: document.getElementById("gripper-authorize-permissions"),
    gripperCheckReason: document.getElementById("gripper-check-reason"),
    gripperPermissionCsrf: document.querySelector('meta[name="gripper-permission-csrf"]'),
    gripperPasswordTransport: document.querySelector('meta[name="gripper-password-transport"]'),
    gripperPermissionDialog: document.getElementById("gripper-permission-dialog"),
    gripperPermissionForm: document.getElementById("gripper-permission-form"),
    gripperPermissionSummary: document.getElementById("gripper-permission-summary"),
    gripperPermissionWarning: document.getElementById("gripper-permission-warning"),
    gripperPasswordField: document.getElementById("gripper-password-field"),
    gripperSudoPassword: document.getElementById("gripper-sudo-password"),
    gripperPermissionError: document.getElementById("gripper-permission-error"),
    gripperPermissionCancel: document.getElementById("gripper-permission-cancel"),
    gripperPermissionSubmit: document.getElementById("gripper-permission-submit"),
    grippers: Object.fromEntries(GRIPPER_SIDES.map((side) => [side, {
      status: document.getElementById(`gripper-${side}-status`),
      port: document.getElementById(`gripper-${side}-port`),
      position: document.getElementById(`gripper-${side}-position`),
      value: document.getElementById(`gripper-${side}-value`),
      minLabel: document.getElementById(`gripper-${side}-min-label`),
      maxLabel: document.getElementById(`gripper-${side}-max-label`),
    }])),
    cameraGrid: document.getElementById("camera-grid"),
    logDetails: document.querySelector(".rail-logs"),
    logDeviceSelect: document.getElementById("log-device-select"),
    logAutoscroll: document.getElementById("log-autoscroll"),
    clearLogsButton: document.getElementById("clear-logs-button"),
    logView: document.getElementById("log-view"),
    logEmpty: document.getElementById("log-empty"),
    confirmDialog: document.getElementById("confirm-dialog"),
    confirmTitle: document.getElementById("confirm-title"),
    confirmMessage: document.getElementById("confirm-message"),
    confirmActionButton: document.getElementById("confirm-action-button"),
    toastRegion: document.getElementById("toast-region"),
  };

  const app = {
    configs: [],
    recommendedConfig: "",
    state: {},
    devices: [],
    selectedConfig: "",
    activeConfig: "",
    configSelectionBeforeOpen: "",
    configSwitching: false,
    configSelectionDeferred: false,
    configOptionsDeferred: false,
    configRefreshPromise: null,
    configDetailRequest: 0,
    preflight: null,
    runtimeMetrics: null,
    runtimeTuning: null,
    runtimeTuningDraft: new Map(),
    generatedConfigSource: "",
    generatedConfigPending: false,
    currentConfigSavePending: false,
    diagnosticsLoading: false,
    diagnosticsRequest: 0,
    preflightRequest: 0,
    runtimeMetricsRequest: 0,
    preflightController: null,
    runtimeMetricsController: null,
    diagnosticsController: null,
    checkpointMutationPending: false,
    checkpointMutationSerial: 0,
    checkpointMutationController: null,
    checkpointMutationContext: null,
    checkpointOptimistic: null,
    checkpointBrowserRequest: 0,
    checkpointBrowserController: null,
    checkpointBrowserContext: null,
    checkpointBrowserPayload: null,
    checkpointBrowserMode: "checkpoint",
    rolloutBrowserEnableOnSelect: false,
    inferenceOptionsMutationPending: false,
    inferenceOptionsMutationSerial: 0,
    inferenceOptionsMutationController: null,
    inferenceOptionsMutationContext: null,
    rolloutStorageAlertStatus: "disabled",
    pendingActions: new Set(),
    cameraCards: new Map(),
    cameraSlots: Object.fromEntries(CAMERA_SLOT_ORDER.map((role) => [role, {
      role,
      deviceKey: "",
      configured: false,
      required: false,
      inferenceRequired: false,
      recordingConfigured: false,
      source: "unconfigured",
    }])),
    robotSlots: Object.fromEntries(ROBOT_SLOT_ORDER.map((role) => [role, {
      role,
      deviceKey: "",
      configured: false,
      required: false,
      inferenceRequired: false,
      source: "unconfigured",
    }])),
    previewStatuses: new Map(),
    previewRetryAt: new Map(),
    previewStatusRequests: new Map(),
    previewFrameTimers: new Map(),
    previewFrameInFlight: new Set(),
    cameraVisibility: new Map(),
    cameraObserver: null,
    workspaceSplitWidth: null,
    workspaceSplitPointerId: null,
    logCursor: null,
    logKey: "",
    logItems: [],
    logFingerprints: new Set(),
    refreshing: false,
    stateRefreshPromise: null,
    stateRequestController: null,
    stateRequestSerial: 0,
    stateAppliedRequestSerial: 0,
    stateEpoch: "",
    stateSequence: -1,
    stateEventRevision: -1,
    stateObservedRevision: -1,
    stateStablePolls: 0,
    stateTransportFailed: true,
    stateTransportReason: "等待首个真实设备状态",
    stateTransportDeadlineAtMs: 0,
    stateEvents: null,
    stateEventsConnected: false,
    logLoading: false,
    logRequestController: null,
    hardwareLoading: false,
    hardwareRequest: 0,
    discovery: {
      revision: null,
      robots: [],
      cameras: [],
      recommendedOverrides: {},
    },
    configDevices: [],
    bindingConfigName: "",
    bindingDirty: false,
    bindingDirtyKeys: new Set(),
    bindingPending: new Set(),
    bindingRequests: new Map(),
    bindingDiscoveryRefreshPromise: null,
    bindingErrors: new Map(),
    controlFailureNotice: "",
    failureLogKey: "",
    workflowPending: new Set(),
    forceStopPending: false,
    controlTransitions: new Map(),
    controlRequests: new Map(),
    gripperCalibratePending: false,
    gripperMovePending: new Set(),
    gripperQueuedTargets: new Map(),
    gripperDrafts: new Map(),
    gripperDesiredTargets: new Map(),
    gripperTargetSessions: new Map(),
    gripperTargetTimers: new Map(),
    gripperEditing: new Set(),
    gripperCalibrationSync: null,
    gripperCalibrationSliderValues: new Map(),
    gripperActualValues: new Map(),
    gripperPositionSync: new Map(),
    gripperProbeSync: new Map(),
    gripperProbeQueued: new Set(),
    gripperProbePending: new Set(),
    gripperProbePromise: null,
    gripperSyncEpoch: 0,
    gripperSnapshotRequest: 0,
    gripperSnapshotRevision: null,
    gripperSideSnapshotRevisions: new Map(),
    gripperSideSnapshotRequests: new Map(),
    gripperTopSnapshotRequest: 0,
    gripperErrors: new Map(),
    gripperPortPending: new Set(),
    gripperPortDrafts: new Map(),
    gripperPermissionPending: false,
    gripperPermissionMode: "sudo",
    gripperPermissionPromptPromise: null,
    gripperPermissionPromptResolve: null,
    gripperPermissionPromptSignature: "",
    gripperPermissionContext: null,
    gripperPermissionDismissedSignature: "",
    gripperPermissionAutoScheduled: false,
    recordingActionPending: "",
    inferenceControlPending: "",
    recordingObservedAtMs: 0,
    robotBindings: { left: "", right: "" },
    cameraBindings: new Map(),
    initialStateObserved: false,
    bootstrapAttempted: false,
    bootstrapFailureNotice: "",
    bootstrapManualChoiceRequired: false,
    pollTimers: {
      state: null,
      preview: null,
      logs: null,
      recordingClock: null,
      diagnostics: null,
    },
    renderKeys: {
      cameraOrder: "",
      logDevices: "",
    },
    perf: {
      startedAtMs: Date.now(),
      requests: 0,
      requestFailures: 0,
      requestAborts: 0,
      responseBytes: 0,
      statePolls: 0,
      previewPolls: 0,
      logPolls: 0,
      overlappingPollsSkipped: 0,
      optionReplacements: 0,
      cameraCardReorders: 0,
      timelineNodesCreated: 0,
      timelineNodesRemoved: 0,
      logRowsCreated: 0,
    },
  };

  function countPerf(name, amount = 1) {
    if (Object.prototype.hasOwnProperty.call(app.perf, name)) app.perf[name] += amount;
  }

  globalThis.__controlPanelPerf = {
    snapshot() {
      return Object.freeze({
        ...app.perf,
        uptimeMs: Math.max(0, Date.now() - app.perf.startedAtMs),
        hidden: Boolean(document.hidden),
        cameraStreams: [...app.cameraCards.values()].filter((card) => (
          card.querySelector("img")?.dataset.streaming === "1"
        )).length,
      });
    },
    reset() {
      const startedAtMs = Date.now();
      for (const key of Object.keys(app.perf)) {
        app.perf[key] = key === "startedAtMs" ? startedAtMs : 0;
      }
    },
  };

  class ApiError extends Error {
    constructor(message, status, payload) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.payload = payload;
    }
  }

  async function api(path, options = {}) {
    const request = { ...options };
    request.headers = { Accept: "application/json", ...(options.headers || {}) };
    if (request.body && !(request.body instanceof FormData)) {
      request.headers["Content-Type"] = "application/json";
    }

    let response;
    countPerf("requests");
    try {
      response = await fetch(path, request);
    } catch (error) {
      if (error?.name === "AbortError") {
        countPerf("requestAborts");
        throw error;
      }
      countPerf("requestFailures");
      throw new ApiError("无法连接控制服务", 0, error);
    }

    const contentLength = Number(response.headers.get("content-length"));
    if (Number.isFinite(contentLength) && contentLength > 0) {
      countPerf("responseBytes", contentLength);
    }

    const contentType = response.headers.get("content-type") || "";
    let payload = null;
    try {
      payload = contentType.includes("application/json")
        ? await response.json()
        : await response.text();
    } catch (_error) {
      payload = null;
    }

    if (!response.ok) {
      countPerf("requestFailures");
      const message =
        (payload && typeof payload === "object" && (payload.error || payload.message || payload.detail)) ||
        (typeof payload === "string" && payload.trim()) ||
        `请求失败（HTTP ${response.status}）`;
      throw new ApiError(String(message), response.status, payload);
    }
    return payload;
  }

  function gripperPasswordTransportAllowed() {
    return elements.gripperPasswordTransport?.content === "allowed";
  }

  function gripperPermissionCsrfToken() {
    return String(elements.gripperPermissionCsrf?.content || "");
  }

  function gripperPermissionErrorCode(error) {
    if (!(error instanceof ApiError) || !error.payload || typeof error.payload !== "object") {
      return "";
    }
    return String(error.payload.code || "").trim();
  }

  function isGripperPermissionRequiredError(error) {
    return [
      "ttyusb_permission_required",
      "can_interface_permission_required",
    ].includes(gripperPermissionErrorCode(error));
  }

  async function withGripperPermissionRetry(operation) {
    const authorize = typeof arguments[1] === "function" ? arguments[1] : () => true;
    if (!authorize()) return null;
    try {
      return await operation();
    } catch (error) {
      if (!isGripperPermissionRequiredError(error)) throw error;
      const authorized = await requestGripperPermission({ force: true });
      if (!authorized || !authorize()) return null;
      return operation();
    }
  }

  function firstDefined(...values) {
    return values.find((value) => value !== undefined && value !== null);
  }

  function asNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function normalizeConfigName(value) {
    if (!value) return "";
    if (typeof value === "string") return value.replace(/\.ya?ml$/i, "");
    return normalizeConfigName(firstDefined(value.name, value.config_name, value.id));
  }

  function extractConfigs(payload) {
    const source = Array.isArray(payload)
      ? payload
      : firstDefined(payload?.items, payload?.configs, payload?.data, []);
    if (!Array.isArray(source)) return [];

    const seen = new Set();
    return source
      .map((item) => {
        const name = normalizeConfigName(item);
        if (!name || seen.has(name)) return null;
        seen.add(name);
        return {
          name,
          label: typeof item === "object" ? firstDefined(item.label, item.title, name) : name,
          valid: typeof item === "object" ? item.valid !== false : true,
          error: typeof item === "object" ? firstDefined(item.error, item.validation_error, "") : "",
          default: typeof item === "object" && item.default === true,
          workflowKind: typeof item === "object" ? firstDefined(item.workflow_kind, "") : "",
          policyClass: typeof item === "object" ? firstDefined(item.policy_class, "") : "",
          hardwareRequired: typeof item === "object" && item.hardware_required === true,
          status: typeof item === "object" ? firstDefined(item.status, "") : "",
          controlPanelSupported: typeof item !== "object" || item.control_panel_supported !== false,
          supportReason: typeof item === "object" ? firstDefined(item.support_reason, "") : "",
        };
      })
      .filter(Boolean)
      .sort((a, b) => a.name.localeCompare(b.name, "zh-CN", { numeric: true }));
  }

  function stableConfigOrder(configs) {
    return configs
      .map((config, index) => ({ config, index }))
      .sort((left, right) => (
        left.config.name.localeCompare(right.config.name, "en", {
          numeric: true,
          sensitivity: "base",
        }) || left.index - right.index
      ))
      .map(({ config }) => config);
  }

  function chooseBootstrapConfig(configs, recommendedConfig = "") {
    const eligible = stableConfigOrder(configs.filter((config) => config.valid !== false));
    if (!eligible.length) return "";
    const byName = new Map(eligible.map((config) => [config.name, config]));
    const recommended = normalizeConfigName(recommendedConfig);
    // The server recommendation is derived from the same Hydra default used
    // by device_starter. Do not embed a Control Panel-specific config name.
    if (recommended && byName.has(recommended)) return recommended;
    const dualPi05 = eligible.find((config) => {
      const name = config.name.toLowerCase();
      return name.includes("dual") && name.includes("pi05");
    });
    if (dualPi05) return dualPi05.name;
    const markedDefault = eligible.find((config) => config.default === true);
    return markedDefault?.name || eligible[0].name;
  }

  function normalizeStatus(raw) {
    const processInfo = raw?.process && typeof raw.process === "object" ? raw.process : {};
    const lifecycleCandidate = String(
      firstDefined(raw?.lifecycle_status, processInfo.lifecycle_status, "")
    ).toLowerCase().trim();
    const candidate = String(
      firstDefined(raw?.status, raw?.state, processInfo.status, processInfo.state, "")
    ).toLowerCase().trim();

    const aliases = {
      alive: "running",
      active: "running",
      up: "running",
      inactive: "stopped",
      down: "stopped",
      exited: "failed",
      error: "failed",
      dead: "failed",
      external: "orphaned",
      "stop-timeout": "stop_timeout",
      "stop timeout": "stop_timeout",
      safety_stop_timeout: "stop_timeout",
    };
    // A safety-stop timeout means the worker may still be alive. Preserve the
    // dangerous lifecycle state even when a legacy payload also says running.
    const lifecycleStatus = aliases[lifecycleCandidate] || lifecycleCandidate;
    if (lifecycleStatus === "stop_timeout") return "stop_timeout";
    const status = aliases[candidate] || candidate;
    if (STATUS_LABELS[status]) return status;

    const running = firstDefined(raw?.running, processInfo.running, processInfo.alive);
    if (running === true) return "running";
    if (running === false) return "stopped";
    return "stopped";
  }

  function timestampAgeMs(timestamp) {
    const value = asNumber(timestamp);
    if (value === null || value <= 0) return null;
    let epochMs = value;
    if (value > 1e17) epochMs = value / 1e6;
    else if (value > 1e14) epochMs = value / 1e6;
    else if (value > 1e11) epochMs = value;
    else epochMs = value * 1000;
    return Math.max(0, Date.now() - epochMs);
  }

  function normalizeShm(raw, className, id, fps) {
    const shm = raw?.shm && typeof raw.shm === "object" ? raw.shm : {};
    const name = String(
      firstDefined(shm.name, raw?.shm_name, raw?.shared_memory_name, `${className}_${id}_data`)
    );
    const connected = Boolean(
      firstDefined(shm.connected, shm.exists, shm.available, raw?.shm_connected, raw?.shm_exists, false)
    );
    const ageMs = firstDefined(
      asNumber(shm.age_ms),
      asNumber(raw?.shm_age_ms),
      timestampAgeMs(firstDefined(shm.last_timestamp_ns, shm.timestamp_ns, raw?.last_timestamp_ns))
    );
    const explicitStatus = String(firstDefined(shm.status, raw?.shm_status, "")).toLowerCase();
    const freshnessLimit = Math.max(2000, fps ? 5000 / fps : 2000);
    let status = explicitStatus;
    if (!status) {
      if (!connected) status = "missing";
      else if (ageMs !== null && ageMs > freshnessLimit) status = "stale";
      else status = "healthy";
    }
    const healthy = firstDefined(shm.healthy, raw?.shm_healthy, status === "healthy") === true;
    return {
      name,
      connected,
      healthy,
      status,
      ageMs,
      fps: firstDefined(asNumber(shm.fps), asNumber(raw?.actual_fps), fps),
      shape: firstDefined(shm.shape, raw?.shape, raw?.data_shape),
      dtype: firstDefined(shm.dtype, raw?.dtype, raw?.data_dtype),
    };
  }

  function hasPreviewCapability(raw, className) {
    const capabilities = raw?.capabilities;
    if (Array.isArray(capabilities) && capabilities.includes("preview")) return true;
    if (capabilities && typeof capabilities === "object" && capabilities.preview === true) return true;
    if (raw?.is_camera === true || raw?.camera === true || raw?.preview === true) return true;
    return /(camera|realsense|iphone)/i.test(className);
  }

  function hasCaptureBindingCapability(raw, className) {
    const capabilities = raw?.capabilities;
    if (Array.isArray(capabilities)) return capabilities.includes("capture_binding");
    if (capabilities && typeof capabilities === "object") {
      if (capabilities.capture_binding === false) return false;
      if (capabilities.capture_binding === true) return true;
    }
    return CAMERA_BINDING_CLASSES.has(String(className || "").toLowerCase());
  }

  function normalizeDevice(raw, index) {
    const config = raw?.config && typeof raw.config === "object" ? raw.config : {};
    const className = String(
      firstDefined(raw?.class, raw?.device_class, raw?.type, config.class, `Device${index + 1}`)
    );
    const id = firstDefined(raw?.id, raw?.device_class_id, config.device_class_id, index);
    const key = String(firstDefined(raw?.key, raw?.device_key, `${className}:${id}`));
    const processInfo = raw?.process && typeof raw.process === "object" ? raw.process : {};
    const fps = firstDefined(asNumber(raw?.fps), asNumber(config.fps));
    const status = normalizeStatus(raw);
    const shm = normalizeShm(raw, className, id, fps);
    return {
      raw,
      config,
      className,
      id,
      key,
      label: String(firstDefined(raw?.label, raw?.name, config.name, className)),
      status,
      running: firstDefined(raw?.running, processInfo.running, processInfo.alive) === true,
      ready: raw?.ready === true,
      desiredState: String(firstDefined(raw?.desired_state, raw?.desired?.state, "stopped")),
      pid: firstDefined(raw?.pid, processInfo.pid),
      exitCode: firstDefined(raw?.exit_code, processInfo.exit_code),
      restartCount: firstDefined(raw?.restart_count, processInfo.restart_count, 0),
      lastError: String(firstDefined(raw?.last_error, raw?.error, processInfo.error, "")),
      reason: String(firstDefined(raw?.reason, raw?.observed?.reason, "")),
      safety: raw?.safety && typeof raw.safety === "object" ? raw.safety : {},
      progress: raw?.progress && typeof raw.progress === "object" ? raw.progress : {},
      ownership: raw?.ownership && typeof raw.ownership === "object" ? raw.ownership : {},
      fps,
      shape: firstDefined(raw?.shape, raw?.data_shape, config.data_shape, shm.shape),
      dtype: firstDefined(raw?.dtype, raw?.data_dtype, config.data_dtype, shm.dtype),
      shm,
      isCamera: hasPreviewCapability(raw, className),
      supportsCaptureBinding: hasCaptureBindingCapability(raw, className),
      risk: String(firstDefined(raw?.risk, raw?.capabilities?.risk, "")),
      capabilities: raw?.capabilities || {},
    };
  }

  function extractDevices(payload) {
    const source = Array.isArray(payload)
      ? payload
      : firstDefined(payload?.devices, payload?.items, payload?.data?.devices, []);
    if (!Array.isArray(source)) return [];
    return source.map(normalizeDevice);
  }

  function normalizeCameraSlots(payload) {
    const source = payload && typeof payload === "object" && !Array.isArray(payload)
      ? payload
      : {};
    return Object.fromEntries(CAMERA_SLOT_ORDER.map((role) => {
      const raw = source[role] && typeof source[role] === "object" ? source[role] : {};
      const deviceKey = typeof raw.device_key === "string" ? raw.device_key.trim() : "";
      const required = raw.required === true || raw.inference_required === true;
      return [role, {
        role,
        deviceKey,
        configured: raw.configured === true && Boolean(deviceKey),
        required,
        inferenceRequired: required,
        recordingConfigured: raw.recording_configured === true,
        source: String(firstDefined(raw.source, "unconfigured")),
      }];
    }));
  }

  function cameraSlotsFromPayload(payload) {
    return firstDefined(
      payload?.effective_config?.camera_slots,
      payload?.camera_slots,
      payload?.base_config?.camera_slots
    );
  }

  function syncCameraSlots(payload, { clear = false } = {}) {
    const slots = cameraSlotsFromPayload(payload);
    if (slots && typeof slots === "object" && !Array.isArray(slots)) {
      app.cameraSlots = normalizeCameraSlots(slots);
      return true;
    }
    if (clear) app.cameraSlots = normalizeCameraSlots({});
    return false;
  }

  function normalizeRobotSlots(payload) {
    const source = payload && typeof payload === "object" && !Array.isArray(payload)
      ? payload
      : {};
    return Object.fromEntries(ROBOT_SLOT_ORDER.map((role) => {
      const raw = source[role] && typeof source[role] === "object" ? source[role] : {};
      const deviceKey = typeof raw.device_key === "string" ? raw.device_key.trim() : "";
      const required = raw.required === true || raw.inference_required === true;
      return [role, {
        role,
        deviceKey,
        configured: raw.configured === true && Boolean(deviceKey),
        required,
        inferenceRequired: required,
        source: String(firstDefined(raw.source, "unconfigured")),
      }];
    }));
  }

  function robotSlotsFromPayload(payload) {
    return firstDefined(
      payload?.effective_config?.robot_slots,
      payload?.robot_slots,
      payload?.base_config?.robot_slots
    );
  }

  function syncRobotSlots(payload, { clear = false } = {}) {
    const slots = robotSlotsFromPayload(payload);
    if (slots && typeof slots === "object" && !Array.isArray(slots)) {
      app.robotSlots = normalizeRobotSlots(slots);
      return true;
    }
    if (clear) app.robotSlots = normalizeRobotSlots({});
    return false;
  }

  function normalizeHardwareDiscovery(payload) {
    const source = payload?.hardware && typeof payload.hardware === "object"
      ? payload.hardware
      : payload?.data && typeof payload.data === "object"
        ? payload.data
        : payload || {};
    const recommendedOverrides = firstDefined(
      source.recommended?.overrides,
      source.recommended_overrides,
      source.overrides,
      {}
    );
    const robotSource = Array.isArray(source.robots) ? source.robots : [];
    const cameraSource = Array.isArray(source.cameras) ? source.cameras : [];
    const robots = robotSource.map((item) => {
      const id = String(firstDefined(item?.id, item?.binding_id, item?.opaque_id, ""));
      const rdkSn = String(
        firstDefined(item?.rdk_sn, item?.serial_number, item?.serial, item?.name, "")
      );
      const explicitVariant = String(firstDefined(item?.variant, "")).trim();
      const variant = explicitVariant
        ? (/4r/i.test(explicitVariant) ? "4r" : "standard")
        : (/4r/i.test(rdkSn) || /^Rizon4R$/i.test(String(item?.model || "")) ? "4r" : "standard");
      return {
        id,
        rdkSn,
        model: String(firstDefined(item?.model, "Rizon")),
        variant,
        detected: firstDefined(item?.detected, true) !== false,
        source: String(firstDefined(item?.source, "")),
        confidence: asNumber(item?.confidence),
      };
    }).filter((item) => item.id);
    const cameras = cameraSource.map((item) => ({
      id: String(firstDefined(item?.id, item?.binding_id, item?.opaque_id, "")),
      stablePath: String(firstDefined(item?.stable_path, "")),
      kernelNode: String(firstDefined(item?.kernel_node, item?.device_node, item?.devnode, "")),
      name: String(firstDefined(item?.name, item?.model, "Webcam")),
      serial: String(firstDefined(item?.serial, "")),
      detected: firstDefined(item?.detected, true) !== false,
      capture: firstDefined(item?.capture, item?.capture_capable, true) !== false,
    })).filter((item) => item.id);
    return {
      revision: firstDefined(source.revision, payload?.revision, null),
      robots,
      cameras,
      recommendedOverrides: recommendedOverrides && typeof recommendedOverrides === "object"
        ? recommendedOverrides
        : {},
    };
  }

  function isRizonDevice(device) {
    return String(device?.className || "").toLowerCase() === "rizonrobot";
  }

  function isI2rtHeadDevice(device) {
    return String(device?.className || "").toLowerCase() === "yamheadrobot";
  }

  function bindingDevices() {
    return app.configDevices.length ? app.configDevices : app.devices;
  }

  function bindingRecommendation(deviceKey) {
    const value = app.discovery.recommendedOverrides?.[deviceKey];
    return value && typeof value === "object" ? value : {};
  }

  function effectiveBindingId(deviceKey, field) {
    const value = app.state.effective_bindings?.[deviceKey]?.[field];
    return typeof value?.id === "string" ? value.id : "";
  }

  function availableRobotBinding(id, side) {
    if (!id) return null;
    const expectedVariant = side === "right" ? "4r" : "standard";
    return app.discovery.robots.find(
      (robot) => robot.id === id && robot.detected && robot.variant === expectedVariant
    ) || null;
  }

  function availableCameraBinding(id) {
    if (!id) return null;
    return app.discovery.cameras.find(
      (camera) => camera.id === id && camera.detected && camera.capture
    ) || null;
  }

  function cameraBindingFromLogicalId(device, used) {
    const logicalId = String(device?.id ?? "");
    const suffix = logicalId ? `/video${logicalId}` : "";
    return app.discovery.cameras.find((camera) => {
      if (!camera.detected || !camera.capture || used.has(camera.id)) return false;
      return camera.id === logicalId || (suffix && camera.kernelNode.endsWith(suffix));
    })?.id || "";
  }

  function resetBindingsForSelectedConfig({ preserve = false } = {}) {
    const configName = elements.configSelect.value || app.selectedConfig || app.activeConfig || "";
    if (!preserve || app.bindingConfigName !== configName) {
      app.bindingDirtyKeys.clear();
      app.bindingErrors.clear();
      app.bindingDirty = false;
    }
    const devices = bindingDevices();
    const robotDevice = devices.find(isRizonDevice) || null;
    const robotRecommendation = bindingRecommendation(robotDevice?.key || "");
    const previousLeft = app.robotBindings.left;
    const previousRight = app.robotBindings.right;

    let left = preserve && availableRobotBinding(previousLeft, "left") ? previousLeft : "";
    if (!left && availableRobotBinding(effectiveBindingId(robotDevice?.key || "", "robot_left"), "left")) {
      left = effectiveBindingId(robotDevice?.key || "", "robot_left");
    }
    if (!left && availableRobotBinding(robotRecommendation.robot_left, "left")) left = robotRecommendation.robot_left;
    if (!left) left = app.discovery.robots.find((item) => item.detected && item.variant === "standard")?.id || "";

    let right = preserve && availableRobotBinding(previousRight, "right") ? previousRight : "";
    if (!right && availableRobotBinding(effectiveBindingId(robotDevice?.key || "", "robot_right"), "right")) {
      right = effectiveBindingId(robotDevice?.key || "", "robot_right");
    }
    if (!right && availableRobotBinding(robotRecommendation.robot_right, "right")) right = robotRecommendation.robot_right;
    if (!right) right = app.discovery.robots.find((item) => item.detected && item.variant === "4r" && item.id !== left)?.id || "";
    if (right === left) right = "";
    app.robotBindings = { left, right };

    const previousCameras = app.cameraBindings;
    const nextCameras = new Map();
    const used = new Set();
    const preservedCameraKeys = new Set();
    const cameraDevices = devices.filter(
      (item) => item.isCamera && item.supportsCaptureBinding
    );
    for (const device of cameraDevices) {
      const recommendation = bindingRecommendation(device.key);
      const preservePrevious = preserve
        && app.bindingConfigName === configName
        && previousCameras.has(device.key);
      const previous = preservePrevious ? previousCameras.get(device.key) : "";
      let selected = availableCameraBinding(previous) && !used.has(previous) ? previous : "";
      if (preservePrevious) {
        preservedCameraKeys.add(device.key);
      } else {
        // The backend binding is the runtime truth. Prefer it over inferring a
        // physical camera from the logical video id; otherwise the dropdown can
        // disagree with the selected config until the user unbinds/rebinds it.
        const effective = configName === app.selectedConfig
          ? effectiveBindingId(device.key, "capture_device")
          : "";
        if (!selected && availableCameraBinding(effective) && !used.has(effective)) {
          selected = effective;
        }
        if (!selected) selected = cameraBindingFromLogicalId(device, used);
        if (
          !selected
          && availableCameraBinding(recommendation.capture_device)
          && !used.has(recommendation.capture_device)
        ) {
          selected = recommendation.capture_device;
        }
      }
      nextCameras.set(device.key, selected);
      if (selected) used.add(selected);
    }
    for (const device of cameraDevices) {
      if (nextCameras.get(device.key) || preservedCameraKeys.has(device.key)) continue;
      const selected = app.discovery.cameras.find(
        (camera) => camera.detected && camera.capture && !used.has(camera.id)
      )?.id || "";
      nextCameras.set(device.key, selected);
      if (selected) used.add(selected);
    }
    app.cameraBindings = nextCameras;
    app.bindingConfigName = configName;
    renderBindingControls();
  }

  async function loadHardwareDiscovery({
    manual = false,
    preserve = true,
    configName = "",
    throwOnError = false,
    reconcileBindings = true,
  } = {}) {
    const requestId = ++app.hardwareRequest;
    app.hardwareLoading = true;
    try {
      const query = configName ? `?config_name=${encodeURIComponent(configName)}` : "";
      const payload = await api(`${HARDWARE_DISCOVERY_URL}${query}`, { cache: "no-store" });
      if (requestId !== app.hardwareRequest) return;
      app.discovery = normalizeHardwareDiscovery(payload);
      if (reconcileBindings) resetBindingsForSelectedConfig({ preserve });
      if (manual) showToast("硬件发现已刷新", "success");
    } catch (error) {
      if (requestId !== app.hardwareRequest) return;
      if (throwOnError) throw error;
      app.discovery = { revision: null, robots: [], cameras: [], recommendedOverrides: {} };
      if (reconcileBindings) resetBindingsForSelectedConfig({ preserve: false });
      if (manual) showToast(`硬件发现失败：${errorMessage(error)}`, "error");
    } finally {
      if (requestId === app.hardwareRequest) {
        app.hardwareLoading = false;
        renderBindingControls();
      }
    }
  }

  function setServerStatus(mode, text) {
    elements.serverStateDot.className = `status-dot status-dot--${mode}`;
    elements.serverStateText.textContent = text;
  }

  function setGlobalNotice(message = "") {
    elements.globalNotice.hidden = !message;
    elements.globalNotice.textContent = message;
  }

  function recordControlFailure(scope, message) {
    const detail = String(message || "设备未达到请求状态");
    app.controlFailureNotice = detail;
    if (String(scope || "").startsWith("device:")) {
      app.failureLogKey = String(scope).slice("device:".length);
    }
    setGlobalNotice(detail);
    showToast(detail, "error");
  }

  function clearControlFailure() {
    app.controlFailureNotice = "";
    app.failureLogKey = "";
  }

  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.textContent = message;
    elements.toastRegion.appendChild(toast);
    window.setTimeout(() => toast.remove(), 4200);
  }

  function errorMessage(error) {
    if (error instanceof ApiError && error.status === 409) {
      return `${error.message}（当前状态不允许此操作）`;
    }
    return error?.message || "发生未知错误";
  }

  function selectedConfigFromPayload(payload) {
    if (!payload || typeof payload !== "object") return null;
    for (const field of ["selected_config", "selected"]) {
      if (Object.prototype.hasOwnProperty.call(payload, field)) {
        return normalizeConfigName(payload[field]);
      }
    }
    return null;
  }

  function normalizeCheckpointState(raw = null) {
    const source = raw && typeof raw === "object"
      ? raw
      : firstDefined(
        app.state?.checkpoint,
        app.state?.workflow?.server?.checkpoint,
        app.state?.workflow?.inference?.checkpoint,
        {}
      );
    const generation = asNumber(source?.generation);
    const defaultPath = String(source?.default_path || source?.defaultPath || "").trim();
    const overridePath = String(source?.override_path || source?.overridePath || "").trim();
    const effectivePath = String(source?.effective_path || source?.effectivePath || "").trim();
    const blockers = Array.isArray(source?.blockers)
      ? source.blockers.map((item) => String(item || "").trim()).filter(Boolean)
      : [];
    const error = String(firstDefined(source?.path_error, source?.error, "") || "").trim();
    return {
      configName: normalizeConfigName(firstDefined(source?.config_name, app.selectedConfig)),
      generation: generation === null ? 0 : Math.max(0, Math.trunc(generation)),
      defaultPath,
      overridePath,
      effectivePath,
      overrideActive: source?.override_active === true || Boolean(overridePath),
      exists: source?.exists === true,
      paramsExists: source?.params_exists === true,
      paramsRequired: source?.params_required !== false,
      ownerStage: String(source?.owner_stage || "inference"),
      valid: source?.valid === true,
      canEdit: source?.can_edit === true || source?.editable === true,
      blockers,
      error,
    };
  }

  function checkpointContextFromState() {
    const checkpoint = normalizeCheckpointState();
    const revision = asNumber(app.state?.revision);
    return {
      selectedConfig: normalizeConfigName(app.selectedConfig),
      revision: revision === null ? null : Math.trunc(revision),
      generation: checkpoint.generation,
    };
  }

  function checkpointContextsEqual(left, right, { includeRevision = true } = {}) {
    if (!left || !right) return false;
    return Boolean(
      left.selectedConfig
      && left.selectedConfig === right.selectedConfig
      && left.generation === right.generation
      && (!includeRevision || left.revision === right.revision)
    );
  }

  function checkpointContextFromPayload(payload) {
    const state = stateFromPayload(payload);
    const checkpoint = normalizeCheckpointState(state?.checkpoint);
    const revision = asNumber(state?.revision);
    return {
      selectedConfig: normalizeConfigName(selectedConfigFromPayload(state)),
      revision: revision === null ? null : Math.trunc(revision),
      generation: checkpoint.generation,
      checkpoint,
    };
  }

  function checkpointBrowsePayloadMatches(context, payload) {
    if (!context || !payload || typeof payload !== "object") return false;
    const revision = asNumber(payload.revision);
    const generation = asNumber(payload.generation);
    return Boolean(
      normalizeConfigName(payload.config_name) === context.selectedConfig
      && revision !== null
      && Math.trunc(revision) === context.revision
      && generation !== null
      && Math.trunc(generation) === context.generation
      && checkpointContextsEqual(context, checkpointContextFromState())
    );
  }

  function checkpointStateForRender() {
    const current = normalizeCheckpointState();
    const optimistic = app.checkpointOptimistic;
    if (!optimistic) return current;
    if (optimistic.selectedConfig !== app.selectedConfig) {
      app.checkpointOptimistic = null;
      return current;
    }
    const currentRevision = asNumber(app.state?.revision);
    if (
      current.generation >= optimistic.generation
      && currentRevision !== null
      && currentRevision >= optimistic.revision
    ) {
      app.checkpointOptimistic = null;
      return current;
    }
    return normalizeCheckpointState(optimistic.checkpoint);
  }

  function normalizeRolloutStorage(raw = null) {
    const source = raw && typeof raw === "object" ? raw : {};
    const status = ["disabled", "unavailable", "healthy", "warning", "critical"]
      .includes(String(source.status || ""))
      ? String(source.status)
      : "unavailable";
    const number = (value) => {
      const parsed = asNumber(value);
      return parsed === null ? null : Math.max(0, parsed);
    };
    return {
      status,
      enabled: source.enabled === true,
      path: String(source.path || "").trim(),
      error: String(source.error || "").trim(),
      totalBytes: number(source.total_bytes),
      freeBytes: number(source.free_bytes),
      reserveBytes: number(source.reserve_bytes),
      usableBytes: number(source.usable_bytes),
      freeFraction: number(source.free_fraction),
      estimatedBytesPerSecond: number(source.estimated_bytes_per_second),
      estimatedRemainingSeconds: number(source.estimated_remaining_seconds),
      estimateSource: String(source.estimate_source || "configuration"),
      observedSamples: number(source.observed_samples) || 0,
      recordingActive: source.recording_active === true,
    };
  }

  function normalizeInferenceOptions(raw = null) {
    const source = raw && typeof raw === "object"
      ? raw
      : firstDefined(
        app.state?.inference_options,
        app.state?.workflow?.inference_options,
        {}
      );
    const generation = asNumber(source?.generation);
    const blockers = Array.isArray(source?.blockers)
      ? source.blockers.map((item) => String(item || "").trim()).filter(Boolean)
      : [];
    return {
      configName: normalizeConfigName(firstDefined(source?.config_name, app.selectedConfig)),
      generation: generation === null ? 0 : Math.max(0, Math.trunc(generation)),
      testMode: source?.test_mode === true,
      mockSupported: source?.mock_supported === true,
      mockError: String(source?.mock_error || "").trim(),
      rolloutEnabled: source?.rollout_enabled === true,
      rolloutOutputRoot: String(source?.rollout_output_root || "").trim(),
      rolloutEffectiveOutputRoot: String(source?.rollout_effective_output_root || "").trim(),
      rolloutPathValid: source?.rollout_path_valid === true,
      rolloutPathError: String(source?.rollout_path_error || "").trim(),
      rolloutStorage: normalizeRolloutStorage(source?.rollout_storage),
      canEdit: source?.can_edit === true || source?.editable === true,
      blockers,
    };
  }

  function inferenceOptionsContextFromState() {
    const options = normalizeInferenceOptions();
    const revision = asNumber(app.state?.revision);
    return {
      selectedConfig: normalizeConfigName(app.selectedConfig),
      revision: revision === null ? null : Math.trunc(revision),
      generation: options.generation,
    };
  }

  function inferenceOptionsContextsEqual(left, right, { includeRevision = true } = {}) {
    if (!left || !right) return false;
    return Boolean(
      left.selectedConfig
      && left.selectedConfig === right.selectedConfig
      && left.generation === right.generation
      && (!includeRevision || left.revision === right.revision)
    );
  }

  function inferenceOptionsBrowsePayloadMatches(context, payload) {
    if (!context || !payload || typeof payload !== "object") return false;
    const revision = asNumber(payload.revision);
    const generation = asNumber(payload.generation);
    return Boolean(
      normalizeConfigName(payload.config_name) === context.selectedConfig
      && revision !== null
      && Math.trunc(revision) === context.revision
      && generation !== null
      && Math.trunc(generation) === context.generation
      && inferenceOptionsContextsEqual(context, inferenceOptionsContextFromState())
    );
  }

  function configChoiceExists(name) {
    return Boolean(name) && app.configs.some((item) => item.name === name);
  }

  function configFallbackChoice(value = "") {
    if (configChoiceExists(value)) return value;
    return chooseBootstrapConfig(app.configs, app.recommendedConfig);
  }

  function syncConfigSelectToServer({ force = false, fallbackValue = "" } = {}) {
    if (!force && (app.configSwitching || document.activeElement === elements.configSelect)) {
      app.configSelectionDeferred = true;
      return false;
    }

    const selected = normalizeConfigName(app.selectedConfig);
    if (
      selected
      && !configChoiceExists(selected)
      && typeof Option === "function"
      && typeof elements.configSelect.add === "function"
    ) {
      const option = new Option(`${selected}（当前配置）`, selected);
      option.title = "服务端当前实际生效配置；配置列表中暂不可用";
      elements.configSelect.add(option);
    }
    const desired = selected || configFallbackChoice(fallbackValue);
    if (app.bootstrapManualChoiceRequired && !selected) {
      elements.configSelect.value = "";
      elements.configSelect.selectedIndex = -1;
    } else if (desired) {
      elements.configSelect.value = desired;
    } else {
      elements.configSelect.value = "";
      elements.configSelect.selectedIndex = -1;
    }
    app.configSelectionBeforeOpen = selected || desired || "";
    app.configSelectionDeferred = false;
    return true;
  }

  function renderConfigOptions({ preserveValue = true } = {}) {
    const currentValue = preserveValue ? elements.configSelect.value : "";
    elements.configSelect.replaceChildren();

    if (!app.configs.length) {
      const option = new Option("未找到可用配置", "");
      elements.configSelect.add(option);
      elements.configSelect.disabled = true;
      return;
    }

    for (const config of app.configs) {
      const label = config.valid ? config.label : `${config.label}（无效）`;
      const option = new Option(label, config.name);
      option.disabled = !config.valid;
      option.title = config.error || config.name;
      elements.configSelect.add(option);
    }
    elements.configSelect.disabled = false;

    syncConfigSelectToServer({
      force: true,
      fallbackValue: app.configSwitching ? currentValue : "",
    });
  }

  async function loadConfigs({ render = true } = {}) {
    const payload = await api("/api/configs");
    app.configs = extractConfigs(payload);
    app.recommendedConfig = normalizeConfigName(payload?.recommended_config);
    const selected = selectedConfigFromPayload(payload);
    // `/api/state` is the canonical lifecycle snapshot. `/api/configs` only
    // seeds the chooser before the first state snapshot so a slower config-list
    // response cannot overwrite a newer selection observed by state polling.
    if (selected !== null && !app.initialStateObserved) setSelectedConfig(selected);
    if (render && document.activeElement !== elements.configSelect) {
      renderConfigOptions();
      app.configOptionsDeferred = false;
    } else {
      app.configOptionsDeferred = true;
    }
    return payload;
  }

  async function loadConfigDetail(name, { throwOnError = false } = {}) {
    const requestId = ++app.configDetailRequest;
    if (!name) {
      elements.configDetail.textContent = "";
      elements.configDetail.removeAttribute("title");
      return;
    }
    elements.configDetail.textContent = "正在检查配置…";
    try {
      const payload = await api(`/api/configs/${encodeURIComponent(name)}`);
      if (requestId !== app.configDetailRequest) return;
      const devices = extractDevices(payload);
      app.configDevices = devices;
      if (name === app.selectedConfig || !app.activeConfig) {
        syncCameraSlots(payload);
        syncRobotSlots(payload);
      }
      resetBindingsForSelectedConfig({ preserve: app.bindingConfigName === name });
      const error = firstDefined(
        payload?.error,
        payload?.validation_error,
        Array.isArray(payload?.errors) && payload.errors.length ? payload.errors.join("；") : undefined
      );
      if (error) {
        elements.configDetail.textContent = `配置错误：${error}`;
        elements.configDetail.title = String(error);
        if (throwOnError) throw new ApiError(String(error), 422, payload);
        return;
      }
      const suffix = name === app.selectedConfig ? " · 当前配置" : "";
      const detail = configDetailSummary(payload, devices, suffix);
      elements.configDetail.textContent = detail.text;
      elements.configDetail.title = detail.title;
      if (name === app.selectedConfig) {
        void loadPreflight();
        void loadRuntimeDiagnostics({ includeTuning: true });
      }
    } catch (error) {
      if (requestId !== app.configDetailRequest) return;
      elements.configDetail.textContent = `无法读取配置：${errorMessage(error)}`;
      elements.configDetail.title = errorMessage(error);
      if (throwOnError) throw error;
    }
  }

  function configDetailSummary(payload, devices, suffix = "") {
    const summary = payload?.effective_config && typeof payload.effective_config === "object"
      ? payload.effective_config
      : (payload && typeof payload === "object" ? payload : {});
    const metadata = summary.metadata && typeof summary.metadata === "object"
      ? summary.metadata
      : summary;
    const profile = summary.execution_profile && typeof summary.execution_profile === "object"
      ? summary.execution_profile
      : {};
    const policy = profile.policy && typeof profile.policy === "object" ? profile.policy : {};
    const executor = profile.action_executor && typeof profile.action_executor === "object"
      ? profile.action_executor
      : {};
    const managerValue = firstDefined(
      typeof executor.chunk_manager === "object" ? executor.chunk_manager?.type : executor.chunk_manager,
      executor.manager_type,
    );
    const managerOptions = firstDefined(
      executor.chunk_manager_options,
      typeof executor.chunk_manager === "object" ? executor.chunk_manager : null,
    );
    const gripperSlots = summary.gripper_slots && typeof summary.gripper_slots === "object"
      ? summary.gripper_slots
      : {};
    const gripperCount = Object.values(gripperSlots).filter((item) => item?.configured === true).length;
    const workflowKind = String(firstDefined(metadata.workflow_kind, "") || "").trim();
    const policyClass = String(firstDefined(metadata.policy_class, policy.class, "") || "").trim();
    const policyFps = asNumber(policy.fps);
    const controlFps = asNumber(policy.control_fps);
    const visible = [
      String(devices.length) + " 个设备",
      workflowKind || null,
      policyClass || null,
      policyFps === null ? null : "Policy " + diagnosticNumber(policyFps, " Hz"),
      controlFps === null ? null : "Control " + diagnosticNumber(controlFps, " Hz"),
      managerValue ? "ActionManager " + String(managerValue) : null,
      gripperCount ? "夹爪 " + String(gripperCount) + " 路" : null,
    ].filter(Boolean);
    const optionText = managerOptions && typeof managerOptions === "object"
      ? Object.entries(managerOptions)
        .slice(0, 8)
        .map(([key, value]) => String(key) + "=" + String(value))
        .join(" · ")
      : "";
    const title = [
      visible.join(" · ") + suffix,
      optionText ? "ActionManager options: " + optionText : null,
      metadata.hardware_required === true ? "需要真实硬件" : "无需真实硬件",
    ].filter(Boolean).join("\n");
    return { text: visible.join(" · ") + suffix, title };
  }

  const DIAGNOSTIC_STATUS_LABELS = Object.freeze({
    ready: "可启动",
    blocked: "已阻塞",
    warning: "有警告",
    unknown: "待确认",
    ok: "正常",
    not_required: "未配置",
    blocking: "阻塞",
  });

  function diagnosticNumber(value, unit = "") {
    const number = asNumber(value);
    if (number === null) return "—";
    const rounded = Math.abs(number) >= 100
      ? Math.round(number)
      : Number.isInteger(number)
        ? number
        : number.toFixed(1);
    return String(rounded) + unit;
  }

  function diagnosticStatusLabel(status) {
    return DIAGNOSTIC_STATUS_LABELS[String(status || "").toLowerCase()]
      || String(status || "未知");
  }

  function diagnosticRate(actual, configured, { reported = false } = {}) {
    const measured = asNumber(actual);
    if (measured !== null) {
      return diagnosticNumber(measured, " Hz") + (reported ? " · SHM" : " · 实测");
    }
    const configuredValue = asNumber(configured);
    return configuredValue === null
      ? "—"
      : diagnosticNumber(configuredValue, " Hz") + " · 配置";
  }

  function renderPreflight(report = app.preflight) {
    if (!elements.preflightSummary || !elements.preflightChecks) return;
    if (!report || typeof report !== "object") {
      setText(elements.preflightSummary, "尚未检查");
      setText(elements.preflightStatus, app.selectedConfig ? "正在执行启动预检…" : "选择配置后自动检查。");
      elements.preflightSummary.removeAttribute("data-status");
      elements.preflightChecks.replaceChildren();
      return;
    }
    const overall = String(report.overall || "unknown").toLowerCase();
    const summary = report.summary && typeof report.summary === "object" ? report.summary : {};
    const blocking = Number(summary.blocking) || 0;
    const warning = Number(summary.warning) || 0;
    const unknown = Number(summary.unknown) || 0;
    setText(
      elements.preflightSummary,
      diagnosticStatusLabel(overall)
        + " · 阻塞 "
        + blocking
        + " · 警告 "
        + warning
        + (unknown ? " · 待确认 " + unknown : "")
    );
    elements.preflightSummary.dataset.status = overall;
    const generated = report.generated_at ? String(report.generated_at).replace("T", " ").slice(0, 19) : "";
    const stages = report.stages && typeof report.stages === "object" ? report.stages : {};
    const runtimeStage = stages.runtime && typeof stages.runtime === "object" ? stages.runtime : null;
    const inferenceStage = stages.inference && typeof stages.inference === "object" ? stages.inference : null;
    const stageText = runtimeStage || inferenceStage
      ? "Runtime " + diagnosticStatusLabel(runtimeStage?.status || "unknown")
        + " / Inference " + diagnosticStatusLabel(inferenceStage?.status || "unknown")
      : "";
    setText(
      elements.preflightStatus,
      (generated ? "最近检查 " + generated : "已完成启动预检")
        + (stageText ? " · " + stageText : "")
    );
    elements.preflightChecks.replaceChildren();
    const checks = Array.isArray(report.checks) ? report.checks : [];
    for (const check of checks) {
      if (!check || typeof check !== "object") continue;
      const status = String(check.status || "unknown").toLowerCase();
      const row = makeTextElement("li", "preflight-check", "");
      row.dataset.status = status;
      const label = makeTextElement("span", "preflight-check__label", String(check.label || check.id || "检查"));
      const state = makeTextElement("span", "preflight-check__status", diagnosticStatusLabel(status));
      row.append(label, state);
      if (check.remediation) row.title = String(check.remediation);
      elements.preflightChecks.appendChild(row);
    }
  }

  function renderRuntimeDiagnostics(payload = app.runtimeMetrics) {
    if (!payload || typeof payload !== "object") {
      setText(elements.debugPolicyFps, "—");
      setText(elements.debugControlFps, "—");
      setText(elements.debugActionManager, "—");
      setText(elements.debugActionFps, "—");
      setText(elements.debugDeviceFps, "—");
      setText(elements.debugActionQueue, "—");
      return;
    }
    const policy = payload.policy && typeof payload.policy === "object" ? payload.policy : {};
    const executor = payload.action_executor && typeof payload.action_executor === "object"
      ? payload.action_executor
      : {};
    const manager = payload.device_manager && typeof payload.device_manager === "object"
      ? payload.device_manager
      : {};
    const devices = Array.isArray(payload.devices) ? payload.devices : [];
    const reported = devices
      .map((item) => asNumber(item?.reported_fps ?? item?.shm_fps))
      .filter((value) => value !== null);
    const deviceFps = reported.length
      ? reported.reduce((sum, value) => sum + value, 0) / reported.length
      : null;
    const executorKind = firstDefined(
      typeof executor.chunk_manager === "object"
        ? executor.chunk_manager?.type
        : executor.chunk_manager,
      executor.manager_type,
      executor.executor_type
    );
    const managerOptions = firstDefined(
      executor.chunk_manager_options,
      typeof executor.chunk_manager === "object"
        ? executor.chunk_manager
        : null
    );
    const executorFps = asNumber(executor.actual_fps);
    const queue = asNumber(executor.queue_depth);
    const latency = firstDefined(
      asNumber(executor.dispatch_latency_ms),
      asNumber(policy.inference_latency_ms)
    );
    setText(
      elements.debugPolicyFps,
      diagnosticRate(policy.actual_fps, policy.configured_fps)
    );
    setText(
      elements.debugControlFps,
      diagnosticRate(null, policy.control_fps)
    );
    setText(elements.debugActionManager, executorKind ? String(executorKind) : "—");
    if (elements.debugActionManager) {
      const optionText = managerOptions && typeof managerOptions === "object"
        ? Object.entries(managerOptions)
          .slice(0, 8)
          .map(([key, value]) => key + "=" + String(value))
          .join(" · ")
        : "";
      elements.debugActionManager.title = optionText
        ? String(executorKind || "ActionManager") + " · " + optionText
        : String(executorKind || "");
    }
    setText(
      elements.debugActionFps,
      diagnosticRate(executorFps, executor.configured_fps)
    );
    const managerActual = asNumber(manager.actual_fps);
    setText(
      elements.debugDeviceFps,
      diagnosticRate(
        managerActual !== null ? managerActual : deviceFps,
        managerActual !== null ? null : manager.target_fps,
        { reported: managerActual === null && deviceFps !== null }
      )
    );
    setText(
      elements.debugActionQueue,
      (queue === null ? "—" : String(Math.round(queue)))
        + " / "
        + (latency === null ? "—" : diagnosticNumber(latency, " ms"))
    );
  }

  function runtimeTuningInputId(key) {
    return "runtime-tuning-" + String(key || "").replace(/[^A-Za-z0-9_-]+/g, "-");
  }

  function selectedRuntimeTuningContext() {
    return {
      configName: normalizeConfigName(app.state?.selected_config || app.selectedConfig || ""),
      revision: asNumber(app.state?.revision),
    };
  }

  function runtimeTuningMatchesContext(payload, context = selectedRuntimeTuningContext()) {
    if (!payload || typeof payload !== "object") return false;
    const payloadName = normalizeConfigName(payload.config_name || payload.selected_config || "");
    const payloadRevision = asNumber(payload.revision);
    return Boolean(
      context.configName
      && payloadName === context.configName
      && Number.isInteger(context.revision)
      && payloadRevision === context.revision
    );
  }

  function invalidateRuntimeTuning({ preserveDraft = false } = {}) {
    app.runtimeTuning = null;
    if (!preserveDraft) {
      app.runtimeTuningDraft.clear();
      app.generatedConfigSource = "";
    }
    if (!app.generatedConfigPending) setGeneratedConfigStatus("");
    renderRuntimeTuning();
  }

  function setSelectedConfig(nextValue) {
    const next = nextValue === null ? null : normalizeConfigName(nextValue);
    if (next === app.selectedConfig) return false;
    app.selectedConfig = next;
    invalidateRuntimeTuning();
    return true;
  }

  function generatedConfigDefaultId() {
    const config = normalizeConfigName(app.selectedConfig || "");
    const raw = config.split("/").pop() || "run";
    const base = raw
      .replace(/[^A-Za-z0-9_.-]+/g, "-")
      .replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, "")
      .slice(0, 48);
    const stamp = new Date().toISOString()
      .replace(/[-:]/g, "")
      .replace(/\.\d{3}Z$/, "Z");
    return (base || "run") + "-" + stamp;
  }

  function setGeneratedConfigStatus(message, status = "") {
    if (!elements.generatedConfigStatus) return;
    setText(elements.generatedConfigStatus, message || "");
    if (status) elements.generatedConfigStatus.dataset.status = status;
    else delete elements.generatedConfigStatus.dataset.status;
  }

  function renderGeneratedConfigAvailability(payload = app.runtimeTuning) {
    const capabilities = payload?.capabilities && typeof payload.capabilities === "object"
      ? payload.capabilities
      : {};
    const selectedConfig = normalizeConfigName(app.selectedConfig || "");
    if (app.generatedConfigSource !== selectedConfig) {
      app.generatedConfigSource = selectedConfig;
      app.runtimeTuningDraft.clear();
      if (elements.generatedConfigId) elements.generatedConfigId.value = "";
      setGeneratedConfigStatus("");
    }
    if (elements.generatedConfigId && !elements.generatedConfigId.value.trim() && selectedConfig) {
      elements.generatedConfigId.value = generatedConfigDefaultId();
    }
    const canSave = Boolean(
      selectedConfig
      && runtimeTuningMatchesContext(payload)
      && capabilities.apply_supported === true
      && !app.configSwitching
      && !workflowLocksConfigMutation()
      && !app.pendingActions.size
      && !app.generatedConfigPending
      && !app.currentConfigSavePending
    );
    const canExport = Boolean(
      canSave
      && !app.devices.some((device) => ACTIVE_STATUSES.has(device.status))
    );
    if (elements.generatedConfigId) elements.generatedConfigId.disabled = !canExport;
    if (elements.generatedConfigExport) {
      elements.generatedConfigExport.disabled = !canExport;
      elements.generatedConfigExport.textContent = app.generatedConfigPending
        ? "生成中…"
        : "生成配置";
    }
    if (elements.currentConfigSave) {
      elements.currentConfigSave.disabled = !canSave;
      elements.currentConfigSave.textContent = app.currentConfigSavePending
        ? "保存中…"
        : "保存当前 config";
    }
  }

  function runtimeTuningOverrides() {
    if (!runtimeTuningMatchesContext(app.runtimeTuning)) {
      return {
        error: "调参配置已过期，请等待当前配置的参数重新加载。",
        overrides: {},
      };
    }
    const controls = Array.isArray(app.runtimeTuning?.controls)
      ? app.runtimeTuning.controls
      : [];
    const allowedSections = new Set(["policy"]);
    const overrides = {};
    for (const control of controls) {
      if (!control || typeof control !== "object") continue;
      if (control.writable !== true || control.section_enabled !== true) continue;
      const key = String(control.key || "");
      const [section, field, extra] = key.split(".");
      if (!section || !field || extra || !allowedSections.has(section)) continue;
      const input = document.getElementById(runtimeTuningInputId(key));
      if (!(input instanceof HTMLInputElement) || input.disabled) continue;
      const numeric = asNumber(input.value);
      const minimum = asNumber(control.min);
      const maximum = asNumber(control.max);
      if (
        numeric === null
        || (minimum !== null && numeric < minimum)
        || (maximum !== null && numeric > maximum)
        || !input.validity.valid
      ) {
        return {
          error: String(control.label || key) + " 不在允许范围内。",
          overrides: {},
        };
      }
      const baseline = asNumber(control.value);
      if (baseline === null || numeric !== baseline) {
        if (!Object.prototype.hasOwnProperty.call(overrides, section)) {
          overrides[section] = {};
        }
        overrides[section][field] = numeric;
      }
    }
    return { error: "", overrides };
  }

  async function exportGeneratedConfig() {
    if (app.generatedConfigPending || app.configSwitching || workflowLocksConfigMutation()) return;
    const selectedConfig = String(app.state?.selected_config || app.selectedConfig || "").trim();
    const expectedRevision = asNumber(app.state?.revision);
    const generatedId = String(elements.generatedConfigId?.value || "").trim();
    if (!selectedConfig || expectedRevision === null || !Number.isInteger(expectedRevision)) {
      setGeneratedConfigStatus("当前配置状态已变化，请刷新后重试。", "error");
      return;
    }
    if (
      !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/.test(generatedId)
      || /\.ya?ml$/i.test(generatedId)
    ) {
      setGeneratedConfigStatus("配置名只能包含字母、数字、点、下划线或连字符。", "error");
      return;
    }
    const tuning = runtimeTuningOverrides();
    if (tuning.error) {
      setGeneratedConfigStatus(tuning.error, "error");
      return;
    }

    app.generatedConfigPending = true;
    renderGeneratedConfigAvailability();
    try {
      const result = await api("/api/configs/generated", {
        method: "POST",
        body: JSON.stringify({
          generated_id: generatedId,
          expected_selected_config: selectedConfig,
          expected_revision: expectedRevision,
          tuning_overrides: tuning.overrides,
          overwrite: false,
        }),
      });
      setGeneratedConfigStatus(
        "已生成 " + String(result?.config_name || ""),
        "success"
      );
      showToast("已生成独立运行配置", "success");
      await loadConfigs();
    } catch (error) {
      const message = errorMessage(error);
      setGeneratedConfigStatus(message, "error");
      if (error instanceof ApiError && error.status === 409) {
        scheduleStatePoll(0);
      }
      showToast(message, "error");
    } finally {
      app.generatedConfigPending = false;
      renderGeneratedConfigAvailability();
    }
  }

  async function saveCurrentConfig() {
    if (
      app.currentConfigSavePending
      || app.generatedConfigPending
      || app.configSwitching
      || workflowLocksConfigMutation()
    ) return;
    const selectedConfig = String(app.state?.selected_config || app.selectedConfig || "").trim();
    const expectedRevision = asNumber(app.state?.revision);
    if (!selectedConfig || expectedRevision === null || !Number.isInteger(expectedRevision)) {
      setGeneratedConfigStatus("当前配置状态已变化，请刷新后重试。", "error");
      return;
    }
    const tuning = runtimeTuningOverrides();
    if (tuning.error) {
      setGeneratedConfigStatus(tuning.error, "error");
      return;
    }
    const confirmed = window.confirm(
      `将当前机器人、相机、夹爪端口和 FPS 写回 ${selectedConfig}，`
        + "并覆盖源文件。继续吗？"
    );
    if (!confirmed) return;

    app.currentConfigSavePending = true;
    renderGeneratedConfigAvailability();
    try {
      const result = await api("/api/config/save", {
        method: "POST",
        body: JSON.stringify({
          expected_selected_config: selectedConfig,
          expected_revision: expectedRevision,
          tuning_overrides: tuning.overrides,
        }),
      });
      const reloadDeferred = result?.reload_deferred === true;
      const savedConfig = String(result?.selected_config || selectedConfig);
      setGeneratedConfigStatus(
        reloadDeferred
          ? `已保存到 ${savedConfig}；重新选择 config 或重启 Control Panel 后加载`
          : `已保存到 ${savedConfig}`,
        "success"
      );
      showToast(
        reloadDeferred
          ? "配置已保存；当前设备保持运行，重新选择 config 后生效"
          : "当前配置已保存，重启运行时后生效",
        "success"
      );
      app.stateRequestController?.abort();
      if (app.stateRefreshPromise) await app.stateRefreshPromise;
      await refreshState(false);
      await loadConfigDetail(selectedConfig);
      if (!reloadDeferred) app.runtimeTuningDraft.clear();
      await loadRuntimeDiagnostics({ includeTuning: true, force: true });
    } catch (error) {
      const message = errorMessage(error);
      setGeneratedConfigStatus(message, "error");
      if (error instanceof ApiError && error.status === 409) scheduleStatePoll(0);
      showToast(message, "error");
    } finally {
      app.currentConfigSavePending = false;
      renderGeneratedConfigAvailability();
    }
  }

  function renderRuntimeTuning(payload = app.runtimeTuning) {
    if (!elements.runtimeTuningControls) return;
    if (payload && !runtimeTuningMatchesContext(payload)) payload = null;
    elements.runtimeTuningControls.replaceChildren();
    const controls = Array.isArray(payload?.controls) ? payload.controls : [];
    if (!controls.length) {
      elements.runtimeTuningControls.appendChild(
        makeTextElement("p", "diagnostics-status", "当前配置没有可展示的调参项。")
      );
      renderGeneratedConfigAvailability(payload);
      return;
    }
    for (const control of controls) {
      if (!control || typeof control !== "object") continue;
      const key = String(control.key || "");
      const row = makeTextElement("div", "runtime-tuning-row", "");
      const writable = control.writable === true
        && control.section_enabled === true
        && asNumber(control.value) !== null;
      row.dataset.disabled = writable ? "false" : "true";
      const inputId = runtimeTuningInputId(key);
      const label = makeTextElement(
        "label",
        "",
        String(control.label || key || "参数")
      );
      label.htmlFor = inputId;
      const input = document.createElement("input");
      input.id = inputId;
      input.type = "number";
      input.inputMode = "decimal";
      input.min = String(control.min ?? "");
      input.max = String(control.max ?? "");
      input.step = String(control.step ?? "any");
      const draft = app.runtimeTuningDraft.get(key);
      const value = draft === undefined ? control.value : draft;
      input.value = value === null || value === undefined ? "" : String(value);
      input.disabled = !writable;
      input.title = String(control.apply_mode || "需重启");
      input.addEventListener("input", () => {
        app.runtimeTuningDraft.set(key, input.value);
        setGeneratedConfigStatus("");
      });
      row.append(label, input);
      elements.runtimeTuningControls.appendChild(row);
    }
    renderGeneratedConfigAvailability(payload);
  }

  async function loadPreflight({ manual = false } = {}) {
    const configName = normalizeConfigName(app.selectedConfig || elements.configSelect?.value);
    if (!configName) {
      app.preflight = null;
      renderPreflight();
      return null;
    }
    const requestId = ++app.preflightRequest;
    app.preflightController?.abort();
    const controller = new AbortController();
    app.preflightController = controller;
    setText(elements.preflightStatus, "正在执行启动预检…");
    try {
      const payload = await api("/api/preflight?config_name=" + encodeURIComponent(configName), {
        signal: controller.signal,
        cache: "no-store",
      });
      if (requestId !== app.preflightRequest || app.preflightController !== controller) return null;
      app.preflight = payload;
      renderPreflight(payload);
      if (manual && payload?.overall === "blocked") {
        showToast("启动预检未通过，请先处理阻塞项。", "error");
      }
      return payload;
    } catch (error) {
      if (error?.name === "AbortError") return null;
      if (requestId !== app.preflightRequest) return null;
      app.preflight = null;
      setText(elements.preflightStatus, "预检失败：" + errorMessage(error));
      return null;
    } finally {
      if (app.preflightController === controller) app.preflightController = null;
    }
  }

  async function loadRuntimeDiagnostics({ includeTuning = false, force = false } = {}) {
    if (!app.selectedConfig) return null;
    if (app.diagnosticsLoading) {
      if (!force) return null;
      app.runtimeMetricsController?.abort();
    }
    const tuningContext = selectedRuntimeTuningContext();
    const requestId = ++app.runtimeMetricsRequest;
    app.runtimeMetricsController?.abort();
    const controller = new AbortController();
    app.runtimeMetricsController = controller;
    app.diagnosticsLoading = true;
    try {
      const requests = [
        api("/api/runtime/metrics", {
          signal: controller.signal,
          cache: "no-store",
        }),
      ];
      if (includeTuning || !app.runtimeTuning) {
        requests.push(
          api("/api/runtime/tuning", {
            signal: controller.signal,
            cache: "no-store",
          })
        );
      }
      const results = await Promise.allSettled(requests);
      if (requestId !== app.runtimeMetricsRequest || app.runtimeMetricsController !== controller) return null;
      if (results[0].status === "fulfilled") {
        app.runtimeMetrics = results[0].value;
        renderRuntimeDiagnostics(app.runtimeMetrics);
      }
      if (results[1]?.status === "fulfilled") {
        const schema = results[1].value;
        if (
          runtimeTuningMatchesContext(schema, tuningContext)
          && runtimeTuningMatchesContext(schema)
        ) {
          app.runtimeTuning = schema;
          renderRuntimeTuning(schema);
        }
      }
      return app.runtimeMetrics;
    } catch (error) {
      if (error?.name !== "AbortError") setText(elements.debugActionQueue, "指标不可用");
      return null;
    } finally {
      if (app.runtimeMetricsController === controller) {
        app.runtimeMetricsController = null;
        app.diagnosticsLoading = false;
      }
    }
  }

  function ensureDiagnosticsPolling(immediate = false) {
    if (document.hidden || !app.selectedConfig) {
      clearPollTimer("diagnostics");
      return;
    }
    if (app.pollTimers.diagnostics !== null && !immediate) return;
    clearPollTimer("diagnostics");
    app.pollTimers.diagnostics = window.setTimeout(async () => {
      app.pollTimers.diagnostics = null;
      await loadRuntimeDiagnostics();
      ensureDiagnosticsPolling();
    }, immediate ? 0 : POLL_INTERVALS.diagnostics);
  }

  function buildBindingOverrides() {
    const overrides = {};
    const devices = bindingDevices();
    const robotDevice = devices.find(isRizonDevice);
    if (robotDevice) {
      const recommendation = bindingRecommendation(robotDevice.key);
      const robotOverride = {};
      const left = availableRobotBinding(app.robotBindings.left, "left")?.id;
      const right = availableRobotBinding(app.robotBindings.right, "right")?.id;
      const networkInterface = recommendation.network_interface;
      if (left) robotOverride.robot_left = left;
      if (right) robotOverride.robot_right = right;
      if (typeof networkInterface === "string" && networkInterface.trim()) {
        robotOverride.network_interface = networkInterface;
      }
      if (Object.keys(robotOverride).length) overrides[robotDevice.key] = robotOverride;
    }
    for (const device of devices.filter((item) => item.isCamera && item.supportsCaptureBinding)) {
      const captureDevice = availableCameraBinding(app.cameraBindings.get(device.key))?.id;
      overrides[device.key] = { capture_device: captureDevice || null };
    }
    return overrides;
  }

  function bindingOverrideForDevice(deviceKey) {
    return buildBindingOverrides()[deviceKey] || {};
  }

  function bindingSelectionMatchesEffective(device) {
    if (!device?.key) return true;
    const fields = isRizonDevice(device)
      ? ["robot_left", "robot_right", "network_interface"]
      : device.isCamera && device.supportsCaptureBinding
        ? ["capture_device"]
        : [];
    if (!fields.length) return true;
    const selected = bindingOverrideForDevice(device.key);
    return fields.every((field) => (
      String(selected[field] || "") === effectiveBindingId(device.key, field)
    ));
  }

  function buildBindingUpdateOverrides(deviceKeys) {
    return Object.fromEntries(
      [...new Set(deviceKeys)].map((key) => [key, bindingOverrideForDevice(key)])
    );
  }

  function completeCameraBindingRequestKeys(deviceKey, affectedKeys = []) {
    const target = bindingDevices().find((device) => device.key === deviceKey);
    if (!target?.isCamera || !target.supportsCaptureBinding) {
      return [...new Set([deviceKey, ...affectedKeys].filter(Boolean))];
    }
    // Freeze only the edited slot and any slot displaced by an atomic swap.
    // Including unrelated running cameras makes an unchanged binding look
    // like a mutation and incorrectly couples otherwise independent controls.
    return [...new Set([deviceKey, ...affectedKeys].filter(Boolean))];
  }

  function configuredCameraBindingId(device) {
    return cameraBindingFromLogicalId(device, new Set());
  }

  function resolvedCameraBindingId(device) {
    const selected = availableCameraBinding(app.cameraBindings.get(device?.key))?.id;
    if (app.cameraBindings.has(device?.key)) return selected || "";
    return configuredCameraBindingId(device);
  }

  function applyCameraBindingSelection(deviceKey, selectedBindingId) {
    const targetDevice = bindingDevices().find((device) => device.key === deviceKey);
    if (!targetDevice) return [];
    const previousPhysical = resolvedCameraBindingId(targetDevice);
    const selected = availableCameraBinding(selectedBindingId)?.id || "";
    const selectedPhysical = selected;
    const affectedKeys = [deviceKey];
    if (selectedPhysical) {
      for (const other of bindingDevices().filter(
        (device) => device.key !== deviceKey
          && device.isCamera
          && device.supportsCaptureBinding
      )) {
        if (resolvedCameraBindingId(other) !== selectedPhysical) continue;
        app.cameraBindings.set(
          other.key,
          previousPhysical && previousPhysical !== selectedPhysical ? previousPhysical : ""
        );
        affectedKeys.push(other.key);
      }
    }
    app.cameraBindings.set(deviceKey, selected);
    return affectedKeys;
  }

  function workflowLocksBindings() {
    const workflow = app.state.workflow || {};
    return Boolean(
      workflow.runtime?.active ||
      workflow.inference?.running ||
      app.workflowPending.size
    );
  }

  function workflowLocksConfigMutation() {
    const workflow = app.state.workflow || {};
    const activeStatuses = new Set(["starting", "running", "stopping"]);
    const runtime = workflow.runtime || {};
    const inference = workflow.inference || {};
    return Boolean(
      runtime.active
      || runtime.running
      || activeStatuses.has(String(runtime.status || "").toLowerCase())
      || inference.active
      || inference.running
      || activeStatuses.has(String(inference.status || "").toLowerCase())
      || app.workflowPending.size
    );
  }

  function bindingTargetLocked(device) {
    return Boolean(
      app.configSwitching ||
      !controlsMatchActiveConfig() ||
      workflowLocksBindings() ||
      (device && ACTIVE_STATUSES.has(device.status)) ||
      (device && app.pendingActions.has(device.key)) ||
      (device && app.bindingPending.has(device.key))
    );
  }

  function refreshHardwareDiscoveryForBindings() {
    if (app.hardwareLoading || app.configSwitching || workflowLocksBindings()) return;
    const configName = elements.configSelect.value || app.selectedConfig || "";
    void loadHardwareDiscovery({ manual: true, preserve: true, configName });
  }

  function bindingEditorActive(select) {
    return Boolean(select && document.activeElement === select);
  }

  function activeConfigSwitchDevices() {
    const switchStatuses = new Set(["running", "starting", "degraded", "stop_timeout"]);
    return app.devices.filter((device) => switchStatuses.has(device.status));
  }

  async function restoreConfigChoice(value) {
    const fallback = app.selectedConfig || configFallbackChoice(value);
    syncConfigSelectToServer({ force: true, fallbackValue: fallback });
    if (fallback) await loadConfigDetail(fallback);
  }

  async function selectConfigWithConfirmation(name, previousValue) {
    if (!name || app.configSwitching) return;
    clearControlFailure();
    if (app.checkpointMutationPending) {
      showToast("Checkpoint 正在更新，请完成后再切换 config。", "error");
      await restoreConfigChoice(previousValue);
      return;
    }
    if (name === app.selectedConfig) {
      app.configSelectionBeforeOpen = name;
      await loadConfigDetail(name);
      return;
    }

    invalidateCheckpointBrowser();
    app.checkpointOptimistic = null;

    const previousBindings = {
      discovery: app.discovery,
      configDevices: app.configDevices,
      bindingConfigName: app.bindingConfigName,
      bindingDirty: app.bindingDirty,
      robotBindings: { ...app.robotBindings },
      cameraBindings: new Map(app.cameraBindings),
    };
    const restorePreviousBindings = () => {
      app.discovery = previousBindings.discovery;
      app.configDevices = previousBindings.configDevices;
      app.bindingConfigName = previousBindings.bindingConfigName;
      app.bindingDirty = previousBindings.bindingDirty;
      app.robotBindings = { ...previousBindings.robotBindings };
      app.cameraBindings = new Map(previousBindings.cameraBindings);
      renderBindingControls();
    };

    app.configSwitching = true;
    elements.configSelect.disabled = true;
    renderGrippers();
    renderCheckpointControl();
    try {
      if (app.configRefreshPromise) await app.configRefreshPromise;
      const [detailResult] = await Promise.allSettled([
        loadConfigDetail(name, { throwOnError: true }),
        loadHardwareDiscovery({
          configName: name,
          preserve: false,
          throwOnError: false,
        }),
      ]);
      if (detailResult.status === "rejected") throw detailResult.reason;
    } catch (error) {
      restorePreviousBindings();
      const message = `无法读取候选配置：${errorMessage(error)}`;
      setGlobalNotice(message);
      showToast(message, "error");
      app.configSwitching = false;
      elements.configSelect.disabled = !app.configs.length;
      renderGrippers();
      renderCheckpointControl();
      await restoreConfigChoice(previousValue);
      return;
    }
    // Hardware inventory is advisory while selecting a config. It is loaded in
    // parallel with the candidate detail above so unplugged or slow hardware
    // discovery does not add another serial wait before confirmation.

    const activeDevices = activeConfigSwitchDevices();
    const keys = activeDevices.map((device) => device.key);
    const statusCounts = ["running", "starting", "degraded", "stop_timeout"]
      .map((status) => `${status}=${activeDevices.filter((device) => device.status === status).length}`)
      .join("，");
    const detail = keys.length
      ? `当前 ${keys.length} 个活跃设备（${statusCounts}）：${keys.join("、")}。`
      : `当前没有活跃设备（${statusCounts}）。`;
    const workflow = app.state.workflow || {};
    const workflowStages = [];
    if (workflow.inference?.running) workflowStages.push("OpenPI 推理");
    if (workflow.runtime?.active) workflowStages.push("运行时");
    const workflowDetail = workflowStages.length
      ? ` 当前还在运行：${workflowStages.join("、")}。`
      : "";
    const confirmed = await confirmOperation(
      "切换运行配置",
      `${detail}${workflowDetail} 切换到 ${name} 前，控制面板会依次停止推理、ActionExecutor、DeviceManager 和设备；若设备停机超时，将保留原配置。`,
      "安全停止并切换"
    );
    if (!confirmed) {
      restorePreviousBindings();
      app.configSwitching = false;
      elements.configSelect.disabled = !app.configs.length;
      renderGrippers();
      renderCheckpointControl();
      await restoreConfigChoice(previousValue);
      return;
    }

    app.stateRequestController?.abort();
    if (app.stateRefreshPromise) await app.stateRefreshPromise;
    invalidateGripperSyncContext();
    try {
      const payload = await api("/api/config/select", {
        method: "POST",
        body: JSON.stringify({
          name,
          discovery_revision: app.discovery.revision,
          overrides: buildBindingOverrides(),
          stop_active: true,
        }),
      });
      app.bindingDirty = false;
      app.bindingDirtyKeys.clear();
      app.bindingErrors.clear();
      const selected = selectedConfigFromPayload(stateFromPayload(payload));
      setSelectedConfig(selected === null ? name : selected);
      app.activeConfig = normalizeConfigName(payload?.active_config);
      syncConfigSelectToServer({ force: true });
      app.bootstrapFailureNotice = "";
      app.bootstrapManualChoiceRequired = false;
      setGlobalNotice("");
      showToast(`已安全切换到配置 ${name}`, "success");
      await refreshState(false);
      requestGripperActualAlignment(GRIPPER_SIDES, "config_changed");
      await loadConfigDetail(name);
    } catch (error) {
      const message = errorMessage(error);
      setGlobalNotice(message);
      showToast(message, "error");
      restorePreviousBindings();
      const selected = selectedConfigFromPayload(error?.payload);
      if (selected !== null) setSelectedConfig(selected);
      await refreshState(false);
      await restoreConfigChoice(app.selectedConfig || previousValue);
    } finally {
      app.configSwitching = false;
      elements.configSelect.disabled = !app.configs.length;
      renderBindingControls();
      renderGrippers();
      renderCheckpointControl();
      void flushGripperActualProbes();
    }
  }

  function stateFromPayload(payload) {
    if (!payload || typeof payload !== "object") return {};
    return payload.state && typeof payload.state === "object" ? payload.state : payload;
  }

  function stateSyncMetadata(state = app.state) {
    const value = state?.state_sync;
    return value && typeof value === "object" ? value : {};
  }

  function isActiveObservedTransition(value) {
    return ["pending", "acknowledged"].includes(String(value?.status || "").toLowerCase());
  }

  function observedTransition(scope) {
    const transitions = stateSyncMetadata().transitions;
    const value = transitions && typeof transitions === "object" ? transitions[scope] : null;
    return value && typeof value === "object" ? value : null;
  }

  function activeControlTransition(scope) {
    const local = app.controlTransitions.get(scope) || null;
    const observed = observedTransition(scope);
    const server = isActiveObservedTransition(observed) ? observed : null;
    if (!local) return server;
    if (!server || local.token === server.token) return local;
    // A newer stop must take visual and control precedence over an older start,
    // regardless of whether it originated in this tab or another tab.
    if (local.desired === "stopped") return local;
    if (server.desired === "stopped") return server;
    return local;
  }

  function controlTransitionDesired(scope, fallbackStatus = "") {
    const desired = String(activeControlTransition(scope)?.desired || "").toLowerCase();
    if (["running", "stopped"].includes(desired)) return desired;
    const status = String(fallbackStatus || "").toLowerCase();
    if (status === "starting") return "running";
    if (status === "stopping") return "stopped";
    return "";
  }

  function controlStateTransportStale() {
    const sync = stateSyncMetadata();
    const localTransportStale = Boolean(
      app.stateTransportFailed ||
      !app.stateTransportDeadlineAtMs ||
      Date.now() >= app.stateTransportDeadlineAtMs
    );
    return localTransportStale || sync.stale === true || sync.authoritative === false;
  }

  function controlSafetyFailClosed() {
    return Boolean(
      app.state.safety?.fail_closed === true &&
      !initializationOnlyFailClosed()
    );
  }

  function controlsFailClosed(action = "") {
    // A stop request only reduces authority and must remain available even when
    // the observation is stale. Starts and position/calibration requests fail
    // closed until a fresh authoritative snapshot arrives.
    return action !== "stop" && (
      controlStateTransportStale() || controlSafetyFailClosed()
    );
  }

  function workflowControlsFailClosed(stageName, action) {
    // FastWAM Server only loads a model and does not touch motion hardware.
    // It can therefore be started before DeviceManager/ActionExecutor. Keep
    // the authoritative state requirement, but do not inherit motion-device
    // fail-closed state from the runtime workflow.
    if (stageName === "server" && action === "start") {
      return controlStateTransportStale();
    }
    return controlsFailClosed(action);
  }

  function deviceInitializationInProgress(device) {
    const safety = device?.safety || {};
    const estop = safety.estop;
    const explicitSafetyFailure = Boolean(
      estop === true ||
      estop?.active === true ||
      safety.fault === true ||
      safety.operational === false ||
      safety.reduced === true
    );
    if (explicitSafetyFailure) return false;
    const phase = deviceProgressPhase(device);
    return Boolean(
      safety.initializing === true ||
      device?.status === "starting" ||
      (
        device?.desiredState === "running" &&
        device?.running === true &&
        device?.ready !== true &&
        phase !== "ready" &&
        DEVICE_START_PROGRESS_PHASES.has(phase)
      )
    );
  }

  function initializationOnlyFailClosed() {
    if (app.state.safety?.estop?.active === true) return false;
    const blocked = app.devices.filter(
      (device) => device?.safety?.fail_closed === true
    );
    return Boolean(
      blocked.length && blocked.every(deviceInitializationInProgress)
    );
  }

  function markStateTransportHealthy(sync = stateSyncMetadata()) {
    const staleAfterMs = Math.max(
      1500,
      asNumber(sync.stale_after_ms) || POLL_INTERVALS.stateWatchdog + 1500
    );
    const observedAgeMs = Math.max(0, asNumber(sync.age_ms) || 0);
    const remainingMs = Math.max(0, staleAfterMs - observedAgeMs);
    app.stateTransportFailed = remainingMs <= 0;
    app.stateTransportReason = remainingMs > 0 ? "" : "收到的设备观测已经过期";
    app.stateTransportDeadlineAtMs = Date.now() + remainingMs;
  }

  function markStateTransportStale(reason, { render = true } = {}) {
    app.stateTransportFailed = true;
    app.stateTransportReason = String(reason || "设备状态连接已中断");
    app.stateTransportDeadlineAtMs = 0;
    setServerStatus("offline", "真实设备状态连接中断");
    setGlobalNotice(`${app.stateTransportReason}；启动和控制按钮已安全锁定，停止操作仍可用。`);
    if (!render) return;
    renderBindingControls();
    renderWorkflow();
    renderGrippers();
    renderRecording();
  }

  function controlActionStillAuthorized(
    action,
    expectedEpoch = app.stateEpoch,
    { notify = true, workflowStage = "" } = {}
  ) {
    if (action === "stop") return true;
    const epochChanged = Boolean(
      expectedEpoch && app.stateEpoch && expectedEpoch !== app.stateEpoch
    );
    if (!epochChanged && !workflowControlsFailClosed(workflowStage, action)) return true;
    if (notify) {
      const message = epochChanged
        ? "控制服务状态代次已变化，请确认最新设备状态后重试。"
        : "真实设备状态不可确认；该操作已安全取消。";
      setGlobalNotice(message);
      showToast(message, "error");
    }
    scheduleStatePoll(0);
    return false;
  }

  async function ensureControlActionAuthorized(
    action,
    expectedEpoch = app.stateEpoch,
    workflowStage = ""
  ) {
    const initialAuthorization = workflowStage
      ? controlActionStillAuthorized(
          action,
          expectedEpoch,
          { notify: false, workflowStage },
        )
      : controlActionStillAuthorized(action, expectedEpoch, { notify: false });
    if (initialAuthorization) {
      return true;
    }
    const epochChanged = () => Boolean(
      expectedEpoch && app.stateEpoch && expectedEpoch !== app.stateEpoch
    );
    if (
      epochChanged()
      || (workflowStage !== "server" && controlSafetyFailClosed())
    ) {
      return controlActionStillAuthorized(action, expectedEpoch, { workflowStage });
    }

    // Discovery/binding can briefly hold the controller lock longer than the
    // state freshness window. Keep this action local and asynchronous while
    // waiting for the monitor's next authoritative sample. Real safety faults
    // and server epoch changes still cancel immediately.
    const recoveryDeadline = Date.now() + POLL_INTERVALS.controlRecoveryTimeout;
    while (Date.now() < recoveryDeadline) {
      await refreshState(false);
      if (controlActionStillAuthorized(
        action,
        expectedEpoch,
        { notify: false, workflowStage },
      )) {
        return true;
      }
      if (
        epochChanged() ||
        (workflowStage !== "server" && controlSafetyFailClosed()) ||
        !controlStateTransportStale()
      ) break;
      const remaining = recoveryDeadline - Date.now();
      if (remaining <= 0) break;
      await new Promise((resolve) => window.setTimeout(
        resolve,
        Math.min(POLL_INTERVALS.controlRecoveryRetry, remaining)
      ));
    }
    return controlActionStillAuthorized(action, expectedEpoch, { workflowStage });
  }

  function transitionToken() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `control-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function beginClientTransition(scope, action, desired, timeoutMs) {
    const value = {
      token: transitionToken(),
      scope,
      action,
      desired,
      timeoutMs,
      startedAtMs: Date.now(),
      epoch: app.stateEpoch,
    };
    app.controlTransitions.set(scope, value);
    return value;
  }

  function yieldForPendingPaint() {
    return new Promise((resolve) => {
      if (
        document.hidden ||
        typeof window.requestAnimationFrame !== "function"
      ) {
        window.setTimeout(resolve, 0);
        return;
      }
      window.requestAnimationFrame(() => window.setTimeout(resolve, 0));
    });
  }

  function clearClientTransition(scope, expectedToken = "") {
    const value = app.controlTransitions.get(scope);
    if (!value || (expectedToken && value.token !== expectedToken)) return null;
    app.controlTransitions.delete(scope);
    if (!app.controlRequests.has(scope)) {
      if (scope.startsWith("device:")) app.pendingActions.delete(scope.slice("device:".length));
      if (scope.startsWith("workflow:")) app.workflowPending.delete(scope.slice("workflow:".length));
    }
    return value;
  }

  function acceptStateSnapshot(nextState, requestSerial = 0) {
    const sync = stateSyncMetadata(nextState);
    const epoch = String(sync.epoch || "");
    const sequence = asNumber(sync.sequence);
    const eventRevision = asNumber(sync.event_revision);
    const observedRevision = asNumber(sync.observed_revision);
    if (requestSerial > 0 && requestSerial < app.stateAppliedRequestSerial) return false;
    if (epoch && app.stateEpoch && epoch !== app.stateEpoch) {
      if (requestSerial > 0 && requestSerial < app.stateAppliedRequestSerial) return false;
      app.controlTransitions.clear();
      app.controlRequests.clear();
      app.pendingActions.clear();
      app.workflowPending.clear();
      invalidateGripperSyncContext();
      app.stateSequence = -1;
      app.stateEventRevision = -1;
      app.stateObservedRevision = -1;
      app.stateStablePolls = 0;
    }
    if (epoch && epoch === app.stateEpoch) {
      const staleEvent = eventRevision !== null && eventRevision < app.stateEventRevision;
      const currentSync = stateSyncMetadata();
      const freshnessChanged = Boolean(
        sync.stale !== currentSync.stale ||
        sync.authoritative !== currentSync.authoritative ||
        sync.last_error !== currentSync.last_error
      );
      const duplicateOrOlder = Boolean(
        !freshnessChanged &&
        eventRevision !== null &&
        eventRevision === app.stateEventRevision &&
        sequence !== null &&
        sequence <= app.stateSequence
      );
      if (staleEvent || duplicateOrOlder) return false;
    }
    if (epoch) app.stateEpoch = epoch;
    app.stateAppliedRequestSerial = Math.max(app.stateAppliedRequestSerial, requestSerial);
    if (sequence !== null) app.stateSequence = sequence;
    if (eventRevision !== null) app.stateEventRevision = eventRevision;
    if (observedRevision !== null) {
      if (observedRevision === app.stateObservedRevision) app.stateStablePolls += 1;
      else app.stateStablePolls = 0;
      app.stateObservedRevision = observedRevision;
    }
    return true;
  }

  function observedOutcome(transition) {
    if (transition.scope.startsWith("device:")) {
      const key = transition.scope.slice("device:".length);
      const device = app.devices.find((item) => item.key === key);
      if (!device) return { status: "failed", reason: "设备已从当前配置消失" };
      const safetyReason = device.safety?.reason || device.reason || device.lastError;
      if (transition.desired === "running") {
        const initializing = deviceInitializationInProgress(device);
        if (device.safety?.fail_closed === true && !initializing) {
          return { status: "failed", reason: safetyReason || "设备安全状态已 fail-closed" };
        }
        if (device.status === "running" && device.ready) return { status: "completed" };
        if (!initializing && ["failed", "degraded", "stop_timeout", "orphaned", "external"].includes(device.status)) {
          return { status: "failed", reason: safetyReason || `设备状态：${device.status}` };
        }
      } else if (transition.desired === "stopped") {
        if (device.status === "stopped" && !device.running && !device.shm.connected) return { status: "completed" };
        if (["stop_timeout", "orphaned", "external", "failed"].includes(device.status)) {
          return { status: "failed", reason: safetyReason || `设备停止状态：${device.status}` };
        }
      }
      return { status: "pending" };
    }
    if (transition.scope.startsWith("workflow:")) {
      const name = transition.scope.slice("workflow:".length);
      const stage = workflowStage(name);
      const active = name === "runtime" ? Boolean(stage.active) : Boolean(stage.running);
      if (transition.desired === "running" && stage.status === "running" && active) {
        if (name !== "runtime" || stage.ready) return { status: "completed" };
      }
      if (transition.desired === "stopped" && stage.status === "stopped" && !active) {
        return { status: "completed" };
      }
      if (["failed", "error"].includes(String(stage.status || "").toLowerCase())) {
        return { status: "failed", reason: stage.reason || stage.error || `${name} 状态异常` };
      }
    }
    return { status: "pending" };
  }

  function reconcileControlTransitions(nowMs = Date.now()) {
    for (const [scope, local] of [...app.controlTransitions]) {
      if (local.epoch && app.stateEpoch && local.epoch !== app.stateEpoch) {
        clearClientTransition(scope, local.token);
        continue;
      }
      const server = observedTransition(scope);
      if (
        isActiveObservedTransition(server)
        && server.token !== local.token
        && server.desired === "stopped"
        && local.desired === "running"
      ) {
        clearClientTransition(scope, local.token);
        continue;
      }
      if (server && server.token === local.token) {
        const status = String(server.status || "").toLowerCase();
        if (["completed", "failed", "timed_out"].includes(status)) {
          clearClientTransition(scope, local.token);
          if (status !== "completed") {
            const message = server.reason || "设备未达到请求状态";
            recordControlFailure(scope, message);
          }
          continue;
        }
      }
      const outcome = observedOutcome(local);
      if (outcome.status === "completed") {
        clearClientTransition(scope, local.token);
        continue;
      }
      if (outcome.status === "failed" || nowMs - local.startedAtMs >= local.timeoutMs) {
        clearClientTransition(scope, local.token);
        const message = outcome.reason || "状态切换超时；已按真实设备状态恢复控件";
        recordControlFailure(scope, message);
      }
    }
  }

  function applyObservedState(
    nextState,
    devicePayload,
    {
      manual = false,
      requestSerial = ++app.stateRequestSerial,
      gripperEpoch = app.gripperSyncEpoch,
      gripperRequestId = nextGripperSnapshotRequest(),
    } = {}
  ) {
    if (!acceptStateSnapshot(nextState, requestSerial)) return false;
    app.initialStateObserved = true;
    const previousSelectedConfig = app.selectedConfig;
    const previousRevision = asNumber(app.state?.revision);
    const nextGrippers = nextState.grippers;
    const previousGrippers = app.state.grippers;
    app.state = { ...nextState };
    delete app.state.grippers;
    if (previousGrippers && typeof previousGrippers === "object") {
      app.state.grippers = previousGrippers;
    }
    markStateTransportHealthy(stateSyncMetadata(nextState));
    app.recordingObservedAtMs = Date.now();
    const selected = selectedConfigFromPayload(app.state);
    const selectedConfigChanged = selected !== null && setSelectedConfig(selected);
    const revisionChanged = previousRevision !== asNumber(app.state?.revision);
    if (!selectedConfigChanged && revisionChanged) {
      invalidateRuntimeTuning({ preserveDraft: app.currentConfigSavePending });
    }
    if (previousSelectedConfig && app.selectedConfig !== previousSelectedConfig) {
      app.checkpointOptimistic = null;
      invalidateCheckpointBrowser();
      app.inferenceOptionsMutationController?.abort();
    }
    if (
      app.checkpointMutationPending
      && app.checkpointMutationContext
      && !checkpointContextsEqual(
        app.checkpointMutationContext,
        checkpointContextFromState()
      )
    ) {
      app.checkpointMutationController?.abort();
    }
    if (
      app.inferenceOptionsMutationPending
      && !inferenceOptionsContextsEqual(
        app.inferenceOptionsMutationContext,
        inferenceOptionsContextFromState()
      )
    ) {
      app.inferenceOptionsMutationController?.abort();
    }
    if (Object.prototype.hasOwnProperty.call(app.state, "active_config")) {
      app.activeConfig = normalizeConfigName(app.state.active_config);
    } else if (Object.prototype.hasOwnProperty.call(app.state, "config")) {
      app.activeConfig = normalizeConfigName(app.state.config);
    }
    syncCameraSlots(nextState, { clear: !app.selectedConfig });
    syncRobotSlots(nextState, { clear: !app.selectedConfig });
    applyGripperSnapshot(nextGrippers, {
      epoch: gripperEpoch,
      requestId: gripperRequestId,
      controllerRevision: nextState.revision,
    });
    observeGripperCalibrationSync();
    observeGripperPositionSync();
    observeGripperProbeSync();
    syncGripperActualValues();
    driveGripperTargetSessions();
    syncConfigSelectToServer();

    app.devices = extractDevices(devicePayload || nextState.devices || []);
    reconcileControlTransitions();
    renderBindingControls();
    renderMetrics();
    renderWorkflow();
    renderPreflight();
    if (selectedConfigChanged || revisionChanged) {
      void loadRuntimeDiagnostics({ includeTuning: true });
      void loadPreflight();
    }
    renderRuntimeDiagnostics();
    renderGrippers();
    ensureDiagnosticsPolling();
    renderRecording();
    renderLogDeviceOptions();
    revealPendingFailureLog();
    void flushGripperActualProbes();

    const serverLabel = String(
      firstDefined(app.state.server?.name, app.state.server?.hostname, app.state.server_name, "控制服务在线")
    );
    const sync = stateSyncMetadata();
    const deviceFailure = currentDeviceFailureNotice();
    if (sync.stale === true || sync.authoritative === false) {
      setServerStatus("pending", "真实设备状态已过期");
      setGlobalNotice(sync.last_error || "正在等待新的设备观测；启动和控制按钮已安全锁定。停止操作仍可用。");
    } else {
      setServerStatus("online", serverLabel);
      if (
        app.state.safety?.fail_closed === true &&
        !initializationOnlyFailClosed()
      ) {
        setGlobalNotice(app.state.safety?.reason || "设备安全状态已触发 fail-closed；仅停止操作可用。");
      } else if (app.bindingErrors.size) {
        setGlobalNotice(`绑定未生效：${[...app.bindingErrors.values()][0]}`);
      } else if (app.bootstrapFailureNotice) {
        setGlobalNotice(app.bootstrapFailureNotice);
      } else if (deviceFailure) {
        setGlobalNotice(deviceFailure);
      } else if (app.controlFailureNotice) {
        setGlobalNotice(app.controlFailureNotice);
      } else {
        setGlobalNotice("");
      }
    }
    if (manual) showToast("状态已刷新", "success");
    return true;
  }

  async function refreshState(manual = false) {
    if (app.refreshing) {
      countPerf("overlappingPollsSkipped");
      return app.stateRefreshPromise;
    }
    app.refreshing = true;
    countPerf("statePolls");
    const requestSerial = ++app.stateRequestSerial;
    const controller = new AbortController();
    let requestTimedOut = false;
    const requestTimeout = window.setTimeout(() => {
      requestTimedOut = true;
      controller.abort();
    }, POLL_INTERVALS.stateRequestTimeout);
    const gripperEpoch = app.gripperSyncEpoch;
    const gripperRequestId = nextGripperSnapshotRequest();
    app.stateRequestController = controller;
    const request = (async () => {
      try {
        let statePayload = null;
        let stateError = null;
        try {
          statePayload = await api("/api/state", {
            signal: controller.signal,
            cache: "no-store",
          });
        } catch (error) {
          stateError = error;
        }

        // `/api/state` is the canonical snapshot and normally carries the full
        // device list. Keep `/api/devices` only as a compatibility fallback so
        // a normal poll is one request instead of downloading duplicate data.
        const nextState = statePayload ? stateFromPayload(statePayload) : null;
        let devicePayload = nextState?.devices;
        if (!Array.isArray(devicePayload)) {
          try {
            devicePayload = await api("/api/devices", { signal: controller.signal });
          } catch (deviceError) {
            if (stateError) throw stateError;
            throw deviceError;
          }
        }
        if (stateError && !devicePayload) throw stateError;
        if (!nextState) throw stateError || new ApiError("状态响应为空", 0, null);
        applyObservedState(nextState, devicePayload, {
          manual,
          requestSerial,
          gripperEpoch,
          gripperRequestId,
        });
      } catch (error) {
        if (error?.name === "AbortError" && !requestTimedOut) return;
        if (requestTimedOut) {
          markStateTransportStale("真实设备状态请求超时");
          return;
        }
        markStateTransportStale(errorMessage(error));
      } finally {
        window.clearTimeout(requestTimeout);
        if (app.stateRequestController === controller) app.stateRequestController = null;
        app.refreshing = false;
        app.stateRefreshPromise = null;
      }
    })();
    app.stateRefreshPromise = request;
    return request;
  }

  function initialStateAllowsBootstrap() {
    if (!app.initialStateObserved) return false;
    const selected = normalizeConfigName(
      firstDefined(app.state.selected_config, app.state.selected)
    );
    if (selected) return false;
    if (app.devices.some((device) => (
      ACTIVE_STATUSES.has(device.status) || device.status === "orphaned"
    ))) return false;

    const workflow = app.state.workflow || {};
    const activeWorkflowStatuses = new Set(["starting", "running", "stopping"]);
    const runtime = workflow.runtime || {};
    const inference = workflow.inference || {};
    const runtimeActive = Boolean(
      runtime.active || runtime.running || activeWorkflowStatuses.has(String(runtime.status || "").toLowerCase())
    );
    const inferenceActive = Boolean(
      inference.active || inference.running || activeWorkflowStatuses.has(String(inference.status || "").toLowerCase())
    );
    return !runtimeActive && !inferenceActive;
  }

  async function maybeAutoApplyDefaultConfig() {
    if (app.bootstrapAttempted) return { status: "already_attempted" };
    app.bootstrapAttempted = true;

    if (!initialStateAllowsBootstrap()) return { status: "skipped", reason: "state_not_idle" };
    const expectedRevision = firstDefined(app.state.revision, null);
    if (expectedRevision === null) return { status: "skipped", reason: "revision_unavailable" };
    const candidate = chooseBootstrapConfig(app.configs, app.recommendedConfig);
    if (!candidate) return { status: "skipped", reason: "config_unavailable" };

    app.configSwitching = true;
    elements.configSelect.disabled = true;
    renderGrippers();
    invalidateCheckpointBrowser();
    app.checkpointOptimistic = null;
    renderCheckpointControl();
    elements.configSelect.value = candidate;
    try {
      await loadConfigDetail(candidate, { throwOnError: true });
      await loadHardwareDiscovery({
        configName: candidate,
        preserve: false,
        throwOnError: false,
      });
      invalidateGripperSyncContext();
      const payload = await api("/api/config/select", {
        method: "POST",
        body: JSON.stringify({
          name: candidate,
          discovery_revision: app.discovery.revision,
          overrides: buildBindingOverrides(),
          stop_active: false,
          auto_apply: true,
          expected_revision: expectedRevision,
          expected_selected_config: null,
        }),
      });
      app.bindingDirty = false;
      app.bindingDirtyKeys.clear();
      app.bindingErrors.clear();
      const selected = selectedConfigFromPayload(stateFromPayload(payload));
      setSelectedConfig(selected === null ? candidate : selected);
      app.activeConfig = normalizeConfigName(payload?.active_config);
      syncConfigSelectToServer({ force: true });
      app.bootstrapFailureNotice = "";
      app.bootstrapManualChoiceRequired = false;
      setGlobalNotice("");
      showToast(`已自动应用默认配置 ${candidate}`, "success");
      await refreshState(false);
      requestGripperActualAlignment(GRIPPER_SIDES, "config_auto_applied");
      await loadConfigDetail(candidate);
      return { status: "applied", candidate, payload };
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        const conflictSelection = normalizeConfigName(error.payload?.selected_config);
        await refreshState(false);
        const currentSelection = normalizeConfigName(selectedConfigFromPayload(app.state))
          || conflictSelection
          || app.selectedConfig;
        if (currentSelection) {
          setSelectedConfig(currentSelection);
          app.activeConfig = normalizeConfigName(
            firstDefined(app.state.active_config, currentSelection)
          );
          app.configSelectionBeforeOpen = currentSelection;
          app.bootstrapFailureNotice = "";
          app.bootstrapManualChoiceRequired = false;
          syncConfigSelectToServer({ force: true });
          setGlobalNotice("");
          requestGripperActualAlignment(GRIPPER_SIDES, "config_auto_converged");
          await loadConfigDetail(currentSelection);
          return {
            status: "converged",
            candidate,
            selectedConfig: currentSelection,
          };
        }
      }
      const message = `自动应用默认配置 ${candidate} 失败：${errorMessage(error)}。请从下拉框手动选择。`;
      app.bootstrapFailureNotice = message;
      app.bootstrapManualChoiceRequired = true;
      setGlobalNotice(message);
      showToast(message, "error");
      elements.configSelect.value = "";
      elements.configSelect.selectedIndex = -1;
      app.configSelectionBeforeOpen = app.selectedConfig || "";
      return { status: "failed", candidate, error };
    } finally {
      app.configSwitching = false;
      elements.configSelect.disabled = !app.configs.length;
      renderBindingControls();
      renderGrippers();
      renderCheckpointControl();
      void flushGripperActualProbes();
    }
  }

  function shapeText(shape) {
    if (Array.isArray(shape)) return shape.join("×");
    if (shape && typeof shape === "object") return Object.values(shape).join("×");
    return shape ? String(shape) : "—";
  }

  function formatAge(ageMs) {
    const value = asNumber(ageMs);
    if (value === null) return "无帧";
    if (value < 1000) return `${Math.round(value)} ms`;
    if (value < 60000) return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)} s`;
    return `${Math.floor(value / 60000)} min`;
  }

  function shmHealthText(device) {
    const { shm } = device;
    if (shm.status === "healthy" || shm.healthy) return `新鲜 · ${formatAge(shm.ageMs)}`;
    if (shm.status === "stale") return `已过期 · ${formatAge(shm.ageMs)}`;
    if (shm.connected) return `已连接 · ${formatAge(shm.ageMs)}`;
    return "未连接";
  }

  function actionCapability(device, action) {
    const capabilities = device.capabilities;
    if (Array.isArray(capabilities)) return capabilities.includes(action);
    if (capabilities && typeof capabilities === "object" && typeof capabilities[action] === "boolean") {
      return capabilities[action];
    }
    if (device.status === "orphaned") return false;
    if (TRANSITION_STATUSES.has(device.status)) return false;
    if (action === "start") return device.status === "stopped" || device.status === "failed";
    if (action === "stop") return ACTIVE_STATUSES.has(device.status);
    if (action === "restart") return ["running", "degraded", "failed", "stop_timeout"].includes(device.status);
    return false;
  }

  function presentationStatus(status) {
    // Reuse the existing failed-state visual treatment for this safety-critical
    // state while retaining the precise lifecycle status in data attributes.
    return status === "stop_timeout" ? "failed" : status;
  }

  function deviceErrorText(device) {
    if (device.status === "stop_timeout") {
      const detail = device.lastError || "设备进程可能仍在运行；请确认现场安全后重试停止。";
      return `安全停机超时 · 最后错误：${detail}`;
    }
    return device.lastError ? `最后错误：${device.lastError}` : "";
  }

  function currentDeviceFailureNotice() {
    const device = app.devices.find((item) => (
      ERROR_STATUSES.has(item.status) && !deviceInitializationInProgress(item)
    ));
    if (!device) return "";
    const label = STATUS_LABELS[device.status] || device.status;
    const detail = device.lastError || device.safety?.reason || device.reason || "";
    return `${device.key} ${label}${detail ? `：${detail}` : ""}`;
  }

  function makeTextElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = text;
    return element;
  }

  function setText(element, value) {
    if (!element) return;
    const next = String(value ?? "");
    if (element.textContent !== next) element.textContent = next;
  }

  function setClassName(element, value) {
    if (element && element.className !== value) element.className = value;
  }

  function setStyleValue(element, property, value) {
    if (element && element.style[property] !== value) element.style[property] = value;
  }

  function setCheckpointPath(element, value, emptyText = "—") {
    if (!element) return;
    const path = String(value || "").trim();
    setText(element, path || emptyText);
    element.title = path;
  }

  function checkpointEditorLocked(
    checkpoint = checkpointStateForRender(),
    { independentServerStage = false } = {}
  ) {
    const ownerStage = checkpoint.ownerStage || "inference";
    const owner = workflowStage(ownerStage);
    const transition = observedTransition(`workflow:${ownerStage}`);
    const ownerTransitionPending = ownerStage === "inference"
      ? app.workflowPending.has("inference")
      : app.workflowPending.has(ownerStage);
    const ownerTransitioning = Boolean(
      ownerTransitionPending
      || isActiveObservedTransition(transition)
      || ["starting", "running", "stopping"].includes(
        String(owner.status || "").toLowerCase()
      )
      || owner.running === true
    );
    return Boolean(
      !checkpoint.configName
      || !checkpoint.canEdit
      || app.configSwitching
      || app.checkpointMutationPending
      || ownerTransitioning
      // A FastWAM Server start owns only the model checkpoint and endpoint;
      // hardware fail-closed state must not block that independent stage.
      || (independentServerStage
        ? controlStateTransportStale()
        : controlsFailClosed("checkpoint"))
    );
  }

  function inferenceOptionsEditorLocked(options = normalizeInferenceOptions()) {
    const inference = workflowStage("inference");
    const transition = observedTransition("workflow:inference");
    const inferenceTransitioning = Boolean(
      app.workflowPending.has("inference")
      || isActiveObservedTransition(transition)
      || ["starting", "running", "stopping"].includes(
        String(inference.status || "").toLowerCase()
      )
      || inference.running === true
    );
    return Boolean(
      !options.configName
      || !options.canEdit
      || app.configSwitching
      || app.inferenceOptionsMutationPending
      || inferenceTransitioning
      || controlsFailClosed("inference-options")
    );
  }

  function invalidateCheckpointBrowser({ close = true } = {}) {
    app.checkpointBrowserRequest += 1;
    app.checkpointBrowserController?.abort();
    app.checkpointBrowserController = null;
    app.checkpointBrowserContext = null;
    app.checkpointBrowserPayload = null;
    app.rolloutBrowserEnableOnSelect = false;
    if (close && elements.checkpointBrowserDialog?.open) {
      elements.checkpointBrowserDialog.close("cancel");
    }
  }

  function renderCheckpointControl() {
    const checkpoint = checkpointStateForRender();
    const inferenceOptions = normalizeInferenceOptions();
    setCheckpointPath(elements.checkpointDefaultPath, checkpoint.defaultPath);
    setCheckpointPath(elements.checkpointOverridePath, checkpoint.overridePath, "未设置");
    setCheckpointPath(elements.checkpointEffectivePath, checkpoint.effectivePath);
    if (elements.checkpointSourceBadge) {
      elements.checkpointSourceBadge.dataset.source = checkpoint.overrideActive ? "override" : "default";
      setText(elements.checkpointSourceBadge, checkpoint.overrideActive ? "临时 override" : "默认");
    }

    const locked = checkpointEditorLocked(checkpoint);
    if (elements.checkpointBrowseButton) elements.checkpointBrowseButton.disabled = locked;
    if (elements.checkpointClearButton) {
      elements.checkpointClearButton.disabled = locked || !checkpoint.overrideActive;
    }
    if (elements.checkpointStatus) {
      let message = "临时选择仅对当前 config 和本次服务会话生效。";
      let mode = "ok";
      if (app.checkpointMutationPending) {
        message = "正在更新 checkpoint，请稍候…";
        mode = "pending";
      } else if (!checkpoint.configName) {
        message = "请先选择运行配置。";
        mode = "pending";
      } else if (checkpoint.blockers.length) {
        message = checkpoint.blockers[0];
        mode = "pending";
      } else if (inferenceOptions.testMode && checkpoint.ownerStage === "inference") {
        message = "空壳推理模式不会读取 checkpoint；源 config 未修改。";
      } else if (checkpoint.error) {
        message = checkpoint.error;
        mode = "error";
      } else if (checkpoint.overrideActive) {
        message = checkpoint.ownerStage === "server"
          ? "FastWAM Server 启动时将使用临时 override；源 config 未修改。"
          : "推理启动时将使用临时 override；源 config 未修改。";
      }
      setText(elements.checkpointStatus, message);
      elements.checkpointStatus.dataset.status = mode;
    }

    if (
      elements.checkpointBrowserDialog?.open
      && app.checkpointBrowserMode === "checkpoint"
    ) {
      const current = checkpointContextFromState();
      if (
        !checkpointContextsEqual(app.checkpointBrowserContext, current)
        || locked
      ) {
        invalidateCheckpointBrowser();
        showToast("Checkpoint 状态已变化，请重新打开选择器。", "error");
      }
    }
  }

  function formatStorageBytes(value) {
    const bytes = asNumber(value);
    if (bytes === null) return "—";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let scaled = Math.max(0, bytes);
    let index = 0;
    while (scaled >= 1024 && index < units.length - 1) {
      scaled /= 1024;
      index += 1;
    }
    const digits = scaled >= 100 || index === 0 ? 0 : scaled >= 10 ? 1 : 2;
    return `${scaled.toFixed(digits)} ${units[index]}`;
  }

  function formatRolloutDuration(value) {
    const seconds = asNumber(value);
    if (seconds === null) return "无法估算";
    if (seconds < 60) return "不足 1 分钟";
    const minutes = Math.floor(seconds / 60);
    const days = Math.floor(minutes / 1440);
    const hours = Math.floor((minutes % 1440) / 60);
    const remainingMinutes = minutes % 60;
    if (days > 0) return `${days} 天 ${hours} 小时`;
    if (hours > 0) return `${hours} 小时 ${remainingMinutes} 分钟`;
    return `${minutes} 分钟`;
  }

  function renderRolloutStorageBudget(options) {
    const container = elements.rolloutStorageBudget;
    if (!container) return;
    if (!options.rolloutEnabled) {
      container.hidden = true;
      app.rolloutStorageAlertStatus = "disabled";
      return;
    }
    container.hidden = false;
    const storage = options.rolloutStorage;
    container.dataset.status = storage.status;
    const labels = {
      healthy: ["存储空间充足", "正常"],
      warning: ["Rollout 空间即将不足", "预警"],
      critical: ["Rollout 空间严重不足", "严重"],
      unavailable: ["无法确认 Rollout 空间", "未知"],
    };
    const [title, badge] = labels[storage.status] || labels.unavailable;
    setText(elements.rolloutStorageTitle, title);
    setText(elements.rolloutStorageBadge, badge);
    setText(elements.rolloutStorageFree, formatStorageBytes(storage.freeBytes));
    setText(
      elements.rolloutStorageRemaining,
      formatRolloutDuration(storage.estimatedRemainingSeconds)
    );
    const freePercent = storage.totalBytes
      ? Math.max(0, Math.min(100, storage.freeBytes / storage.totalBytes * 100))
      : 0;
    if (elements.rolloutStorageBar) {
      elements.rolloutStorageBar.style.width = `${freePercent.toFixed(1)}%`;
    }
    const rate = formatStorageBytes(storage.estimatedBytesPerSecond);
    const reserve = formatStorageBytes(storage.reserveBytes);
    const source = storage.estimateSource === "observed"
      ? `录制实测（${Math.trunc(storage.observedSamples)} 个样本）`
      : "当前 config 保守估算";
    setText(
      elements.rolloutStorageDetail,
      storage.error || `按 ${rate}/秒 · ${source}；预留 ${reserve} 不用于录制。`
    );

    const rank = { disabled: 0, unavailable: 0, healthy: 0, warning: 1, critical: 2 };
    if ((rank[storage.status] || 0) > (rank[app.rolloutStorageAlertStatus] || 0)) {
      showToast(
        storage.status === "critical"
          ? `Rollout 存储空间严重不足，预计仅可录 ${formatRolloutDuration(storage.estimatedRemainingSeconds)}。`
          : `Rollout 存储空间预警，预计可录 ${formatRolloutDuration(storage.estimatedRemainingSeconds)}。`,
        storage.status === "critical" ? "error" : "warning"
      );
    }
    app.rolloutStorageAlertStatus = storage.status;
  }

  function renderInferenceOptionsControl() {
    const options = normalizeInferenceOptions();
    const locked = inferenceOptionsEditorLocked(options);
    if (elements.testInferenceToggle) {
      const testLocked = locked || (!options.testMode && !options.mockSupported);
      elements.testInferenceToggle.disabled = testLocked;
      elements.testInferenceToggle.setAttribute("aria-checked", String(options.testMode));
      elements.testInferenceToggle.setAttribute(
        "aria-busy", String(app.inferenceOptionsMutationPending)
      );
      elements.testInferenceToggle.classList.toggle("is-on", options.testMode);
      elements.testInferenceToggle.classList.toggle(
        "is-busy", app.inferenceOptionsMutationPending
      );
      setText(
        elements.testInferenceToggle,
        app.inferenceOptionsMutationPending
          ? "更新中"
          : options.testMode ? "已开启" : "未开启"
      );
      elements.testInferenceToggle.title = testLocked
        ? options.blockers[0] || options.mockError || "当前状态不能修改推理模式"
        : options.testMode ? "关闭空壳推理测试" : "开启空壳推理测试";
    }
    if (elements.testInferenceStatus) {
      setText(
        elements.testInferenceStatus,
        options.testMode
          ? "不加载 checkpoint；双臂 TCP 保持，双夹爪同步往返"
          : options.mockSupported
            ? "保持双臂 TCP，仅同步往返开合夹爪"
            : options.mockError || "当前 config 不支持空壳推理"
      );
    }
    if (elements.rolloutToggle) {
      elements.rolloutToggle.disabled = locked;
      elements.rolloutToggle.setAttribute("aria-checked", String(options.rolloutEnabled));
      elements.rolloutToggle.setAttribute(
        "aria-busy", String(app.inferenceOptionsMutationPending)
      );
      elements.rolloutToggle.classList.toggle("is-on", options.rolloutEnabled);
      elements.rolloutToggle.classList.toggle(
        "is-busy", app.inferenceOptionsMutationPending
      );
      setText(
        elements.rolloutToggle,
        app.inferenceOptionsMutationPending
          ? "更新中"
          : options.rolloutEnabled ? "已开启" : "未开启"
      );
      elements.rolloutToggle.title = locked
        ? options.blockers[0] || "当前状态不能修改 rollout 录制"
        : options.rolloutEnabled ? "关闭 rollout 录制" : "选择保存目录并开启";
    }
    setCheckpointPath(
      elements.rolloutOutputPath,
      options.rolloutOutputRoot,
      "未选择"
    );
    if (elements.rolloutPathButton) {
      elements.rolloutPathButton.disabled = locked;
      setText(
        elements.rolloutPathButton,
        options.rolloutOutputRoot ? "更改目录" : "选择目录"
      );
      elements.rolloutPathButton.title = options.rolloutPathError || "选择服务器保存根目录";
    }
    renderRolloutStorageBudget(options);
    if (
      elements.checkpointBrowserDialog?.open
      && app.checkpointBrowserMode === "rollout"
    ) {
      const current = inferenceOptionsContextFromState();
      if (
        !inferenceOptionsContextsEqual(app.checkpointBrowserContext, current)
        || locked
      ) {
        invalidateCheckpointBrowser();
        showToast("推理录制设置已变化，请重新打开选择器。", "error");
      }
    }
  }

  function checkpointBrowserSetError(message = "") {
    if (!elements.checkpointBrowserError) return;
    elements.checkpointBrowserError.hidden = !message;
    setText(elements.checkpointBrowserError, message);
  }

  function setCheckpointBrowserInputPath(value = "") {
    if (!elements.checkpointBrowserPath) return;
    const path = String(value || "");
    elements.checkpointBrowserPath.value = path;
    elements.checkpointBrowserPath.title = path;
  }

  function checkpointBrowserInputPath() {
    return String(elements.checkpointBrowserPath?.value || "").trim();
  }

  function validateCheckpointBrowserInput(rawPath) {
    const path = String(rawPath || "").trim();
    if (!path) return { path: "", error: "请输入服务器目录路径。" };
    if (path.length > CHECKPOINT_BROWSER_PATH_MAX_LENGTH) {
      return { path, error: "路径过长，最多允许 4096 个字符。" };
    }
    if (/[\u0000-\u001f]/.test(path)) {
      return { path, error: "路径包含不可见控制字符，请重新粘贴。" };
    }
    if (!(path.startsWith("/") || path === "~" || path.startsWith("~/"))) {
      return { path, error: "请输入 Linux 服务器绝对路径，例如 /home/syb/recordings。" };
    }
    if (path.split("/").includes("..")) {
      return { path, error: "路径不能包含 ..。" };
    }
    return { path, error: "" };
  }

  function checkpointBrowserFriendlyError(error) {
    const message = errorMessage(error);
    const normalized = message.toLowerCase();
    if (normalized.includes("does not exist")) return "目录不存在，请检查路径后重试。";
    if (normalized.includes("outside the allowed browse roots")) {
      return "该路径不在允许访问的位置内，请使用页面提供的用户目录或挂载盘。";
    }
    if (normalized.includes("too deeply nested")) return "目录层级过深，请选择更上层的目录。";
    if (normalized.includes("must be a directory")) return "该路径不是目录。";
    if (normalized.includes("must be absolute")) return "请输入 Linux 服务器绝对路径。";
    if (normalized.includes("must not contain ..")) return "路径不能包含 ..。";
    if (
      normalized.includes("not readable")
      || normalized.includes("must be readable")
      || normalized.includes("unable to browse")
    ) {
      return "目录不可读取，请检查服务器权限。";
    }
    if (normalized.includes("path is invalid")) return "路径格式无效，请重新输入。";
    if (normalized.includes("no checkpoint browse roots")) {
      return "服务未配置可浏览的目录。";
    }
    return message;
  }

  function syncCheckpointBrowserDraftState() {
    const inputPath = checkpointBrowserInputPath();
    const payload = app.checkpointBrowserPayload;
    const loadedPath = String(payload?.path || "").trim();
    const matchesLoaded = Boolean(inputPath && loadedPath && inputPath === loadedPath);
    if (elements.checkpointBrowserSelect) {
      elements.checkpointBrowserSelect.disabled = !(
        matchesLoaded && payload?.current_selectable === true
      );
    }
    if (elements.checkpointBrowserGo) {
      elements.checkpointBrowserGo.disabled = !inputPath;
    }
    return matchesLoaded;
  }

  function navigateCheckpointBrowserFromInput() {
    const result = validateCheckpointBrowserInput(checkpointBrowserInputPath());
    if (result.error) {
      checkpointBrowserSetError(result.error);
      setText(elements.checkpointBrowserStatus, "路径尚未加载。修正后按 Enter 或点击“打开”。");
      syncCheckpointBrowserDraftState();
      return;
    }
    setCheckpointBrowserInputPath(result.path);
    void loadCheckpointBrowser(result.path);
  }

  function renderCheckpointBrowser(payload = app.checkpointBrowserPayload) {
    if (!payload || typeof payload !== "object") return;
    app.checkpointBrowserPayload = payload;
    const path = String(payload.path || "");
    setCheckpointBrowserInputPath(path);
    if (elements.checkpointBrowserUp) {
      elements.checkpointBrowserUp.disabled = !payload.parent;
      elements.checkpointBrowserUp.dataset.path = String(payload.parent || "");
    }
    if (elements.checkpointBrowserRoots) {
      const roots = Array.isArray(payload.roots) ? payload.roots.map(String) : [];
      const rootButtons = roots.map((root) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "checkpoint-browser-root-button";
        button.dataset.path = root;
        button.textContent = root === "/home"
          ? "用户目录 /home"
          : root === "/mnt" ? "挂载盘 /mnt" : root;
        const active = root === payload.root;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", String(active));
        button.addEventListener("click", () => void loadCheckpointBrowser(root));
        return button;
      });
      elements.checkpointBrowserRoots.replaceChildren(...rootButtons);
      elements.checkpointBrowserRoots.hidden = rootButtons.length < 2;
    }
    if (elements.checkpointBrowserList) {
      const entries = Array.isArray(payload.entries) ? payload.entries : [];
      const rows = entries.map((entry) => {
        const navigable = entry?.navigable === true;
        const selectable = entry?.selectable === true;
        const row = document.createElement(navigable || selectable ? "button" : "div");
        row.className = "checkpoint-browser-entry";
        if (navigable || selectable) {
          row.type = "button";
          row.dataset.path = String(entry.path || "");
          row.addEventListener("click", () => {
            if (navigable) {
              void loadCheckpointBrowser(row.dataset.path);
            } else if (app.checkpointBrowserMode === "checkpoint") {
              // FastWAM may use a concrete .pt file instead of a checkpoint
              // directory. Selecting it applies the same session override
              // endpoint used by the directory submit button.
              void mutateCheckpointOverride(row.dataset.path);
            }
          });
        }
        const kind = makeTextElement(
          "span",
          "checkpoint-browser-entry__kind",
          entry?.kind === "directory" ? "目录" : "文件"
        );
        const name = makeTextElement(
          "span",
          "checkpoint-browser-entry__name",
          String(entry?.name || "")
        );
        name.title = String(entry?.path || "");
        row.append(kind, name);
        if (entry?.selectable === true) {
          row.append(makeTextElement(
            "span",
            "checkpoint-browser-entry__badge",
            app.checkpointBrowserMode === "rollout" ? "可写" : "Checkpoint"
          ));
        }
        return row;
      });
      if (!rows.length) {
        rows.push(makeTextElement("p", "checkpoint-browser-empty", "当前目录没有可浏览条目。"));
      }
      elements.checkpointBrowserList.replaceChildren(...rows);
      elements.checkpointBrowserList.setAttribute("aria-busy", "false");
    }
    if (elements.checkpointBrowserSelect) {
      elements.checkpointBrowserSelect.disabled = payload.current_selectable !== true;
    }
    if (elements.checkpointBrowserGo) elements.checkpointBrowserGo.disabled = false;
    const requiresParams = payload.requires_params !== false;
    const status = payload.truncated === true
      ? `仅显示前 ${Array.isArray(payload.entries) ? payload.entries.length : 0} 项。`
      : payload.current_selectable === true
        ? app.checkpointBrowserMode === "rollout"
          ? "当前目录可写，可作为 rollout 保存根目录。"
          : requiresParams
            ? "当前目录是有效 checkpoint，可直接选择。"
            : "当前目录是有效 FastWAM checkpoint，可直接选择。"
        : app.checkpointBrowserMode === "rollout"
          ? "进入一个可写目录后才能选择。"
          : requiresParams
            ? "进入包含可读 params/ 的目录后才能选择。"
            : "进入 FastWAM checkpoint 根目录后才能选择。";
    setText(elements.checkpointBrowserStatus, status);
    checkpointBrowserSetError("");
  }

  async function loadCheckpointBrowser(path = "") {
    const context = app.checkpointBrowserContext;
    if (!context || !elements.checkpointBrowserDialog?.open) return;
    const requestId = ++app.checkpointBrowserRequest;
    app.checkpointBrowserController?.abort();
    const controller = new AbortController();
    app.checkpointBrowserController = controller;
    if (elements.checkpointBrowserList) {
      elements.checkpointBrowserList.setAttribute("aria-busy", "true");
    }
    if (elements.checkpointBrowserSelect) elements.checkpointBrowserSelect.disabled = true;
    if (elements.checkpointBrowserGo) elements.checkpointBrowserGo.disabled = true;
    setText(elements.checkpointBrowserStatus, "正在读取服务器目录…");
    checkpointBrowserSetError("");
    try {
      const query = path ? `?path=${encodeURIComponent(path)}` : "";
      const endpoint = app.checkpointBrowserMode === "rollout"
        ? "/api/rollout/browse"
        : "/api/checkpoints/browse";
      const payload = await api(`${endpoint}${query}`, {
        signal: controller.signal,
        cache: "no-store",
      });
      if (
        requestId !== app.checkpointBrowserRequest
        || app.checkpointBrowserController !== controller
        || !elements.checkpointBrowserDialog?.open
      ) return;
      const contextMatches = app.checkpointBrowserMode === "rollout"
        ? inferenceOptionsBrowsePayloadMatches(context, payload)
        : checkpointBrowsePayloadMatches(context, payload);
      if (!contextMatches) {
        invalidateCheckpointBrowser();
        showToast("配置上下文已变化，旧目录响应已丢弃。", "error");
        scheduleStatePoll(0);
        return;
      }
      renderCheckpointBrowser(payload);
    } catch (error) {
      if (error?.name === "AbortError") return;
      if (requestId !== app.checkpointBrowserRequest) return;
      checkpointBrowserSetError(checkpointBrowserFriendlyError(error));
      setText(elements.checkpointBrowserStatus, "路径未加载，请修改路径或返回已知目录重试。");
      if (elements.checkpointBrowserList) {
        elements.checkpointBrowserList.setAttribute("aria-busy", "false");
      }
    } finally {
      if (app.checkpointBrowserController === controller) {
        app.checkpointBrowserController = null;
      }
      if (elements.checkpointBrowserDialog?.open) syncCheckpointBrowserDraftState();
    }
  }

  function openServerPathBrowser(
    mode = "checkpoint",
    { enableRolloutOnSelect = false } = {}
  ) {
    const isRollout = mode === "rollout";
    const checkpoint = checkpointStateForRender();
    const options = normalizeInferenceOptions();
    if (
      isRollout
        ? inferenceOptionsEditorLocked(options)
        : checkpointEditorLocked(checkpoint)
    ) {
      const blockers = isRollout ? options.blockers : checkpoint.blockers;
      showToast(
        blockers[0] || `当前状态不能修改${isRollout ? "录制目录" : " checkpoint"}。`,
        "error"
      );
      return;
    }
    if (
      !elements.checkpointBrowserDialog
      || typeof elements.checkpointBrowserDialog.showModal !== "function"
    ) {
      showToast("当前浏览器不支持服务器文件选择窗口。", "error");
      return;
    }
    invalidateCheckpointBrowser({ close: false });
    app.checkpointBrowserMode = isRollout ? "rollout" : "checkpoint";
    app.rolloutBrowserEnableOnSelect = isRollout && enableRolloutOnSelect;
    app.checkpointBrowserContext = isRollout
      ? inferenceOptionsContextFromState()
      : checkpointContextFromState();
    if (app.checkpointBrowserContext.revision === null) {
      app.checkpointBrowserContext = null;
      showToast("控制器 revision 不可用，请刷新状态后重试。", "error");
      return;
    }
    setText(elements.checkpointBrowserEyebrow, "服务器目录浏览");
    setText(
      elements.checkpointBrowserTitle,
      isRollout ? "选择 Rollout 保存根目录" : "选择临时 checkpoint"
    );
    setText(
      elements.checkpointBrowserHint,
      isRollout
        ? "浏览或粘贴服务器目录；成功加载且可写后才能选择。录制会直接写入该目录。"
        : checkpointStateForRender().paramsRequired
          ? "浏览或粘贴服务器目录；仅可选择允许位置内、且包含可读 params/ 的 checkpoint，不会修改源 config。"
          : "浏览或粘贴服务器目录或权重文件；选择 FastWAM checkpoint 根目录或具体 .pt 文件，不要求 params/，不会修改源 config。"
    );
    setText(
      elements.checkpointBrowserSelect,
      isRollout
        ? app.rolloutBrowserEnableOnSelect
          ? "选择并开启录制"
          : "选择为保存及回放目录"
        : "选择当前目录"
    );
    setCheckpointBrowserInputPath("");
    if (elements.checkpointBrowserGo) elements.checkpointBrowserGo.disabled = true;
    elements.checkpointBrowserDialog.showModal();
    void loadCheckpointBrowser();
  }

  function openCheckpointBrowser() {
    openServerPathBrowser("checkpoint");
  }

  function openRolloutBrowser({ enableOnSelect = false } = {}) {
    openServerPathBrowser("rollout", { enableRolloutOnSelect: enableOnSelect });
  }

  async function mutateCheckpointOverride(path = null) {
    if (app.checkpointMutationPending) return;
    const context = checkpointContextFromState();
    const checkpoint = checkpointStateForRender();
    if (!context.selectedConfig || context.revision === null || checkpointEditorLocked(checkpoint)) {
      showToast(checkpoint.blockers[0] || "Checkpoint 上下文不可用，请刷新状态后重试。", "error");
      return;
    }
    if (path !== null && !String(path).trim()) return;
    if (path !== null) invalidateCheckpointBrowser();
    const requestId = ++app.checkpointMutationSerial;
    app.checkpointMutationController?.abort();
    const controller = new AbortController();
    app.checkpointMutationController = controller;
    app.checkpointMutationContext = context;
    app.checkpointMutationPending = true;
    renderCheckpointControl();
    try {
      const body = {
        expected_selected_config: context.selectedConfig,
        expected_revision: context.revision,
        expected_generation: context.generation,
      };
      if (path !== null) body.path = String(path);
      const payload = await api("/api/checkpoint/override", {
        method: path === null ? "DELETE" : "POST",
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (
        requestId !== app.checkpointMutationSerial
        || app.checkpointMutationController !== controller
      ) return;
      const response = checkpointContextFromPayload(payload);
      const current = checkpointContextFromState();
      const capturedStillCurrent = checkpointContextsEqual(context, current);
      const responseAlreadyObserved = Boolean(
        response.selectedConfig === context.selectedConfig
        && current.selectedConfig === response.selectedConfig
        && response.revision !== null
        && current.revision !== null
        && current.revision >= response.revision
        && current.generation === response.generation
      );
      if (!capturedStillCurrent && !responseAlreadyObserved) {
        invalidateCheckpointBrowser();
        scheduleStatePoll(0);
        await refreshState(false);
        return;
      }
      if (capturedStillCurrent) {
        app.checkpointOptimistic = {
          selectedConfig: response.selectedConfig,
          revision: response.revision,
          generation: response.generation,
          checkpoint: response.checkpoint,
        };
      }
      invalidateCheckpointBrowser();
      showToast(path === null ? "已清除临时 checkpoint override" : "已应用临时 checkpoint override", "success");
      scheduleStatePoll(0);
      await refreshState(false);
    } catch (error) {
      if (error?.name === "AbortError") {
        scheduleStatePoll(0);
        return;
      }
      const stale = error instanceof ApiError && error.status === 409;
      const message = stale
        ? "Checkpoint 或 config 状态已变化，已刷新，请重试。"
        : errorMessage(error);
      setGlobalNotice(message);
      showToast(message, "error");
      if (stale) await refreshState(false);
    } finally {
      if (requestId === app.checkpointMutationSerial) {
        if (app.checkpointMutationController === controller) {
          app.checkpointMutationController = null;
        }
        app.checkpointMutationContext = null;
        app.checkpointMutationPending = false;
        renderCheckpointControl();
      }
    }
  }

  function selectOptionSignature(items) {
    return JSON.stringify(items.map((item) => [
      item.value,
      item.label,
      item.disabled === true,
      item.title || "",
    ]));
  }

  function syncSelectOptions(select, items, selectedValue) {
    if (!select || bindingEditorActive(select)) return;
    const signature = selectOptionSignature(items);
    if (select.dataset.optionsSignature !== signature) {
      const options = items.map((item) => {
        const option = new Option(item.label, item.value);
        option.disabled = item.disabled === true;
        option.title = item.title || "";
        return option;
      });
      select.replaceChildren(...options);
      select.dataset.optionsSignature = signature;
      countPerf("optionReplacements");
    }
    const desired = items.some((item) => item.value === selectedValue && item.disabled !== true)
      ? selectedValue
      : "";
    if (select.value !== desired) select.value = desired;
  }

  function robotBindingLabel(robot) {
    const identity = robot.rdkSn || robot.model || "Rizon";
    return `${identity}${robot.detected ? "" : " · 未检测"}`;
  }

  function cameraBindingLabel(camera) {
    const detail = camera.serial || camera.kernelNode || camera.stablePath;
    return `${camera.name}${detail ? ` · ${detail}` : ""}${camera.detected && camera.capture ? "" : " · 未检测"}`;
  }

  function replaceRobotBindingOptions(select, side, selected, otherSelected, locked) {
    if (!select) return;
    const expectedVariant = side === "right" ? "4r" : "standard";
    const choices = app.discovery.robots.filter((robot) => robot.variant === expectedVariant);
    const items = [{
      label: app.hardwareLoading
        ? "正在发现…"
        : choices.some((item) => item.detected)
          ? "使用配置值"
          : "使用配置值（未检测到设备）",
      value: "",
      disabled: false,
    }, ...choices.map((robot) => ({
      label: robotBindingLabel(robot),
      value: robot.id,
      disabled: !robot.detected || robot.id === otherSelected,
    }))];
    syncSelectOptions(
      select,
      items,
      choices.some((item) => item.id === selected && item.detected) ? selected : ""
    );
    select.disabled = locked;
    select.title = locked
      ? "停止此设备后可修改绑定"
      : choices.length
        ? "选择临时 override，或使用 config 默认序列号"
        : "未发现可选序列号；可继续使用 config 默认值或重新扫描";
  }

  function deviceToggleAction(device) {
    if (!device || device.status === "orphaned") return "";
    const transition = activeControlTransition(`device:${device.key}`);
    // Once a stop supersedes an in-flight start, disable repeated clicks even
    // if the latest device sample still reports the previous `starting` state.
    if (transition?.desired === "stopped") return "";
    // A pending/acknowledged Start already owns or is trying to own hardware.
    // The only safe action at that point is Stop, and it must remain available
    // so a newer safety token can supersede the in-flight Start.
    if (
      device.status === "starting" ||
      (isActiveObservedTransition(transition) && transition?.desired === "running")
    ) return "stop";
    if (TRANSITION_STATUSES.has(device.status)) return "";
    // A startup failure is reported before the worker's finally block has
    // necessarily finished. Do not allow a second worker to start until the
    // failed process is confirmed gone; Stop remains available to finish the
    // cleanup immediately.
    if (device.status === "failed") return device.running ? "stop" : "start";
    if (device.status === "stopped") return "start";
    if (["running", "degraded", "stop_timeout"].includes(device.status)) return "stop";
    return "";
  }

  function controlsMatchActiveConfig() {
    const selected = elements.configSelect.value;
    const applied = app.selectedConfig;
    return !app.configSwitching && Boolean(applied) && selected === applied;
  }

  function deviceProgressPhase(device) {
    const reported = String(device?.progress?.phase || "").trim().toLowerCase();
    if (reported) return reported;
    const ownership = String(device?.ownership?.status || "").trim().toLowerCase();
    if (["worker_started", "preflight", "preflight_clear"].includes(ownership)) {
      return "checking_shm";
    }
    if (["claiming", "owned"].includes(ownership)) return "claiming_shm";
    return "";
  }

  function deviceProgressLabel(device, fallback = "", desired = "") {
    const phase = deviceProgressPhase(device);
    if (
      (desired === "running" && !DEVICE_START_PROGRESS_PHASES.has(phase)) ||
      (desired === "stopped" && !DEVICE_STOP_PROGRESS_PHASES.has(phase))
    ) {
      return String(fallback || "");
    }
    return DEVICE_PROGRESS_LABELS[phase] || String(fallback || "");
  }

  function updateDeviceToggle(button, device, label, bindingsReady) {
    if (!button) return;
    const scope = device ? `device:${device.key}` : "";
    const action = deviceToggleAction(device);
    const transition = scope ? activeControlTransition(scope) : null;
    const pending = Boolean(device && transition);
    const transitioning = Boolean(device && TRANSITION_STATUSES.has(device.status));
    const checked = Boolean(device && ACTIVE_STATUSES.has(device.status));
    button.dataset.deviceKey = device?.key || "";
    button.dataset.action = action;
    button.setAttribute("aria-checked", String(checked));
    button.setAttribute("aria-busy", String(pending || transitioning));
    button.classList.toggle("is-on", checked);
    button.classList.toggle("is-busy", pending || transitioning);
    if (!device) button.textContent = `${label} 未配置`;
    else if (pending) {
      const desired = controlTransitionDesired(scope, device.status);
      const fallback = desired === "stopped"
        ? (isI2rtHeadDevice(device) ? "归零确认中" : "停止中")
        : "启动中";
      const progressLabel = deviceProgressLabel(device, fallback, desired);
      button.textContent = desired === "running" && action === "stop"
        ? `${progressLabel} · 紧急停止`
        : progressLabel;
    }
    else if (device.status === "starting" && action === "stop") {
      button.textContent = `${deviceProgressLabel(device, "启动中", "running")} · 紧急停止`;
    }
    else if (TRANSITION_STATUSES.has(device.status)) button.textContent = STATUS_LABELS[device.status];
    else if (action === "stop" && isI2rtHeadDevice(device)) {
      button.textContent = device.status === "stop_timeout"
        ? "重试归零并断连 Yam 头臂"
        : "归零并断连 Yam 头臂";
    }
    else if (action === "stop") button.textContent = device.status === "stop_timeout" ? `重试停止 ${label}` : `停止 ${label}`;
    else button.textContent = `启动 ${label}`;
    button.title = (pending || transitioning)
      ? (
          action === "stop" && controlTransitionDesired(scope, device.status) === "running"
            ? `${deviceProgressLabel(device, "启动中", "running")}；点击立即发出停止请求`
            : deviceProgressLabel(
                device,
                STATUS_LABELS[device.status] || "处理中",
                pending ? controlTransitionDesired(scope, device.status) : ""
              )
        )
      : "";
    button.disabled = !action || (pending && action !== "stop") || controlsFailClosed(action) || !controlsMatchActiveConfig() || (action === "start" && !bindingsReady);
  }

  function robotSlotRequirement(slot) {
    if (!slot?.configured || !slot.deviceKey) {
      return { label: "未使用", kind: "unused" };
    }
    if (slot.inferenceRequired) {
      return { label: "推理必需", kind: "required" };
    }
    return { label: "可选", kind: "optional" };
  }

  function updateRobotSlotRequirement(element, slot) {
    if (!element) return;
    const requirement = robotSlotRequirement(slot);
    setText(element, requirement.label);
    if (element.dataset) element.dataset.kind = requirement.kind;
  }

  function renderRobotBindingControls() {
    const rizonSlot = app.robotSlots.rizon;
    const i2rtHeadSlot = app.robotSlots.i2rt_head;
    const configuredRobot = bindingDevices().find(
      (device) => device.key === rizonSlot.deviceKey
    ) || bindingDevices().find(isRizonDevice) || null;
    const runtimeRobot = app.devices.find(
      (device) => device.key === rizonSlot.deviceKey
    ) || app.devices.find(isRizonDevice) || null;
    const configuredI2rtHead = bindingDevices().find(
      (device) => device.key === i2rtHeadSlot.deviceKey
    ) || bindingDevices().find(isI2rtHeadDevice) || null;
    const runtimeI2rtHead = app.devices.find(
      (device) => device.key === i2rtHeadSlot.deviceKey
    ) || app.devices.find(isI2rtHeadDevice) || null;
    const locked = bindingTargetLocked(runtimeRobot);
    replaceRobotBindingOptions(
      elements.robotLeftBindingSelect,
      "left",
      app.robotBindings.left,
      app.robotBindings.right,
      locked
    );
    replaceRobotBindingOptions(
      elements.robotRightBindingSelect,
      "right",
      app.robotBindings.right,
      app.robotBindings.left,
      locked
    );
    // An empty side deliberately means "keep the serial from the base config".
    // Discovery is an optional session override, not a prerequisite for start.
    const bindingsReady = Boolean(configuredRobot);
    updateRobotSlotRequirement(elements.rizonDeviceRequirement, rizonSlot);
    updateRobotSlotRequirement(elements.i2rtHeadDeviceRequirement, i2rtHeadSlot);
    updateDeviceToggle(elements.robotDeviceToggle, runtimeRobot, "Rizon", bindingsReady);
    updateDeviceToggle(
      elements.i2rtHeadDeviceToggle,
      runtimeI2rtHead,
      "Yam 头臂",
      Boolean(configuredI2rtHead)
    );
    if (elements.hardwareDiscoveryRefresh) {
      elements.hardwareDiscoveryRefresh.disabled = Boolean(
        app.hardwareLoading ||
        app.configSwitching ||
        workflowLocksBindings()
      );
      elements.hardwareDiscoveryRefresh.classList.toggle("is-busy", app.hardwareLoading);
      elements.hardwareDiscoveryRefresh.title = app.hardwareLoading
        ? "正在扫描硬件"
        : "重新扫描摄像头和机械臂";
    }
  }

  function renderBindingControls() {
    renderRobotBindingControls();
    renderCameras();
    renderStageGuidance();
  }

  function stageGuidanceDeviceEntries() {
    const entries = [];
    const robotButtons = {
      rizon: elements.robotDeviceToggle,
      i2rt_head: elements.i2rtHeadDeviceToggle,
    };
    for (const role of ROBOT_SLOT_ORDER) {
      const slot = app.robotSlots[role];
      if (!slot?.configured || !slot.inferenceRequired || !slot.deviceKey) continue;
      entries.push({
        button: robotButtons[role],
        device: app.devices.find((item) => item.key === slot.deviceKey) || null,
      });
    }
    for (const role of CAMERA_SLOT_ORDER) {
      const slot = app.cameraSlots[role];
      if (!slot?.configured || !slot.inferenceRequired || !slot.deviceKey) continue;
      entries.push({
        button: app.cameraCards
          .get(role)
          ?.querySelector('[data-role="camera-device-toggle"]') || null,
        device: app.devices.find((item) => item.key === slot.deviceKey) || null,
      });
    }
    return entries;
  }

  function setStageGuidance(button, active) {
    if (!button) return;
    if (
      active
      && !button.disabled
      && button.dataset.action === "start"
    ) {
      button.dataset.guidance = "next";
      return;
    }
    button.removeAttribute("data-guidance");
  }

  function renderStageGuidance() {
    const deviceEntries = stageGuidanceDeviceEntries();
    const buttons = new Set([
      elements.robotDeviceToggle,
      elements.i2rtHeadDeviceToggle,
      elements.runtimeToggle,
      elements.serverToggle,
      elements.inferenceToggle,
      ...[...app.cameraCards.values()].map((card) => (
        card.querySelector('[data-role="camera-device-toggle"]')
      )),
      ...deviceEntries.map((entry) => entry.button),
    ]);
    for (const button of buttons) setStageGuidance(button, false);

    if (
      !app.selectedConfig
      || !app.state.effective_config
      || !app.state.workflow
      || !controlsMatchActiveConfig()
      || controlsFailClosed("start")
    ) return;

    const fastwam = app.state.workflow?.kind === "fastwam";
    const server = workflowStage("server");
    if (fastwam && !server.running) {
      setStageGuidance(elements.serverToggle, server.can_start === true);
    }

    const waitingForDevices = deviceEntries.filter(
      (entry) => entry.device?.ready !== true
    );
    if (waitingForDevices.length) {
      for (const entry of waitingForDevices) {
        setStageGuidance(
          entry.button,
          deviceToggleAction(entry.device) === "start"
        );
      }
      return;
    }

    const runtime = workflowStage("runtime");
    if (!runtime.ready) {
      setStageGuidance(elements.runtimeToggle, runtime.can_start === true);
      return;
    }

    const inference = workflowStage("inference");
    if (!inference.running && (!fastwam || server.ready === true || normalizeInferenceOptions().testMode)) {
      setStageGuidance(elements.inferenceToggle, inference.can_start === true);
    }
  }

  function renderMetrics() {
    const running = app.devices.filter((device) => ACTIVE_STATUSES.has(device.status)).length;
    const healthy = app.devices.filter((device) => device.shm.healthy || device.shm.status === "healthy").length;
    const errors = app.devices.filter((device) => ERROR_STATUSES.has(device.status)).length;
    setText(elements.metricConfigured, app.devices.length);
    setText(elements.metricRunning, running);
    setText(elements.metricHealthy, healthy);
    setText(elements.metricErrors, errors);
  }

  function workflowStage(name) {
    const stage = app.state.workflow?.[name];
    return stage && typeof stage === "object" ? stage : {};
  }

  function workflowBlockerText(stage) {
    if (!Array.isArray(stage?.blockers)) return "";
    return [...new Set(
      stage.blockers
        .map((item) => String(item || "").trim())
        .filter(Boolean),
    )].join("；");
  }

  function fastwamServerProgress(stage, fallback = "FastWAM 模型加载中") {
    const progress = stage?.progress && typeof stage.progress === "object"
      ? stage.progress
      : {};
    const phase = String(progress.phase || "").trim().toLowerCase();
    const label = FASTWAM_SERVER_PROGRESS_LABELS[phase] || fallback;
    const elapsedMs = asNumber(progress.elapsed_ms);
    const elapsed = elapsedMs !== null && elapsedMs >= 1000
      ? ` · ${Math.floor(elapsedMs / 1000)} 秒`
      : "";
    return { phase, button: label, status: `${label}${elapsed}` };
  }

  function ownedChildProcessPids() {
    const pids = [];
    const add = (value) => {
      const pid = Number(value);
      if (Number.isInteger(pid) && pid > 0) pids.push(pid);
    };
    for (const device of app.devices) add(device.pid);
    const workflow = app.state.workflow || {};
    add(workflow.runtime?.action_executor?.pid);
    add(workflow.runtime?.device_manager?.pid);
    add(workflow.server?.pid);
    add(workflow.inference?.pid);
    const gripperWorker = app.state.grippers?.worker || {};
    for (const pid of gripperWorker.pids || []) add(pid);
    return [...new Set(pids)];
  }

  function renderForceStopAll() {
    const button = elements.forceStopAll;
    if (!button) return;
    const serverPending = app.state.server?.force_stop_in_progress === true;
    const pending = app.forceStopPending || serverPending;
    const pids = ownedChildProcessPids();
    button.classList.toggle("is-busy", pending);
    button.textContent = pending ? "正在强制关闭" : "强制关闭所有进程";
    button.disabled = pending || pids.length === 0;
    if (elements.forceStopHint) {
      elements.forceStopHint.textContent = pending
        ? "正在强制终止控制面板拥有的子进程，网页服务会继续运行。"
        : pids.length
          ? `检测到 ${pids.length} 个控制面板拥有的子进程。强制关闭不会执行机械臂复位、优雅 Disable 或录制收尾。`
          : "当前没有控制面板拥有的活动子进程；外部占用进程不会被此按钮终止。";
    }
  }

  async function mutateInferenceOptions(kind, enabled, path = null) {
    if (app.inferenceOptionsMutationPending) return;
    const context = inferenceOptionsContextFromState();
    const options = normalizeInferenceOptions();
    if (
      !context.selectedConfig
      || context.revision === null
      || inferenceOptionsEditorLocked(options)
    ) {
      showToast(
        options.blockers[0] || "推理选项上下文不可用，请刷新后重试。",
        "error"
      );
      return;
    }
    if (kind === "test" && enabled && !options.mockSupported) {
      showToast(options.mockError || "当前 config 不支持空壳推理。", "error");
      return;
    }
    if (kind === "rollout" && path !== null && !String(path).trim()) return;
    if (kind === "rollout" && path !== null) invalidateCheckpointBrowser();
    const requestId = ++app.inferenceOptionsMutationSerial;
    app.inferenceOptionsMutationController?.abort();
    const controller = new AbortController();
    app.inferenceOptionsMutationController = controller;
    app.inferenceOptionsMutationContext = context;
    app.inferenceOptionsMutationPending = true;
    renderInferenceOptionsControl();
    try {
      const body = {
        enabled: enabled === true,
        expected_selected_config: context.selectedConfig,
        expected_revision: context.revision,
        expected_generation: context.generation,
      };
      if (kind === "rollout" && path !== null) body.path = String(path);
      const endpoint = kind === "test"
        ? "/api/inference/test-mode"
        : "/api/rollout/settings";
      await api(endpoint, {
        method: "POST",
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (
        requestId !== app.inferenceOptionsMutationSerial
        || app.inferenceOptionsMutationController !== controller
      ) return;
      showToast(
        kind === "test"
          ? `空壳推理测试已${enabled ? "开启" : "关闭"}`
          : path !== null && !enabled
            ? "Rollout 保存及回放目录已更新"
            : `Rollout 录制已${enabled ? "开启" : "关闭"}`,
        "success"
      );
      scheduleStatePoll(0);
      await refreshState(false);
    } catch (error) {
      if (error?.name === "AbortError") {
        scheduleStatePoll(0);
        return;
      }
      const stale = error instanceof ApiError && error.status === 409;
      const message = stale
        ? "推理选项或 config 状态已变化，已刷新，请重试。"
        : errorMessage(error);
      setGlobalNotice(message);
      showToast(message, "error");
      if (stale) await refreshState(false);
    } finally {
      if (requestId === app.inferenceOptionsMutationSerial) {
        if (app.inferenceOptionsMutationController === controller) {
          app.inferenceOptionsMutationController = null;
        }
        app.inferenceOptionsMutationContext = null;
        app.inferenceOptionsMutationPending = false;
        renderInferenceOptionsControl();
      }
    }
  }

  function renderWorkflowButton(button, stageName, stage, labels) {
    if (!button) return;
    const scope = `workflow:${stageName}`;
    const transition = activeControlTransition(scope);
    const pending = Boolean(transition);
    const running = stageName === "runtime" ? Boolean(stage.active) : Boolean(stage.running);
    const action = stage.can_stop ? "stop" : stage.can_start ? "start" : "";
    const status = String(stage.status || "").toLowerCase();
    const desired = controlTransitionDesired(scope, status);
    button.dataset.stage = stageName;
    button.dataset.action = action;
    button.classList.toggle("is-running", running);
    button.classList.toggle("is-busy", pending || ["starting", "stopping"].includes(status));
    if (desired === "stopped" && (pending || status === "stopping")) button.textContent = labels.stopping;
    else if (desired === "running" && (pending || status === "starting")) button.textContent = labels.starting;
    else if (action === "stop") button.textContent = labels.stop;
    else button.textContent = labels.start;
    button.disabled = !action
      || (pending && action !== "stop")
      || workflowControlsFailClosed(stageName, action)
      || app.configSwitching;
  }

  function renderWorkflow() {
    const runtime = workflowStage("runtime");
    const server = workflowStage("server");
    const inference = workflowStage("inference");
    const inferenceOptions = normalizeInferenceOptions();
    const workflow = app.state.workflow || {};
    const fastwam = workflow.kind === "fastwam";
    const serverProgress = fastwamServerProgress(server);
    if (elements.serverControl) elements.serverControl.hidden = !fastwam;
    renderWorkflowButton(elements.runtimeToggle, "runtime", runtime, {
      start: "启动运行时",
      starting: "运行时启动中",
      stop: "停止运行时",
      stopping: "运行时停止中",
    });
    renderWorkflowButton(elements.inferenceToggle, "inference", inference, {
      start: inferenceOptions.testMode ? "开始空壳推理" : "开始推理",
      starting: inferenceOptions.testMode ? "空壳推理启动中" : "推理启动中",
      stop: inferenceOptions.testMode ? "停止空壳推理" : "停止推理",
      stopping: inferenceOptions.testMode ? "空壳推理停止中" : "推理停止中",
    });
    if (fastwam) {
      renderWorkflowButton(elements.serverToggle, "server", server, {
        start: "启动模型 Server",
        starting: serverProgress.button,
        stop: "停止模型 Server",
        stopping: "FastWAM Server 停止中",
      });
      if (elements.serverToggle) {
        elements.serverToggle.title = server.status === "starting"
          ? serverProgress.status
          : "";
      }
    } else if (elements.serverToggle) {
      elements.serverToggle.disabled = true;
      elements.serverToggle.removeAttribute("data-stage");
      elements.serverToggle.removeAttribute("data-action");
    }

    const runtimeStatus = WORKFLOW_STATUS_LABELS[runtime.status] || "未配置";
    const inferenceStatus = WORKFLOW_STATUS_LABELS[inference.status] || "未配置";
    if (elements.runtimeStatus) {
      elements.runtimeStatus.textContent = runtime.ready
        ? "DeviceManager 与 ActionExecutor 已就绪"
        : runtime.error || runtime.blockers?.[0] || runtimeStatus;
    }
    if (elements.inferenceStatus) {
      elements.inferenceStatus.textContent = inference.running
        ? inferenceOptions.testMode
          ? "空壳推理正在运行：TCP 保持，夹爪同步往返"
          : `${workflow.inference_label || "Policy"} inference 正在运行`
        : inference.error || inference.blockers?.[0] || inferenceStatus;
    }
    if (fastwam && elements.serverStatus) {
      const serverStatus = WORKFLOW_STATUS_LABELS[server.status] || "未配置";
      const serverBlockers = workflowBlockerText(server);
      elements.serverStatus.textContent = server.running
        ? `FastWAM 模型 Server 已就绪 (${server.endpoint || ""})`
        : server.status === "starting"
          ? `${serverProgress.status} (${server.endpoint || "端口分配中"})`
          : server.error || serverBlockers || serverStatus;
    }

    const blockers = [];
    if (!runtime.ready && !runtime.active) {
      for (const item of runtime.blockers || []) blockers.push(`运行时：${item}`);
    }
    if (!inference.running) {
      for (const item of inference.blockers || []) blockers.push(`推理：${item}`);
    }
    if (fastwam && !server.running) {
      for (const item of server.blockers || []) blockers.push(`Server：${item}`);
    }
    if (elements.workflowBlockers) {
      const unique = [...new Set(blockers)];
      elements.workflowBlockers.hidden = unique.length === 0;
      elements.workflowBlockers.textContent = unique.join("\n");
    }

    const profiles = app.state.workflow?.profiles || {};
    if (elements.workflowProfile) {
      const runtimeProfile = profiles.runtime?.name || "device";
      const inferenceProfile = profiles.inference?.name || "openpi";
      const inferenceLabel = inferenceOptions.testMode ? "test-shell" : inferenceProfile;
      elements.workflowProfile.textContent = fastwam
        ? `${runtimeProfile} / ${profiles.server?.name || "fastwam-server"} / ${inferenceLabel}`
        : `${runtimeProfile} / ${inferenceLabel}`;
    }
    renderCheckpointControl();
    renderInferenceOptionsControl();
    renderForceStopAll();
    renderStageGuidance();
  }

  function gripperBlockers(...values) {
    const blockers = [];
    for (const value of values) {
      if (Array.isArray(value)) blockers.push(...value);
      else if (value) blockers.push(value);
    }
    return [...new Set(blockers.map((item) => String(item || "").trim()).filter(Boolean))];
  }

  function normalizeGripperCapability(value, fallbackBlockers = []) {
    if (value === true) return { can: true, blockers: [] };
    if (value === false) return { can: false, blockers: gripperBlockers(fallbackBlockers) };
    if (!value || typeof value !== "object") {
      return { can: false, blockers: gripperBlockers(fallbackBlockers) };
    }
    return {
      can: firstDefined(value.can, value.allowed, value.enabled) === true,
      blockers: gripperBlockers(value.blockers, value.reasons, value.reason, fallbackBlockers),
    };
  }

  function gripperCapabilityValue(source, action) {
    if (!source || typeof source !== "object") return undefined;
    const capabilities = source.capabilities && typeof source.capabilities === "object"
      ? source.capabilities
      : {};
    return firstDefined(
      capabilities[action],
      capabilities[`can_${action}`],
      source[`can_${action}`],
      action === "move" ? capabilities.position : undefined,
      action === "move" ? capabilities.can_position : undefined,
      action === "move" ? source.can_position : undefined
    );
  }

  function gripperPercent(source, keys) {
    for (const key of keys) {
      const raw = source?.[key];
      if (raw === undefined || raw === null || raw === "") continue;
      const value = asNumber(raw);
      if (value !== null) return Math.min(100, Math.max(0, value));
    }
    return null;
  }

  function localizeGripperEndpoint(value) {
    const text = String(value || "").trim();
    if (!text) return "";
    if (/^(closed|close)$/i.test(text)) return "闭合";
    if (/^(open|opened)$/i.test(text)) return "张开";
    if (/^(minimum|min)$/i.test(text)) return "最小";
    if (/^(maximum|max)$/i.test(text)) return "最大";
    return text;
  }

  function gripperEndpointLabel(source, sideSource, percent) {
    const key = percent === 0 ? "zero" : "hundred";
    const alternateKey = percent === 0 ? "min" : "max";
    const label = firstDefined(
      sideSource?.position_labels?.[key],
      sideSource?.position_semantics?.[key],
      sideSource?.[`${key}_label`],
      sideSource?.[`${alternateKey}_label`],
      source?.position_labels?.[key],
      source?.position_semantics?.[key],
      source?.[`${key}_label`],
      source?.[`${alternateKey}_label`]
    );
    const localized = localizeGripperEndpoint(label);
    if (!localized) return `${percent}% ${percent === 0 ? "闭合" : "张开"}`;
    return localized.includes("%") ? localized : `${percent}% ${localized}`;
  }

  function normalizeGrippers(raw = app.state.grippers) {
    const source = raw && typeof raw === "object" ? raw : {};
    const sourceStatus = String(source.status || "").trim().toLowerCase();
    const explicitBusySides = new Set(
      (Array.isArray(source.busy_sides) ? source.busy_sides : [])
        .map((side) => String(side).trim().toLowerCase())
        .filter((side) => GRIPPER_SIDES.includes(side))
    );
    const availablePorts = [...new Set(
      (Array.isArray(source.available_ports) ? source.available_ports : [])
        .map((port) => String(port))
        .filter((port) => /^\/dev\/ttyUSB[0-9]+$/.test(port))
    )].sort((left, right) => {
      const leftIndex = Number(left.replace("/dev/ttyUSB", ""));
      const rightIndex = Number(right.replace("/dev/ttyUSB", ""));
      return leftIndex - rightIndex || left.localeCompare(right);
    });
    const sharedBlockers = gripperBlockers(source.blockers, source.reasons, source.reason);
    const sharedCapabilities = {
      calibrate: normalizeGripperCapability(
        gripperCapabilityValue(source, "calibrate"),
        sharedBlockers
      ),
      move: normalizeGripperCapability(
        gripperCapabilityValue(source, "move"),
        sharedBlockers
      ),
      queuePosition: normalizeGripperCapability(
        firstDefined(
          gripperCapabilityValue(source, "queue_position"),
          gripperCapabilityValue(source, "move")
        ),
        sharedBlockers
      ),
      probe: normalizeGripperCapability(
        gripperCapabilityValue(source, "probe"),
        sharedBlockers
      ),
    };
    const sides = {};
    for (const side of GRIPPER_SIDES) {
      const sideSource = firstDefined(source.sides?.[side], source[side], {});
      const rawStatus = String(sideSource.status || source.status || "unavailable").trim().toLowerCase();
      const status = Object.prototype.hasOwnProperty.call(GRIPPER_STATUS_LABELS, rawStatus)
        ? rawStatus
        : "unavailable";
      const sideBlockers = gripperBlockers(
        sideSource.blockers,
        sideSource.reasons,
        sideSource.reason,
        sharedBlockers
      );
      const calibrateValue = gripperCapabilityValue(sideSource, "calibrate");
      const moveValue = gripperCapabilityValue(sideSource, "move");
      const queuePositionValue = gripperCapabilityValue(sideSource, "queue_position");
      const probeValue = gripperCapabilityValue(sideSource, "probe");
      const calibrationArtifact = firstDefined(
        sideSource.calibration_artifact,
        sideSource.calibrationArtifact,
        null
      );
      const sideBusy = Boolean(
        sideSource.busy === true ||
        explicitBusySides.has(side) ||
        ["moving", "queued", "probing"].includes(status)
      );
      if (sideBusy) explicitBusySides.add(side);
      sides[side] = {
        source: sideSource,
        status,
        configured: sideSource.configured === true,
        calibrationArtifact: calibrationArtifact && typeof calibrationArtifact === "object"
          ? calibrationArtifact
          : null,
        revision: firstDefined(sideSource.revision, source.side_revisions?.[side], null),
        port: /^\/dev\/ttyUSB[0-9]+$/.test(String(sideSource.port || ""))
          ? String(sideSource.port)
          : "",
        portPresent: sideSource.port_present === true,
        actualPercent: gripperPercent(sideSource, [
          "actual_percent", "position_percent", "current_percent", "opening_percent",
        ]),
        targetPercent: gripperPercent(sideSource, ["target_percent", "commanded_percent"]),
        minLabel: gripperEndpointLabel(source, sideSource, 0),
        maxLabel: gripperEndpointLabel(source, sideSource, 100),
        error: String(firstDefined(sideSource.error, sideSource.last_error, "")).trim(),
        busy: sideBusy,
        capabilities: {
          calibrate: calibrateValue === undefined
            ? sharedCapabilities.calibrate
            : normalizeGripperCapability(calibrateValue, sideBlockers),
          move: moveValue === undefined
            ? sharedCapabilities.move
            : normalizeGripperCapability(moveValue, sideBlockers),
          queuePosition: queuePositionValue === undefined
            ? (moveValue === undefined
                ? sharedCapabilities.queuePosition
                : normalizeGripperCapability(moveValue, sideBlockers))
            : normalizeGripperCapability(queuePositionValue, sideBlockers),
          probe: probeValue === undefined
            ? sharedCapabilities.probe
            : normalizeGripperCapability(probeValue, sideBlockers),
        },
      };
    }
    const explicitCalibrate = gripperCapabilityValue(source, "calibrate");
    const calibrate = explicitCalibrate === undefined
      ? {
          can: GRIPPER_SIDES.every((side) => sides[side].capabilities.calibrate.can),
          blockers: gripperBlockers(...GRIPPER_SIDES.map((side) => (
            sides[side].capabilities.calibrate.blockers
          ))),
        }
      : sharedCapabilities.calibrate;
    const globalBusy = Boolean(
      source.global_busy === true ||
      sourceStatus === "calibrating" ||
      GRIPPER_SIDES.some((side) => sides[side].status === "calibrating") ||
      (source.busy === true && explicitBusySides.size === 0)
    );
    const permissionSource = source.permission && typeof source.permission === "object"
      ? source.permission
      : {};
    const permissionAffectedPorts = [...new Set(
      (Array.isArray(permissionSource.affected_ports) ? permissionSource.affected_ports : [])
        .map((port) => String(port))
        .filter((port) => /^\/dev\/ttyUSB[0-9]+$/.test(port))
    )].sort((left, right) => {
      const leftIndex = Number(left.replace("/dev/ttyUSB", ""));
      const rightIndex = Number(right.replace("/dev/ttyUSB", ""));
      return leftIndex - rightIndex || left.localeCompare(right);
    });
    const permissionTargetPorts = [...new Set(
      (Array.isArray(permissionSource.target_ports) ? permissionSource.target_ports : [])
        .map((port) => String(port))
        .filter((port) => /^\/dev\/ttyUSB[0-9]+$/.test(port))
    )].sort((left, right) => {
      const leftIndex = Number(left.replace("/dev/ttyUSB", ""));
      const rightIndex = Number(right.replace("/dev/ttyUSB", ""));
      return leftIndex - rightIndex || left.localeCompare(right);
    });
    const permissionAffectedCanInterfaces = [...new Set(
      (Array.isArray(permissionSource.affected_can_interfaces)
        ? permissionSource.affected_can_interfaces
        : [])
        .map((interfaceName) => String(interfaceName))
        .filter((interfaceName) => /^can[0-9]+$/.test(interfaceName))
    )].sort();
    const permissionTargetCanInterfaces = [...new Set(
      (Array.isArray(permissionSource.target_can_interfaces)
        ? permissionSource.target_can_interfaces
        : [])
        .map((interfaceName) => String(interfaceName))
        .filter((interfaceName) => /^can[0-9]+$/.test(interfaceName))
    )].sort();
    const unavailableCanInterfaces = [...new Set(
      (Array.isArray(permissionSource.unavailable_can_interfaces)
        ? permissionSource.unavailable_can_interfaces
        : [])
        .map((interfaceName) => String(interfaceName))
        .filter((interfaceName) => /^can[0-9]+$/.test(interfaceName))
    )].sort();
    return {
      revision: firstDefined(source.revision, app.state.revision, null),
      status: sourceStatus,
      pendingCommand: firstDefined(source.pending_command, null),
      busy: source.busy === true,
      globalBusy,
      busySides: explicitBusySides,
      progress: source.progress && typeof source.progress === "object" ? source.progress : null,
      blockers: sharedBlockers,
      bindingBlockers: gripperBlockers(source.binding_blockers),
      canBindPorts: source.can_bind_ports === true,
      availablePorts,
      error: String(firstDefined(source.error, source.last_error, "")).trim(),
      permission: {
        contextId: String(firstDefined(
          permissionSource.context_id,
          permissionSource.contextId,
          ""
        )).trim().slice(0, 256),
        required: permissionSource.required === true && Boolean(
          permissionAffectedPorts.length || permissionAffectedCanInterfaces.length
        ),
        targetPorts: permissionTargetPorts,
        affectedPorts: permissionAffectedPorts,
        targetCanInterfaces: permissionTargetCanInterfaces,
        affectedCanInterfaces: permissionAffectedCanInterfaces,
        unavailableCanInterfaces,
        expectedCanBitrate: asNumber(permissionSource.expected_can_bitrate) || 1000000,
        inspectionError: String(permissionSource.inspection_error || "").trim(),
      },
      capabilities: { ...sharedCapabilities, calibrate },
      sides,
    };
  }

  function gripperPermissionSignature(permission) {
    return JSON.stringify({
      contextId: String(permission?.contextId || ""),
      targetPorts: permission?.targetPorts || [],
      affectedPorts: permission?.affectedPorts || [],
      targetCanInterfaces: permission?.targetCanInterfaces || [],
      affectedCanInterfaces: permission?.affectedCanInterfaces || [],
    });
  }

  function setGripperPermissionDialogMode(mode, permission) {
    const nextMode = mode;
    app.gripperPermissionMode = nextMode;
    const ports = permission?.targetPorts || [];
    const affected = permission?.affectedPorts || [];
    const canInterfaces = permission?.targetCanInterfaces || [];
    const affectedCanInterfaces = permission?.affectedCanInterfaces || [];
    const summaries = [];
    if (ports.length) {
      summaries.push(`夹爪串口：${ports.join("、")}（缺少权限：${affected.join("、") || "无"}）`);
    }
    if (canInterfaces.length) {
      summaries.push(`Yam CAN：${canInterfaces.join("、")}（需要配置：${affectedCanInterfaces.join("、") || "无"}）`);
    }
    if (elements.gripperPermissionSummary) {
      elements.gripperPermissionSummary.textContent = summaries.length
        ? `本次仅处理当前 config 的目标：${summaries.join("；")}。绑定或 config 变化后本次确认会失效。`
        : "当前设备权限上下文不可用，请关闭弹窗并刷新状态。";
    }
    if (elements.gripperPermissionWarning) {
      const operations = [];
      if (affected.length) operations.push("ttyUSB 将执行固定的 sudo chmod 0666");
      if (affectedCanInterfaces.length) {
        operations.push(`CAN 将执行固定的 sudo ip link down/up，并设置 bitrate=${permission.expectedCanBitrate || 1000000}`);
      }
      elements.gripperPermissionWarning.textContent = `权限变更：${operations.join("；")}。sudo 密码仅通过本次请求的 stdin 使用且不会保存。`;
    }
    const showPassword = nextMode === "sudo";
    if (elements.gripperPasswordField) elements.gripperPasswordField.hidden = !showPassword;
    if (elements.gripperSudoPassword) {
      elements.gripperSudoPassword.disabled = !showPassword || app.gripperPermissionPending;
      if (!showPassword) elements.gripperSudoPassword.value = "";
    }
    if (elements.gripperPermissionSubmit) {
      elements.gripperPermissionSubmit.textContent = nextMode === "stale"
        ? "绑定已变化，请关闭重开"
        : "使用 sudo 授权";
      elements.gripperPermissionSubmit.disabled = app.gripperPermissionPending || nextMode === "stale";
    }
    if (showPassword && !app.gripperPermissionPending) {
      window.setTimeout(() => elements.gripperSudoPassword?.focus(), 0);
    }
  }

  function configureGripperPermissionDialog(permission) {
    const mode = "sudo";
    if (elements.gripperPermissionError) {
      elements.gripperPermissionError.textContent = "";
      elements.gripperPermissionError.hidden = true;
    }
    setGripperPermissionDialogMode(mode, permission);
  }

  function settleGripperPermissionPrompt(authorized) {
    if (elements.gripperSudoPassword) elements.gripperSudoPassword.value = "";
    if (elements.gripperPermissionDialog?.open) elements.gripperPermissionDialog.close();
    const resolve = app.gripperPermissionPromptResolve;
    app.gripperPermissionPromptResolve = null;
    app.gripperPermissionPromptPromise = null;
    app.gripperPermissionPromptSignature = "";
    app.gripperPermissionContext = null;
    if (typeof resolve === "function") resolve(Boolean(authorized));
  }

  function cancelGripperPermissionPrompt() {
    if (app.gripperPermissionPending) return;
    app.gripperPermissionDismissedSignature = app.gripperPermissionPromptSignature;
    settleGripperPermissionPrompt(false);
  }

  function requestGripperPermission({ force = false } = {}) {
    const grippers = normalizeGrippers();
    const permission = grippers.permission;
    if (!permission.required) return Promise.resolve(true);
    if (
      !Number.isInteger(grippers.revision)
      || !(permission.targetPorts.length || permission.targetCanInterfaces.length)
    ) {
      return Promise.resolve(false);
    }
    const signature = gripperPermissionSignature(permission);
    if (!force && signature === app.gripperPermissionDismissedSignature) {
      return Promise.resolve(false);
    }
    if (app.gripperPermissionPromptPromise) return app.gripperPermissionPromptPromise;
    if (!elements.gripperPermissionDialog || typeof elements.gripperPermissionDialog.showModal !== "function") {
      return Promise.resolve(false);
    }
    app.gripperPermissionPromptSignature = signature;
    app.gripperPermissionContext = Object.freeze({
      revision: grippers.revision,
      ports: Object.freeze([...permission.targetPorts]),
      canInterfaces: Object.freeze([...permission.targetCanInterfaces]),
    });
    configureGripperPermissionDialog(permission);
    app.gripperPermissionPromptPromise = new Promise((resolve) => {
      app.gripperPermissionPromptResolve = resolve;
    });
    elements.gripperPermissionDialog.showModal();
    return app.gripperPermissionPromptPromise;
  }

  function maybePromptGripperPermission(permission) {
    if (!permission?.required) {
      app.gripperPermissionDismissedSignature = "";
      return;
    }
    if (
      app.gripperPermissionPending ||
      app.gripperPermissionPromptPromise ||
      app.gripperPermissionAutoScheduled ||
      gripperPermissionSignature(permission) === app.gripperPermissionDismissedSignature
    ) return;
    app.gripperPermissionAutoScheduled = true;
    window.setTimeout(() => {
      app.gripperPermissionAutoScheduled = false;
      void requestGripperPermission();
    }, 0);
  }

  async function submitGripperPermissionRepair() {
    if (app.gripperPermissionPending || app.gripperPermissionMode === "stale") return;
    const permission = normalizeGrippers().permission;
    const context = app.gripperPermissionContext;
    const csrfToken = gripperPermissionCsrfToken();
    if (
      !csrfToken
      || !context
      || !Number.isInteger(context.revision)
      || !(context.ports.length || context.canInterfaces.length)
    ) {
      if (elements.gripperPermissionError) {
        elements.gripperPermissionError.textContent = "授权页面令牌已失效，请刷新页面。";
        elements.gripperPermissionError.hidden = false;
      }
      return;
    }

    app.gripperPermissionPending = true;
    if (elements.gripperPermissionCancel) elements.gripperPermissionCancel.disabled = true;
    if (elements.gripperPermissionError) {
      elements.gripperPermissionError.textContent = "";
      elements.gripperPermissionError.hidden = true;
    }
    setGripperPermissionDialogMode(app.gripperPermissionMode, permission);
    let password = "";
    let requestBody = null;
    try {
      password = String(elements.gripperSudoPassword?.value || "");
      if (!password) {
        throw new ApiError("请输入 sudo 密码", 422, { code: "invalid_sudo_password" });
      }
      requestBody = JSON.stringify({
        revision: context.revision,
        ports: context.ports,
        can_interfaces: context.canInterfaces,
        password,
      });
      if (elements.gripperSudoPassword) elements.gripperSudoPassword.value = "";
      password = "";
      await api("/api/hardware/permissions/sudo", {
        method: "POST",
        body: requestBody,
        cache: "no-store",
        credentials: "same-origin",
        referrerPolicy: "no-referrer",
        headers: { "X-CSRF-Token": csrfToken },
      });
      requestBody = null;
      app.gripperPermissionDismissedSignature = app.gripperPermissionPromptSignature;
      try {
        await refreshState(false);
      } catch (_error) {
        // Permission repair already succeeded; the normal poll will resync.
      }
      requestGripperActualAlignment(GRIPPER_SIDES, "permission_repaired");
      showToast("设备权限已更新", "success");
      settleGripperPermissionPrompt(true);
    } catch (error) {
      requestBody = null;
      password = "";
      const code = gripperPermissionErrorCode(error);
      if (code === "permission_context_changed") {
        app.gripperPermissionDismissedSignature = "";
        setGripperPermissionDialogMode("stale", permission);
        scheduleStatePoll(0);
      } else {
        setGripperPermissionDialogMode("sudo", permission);
      }
      if (elements.gripperPermissionError) {
        elements.gripperPermissionError.textContent = errorMessage(error);
        elements.gripperPermissionError.hidden = false;
      }
    } finally {
      requestBody = null;
      password = "";
      app.gripperPermissionPending = false;
      if (elements.gripperPermissionCancel) elements.gripperPermissionCancel.disabled = false;
      setGripperPermissionDialogMode(app.gripperPermissionMode, permission);
    }
  }

  function renderGripperPortSelect(side, state, grippers, busy) {
    const select = elements.grippers[side]?.port;
    if (!select) return;
    const draft = app.gripperPortDrafts.get(side);
    const selected = draft ?? state.port;
    const values = [...grippers.availablePorts];
    if (selected && !values.includes(selected)) values.unshift(selected);
    const signature = JSON.stringify({ values, selected, present: state.portPresent });
    if (
      select.dataset.renderSignature !== signature &&
      typeof document.createElement === "function"
    ) {
      const options = values.map((port) => {
        const option = document.createElement("option");
        option.textContent = port === selected && !grippers.availablePorts.includes(port)
          ? `${port}（已断开）`
          : port;
        option.value = port;
        return option;
      });
      if (!options.length) {
        const empty = document.createElement("option");
        empty.textContent = "未发现可用 ttyUSB";
        empty.value = "";
        empty.disabled = true;
        options.push(empty);
      }
      select.replaceChildren(...options);
      select.dataset.renderSignature = signature;
      countPerf("optionReplacements");
    }
    select.value = selected && values.includes(selected) ? selected : "";
    select.disabled = Boolean(
      !state.configured ||
      !grippers.canBindPorts ||
      busy ||
      app.configSwitching ||
      app.gripperPortPending.size ||
      !grippers.availablePorts.length
    );
    select.title = grippers.bindingBlockers.join("；");
  }

  function renderGrippers() {
    const grippers = normalizeGrippers();
    const calibrationBusy = Boolean(
      app.gripperCalibratePending ||
      grippers.globalBusy ||
      grippers.busySides.size ||
      app.gripperMovePending.size ||
      app.gripperTargetSessions.size ||
      app.gripperPortPending.size
    );
    const calibrationRunning = Boolean(
      app.gripperCalibratePending ||
      grippers.globalBusy ||
      grippers.status === "calibrating"
    );
    const calibrate = grippers.capabilities.calibrate;
    if (elements.gripperAuthorizePermissions) {
      elements.gripperAuthorizePermissions.hidden = !grippers.permission.required;
      elements.gripperAuthorizePermissions.disabled = app.gripperPermissionPending;
      elements.gripperAuthorizePermissions.classList.toggle(
        "is-busy",
        app.gripperPermissionPending
      );
      elements.gripperAuthorizePermissions.textContent = app.gripperPermissionPending
        ? "正在修复权限"
        : "修复设备权限";
      elements.gripperAuthorizePermissions.title = grippers.permission.required
        ? [
            grippers.permission.affectedPorts.length
              ? `串口：${grippers.permission.affectedPorts.join("、")}`
              : "",
            grippers.permission.affectedCanInterfaces.length
              ? `CAN：${grippers.permission.affectedCanInterfaces.join("、")}`
              : "",
          ].filter(Boolean).join("；")
        : "";
    }
    if (elements.gripperCalibrateAll) {
      elements.gripperCalibrateAll.disabled = !calibrate.can || calibrationBusy || controlsFailClosed("calibrate") || app.configSwitching;
      elements.gripperCalibrateAll.classList.toggle("is-busy", app.gripperCalibratePending);
      const progressSide = grippers.progress?.side === "left"
        ? "左夹爪"
        : grippers.progress?.side === "right"
          ? "右夹爪"
          : "双夹爪";
      const progressPhase = String(firstDefined(grippers.progress?.phase_label, grippers.progress?.phase, "")).trim();
      elements.gripperCalibrateAll.textContent = calibrationRunning
        ? `${progressSide}标定中${progressPhase ? ` · ${progressPhase}` : ""}`
        : "一键标定双夹爪";
      elements.gripperCalibrateAll.title = calibrate.blockers.join("；");
    }

    const visibleReasons = [];
    if (grippers.error) visibleReasons.push(grippers.error);
    if (grippers.permission.required) {
      if (grippers.permission.affectedPorts.length) {
        visibleReasons.push(
          `串口权限不足：${grippers.permission.affectedPorts.join("、")}`
        );
      }
      if (grippers.permission.affectedCanInterfaces.length) {
        visibleReasons.push(
          `Yam CAN 尚未启用：${grippers.permission.affectedCanInterfaces.join("、")}`
        );
      }
    }
    if (grippers.permission.inspectionError) {
      visibleReasons.push("无法安全检查夹爪串口权限");
    }
    for (const side of GRIPPER_SIDES) {
      const state = grippers.sides[side];
      const ui = elements.grippers[side];
      if (!ui) continue;
      const moving = Boolean(
        app.gripperMovePending.has(side) ||
        app.gripperPositionSync.has(side) ||
        app.gripperTargetSessions.has(side) ||
        ["moving", "queued"].includes(state.status)
      );
      const retryingContainedError = Boolean(
        app.gripperTargetSessions.has(side) &&
        state.error &&
        app.state.grippers?.hardware_unknown !== true &&
        state.source?.hardware_unknown !== true
      );
      const visibleStateError = retryingContainedError ? "" : state.error;
      const artifact = state.calibrationArtifact;
      const artifactStatus = String(artifact?.status || "").trim().toLowerCase();
      const artifactReason = String(artifact?.reason || "").trim();
      const draft = app.gripperDrafts.get(side);
      const displayedTarget = gripperSliderDisplayPercent(side, state, draft);
      renderGripperPortSelect(side, state, grippers, calibrationBusy);
      if (ui.status) {
        ui.status.dataset.status = moving ? "moving" : state.status;
        ui.status.textContent = moving ? "移动中" : GRIPPER_STATUS_LABELS[state.status];
        ui.status.title = [
          visibleStateError,
          artifactStatus && artifactStatus !== "valid"
            ? `标定 artifact：${GRIPPER_ARTIFACT_STATUS_LABELS[artifactStatus] || artifactStatus}`
            : "",
          artifactReason,
          state.capabilities.queuePosition.blockers.join("；"),
        ].filter(Boolean).join("；");
      }
      if (ui.minLabel) ui.minLabel.textContent = state.minLabel;
      if (ui.maxLabel) ui.maxLabel.textContent = state.maxLabel;
      if (ui.position) {
        if (!app.gripperEditing.has(side) && displayedTarget !== null) {
          ui.position.value = String(Math.round(displayedTarget));
        }
        // A position worker is side-local and the backend safely coalesces its
        // next target.  Keep the slider interactive while it is moving; only
        // global safety/topology/calibration interlocks disable new targets.
        ui.position.disabled = !state.capabilities.queuePosition.can ||
          app.gripperCalibratePending ||
          grippers.globalBusy ||
          app.gripperPortPending.size > 0 ||
          controlsFailClosed("position") ||
          app.configSwitching;
        ui.position.setAttribute(
          "aria-valuetext",
          displayedTarget === null ? "实际位置未知" : `${Math.round(displayedTarget)}%`
        );
        ui.position.title = state.capabilities.queuePosition.blockers.join("；");
      }
      if (ui.value) {
        const actual = state.actualPercent === null ? "实际 —" : `实际 ${Math.round(state.actualPercent)}%`;
        const desired = firstDefined(
          app.gripperDesiredTargets.get(side),
          draft
        );
        ui.value.textContent = desired === undefined
          ? actual
          : `${actual} · 目标 ${Math.round(desired)}%`;
      }
      if (visibleStateError) {
        visibleReasons.push(`${side === "left" ? "左" : "右"}夹爪：${visibleStateError}`);
      }
      if (artifactStatus && artifactStatus !== "valid") {
        visibleReasons.push(
          `${side === "left" ? "左" : "右"}夹爪：${GRIPPER_ARTIFACT_STATUS_LABELS[artifactStatus] || artifactStatus}`
            + (artifactReason ? `（${artifactReason}）` : "")
        );
      }
      if (!state.capabilities.queuePosition.can) {
        visibleReasons.push(...state.capabilities.queuePosition.blockers);
      }
    }
    if (!calibrate.can) visibleReasons.push(...calibrate.blockers);
    for (const [side, message] of app.gripperErrors) {
      visibleReasons.push(`${side === "all" ? "夹爪" : side === "left" ? "左夹爪" : "右夹爪"}：${message}`);
    }
    if (elements.gripperCheckReason) {
      const pendingText = app.gripperCalibratePending
        ? "正在执行双夹爪全行程标定。"
        : app.gripperPortPending.size
          ? "正在保存夹爪串口绑定。"
        : app.gripperTargetSessions.size || app.gripperMovePending.size
          ? "夹爪正在闭环追踪目标；可继续拖动任一侧滑块。"
          : "";
      const hardControlReason = GRIPPER_SIDES
        .map((side) => grippers.sides[side])
        .filter((state) => !state.capabilities.queuePosition.can)
        .flatMap((state) => state.capabilities.queuePosition.blockers)
        .find(Boolean);
      const reason = hardControlReason || pendingText ||
        [...new Set(visibleReasons)].find(Boolean) ||
        "夹爪已就绪；拖动滑块会限频追踪目标位置。";
      elements.gripperCheckReason.textContent = reason;
      elements.gripperCheckReason.title = reason;
      elements.gripperCheckReason.classList.toggle(
        "is-error",
        Boolean(grippers.error || app.gripperErrors.size || grippers.permission.required)
      );
    }
    maybePromptGripperPermission(grippers.permission);
  }

  function gripperIdempotencyKey() {
    if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
      return globalThis.crypto.randomUUID();
    }
    return `gripper-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function nextGripperSnapshotRequest() {
    app.gripperSnapshotRequest += 1;
    return app.gripperSnapshotRequest;
  }

  function gripperPayloadCommand(payload) {
    return firstDefined(payload?.command, payload?.state?.command, payload?.data?.command);
  }

  function gripperSnapshotRevision(raw, side) {
    return asNumber(firstDefined(raw?.sides?.[side]?.revision, raw?.side_revisions?.[side]));
  }

  function publishAcceptedGripperSnapshotToViewer(acceptedSides) {
    if (!acceptedSides.length) return false;
    const liveRizon = app.devices.some((device) => (
      isRizonDevice(device) &&
      ["running", "degraded"].includes(String(device.status || "").toLowerCase())
    ));
    if (liveRizon) {
      // Rizon SHM publishes gripper state at the device rate. Let that source
      // drive animation while the arm is live; the direct gripper snapshot is
      // the fallback when Rizon is stopped.
      window.__dualArmViewer?.invalidateGripperDirectSources?.(acceptedSides);
      return false;
    }
    const grippers = app.state.grippers && typeof app.state.grippers === "object"
      ? app.state.grippers
      : {};
    const sync = stateSyncMetadata();
    const transportStale = Boolean(
      app.stateTransportFailed ||
      !app.stateTransportDeadlineAtMs ||
      Date.now() >= app.stateTransportDeadlineAtMs ||
      sync.stale === true ||
      sync.authoritative === false
    );
    const globalHardwareUnknown = grippers.hardware_unknown === true;
    const payload = {};
    for (const side of acceptedSides) {
      const sideSource = grippers.sides?.[side];
      const rawActualPercent = sideSource?.actual_percent;
      const actualPercent = rawActualPercent === undefined || rawActualPercent === null || rawActualPercent === ""
        ? null
        : asNumber(rawActualPercent);
      const status = String(sideSource?.status || grippers.status || "unavailable")
        .trim()
        .toLowerCase();
      const known = Boolean(
        !transportStale &&
        !globalHardwareUnknown &&
        sideSource?.hardware_unknown !== true &&
        status === "idle" &&
        actualPercent !== null &&
        actualPercent >= 0 &&
        actualPercent <= 100
      );
      payload[side] = known
        ? { known: true, actual_percent: actualPercent }
        : { known: false };
    }
    const viewerOptions = { source: "control-panel" };
    if (window.__dualArmViewer?.updateGripperStates) {
      window.__dualArmViewer.updateGripperStates(payload, viewerOptions);
    } else if (typeof window.updateGripperStates === "function") {
      window.updateGripperStates(payload, viewerOptions);
    } else {
      // device_control.js can observe the first state before the module viewer
      // mounts. Preserve that authoritative serial snapshot for autoMount.
      window.__pendingGripperViewerStates = {
        raw: payload,
        options: viewerOptions,
      };
    }
    return true;
  }

  function applyGripperSnapshot(raw, {
    epoch = app.gripperSyncEpoch,
    requestId = nextGripperSnapshotRequest(),
    controllerRevision = null,
  } = {}) {
    if (!raw || typeof raw !== "object" || epoch !== app.gripperSyncEpoch) return false;
    const incomingConfig = normalizeConfigName(raw.config_name);
    const expectedConfig = normalizeConfigName(app.selectedConfig);
    if (incomingConfig && expectedConfig && incomingConfig !== expectedConfig) return false;

    const incomingRevision = asNumber(firstDefined(controllerRevision, raw.revision));
    const currentRevision = asNumber(app.gripperSnapshotRevision);
    if (
      incomingRevision !== null &&
      currentRevision !== null &&
      incomingRevision < currentRevision
    ) return false;

    const globalAdvanced = Boolean(
      incomingRevision !== null &&
      (currentRevision === null || incomingRevision > currentRevision)
    );
    const current = app.state.grippers && typeof app.state.grippers === "object"
      ? app.state.grippers
      : {};
    const acceptTop = globalAdvanced || requestId >= app.gripperTopSnapshotRequest;
    const merged = acceptTop ? { ...current, ...raw } : { ...raw, ...current };
    const currentSides = current.sides && typeof current.sides === "object" ? current.sides : {};
    const incomingSides = raw.sides && typeof raw.sides === "object" ? raw.sides : {};
    merged.sides = { ...currentSides };
    const acceptedSides = [];

    for (const side of GRIPPER_SIDES) {
      const incoming = incomingSides[side];
      if (!incoming || typeof incoming !== "object") continue;
      const incomingSideRevision = gripperSnapshotRevision(raw, side);
      const storedSideRevision = asNumber(app.gripperSideSnapshotRevisions.get(side));
      const storedRequest = asNumber(app.gripperSideSnapshotRequests.get(side)) ?? 0;
      const staleSide = Boolean(
        !globalAdvanced &&
        (
          (incomingSideRevision !== null && storedSideRevision !== null && incomingSideRevision < storedSideRevision) ||
          (incomingSideRevision === storedSideRevision && requestId < storedRequest)
        )
      );
      if (staleSide) continue;
      merged.sides[side] = incoming;
      acceptedSides.push(side);
      if (incomingSideRevision !== null) {
        app.gripperSideSnapshotRevisions.set(side, incomingSideRevision);
      }
      app.gripperSideSnapshotRequests.set(side, requestId);
    }

    app.state.grippers = merged;
    publishAcceptedGripperSnapshotToViewer(acceptedSides);
    if (acceptTop) app.gripperTopSnapshotRequest = requestId;
    if (incomingRevision !== null) {
      app.gripperSnapshotRevision = incomingRevision;
      const stateRevision = asNumber(app.state.revision);
      if (stateRevision === null || incomingRevision >= stateRevision) {
        app.state.revision = incomingRevision;
      }
    }
    return true;
  }

  function updateGrippersFromPayload(payload, options = {}) {
    const next = firstDefined(payload?.grippers, payload?.state?.grippers, payload?.data?.grippers);
    const revision = firstDefined(payload?.revision, payload?.state?.revision, payload?.data?.revision);
    return applyGripperSnapshot(next, { ...options, controllerRevision: revision });
  }

  function clearGripperTargetTimer(side) {
    const timer = app.gripperTargetTimers.get(side);
    if (timer !== undefined) window.clearTimeout(timer);
    app.gripperTargetTimers.delete(side);
  }

  function resetGripperTargetControl(sides = GRIPPER_SIDES, {
    clearDesired = true,
    clearActual = false,
  } = {}) {
    for (const side of sides) {
      if (!GRIPPER_SIDES.includes(side)) continue;
      clearGripperTargetTimer(side);
      app.gripperTargetSessions.delete(side);
      app.gripperQueuedTargets.delete(side);
      app.gripperPositionSync.delete(side);
      app.gripperDrafts.delete(side);
      app.gripperErrors.delete(side);
      if (clearDesired) app.gripperDesiredTargets.delete(side);
      if (clearActual) app.gripperActualValues.delete(side);
    }
  }

  function requestGripperActualAlignment(sides = GRIPPER_SIDES, reason = "refresh") {
    const affected = [...new Set(sides.filter((side) => GRIPPER_SIDES.includes(side)))];
    resetGripperTargetControl(affected, { clearDesired: true, clearActual: true });
    queueGripperActualProbe(affected, reason);
  }

  function invalidateGripperSyncContext(sides = GRIPPER_SIDES) {
    const affected = [...new Set(sides.filter((side) => GRIPPER_SIDES.includes(side)))];
    window.__dualArmViewer?.invalidateGripperDirectSources?.(affected);
    const pendingViewerState = window.__pendingGripperViewerStates;
    if (pendingViewerState?.raw && typeof pendingViewerState.raw === "object") {
      const pendingSides = pendingViewerState.raw.sides && typeof pendingViewerState.raw.sides === "object"
        ? pendingViewerState.raw.sides
        : pendingViewerState.raw;
      for (const side of affected) delete pendingSides[side];
    }
    app.gripperSyncEpoch += 1;
    app.gripperProbeQueued.clear();
    app.gripperProbePending.clear();
    resetGripperTargetControl(affected, { clearDesired: true, clearActual: true });
    for (const side of affected) {
      app.gripperActualValues.delete(side);
      app.gripperCalibrationSliderValues.delete(side);
      app.gripperProbeSync.delete(side);
    }
    if (affected.length === GRIPPER_SIDES.length) app.gripperCalibrationSync = null;
    return app.gripperSyncEpoch;
  }

  function setGripperActualValue(side, value) {
    if (value === undefined || value === null || value === "") return false;
    const percent = asNumber(value);
    if (!GRIPPER_SIDES.includes(side) || percent === null) return false;
    app.gripperActualValues.set(side, Math.min(100, Math.max(0, percent)));
    return true;
  }

  function gripperSliderFrozen(side) {
    return Boolean(
      app.gripperEditing.has(side) ||
      app.gripperMovePending.has(side) ||
      app.gripperPositionSync.has(side)
    );
  }

  function syncGripperActualValues(grippers = normalizeGrippers()) {
    for (const side of GRIPPER_SIDES) {
      // Actual feedback is tracked independently from the user's latched
      // target.  Updating it must never rewrite the slider while a manual
      // target exists; gripperSliderDisplayPercent performs that arbitration.
      setGripperActualValue(side, grippers.sides[side].actualPercent);
    }
  }

  function beginGripperPositionSync(payload, side, targetPercent, epoch = app.gripperSyncEpoch) {
    const command = gripperPayloadCommand(payload);
    const commandId = String(command?.command_id || "").trim();
    if (
      epoch !== app.gripperSyncEpoch ||
      !GRIPPER_SIDES.includes(side) ||
      String(command?.action || "").toLowerCase() !== "position" ||
      !commandId ||
      command?.accepted === false
    ) return false;
    const grippers = normalizeGrippers();
    app.gripperPositionSync.set(side, {
      action: "position",
      commandId,
      targetPercent: Math.min(100, Math.max(0, asNumber(targetPercent) ?? 0)),
      revision: asNumber(firstDefined(payload?.revision, payload?.state?.revision, payload?.data?.revision)),
      sideRevision: asNumber(firstDefined(command?.side_revision, command?.side_revisions?.[side], grippers.sides[side].revision)),
      epoch,
      startedAtMs: Date.now(),
      observedCommand: false,
    });
    return true;
  }

  function gripperTargetReached(actualPercent, targetPercent) {
    const actual = asNumber(actualPercent);
    const target = asNumber(targetPercent);
    return actual !== null && target !== null &&
      Math.abs(actual - target) <= GRIPPER_TARGET_TOLERANCE_PERCENT;
  }

  function finishGripperTargetSession(side, state) {
    clearGripperTargetTimer(side);
    app.gripperPositionSync.delete(side);
    app.gripperTargetSessions.delete(side);
    app.gripperQueuedTargets.delete(side);
    app.gripperDrafts.delete(side);
    app.gripperErrors.delete(side);
    setGripperActualValue(side, state?.actualPercent);
    return true;
  }

  function failGripperTargetSession(side, message) {
    clearGripperTargetTimer(side);
    app.gripperPositionSync.delete(side);
    app.gripperTargetSessions.delete(side);
    app.gripperQueuedTargets.delete(side);
    app.gripperDrafts.delete(side);
    if (message) {
      app.gripperErrors.set(side, message);
      showToast(message, "error");
    }
    return true;
  }

  function continueGripperTargetSession(side, state, outcome, nowMs = Date.now()) {
    const session = app.gripperTargetSessions.get(side);
    const desired = asNumber(app.gripperDesiredTargets.get(side));
    if (!session || desired === null) {
      app.gripperPositionSync.delete(side);
      return false;
    }
    if (outcome === "completed") {
      setGripperActualValue(side, state?.actualPercent);
      if (gripperTargetReached(state?.actualPercent, desired)) {
        return finishGripperTargetSession(side, state);
      }
    }
    app.gripperPositionSync.delete(side);
    const grippers = normalizeGrippers();
    const unsafe = Boolean(
      grippers.globalBusy ||
      app.state.grippers?.hardware_unknown === true ||
      state?.source?.hardware_unknown === true
    );
    const expired = nowMs - session.startedAtMs >= GRIPPER_TARGET_SESSION_TIMEOUT_MS;
    const exhausted = session.attempts >= GRIPPER_TARGET_MAX_ATTEMPTS;
    if (unsafe || expired || exhausted) {
      const reason = state?.error ||
        (unsafe ? "夹爪硬件状态未知，已停止自动重试" : "夹爪未能在限定时间内到达目标位置");
      return failGripperTargetSession(side, reason);
    }
    // A contained worker error or a valid but off-target readback is retried
    // with the same latched target.  Do not probe and do not rewrite slider.
    app.gripperErrors.delete(side);
    app.gripperQueuedTargets.set(side, desired);
    scheduleGripperTargetSubmission(side, GRIPPER_TARGET_RETRY_DELAY_MS);
    return true;
  }

  function observeGripperPositionSync(nowMs = Date.now()) {
    const grippers = normalizeGrippers();
    let changed = false;
    for (const [side, pending] of [...app.gripperPositionSync]) {
      if (pending.epoch !== app.gripperSyncEpoch) {
        app.gripperPositionSync.delete(side);
        changed = true;
        continue;
      }
      const state = grippers.sides[side];
      if (nowMs - pending.startedAtMs >= GRIPPER_POSITION_SYNC_TIMEOUT_MS) {
        changed = continueGripperTargetSession(side, state, "timeout", nowMs) || changed;
        continue;
      }
      const revision = asNumber(grippers.revision);
      const sideRevision = asNumber(state.revision);
      if (
        (pending.revision !== null && revision !== null && revision < pending.revision) ||
        (pending.sideRevision !== null && sideRevision !== null && sideRevision < pending.sideRevision)
      ) continue;
      const lastCommandId = String(state?.source?.last_command_id || "");
      const commandMatches = lastCommandId === pending.commandId;
      if (commandMatches) pending.observedCommand = true;
      const completed = Boolean(
        pending.observedCommand &&
        commandMatches &&
        state.status === "idle" &&
        state.actualPercent !== null &&
        String(state?.source?.last_completed_command_id || "") === pending.commandId &&
        String(state?.source?.last_completed_action || "").toLowerCase() === "position"
      );
      const failed = Boolean(
        pending.observedCommand &&
        commandMatches &&
        (state.status === "error" || state.error)
      );
      if (completed) {
        changed = continueGripperTargetSession(side, state, "completed", nowMs) || changed;
      } else if (failed) {
        changed = continueGripperTargetSession(side, state, "failed", nowMs) || changed;
      }
    }
    return changed;
  }

  function driveGripperTargetSessions(nowMs = Date.now()) {
    const grippers = normalizeGrippers();
    for (const [side, session] of [...app.gripperTargetSessions]) {
      const desired = asNumber(app.gripperDesiredTargets.get(side));
      const state = grippers.sides[side];
      if (desired === null) {
        resetGripperTargetControl([side], { clearDesired: true });
        continue;
      }
      if (
        state.status === "idle" &&
        state.actualPercent !== null &&
        gripperTargetReached(state.actualPercent, desired)
      ) {
        finishGripperTargetSession(side, state);
        continue;
      }
      if (nowMs - session.startedAtMs >= GRIPPER_TARGET_SESSION_TIMEOUT_MS) {
        failGripperTargetSession(side, "夹爪未能在限定时间内到达目标位置");
        continue;
      }
      if (
        !app.gripperPositionSync.has(side) &&
        !app.gripperMovePending.has(side) &&
        !app.gripperTargetTimers.has(side)
      ) {
        scheduleGripperTargetSubmission(side, 0);
      }
    }
  }

  function beginGripperProbeSync(payload, side, epoch = app.gripperSyncEpoch) {
    const command = gripperPayloadCommand(payload);
    const commandId = String(command?.command_id || "").trim();
    if (
      epoch !== app.gripperSyncEpoch ||
      !GRIPPER_SIDES.includes(side) ||
      String(command?.action || "").toLowerCase() !== "probe" ||
      !commandId ||
      command?.accepted === false
    ) return false;
    const grippers = normalizeGrippers();
    app.gripperProbeSync.set(side, {
      commandId,
      revision: asNumber(firstDefined(payload?.revision, payload?.state?.revision, payload?.data?.revision)),
      sideRevision: asNumber(firstDefined(command?.side_revision, grippers.sides[side].revision)),
      epoch,
      startedAtMs: Date.now(),
      observedCommand: false,
    });
    return true;
  }

  function observeGripperProbeSync(nowMs = Date.now()) {
    const grippers = normalizeGrippers();
    let changed = false;
    for (const [side, pending] of [...app.gripperProbeSync]) {
      if (pending.epoch !== app.gripperSyncEpoch) {
        app.gripperProbeSync.delete(side);
        changed = true;
        continue;
      }
      const state = grippers.sides[side];
      if (nowMs - pending.startedAtMs >= GRIPPER_PROBE_SYNC_TIMEOUT_MS) {
        app.gripperProbeSync.delete(side);
        changed = true;
        continue;
      }
      const revision = asNumber(grippers.revision);
      const sideRevision = asNumber(state.revision);
      if (
        (pending.revision !== null && revision !== null && revision < pending.revision) ||
        (pending.sideRevision !== null && sideRevision !== null && sideRevision < pending.sideRevision)
      ) continue;
      const commandMatches = String(firstDefined(
        state?.source?.last_probe_command_id,
        state?.source?.last_command_id,
        ""
      )) === pending.commandId;
      if (commandMatches) pending.observedCommand = true;
      const completed = Boolean(
        pending.observedCommand &&
        commandMatches &&
        state.status === "idle" &&
        state.actualPercent !== null &&
        String(firstDefined(
          state?.source?.last_probe_completed_command_id,
          String(state?.source?.last_completed_action || "").toLowerCase() === "probe"
            ? state?.source?.last_completed_command_id
            : "",
          ""
        )) === pending.commandId
      );
      const failed = Boolean(
        pending.observedCommand &&
        commandMatches &&
        (["error", "unavailable"].includes(state.status) || state.error)
      );
      if (completed) setGripperActualValue(side, state.actualPercent);
      if (completed || failed) {
        app.gripperProbeSync.delete(side);
        changed = true;
      }
    }
    return changed;
  }

  function beginGripperCalibrationSync(payload) {
    const command = firstDefined(payload?.command, payload?.state?.command, payload?.data?.command);
    const commandId = String(command?.command_id || "").trim();
    if (String(command?.action || "").toLowerCase() !== "calibrate" || !commandId) return false;
    const sides = command?.side === "both"
      ? [...GRIPPER_SIDES]
      : GRIPPER_SIDES.includes(command?.side)
        ? [command.side]
        : [];
    if (!sides.length) return false;
    resetGripperTargetControl(sides, { clearDesired: true, clearActual: true });
    for (const side of sides) {
      app.gripperCalibrationSliderValues.delete(side);
      app.gripperActualValues.delete(side);
    }
    app.gripperCalibrationSync = {
      commandId,
      sides,
      revision: asNumber(firstDefined(payload?.revision, payload?.state?.revision, payload?.data?.revision)),
      startedAtMs: Date.now(),
      observedCommand: false,
    };
    return observeGripperCalibrationSync();
  }

  function observeGripperCalibrationSync(nowMs = Date.now()) {
    const pending = app.gripperCalibrationSync;
    if (!pending) return false;
    if (nowMs - pending.startedAtMs >= GRIPPER_CALIBRATION_SYNC_TIMEOUT_MS) {
      app.gripperCalibrationSync = null;
      return false;
    }
    const grippers = normalizeGrippers();
    const revision = asNumber(grippers.revision);
    if (pending.revision !== null && revision !== null && revision < pending.revision) {
      // Ignore an older state request that was already in flight when calibration
      // was accepted. It must not cancel the completion tracker.
      return false;
    }
    const sideStates = pending.sides.map((side) => grippers.sides[side]);
    const commandMatches = sideStates.every((state) => (
      String(state?.source?.last_command_id || "") === pending.commandId
    ));
    if (commandMatches) pending.observedCommand = true;
    const failed = Boolean(
      grippers.error ||
      grippers.status === "error" ||
      sideStates.some((state) => state?.status === "error" || state?.error)
    );
    if (failed) {
      app.gripperCalibrationSync = null;
      return false;
    }
    if (!pending.observedCommand || !commandMatches) return false;
    const completed = sideStates.every((state) => (
      state?.status === "idle" &&
      state.actualPercent !== null &&
      state?.source?.calibrated === true &&
      String(state?.source?.last_completed_command_id || "") === pending.commandId &&
      String(state?.source?.last_completed_action || "").toLowerCase() === "calibrate"
    ));
    if (!completed) return false;
    for (const side of pending.sides) {
      app.gripperCalibrationSliderValues.set(side, {
        commandId: pending.commandId,
        actualPercent: grippers.sides[side].actualPercent,
      });
      setGripperActualValue(side, grippers.sides[side].actualPercent);
    }
    app.gripperCalibrationSync = null;
    requestGripperActualAlignment(pending.sides, "calibration_completed");
    return true;
  }

  function gripperSliderDisplayPercent(side, state, draft) {
    const draftPercent = asNumber(draft);
    if (draftPercent !== null) return Math.min(100, Math.max(0, draftPercent));
    const desiredPercent = asNumber(app.gripperDesiredTargets.get(side));
    if (desiredPercent !== null) {
      return Math.min(100, Math.max(0, desiredPercent));
    }
    const positionPending = app.gripperPositionSync.get(side);
    if (positionPending) return positionPending.targetPercent;
    const calibrated = app.gripperCalibrationSliderValues.get(side);
    if (calibrated) {
      const lastCommandId = String(state?.source?.last_command_id || "");
      if (lastCommandId === calibrated.commandId) return calibrated.actualPercent;
      if (lastCommandId) app.gripperCalibrationSliderValues.delete(side);
    }
    const actual = asNumber(app.gripperActualValues.get(side));
    return actual === null ? null : Math.min(100, Math.max(0, actual));
  }

  function discardGripperCalibrationSliderValue(side) {
    app.gripperCalibrationSliderValues.delete(side);
  }

  function gripperProbeBlocked(side, grippers = normalizeGrippers()) {
    const state = grippers.sides[side];
    return Boolean(
      !state?.configured ||
      !state.port ||
      !state.capabilities.probe.can ||
      !Number.isInteger(state.revision) ||
      !Number.isInteger(grippers.revision) ||
      app.configSwitching ||
      app.gripperCalibratePending ||
      app.gripperPortPending.size ||
      gripperSliderFrozen(side) ||
      app.gripperProbePending.has(side) ||
      app.gripperProbeSync.has(side) ||
      grippers.globalBusy ||
      state.busy ||
      ["calibrating", "moving", "queued", "probing"].includes(state.status)
    );
  }

  function queueGripperActualProbe(sides = GRIPPER_SIDES, reason = "refresh") {
    void reason;
    let added = false;
    for (const side of sides) {
      if (
        GRIPPER_SIDES.includes(side) &&
        !app.gripperProbePending.has(side) &&
        !app.gripperProbeSync.has(side)
      ) {
        app.gripperProbeQueued.add(side);
        added = true;
      }
    }
    if (!added) return;
    scheduleStatePoll(0);
    void flushGripperActualProbes();
  }

  async function flushGripperActualProbes() {
    if (app.gripperProbePromise || app.configSwitching) return app.gripperProbePromise;
    const grippers = normalizeGrippers();
    const sides = [...app.gripperProbeQueued].filter((side) => !gripperProbeBlocked(side, grippers));
    if (!sides.length) return null;
    const epoch = app.gripperSyncEpoch;
    const requests = sides.map((side) => {
      app.gripperProbeQueued.delete(side);
      app.gripperProbePending.add(side);
      const requestId = nextGripperSnapshotRequest();
      const sideRevision = grippers.sides[side].revision;
      return (async () => {
        try {
          const payload = await withGripperPermissionRetry(() => api("/api/grippers/probe", {
            method: "POST",
            body: JSON.stringify({
              side,
              revision: grippers.revision,
              side_revision: sideRevision,
              idempotency_key: gripperIdempotencyKey(),
            }),
          }));
          if (!payload) return false;
          if (epoch !== app.gripperSyncEpoch) return false;
          updateGrippersFromPayload(payload, { epoch, requestId });
          beginGripperProbeSync(payload, side, epoch);
          return true;
        } catch (error) {
          if (epoch === app.gripperSyncEpoch && error instanceof ApiError && error.status === 409) {
            app.gripperProbeQueued.add(side);
            scheduleStatePoll(0);
          }
          return false;
        } finally {
          app.gripperProbePending.delete(side);
        }
      })();
    });
    const request = Promise.allSettled(requests).finally(() => {
      if (app.gripperProbePromise === request) app.gripperProbePromise = null;
      observeGripperProbeSync();
      syncGripperActualValues();
      renderGrippers();
      scheduleStatePoll(250);
    });
    app.gripperProbePromise = request;
    return request;
  }

  async function submitGripperPortBinding(side, port) {
    if (!GRIPPER_SIDES.includes(side) || !/^\/dev\/ttyUSB[0-9]+$/.test(port)) return;
    if (app.gripperPortPending.size || app.gripperTargetSessions.size || app.configSwitching) return;
    const grippers = normalizeGrippers();
    if (!grippers.canBindPorts || !grippers.availablePorts.includes(port)) return;
    const current = grippers.sides[side].port;
    if (current === port) return;
    const other = side === "left" ? "right" : "left";
    const bindings = { [side]: port };
    if (grippers.sides[other].port === port && current) bindings[other] = current;
    for (const [bindingSide, bindingPort] of Object.entries(bindings)) {
      app.gripperPortPending.add(bindingSide);
      app.gripperPortDrafts.set(bindingSide, bindingPort);
      app.gripperErrors.delete(bindingSide);
    }
    const affectedSides = Object.keys(bindings);
    const epoch = invalidateGripperSyncContext(affectedSides);
    const requestId = nextGripperSnapshotRequest();
    renderGrippers();
    try {
      const payload = await api("/api/grippers/bindings", {
        method: "POST",
        body: JSON.stringify({
          revision: grippers.revision,
          bindings,
        }),
      });
      updateGrippersFromPayload(payload, { epoch, requestId });
      requestGripperActualAlignment(affectedSides, "tty_binding_changed");
      showToast("夹爪串口绑定已更新", "success");
    } catch (error) {
      const message = errorMessage(error);
      app.gripperErrors.set(side, message);
      showToast(message, "error");
      scheduleStatePoll(0);
    } finally {
      for (const bindingSide of Object.keys(bindings)) {
        app.gripperPortPending.delete(bindingSide);
        app.gripperPortDrafts.delete(bindingSide);
      }
      renderGrippers();
      void flushGripperActualProbes();
    }
  }

  async function runGripperCalibration() {
    if (
      app.gripperCalibratePending ||
      app.gripperMovePending.size ||
      app.gripperTargetSessions.size ||
      app.configSwitching
    ) return;
    const expectedEpoch = app.stateEpoch;
    if (!(await ensureControlActionAuthorized("calibrate", expectedEpoch))) return;
    const grippers = normalizeGrippers();
    if (!grippers.capabilities.calibrate.can || grippers.globalBusy || grippers.busySides.size) return;
    const confirmed = await confirmOperation(
      "标定左右夹爪？",
      "请移除夹爪内及周围的物体和手指。左右夹爪将依次执行两轮全开、全合并触碰机械限位，整个过程最久约 2.5 分钟。",
      "确认开始标定"
    );
    if (!confirmed) return;
    if (!(await ensureControlActionAuthorized("calibrate", expectedEpoch))) return;

    app.gripperCalibratePending = true;
    app.gripperErrors.delete("all");
    const epoch = app.gripperSyncEpoch;
    const requestId = nextGripperSnapshotRequest();
    renderGrippers();
    try {
      if (!(await ensureControlActionAuthorized("calibrate", expectedEpoch))) return;
      const payload = await withGripperPermissionRetry(() => api("/api/grippers/calibrate", {
          method: "POST",
          body: JSON.stringify({
            side: "both",
            revision: grippers.revision,
            idempotency_key: gripperIdempotencyKey(),
          }),
        }),
        () => controlActionStillAuthorized("calibrate", expectedEpoch)
      );
      if (!payload) return;
      updateGrippersFromPayload(payload, { epoch, requestId });
      beginGripperCalibrationSync(payload);
      showToast("双夹爪标定指令已发送", "success");
      scheduleStatePoll(250);
    } catch (error) {
      const message = errorMessage(error);
      app.gripperErrors.set("all", message);
      showToast(message, "error");
      if (error instanceof ApiError && error.status === 409) await refreshState(false);
    } finally {
      app.gripperCalibratePending = false;
      renderGrippers();
    }
  }

  function latchGripperTarget(side, targetPercent) {
    if (!GRIPPER_SIDES.includes(side)) return null;
    const numericTarget = asNumber(targetPercent);
    if (numericTarget === null) return null;
    const target = Math.min(100, Math.max(0, Math.round(numericTarget)));
    const previous = asNumber(app.gripperDesiredTargets.get(side));
    let session = app.gripperTargetSessions.get(side);
    if (!session || previous === null || previous !== target) {
      clearGripperTargetTimer(side);
      session = {
        targetPercent: target,
        startedAtMs: Date.now(),
        attempts: 0,
        lastSubmittedAtMs: 0,
      };
      app.gripperTargetSessions.set(side, session);
    } else {
      session.targetPercent = target;
    }
    discardGripperCalibrationSliderValue(side);
    app.gripperProbeQueued.delete(side);
    app.gripperDesiredTargets.set(side, target);
    app.gripperDrafts.set(side, target);
    app.gripperQueuedTargets.set(side, target);
    app.gripperErrors.delete(side);
    return { target, session };
  }

  function scheduleGripperTargetSubmission(side, delayMs = 0) {
    const session = app.gripperTargetSessions.get(side);
    if (!session || app.gripperTargetTimers.has(side)) return false;
    const desired = asNumber(app.gripperDesiredTargets.get(side));
    const pending = app.gripperPositionSync.get(side);
    if (
      desired !== null &&
      pending &&
      pending.targetPercent === desired
    ) return false;
    const sinceLast = Date.now() - session.lastSubmittedAtMs;
    const rateDelay = Math.max(0, GRIPPER_TARGET_COMMAND_INTERVAL_MS - sinceLast);
    const waitMs = Math.max(0, Number(delayMs) || 0, rateDelay);
    const timer = window.setTimeout(() => {
      app.gripperTargetTimers.delete(side);
      void submitGripperTarget(side);
    }, waitMs);
    app.gripperTargetTimers.set(side, timer);
    return true;
  }

  async function submitGripperTarget(side) {
    if (!GRIPPER_SIDES.includes(side) || app.configSwitching) return false;
    const session = app.gripperTargetSessions.get(side);
    const target = asNumber(app.gripperDesiredTargets.get(side));
    if (!session || target === null) return false;
    if (app.gripperMovePending.has(side)) {
      app.gripperQueuedTargets.set(side, target);
      return false;
    }
    const expectedEpoch = app.stateEpoch;
    let authorizationDeferred = false;
    const authorizeTargetSubmission = () => {
      const authorized = controlActionStillAuthorized(
        "position",
        expectedEpoch,
        { notify: false }
      );
      if (!authorized) {
        authorizationDeferred = true;
        if (Date.now() - session.startedAtMs >= GRIPPER_TARGET_SESSION_TIMEOUT_MS) {
          failGripperTargetSession(
            side,
            "真实设备状态持续不可确认，夹爪目标控制已暂停"
          );
        } else {
          scheduleGripperTargetSubmission(side, GRIPPER_TARGET_RETRY_DELAY_MS);
        }
      }
      return authorized;
    };
    // A brief state-watchdog gap must pause the lane, not discard the user's
    // latest target. No command is sent until a fresh authoritative snapshot
    // arrives; the normal session timeout still reports a genuine outage.
    if (!authorizeTargetSubmission()) return false;
    const grippers = normalizeGrippers();
    const sideState = grippers.sides[side];
    if (grippers.globalBusy || app.state.grippers?.hardware_unknown === true) {
      failGripperTargetSession(
        side,
        sideState.error || "夹爪处于标定或未知硬件状态，目标控制已停止"
      );
      return false;
    }
    if (!sideState.capabilities.queuePosition.can) {
      failGripperTargetSession(
        side,
        sideState.capabilities.queuePosition.blockers[0] || "夹爪当前不可控制"
      );
      return false;
    }
    if (!Number.isInteger(sideState.revision)) {
      scheduleGripperTargetSubmission(side, GRIPPER_TARGET_RETRY_DELAY_MS);
      return false;
    }

    app.gripperMovePending.add(side);
    app.gripperErrors.delete(side);
    session.attempts += 1;
    session.lastSubmittedAtMs = Date.now();
    const epoch = app.gripperSyncEpoch;
    const requestId = nextGripperSnapshotRequest();
    let trackingCompletion = false;
    renderGrippers();
    try {
      if (!authorizeTargetSubmission()) return false;
      const payload = await withGripperPermissionRetry(() => api("/api/grippers/position", {
          method: "POST",
          body: JSON.stringify({
            side,
            target_percent: target,
            revision: grippers.revision,
            side_revision: sideState.revision,
            idempotency_key: gripperIdempotencyKey(),
          }),
        }),
        authorizeTargetSubmission
      );
      if (!payload) {
        if (authorizationDeferred) return false;
        failGripperTargetSession(side, "夹爪控制授权未完成，目标控制已暂停");
        return false;
      }
      updateGrippersFromPayload(payload, { epoch, requestId });
      trackingCompletion = beginGripperPositionSync(payload, side, target, epoch);
      if (!trackingCompletion) {
        const current = normalizeGrippers().sides[side];
        if (gripperTargetReached(current.actualPercent, target)) {
          finishGripperTargetSession(side, current);
        } else {
          scheduleGripperTargetSubmission(side, GRIPPER_TARGET_RETRY_DELAY_MS);
        }
      }
      scheduleStatePoll(200);
      return true;
    } catch (error) {
      const message = errorMessage(error);
      if (error instanceof ApiError && error.status === 409) {
        await refreshState(false);
      }
      const currentSession = app.gripperTargetSessions.get(side);
      const hardwareUnknown = app.state.grippers?.hardware_unknown === true ||
        normalizeGrippers().sides[side].source?.hardware_unknown === true;
      const expired = !currentSession ||
        Date.now() - currentSession.startedAtMs >= GRIPPER_TARGET_SESSION_TIMEOUT_MS ||
        currentSession.attempts >= GRIPPER_TARGET_MAX_ATTEMPTS;
      if (
        hardwareUnknown ||
        expired ||
        (error instanceof ApiError && error.status === 422)
      ) {
        failGripperTargetSession(side, message);
      } else {
        // Keep the user's target latched.  A fresh state snapshot decides
        // whether this was a transient revision/worker error or a fail-closed
        // hardware condition before the next rate-limited submission.
        scheduleGripperTargetSubmission(side, GRIPPER_TARGET_RETRY_DELAY_MS);
        scheduleStatePoll(0);
      }
      return false;
    } finally {
      app.gripperMovePending.delete(side);
      const queued = asNumber(app.gripperQueuedTargets.get(side));
      if (queued !== null && queued === target) app.gripperQueuedTargets.delete(side);
      const desired = asNumber(app.gripperDesiredTargets.get(side));
      const pending = app.gripperPositionSync.get(side);
      if (
        app.gripperTargetSessions.has(side) &&
        desired !== null &&
        (!pending || pending.targetPercent !== desired)
      ) {
        scheduleGripperTargetSubmission(side, 0);
      }
      renderGrippers();
    }
  }

  async function runGripperMove(side, targetPercent) {
    const latched = latchGripperTarget(side, targetPercent);
    if (!latched) return false;
    clearGripperTargetTimer(side);
    const pending = app.gripperPositionSync.get(side);
    if (pending && pending.targetPercent === latched.target) return true;
    if (
      latched.session.lastSubmittedAtMs > 0 &&
      Date.now() - latched.session.lastSubmittedAtMs < GRIPPER_TARGET_COMMAND_INTERVAL_MS
    ) {
      scheduleGripperTargetSubmission(side, 0);
      return false;
    }
    return submitGripperTarget(side);
  }

  function normalizeRecordingCapability(value) {
    if (value === true) return { can: true, blockers: [] };
    if (!value || typeof value !== "object") return { can: false, blockers: [] };
    const rawBlockers = Array.isArray(value.blockers)
      ? value.blockers
      : value.blockers
        ? [value.blockers]
        : [];
    return {
      can: value.can === true,
      blockers: rawBlockers.map((item) => String(item || "").trim()).filter(Boolean),
    };
  }

  function normalizeRecordingTag(value) {
    const tag = String(value || "").trim().toLowerCase();
    if (["accept", "accepted", "success", "positive"].includes(tag)) return "accept";
    if (["reject", "rejected", "failure", "negative"].includes(tag)) return "reject";
    return "";
  }

  function normalizeRecordingCameraRole(value, role) {
    const source = typeof value === "string"
      ? { device_key: value, status: "configured" }
      : value && typeof value === "object" ? value : {};
    const deviceKey = String(firstDefined(source.device_key, source.device, "") || "").trim();
    const topic = String(firstDefined(source.topic, source.camera_id, "") || "").trim();
    const shmName = String(firstDefined(source.shm_name, source.shm, "") || "").trim();
    const reason = String(source.reason || "").trim();
    let status = String(source.status || "").trim().toLowerCase();
    if (source.black_placeholder === true) status = "black_placeholder";
    else if (source.missing === true) status = "missing";
    else if (source.healthy === true && !status) status = "healthy";
    if (!Object.prototype.hasOwnProperty.call(RECORDING_CAMERA_STATUS_LABELS, status)) {
      status = deviceKey || topic ? "configured" : "unknown";
    }
    return {
      role,
      device_key: deviceKey || null,
      topic: topic || null,
      shm_name: shmName || null,
      reason: reason || null,
      status,
    };
  }

  function normalizeRecordingCameraRoles(value) {
    const source = value && typeof value === "object" ? value : {};
    return Object.fromEntries(RECORDING_CAMERA_ROLES.map((role) => (
      [role, normalizeRecordingCameraRole(firstDefined(source[role], source[`${role}_camera`]), role)]
    )));
  }

  function recordingCameraWarning(reason) {
    return /camera mapping missing/i.test(String(reason || "")) ? "未配置映射" : "";
  }

  function normalizeRecordingState(raw = app.state.recording) {
    const source = raw && typeof raw === "object" ? raw : {};
    const episodeSource = source.episode && typeof source.episode === "object"
      ? source.episode
      : null;
    const episode = episodeSource
      ? {
          ...episodeSource,
          id: firstDefined(episodeSource.id, null),
          index: episodeSource.index === undefined || episodeSource.index === null
            ? null
            : asNumber(episodeSource.index),
          frame_count: Math.max(0, asNumber(episodeSource.frame_count) || 0),
          elapsed_ms: Math.max(0, asNumber(episodeSource.elapsed_ms) || 0),
          checkpoint_revision: firstDefined(episodeSource.checkpoint_revision, 0),
        }
      : null;
    const segments = Array.isArray(source.segments)
      ? source.segments.map((item) => {
          const start = Math.max(0, asNumber(item?.start_frame) || 0);
          const end = Math.max(start, asNumber(item?.end_frame) || start);
          return {
            ...item,
            start_frame: start,
            end_frame: end,
            tag: normalizeRecordingTag(item?.tag),
          };
        }).filter((item) => item.tag)
      : [];
    const checkpoints = Array.isArray(source.checkpoints)
      ? source.checkpoints.map((item, index) => ({
          ...item,
          index: asNumber(firstDefined(item?.index, index)),
          frame: Math.max(0, asNumber(item?.frame) || 0),
          elapsed_ms: Math.max(0, asNumber(item?.elapsed_ms) || 0),
        }))
      : [];
    const capabilities = {};
    for (const action of RECORDING_ACTIONS) {
      capabilities[action] = normalizeRecordingCapability(source.capabilities?.[action]);
    }
    return {
      ...source,
      configured: source.configured === true,
      enabled: source.enabled === true,
      required: source.required === true,
      active: source.active === true,
      recorder_ready: source.recorder_ready === true,
      status: String(source.status || "").trim().toLowerCase(),
      error: String(source.error || "").trim(),
      episode,
      segments,
      checkpoints,
      capabilities,
      camera_roles: normalizeRecordingCameraRoles(source.camera_roles),
      pending_command: source.pending_command || null,
    };
  }

  function normalizeInferenceResetState() {
    const source = app.state.workflow?.inference?.reset;
    const reset = source && typeof source === "object" ? source : {};
    return {
      ...reset,
      available: reset.available === true,
      can: reset.can === true,
      paused: reset.paused === true,
      canPause: reset.can_pause === true,
      canResume: reset.can_resume === true,
      reset_pending: reset.reset_pending === true,
      blockers: Array.isArray(reset.blockers)
        ? reset.blockers.map((item) => String(item || "")).filter(Boolean)
        : [],
      error: String(reset.error || "").trim(),
      pending_command: reset.pending_command || null,
      last_command: reset.last_command || null,
    };
  }

  function recordingPresentationStatus(recording) {
    if (!recording.configured) return "disabled";
    if (recording.error) return "error";
    const status = recording.status;
    if (/error|fail/.test(status)) return "error";
    if (/final|sav|flush/.test(status)) return "finalizing";
    if (/transition|home|restart/.test(status)) return "transitioning";
    if (/record|active|running/.test(status) || recording.active) return "recording";
    if (/wait/.test(status)) return "waiting";
    if (/start|init|prepar/.test(status)) return "starting";
    if (/ready/.test(status) || recording.recorder_ready) return "ready";
    return "stopped";
  }

  function formatRecordingElapsed(value) {
    const totalTenths = Math.max(0, Math.floor((asNumber(value) || 0) / 100));
    const tenths = totalTenths % 10;
    const totalSeconds = Math.floor(totalTenths / 10);
    const seconds = totalSeconds % 60;
    const totalMinutes = Math.floor(totalSeconds / 60);
    const minutes = totalMinutes % 60;
    const hours = Math.floor(totalMinutes / 60);
    const secondText = String(seconds).padStart(2, "0");
    if (hours > 0) {
      return `${hours}:${String(minutes).padStart(2, "0")}:${secondText}.${tenths}`;
    }
    return `${String(totalMinutes).padStart(2, "0")}:${secondText}.${tenths}`;
  }

  function liveRecordingElapsedMs(recording) {
    const base = recording.episode?.elapsed_ms || 0;
    if (!recording.active || !app.recordingObservedAtMs) return base;
    return base + Math.max(0, Date.now() - app.recordingObservedAtMs);
  }

  function recordingEpisodeLabel(episode) {
    if (!episode) return "EP —";
    if (episode.index !== null) return `EP ${episode.index}`;
    const id = String(episode.id ?? "—");
    return `EP ${id.length > 12 ? `${id.slice(0, 12)}…` : id}`;
  }

  function recordingCapabilityBlocker(recording) {
    for (const action of RECORDING_ACTIONS) {
      const blocker = recording.capabilities[action]?.blockers?.[0];
      if (blocker) return blocker;
    }
    return "";
  }

  function renderRecordingButtons(recording, statusKey) {
    const showRecordingActions = recording.enabled === true;
    const reset = normalizeInferenceResetState();
    if (elements.recordingActionGrid) {
      elements.recordingActionGrid.hidden = !showRecordingActions;
    }
    if (elements.standaloneResetControl) {
      elements.standaloneResetControl.hidden = showRecordingActions;
    }
    const sharedPending = Boolean(app.recordingActionPending || recording.pending_command);
    for (const button of elements.recordingActionButtons) {
      const action = button.dataset.recordingAction;
      const capability = recording.capabilities[action] || { can: false, blockers: [] };
      const shortcut = RECORDING_ACTION_SHORTCUTS[action];
      const isOwnPending = app.recordingActionPending === action;
      button.disabled = !showRecordingActions || !capability.can || sharedPending || controlsFailClosed("recording") || app.configSwitching;
      button.classList.toggle("is-busy", isOwnPending);
      setText(button.querySelector("[data-recording-shortcut-label]"), shortcut?.key || "");
      if (shortcut) button.setAttribute("aria-keyshortcuts", shortcut.key);
      button.title = [
        ...capability.blockers,
        shortcut ? `快捷键 ${shortcut.key}` : "",
      ].filter(Boolean).join("；");
    }

    const pendingAction = String(
      app.inferenceControlPending
        || reset.pending_command?.action
        || (reset.reset_pending ? "reset" : "")
        || ""
    );
    const controlPending = Boolean(pendingAction);
    const pauseResumeAction = reset.paused ? "resume" : "pause";
    if (elements.standalonePauseButton) {
      elements.standalonePauseButton.disabled = showRecordingActions
        || (reset.paused ? !reset.canResume : !reset.canPause)
        || controlPending
        || controlsFailClosed(pauseResumeAction)
        || app.configSwitching;
      elements.standalonePauseButton.classList.toggle(
        "is-busy",
        pendingAction === "pause" || pendingAction === "resume",
      );
      elements.standalonePauseButton.textContent = reset.paused
        ? "继续推理"
        : "暂停推理";
      elements.standalonePauseButton.title = reset.blockers.join("；");
    }
    if (elements.standaloneResetButton) {
      elements.standaloneResetButton.disabled = showRecordingActions
        || !reset.can
        || controlPending
        || controlsFailClosed("reset")
        || app.configSwitching;
      elements.standaloneResetButton.classList.toggle(
        "is-busy",
        pendingAction === "reset",
      );
      elements.standaloneResetButton.title = reset.blockers.join("；");
    }

    if (elements.recordingActionStatus) {
      const standaloneStatus = pendingAction === "reset"
        ? "复位中"
        : pendingAction === "pause"
          ? "暂停中"
          : pendingAction === "resume"
            ? "继续中"
            : reset.paused
              ? "推理已暂停"
              : reset.can
                ? "推理运行中"
                : "等待推理";
      elements.recordingActionStatus.textContent = showRecordingActions
        ? (recording.pending_command
            ? "命令处理中"
            : RECORDING_STATUS_LABELS[statusKey] || "状态未知")
        : standaloneStatus;
    }
    if (!elements.recordingActionHint) return;
    if (!showRecordingActions) {
      const pendingHint = pendingAction === "reset"
        ? "正在复位，运行时与模型保持加载"
        : pendingAction === "pause"
          ? "正在暂停推理"
          : pendingAction === "resume"
            ? "正在继续推理"
            : "";
      elements.recordingActionHint.textContent = reset.error
        || pendingHint
        || reset.blockers[0]
        || (reset.paused ? "推理已暂停，运行时与模型保持加载" : "推理已就绪");
      return;
    }
    if (recording.error) {
      elements.recordingActionHint.textContent = recording.error;
    } else if (app.recordingActionPending || recording.pending_command) {
      elements.recordingActionHint.textContent = "正在提交录制操作，请稍候。";
    } else if (!recording.configured) {
      elements.recordingActionHint.textContent = "当前配置未启用 infer_rollout 录制。";
    } else if (!recording.episode) {
      elements.recordingActionHint.textContent = "推理启动后等待首条 action 自动开始录制。";
    } else {
      const blocker = recordingCapabilityBlocker(recording);
      elements.recordingActionHint.textContent = blocker || "整段操作会结束并回 Home；片段标记会继续录制。";
    }
  }

  function syncRecordingTrackNodes(container, items, updateNode) {
    if (!container) return;
    while (container.children.length < items.length) {
      container.appendChild(document.createElement("span"));
      countPerf("timelineNodesCreated");
    }
    while (container.children.length > items.length) {
      container.lastElementChild.remove();
      countPerf("timelineNodesRemoved");
    }
    items.forEach((item, index) => updateNode(container.children[index], item, index));
  }

  function renderRecordingTimeline(recording, statusKey) {
    const episode = recording.episode;
    const hasEpisode = Boolean(episode && (episode.id !== null || episode.index !== null || recording.active));
    const elapsedMs = liveRecordingElapsedMs(recording);
    if (elements.recordingStateBadge) {
      elements.recordingStateBadge.dataset.status = statusKey;
      elements.recordingStateBadge.textContent = RECORDING_STATUS_LABELS[statusKey] || "状态未知";
    }
    if (elements.recordingEpisodeLabel) {
      elements.recordingEpisodeLabel.textContent = recordingEpisodeLabel(episode);
    }
    if (elements.recordingFrameCount) {
      elements.recordingFrameCount.textContent = `${Math.round(episode?.frame_count || 0)} frames`;
    }
    if (elements.recordingElapsed) {
      elements.recordingElapsed.textContent = formatRecordingElapsed(elapsedMs);
      elements.recordingElapsed.dateTime = `PT${(elapsedMs / 1000).toFixed(1)}S`;
    }
    if (elements.recordingTimelineError) {
      elements.recordingTimelineError.hidden = !recording.error;
      elements.recordingTimelineError.textContent = recording.error;
      elements.recordingTimelineError.title = recording.error;
    }

    if (!hasEpisode) {
      if (elements.recordingTimelineEmpty) {
        elements.recordingTimelineEmpty.hidden = false;
        elements.recordingTimelineEmpty.textContent = recording.configured
          ? "等待推理写入首条 action 后自动开始录制。"
          : "当前配置未启用推理录制。";
      }
      if (elements.recordingTimelineContent) elements.recordingTimelineContent.hidden = true;
      return;
    }

    if (elements.recordingTimelineEmpty) elements.recordingTimelineEmpty.hidden = true;
    if (elements.recordingTimelineContent) elements.recordingTimelineContent.hidden = false;
    if (elements.recordingScaleEnd) elements.recordingScaleEnd.textContent = formatRecordingElapsed(elapsedMs);

    const maximumFrame = Math.max(
      1,
      episode.frame_count,
      ...recording.segments.map((item) => item.end_frame),
      ...recording.checkpoints.map((item) => item.frame)
    );
    syncRecordingTrackNodes(elements.recordingTrackSegments, recording.segments, (node, segment) => {
      const start = Math.min(maximumFrame, Math.max(0, segment.start_frame));
      const end = Math.min(maximumFrame, Math.max(start, segment.end_frame));
      setClassName(node, `recording-track-segment recording-track-segment--${segment.tag}`);
      setStyleValue(node, "left", `${(start / maximumFrame) * 100}%`);
      setStyleValue(node, "width", `${Math.max(0, ((end - start) / maximumFrame) * 100)}%`);
      const title = `${segment.tag === "accept" ? "接受" : "拒绝"} · frames ${Math.round(start)}–${Math.round(end)}`;
      if (node.title !== title) node.title = title;
    });
    syncRecordingTrackNodes(elements.recordingTrackCheckpoints, recording.checkpoints, (node, checkpoint) => {
      const frame = Math.min(maximumFrame, checkpoint.frame);
      setClassName(node, "recording-track-checkpoint");
      setStyleValue(node, "left", `${(frame / maximumFrame) * 100}%`);
      const title = `Checkpoint ${checkpoint.index ?? ""} · frame ${Math.round(frame)} · ${formatRecordingElapsed(checkpoint.elapsed_ms)}`;
      if (node.title !== title) node.title = title;
    });
    if (elements.recordingTrackCursor) {
      const currentFrame = Math.min(maximumFrame, episode.frame_count);
      elements.recordingTrackCursor.style.left = `${(currentFrame / maximumFrame) * 100}%`;
      elements.recordingTrackCursor.style.right = "auto";
      elements.recordingTrackCursor.classList.toggle("is-live", recording.active);
    }
    if (elements.recordingTrack) {
      elements.recordingTrack.setAttribute(
        "aria-label",
        `${recordingEpisodeLabel(episode)}，${Math.round(episode.frame_count)} frames，${recording.segments.length} 个已标片段，${recording.checkpoints.length} 个 checkpoint`
      );
    }
    if (elements.recordingReference) {
      const reference = recording.reference && typeof recording.reference === "object"
        ? [recording.reference.kind, recording.reference.name].filter(Boolean).join(" · ")
        : "";
      elements.recordingReference.textContent = reference || "按当前 episode 范围缩放";
      elements.recordingReference.title = elements.recordingReference.textContent;
    }
  }

  function renderRecording() {
    const recording = normalizeRecordingState();
    const statusKey = recordingPresentationStatus(recording);
    for (const row of elements.recordingCameraRoles) {
      const camera = recording.camera_roles[row.dataset.cameraRole];
      if (!camera) continue;
      row.dataset.status = camera.status;
      setText(row.querySelector("[data-camera-role-device]"), camera.device_key || "—");
      setText(row.querySelector("[data-camera-role-topic]"), camera.topic || "—");
      setText(row.querySelector("[data-camera-role-shm]"), camera.shm_name || "—");
      const warning = recordingCameraWarning(camera.reason);
      setText(
        row.querySelector("[data-camera-role-status]"),
        `${RECORDING_CAMERA_STATUS_LABELS[camera.status]}${warning ? ` · ${warning}` : ""}`,
      );
      row.title = camera.reason || "";
    }
    renderRecordingButtons(recording, statusKey);
    renderRecordingTimeline(recording, statusKey);
    ensureRecordingClock();
  }

  function renderRecordingClock() {
    const recording = normalizeRecordingState();
    if (!recording.episode) return;
    const elapsedMs = liveRecordingElapsedMs(recording);
    if (elements.recordingElapsed) {
      setText(elements.recordingElapsed, formatRecordingElapsed(elapsedMs));
      const dateTime = `PT${(elapsedMs / 1000).toFixed(1)}S`;
      if (elements.recordingElapsed.dateTime !== dateTime) elements.recordingElapsed.dateTime = dateTime;
    }
    setText(elements.recordingScaleEnd, formatRecordingElapsed(elapsedMs));
  }

  function recordingIdempotencyKey() {
    if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
      return globalThis.crypto.randomUUID();
    }
    return `recording-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function recordingActionMessage(action) {
    return {
      episode_accept: "已提交整段接受，数据保留并将在复位后开始下一段",
      episode_discard: "已提交整段拒绝，数据和片段标签保留并将在复位后开始下一段",
      segment_accept: "已提交片段接受标记，录制继续",
      segment_reject: "已提交片段拒绝标记，录制继续",
    }[action] || "录制操作已提交";
  }

  function recordingShortcutTargetIsEditable(target) {
    if (!target) return false;
    if (target.isContentEditable === true) return true;
    return ["INPUT", "TEXTAREA", "SELECT"].includes(
      String(target.tagName || "").toUpperCase()
    );
  }

  function recordingShortcutDialogIsOpen() {
    return Boolean(
      elements.checkpointBrowserDialog?.open
      || elements.confirmDialog?.open
      || elements.gripperPermissionDialog?.open
    );
  }

  function handleRecordingShortcut(event) {
    if (
      event.defaultPrevented
      || event.repeat
      || event.isComposing
      || event.ctrlKey
      || event.altKey
      || event.metaKey
      || event.shiftKey
      || recordingShortcutTargetIsEditable(event.target)
      || recordingShortcutDialogIsOpen()
    ) {
      return;
    }
    const action = RECORDING_ACTIONS.find(
      (candidate) => RECORDING_ACTION_SHORTCUTS[candidate]?.code === event.code
    );
    if (!action) return;
    const button = elements.recordingActionButtons.find(
      (candidate) => candidate.dataset.recordingAction === action
    );
    if (!button || button.disabled) return;
    event.preventDefault();
    void runRecordingAction(button);
  }

  async function runRecordingAction(button) {
    const action = button?.dataset?.recordingAction;
    if (!RECORDING_ACTIONS.includes(action) || app.recordingActionPending || app.configSwitching) return;
    const expectedEpoch = app.stateEpoch;
    if (!(await ensureControlActionAuthorized("recording", expectedEpoch))) return;
    const recording = normalizeRecordingState();
    const capability = recording.capabilities[action];
    const episode = recording.episode;
    if (!capability?.can || recording.pending_command || !episode || episode.id === null) return;

    if (!(await ensureControlActionAuthorized("recording", expectedEpoch))) return;

    app.recordingActionPending = action;
    renderRecording();
    try {
      if (!(await ensureControlActionAuthorized("recording", expectedEpoch))) return;
      await api(`/api/recording/actions/${action}`, {
        method: "POST",
        body: JSON.stringify({
          episode_id: episode.id,
          expected_checkpoint_revision: episode.checkpoint_revision,
          idempotency_key: recordingIdempotencyKey(),
        }),
      });
      // The HTTP response only acknowledges command acceptance. Recording UI
      // state is updated by the authoritative /api/state observation below.
      showToast(recordingActionMessage(action), "success");
      await refreshState(false);
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        showToast("Episode 或 checkpoint 已变化，状态已刷新，请重试。", "error");
        await refreshState(false);
      } else {
        const message = errorMessage(error);
        setGlobalNotice(message);
        showToast(message, "error");
      }
    } finally {
      app.recordingActionPending = "";
      renderRecording();
    }
  }

  async function runStandaloneReset() {
    if (app.inferenceControlPending || app.configSwitching) return;
    const expectedEpoch = app.stateEpoch;
    if (!(await ensureControlActionAuthorized("reset", expectedEpoch))) return;
    const reset = normalizeInferenceResetState();
    const selectedConfig = String(app.state.selected_config || "");
    const expectedRevision = asNumber(app.state.revision);
    if (!reset.can || reset.pending_command || !selectedConfig || expectedRevision === null) return;

    app.inferenceControlPending = "reset";
    renderRecording();
    await yieldForPendingPaint();
    try {
      if (!(await ensureControlActionAuthorized("reset", expectedEpoch))) return;
      await api("/api/inference/reset", {
        method: "POST",
        body: JSON.stringify({
          idempotency_key: recordingIdempotencyKey(),
          expected_selected_config: selectedConfig,
          expected_revision: expectedRevision,
        }),
      });
      showToast("复位已提交", "success");
      await refreshState(false);
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        await refreshState(false);
      }
      const message = errorMessage(error);
      setGlobalNotice(message);
      showToast(message, "error");
    } finally {
      app.inferenceControlPending = "";
      renderRecording();
    }
  }

  async function runStandalonePauseResume() {
    if (app.inferenceControlPending || app.configSwitching) return;
    const expectedEpoch = app.stateEpoch;
    const control = normalizeInferenceResetState();
    const action = control.paused ? "resume" : "pause";
    const canControl = control.paused ? control.canResume : control.canPause;
    if (!(await ensureControlActionAuthorized(action, expectedEpoch))) return;
    const selectedConfig = String(app.state.selected_config || "");
    const expectedRevision = asNumber(app.state.revision);
    if (!canControl || control.pending_command || !selectedConfig || expectedRevision === null) return;

    app.inferenceControlPending = action;
    renderRecording();
    await yieldForPendingPaint();
    try {
      if (!(await ensureControlActionAuthorized(action, expectedEpoch))) return;
      await api(`/api/inference/${action}`, {
        method: "POST",
        body: JSON.stringify({
          idempotency_key: recordingIdempotencyKey(),
          expected_selected_config: selectedConfig,
          expected_revision: expectedRevision,
        }),
      });
      showToast(action === "pause" ? "暂停已提交" : "继续已提交", "success");
      await refreshState(false);
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        await refreshState(false);
      }
      const message = errorMessage(error);
      setGlobalNotice(message);
      showToast(message, "error");
    } finally {
      app.inferenceControlPending = "";
      renderRecording();
    }
  }

  async function runWorkflowAction(button) {
    const stage = button?.dataset?.stage;
    const action = button?.dataset?.action;
    if (!stage || !action || (app.workflowPending.has(stage) && action !== "stop")) return;
    if (action === "start" && stage !== "server") void loadPreflight();
    const inferenceOptions = normalizeInferenceOptions();
    const checkpointOwnerStage = String(
      app.state.workflow?.checkpoint_owner_stage || "inference"
    );
    const checkpointContext = stage === checkpointOwnerStage && action === "start"
      ? checkpointContextFromState()
      : null;
    const inferenceOptionsContext = stage === "inference" && action === "start"
      ? inferenceOptionsContextFromState()
      : null;
    const checkpointRequired = Boolean(
      checkpointContext
      && !(stage === "inference" && inferenceOptions.testMode)
    );
    if (checkpointContext || inferenceOptionsContext) {
      const checkpoint = checkpointStateForRender();
      if (
        (checkpointRequired && !checkpoint.valid)
        || (checkpointRequired && checkpointEditorLocked(checkpoint, {
          independentServerStage: stage === "server",
        }))
        || (inferenceOptionsContext && inferenceOptionsEditorLocked(inferenceOptions))
        || (checkpointContext && checkpointContext.revision === null)
        || inferenceOptionsContext?.revision === null
      ) {
        showToast(
          inferenceOptions.blockers[0]
          || (checkpointRequired ? checkpoint.error || checkpoint.blockers[0] : "")
          || `${stage === "server" ? "Server" : "推理"}启动上下文不可用，请刷新后重试。`,
          "error"
        );
        return;
      }
    }
    const expectedEpoch = app.stateEpoch;
    if (!(await ensureControlActionAuthorized(action, expectedEpoch, stage))) return;
    const runtimeCanPermissionRequired = Boolean(
      stage === "runtime" &&
      action === "start" &&
      normalizeGrippers().permission.affectedCanInterfaces.length
    );
    if (runtimeCanPermissionRequired) {
      const authorized = await requestGripperPermission({ force: true });
      if (!authorized || !(await ensureControlActionAuthorized(action, expectedEpoch, stage))) return;
    }
    const scope = `workflow:${stage}`;
    const transition = beginClientTransition(
      scope,
      action,
      action === "start" ? "running" : "stopped",
      action === "start" && stage === "runtime"
        ? 120000
        : action === "start" && stage === "server"
          ? 600000
          : action === "start" ? 30000 : 75000
    );
    app.workflowPending.add(stage);
    app.controlRequests.set(scope, transition.token);
    renderWorkflow();
    await yieldForPendingPaint();
    scheduleStatePoll(0);
    try {
      if (!(await ensureControlActionAuthorized(action, expectedEpoch, stage))) {
        clearClientTransition(scope, transition.token);
        return;
      }
      const request = {
        method: "POST",
        headers: {
          "X-Control-Transition": transition.token,
          Prefer: "respond-async",
        },
      };
      if (checkpointContext || inferenceOptionsContext) {
        const body = checkpointContext
          ? {
              expected_selected_config: checkpointContext.selectedConfig,
              expected_revision: checkpointContext.revision,
              expected_generation: checkpointContext.generation,
            }
          : {
              expected_selected_config: inferenceOptionsContext.selectedConfig,
              expected_revision: inferenceOptionsContext.revision,
            };
        if (checkpointContext) {
          body.expected_generation = checkpointContext.generation;
        }
        if (inferenceOptionsContext) {
          body.expected_options_generation = inferenceOptionsContext.generation;
        }
        request.body = JSON.stringify(body);
      }
      const operation = () => api(`/api/workflow/${stage}/${action}`, request);
      const result = action === "start"
        ? await withGripperPermissionRetry(
            operation,
            () => controlActionStillAuthorized(
              action,
              expectedEpoch,
              { workflowStage: stage },
            )
          )
        : await operation();
      if (result === null) {
        clearClientTransition(scope, transition.token);
        return;
      }
      const label = stage === "runtime" ? "运行时" : stage === "server" ? "Server" : "推理";
      showToast(`${label}${action === "start" ? "启动" : "停止"}指令已发送`, "success");
      scheduleStatePoll(100);
    } catch (error) {
      clearClientTransition(scope, transition.token);
      const staleContext = (checkpointContext || inferenceOptionsContext)
        && error instanceof ApiError
        && error.status === 409;
      const message = staleContext
        ? "Checkpoint、推理选项或 config 状态已变化，已刷新，请重试。"
        : errorMessage(error);
      setGlobalNotice(message);
      showToast(message, "error");
      if (staleContext) await refreshState(false);
    } finally {
      if (app.controlRequests.get(scope) === transition.token) {
        app.controlRequests.delete(scope);
      }
      if (!app.controlTransitions.has(scope)) app.workflowPending.delete(stage);
      renderWorkflow();
    }
  }

  function cameraEndpoint(device, suffix) {
    // RealSense and every other camera use the server-owned class/id route.
    // Never derive preview URLs from a client-composed shared-memory name.
    return `/api/devices/${encodeURIComponent(device.className)}/${encodeURIComponent(String(device.id))}/${suffix}`;
  }

  function normalizePreviewMetadata(payload) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) return {};

    // The current API returns metadata directly at the top level. Keep that
    // object intact when `status` is a string; older wrappers may nest the
    // metadata under `metadata`, `data.metadata`, or an object-valued `status`.
    const nested = firstDefined(
      payload.metadata && typeof payload.metadata === "object" ? payload.metadata : undefined,
      payload.data?.metadata && typeof payload.data.metadata === "object" ? payload.data.metadata : undefined,
      payload.status && typeof payload.status === "object" ? payload.status : undefined
    );
    return nested ? { ...payload, ...nested } : payload;
  }

  function cameraSlotRequirement(slot) {
    if (!slot?.configured || !slot.deviceKey) {
      return { label: "未使用", kind: "unused" };
    }
    if (slot.inferenceRequired) {
      return { label: "推理必需", kind: "required" };
    }
    if (slot.recordingConfigured) {
      return { label: "录制可选", kind: "optional" };
    }
    return { label: "可选", kind: "optional" };
  }

  function cameraSlotForDevice(device) {
    return CAMERA_SLOT_ORDER
      .map((role) => app.cameraSlots[role])
      .find((slot) => slot?.deviceKey === device?.key) || null;
  }

  function cameraCardForDevice(device) {
    const slot = cameraSlotForDevice(device);
    return slot ? app.cameraCards.get(slot.role) : null;
  }

  function cameraDeviceForSlot(slot) {
    if (!slot?.deviceKey) return null;
    return app.devices.find((device) => device.key === slot.deviceKey)
      || app.configDevices.find((device) => device.key === slot.deviceKey)
      || null;
  }

  function createCameraCard(slot, device) {
    const card = document.createElement("article");
    card.className = `camera-card camera-card--${slot.role}`;
    card.dataset.cameraRole = slot.role;
    card.dataset.deviceKey = device?.key || slot.deviceKey || "";
    card.dataset.hasDevice = device ? "true" : "false";

    const header = document.createElement("div");
    header.className = "camera-card__header";
    const title = document.createElement("div");
    title.className = "camera-card__title";
    title.append(
      makeTextElement(
        "strong",
        "camera-name",
        device ? `${CAMERA_SLOT_LABELS[slot.role]} · ${device.label}` : CAMERA_SLOT_LABELS[slot.role]
      ),
      makeTextElement(
        "span",
        "camera-shm-name",
        device ? device.shm.name : (slot.deviceKey || "当前 config 未配置")
      )
    );
    const requirement = cameraSlotRequirement(slot);
    const requirementBadge = makeTextElement(
      "span",
      "camera-slot-requirement",
      requirement.label
    );
    requirementBadge.dataset.role = "camera-slot-requirement";
    requirementBadge.dataset.kind = requirement.kind;
    const badge = makeTextElement("span", "state-badge", device ? "已停止" : "未使用");
    badge.dataset.role = "camera-status";
    const headerControls = document.createElement("div");
    headerControls.className = "camera-card__controls";
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "panel-device-toggle panel-device-toggle--camera";
    toggle.dataset.role = "camera-device-toggle";
    if (device) toggle.dataset.deviceKey = device.key;
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-checked", "false");
    toggle.disabled = true;
    toggle.textContent = device ? "启动" : "未使用";
    if (device?.supportsCaptureBinding) {
      const bindingSelect = document.createElement("select");
      bindingSelect.className = "camera-binding-select";
      bindingSelect.dataset.role = "camera-binding-select";
      bindingSelect.dataset.deviceKey = device.key;
      bindingSelect.setAttribute("aria-label", `${CAMERA_SLOT_LABELS[slot.role]}物理摄像头`);
      bindingSelect.disabled = true;
      headerControls.append(bindingSelect);
    } else if (!device) {
      const bindingSelect = document.createElement("select");
      bindingSelect.className = "camera-binding-select";
      bindingSelect.setAttribute("aria-label", `${CAMERA_SLOT_LABELS[slot.role]}物理摄像头`);
      bindingSelect.disabled = true;
      bindingSelect.append(new Option("当前 config 未使用", ""));
      headerControls.append(bindingSelect);
    }
    headerControls.append(requirementBadge, toggle, badge);
    header.append(title, headerControls);

    const frame = document.createElement("div");
    frame.className = "camera-frame";
    const image = document.createElement("img");
    image.alt = `${CAMERA_SLOT_LABELS[slot.role]}实时画面`;
    image.hidden = true;
    image.decoding = "async";
    const placeholder = document.createElement("div");
    placeholder.className = "camera-placeholder";
    placeholder.dataset.role = "camera-placeholder";
    const icon = makeTextElement("span", "camera-placeholder__icon", "▣");
    icon.setAttribute("aria-hidden", "true");
    const placeholderText = makeTextElement(
      "span",
      "",
      device ? "启动相机后显示画面" : "当前 config 未使用此视角"
    );
    placeholderText.dataset.role = "camera-placeholder-text";
    placeholder.append(icon, placeholderText);
    frame.append(image, placeholder);

    const stats = document.createElement("div");
    stats.className = "camera-stats";
    const streamStats = makeTextElement(
      "span",
      "",
      device ? "画面未连接" : "不参与本次运行"
    );
    streamStats.dataset.role = "camera-stream-stats";
    const shape = makeTextElement("span", "", device ? shapeText(device.shape) : "—");
    shape.dataset.role = "camera-shape";
    stats.append(streamStats, shape);

    if (device) {
      image.addEventListener("load", () => {
        finishCameraPreviewFrame(device.key, image);
        if (image.dataset.streaming !== "1") {
          return;
        }
        image.hidden = false;
        placeholder.hidden = true;
        image.dataset.streaming = "1";
        scheduleCameraPreviewFrame(device, card);
      });
      image.addEventListener("error", () => {
        finishCameraPreviewFrame(device.key, image);
        if (image.dataset.streaming !== "1") {
          return;
        }
        image.hidden = true;
        image.dataset.streaming = "0";
        cancelCameraPreviewFrame(device.key);
        releaseCameraImage(image);
        placeholder.hidden = false;
        placeholderText.textContent = "画面连接失败，正在重试…";
        app.previewRetryAt.set(device.key, Date.now() + 2500);
      });
    } else {
      card.classList.add("camera-card--unused");
    }

    card.append(header, frame, stats);
    return card;
  }

  function setupCameraVisibilityObserver() {
    if (typeof globalThis.IntersectionObserver !== "function") return;
    app.cameraObserver = new globalThis.IntersectionObserver((entries) => {
      for (const entry of entries) {
        const card = entry.target;
        const key = card.dataset.deviceKey;
        const visible = entry.isIntersecting && entry.intersectionRatio > 0;
        if (app.cameraVisibility.get(key) === visible) continue;
        app.cameraVisibility.set(key, visible);
        const device = app.devices.find((item) => item.key === key);
        const slot = device ? cameraSlotForDevice(device) : null;
        if (device && slot) updateCameraCard(device, card, slot);
      }
      ensurePreviewPolling(true);
    }, { rootMargin: "160px 0px" });
  }

  function cameraCardCanStream(device, card) {
    if (document.hidden || !card) return false;
    if (typeof document.hasFocus === "function" && !document.hasFocus()) return false;
    if (!app.cameraObserver) return true;
    return app.cameraVisibility.get(device.key) === true;
  }

  function releaseCameraImage(image) {
    if (!image) return;
    const objectUrl = image.dataset.objectUrl || "";
    if (objectUrl && typeof globalThis.URL?.revokeObjectURL === "function") {
      globalThis.URL.revokeObjectURL(objectUrl);
    }
    delete image.dataset.objectUrl;
    if (image.hasAttribute("src")) image.removeAttribute("src");
  }

  function finishCameraPreviewFrame(deviceKey, image) {
    app.previewFrameInFlight.delete(deviceKey);
    if (image) image.dataset.loading = "0";
  }

  function cancelCameraPreviewFrame(deviceKey) {
    const timer = app.previewFrameTimers.get(deviceKey);
    if (timer !== undefined) window.clearTimeout(timer);
    app.previewFrameTimers.delete(deviceKey);
    app.previewFrameInFlight.delete(deviceKey);
  }

  function requestCameraPreviewFrame(device, card) {
    const image = card.querySelector("img");
    if (
      !image ||
      image.dataset.streaming !== "1" ||
      image.dataset.loading === "1" ||
      !cameraCardCanStream(device, card)
    ) return;
    if (
      app.refreshing ||
      app.previewFrameInFlight.size >= MAX_CONCURRENT_PREVIEW_FRAMES
    ) {
      scheduleCameraPreviewFrame(device, card, PREVIEW_FRAME_BACKPRESSURE_MS);
      return;
    }
    const retryAt = app.previewRetryAt.get(device.key) || 0;
    if (Date.now() < retryAt) {
      scheduleCameraPreviewFrame(device, card, retryAt - Date.now());
      return;
    }
    app.previewFrameInFlight.add(device.key);
    image.dataset.loading = "1";
    image.src = `${cameraEndpoint(device, "preview.jpg")}?client_version=${encodeURIComponent(PREVIEW_BROWSER_CLIENT_VERSION)}&t=${Date.now()}`;
  }

  function scheduleCameraPreviewFrame(
    device,
    card,
    delayMs = POLL_INTERVALS.previewFrame
  ) {
    cancelCameraPreviewFrame(device.key);
    const image = card.querySelector("img");
    if (
      !image ||
      image.dataset.streaming !== "1" ||
      !cameraCardCanStream(device, card)
    ) return;
    const timer = window.setTimeout(() => {
      if (app.previewFrameTimers.get(device.key) !== timer) return;
      app.previewFrameTimers.delete(device.key);
      requestCameraPreviewFrame(device, card);
    }, Math.max(0, delayMs));
    app.previewFrameTimers.set(device.key, timer);
  }

  function updateCameraBindingControls(device, card) {
    const select = card.querySelector('[data-role="camera-binding-select"]');
    const toggle = card.querySelector('[data-role="camera-device-toggle"]');
    if (!select) {
      updateDeviceToggle(toggle, device, "相机", true);
      return;
    }
    const selected = app.cameraBindings.get(device.key) || "";
    const usedByLockedOthers = new Set(
      bindingDevices()
        .filter((other) => (
          other.key !== device.key
          && other.isCamera
          && other.supportsCaptureBinding
          && bindingTargetLocked(
            app.devices.find((runtime) => runtime.key === other.key) || other
          )
        ))
        .map(resolvedCameraBindingId)
        .filter(Boolean)
    );
    const available = app.discovery.cameras.filter((camera) => camera.detected && camera.capture);
    const items = [{
      label: app.hardwareLoading
        ? "正在扫描（暂时停放此槽位）"
        : "缺失 / 未绑定（不可启动）",
      value: "",
      disabled: false,
    }, ...app.discovery.cameras.map((camera) => ({
      label: cameraBindingLabel(camera),
      value: camera.id,
      // Stopped slots can exchange cameras atomically. A camera owned by a
      // running slot remains unavailable until that slot is stopped.
      disabled: !camera.detected || !camera.capture || usedByLockedOthers.has(camera.id),
    }))];
    syncSelectOptions(
      select,
      items,
      availableCameraBinding(selected) ? selected : ""
    );
    const locked = bindingTargetLocked(device);
    select.disabled = locked;
    select.title = locked
      ? "停止此设备后可修改绑定"
      : available.length
        ? "选择一个已发现的相机；缺失槽位会保持停放且不可启动"
        : "未发现可用相机；此槽位保持停放且不可启动";
    card.classList.toggle("has-binding", Boolean(select.value));
    card.classList.toggle("has-missing-binding", !select.value);
    // Empty is an explicit parked slot, not an instruction to reuse the
    // config's capture_id. The backend keeps the logical device visible but
    // rejects its start until a detected physical camera is selected.
    // A valid draft selection is enough to enable Start. runToggleButton first
    // commits a selection that differs from effective_bindings, then starts the
    // rebuilt backend device. Requiring the old backend start capability here
    // creates a deadlock for initially parked camera slots.
    const bindingReady = Boolean(availableCameraBinding(selected));
    updateDeviceToggle(toggle, device, "相机", bindingReady);
  }

  function stopCameraPreview(card) {
    const image = card.querySelector("img");
    if (!image) return;
    cancelCameraPreviewFrame(card.dataset.deviceKey || "");
    if (image.dataset.streaming === "0" && !image.hasAttribute("src")) return;
    image.dataset.streaming = "0";
    image.dataset.loading = "0";
    releaseCameraImage(image);
    image.hidden = true;
  }

  function startCameraPreview(device, card) {
    const image = card.querySelector("img");
    if (!image || !cameraCardCanStream(device, card) || image.dataset.streaming === "1") return;
    const retryAt = app.previewRetryAt.get(device.key) || 0;
    if (Date.now() < retryAt) return;
    image.hidden = false;
    image.dataset.streaming = "1";
    image.dataset.loading = "0";
    requestCameraPreviewFrame(device, card);
  }

  function updateCameraCard(device, card, slot) {
    card.dataset.status = presentationStatus(device.status);
    card.dataset.lifecycleStatus = device.status;
    card.dataset.progressPhase = deviceProgressPhase(device);
    const name = card.querySelector(".camera-name");
    const shmName = card.querySelector(".camera-shm-name");
    const badge = card.querySelector('[data-role="camera-status"]');
    const placeholder = card.querySelector('[data-role="camera-placeholder"]');
    const placeholderText = card.querySelector('[data-role="camera-placeholder-text"]');
    const stats = card.querySelector('[data-role="camera-stream-stats"]');
    const shape = card.querySelector('[data-role="camera-shape"]');
    const requirementBadge = card.querySelector('[data-role="camera-slot-requirement"]');
    const previewStatus = app.previewStatuses.get(device.key) || {};
    const requirement = cameraSlotRequirement(slot);
    const transition = activeControlTransition(`device:${device.key}`);
    const transitionDesired = controlTransitionDesired(
      `device:${device.key}`,
      device.status
    );
    const transitioning = Boolean(transition) || TRANSITION_STATUSES.has(device.status);
    const transitionFallback = transitionDesired === "stopped" ? "停止中" : "启动中";
    const progressLabel = deviceProgressLabel(
      device,
      STATUS_LABELS[device.status] || transitionFallback,
      transition ? transitionDesired : ""
    );
    card.classList.toggle("is-busy", transitioning);

    setText(name, `${CAMERA_SLOT_LABELS[slot.role]} · ${device.label}`);
    setText(shmName, device.shm.name);
    setText(requirementBadge, requirement.label);
    if (requirementBadge) requirementBadge.dataset.kind = requirement.kind;
    setClassName(badge, `state-badge state-badge--${presentationStatus(device.status)}`);
    setText(badge, STATUS_LABELS[device.status]);
    setText(shape, shapeText(firstDefined(previewStatus.shape, device.shape)));

    const shouldStream = (
      (device.status === "running" || device.status === "degraded") &&
      cameraCardCanStream(device, card)
    );
    if (shouldStream) {
      setText(placeholderText, device.shm.connected ? "正在连接画面…" : "等待共享内存…");
      startCameraPreview(device, card);
      const fps = firstDefined(asNumber(previewStatus.fps), asNumber(device.shm.fps));
      const age = firstDefined(asNumber(previewStatus.age_ms), asNumber(previewStatus.frame_age_ms), device.shm.ageMs);
      setText(stats, `${fps ? `${fps.toFixed(1)} fps` : "实时流"} · ${formatAge(age)}`);
    } else {
      stopCameraPreview(card);
      placeholder.hidden = false;
      setText(
        placeholderText,
        transitioning ? progressLabel : "启动相机后显示画面"
      );
      setText(
        stats,
        transitioning ? progressLabel : "画面未连接"
      );
    }
    updateCameraBindingControls(device, card);
  }

  function renderCameras() {
    const slots = CAMERA_SLOT_ORDER.map((role) => app.cameraSlots[role]);
    const newCards = [];

    for (const slot of slots) {
      const device = cameraDeviceForSlot(slot);
      let card = app.cameraCards.get(slot.role);
      const nextDeviceKey = device?.key || slot.deviceKey || "";
      const nextHasDevice = device ? "true" : "false";
      if (
        card
        && (
          card.dataset.deviceKey !== nextDeviceKey
          || card.dataset.hasDevice !== nextHasDevice
        )
      ) {
        const previousKey = card.dataset.deviceKey;
        stopCameraPreview(card);
        app.cameraObserver?.unobserve(card);
        card.remove();
        app.cameraCards.delete(slot.role);
        if (previousKey) {
          cancelPreviewStatusRequest(previousKey);
          app.previewStatuses.delete(previousKey);
          app.previewRetryAt.delete(previousKey);
          app.cameraVisibility.delete(previousKey);
        }
        card = null;
      }
      if (!card) {
        card = createCameraCard(slot, device);
        app.cameraCards.set(slot.role, card);
        if (device && app.cameraObserver) app.cameraVisibility.set(device.key, false);
        newCards.push(card);
      }
      if (device) updateCameraCard(device, card, slot);
    }

    const orderKey = slots
      .map((slot) => `${slot.role}:${slot.deviceKey}:${slot.required ? 1 : 0}`)
      .join("\u0000");
    if (app.renderKeys.cameraOrder !== orderKey) {
      for (const slot of slots) elements.cameraGrid.appendChild(app.cameraCards.get(slot.role));
      app.renderKeys.cameraOrder = orderKey;
      countPerf("cameraCardReorders");
    }
    for (const card of newCards) {
      if (card.dataset.hasDevice === "true") app.cameraObserver?.observe(card);
    }

    ensurePreviewPolling();
  }

  function cancelPreviewStatusRequest(deviceKey) {
    const entry = app.previewStatusRequests.get(deviceKey);
    if (!entry) return;
    entry.controller.abort();
    app.previewStatusRequests.delete(deviceKey);
  }

  async function refreshPreviewStatus(device) {
    const existing = app.previewStatusRequests.get(device.key);
    if (existing) {
      countPerf("overlappingPollsSkipped");
      return existing.promise;
    }
    const controller = new AbortController();
    const entry = { controller, promise: null };
    const timeout = window.setTimeout(
      () => controller.abort(),
      POLL_INTERVALS.previewRequestTimeout
    );
    const request = (async () => {
      try {
        const payload = await api(cameraEndpoint(device, "preview/status"), {
          signal: controller.signal,
        });
        app.previewStatuses.set(device.key, normalizePreviewMetadata(payload));
      } catch (error) {
        if (error?.name !== "AbortError" && (
          !(error instanceof ApiError) || ![404, 405].includes(error.status)
        )) {
          app.previewStatuses.set(device.key, { error: errorMessage(error) });
        }
      } finally {
        window.clearTimeout(timeout);
        if (app.previewStatusRequests.get(device.key) === entry) {
          app.previewStatusRequests.delete(device.key);
        }
        const current = app.devices.find((item) => item.key === device.key);
        const card = current ? cameraCardForDevice(current) : null;
        const slot = current ? cameraSlotForDevice(current) : null;
        if (current && card && slot) updateCameraCard(current, card, slot);
      }
    })();
    entry.promise = request;
    app.previewStatusRequests.set(device.key, entry);
    return request;
  }

  async function refreshPreviewStatuses() {
    if (document.hidden) return;
    const cameras = app.devices.filter(
      (device) => (
        device.isCamera &&
        (device.status === "running" || device.status === "degraded") &&
        cameraCardCanStream(device, cameraCardForDevice(device))
      )
    );
    if (!cameras.length) return;
    countPerf("previewPolls");
    for (const device of cameras) void refreshPreviewStatus(device);
  }

  function isMotionRisk(device) {
    return device.risk.toLowerCase() === "motion" || /(robot|teleoperator|jointslider|airexoskeleton)/i.test(device.className);
  }

  function confirmOperation(title, message, confirmText) {
    if (!elements.confirmDialog || typeof elements.confirmDialog.showModal !== "function") {
      return Promise.resolve(window.confirm(message));
    }
    elements.confirmTitle.textContent = title;
    elements.confirmMessage.textContent = message;
    elements.confirmActionButton.textContent = confirmText;
    elements.confirmDialog.showModal();
    return new Promise((resolve) => {
      elements.confirmDialog.addEventListener(
        "close",
        () => resolve(elements.confirmDialog.returnValue === "confirm"),
        { once: true }
      );
    });
  }

  async function runForceStopAll() {
    if (app.forceStopPending || app.state.server?.force_stop_in_progress === true) return;
    const firstConfirmation = await confirmOperation(
      "强制关闭所有进程",
      "该操作会立即终止控制面板拥有的设备、运行时、推理、录制和夹爪子进程。不会执行机械臂复位、Yam 归零确认、优雅 Disable、夹爪保持确认或录制收尾。外部占用进程不会被终止。",
      "继续确认"
    );
    if (!firstConfirmation) return;
    const finalConfirmation = await confirmOperation(
      "最终安全确认",
      "机械臂可能仍处于使能或保持状态，未完成的录制可能只能按异常数据恢复。确定立即发送 SIGKILL 吗？",
      "立即强制关闭"
    );
    if (!finalConfirmation) return;

    app.forceStopPending = true;
    renderForceStopAll();
    await yieldForPendingPaint();
    scheduleStatePoll(0);
    try {
      const result = await api("/api/processes/force-stop-all", {
        method: "POST",
        body: JSON.stringify({ confirmation: "FORCE_STOP_ALL" }),
      });
      const killed = Array.isArray(result?.killed_pids) ? result.killed_pids.length : 0;
      if (result?.ok === true) {
        clearControlFailure();
        showToast(`已强制关闭 ${killed} 个控制面板子进程`, "success");
      } else {
        const stages = Array.isArray(result?.incomplete_stages)
          ? result.incomplete_stages.join("、")
          : "未知阶段";
        recordControlFailure("force:all", `强制关闭未完全完成：${stages}`);
      }
    } catch (error) {
      recordControlFailure("force:all", errorMessage(error));
    } finally {
      app.forceStopPending = false;
      renderForceStopAll();
      scheduleStatePoll(0);
      await refreshState(false);
    }
  }

  async function runDeviceAction(device, action) {
    if (app.pendingActions.has(device.key) && action !== "stop") return;
    const expectedEpoch = app.stateEpoch;
    if (!(await ensureControlActionAuthorized(action, expectedEpoch))) return;

    if (action === "stop" && isI2rtHeadDevice(device)) {
      const confirmed = await confirmOperation(
        "Yam 头臂安全断连",
        "断连前将先回到 joint 全 0，并连续读取真实 joint 状态确认到位。此断连零位不同于启动后的工作 homepose；如果无法确认到位，系统会保持连接并继续锁住零位目标。",
        "确认归零并断连"
      );
      if (!confirmed) return;
    } else if ((action === "start" && isMotionRisk(device)) || action === "restart") {
      const verb = action === "start" ? "启动" : "重启";
      const confirmed = await confirmOperation(
        `${verb}设备`,
        `${verb} ${device.className}:${device.id}${isMotionRisk(device) ? " 可能使机械设备运动。" : " 会短暂中断数据。"}`,
        `确认${verb}`
      );
      if (!confirmed) return;
    }
    if (!(await ensureControlActionAuthorized(action, expectedEpoch))) return;
    const yamCanPermissionRequired = Boolean(
      action !== "stop" &&
      isI2rtHeadDevice(device) &&
      normalizeGrippers().permission.affectedCanInterfaces.length
    );
    if (yamCanPermissionRequired) {
      const authorized = await requestGripperPermission({ force: true });
      if (!authorized || !(await ensureControlActionAuthorized(action, expectedEpoch))) return;
    }

    clearControlFailure();
    app.pendingActions.add(device.key);
    const scope = `device:${device.key}`;
    const transition = beginClientTransition(
      scope,
      action,
      action === "stop" ? "stopped" : "running",
      action === "stop" ? 75000 : 45000
    );
    app.controlRequests.set(scope, transition.token);
    renderBindingControls();
    await yieldForPendingPaint();
    scheduleStatePoll(0);

    try {
      if (!(await ensureControlActionAuthorized(action, expectedEpoch))) {
        clearClientTransition(scope, transition.token);
        return;
      }
      const operation = () => api(cameraEndpoint(device, action), {
        method: "POST",
        headers: {
          "X-Control-Transition": transition.token,
          Prefer: "respond-async",
        },
      });
      const result = isI2rtHeadDevice(device) && action !== "stop"
        ? await withGripperPermissionRetry(
            operation,
            () => controlActionStillAuthorized(action, expectedEpoch)
          )
        : await operation();
      if (result === null) {
        clearClientTransition(scope, transition.token);
        return;
      }
      const actionLabel = action === "stop"
        ? (isI2rtHeadDevice(device) ? "归零并断连指令" : "停止指令")
        : action === "restart"
          ? "重启指令"
          : "启动指令";
      showToast(`${device.className}:${device.id} ${actionLabel}已发送`, "success");
      scheduleStatePoll(100);
    } catch (error) {
      clearClientTransition(scope, transition.token);
      const message = errorMessage(error);
      recordControlFailure(scope, message);
      revealPendingFailureLog();
    } finally {
      if (app.controlRequests.get(scope) === transition.token) {
        app.controlRequests.delete(scope);
      }
      if (!app.controlTransitions.has(scope)) app.pendingActions.delete(device.key);
      renderBindingControls();
    }
  }

  function renderLogDeviceOptions() {
    const current = elements.logDeviceSelect.value;
    const signature = app.devices.map((device) => [device.key, device.className, device.id]);
    const key = JSON.stringify(signature);
    if (app.renderKeys.logDevices === key) return;
    const options = [new Option("全部设备", "")];
    for (const device of app.devices) {
      options.push(new Option(`${device.className}:${device.id}`, device.key));
    }
    elements.logDeviceSelect.replaceChildren(...options);
    app.renderKeys.logDevices = key;
    countPerf("optionReplacements");
    if ([...elements.logDeviceSelect.options].some((option) => option.value === current)) {
      elements.logDeviceSelect.value = current;
    }
  }

  function normalizeLogItem(item, index) {
    if (typeof item === "string") {
      return { seq: null, timestamp: "", level: "", message: item, fingerprint: item };
    }
    const timestamp = firstDefined(item?.timestamp, item?.ts, item?.time, "");
    const level = String(firstDefined(item?.level, item?.severity, "")).toLowerCase();
    const message = String(firstDefined(item?.message, item?.msg, item?.text, JSON.stringify(item)));
    const seq = firstDefined(item?.seq, item?.id, item?.cursor, null);
    return {
      seq,
      timestamp,
      level,
      message,
      fingerprint: seq !== null ? `seq:${seq}` : `${timestamp}|${level}|${message}|${index}`,
    };
  }

  function formatLogTime(value) {
    if (!value) return "";
    if (typeof value === "string" && !/^\d+$/.test(value)) return value;
    const numeric = asNumber(value);
    if (numeric === null) return String(value);
    let epochMs = numeric;
    if (numeric > 1e14) epochMs = numeric / 1e6;
    else if (numeric < 1e11) epochMs = numeric * 1000;
    const date = new Date(epochMs);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleTimeString("zh-CN", { hour12: false });
  }

  function createLogRow(item) {
    const row = document.createElement("div");
    row.className = "log-line";
    const time = makeTextElement("span", "log-time", formatLogTime(item.timestamp));
    const level = makeTextElement("span", `log-level log-level--${item.level || "plain"}`, item.level ? item.level.toUpperCase() : "·");
    const message = makeTextElement("span", "log-message", item.message);
    row.append(time, level, message);
    countPerf("logRowsCreated");
    return row;
  }

  function appendLogRows(items) {
    if (!items.length) return;
    if (elements.logView.querySelector(".log-empty")) elements.logView.replaceChildren();
    const fragment = document.createDocumentFragment();
    for (const item of items) fragment.appendChild(createLogRow(item));
    elements.logView.appendChild(fragment);
    if (elements.logAutoscroll.checked) elements.logView.scrollTop = elements.logView.scrollHeight;
  }

  function renderLogs() {
    elements.logView.replaceChildren();
    if (!app.logItems.length) {
      const empty = makeTextElement("p", "log-empty", "暂无日志。");
      empty.id = "log-empty";
      elements.logView.appendChild(empty);
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const item of app.logItems) {
      fragment.appendChild(createLogRow(item));
    }
    elements.logView.appendChild(fragment);
    if (elements.logAutoscroll.checked) elements.logView.scrollTop = elements.logView.scrollHeight;
  }

  function resetLogs() {
    app.logRequestController?.abort();
    app.logCursor = null;
    app.logItems = [];
    app.logFingerprints.clear();
    renderLogs();
  }

  function revealPendingFailureLog() {
    const deviceKey = app.failureLogKey;
    if (!deviceKey || !elements.logDetails || !elements.logDeviceSelect) return;
    const options = elements.logDeviceSelect.options
      ? [...elements.logDeviceSelect.options]
      : [];
    if (!options.some((option) => option.value === deviceKey)) return;
    app.failureLogKey = "";
    elements.logDetails.open = true;
    const hint = elements.logDetails.querySelector(".summary-hint");
    if (hint) hint.textContent = "收起";
    if (app.logKey !== deviceKey || elements.logDeviceSelect.value !== deviceKey) {
      app.logKey = deviceKey;
      elements.logDeviceSelect.value = deviceKey;
      resetLogs();
    }
    ensureLogPolling(true);
  }

  async function loadLogs() {
    if (app.logLoading || document.hidden || !elements.logDetails?.open) return;
    app.logLoading = true;
    countPerf("logPolls");
    const controller = new AbortController();
    app.logRequestController = controller;
    try {
      const params = new URLSearchParams();
      if (app.logKey) {
        // `key` is the public UI contract; `device_key` keeps compatibility with
        // the current Flask controller implementation.
        params.set("key", app.logKey);
        params.set("device_key", app.logKey);
      }
      if (app.logCursor !== null && app.logCursor !== undefined) params.set("cursor", String(app.logCursor));
      const payload = await api(`/api/logs${params.size ? `?${params.toString()}` : ""}`, {
        signal: controller.signal,
      });
      const source = Array.isArray(payload)
        ? payload
        : firstDefined(payload?.items, payload?.logs, payload?.entries, []);
      if (!Array.isArray(source)) return;

      const normalized = source.map(normalizeLogItem);
      const appended = [];
      for (const item of normalized) {
        if (app.logFingerprints.has(item.fingerprint)) continue;
        app.logFingerprints.add(item.fingerprint);
        app.logItems.push(item);
        appended.push(item);
      }
      let trimmed = false;
      if (app.logItems.length > MAX_LOG_LINES) {
        app.logItems = app.logItems.slice(-MAX_LOG_LINES);
        app.logFingerprints = new Set(app.logItems.map((item) => item.fingerprint));
        trimmed = true;
      }
      app.logCursor = firstDefined(payload?.next_cursor, payload?.cursor, app.logCursor);
      if (trimmed) renderLogs();
      else appendLogRows(appended);
    } catch (error) {
      if (error?.name === "AbortError") return;
      if (!(error instanceof ApiError) || error.status !== 404) {
        showToast(`日志读取失败：${errorMessage(error)}`, "error");
      }
    } finally {
      if (app.logRequestController === controller) app.logRequestController = null;
      app.logLoading = false;
    }
  }

  function refreshConfigSourcesForInteraction() {
    if (app.configRefreshPromise || app.configSwitching) return app.configRefreshPromise;
    app.configSelectionBeforeOpen = app.selectedConfig || elements.configSelect.value || app.configSelectionBeforeOpen;
    app.configRefreshPromise = Promise.allSettled([
      loadConfigs({ render: false }),
      loadHardwareDiscovery({ manual: false, preserve: true }),
    ]).finally(() => {
      app.configRefreshPromise = null;
      if (document.activeElement !== elements.configSelect) {
        if (app.configOptionsDeferred) {
          renderConfigOptions({ preserveValue: true });
          app.configOptionsDeferred = false;
        } else if (app.configSelectionDeferred) {
          syncConfigSelectToServer();
        }
      }
    });
    return app.configRefreshPromise;
  }

  function applyBindingResponseState(payload, {
    requestSerial,
    gripperEpoch,
    gripperRequestId,
  }) {
    const nextState = stateFromPayload(payload);
    if (!nextState || typeof nextState !== "object") return false;
    if (
      requestSerial < app.stateAppliedRequestSerial ||
      gripperEpoch !== app.gripperSyncEpoch
    ) return false;

    const incomingRevision = asNumber(nextState.revision);
    const currentRevision = asNumber(app.state.revision);
    if (
      incomingRevision !== null &&
      currentRevision !== null &&
      incomingRevision < currentRevision
    ) return false;

    const nextSync = stateSyncMetadata(nextState);
    if (Object.keys(nextSync).length) {
      return applyObservedState(nextState, nextState.devices, {
        requestSerial,
        gripperEpoch,
        gripperRequestId,
      });
    }

    const previousSync = stateSyncMetadata();
    const previousGrippers = app.state.grippers;
    const nextGrippers = nextState.grippers;
    app.state = { ...nextState };
    delete app.state.grippers;
    if (previousGrippers && typeof previousGrippers === "object") {
      app.state.grippers = previousGrippers;
    }
    if (Object.keys(previousSync).length) {
      app.state.state_sync = previousSync;
    }
    app.stateAppliedRequestSerial = Math.max(
      app.stateAppliedRequestSerial,
      requestSerial
    );
    applyGripperSnapshot(nextGrippers, {
      epoch: gripperEpoch,
      requestId: gripperRequestId,
      controllerRevision: nextState.revision,
    });
    syncCameraSlots(nextState, { clear: !selectedConfigFromPayload(nextState) });
    syncRobotSlots(nextState, { clear: !selectedConfigFromPayload(nextState) });
    app.devices = extractDevices(nextState.devices || []);
    return true;
  }

  function isHardwareDiscoveryRevisionConflict(error) {
    if (!(error instanceof ApiError) || error.status !== 409) return false;
    const payloadMessage = error.payload && typeof error.payload === "object"
      ? (error.payload.error || error.payload.message || error.payload.detail || "")
      : "";
    const code = error.payload && typeof error.payload === "object"
      ? String(error.payload.code || "").toLowerCase()
      : "";
    const message = String(payloadMessage || error.message || "").toLowerCase();
    return code === "hardware_discovery_changed"
      || message.includes("hardware discovery changed");
  }

  async function refreshHardwareDiscoveryForBindingSubmit() {
    if (app.bindingDiscoveryRefreshPromise) return app.bindingDiscoveryRefreshPromise;
    const configName = app.selectedConfig || elements.configSelect.value || "";
    const refresh = (async () => {
      await loadHardwareDiscovery({
        manual: false,
        preserve: true,
        configName,
        throwOnError: true,
        // The dropdown change already produced an atomic camera swap. Only
        // refresh the inventory/revision here; rebuilding the draft can race
        // that swap and submit the same physical camera for two logical slots.
        reconcileBindings: false,
      });
      const revision = app.discovery.revision;
      if (!revision) {
        throw new ApiError(
          "Hardware discovery changed during refresh; refresh discovery before applying bindings",
          409,
          { code: "hardware_discovery_changed" }
        );
      }
      return revision;
    })();
    app.bindingDiscoveryRefreshPromise = refresh;
    try {
      return await refresh;
    } finally {
      if (app.bindingDiscoveryRefreshPromise === refresh) {
        app.bindingDiscoveryRefreshPromise = null;
      }
    }
  }

  async function postBindingChangeWithFreshDiscovery(bindingOverrides) {
    let lastError = null;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        const discoveryRevision = await refreshHardwareDiscoveryForBindingSubmit();
        return await api("/api/bindings", {
          method: "POST",
          body: JSON.stringify({
            name: app.selectedConfig,
            discovery_revision: discoveryRevision,
            // A camera swap updates both logical slots atomically. Sending
            // only the clicked slot leaves the other slot's config default
            // active and the backend correctly rejects a hidden duplicate.
            overrides: bindingOverrides,
          }),
        });
      } catch (error) {
        lastError = error;
        if (attempt === 0 && isHardwareDiscoveryRevisionConflict(error)) continue;
        throw error;
      }
    }
    throw lastError;
  }

  async function submitBindingChange(deviceKey, { notify = true, affectedKeys = [deviceKey] } = {}) {
    const requestKeys = completeCameraBindingRequestKeys(deviceKey, affectedKeys);
    const existing = requestKeys
      .map((key) => app.bindingRequests.get(key))
      .find(Boolean);
    if (existing) return existing;
    // Freeze the user's binding intent before any asynchronous discovery or
    // state refresh. Camera swaps must remain one atomic, immutable payload.
    const bindingOverrides = buildBindingUpdateOverrides(requestKeys);
    const requestSerial = ++app.stateRequestSerial;
    const gripperEpoch = app.gripperSyncEpoch;
    const gripperRequestId = nextGripperSnapshotRequest();
    const request = (async () => {
      requestKeys.forEach((key) => app.bindingPending.add(key));
      renderBindingControls();
      try {
        const payload = await postBindingChangeWithFreshDiscovery(bindingOverrides);
        const applied = applyBindingResponseState(payload, {
          requestSerial,
          gripperEpoch,
          gripperRequestId,
        });
        if (!applied) scheduleStatePoll(0);
        requestKeys.forEach((key) => {
          app.bindingDirtyKeys.delete(key);
          app.bindingErrors.delete(key);
        });
        app.bindingDirty = app.bindingDirtyKeys.size > 0;
        if (!app.bindingErrors.size) setGlobalNotice("");
        if (notify) showToast(`${deviceKey} 绑定已更新`, "success");
        return true;
      } catch (error) {
        const message = errorMessage(error);
        requestKeys.forEach((key) => {
          app.bindingErrors.set(key, message);
          app.bindingDirtyKeys.add(key);
        });
        app.bindingDirty = true;
        setGlobalNotice(`绑定未生效：${message}`);
        showToast(`绑定未生效：${message}`, "error");
        return false;
      } finally {
        requestKeys.forEach((key) => app.bindingPending.delete(key));
        renderBindingControls();
        renderMetrics();
        renderWorkflow();
        renderGrippers();
      }
    })();
    requestKeys.forEach((key) => app.bindingRequests.set(key, request));
    try {
      return await request;
    } finally {
      requestKeys.forEach((key) => {
        if (app.bindingRequests.get(key) === request) app.bindingRequests.delete(key);
      });
    }
  }

  async function runToggleButton(button) {
    // Keep the original click context stable while discovery and binding
    // application re-render the controls underneath this async handler.
    const deviceKey = String(button?.dataset?.deviceKey || "");
    const action = button?.dataset?.action;
    const selectedConfig = app.selectedConfig;
    let device = app.devices.find((item) => item.key === deviceKey);
    if (!device || !action) return;
    if (
      action === "start" &&
      (
        app.bindingDirtyKeys.has(device.key) ||
        app.bindingRequests.has(device.key) ||
        !bindingSelectionMatchesEffective(device)
      )
    ) {
      const applied = await submitBindingChange(device.key, { notify: false });
      if (!applied) return;
      // The backend start endpoint receives only the logical class/id and
      // resolves the just-applied binding itself. A second state refresh here
      // can observe the short config-rebuild window and incorrectly cancel the
      // original click before the confirmation dialog is shown.
      if (app.selectedConfig !== selectedConfig || !controlsMatchActiveConfig()) {
        return;
      }
      device = app.devices.find((item) => item.key === deviceKey) || device;
    }
    return runDeviceAction(device, action);
  }

  function stopStateEvents() {
    const source = app.stateEvents;
    app.stateEvents = null;
    app.stateEventsConnected = false;
    if (source && typeof source.close === "function") source.close();
  }

  function startStateEvents() {
    if (document.hidden || typeof globalThis.EventSource !== "function" || app.stateEvents) return;
    const source = new globalThis.EventSource("/api/events");
    app.stateEvents = source;
    const handleSnapshot = (event) => {
      if (app.stateEvents !== source || document.hidden) return;
      try {
        const payload = JSON.parse(String(event.data || "{}"));
        const nextState = stateFromPayload(payload);
        applyObservedState(nextState, nextState.devices, {
          requestSerial: ++app.stateRequestSerial,
        });
        app.stateEventsConnected = true;
        scheduleStatePoll();
      } catch (error) {
        markStateTransportStale(`状态事件解析失败：${errorMessage(error)}`);
      }
    };
    source.addEventListener("state", handleSnapshot);
    source.addEventListener("heartbeat", handleSnapshot);
    source.onopen = () => {
      if (app.stateEvents === source) {
        app.stateEventsConnected = true;
        scheduleStatePoll();
      }
    };
    source.onerror = () => {
      if (app.stateEvents !== source || document.hidden) return;
      app.stateEventsConnected = false;
      markStateTransportStale("实时状态事件流已中断");
      scheduleStatePoll(0);
    };
  }

  function clearPollTimer(name) {
    const timer = app.pollTimers[name];
    if (timer !== null) window.clearTimeout(timer);
    app.pollTimers[name] = null;
  }

  function statePollDelay() {
    const workflow = app.state.workflow || {};
    const recording = normalizeRecordingState();
    const serverTransitions = stateSyncMetadata().transitions;
    const serverPending = Boolean(
      serverTransitions &&
      typeof serverTransitions === "object" &&
      Object.values(serverTransitions).some(isActiveObservedTransition)
    );
    const pending = Boolean(
      serverPending ||
      app.pendingActions.size ||
      app.bindingPending.size ||
      app.workflowPending.size ||
      app.forceStopPending ||
      app.gripperCalibratePending ||
      app.gripperMovePending.size ||
      app.gripperTargetSessions.size ||
      app.gripperPositionSync.size ||
      app.gripperProbePending.size ||
      app.gripperProbeSync.size ||
      app.recordingActionPending ||
      recording.pending_command ||
      app.devices.some((device) => TRANSITION_STATUSES.has(device.status)) ||
      [workflow.runtime?.status, workflow.server?.status, workflow.inference?.status].some((status) => (
        ["starting", "stopping"].includes(String(status || "").toLowerCase())
      )) ||
      [normalizeGrippers().status, ...GRIPPER_SIDES.map((side) => normalizeGrippers().sides[side].status)]
        .some((status) => ["calibrating", "moving", "queued", "probing"].includes(status))
    );
    if (pending) return POLL_INTERVALS.statePending;
    const recovering = Boolean(
      app.state.safety?.fail_closed ||
      app.devices.some((device) => ERROR_STATUSES.has(device.status)) ||
      [workflow.runtime?.status, workflow.server?.status, workflow.inference?.status]
        .some((status) => ["failed", "error"].includes(String(status || "").toLowerCase()))
    );
    if (recovering) {
      return app.stateStablePolls < 8 ? POLL_INTERVALS.stateRecovery : POLL_INTERVALS.stateActive;
    }
    if (app.stateEventsConnected && !app.stateTransportFailed) {
      return POLL_INTERVALS.stateWatchdog;
    }
    const active = Boolean(
      recording.active ||
      workflow.runtime?.active ||
      workflow.server?.active ||
      workflow.inference?.running ||
      [normalizeGrippers().status, ...GRIPPER_SIDES.map((side) => normalizeGrippers().sides[side].status)]
        .some((status) => ["calibrating", "moving"].includes(status)) ||
      app.devices.some((device) => ACTIVE_STATUSES.has(device.status))
    );
    if (active) {
      return app.stateStablePolls < 3 ? POLL_INTERVALS.stateActiveFast : POLL_INTERVALS.stateActive;
    }
    return app.stateStablePolls < 2 ? POLL_INTERVALS.stateIdleFast : POLL_INTERVALS.stateIdle;
  }

  function scheduleStatePoll(delay = statePollDelay()) {
    clearPollTimer("state");
    if (document.hidden) return;
    app.pollTimers.state = window.setTimeout(async () => {
      app.pollTimers.state = null;
      await refreshState(false);
      scheduleStatePoll();
    }, delay);
  }

  function visibleRunningCameras() {
    return app.devices.filter((device) => (
      device.isCamera &&
      (device.status === "running" || device.status === "degraded") &&
      cameraCardCanStream(device, cameraCardForDevice(device))
    ));
  }

  function ensurePreviewPolling(immediate = false) {
    if (document.hidden || !visibleRunningCameras().length) {
      clearPollTimer("preview");
      return;
    }
    if (app.pollTimers.preview !== null && !immediate) return;
    clearPollTimer("preview");
    app.pollTimers.preview = window.setTimeout(async () => {
      app.pollTimers.preview = null;
      await refreshPreviewStatuses();
      ensurePreviewPolling();
    }, immediate ? 0 : POLL_INTERVALS.preview);
  }

  function ensureLogPolling(immediate = false) {
    if (document.hidden || !elements.logDetails?.open) {
      clearPollTimer("logs");
      return;
    }
    if (app.pollTimers.logs !== null && !immediate) return;
    clearPollTimer("logs");
    app.pollTimers.logs = window.setTimeout(async () => {
      app.pollTimers.logs = null;
      await loadLogs();
      ensureLogPolling();
    }, immediate ? 0 : POLL_INTERVALS.logs);
  }

  function ensureRecordingClock() {
    const recording = normalizeRecordingState();
    if (document.hidden || !recording.active || !recording.episode) {
      clearPollTimer("recordingClock");
      return;
    }
    if (app.pollTimers.recordingClock !== null) return;
    app.pollTimers.recordingClock = window.setTimeout(() => {
      app.pollTimers.recordingClock = null;
      renderRecordingClock();
      ensureRecordingClock();
    }, POLL_INTERVALS.recordingClock);
  }

  function suspendPolling() {
    for (const name of Object.keys(app.pollTimers)) clearPollTimer(name);
    stopStateEvents();
    app.stateTransportFailed = true;
    app.stateTransportReason = "页面隐藏期间未持续观测设备状态";
    app.stateTransportDeadlineAtMs = 0;
    app.stateRequestController?.abort();
    for (const deviceKey of [...app.previewStatusRequests.keys()]) {
      cancelPreviewStatusRequest(deviceKey);
    }
    app.logRequestController?.abort();
    app.preflightController?.abort();
    app.runtimeMetricsController?.abort();
    app.checkpointMutationController?.abort();
    invalidateCheckpointBrowser();
    for (const card of app.cameraCards.values()) stopCameraPreview(card);
  }

  function resumePolling() {
    markStateTransportStale("页面已恢复，正在重新确认真实设备状态");
    startStateEvents();
    const resumeState = () => scheduleStatePoll(0);
    if (app.stateRefreshPromise) app.stateRefreshPromise.finally(resumeState);
    else resumeState();
    renderCameras();
    ensurePreviewPolling(true);
    ensureLogPolling(true);
    ensureDiagnosticsPolling(true);
    renderRecordingClock();
    ensureRecordingClock();
  }

  function workspaceSplitBounds() {
    const layout = elements.mainContent;
    if (!layout || typeof layout.getBoundingClientRect !== "function") return null;
    const rect = layout.getBoundingClientRect();
    const styles = typeof globalThis.getComputedStyle === "function"
      ? globalThis.getComputedStyle(layout)
      : null;
    const paddingLeft = Number.parseFloat(styles?.paddingLeft || "0") || 0;
    const paddingRight = Number.parseFloat(styles?.paddingRight || "0") || 0;
    const splitterWidth = elements.workspaceSplitter?.getBoundingClientRect?.().width || 10;
    const available = Math.max(0, rect.width - paddingLeft - paddingRight - splitterWidth);
    const minimum = Math.min(WORKSPACE_SPLIT_MIN_PX, available);
    const maximum = Math.max(
      minimum,
      Math.min(620, available - WORKSPACE_VISUAL_MIN_PX)
    );
    return { rect, paddingLeft, minimum, maximum, available };
  }

  function defaultWorkspaceSplitWidth(bounds = workspaceSplitBounds()) {
    if (!bounds) return WORKSPACE_SPLIT_MIN_PX;
    return Math.min(
      bounds.maximum,
      Math.max(bounds.minimum, Math.round(bounds.available * 0.2))
    );
  }

  function applyWorkspaceSplitWidth(rawWidth, { persist = false } = {}) {
    const bounds = workspaceSplitBounds();
    if (!bounds || !elements.mainContent || !elements.workspaceSplitter) return null;
    const numeric = Number(rawWidth);
    const requested = Number.isFinite(numeric) ? numeric : defaultWorkspaceSplitWidth(bounds);
    const width = Math.round(Math.min(bounds.maximum, Math.max(bounds.minimum, requested)));
    app.workspaceSplitWidth = width;
    elements.mainContent.style.setProperty("--control-rail-width", `${width}px`);
    elements.workspaceSplitter.setAttribute("aria-valuemin", String(Math.round(bounds.minimum)));
    elements.workspaceSplitter.setAttribute("aria-valuemax", String(Math.round(bounds.maximum)));
    elements.workspaceSplitter.setAttribute("aria-valuenow", String(width));
    if (persist) {
      try {
        globalThis.localStorage?.setItem(WORKSPACE_SPLIT_STORAGE_KEY, String(width));
      } catch (_error) {
        // Layout persistence is best effort and never affects device control.
      }
    }
    return width;
  }

  function restoreWorkspaceSplitWidth() {
    let stored = null;
    try {
      stored = globalThis.localStorage?.getItem(WORKSPACE_SPLIT_STORAGE_KEY) || null;
    } catch (_error) {
      stored = null;
    }
    applyWorkspaceSplitWidth(stored === null ? undefined : stored);
  }

  function beginWorkspaceSplitDrag(event) {
    if (event.button !== undefined && event.button !== 0) return;
    app.workspaceSplitPointerId = event.pointerId;
    elements.workspaceSplitter?.setPointerCapture?.(event.pointerId);
    if (elements.workspaceSplitter) elements.workspaceSplitter.dataset.dragging = "true";
    if (document.documentElement) document.documentElement.dataset.workspaceResizing = "true";
    event.preventDefault();
  }

  function moveWorkspaceSplitDrag(event) {
    if (app.workspaceSplitPointerId === null || event.pointerId !== app.workspaceSplitPointerId) return;
    const bounds = workspaceSplitBounds();
    if (!bounds) return;
    applyWorkspaceSplitWidth(event.clientX - bounds.rect.left - bounds.paddingLeft);
  }

  function endWorkspaceSplitDrag(event) {
    if (app.workspaceSplitPointerId === null || event.pointerId !== app.workspaceSplitPointerId) return;
    elements.workspaceSplitter?.releasePointerCapture?.(event.pointerId);
    app.workspaceSplitPointerId = null;
    if (elements.workspaceSplitter) delete elements.workspaceSplitter.dataset.dragging;
    if (document.documentElement) delete document.documentElement.dataset.workspaceResizing;
    applyWorkspaceSplitWidth(app.workspaceSplitWidth, { persist: true });
  }

  function handleWorkspaceSplitKey(event) {
    const bounds = workspaceSplitBounds();
    if (!bounds) return;
    let width = app.workspaceSplitWidth ?? defaultWorkspaceSplitWidth(bounds);
    if (event.key === "ArrowLeft") width -= event.shiftKey ? 48 : 16;
    else if (event.key === "ArrowRight") width += event.shiftKey ? 48 : 16;
    else if (event.key === "Home") width = bounds.minimum;
    else if (event.key === "End") width = bounds.maximum;
    else return;
    event.preventDefault();
    applyWorkspaceSplitWidth(width, { persist: true });
  }

  function bindEvents() {
    restoreWorkspaceSplitWidth();
    elements.workspaceSplitter?.addEventListener("pointerdown", beginWorkspaceSplitDrag);
    elements.workspaceSplitter?.addEventListener("pointermove", moveWorkspaceSplitDrag);
    elements.workspaceSplitter?.addEventListener("pointerup", endWorkspaceSplitDrag);
    elements.workspaceSplitter?.addEventListener("pointercancel", endWorkspaceSplitDrag);
    elements.workspaceSplitter?.addEventListener("keydown", handleWorkspaceSplitKey);
    elements.workspaceSplitter?.addEventListener("dblclick", () => {
      applyWorkspaceSplitWidth(defaultWorkspaceSplitWidth(), { persist: true });
    });
    globalThis.addEventListener?.("resize", () => {
      if (app.workspaceSplitWidth !== null) applyWorkspaceSplitWidth(app.workspaceSplitWidth);
    });
    elements.configSelect.addEventListener("pointerdown", refreshConfigSourcesForInteraction);
    elements.configSelect.addEventListener("focus", refreshConfigSourcesForInteraction);
    elements.configSelect.addEventListener("blur", () => {
      window.setTimeout(() => {
        if (!app.configSwitching) {
          if (app.configOptionsDeferred) {
            renderConfigOptions({ preserveValue: true });
            app.configOptionsDeferred = false;
          } else if (app.configSelectionDeferred) {
            syncConfigSelectToServer();
          }
        }
      }, 0);
    });
    elements.configSelect.addEventListener("change", () => {
      const name = elements.configSelect.value;
      const previousValue = app.selectedConfig || app.configSelectionBeforeOpen || "";
      app.bootstrapManualChoiceRequired = false;
      app.bindingDirty = false;
      app.bindingDirtyKeys.clear();
      app.bindingErrors.clear();
      app.gripperPortDrafts.clear();
      app.gripperErrors.clear();
      app.configDevices = [];
      syncRobotSlots({}, { clear: true });
      selectConfigWithConfirmation(name, previousValue);
    });

    elements.checkpointBrowseButton?.addEventListener("click", openCheckpointBrowser);
    elements.checkpointClearButton?.addEventListener("click", () => {
      void mutateCheckpointOverride(null);
    });
    elements.testInferenceToggle?.addEventListener("click", () => {
      const options = normalizeInferenceOptions();
      void mutateInferenceOptions("test", !options.testMode);
    });
    elements.rolloutToggle?.addEventListener("click", () => {
      const options = normalizeInferenceOptions();
      if (options.rolloutEnabled) {
        void mutateInferenceOptions("rollout", false);
      } else {
        openRolloutBrowser({ enableOnSelect: true });
      }
    });
    elements.rolloutPathButton?.addEventListener("click", () => {
      openRolloutBrowser({ enableOnSelect: false });
    });
    elements.checkpointBrowserForm?.addEventListener("submit", (event) => {
      event.preventDefault();
      const payload = app.checkpointBrowserPayload;
      if (
        !payload
        || payload.current_selectable !== true
        || !payload.path
        || checkpointBrowserInputPath() !== String(payload.path)
      ) return;
      if (app.checkpointBrowserMode === "rollout") {
        const options = normalizeInferenceOptions();
        void mutateInferenceOptions(
          "rollout",
          options.rolloutEnabled || app.rolloutBrowserEnableOnSelect,
          String(payload.path)
        );
      } else {
        void mutateCheckpointOverride(String(payload.path));
      }
    });
    elements.checkpointBrowserRefresh?.addEventListener("click", () => {
      navigateCheckpointBrowserFromInput();
    });
    elements.checkpointBrowserUp?.addEventListener("click", () => {
      const path = String(elements.checkpointBrowserUp?.dataset?.path || "");
      if (path) void loadCheckpointBrowser(path);
    });
    elements.checkpointBrowserGo?.addEventListener("click", () => {
      navigateCheckpointBrowserFromInput();
    });
    elements.checkpointBrowserPath?.addEventListener("input", () => {
      app.checkpointBrowserRequest += 1;
      app.checkpointBrowserController?.abort();
      app.checkpointBrowserController = null;
      if (elements.checkpointBrowserList) {
        elements.checkpointBrowserList.setAttribute("aria-busy", "false");
      }
      checkpointBrowserSetError("");
      if (!syncCheckpointBrowserDraftState()) {
        setText(
          elements.checkpointBrowserStatus,
          "路径尚未加载，按 Enter 或点击“打开”进行验证。"
        );
      }
    });
    elements.checkpointBrowserPath?.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.isComposing) return;
      event.preventDefault();
      event.stopPropagation();
      navigateCheckpointBrowserFromInput();
    });
    elements.checkpointBrowserCancel?.addEventListener("click", () => {
      invalidateCheckpointBrowser();
    });
    elements.checkpointBrowserDialog?.addEventListener("cancel", (event) => {
      event.preventDefault();
      invalidateCheckpointBrowser();
    });
    elements.checkpointBrowserDialog?.addEventListener("close", () => {
      invalidateCheckpointBrowser({ close: false });
    });

    elements.robotLeftBindingSelect?.addEventListener("change", () => {
      const selected = availableRobotBinding(elements.robotLeftBindingSelect.value, "left")?.id || "";
      app.robotBindings.left = selected;
      if (selected && app.robotBindings.right === selected) app.robotBindings.right = "";
      const deviceKey = bindingDevices().find(isRizonDevice)?.key || "";
      if (!deviceKey) return;
      app.bindingDirty = true;
      app.bindingDirtyKeys.add(deviceKey);
      app.bindingErrors.delete(deviceKey);
      renderBindingControls();
      void submitBindingChange(deviceKey);
    });
    elements.robotRightBindingSelect?.addEventListener("change", () => {
      const selected = availableRobotBinding(elements.robotRightBindingSelect.value, "right")?.id || "";
      app.robotBindings.right = selected;
      if (selected && app.robotBindings.left === selected) app.robotBindings.left = "";
      const deviceKey = bindingDevices().find(isRizonDevice)?.key || "";
      if (!deviceKey) return;
      app.bindingDirty = true;
      app.bindingDirtyKeys.add(deviceKey);
      app.bindingErrors.delete(deviceKey);
      renderBindingControls();
      void submitBindingChange(deviceKey);
    });
    elements.robotDeviceToggle?.addEventListener("click", () => runToggleButton(elements.robotDeviceToggle));
    elements.i2rtHeadDeviceToggle?.addEventListener("click", () => runToggleButton(elements.i2rtHeadDeviceToggle));
    elements.hardwareDiscoveryRefresh?.addEventListener(
      "click",
      refreshHardwareDiscoveryForBindings
    );
    elements.gripperCalibrateAll?.addEventListener("click", () => void runGripperCalibration());
    elements.gripperAuthorizePermissions?.addEventListener("click", () => {
      app.gripperPermissionDismissedSignature = "";
      void requestGripperPermission({ force: true });
    });
    elements.gripperPermissionForm?.addEventListener("submit", (event) => {
      event.preventDefault();
      void submitGripperPermissionRepair();
    });
    elements.gripperPermissionCancel?.addEventListener("click", cancelGripperPermissionPrompt);
    elements.gripperPermissionDialog?.addEventListener("cancel", (event) => {
      event.preventDefault();
      cancelGripperPermissionPrompt();
    });
    for (const side of GRIPPER_SIDES) {
      const portSelect = elements.grippers[side]?.port;
      portSelect?.addEventListener("change", () => {
        const port = String(portSelect.value || "");
        void submitGripperPortBinding(side, port);
      });
      const slider = elements.grippers[side]?.position;
      if (!slider) continue;
      slider.addEventListener("pointerdown", () => app.gripperEditing.add(side));
      slider.addEventListener("pointerup", () => {
        window.setTimeout(() => {
          app.gripperEditing.delete(side);
          const target = asNumber(app.gripperDesiredTargets.get(side));
          if (target !== null) void runGripperMove(side, target);
          renderGrippers();
        }, 0);
      });
      slider.addEventListener("keydown", (event) => {
        if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"].includes(event.key)) {
          app.gripperEditing.add(side);
        }
      });
      slider.addEventListener("input", () => {
        // Keep the latest user target authoritative while dragging.  Commands
        // are rate-limited to the backend's 5 Hz coalescing lane, and the
        // slider remains editable while either gripper is moving.
        app.gripperEditing.add(side);
        const numericValue = asNumber(slider.value);
        if (numericValue === null) return;
        const latched = latchGripperTarget(side, numericValue);
        if (!latched) return;
        scheduleGripperTargetSubmission(side, 0);
        renderGrippers();
      });
      slider.addEventListener("change", () => {
        const numericValue = asNumber(slider.value);
        if (numericValue === null) return;
        const target = Math.min(100, Math.max(0, Math.round(numericValue)));
        discardGripperCalibrationSliderValue(side);
        app.gripperEditing.delete(side);
        app.gripperDrafts.set(side, target);
        void runGripperMove(side, target);
      });
      slider.addEventListener("pointercancel", () => {
        app.gripperEditing.delete(side);
        const target = asNumber(app.gripperDesiredTargets.get(side));
        if (target !== null) scheduleGripperTargetSubmission(side, 0);
        renderGrippers();
      });
      slider.addEventListener("blur", () => {
        app.gripperEditing.delete(side);
        const target = asNumber(app.gripperDesiredTargets.get(side));
        if (target !== null) scheduleGripperTargetSubmission(side, 0);
        renderGrippers();
      });
    }
    elements.runtimeToggle?.addEventListener("click", () => runWorkflowAction(elements.runtimeToggle));
    elements.serverToggle?.addEventListener("click", () => runWorkflowAction(elements.serverToggle));
    elements.inferenceToggle?.addEventListener("click", () => runWorkflowAction(elements.inferenceToggle));
    elements.preflightRefresh?.addEventListener("click", () => void loadPreflight({ manual: true }));
    elements.runtimeMetricsRefresh?.addEventListener("click", () => {
      void loadRuntimeDiagnostics({ includeTuning: true });
      void loadPreflight({ manual: true });
    });
    elements.generatedConfigExport?.addEventListener("click", () => {
      void exportGeneratedConfig();
    });
    elements.currentConfigSave?.addEventListener("click", () => {
      void saveCurrentConfig();
    });
    elements.forceStopAll?.addEventListener("click", () => void runForceStopAll());
    for (const button of elements.recordingActionButtons) {
      button.addEventListener("click", () => runRecordingAction(button));
    }
    elements.standaloneResetButton?.addEventListener(
      "click",
      () => void runStandaloneReset(),
    );
    elements.standalonePauseButton?.addEventListener(
      "click",
      () => void runStandalonePauseResume(),
    );
    document.addEventListener("keydown", handleRecordingShortcut);

    elements.cameraGrid.addEventListener("change", (event) => {
      const select = event.target.closest('[data-role="camera-binding-select"][data-device-key]');
      if (!select) return;
      const deviceKey = select.dataset.deviceKey;
      const affectedKeys = applyCameraBindingSelection(deviceKey, select.value);
      if (!affectedKeys.length) return;
      app.bindingDirty = true;
      affectedKeys.forEach((key) => {
        app.bindingDirtyKeys.add(key);
        app.bindingErrors.delete(key);
      });
      renderBindingControls();
      void submitBindingChange(deviceKey, { affectedKeys });
    });
    elements.cameraGrid.addEventListener("pointerdown", (event) => {
      const select = event.target.closest('[data-role="camera-binding-select"][data-device-key]');
      if (!select || app.discovery.cameras.length || bindingTargetLocked(
        app.devices.find((device) => device.key === select.dataset.deviceKey)
      )) return;
      refreshHardwareDiscoveryForBindings();
    });
    elements.cameraGrid.addEventListener("click", (event) => {
      const button = event.target.closest('[data-role="camera-device-toggle"][data-device-key]');
      if (button) runToggleButton(button);
    });

    elements.logDeviceSelect.addEventListener("change", () => {
      app.logKey = elements.logDeviceSelect.value;
      resetLogs();
      ensureLogPolling(true);
    });
    elements.clearLogsButton.addEventListener("click", resetLogs);
    elements.logDetails?.addEventListener("toggle", () => {
      const hint = elements.logDetails.querySelector(".summary-hint");
      if (hint) hint.textContent = elements.logDetails.open ? "收起" : "展开";
      if (elements.logDetails.open) ensureLogPolling(true);
      else {
        clearPollTimer("logs");
        app.logRequestController?.abort();
      }
    });

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) suspendPolling();
      else resumePolling();
    });
    globalThis.addEventListener?.("blur", () => {
      for (const card of app.cameraCards.values()) {
        stopCameraPreview(card);
      }
    });
    globalThis.addEventListener?.("focus", () => {
      renderCameras();
      ensurePreviewPolling(true);
    });
  }

  async function initialize() {
    bindEvents();
    setupCameraVisibilityObserver();
    setServerStatus("pending", "正在连接");
    const [configsResult] = await Promise.allSettled([
      loadConfigs(),
      refreshState(false),
      loadHardwareDiscovery({ preserve: false }),
    ]);
    if (configsResult.status === "rejected") {
      setGlobalNotice(errorMessage(configsResult.reason));
    }
    syncConfigSelectToServer();
    const bootstrapResult = await maybeAutoApplyDefaultConfig();
    if (bootstrapResult.status !== "applied") {
      const selected = elements.configSelect.value;
      app.configSelectionBeforeOpen = selected;
      if (selected) loadConfigDetail(selected);
    }
    if (app.selectedConfig) {
      requestGripperActualAlignment(GRIPPER_SIDES, "startup");
    }

    scheduleStatePoll();
    startStateEvents();
    ensurePreviewPolling();
    ensureLogPolling();
    ensureDiagnosticsPolling(true);
    ensureRecordingClock();
  }

  initialize();
})();
