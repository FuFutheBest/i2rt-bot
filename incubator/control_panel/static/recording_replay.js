const RECORDINGS_URL = "/api/recordings";
const RECORDING_EXPORTS_URL = "/api/recording-exports";
const JOINT_DIMENSION = 7;
const TCP_DIMENSION = 7;
const GRIPPER_DIMENSION = 1;
const LIST_PAGE_SIZE = 50;
const MAX_SELECTED_EPISODES = 1000;
const TRAJECTORY_WINDOW_FRAMES = 1800;
const TRAJECTORY_MAX_POINTS = 1800;
const TRAJECTORY_CACHE_ENTRIES = 3;
const TCP_OVERVIEW_MAX_POINTS = 900;
const TCP_TRAIL_FRACTION = 0.1;
const MAX_PLAYBACK_FPS = 30;
const DEFAULT_PLAYBACK_FPS = 30;
const SCRUB_DEBOUNCE_MS = 100;
const FILTER_DEBOUNCE_MS = 250;
const EXPORT_POLL_MS = 1500;
const DENSE_DOM_NODE_LIMIT = 1000;
const CACHE_POLL_DELAYS_MS = Object.freeze([250, 500, 1000, 2000]);
const SPEEDS = new Set([0.25, 0.5, 1, 2]);
const CAMERA_ROLES = ["head", "left", "right"];
const CAMERA_ROLE_LABELS = { head: "Head", left: "Left", right: "Right" };
const CAMERA_ROLE_STATUS_LABELS = {
  healthy: "正常",
  configured: "已配置",
  not_configured: "未配置",
  missing: "缺失",
  black_placeholder: "黑帧占位",
  unknown: "未上报",
};

function isObject(value) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function safeString(value, fallback = "", limit = 240) {
  return typeof value === "string" && value.trim()
    ? value.trim().slice(0, limit)
    : fallback;
}

function nonnegativeInteger(value, fallback = 0) {
  const number = Number(value);
  return Number.isInteger(number) && number >= 0 ? number : fallback;
}

function finiteNumber(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function normalizeResult(value) {
  const result = safeString(value, "unknown", 48).toLowerCase();
  if (["success", "accepted", "accept"].includes(result)) return "success";
  if (["failure", "failed", "rejected", "reject", "discarded"].includes(result)) return "failure";
  return result;
}

function statusTone(status, result) {
  if (result === "success" || ["ready", "complete", "completed", "finalized"].includes(status)) return "success";
  if (result === "failure" || ["error", "failed", "incomplete", "corrupt"].includes(status)) return "danger";
  if (["recording", "finalizing", "pending", "unknown"].includes(status)) return "warning";
  return "neutral";
}

function normalizeRecordingItem(value) {
  if (!isObject(value)) return null;
  const id = safeString(value.id, "", 256);
  if (!id) return null;
  const episodeIndex = finiteNumber(value.episode_index);
  return Object.freeze({
    id,
    name: safeString(value.name, id, 256),
    episodeName: safeString(value.episode_name, id.split(":").at(-1), 160),
    date: safeString(value.date, "", 16) || "legacy",
    status: safeString(value.status, "unknown", 48).toLowerCase(),
    result: normalizeResult(value.result),
    episodeIndex: Number.isInteger(episodeIndex) ? episodeIndex : null,
    episodeId: safeString(value.episode_id, "", 160) || null,
    taskName: safeString(value.task_name, "", 300) || null,
    updatedAt: safeString(value.updated_at, "", 80) || null,
    hasMetadata: value.has_metadata === true,
    hasDenseLabels: value.has_dense_labels === true,
    hasConversionCheck: value.has_conversion_check === true,
  });
}

function normalizeRecordingList(payload) {
  if (!isObject(payload)) throw new Error("录制列表响应格式无效");
  const items = Array.isArray(payload.items)
    ? payload.items.map(normalizeRecordingItem).filter(Boolean)
    : [];
  const rootConfigured = payload.root_configured === true;
  const rootState = safeString(payload.root_state, "", 32).toLowerCase();
  const missingButConfigured = rootState === "missing" && rootConfigured;
  const stateUnavailable = ["unreadable", "unconfigured"].includes(rootState);
  const rootAvailable = missingButConfigured || (!stateUnavailable && (
    payload.root_available === true
    || (payload.root_available == null && rootConfigured)
  ));
  const dates = Array.isArray(payload.dates)
    ? payload.dates.map((value) => {
      if (!isObject(value)) return null;
      const id = safeString(value.id, "", 16);
      if (!id || (id !== "legacy" && !/^\d{8}$/.test(id))) return null;
      return Object.freeze({
        id,
        label: safeString(value.label, id, 40),
        count: nonnegativeInteger(value.count, 0),
        accepted: nonnegativeInteger(value.accepted, 0),
        rejected: nonnegativeInteger(value.rejected, 0),
      });
    }).filter(Boolean)
    : [];
  const filters = isObject(payload.filters) ? payload.filters : {};
  return Object.freeze({
    items: Object.freeze(items),
    page: Math.max(1, nonnegativeInteger(payload.page, 1)),
    pageSize: Math.max(1, nonnegativeInteger(payload.page_size, LIST_PAGE_SIZE)),
    total: nonnegativeInteger(payload.total, items.length),
    totalPages: Math.max(0, nonnegativeInteger(payload.total_pages, items.length ? 1 : 0)),
    rootConfigured,
    rootAvailable,
    rootState,
    dates: Object.freeze(dates),
    filters: Object.freeze({
      date: safeString(filters.date, "all", 16),
      status: safeString(filters.status, "all", 32),
      query: safeString(filters.query, "", 120),
    }),
    cachePersistent: payload.cache?.persistent === true,
  });
}

function normalizeRecordingExportJob(value) {
  if (!isObject(value)) return null;
  const jobId = safeString(value.job_id, "", 64);
  const state = safeString(value.state, "unknown", 32).toLowerCase();
  if (!jobId || !["queued", "waiting", "running", "ready", "error"].includes(state)) {
    return null;
  }
  const episodeCount = nonnegativeInteger(value.episode_count, 0);
  return Object.freeze({
    jobId,
    state,
    phase: safeString(value.phase, "unknown", 64).toLowerCase(),
    episodeCount,
    conversionReadyCount: Math.min(
      episodeCount,
      nonnegativeInteger(value.conversion_ready_count, 0),
    ),
    conversionStates: Object.freeze(
      isObject(value.conversion_states) ? { ...value.conversion_states } : {},
    ),
    datasetName: safeString(value.dataset_name, "", 128) || null,
    datasetRoot: safeString(value.dataset_root, "", 1000) || null,
    videoMaterialization: Object.freeze(
      isObject(value.video_materialization) ? { ...value.video_materialization } : {},
    ),
    validation: Object.freeze(isObject(value.validation) ? { ...value.validation } : {}),
    error: safeString(value.error, "", 1600) || null,
    createdAt: safeString(value.created_at, "", 80) || null,
    updatedAt: safeString(value.updated_at, "", 80) || null,
    finishedAt: safeString(value.finished_at, "", 80) || null,
  });
}

function normalizeRecordingExportState(payload) {
  if (!isObject(payload)) throw new Error("录制导出状态响应格式无效");
  const jobs = Array.isArray(payload.jobs)
    ? payload.jobs.map(normalizeRecordingExportJob).filter(Boolean)
    : [];
  return Object.freeze({
    available: payload.available === true,
    blockers: Object.freeze(
      Array.isArray(payload.blockers)
        ? payload.blockers.map((value) => safeString(value, "", 500)).filter(Boolean)
        : [],
    ),
    jobs: Object.freeze(jobs),
  });
}

function recordingExportPhaseLabel(job) {
  if (!job) return "待提交";
  const labels = {
    checking_conversion: "检查单 episode 转换",
    waiting_conversion: "等待单 episode 转换",
    waiting_idle: "等待设备与推理空闲",
    ready_to_merge: "等待合并",
    merging: "正在合并",
    published: "导出完成",
    conversion_blocked: "转换未就绪",
    validation_failed: "选择校验失败",
    failed: "导出失败",
    recovered: "恢复导出任务",
  };
  return labels[job.phase]
    || (job.state === "ready" ? "导出完成" : job.state === "error" ? "导出失败" : "处理中");
}

function recordingItemExportEligible(item) {
  return item?.result === "success";
}

async function collectSuccessfulRecordingItems(
  fetchJson,
  { date = "all", query = "", signal = undefined, pageSize = 100 } = {},
) {
  const accepted = new Map();
  let page = 1;
  let totalPages = 1;
  do {
    const parameters = new URLSearchParams({
      page: String(page),
      page_size: String(pageSize),
      status: "all",
      result: "success",
      date,
      query,
    });
    const list = normalizeRecordingList(await fetchJson(
      `${RECORDINGS_URL}?${parameters}`,
      { signal },
    ));
    for (const item of list.items) {
      if (item.status !== "recording" && recordingItemExportEligible(item)) {
        accepted.set(item.id, item);
        if (accepted.size > MAX_SELECTED_EPISODES) {
          throw new Error(`success episode 超过 ${MAX_SELECTED_EPISODES} 条，请先按日期或搜索条件缩小范围`);
        }
      }
    }
    totalPages = list.totalPages;
    page += 1;
  } while (page <= totalPages);
  return Object.freeze([...accepted.values()]);
}

function normalizeJointAvailability(value, side) {
  const source = isObject(value) ? value : {};
  const dimension = nonnegativeInteger(source.dimension, 0);
  const count = nonnegativeInteger(source.count, 0);
  const available = source.available === true && dimension === JOINT_DIMENSION && count > 0;
  return Object.freeze({
    side,
    available,
    topic: safeString(source.topic, "", 300) || null,
    count,
    dimension,
  });
}

function normalizeTcpAvailability(value, side) {
  const source = isObject(value) ? value : {};
  const dimension = nonnegativeInteger(source.dimension, 0);
  const count = nonnegativeInteger(source.count, 0);
  const available = source.available === true && dimension === TCP_DIMENSION && count > 0;
  return Object.freeze({
    side,
    available,
    topic: safeString(source.topic, "", 300) || null,
    count,
    dimension,
  });
}

function normalizeGripperAvailability(value, side) {
  const source = isObject(value) ? value : {};
  const dimension = nonnegativeInteger(source.dimension, 0);
  const count = nonnegativeInteger(source.count, 0);
  const available = source.available === true
    && dimension === GRIPPER_DIMENSION
    && count > 0;
  return Object.freeze({
    side,
    available,
    topic: safeString(source.topic, "", 300) || null,
    count,
    dimension,
  });
}

function normalizeGripperCurveSide(value, side) {
  const source = isObject(value) ? value : {};
  const timestamps = Array.isArray(source.timestamps_ns)
    ? source.timestamps_ns.map(Number)
    : [];
  const values = Array.isArray(source.values) ? source.values.map(Number) : [];
  const available = source.available === true
    && timestamps.length > 0
    && timestamps.length === values.length
    && timestamps.every(Number.isFinite)
    && values.every(Number.isFinite)
    && timestamps.every((timestamp, index) => (
      index === 0 || timestamp > timestamps[index - 1]
    ));
  return Object.freeze({
    side,
    available,
    topic: safeString(source.topic, "", 300) || null,
    timestampsNs: Object.freeze(available ? timestamps : []),
    values: Object.freeze(available ? values : []),
  });
}

function normalizeGripperCurve(value) {
  const source = isObject(value) ? value : {};
  return Object.freeze({
    left: normalizeGripperCurveSide(source.left, "left"),
    right: normalizeGripperCurveSide(source.right, "right"),
  });
}

function normalizeRecordingVideo(value, role) {
  const source = isObject(value) ? value : {};
  const url = safeString(source.url, "", 500);
  const available = source.available === true
    && url.startsWith(`${RECORDINGS_URL}/`)
    && url.includes(`/videos/${role}`);
  return Object.freeze({
    role,
    available,
    url: available ? url : null,
    feature: safeString(source.feature, "", 200) || null,
    sizeBytes: nonnegativeInteger(source.size_bytes, 0),
  });
}

function normalizeRecordingVideos(value) {
  const source = isObject(value) ? value : {};
  return Object.freeze(Object.fromEntries(CAMERA_ROLES.map((role) => (
    [role, normalizeRecordingVideo(source[role], role)]
  ))));
}

function normalizeReplayVisualization(value) {
  const source = isObject(value) ? value : {};
  const configuredTools = isObject(source.configured_tools) ? source.configured_tools : {};
  return Object.freeze({
    configName: safeString(source.config_name, "", 160) || null,
    source: safeString(source.source, "unknown", 64),
    configuredTools: Object.freeze({
      left: isObject(configuredTools.left) ? configuredTools.left : null,
      right: isObject(configuredTools.right) ? configuredTools.right : null,
    }),
  });
}

function normalizeIssue(value) {
  if (!isObject(value)) return null;
  const message = safeString(value.message, "", 500);
  if (!message) return null;
  const rawSeverity = safeString(value.severity, "warning", 24).toLowerCase();
  const severity = ["error", "warning", "info"].includes(rawSeverity) ? rawSeverity : "warning";
  return Object.freeze({
    code: safeString(value.code, "recording_issue", 96),
    message,
    severity,
  });
}

function normalizeCameraRole(value, role) {
  const source = typeof value === "string"
    ? { device_key: value, status: "configured" }
    : isObject(value) ? value : {};
  const deviceKey = safeString(source.device_key ?? source.device, "", 240) || null;
  const topic = safeString(source.topic ?? source.camera_id, "", 300) || null;
  const shmName = safeString(source.shm_name ?? source.shm, "", 300) || null;
  const rawStatus = safeString(source.status, "", 48).toLowerCase();
  let status = rawStatus;
  if (source.black_placeholder === true || ["placeholder", "black"].includes(status)) status = "black_placeholder";
  else if (source.missing === true || status === "unavailable") status = "missing";
  else if (source.healthy === true || ["available", "recorded"].includes(status)) status = "healthy";
  if (!Object.prototype.hasOwnProperty.call(CAMERA_ROLE_STATUS_LABELS, status)) {
    status = deviceKey || topic ? "configured" : "unknown";
  }
  return Object.freeze({
    role,
    deviceKey,
    topic,
    shmName,
    status,
    count: nonnegativeInteger(source.count, 0),
    reason: safeString(source.reason, "", 400) || null,
  });
}

function normalizeCameraRoles(value) {
  const source = isObject(value) ? value : {};
  return Object.freeze(Object.fromEntries(CAMERA_ROLES.map((role) => (
    [role, normalizeCameraRole(source[role] ?? source[`${role}_camera`], role)]
  ))));
}

function cameraRoleIssues(cameraRoles) {
  return CAMERA_ROLES.flatMap((role) => {
    const camera = cameraRoles[role];
    if (!camera || !["missing", "black_placeholder"].includes(camera.status)) return [];
    const placeholder = camera.status === "black_placeholder";
    return [Object.freeze({
      code: `camera_${role}_${placeholder ? "black_placeholder" : "missing"}`,
      severity: "warning",
      message: placeholder
        ? `${CAMERA_ROLE_LABELS[role]} camera 将使用黑帧占位，episode 不含真实画面。`
        : `${CAMERA_ROLE_LABELS[role]} camera 缺失，episode 不含该视角。`,
    })];
  });
}

function cameraRoleWarning(reason) {
  return /camera mapping missing/i.test(String(reason || "")) ? "未配置映射" : "";
}

function normalizeDenseLabel(value) {
  const label = safeString(value, "other", 48).toLowerCase();
  if (["accept", "accepted", "success", "positive"].includes(label)) return "accept";
  if (["reject", "rejected", "failure", "negative"].includes(label)) return "reject";
  return "other";
}

function normalizeDenseSegment(value) {
  if (!isObject(value)) return null;
  const start = nonnegativeInteger(value.start_frame ?? value.start, -1);
  const end = nonnegativeInteger(value.end_frame ?? value.end, -1);
  if (start < 0 || end <= start) return null;
  return Object.freeze({
    start,
    end,
    label: normalizeDenseLabel(value.label ?? value.tag ?? value.result),
    checkpointId: safeString(value.checkpoint_id, "", 96) || null,
  });
}

function normalizeDenseCheckpoint(value) {
  if (!isObject(value)) return null;
  const frame = nonnegativeInteger(value.frame, -1);
  if (frame < 0) return null;
  const revision = finiteNumber(value.revision);
  return Object.freeze({
    frame,
    revision: Number.isInteger(revision) ? revision : null,
    timestampNs: finiteNumber(value.timestamp_ns),
    label: normalizeDenseLabel(value.label),
  });
}

function sampleEvenly(items, limit) {
  const budget = Math.max(0, nonnegativeInteger(limit, 0));
  if (!budget || !items.length) return [];
  if (items.length <= budget) return [...items];
  if (budget === 1) return [items[0]];
  return Array.from({ length: budget }, (_, index) => (
    items[Math.round((index * (items.length - 1)) / (budget - 1))]
  ));
}

function bucketDenseSegments(segments, totalFrames, limit) {
  const budget = Math.max(0, nonnegativeInteger(limit, 0));
  if (!budget || !segments.length) return [];
  if (segments.length <= budget) return [...segments];
  const total = Math.max(1, nonnegativeInteger(totalFrames, 1));
  const buckets = Array.from({ length: budget }, () => null);
  for (const segment of segments) {
    const start = clamp(segment.start, 0, total);
    const end = clamp(segment.end, start, total);
    const midpoint = start + (end - start) / 2;
    const index = clamp(Math.floor((midpoint / total) * budget), 0, budget - 1);
    const previous = buckets[index];
    buckets[index] = previous
      ? Object.freeze({
        ...previous,
        start: Math.min(previous.start, start),
        end: Math.max(previous.end, end),
        label: previous.label === "reject" || segment.label === "reject" ? "reject" : previous.label,
      })
      : Object.freeze({ ...segment, start, end });
  }
  return buckets.filter(Boolean);
}

function planDenseTimeline(segments, checkpoints, totalFrames, maxNodes = DENSE_DOM_NODE_LIMIT) {
  const sourceSegments = Array.isArray(segments) ? segments : [];
  const sourceCheckpoints = Array.isArray(checkpoints) ? checkpoints : [];
  const budget = Math.max(1, nonnegativeInteger(maxNodes, DENSE_DOM_NODE_LIMIT));
  const sourceCount = sourceSegments.length + sourceCheckpoints.length;
  if (sourceCount <= budget) {
    return Object.freeze({
      segments: Object.freeze([...sourceSegments]),
      checkpoints: Object.freeze([...sourceCheckpoints]),
      truncated: false,
    });
  }
  let segmentBudget = sourceSegments.length
    ? Math.max(1, Math.round((budget * sourceSegments.length) / sourceCount))
    : 0;
  let checkpointBudget = sourceCheckpoints.length ? Math.max(1, budget - segmentBudget) : 0;
  if (segmentBudget + checkpointBudget > budget) {
    if (checkpointBudget > segmentBudget) checkpointBudget -= 1;
    else segmentBudget -= 1;
  }
  const plannedSegments = bucketDenseSegments(sourceSegments, totalFrames, segmentBudget);
  const plannedCheckpoints = sampleEvenly(sourceCheckpoints, checkpointBudget);
  return Object.freeze({
    segments: Object.freeze(plannedSegments),
    checkpoints: Object.freeze(plannedCheckpoints),
    truncated: true,
  });
}

function normalizeRecordingSummary(payload, expectedId = null) {
  if (!isObject(payload)) throw new Error("Episode summary 响应格式无效");
  const id = safeString(payload.id, "", 256);
  if (!id || (expectedId && id !== expectedId)) throw new Error("Episode summary 标识不匹配");
  const dataFormat = safeString(payload.data_format, "NEDF3", 16).toUpperCase();
  const topics = Array.isArray(payload.topics)
    ? payload.topics.map((topic) => ({
      name: safeString(topic?.name, "", 400),
      count: nonnegativeInteger(topic?.count, 0),
      kind: "topic",
    })).filter((topic) => topic.name)
    : [];
  const cameras = Array.isArray(payload.cameras)
    ? payload.cameras.map((camera) => ({
      name: safeString(camera?.id, "", 240),
      count: nonnegativeInteger(camera?.count, 0),
      kind: "camera",
    })).filter((camera) => camera.name)
    : [];
  const issues = Array.isArray(payload.issues)
    ? payload.issues.map(normalizeIssue).filter(Boolean)
    : [];
  const cameraRoles = normalizeCameraRoles(payload.camera_roles);
  for (const issue of cameraRoleIssues(cameraRoles)) {
    if (!issues.some((item) => item.code === issue.code)) issues.push(issue);
  }
  const denseSource = isObject(payload.dense) ? payload.dense : {};
  const segments = Array.isArray(denseSource.segments)
    ? denseSource.segments.map(normalizeDenseSegment).filter(Boolean)
    : [];
  const checkpoints = Array.isArray(denseSource.checkpoints)
    ? denseSource.checkpoints.map(normalizeDenseCheckpoint).filter(Boolean)
    : [];
  const range = isObject(payload.time_range_ns) ? payload.time_range_ns : {};
  const startNs = finiteNumber(range.start);
  const endNs = finiteNumber(range.end);
  const validRange = startNs !== null && endNs !== null && endNs >= startNs;
  const joints = isObject(payload.joints) ? payload.joints : {};
  const left = normalizeJointAvailability(joints.left, "left");
  const right = normalizeJointAvailability(joints.right, "right");
  const tcpSource = isObject(payload.tcp) ? payload.tcp : {};
  const tcpLeft = normalizeTcpAvailability(tcpSource.left, "left");
  const tcpRight = normalizeTcpAvailability(tcpSource.right, "right");
  const gripperSource = isObject(payload.grippers) ? payload.grippers : {};
  const gripperLeft = normalizeGripperAvailability(gripperSource.left, "left");
  const gripperRight = normalizeGripperAvailability(gripperSource.right, "right");
  return Object.freeze({
    id,
    dataFormat,
    status: safeString(payload.status, "unknown", 48).toLowerCase(),
    result: normalizeResult(payload.result),
    episodeIndex: Number.isInteger(Number(payload.episode_index)) ? Number(payload.episode_index) : null,
    episodeId: safeString(payload.episode_id, "", 160) || null,
    configName: safeString(payload.config_name, "", 160) || null,
    taskName: safeString(payload.task_name, "", 300) || null,
    timeRangeNs: Object.freeze({ start: validRange ? startNs : null, end: validRange ? endNs : null }),
    topics: Object.freeze([...topics, ...cameras]),
    cameraRoles,
    joints: Object.freeze({ left, right }),
    tcp: Object.freeze({ left: tcpLeft, right: tcpRight }),
    grippers: Object.freeze({ left: gripperLeft, right: gripperRight }),
    gripperCurve: normalizeGripperCurve(payload.gripper_curve),
    visualization: normalizeReplayVisualization(payload.visualization),
    videos: normalizeRecordingVideos(payload.videos),
    dense: Object.freeze({
      available: denseSource.available === true,
      segments: Object.freeze(segments),
      checkpoints: Object.freeze(checkpoints),
    }),
    issues: Object.freeze(issues),
  });
}

function validJointVector(value) {
  return Array.isArray(value)
    && value.length === JOINT_DIMENSION
    && value.every((component) => Number.isFinite(Number(component)));
}

function normalizeTrajectoryPayload(payload, expectedId = null) {
  if (!isObject(payload)) throw new Error("关节轨迹响应格式无效");
  const id = safeString(payload.id, "", 256);
  if (!id || (expectedId && id !== expectedId)) throw new Error("关节轨迹标识不匹配");
  const timestamps = Array.isArray(payload.timestamps_ns) ? payload.timestamps_ns.map(Number) : [];
  const left = Array.isArray(payload.left) ? payload.left : [];
  const right = Array.isArray(payload.right) ? payload.right : [];
  if (!timestamps.length || left.length !== timestamps.length || right.length !== timestamps.length) {
    throw new Error("左右臂轨迹与时间戳长度不一致");
  }
  if (!left.every(validJointVector) || !right.every(validJointVector)) {
    throw new Error("轨迹必须包含左右臂真实 7D joint 数据");
  }
  if (!timestamps.every(Number.isFinite)) throw new Error("轨迹时间戳无效");
  for (let index = 1; index < timestamps.length; index += 1) {
    if (timestamps[index] <= timestamps[index - 1]) throw new Error("轨迹时间戳必须严格递增");
  }
  const start = nonnegativeInteger(payload.start, 0);
  const stride = Math.max(1, nonnegativeInteger(payload.stride, 1));
  const totalFrames = nonnegativeInteger(payload.total_frames, 0);
  const returnedFrames = nonnegativeInteger(payload.returned_frames, timestamps.length);
  if (returnedFrames !== timestamps.length || totalFrames <= 0 || start >= totalFrames) {
    throw new Error("轨迹帧范围无效");
  }
  const computedEnd = Math.min(totalFrames, start + stride * timestamps.length);
  const end = clamp(nonnegativeInteger(payload.end, computedEnd), start + 1, totalFrames);
  return Object.freeze({
    id,
    start,
    end,
    stride,
    totalFrames,
    returnedFrames,
    timestampsNs: Object.freeze(timestamps),
    left: Object.freeze(left.map((vector) => Object.freeze(vector.map(Number)))),
    right: Object.freeze(right.map((vector) => Object.freeze(vector.map(Number)))),
    issues: Object.freeze(Array.isArray(payload.issues) ? payload.issues.map(normalizeIssue).filter(Boolean) : []),
  });
}

function validPlaybackVector(value, dimension) {
  return Array.isArray(value)
    && value.length === dimension
    && value.every((component) => Number.isFinite(Number(component)));
}

function normalizePlaybackStream(value, timestamps, label, dimension) {
  const source = isObject(value) ? value : {};
  const leftAvailable = source.left_available === true;
  const rightAvailable = source.right_available === true;
  const left = Array.isArray(source.left) ? source.left : [];
  const right = Array.isArray(source.right) ? source.right : [];
  for (const [side, available, values] of [
    ["left", leftAvailable, left],
    ["right", rightAvailable, right],
  ]) {
    if (!available) continue;
    if (
      values.length !== timestamps.length
      || !values.every((vector) => validPlaybackVector(vector, dimension))
    ) {
      throw new Error(`${label} ${side} 轨迹必须是与时间戳对齐的真实 ${dimension}D 数据`);
    }
  }
  return Object.freeze({
    available: leftAvailable || rightAvailable,
    leftAvailable,
    rightAvailable,
    left: Object.freeze(leftAvailable
      ? left.map((vector) => Object.freeze(vector.map(Number)))
      : []),
    right: Object.freeze(rightAvailable
      ? right.map((vector) => Object.freeze(vector.map(Number)))
      : []),
  });
}

function normalizePlaybackTrajectoryPayload(payload, expectedId = null) {
  if (!isObject(payload)) throw new Error("回放轨迹响应格式无效");
  const id = safeString(payload.id, "", 256);
  if (!id || (expectedId && id !== expectedId)) throw new Error("回放轨迹标识不匹配");
  const timestamps = Array.isArray(payload.timestamps_ns)
    ? payload.timestamps_ns.map(Number)
    : [];
  if (!timestamps.length || !timestamps.every(Number.isFinite)) {
    throw new Error("回放轨迹时间戳无效");
  }
  for (let index = 1; index < timestamps.length; index += 1) {
    if (timestamps[index] <= timestamps[index - 1]) {
      throw new Error("回放轨迹时间戳必须严格递增");
    }
  }
  const joints = normalizePlaybackStream(payload.joints, timestamps, "joint", JOINT_DIMENSION);
  const tcp = normalizePlaybackStream(payload.tcp, timestamps, "TCP", TCP_DIMENSION);
  const grippers = normalizePlaybackStream(
    payload.grippers,
    timestamps,
    "gripper",
    GRIPPER_DIMENSION,
  );
  if (!joints.available && !tcp.available) throw new Error("回放轨迹不含 TCP 或 joint 数据");
  const start = nonnegativeInteger(payload.start, 0);
  const stride = Math.max(1, nonnegativeInteger(payload.stride, 1));
  const totalFrames = nonnegativeInteger(payload.total_frames, 0);
  const returnedFrames = nonnegativeInteger(payload.returned_frames, timestamps.length);
  if (returnedFrames !== timestamps.length || totalFrames <= 0 || start >= totalFrames) {
    throw new Error("回放轨迹帧范围无效");
  }
  const computedEnd = Math.min(totalFrames, start + stride * timestamps.length);
  const end = clamp(nonnegativeInteger(payload.end, computedEnd), start + 1, totalFrames);
  return Object.freeze({
    id,
    start,
    end,
    stride,
    totalFrames,
    returnedFrames,
    timestampsNs: Object.freeze(timestamps),
    joints,
    tcp,
    grippers,
    issues: Object.freeze(Array.isArray(payload.issues)
      ? payload.issues.map(normalizeIssue).filter(Boolean)
      : []),
  });
}

function planTrajectoryRequest(targetFrame, totalFrames, options = {}) {
  const total = Math.max(1, nonnegativeInteger(totalFrames, 1));
  const target = clamp(nonnegativeInteger(targetFrame, 0), 0, total - 1);
  const windowFrames = Math.max(1, nonnegativeInteger(options.windowFrames, TRAJECTORY_WINDOW_FRAMES));
  const maxPoints = Math.max(2, nonnegativeInteger(options.maxPoints, TRAJECTORY_MAX_POINTS));
  const start = Math.floor(target / windowFrames) * windowFrames;
  const end = Math.min(total, start + windowFrames);
  const stride = Math.max(1, Math.ceil((end - start) / maxPoints));
  return Object.freeze({ start, end, stride, maxPoints });
}

function trajectoryPlanKey(targetFrame, totalFrames) {
  const plan = planTrajectoryRequest(targetFrame, totalFrames);
  return `${plan.start}:${plan.end}:${plan.stride}`;
}

function cacheStateFromPayload(payload) {
  const raw = safeString(payload?.cache_state ?? payload?.cache?.state, "", 32).toLowerCase();
  return ["building", "ready", "failed"].includes(raw) ? raw : "ready";
}

function cacheRetryDelay(payload, attempt) {
  const requested = nonnegativeInteger(payload?.retry_after_ms, 0);
  if (requested > 0) return clamp(requested, 100, 5000);
  return CACHE_POLL_DELAYS_MS[Math.min(attempt, CACHE_POLL_DELAYS_MS.length - 1)];
}

function cacheStateError(payload, fallback = "快速回放缓存不可用") {
  const state = cacheStateFromPayload(payload);
  if (state === "ready") return null;
  const error = new Error(
    state === "building"
      ? "正在准备快速回放"
      : safeString(payload?.error, fallback, 500),
  );
  error.code = `trajectory_cache_${state}`;
  error.cacheState = state;
  error.retryAfterMs = cacheRetryDelay(payload, 0);
  error.payload = payload;
  return error;
}

function abortError() {
  const error = new Error("Request aborted");
  error.name = "AbortError";
  return error;
}

function waitForCacheRetry(delayMs, signal) {
  if (signal?.aborted) return Promise.reject(abortError());
  return new Promise((resolve, reject) => {
    const timeout = globalThis.setTimeout(resolve, delayMs);
    signal?.addEventListener("abort", () => {
      globalThis.clearTimeout(timeout);
      reject(abortError());
    }, { once: true });
  });
}

async function pollCacheOperation(operation, { signal, onBuilding } = {}) {
  let attempt = 0;
  while (true) {
    if (signal?.aborted) throw abortError();
    let value;
    let stateError = null;
    try {
      value = await operation();
      stateError = cacheStateError(value);
    } catch (error) {
      stateError = error;
    }
    if (!stateError) return value;
    if (stateError.name === "AbortError" || signal?.aborted) throw abortError();
    if (stateError.cacheState !== "building") throw stateError;
    onBuilding?.(attempt, stateError.payload);
    await waitForCacheRetry(cacheRetryDelay(stateError.payload, attempt), signal);
    attempt += 1;
  }
}

function interpolateVectors(left, right, alpha) {
  return left.map((value, index) => value + (right[index] - value) * alpha);
}

function interpolatePlaybackStream(stream, lowerIndex, upperIndex, alpha) {
  if (!stream?.available) {
    return Object.freeze({
      available: false,
      leftAvailable: false,
      rightAvailable: false,
      left: null,
      right: null,
    });
  }
  const interpolateSide = (side, available) => available
    ? Object.freeze(interpolateVectors(
      stream[side][lowerIndex],
      stream[side][upperIndex],
      alpha,
    ))
    : null;
  return Object.freeze({
    available: stream.available,
    leftAvailable: stream.leftAvailable,
    rightAvailable: stream.rightAvailable,
    left: interpolateSide("left", stream.leftAvailable),
    right: interpolateSide("right", stream.rightAvailable),
  });
}

function interpolateTrajectoryFrame(chunk, frame) {
  if (!chunk || frame < chunk.start || frame >= chunk.end) return null;
  const position = (frame - chunk.start) / chunk.stride;
  const lowerIndex = clamp(Math.floor(position), 0, chunk.returnedFrames - 1);
  const upperIndex = clamp(Math.ceil(position), 0, chunk.returnedFrames - 1);
  const alpha = clamp(position - lowerIndex, 0, 1);
  const timestamp = chunk.timestampsNs[lowerIndex]
    + (chunk.timestampsNs[upperIndex] - chunk.timestampsNs[lowerIndex]) * alpha;
  if (!chunk.joints && !chunk.tcp) {
    return Object.freeze({
      frame,
      timestampNs: timestamp,
      left: Object.freeze(interpolateVectors(chunk.left[lowerIndex], chunk.left[upperIndex], alpha)),
      right: Object.freeze(interpolateVectors(chunk.right[lowerIndex], chunk.right[upperIndex], alpha)),
    });
  }
  const joints = interpolatePlaybackStream(chunk.joints, lowerIndex, upperIndex, alpha);
  const tcp = interpolatePlaybackStream(chunk.tcp, lowerIndex, upperIndex, alpha);
  const grippers = interpolatePlaybackStream(chunk.grippers, lowerIndex, upperIndex, alpha);
  const trailLength = Math.max(2, Math.ceil(chunk.returnedFrames * TCP_TRAIL_FRACTION));
  const trailStart = Math.max(0, lowerIndex - trailLength + 1);
  const tcpTrail = Object.freeze({
    left: Object.freeze(tcp.leftAvailable
      ? chunk.tcp.left.slice(trailStart, lowerIndex + 1)
      : []),
    right: Object.freeze(tcp.rightAvailable
      ? chunk.tcp.right.slice(trailStart, lowerIndex + 1)
      : []),
  });
  return Object.freeze({
    frame,
    timestampNs: timestamp,
    joints,
    tcp,
    grippers,
    tcpTrail,
    left: joints.left,
    right: joints.right,
  });
}

class TrajectoryChunkStore {
  constructor({ fetchJson, maxEntries = TRAJECTORY_CACHE_ENTRIES } = {}) {
    if (typeof fetchJson !== "function") throw new Error("TrajectoryChunkStore needs fetchJson");
    this.fetchJson = fetchJson;
    this.maxEntries = Math.max(1, nonnegativeInteger(maxEntries, TRAJECTORY_CACHE_ENTRIES));
    this.recordingId = null;
    this.generation = 0;
    this.chunks = new Map();
    this.pending = new Map();
    this.overviewPending = null;
  }

  reset(recordingId = null) {
    this.recordingId = recordingId;
    this.generation += 1;
    this.chunks.clear();
    this.pending.clear();
    this.overviewPending = null;
  }

  _key(plan) {
    return `${plan.start}:${plan.end}:${plan.stride}`;
  }

  _remember(key, chunk) {
    this.chunks.delete(key);
    this.chunks.set(key, chunk);
    while (this.chunks.size > this.maxEntries) this.chunks.delete(this.chunks.keys().next().value);
  }

  sample(frame) {
    for (const [key, chunk] of [...this.chunks.entries()].reverse()) {
      const sample = interpolateTrajectoryFrame(chunk, frame);
      if (!sample) continue;
      this._remember(key, chunk);
      return sample;
    }
    return null;
  }

  async load(recordingId, targetFrame, totalFrames, signal) {
    if (this.recordingId !== recordingId) this.reset(recordingId);
    const plan = planTrajectoryRequest(targetFrame, totalFrames);
    const key = this._key(plan);
    if (this.chunks.has(key)) return this.chunks.get(key);
    if (this.pending.has(key)) return this.pending.get(key);
    const query = new URLSearchParams({
      start: String(plan.start),
      end: String(plan.end),
      stride: String(plan.stride),
      max_points: String(plan.maxPoints),
    });
    const generation = this.generation;
    let promise;
    promise = this.fetchJson(
      `${RECORDINGS_URL}/${encodeURIComponent(recordingId)}/playback-trajectory?${query}`,
      { signal },
    ).then((payload) => {
      const cacheError = cacheStateError(payload);
      if (cacheError) throw cacheError;
      const chunk = normalizePlaybackTrajectoryPayload(payload, recordingId);
      if (this.generation === generation && this.recordingId === recordingId) {
        this._remember(key, chunk);
      }
      return chunk;
    }).finally(() => {
      if (this.pending.get(key) === promise) this.pending.delete(key);
    });
    this.pending.set(key, promise);
    return promise;
  }

  async loadOverview(recordingId, totalFrames, signal) {
    if (this.recordingId !== recordingId) this.reset(recordingId);
    if (this.overviewPending) return this.overviewPending;
    const total = Math.max(1, nonnegativeInteger(totalFrames, 1));
    const stride = Math.max(1, Math.ceil(total / TCP_OVERVIEW_MAX_POINTS));
    const query = new URLSearchParams({
      start: "0",
      end: String(total),
      stride: String(stride),
      max_points: String(TCP_OVERVIEW_MAX_POINTS),
    });
    const generation = this.generation;
    let promise;
    promise = this.fetchJson(
      `${RECORDINGS_URL}/${encodeURIComponent(recordingId)}/playback-trajectory?${query}`,
      { signal },
    ).then((payload) => {
      const cacheError = cacheStateError(payload);
      if (cacheError) throw cacheError;
      return normalizePlaybackTrajectoryPayload(payload, recordingId);
    }).finally(() => {
      if (this.overviewPending === promise && this.generation === generation) {
        this.overviewPending = null;
      }
    });
    this.overviewPending = promise;
    return promise;
  }
}

function formatDuration(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const remainder = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function summaryDurationSeconds(summary) {
  const { start, end } = summary?.timeRangeNs || {};
  return start === null || end === null ? null : Math.max(0, (end - start) / 1e9);
}

function availablePairFrameCount(value) {
  const counts = [value?.left, value?.right]
    .filter((item) => item?.available)
    .map((item) => item.count)
    .filter((count) => count > 0);
  return counts.length ? Math.min(...counts) : 0;
}

function summaryPlaybackFrameCount(summary) {
  return availablePairFrameCount(summary?.tcp)
    || availablePairFrameCount(summary?.joints)
    || Math.max(0, ...Object.values(summary?.cameraRoles || {}).map((item) => item.count || 0));
}

async function defaultFetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    cache: "no-store",
    ...options,
  });
  let payload = null;
  try { payload = await response.json(); } catch (_error) { /* handled below */ }
  if (!response.ok) {
    const error = cacheStateError(payload, `请求失败 (${response.status})`)
      || new Error(safeString(payload?.error, `请求失败 (${response.status})`, 500));
    error.status = response.status;
    if (!error.code) error.code = safeString(payload?.code, "request_failed", 96);
    error.payload = payload;
    throw error;
  }
  return payload;
}

function createElement(tag, className = "", text = "") {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function initializeRecordingReplay({ fetchJson = defaultFetchJson } = {}) {
  const elements = {
    rootState: document.querySelector("#recording-root-state"),
    connection: document.querySelector("#recording-connection-state"),
    refresh: document.querySelector("#recordings-refresh"),
    pagePrevious: document.querySelector("#recordings-page-previous"),
    pageNext: document.querySelector("#recordings-page-next"),
    pageReadout: document.querySelector("#recordings-page-readout"),
    selectAccepted: document.querySelector("#recordings-select-accepted"),
    dateFilter: document.querySelector("#recordings-date-filter"),
    statusFilter: document.querySelector("#recordings-status-filter"),
    search: document.querySelector("#recordings-search"),
    count: document.querySelector("#recording-count"),
    selectionCount: document.querySelector("#recording-selection-count"),
    selectionClear: document.querySelector("#recording-selection-clear"),
    exportOpen: document.querySelector("#recording-export-open"),
    deleteOpen: document.querySelector("#recording-delete-open"),
    deleteDialog: document.querySelector("#recording-delete-dialog"),
    deleteForm: document.querySelector("#recording-delete-form"),
    deleteClose: document.querySelector("#recording-delete-close"),
    deleteCancel: document.querySelector("#recording-delete-cancel"),
    deleteConfirm: document.querySelector("#recording-delete-confirm"),
    deleteSummary: document.querySelector("#recording-delete-summary"),
    deleteError: document.querySelector("#recording-delete-error"),
    exportActivity: document.querySelector("#recording-export-activity"),
    exportDialog: document.querySelector("#recording-export-dialog"),
    exportForm: document.querySelector("#recording-export-form"),
    exportClose: document.querySelector("#recording-export-close"),
    exportCancel: document.querySelector("#recording-export-cancel"),
    exportSubmit: document.querySelector("#recording-export-submit"),
    exportSelection: document.querySelector("#recording-export-selection"),
    exportState: document.querySelector("#recording-export-state"),
    exportName: document.querySelector("#recording-export-name"),
    exportProgress: document.querySelector("#recording-export-progress"),
    exportProgressBar: document.querySelector("#recording-export-progress-bar"),
    exportProgressLabel: document.querySelector("#recording-export-progress-label"),
    exportResult: document.querySelector("#recording-export-result"),
    exportPath: document.querySelector("#recording-export-path"),
    exportError: document.querySelector("#recording-export-error"),
    list: document.querySelector("#recording-episode-list"),
    facts: document.querySelector("#recording-episode-facts"),
    issues: document.querySelector("#recording-issues"),
    viewerRoot: document.querySelector("#recording-arm-viewer-root"),
    viewerSource: document.querySelector("#recording-viewer-source"),
    unavailable: document.querySelector("#recording-viewer-unavailable"),
    unavailableReason: document.querySelector("#recording-viewer-unavailable-reason"),
    jointHealth: document.querySelector("#recording-joint-health"),
    cameraRoleHealth: document.querySelector("#recording-camera-role-health"),
    topicFormat: document.querySelector("#recording-topic-format"),
    topicList: document.querySelector("#recording-topic-list"),
    play: document.querySelector("#recording-play-toggle"),
    speed: document.querySelector("#recording-speed"),
    scrubber: document.querySelector("#recording-scrubber"),
    denseTrack: document.querySelector("#recording-dense-track"),
    frameReadout: document.querySelector("#recording-frame-readout"),
    timeReadout: document.querySelector("#recording-time-readout"),
    gripperChart: document.querySelector("#recording-gripper-chart"),
    gripperLeftPath: document.querySelector("#recording-gripper-left-path"),
    gripperRightPath: document.querySelector("#recording-gripper-right-path"),
    gripperCursor: document.querySelector("#recording-gripper-cursor"),
    gripperLeftValue: document.querySelector("#recording-gripper-left-value"),
    gripperRightValue: document.querySelector("#recording-gripper-right-value"),
    gripperEmpty: document.querySelector("#recording-gripper-empty"),
    videos: Object.fromEntries(CAMERA_ROLES.map((role) => [
      role,
      document.querySelector(`#recording-video-${role}`),
    ])),
    videoCards: Object.fromEntries(CAMERA_ROLES.map((role) => [
      role,
      document.querySelector(`[data-video-role="${role}"]`),
    ])),
  };
  if (
    Object.entries(elements).some(([key, element]) => (
      key !== "videos" && key !== "videoCards" && !element
    ))
    || CAMERA_ROLES.some((role) => !elements.videos[role] || !elements.videoCards[role])
  ) return null;

  const app = {
    items: [],
    dates: [],
    dateFilter: "all",
    statusFilter: "all",
    searchQuery: "",
    searchTimer: 0,
    selectedExportIds: new Set(),
    selectedItems: new Map(),
    acceptedSelectionController: null,
    acceptedSelectionLoading: false,
    acceptedSelectionError: null,
    deleteSubmitting: false,
    exportJobId: null,
    exportJob: null,
    exportController: null,
    exportPollTimer: 0,
    exportSubmitting: false,
    page: 1,
    totalPages: 0,
    selectedId: null,
    summary: null,
    totalFrames: 0,
    currentFrame: 0,
    frameFloat: 0,
    estimatedFps: DEFAULT_PLAYBACK_FPS,
    speed: 1,
    playing: false,
    animationHandle: 0,
    lastAnimationTime: null,
    lastRenderedTime: Number.NEGATIVE_INFINITY,
    listController: null,
    summaryController: null,
    trajectoryRequest: null,
    trajectoryPrefetch: null,
    trajectoryRequestSerial: 0,
    scrubTimer: 0,
    scrubbing: false,
    scrubberPointerActive: false,
    selectionRevision: 0,
    frameRevision: 0,
    lastVideoSyncAt: 0,
    gripperCurve: null,
    gripperDomain: null,
    disposed: false,
  };
  let viewer = null;
  let viewerPromise = null;
  const trajectories = new TrajectoryChunkStore({ fetchJson });

  async function ensureViewer(expectedSelectionRevision = app.selectionRevision) {
    if (viewer) {
      await viewer.ready;
      return viewer;
    }
    if (!viewerPromise) {
      viewerPromise = import("./arm_viewer.js?v=20260726-recording-gripper-v4").then(({ DualArmViewer }) => {
        if (app.disposed || expectedSelectionRevision !== app.selectionRevision) {
          viewerPromise = null;
          return null;
        }
        viewer = new DualArmViewer(elements.viewerRoot, {
          robotStateUrl: null,
          homePoseCandidatesUrl: null,
          statusMode: "external",
        });
        viewer.sceneRoot.visible = false;
        return viewer.ready.then(() => viewer);
      }).catch((error) => {
        viewerPromise = null;
        throw error;
      });
    }
    return viewerPromise;
  }

  function setConnection(state, label) {
    elements.connection.dataset.state = state;
    elements.connection.lastElementChild.textContent = label;
  }

  function stopPlayback() {
    app.playing = false;
    app.lastAnimationTime = null;
    if (app.animationHandle) cancelAnimationFrame(app.animationHandle);
    app.animationHandle = 0;
    for (const video of Object.values(elements.videos)) video.pause();
    elements.play.innerHTML = "&#9654;";
    elements.play.title = "播放";
    elements.play.setAttribute("aria-label", "播放");
  }

  function masterRecordedVideo() {
    for (const role of ["head", "left", "right"]) {
      const video = elements.videos[role];
      if (video.readyState >= 1 && Number.isFinite(video.duration) && video.duration > 0) {
        return video;
      }
    }
    return null;
  }

  function currentVideoRatio() {
    const master = masterRecordedVideo();
    if (!master) return null;
    return clamp(master.currentTime / master.duration, 0, 1);
  }

  function syncVideosToRatio(ratio, { force = false } = {}) {
    const normalized = clamp(Number(ratio) || 0, 0, 1);
    const now = performance.now();
    for (const video of Object.values(elements.videos)) {
      if (video.readyState < 1 || !Number.isFinite(video.duration) || video.duration <= 0) continue;
      const target = normalized * video.duration;
      if (force || Math.abs(video.currentTime - target) > 0.18) video.currentTime = target;
    }
    app.lastVideoSyncAt = now;
  }

  function syncVideosToFrame(frame, { force = false } = {}) {
    const maximum = Math.max(1, app.totalFrames - 1);
    syncVideosToRatio(clamp(frame / maximum, 0, 1), { force });
  }

  function correctVideoDrift(force = false) {
    const ratio = currentVideoRatio();
    if (ratio === null) return;
    const now = performance.now();
    if (!force && now - app.lastVideoSyncAt < 500) return;
    const master = masterRecordedVideo();
    for (const video of Object.values(elements.videos)) {
      if (
        video === master
        || video.readyState < 1
        || !Number.isFinite(video.duration)
        || video.duration <= 0
      ) continue;
      const target = ratio * video.duration;
      if (force || Math.abs(video.currentTime - target) > 0.18) video.currentTime = target;
    }
    app.lastVideoSyncAt = now;
  }

  function clearRecordedVideos() {
    for (const role of CAMERA_ROLES) {
      const video = elements.videos[role];
      const card = elements.videoCards[role];
      video.pause();
      video.onloadedmetadata = null;
      video.oncanplay = null;
      video.onended = null;
      video.onerror = null;
      video.removeAttribute("src");
      video.load();
      card.dataset.state = "unavailable";
      card.querySelector(".recording-video-placeholder").textContent = "视频未就绪";
    }
  }

  function setupRecordedVideos(summary) {
    clearRecordedVideos();
    for (const role of CAMERA_ROLES) {
      const source = summary.videos[role];
      const video = elements.videos[role];
      const card = elements.videoCards[role];
      const placeholder = card.querySelector(".recording-video-placeholder");
      if (!source.available || !source.url) {
        const camera = summary.cameraRoles[role];
        placeholder.textContent = camera?.count
          ? `${summary.dataFormat} 视频缓存不可用`
          : `${summary.dataFormat} 中无该视角`;
        continue;
      }
      card.dataset.state = "loading";
      placeholder.textContent = "正在读取视频";
      video.playbackRate = app.speed;
      video.onloadedmetadata = () => {
        if (summary !== app.summary) return;
        video.playbackRate = app.speed;
        syncVideosToFrame(app.currentFrame, { force: true });
      };
      video.oncanplay = () => {
        if (summary !== app.summary) return;
        card.dataset.state = "ready";
        if (app.playing) video.play().catch(() => {});
      };
      video.onended = () => {
        if (video === masterRecordedVideo()) {
          showFrame(Math.max(0, app.totalFrames - 1), {
            preserveFloat: true,
            syncVideos: false,
          });
          stopPlayback();
        }
      };
      video.onerror = () => {
        if (summary !== app.summary) return;
        card.dataset.state = "error";
        placeholder.textContent = "视频读取失败";
      };
      video.src = source.url;
      video.load();
    }
  }

  function setViewerUnavailable(reason) {
    stopPlayback();
    if (viewer) {
      viewer.sceneRoot.visible = false;
      viewer._invalidateRender?.("recording-unavailable");
      viewer.status.dataset.state = "no-data";
      viewer.status.textContent = "录制中没有可回放的 TCP 或 joint 数据";
    }
    elements.unavailable.hidden = false;
    elements.unavailableReason.textContent = reason;
    elements.play.disabled = true;
    elements.scrubber.disabled = true;
  }

  function setViewerAvailable({ jointsAvailable = false, tcpAvailable = false } = {}) {
    if (!viewer) return;
    viewer.sceneRoot.visible = true;
    viewer.setArmModelsVisible?.(jointsAvailable);
    viewer._invalidateRender?.("recording-available");
    elements.unavailable.hidden = true;
    elements.play.disabled = app.totalFrames <= 1;
    elements.scrubber.disabled = app.totalFrames <= 1;
    viewer.status.dataset.state = "replay";
    viewer.status.textContent = jointsAvailable
      ? "录制 URDF + TCP 轨迹回放"
      : tcpAvailable
        ? "录制 TCP 轨迹回放"
        : "录制视频回放";
  }

  function applyRecordedVisualization(summary) {
    if (!viewer) return;
    const tools = summary?.visualization?.configuredTools || {};
    viewer.updateConfiguredTools?.(tools.left, tools.right);
  }

  function setViewerPreparing() {
    stopPlayback();
    if (viewer) {
      viewer.sceneRoot.visible = true;
      viewer._invalidateRender?.("recording-cache-building");
      viewer.status.dataset.state = "loading";
      viewer.status.textContent = "正在准备快速回放";
    }
    elements.unavailable.hidden = false;
    elements.unavailableReason.textContent = "正在准备快速回放，请稍候…";
    elements.play.disabled = true;
    elements.scrubber.disabled = app.totalFrames <= 1;
  }

  function setViewerCacheFailed(message) {
    stopPlayback();
    if (viewer) {
      viewer.sceneRoot.visible = true;
      viewer._invalidateRender?.("recording-cache-failed");
      viewer.status.dataset.state = "error";
      viewer.status.textContent = "快速回放准备失败";
    }
    elements.unavailable.hidden = false;
    elements.unavailableReason.textContent = `${message || "快速回放准备失败"}；可刷新或重新选择 episode 重试。`;
    elements.play.disabled = true;
    elements.scrubber.disabled = app.totalFrames <= 1;
  }

  function renderDateFilters() {
    const options = [
      Object.assign(document.createElement("option"), {
        value: "all",
        textContent: `全部日期 (${app.dates.reduce((sum, row) => sum + row.count, 0)})`,
      }),
      ...app.dates.map((row) => Object.assign(document.createElement("option"), {
        value: row.id,
        textContent: `${row.label} (${row.count})`,
      })),
    ];
    elements.dateFilter.replaceChildren(...options);
    if (!options.some((option) => option.value === app.dateFilter)) {
      app.dateFilter = "all";
    }
    elements.dateFilter.value = app.dateFilter;
    elements.statusFilter.value = app.statusFilter;
    if (elements.search.value !== app.searchQuery) {
      elements.search.value = app.searchQuery;
    }
  }

  function selectedExportCount() {
    return app.selectedExportIds.size;
  }

  function catalogEpisodeCount() {
    if (app.dateFilter === "all") {
      return app.dates.reduce((total, row) => total + row.count, 0);
    }
    return app.dates.find((row) => row.id === app.dateFilter)?.count || 0;
  }

  function cancelAcceptedSelection() {
    app.acceptedSelectionController?.abort();
    app.acceptedSelectionController = null;
    app.acceptedSelectionLoading = false;
  }

  function renderExportSelection() {
    const count = selectedExportCount();
    const selectedItems = [...app.selectedItems.values()];
    const exportableCount = selectedItems.filter(recordingItemExportEligible).length;
    elements.selectionCount.textContent = app.acceptedSelectionLoading
      ? "选择中..."
      : app.acceptedSelectionError
        ? "选择失败"
        : `已选 ${count}`;
    elements.selectionCount.title = app.acceptedSelectionError || "";
    elements.selectionClear.disabled = count === 0;
    elements.exportOpen.disabled = count === 0 || exportableCount !== count;
    elements.deleteOpen.disabled = count === 0 || app.deleteSubmitting;
    elements.exportSubmit.disabled = count === 0 || exportableCount !== count || app.exportSubmitting;
    elements.exportSelection.textContent = `${count} episodes`;
    elements.selectAccepted.disabled = app.acceptedSelectionLoading || catalogEpisodeCount() === 0;
    elements.selectAccepted.textContent = app.acceptedSelectionLoading ? "选择中..." : "全部接受";
    elements.selectAccepted.setAttribute("aria-busy", String(app.acceptedSelectionLoading));
    elements.selectAccepted.title = app.acceptedSelectionError
      ? `选择失败：${app.acceptedSelectionError}`
      : "选择当前日期和搜索范围内的全部 success episode";
  }

  async function selectAllAcceptedRecordings() {
    if (app.acceptedSelectionLoading) return;
    cancelAcceptedSelection();
    const controller = new AbortController();
    app.acceptedSelectionController = controller;
    app.acceptedSelectionLoading = true;
    app.acceptedSelectionError = null;
    renderExportSelection();
    try {
      const items = await collectSuccessfulRecordingItems(fetchJson, {
        date: app.dateFilter,
        query: app.searchQuery,
        signal: controller.signal,
      });
      if (controller !== app.acceptedSelectionController) return;
      app.selectedExportIds.clear();
      app.selectedItems.clear();
      for (const item of items) {
        app.selectedExportIds.add(item.id);
        app.selectedItems.set(item.id, item);
      }
      renderList();
    } catch (error) {
      if (error.name === "AbortError") return;
      app.acceptedSelectionError = error.message;
    } finally {
      if (controller === app.acceptedSelectionController) {
        app.acceptedSelectionController = null;
        app.acceptedSelectionLoading = false;
        renderExportSelection();
      }
    }
  }

  function defaultExportName() {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, "0");
    return [
      "selected-",
      now.getFullYear(),
      pad(now.getMonth() + 1),
      pad(now.getDate()),
      "-",
      pad(now.getHours()),
      pad(now.getMinutes()),
      pad(now.getSeconds()),
    ].join("");
  }

  function renderExportJob(job = app.exportJob) {
    app.exportJob = job || null;
    if (!job) {
      elements.exportState.textContent = "待提交";
      elements.exportProgress.hidden = true;
      elements.exportResult.hidden = true;
      elements.exportError.hidden = true;
      elements.exportActivity.hidden = true;
      return;
    }
    const phaseLabel = recordingExportPhaseLabel(job);
    elements.exportState.textContent = phaseLabel;
    elements.exportProgress.hidden = false;
    elements.exportProgressBar.max = Math.max(1, job.episodeCount);
    if (job.phase === "merging") {
      elements.exportProgressBar.removeAttribute("value");
      elements.exportProgressLabel.textContent = `${job.episodeCount} episodes`;
    } else {
      elements.exportProgressBar.value = job.conversionReadyCount;
      elements.exportProgressLabel.textContent = `${job.conversionReadyCount} / ${job.episodeCount}`;
    }
    elements.exportResult.hidden = job.state !== "ready" || !job.datasetRoot;
    elements.exportPath.textContent = job.datasetRoot || "";
    elements.exportError.hidden = !job.error;
    elements.exportError.textContent = job.error || "";
    elements.exportActivity.hidden = false;
    elements.exportActivity.dataset.state = job.state;
    elements.exportActivity.textContent = `LeRobot · ${phaseLabel} · ${job.datasetName || "dataset"}`;
  }

  function openExportDialog() {
    elements.exportName.value = defaultExportName();
    renderExportSelection();
    renderExportJob();
    if (typeof elements.exportDialog.showModal === "function") {
      if (!elements.exportDialog.open) elements.exportDialog.showModal();
    } else {
      elements.exportDialog.setAttribute("open", "");
    }
  }

  function closeExportDialog() {
    if (typeof elements.exportDialog.close === "function") elements.exportDialog.close();
    else elements.exportDialog.removeAttribute("open");
  }

  function scheduleExportPoll() {
    if (app.exportPollTimer) window.clearTimeout(app.exportPollTimer);
    app.exportPollTimer = 0;
    if (!app.exportJob || ["ready", "error"].includes(app.exportJob.state) || app.disposed) return;
    app.exportPollTimer = window.setTimeout(
      () => {
        app.exportPollTimer = 0;
        loadExportState();
      },
      document.visibilityState === "hidden" ? 5000 : EXPORT_POLL_MS,
    );
  }

  async function loadExportState() {
    app.exportController?.abort();
    const controller = new AbortController();
    app.exportController = controller;
    try {
      const state = normalizeRecordingExportState(await fetchJson(
        `${RECORDING_EXPORTS_URL}?limit=20`,
        { signal: controller.signal },
      ));
      if (controller !== app.exportController) return;
      const job = (
        state.jobs.find((item) => item.jobId === app.exportJobId)
        || state.jobs[0]
        || null
      );
      if (job) app.exportJobId = job.jobId;
      renderExportJob(job);
      if (!state.available && !job && state.blockers.length) {
        elements.exportActivity.hidden = false;
        elements.exportActivity.dataset.state = "error";
        elements.exportActivity.textContent = state.blockers[0];
      }
    } catch (error) {
      if (error.name === "AbortError") return;
      if (app.exportJob && !["ready", "error"].includes(app.exportJob.state)) {
        elements.exportActivity.hidden = false;
        elements.exportActivity.dataset.state = "error";
        elements.exportActivity.textContent = `LeRobot · 状态读取失败 · ${error.message}`;
      }
    } finally {
      if (controller === app.exportController) app.exportController = null;
      scheduleExportPoll();
    }
  }

  async function submitRecordingExport() {
    const episodeIds = [...app.selectedExportIds];
    if (!episodeIds.length || app.exportSubmitting) return;
    app.exportSubmitting = true;
    renderExportSelection();
    elements.exportError.hidden = true;
    elements.exportState.textContent = "正在提交";
    try {
      const job = normalizeRecordingExportJob(await fetchJson(RECORDING_EXPORTS_URL, {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({
          episode_ids: episodeIds,
          dataset_name: elements.exportName.value.trim(),
        }),
      }));
      if (!job) throw new Error("录制导出响应格式无效");
      app.exportJobId = job.jobId;
      app.selectedExportIds.clear();
      app.selectedItems.clear();
      renderList();
      renderExportJob(job);
      scheduleExportPoll();
    } catch (error) {
      elements.exportState.textContent = "提交失败";
      elements.exportError.hidden = false;
      elements.exportError.textContent = error.message;
    } finally {
      app.exportSubmitting = false;
      renderExportSelection();
    }
  }

  function closeDeleteDialog() {
    if (typeof elements.deleteDialog.close === "function") elements.deleteDialog.close();
    else elements.deleteDialog.removeAttribute("open");
  }

  function openDeleteDialog() {
    const count = selectedExportCount();
    if (!count || app.deleteSubmitting) return;
    elements.deleteSummary.textContent = `将永久删除选中的 ${count} 个 episode。`;
    elements.deleteError.hidden = true;
    elements.deleteError.textContent = "";
    if (typeof elements.deleteDialog.showModal === "function") {
      if (!elements.deleteDialog.open) elements.deleteDialog.showModal();
    } else {
      elements.deleteDialog.setAttribute("open", "");
    }
  }

  async function submitRecordingDeletion() {
    const episodeIds = [...app.selectedExportIds];
    if (!episodeIds.length || app.deleteSubmitting) return;
    app.deleteSubmitting = true;
    elements.deleteConfirm.disabled = true;
    elements.deleteError.hidden = true;
    renderExportSelection();
    try {
      const payload = await fetchJson(RECORDINGS_URL, {
        method: "DELETE",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({ episode_ids: episodeIds, confirm: true }),
      });
      const deleted = Array.isArray(payload.deleted) ? payload.deleted : episodeIds;
      for (const episodeId of deleted) {
        app.selectedExportIds.delete(episodeId);
        app.selectedItems.delete(episodeId);
      }
      closeDeleteDialog();
      await loadList({ preserveSelection: false, page: app.page });
    } catch (error) {
      elements.deleteError.hidden = false;
      elements.deleteError.textContent = error.message;
    } finally {
      app.deleteSubmitting = false;
      elements.deleteConfirm.disabled = false;
      renderExportSelection();
    }
  }

  function renderList() {
    for (const item of app.items) {
      if (app.selectedExportIds.has(item.id)) app.selectedItems.set(item.id, item);
    }
    const nodes = app.items.map((item) => {
      const row = createElement("div", "episode-row");
      const checkbox = createElement("input", "episode-select");
      checkbox.type = "checkbox";
      checkbox.checked = app.selectedExportIds.has(item.id);
      checkbox.disabled = item.status === "recording";
      checkbox.setAttribute("aria-label", `选择 ${item.episodeName}`);
      checkbox.title = checkbox.disabled ? "正在录制的 episode 不能删除" : "加入批量操作";
      checkbox.addEventListener("change", () => {
        cancelAcceptedSelection();
        app.acceptedSelectionError = null;
        if (checkbox.checked) {
          app.selectedExportIds.add(item.id);
          app.selectedItems.set(item.id, item);
        } else {
          app.selectedExportIds.delete(item.id);
          app.selectedItems.delete(item.id);
        }
        renderExportSelection();
      });
      const button = createElement("button", "episode-button");
      button.type = "button";
      button.role = "option";
      button.dataset.recordingId = item.id;
      button.setAttribute("aria-selected", String(item.id === app.selectedId));
      button.appendChild(createElement("span", "episode-name", item.episodeName));
      const badge = createElement("span", "status-badge", item.result === "unknown" ? item.status : item.result);
      badge.dataset.tone = statusTone(item.status, item.result);
      button.appendChild(badge);
      const metadata = [
        item.date === "legacy" ? "历史" : item.date,
        item.episodeIndex === null ? null : `#${item.episodeIndex}`,
        item.taskName,
        item.updatedAt,
      ].filter(Boolean).join(" · ");
      button.appendChild(createElement("span", "episode-meta", metadata || "无 metadata 摘要"));
      button.addEventListener("click", () => selectRecording(item.id, {
        preserveTrajectory: item.id === app.selectedId,
      }));
      row.append(checkbox, button);
      return row;
    });
    elements.list.replaceChildren(...(nodes.length ? nodes : [createElement("div", "placeholder-row", "没有可检查的录制。")]));
    renderExportSelection();
  }

  function factRow(label, value) {
    const row = document.createElement("div");
    row.append(createElement("dt", "", label), createElement("dd", "", value));
    return row;
  }

  function gripperCurvePath(side, domain) {
    if (!side.available || !side.values.length) return "";
    const timeSpan = Math.max(1, domain.endNs - domain.startNs);
    const valueSpan = Math.max(1e-9, domain.maximum - domain.minimum);
    return side.values.map((value, index) => {
      const x = ((side.timestampsNs[index] - domain.startNs) / timeSpan) * 1000;
      const y = 108 - ((value - domain.minimum) / valueSpan) * 96;
      return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
  }

  function nearestGripperValue(side, timestampNs) {
    if (!side?.available || !side.timestampsNs.length) return null;
    let low = 0;
    let high = side.timestampsNs.length - 1;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (side.timestampsNs[middle] < timestampNs) low = middle + 1;
      else high = middle;
    }
    const previous = Math.max(0, low - 1);
    const index = Math.abs(side.timestampsNs[previous] - timestampNs)
      <= Math.abs(side.timestampsNs[low] - timestampNs)
      ? previous
      : low;
    return side.values[index];
  }

  function renderGripperCurve(curve) {
    app.gripperCurve = curve;
    const availableSides = [curve?.left, curve?.right].filter(
      (side) => side?.available && side.values.length,
    );
    if (!availableSides.length) {
      app.gripperDomain = null;
      elements.gripperChart.dataset.state = "empty";
      elements.gripperLeftPath.setAttribute("d", "");
      elements.gripperRightPath.setAttribute("d", "");
      elements.gripperCursor.setAttribute("x1", "0");
      elements.gripperCursor.setAttribute("x2", "0");
      elements.gripperLeftValue.textContent = "左 —";
      elements.gripperRightValue.textContent = "右 —";
      return;
    }
    const allValues = availableSides.flatMap((side) => [...side.values]);
    let minimum = Math.min(...allValues);
    let maximum = Math.max(...allValues);
    if (maximum - minimum < 1e-9) {
      const padding = Math.max(0.001, Math.abs(maximum) * 0.05);
      minimum -= padding;
      maximum += padding;
    }
    const domain = Object.freeze({
      startNs: Math.min(...availableSides.map((side) => side.timestampsNs[0])),
      endNs: Math.max(...availableSides.map((side) => side.timestampsNs.at(-1))),
      minimum,
      maximum,
    });
    app.gripperDomain = domain;
    elements.gripperChart.dataset.state = "ready";
    elements.gripperLeftPath.setAttribute("d", gripperCurvePath(curve.left, domain));
    elements.gripperRightPath.setAttribute("d", gripperCurvePath(curve.right, domain));
    updateGripperReadout();
  }

  function updateGripperReadout() {
    const domain = app.gripperDomain;
    const curve = app.gripperCurve;
    if (!domain || !curve) return;
    const maximumFrame = Math.max(1, app.totalFrames - 1);
    const ratio = clamp(app.currentFrame / maximumFrame, 0, 1);
    const cursor = ratio * 1000;
    const timestampNs = domain.startNs + ratio * (domain.endNs - domain.startNs);
    elements.gripperCursor.setAttribute("x1", cursor.toFixed(2));
    elements.gripperCursor.setAttribute("x2", cursor.toFixed(2));
    const left = nearestGripperValue(curve.left, timestampNs);
    const right = nearestGripperValue(curve.right, timestampNs);
    elements.gripperLeftValue.textContent = left === null ? "左 —" : `左 ${left.toFixed(4)}`;
    elements.gripperRightValue.textContent = right === null ? "右 —" : `右 ${right.toFixed(4)}`;
  }

  function renderSummary(summary) {
    const duration = summaryDurationSeconds(summary);
    const frameCount = summaryPlaybackFrameCount(summary);
    elements.facts.replaceChildren(
      factRow("格式", summary.dataFormat),
      factRow("状态", summary.status),
      factRow("结果", summary.result),
      factRow("帧数", frameCount ? String(frameCount) : "-"),
      factRow("时长", duration === null ? "-" : formatDuration(duration)),
    );
    const issueNodes = summary.issues.map((issue) => {
      const node = createElement("div", "issue-item");
      node.dataset.severity = issue.severity;
      node.append(createElement("span", "issue-code", issue.code), document.createTextNode(issue.message));
      return node;
    });
    elements.issues.replaceChildren(...(issueNodes.length ? issueNodes : [createElement("p", "muted-copy", "未发现结构问题。")]));
    elements.viewerSource.textContent = summary.episodeId || summary.id;

    const jointRows = [summary.joints.left, summary.joints.right].map((joint) => {
      const row = createElement("div", "health-row");
      row.dataset.available = String(joint.available);
      row.append(
        createElement("span", "health-dot"),
        createElement("span", "health-name", joint.side === "left" ? "左臂 Joint" : "右臂 Joint"),
        createElement("span", "health-detail", joint.available ? `${joint.dimension}D · ${joint.count}` : "不可用"),
      );
      if (joint.topic) row.title = joint.topic;
      return row;
    });
    const tcpRows = [summary.tcp.left, summary.tcp.right].map((tcp) => {
      const row = createElement("div", "health-row");
      row.dataset.available = String(tcp.available);
      row.append(
        createElement("span", "health-dot"),
        createElement("span", "health-name", tcp.side === "left" ? "左臂 TCP" : "右臂 TCP"),
        createElement("span", "health-detail", tcp.available ? `${tcp.dimension}D · ${tcp.count}` : "不可用"),
      );
      if (tcp.topic) row.title = tcp.topic;
      return row;
    });
    const gripperRows = [summary.grippers.left, summary.grippers.right].map((gripper) => {
      const row = createElement("div", "health-row");
      row.dataset.available = String(gripper.available);
      row.append(
        createElement("span", "health-dot"),
        createElement(
          "span",
          "health-name",
          gripper.side === "left" ? "左夹爪" : "右夹爪",
        ),
        createElement(
          "span",
          "health-detail",
          gripper.available ? `1D · ${gripper.count}` : "不可用",
        ),
      );
      if (gripper.topic) row.title = gripper.topic;
      return row;
    });
    elements.jointHealth.replaceChildren(...jointRows, ...tcpRows, ...gripperRows);

    const cameraRows = CAMERA_ROLES.map((role) => {
      const camera = summary.cameraRoles[role];
      const row = createElement("div", "camera-role-card");
      row.dataset.available = String(["healthy", "configured"].includes(camera.status));
      row.dataset.status = camera.status;
      const warning = cameraRoleWarning(camera.reason);
      const heading = createElement("div", "camera-role-card__heading");
      heading.append(
        createElement("span", "health-dot"),
        createElement("span", "health-name", `${CAMERA_ROLE_LABELS[role]} Camera`),
        createElement(
          "span",
          "health-detail",
          `${CAMERA_ROLE_STATUS_LABELS[camera.status]}${warning ? ` · ${warning}` : ""}`,
        ),
      );
      const fields = createElement("div", "camera-role-card__fields");
      for (const [label, value] of [
        ["Device", camera.deviceKey],
        ["Topic", camera.topic],
        ["SHM", camera.shmName],
      ]) {
        const field = createElement("div", "camera-role-card__field");
        field.append(
          createElement("span", "camera-role-card__field-label", label),
          createElement("code", "camera-role-card__field-value", value || "—"),
        );
        fields.appendChild(field);
      }
      row.append(heading, fields);
      row.title = camera.reason || "";
      return row;
    });
    elements.cameraRoleHealth.replaceChildren(...cameraRows);
    elements.topicFormat.textContent = summary.dataFormat;

    const topicRows = summary.topics.map((topic) => {
      const row = createElement("div", "topic-row");
      row.append(
        createElement("span", "topic-name", `${topic.kind === "camera" ? "camera:" : ""}${topic.name}`),
        createElement("span", "topic-count", String(topic.count)),
      );
      return row;
    });
    elements.topicList.replaceChildren(...(topicRows.length ? topicRows : [createElement("div", "placeholder-row", "没有 topic 统计。")]));
    renderGripperCurve(summary.gripperCurve);
    renderDenseSegments(summary.dense.segments, summary.dense.checkpoints, frameCount);
  }

  function renderDenseSegments(segments, checkpoints, totalFrames) {
    if (!totalFrames) {
      elements.denseTrack.replaceChildren();
      delete elements.denseTrack.dataset.truncated;
      return;
    }
    const plan = planDenseTimeline(segments, checkpoints, totalFrames);
    const nodes = plan.segments.map((segment) => {
      const start = clamp(segment.start, 0, totalFrames);
      const end = clamp(segment.end, start, totalFrames);
      const node = createElement("span", "dense-segment");
      node.dataset.label = segment.label;
      node.style.left = `${(start / totalFrames) * 100}%`;
      node.style.width = `${Math.max(0.1, ((end - start) / totalFrames) * 100)}%`;
      node.title = `${segment.label === "accept" ? "接受" : segment.label === "reject" ? "拒绝" : "其他"}片段 · frame [${start}, ${end})`;
      return node;
    });
    plan.checkpoints.forEach((checkpoint) => {
      const node = createElement("span", "dense-checkpoint");
      node.dataset.label = checkpoint.label;
      node.style.left = `${(clamp(checkpoint.frame, 0, totalFrames) / totalFrames) * 100}%`;
      node.title = checkpoint.revision === null
        ? `Checkpoint · frame ${checkpoint.frame}`
        : `Checkpoint r${checkpoint.revision} · frame ${checkpoint.frame}`;
      nodes.push(node);
    });
    elements.denseTrack.replaceChildren(...nodes);
    elements.denseTrack.dataset.truncated = String(plan.truncated);
  }

  function updateReadout(sample = null) {
    const maximum = Math.max(0, app.totalFrames - 1);
    elements.frameReadout.textContent = app.totalFrames
      ? `Frame ${app.currentFrame} / ${maximum}`
      : "Frame - / -";
    const duration = summaryDurationSeconds(app.summary) || 0;
    let elapsed = maximum ? (app.currentFrame / maximum) * duration : 0;
    const rangeStart = app.summary?.timeRangeNs?.start;
    if (sample && rangeStart !== null && rangeStart !== undefined) {
      elapsed = Math.max(0, (sample.timestampNs - rangeStart) / 1e9);
    }
    elements.timeReadout.textContent = `${formatDuration(elapsed)} / ${formatDuration(duration)}`;
    elements.scrubber.value = String(app.currentFrame);
    updateGripperReadout();
  }

  function estimateFps(chunk) {
    if (!chunk || chunk.timestampsNs.length < 2) return app.estimatedFps;
    const durationSeconds = (chunk.timestampsNs.at(-1) - chunk.timestampsNs[0]) / 1e9;
    if (durationSeconds <= 0) return app.estimatedFps;
    const estimate = ((chunk.timestampsNs.length - 1) * chunk.stride) / durationSeconds;
    return Number.isFinite(estimate) ? clamp(estimate, 1, 240) : app.estimatedFps;
  }

  function abortTrajectorySlot(slot) {
    const request = app[slot];
    if (!request) return;
    request.controller.abort();
    app[slot] = null;
  }

  function abortTrajectoryRequests() {
    abortTrajectorySlot("trajectoryRequest");
    abortTrajectorySlot("trajectoryPrefetch");
  }

  function acquireTrajectoryRequest(target, { prefetch = false } = {}) {
    const key = trajectoryPlanKey(target, app.totalFrames);
    if (!prefetch && app.trajectoryPrefetch?.key === key) {
      abortTrajectorySlot("trajectoryRequest");
      const promoted = app.trajectoryPrefetch;
      app.trajectoryPrefetch = null;
      app.trajectoryRequest = promoted;
      promoted.prefetch = false;
      setViewerPreparing();
      return promoted;
    }
    const slot = prefetch ? "trajectoryPrefetch" : "trajectoryRequest";
    const current = app[slot];
    if (current?.key === key && !current.controller.signal.aborted) return current;
    abortTrajectorySlot(slot);
    const request = {
      key,
      prefetch,
      serial: ++app.trajectoryRequestSerial,
      selectionRevision: app.selectionRevision,
      controller: new AbortController(),
    };
    app[slot] = request;
    return request;
  }

  async function loadTrajectoryWindow(target, { prefetch = false } = {}) {
    if (prefetch && app.scrubbing) return null;
    const request = acquireTrajectoryRequest(target, { prefetch });
    if (!request.promise) {
      request.promise = pollCacheOperation(
        () => trajectories.load(
          app.selectedId,
          target,
          app.totalFrames,
          request.controller.signal,
        ),
        {
          signal: request.controller.signal,
          onBuilding: () => {
            if (!request.prefetch) setViewerPreparing();
          },
        },
      ).finally(() => {
        if (app.trajectoryRequest === request) app.trajectoryRequest = null;
        if (app.trajectoryPrefetch === request) app.trajectoryPrefetch = null;
      });
    }
    const chunk = await request.promise;
    if (
      request.controller.signal.aborted
      || request.selectionRevision !== app.selectionRevision
    ) return null;
    return chunk;
  }

  async function loadTcpOverview(signal, revision) {
    const overview = await pollCacheOperation(
      () => trajectories.loadOverview(
        app.selectedId,
        app.totalFrames,
        signal,
      ),
      { signal, onBuilding: setViewerPreparing },
    );
    if (revision !== app.selectionRevision || !viewer) return null;
    viewer.setRecordedTcpTrajectories?.(
      overview.tcp.leftAvailable ? overview.tcp.left : [],
      overview.tcp.rightAvailable ? overview.tcp.right : [],
    );
    return overview;
  }

  function applyPlaybackSample(sample, target) {
    if (!sample || !viewer) return;
    const jointsAvailable = Boolean(
      sample.joints?.leftAvailable && sample.joints?.rightAvailable
    );
    const tcpAvailable = Boolean(
      sample.tcp?.leftAvailable || sample.tcp?.rightAvailable
    );
    viewer.setArmModelsVisible?.(jointsAvailable);
    if (jointsAvailable) {
      viewer.updateJointStates(sample.joints.left, sample.joints.right);
    }
    viewer.updateTcpPoses(
      sample.tcp?.leftAvailable ? sample.tcp.left : null,
      sample.tcp?.rightAvailable ? sample.tcp.right : null,
    );
    viewer.updateRecordedTcpTrails?.(
      sample.tcp?.leftAvailable ? sample.tcpTrail.left : [],
      sample.tcp?.rightAvailable ? sample.tcpTrail.right : [],
    );
    viewer.updateGripperPositions?.({
      left: sample.grippers?.leftAvailable
        ? { value: sample.grippers.left[0], known: true, source: "recording-replay" }
        : { known: false, source: "recording-unavailable" },
      right: sample.grippers?.rightAvailable
        ? { value: sample.grippers.right[0], known: true, source: "recording-replay" }
        : { known: false, source: "recording-unavailable" },
    });
    viewer.status.dataset.state = "replay";
    viewer.status.textContent = jointsAvailable
      ? `URDF + TCP 轨迹 · Frame ${target}`
      : tcpAvailable
        ? `TCP 轨迹 · Frame ${target}`
        : `录制回放 · Frame ${target}`;
  }

  function previewScrubFrame(frame) {
    if (!app.summary || !app.totalFrames) return 0;
    const target = clamp(Math.round(frame), 0, app.totalFrames - 1);
    app.currentFrame = target;
    app.frameFloat = target;
    const sample = trajectories.sample(target);
    if (sample && viewer) {
      applyPlaybackSample(sample, target);
    }
    syncVideosToFrame(target);
    updateReadout(sample);
    return target;
  }

  function scheduleScrubFrame(frame) {
    const target = previewScrubFrame(frame);
    const targetKey = trajectoryPlanKey(target, app.totalFrames);
    if (app.trajectoryRequest?.key !== targetKey) abortTrajectorySlot("trajectoryRequest");
    app.scrubbing = true;
    abortTrajectorySlot("trajectoryPrefetch");
    if (app.scrubTimer) window.clearTimeout(app.scrubTimer);
    app.scrubTimer = window.setTimeout(() => {
      app.scrubTimer = 0;
      app.scrubbing = app.scrubberPointerActive;
      showFrame(target, { allowPrefetch: !app.scrubbing });
    }, SCRUB_DEBOUNCE_MS);
  }

  function finishScrubbing() {
    if (app.scrubTimer) window.clearTimeout(app.scrubTimer);
    app.scrubTimer = 0;
    app.scrubberPointerActive = false;
    app.scrubbing = false;
    showFrame(app.currentFrame, { allowPrefetch: true });
  }

  async function showFrame(
    frame,
    { preserveFloat = false, allowPrefetch = true, syncVideos = !app.playing } = {},
  ) {
    if (!app.summary || !app.totalFrames) return;
    const target = clamp(Math.round(frame), 0, app.totalFrames - 1);
    const targetKey = trajectoryPlanKey(target, app.totalFrames);
    if (app.trajectoryRequest?.key !== targetKey) abortTrajectorySlot("trajectoryRequest");
    app.currentFrame = target;
    if (!preserveFloat) app.frameFloat = target;
    if (syncVideos) syncVideosToFrame(target);
    const revision = ++app.frameRevision;
    let sample = trajectories.sample(target);
    updateReadout(sample);
    if (!sample) {
      try {
        const chunk = await loadTrajectoryWindow(target);
        if (!chunk) return;
        app.estimatedFps = estimateFps(chunk);
        if (revision !== app.frameRevision || target !== app.currentFrame) return;
        sample = trajectories.sample(target);
      } catch (error) {
        if (error.name === "AbortError" || revision !== app.frameRevision) return;
        if (error.cacheState === "failed") {
          setViewerCacheFailed(error.message);
          return;
        }
        setViewerUnavailable(
          error.code === "joint_trajectory_unavailable"
            ? "该 episode 没有可读取的左右臂 joint trajectory；不会使用 TCP 做 IK。"
            : error.message,
        );
        return;
      }
    }
    if (!sample || revision !== app.frameRevision) return;
    if (!viewer) return;
    if (!elements.unavailable.hidden || !viewer.sceneRoot.visible) setViewerAvailable();
    applyPlaybackSample(sample, target);
    updateReadout(sample);
    const nextTarget = target + Math.floor(TRAJECTORY_WINDOW_FRAMES * 0.8);
    if (allowPrefetch && !app.scrubbing && nextTarget < app.totalFrames) {
      const nextKey = trajectoryPlanKey(nextTarget, app.totalFrames);
      if (nextKey !== targetKey && !trajectories.sample(nextTarget)) {
        loadTrajectoryWindow(nextTarget, { prefetch: true }).catch(() => {});
      }
    }
  }

  function animationTick(timestamp) {
    app.animationHandle = 0;
    if (!app.playing || app.disposed || document.visibilityState === "hidden") return;
    const master = masterRecordedVideo();
    if (master && !master.paused) {
      correctVideoDrift();
      app.frameFloat = currentVideoRatio() * Math.max(0, app.totalFrames - 1);
    } else {
      if (app.lastAnimationTime === null) app.lastAnimationTime = timestamp;
      const elapsedMs = Math.max(0, timestamp - app.lastAnimationTime);
      app.frameFloat += (elapsedMs / 1000) * app.estimatedFps * app.speed;
    }
    app.lastAnimationTime = timestamp;
    if (app.frameFloat >= app.totalFrames - 1) {
      app.frameFloat = app.totalFrames - 1;
      showFrame(app.frameFloat, { preserveFloat: true, syncVideos: false });
      stopPlayback();
      return;
    }
    if (timestamp - app.lastRenderedTime >= 1000 / MAX_PLAYBACK_FPS) {
      app.lastRenderedTime = timestamp;
      showFrame(app.frameFloat, { preserveFloat: true, syncVideos: false });
    }
    app.animationHandle = requestAnimationFrame(animationTick);
  }

  function startPlayback() {
    if (elements.play.disabled || app.playing || !app.totalFrames) return;
    if (app.currentFrame >= app.totalFrames - 1) {
      app.currentFrame = 0;
      app.frameFloat = 0;
    }
    app.playing = true;
    app.lastAnimationTime = null;
    syncVideosToFrame(app.currentFrame, { force: true });
    for (const video of Object.values(elements.videos)) {
      if (video.readyState < 1 || !Number.isFinite(video.duration) || video.duration <= 0) continue;
      video.playbackRate = app.speed;
      video.play().catch(() => {});
    }
    elements.play.innerHTML = "&#10074;&#10074;";
    elements.play.title = "暂停";
    elements.play.setAttribute("aria-label", "暂停");
    app.animationHandle = requestAnimationFrame(animationTick);
  }

  async function selectRecording(recordingId, { preserveTrajectory = false } = {}) {
    if (!recordingId) return;
    stopPlayback();
    const sameRecording = recordingId === app.selectedId;
    const keepTrajectory = sameRecording && preserveTrajectory;
    const resumeFrame = keepTrajectory ? app.currentFrame : 0;
    const previousTotalFrames = app.totalFrames;
    app.selectedId = recordingId;
    app.summary = null;
    if (!keepTrajectory) {
      app.totalFrames = 0;
      app.currentFrame = 0;
      app.frameFloat = 0;
    }
    app.selectionRevision += 1;
    const revision = app.selectionRevision;
    app.summaryController?.abort();
    abortTrajectoryRequests();
    if (app.scrubTimer) window.clearTimeout(app.scrubTimer);
    app.scrubTimer = 0;
    app.scrubbing = false;
    app.scrubberPointerActive = false;
    const summaryController = new AbortController();
    app.summaryController = summaryController;
    if (!keepTrajectory) trajectories.reset(recordingId);
    if (!sameRecording) renderList();
    viewer?.clearRecordedTcpTrajectories?.();
    clearRecordedVideos();
    renderGripperCurve(null);
    setViewerUnavailable("正在检查 TCP、joint 与同步视频...");
    elements.viewerSource.textContent = recordingId;
    try {
      const payload = await pollCacheOperation(
        () => fetchJson(`${RECORDINGS_URL}/${encodeURIComponent(recordingId)}/summary`, {
          signal: summaryController.signal,
        }),
        {
          signal: summaryController.signal,
          onBuilding: setViewerPreparing,
        },
      );
      if (revision !== app.selectionRevision) return;
      const summary = normalizeRecordingSummary(payload, recordingId);
      app.summary = summary;
      renderSummary(summary);
      setupRecordedVideos(summary);
      const jointsAvailable = summary.joints.left.available && summary.joints.right.available;
      const tcpAvailable = summary.tcp.left.available || summary.tcp.right.available;
      app.totalFrames = summaryPlaybackFrameCount(summary);
      if (keepTrajectory && trajectories.recordingId !== recordingId) trajectories.reset(recordingId);
      if (keepTrajectory && previousTotalFrames !== app.totalFrames) trajectories.reset(recordingId);
      elements.scrubber.max = String(Math.max(0, app.totalFrames - 1));
      updateReadout();
      if (!jointsAvailable && !tcpAvailable) {
        setViewerUnavailable("该 episode 没有可读取的 TCP 或 joint trajectory。");
        return;
      }
      let activeViewer = await ensureViewer(revision);
      if (!activeViewer && revision === app.selectionRevision) {
        activeViewer = await ensureViewer(revision);
      }
      if (!activeViewer || revision !== app.selectionRevision) return;
      applyRecordedVisualization(summary);
      setViewerAvailable({ jointsAvailable, tcpAvailable });
      if (tcpAvailable) {
        try {
          await loadTcpOverview(summaryController.signal, revision);
        } catch (error) {
          if (error.name === "AbortError" || revision !== app.selectionRevision) return;
          activeViewer.clearRecordedTcpTrajectories?.();
        }
      }
      await showFrame(clamp(resumeFrame, 0, app.totalFrames - 1));
    } catch (error) {
      if (error.name === "AbortError" || revision !== app.selectionRevision) return;
      if (error.cacheState === "failed") {
        setConnection("online", "已连接");
        setViewerCacheFailed(error.message);
        return;
      }
      setConnection("offline", "读取失败");
      setViewerUnavailable(error.message);
    }
  }

  async function loadList({ preserveSelection = true, page = app.page } = {}) {
    stopPlayback();
    cancelAcceptedSelection();
    app.listController?.abort();
    app.summaryController?.abort();
    abortTrajectoryRequests();
    app.listController = new AbortController();
    elements.refresh.disabled = true;
    setConnection("pending", "刷新中");
    try {
      const query = new URLSearchParams({
        page: String(Math.max(1, page)),
        page_size: String(LIST_PAGE_SIZE),
        status: app.statusFilter,
        date: app.dateFilter,
        query: app.searchQuery,
      });
      const list = normalizeRecordingList(await fetchJson(`${RECORDINGS_URL}?${query}`, {
        signal: app.listController.signal,
      }));
      app.items = [...list.items];
      app.dates = [...list.dates];
      app.dateFilter = list.filters.date;
      app.statusFilter = list.filters.status;
      app.searchQuery = list.filters.query;
      app.page = list.page;
      app.totalPages = list.totalPages;
      elements.count.textContent = String(list.total);
      elements.pageReadout.textContent = `${list.page} / ${Math.max(1, list.totalPages)}`;
      elements.pagePrevious.disabled = list.page <= 1;
      elements.pageNext.disabled = list.totalPages <= 0 || list.page >= list.totalPages;
      renderDateFilters();
      elements.rootState.textContent = !list.rootConfigured
        ? "录制根目录未配置"
        : list.rootState === "missing"
          ? "录制目录尚未创建"
        : !list.rootAvailable
          ? "录制目录暂不可用"
          : `${list.total} 个 episode · ${list.cachePersistent ? "持久快读缓存" : "临时缓存"}`;
      setConnection("online", "已连接");
      const previousSelectedId = app.selectedId;
      const selectedExists = app.items.some((item) => item.id === app.selectedId);
      if (!preserveSelection || !selectedExists) app.selectedId = app.items[0]?.id || null;
      renderList();
      if (app.selectedId) {
        await selectRecording(app.selectedId, {
          preserveTrajectory: preserveSelection && previousSelectedId === app.selectedId,
        });
      } else {
        setViewerUnavailable(
          !list.rootConfigured
            ? "录制根目录未配置。"
            : list.rootState === "missing"
              ? "尚无录制；首次录制后会自动创建当日目录。"
            : !list.rootAvailable
              ? "录制目录暂不可用。"
              : "没有可检查的 episode。",
        );
      }
    } catch (error) {
      if (error.name === "AbortError") return;
      setConnection("offline", "连接失败");
      elements.rootState.textContent = error.message;
      elements.list.replaceChildren(createElement("div", "placeholder-row", error.message));
      setViewerUnavailable("无法读取录制列表。");
    } finally {
      elements.refresh.disabled = false;
    }
  }

  elements.refresh.addEventListener("click", () => loadList());
  elements.selectionClear.addEventListener("click", () => {
    cancelAcceptedSelection();
    app.acceptedSelectionError = null;
    app.selectedExportIds.clear();
    app.selectedItems.clear();
    renderList();
  });
  elements.selectAccepted.addEventListener("click", selectAllAcceptedRecordings);
  elements.exportOpen.addEventListener("click", openExportDialog);
  elements.deleteOpen.addEventListener("click", openDeleteDialog);
  elements.deleteClose.addEventListener("click", closeDeleteDialog);
  elements.deleteCancel.addEventListener("click", closeDeleteDialog);
  elements.deleteForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitRecordingDeletion();
  });
  elements.exportActivity.addEventListener("click", openExportDialog);
  elements.exportClose.addEventListener("click", closeExportDialog);
  elements.exportCancel.addEventListener("click", closeExportDialog);
  elements.exportForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitRecordingExport();
  });
  elements.dateFilter.addEventListener("change", () => {
    app.dateFilter = elements.dateFilter.value || "all";
    loadList({ preserveSelection: false, page: 1 });
  });
  elements.statusFilter.addEventListener("change", () => {
    app.statusFilter = elements.statusFilter.value || "all";
    loadList({ preserveSelection: false, page: 1 });
  });
  elements.search.addEventListener("input", () => {
    if (app.searchTimer) window.clearTimeout(app.searchTimer);
    app.searchTimer = window.setTimeout(() => {
      app.searchTimer = 0;
      app.searchQuery = elements.search.value.trim().slice(0, 120);
      loadList({ preserveSelection: false, page: 1 });
    }, FILTER_DEBOUNCE_MS);
  });
  elements.pagePrevious.addEventListener("click", () => {
    if (app.page > 1) loadList({ preserveSelection: false, page: app.page - 1 });
  });
  elements.pageNext.addEventListener("click", () => {
    if (app.page < app.totalPages) loadList({ preserveSelection: false, page: app.page + 1 });
  });
  elements.play.addEventListener("click", () => (app.playing ? stopPlayback() : startPlayback()));
  elements.speed.addEventListener("change", () => {
    const speed = Number(elements.speed.value);
    app.speed = SPEEDS.has(speed) ? speed : 1;
    for (const video of Object.values(elements.videos)) video.playbackRate = app.speed;
  });
  elements.scrubber.addEventListener("pointerdown", () => {
    app.scrubberPointerActive = true;
    app.scrubbing = true;
    abortTrajectorySlot("trajectoryPrefetch");
  });
  elements.scrubber.addEventListener("input", () => {
    stopPlayback();
    scheduleScrubFrame(Number(elements.scrubber.value));
  });
  elements.scrubber.addEventListener("pointerup", finishScrubbing);
  elements.scrubber.addEventListener("pointercancel", finishScrubbing);
  elements.denseTrack.addEventListener("click", (event) => {
    if (!app.totalFrames) return;
    const bounds = elements.denseTrack.getBoundingClientRect();
    const ratio = bounds.width ? clamp((event.clientX - bounds.left) / bounds.width, 0, 1) : 0;
    stopPlayback();
    if (app.scrubTimer) window.clearTimeout(app.scrubTimer);
    app.scrubTimer = 0;
    app.scrubbing = false;
    app.scrubberPointerActive = false;
    abortTrajectorySlot("trajectoryPrefetch");
    showFrame(Math.round(ratio * (app.totalFrames - 1)));
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") stopPlayback();
    else if (app.exportJob && !["ready", "error"].includes(app.exportJob.state)) {
      loadExportState();
    }
  });
  window.addEventListener("pagehide", () => {
    app.disposed = true;
    stopPlayback();
    app.listController?.abort();
    app.acceptedSelectionController?.abort();
    app.summaryController?.abort();
    abortTrajectoryRequests();
    if (app.scrubTimer) window.clearTimeout(app.scrubTimer);
    if (app.searchTimer) window.clearTimeout(app.searchTimer);
    if (app.exportPollTimer) window.clearTimeout(app.exportPollTimer);
    app.exportController?.abort();
    trajectories.reset();
    clearRecordedVideos();
    viewer?.dispose();
  }, { once: true });

  setViewerUnavailable("选择包含 TCP 或 joint topic 的 episode。");
  loadList({ preserveSelection: false });
  loadExportState();
  return Object.freeze({
    app,
    get viewer() { return viewer; },
    trajectories,
    loadList,
    selectRecording,
    showFrame,
    dispose: () => viewer?.dispose(),
  });
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  const start = () => initializeRecordingReplay();
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
}

export {
  CACHE_POLL_DELAYS_MS,
  DENSE_DOM_NODE_LIMIT,
  GRIPPER_DIMENSION,
  JOINT_DIMENSION,
  TCP_DIMENSION,
  MAX_PLAYBACK_FPS,
  SCRUB_DEBOUNCE_MS,
  TrajectoryChunkStore,
  cacheRetryDelay,
  cacheStateFromPayload,
  formatDuration,
  initializeRecordingReplay,
  interpolateTrajectoryFrame,
  normalizeRecordingItem,
  normalizeRecordingList,
  collectSuccessfulRecordingItems,
  normalizeRecordingExportJob,
  normalizeRecordingExportState,
  normalizeRecordingSummary,
  normalizeGripperCurve,
  normalizePlaybackTrajectoryPayload,
  normalizeTrajectoryPayload,
  planDenseTimeline,
  planTrajectoryRequest,
  pollCacheOperation,
  recordingExportPhaseLabel,
  recordingItemExportEligible,
  summaryDurationSeconds,
};
