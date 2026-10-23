import assert from "node:assert/strict";
import test from "node:test";

import {
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
  collectSuccessfulRecordingItems,
  interpolateTrajectoryFrame,
  normalizeRecordingList,
  normalizeRecordingExportJob,
  normalizeRecordingExportState,
  normalizeRecordingSummary,
  normalizePlaybackTrajectoryPayload,
  normalizeTrajectoryPayload,
  planDenseTimeline,
  planTrajectoryRequest,
  pollCacheOperation,
  recordingExportPhaseLabel,
  recordingItemExportEligible,
} from "../static/recording_replay.js";

const RECORDING_ID = "episode_0007";

function vector(offset) {
  return Array.from({ length: JOINT_DIMENSION }, (_, index) => offset + index * 0.1);
}

function summaryFixture(overrides = {}) {
  return {
    id: RECORDING_ID,
    data_format: "NEDF3",
    status: "finalized",
    result: "success",
    episode_index: 7,
    episode_id: "episode-uuid",
    config_name: "recorded_config",
    task_name: "sort chips",
    time_range_ns: { start: 1_000_000_000, end: 3_000_000_000 },
    topics: [{ name: "/lowdim/robot_left_joint", count: 61 }],
    cameras: [{ id: "WebCameraOpenPi_4", count: 60 }],
    camera_roles: {
      head: { device_key: "WebCameraOpenPi_2", topic: "/camera/color/compressed/jpg/webcameraopenpi_2", shm_name: "WebCameraOpenPi_2_data", status: "healthy", count: 60 },
      left: { device_key: "WebCameraOpenPi_0", topic: "/camera/color/compressed/jpg/webcameraopenpi_0", shm_name: "WebCameraOpenPi_0_data", status: "healthy", count: 60 },
      right: { device_key: "WebCameraOpenPi_4", topic: "/camera/color/compressed/jpg/webcameraopenpi_4", shm_name: "WebCameraOpenPi_4_data", status: "healthy", count: 60 },
    },
    joints: {
      left: { available: true, topic: "/lowdim/robot_left_joint", count: 60, dimension: 7 },
      right: { available: true, topic: "/lowdim/robot_right_joint", count: 60, dimension: 7 },
    },
    tcp: {
      left: { available: true, topic: "/lowdim/robot_left", count: 60, dimension: 7 },
      right: { available: true, topic: "/lowdim/robot_right", count: 60, dimension: 7 },
    },
    grippers: {
      left: { available: true, topic: "/lowdim/gripper_left", count: 3, dimension: 1 },
      right: { available: true, topic: "/lowdim/gripper_right", count: 3, dimension: 1 },
    },
    gripper_curve: {
      left: {
        available: true,
        topic: "/lowdim/gripper_left",
        timestamps_ns: [1_000_000_000, 2_000_000_000, 3_000_000_000],
        values: [0.01, 0.04, 0.08],
      },
      right: {
        available: true,
        topic: "/lowdim/gripper_right",
        timestamps_ns: [1_000_000_000, 2_000_000_000, 3_000_000_000],
        values: [0.08, 0.04, 0.01],
      },
    },
    visualization: {
      config_name: "recorded_config",
      source: "recorded-config",
      configured_tools: {
        left: { status: "not-configured", visual: { model: "umi3" } },
        right: { status: "not-configured", visual: { model: "umi3" } },
      },
    },
    videos: {
      head: { available: true, url: `/api/recordings/${RECORDING_ID}/videos/head`, size_bytes: 1000 },
      left: { available: true, url: `/api/recordings/${RECORDING_ID}/videos/left`, size_bytes: 1000 },
      right: { available: true, url: `/api/recordings/${RECORDING_ID}/videos/right`, size_bytes: 1000 },
    },
    dense: {
      available: true,
      segments: [
        { start_frame: 0, end_frame: 20, label: "accept" },
        { start_frame: 20, end_frame: 40, label: "reject" },
      ],
      checkpoints: [{ revision: 2, frame: 40, timestamp_ns: 2_000_000_000, label: "reject" }],
    },
    issues: [],
    ...overrides,
  };
}

function trajectoryFixture(overrides = {}) {
  return {
    id: RECORDING_ID,
    start: 0,
    end: 3,
    stride: 1,
    total_frames: 60,
    returned_frames: 3,
    timestamps_ns: [1_000_000_000, 1_033_333_333, 1_066_666_666],
    left: [vector(0), vector(1), vector(2)],
    right: [vector(10), vector(11), vector(12)],
    joints: {
      available: true,
      left_available: true,
      right_available: true,
      left: [vector(0), vector(1), vector(2)],
      right: [vector(10), vector(11), vector(12)],
    },
    tcp: {
      available: true,
      left_available: true,
      right_available: true,
      left: [vector(20), vector(21), vector(22)],
      right: [vector(30), vector(31), vector(32)],
    },
    grippers: {
      available: true,
      left_available: true,
      right_available: true,
      left: [[0.01], [0.04], [0.08]],
      right: [[0.08], [0.04], [0.01]],
    },
    issues: [],
    ...overrides,
  };
}

test("recording list normalizes a bounded read-only episode summary", () => {
  const list = normalizeRecordingList({
    items: [{
      id: RECORDING_ID,
      name: "episode_0007",
      status: "finalized",
      result: "success",
      episode_index: 7,
      task_name: "sort chips",
      has_metadata: true,
    }],
    page: 1,
    page_size: 50,
    total: 1,
    total_pages: 1,
    root_configured: true,
  });
  assert.equal(list.rootConfigured, true);
  assert.equal(list.rootAvailable, true);
  assert.equal(list.items.length, 1);
  assert.equal(list.items[0].episodeIndex, 7);
  assert.equal(list.items[0].result, "success");
});

test("recording list exposes dated catalog filters and persistent cache state", () => {
  const list = normalizeRecordingList({
    items: [{
      id: "20260716:episode_0007",
      episode_name: "episode_0007",
      date: "20260716",
      status: "accepted",
      result: "success",
    }],
    dates: [
      { id: "20260716", label: "20260716", count: 12, accepted: 10, rejected: 2 },
      { id: "legacy", label: "历史（未分日期）", count: 100 },
    ],
    filters: { date: "20260716", status: "accepted", query: "0007" },
    cache: { persistent: true },
    root_configured: true,
    root_available: true,
  });
  assert.equal(list.items[0].episodeName, "episode_0007");
  assert.equal(list.items[0].date, "20260716");
  assert.equal(list.dates[1].id, "legacy");
  assert.deepEqual(list.filters, {
    date: "20260716",
    status: "accepted",
    query: "0007",
  });
  assert.equal(list.cachePersistent, true);
});

test("recording list preserves old API compatibility and explicit unavailable roots", () => {
  const legacy = normalizeRecordingList({ items: [], root_configured: true });
  assert.equal(legacy.rootAvailable, true);
  const unavailable = normalizeRecordingList({
    items: [],
    root_configured: true,
    root_available: false,
  });
  assert.equal(unavailable.rootConfigured, true);
  assert.equal(unavailable.rootAvailable, false);
  const missing = normalizeRecordingList({
    items: [],
    root_configured: true,
    root_available: false,
    root_state: "missing",
  });
  assert.equal(missing.rootAvailable, true);
  assert.equal(missing.rootState, "missing");
  const unreadable = normalizeRecordingList({
    items: [],
    root_configured: true,
    root_available: true,
    root_state: "unreadable",
  });
  assert.equal(unreadable.rootState, "unreadable");
  assert.equal(unreadable.rootAvailable, false);
});

test("recording export state is bounded and accepted episodes are eligible", () => {
  const job = normalizeRecordingExportJob({
    job_id: "a".repeat(32),
    state: "waiting",
    phase: "waiting_conversion",
    episode_count: 3,
    conversion_ready_count: 2,
    dataset_name: "selected-test",
    dataset_root: "/data/selected-test",
  });
  assert.equal(job.episodeCount, 3);
  assert.equal(job.conversionReadyCount, 2);
  assert.equal(recordingExportPhaseLabel(job), "等待单 episode 转换");
  const state = normalizeRecordingExportState({
    available: true,
    blockers: [],
    jobs: [job && {
      job_id: job.jobId,
      state: job.state,
      phase: job.phase,
      episode_count: job.episodeCount,
      conversion_ready_count: job.conversionReadyCount,
    }],
  });
  assert.equal(state.jobs.length, 1);
  assert.equal(recordingItemExportEligible({ result: "success" }), true);
  assert.equal(recordingItemExportEligible({ result: "failure" }), false);
});

test("accepted bulk selection traverses every page and keeps only unique success episodes", async () => {
  const calls = [];
  const pages = {
    1: {
      items: [
        { id: "20260805:episode_0001", status: "accepted", result: "success" },
        { id: "20260805:episode_0002", status: "rejected", result: "failure" },
      ],
      page: 1,
      page_size: 100,
      total: 4,
      total_pages: 3,
      root_configured: true,
    },
    2: {
      items: [
        { id: "20260805:episode_0003", status: "accepted", result: "success" },
        { id: "20260805:episode_0001", status: "accepted", result: "success" },
      ],
      page: 2,
      page_size: 100,
      total: 4,
      total_pages: 3,
      root_configured: true,
    },
    3: {
      items: [{ id: "20260805:episode_0004", status: "accepted", result: "accepted" }],
      page: 3,
      page_size: 100,
      total: 4,
      total_pages: 3,
      root_configured: true,
    },
  };
  const items = await collectSuccessfulRecordingItems(async (url) => {
    calls.push(url);
    const page = Number(new URL(url, "http://localhost").searchParams.get("page"));
    return pages[page];
  }, { date: "20260805", query: "fold" });
  assert.deepEqual(items.map((item) => item.id), [
    "20260805:episode_0001",
    "20260805:episode_0003",
    "20260805:episode_0004",
  ]);
  assert.equal(calls.length, 3);
  for (const url of calls) {
    const parameters = new URL(url, "http://localhost").searchParams;
    assert.equal(parameters.get("status"), "all");
    assert.equal(parameters.get("result"), "success");
    assert.equal(parameters.get("date"), "20260805");
    assert.equal(parameters.get("query"), "fold");
    assert.equal(parameters.get("page_size"), "100");
  }
});

test("accepted bulk selection requires narrowing a result larger than the export limit", async () => {
  await assert.rejects(
    collectSuccessfulRecordingItems(async () => ({
      items: Array.from({ length: 1001 }, (_, index) => ({
        id: `episode_${String(index).padStart(4, "0")}`,
        status: "accepted",
        result: "success",
      })),
      page: 1,
      page_size: 1001,
      total: 1001,
      total_pages: 1,
      root_configured: true,
    })),
    /超过 1000 条/,
  );
});

test("summary requires genuine 7D joints on both arms", () => {
  const ready = normalizeRecordingSummary(summaryFixture(), RECORDING_ID);
  assert.equal(ready.dataFormat, "NEDF3");
  assert.equal(ready.joints.left.available, true);
  assert.equal(ready.joints.right.available, true);
  assert.equal(ready.tcp.left.available, true);
  assert.equal(ready.tcp.right.dimension, TCP_DIMENSION);
  assert.equal(ready.grippers.left.dimension, GRIPPER_DIMENSION);
  assert.equal(ready.grippers.right.available, true);
  assert.deepEqual(ready.gripperCurve.left.values, [0.01, 0.04, 0.08]);
  assert.equal(ready.configName, "recorded_config");
  assert.equal(ready.visualization.configuredTools.left.visual.model, "umi3");
  assert.equal(ready.videos.head.available, true);
  assert.equal(ready.cameraRoles.head.deviceKey, "WebCameraOpenPi_2");
  assert.equal(ready.cameraRoles.head.topic, "/camera/color/compressed/jpg/webcameraopenpi_2");
  assert.equal(ready.cameraRoles.head.shmName, "WebCameraOpenPi_2_data");
  assert.equal(ready.cameraRoles.left.status, "healthy");
  assert.deepEqual(ready.dense.segments.map((segment) => segment.label), ["accept", "reject"]);
  assert.deepEqual(ready.dense.checkpoints[0], {
    frame: 40,
    revision: 2,
    timestampNs: 2_000_000_000,
    label: "reject",
  });

  const defaultFormat = normalizeRecordingSummary(summaryFixture({ data_format: undefined }), RECORDING_ID);
  assert.equal(defaultFormat.dataFormat, "NEDF3");

  const missing = normalizeRecordingSummary(summaryFixture({
    joints: {
      left: { available: true, topic: "/left", count: 60, dimension: 7 },
      right: { available: true, topic: "/right", count: 60, dimension: 6 },
    },
  }), RECORDING_ID);
  assert.equal(missing.joints.left.available, true);
  assert.equal(missing.joints.right.available, false);

  const missingHead = normalizeRecordingSummary(summaryFixture({
    camera_roles: {
      head: { device_key: "WebCameraOpenPi_2", status: "missing" },
      left: { device_key: "WebCameraOpenPi_0", status: "healthy" },
      right: { device_key: "WebCameraOpenPi_4", status: "black_placeholder" },
    },
  }), RECORDING_ID);
  assert.equal(missingHead.cameraRoles.head.status, "missing");
  assert.equal(missingHead.cameraRoles.head.shmName, null);
  assert.equal(missingHead.cameraRoles.right.status, "black_placeholder");
  assert.deepEqual(
    missingHead.issues.map((issue) => issue.code),
    ["camera_head_missing", "camera_right_black_placeholder"],
  );
});

test("legacy camera mappings keep an explicit visible warning reason", () => {
  const legacy = normalizeRecordingSummary(summaryFixture({
    camera_roles: {
      head: { status: "unknown", reason: "camera mapping missing; LeRobot conversion unavailable" },
      left: { status: "unknown", reason: "camera mapping missing; LeRobot conversion unavailable" },
      right: { status: "unknown", reason: "camera mapping missing; LeRobot conversion unavailable" },
    },
  }), RECORDING_ID);
  assert.match(legacy.cameraRoles.head.reason, /camera mapping missing/);
});

test("trajectory rejects non-7D data and non-monotonic timestamps", () => {
  assert.throws(
    () => normalizeTrajectoryPayload(trajectoryFixture({ left: [[1], [2], [3]] }), RECORDING_ID),
    /真实 7D joint/,
  );
  assert.throws(
    () => normalizeTrajectoryPayload(trajectoryFixture({ timestamps_ns: [1, 1, 2] }), RECORDING_ID),
    /严格递增/,
  );
});

test("trajectory frame interpolation is joint-space only", () => {
  const chunk = normalizeTrajectoryPayload(trajectoryFixture({
    start: 10,
    end: 15,
    stride: 2,
    total_frames: 20,
    timestamps_ns: [100, 200, 300],
  }), RECORDING_ID);
  const sample = interpolateTrajectoryFrame(chunk, 11);
  assert.equal(sample.frame, 11);
  assert.equal(sample.timestampNs, 150);
  assert.equal(sample.left[0], 0.5);
  assert.equal(sample.right[0], 10.5);
  assert.equal(interpolateTrajectoryFrame(chunk, 15), null);
});

test("playback trajectory keeps real joints and TCP on one timeline", () => {
  const chunk = normalizePlaybackTrajectoryPayload(trajectoryFixture({
    start: 10,
    end: 15,
    stride: 2,
    total_frames: 20,
    timestamps_ns: [100, 200, 300],
  }), RECORDING_ID);
  const sample = interpolateTrajectoryFrame(chunk, 11);
  assert.equal(sample.timestampNs, 150);
  assert.equal(sample.joints.left[0], 0.5);
  assert.equal(sample.tcp.left[0], 20.5);
  assert.equal(sample.grippers.left[0], 0.025);
  assert.equal(sample.grippers.right[0], 0.06);
  assert.ok(sample.tcpTrail.left.length >= 1);
});

test("chunk planner uses stride to stay under the point budget", () => {
  assert.deepEqual(
    planTrajectoryRequest(12_345, 100_000, { windowFrames: 10_000, maxPoints: 2_000 }),
    { start: 10_000, end: 20_000, stride: 5, maxPoints: 2_000 },
  );
  assert.equal(MAX_PLAYBACK_FPS, 30);
  assert.equal(SCRUB_DEBOUNCE_MS, 100);
});

test("async replay cache states use bounded retry timing and legacy ready fallback", () => {
  assert.deepEqual([...CACHE_POLL_DELAYS_MS], [250, 500, 1000, 2000]);
  assert.equal(cacheStateFromPayload({}), "ready");
  assert.equal(cacheStateFromPayload({ cache_state: "building" }), "building");
  assert.equal(cacheStateFromPayload({ cache_state: "failed" }), "failed");
  assert.equal(cacheRetryDelay({}, 0), 250);
  assert.equal(cacheRetryDelay({}, 10), 2000);
  assert.equal(cacheRetryDelay({ retry_after_ms: 750 }, 0), 750);
});

test("cache polling aborts before a replaced selection can issue another request", async () => {
  const controller = new AbortController();
  let calls = 0;
  const pending = pollCacheOperation(
    async () => {
      calls += 1;
      return { cache_state: "building", retry_after_ms: 100 };
    },
    {
      signal: controller.signal,
      onBuilding: () => controller.abort(),
    },
  );
  await assert.rejects(pending, (error) => error.name === "AbortError");
  assert.equal(calls, 1);
});

test("dense timeline buckets pathological sidecars below the DOM budget", () => {
  const segments = Array.from({ length: 5000 }, (_, index) => ({
    start: index * 2,
    end: index * 2 + 2,
    label: index % 2 ? "accept" : "reject",
  }));
  const checkpoints = Array.from({ length: 5000 }, (_, frame) => ({
    frame: frame * 2,
    revision: frame,
    label: "accept",
  }));
  const planned = planDenseTimeline(segments, checkpoints, 10_000);
  assert.equal(DENSE_DOM_NODE_LIMIT, 1000);
  assert.equal(planned.truncated, true);
  assert.ok(planned.segments.length + planned.checkpoints.length <= DENSE_DOM_NODE_LIMIT);
  assert.equal(planned.checkpoints[0], checkpoints[0]);
  assert.equal(planned.checkpoints.at(-1), checkpoints.at(-1));
});

test("trajectory store coalesces fetches and serves cached interpolation", async () => {
  const calls = [];
  const store = new TrajectoryChunkStore({
    fetchJson: async (url) => {
      calls.push(url);
      return trajectoryFixture();
    },
  });
  const first = store.load(RECORDING_ID, 1, 60);
  const second = store.load(RECORDING_ID, 1, 60);
  assert.equal(await first, await second);
  assert.equal(calls.length, 1);
  assert.match(calls[0], /start=0/);
  assert.match(calls[0], /max_points=1800/);
  assert.equal(store.sample(1).left[0], 1);
});

test("trajectory store never commits a stale episode response", async () => {
  let release;
  const deferred = new Promise((resolve) => { release = resolve; });
  const store = new TrajectoryChunkStore({ fetchJson: () => deferred });
  const oldRequest = store.load("old-episode", 1, 60);
  store.reset("new-episode");
  release({ ...trajectoryFixture(), id: "old-episode" });
  await oldRequest;
  assert.equal(store.sample(1), null);
  assert.equal(store.recordingId, "new-episode");
});

test("trajectory store surfaces building without caching a partial response", async () => {
  const store = new TrajectoryChunkStore({
    fetchJson: async () => ({
      cache_state: "building",
      retry_after_ms: 250,
      episode_id: RECORDING_ID,
    }),
  });
  await assert.rejects(
    store.load(RECORDING_ID, 1, 60),
    (error) => error.cacheState === "building",
  );
  assert.equal(store.chunks.size, 0);
  assert.equal(store.pending.size, 0);
});
