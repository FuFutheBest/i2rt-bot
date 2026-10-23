/*
 * Local, dependency-free entry point for the dual-arm visualizer.
 *
 * The viewer intentionally lives behind #arm-viewer-root so the device
 * control page can decide where the full-bleed canvas belongs.  All runtime
 * dependencies and robot meshes are served from this repository; no CDN is
 * required.
 */
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { MTLLoader } from "./vendor/MTLLoader.js";
import { OBJLoader } from "./vendor/OBJLoader.js";

// Resolve assets relative to this module rather than the current page URL.
// This keeps `/device-control`, `/`, and an embedded iframe equally valid.
const DEFAULT_ASSET_ROOT = new URL("./assets/flexiv/", import.meta.url).href;
const ARM_MODEL_SPECS = Object.freeze({
  left: Object.freeze({
    variant: "Rizon4",
    urdfUrl: new URL("./assets/flexiv/flexiv_Rizon4_kinematics.urdf", import.meta.url).href,
  }),
  right: Object.freeze({
    variant: "Rizon4R",
    urdfUrl: new URL("./assets/flexiv/flexiv_Rizon4R_kinematics.urdf", import.meta.url).href,
  }),
});
const YAM_MODEL_SPEC = Object.freeze({
  model: "yam",
  jointCount: 6,
  jointNames: Object.freeze(["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]),
  urdfUrl: new URL("./assets/yam/yam_no_gripper.urdf", import.meta.url).href,
  publicUrdfUrl: "/static/assets/yam/yam_no_gripper.urdf",
});
const DEFAULT_ROBOT_STATE_URL = "/api/robots/state";
const DEFAULT_HOME_POSE_CANDIDATES_URL = null;
const JOINT_COUNT = 7;
const DEFAULT_POSE = [0, -0.42, 0, 1.05, 0, 0.6, 0];
const ARM_BASE_Y_OFFSET = 0.2;
const ARM_BASE_POSITIONS = Object.freeze({
  left: Object.freeze([0, ARM_BASE_Y_OFFSET, 0]),
  right: Object.freeze([0, -ARM_BASE_Y_OFFSET, 0]),
});
const TCP_DISPLAY_Y_OFFSETS = Object.freeze({
  left: ARM_BASE_Y_OFFSET,
  right: -ARM_BASE_Y_OFFSET,
});
const GROUND_FLEXIV_POSITION = Object.freeze([0, 0, -0.5]);
const GRID_SCENE_LIFT = 0.002;
const DEFAULT_STATE_POLL_INTERVAL_MS = 100;
const IDLE_STATE_POLL_INTERVAL_MS = 500;
const DEFAULT_HOME_POSE_CANDIDATE_POLL_INTERVAL_MS = 500;
const READY_HOME_POSE_CANDIDATE_POLL_INTERVAL_MS = 5000;
const IDLE_HOME_POSE_CANDIDATE_POLL_INTERVAL_MS = 2000;
const MAX_HOME_POSE_GHOSTS_PER_SIDE = 6;
const HOME_POSE_GHOST_OPACITY = 0.12;
const HOME_POSE_GHOST_POINT_SIZE = 0.009;
const DEFAULT_VIEW_PADDING = 1.22;
const DEFAULT_CAMERA_ELEVATION = 0.1;
const TCP_AXIS_SIZE = 0.14;
const CONFIG_TOOL_AXIS_SIZE = 0.12;
const WORLD_AXIS_SIZE = 0.34;
const MAX_DEVICE_PIXEL_RATIO = 1.5;
const MAX_RENDER_FPS = 30;
const CONTROL_DAMPING_BURST_MS = 1200;
const JOINT_UPDATE_EPSILON_RAD = 1e-4;
const TCP_UPDATE_EPSILON = 1e-5;
const GRIPPER_UPDATE_EPSILON_RAD = 1e-4;
const GRIPPER_DIRECT_SOURCE_TTL_MS = 10000;
const TOOL_JOINT_NEUTRAL_FRACTION = 0.5;
const TOOL_VISUAL_VARIANT_STORAGE_KEY = "superinference.tool-visual-variant";
const TOOL_VISUAL_VARIANT_KEYS = Object.freeze(["soft", "hard"]);
const LEFT_COLOR = new THREE.Color(0x35c7c9);
const RIGHT_COLOR = new THREE.Color(0xf0a45b);
const RECORDED_TCP_TRAIL_POINTS = 256;
const AXIS_COLORS = Object.freeze({ X: "#ff4d4d", Y: "#42d66b", Z: "#4d85ff" });
const HOME_POSE_CANDIDATE_STATUSES = new Set([
  "no_config",
  "unavailable",
  "generating",
  "ready",
  "partial",
  "error",
]);

function numericVectorsNearlyEqual(left, right, epsilon) {
  if (left === right) return true;
  if (!left || !right || left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    if (Math.abs(Number(left[index]) - Number(right[index])) > epsilon) return false;
  }
  return true;
}

function configuredToolSignature(tool) {
  const transform = tool?.transform;
  const asset = tool?.asset;
  const visual = tool?.visual;
  const visualAsset = visual?.asset;
  const visualMount = visual?.mountTransform;
  const articulation = visual?.articulation;
  return [
    tool?.status || "",
    tool?.resolvedToolName || tool?.name || "",
    tool?.resolvedToolSource || "",
    tool?.resolution || "",
    transform?.frame || "",
    ...(transform?.translation || []),
    ...(transform?.quaternion || []),
    asset?.kind || "",
    asset?.url || "",
    asset?.materialUrl || "",
    ...(asset?.scale || []),
    visual?.model || "",
    visual?.selectedVariant || "",
    visual?.mirrorAxis || "",
    visualAsset?.kind || "",
    visualAsset?.url || "",
    visualAsset?.materialUrl || "",
    ...(visualAsset?.scale || []),
    visualMount?.frame || "",
    ...(visualMount?.translation || []),
    ...(visualMount?.quaternion || []),
    articulation?.jointName || "",
    articulation?.jointLowerRad ?? "",
    articulation?.jointUpperRad ?? "",
    articulation?.closedJointRad ?? "",
    articulation?.openJointRad ?? "",
    articulation?.valueMode || "",
    articulation?.armLengthM ?? "",
    articulation?.maxWidthM ?? "",
    articulation?.apexAngleMaxDeg ?? "",
  ].join("|");
}

function toolArticulationSignature(articulation) {
  if (!articulation) return "";
  return [
    articulation.jointName,
    articulation.jointLowerRad,
    articulation.jointUpperRad,
    articulation.closedJointRad,
    articulation.openJointRad,
    articulation.valueMode,
    articulation.armLengthM,
    articulation.maxWidthM,
    articulation.apexAngleMaxDeg,
  ].join("|");
}

function configuredToolVisualTransform(tool) {
  if (tool?.visual?.mountTransform) return tool.visual.mountTransform;
  return tool?.status === "available" ? tool.transform : null;
}

class ToolAssetRequestGate {
  constructor() {
    this._epochs = { left: 0, right: 0 };
  }

  next(side) {
    this._epochs[side] = (this._epochs[side] || 0) + 1;
    return this._epochs[side];
  }

  current(side, epoch) {
    return this._epochs[side] === epoch;
  }

  settle(side, epoch, promise) {
    return Promise.resolve(promise).then(
      (value) => ({ accepted: this.current(side, epoch), value, error: null }),
      (error) => ({ accepted: this.current(side, epoch), value: null, error }),
    );
  }

  invalidate() {
    this.next("left");
    this.next("right");
  }
}

class DirtyRenderLoop {
  constructor({
    render,
    update = () => false,
    canRender = () => true,
    maxFps = MAX_RENDER_FPS,
    requestFrame = (callback) => globalThis.requestAnimationFrame(callback),
    cancelFrame = (handle) => globalThis.cancelAnimationFrame(handle),
    now = () => globalThis.performance.now(),
  }) {
    if (typeof render !== "function") throw new Error("DirtyRenderLoop needs a render callback.");
    this._render = render;
    this._update = update;
    this._canRender = canRender;
    this._requestFrame = requestFrame;
    this._cancelFrame = cancelFrame;
    this._now = now;
    this._minimumFrameIntervalMs = 1000 / Math.max(1, Number(maxFps) || MAX_RENDER_FPS);
    this._frameHandle = null;
    this._dirty = false;
    this._disposed = false;
    this._dampingUntil = 0;
    this._lastRenderTime = Number.NEGATIVE_INFINITY;
    this._lastReason = null;
    this._frameCount = 0;
    this._renderCount = 0;
    this._frameRequestCount = 0;
    this._invalidationCount = 0;
    this._tick = (timestamp) => this._onFrame(timestamp);
  }

  invalidate(reason = "update") {
    if (this._disposed) return;
    this._dirty = true;
    this._lastReason = reason;
    this._invalidationCount += 1;
    this._ensureFrame();
  }

  beginDamping(durationMs = CONTROL_DAMPING_BURST_MS) {
    if (this._disposed) return;
    this._dampingUntil = Math.max(this._dampingUntil, this._now() + Math.max(0, durationMs));
    this.invalidate("controls-damping");
  }

  pause() {
    if (this._frameHandle !== null) this._cancelFrame(this._frameHandle);
    this._frameHandle = null;
    this._dampingUntil = 0;
  }

  resume() {
    if (this._disposed) return;
    if (this._dirty) this._ensureFrame();
  }

  _ensureFrame() {
    if (this._frameHandle !== null || !this._canRender()) return;
    this._frameHandle = this._requestFrame(this._tick);
    this._frameRequestCount += 1;
  }

  _onFrame(timestamp) {
    this._frameHandle = null;
    this._frameCount += 1;
    if (this._disposed || !this._canRender()) return;
    const now = Number.isFinite(timestamp) ? timestamp : this._now();
    if (now - this._lastRenderTime + 0.25 < this._minimumFrameIntervalMs) {
      this._ensureFrame();
      return;
    }

    const dampingActive = now < this._dampingUntil;
    if (dampingActive && this._update()) this._dirty = true;
    if (this._dirty) {
      this._dirty = false;
      this._render();
      this._renderCount += 1;
      this._lastRenderTime = now;
    }
    if (dampingActive) this._ensureFrame();
  }

  snapshot() {
    return Object.freeze({
      frame_count: this._frameCount,
      render_count: this._renderCount,
      frame_request_count: this._frameRequestCount,
      invalidation_count: this._invalidationCount,
      dirty: this._dirty,
      scheduled: this._frameHandle !== null,
      damping: this._now() < this._dampingUntil,
      max_fps: Math.round(1000 / this._minimumFrameIntervalMs),
      last_reason: this._lastReason,
    });
  }

  dispose() {
    if (this._disposed) return;
    this.pause();
    this._disposed = true;
  }
}

const DEG = Math.PI / 180;
const FLEXIV_TO_THREE_QUATERNION = new THREE.Quaternion()
  .setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2)
  .normalize();

function composeFlexivBaseWorldQuaternion() {
  const rotateWorldZFirst = new THREE.Quaternion()
    .setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI);
  const rotateWorldYSecond = new THREE.Quaternion()
    .setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
  // Three.js multiplyQuaternions(a, b) stores a * b. Applied to a vector,
  // b acts first, so qY * qZ means world-Z 180 deg, then world-Y 90 deg.
  return new THREE.Quaternion()
    .multiplyQuaternions(rotateWorldYSecond, rotateWorldZFirst)
    .normalize();
}

function flexivWorldRotationToThree(rotation) {
  // Change coordinates by conjugation: C * R_flexiv * C^-1, where C maps
  // Flexiv (x, y, z) to Three.js (x, z, -y).
  return FLEXIV_TO_THREE_QUATERNION.clone()
    .multiply(rotation)
    .multiply(FLEXIV_TO_THREE_QUATERNION.clone().invert())
    .normalize();
}

function composeYamBaseSceneQuaternion() {
  const rotateWorldZFirst = new THREE.Quaternion()
    .setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI);
  const rotateWorldXSecond = new THREE.Quaternion()
    .setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
  // Keep the original URDF-to-Three coordinate conversion as the innermost
  // transform. The previous implementation replaced it, so relative to the
  // old displayed pose Rx(+90) cancelled Rx(-90) and only Rz(180) remained
  // visible. Applied right-to-left: original alignment, Z 180, then X -90.
  return rotateWorldXSecond
    .multiply(rotateWorldZFirst)
    .multiply(FLEXIV_TO_THREE_QUATERNION)
    .normalize();
}

function flexivFrameRotationToThree(rotation) {
  // A TCP pose quaternion stores the TCP basis vectors in Flexiv world
  // coordinates. Only those world-vector coordinates change here, so the
  // frame orientation is C * R (not C * R * C^-1). The removed trailing
  // C^-1 was equivalent to an unwanted local +90-degree X rotation; omitting
  // it rotates the displayed frame -90 degrees about its own X, making its
  // new local Z axis coincide with the previously displayed local Y axis.
  // URDF placement and configured flange_T_tool transforms are deliberately
  // not part of this absolute live-TCP path.
  return FLEXIV_TO_THREE_QUATERNION.clone()
    .multiply(rotation)
    .normalize();
}

function recordedTcpScenePoints(values, side) {
  if (!Array.isArray(values)) return [];
  const points = [];
  for (const value of values) {
    const pose = normalizedTcpPose(value);
    if (!pose) continue;
    points.push(absoluteTcpPoseToSceneTransform(pose, side).position);
  }
  return points;
}

function replaceStaticLinePoints(line, points) {
  const previous = line.geometry;
  line.geometry = new THREE.BufferGeometry().setFromPoints(points);
  line.geometry.setDrawRange(0, points.length);
  line.visible = points.length > 1;
  previous?.dispose?.();
}

function updateDynamicLinePoints(line, points) {
  const source = points.length > RECORDED_TCP_TRAIL_POINTS
    ? points.slice(points.length - RECORDED_TCP_TRAIL_POINTS)
    : points;
  const attribute = line.geometry.getAttribute("position");
  for (let index = 0; index < source.length; index += 1) {
    attribute.setXYZ(index, source[index].x, source[index].y, source[index].z);
  }
  attribute.needsUpdate = true;
  line.geometry.setDrawRange(0, source.length);
  line.visible = source.length > 1;
}

function createRecordedTcpTrajectoryVisual(side, color) {
  const group = new THREE.Group();
  group.name = `${side}-recorded-tcp-trajectory`;
  const full = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({
      color: color.clone(),
      transparent: true,
      opacity: 0.2,
      depthWrite: false,
    }),
  );
  full.name = `${side}-recorded-tcp-full`;
  full.visible = false;
  const trailGeometry = new THREE.BufferGeometry();
  const trailAttribute = new THREE.BufferAttribute(
    new Float32Array(RECORDED_TCP_TRAIL_POINTS * 3),
    3,
  );
  trailAttribute.setUsage(THREE.DynamicDrawUsage);
  trailGeometry.setAttribute("position", trailAttribute);
  trailGeometry.setDrawRange(0, 0);
  const trail = new THREE.Line(
    trailGeometry,
    new THREE.LineBasicMaterial({
      color: color.clone().offsetHSL(0, 0.08, 0.2),
      transparent: true,
      opacity: 0.98,
      depthWrite: false,
    }),
  );
  trail.name = `${side}-recorded-tcp-trail`;
  trail.visible = false;
  const endpointGeometry = new THREE.SphereGeometry(0.018, 14, 10);
  const start = new THREE.Mesh(
    endpointGeometry,
    new THREE.MeshBasicMaterial({
      color: color.clone(),
      transparent: true,
      opacity: 0.36,
      depthWrite: false,
    }),
  );
  start.name = `${side}-recorded-tcp-start`;
  start.visible = false;
  const end = new THREE.Mesh(
    new THREE.SphereGeometry(0.026, 16, 12),
    new THREE.MeshBasicMaterial({
      color: color.clone(),
      transparent: true,
      opacity: 0.76,
      depthWrite: false,
    }),
  );
  end.name = `${side}-recorded-tcp-end`;
  end.visible = false;
  group.add(full, trail, start, end);
  return { group, full, trail, start, end };
}

function tcpDisplayWorldMatrix(side) {
  const yOffset = TCP_DISPLAY_Y_OFFSETS[side];
  if (!Number.isFinite(yOffset)) return null;
  // The SHM pose is expressed in each robot's local Flexiv world. Display it
  // in the dual-arm station frame by applying T_y * R_z(pi): rotate around Z
  // first, then translate the left/right origin by +/-0.2 m along world Y.
  return new THREE.Matrix4()
    .makeRotationZ(Math.PI)
    .setPosition(0, yOffset, 0);
}

function absoluteTcpPoseToSceneTransform(values, side) {
  const pose = normalizedTcpPose(values);
  const stationFromRobot = tcpDisplayWorldMatrix(side);
  if (!pose || !stationFromRobot) return null;
  const [x, y, z, qw, qx, qy, qz] = pose;
  const robotFromTcp = new THREE.Matrix4().compose(
    new THREE.Vector3(x, y, z),
    new THREE.Quaternion(qx, qy, qz, qw).normalize(),
    new THREE.Vector3(1, 1, 1),
  );
  const stationFromTcp = stationFromRobot.multiply(robotFromTcp);
  const flexivPosition = new THREE.Vector3();
  const flexivRotation = new THREE.Quaternion();
  stationFromTcp.decompose(flexivPosition, flexivRotation, new THREE.Vector3());
  return Object.freeze({
    position: flexivWorldToThreePosition(flexivPosition.toArray()),
    quaternion: flexivFrameRotationToThree(flexivRotation),
  });
}

function armBaseSceneQuaternion() {
  return flexivWorldRotationToThree(composeFlexivBaseWorldQuaternion());
}

function validJointVector(values) {
  return (Array.isArray(values) || ArrayBuffer.isView(values))
    && values.length >= JOINT_COUNT
    && Array.from(values).slice(0, JOINT_COUNT).every((value) => Number.isFinite(Number(value)));
}

function normalizedJointVector(values) {
  return validJointVector(values)
    ? Array.from(values).slice(0, JOINT_COUNT).map(Number)
    : null;
}

function normalizedExactNumericVector(values, length) {
  if (!(Array.isArray(values) || ArrayBuffer.isView(values)) || values.length !== length) return null;
  const result = Array.from(values).map(Number);
  return result.every(Number.isFinite) ? result : null;
}

function normalizedYamState(value) {
  const raw = value && typeof value === "object" ? value : {};
  const configured = raw.configured === true;
  const model = raw.model === YAM_MODEL_SPEC.model ? raw.model : null;
  const rawUrdfUrl = raw.urdf_url ?? raw.urdfUrl;
  const urdfUrl = rawUrdfUrl === YAM_MODEL_SPEC.publicUrdfUrl
    || rawUrdfUrl === YAM_MODEL_SPEC.urdfUrl
    ? YAM_MODEL_SPEC.urdfUrl
    : null;
  const initialJoints = normalizedExactNumericVector(
    raw.initial_joints ?? raw.initialJoints,
    YAM_MODEL_SPEC.jointCount,
  );
  const liveJoints = raw.fresh === true
    ? normalizedExactNumericVector(raw.joints ?? raw.liveJoints, YAM_MODEL_SPEC.jointCount)
    : null;
  const basePositionWorld = normalizedExactNumericVector(
    raw.base_position_world ?? raw.basePositionWorld,
    3,
  ) || [0, 0, 0.2];
  return Object.freeze({
    configured: configured && model === YAM_MODEL_SPEC.model && Boolean(urdfUrl),
    model,
    status: typeof raw.status === "string" ? raw.status.slice(0, 64) : "not-configured",
    urdfUrl,
    basePositionWorld: Object.freeze(basePositionWorld),
    initialJoints: initialJoints ? Object.freeze(initialJoints) : null,
    liveJoints: liveJoints ? Object.freeze(liveJoints) : null,
    fresh: Boolean(liveJoints),
    source: liveJoints ? "live" : initialJoints ? "initpos" : "unavailable",
    timestamp: raw.timestamp_ns ?? raw.timestamp ?? null,
  });
}

function validTcpPose(values) {
  if (!(Array.isArray(values) || ArrayBuffer.isView(values)) || values.length < 7) return false;
  const pose = Array.from(values).slice(0, 7).map(Number);
  if (!pose.every(Number.isFinite)) return false;
  const quaternionNormSq = pose.slice(3, 7).reduce((sum, value) => sum + value * value, 0);
  return quaternionNormSq > 1e-12;
}

function normalizedTcpPose(values) {
  return validTcpPose(values) ? Array.from(values).slice(0, 7).map(Number) : null;
}

function normalizedToolName(value) {
  const name = typeof value === "string" ? value.trim() : "";
  return name || null;
}

const EMPTY_CONFIGURED_TOOL = Object.freeze({
  name: null,
  resolvedToolName: null,
  resolvedToolSource: null,
  resolution: "none",
  resolutionWarning: null,
  activeElementCount: 0,
  status: "not-configured",
  transform: null,
  asset: null,
  visual: null,
  source: null,
  reason: "rizon-tool-not-configured",
});

function normalizedToolAssetPath(value, extensions) {
  if (typeof value !== "string") return null;
  const path = value.trim();
  if (!path.startsWith("/static/") || /[%:\\?#]/.test(path)) return null;
  if (path.split("/").some((part) => part === "..")) return null;
  const lower = path.toLowerCase();
  return extensions.some((extension) => lower.endsWith(extension)) ? path : null;
}

function normalizedToolAsset(value) {
  if (!value || typeof value !== "object") return null;
  const kind = value.kind === "urdf"
    ? "urdf"
    : value.kind === "obj"
      ? "obj"
      : value.kind === "stl"
        ? "stl"
        : null;
  if (!kind) return null;
  const extensions = kind === "urdf" ? [".urdf"] : kind === "obj" ? [".obj"] : [".stl"];
  const url = normalizedToolAssetPath(value.url, extensions);
  if (!url) return null;
  const materialUrl = value.material_url ?? value.materialUrl;
  const normalizedMaterialUrl = materialUrl == null
    ? null
    : normalizedToolAssetPath(materialUrl, [".mtl"]);
  if (materialUrl != null && (kind !== "obj" || !normalizedMaterialUrl)) return null;
  const scale = Array.isArray(value.scale) && value.scale.length === 3
    ? value.scale.map(Number)
    : [1, 1, 1];
  if (!scale.every((component) => Number.isFinite(component) && component > 0 && component <= 1000)) {
    return null;
  }
  return Object.freeze({
    kind,
    url,
    materialUrl: normalizedMaterialUrl,
    scale: Object.freeze(scale),
    source: typeof value.source === "string" ? value.source.slice(0, 256) : null,
  });
}

function normalizedToolVariantKey(value) {
  const key = typeof value === "string" ? value.trim().toLowerCase() : "";
  return /^[a-z][a-z0-9_-]{0,31}$/.test(key) ? key : null;
}

function normalizedToolVariants(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return Object.freeze({});
  const variants = {};
  for (const [rawKey, rawAsset] of Object.entries(value).slice(0, 8)) {
    const key = normalizedToolVariantKey(rawKey);
    const asset = normalizedToolAsset(rawAsset?.asset ?? rawAsset);
    if (key && asset) variants[key] = asset;
  }
  return Object.freeze(variants);
}

function storedToolVisualVariant() {
  if (typeof window === "undefined") return "soft";
  try {
    const value = normalizedToolVariantKey(window.localStorage?.getItem(TOOL_VISUAL_VARIANT_STORAGE_KEY));
    return TOOL_VISUAL_VARIANT_KEYS.includes(value) ? value : "soft";
  } catch (_error) {
    return "soft";
  }
}

function storeToolVisualVariant(value) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage?.setItem(TOOL_VISUAL_VARIANT_STORAGE_KEY, value);
  } catch (_error) {
    // Storage can be disabled without affecting the current viewer session.
  }
}

function normalizedToolTransform(value) {
  if (!value || typeof value !== "object" || value.frame !== "flange") return null;
  const quaternionOrder = value.quaternion_order ?? value.quaternionOrder;
  if (quaternionOrder !== "wxyz") return null;
  if (!Array.isArray(value.translation) || value.translation.length !== 3) return null;
  if (!Array.isArray(value.quaternion) || value.quaternion.length !== 4) return null;
  const translation = value.translation.map(Number);
  const quaternion = value.quaternion.map(Number);
  if (![...translation, ...quaternion].every(Number.isFinite)) return null;
  const norm = Math.hypot(...quaternion);
  if (!Number.isFinite(norm) || norm <= 1e-12) return null;
  return Object.freeze({
    frame: "flange",
    translation: Object.freeze(translation),
    quaternion: Object.freeze(quaternion.map((component) => component / norm)),
    quaternionOrder: "wxyz",
  });
}

function normalizedToolArticulation(value) {
  if (!value || typeof value !== "object") return null;
  const jointName = typeof (value.joint_name ?? value.jointName) === "string"
    ? String(value.joint_name ?? value.jointName).trim()
    : "";
  if (!/^[A-Za-z_][A-Za-z0-9_.-]{0,127}$/.test(jointName)) return null;
  const jointLowerRad = Number(
    value.joint_lower_rad ?? value.jointLowerRad ?? value.lower_rad ?? value.lowerRad,
  );
  const jointUpperRad = Number(
    value.joint_upper_rad ?? value.jointUpperRad ?? value.upper_rad ?? value.upperRad,
  );
  const closedJointRad = Number(
    value.closed_joint_rad ?? value.closedJointRad ?? jointLowerRad,
  );
  const valueMode = String(value.value_mode ?? value.valueMode ?? "").trim().toLowerCase();
  if (
    !Number.isFinite(jointLowerRad)
    || !Number.isFinite(jointUpperRad)
    || !Number.isFinite(closedJointRad)
    || jointUpperRad <= jointLowerRad
    || closedJointRad < jointLowerRad
    || closedJointRad > jointUpperRad
  ) return null;
  if (valueMode !== "normalized" && valueMode !== "width_m") return null;

  let armLengthM = null;
  let maxWidthM = null;
  let apexAngleMaxDeg = null;
  if (valueMode === "width_m") {
    armLengthM = Number(value.arm_length_m ?? value.armLengthM);
    maxWidthM = Number(value.max_width_m ?? value.maxWidthM);
    apexAngleMaxDeg = Number(value.apex_angle_max_deg ?? value.apexAngleMaxDeg);
    if (
      !Number.isFinite(armLengthM)
      || !Number.isFinite(maxWidthM)
      || !Number.isFinite(apexAngleMaxDeg)
      || armLengthM <= 0
      || maxWidthM <= 0
      || maxWidthM > 2 * armLengthM + 1e-9
      || apexAngleMaxDeg <= 0
      || apexAngleMaxDeg > 180
    ) return null;
  }
  return Object.freeze({
    jointName,
    jointLowerRad,
    jointUpperRad,
    closedJointRad,
    openJointRad: Math.abs(jointUpperRad - closedJointRad) >= Math.abs(jointLowerRad - closedJointRad)
      ? jointUpperRad
      : jointLowerRad,
    valueMode,
    armLengthM,
    maxWidthM,
    apexAngleMaxDeg,
  });
}

function normalizedToolVisual(value) {
  if (!value || typeof value !== "object") return null;
  // Raw API descriptors keep kind/url on `visual`; normalized descriptors
  // keep the same asset under `visual.asset`. Accept both so repeated
  // normalization (robotStateFromPayload -> updateConfiguredTools) is lossless.
  const asset = normalizedToolAsset(value.asset ?? value);
  const mountTransform = normalizedToolTransform(
    value.mount_transform ?? value.mountTransform ?? value.mount,
  );
  const articulation = normalizedToolArticulation(value.articulation);
  if (!asset || !mountTransform || !articulation) return null;
  const model = typeof value.model === "string" ? value.model.trim().slice(0, 64) : "";
  const variants = normalizedToolVariants(value.variants);
  const requestedDefault = normalizedToolVariantKey(
    value.default_variant ?? value.defaultVariant,
  );
  const defaultVariant = requestedDefault && variants[requestedDefault]
    ? requestedDefault
    : Object.keys(variants)[0] || null;
  const requestedSelected = normalizedToolVariantKey(
    value.selected_variant ?? value.selectedVariant,
  );
  const selectedVariant = requestedSelected && variants[requestedSelected]
    ? requestedSelected
    : defaultVariant;
  const mirrorAxisValue = value.mirror_axis ?? value.mirrorAxis;
  const mirrorAxis = mirrorAxisValue === "y" ? "y" : null;
  return Object.freeze({
    model: model || null,
    asset: selectedVariant ? variants[selectedVariant] : asset,
    variants,
    defaultVariant,
    selectedVariant,
    mirrorAxis,
    mountTransform,
    articulation,
    source: typeof value.source === "string" ? value.source.slice(0, 256) : null,
    mountSource: typeof (value.mount_source ?? value.mountSource) === "string"
      ? String(value.mount_source ?? value.mountSource).slice(0, 256)
      : null,
  });
}

function selectConfiguredToolVariant(tool, preferredVariant) {
  const visual = tool.visual;
  if (!visual || !Object.keys(visual.variants).length) return tool;
  const preferred = normalizedToolVariantKey(preferredVariant);
  const selectedVariant = preferred && visual.variants[preferred]
    ? preferred
    : visual.defaultVariant;
  if (!selectedVariant || !visual.variants[selectedVariant]) return tool;
  if (visual.selectedVariant === selectedVariant && visual.asset === visual.variants[selectedVariant]) {
    return tool;
  }
  return Object.freeze({
    ...tool,
    visual: Object.freeze({
      ...visual,
      asset: visual.variants[selectedVariant],
      selectedVariant,
    }),
  });
}

function configuredToolWithVariant(value, preferredVariant) {
  return selectConfiguredToolVariant(normalizedConfiguredTool(value), preferredVariant);
}

function gripperJointAngleFromValue(value, articulation, inputMode = "raw") {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || !articulation) return null;
  let jointFraction = null;
  if (articulation.valueMode === "normalized") {
    jointFraction = THREE.MathUtils.clamp(
      inputMode === "percent" ? numeric / 100 : numeric,
      0,
      1,
    );
  } else if (articulation.valueMode === "width_m") {
    const apexMaxRad = THREE.MathUtils.degToRad(articulation.apexAngleMaxDeg);
    let sineArgument = null;
    if (inputMode === "percent") {
      const widthFraction = THREE.MathUtils.clamp(numeric / 100, 0, 1);
      sineArgument = widthFraction * Math.sin(apexMaxRad / 2);
    } else {
      if (numeric < 0 || numeric > articulation.maxWidthM) return null;
      sineArgument = numeric / (2 * articulation.armLengthM);
    }
    const apex = 2 * Math.asin(THREE.MathUtils.clamp(sineArgument, 0, 1));
    jointFraction = THREE.MathUtils.clamp(apex / apexMaxRad, 0, 1);
  }
  if (!Number.isFinite(jointFraction)) return null;
  return articulation.closedJointRad
    + jointFraction * (articulation.openJointRad - articulation.closedJointRad);
}

function normalizedGripperPosition(value, articulation, defaultSource = "unknown") {
  const objectValue = value && typeof value === "object" ? value : null;
  const explicitKnown = objectValue?.known;
  const source = typeof objectValue?.source === "string"
    ? objectValue.source.slice(0, 64)
    : defaultSource;
  if (explicitKnown === false) {
    return Object.freeze({ known: false, rawValue: null, actualPercent: null, jointAngle: null, source });
  }
  const existingJointAngleValue = objectValue?.jointAngle ?? objectValue?.joint_angle;
  const existingJointAngle = Number(existingJointAngleValue);
  if (existingJointAngleValue != null && Number.isFinite(existingJointAngle) && articulation) {
    const rawValueCandidate = objectValue?.rawValue ?? objectValue?.raw_value;
    const actualPercentCandidate = objectValue?.actualPercent ?? objectValue?.actual_percent;
    const jointAngle = THREE.MathUtils.clamp(
      existingJointAngle,
      articulation.jointLowerRad,
      articulation.jointUpperRad,
    );
    return Object.freeze({
      known: true,
      rawValue: rawValueCandidate != null && Number.isFinite(Number(rawValueCandidate))
        ? Number(rawValueCandidate)
        : null,
      actualPercent: actualPercentCandidate != null && Number.isFinite(Number(actualPercentCandidate))
        ? THREE.MathUtils.clamp(Number(actualPercentCandidate), 0, 100)
        : null,
      jointAngle,
      source,
    });
  }
  const percentCandidate = objectValue
    ? objectValue.actual_percent ?? objectValue.actualPercent ?? objectValue.percent
    : undefined;
  const usesPercent = percentCandidate !== undefined && percentCandidate !== null;
  const rawCandidate = usesPercent
    ? percentCandidate
    : objectValue
      ? objectValue.value ?? objectValue.raw_value ?? objectValue.rawValue ?? objectValue.position
      : value;
  if (rawCandidate === undefined || rawCandidate === null || rawCandidate === "") {
    return Object.freeze({ known: false, rawValue: null, actualPercent: null, jointAngle: null, source });
  }
  const numeric = Number(rawCandidate);
  if (!Number.isFinite(numeric)) {
    return Object.freeze({ known: false, rawValue: null, actualPercent: null, jointAngle: null, source });
  }
  const jointAngle = gripperJointAngleFromValue(
    numeric,
    articulation,
    usesPercent ? "percent" : "raw",
  );
  if (!Number.isFinite(jointAngle)) {
    return Object.freeze({ known: false, rawValue: numeric, actualPercent: null, jointAngle: null, source });
  }
  return Object.freeze({
    known: explicitKnown !== false,
    rawValue: usesPercent ? null : numeric,
    actualPercent: usesPercent ? THREE.MathUtils.clamp(numeric, 0, 100) : null,
    jointAngle,
    source,
  });
}

function normalizedConfiguredTool(value) {
  if (!value || typeof value !== "object") return EMPTY_CONFIGURED_TOOL;
  const resolvedToolName = normalizedToolName(value.resolved_tool_name ?? value.resolvedToolName ?? value.name);
  const transform = normalizedToolTransform(value.transform);
  const asset = normalizedToolAsset(value.asset);
  const visual = normalizedToolVisual(value.visual);
  let status = typeof value.status === "string" ? value.status : "invalid-config";
  if (status === "available" && !transform) status = "invalid-transform";
  if (status !== "available" && transform) status = "available";
  const activeElementCount = Number(value.active_element_count ?? value.activeElementCount);
  return Object.freeze({
    name: resolvedToolName,
    resolvedToolName,
    resolvedToolSource: typeof (value.resolved_tool_source ?? value.resolvedToolSource) === "string"
      ? (value.resolved_tool_source ?? value.resolvedToolSource).slice(0, 256)
      : null,
    resolution: typeof value.resolution === "string" ? value.resolution.slice(0, 64) : "none",
    resolutionWarning: typeof (value.resolution_warning ?? value.resolutionWarning) === "string"
      ? (value.resolution_warning ?? value.resolutionWarning).slice(0, 512)
      : null,
    activeElementCount: Number.isInteger(activeElementCount) && activeElementCount >= 0
      ? activeElementCount
      : 0,
    status,
    transform,
    asset,
    visual,
    source: typeof value.source === "string" ? value.source : null,
    reason: typeof value.reason === "string" ? value.reason : null,
  });
}

function normalizedExactJointVector(values) {
  if (!(Array.isArray(values) || ArrayBuffer.isView(values)) || values.length !== JOINT_COUNT) {
    return null;
  }
  const joints = Array.from(values).map(Number);
  return joints.every(Number.isFinite) ? joints : null;
}

function normalizedCandidatePose(value) {
  if (!value || typeof value !== "object") return null;
  if (!Array.isArray(value.translation) || value.translation.length !== 3) return null;
  if (!Array.isArray(value.quaternion_wxyz) || value.quaternion_wxyz.length !== 4) return null;
  const translation = value.translation.map(Number);
  const quaternion = value.quaternion_wxyz.map(Number);
  if (![...translation, ...quaternion].every(Number.isFinite)) return null;
  const norm = Math.hypot(...quaternion);
  if (!Number.isFinite(norm) || norm <= 1e-12) return null;
  return Object.freeze({
    translation: Object.freeze(translation),
    quaternionWxyz: Object.freeze(quaternion.map((component) => component / norm)),
  });
}

function normalizedHomePoseCandidate(value, side, index) {
  if (!value || typeof value !== "object") return null;
  const joints = normalizedExactJointVector(value.joints);
  if (!joints) return null;
  const rawId = typeof value.id === "string" ? value.id.trim() : "";
  const residual = value.residual && typeof value.residual === "object" ? value.residual : {};
  const positionResidual = Number(residual.position_m);
  const orientationResidual = Number(residual.orientation_rad);
  const jointDistance = Number(value.joint_distance);
  return Object.freeze({
    id: rawId && rawId.length <= 96 ? rawId : `${side}-${String(index + 1).padStart(2, "0")}`,
    joints: Object.freeze(joints),
    fkPose: normalizedCandidatePose(value.fk_pose),
    residual: Object.freeze({
      positionM: Number.isFinite(positionResidual) ? positionResidual : null,
      orientationRad: Number.isFinite(orientationResidual) ? orientationResidual : null,
    }),
    jointDistance: Number.isFinite(jointDistance) ? jointDistance : null,
    collisionChecked: value.collision_checked === true,
    source: typeof value.source === "string" ? value.source.slice(0, 64) : null,
  });
}

function emptyHomePoseCandidateArm(side, status = "unavailable") {
  return Object.freeze({
    model: ARM_MODEL_SPECS[side].variant,
    status,
    targetFrame: "flange",
    candidates: Object.freeze([]),
    error: null,
  });
}

function normalizedHomePoseCandidateArm(value, side, allowCandidates) {
  if (!value || typeof value !== "object") return emptyHomePoseCandidateArm(side);
  const expectedModel = ARM_MODEL_SPECS[side].variant;
  const model = typeof value.model === "string" ? value.model : "";
  const status = typeof value.status === "string" ? value.status : "unavailable";
  const targetFrame = typeof value.target_frame === "string" ? value.target_frame : "flange";
  const candidates = [];
  if (allowCandidates && status === "ready" && model === expectedModel && Array.isArray(value.candidates)) {
    for (const item of value.candidates) {
      const normalized = normalizedHomePoseCandidate(item, side, candidates.length);
      if (normalized) candidates.push(normalized);
      if (candidates.length >= MAX_HOME_POSE_GHOSTS_PER_SIDE) break;
    }
  }
  return Object.freeze({
    model: model || expectedModel,
    status: model && model !== expectedModel ? "model-mismatch" : status,
    targetFrame,
    candidates: Object.freeze(candidates),
    error: typeof value.error === "string" ? value.error.slice(0, 256) : null,
  });
}

function emptyHomePoseCandidates(status = "unavailable") {
  return Object.freeze({
    schemaVersion: 1,
    status,
    key: null,
    configName: null,
    params: Object.freeze({ preset: null, count: null, seed: null }),
    arms: Object.freeze({
      left: emptyHomePoseCandidateArm("left", status),
      right: emptyHomePoseCandidateArm("right", status),
    }),
  });
}

function normalizedHomePoseCandidates(payload) {
  if (!payload || typeof payload !== "object" || Number(payload.schema_version) !== 1) {
    return emptyHomePoseCandidates();
  }
  const rawStatus = typeof payload.status === "string" ? payload.status : "unavailable";
  const status = HOME_POSE_CANDIDATE_STATUSES.has(rawStatus) ? rawStatus : "unavailable";
  const rawKey = typeof payload.key === "string" ? payload.key.toLowerCase() : "";
  const key = /^[0-9a-f]{64}$/.test(rawKey) ? rawKey : null;
  const renderable = (status === "ready" || status === "partial") && Boolean(key);
  const rawParams = payload.params && typeof payload.params === "object" ? payload.params : {};
  const count = Number(rawParams.count);
  const seed = Number(rawParams.seed);
  const arms = payload.arms && typeof payload.arms === "object" ? payload.arms : {};
  return Object.freeze({
    schemaVersion: 1,
    status: renderable ? status : status === "ready" || status === "partial" ? "unavailable" : status,
    key,
    configName: typeof payload.config_name === "string" ? payload.config_name.slice(0, 160) : null,
    params: Object.freeze({
      preset: typeof rawParams.preset === "string" ? rawParams.preset.slice(0, 32) : null,
      count: Number.isInteger(count) ? count : null,
      seed: Number.isInteger(seed) ? seed : null,
    }),
    arms: Object.freeze({
      left: normalizedHomePoseCandidateArm(arms.left, "left", renderable),
      right: normalizedHomePoseCandidateArm(arms.right, "right", renderable),
    }),
  });
}

function robotStateFromPayload(payload) {
  if (!payload || typeof payload !== "object") return null;
  const positions = payload.joints || payload.joint_positions || payload;
  const globallyFresh = payload.status === "ok"
    && payload.fresh !== false
    && payload.metadata?.fresh !== false;
  const liveLeft = globallyFresh ? normalizedJointVector(positions.left) : null;
  const liveRight = globallyFresh ? normalizedJointVector(positions.right) : null;
  const home = payload.home && typeof payload.home === "object" ? payload.home : {};
  const homeLeft = normalizedJointVector(home.joints_left);
  const homeRight = normalizedJointVector(home.joints_right);
  const tcpPoses = payload.tcp_poses || payload.tcp || payload.tool_poses || {};
  const toolNames = payload.tool_names || payload.active_tools || payload.tools || {};
  const configuredTools = payload.configured_tools && typeof payload.configured_tools === "object"
    ? payload.configured_tools
    : {};
  const normalizedTools = {
    left: normalizedConfiguredTool(configuredTools.left),
    right: normalizedConfiguredTool(configuredTools.right),
  };
  const gripperPositions = payload.gripper_positions && typeof payload.gripper_positions === "object"
    ? payload.gripper_positions
    : {};
  const unknownGripper = (source) => normalizedGripperPosition(
    { known: false, source },
    null,
    source,
  );
  return {
    // Each arm independently prefers a fresh SHM vector.  An absent/stale arm
    // uses the selected config's home pose without changing that config.
    left: liveLeft || homeLeft || [...DEFAULT_POSE],
    right: liveRight || homeRight || [...DEFAULT_POSE],
    home: { left: homeLeft, right: homeRight },
    fresh: { left: Boolean(liveLeft), right: Boolean(liveRight) },
    sources: {
      left: liveLeft ? "live" : homeLeft ? "home" : "default",
      right: liveRight ? "live" : homeRight ? "home" : "default",
    },
    tcpPoses: {
      left: liveLeft ? normalizedTcpPose(tcpPoses.left ?? payload.left_tcp_pose) : null,
      right: liveRight ? normalizedTcpPose(tcpPoses.right ?? payload.right_tcp_pose) : null,
    },
    toolNames: {
      left: normalizedToolName(toolNames.left ?? payload.left_tool_name ?? payload.tool_name),
      right: normalizedToolName(toolNames.right ?? payload.right_tool_name ?? payload.tool_name),
    },
    configuredTools: {
      left: normalizedTools.left,
      right: normalizedTools.right,
    },
    gripperPositions: {
      left: globallyFresh
        ? normalizedGripperPosition(
          gripperPositions.left,
          normalizedTools.left.visual?.articulation,
          "robot-state",
        )
        : unknownGripper("robot-state-stale"),
      right: globallyFresh
        ? normalizedGripperPosition(
          gripperPositions.right,
          normalizedTools.right.visual?.articulation,
          "robot-state",
        )
        : unknownGripper("robot-state-stale"),
    },
    yam: normalizedYamState(payload.yam_head ?? payload.yam),
    timestamp: payload.timestamp_ns ?? payload.timestamp ?? null,
  };
}

function configuredToolStatusSummary(tools, liveTcpVisible = false) {
  const entries = ["left", "right"].map((side) => ({
    side,
    tool: normalizedConfiguredTool(tools?.[side]),
  }));
  const available = entries.filter(({ tool }) => tool.status === "available" && tool.transform);
  if (available.length) {
    const names = [...new Set(available.map(({ tool }) => tool.name).filter(Boolean))];
    const label = names.length ? names.join(" / ") : "configured";
    const prefix = available.some(({ tool }) => tool.resolution === "elements-active-tool")
      ? "Elements Tool"
      : "Config Tool";
    return liveTcpVisible
      ? `${prefix} ${label}（实时 TCP 坐标轴覆盖）`
      : `${prefix} ${label} · flange→tool 坐标轴`;
  }
  if (entries.some(({ tool }) => tool.status === "controller-active-unknown")) {
    return "Tool: 控制器活动工具未知（config tool_name 为空）";
  }
  const knownNames = [...new Set(entries
    .filter(({ tool }) => tool.status === "name-known-pose-unknown")
    .map(({ tool }) => tool.name)
    .filter(Boolean))];
  if (knownNames.length) {
    return `Tool ${knownNames.join(" / ")}：config 未提供 flange→tool 位姿`;
  }
  if (entries.some(({ tool }) => tool.status === "invalid-transform")) {
    return "Tool: config 中的 flange→tool 位姿无效";
  }
  if (entries.some(({ tool }) => tool.status === "invalid-config")) {
    return "Tool: config 选择无效";
  }
  return "";
}

function createAxisLabel(text, color, worldSize) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.beginPath();
  context.arc(64, 64, 43, 0, Math.PI * 2);
  context.fillStyle = "rgba(9, 16, 24, 0.9)";
  context.fill();
  context.lineWidth = 8;
  context.strokeStyle = color;
  context.stroke();
  context.fillStyle = color;
  context.font = "700 66px system-ui, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 64, 67);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const label = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  }));
  label.name = `${text.toLowerCase()}-axis-direction-label`;
  label.scale.setScalar(worldSize);
  label.renderOrder = 20;
  label.userData.viewerHelper = true;
  label.userData.axisDirection = text;
  return label;
}

function flexivWorldToThreePosition(position) {
  const [x, y, z] = position;
  return new THREE.Vector3(x, z, -y);
}

function groundScenePosition() {
  return flexivWorldToThreePosition(GROUND_FLEXIV_POSITION);
}

function createLabeledAxes(name, size, labelSize, axisLines = null) {
  const frame = new THREE.Group();
  frame.name = name;
  frame.userData.viewerHelper = true;
  const axes = axisLines || new THREE.AxesHelper(size);
  axes.name = `${name}-lines`;
  axes.userData.viewerHelper = true;
  frame.add(axes);

  const labels = [
    ["X", AXIS_COLORS.X, new THREE.Vector3(size * 1.18, 0, 0)],
    ["Y", AXIS_COLORS.Y, new THREE.Vector3(0, size * 1.18, 0)],
    ["Z", AXIS_COLORS.Z, new THREE.Vector3(0, 0, size * 1.18)],
  ];
  for (const [axis, color, position] of labels) {
    const label = createAxisLabel(axis, color, labelSize);
    label.position.copy(position);
    frame.add(label);
  }
  return frame;
}

function numberAttr(node, name, fallback = 0) {
  const value = Number.parseFloat(node?.getAttribute(name));
  return Number.isFinite(value) ? value : fallback;
}

function vectorAttr(node, name, fallback = [0, 0, 0]) {
  const raw = node?.getAttribute(name);
  if (!raw) return new THREE.Vector3(...fallback);
  const values = raw.trim().split(/\s+/).map(Number);
  if (values.length < 3 || values.some((value) => !Number.isFinite(value))) {
    return new THREE.Vector3(...fallback);
  }
  return new THREE.Vector3(values[0], values[1], values[2]);
}

function transformFromURDF(element, target) {
  const origin = element?.querySelector(":scope > origin");
  if (!origin) return;
  target.position.copy(vectorAttr(origin, "xyz"));
  const rpy = vectorAttr(origin, "rpy");
  target.rotation.set(rpy.x, rpy.y, rpy.z, "XYZ");
}

function _resolveMeshAssetKey(filename) {
  const raw = typeof filename === "string" ? filename.trim().replace(/\\/g, "/") : "";
  if (!raw) return "";
  const withoutScheme = raw.replace(/^package:\/\//, "").replace(/^\/+/, "");
  if (!withoutScheme || withoutScheme.split("/").some((part) => part === "..")) return "";

  // The Flexiv descriptions use `package://meshes/Rizon4/...`: in this
  // repository `meshes` is a real directory, not a package name to discard.
  if (withoutScheme.startsWith("meshes/")) return withoutScheme;
  const nestedMeshRoot = withoutScheme.indexOf("/meshes/");
  if (nestedMeshRoot >= 0) return withoutScheme.slice(nestedMeshRoot + 1);
  if (/^Rizon4R?\//.test(withoutScheme)) return `meshes/${withoutScheme}`;
  return withoutScheme;
}

function rgbaAttr(node) {
  const raw = node?.getAttribute("rgba");
  if (!raw) return null;
  const values = raw.trim().split(/\s+/).map(Number);
  if (values.length !== 4 || values.some((value) => !Number.isFinite(value))) return null;
  return Object.freeze([
    THREE.MathUtils.clamp(values[0], 0, 1),
    THREE.MathUtils.clamp(values[1], 0, 1),
    THREE.MathUtils.clamp(values[2], 0, 1),
    THREE.MathUtils.clamp(values[3], 0, 1),
  ]);
}

function parseURDF(xmlText) {
  const document = new DOMParser().parseFromString(xmlText, "application/xml");
  const parserError = document.querySelector("parsererror");
  if (parserError) throw new Error("The URDF could not be parsed.");

  const materialColors = new Map();
  document.querySelectorAll("robot > material").forEach((materialElement) => {
    const name = materialElement.getAttribute("name") || "";
    const color = rgbaAttr(materialElement.querySelector(":scope > color"));
    if (name && color) materialColors.set(name, color);
  });

  const links = new Map();
  document.querySelectorAll("robot > link").forEach((linkElement) => {
    const visuals = [];
    linkElement.querySelectorAll(":scope > visual").forEach((visualElement) => {
      const mesh = visualElement.querySelector(":scope > geometry > mesh");
      if (!mesh) return;
      const filename = mesh.getAttribute("filename") || "";
      const normalized = _resolveMeshAssetKey(filename);
      if (!normalized) return;
      const materialElement = visualElement.querySelector(":scope > material");
      const inlineColor = rgbaAttr(materialElement?.querySelector(":scope > color"));
      const materialName = materialElement?.getAttribute("name") || "";
      visuals.push({
        filename: normalized,
        scale: vectorAttr(mesh, "scale", [1, 1, 1]),
        transform: visualElement,
        color: inlineColor || materialColors.get(materialName) || null,
      });
    });
    links.set(linkElement.getAttribute("name"), { visuals });
  });

  const joints = [];
  const childNames = new Set();
  document.querySelectorAll("robot > joint").forEach((jointElement) => {
    const parent = jointElement.querySelector(":scope > parent")?.getAttribute("link");
    const child = jointElement.querySelector(":scope > child")?.getAttribute("link");
    if (!parent || !child) return;
    const limit = jointElement.querySelector(":scope > limit");
    const axis = vectorAttr(jointElement.querySelector(":scope > axis"), "xyz", [0, 0, 1]);
    if (axis.lengthSq() < 1e-8) axis.set(0, 0, 1);
    joints.push({
      name: jointElement.getAttribute("name") || child,
      type: jointElement.getAttribute("type") || "fixed",
      parent,
      child,
      axis: axis.normalize(),
      lower: numberAttr(limit, "lower", -Math.PI),
      upper: numberAttr(limit, "upper", Math.PI),
      transform: jointElement,
    });
    childNames.add(child);
  });

  const roots = [...links.keys()].filter((name) => !childNames.has(name));
  return {
    variant: document.documentElement.getAttribute("name") || "Flexiv",
    links,
    joints,
    root: roots.includes("world") ? "world" : roots[0],
  };
}

function materialForArm(material, color) {
  const copy = material?.clone ? material.clone() : new THREE.MeshStandardMaterial();
  if (copy.color) {
    // Keep the source mesh's light/dark variation while applying a clear arm
    // identity tint.  This also works when MTLLoader returns MeshPhongMaterial.
    copy.color.multiply(color);
  }
  if (copy.emissive) copy.emissive.multiplyScalar(0.08);
  copy.side = THREE.DoubleSide;
  return copy;
}

function tintObject(object, color) {
  object.traverse((node) => {
    if (!node.isMesh) return;
    if (Array.isArray(node.material)) {
      node.material = node.material.map((material) => materialForArm(material, color));
    } else {
      node.material = materialForArm(node.material, color);
    }
    node.castShadow = true;
    node.receiveShadow = true;
  });
}

function cloneAsset(asset, color) {
  const clone = asset.clone(true);
  tintObject(clone, color);
  return clone;
}

function cloneToolAssetSource(source, color) {
  const clone = source.clone(true);
  const preserveToolColors = source.userData?.preserveToolColors === true;
  clone.traverse((node) => {
    if (!node.isMesh) return;
    const cloneMaterial = (material) => {
      const copy = preserveToolColors
        ? material?.clone?.() || new THREE.MeshStandardMaterial()
        : materialForArm(material, color);
      copy.side = THREE.DoubleSide;
      return copy;
    };
    node.material = Array.isArray(node.material)
      ? node.material.map(cloneMaterial)
      : cloneMaterial(node.material);
    node.castShadow = true;
    node.receiveShadow = true;
  });
  initializeToolJointRuntime(clone);
  return clone;
}

function disposeToolAssetClone(object) {
  const materials = new Set();
  object?.traverse((node) => {
    const nodeMaterials = Array.isArray(node.material) ? node.material : [node.material];
    nodeMaterials.forEach((material) => {
      if (material?.dispose) materials.add(material);
    });
  });
  materials.forEach((material) => material.dispose());
}

function createConfiguredToolPlaceholder(side) {
  const color = side === "left" ? LEFT_COLOR : RIGHT_COLOR;
  const group = new THREE.Group();
  group.name = `${side}-configured-tool-placeholder`;
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: color.clone().lerp(new THREE.Color(0xffffff), 0.32),
    metalness: 0.38,
    roughness: 0.48,
    transparent: true,
    opacity: 0.88,
  });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.034, 0.12, 20), bodyMaterial);
  body.name = `${side}-configured-tool-placeholder-body`;
  body.rotation.x = Math.PI / 2;
  body.position.z = -0.06;
  const tip = new THREE.Mesh(
    new THREE.ConeGeometry(0.023, 0.055, 20),
    bodyMaterial.clone(),
  );
  tip.name = `${side}-configured-tool-placeholder-tip`;
  tip.rotation.x = -Math.PI / 2;
  tip.position.z = 0.026;
  group.add(body, tip);
  return group;
}

function toolUrdfMeshUrl(filename, urdfUrl) {
  const rawFilename = typeof filename === "string" ? filename.trim() : "";
  const rawUrdfUrl = typeof urdfUrl === "string" ? urdfUrl.trim() : "";
  if (!rawFilename || !rawUrdfUrl) return null;
  if (/[%\\?#]/.test(rawFilename) || rawFilename.startsWith("//")) return null;
  if (/[%\\?#]/.test(rawUrdfUrl) || rawUrdfUrl.startsWith("//")) return null;
  const filenameScheme = rawFilename.match(/^([a-z][a-z0-9+.-]*):/i);
  if (filenameScheme && !rawFilename.startsWith("package://")) return null;

  let relative = _resolveMeshAssetKey(rawFilename);
  if (!relative || relative.includes(":") || relative.startsWith("//")) return null;
  const meshRoot = relative.indexOf("meshes/");
  if (meshRoot > 0) relative = relative.slice(meshRoot);

  try {
    const fallbackBase = new URL("http://localhost/");
    let pageUrl = new URL(globalThis.location?.href || fallbackBase.href, fallbackBase);
    if (!/^https?:$/.test(pageUrl.protocol)) pageUrl = fallbackBase;
    const resolvedUrdfUrl = new URL(rawUrdfUrl, pageUrl);
    if (resolvedUrdfUrl.origin !== pageUrl.origin) return null;
    if (!resolvedUrdfUrl.pathname.startsWith("/static/")) return null;
    const url = new URL(relative, new URL(".", resolvedUrdfUrl));
    if (url.origin !== pageUrl.origin || !url.pathname.startsWith("/static/")) return null;
    return url.href;
  } catch {
    return null;
  }
}

function applyUrdfVisualMaterial(object, rgba) {
  if (!rgba) return;
  const [red, green, blue, alpha] = rgba;
  object.traverse((node) => {
    if (!node.isMesh) return;
    const previous = Array.isArray(node.material) ? node.material : [node.material];
    previous.forEach((material) => material?.dispose?.());
    node.material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(red, green, blue),
      metalness: 0.28,
      roughness: 0.48,
      transparent: alpha < 0.999,
      opacity: alpha,
      side: THREE.DoubleSide,
    });
  });
}

function initializeToolJointRuntime(root) {
  const joints = new Map();
  root?.traverse((node) => {
    const definition = node.userData?.toolJointDefinition;
    if (!definition || typeof definition !== "object") return;
    const axis = Array.isArray(definition.axis) ? definition.axis.map(Number) : [];
    const baseQuaternion = Array.isArray(definition.baseQuaternion)
      ? definition.baseQuaternion.map(Number)
      : [];
    if (
      axis.length !== 3
      || baseQuaternion.length !== 4
      || [...axis, ...baseQuaternion].some((value) => !Number.isFinite(value))
    ) return;
    node.userData.axis = new THREE.Vector3(...axis).normalize();
    node.userData.limit = [Number(definition.lower), Number(definition.upper)];
    node.userData.jointName = String(definition.name || node.name);
    node.userData.baseQuaternion = new THREE.Quaternion(...baseQuaternion).normalize();
    node.userData.axisAngleQuaternion = new THREE.Quaternion();
    joints.set(node.userData.jointName, node);
  });
  root.userData.toolJoints = joints;
  return joints;
}

function buildDynamicUrdfObject(model, assets, prefix) {
  const links = new Map();
  model.links.forEach((_, name) => {
    const group = new THREE.Group();
    group.name = `${prefix}-${name}`;
    links.set(name, group);
  });
  model.joints.forEach((joint) => {
    const parent = links.get(joint.parent);
    const child = links.get(joint.child);
    if (!parent || !child) return;
    const jointGroup = new THREE.Group();
    jointGroup.name = `${prefix}-${joint.name}-origin`;
    transformFromURDF(joint.transform, jointGroup);
    const motion = new THREE.Group();
    motion.name = `${prefix}-${joint.name}`;
    motion.userData.toolJointDefinition = {
      name: joint.name,
      axis: joint.axis.toArray(),
      lower: joint.lower,
      upper: joint.upper,
      baseQuaternion: motion.quaternion.toArray(),
    };
    jointGroup.add(motion);
    motion.add(child);
    parent.add(jointGroup);
  });
  const color = new THREE.Color(0xe7edf2);
  model.links.forEach((linkData, linkName) => {
    const linkGroup = links.get(linkName);
    linkData.visuals.forEach((visual) => {
      const asset = assets.get(visual.filename);
      if (!asset) return;
      const mesh = cloneAsset(asset, color);
      mesh.name = `${prefix}-${linkName}-${visual.filename.split("/").pop()}`;
      mesh.scale.copy(visual.scale);
      transformFromURDF(visual.transform, mesh);
      applyUrdfVisualMaterial(mesh, visual.color);
      linkGroup.add(mesh);
    });
  });
  const root = links.get(model.root) || links.values().next().value;
  if (!root) throw new Error("Tool URDF has no renderable root link.");
  root.userData.preserveToolColors = true;
  root.userData.toolUrdf = true;
  initializeToolJointRuntime(root);
  root.updateMatrixWorld(true);
  return root;
}

function homePoseGhostMaterial(material, color) {
  const copy = materialForArm(material, color);
  copy.transparent = true;
  copy.opacity = HOME_POSE_GHOST_OPACITY;
  copy.depthWrite = false;
  copy.depthTest = true;
  copy.needsUpdate = true;
  return copy;
}

function freezeStaticHierarchy(root) {
  root.traverse((node) => {
    node.updateMatrix();
    node.matrixAutoUpdate = false;
  });
  root.updateMatrixWorld(true);
}

function setJointAngle(joint, value) {
  const axisAngle = joint.userData.axisAngleQuaternion;
  axisAngle.setFromAxisAngle(joint.userData.axis, value);
  // The joint's initial orientation is part of its local transform. Compose
  // the commanded axis rotation after it instead of replacing that basis.
  joint.quaternion.copy(joint.userData.baseQuaternion).multiply(axisAngle);
}

function meshCount(object) {
  let count = 0;
  object?.traverse((node) => {
    if (node.isMesh && !node.userData.viewerHelper) count += 1;
  });
  return count;
}

function vectorSnapshot(vector) {
  return Object.freeze(vector.toArray());
}

function quaternionSnapshot(quaternion) {
  return Object.freeze(quaternion.toArray());
}

function boundsSnapshot(object) {
  if (!object) return null;
  const bounds = new THREE.Box3().setFromObject(object);
  if (bounds.isEmpty()) return null;
  return Object.freeze({
    min: vectorSnapshot(bounds.min),
    max: vectorSnapshot(bounds.max),
    center: vectorSnapshot(bounds.getCenter(new THREE.Vector3())),
    size: vectorSnapshot(bounds.getSize(new THREE.Vector3())),
  });
}

function rendererInfoSnapshot(renderer) {
  const info = renderer?.info;
  return Object.freeze({
    calls: Number(info?.render?.calls) || 0,
    triangles: Number(info?.render?.triangles) || 0,
    points: Number(info?.render?.points) || 0,
    lines: Number(info?.render?.lines) || 0,
    geometries: Number(info?.memory?.geometries) || 0,
    textures: Number(info?.memory?.textures) || 0,
    programs: Array.isArray(info?.programs) ? info.programs.length : 0,
  });
}

function disposeObjectResources(objects) {
  const geometries = new Set();
  const materials = new Set();
  const textures = new Set();
  for (const object of objects) {
    object?.traverse((node) => {
      if (node.geometry?.dispose) geometries.add(node.geometry);
      const nodeMaterials = Array.isArray(node.material) ? node.material : [node.material];
      for (const material of nodeMaterials) {
        if (!material?.dispose) continue;
        materials.add(material);
        for (const value of Object.values(material)) {
          if (value?.isTexture && value.dispose) textures.add(value);
        }
      }
    });
  }
  textures.forEach((texture) => texture.dispose());
  materials.forEach((material) => material.dispose());
  geometries.forEach((geometry) => geometry.dispose());
}

function disposeToolAssetCache(
  cache,
  disposer = (source) => disposeObjectResources([source]),
) {
  const pending = [...cache.values()].map((value) => Promise.resolve(value).then(
    (source) => {
      disposer(source);
      return true;
    },
    () => false,
  ).catch(() => false));
  cache.clear();
  return Promise.all(pending);
}

function tcpFrameSnapshot(arm) {
  const frame = arm?.tcpFrame;
  if (!frame) return null;
  return Object.freeze({
    visible: frame.visible,
    position: vectorSnapshot(frame.position),
    quaternionXyzw: Object.freeze(frame.quaternion.toArray()),
  });
}

function configuredToolFrameSnapshot(arm) {
  const frame = arm?.configuredToolFrame;
  if (!frame) return null;
  return Object.freeze({
    visible: frame.visible,
    localPosition: vectorSnapshot(frame.position),
    localQuaternionXyzw: quaternionSnapshot(frame.quaternion),
    worldPosition: vectorSnapshot(frame.getWorldPosition(new THREE.Vector3())),
    worldQuaternionXyzw: quaternionSnapshot(
      frame.getWorldQuaternion(new THREE.Quaternion()),
    ),
  });
}

const EMPTY_DEBUG_SNAPSHOT = Object.freeze({
  loadedArmCount: 0,
  meshCounts: Object.freeze({ left: 0, right: 0 }),
  modelVariants: Object.freeze({ left: null, right: null }),
  assetCounts: Object.freeze({ left: 0, right: 0 }),
  baseQuaternionsXyzw: Object.freeze({ left: null, right: null }),
  modelQuaternionsXyzw: Object.freeze({ left: null, right: null }),
  worldBounds: Object.freeze({ left: null, right: null, combined: null }),
  tcpFrames: Object.freeze({ left: null, right: null }),
  configuredToolFrames: Object.freeze({ left: null, right: null }),
  configuredTools: Object.freeze({ left: EMPTY_CONFIGURED_TOOL, right: EMPTY_CONFIGURED_TOOL }),
  toolAssets: Object.freeze({ left: null, right: null }),
  gripperPositions: Object.freeze({ left: null, right: null }),
  armBasePositions: Object.freeze({ left: null, right: null }),
  groundPlane: Object.freeze({
    flexivWorld: GROUND_FLEXIV_POSITION,
    floorScene: null,
    gridScene: null,
  }),
  viewTarget: Object.freeze({
    flexivWorld: GROUND_FLEXIV_POSITION,
    scene: vectorSnapshot(groundScenePosition()),
    controlsScene: null,
    defaultScene: null,
  }),
  worldFramePosition: null,
  cameraPosition: null,
  identityLabels: Object.freeze({ left: null, right: null }),
  homePoseGhosts: Object.freeze({
    status: "unavailable",
    key: null,
    fresh: Object.freeze({ left: false, right: false }),
    counts: Object.freeze({ left: 0, right: 0 }),
    visible: Object.freeze({ left: false, right: false }),
    materialCounts: Object.freeze({ left: 0, right: 0 }),
  }),
  performance: Object.freeze({
    scheduler: null,
    renderer: rendererInfoSnapshot(null),
    counters: Object.freeze({}),
    pixelRatio: null,
    visible: false,
  }),
});

let activeViewer = null;

function installDebugGetter() {
  if (typeof window === "undefined") return;
  const descriptor = Object.getOwnPropertyDescriptor(window, "__armViewerDebug");
  if (descriptor && !descriptor.configurable) return;
  Object.defineProperty(window, "__armViewerDebug", {
    configurable: true,
    enumerable: false,
    get: () => activeViewer?.getDebugSnapshot() || EMPTY_DEBUG_SNAPSHOT,
  });
}

function loadText(url) {
  return fetch(url).then((response) => {
    if (!response.ok) throw new Error(`Unable to load ${url} (${response.status}).`);
    return response.text();
  });
}

function loadArrayBuffer(url) {
  return fetch(url).then((response) => {
    if (!response.ok) throw new Error(`Unable to load ${url} (${response.status}).`);
    return response.arrayBuffer();
  });
}

function parseBinarySTL(input) {
  const bytes = input instanceof ArrayBuffer
    ? new Uint8Array(input)
    : ArrayBuffer.isView(input)
      ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
      : null;
  if (!bytes || bytes.byteLength < 84) throw new Error("Binary STL is shorter than its header.");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const triangleCount = view.getUint32(80, true);
  if (triangleCount <= 0 || triangleCount > 2_000_000) {
    throw new Error("Binary STL triangle count is outside the supported range.");
  }
  const expectedLength = 84 + triangleCount * 50;
  if (expectedLength !== bytes.byteLength) {
    throw new Error("Binary STL byte length does not match its triangle count.");
  }
  const positions = new Float32Array(triangleCount * 9);
  let target = 0;
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const offset = 84 + triangle * 50;
    for (let component = 0; component < 12; component += 1) {
      const value = view.getFloat32(offset + component * 4, true);
      if (!Number.isFinite(value)) throw new Error("Binary STL contains a non-finite coordinate.");
      if (component >= 3) positions[target++] = value;
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  geometry.userData.binaryStlTriangleCount = triangleCount;
  return geometry;
}

class DualArmViewer {
  constructor(root, options = {}) {
    if (!root) throw new Error("DualArmViewer needs a mount element.");
    this.root = root;
    this.options = options;
    this.statusMode = options.statusMode === "external" ? "external" : "robot-state";
    this.urdfUrls = Object.freeze({
      left: options.leftUrdfUrl || options.urdfUrls?.left || ARM_MODEL_SPECS.left.urdfUrl,
      right: options.rightUrdfUrl || options.urdfUrls?.right || ARM_MODEL_SPECS.right.urdfUrl,
    });
    this.assetRoot = options.assetRoot || DEFAULT_ASSET_ROOT;
    this.robotStateUrl = options.robotStateUrl === undefined ? DEFAULT_ROBOT_STATE_URL : options.robotStateUrl;
    this.statePollIntervalMs = Math.max(50, Number(options.statePollIntervalMs) || DEFAULT_STATE_POLL_INTERVAL_MS);
    this.homePoseCandidatesUrl = options.homePoseCandidatesUrl === undefined
      ? DEFAULT_HOME_POSE_CANDIDATES_URL
      : options.homePoseCandidatesUrl;
    this.homePoseCandidatePollIntervalMs = Math.max(
      250,
      Number(options.homePoseCandidatePollIntervalMs) || DEFAULT_HOME_POSE_CANDIDATE_POLL_INTERVAL_MS,
    );
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b1118);
    this.scene.fog = new THREE.Fog(0x0b1118, 5.2, 11);
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.01, 100);
    this.camera.position.set(2.45, 2.15, 3.55);
    this.controls = null;
    this.renderer = null;
    this.floor = null;
    this.grid = null;
    this.sceneRoot = new THREE.Group();
    this.sceneRoot.name = "solid-arm-scene-root";
    this.recordedTcpTrajectoryRoot = new THREE.Group();
    this.recordedTcpTrajectoryRoot.name = "recorded-tcp-trajectory-root";
    this.recordedTcpTrajectoryVisuals = {
      left: createRecordedTcpTrajectoryVisual("left", LEFT_COLOR),
      right: createRecordedTcpTrajectoryVisual("right", RIGHT_COLOR),
    };
    this.recordedTcpTrajectoryRoot.add(
      this.recordedTcpTrajectoryVisuals.left.group,
      this.recordedTcpTrajectoryVisuals.right.group,
    );
    this.sceneRoot.add(this.recordedTcpTrajectoryRoot);
    this.recordedTcpScenePoints = [];
    this.armModelsVisible = true;
    // Ghosts are intentionally outside sceneRoot so alternative poses never
    // change the default camera fit or the solid-arm debug bounds.
    this.homePoseGhostRoot = new THREE.Group();
    this.homePoseGhostRoot.name = "home-pose-ghost-root";
    this.homePoseGhostSideRoots = { left: new THREE.Group(), right: new THREE.Group() };
    for (const side of ["left", "right"]) {
      this.homePoseGhostSideRoots[side].name = `${side}-home-pose-ghosts`;
      this.homePoseGhostSideRoots[side].visible = false;
      this.homePoseGhostRoot.add(this.homePoseGhostSideRoots[side]);
    }
    this.scene.add(this.sceneRoot, this.homePoseGhostRoot);
    this.arms = { left: null, right: null };
    this.yam = null;
    this.yamState = normalizedYamState(null);
    this.yamJointState = null;
    this.yamAssetState = { status: "idle", error: null };
    this._yamLoadPromise = null;
    this._yamLoadEpoch = 0;
    this.jointStates = { left: [...DEFAULT_POSE], right: [...DEFAULT_POSE] };
    this.homeJointStates = { left: null, right: null };
    this.jointSources = { left: "default", right: "default" };
    this.tcpStates = { left: null, right: null };
    this.toolNames = { left: null, right: null };
    this.configuredTools = {
      left: EMPTY_CONFIGURED_TOOL,
      right: EMPTY_CONFIGURED_TOOL,
    };
    this.configuredToolDescriptors = {
      left: EMPTY_CONFIGURED_TOOL,
      right: EMPTY_CONFIGURED_TOOL,
    };
    this.toolVisualVariant = TOOL_VISUAL_VARIANT_KEYS.includes(options.toolVisualVariant)
      ? options.toolVisualVariant
      : storedToolVisualVariant();
    this.toolVariantControl = null;
    this.toolVariantButtons = [];
    this._onToolVariantClick = (event) => {
      const variant = event.currentTarget?.dataset?.toolVisualVariant;
      this.setToolVisualVariant(variant);
    };
    this.toolAssetCache = new Map();
    this.toolAssetRequestGate = new ToolAssetRequestGate();
    this.toolAssetObjects = { left: null, right: null };
    this.toolJointByName = { left: new Map(), right: new Map() };
    this.gripperVisualStates = {
      left: { known: false, rawValue: null, actualPercent: null, jointAngle: null, source: "initial" },
      right: { known: false, rawValue: null, actualPercent: null, jointAngle: null, source: "initial" },
    };
    this.gripperDirectSourceFreshUntil = { left: 0, right: 0 };
    this.gripperDirectValues = { left: null, right: null };
    this.gripperArticulationSignatures = { left: "", right: "" };
    this.toolAssetStates = {
      left: { status: "none", url: null, error: null },
      right: { status: "none", url: null, error: null },
    };
    this.jointByName = { left: new Map(), right: new Map() };
    // Models and mesh caches are deliberately side-specific. Rizon4 and
    // Rizon4R share basenames such as link0.obj, but they are not equivalent.
    this.armAssets = { left: new Map(), right: new Map() };
    this.armModels = { left: null, right: null };
    this.homePoseCandidateState = emptyHomePoseCandidates();
    this.homePoseGhostEntries = { left: [], right: [] };
    this.homePoseGhostFresh = { left: false, right: false };
    this.homePoseGhostMaterialCaches = { left: new Map(), right: new Map() };
    this.homePoseGhostPointGeometry = new THREE.SphereGeometry(HOME_POSE_GHOST_POINT_SIZE, 12, 8);
    this.homePoseGhostPointMaterials = {
      left: new THREE.MeshBasicMaterial({
        color: LEFT_COLOR.clone(),
        transparent: true,
        opacity: 0.72,
        depthWrite: false,
      }),
      right: new THREE.MeshBasicMaterial({
        color: RIGHT_COLOR.clone(),
        transparent: true,
        opacity: 0.72,
        depthWrite: false,
      }),
    };
    this._renderedHomePoseCandidateSignature = null;
    this._resizeObserver = null;
    this._intersectionObserver = null;
    this._renderLoop = null;
    this._pageVisible = typeof document === "undefined" || document.visibilityState !== "hidden";
    this._panelVisible = true;
    this._renderWidth = 0;
    this._renderHeight = 0;
    this._statePollTimer = 0;
    this._stateRequestController = null;
    this._homePoseCandidatePollTimer = 0;
    this._homePoseCandidateRequestController = null;
    this._modelLoaded = false;
    this._liveStateAvailable = false;
    this._liveFlexivStateAvailable = false;
    this._lastRobotTimestamp = null;
    this._defaultView = null;
    this._perf = {
      joint_applies: 0,
      joint_skips: 0,
      tcp_applies: 0,
      tcp_skips: 0,
      tool_applies: 0,
      tool_skips: 0,
      tool_asset_loads: 0,
      tool_asset_failures: 0,
      tool_asset_stale_drops: 0,
      gripper_applies: 0,
      gripper_skips: 0,
      gripper_unknowns: 0,
      yam_applies: 0,
      yam_skips: 0,
      yam_asset_loads: 0,
      yam_asset_failures: 0,
      ghost_rebuilds: 0,
      ghost_visibility_changes: 0,
      resize_applies: 0,
      resize_skips: 0,
      state_polls: 0,
      candidate_polls: 0,
    };
    this._onDocumentVisibilityChange = () => this._handleVisibilityChange();
    this._disposed = false;
    this._setupDOM();
    this._setupToolVariantControls();
    this._setupScene();
    this._scheduleStatePoll(0);
    this._scheduleHomePoseCandidatePoll(0);
    this.ready = this._load();
    void this.ready.catch((error) => this._setState("error", error));
  }

  _setupDOM() {
    this.root.replaceChildren();
    this.root.classList.add("arm-viewer-root");
    this.root.setAttribute("aria-label", "Dual Flexiv arm 3D viewer");
    this.root.style.position = "relative";
    this.root.style.minHeight = this.root.style.minHeight || "360px";
    this.root.style.overflow = "hidden";

    this.canvasHost = document.createElement("div");
    this.canvasHost.className = "arm-viewer-canvas";
    Object.assign(this.canvasHost.style, {
      position: "absolute",
      inset: "0",
      minHeight: "inherit",
    });
    this.root.appendChild(this.canvasHost);

    this.status = document.createElement("div");
    this.status.className = "arm-viewer-status";
    this.status.setAttribute("role", "status");
    Object.assign(this.status.style, {
      position: "absolute",
      left: "16px",
      bottom: "14px",
      zIndex: "2",
      color: "#dce8f0",
      font: "12px/1.4 system-ui, sans-serif",
      pointerEvents: "none",
      maxWidth: "calc(100% - 32px)",
    });
    this.root.appendChild(this.status);
    this._setState("loading", "Loading Flexiv Rizon 4 / 4R models...");

    this.axisLegend = document.createElement("div");
    this.axisLegend.className = "arm-viewer-axis-legend";
    this.axisLegend.setAttribute("aria-label", "World axes: X red, Y green, Z blue");
    Object.assign(this.axisLegend.style, {
      position: "absolute",
      top: "14px",
      left: "16px",
      zIndex: "2",
      color: "#dce8f0",
      font: "700 11px/1.4 system-ui, sans-serif",
      letterSpacing: "0.04em",
      pointerEvents: "none",
    });
    const worldLabel = document.createElement("span");
    worldLabel.textContent = "WORLD  ";
    this.axisLegend.appendChild(worldLabel);
    for (const axis of ["X", "Y", "Z"]) {
      const label = document.createElement("span");
      label.textContent = `${axis}${axis === "Z" ? "" : "  "}`;
      label.style.color = AXIS_COLORS[axis];
      this.axisLegend.appendChild(label);
    }
    this.root.appendChild(this.axisLegend);

    this.resetButton = document.createElement("button");
    this.resetButton.type = "button";
    this.resetButton.className = "arm-viewer-reset";
    this.resetButton.textContent = "Reset view";
    this.resetButton.title = "Reset 3D view";
    Object.assign(this.resetButton.style, {
      position: "absolute",
      top: "12px",
      right: "12px",
      zIndex: "3",
      cursor: "pointer",
    });
    this.resetButton.addEventListener("click", () => this.resetView());
    this.root.appendChild(this.resetButton);
  }

  _setupToolVariantControls() {
    this.toolVariantControl = document.getElementById("tool-variant-control");
    this.toolVariantButtons = typeof document.querySelectorAll === "function"
      ? [...document.querySelectorAll("[data-tool-visual-variant]")]
      : [];
    for (const button of this.toolVariantButtons) {
      button.addEventListener("click", this._onToolVariantClick);
    }
    this._syncToolVariantControls();
  }

  _syncToolVariantControls() {
    const available = new Set();
    for (const side of ["left", "right"]) {
      const variants = this.configuredToolDescriptors?.[side]?.visual?.variants || {};
      Object.keys(variants).forEach((key) => available.add(key));
    }
    const loading = ["left", "right"].some(
      (side) => this.toolAssetStates?.[side]?.status === "loading",
    );
    const hasSelectableVariants = TOOL_VISUAL_VARIANT_KEYS.every(
      (key) => available.has(key),
    );
    if (this.toolVariantControl) {
      this.toolVariantControl.dataset.loading = loading ? "true" : "false";
      this.toolVariantControl.dataset.available = hasSelectableVariants ? "true" : "false";
      this.toolVariantControl.title = hasSelectableVariants
        ? `当前模型：${this.toolVisualVariant === "hard" ? "硬指夹爪" : "软指夹爪"}`
        : "当前配置未加载可切换的 UMI3 夹爪模型";
    }
    for (const button of this.toolVariantButtons || []) {
      const variant = normalizedToolVariantKey(button.dataset.toolVisualVariant);
      const selected = variant === this.toolVisualVariant;
      button.disabled = !variant || !available.has(variant);
      button.setAttribute("aria-pressed", selected ? "true" : "false");
      button.setAttribute("aria-busy", selected && loading ? "true" : "false");
      button.dataset.active = selected ? "true" : "false";
    }
  }

  setToolVisualVariant(value) {
    const variant = normalizedToolVariantKey(value);
    if (!variant || !TOOL_VISUAL_VARIANT_KEYS.includes(variant)) return false;
    if (variant === this.toolVisualVariant) {
      this._syncToolVariantControls();
      return this;
    }
    this.toolVisualVariant = variant;
    storeToolVisualVariant(variant);
    for (const side of ["left", "right"]) {
      this._applyConfiguredTool(side, this.configuredToolDescriptors[side]);
    }
    this._syncToolVariantControls();
    this._showRobotStateStatus();
    return this;
  }

  _setupScene() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_DEVICE_PIXEL_RATIO));
    this.renderer.setSize(1, 1, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.canvasHost.appendChild(this.renderer.domElement);

    const defaultViewTarget = groundScenePosition();
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.target.copy(defaultViewTarget);
    this.camera.lookAt(defaultViewTarget);
    this.controls.minDistance = 0.7;
    this.controls.maxDistance = 7;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this._renderLoop = new DirtyRenderLoop({
      render: () => this.renderer?.render(this.scene, this.camera),
      update: () => Boolean(this.controls?.update()),
      canRender: () => this._canWork(),
      maxFps: MAX_RENDER_FPS,
    });
    this.controls.addEventListener("start", () => this._renderLoop?.beginDamping());
    this.controls.addEventListener("change", () => this._invalidateRender("controls-change"));
    this.controls.addEventListener("end", () => this._renderLoop?.beginDamping());

    const key = new THREE.DirectionalLight(0xffffff, 3.2);
    key.position.set(2.5, 5, 3.5);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    this.scene.add(key);
    const fill = new THREE.HemisphereLight(0xc9e4ff, 0x142333, 1.65);
    this.scene.add(fill);
    const rim = new THREE.DirectionalLight(0x5ec9d4, 1.25);
    rim.position.set(-3, 2.8, -1.5);
    this.scene.add(rim);

    this.floor = new THREE.Mesh(
      new THREE.PlaneGeometry(8, 8),
      new THREE.MeshStandardMaterial({ color: 0x111b25, roughness: 0.92, metalness: 0.05 }),
    );
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.position.copy(defaultViewTarget);
    this.floor.receiveShadow = true;
    this.scene.add(this.floor);
    this.grid = new THREE.GridHelper(6, 24, 0x294254, 0x1a2b37);
    this.grid.position.copy(defaultViewTarget);
    this.grid.position.y += GRID_SCENE_LIFT;
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.48;
    this.scene.add(this.grid);

    // Flexiv reports poses in a Z-up world frame.  Keep that convention in
    // the labelled reference triad while adapting it to Three.js's Y-up scene.
    this.worldFrame = createLabeledAxes("world-coordinate-axes", WORLD_AXIS_SIZE, 0.075);
    this.worldFrame.quaternion.copy(FLEXIV_TO_THREE_QUATERNION);
    this.worldFrame.position.set(0, 0, 0);
    this.scene.add(this.worldFrame);

    this._resizeObserver = new ResizeObserver(() => this._resize());
    this._resizeObserver.observe(this.canvasHost);
    document.addEventListener("visibilitychange", this._onDocumentVisibilityChange);
    if (typeof IntersectionObserver === "function") {
      this._intersectionObserver = new IntersectionObserver((entries) => {
        const entry = entries[entries.length - 1];
        const visible = Boolean(entry?.isIntersecting && entry.intersectionRatio > 0);
        if (visible === this._panelVisible) return;
        this._panelVisible = visible;
        this._handleVisibilityChange();
      }, { threshold: 0 });
      this._intersectionObserver.observe(this.root);
    }
    this._resize();
    this._invalidateRender("scene-ready");
  }

  async _load() {
    const sides = ["left", "right"];
    const urdfTexts = await Promise.all(sides.map((side) => loadText(this.urdfUrls[side])));
    sides.forEach((side, index) => {
      this.armModels[side] = parseURDF(urdfTexts[index]);
    });

    for (const side of sides) {
      const model = this.armModels[side];
      const paths = new Set();
      model.links.forEach((link) => link.visuals.forEach((visual) => paths.add(visual.filename)));
      let loaded = 0;
      for (const relativePath of paths) {
        this._setState(
          "loading",
          `Loading ${model.variant} mesh ${++loaded}/${paths.size}...`,
        );
        this.armAssets[side].set(relativePath, await this._loadOBJ(relativePath));
      }
    }
    this.arms.left = this._buildArm(
      "left",
      LEFT_COLOR,
      flexivWorldToThreePosition(ARM_BASE_POSITIONS.left),
    );
    this.arms.right = this._buildArm(
      "right",
      RIGHT_COLOR,
      flexivWorldToThreePosition(ARM_BASE_POSITIONS.right),
    );
    this.sceneRoot.add(this.arms.left.root, this.arms.right.root);
    this.arms.left.root.visible = this.armModelsVisible;
    this.arms.right.root.visible = this.armModelsVisible;
    this.updateJointStates(this.jointStates.left, this.jointStates.right);
    this.updateConfiguredTools(this.configuredTools.left, this.configuredTools.right);
    this.updateTcpPoses(this.tcpStates.left, this.tcpStates.right);
    this._modelLoaded = true;
    if (!this.armModelsVisible && this.recordedTcpScenePoints.length) {
      this._frameRecordedTcpTrajectories(this.recordedTcpScenePoints);
    } else {
      this._frameModel();
    }
    const candidateState = this.homePoseCandidateState;
    if ((candidateState.status === "ready" || candidateState.status === "partial") && candidateState.key) {
      this._rebuildHomePoseGhosts(`${candidateState.status}:${candidateState.key}`);
    }
    this._showRobotStateStatus();
  }

  _loadOBJ(relativePath) {
    const objUrl = new URL(relativePath, this.assetRoot).href;
    const mtlUrl = objUrl.replace(/\.obj$/i, ".mtl");
    return this._loadOBJUrls(objUrl, mtlUrl);
  }

  _loadOBJUrls(objUrl, mtlUrl = null) {
    return new Promise((resolve, reject) => {
      const loadWithoutMaterial = () => {
        const objLoader = new OBJLoader();
        objLoader.load(objUrl, resolve, undefined, reject);
      };
      if (!mtlUrl) {
        loadWithoutMaterial();
        return;
      }
      const mtlLoader = new MTLLoader();
      mtlLoader.setPath("");
      mtlLoader.setResourcePath(new URL(".", mtlUrl).href);
      mtlLoader.load(mtlUrl, (materials) => {
        materials.preload();
        const objLoader = new OBJLoader();
        objLoader.setMaterials(materials);
        objLoader.load(objUrl, resolve, undefined, reject);
      }, undefined, () => {
        // OBJLoader can still render a valid OBJ without an MTL file.  This
        // fallback keeps the model useful when a vendor strips material files.
        loadWithoutMaterial();
      });
    });
  }

  async _loadSTLUrl(stlUrl) {
    const geometry = parseBinarySTL(await loadArrayBuffer(stlUrl));
    const material = new THREE.MeshStandardMaterial({
      color: 0xe7edf2,
      metalness: 0.24,
      roughness: 0.52,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = stlUrl.split("/").pop() || "tool-stl";
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  _toolAssetCacheKey(asset) {
    return [asset.kind, asset.url, asset.materialUrl || ""].join("|");
  }

  _loadToolAssetSource(asset) {
    const key = this._toolAssetCacheKey(asset);
    if (this.toolAssetCache.has(key)) return this.toolAssetCache.get(key);
    this._perf.tool_asset_loads += 1;
    const loader = asset.kind === "urdf"
      ? this._loadToolUrdf(asset.url)
      : asset.kind === "stl"
        ? this._loadSTLUrl(asset.url)
        : this._loadOBJUrls(asset.url, asset.materialUrl || asset.url.replace(/\.obj$/i, ".mtl"));
    const promise = loader
      .catch((error) => {
        this.toolAssetCache.delete(key);
        throw error;
      });
    this.toolAssetCache.set(key, promise);
    return promise;
  }

  async _loadToolUrdf(urdfUrl) {
    const model = parseURDF(await loadText(urdfUrl));
    const paths = new Set();
    model.links.forEach((link) => link.visuals.forEach((visual) => paths.add(visual.filename)));
    const loadedAssets = await Promise.all([...paths].map(async (relativePath) => {
      const objUrl = toolUrdfMeshUrl(relativePath, urdfUrl);
      if (!objUrl) {
        throw new Error(`Unsupported or unsafe tool URDF mesh: ${relativePath}`);
      }
      const lower = objUrl.toLowerCase();
      if (lower.endsWith(".obj")) {
        return [relativePath, await this._loadOBJUrls(objUrl, objUrl.replace(/\.obj$/i, ".mtl"))];
      }
      if (lower.endsWith(".stl")) return [relativePath, await this._loadSTLUrl(objUrl)];
      throw new Error(`Unsupported tool URDF mesh extension: ${relativePath}`);
    }));
    return buildDynamicUrdfObject(
      model,
      new Map(loadedAssets),
      "configured-tool-urdf",
    );
  }

  _clearConfiguredToolAsset(side) {
    const arm = this.arms[side];
    const current = this.toolAssetObjects[side];
    if (current && arm?.configuredToolAssetMount) {
      arm.configuredToolAssetMount.remove(current);
      disposeToolAssetClone(current);
    }
    this.toolAssetObjects[side] = null;
    this.toolJointByName[side] = new Map();
  }

  _setConfiguredToolPlaceholder(side, visible) {
    const placeholder = this.arms[side]?.configuredToolPlaceholder;
    if (placeholder) placeholder.visible = Boolean(visible);
  }

  _updateConfiguredToolAsset(side, tool) {
    const epoch = this.toolAssetRequestGate.next(side);
    this._clearConfiguredToolAsset(side);
    const visualTransform = configuredToolVisualTransform(tool);
    const asset = tool.visual?.asset || tool.asset;
    const hasVisualMount = Boolean(visualTransform);
    if (!hasVisualMount) {
      this._setConfiguredToolPlaceholder(side, false);
      this.toolAssetStates[side] = { status: "none", url: null, error: null };
      this._syncToolVariantControls();
      return;
    }
    if (!asset) {
      this._setConfiguredToolPlaceholder(side, true);
      this.toolAssetStates[side] = { status: "placeholder", url: null, error: null };
      this._syncToolVariantControls();
      return;
    }

    this._setConfiguredToolPlaceholder(side, true);
    this.toolAssetStates[side] = { status: "loading", url: asset.url, error: null };
    this._syncToolVariantControls();
    this.toolAssetRequestGate.settle(
      side,
      epoch,
      this._loadToolAssetSource(asset),
    ).then(({ accepted, value: source, error }) => {
      if (this._disposed || !accepted) {
        this._perf.tool_asset_stale_drops += 1;
        return;
      }
      if (error) {
        this._perf.tool_asset_failures += 1;
        this._setConfiguredToolPlaceholder(side, true);
        this.toolAssetStates[side] = {
          status: "error",
          url: asset.url,
          error: String(error?.message || error).slice(0, 256),
        };
        this._syncToolVariantControls();
        this._showRobotStateStatus();
        this._invalidateRender("configured-tool-asset-error");
        return;
      }
      const mounted = cloneToolAssetSource(
        source,
        side === "left" ? LEFT_COLOR : RIGHT_COLOR,
      );
      mounted.name = `${side}-configured-tool-asset`;
      mounted.scale.set(...asset.scale);
      if (tool.visual?.mirrorAxis === "y") mounted.scale.y *= -1;
      mounted.userData.armSide = side;
      mounted.userData.toolVisualVariant = tool.visual?.selectedVariant || null;
      mounted.userData.toolMirrorAxis = tool.visual?.mirrorAxis || null;
      this.toolJointByName[side] = mounted.userData.toolJoints instanceof Map
        ? mounted.userData.toolJoints
        : initializeToolJointRuntime(mounted);
      mounted.updateMatrix();
      mounted.updateMatrixWorld(true);
      this.arms[side]?.configuredToolAssetMount?.add(mounted);
      this.toolAssetObjects[side] = mounted;
      this._applyGripperStateToMountedTool(side, true);
      this._setConfiguredToolPlaceholder(side, false);
      this.toolAssetStates[side] = { status: "ready", url: asset.url, error: null };
      this._syncToolVariantControls();
      this._invalidateRender("configured-tool-asset-ready");
    });
  }

  _buildArm(side, color, basePosition) {
    const robotModel = this.armModels[side];
    const assets = this.armAssets[side];
    if (!robotModel || !assets) throw new Error(`No ${side} Flexiv model was loaded.`);
    const armRoot = new THREE.Group();
    armRoot.name = `${side}-arm`;
    armRoot.position.copy(basePosition);
    // Both station poses are specified in Flexiv's Z-up world frame: first
    // world-Z 180 deg, then world-Y 90 deg. Convert that world rotation to the
    // Three.js Y-up convention before composing the URDF-frame conversion.
    armRoot.quaternion.copy(armBaseSceneQuaternion());
    const urdfFrame = new THREE.Group();
    urdfFrame.name = `${side}-urdf-z-up-frame`;
    urdfFrame.quaternion.copy(FLEXIV_TO_THREE_QUATERNION);
    armRoot.add(urdfFrame);

    // tcp_pose is an absolute O_T_TCP pose in the shared Flexiv world. It must
    // not inherit this arm's model placement or the base rotation a second
    // time, so the helper is parented directly under the scene root.
    const tcpFrame = new THREE.Group();
    tcpFrame.name = `${side}-active-tcp-frame`;
    tcpFrame.visible = false;
    const tcpAxes = createLabeledAxes(
      `${side}-active-tcp-axes`,
      TCP_AXIS_SIZE,
      0.038,
      new THREE.AxesHelper(TCP_AXIS_SIZE),
    );
    const tcpMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.014, 16, 12),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.55 }),
    );
    tcpMarker.name = `${side}-active-tcp-origin`;
    tcpMarker.userData.viewerHelper = true;
    tcpMarker.userData.armSide = side;
    tcpFrame.add(tcpAxes, tcpMarker);
    this.sceneRoot.add(tcpFrame);

    // The config transform is a static flange_T_tool pose.  It is parented to
    // the URDF flange so the current/home joint hierarchy performs FK.  This
    // frame is hidden whenever a fresh absolute SHM TCP pose is available.
    const configuredToolFrame = new THREE.Group();
    configuredToolFrame.name = `${side}-configured-tool-frame`;
    configuredToolFrame.visible = false;
    const configuredToolAxes = createLabeledAxes(
      `${side}-configured-tool-axes`,
      CONFIG_TOOL_AXIS_SIZE,
      0.034,
      new THREE.AxesHelper(CONFIG_TOOL_AXIS_SIZE),
    );
    const configuredToolMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.012, 16, 12),
      new THREE.MeshStandardMaterial({
        color: 0xffe27a,
        emissive: 0xffb84d,
        emissiveIntensity: 0.5,
      }),
    );
    configuredToolMarker.name = `${side}-configured-tool-origin`;
    configuredToolMarker.userData.viewerHelper = true;
    configuredToolFrame.add(configuredToolAxes, configuredToolMarker);
    const configuredToolVisualFrame = new THREE.Group();
    configuredToolVisualFrame.name = `${side}-configured-tool-visual-frame`;
    configuredToolVisualFrame.visible = false;
    const configuredToolAssetMount = new THREE.Group();
    configuredToolAssetMount.name = `${side}-configured-tool-asset-mount`;
    const configuredToolPlaceholder = createConfiguredToolPlaceholder(side);
    configuredToolPlaceholder.visible = false;
    configuredToolVisualFrame.add(
      configuredToolAssetMount,
      configuredToolPlaceholder,
    );

    const links = new Map();
    robotModel.links.forEach((_, name) => {
      const group = new THREE.Group();
      group.name = `${side}-${name}`;
      links.set(name, group);
    });

    const jointByName = new Map();
    const children = new Set();
    robotModel.joints.forEach((joint) => {
      const parent = links.get(joint.parent);
      const child = links.get(joint.child);
      if (!parent || !child) return;
      const jointGroup = new THREE.Group();
      jointGroup.name = `${side}-${joint.name}-origin`;
      transformFromURDF(joint.transform, jointGroup);
      const motion = new THREE.Group();
      motion.name = `${side}-${joint.name}`;
      motion.userData.axis = joint.axis.clone();
      motion.userData.limit = [joint.lower, joint.upper];
      motion.userData.jointName = joint.name;
      motion.userData.baseQuaternion = motion.quaternion.clone();
      motion.userData.axisAngleQuaternion = new THREE.Quaternion();
      jointGroup.add(motion);
      motion.add(child);
      parent.add(jointGroup);
      if (joint.type !== "fixed") jointByName.set(joint.name, motion);
      children.add(joint.child);
    });

    robotModel.links.forEach((linkData, linkName) => {
      const linkGroup = links.get(linkName);
      linkData.visuals.forEach((visual) => {
        const asset = assets.get(visual.filename);
        if (!asset) return;
        const mesh = cloneAsset(asset, color);
        mesh.name = `${side}-${linkName}-${visual.filename.split("/").pop()}`;
        mesh.scale.copy(visual.scale);
        transformFromURDF(visual.transform, mesh);
        linkGroup.add(mesh);
      });
    });

    const rootName = robotModel.root || "world";
    const rootLink = links.get(rootName) || links.get("base_link");
    if (rootLink) urdfFrame.add(rootLink);
    else throw new Error("No root link found in Flexiv URDF.");
    const flangeLink = links.get("flange");
    if (flangeLink) flangeLink.add(configuredToolFrame, configuredToolVisualFrame);
    // Some generated kinematics URDFs have a world link with a fixed
    // base_joint.  Keep an explicit reference to the base for placement.
    this._setInitialPose(jointByName, side);
    return {
      root: armRoot,
      joints: jointByName,
      urdfFrame,
      tcpFrame,
      configuredToolFrame,
      configuredToolVisualFrame,
      configuredToolAssetMount,
      configuredToolPlaceholder,
      flangeLink: flangeLink || null,
    };
  }

  _setInitialPose(jointByName, side) {
    const pose = this.jointStates[side] || DEFAULT_POSE;
    [...jointByName.values()].slice(0, JOINT_COUNT).forEach((joint, index) => {
      const value = Number.isFinite(pose[index]) ? pose[index] : 0;
      setJointAngle(joint, value);
    });
  }

  _homePoseGhostMaterial(side, sourceMaterial) {
    const cache = this.homePoseGhostMaterialCaches[side];
    const key = sourceMaterial?.uuid || "__fallback__";
    if (!cache.has(key)) {
      const color = side === "left" ? LEFT_COLOR : RIGHT_COLOR;
      cache.set(key, homePoseGhostMaterial(sourceMaterial, color));
    }
    return cache.get(key);
  }

  _cloneHomePoseGhostAsset(asset, side) {
    const clone = asset.clone(true);
    clone.traverse((node) => {
      if (!node.isMesh) return;
      if (Array.isArray(node.material)) {
        node.material = node.material.map((material) => this._homePoseGhostMaterial(side, material));
      } else {
        node.material = this._homePoseGhostMaterial(side, node.material);
      }
      node.castShadow = false;
      node.receiveShadow = false;
      node.userData.homePoseGhost = true;
    });
    return clone;
  }

  _buildHomePoseGhost(side, candidate, index) {
    const robotModel = this.armModels[side];
    const assets = this.armAssets[side];
    if (!robotModel || !assets) throw new Error(`No ${side} ghost model assets were loaded.`);
    if (robotModel.variant !== ARM_MODEL_SPECS[side].variant) {
      throw new Error(`Unexpected ${side} ghost model variant: ${robotModel.variant}`);
    }

    const prefix = `${side}-home-pose-ghost-${index + 1}`;
    const armRoot = new THREE.Group();
    armRoot.name = prefix;
    armRoot.position.copy(flexivWorldToThreePosition(ARM_BASE_POSITIONS[side]));
    armRoot.quaternion.copy(armBaseSceneQuaternion());
    armRoot.userData.homePoseGhost = true;
    armRoot.userData.candidateId = candidate.id;
    armRoot.userData.model = robotModel.variant;

    const urdfFrame = new THREE.Group();
    urdfFrame.name = `${prefix}-urdf-z-up-frame`;
    urdfFrame.quaternion.copy(FLEXIV_TO_THREE_QUATERNION);
    armRoot.add(urdfFrame);

    const links = new Map();
    robotModel.links.forEach((_, name) => {
      const group = new THREE.Group();
      group.name = `${prefix}-${name}`;
      links.set(name, group);
    });

    const jointByName = new Map();
    robotModel.joints.forEach((joint) => {
      const parent = links.get(joint.parent);
      const child = links.get(joint.child);
      if (!parent || !child) return;
      const jointGroup = new THREE.Group();
      jointGroup.name = `${prefix}-${joint.name}-origin`;
      transformFromURDF(joint.transform, jointGroup);
      const motion = new THREE.Group();
      motion.name = `${prefix}-${joint.name}`;
      motion.userData.axis = joint.axis.clone();
      motion.userData.limit = [joint.lower, joint.upper];
      motion.userData.jointName = joint.name;
      motion.userData.baseQuaternion = motion.quaternion.clone();
      motion.userData.axisAngleQuaternion = new THREE.Quaternion();
      jointGroup.add(motion);
      motion.add(child);
      parent.add(jointGroup);
      if (joint.type !== "fixed") jointByName.set(joint.name, motion);
    });

    robotModel.links.forEach((linkData, linkName) => {
      const linkGroup = links.get(linkName);
      linkData.visuals.forEach((visual) => {
        const asset = assets.get(visual.filename);
        if (!asset) return;
        const mesh = this._cloneHomePoseGhostAsset(asset, side);
        mesh.name = `${prefix}-${linkName}-${visual.filename.split("/").pop()}`;
        mesh.scale.copy(visual.scale);
        transformFromURDF(visual.transform, mesh);
        linkGroup.add(mesh);
      });
    });

    const rootName = robotModel.root || "world";
    const rootLink = links.get(rootName) || links.get("base_link");
    if (!rootLink) throw new Error(`No ${side} ghost root link found.`);
    urdfFrame.add(rootLink);
    const flangeLink = links.get("flange") || null;
    let flangePoint = null;
    if (flangeLink) {
      flangePoint = new THREE.Mesh(
        this.homePoseGhostPointGeometry,
        this.homePoseGhostPointMaterials[side],
      );
      flangePoint.name = `${prefix}-flange-point`;
      flangePoint.userData.viewerHelper = true;
      flangePoint.userData.homePoseGhostFlangePoint = true;
      flangeLink.add(flangePoint);
    }

    [...jointByName.values()].slice(0, JOINT_COUNT).forEach((joint, jointIndex) => {
      const lower = joint.userData.limit?.[0] ?? -Math.PI;
      const upper = joint.userData.limit?.[1] ?? Math.PI;
      setJointAngle(joint, THREE.MathUtils.clamp(candidate.joints[jointIndex], lower, upper));
    });
    freezeStaticHierarchy(armRoot);
    return Object.freeze({
      id: candidate.id,
      root: armRoot,
      joints: jointByName,
      flangePoint,
      model: robotModel.variant,
    });
  }

  _clearHomePoseGhosts() {
    const hadGhosts = this._renderedHomePoseCandidateSignature !== null
      || this.homePoseGhostEntries.left.length > 0
      || this.homePoseGhostEntries.right.length > 0;
    for (const side of ["left", "right"]) {
      this.homePoseGhostSideRoots[side].clear();
      this.homePoseGhostSideRoots[side].visible = false;
      this.homePoseGhostEntries[side] = [];
    }
    this._renderedHomePoseCandidateSignature = null;
    if (hadGhosts) this._invalidateRender("ghost-clear");
  }

  _rebuildHomePoseGhosts(signature) {
    this._clearHomePoseGhosts();
    if (!this._modelLoaded) return;
    this._perf.ghost_rebuilds += 1;
    for (const side of ["left", "right"]) {
      const armState = this.homePoseCandidateState.arms[side];
      if (armState.status !== "ready" || armState.model !== ARM_MODEL_SPECS[side].variant) continue;
      for (const candidate of armState.candidates.slice(0, MAX_HOME_POSE_GHOSTS_PER_SIDE)) {
        try {
          const entry = this._buildHomePoseGhost(
            side,
            candidate,
            this.homePoseGhostEntries[side].length,
          );
          this.homePoseGhostEntries[side].push(entry);
          this.homePoseGhostSideRoots[side].add(entry.root);
        } catch (_error) {
          // One malformed candidate must not take down the solid-arm viewer.
        }
      }
    }
    this._renderedHomePoseCandidateSignature = signature;
    this._syncHomePoseGhostVisibility();
    this._invalidateRender("ghost-rebuild");
  }

  updateHomePoseCandidates(value) {
    const state = normalizedHomePoseCandidates(value);
    this.homePoseCandidateState = state;
    const renderable = (state.status === "ready" || state.status === "partial") && state.key;
    if (!renderable) {
      this._clearHomePoseGhosts();
      return this;
    }
    const signature = `${state.status}:${state.key}`;
    if (signature !== this._renderedHomePoseCandidateSignature) {
      this._rebuildHomePoseGhosts(signature);
    } else {
      this._syncHomePoseGhostVisibility();
    }
    return this;
  }

  _syncHomePoseGhostVisibility() {
    const state = this.homePoseCandidateState;
    const globallyRenderable = Boolean(
      (state.status === "ready" || state.status === "partial") && state.key,
    );
    for (const side of ["left", "right"]) {
      const armState = state.arms[side];
      const visible = Boolean(
        globallyRenderable
        && armState.status === "ready"
        && armState.model === ARM_MODEL_SPECS[side].variant
        && this.homeJointStates[side]
        && this.homePoseGhostEntries[side].length
        && !this.homePoseGhostFresh[side],
      );
      if (this.homePoseGhostSideRoots[side].visible !== visible) {
        this.homePoseGhostSideRoots[side].visible = visible;
        this._perf.ghost_visibility_changes += 1;
        this._invalidateRender("ghost-visibility");
      }
    }
  }

  updateYamState(value) {
    const next = normalizedYamState(value);
    this.yamState = next;
    if (!next.configured) {
      if (this.yam?.root?.visible) {
        this.yam.root.visible = false;
        if (this._modelLoaded) this._frameModel();
        this._invalidateRender("yam-hidden");
      }
      this._showRobotStateStatus();
      return this;
    }
    if (this.yam) {
      this.yam.root.visible = true;
      this._applyYamPose();
      this._showRobotStateStatus();
      return this;
    }
    void this._ensureYamLoaded();
    return this;
  }

  _ensureYamLoaded() {
    if (this.yam || !this.yamState.configured) return Promise.resolve(this.yam);
    if (this._yamLoadPromise) return this._yamLoadPromise;
    const epoch = ++this._yamLoadEpoch;
    this.yamAssetState = { status: "loading", error: null };
    this._perf.yam_asset_loads += 1;
    this._showRobotStateStatus();
    this._yamLoadPromise = this._loadToolUrdf(this.yamState.urdfUrl)
      .then((modelRoot) => {
        if (this._disposed || epoch !== this._yamLoadEpoch) {
          disposeObjectResources(modelRoot);
          return null;
        }
        const root = new THREE.Group();
        root.name = "yam-head-arm";
        root.quaternion.copy(composeYamBaseSceneQuaternion());
        root.add(modelRoot);
        this.yam = {
          root,
          modelRoot,
          joints: modelRoot.userData.toolJoints instanceof Map
            ? modelRoot.userData.toolJoints
            : initializeToolJointRuntime(modelRoot),
        };
        this.sceneRoot.add(root);
        this.yamAssetState = { status: "ready", error: null };
        root.visible = this.yamState.configured;
        this._applyYamPose(true);
        if (this._modelLoaded) this._frameModel();
        this._showRobotStateStatus();
        this._invalidateRender("yam-asset-ready");
        return this.yam;
      })
      .catch((error) => {
        if (!this._disposed && epoch === this._yamLoadEpoch) {
          this._perf.yam_asset_failures += 1;
          this.yamAssetState = {
            status: "error",
            error: String(error?.message || error).slice(0, 256),
          };
          this._showRobotStateStatus();
          this._invalidateRender("yam-asset-error");
        }
        return null;
      })
      .finally(() => {
        if (epoch === this._yamLoadEpoch) this._yamLoadPromise = null;
      });
    return this._yamLoadPromise;
  }

  _applyYamPose(force = false) {
    if (!this.yam || !this.yamState.configured) return false;
    const target = this.yamState.liveJoints
      || this.yamState.initialJoints
      || new Array(YAM_MODEL_SPEC.jointCount).fill(0);
    const scenePosition = flexivWorldToThreePosition(this.yamState.basePositionWorld);
    const baseChanged = this.yam.root.position.distanceToSquared(scenePosition) > 1e-12;
    if (baseChanged) this.yam.root.position.copy(scenePosition);
    const unchanged = numericVectorsNearlyEqual(
      this.yamJointState,
      target,
      JOINT_UPDATE_EPSILON_RAD,
    );
    if (!force && unchanged && !baseChanged) {
      this._perf.yam_skips += 1;
      return false;
    }
    YAM_MODEL_SPEC.jointNames.forEach((name, index) => {
      const joint = this.yam.joints.get(name);
      if (!joint) return;
      const lower = joint.userData.limit?.[0] ?? -Math.PI;
      const upper = joint.userData.limit?.[1] ?? Math.PI;
      setJointAngle(joint, THREE.MathUtils.clamp(target[index], lower, upper));
    });
    this.yamJointState = [...target];
    this.yam.root.userData.poseSource = this.yamState.source;
    this.yam.root.updateMatrixWorld(true);
    this._perf.yam_applies += 1;
    this._invalidateRender("yam-joints");
    return true;
  }

  updateJointStates(left, right) {
    if (left !== undefined) this._applyJointState("left", left);
    if (right !== undefined) this._applyJointState("right", right);
    return this;
  }

  updateTcpPoses(left, right) {
    if (left !== undefined) this._applyTcpPose("left", left);
    if (right !== undefined) this._applyTcpPose("right", right);
    return this;
  }

  updateConfiguredTools(left, right) {
    if (left !== undefined) this._applyConfiguredTool("left", left);
    if (right !== undefined) this._applyConfiguredTool("right", right);
    return this;
  }

  updateGripperPositions(left, right) {
    let leftValue = left;
    let rightValue = right;
    const envelope = right === undefined
      && left
      && typeof left === "object"
      && (
        left.sides
        || Object.prototype.hasOwnProperty.call(left, "left")
        || Object.prototype.hasOwnProperty.call(left, "right")
        || left.hardware_unknown === true
        || left.hardwareUnknown === true
        || left.busy === true
        || left.stale === true
      );
    if (envelope) {
      const sides = left.sides && typeof left.sides === "object" ? left.sides : left;
      leftValue = undefined;
      rightValue = undefined;
      if (Object.prototype.hasOwnProperty.call(sides, "left")) leftValue = sides.left;
      if (Object.prototype.hasOwnProperty.call(sides, "right")) rightValue = sides.right;
      if (
        left.hardware_unknown === true
        || left.hardwareUnknown === true
        || left.busy === true
        || left.stale === true
      ) {
        leftValue = { known: false, source: "control-panel-unknown" };
        rightValue = { known: false, source: "control-panel-unknown" };
      }
    }
    if (leftValue !== undefined) this._applyGripperPosition("left", leftValue);
    if (rightValue !== undefined) this._applyGripperPosition("right", rightValue);
    return this;
  }

  updateGripperStates(raw, { source = "external-update" } = {}) {
    const envelope = raw && typeof raw === "object" ? raw : {};
    const sides = envelope.sides && typeof envelope.sides === "object" ? envelope.sides : envelope;
    const withSource = (value) => {
      if (value && typeof value === "object") {
        const enriched = value.source ? value : { ...value, source };
        if (value.busy === true || value.stale === true || value.idle === false) {
          return { ...enriched, known: false };
        }
        return enriched;
      }
      return {
        value,
        known: value !== undefined && value !== null && value !== "" && Number.isFinite(Number(value)),
        source,
      };
    };
    const payload = {
      hardware_unknown: envelope.hardware_unknown ?? envelope.hardwareUnknown,
      busy: envelope.busy === true,
      stale: envelope.stale === true,
    };
    if (Object.prototype.hasOwnProperty.call(sides, "left")) payload.left = withSource(sides.left);
    if (Object.prototype.hasOwnProperty.call(sides, "right")) payload.right = withSource(sides.right);
    return this.updateGripperPositions(payload);
  }

  _applyGripperPosition(side, value) {
    const articulation = this.configuredTools[side]?.visual?.articulation || null;
    const next = normalizedGripperPosition(value, articulation, "external-update");
    const previous = this.gripperVisualStates[side];
    const now = Date.now();
    const isDirectSource = next.source === "control-panel"
      || next.source.startsWith("device-control")
      || next.source.startsWith("gripper-control");
    const isRobotStateSource = next.source.startsWith("robot-state");
    this.gripperDirectSourceFreshUntil ||= { left: 0, right: 0 };
    this.gripperDirectValues ||= { left: null, right: null };
    if (isDirectSource) {
      this.gripperDirectSourceFreshUntil[side] = now + GRIPPER_DIRECT_SOURCE_TTL_MS;
      const directPercent = value && typeof value === "object"
        ? Number(value.actual_percent ?? value.actualPercent ?? value.percent)
        : Number.NaN;
      if (next.known || Number.isFinite(directPercent)) {
        this.gripperDirectValues[side] = value && typeof value === "object"
          ? { ...value, source: next.source }
          : { value, source: next.source };
      }
    } else if (
      isRobotStateSource
      && now < (this.gripperDirectSourceFreshUntil[side] || 0)
    ) {
      // The robot SHM may contain the controller's requested/default gripper
      // width rather than the independently measured serial position. Keep the
      // latest direct observation authoritative while it remains fresh.
      this._perf.gripper_skips += 1;
      return;
    }
    if (!next.known) {
      const knownChanged = previous?.known !== false;
      this.gripperVisualStates[side] = {
        ...previous,
        known: false,
        source: next.source,
      };
      this._perf.gripper_unknowns += 1;
      const mounted = this.toolAssetObjects[side];
      if (mounted) mounted.userData.gripperKnown = false;
      if (knownChanged) this._showRobotStateStatus();
      return;
    }
    const knownChanged = previous?.known !== true;
    const unchanged = Number.isFinite(previous?.jointAngle)
      && Math.abs(previous.jointAngle - next.jointAngle) <= GRIPPER_UPDATE_EPSILON_RAD;
    this.gripperVisualStates[side] = { ...next };
    if (unchanged) {
      this._perf.gripper_skips += 1;
      const mounted = this.toolAssetObjects[side];
      if (mounted) {
        mounted.userData.gripperKnown = true;
        mounted.userData.gripperSource = next.source;
      }
      if (knownChanged) this._showRobotStateStatus();
      return;
    }
    this._perf.gripper_applies += 1;
    this._applyGripperStateToMountedTool(side);
    if (knownChanged) this._showRobotStateStatus();
  }

  _applyGripperStateToMountedTool(side, force = false) {
    const articulation = this.configuredTools[side]?.visual?.articulation;
    const joint = articulation ? this.toolJointByName[side]?.get(articulation.jointName) : null;
    const mounted = this.toolAssetObjects[side];
    if (!articulation || !joint || !mounted) return false;
    const state = this.gripperVisualStates[side];
    const neutral = articulation.closedJointRad
      + TOOL_JOINT_NEUTRAL_FRACTION * (articulation.openJointRad - articulation.closedJointRad);
    const target = THREE.MathUtils.clamp(
      Number.isFinite(state?.jointAngle) ? state.jointAngle : neutral,
      Math.max(articulation.jointLowerRad, joint.userData.limit?.[0] ?? articulation.jointLowerRad),
      Math.min(articulation.jointUpperRad, joint.userData.limit?.[1] ?? articulation.jointUpperRad),
    );
    const unchanged = Number.isFinite(joint.userData.gripperJointAngle)
      && Math.abs(joint.userData.gripperJointAngle - target) <= GRIPPER_UPDATE_EPSILON_RAD;
    mounted.userData.gripperKnown = state?.known === true;
    mounted.userData.gripperSource = state?.source || "unknown";
    mounted.userData.gripperJointName = articulation.jointName;
    mounted.userData.gripperJointAngle = target;
    if (!force && unchanged) return false;
    setJointAngle(joint, target);
    joint.userData.gripperJointAngle = target;
    mounted.updateMatrixWorld(true);
    this._invalidateRender("gripper-position");
    return true;
  }

  _applyConfiguredTool(side, value) {
    const descriptor = normalizedConfiguredTool(value);
    this.configuredToolDescriptors ||= {
      left: EMPTY_CONFIGURED_TOOL,
      right: EMPTY_CONFIGURED_TOOL,
    };
    this.configuredToolDescriptors[side] = descriptor;
    const tool = selectConfiguredToolVariant(descriptor, this.toolVisualVariant);
    this.configuredTools[side] = tool;
    this._syncToolVariantControls?.();
    const articulationSignature = toolArticulationSignature(tool.visual?.articulation);
    if (this.gripperArticulationSignatures[side] !== articulationSignature) {
      this.gripperArticulationSignatures[side] = articulationSignature;
      this.gripperDirectSourceFreshUntil ||= { left: 0, right: 0 };
      this.gripperDirectValues ||= { left: null, right: null };
      this.gripperVisualStates[side] = {
        known: false,
        rawValue: null,
        actualPercent: null,
        jointAngle: null,
        source: "tool-config-changed",
      };
      const directValue = this.gripperDirectValues[side];
      if (
        directValue
        && Date.now() < (this.gripperDirectSourceFreshUntil[side] || 0)
      ) {
        this._applyGripperPosition(side, directValue);
      } else {
        this.gripperDirectValues[side] = null;
        this.gripperDirectSourceFreshUntil[side] = 0;
      }
    }
    const frame = this.arms[side]?.configuredToolFrame;
    if (!frame) return;
    const signature = configuredToolSignature(tool);
    if (frame.userData.toolSignature === signature) {
      this._perf.tool_skips += 1;
      return;
    }
    frame.userData.toolSignature = signature;
    this._perf.tool_applies += 1;
    this._updateConfiguredToolAsset(side, tool);
    const transform = tool.status === "available" ? tool.transform : null;
    const visualTransform = configuredToolVisualTransform(tool);
    if ((!transform && !visualTransform) || !this.arms[side]?.flangeLink) {
      frame.visible = false;
      if (this.arms[side]?.configuredToolVisualFrame) {
        this.arms[side].configuredToolVisualFrame.visible = false;
        this.arms[side].configuredToolVisualFrame.userData.configured = false;
      }
      frame.userData.configured = false;
      frame.userData.toolName = null;
      frame.userData.toolSource = null;
      this._syncTcpFrameVisibility(side);
      this._invalidateRender("configured-tool");
      return;
    }
    if (transform) {
      const [x, y, z] = transform.translation;
      const [qw, qx, qy, qz] = transform.quaternion;
      frame.position.set(x, y, z);
      frame.quaternion.set(qx, qy, qz, qw).normalize();
    }
    const visualFrame = this.arms[side]?.configuredToolVisualFrame;
    if (visualFrame && visualTransform) {
      const [x, y, z] = visualTransform.translation;
      const [qw, qx, qy, qz] = visualTransform.quaternion;
      visualFrame.position.set(x, y, z);
      visualFrame.quaternion.set(qx, qy, qz, qw).normalize();
      visualFrame.userData.configured = true;
    }
    frame.userData.configured = Boolean(transform);
    frame.userData.toolName = tool.resolvedToolName;
    frame.userData.toolSource = tool.resolvedToolSource;
    this._syncTcpFrameVisibility(side);
    this._invalidateRender("configured-tool");
  }

  _syncTcpFrameVisibility(side) {
    const arm = this.arms[side];
    if (!arm) return;
    const hasFreshAbsoluteTcp = Boolean(this.tcpStates[side]);
    const configuredVisible = Boolean(
      !hasFreshAbsoluteTcp
      && arm.configuredToolFrame.userData.configured
      && arm.flangeLink,
    );
    const visualVisible = Boolean(
      arm.configuredToolVisualFrame.userData.configured && arm.flangeLink,
    );
    const changed = arm.tcpFrame.visible !== hasFreshAbsoluteTcp
      || arm.configuredToolFrame.visible !== configuredVisible
      || arm.configuredToolVisualFrame.visible !== visualVisible;
    arm.tcpFrame.visible = hasFreshAbsoluteTcp;
    arm.configuredToolFrame.visible = configuredVisible;
    arm.configuredToolVisualFrame.visible = visualVisible;
    if (changed) this._invalidateRender("tcp-visibility");
  }

  _applyTcpPose(side, values) {
    const pose = normalizedTcpPose(values);
    this.tcpStates[side] = pose;
    const frame = this.arms[side]?.tcpFrame;
    if (!frame) return;
    if (numericVectorsNearlyEqual(frame.userData.pose, pose, TCP_UPDATE_EPSILON)) {
      this._perf.tcp_skips += 1;
      return;
    }
    frame.userData.pose = pose ? [...pose] : null;
    this._perf.tcp_applies += 1;
    if (!pose) {
      frame.userData.fresh = false;
      this._syncTcpFrameVisibility(side);
      this._invalidateRender("tcp-pose");
      return;
    }
    const sceneTransform = absoluteTcpPoseToSceneTransform(pose, side);
    frame.position.copy(sceneTransform.position);
    frame.quaternion.copy(sceneTransform.quaternion);
    frame.userData.fresh = true;
    this._syncTcpFrameVisibility(side);
    this._invalidateRender("tcp-pose");
  }

  invalidateGripperDirectSources(sides = ["left", "right"]) {
    this.gripperDirectSourceFreshUntil ||= { left: 0, right: 0 };
    this.gripperDirectValues ||= { left: null, right: null };
    for (const side of sides) {
      if (side !== "left" && side !== "right") continue;
      this.gripperDirectSourceFreshUntil[side] = 0;
      this.gripperDirectValues[side] = null;
    }
    return this;
  }

  setArmModelsVisible(visible) {
    this.armModelsVisible = Boolean(visible);
    for (const side of ["left", "right"]) {
      if (this.arms[side]?.root) {
        this.arms[side].root.visible = this.armModelsVisible;
      }
    }
    this._invalidateRender("arm-model-visibility");
    return this;
  }

  setRecordedTcpTrajectories(left, right) {
    const allPoints = [];
    for (const [side, values] of [["left", left], ["right", right]]) {
      const visual = this.recordedTcpTrajectoryVisuals[side];
      const points = recordedTcpScenePoints(values, side);
      allPoints.push(...points);
      replaceStaticLinePoints(visual.full, points);
      visual.start.visible = points.length > 0;
      visual.end.visible = points.length > 0;
      if (points.length) {
        visual.start.position.copy(points[0]);
        visual.end.position.copy(points[points.length - 1]);
      }
    }
    this.recordedTcpScenePoints = allPoints;
    this.recordedTcpTrajectoryRoot.visible = true;
    if (this._modelLoaded) {
      if (this.armModelsVisible) this._frameModel();
      else this._frameRecordedTcpTrajectories(allPoints);
    }
    this._invalidateRender("recorded-tcp-trajectory");
    return this;
  }

  updateRecordedTcpTrails(left, right) {
    for (const [side, values] of [["left", left], ["right", right]]) {
      updateDynamicLinePoints(
        this.recordedTcpTrajectoryVisuals[side].trail,
        recordedTcpScenePoints(values, side),
      );
    }
    this._invalidateRender("recorded-tcp-trail");
    return this;
  }

  clearRecordedTcpTrajectories() {
    this.recordedTcpScenePoints = [];
    for (const visual of Object.values(this.recordedTcpTrajectoryVisuals)) {
      replaceStaticLinePoints(visual.full, []);
      updateDynamicLinePoints(visual.trail, []);
      visual.start.visible = false;
      visual.end.visible = false;
    }
    this._invalidateRender("recorded-tcp-clear");
    return this;
  }

  _applyJointState(side, values) {
    const arm = this.arms[side];
    const normalized = this._normalizeJointState(values, arm?.joints);
    const unchanged = numericVectorsNearlyEqual(
      this.jointStates[side],
      normalized,
      JOINT_UPDATE_EPSILON_RAD,
    );
    this.jointStates[side] = normalized;
    if (!arm || unchanged) {
      if (arm && unchanged) this._perf.joint_skips += 1;
      return;
    }
    this._perf.joint_applies += 1;
    let index = 0;
    arm.joints.forEach((joint) => {
      const raw = normalized[index++] ?? 0;
      const lower = joint.userData.limit?.[0] ?? -Math.PI;
      const upper = joint.userData.limit?.[1] ?? Math.PI;
      const value = THREE.MathUtils.clamp(Number(raw) || 0, lower, upper);
      setJointAngle(joint, value);
    });
    this._invalidateRender("joint-state");
  }

  _normalizeJointState(values, joints) {
    if (Array.isArray(values) || ArrayBuffer.isView(values)) {
      return Array.from(values).slice(0, JOINT_COUNT).map((value) => Number(value) || 0);
    }
    const result = [];
    const names = joints ? [...joints.keys()] : [];
    for (let index = 0; index < JOINT_COUNT; index += 1) {
      const name = names[index] || `joint${index + 1}`;
      result.push(Number(values?.[name] ?? values?.[index] ?? 0) || 0);
    }
    return result;
  }

  resetView() {
    if (this._defaultView) {
      this.camera.position.copy(this._defaultView.position);
      this.controls.target.copy(this._defaultView.target);
    } else {
      this.camera.position.set(2.45, 2.15, 3.55);
      this.controls.target.copy(groundScenePosition());
    }
    this.camera.lookAt(this.controls.target);
    this.controls.update();
    this._invalidateRender("reset-view");
  }

  _frameModel() {
    const bounds = new THREE.Box3().setFromObject(this.sceneRoot);
    if (bounds.isEmpty()) return;
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * Math.max(0.1, this.camera.aspect));
    const fitHeight = (size.y * DEFAULT_VIEW_PADDING * 0.5) / Math.tan(verticalFov / 2);
    const fitWidth = (size.x * DEFAULT_VIEW_PADDING * 0.5) / Math.tan(horizontalFov / 2);
    const defaultViewTarget = groundScenePosition();
    const distance = Math.max(2.25, fitHeight, fitWidth) + size.z * 0.5;
    this.controls.target.copy(defaultViewTarget);
    // A slightly diagonal view makes Flexiv-world Y separation visible after
    // it maps to the Three.js depth axis, while retaining stable framing.
    this.camera.position.set(
      center.x + distance * 0.38,
      center.y + distance * DEFAULT_CAMERA_ELEVATION,
      center.z + distance * 0.92,
    );
    this.camera.lookAt(defaultViewTarget);
    this.camera.near = 0.01;
    this.camera.far = Math.max(30, distance + size.length() * 6);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this._defaultView = Object.freeze({
      position: this.camera.position.clone(),
      target: this.controls.target.clone(),
    });
    this._invalidateRender("frame-model");
  }

  _frameRecordedTcpTrajectories(points) {
    if (!Array.isArray(points) || !points.length) return;
    const bounds = new THREE.Box3().setFromPoints(points).expandByScalar(0.16);
    if (bounds.isEmpty()) return;
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
    const horizontalFov = 2 * Math.atan(
      Math.tan(verticalFov / 2) * Math.max(0.1, this.camera.aspect),
    );
    const fitHeight = (size.y * 1.45 * 0.5) / Math.tan(verticalFov / 2);
    const fitWidth = (size.x * 1.45 * 0.5) / Math.tan(horizontalFov / 2);
    const distance = Math.max(0.65, fitHeight, fitWidth) + size.z * 0.5;
    this.controls.target.copy(center);
    this.camera.position.set(
      center.x + distance * 0.42,
      center.y + distance * 0.48,
      center.z + distance * 0.9,
    );
    this.camera.lookAt(center);
    this.camera.near = 0.01;
    this.camera.far = Math.max(10, distance + size.length() * 8);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this._defaultView = Object.freeze({
      position: this.camera.position.clone(),
      target: center.clone(),
    });
    this._invalidateRender("frame-recorded-tcp");
  }

  getDebugSnapshot() {
    const leftRoot = this.arms.left?.root || null;
    const rightRoot = this.arms.right?.root || null;
    this.scene.updateMatrixWorld(true);
    return Object.freeze({
      loadedArmCount: [leftRoot, rightRoot].filter((root) => root?.parent === this.sceneRoot).length,
      meshCounts: Object.freeze({ left: meshCount(leftRoot), right: meshCount(rightRoot) }),
      modelVariants: Object.freeze({
        left: this.armModels.left?.variant || null,
        right: this.armModels.right?.variant || null,
      }),
      assetCounts: Object.freeze({
        left: this.armAssets.left.size,
        right: this.armAssets.right.size,
      }),
      baseQuaternionsXyzw: Object.freeze({
        left: leftRoot ? quaternionSnapshot(leftRoot.quaternion) : null,
        right: rightRoot ? quaternionSnapshot(rightRoot.quaternion) : null,
      }),
      modelQuaternionsXyzw: Object.freeze({
        left: this.arms.left?.urdfFrame
          ? quaternionSnapshot(this.arms.left.urdfFrame.getWorldQuaternion(new THREE.Quaternion()))
          : null,
        right: this.arms.right?.urdfFrame
          ? quaternionSnapshot(this.arms.right.urdfFrame.getWorldQuaternion(new THREE.Quaternion()))
          : null,
      }),
      worldBounds: Object.freeze({
        left: boundsSnapshot(leftRoot),
        right: boundsSnapshot(rightRoot),
        combined: boundsSnapshot(this.sceneRoot),
      }),
      tcpFrames: Object.freeze({
        left: tcpFrameSnapshot(this.arms.left),
        right: tcpFrameSnapshot(this.arms.right),
      }),
      configuredToolFrames: Object.freeze({
        left: configuredToolFrameSnapshot(this.arms.left),
        right: configuredToolFrameSnapshot(this.arms.right),
      }),
      toolAssets: Object.freeze({
        left: Object.freeze({
          ...this.toolAssetStates.left,
          visualVisible: Boolean(this.arms.left?.configuredToolVisualFrame?.visible),
          meshCount: meshCount(this.toolAssetObjects.left),
        }),
        right: Object.freeze({
          ...this.toolAssetStates.right,
          visualVisible: Boolean(this.arms.right?.configuredToolVisualFrame?.visible),
          meshCount: meshCount(this.toolAssetObjects.right),
        }),
      }),
      gripperPositions: Object.freeze({
        left: Object.freeze({ ...this.gripperVisualStates.left }),
        right: Object.freeze({ ...this.gripperVisualStates.right }),
      }),
      yam: Object.freeze({
        configured: this.yamState.configured,
        model: this.yamState.model,
        status: this.yamState.status,
        source: this.yamState.source,
        fresh: this.yamState.fresh,
        asset: Object.freeze({ ...this.yamAssetState }),
        visible: Boolean(this.yam?.root?.visible),
        meshCount: meshCount(this.yam?.root),
        base: Object.freeze({
          flexivWorld: this.yamState.basePositionWorld,
          scene: this.yam?.root ? vectorSnapshot(this.yam.root.position) : null,
        }),
        joints: this.yamJointState ? Object.freeze([...this.yamJointState]) : null,
        bounds: boundsSnapshot(this.yam?.root),
      }),
      armBasePositions: Object.freeze({
        left: Object.freeze({
          flexivWorld: ARM_BASE_POSITIONS.left,
          scene: leftRoot ? vectorSnapshot(leftRoot.position) : null,
        }),
        right: Object.freeze({
          flexivWorld: ARM_BASE_POSITIONS.right,
          scene: rightRoot ? vectorSnapshot(rightRoot.position) : null,
        }),
      }),
      groundPlane: Object.freeze({
        flexivWorld: GROUND_FLEXIV_POSITION,
        floorScene: this.floor ? vectorSnapshot(this.floor.position) : null,
        gridScene: this.grid ? vectorSnapshot(this.grid.position) : null,
      }),
      viewTarget: Object.freeze({
        flexivWorld: GROUND_FLEXIV_POSITION,
        scene: vectorSnapshot(groundScenePosition()),
        controlsScene: this.controls ? vectorSnapshot(this.controls.target) : null,
        defaultScene: this._defaultView ? vectorSnapshot(this._defaultView.target) : null,
      }),
      worldFramePosition: this.worldFrame
        ? vectorSnapshot(this.worldFrame.position)
        : null,
      cameraPosition: vectorSnapshot(this.camera.position),
      identityLabels: Object.freeze({ left: null, right: null }),
      homePoseGhosts: Object.freeze({
        status: this.homePoseCandidateState.status,
        key: this.homePoseCandidateState.key,
        fresh: Object.freeze({ ...this.homePoseGhostFresh }),
        counts: Object.freeze({
          left: this.homePoseGhostEntries.left.length,
          right: this.homePoseGhostEntries.right.length,
        }),
        visible: Object.freeze({
          left: this.homePoseGhostSideRoots.left.visible,
          right: this.homePoseGhostSideRoots.right.visible,
        }),
        materialCounts: Object.freeze({
          left: this.homePoseGhostMaterialCaches.left.size,
          right: this.homePoseGhostMaterialCaches.right.size,
        }),
        models: Object.freeze({
          left: this.homePoseCandidateState.arms.left.model,
          right: this.homePoseCandidateState.arms.right.model,
        }),
        candidateIds: Object.freeze({
          left: Object.freeze(this.homePoseGhostEntries.left.map((entry) => entry.id)),
          right: Object.freeze(this.homePoseGhostEntries.right.map((entry) => entry.id)),
        }),
        flangePointCounts: Object.freeze({
          left: this.homePoseGhostEntries.left.filter((entry) => entry.flangePoint).length,
          right: this.homePoseGhostEntries.right.filter((entry) => entry.flangePoint).length,
        }),
      }),
      performance: Object.freeze({
        scheduler: this._renderLoop?.snapshot() || null,
        renderer: rendererInfoSnapshot(this.renderer),
        counters: Object.freeze({ ...this._perf }),
        pixelRatio: this.renderer?.getPixelRatio() ?? null,
        visible: this._canWork(),
        renderSize: Object.freeze([this._renderWidth, this._renderHeight]),
        statePollIntervalMs: this._nextStatePollDelay(),
        candidatePollIntervalMs: this._nextHomePoseCandidatePollDelay(),
      }),
      toolNames: Object.freeze({ ...this.toolNames }),
      configuredTools: Object.freeze({ ...this.configuredTools }),
    });
  }

  _resize() {
    if (!this.renderer || !this.canvasHost) return;
    const width = Math.max(1, this.canvasHost.clientWidth || this.root.clientWidth || 1);
    const height = Math.max(1, this.canvasHost.clientHeight || this.root.clientHeight || 1);
    if (width === this._renderWidth && height === this._renderHeight) {
      this._perf.resize_skips += 1;
      return;
    }
    this._renderWidth = width;
    this._renderHeight = height;
    this._perf.resize_applies += 1;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    this._invalidateRender("resize");
  }

  _setState(state, message) {
    if (!this.status) return;
    if (this.status.dataset.state === state && this.status.textContent === message) return;
    this.status.dataset.state = state;
    this.status.textContent = message;
  }

  _canWork() {
    return !this._disposed && this._pageVisible && this._panelVisible;
  }

  _invalidateRender(reason) {
    this._renderLoop?.invalidate(reason);
  }

  _handleVisibilityChange() {
    this._pageVisible = typeof document === "undefined" || document.visibilityState !== "hidden";
    clearTimeout(this._statePollTimer);
    clearTimeout(this._homePoseCandidatePollTimer);
    if (!this._canWork()) {
      this._renderLoop?.pause();
      this._stateRequestController?.abort();
      this._homePoseCandidateRequestController?.abort();
      return;
    }
    this._renderLoop?.resume();
    this._invalidateRender("visibility-resume");
    this._scheduleStatePoll(0);
    this._scheduleHomePoseCandidatePoll(0);
  }

  _nextStatePollDelay() {
    return this._liveStateAvailable ? this.statePollIntervalMs : Math.max(
      this.statePollIntervalMs,
      IDLE_STATE_POLL_INTERVAL_MS,
    );
  }

  _nextHomePoseCandidatePollDelay() {
    const state = this.homePoseCandidateState;
    if ((state.status === "ready" || state.status === "partial") && state.key) {
      return Math.max(this.homePoseCandidatePollIntervalMs, READY_HOME_POSE_CANDIDATE_POLL_INTERVAL_MS);
    }
    if (state.status === "generating") return this.homePoseCandidatePollIntervalMs;
    return Math.max(this.homePoseCandidatePollIntervalMs, IDLE_HOME_POSE_CANDIDATE_POLL_INTERVAL_MS);
  }

  _scheduleStatePoll(delay = this._nextStatePollDelay()) {
    if (!this._canWork() || !this.robotStateUrl) return;
    clearTimeout(this._statePollTimer);
    this._statePollTimer = window.setTimeout(() => this._pollRobotState(), delay);
  }

  _scheduleHomePoseCandidatePoll(delay = this._nextHomePoseCandidatePollDelay()) {
    if (!this._canWork() || !this.homePoseCandidatesUrl) return;
    clearTimeout(this._homePoseCandidatePollTimer);
    this._homePoseCandidatePollTimer = window.setTimeout(
      () => this._pollHomePoseCandidates(),
      delay,
    );
  }

  async _pollHomePoseCandidates() {
    if (!this._canWork()) return;
    this._perf.candidate_polls += 1;
    const controller = new AbortController();
    this._homePoseCandidateRequestController = controller;
    const timeout = window.setTimeout(
      () => controller.abort(),
      Math.max(1500, this.homePoseCandidatePollIntervalMs * 4),
    );
    try {
      const response = await fetch(this.homePoseCandidatesUrl, {
        headers: { Accept: "application/json" },
        cache: "no-store",
        signal: controller.signal,
      });
      this.updateHomePoseCandidates(response.ok ? await response.json() : null);
    } catch (_error) {
      if (!this._disposed && this._canWork() && !controller.signal.aborted) {
        this.updateHomePoseCandidates(null);
      }
    } finally {
      window.clearTimeout(timeout);
      if (this._homePoseCandidateRequestController === controller) {
        this._homePoseCandidateRequestController = null;
      }
      this._scheduleHomePoseCandidatePoll();
    }
  }

  async _pollRobotState() {
    if (!this._canWork()) return;
    this._perf.state_polls += 1;
    const controller = new AbortController();
    this._stateRequestController = controller;
    const timeout = window.setTimeout(
      () => controller.abort(),
      Math.max(1000, this.statePollIntervalMs * 5),
    );
    try {
      const response = await fetch(this.robotStateUrl, {
        headers: { Accept: "application/json" },
        cache: "no-store",
        signal: controller.signal,
      });
      const payload = response.ok ? await response.json() : null;
      const state = robotStateFromPayload(payload);
      if (!state) {
        this._useNeutralRobotState();
      } else {
        this.homeJointStates = state.home;
        this.homePoseGhostFresh = { ...state.fresh };
        this.jointSources = state.sources;
        this._liveFlexivStateAvailable = state.fresh.left || state.fresh.right;
        this._liveStateAvailable = this._liveFlexivStateAvailable || state.yam.fresh;
        this._lastRobotTimestamp = state.timestamp;
        this.toolNames = state.toolNames;
        this.updateJointStates(state.left, state.right);
        this.updateConfiguredTools(
          state.configuredTools.left,
          state.configuredTools.right,
        );
        this.updateGripperPositions(state.gripperPositions);
        this.updateTcpPoses(state.tcpPoses.left, state.tcpPoses.right);
        this.updateYamState(state.yam);
        this._syncHomePoseGhostVisibility();
        this._showRobotStateStatus();
      }
    } catch (_error) {
      // Missing config/SHM and a temporarily unavailable server are expected
      // control-plane states. Keep the viewer quiet and visibly neutral.
      if (this._canWork() && !controller.signal.aborted) this._useNeutralRobotState();
    } finally {
      window.clearTimeout(timeout);
      if (this._stateRequestController === controller) this._stateRequestController = null;
      this._scheduleStatePoll();
    }
  }

  _useNeutralRobotState() {
    const left = this.homeJointStates.left || DEFAULT_POSE;
    const right = this.homeJointStates.right || DEFAULT_POSE;
    this.updateJointStates(left, right);
    this.jointSources = {
      left: this.homeJointStates.left ? "home" : "default",
      right: this.homeJointStates.right ? "home" : "default",
    };
    this.updateTcpPoses(null, null);
    this.updateGripperPositions({
      left: { known: false, source: "robot-state-unavailable" },
      right: { known: false, source: "robot-state-unavailable" },
    });
    this.homePoseGhostFresh = { left: false, right: false };
    this._syncHomePoseGhostVisibility();
    this.toolNames = { left: null, right: null };
    if (this.yamState.configured) {
      this.yamState = Object.freeze({
        ...this.yamState,
        liveJoints: null,
        fresh: false,
        source: this.yamState.initialJoints ? "initpos" : "unavailable",
      });
      this._applyYamPose();
    }
    this._liveFlexivStateAvailable = false;
    this._liveStateAvailable = false;
    this._lastRobotTimestamp = null;
    this._showRobotStateStatus();
  }

  _showRobotStateStatus() {
    if (this.statusMode === "external") return;
    if (!this._modelLoaded) return;
    const hasActiveTcp = Boolean(this.arms.left?.tcpFrame.visible || this.arms.right?.tcpFrame.visible);
    const configuredToolSummary = configuredToolStatusSummary(
      this.configuredTools,
      hasActiveTcp,
    );
    const toolWarnings = [...new Set(Object.values(this.configuredTools)
      .map((tool) => tool.resolutionWarning)
      .filter(Boolean))];
    const assetFailures = ["left", "right"].filter(
      (side) => this.toolAssetStates[side].status === "error",
    );
    const unknownGripperSides = ["left", "right"].filter((side) => (
      this.configuredTools[side]?.visual?.articulation
      && this.gripperVisualStates[side]?.known !== true
    ));
    const yamSummary = this.yamState.configured
      ? `Yam ${this.yamState.fresh ? "live" : "initpos"}`
      : null;
    const yamDetails = this.yamAssetState.status === "error"
      ? `Yam model failed: ${this.yamAssetState.error}`
      : this.yamState.configured && this.yamAssetState.status === "loading"
        ? "Yam model loading"
        : null;
    const toolDetails = [
      ...toolWarnings.map((warning) => `Tool warning: ${warning}`),
      ...(assetFailures.length
        ? [`Tool model failed (${assetFailures.join("/")}); placeholder shown`]
        : []),
      ...(unknownGripperSides.length
        ? [`夹爪姿态未知（${unknownGripperSides.join("/")}，保留上次可信或中性姿态）`]
        : []),
    ].join(" · ");
    if (!this._liveFlexivStateAvailable) {
      const hasHome = Object.values(this.jointSources).includes("home");
      const toolSuffix = [configuredToolSummary, toolDetails, yamSummary, yamDetails]
        .filter(Boolean)
        .map((detail) => ` · ${detail}`)
        .join("");
      this._setState(
        "no-data",
        `Flexiv Rizon 4 + 4R | ${hasHome ? "Home 姿态 · " : ""}无实时数据${toolSuffix}`,
      );
      return;
    }
    let suffix = "实时数据";
    const homeSides = ["left", "right"].filter((side) => this.jointSources[side] === "home");
    if (homeSides.length) suffix += ` · ${homeSides.join("/")} Home`;
    const activeToolNames = [...new Set(Object.values(this.toolNames).filter(Boolean))];
    if (activeToolNames.length) suffix += ` · Tool ${activeToolNames.join(" / ")}`;
    else if (hasActiveTcp) suffix += " · Active TCP（SHM absolute）";
    if (configuredToolSummary) suffix += ` · ${configuredToolSummary}`;
    if (toolDetails) suffix += ` · ${toolDetails}`;
    if (yamSummary) suffix += ` | ${yamSummary}`;
    if (yamDetails) suffix += ` | ${yamDetails}`;
    const numericTimestamp = Number(this._lastRobotTimestamp);
    if (Number.isFinite(numericTimestamp) && numericTimestamp > 0) {
      let timestampMs = numericTimestamp;
      if (numericTimestamp > 1e17) timestampMs = numericTimestamp / 1e6;
      else if (numericTimestamp > 1e14) timestampMs = numericTimestamp / 1e3;
      else if (numericTimestamp < 1e11) timestampMs = numericTimestamp * 1000;
      const ageMs = Math.max(0, Date.now() - timestampMs);
      suffix += ` · ${Math.round(ageMs)} ms`;
    }
    this._setState("live", `Flexiv Rizon 4 + 4R | ${suffix}`);
  }

  dispose() {
    this._disposed = true;
    for (const button of this.toolVariantButtons || []) {
      button.removeEventListener("click", this._onToolVariantClick);
    }
    this.toolVariantButtons = [];
    this.toolVariantControl = null;
    this._yamLoadEpoch += 1;
    this.toolAssetRequestGate.invalidate();
    for (const side of ["left", "right"]) this._clearConfiguredToolAsset(side);
    void disposeToolAssetCache(this.toolAssetCache);
    if (activeViewer === this) activeViewer = null;
    if (typeof window !== "undefined" && window.__dualArmViewer === this) window.__dualArmViewer = null;
    this._renderLoop?.dispose();
    clearTimeout(this._statePollTimer);
    clearTimeout(this._homePoseCandidatePollTimer);
    this._stateRequestController?.abort();
    this._homePoseCandidateRequestController?.abort();
    const ownedRoots = [this.sceneRoot, this.homePoseGhostRoot, this.floor, this.grid, this.worldFrame];
    this._clearHomePoseGhosts();
    for (const side of ["left", "right"]) {
      this.homePoseGhostMaterialCaches[side].forEach((material) => material.dispose());
      this.homePoseGhostMaterialCaches[side].clear();
      this.homePoseGhostPointMaterials[side].dispose();
    }
    this.homePoseGhostPointGeometry.dispose();
    this._resizeObserver?.disconnect();
    this._intersectionObserver?.disconnect();
    document.removeEventListener("visibilitychange", this._onDocumentVisibilityChange);
    this.controls?.dispose();
    disposeObjectResources(ownedRoots);
    for (const side of ["left", "right"]) {
      disposeObjectResources(this.armAssets[side].values());
      this.armAssets[side].clear();
    }
    this.renderer?.dispose();
    this.root.replaceChildren();
  }
}

function mountArmViewer(root = document.querySelector("#arm-viewer-root"), options = {}) {
  if (!root) return null;
  if (root.__dualArmViewer) root.__dualArmViewer.dispose();
  const viewer = new DualArmViewer(root, options);
  root.__dualArmViewer = viewer;
  if (typeof window !== "undefined") {
    activeViewer = viewer;
    window.__dualArmViewer = viewer;
  }
  return viewer;
}

// The page can mount explicitly after rendering its layout, while the default
// path supports a plain <script type="module"> include at the end of <body>.
if (typeof window !== "undefined") {
  installDebugGetter();
  window.DualArmViewer = DualArmViewer;
  window.mountArmViewer = mountArmViewer;
  window.updateJointStates = (left, right) => window.__dualArmViewer?.updateJointStates(left, right);
  window.setToolVisualVariant = (variant) => window.__dualArmViewer?.setToolVisualVariant(variant);
  window.updateGripperPositions = (left, right) => {
    if (window.__dualArmViewer) return window.__dualArmViewer.updateGripperPositions(left, right);
    window.__pendingGripperViewerPositions = right === undefined ? left : { left, right };
    return null;
  };
  window.updateGripperStates = (raw, options) => {
    if (window.__dualArmViewer) return window.__dualArmViewer.updateGripperStates(raw, options);
    window.__pendingGripperViewerStates = { raw, options };
    return null;
  };
  const autoMount = () => {
    const root = document.querySelector("#arm-viewer-root");
    if (!root) return;
    const viewer = mountArmViewer(root);
    window.__dualArmViewer = viewer;
    if (window.__pendingGripperViewerPositions !== undefined) {
      viewer.updateGripperPositions(window.__pendingGripperViewerPositions);
      delete window.__pendingGripperViewerPositions;
    }
    if (window.__pendingGripperViewerStates !== undefined) {
      const pending = window.__pendingGripperViewerStates;
      viewer.updateGripperStates(pending.raw, pending.options);
      delete window.__pendingGripperViewerStates;
    }
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", autoMount, { once: true });
  else autoMount();
}

export {
  ARM_BASE_POSITIONS,
  absoluteTcpPoseToSceneTransform,
  DualArmViewer,
  FLEXIV_TO_THREE_QUATERNION,
  GROUND_FLEXIV_POSITION,
  GRID_SCENE_LIFT,
  HOME_POSE_GHOST_OPACITY,
  MAX_HOME_POSE_GHOSTS_PER_SIDE,
  YAM_MODEL_SPEC,
  ToolAssetRequestGate,
  armBaseSceneQuaternion,
  composeFlexivBaseWorldQuaternion,
  composeYamBaseSceneQuaternion,
  cloneToolAssetSource,
  configuredToolWithVariant,
  configuredToolVisualTransform,
  flexivFrameRotationToThree,
  flexivWorldRotationToThree,
  flexivWorldToThreePosition,
  groundScenePosition,
  mountArmViewer,
  normalizedHomePoseCandidates,
  normalizedYamState,
  normalizedConfiguredTool,
  normalizedGripperPosition,
  normalizedToolAsset,
  normalizedToolArticulation,
  normalizedToolVariants,
  normalizedToolTransform,
  normalizedToolVisual,
  disposeToolAssetCache,
  gripperJointAngleFromValue,
  parseBinarySTL,
  robotStateFromPayload,
  toolUrdfMeshUrl,
  DirtyRenderLoop,
  numericVectorsNearlyEqual,
};
