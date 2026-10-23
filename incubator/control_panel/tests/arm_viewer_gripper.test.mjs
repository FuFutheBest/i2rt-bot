import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

import * as THREE from "../static/vendor/three.module.min.js";
import {
  absoluteTcpPoseToSceneTransform,
  cloneToolAssetSource,
  configuredToolWithVariant,
  configuredToolVisualTransform,
  composeYamBaseSceneQuaternion,
  DualArmViewer,
  FLEXIV_TO_THREE_QUATERNION,
  YAM_MODEL_SPEC,
  flexivWorldToThreePosition,
  flexivWorldRotationToThree,
  gripperJointAngleFromValue,
  normalizedConfiguredTool,
  normalizedGripperPosition,
  normalizedYamState,
  parseBinarySTL,
  robotStateFromPayload,
} from "../static/arm_viewer.js";

const UMI3_SOFT_URL = "/static/assets/tools/umi3_camera/soft/umi3_gripper.urdf";
const UMI3_HARD_URL = "/static/assets/tools/umi3_camera/hard/umi3_gripper.urdf";

const mountTransform = {
  frame: "flange",
  translation: [0, 0, 0.02],
  quaternion: [1, 0, 0, 0],
  quaternion_order: "wxyz",
};

const widthArticulation = {
  joint_name: "gripper_joint",
  joint_lower_rad: 0,
  joint_upper_rad: Math.PI / 2,
  closed_joint_rad: 0,
  value_mode: "width_m",
  arm_length_m: 0.1,
  max_width_m: 2 * 0.1 * Math.sin(Math.PI / 4),
  apex_angle_max_deg: 90,
};

function configuredTool(articulation = widthArticulation) {
  return {
    resolved_tool_name: "umi3",
    status: "available",
    transform: mountTransform,
    visual: {
      model: "umi3",
      kind: "urdf",
      url: UMI3_SOFT_URL,
      default_variant: "soft",
      variants: {
        soft: { kind: "urdf", url: UMI3_SOFT_URL },
        hard: { kind: "urdf", url: UMI3_HARD_URL },
      },
      mirror_axis: "y",
      mount_transform: mountTransform,
      source: "elements[0].visual",
      mount_source: "elements[0].visual.mount_transform",
      articulation,
    },
  };
}

function viewerStateHarness() {
  const viewer = Object.create(DualArmViewer.prototype);
  viewer.configuredTools = {
    left: normalizedConfiguredTool(configuredTool()),
    right: normalizedConfiguredTool(configuredTool()),
  };
  viewer.gripperVisualStates = {
    left: { known: false, rawValue: null, actualPercent: null, jointAngle: null, source: "initial" },
    right: { known: false, rawValue: null, actualPercent: null, jointAngle: null, source: "initial" },
  };
  viewer.toolAssetObjects = { left: null, right: null };
  viewer.toolJointByName = { left: new Map(), right: new Map() };
  viewer._perf = { gripper_applies: 0, gripper_skips: 0, gripper_unknowns: 0 };
  viewer._invalidateRender = () => {};
  return viewer;
}

function configuredToolApplyHarness() {
  const viewer = Object.create(DualArmViewer.prototype);
  const flangeLink = new THREE.Group();
  const configuredToolFrame = new THREE.Group();
  const configuredToolVisualFrame = new THREE.Group();
  const tcpFrame = new THREE.Group();
  viewer.arms = {
    left: {
      flangeLink,
      configuredToolFrame,
      configuredToolVisualFrame,
      tcpFrame,
    },
    right: null,
  };
  viewer.configuredTools = { left: null, right: null };
  viewer.configuredToolDescriptors = { left: null, right: null };
  viewer.toolVisualVariant = "soft";
  viewer.toolVariantButtons = [];
  viewer.tcpStates = { left: null, right: null };
  viewer.gripperArticulationSignatures = { left: "", right: "" };
  viewer.gripperVisualStates = {
    left: { known: false, rawValue: null, actualPercent: null, jointAngle: null, source: "initial" },
    right: { known: false, rawValue: null, actualPercent: null, jointAngle: null, source: "initial" },
  };
  viewer.toolAssetStates = {
    left: { status: "none", url: null, error: null },
    right: { status: "none", url: null, error: null },
  };
  viewer._perf = { tool_applies: 0, tool_skips: 0 };
  viewer._invalidateRender = () => {};
  viewer._updateConfiguredToolAsset = (side, tool) => {
    viewer.capturedConfiguredTool = tool;
    const asset = tool.visual?.asset || tool.asset;
    viewer.toolAssetStates[side] = asset
      ? { status: "ready", url: asset.url, error: null }
      : { status: "none", url: null, error: null };
  };
  return viewer;
}

test("nested UMI3 visual keeps a flange mount independent of the TCP transform", () => {
  const tool = normalizedConfiguredTool(configuredTool());
  assert.equal(tool.visual.model, "umi3");
  assert.equal(tool.visual.asset.kind, "urdf");
  assert.equal(
    tool.visual.asset.url,
    UMI3_SOFT_URL,
  );
  assert.deepEqual(Object.keys(tool.visual.variants), ["soft", "hard"]);
  assert.equal(tool.visual.defaultVariant, "soft");
  assert.equal(tool.visual.selectedVariant, "soft");
  assert.equal(tool.visual.mirrorAxis, "y");
  assert.deepEqual(tool.visual.mountTransform.translation, [0, 0, 0.02]);
  assert.equal(tool.visual.articulation.jointName, "gripper_joint");
  assert.equal(tool.visual.articulation.closedJointRad, 0);
});

test("nested visual remains mountable when the controller TCP is not configured", () => {
  const raw = configuredTool();
  raw.status = "not-configured";
  raw.transform = null;
  const tool = normalizedConfiguredTool(raw);
  assert.equal(tool.status, "not-configured");
  assert.equal(configuredToolVisualTransform(tool), tool.visual.mountTransform);
});

test("robot-state normalized UMI3 visual survives updateConfiguredTools double normalization", () => {
  const raw = configuredTool();
  raw.status = "not-configured";
  raw.transform = null;
  const state = robotStateFromPayload({
    status: "not_found",
    configured_tools: { left: raw },
    gripper_positions: { left: null, right: null },
  });
  const once = state.configuredTools.left;
  assert.equal(once.visual.asset.kind, "urdf");
  assert.equal(once.visual.asset.url, UMI3_SOFT_URL);

  const twice = normalizedConfiguredTool(once);
  assert.deepEqual(twice, once);
  assert.equal(twice.visual.asset.kind, "urdf");
  assert.equal(twice.visual.mountTransform.frame, "flange");
  assert.equal(twice.visual.articulation.jointName, "gripper_joint");

  const viewer = configuredToolApplyHarness();
  viewer.updateConfiguredTools(once, undefined);
  assert.equal(viewer.configuredTools.left.visual.asset.kind, "urdf");
  assert.equal(viewer.capturedConfiguredTool.visual.asset.kind, "urdf");
  assert.equal(viewer.toolAssetStates.left.status, "ready");
  assert.equal(
    viewer.toolAssetStates.left.url,
    UMI3_SOFT_URL,
  );
  assert.equal(viewer.arms.left.configuredToolVisualFrame.userData.configured, true);
});

test("normalized nested asset accepts camelCase materialUrl on repeat normalization", () => {
  const raw = configuredTool({ ...widthArticulation, value_mode: "normalized" });
  delete raw.visual.default_variant;
  delete raw.visual.variants;
  delete raw.visual.mirror_axis;
  raw.visual.kind = "obj";
  raw.visual.url = "/static/assets/tools/example/tool.obj";
  raw.visual.material_url = "/static/assets/tools/example/tool.mtl";
  raw.visual.scale = [2, 3, 4];
  const once = normalizedConfiguredTool(raw);
  const twice = normalizedConfiguredTool(once);
  assert.deepEqual(twice, once);
  assert.equal(twice.visual.asset.kind, "obj");
  assert.equal(twice.visual.asset.url, "/static/assets/tools/example/tool.obj");
  assert.equal(twice.visual.asset.materialUrl, "/static/assets/tools/example/tool.mtl");
  assert.deepEqual(twice.visual.asset.scale, [2, 3, 4]);
});

test("normalized and physical-width values map closed to q=0 and open to q=pi/2", () => {
  const normalized = normalizedConfiguredTool(configuredTool({
    ...widthArticulation,
    value_mode: "normalized",
  })).visual.articulation;
  assert.equal(gripperJointAngleFromValue(0, normalized), 0);
  assert.equal(gripperJointAngleFromValue(1, normalized), Math.PI / 2);
  assert.equal(gripperJointAngleFromValue(2, normalized), Math.PI / 2);

  const width = normalizedConfiguredTool(configuredTool()).visual.articulation;
  assert.ok(Math.abs(gripperJointAngleFromValue(0, width) - 0) < 1e-12);
  assert.ok(Math.abs(gripperJointAngleFromValue(width.maxWidthM, width) - Math.PI / 2) < 1e-12);
  const halfWidth = normalizedGripperPosition(
    { known: true, actual_percent: 50, source: "device-control-idle" },
    width,
  );
  const expected = 2 * Math.asin(0.5 * Math.sin(Math.PI / 4));
  assert.equal(halfWidth.known, true);
  assert.ok(Math.abs(halfWidth.jointAngle - expected) < 1e-12);
});

test("robot state accepts same-frame gripper_positions only while globally fresh", () => {
  const fresh = robotStateFromPayload({
    status: "ok",
    fresh: true,
    joints: { left: [0, 0, 0, 0, 0, 0, 0], right: [0, 0, 0, 0, 0, 0, 0] },
    configured_tools: { left: configuredTool(), right: configuredTool() },
    gripper_positions: { left: 0, right: widthArticulation.max_width_m },
  });
  assert.equal(fresh.gripperPositions.left.known, true);
  assert.equal(fresh.gripperPositions.left.jointAngle, 0);
  assert.equal(fresh.gripperPositions.left.actualPercent, null);
  assert.equal(
    normalizedGripperPosition(
      fresh.gripperPositions.left,
      normalizedConfiguredTool(configuredTool()).visual.articulation,
    ).actualPercent,
    null,
  );
  assert.ok(Math.abs(fresh.gripperPositions.right.jointAngle - Math.PI / 2) < 1e-12);

  const stale = robotStateFromPayload({
    status: "ok",
    fresh: false,
    configured_tools: { left: configuredTool(), right: configuredTool() },
    gripper_positions: { left: 0, right: widthArticulation.max_width_m },
  });
  assert.equal(stale.gripperPositions.left.known, false);
  assert.equal(stale.gripperPositions.right.known, false);
});

test("null, NaN, and physically impossible widths stay unknown instead of faking closed", () => {
  const articulation = normalizedConfiguredTool(configuredTool()).visual.articulation;
  for (const value of [null, Number.NaN, -0.001, articulation.maxWidthM + 0.001]) {
    const position = normalizedGripperPosition(value, articulation, "robot-state");
    assert.equal(position.known, false, String(value));
    assert.equal(position.jointAngle, null, String(value));
  }
  const state = robotStateFromPayload({
    status: "ok",
    fresh: true,
    configured_tools: { left: configuredTool(), right: configuredTool() },
    gripper_positions: { left: null, right: null },
  });
  assert.equal(state.gripperPositions.left.known, false);
  assert.equal(state.gripperPositions.right.known, false);
});

test("viewer keeps the last trusted pose on unknown/busy updates and skips repeated values", () => {
  const viewer = viewerStateHarness();
  viewer.updateGripperStates({ left: { known: true, actual_percent: 40 } }, { source: "idle" });
  const trustedAngle = viewer.gripperVisualStates.left.jointAngle;
  assert.equal(viewer.gripperVisualStates.left.known, true);
  assert.equal(viewer.gripperVisualStates.right.jointAngle, null);
  assert.equal(viewer._perf.gripper_applies, 1);

  viewer.updateGripperStates({ left: { known: true, actual_percent: 40 } }, { source: "idle" });
  assert.equal(viewer._perf.gripper_skips, 1);
  viewer.updateGripperStates({ left: { known: false } }, { source: "stale" });
  assert.equal(viewer.gripperVisualStates.left.known, false);
  assert.equal(viewer.gripperVisualStates.left.jointAngle, trustedAngle);
  viewer.updateGripperStates(
    { left: { known: true, actual_percent: 80 }, busy: true },
    { source: "busy" },
  );
  assert.equal(viewer.gripperVisualStates.left.known, false);
  assert.equal(viewer.gripperVisualStates.left.jointAngle, trustedAngle);
});

test("soft/hard visual switching keeps articulation and selects only allowlisted assets", () => {
  const soft = normalizedConfiguredTool(configuredTool());
  const hard = configuredToolWithVariant(soft, "hard");
  assert.equal(soft.visual.asset.url, UMI3_SOFT_URL);
  assert.equal(hard.visual.asset.url, UMI3_HARD_URL);
  assert.equal(hard.visual.selectedVariant, "hard");
  assert.deepEqual(hard.visual.articulation, soft.visual.articulation);
  assert.deepEqual(hard.visual.mountTransform, soft.visual.mountTransform);
  assert.equal(configuredToolWithVariant(soft, "unknown").visual.asset.url, UMI3_SOFT_URL);
});

test("viewer switches both grippers without resetting the trusted joint angle", () => {
  const viewer = configuredToolApplyHarness();
  viewer.updateConfiguredTools(configuredTool(), configuredTool());
  viewer.gripperVisualStates.left = {
    known: true,
    rawValue: null,
    actualPercent: 47,
    jointAngle: 0.73,
    source: "control-panel",
  };

  assert.equal(viewer.setToolVisualVariant("hard"), viewer);
  assert.equal(viewer.configuredTools.left.visual.asset.url, UMI3_HARD_URL);
  assert.equal(viewer.configuredTools.right.visual.asset.url, UMI3_HARD_URL);
  assert.equal(viewer.gripperVisualStates.left.jointAngle, 0.73);
  assert.equal(viewer.gripperVisualStates.left.source, "control-panel");
  assert.equal(viewer.setToolVisualVariant("unsupported"), false);
});

test("fresh serial gripper state wins over conflicting robot SHM width", () => {
  const viewer = viewerStateHarness();
  viewer.updateGripperStates(
    { left: { known: true, actual_percent: 86.4 } },
    { source: "control-panel" },
  );
  const directAngle = viewer.gripperVisualStates.left.jointAngle;

  viewer.updateGripperPositions({
    left: normalizedGripperPosition(
      widthArticulation.max_width_m,
      normalizedConfiguredTool(configuredTool()).visual.articulation,
      "robot-state",
    ),
  });

  assert.equal(viewer.gripperVisualStates.left.source, "control-panel");
  assert.equal(viewer.gripperVisualStates.left.jointAngle, directAngle);
  assert.equal(viewer._perf.gripper_skips, 1);

  viewer.gripperDirectSourceFreshUntil.left = Date.now() - 1;
  viewer.updateGripperPositions({
    left: normalizedGripperPosition(
      widthArticulation.max_width_m,
      normalizedConfiguredTool(configuredTool()).visual.articulation,
      "robot-state",
    ),
  });
  assert.equal(viewer.gripperVisualStates.left.source, "robot-state");
  assert.ok(Math.abs(viewer.gripperVisualStates.left.jointAngle - Math.PI / 2) < 1e-12);
});

test("serial state arriving before tool config is applied after articulation loads", () => {
  const viewer = viewerStateHarness();
  viewer.configuredTools.left = normalizedConfiguredTool(null);
  viewer.updateGripperStates(
    { left: { known: true, actual_percent: 42 } },
    { source: "control-panel" },
  );
  assert.equal(viewer.gripperVisualStates.left.known, false);
  assert.equal(viewer.gripperDirectValues.left.actual_percent, 42);

  viewer.arms = { left: null, right: null };
  viewer.gripperArticulationSignatures = { left: "", right: "" };
  viewer.toolAssetStates = {
    left: { status: "none", url: null, error: null },
    right: { status: "none", url: null, error: null },
  };
  viewer._updateConfiguredToolAsset = () => {};
  viewer.updateConfiguredTools(configuredTool(), undefined);

  assert.equal(viewer.gripperVisualStates.left.known, true);
  assert.equal(viewer.gripperVisualStates.left.actualPercent, 42);
  assert.equal(viewer.gripperVisualStates.left.source, "control-panel");

  viewer.updateGripperPositions({
    left: normalizedGripperPosition(
      widthArticulation.max_width_m,
      normalizedConfiguredTool(configuredTool()).visual.articulation,
      "robot-state",
    ),
  });
  assert.equal(viewer.gripperVisualStates.left.actualPercent, 42);
});

test("absolute TCP conversion applies station Z-180 rotation and side Y offsets", () => {
  const pose = [
    0.9476, 0.0362, -0.4530,
    1, 0, 0, 0,
  ];
  const left = absoluteTcpPoseToSceneTransform(pose, "left");
  const right = absoluteTcpPoseToSceneTransform(pose, "right");
  assert.ok(left);
  assert.ok(right);
  assert.deepEqual(
    left.position.toArray().map((value) => Number(value.toFixed(4))),
    [-0.9476, -0.453, -0.1638],
  );
  assert.deepEqual(
    right.position.toArray().map((value) => Number(value.toFixed(4))),
    [-0.9476, -0.453, 0.2362],
  );
  assert.ok(left.position.x < 0);
  assert.ok(Math.abs(left.position.x) > Math.abs(left.position.y));
  assert.ok(Math.abs(left.position.x) > Math.abs(left.position.z));
  const leftAxes = [
    new THREE.Vector3(1, 0, 0).applyQuaternion(left.quaternion),
    new THREE.Vector3(0, 1, 0).applyQuaternion(left.quaternion),
    new THREE.Vector3(0, 0, 1).applyQuaternion(left.quaternion),
  ];
  const expectedAxes = [
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(0, 1, 0),
  ];
  leftAxes.forEach((axis, index) => {
    assert.ok(axis.distanceTo(expectedAxes[index]) < 1e-9);
  });
  const rightAxes = [
    new THREE.Vector3(1, 0, 0).applyQuaternion(right.quaternion),
    new THREE.Vector3(0, 1, 0).applyQuaternion(right.quaternion),
    new THREE.Vector3(0, 0, 1).applyQuaternion(right.quaternion),
  ];
  rightAxes.forEach((axis, index) => {
    assert.ok(axis.distanceTo(leftAxes[index]) < 1e-9);
  });
  assert.equal(absoluteTcpPoseToSceneTransform(pose, "unknown"), null);
});

test("absolute TCP orientation converts frame basis without an extra URDF conjugation", () => {
  const rawRotation = new THREE.Quaternion()
    .setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.37)
    .normalize();
  const pose = [
    0.5, -0.1, 0.25,
    rawRotation.w, rawRotation.x, rawRotation.y, rawRotation.z,
  ];
  const actual = absoluteTcpPoseToSceneTransform(pose, "left");
  const stationRotation = new THREE.Quaternion()
    .setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI)
    .multiply(rawRotation)
    .normalize();
  const expected = FLEXIV_TO_THREE_QUATERNION.clone()
    .multiply(stationRotation)
    .normalize();
  assert.ok(actual.quaternion.angleTo(expected) < 1e-9);

  const oldConjugated = flexivWorldRotationToThree(stationRotation);
  const localXMinus90 = new THREE.Quaternion()
    .setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
  assert.ok(actual.quaternion.angleTo(oldConjugated.multiply(localXMinus90)) < 1e-9);
});

test("strict binary STL parser checks exact length and produces one triangle", () => {
  const buffer = new ArrayBuffer(84 + 50);
  const view = new DataView(buffer);
  view.setUint32(80, 1, true);
  const values = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0];
  values.forEach((value, index) => view.setFloat32(84 + index * 4, value, true));
  const geometry = parseBinarySTL(buffer);
  assert.equal(geometry.userData.binaryStlTriangleCount, 1);
  assert.equal(geometry.getAttribute("position").count, 3);
  assert.throws(() => parseBinarySTL(buffer.slice(0, -1)), /byte length/);
  geometry.dispose();
});

test("vendored UMI3 binary STL files parse with the expected triangle counts", () => {
  const expected = new Map([
    ["hard/meshes/base_link.stl", 92_769],
    ["hard/meshes/index_link.stl", 18_968],
    ["soft/meshes/base_link.stl", 92_715],
    ["soft/meshes/index_link.stl", 19_082],
  ]);
  for (const [name, triangleCount] of expected) {
    const bytes = fs.readFileSync(new URL(`../static/assets/tools/umi3_camera/${name}`, import.meta.url));
    const input = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const geometry = parseBinarySTL(input);
    assert.equal(geometry.userData.binaryStlTriangleCount, triangleCount);
    assert.equal(geometry.getAttribute("position").count, triangleCount * 3);
    assert.equal(geometry.boundingBox.isEmpty(), false);
    geometry.dispose();
  }
});

test("vendored Yam binary STL files parse with the expected triangle counts", () => {
  const expected = new Map([
    ["base.stl", 8754],
    ["link1.stl", 6172],
    ["link2.stl", 29914],
    ["link3.stl", 30150],
    ["link4.stl", 3708],
    ["link5.stl", 1136],
  ]);
  for (const [name, triangleCount] of expected) {
    const bytes = fs.readFileSync(new URL(`../static/assets/yam/meshes/${name}`, import.meta.url));
    const input = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const geometry = parseBinarySTL(input);
    assert.equal(geometry.userData.binaryStlTriangleCount, triangleCount);
    assert.equal(geometry.getAttribute("position").count, triangleCount * 3);
    geometry.dispose();
  }
});

test("Yam state prefers fresh SHM joints and otherwise keeps config initpos", () => {
  const init = {
    configured: true,
    model: "yam",
    status: "not_found",
    urdf_url: "/static/assets/yam/yam_no_gripper.urdf",
    base_position_world: [0, 0, 0.2],
    initial_joints: [0, 2, 0.73, 0.82, 0, 0.02],
    fresh: false,
    joints: null,
  };
  const neutral = normalizedYamState(init);
  assert.equal(neutral.configured, true);
  assert.equal(neutral.source, "initpos");
  assert.deepEqual(neutral.initialJoints, init.initial_joints);
  assert.equal(neutral.liveJoints, null);
  assert.equal(neutral.urdfUrl, YAM_MODEL_SPEC.urdfUrl);

  const live = robotStateFromPayload({
    status: "not_found",
    yam_head: {
      ...init,
      status: "ok",
      fresh: true,
      joints: [0.1, 2.1, 0.7, 0.8, -0.1, 0.03],
    },
  }).yam;
  assert.equal(live.source, "live");
  assert.deepEqual(live.liveJoints, [0.1, 2.1, 0.7, 0.8, -0.1, 0.03]);
});

test("Yam viewer places its base 0.2m above world origin and applies six joints", () => {
  const viewer = Object.create(DualArmViewer.prototype);
  const root = new THREE.Group();
  const joints = new Map();
  for (const name of YAM_MODEL_SPEC.jointNames) {
    const joint = new THREE.Group();
    joint.userData.axis = new THREE.Vector3(0, 0, 1);
    joint.userData.limit = [-Math.PI, Math.PI];
    joint.userData.baseQuaternion = new THREE.Quaternion();
    joint.userData.axisAngleQuaternion = new THREE.Quaternion();
    joints.set(name, joint);
  }
  viewer.yam = { root, joints };
  viewer.yamState = normalizedYamState({
    configured: true,
    model: "yam",
    status: "not_found",
    urdf_url: "/static/assets/yam/yam_no_gripper.urdf",
    base_position_world: [0, 0, 0.2],
    initial_joints: [0, 2, 0.73, 0.82, 0, 0.02],
    fresh: false,
  });
  viewer.yamJointState = null;
  viewer._perf = { yam_applies: 0, yam_skips: 0 };
  viewer._invalidateRender = () => {};

  assert.equal(viewer._applyYamPose(true), true);
  assert.deepEqual(root.position.toArray(), flexivWorldToThreePosition([0, 0, 0.2]).toArray());
  assert.deepEqual(viewer.yamJointState, [0, 2, 0.73, 0.82, 0, 0.02]);
  assert.equal(root.userData.poseSource, "initpos");
  assert.equal(viewer._applyYamPose(), false);
  assert.equal(viewer._perf.yam_skips, 1);
});

test("Yam base preserves scene conversion then adds world Z 180 and X -90", () => {
  const rotation = composeYamBaseSceneQuaternion();
  const relativeToOriginal = rotation.clone()
    .multiply(FLEXIV_TO_THREE_QUATERNION.clone().invert());
  const actualAxes = [
    new THREE.Vector3(1, 0, 0).applyQuaternion(relativeToOriginal),
    new THREE.Vector3(0, 1, 0).applyQuaternion(relativeToOriginal),
    new THREE.Vector3(0, 0, 1).applyQuaternion(relativeToOriginal),
  ];
  const expectedAxes = [
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(0, 1, 0),
  ];
  actualAxes.forEach((axis, index) => {
    assert.ok(axis.distanceTo(expectedAxes[index]) < 1e-9);
  });
});

test("tool clones share cached geometry but keep independent materials and joints", () => {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshStandardMaterial({ color: 0xffffff });
  const source = new THREE.Group();
  source.userData.preserveToolColors = true;
  const motion = new THREE.Group();
  motion.userData.toolJointDefinition = {
    name: "gripper_joint",
    axis: [0, -1, 0],
    lower: 0,
    upper: Math.PI / 2,
    baseQuaternion: [0, 0, 0, 1],
  };
  motion.add(new THREE.Mesh(geometry, material));
  source.add(motion);

  const left = cloneToolAssetSource(source, new THREE.Color(0x00ffff));
  const right = cloneToolAssetSource(source, new THREE.Color(0xff8800));
  const leftMesh = left.getObjectByProperty("isMesh", true);
  const rightMesh = right.getObjectByProperty("isMesh", true);
  assert.equal(leftMesh.geometry, geometry);
  assert.equal(rightMesh.geometry, geometry);
  assert.notEqual(leftMesh.material, rightMesh.material);
  assert.notEqual(left.userData.toolJoints, right.userData.toolJoints);
  assert.equal(left.userData.toolJoints.get("gripper_joint").userData.axis.y, -1);

  leftMesh.material.dispose();
  rightMesh.material.dispose();
  geometry.dispose();
  material.dispose();
});
