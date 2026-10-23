import assert from "node:assert/strict";
import test from "node:test";

import {
  disposeToolAssetCache,
  ToolAssetRequestGate,
  normalizedConfiguredTool,
  normalizedToolAsset,
  robotStateFromPayload,
  toolUrdfMeshUrl,
} from "../static/arm_viewer.js";

const transform = {
  frame: "flange",
  translation: [0.01, -0.02, 0.31],
  quaternion: [1, 0, 0, 0],
  quaternion_order: "wxyz",
};

test("Elements fallback fields and same-origin asset survive payload normalization", () => {
  const tool = normalizedConfiguredTool({
    name: "legacy-name",
    resolved_tool_name: "umi3_tcp",
    resolved_tool_source: "gripper.elements[0]",
    resolution: "elements-active-tool",
    resolution_warning: "multiple active tool elements; selected umi3_tcp",
    active_element_count: 2,
    status: "available",
    transform,
    asset: {
      kind: "obj",
      url: "/static/assets/tools/umi3_tcp.obj",
      material_url: "/static/assets/tools/umi3_tcp.mtl",
      scale: [1, 1, 1],
    },
  });
  assert.equal(tool.resolvedToolName, "umi3_tcp");
  assert.equal(tool.name, "umi3_tcp");
  assert.equal(tool.resolvedToolSource, "gripper.elements[0]");
  assert.equal(tool.resolution, "elements-active-tool");
  assert.equal(tool.activeElementCount, 2);
  assert.equal(tool.asset.kind, "obj");
});

test("tool assets reject external, traversal, and unsupported paths", () => {
  assert.equal(normalizedToolAsset({ kind: "obj", url: "https://example.invalid/tool.obj" }), null);
  assert.equal(normalizedToolAsset({ kind: "obj", url: "/static/../private/tool.obj" }), null);
  assert.equal(normalizedToolAsset({ kind: "obj", url: "/static/%2e%2e/private/tool.obj" }), null);
  assert.equal(normalizedToolAsset({ kind: "obj", url: "/static/assets/tool:copy.obj" }), null);
  assert.equal(normalizedToolAsset({ kind: "obj", url: "/static/assets/tool.stl" }), null);
  assert.equal(normalizedToolAsset({ kind: "urdf", url: "/static/assets/tools/tool.urdf" }).kind, "urdf");
});

test("tool URDF mesh URLs resolve relative and package paths under static", () => {
  assert.equal(
    toolUrdfMeshUrl("visual/tool.obj", "/static/assets/tools/tool.urdf"),
    "http://localhost/static/assets/tools/visual/tool.obj",
  );
  assert.equal(
    toolUrdfMeshUrl("package://meshes/vendor/tool.obj", "/static/assets/tools/tool.urdf"),
    "http://localhost/static/assets/tools/meshes/vendor/tool.obj",
  );
  const resolved = new URL(toolUrdfMeshUrl(
    "package://vendor/meshes/tool.obj",
    "http://localhost/static/assets/tools/tool.urdf",
  ));
  assert.equal(resolved.origin, "http://localhost");
  assert.equal(resolved.pathname, "/static/assets/tools/meshes/tool.obj");
  assert.ok(resolved.pathname.startsWith("/static/"));
});

test("tool URDF mesh URLs reject encoded traversal, backslashes, and unsafe schemes", () => {
  const urdf = "/static/assets/tools/tool.urdf";
  for (const filename of [
    "%2e%2e/private.obj",
    "%2E%2E/private.obj",
    "package://meshes/%2e%2e/private.obj",
    "..\\private.obj",
    "//example.invalid/tool.obj",
    "https://example.invalid/tool.obj",
    "file:///private/tool.obj",
    "data:text/plain,tool",
    "package:meshes/tool.obj",
  ]) {
    assert.equal(toolUrdfMeshUrl(filename, urdf), null, filename);
  }
  assert.equal(toolUrdfMeshUrl("visual/tool.obj", "/static/%2E%2E/private/tool.urdf"), null);
  assert.equal(toolUrdfMeshUrl("visual/tool.obj", "/private/tool.urdf"), null);
  assert.equal(toolUrdfMeshUrl("visual/tool.obj", "//example.invalid/static/tool.urdf"), null);
});

test("viewer disposal clears the cache and disposes pending stale sources", async () => {
  let resolvePending;
  const pending = new Promise((resolve) => { resolvePending = resolve; });
  const cache = new Map([
    ["ready", Promise.resolve({ id: "ready" })],
    ["pending", pending],
    ["failed", Promise.reject(new Error("load failed"))],
  ]);
  const disposed = [];
  const completion = disposeToolAssetCache(cache, (source) => disposed.push(source.id));
  assert.equal(cache.size, 0);
  resolvePending({ id: "pending-stale" });
  assert.deepEqual(await completion, [true, true, false]);
  assert.deepEqual(disposed.sort(), ["pending-stale", "ready"]);
});

test("config switch payload clears the previous resolved tool", () => {
  const first = robotStateFromPayload({
    status: "not_found",
    home: {},
    configured_tools: {
      left: {
        resolved_tool_name: "umi3_tcp",
        resolved_tool_source: "elements[0]",
        resolution: "elements-active-tool",
        status: "available",
        transform,
      },
    },
  });
  const second = robotStateFromPayload({
    status: "not_found",
    home: {},
    configured_tools: {
      left: {
        resolved_tool_name: null,
        resolved_tool_source: null,
        resolution: "none",
        status: "not-configured",
        transform: null,
      },
    },
  });
  assert.equal(first.configuredTools.left.resolvedToolName, "umi3_tcp");
  assert.equal(second.configuredTools.left.resolvedToolName, null);
  assert.equal(second.configuredTools.left.asset, null);
});

test("stale async tool asset completion cannot win after a config switch", async () => {
  const gate = new ToolAssetRequestGate();
  const oldEpoch = gate.next("left");
  let resolveOld;
  const oldPromise = new Promise((resolve) => { resolveOld = resolve; });
  const oldOutcome = gate.settle("left", oldEpoch, oldPromise);
  const newEpoch = gate.next("left");
  const newOutcome = await gate.settle("left", newEpoch, Promise.resolve("new-model"));
  resolveOld("old-model");
  const staleOutcome = await oldOutcome;
  assert.deepEqual(newOutcome, { accepted: true, value: "new-model", error: null });
  assert.equal(staleOutcome.accepted, false);
  assert.equal(staleOutcome.value, "old-model");
});

test("current mesh load failure is accepted for placeholder fallback", async () => {
  const gate = new ToolAssetRequestGate();
  const epoch = gate.next("right");
  const failure = new Error("mesh failed to load");
  const outcome = await gate.settle("right", epoch, Promise.reject(failure));
  assert.equal(outcome.accepted, true);
  assert.equal(outcome.value, null);
  assert.equal(outcome.error, failure);
});
