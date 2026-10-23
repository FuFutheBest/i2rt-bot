import assert from "node:assert/strict";
import test from "node:test";

import { DirtyRenderLoop, numericVectorsNearlyEqual } from "../static/arm_viewer.js";

function harness() {
  let now = 0;
  let nextHandle = 1;
  const callbacks = new Map();
  let renders = 0;
  let updates = 0;
  let visible = true;
  const loop = new DirtyRenderLoop({
    render: () => { renders += 1; },
    update: () => { updates += 1; return updates < 3; },
    canRender: () => visible,
    requestFrame: (callback) => {
      const handle = nextHandle++;
      callbacks.set(handle, callback);
      return handle;
    },
    cancelFrame: (handle) => callbacks.delete(handle),
    now: () => now,
    maxFps: 30,
  });
  const advance = (milliseconds = 34) => {
    now += milliseconds;
    const pending = [...callbacks.values()];
    callbacks.clear();
    pending.forEach((callback) => callback(now));
  };
  return {
    loop,
    advance,
    setVisible(value) { visible = value; },
    get callbacks() { return callbacks.size; },
    get renders() { return renders; },
  };
}

test("dirty loop renders once then becomes idle", () => {
  const state = harness();
  state.loop.invalidate("initial");
  assert.equal(state.callbacks, 1);
  state.advance();
  assert.equal(state.renders, 1);
  assert.equal(state.callbacks, 0);
  assert.equal(state.loop.snapshot().render_count, 1);
});

test("multiple invalidations coalesce and frame rate is capped", () => {
  const state = harness();
  state.loop.invalidate("joint");
  state.loop.invalidate("tcp");
  assert.equal(state.callbacks, 1);
  state.advance(10);
  assert.equal(state.renders, 1);
  state.loop.invalidate("joint");
  state.advance(10);
  assert.equal(state.renders, 1);
  assert.equal(state.callbacks, 1);
  state.advance(24);
  assert.equal(state.renders, 2);
});

test("visibility pause cancels work and resume preserves dirty frame", () => {
  const state = harness();
  state.loop.invalidate("model");
  state.loop.pause();
  assert.equal(state.callbacks, 0);
  state.setVisible(false);
  state.loop.resume();
  assert.equal(state.callbacks, 0);
  state.setVisible(true);
  state.loop.resume();
  assert.equal(state.callbacks, 1);
  state.advance();
  assert.equal(state.renders, 1);
});

test("damping burst schedules only a bounded series", () => {
  const state = harness();
  state.loop.beginDamping(70);
  state.advance(34);
  state.advance(34);
  state.advance(34);
  assert.equal(state.callbacks, 0);
  assert.ok(state.renders >= 1);
});

test("dispose cancels scheduled work permanently", () => {
  const state = harness();
  state.loop.invalidate("initial");
  state.loop.dispose();
  assert.equal(state.callbacks, 0);
  state.loop.invalidate("ignored");
  assert.equal(state.callbacks, 0);
});

test("numeric epsilon skips repeated state but catches real movement", () => {
  assert.equal(numericVectorsNearlyEqual([1, 2], [1.00001, 2], 1e-4), true);
  assert.equal(numericVectorsNearlyEqual([1, 2], [1.001, 2], 1e-4), false);
  assert.equal(numericVectorsNearlyEqual(null, null, 1e-4), true);
  assert.equal(numericVectorsNearlyEqual(null, [1], 1e-4), false);
});
