import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const source = readFileSync(resolve(root, "static/device_control.js"), "utf8");
const template = readFileSync(resolve(root, "templates/device_control.html"), "utf8");
const css = readFileSync(resolve(root, "static/device_control.css"), "utf8");

function body(name, nextName) {
  const start = source.indexOf(`  function ${name}(`);
  const asyncStart = source.indexOf(`  async function ${name}(`);
  const actualStart = start >= 0 ? start : asyncStart;
  assert.ok(actualStart >= 0, `${name} missing`);
  const end = source.indexOf(`  function ${nextName}(`, actualStart + 1);
  const asyncEnd = source.indexOf(`  async function ${nextName}(`, actualStart + 1);
  const candidates = [end, asyncEnd].filter((value) => value >= 0);
  assert.ok(candidates.length, `${nextName} boundary missing`);
  return source.slice(actualStart, Math.min(...candidates));
}

test("checkpoint UI exposes default override effective and server browser", () => {
  for (const id of [
    "checkpoint-default-path",
    "checkpoint-override-path",
    "checkpoint-effective-path",
    "checkpoint-browse-button",
    "checkpoint-clear-button",
    "checkpoint-browser-dialog",
    "checkpoint-browser-list",
    "checkpoint-browser-select",
  ]) assert.match(template, new RegExp(`id="${id}"`));
  const scriptVersion = template.match(/device_control\.js\?v=([^"']+)/);
  const styleVersion = template.match(/device_control\.css\?v=([^"']+)/);
  assert.ok(scriptVersion, "device_control.js cache version missing");
  assert.ok(styleVersion, "device_control.css cache version missing");
  assert.equal(scriptVersion[1], styleVersion[1]);
  assert.match(css, /\.checkpoint-browser-dialog/);
  assert.match(css, /\.checkpoint-paths/);
});

test("browse responses are context-bound and stale requests are discarded", () => {
  const matcher = body("checkpointBrowsePayloadMatches", "checkpointStateForRender");
  assert.match(matcher, /payload\.config_name/);
  assert.match(matcher, /payload\.revision/);
  assert.match(matcher, /payload\.generation/);
  assert.match(matcher, /checkpointContextsEqual\(context, checkpointContextFromState\(\)\)/);

  const browse = body("loadCheckpointBrowser", "openCheckpointBrowser");
  assert.match(browse, /\+\+app\.checkpointBrowserRequest/);
  assert.match(browse, /checkpointBrowserController\?\.abort\(\)/);
  assert.match(browse, /requestId !== app\.checkpointBrowserRequest/);
  assert.match(browse, /checkpointBrowsePayloadMatches\(context, payload\)/);
});

test("set clear and inference start carry checkpoint CAS context", () => {
  const mutate = body("mutateCheckpointOverride", "selectOptionSignature");
  assert.match(mutate, /expected_selected_config: context\.selectedConfig/);
  assert.match(mutate, /expected_revision: context\.revision/);
  assert.match(mutate, /expected_generation: context\.generation/);
  assert.match(mutate, /method: path === null \? "DELETE" : "POST"/);
  assert.match(mutate, /checkpointMutationController\?\.abort\(\)/);
  assert.match(mutate, /signal: controller\.signal/);
  assert.match(mutate, /checkpointContextsEqual\(context, current\)/);
  assert.match(mutate, /current\.generation === response\.generation/);
  assert.match(mutate, /!capturedStillCurrent && !responseAlreadyObserved/);
  assert.match(mutate, /error\?\.name === "AbortError"/);
  assert.match(mutate, /error\.status === 409/);

  const workflow = body("runWorkflowAction", "cameraEndpoint");
  assert.match(workflow, /stage === "inference" && action === "start"/);
  assert.match(workflow, /expected_selected_config: checkpointContext\.selectedConfig/);
  assert.match(workflow, /expected_revision: checkpointContext\.revision/);
  assert.match(workflow, /expected_generation: checkpointContext\.generation/);
});

test("checkpoint editor locks on inference transitions but not runtime activity", () => {
  const locked = body("checkpointEditorLocked", "invalidateCheckpointBrowser");
  assert.match(locked, /workflowPending\.has\("inference"\)/);
  assert.match(locked, /\["starting", "running", "stopping"\]/);
  assert.doesNotMatch(locked, /workflowStage\("runtime"\)/);
});
