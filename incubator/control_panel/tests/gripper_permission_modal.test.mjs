import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const source = readFileSync(resolve(root, "static/device_control.js"), "utf8");
const template = readFileSync(resolve(root, "templates/device_control.html"), "utf8");

test("plain HTTP serializes the password only in the unified sudo request body", () => {
  const functionStart = source.indexOf("async function submitGripperPermissionRepair()");
  const functionEnd = source.indexOf("function renderGripperPortSelect", functionStart);
  assert.ok(functionStart > 0 && functionEnd > functionStart);
  const body = source.slice(functionStart, functionEnd);
  const serialize = body.indexOf("ports: context.ports,");
  const serializeCan = body.indexOf("can_interfaces: context.canInterfaces,", serialize);
  const serializePassword = body.indexOf("password,", serialize);
  const clearInput = body.indexOf('elements.gripperSudoPassword.value = ""');
  const send = body.indexOf('await api("/api/hardware/permissions/sudo"');
  assert.doesNotMatch(body, /secure_transport_required|普通 HTTP 禁止/);
  assert.ok(serialize >= 0 && serializeCan > serialize && serializePassword > serializeCan);
  assert.ok(serialize < clearInput);
  assert.ok(clearInput < send);
  assert.match(body, /cache:\s*"no-store"/);
  assert.match(body, /credentials:\s*"same-origin"/);
  assert.match(body, /referrerPolicy:\s*"no-referrer"/);
  assert.match(body, /"X-CSRF-Token": csrfToken/);
});

test("password modal describes fixed ttyUSB and CAN commands", () => {
  assert.match(template, /id="gripper-permission-dialog"/);
  assert.match(template, /type="password"/);
  assert.match(template, /autocomplete="off"/);
  assert.match(source, /sudo chmod 0666/);
  assert.match(source, /sudo ip link down\/up/);
  assert.match(source, /const mode = "sudo"/);
  assert.doesNotMatch(source, /helper|permissions\/repair|permission_helper_required/i);
  assert.match(source, /addEventListener\("click", cancelGripperPermissionPrompt\)/);
  const cancelStart = source.indexOf("function cancelGripperPermissionPrompt()");
  const cancelEnd = source.indexOf("function requestGripperPermission", cancelStart);
  const cancelBody = source.slice(cancelStart, cancelEnd);
  assert.doesNotMatch(cancelBody, /api\(|fetch\(|submitGripperPermissionRepair/);
});

test("password is not routed through persistent browser storage or a URL", () => {
  assert.doesNotMatch(source, /localStorage[^\n]*password/i);
  assert.doesNotMatch(source, /sessionStorage[^\n]*password/i);
  assert.doesNotMatch(source, /location[^\n]*password/i);
  assert.match(source, /api\("\/api\/hardware\/permissions\/sudo"/);
  assert.doesNotMatch(source, /[?&]password=/i);
});

test("hardware actions retry only after structured ttyUSB or CAN recovery", () => {
  const retryStart = source.indexOf("async function withGripperPermissionRetry(operation)");
  const retryEnd = source.indexOf("function firstDefined", retryStart);
  const retryBody = source.slice(retryStart, retryEnd);
  assert.match(retryBody, /isGripperPermissionRequiredError/);
  assert.match(retryBody, /requestGripperPermission\(\{ force: true \}\)/);
  assert.equal((retryBody.match(/return operation\(\)/g) || []).length, 1);
  assert.match(source, /withGripperPermissionRetry\(\(\) => api\("\/api\/grippers\/probe"/);
  assert.match(source, /withGripperPermissionRetry\(\(\) => api\("\/api\/grippers\/calibrate"/);
  assert.match(source, /withGripperPermissionRetry\(\(\) => api\("\/api\/grippers\/position"/);
  const deviceStart = source.slice(
    source.indexOf("async function runDeviceAction"),
    source.indexOf("function renderLogDeviceOptions"),
  );
  assert.match(deviceStart, /isI2rtHeadDevice\(device\) && action !== "stop"/);
  assert.match(deviceStart, /withGripperPermissionRetry/);
  const workflowStart = source.slice(
    source.indexOf("async function runWorkflowAction"),
    source.indexOf("function cameraEndpoint"),
  );
  assert.match(workflowStart, /action === "start"/);
  assert.match(workflowStart, /withGripperPermissionRetry/);
});

test("modal freezes disclosed ports and revision into the sudo permission post", () => {
  assert.match(source, /app\.gripperPermissionContext = Object\.freeze\(\{/);
  assert.match(source, /revision: grippers\.revision/);
  assert.match(source, /ports: Object\.freeze\(\[\.\.\.permission\.targetPorts\]\)/);
  assert.match(source, /canInterfaces: Object\.freeze\(\[\.\.\.permission\.targetCanInterfaces\]\)/);
  const submitStart = source.indexOf("async function submitGripperPermissionRepair()");
  const submitEnd = source.indexOf("function renderGripperPortSelect", submitStart);
  const submit = source.slice(submitStart, submitEnd);
  assert.match(submit, /revision: context\.revision/);
  assert.match(submit, /ports: context\.ports/);
  assert.match(submit, /can_interfaces: context\.canInterfaces/);
  assert.match(submit, /permission_context_changed/);
});
