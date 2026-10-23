import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const source = readFileSync(resolve(root, "static/device_control.js"), "utf8");
const marker = /  initialize\(\);\r?\n\}\)\(\);/;
assert.match(source, marker);

const timers = [];
const nodes = new Map();
let dialogOpenCount = 0;

function node(id = "") {
  return {
    id,
    value: "",
    content: "",
    disabled: false,
    hidden: false,
    open: false,
    textContent: "",
    title: "",
    dataset: {},
    classList: {
      toggle() {},
      add() {},
      remove() {},
    },
    setAttribute() {},
    removeAttribute() {},
    hasAttribute() { return false; },
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
    appendChild() {},
    replaceChildren() {},
    focus() {},
    close() { this.open = false; },
    showModal() {
      this.open = true;
      dialogOpenCount += 1;
    },
  };
}

globalThis.document = {
  hidden: false,
  activeElement: null,
  getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, node(id));
    return nodes.get(id);
  },
  querySelector(selector) {
    if (!nodes.has(selector)) nodes.set(selector, node(selector));
    return nodes.get(selector);
  },
  querySelectorAll() { return []; },
  createElement(tag) { return node(tag); },
};
globalThis.window = {
  setTimeout(callback) {
    timers.push(callback);
    return timers.length;
  },
  clearTimeout() {},
  confirm() { return false; },
};
globalThis.Option = class Option {
  constructor(label, value) {
    this.label = label;
    this.value = value;
  }
};

const injected = String.raw`  globalThis.__gripperPermissionContextContract = {
    reset() {
      app.state = {};
      app.selectedConfig = "";
      app.gripperPermissionPending = false;
      app.gripperPermissionMode = "sudo";
      app.gripperPermissionPromptPromise = null;
      app.gripperPermissionPromptResolve = null;
      app.gripperPermissionPromptSignature = "";
      app.gripperPermissionContext = null;
      app.gripperPermissionDismissedSignature = "";
      app.gripperPermissionAutoScheduled = false;
      app.stateAppliedRequestSerial = 0;
      app.stateRequestSerial = 0;
      app.gripperSyncEpoch = 0;
      app.gripperSnapshotRevision = null;
      app.gripperSnapshotRequest = 0;
      app.gripperTopSnapshotRequest = 0;
      app.gripperSideSnapshotRevisions.clear();
      app.gripperSideSnapshotRequests.clear();
    },
    setPermission({
      revision,
      contextId,
      required,
      targetPorts = [],
      affectedPorts = [],
      targetCanInterfaces = [],
      affectedCanInterfaces = [],
    }) {
      app.state = {
        ...app.state,
        revision,
        grippers: {
          revision,
          permission: {
            context_id: contextId,
            required,
            target_ports: targetPorts,
            affected_ports: affectedPorts,
            target_can_interfaces: targetCanInterfaces,
            affected_can_interfaces: affectedCanInterfaces,
          },
          sides: {},
        },
      };
      return normalizeGrippers().permission;
    },
    maybe(permission) {
      maybePromptGripperPermission(permission);
    },
    force() {
      return requestGripperPermission({ force: true });
    },
    cancel() {
      cancelGripperPermissionPrompt();
    },
    dismissed() {
      return app.gripperPermissionDismissedSignature;
    },
    signature(permission) {
      return gripperPermissionSignature(permission);
    },
    applyBinding(payload, requestSerial, gripperEpoch, gripperRequestId = requestSerial) {
      return applyBindingResponseState(payload, {
        requestSerial,
        gripperEpoch,
        gripperRequestId,
      });
    },
    applyGrippers(raw, epoch, requestId, controllerRevision) {
      return applyGripperSnapshot(raw, {
        epoch, requestId, controllerRevision,
      });
    },
    setBindingBaseline({ revision, requestSerial, gripperEpoch, contextId }) {
      app.state = {
        revision,
        state_sync: {
          epoch: "epoch-a",
          sequence: 7,
          event_revision: 7,
          observed_revision: revision,
          stale: false,
          authoritative: true,
        },
        grippers: {
          revision,
          permission: {
            context_id: contextId,
            required: false,
            target_ports: ["/dev/ttyUSB0"],
            affected_ports: [],
          },
          sides: {},
        },
        devices: [],
      };
      app.stateAppliedRequestSerial = requestSerial;
      app.gripperSyncEpoch = gripperEpoch;
      app.gripperSnapshotRevision = revision;
      app.gripperTopSnapshotRequest = 0;
    },
    state() {
      return app.state;
    },
  };
})();`;

eval(source.replace(marker, injected));

function flushTimers() {
  while (timers.length) {
    const callback = timers.shift();
    callback();
  }
}

test("permission prompts are keyed by device context rather than controller revision", async () => {
  const contract = globalThis.__gripperPermissionContextContract;
  contract.reset();
  dialogOpenCount = 0;
  timers.length = 0;

  const first = contract.setPermission({
    revision: 10,
    contextId: "tty-context-a",
    required: true,
    targetPorts: ["/dev/ttyUSB0"],
    affectedPorts: ["/dev/ttyUSB0"],
  });
  assert.equal(first.contextId, "tty-context-a");
  assert.doesNotMatch(contract.signature(first), /revision/);

  contract.maybe(first);
  flushTimers();
  assert.equal(dialogOpenCount, 1);
  contract.cancel();
  const dismissed = contract.dismissed();
  assert.ok(dismissed);

  const sameDeviceNewRevision = contract.setPermission({
    revision: 11,
    contextId: "tty-context-a",
    required: true,
    targetPorts: ["/dev/ttyUSB0"],
    affectedPorts: ["/dev/ttyUSB0"],
  });
  contract.maybe(sameDeviceNewRevision);
  flushTimers();
  assert.equal(dialogOpenCount, 1);
  assert.equal(contract.dismissed(), dismissed);

  const accessible = contract.setPermission({
    revision: 12,
    contextId: "tty-context-a",
    required: false,
    targetPorts: ["/dev/ttyUSB0"],
    affectedPorts: [],
  });
  contract.maybe(accessible);
  assert.equal(contract.dismissed(), "");

  const replacement = contract.setPermission({
    revision: 13,
    contextId: "tty-context-b",
    required: true,
    targetPorts: ["/dev/ttyUSB0"],
    affectedPorts: ["/dev/ttyUSB0"],
  });
  contract.maybe(replacement);
  flushTimers();
  assert.equal(dialogOpenCount, 2);
  contract.cancel();

  const forced = contract.force();
  assert.equal(dialogOpenCount, 3);
  contract.cancel();
  assert.equal(await forced, false);
});

test("CAN-only permission state opens the same password prompt", () => {
  const contract = globalThis.__gripperPermissionContextContract;
  contract.reset();
  dialogOpenCount = 0;
  timers.length = 0;

  const permission = contract.setPermission({
    revision: 30,
    contextId: "can-context-a",
    required: true,
    targetCanInterfaces: ["can0"],
    affectedCanInterfaces: ["can0"],
  });
  assert.equal(permission.required, true);
  assert.deepEqual(permission.targetPorts, []);
  assert.deepEqual(permission.targetCanInterfaces, ["can0"]);

  contract.maybe(permission);
  flushTimers();
  assert.equal(dialogOpenCount, 1);
  contract.cancel();
});

test("stale binding responses cannot overwrite newer state or gripper permission context", () => {
  const contract = globalThis.__gripperPermissionContextContract;
  contract.reset();
  contract.setBindingBaseline({
    revision: 20,
    requestSerial: 8,
    gripperEpoch: 4,
    contextId: "tty-context-new",
  });

  const staleByRequest = contract.applyBinding({
    revision: 21,
    grippers: {
      revision: 21,
      permission: {
        context_id: "tty-context-old-request",
        required: true,
        target_ports: ["/dev/ttyUSB0"],
        affected_ports: ["/dev/ttyUSB0"],
      },
      sides: {},
    },
    devices: [],
  }, 7, 4);
  assert.equal(staleByRequest, false);
  assert.equal(contract.state().revision, 20);

  const staleByRevision = contract.applyBinding({
    revision: 19,
    grippers: {
      revision: 19,
      permission: {
        context_id: "tty-context-old-revision",
        required: true,
        target_ports: ["/dev/ttyUSB0"],
        affected_ports: ["/dev/ttyUSB0"],
      },
      sides: {},
    },
    devices: [],
  }, 9, 4);
  assert.equal(staleByRevision, false);
  assert.equal(contract.state().revision, 20);

  const staleByEpoch = contract.applyBinding({
    revision: 21,
    grippers: { revision: 21, sides: {} },
    devices: [],
  }, 9, 3);
  assert.equal(staleByEpoch, false);

  contract.applyGrippers({
    revision: 20,
    permission: {
      context_id: "tty-context-newer-gripper",
      required: false,
      target_ports: ["/dev/ttyUSB0"],
      affected_ports: [],
    },
    sides: {},
  }, 4, 12, 20);
  const staleGripperPart = contract.applyBinding({
    revision: 20,
    grippers: {
      revision: 20,
      permission: {
        context_id: "tty-context-old-gripper",
        required: true,
        target_ports: ["/dev/ttyUSB0"],
        affected_ports: ["/dev/ttyUSB0"],
      },
      sides: {},
    },
    devices: [],
  }, 9, 4, 10);
  assert.equal(staleGripperPart, true);
  assert.equal(
    contract.state().grippers.permission.context_id,
    "tty-context-newer-gripper",
  );

  const accepted = contract.applyBinding({
    revision: 21,
    grippers: {
      revision: 21,
      permission: {
        context_id: "tty-context-current",
        required: false,
        target_ports: ["/dev/ttyUSB0"],
        affected_ports: [],
      },
      sides: {},
    },
    devices: [],
  }, 10, 4, 13);
  assert.equal(accepted, true);
  assert.equal(contract.state().revision, 21);
  assert.equal(contract.state().state_sync.epoch, "epoch-a");
  assert.equal(
    contract.state().grippers.permission.context_id,
    "tty-context-current",
  );
});
