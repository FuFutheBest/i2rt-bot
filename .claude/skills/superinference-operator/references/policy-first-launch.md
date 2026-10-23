# Policy-First Launch Without Control Panel

Use this workflow when the user names a policy and wants the repository stack
started directly with `device_starter.py`.

## Core Model

Treat a run as one immutable launch plan, not as a single command:

```text
policy class
  -> policy consumer
  -> policy observation/control contract
  -> compatible device + device_manager + action_executor groups
  -> reviewed Hydra config name or isolated generated overlay
```

`device_starter.py` starts device producers, the device manager, and the action
executor described by the selected config. The policy consumer is a separate
process that publishes policy actions for that executor to consume. Do not start
two consumers for one action SHM and do not replace a policy class while keeping
an incompatible device or action-executor group.

## Responsibility Boundary

The operator skill owns the decision process: translate policy and hardware
intent into candidate configs, compare their complete resolved contracts,
decide whether reuse is safe, choose three verified interpreter prefixes, and
obtain confirmation at the hardware boundary. It must stop when policy intent
is insufficient to choose a camera/robot/gripper contract.

tests/config/policy_launch_plan.py is a read-only Hydra and contract checker.
Its batch mode checks explicitly packaged native OpenPI launch files for CI and
configuration-regression coverage. Its single-config mode reports
policy-to-consumer routing and static SHM/device details for a new or failing
preset. Neither mode starts a process or opens hardware. The checker does not
discover an interpreter, select or generate a config, or authorize a launch;
explicit Conda flags are compatibility formatting only, not workflow metadata.
Routine operators should use the Skill or Control Panel workflow instead of
running this checker on every start.

## Policy-to-Consumer Routing

| `policy.class` | Consumer module | Required environment probe |
| --- | --- | --- |
| `OpenPi*` | `consumer.inference.openpi_policy_infer` | OpenPI fork/source, `pi_config_name`, checkpoint, observation SHM names |
| `Diffusion*` | `consumer.inference.diffusion_policy_infer` | policy checkpoint, camera/robot SHM contract, action shape |
| `LingBotVAPolicy` | `consumer.inference.lingbot_va_infer` | LingBot checkpoint/server settings, camera keys, robot SHM |
| `NMXVLAPolicy` | `consumer.inference.nmx_vla_infer` | NMX checkpoint/service settings, camera keys, robot SHM |
| `FastWAMPolicy` | `consumer/fastwam_server.py` + `consumer.inference.fastwam_policy_infer` | FastWAM checkpoint layout, websocket endpoint, summary/policy SHM contract |

For an unrecognized class, stop and inspect `policy.POLICY_CLASSES` and the
consumer module. Never guess a launcher from a filename.

FastWAM is the one policy family here with two model processes. The Server
loads the checkpoint and owns model memory; the lightweight client connects to
the Server and must not receive a second checkpoint override. Resolve and
review one native config for both processes. A valid checkpoint is a directory
containing `config.yaml`, `dataset_statistics.json`, and at least one
`checkpoints/weights/*.pt` file.

For a direct FastWAM launch, keep the device runtime in its approved device
environment. Run the Server in its separately verified NMX/CUDA environment,
then run the lightweight client with the separately verified policy prefix:

```bash
<fastwam-server-prefix> consumer/fastwam_server.py --config-name <native-hydra-name>
<policy-prefix> -m consumer.inference.fastwam_policy_infer --config-name <native-hydra-name>
```

Wait for the Server's identity-checked `/healthz` readiness before starting the
client. Use `policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam` or its `_rtc`
variant for both direct launch and Control Panel. They share the dedicated
FastWAM two-wrist device group; there is no separate panel config or hardware
binding. The group remains separate from OpenPI so camera settings can diverge,
and both currently use `usb_port_requirement: any` during the camera-port
investigation. Direct `DeviceStarter` also takes an advisory resource lock,
while the panel's independent `server` stage represents the same process
boundary.

## Environment Resolution

Treat the planner, device runtime, and policy consumer as three independent
interpreter decisions. Represent each as a command prefix, not as a presumed
Conda environment name. Examples include an absolute venv interpreter, a
Conda prefix ending in python, or a uv-project command. A uv prefix must name a
known project working directory or explicit project path and must not silently
sync or install dependencies. The repository root is set up through a
uv-created venv but is not itself a universal uv project; third_party/openpi
has its own uv project and Python version requirements.

Resolve each prefix in this order:

1. Use an exact interpreter/prefix explicitly supplied by the operator.
2. Reuse a machine-local launcher command or interpreter path that was already
   verified for the selected config and workstation.
3. Consider the active environment only after a read-only import/version probe
   proves it owns the required dependencies.
4. Otherwise report the missing prefix/dependency and request a choice. Do not
   silently activate, install into, or switch environments.

Never infer the device or policy prefix from a policy name, checkpoint path,
config filename, or an arbitrary available Conda environment. The
policy.infer_rollout.python_path value controls only an auxiliary rollout-saver
subprocess. When rollout saving is enabled, verify that fourth executable and
its saver dependencies separately; never promote it to either main-process
prefix. A successful import probe still does not prove hardware connectivity or
authorize startup.

For the currently documented split UMI/OpenPI workstation path,
scripts/lingbotva/run_infer_split.sh can be reused only after the operator
confirms its two Conda environments and config match the target. It is not a
generic OpenPI launcher.

## Config names and reuse

A Hydra source name is the YAML path relative to `configs/`, with its
suffix removed. It is shared by `device_starter.py`, the policy consumer, and
Control Panel; it does not identify a panel-specific config:

| YAML file | Control Panel name | Native Hydra name |
| --- | --- | --- |
| `configs/foo.yaml` | `foo` | `foo` |
| `configs/policy_infer/openpi/foo.yaml` | `policy_infer/openpi/foo` | `policy_infer/openpi/foo` |

Pass the shared Hydra source name to `device_starter.py`, the policy consumer,
and Control Panel. For example, a file under `configs/policy_infer/openpi/` must be launched
as `--config-name policy_infer/openpi/foo` in each applicable entrypoint.

For CI or after adding/changing a native OpenPI launch config, check the
curated set without hardware:

~~~bash
<planner-prefix> tests/config/policy_launch_plan.py --all-supported --json
~~~

The batch command scans explicitly packaged native OpenPI launch files,
excluding generated_*.yaml artifacts and Hydra config groups. It exits non-zero
when any preset cannot compose or has a missing policy consumer, enabled device,
or control channel. Its JSON report is intended for CI logs and machine
inspection.

The same check is covered by the hardware-free planner tests, so normal CI
can run:

~~~bash
python -m pytest tests/config/test_policy_launch_plan.py -q
~~~

For a panelless run with a new, ambiguous, or failing preset, inspect one
native config:

~~~bash
<planner-prefix> tests/config/policy_launch_plan.py --config-name openpi/<name> --json
~~~

This command composes Hydra and reports the resolved policy class, devices, SHM
contract, and execution parameters. Its default entrypoints use neutral
prefix placeholders until the Skill resolves the planner, device, and policy
prefixes. Do not execute those placeholders directly. Routine launches do not
require a user to run the checker manually; the Skill or Control Panel should
invoke it when the contract is new or uncertain.

Reject a plan with unresolved interpolation, a missing checkpoint/calibration
artifact, an unknown consumer, or a policy/device/control mismatch.

Reuse an existing config only when the resolved contract matches:

1. Search candidates by resolved `policy.class`, not filenames.
2. Compare the full `policy.params`, `policy.obs.devices`,
   `policy.controls`, enabled device classes/IDs, ActionExecutor,
   DeviceManager, FPS, chunk length, and SHM shapes/dtypes.

The repository-wide composition contracts can also be checked without
hardware:

```bash
python -m pytest tests/config -q
```

Inspect the selected config in this order: top-level preset, `defaults`,
`devices`, `device_manager`, `action_executor`,
`policy`, and `runtime`.

## Isolated generated config

Generate a new config only when no existing resolved contract matches. Never
edit or overwrite a source preset. The generated artifact must be a complete
effective configuration, not a small YAML fragment that changes only
`policy.class`. Standalone preflight accepts config-group names for read-only
inspection, but generation rejects `devices/*`, `device_manager/*`,
`action_executor/*`, and other non-launch namespaces; use a root-level,
`openpi/*` preset as the source. Device-only presets may
omit `policy` when they explicitly use the `device` workflow, but they still
need the enabled device list and the manager contract required by their
runtime.

For a stopped Control Panel session, use `POST /api/configs/generated`.
It writes `configs/generated_<id>.yaml`, preserves configured LKTech calibration
paths without copying their JSON files, and stores binding/calibration artifact
provenance. The selected source preset is never changed.

For a panelless run, first inspect the complete read-only contract with the
launch planner. When USB or camera bindings need to be refreshed, use the
interactive
[`scripts/diagnostics/rollout_device_preflight.py`](../../../../scripts/diagnostics/rollout_device_preflight.py)
workflow against a complete top-level rollout source; it opens hardware and may
move grippers. It rejects `configs/devices/*.yaml`, `configs/devices/infer/*.yaml`, and thin FastWAM aliases.
Generate only after the read-only launch plan identifies a complete reusable
base. The workflow can copy its fully resolved contract into an isolated
top-level Hydra preset, preserve the complete `policy` node, and add provenance
including the source native Hydra name, source revision, binding identities, and
observed calibration artifacts. See the
[`root command reference`](../../../../useful_tools_commands.md) for the interactive
workflow and its persistence behavior. Do not use `params: {}`
for an OpenPI policy: `OpenPiUmiPolicy` requires at least
`pi_config_name` and `ckpt_dir`, and the policy's observation,
control, FPS, chunk, task, robot-count, and calibration settings must remain
compatible with the checkpoint.

After generation, compose the new preset with the launch-plan command and
inspect its resolved device keys, policy observation keys, control SHM
shape/dtype, checkpoint path, signal files, and referenced calibration paths. If
any value is unresolved or workstation-specific, stop for a user decision
instead of silently falling back to another config.

## Launch Sequence

Once the plan has been shown and the user has authorized the hardware run:

For a standalone launch, the final reviewed config should have passed the launch
planner and any required hardware binding workflow; repeat those checks if a
binding, calibration, or environment value changed after the launch plan.

1. Start the reviewed config from the repository root:

   ```bash
   <device-prefix> device_starter.py --config-name <native-hydra-name>
   ```

2. Wait for configured device processes and summary SHM. Confirm the action
   executor target SHM exists and has the expected shape/dtype.
3. For a non-FastWAM policy, start exactly the consumer selected by the policy
   routing table with the separately verified policy prefix:

   ```bash
   <policy-prefix> -m <consumer-module> --config-name <native-hydra-name>
   ```
4. For `FastWAMPolicy`, start the Server before the client, verify its
   identity-checked `/healthz`, and then start the client:

   ```bash
   <fastwam-server-prefix> consumer/fastwam_server.py --config-name <native-hydra-name>
   <policy-prefix> -m consumer.inference.fastwam_policy_infer --config-name <native-hydra-name>
   ```

5. Verify logs show observation SHM attachment and action publication before
   enabling any motion-producing mode.
6. Stop the policy client first. For FastWAM, stop the Server next; then stop
   `device_starter.py` with Ctrl-C. Confirm child processes and shared-memory
   producers have exited.

Starting `device_starter.py` is hardware-facing even when the policy consumer is
not started. A dry-run, config composition, or launch-plan request must not start
it. For motion, homing, replay, or policy control, request confirmation at the
final boundary and report the exact config, policy, environment, and targets.

## Launch Plan Output

Before execution, report a compact plan containing:

- resolved config name and whether it is reused or generated;
- workflow_kind, policy class, consumer module, and the selected planner/device/policy prefixes;
- enabled device classes/IDs, camera and robot SHM keys;
- action control SHM names, shapes, dtypes, FPS, and chunk length;
- checkpoint and calibration paths; missing, stale, invalid, or unverified
  gripper artifacts block a hardware launch;
- final device_starter and consumer commands rendered with those prefixes;
- the first blocking mismatch, if the plan is not executable.
