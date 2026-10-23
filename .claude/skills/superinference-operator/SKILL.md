---
name: superinference-operator
description: Use for day-to-day SuperInference operation requests involving robot state probes, runtime lifecycle, policy launch, SHM/config inspection, recording, calibration, replay, or hardware-facing troubleshooting.
---

# Superinference Operator

This is the routing and safety layer for operating the repository. Convert the user's
intent into an existing repository entrypoint, run it from the repository root, and
report the result with the target, environment, units, and any safety limitation.
Do not invent a new script when an existing entrypoint below already owns the task.

## Operating Boundaries

- Treat every command that can connect to hardware, energize an arm, move an actuator,
  overwrite calibration, or replay a trajectory as hardware-facing.
- Read-only inspection is allowed when it is explicitly requested. Motion, enabling,
  fault clearing, homing, calibration, replay, and policy control require a final
  confirmation immediately before execution unless the user has already explicitly
  authorized that exact operation and target.
- Never use `device_starter.py` as a harmless config validator. It starts configured
  device processes and can connect to physical hardware.
- Run commands from the repository root (`git rev-parse --show-toplevel`) unless
  the command documents another working directory.
- Use the interpreter from the environment that owns the required dependency. Do not
  silently switch between the core Python environment, a model environment, and a
  vendor SDK environment.
- Preserve user changes in a dirty worktree. Do not reset, checkout, or overwrite
  unrelated files.

## Environment Decision

For a policy-first request, choose independent command prefixes rather than
assuming one global environment:

- planner prefix: imports Hydra and OmegaConf only to compose and inspect a
  config;
- device prefix: owns the configured vendor SDKs and launches device_starter;
- policy prefix: owns the selected model stack and launches its consumer.
- FastWAM Server prefix: owns the NMX/CUDA model stack; it may differ from the
  lightweight policy-client prefix.

Resolve every prefix through
[`runtime-profile-management`](../runtime-profile-management/SKILL.md). Run
`runtime_profiles.py doctor --config-name <name>` before a model or hardware
launch, use `run --role <role>` for execution, and use `command --json` when only
reporting the resolved argv. An explicit user path or existing role-specific
`SUPERINFERENCE_*_PYTHON` variable remains a higher-priority one-off override.
Do not parse the local registry in this Skill or infer an environment from a
policy class, checkpoint path, config filename, or available Conda environment
names.

When `policy.infer_rollout` is enabled, its Python belongs only to a fourth,
auxiliary rollout-saver subprocess. Native launches first use a verified
config-specific `recording` profile when one is bound, then the explicit
config/current-interpreter fallback. Control Panel or split-SDK deployments may
still override it with `SUPERINFERENCE_RECORDER_PYTHON`. Never hardcode
machine-local interpreter paths in reusable configs, and never treat the recorder
interpreter as evidence that DeviceStarter or the policy consumer can run with
that profile.

For a fresh machine without verified prefixes, do not start resolving environments
ad hoc. First use
[`workstation-inference-bootstrap`](../workstation-inference-bootstrap/SKILL.md).
It owns OpenPI submodule initialization, source-environment creation, GPU checks,
and Flexiv robot-software-to-RDK compatibility. Return here only after it reports
the installed absolute interpreter paths and an approved RDK selection.

## Intent Router

Choose the narrowest existing entrypoint that satisfies the request.

### Current robot state or joint angles

For a live Flexiv Rizon state read, use the existing adapter utility:

```bash
python devices/flexiv_adapter.py \
  --task read_states \
  --robot-sn "<robot-serial-number>" \
  --backend python
```

`--backend pybind` is valid only when the compiled `flexiv_rdk_ext` binding is
available. The command is read-only and prints `joints` in radians, `pose_wxyz`,
tool metadata, and the adapter's connection errors. It does not start the device
runtime. Normalize shorthand requests such as “device_starter read_states” to this
command; `device_starter.py` currently has no `read_states` subcommand.

The same utility also exposes three explicit tasks:

- `test_rdk`: connect and construct the adapter/robot only; it does not enable;
- `move_to_joints`: seven joint targets in radians;
- `move_to_pose`: `[x, y, z, qw, qx, qy, qz]`, with position in meters and a
  `wxyz` quaternion.

Both motion tasks clear faults when present, enable the robot, wait for operational
state, switch control mode, and send a target. Require an explicit `--target` and
final motion authorization; never rely on the utility's example default target.

When the user asks for health of multiple configured arms, use the read-only form of
the diagnostic script instead:

```bash
python scripts/legacy/check_robot_arm.py \
  --config configs/devices/infer/<device-config>.yaml
```

Do not add `--enable` unless the user explicitly approves energizing the arms. Do not
claim that a state read came from SHM: the adapter command reads the Flexiv SDK
directly. If the user specifically asks for the last state published by a running
runtime, identify the summary-SHM source and use an existing SHM reader or inspect
the configured device contract rather than restarting the runtime.

Do not silently use the adapter's default serial number when the target robot is
ambiguous. Resolve it from the selected device config or ask for the serial number.

### Runtime lifecycle

Start devices with a reviewed top-level Hydra config:

```bash
<device-prefix> device_starter.py --config-name <config-name>
```

DeviceStarter detects initialization from changing live frames in every enabled
device SHM before starting DeviceManager; configs do not specify a fixed startup
delay.

The optional `-v`/`--visualize` flag enables runtime visualizers. Start the policy
consumer separately, after the device and summary SHM are available:

```bash
<policy-prefix> -m consumer.inference.openpi_policy_infer --config-name <config-name>
<policy-prefix> -m consumer.inference.lingbot_va_infer --config-name <config-name>
<policy-prefix> -m consumer.inference.nmx_vla_infer --config-name <config-name>
```

A Hydra config name is the path under `configs/` without its suffix. In
particular, `configs/policy_infer/openpi/foo.yaml` is `policy_infer/openpi/foo` for these commands.
`policy_infer/openpi/foo` is the shared source name, not a Control Panel-native config:
Control Panel and `device_starter.py` consume that same source and normalized
mapping. Do not maintain a second producer stack or duplicate hardware bindings
for an ordinary OpenPI launch.

To open the panel on the exact same source used by a direct device launch, pass
the name explicitly:

```bash
<device-prefix> -m incubator.control_panel --config-name openpi/<name>
```

With no `--config-name`, both entrypoints use Hydra's shared default `config`.

The canonical OpenPI and FastWAM configs keep Hydra's working output at the
repository root but route Hydra job logs to `logs/<job-name>.log`. These files
are separate from the Loguru-managed `logs/app.log` and `logs/error.log` sinks.

When a session needs a persisted effective configuration, use the stopped-only
generated-config endpoint. It writes `configs/generated_<id>.yaml`, returns the
native name `generated_<id>`, and leaves the selected source YAML unchanged.
That artifact can be inspected with the launch planner and started directly by
`device_starter.py` using the same `--config-name generated_<id>` argument.

FastWAM is the supported two-process model workflow. Start its model Server and
wait for identity-checked `/healthz` readiness before starting the lightweight
client:

```bash
<fastwam-server-prefix> consumer/fastwam_server.py \
  --config-name policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam
<policy-prefix> -m consumer.inference.fastwam_policy_infer \
  --config-name policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam
```

The Server owns `policy.params.ckpt_path` and model memory; the client only
consumes summary SHM and publishes policy SHM. In the Control Panel, FastWAM
adds an independent `server` stage that may load before DeviceManager, while
`inference` still requires both the device runtime and a ready Server. The panel
and direct launcher both use `policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam`
or its `_rtc` variant; there is no separate panel config or hardware binding.
These FastWAM entries share their dedicated FastWAM device group across
server/client/panel; they intentionally do not reuse the OpenPI device group so
camera settings can diverge. Both groups currently set
`usb_port_requirement: any` during the camera-port investigation.

The planner is primarily an automated launch-contract checker, not a routine
operator command. In CI or after adding/changing a native OpenPI launch config,
check the curated set in one pass:

~~~bash
<planner-prefix> tests/config/policy_launch_plan.py --all-supported --json
~~~

This checks native OpenPI launch files under configs/policy_infer/openpi/, starts no process,
opens no hardware, and exits
non-zero when a preset cannot compose or has a missing policy consumer, enabled
device, or control channel. Generated configs and Hydra config groups are
intentionally outside this repository-contract scope.

The same check is covered by the hardware-free planner tests, so normal CI
can run:

~~~bash
python -m pytest tests/config/test_policy_launch_plan.py -q
~~~

For a single new, ambiguous, or failing preset, inspect its contract directly:

~~~bash
<planner-prefix> tests/config/policy_launch_plan.py --config-name <native-hydra-name> --json
~~~

The single-config report is evidence for native Hydra resolution, consumer
routing, and the static SHM/device contract. It does not choose an interpreter,
select or generate a config, or authorize startup. Resolve device and policy
prefixes through the Environment Decision above before rendering final
commands. Normal launches should be driven by this Skill or the Control Panel;
users do not need to run the planner on every routine start. Stop a foreground
process with Ctrl-C so its SHM and vendor handles can be cleaned up. For an
already-running stack, inspect processes and logs before starting a second copy.

The repository launchers (`rollout_device_preflight.py`, `device_starter.py`,
`python -m incubator.control_panel`, and `visualize_flexiv_assembly.py`) perform a
startup process preflight. It discovers same-user instances of these launchers
and their multiprocessing descendants, sends `SIGTERM`, and waits up to ten
seconds before continuing. A survivor blocks the new start and reports its PID
and command line; startup cleanup never sends `SIGKILL` because a motion worker
may still need its robot-rest and shared-memory cleanup path. The control panel
has `--no-process-cleanup` for isolated tests or an operator who has explicitly
verified that another launcher must remain running.

For an operator who has physically verified that the robot is stopped and needs
to clear an unresponsive process, pass `--force-process-cleanup` to the control
panel, rollout preflight, or assembly viewer. `device_starter.py` accepts the
same opt-in through `SUPERINFERENCE_FORCE_PROCESS_CLEANUP=1`. Forced cleanup
first asks any owning `systemd --user` service to stop (preventing respawn),
then sends `SIGKILL` to remaining matched processes and descendants. Do not use
this mode while a robot may still be moving; normal graceful cleanup remains
the default.

When a launcher is supervised by `systemd --user`, stopping its PID alone can
cause an immediate replacement PID. The post-cleanup scan treats that replacement
as a survivor; stop the owning user service before retrying a hardware start.

For a policy-first launch without Control Panel, read
[references/policy-first-launch.md](references/policy-first-launch.md). It defines
how to resolve a policy to its consumer and compatible device/action-executor
contract, reuse an existing preset, or create an isolated generated overlay
without overwriting source configs. Use the single-config planner only for a
new or uncertain contract; use the batch mode for CI and regression checks.

#### No-Control-Panel device preflight

For an interactive rollout hardware binding (USB numbers may have changed), use
the rollout-specific workflow:

Preflight prompts, status messages and owned errors are in Chinese. Keep config
keys, serials, paths and control keys unchanged. Camera instructions are printed
in the terminal in Chinese; the OpenCV Hershey overlay shows only device paths
and key symbols because it cannot render Chinese. Third-party SDK/Qt diagnostics
remain verbatim for troubleshooting.

```bash
<device-prefix> python scripts/diagnostics/rollout_device_preflight.py \
  configs/<top-level-rollout-config>.yaml [--gripper_calibrate]
```

It enumerates `/dev/ttyUSB*` first. Each gripper is calibrated when requested.
Without calibration, the current gripper keeps moving between 30%-70% of its
configured maximum opening until the operator answers `l`/`r`; it is then stopped
and disabled before the next device.
Non-calibration identification matches each discovered serial port to the calibration
JSON's recorded port (falling back to the configured port only if that field is absent),
never to the left/right list index. An unknown or ambiguous match blocks motion and
requires fresh binding/calibration. The nonblocking SDK target alternates every
0.5 seconds by default; operator input interrupts that wait. This interval does
not guarantee arrival at each target before reversing.
The selected ports and calibration metadata are written back to
the referenced devices YAML/config-referenced JSON. The input must be a top-level
Hydra rollout config with `camera.left_capture_id/right_capture_id`; passing `configs/devices/*.yaml` or
`configs/devices/infer/*.yaml`
is rejected before hardware access. It then previews each capture-capable
`/dev/video*` node. In the live window, press `l`/`r` to bind the left/right camera,
or `s` to skip an unwanted camera and continue to the next node. These letter keys
also accept uppercase. `q` or Escape aborts the entire binding workflow. Skipping
releases the current capture and closes its preview; both left and right cameras
must still be selected before either YAML is saved. Camera indices are written to
the top-level config; metadata-only video nodes are skipped automatically.
Before interactive USB identification, the preflight passively reads Flexiv
Elements cache records. If multiple distinct serials are found, it lists lettered
options (`a`, `b`, `c`, ...) with serial, model and cached IP. Select exactly as many
distinct options as the existing `robot_sn` entries, separated by commas or spaces,
in configuration order (left then right for dual arms). A scalar `robot_sn` requires
one choice and remains scalar. Duplicate cache records are collapsed by serial.
Invalid counts, duplicate choices and unknown letters prompt again; `abort` or EOF
stops before gripper motion. Entering `s` or `skip` preserves the current `robot_sn` and
`network_interface_whitelist` without using cached candidates, then continues with
gripper and camera binding. Explicit selection overrides model-based guessing.
If exactly one local IPv4 interface contains all selected cached robot IPs, it
also updates `network_interface_whitelist`. Every confirmed/change/skipped
decision is printed to the terminal. Cache discovery does not prove current
RDK reachability; unconfirmed ambiguous candidates never overwrite the config.
This command intentionally opens hardware and may move grippers, so run it only
with the intended rig connected and keep the operator clear of moving jaws.
After a successful pass, use the same top-level config for `doctor`, DeviceStarter,
the policy consumer, or Control Panel. The preflight has already persisted the
confirmed camera IDs, gripper ports/calibration metadata, and any unambiguous robot
binding updates; no manual config edit is required unless the rig or deployment
values change.

### Configuration and software checks

Read configuration in this order: top-level preset, `defaults`, `devices`,
`device_manager`, `action_executor`, `policy`, then `runtime`. Check device and
policy registry names, SHM names/shapes/dtypes, control targets, FPS/chunk lengths,
signal-file paths, robot serials, calibration paths, and absolute workstation paths.

Do not create separate configs when the only differences are wrist camera IDs,
gripper port/identity/calibration paths, checkpoint paths, task text,
`pi_config_name`, `chunk_length`, or `fps`. Keep one descriptive canonical config
and supply those deployment values through the normal override or binding workflow.
`tests/config/test_no_redundant_configs.py` enforces this rule structurally while
preserving configs that differ in control, safety, history/cache, scaling,
recording, or device behavior.

For OpenPi checkpoints, validate the registered `pi_config_name`, model variant,
action horizon, asset ID, normalization-stat dimensions, and task prompt as one
deployment unit. A readable `params/` directory alone is insufficient. Keep one
canonical checkpoint directory containing the real `params/` and
`assets/<asset_id>/norm_stats.json`; do not create a shadow checkpoint, symlinked
`params/`, or temporary compatibility directory. Point both direct inference and
Control Panel at that canonical `ckpt_dir`.

The native OpenPI RTC config declares `policy.params.ckpt_dir`, `pi_config_name`,
and `task` together in the tracked `policy_infer` YAML, like the non-RTC config;
it does not load a hidden `configs/local` overlay. Keep these three values aligned
as one checkpoint contract and validate the composed values using the bound policy
profile without restoring model weights.

When a checkpoint's asset directory does not match an existing OpenPI registry
entry, inspect the checkpoint first. Add a narrowly scoped local registry alias in
the selected OpenPI source whose model settings match the training run and whose
`asset_id` matches the actual directory. Set `policy.openpi_src` so direct inference
and Control Panel preflight resolve the same source. If the deployment intentionally
keeps that registration workstation-local, do not commit the submodule or parent
gitlink; future AI sessions must reapply/verify the alias before launch. Copy norm
stats into the canonical checkpoint only with explicit operator approval, and never
use copying to hide an unknown model/asset mismatch.

Before model restore, run the shared checkpoint contract validation (Control Panel
does this automatically, and `OpenPiUmiPolicy` repeats it in direct inference). It
must resolve `pi_config_name`, require a real non-symlinked and non-empty `params/`,
load the registered asset's `norm_stats.json` with OpenPI, and match `state` and
`actions` dimensions to their policy-specific expectations (Control Panel derives
the action dimension from `policy.controls`; policy adapters enforce any state
dimension constraint). A failure here is a config or checkpoint registration
problem, not a reason to create another checkpoint directory.

For a source checkout, start Control Panel with the intended OpenPI Python. Device
workers, inference, and native recording inherit that active interpreter unless
an explicit recorder override is supplied; `--device-python` and
`--recording-python` are therefore unnecessary for the canonical OpenPI config.
Packaged releases keep using the interpreter wrappers supplied through their
`SUPERINFERENCE_*_PYTHON` environment. A repository checkout is not in packaged
release mode unless it is launched through an extracted release wrapper or
`SUPERINFERENCE_RELEASE_ROOT` is set; the presence of
`incubator/control_panel/release/` does not change source
runtime behavior.

Use hardware-free checks instead of starting the runtime:

```bash
python -m pytest tests/config -q
python -m pytest tests/config tests/devices tests/architecture -q
```

`device_starter.py --help` is not guaranteed to be dependency-free; missing Hydra
or OmegaConf is an environment problem, not evidence that a hardware command is
safe to retry.

Action chunk boundary logs (`pred_chunk_start` and `pred_chunk_end`) must come
from ActionExecutor's post-chunk-manager timeline. Do not reconstruct them in
PolicyConnector by comparing floating-point actions against control SHM: chunk
selection, alignment, and interpolation can legitimately change the dispatched
values and produce false stale-match warnings even while control is healthy.

#### Runtime config metadata and Control Panel filtering

The Control Panel exposes metadata for every config in `GET /api/configs` and
`GET /api/configs/<name>`:

- `workflow_kind`: inferred from `policy.class` (`device`, `openpi`,
  `fastwam`, `diffusion`, `lingbotva`, `nmx_vla`, or `unknown`).
- `policy_class`: the resolved policy class, when present.
- `hardware_required`: inferred from enabled device classes; set explicitly when
  a workflow has non-obvious hardware requirements.
- `status`: `available` or `invalid` by default; a config may override it with a
  lifecycle label such as `experimental` or `deprecated`.
- `control_panel_supported`: whether this panel may display/select the config.

Existing configs receive these fields without migration. Add a top-level
`metadata` mapping when a config needs an explicit value, for example:

```yaml
metadata:
  workflow_kind: openpi
  policy_class: OpenPiUmiPolicy
  hardware_required: true
  status: available
  control_panel_supported: true
```

The current panel keeps device-only configs, OpenPI policies, and the integrated
FastWAM workflow, and hides other policy families until their matching launcher
is integrated. The backend enforces the same capability check during config
selection; hiding an option in the browser is not the safety boundary.

FastWAM model startup uses a private readiness/progress directory and chooses a
free endpoint from `policy.params.port_range` (default `18090-18100`). Treat a
port listener without matching run id, PID, config name, and `model_loaded` as
unready. Stop the Server through the lifecycle path so its process group and
private markers are cleaned up; reserve `force_stop_all` for emergencies.

### Recording, episode control, and RTC

Route deep changes or operational questions to the existing focused Skills:

- Rollout streams, NEDF3, ActionStateXYZ, and TimestampEvent:
  [`infer-rollout-recording`](../infer-rollout-recording/SKILL.md).
- Keyboard episode transitions, reset barriers, and homing state cleanup:
  [`infer-rollout-episode-control`](../infer-rollout-episode-control/SKILL.md).
- Latency alignment, interpolation, and original action indices:
  [`async-action-chunk-alignment`](../async-action-chunk-alignment/SKILL.md).
- LingBotVA RTC and prefix/inpainting behavior:
  [`lingbot-va-rtc`](../lingbot-va-rtc/SKILL.md).

Do not duplicate their field dictionaries or timing contracts in this routing Skill.

### Calibration, gripper, and replay

These operations can move hardware or overwrite artifacts. Before running them,
verify the exact target, ports/serials, calibration files, workspace limits,
emergency-stop access, and whether another process owns the device.

Control Panel and direct launches use the same source config. Keep their full
runtime sequence in this Skill; the root
[`useful_tools_commands.md`](../../../useful_tools_commands.md) intentionally lists only a
few small Flexiv and rollout-device utilities.

- LKTech calibration, width semantics, clipping, and serial ownership:
  [`lktech-gripper-control`](../lktech-gripper-control/SKILL.md).
- Camera/mask/force/safety calibration: use the matching script under
  `scripts/calibration/` and read its `--help` plus module docstring first.
- Robot trajectory replay: use `scripts/replay/robot_record_replay.py` or the
  matching gripper replay script; validate dataset shape, units, limits, and
  control mode before execution.
- Connectivity and fault diagnosis: use
  `scripts/legacy/check_robot_arm.py` without `--enable` first.

### Data and camera inspection

Prefer offline tools when the request concerns existing data:

- Episode structure: `python scripts/analysis/inspect_episode_data.py ...`
- NEDF3 episode replay: use the control-panel recording replay UI.
- OpenPI camera frames/model inputs: `scripts/openpi/dump_raw_camera_frames.py`
  or `scripts/openpi/dump_model_input.py`.

For the concise human command reference, including the standalone Flexiv assembly
viewer and rollout device preflight, use
[`useful_tools_commands.md`](../../../useful_tools_commands.md). Direct inference
startup references live in `openpi_inference_commands.md`,
`fastwam_inference_commands.md`, and `robotwin_inference_commands.md`. Prompt templates for
environment setup and launch-command generation live in
[`useful_prompts.md`](../../../useful_prompts.md).

The hardware-reading `scripts/visualization/visualize_flexiv_assembly.py`
viewer is a standalone process. It is not embedded in Control Panel; the panel's
arm view consumes its own runtime/SHM state path. For current OpenPI rollout
configs, keep `WebCameraOpenPi.hardware_latency_ms: 0.0`: delay compensation is
handled by the OpenPI rollout contract, and adding a second camera offset shifts
observation alignment twice.

Stop the owning device process before opening a camera directly with
`dump_raw_camera_frames.py`; a busy camera can make the probe fail or select the
wrong source. Treat files under `logs/`, rollout directories, and calibration
directories as user data; do not delete or replace them without explicit scope.

### Code changes, tests, and CI

For implementation or review work, use the repository's normal test/lint commands
and route CI/module-boundary questions to
[`superinference-ci`](../superinference-ci/SKILL.md). Keep the
hardware/model/manual jobs separate from the default hardware-free suite.

## Preflight Before Any Hardware Command

1. Identify the exact robot/device, serial number or config, arm/side, and requested
   operation.
2. Classify the command as read-only, energizing, moving, or artifact-changing.
3. Resolve the Python environment and check that the required SDK/model dependency
   is importable.
4. Check for conflicting processes, serial-port owners, stale signal files, and
   existing runtime logs when relevant.
5. Review control mode, calibration, workspace/velocity limits, and emergency-stop
   access for anything beyond a read-only probe.
6. Ask for confirmation at the final motion/destructive boundary when authorization
   is not explicit.

For a read-only Flexiv state request, the minimum preflight is the explicit serial
number, selected backend, and SDK importability. Do not launch `device_starter.py`
just to obtain a one-shot state.

## Reporting Contract

After a command, report:

- the exact command and interpreter/environment used;
- target robot/device and config, if any;
- success/failure and exit code;
- values with units and conventions (`rad`, `m`, `wxyz`, timestamps);
- whether the result is direct SDK state, summary-SHM state, predicted action, or
  dispatched command;
- the first actionable error and a bounded next diagnostic step.

Never present a control-SHM dispatch as measured physical motion. For policy and
recording questions, preserve the distinction between predicted, aligned,
dispatched, and measured data.

## Common Failure Routing

- `ModuleNotFoundError`: report the missing package and identify the required
  profile/environment; do not switch environments silently.
- Flexiv connection timeout or malformed serial: run the read-only
  `check_robot_arm.py` diagnostic with the intended config and inspect NIC whitelist
  and serial spelling.
- SHM not found: verify the producer is running, the configured SHM name/shape/dtype,
  and whether a stale process owns the name. Do not unlink another process's SHM as
  a first response.
- Camera busy: stop the owning runtime or use its SHM output instead of opening the
  physical camera directly.
- Unknown `control_panel/*` config: the dedicated wrapper configs were removed;
  resolve the YAML path relative to `configs/` and pass that shared source name
  to both launchers, for example `policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi`
  or `policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc`.
- OpenPI inference exits with code 1 immediately after launch: reproduce source
  activation with the configured interpreter before blaming the checkpoint. Relative
  `policy.openpi_src` paths are resolved from the repository root, not `consumer/`;
  an erroneous `consumer/third_party/openpi/src` path means the entrypoint computed
  its root one directory too shallow.
- Calibration span, replay shape, or policy action mismatch: stop before motion and
  compare the relevant config/device contract; do not “try once” on hardware.

## Validation Of This Skill

When editing this Skill, run the skill validator from the Codex skill package:

```bash
python "$HOME/.codex/skills/.system/skill-creator/scripts/quick_validate.py" \
  .claude/skills/superinference-operator
```
