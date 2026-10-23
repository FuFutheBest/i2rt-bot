---
name: lktech-gripper-control
description: >
  Use when changing LKTech gripper calibration, gripper min/max angle,
  gripper encoder over-rotation handling, gripper.max_width clipping,
  homing, or runtime command paths in devices/robots/rizon/gripper.py and calibration/lktech_gripper.
---

# LKTech Gripper Control

## Runtime Width Semantics

Structured Rizon configs enable each gripper through its `gripper` primitive in
`state_feedback.format.type` or `control.format.type`. `gripper.enabled` is
removed from native configs. The panel's gripper profiles and preflight slots
must use `gripper_contract.rizon_gripper_enabled_sides` so omitted legacy flags
cannot hide a configured gripper. A settings block alone must not enable an
arm whose formats contain no gripper primitive. Legacy display-only mappings
retain their explicit enable flags at the compatibility boundary.

`RizonRobot` supports LKTech grippers through
`devices/robots/rizon/gripper.py`; `devices/robot.py` remains a lazy
compatibility facade.

The control panel uses one transport: direct LKTech serial access through
short-lived, side-scoped workers owned by `GripperControlManager`. There is no
persistent gripper server or WebSocket hop. Each worker acquires the same
per-port lock, performs a final passive hardware check, executes one command,
then disables the SDK handle before exiting.

- `gripper.value_mode` selects the published state-feedback unit: `width_m`
  publishes physical opening width, while `normalized` publishes 0=closed and
  1=open.
- `gripper.command_value_mode` independently selects the control-stream unit.
  This is required when a teleoperator commands a normalized opening but the
  recorded state must remain in meters.
- `gripper.min_width` and `gripper.max_width` define the conversion and clipping
  range. When `max_width` is omitted, the structured config derives it as
  `2 * arm_length_m * sin(apex_angle_max_deg / 2)`.

## I2RT Leader-Follower Teleoperation

Configurable YAM leader-follower teleoperation is owned by SuperInference under
`devices/robots/i2rt_teleop/`; the public `third_party/i2rt` submodule remains
independent of the LKTech SDK. Launch it with:

```bash
python consumer/i2rt_bimanual_teleop.py \
  --config configs/i2rt_bimanual_teleop.yaml
```

Run `--check` first to validate the YAML and imports without opening CAN or
serial hardware. Each follower always presents `[j1..j6, gripper]` to the
coordinator and recorder:

- `i2rt_can` sends all seven values to the YAM CAN chain.
- `lktech` starts the YAM with `no_gripper`, sends six joints over CAN, and
  sends the normalized seventh value through the LKTech serial adapter.
- `none` starts a six-joint YAM and publishes `0.0` as gripper feedback.

The LKTech adapter loads an existing calibration JSON, acquires a per-port
process lock, clips commands before calling `Move()`, and disables the motor on
shutdown. Teleoperation never auto-calibrates or automatically homes the
gripper. Confirm port-to-side identity after every USB re-enumeration before
changing a follower backend to `lktech`.

The I2RT DM3 web service owns the composite `device_starter.py` process and
derives the operator-visible teleoperation phase from coordinator log events:
ready/safe hold, follower alignment, active control, disabled/safe hold, and
watchdog fault. A watchdog fault clears device readiness and prevents a new
NEDF3 saver or episode from starting until the device stack is restarted; an
episode already in progress may still be finalized or discarded. The
leader-arm buttons and browser controls use the same toggle/discard semantics.

I2RT device startup and dual-gripper calibration are mutually exclusive. Keep
the check-and-launch operation atomic in `I2RTProcessManager`: calibration must
reject a live device process, and device startup must reject a live calibration
process. Web and desktop controls may reflect this state early, but they are not
the safety boundary; direct controller calls must still be rejected by the
process manager.

The I2RT DM3 web log endpoint maintains independent monotonic offsets for the
device, saver, calibration, and upload streams. Keep all four query offsets in
the route allowlist so polling returns only newly appended lines.

The D405/YAM collection variant uses
`configs/i2rt_d405_yam_teleop.yaml` with the built-in `crank_4310` CAN grippers
on both followers and no LKTech serial or calibration worker. Its web controller
checks the operator-confirmed gripper type and both D405 serials before starting
devices. Keep camera serials in the machine-local runtime profile or environment.
The current D405 rig maps left leader/follower to `can1`/`can3` and right
leader/follower to `can0`/`can2`; these kernel-assigned numbers can change after
replugging, so reconfirm cable identity and 1 Mbit/s on all four interfaces
before hardware startup. The D405 controller reads machine-local
`SI_I2RT_CAN_{LEADER,FOLLOWER}_{LEFT,RIGHT}_SERIAL` bindings and rejects swapped
adapters, down links, and bitrates other than 1 Mbit/s.

The coordinator starts all four leader/follower `get_sample` Portal futures
before waiting for any result. Keep this fan-out boundary when changing state
collection: serial waits make the source publication rate equal to the sum of
four RPC latencies, while concurrent requests make it track the slowest worker.
The D405/YAM source runs at 100 Hz so the NEDF3 saver can select robot samples at
90 Hz.

The worker keeps the RPC server, endpoint I/O, and watchdog in one process but
uses separate execution paths. Only the I/O thread calls `get_info`,
`command_joint_pos`, feedback-enable, or safe-state methods; the watchdog thread
only inspects lock-protected timestamps and phase state. It uses the existing
`feedback_timeout_s`, `command_lease_timeout_s`, and
`watchdog_check_interval_s` values—no extra watchdog configuration is needed.
Consequently a stuck endpoint call cannot hold the runtime lock and block
`get_status`; the watchdog latches `FAULT`, and the coordinator/process owner
can shut down the worker safely. Keep `run_once()` deterministic tests when
changing this contract and exercise the blocked-I/O case with:

Worker signal handling must remain effective before the RPC server exists. A
signal handler records the stop request and wakes runtime startup without
blocking; endpoint construction, runtime initialization, and RPC binding each
check that request before advancing. The RPC server runs nonblocking while the
worker main thread supervises it, so a stop request always reaches the common
`finally` cleanup path. Process-level SIGKILL remains only the fallback for a
hardware call that never returns.

```bash
uv run --with pytest --with portal pytest -q \
  tests/devices/test_i2rt_teleop_watchdog.py
```

## Hardware Map

Rizon dual-arm UMI configs such as
`configs/devices/infer/rizon_dual_eef_abs_fisheye_webcam.yaml` use:

| Arm | Serial port | Persisted calibration file |
|-----|-------------|----------------------------|
| Left | `/dev/ttyUSB0` | `calibration/lktech_gripper/LKTech-Gripper-1_2_id1_calibration.json` |
| Right | `/dev/ttyUSB1` | `calibration/lktech_gripper/LKTech-Gripper-2_2_id1_calibration.json` |

USB enumeration can swap after reconnect. Confirm the target port matches the
intended arm; the persisted file records the `port` it was calibrated on.

## Calibration Workflow

The repository calibration helper prints connection, current sampling, calibration
progress and result labels in Chinese for the rollout preflight. Preserve machine
readable YAML/JSON keys, units and third-party SDK diagnostics.

Recalibration uses `read_multi_turn_angle()`. The absolute encoder value shifts
every power cycle and can occasionally jump a full turn (+/-360 degrees), so
only the travel span is meaningful.

1. Free the serial port.
   Stop any running `device_starter.py`, device manager, or `*_policy_infer.py`
   process that owns the grippers. Check sibling repos if needed. Prefer SIGINT
   for clean Flexiv/RDK shutdown; escalate only if the process hangs and no
   longer owns `ttyUSB` fds or children.
2. Confirm no port holders; this should print nothing:

```bash
for pid in $(ls /proc | grep -E '^[0-9]+$'); do for fd in /proc/$pid/fd/*; do
  t=$(readlink "$fd" 2>/dev/null); case "$t" in *ttyUSB*) echo "pid=$pid -> $t";; esac
done; done 2>/dev/null | sort -u
```

3. Calibrate into a temp dir. Single-port runs name the output file
   `...-1_1-...json`, so do not point `--calibration-dir` at the production
   folder.

```bash
CALIB_TMP=$(mktemp -d /tmp/left_gripper_calib_XXXXXX)
python scripts/calibration/calibrate_lk_gripper.py \
  --port /dev/ttyUSB0 --id 1 --gripper-type soft --sequential \
  --calibration-dir "$CALIB_TMP"
```

This physically drives the jaw to both mechanical limits using low-speed IQ
contact detection. Ensure the jaw is clear before running.

4. Validate the temp JSON span:
   `span = abs(max_angle_deg - min_angle_deg)`.
   For the soft preset (`apex_angle_max_deg = 90`, `gear_ratio ~= 11.4`), expect
   about `1020-1035` degrees. Accept only spans in `[950, 1100]` degrees. A span
   near `660` or `1390` degrees indicates a +/-360 degree multi-turn wrap; re-run
   calibration and do not persist it.
5. Persist only validated values. Copy `min_angle_deg`, `max_angle_deg`, and
   `gear_ratio` into the correct production file (`-1_2-` for left, `-2_2-` for
   right), preserving that file's `label` and `port`. Keep preset fields such as
   `gripper_type`, torques, `arm_length_m`, `apex_angle_max_deg`, and `max_width`
   unchanged unless the preset changed.
6. Restart the inference/device stack if it was stopped in step 1.

Preferred dual-arm path:

```bash
python scripts/calibration/auto_calibrate_grippers.py --config configs/devices/infer/<device>.yaml
```

To calibrate both grippers in one shot and get the `-1_2-`/`-2_2-` names
directly:

```bash
python scripts/calibration/calibrate_lk_gripper.py --port /dev/ttyUSB0 /dev/ttyUSB1 --id 1 --gripper-type soft
```

Calibration internals live in
`third_party/lktech_motor_sdk/example/lktech_gripper.py`
(`_calibrate_encoder_range`, `_get_stable_multi_turn_angle`). Runtime rebasing
to the home endpoint happens via `_rebase_multi_turn_calibration_to_endpoint`;
negative absolute angles are fine as long as the span is valid.

## Clipping Pattern

Do not rely on `third_party/lktech_motor_sdk/example/lktech_gripper.py::Move()`
as the first clipping point during inference. It logs a warning every time a
requested width exceeds the physical maximum, which is noisy at policy control rates.

Preferred flow in `devices/robots/rizon/gripper.py`:

1. Configure `gripper.max_width` from the LKTech calibrated physical max, or
   provide `arm_length_m` and `apex_angle_max_deg` so it is derived.
2. Convert `command_value_mode` to physical width and clip against the structured
   per-arm `min_width`/`max_width` immediately before `gripper.Move()`.
3. Keep `last_gripper_value` in `value_mode`, and keep the last commanded width
   separately in meters for the R1 feedback extension.

This keeps policy outputs such as `0.0952 m` from repeatedly triggering SDK warnings
when the calibrated/SDK maximum is about `0.0925 m`.

## Common Issues

- Repeated warning: `requested width ... exceeds max ...; clipping`
  - Cause: policy/control layer sends a width above LKTech physical max.
  - Fix: set device config `gripper.max_width` and clip before `gripper.Move()`.
- Config has `max_width: null`
  - The runtime derives the maximum from `arm_length_m` and
    `apex_angle_max_deg`; verify those geometry values match the installed jaw.
- Home width equals or slightly exceeds the physical max
  - Set `home_width` no larger than `gripper.max_width`.
- `TypeError: float() argument ... not 'NoneType'` at `gear_ratio=float(args.gear_ratio)`
  - Cause: `auto_calibrate_grippers` intentionally leaves `gear_ratio=None` for post-sweep inference,
    but older `calibrate_gripper` cast it before measuring span.
  - Fix: use current `scripts/calibration/calibrate_lk_gripper.py` which accepts `None` and infers after calibration.
- Assembly viewer shows `No module named 'devices'` for both arms
  - Cause: direct file execution puts `scripts/visualization` rather than the repository root on `sys.path`.
  - Fix: use the current viewer, which inserts its resolved repository root before importing the Flexiv adapter.
    Validate this behavior from a working directory outside the repository so test-runner path setup cannot hide a regression.
- Assembly appears to fly or tumble when using the mouse wheel
  - Cause: Flexiv URDF is Z-up, while an unconfigured Three.js camera and grid are Y-up.
  - Fix: keep the camera and orbit controls Z-up, rotate the grid onto the XY plane, disable zoom-to-cursor, and bound
    camera distance. Browser validation must confirm arm root positions are invariant across repeated wheel events.
- Assembly viewer zoom is too abrupt or panning feels unavailable
  - Fix: disable OrbitControls wheel zoom and use the viewer's normalized, clamped wheel handler. It groups a wheel-event
    burst into one gesture and changes camera distance by about 35% per gesture. Keep screen-space panning and keyboard
    pan enabled. Validate right-drag changes only the orbit target while arm roots remain invariant.
- Rizon 4R visual is missing its base shell
  - Cause: unlike the Rizon 4 URDF, Rizon 4R has no `world_to_base_link` joint, so a child-link-only renderer never
    visits its root `base_link` visual.
  - Fix: identify and render the URDF root link independently before traversing joints; do not special-case the model name.
- Rollout device preflight crashes on `float(None)`
  - Cause: `max_width: null` is valid in `width_m` mode, but the preflight previously cast the explicit null before
    consulting calibration data or the gripper preset.
  - Fix: treat scalar null and null list entries as the `0.0925 m` provisional default. Calibration JSON still takes
    precedence when present, and the preflight must preserve the null value in the YAML config.
- Rollout binding offers unresolved `${camera.left_capture_id}` as a camera choice
  - Cause: the operator passed a leaf `configs/devices/*.yaml` or `configs/devices/infer/*.yaml`, while camera IDs are owned by its top-level Hydra config.
  - Fix: require the top-level rollout config before moving hardware, resolve its devices default for gripper writes, and
    identify each capture-capable video node from a live preview using `l`/`r`. Write camera IDs only to the top-level file.
- A gripper stops moving before the operator can identify its side
  - Fix: in non-calibration rollout binding, repeat the bounded 30%-70% opening motion until the operator enters `l` or
    `r`. Stop the worker and disable the motor immediately after input, on errors, and on interruption.
- Non-calibration rollout identification barely moves or uses the wrong range
  - Match the saved calibration's `port` to the discovered port; USB enumeration order is not left/right order.
    A camera-stage failure may leave updated calibration JSON alongside stale YAML ports. Reject unknown or
    ambiguous matches before starting identification motion.
  - `Move(width, velocity)` returns before arrival, and velocity is calibrated motor spans per second. Identification
    intentionally reverses every 0.5 seconds by default for a faster cycle, so it may not reach each target. Keep this wait
    interruptible so answering `l`/`r` stops motion immediately. Stored port matching cannot establish identity
    after USB re-enumeration; reconfirm the physical binding and calibration after reconnect or power cycling.

## Validation

At minimum:

```bash
python3 -m compileall -q devices/robots/rizon
```

When project tooling is available:

```bash
uv run pytest --strict-markers -m "not manual"
```

## Assembly Visualization

The repository contains no LKTech gripper URDF. Use the independent viewer for
Rizon/LK assembly inspection:

```bash
python scripts/visualization/visualize_flexiv_assembly.py \
  "Rizon 4-063256" "Rizon 4R-062088" --gripper-type umi3_camera
```

The viewer is read-only: it calls `states()` and reads the tool's
`params().tcp_location`; it never enables or moves a robot. `--offline` uses a
nominal pose. The RDK pose is the only available flange-to-tool relationship;
when it cannot be read, the viewer shows an identity transform warning. The
viewer uses the verified UMI3 soft-finger CAD assets under
`incubator/control_panel/static/assets/tools/umi3_camera/soft/`. Those assets
include the LK-driven gripper assembly, Flexiv flange adapter, wrist camera,
and mount; their source and output hashes are recorded in the adjacent
`generation_manifest.json`. Do not fall back to primitive substitute geometry
if these assets fail to load.
