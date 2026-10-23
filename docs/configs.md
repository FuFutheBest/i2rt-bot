# Configuration

SuperInference uses Hydra composition rooted at `configs/`. `device_starter.py` defaults to
`configs/config.yaml`; model-specific runs select a native name such as
`policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi` with `--config-name`.

## Config groups

- `configs/devices/`: common producer processes, collection/replay presets, hardware identifiers, robot control settings,
  and calibration assets.
- `configs/devices/infer/`: policy-inference-specific device presets.
- `configs/device_manager/`: summary shared-memory aggregation and synchronization.
- `configs/action_executor/`: policy chunk consumption and robot control targets.
- `configs/consumer/`: shared consumer defaults.
- `configs/runtime/`: signal-file paths, process switches, camera overlay defaults, and CPU affinity.
- `configs/anomaly_config/` and `configs/trajectory/`: feature-specific settings.
- `configs/policy_infer/diffusion/` and `configs/policy_infer/nmx_vla/`: legacy
  DiffusionPolicy and NMX-VLA launch presets.
- `configs/policy_infer/fastwam/`: FastWAM Server and policy-client presets; they use a
  FastWAM-specific `WebCameraOpenPi` device group, separate from OpenPI.
- `configs/policy_infer/lingbotva/`: LingBotVA launch presets.
- `configs/policy_infer/robotwin/`: RoboTwin direct-drive simulation presets for `joint14`, ACT, and optional `eef16` action
  schemas.
- `configs/policy_infer/openpi/`: the canonical OpenPI real-device presets. It contains only the normal and RTC variants;
  both compose the same device groups and differ in `policy.params.enable_rtc`.

`configs/` is the only runtime configuration source. The optional application under
`incubator/control_panel/` consumes the same Hydra config name and normalized
mapping as `device_starter.py`, for example
`policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi`; it does not maintain a config
mirror or a Control Panel-native preset.

`WebCameraOpenPi.usb_port_requirement` accepts `any`, `usb_2.0`, and
`usb_3.0`. It validates the negotiated V4L2 USB link speed (`480 Mbps` for
USB 2.0 and at least `5000 Mbps` for USB 3.0); it does not identify a physical
USB receptacle. Use an explicit stable `capture_id` path when a fixed camera
identity is also required. The current FastWAM and OpenPI device presets use
`any` while their camera-port bindings are under investigation.

## Device timestamp synchronization

`device_manager.type: synchronized` accepts `sync_mode`:

- `binary_match` is the default when omitted. It uses `master_device_class_id` as a
  logical device ID (the position in the complete configured list, before
  filtering disabled devices) and selects the nearest timestamped frame from
  every other device ring buffer.
- `camera_align` requires `master_device_class_id` to select an image device.
  The master image is copied directly, other image streams retain nearest-frame
  matching, and every non-image stream must declare `state_feedback.format`.
  The manager prefers bracketed interpolation. Robot states may instead be
  projected forward from their latest two samples by at most 1.5 configured
  state-feedback periods. If alignment is unavailable within this bound, no
  summary frame is published. The former name `camera_interpolate` is no longer
  accepted; use `camera_align`.

## Low-dimensional device contract

### Device constructor contract

Every registered device constructor has the same required stream argument:

```yaml
state_feedback:
  fps: 100.0                 # publishing rate in Hz
  buffer_size: 100           # number of frames retained in the SHM ring
  hardware_latency_ms: 0.0   # measured source latency, in milliseconds
  format:
    shape: [7]               # NumPy shape of one published frame
    dtype: float64           # SHM element dtype
    type: [j1-7]             # low-dimensional primitives, or rgb for cameras
```

`state_feedback` is mandatory for cameras, low-dimensional readers, and robots;
there are no constructor defaults for these stream fields. Camera formats must
use `type: rgb` and a `[height, width, 3]` shape. Low-dimensional formats use an
ordered primitive list whose widths must sum to `prod(shape)`. A writable device
(capability `low_dim_rw` or `low_dim_w`) additionally requires the robot command
contract below. DeviceStarter parses these contracts once and passes the typed
objects directly to the device class; device classes must declare
`state_feedback` (and `control` when writable) in their constructor signature.

Common device-specific constructor arguments are kept beside this required
contract:

- cameras: `capture_id`, backend/format controls, rotation/exposure, and
  `usb_port_requirement`;
- robots: `robot_sn`, `home`, `gripper`, `motion_limits`, and signal-file paths;
- trackers and serial teleoperators: `tracker_roles`, `port`, encoder selectors,
  and calibration files;
- simulators: URDF/task paths and GUI/rendering switches.

These options describe hardware or behavior; they do not replace or duplicate
the required stream contract.

Low-dimensional source configs group the published state stream by meaning.
`dtype` belongs to the format because it is part of the SHM data contract; `fps`,
buffering, and latency belong to `state_feedback` because they describe how that
stream is published:

```yaml
state_feedback:
  fps: 120.0
  buffer_size: 1000
  hardware_latency_ms: 10.0
  format:
    shape: [22]
    dtype: float64
    # q, dq, measured TCP translation/quaternion, gripper width
    type: [j1-7, dq1-7, xyz, qwqxqyqz, gripper]
```

The primitive widths must sum to `prod(shape)`. `xyz`, `scalar`, `gripper`,
`motor_angle`, `wrench`, `twist`, `jN-M`, `dqN-M`, and `linear:N` interpolate
linearly. `qwqxqyqz` is a WXYZ quaternion and `qxqyqzqw` is XYZW; both use
SLERP. Unknown fields and primitive-width mismatches fail during config
validation.

Controllable devices use a separate command contract:

```yaml
control:
  type: eef_abs  # joints | eef_abs | eef_rel
  # Omit source_selection for the device-owned command stream. For an external
  # teleoperator, describe the endpoint instead of spelling its SHM name.
  source_selection:
    mode: single_source
    source:
      device: Teleoperator
      device_class_id: 0
      port: state
  execution:
    mode: synchronous  # synchronous | legacy_async
    fps: 100.0
  buffer_size: 100
  format:
    shape: [16]
    dtype: float64
    # Per arm: TCP xyz, command quaternion (XYZW), gripper command.
    type: [xyz, qxqyqzqw, gripper, xyz, qxqyqzqw, gripper]
  cartesian_impedance:
    translation_n_m: [10000.0, 10000.0, 10000.0]
    rotation_nm_rad: [1000.0, 1000.0, 1000.0]
```

`enabled` at the device level means whether DeviceStarter launches that device.
The command format and its primitives define the complete actuator layout. For
Rizon, a `gripper` primitive
in `state_feedback.format.type` requests gripper feedback and one in
`control.format.type` requests gripper commands. There is no separate
`control.grippers_enabled`, `control.head_enabled`, `gripper.enabled`, or
`using_tcp_pose` switch. For X1, the optional `gripper` and `j1-2` command
primitives respectively enable gripper and head commands. The unused Rizon
`mirror_mode` and `manual_confirm` keys are not part of the contract.
`gripper.value_mode` defines the feedback unit; `gripper.command_value_mode`
defines the command unit independently. For example, R1 collection publishes
physical width with `value_mode: width_m` while accepting Quest trigger values
with `command_value_mode: normalized`.

`control.source_selection` describes where command frames come from. Its
endpoint form is `{device, device_class_id, port}`, where `device: self` refers to
the robot's own command stream and a source `port` is `state` or `command`.
The default is the device-owned `command` stream, so it may be omitted. A
shared-control robot can declare two named sources and a `control_flag`
selector endpoint:

```yaml
source_selection:
  mode: dual_source
  sources:
    teleoperation: {device: KeyboardTeleoperator, device_class_id: 0, port: state}
    policy: {device: self, port: command}
  dual_source_selector:
    endpoint: {device: KeyboardTeleoperator, device_class_id: 0, port: control_flag}
```

The runtime derives standard SHM names from these endpoints; raw `*_data`,
`*_control`, and `*_control_flag` names are not part of a device config.
The selector protocol is fixed and is not configured: `0` selects `policy` and
`1` selects `teleoperation`.

Rizon validates `control.type` against `control.format.type` before connecting
to hardware. The shared contract accepts three values: `joints` is an absolute
joint target, `eef_abs` is an absolute TCP target, and `eef_rel` is relative to
the current TCP pose. Each robot adapter accepts the applicable subset; there
are no aliases or secondary internal control-mode names. `joints` requires one
`j1-7` primitive per arm.
Every TCP mode requires one `xyz` and one quaternion primitive per arm, with an
optional `gripper`; quaternion order is taken from the primitive name rather
than assumed from fixed array offsets. State feedback is serialized in the exact
primitive order declared by `state_feedback.format.type`. The R1 74/88-value
extensions are also validated as ordered format primitives instead of selecting
an implicit layout from the shape alone.

Motion limits are grouped first by derivative, then by the object being limited:

```yaml
motion_limits:
  velocity:
    joints_rad_s:
      left: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
      right: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    tcp_linear_m_s: 0.5
    tcp_angular_rad_s: 0.8
  acceleration:
    joints_rad_s2:
      left: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
      right: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    tcp_linear_m_s2: 5.0
    tcp_angular_rad_s2: 2.0
  # null disables the limit; otherwise [Fx, Fy, Fz, Mx, My, Mz] in N / N*m.
  max_tcp_contact_wrench: null
```

The old `max_velocity` vector limited each joint in joint space; the old
`max_angular_vel` scalar limited TCP orientation speed in Cartesian space. They
were not two names for the same setting. The new names make that distinction
and every unit explicit. Runtime config validation does not create flat robot
fields such as `data_shape`, `control_shape`, `control_type`, or `fps`. Every
registered robot entrypoint consumes `state_feedback` and, when it has a standard
command stream, `control` directly. Their public constructors do not accept
duplicate shape, dtype, rate, or buffer arguments.

## Device capabilities and manager validation

Every registered device class has one framework-facing capability:

- `camera`: publishes image frames.
- `low_dim_r`: publishes a typed low-dimensional state stream.
- `low_dim_rw`: publishes typed state and consumes a standard control stream.
- `low_dim_w`: consumes a standard control stream and has no state stream.

The capability describes SuperInference SHM channels, not whether the physical
hardware can move. For example, `YamHeadRobot` consumes a custom Quest channel
and publishes state, so its manager-facing capability is `low_dim_r`.

Supporting components that read a robot command stream use the same logical
endpoint form rather than a raw SHM name. For example, the planned Yam-head
tracking strategies declare
`control_endpoint: {device: RizonRobot, device_class_id: 2, port: command}`.

`DeviceManager` aggregates only readable capabilities. On connection it checks
the registry class and device ID against the SHM header. For cameras it checks
the configured image shape and dtype. For low-dimensional devices it requires
`state_feedback.format`, validates that primitive widths sum to its shape, and
then compares the format shape and dtype with the actual SHM header. A
`low_dim_rw` entry must also declare `control`. Contract disagreement fails
startup instead of allowing a mis-shaped stream into the summary SHM.

Device-manager groups are behavior-level presets: `base`, `sync`,
`camera_sync`, and `robotwin`. Each launch config owns its logical master ID
(`master_device_class_id`) and summary history (`params.buffer_size`). The
canonical OpenPi and FastWAM presets use `camera_sync` with the first wrist
camera at index `0`; ordinary Rizon/UMI workflows use `sync` and retain the
default binary-match mode.

Use the YAML path relative to `configs/` without its suffix as the config name. A file at
`configs/policy_infer/openpi/foo.yaml` is therefore selected with
`--config-name policy_infer/openpi/foo` by both launchers.
For example, the same source can be selected in either entrypoint:

```bash
python device_starter.py --config-name policy_infer/openpi/foo
python -m incubator.control_panel --config-name policy_infer/openpi/foo
```

A typical top-level file composes groups like this:

```yaml
# @package _global_

defaults:
  - /devices/infer@devices: rizon_single_eef_abs_iphone_umi
  - /device_manager: sync
  - /action_executor: base
  - /consumer: default
  - /runtime: default
  - _self_

# Config-group contents retain a historical device_manager envelope. This is
# flattened before DeviceStarter and Control Panel consume the configuration.
device_manager:
  device_manager:
    master_device_class_id: 1
    params:
      buffer_size: 100000
```

Top-level configs placed in a subsystem directory use `# @package _global_` and absolute config-group defaults so
their runtime keys remain at the root. A config that extends another subsystem preset should likewise reference it
with an absolute entry such as `/policy_infer/robotwin/config_infer_robotwin_replay_joint14`.

## Cross-file contracts

Check these values together:

- Every enabled `devices[*].class` must be a key in `devices.DEVICE_CLASSES`.
- `policy.class` must be a key in `policy.POLICY_CLASSES` and must resolve successfully in the installed
  environment.
- `policy.obs.devices` names must match configured device SHM keys.
- Each `policy.controls[*].endpoint` must name the target robot with `device`, `device_class_id`, and
  `port: command`; its `shape` and `dtype` must match that robot control stream. The runtime derives
  the canonical `*_control` SHM name and does not accept a raw `shm` field.
- `action_executor.robot_control_shms`, `execution_fps`, and `chunk_length` commonly interpolate the policy block.
- With `action_executor.execution_clock: sim_time`, the executor derives `<device>_<device_class_id>_sim_clock`
  from its first command endpoint; `sim_clock_shm` is not a supported configuration field.
- Device manager `type` and action executor chunk-manager `type` must be registered in `manager.MANAGER_CLASSES`
  and `manager.utils.CHUNK_MANAGERS`.
- All processes participating in reset or episode coordination must receive the same `runtime.signal_files`.

## Inspect and validate

Read a config in this order: top-level file, its `defaults`, device group, device-manager group, action-executor
group, policy block, then runtime settings.

```bash
python -m pytest tests/config -q
python tests/config/policy_launch_plan.py --all-supported --json
```

Both commands are hardware-free. Before a hardware run, resolve the bound runtime profiles and independently
validate serial numbers, IP addresses, calibration files, control mode, workspace limits, CPU assignment, and
emergency-stop behavior.

Avoid committing workstation-specific absolute paths. Prefer environment interpolation, a local override, or a new
portable preset. For deep feature workflows, follow the relevant `.claude/skills/*/SKILL.md`; this page owns only
the stable composition contract.
