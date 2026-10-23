---
name: infer-rollout-recording
description: Use when configuring, inspecting, visualizing, or changing NEDF3 inference-rollout recording and the ActionStateXYZ/TimestampEvent latency traces, especially when distinguishing predicted, original, aligned, and executed actions.
---

# Infer Rollout Recording

Use this skill for `policy.infer_rollout`, the compatibility entrypoint
`consumer/io/async_infer_data_saver_NEDF3.py`, the implementation
`incubator/r1/async_infer_data_saver_NEDF3.py`,
`utils/action_state_xyz_logger.py`, `utils/timestamp_event_logger.py`, and
ActionExecutor action-trace work. Keep recording and trace semantics separate:

- NEDF3 is the episode dataset: camera frames, observations, predicted chunks,
  policy outputs, and commands read from control SHMs.
- ActionStateXYZ JSONL is a lightweight position/timing diagnostic trace.
- TimestampEvent JSONL is event and SHM lifecycle instrumentation.

Neither JSONL stream is the source of truth for complete robot pose or episode data.

`policy.observation_recorder` is a separate NPZ/JSONL stream for SuperOffline,
owned by [`superoffline-comparison`](../superoffline-comparison/SKILL.md). Native
OpenPi and FastWAM variants enable it with explicit per-variant output directories
under `logs/offline_observations/`. It starts with the connected policy consumer,
independently of NEDF3 episode recording and `infer_rollout.enabled`; starting a
FastWAM server alone does not record observations. Its normal shutdown finalizes
the manifest and logs frame/inference counts and the absolute output directory.

## Canonical Contract and Skill Boundaries

The I2RT D405/YAM Web collector runs its device and browser workers in the
I2RT profile, while its NEDF3 saver resolves a separate verified `recording`
profile for `i2rt_d405_yam_collect_async`. The two environments cannot share
the reviewed NumPy pins: I2RT requires 2.2.6 and the approved NEDF3 recording
profile requires 1.26.4. Saver startup fails if that recording binding is
missing or stale; do not fall back to the device interpreter. The older I2RT
collector retains its current-interpreter fallback.

RealSense frames written to shared memory must use Unix epoch nanoseconds from
the host clock. `pyrealsense2` frame timestamps are device-relative
milliseconds; storing them as epoch time makes live D405 streams fail device
readiness as stale even while frames are arriving.
The RealSense capture loop is paced by blocking `wait_for_frames()` calls. Do
not add the configured frame interval as a second wait after each captured
frame; the duplicate delay can reduce one camera to roughly 20-25 FPS when two
D405 processes run concurrently.
RealSense stream resolution must match the configured SHM shape. The adapter
drops a mismatched frame instead of silently resizing it; configure a native
camera mode and preserve its geometry through recording.
The I2RT NEDF3 saver checks real camera SHMs from its 90 Hz main loop and writes
only a new timestamp. Polling a 30 FPS producer at exactly 30 Hz causes phase
drift and persistent missed frames. Synthetic camera placeholders remain
scheduled at the configured 30 Hz camera rate.
Camera SHM is RGB. OpenCV JPEG encoding requires an explicit RGB-to-BGR
conversion. The corrected contract is `i2rt-nedf3-v2`; the offline Rerun reader
must repair the historical red/blue swap for `i2rt-nedf3-v1` Episodes and must
not apply that repair to v2 payloads.
The terminal loop advances an absolute monotonic sampling deadline and waits
only for the time left after each collection pass. Camera JPEG encoding and
RGBD message construction run in the existing writer thread so the 90 Hz robot
sampler does not pay camera-processing latency. I2RT camera geometry is saved
exactly as captured; do not add recorder-side crop or resize. Preserve the
writer-thread boundary when changing the saver queue or message construction.
The loop polls at twice the highest configured output rate and tracks robot
deadlines per leader/follower stream. A deadline advances only after a new SHM
frame is accepted, allowing a later poll in the same 90 Hz slot to recover when
the first poll races the roughly 97 Hz source. Camera SHM reads retain their own
capped polling deadline and do not run at the faster robot retry rate.
Saved I2RT episodes are visualized offline with
`consumer.io.i2rt_nedf3_rerun` through the verified `recording` profile. The
reader streams MCAP shards and decodes their embedded protobuf descriptors; it
must not import the hardware stack or load all decoded camera frames into RAM.
The NumPy 1.26.4 recording profile uses `rerun-sdk==0.22.1`, because Rerun 0.23+
requires NumPy 2.x. Keep visualization outside the saver process so viewer work
cannot affect collection frequency.
The I2RT DM3 Web UI may list finalized episodes below the managed `data/` root
and launch the operator-selected folder through this recording profile only
while device, saver, and calibration processes are idle. Resolve only validated
relative paths below that root; reject traversal, incomplete, in-progress, and
quarantined directories.
The I2RT DM3 collection page previews both wrist SHMs through local JPEG
snapshot endpoints and presents the synthetic black head stream as an explicit
placeholder. Preview readers attach to existing SHM and must never open a
second RealSense pipeline or alter recorder sampling.
Use one persistent latest-frame MJPEG connection per preview. The D405 wrist
streams are native 640x480 RGB at 30 FPS; preserve that frame for monitoring
and recording, encode at JPEG quality 90, and convert the RGB publisher output
to BGR before calling OpenCV's JPEG encoder. The legacy webcam collector may
still supply 640x360 wrists, which are also recorded unchanged. Per-frame HTTP
snapshot polling is too slow and soft for live operator monitoring.
The D405 device entries also declare `device_name: RealSenseCamera`, matching
the adapter's actual shared-memory prefix; otherwise DeviceStarter waits for
the nonexistent `RealSense_<id>_data` streams until its readiness timeout.
The selected I2RT system profile owns optional-head behavior end to end. A
two-camera D405 launch sets `SI_I2RT_HEAD_CAMERA_ENABLED=false`, validates only
the bound wrist serials, and materializes an ignored local saver snapshot with
`devices.cameras.head: null` plus a 640x480 synthetic head. A three-camera
profile enables and validates the real head D405 and keeps its real SHM mapping.

For mirrored I2RT hardware, use the follower's seven-value
`joint_directions` mapping in the teleoperation config. Apply the same mapping
to alignment, active follower commands, and follower-to-leader feedback. The
seventh value is the gripper axis, so a right wrist J6 correction is
`[1, 1, 1, 1, 1, -1, 1]`.

Read the control-panel
[`NEDF3_ROLLOUT_CONTRACT.md`](../../../incubator/control_panel/docs/NEDF3_ROLLOUT_CONTRACT.md)
before changing a topic, event, field, timestamp, stage label, recorder path, or
visualizer. Keep this skill focused on the operational workflow.

Recording changes often touch adjacent behavior. Use the dedicated skills
instead of silently changing those contracts:

- Read
  [`async-action-chunk-alignment`](../async-action-chunk-alignment/SKILL.md)
  before changing `all_actions`, `aligned_actions`, chunk timestamps,
  `original_action_offset`, drop/resampling/smoothing, or dispatch alignment.
- Read
  [`infer-rollout-episode-control`](../infer-rollout-episode-control/SKILL.md)
  before changing start/stop/result signals, auto-restart, reset barriers,
  homing, or state clearing across episodes.

The operational stage order is `predicted` -> `policy_original` ->
`action_executor_aligned` -> dispatched (`action_stage=executed`) -> measured.
Here `executed` is a compatibility label for a control-SHM dispatch, not
physical motion. In TimestampEvent JSONL, `event_ns` is always log-call wall
time; in ActionStateXYZ JSONL, `event_ns` aliases the event-specific
`timestamp_ns` and may instead be sample or extrapolated trajectory time.

## Operational Workflow

1. Identify which of the three streams owns the requested information. Do not
   add episode payloads to JSONL or diagnostic-only fields to NEDF3 without a
   clear consumer.
2. Trace the producer to the source timestamp and array before editing the
   logger. Record source shape, dtype, units, clock meaning, stage, and aliases
   in this skill and the NEDF3 rollout contract.
3. For action traces, preserve the separate original, aligned, dispatched, and
   measured sources of truth. Never infer dispatch by matching floating-point
   action values.
4. Check the NEDF3 replay UI and ActionStateXYZ HTML visualizer dependencies.
   TimestampEvent JSONL has no direct HTML visualizer.
5. Keep the optional inference recorder implementation under
   `incubator/r1/async_infer_data_saver_NEDF3.py`. The stable
   `consumer/io/async_infer_data_saver_NEDF3.py` entrypoint is a lazy facade for
   existing launchers and recording profiles. The implementation inherits the
   canonical R1/Flexiv NEDF3 writer and writes `nmx/hal/...` topics. Do not
   reintroduce a parallel legacy writer or format fallback.
6. Extend focused contract tests for aliases, vector extraction, path
   resolution, and producer payloads; hardware is not required for these.
7. Run focused validation and update both this workflow and the field
   dictionary when the operational contract changes.

## PolicyConnector Composition Boundaries

`consumer/inference/common/policy_connector.py` is the compatibility-facing
orchestrator. Runtime behavior is composed from focused components:

- `policy_loop.py`: startup/shutdown and per-iteration inference orchestration.
- `policy_runtime_io.py`: policy/summary SHM metadata, reconnects, action-index
  reads, action configuration, and merged rollout chunks.
- `observation_builder.py`: current observations, history collection, and policy
  input preparation.
- `policy_visualization.py`: inference markers and action/state XYZ diagnostics.

Keep compatibility methods on `PolicyConnector` as thin delegations. Put new
logic in the component that owns the responsibility; do not reintroduce mixin
inheritance or grow the connector into the implementation boundary again.

## NEDF3 Topic Reference

The writer uses each sample's SHM frame timestamp as both NEDF `log_time` and
`publish_time`. Timestamps are Unix-wall-clock nanoseconds.

| Topic or stream | Source | Payload and units |
| --- | --- | --- |
| Camera stream keyed by `<DeviceType>_<id>` | Summary SHM camera frame | JPEG image; recorded resolution is original unless `infer_rollout.image_size: [width, height]` is set |
| `nmx/hal/arm/{left,right}/tcp/cmd` and `nmx/hal/hand/{left,right}/cmd` | 16D `RizonRobot_*_control` SHM | Executed `[xyz, quat_xyzw, gripper_width_m]` command for each arm |
| `nmx/hal/arm/left/extra` (`policy_action_chunk`) | Enabled `action_chunk` policy-output SHM | Full predicted `[chunk_length, 16]` array plus source and shape in an NEDF3 `Metadata` message |
| `/lowdim/robot_left` | `RizonRobot` 44D state slice `[14:21]` | 7D `[x, y, z, qw, qx, qy, qz]`; position meters, unit quaternion |
| `/lowdim/gripper_left` | `RizonRobot` state index `21` | 1D gripper width, meters |
| `/lowdim/robot_right` | `RizonRobot` 44D state slice `[36:43]` | 7D `[x, y, z, qw, qx, qy, qz]`; position meters, unit quaternion |
| `/lowdim/gripper_right` | `RizonRobot` state index `43` | 1D gripper width, meters |
| `/lowdim/robot_single` and `/lowdim/gripper_single` | 8D single-arm robot state | Pose `[x, y, z, qw, qx, qy, qz]` and width |
| `/lowdim/tcp_single` | 7D robot state | `[x, y, z, qw, qx, qy, qz]` |
| `/lowdim/<DeviceType>_<id>` | Other summary-SHM low-dimensional device | Flattened native vector; consult its device header for shape, dtype, and units |
| `/lowdim/action/<shm_name>` | A configured `infer_rollout.control_shms` entry | Flattened command exactly as read from that control SHM |
| `/lowdim/action/<policy-output-shm>` | Enabled `policy_outputs` SHM | Value, Q values, Q timestamps, or merged predicted action chunk |

All device SHM names end in the global logical `device_class_id`, independent
of physical `capture_id` and canonical NEDF camera IDs. The OpenPI inference
device config order is left wrist, right wrist,
then optional head, and the recorder maps those entries to `camera_1`,
`camera_0`, and `camera_2` respectively. Keep anomaly detection on the same
resolved mapping; parsing `/dev/videoN` suffixes directly can either swap roles
or request the unsupported `camera_3` topic. Collection presets list recording
cameras as right, left, head. `utils.recording_device_roles.nedf_camera_entries`
maps the camera-only order, ignoring preceding ZED/robot/teleoperator entries.
The Windows recorder likewise maps the first/second rotary encoder to left/right,
not by the numeric SHM suffix. Keep global IDs contiguous in the full YAML list.

The corresponding LeRobot feature order is right wrist from `camera_0`, left
wrist from `camera_1`, and head from `camera_2`. An explicit
`head_camera: null` is authoritative: emit the black head-camera placeholder
even when episode metadata mentions another camera. Do not silently promote a
wrist or metadata-only camera into the head feature.
For a two-camera episode with that synthetic head placeholder, set NEDF3
`main_camera` to the real right wrist (`camera_0`); use `camera_2` only when a
real head stream is present.

The R1 recorder writes measured arm joint positions from the Rizon dual-arm
state frame slices `[0:7]` and `[22:29]` to the left/right
`nmx/hal/arm/{side}/joint/state` topics, and the corresponding velocity slices
to `joint/vel`. These are `ArmState` protobuf messages with `joint0` through
`joint6` positions in radians; the 44D base frame is sufficient for both
streams.

`PolicyRuntimeIO._merge_action_chunks`, exposed through
`PolicyConnector._merge_action_chunks`, concatenates predicted device chunks in
**sorted device-name order**, producing shape `[chunk_length, sum(action_dim)]`.
This is a recording representation, not proof that every row was dispatched.
The ActionExecutor policy SHM still keeps per-device chunks.

Do not globally reinterpret quaternion order. The Rizon state split is
explicitly `wxyz`, while policy action quaternion order can be policy/checkpoint
specific (for example LingBotVA metadata supports `xyzw`, `wxyz`, or Euler modes).
The JSONL xyz logger intentionally extracts only position and gripper width; it does
not certify or transform quaternion channels.

## ActionStateXYZ JSONL

Default location:
`logs/action_state_xyz/<run_id>/<source>_xyz.jsonl`.

Environment controls:

- `SI_ACTION_STATE_XYZ_LOG_ENABLED=0` disables the logger.
- `SI_ACTION_STATE_XYZ_LOG_FILE=/path/trace.jsonl` selects one exact file.
- `SI_ACTION_STATE_XYZ_LOG_DIR=/path` selects the base directory.
- `SI_ACTION_STATE_XYZ_RUN_ID=<id>` selects the run directory.
- `SI_TIMESTAMP_RUN_ID=<id>` is the fallback shared run ID.
- `SI_ACTION_STATE_XYZ_QUEUE_SIZE=<count>` sets the non-blocking queue capacity.

Every record contains:

- `event_type`: event contract listed below.
- `timestamp_ns`: event/sample time as Unix-wall-clock nanoseconds.
- `timestamp_time_utc`: ISO-8601 UTC rendering.
- `event_ns`, `event_time_utc`: backward-compatible aliases of the preceding fields.
- `source`: process component, normally `policy_connector` or `action_executor`.
- `pid`: writer process ID.

XYZ action/state records additionally contain:

- `device_name`, `vector_kind`, `vector_dim`, and `arm`.
- `x`, `y`, `z`, and duplicate `xyz`; position units are inherited from the source
  contract and are meters for Rizon Cartesian actions/states.
- `w`, `width`, and `gripper_width` when an 8D arm segment is available; all three
  are aliases, normally meters.

Vector extraction:

- 16D action: left xyz `[0:3]`, width `[7]`; right xyz `[8:11]`, width `[15]`.
- 8D action: xyz `[0:3]`, width `[7]`.
- 44D state: left xyz `[14:17]`, width `[21]`; right xyz `[36:39]`, width `[43]`.
- 22D state: xyz `[14:17]`, width `[21]`.

Important event types:

- `observation_xyz`: robot state read for a policy observation.
- `relative_xyz`, `history_relative_xyz`, `query_xyz`, `reference_xyz`: policy-side
  diagnostics; they are not absolute commanded poses unless that producer says so.
- `inference_start`, `inference_active`, `inference_end`: inference timing markers.
- `chunk_switch`: ActionExecutor accepted/aligned a chunk and reached first dispatch.
- `pred_chunk_start`, `pred_chunk_end`: boundaries of the original policy chunk on
  the offset-aware timeline.
- `all_actions`: the **original policy output** before chunk-manager drop,
  interpolation, or smoothing. `chunk_stage=policy_original`.
- `aligned_actions`: the exact snapshot after chunk-manager processing.
  `chunk_stage=action_executor_aligned`; `original_action_offset` can be fractional.
- `action_xyz` from `source=action_executor`: command written to a robot control
  SHM. New records carry `action_stage=executed`,
  `dispatch_timestamp_ns`, and `executed_action_timestamp_ns`.

The last item is commonly called “executed” in analysis, but technically means
**dispatched command**, not measured physical motion. Compare it with
`observation_xyz` to assess tracking.

### Original and aligned timelines

`policy_write_timestamp_ns` is when PolicyConnector published a chunk.
`observation_timestamp_ns` identifies the observation used by inference.
`switch_timestamp_ns` is represented by the `chunk_switch.timestamp_ns`.

For original rows:

`original_start_timestamp_ns =
switch_timestamp_ns - round(original_start_offset * original_step_interval_ns)`.

Each `all_actions` row advances by `original_step_interval_ns`. Each
`aligned_actions` row starts at dispatch and advances by
`aligned_step_interval_ns`. `original_action_offset` maps aligned samples back to
the original policy trajectory and may be fractional after resampling.

Do not reconstruct or reintroduce `pred_abs_xyz`. The original and aligned arrays
owned by ActionExecutor are the authoritative predicted traces.

## TimestampEvent JSONL

Default location:
`logs/timestamp_events/<source>_<run_id>.jsonl`.

Environment controls:

- `SI_TIMESTAMP_LOG_ENABLED=0`
- `SI_TIMESTAMP_LOG_FILE=/path/events.jsonl`
- `SI_TIMESTAMP_LOG_DIR=/path`
- `SI_TIMESTAMP_RUN_ID=<id>`

Base fields are `event_type`, `event_ns`, `event_time_utc`, `source`, and `pid`.
`timestamp_ns` and `timestamp_time_utc` are exact backward-compatible aliases.
NumPy values are summarized as `shape`, `dtype`, `preview`, `min`, and `max`; this
trace does not store complete large arrays.

Key events and fields:

- `action_chunk_written`: `observation_timestamp_ns`,
  `policy_write_timestamp_ns`, `chunk_shape`, `device_name`.
- `action_dispatched`: `action_timestamp_ns`, `dispatch_timestamp_ns`, and
  `executed_action_timestamp_ns` are aliases for the control-SHM frame timestamp;
  `action_stage=executed`.
- History events may have specific names plus `legacy_event_type`; preserve both
  when changing readers.

For latency, subtract only timestamps from the same wall-clock domain:

- policy publication latency:
  `policy_write_timestamp_ns - observation_timestamp_ns`;
- queue/alignment-to-dispatch latency:
  `dispatch_timestamp_ns - policy_write_timestamp_ns`;
- observation-to-dispatch latency:
  `dispatch_timestamp_ns - observation_timestamp_ns`.

Negative or implausibly large values usually indicate mixed runs, clock adjustment,
or comparison with an extrapolated action timestamp.

## Configuration Workflow

### Control Panel Readiness Handshake

When inference is launched by the Control Panel, the recorder must instantiate
`RecordingControlPlane` from `SUPERINFERENCE_RECORDING_CONTROL_DIR` and publish
`status/recorder.json` with `ready: true` and state `armed` or `recording`.
The legacy `rollout_ready` signal file only starts the policy-side saver and is
not sufficient for the panel's required-recorder gate. A timeout saying
`录制器未在超时前进入 armed/recording 状态` usually means the recorder exited
before this status was published, or was launched without the panel's run
environment. Native launches without those environment variables continue to
use the legacy ready/signal-file path.

Under `policy.infer_rollout`:

1. Set `enabled: true`, `summary_shm_name`, `output_dir`, and recorder
   dependencies. Native launches use a verified config-specific `recording`
   profile when bound, then the active `sys.executable` unless
   `SUPERINFERENCE_RECORDER_PYTHON` or an explicit `python_path` is set; do not
   hardcode machine-local interpreter paths in reusable configs. The selected
   interpreter must provide `dm3-core==0.2.18`, `dm3_core.nedf`, and `nmx_msg`;
   do not install the legacy cp310-only `nmx_nedf_api` wheel.
   Approved source deployments can obtain the exact `dm3-core` wheel from the
   internal Nexus `pypi-releases` repository and `nmx-msg` from `pypi-group`;
   verify the hashes documented in `incubator/nedf3/I2RT_DM3/README.md` before
   running the recording-profile installer.
   The recorder script is resolved from the package root through the stable
   `consumer/io/async_infer_data_saver_NEDF3.py` facade, independent of the
   current working directory.
   Native FastWAM configs carry the full rollout contract with `enabled: false`
   by default, but their code-owned runtime role contract still requires a
   verified config-specific `recording` binding. Environment readiness must not
   depend on whether an operator has enabled recording for the current run.
2. Set `control_shms: ${policy.controls}` to record dispatched robot commands.
3. Enable only available `policy_outputs`. `action_chunk` records the merged
   predicted chunk; disable value/Q outputs for policies that do not produce them.
4. Choose `auto_start`. Otherwise PolicyConnector starts episode recording at the
   first action write and episode control drives stop/result transitions.
5. Keep `queue_size` large enough for camera bursts. A queue drop means the NEDF3
   episode is incomplete even if JSONL traces are intact.

Do not tune `device_delays`, `device_dimension_delays`, `drop_first_actions`, or
policy delay parameters as part of recording instrumentation.

Control Panel passes its validated interpreter through
`SUPERINFERENCE_RECORDER_PYTHON`. `PolicyConnector` must give that variable
precedence over the local runtime binding and `policy.infer_rollout.python_path`;
native/manual launches that set neither use the current `sys.executable`. This guarantees the profile
checked by Control Panel is the process that imports `dm3_core` and writes NEDF3
data while keeping source-checkout configs portable.

The panel resolves that validated interpreter from, in order: its explicit CLI
argument, `SUPERINFERENCE_RECORDING_PYTHON` or its recorder alias, a verified
config-specific `recording` profile, then `policy.infer_rollout.python_path`.
Direct inference follows the same rule before its active-interpreter fallback and
does not guess a repository `.venv`. Profile configuration and verification are
owned by [`runtime-profile-management`](../runtime-profile-management/SKILL.md).

For camera selectors, an existing server `effective_bindings` entry is the
runtime source of truth. On page load or discovery refresh, prefer it over
guessing from the logical SHM suffix or discovery recommendations.
Otherwise the dropdown can show an unapplied draft and force an unnecessary
unbind/rebind cycle before the camera start control becomes available.
An initially parked backend camera may report `capabilities.start=false`; the
frontend must still enable Start when its draft selects a discovered capture
device. The Start handler commits the binding first and only then starts the
rebuilt device, so backend binding validation remains authoritative.

## Common Issues

- Control Panel rollout paths are YAML-owned by default. Resolve relative
  `policy.infer_rollout.output_dir` values from the selected config's
  repository root and write episodes directly below that root, exactly like a
  direct launch. Do not append a host-date `YYYYMMDD` directory. Existing dated
  children remain discoverable as legacy replay roots but are never selected
  for a new recording. A path remembered from an earlier process or config
  must not override this YAML default; only a browser path explicitly selected
  in the current config session may do so. The directory button updates the
  save and offline-replay root without changing the rollout enable switch, while
  opening the selector from the disabled rollout switch enables recording
  after selection. Recording replay reads either root without an inference
  process ever having started. The explicit
  `保存当前 config` action writes effective bindings, gripper ports, rollout
  settings, and restart-aware FPS values back to the selected source YAML and
  atomically replaces changed YAML files without creating `.bak` sidecars.

- Repeated `Unknown device type: WebCameraOpenPi` means the R1 recorder camera
  whitelist has regressed; frames are being skipped even though DeviceManager is
  healthy.
- Repeated `Unknown device type: ControlSHM` means executed controls and policy
  chunks are not entering the episode. A 16D `*_control` stream is a dual-arm
  command; a 2D stream with width 16 is a predicted action chunk.
- `dm3-core==0.2.16` must not be used for recording. Its chunk writer truncates
  the complete output path at the first dot, so a checkout path containing a
  dot can write MCAP shards outside the episode. Use
  `dm3-core==0.2.18` or newer behavior verified by a real write beneath a
  dotted parent. Before publishing a successful conversion receipt, require a
  nonempty episode-local `data/mcap_index.json` and every indexed regular,
  nonempty MCAP shard. The index must also contain each connected camera's
  RGBD topic and the connected Rizon/Yam state topics; a metadata-only or
  action-only episode must remain incomplete and stop the recorder.
- `No data found in episode` can be a false anomaly-detector result when its
  camera feature map is empty. Inspect `data/mcap_index.json` before declaring
  the episode empty, and reuse the recorder's canonical camera mapping. The
  original anomaly check runs synchronously immediately after finalization and
  may report this transient SDK read failure; it does not determine whether the
  MCAP was successfully saved.
- `NEDF recording could not be inspected` can be caused by invoking the trusted
  inspector by absolute path with the wrong package root or Python. The replay
  service must dynamically use the selected workflow's resolved recording
  Python rather than snapshotting the Control Panel interpreter before config
  selection. The inspector resolves the repository root independently of cwd;
  verify that the selected recording interpreter imports
  `dm3_core`, `nmx_msg`, `cv2`, `yaml`, and `numpy`.
- The R1 saver lives at `incubator/r1/...`, so its repository root is
  `Path(__file__).resolve().parents[2]`. Using `parents[1]` starts the anomaly
  subprocess from `incubator/`, where `python -m incubator...` fails immediately
  with exit code 1 before reading the episode.
- Trusted conversion is fail-closed on the SHA-256 values for the accepted-
  episode wrapper, NEDF3 converter, and shard merger in
  `incubator/control_panel/converter_profile.py`. After changing any of those
  sources, regenerate and review all three expected hashes together, bump the
  dated `DEFAULT_PROFILE_ID`, and rerun the activation smoke test so a new
  receipt is issued. Never weaken the mismatch checks or reuse an activation
  receipt from the previous profile.

## Visualization and Inspection

- NEDF3 episode: inspect it through the control-panel recording replay UI. The
  dedicated legacy episode visualizer has been removed.
- Action/latency trace: open
  `visualizers/action_state_xyz_visualizer.html` and load one or more
  `*_xyz.jsonl` files. Preserve the “original full chunk” (`all_actions`) and
  “post alignment” (`aligned_actions`) series.
- Timestamp events: parse each JSONL line independently and join by run,
  `device_name`, observation timestamp, and chunk index where available. Do not
  join solely by row order.

## Validation

After instrumentation changes:

```bash
RUNTIME_CONFIG=<config-with-recording-profile>
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$RUNTIME_CONFIG" --role recording -- \
  -m py_compile \
  consumer/inference/common/rollout_controller.py \
  utils/action_state_xyz_logger.py \
  utils/timestamp_event_logger.py \
  manager/base_policy_runner.py
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$RUNTIME_CONFIG" --role recording -- \
  -m pytest -q \
  tests/test_rollout_trace_logger_contracts.py \
  tests/test_action_executor_aligned_chunk_logging.py
```

Also confirm the visualizer still contains `all_actions` and `aligned_actions`, and
does not contain `pred_abs_xyz`.
