# NEDF3 Rollout Contract

This document is the compatibility boundary for the control-panel rollout path:

```text
recording -> NEDF3 -> LeRobot v2.1
```

## Reference Sources

- Raw NEDF3: `origin/chenyiming/R1_teleoperation`, especially
  `incubator/r1/async_data_saver_NEDF3_r1.py`. The older control-panel UMI/NEDF3
  experiment is deliberately unsupported.
- LeRobot: `/mnt/workspace/shenyibo/datasets/waic_chip_0715` and
  `/mnt/workspace/shenyibo/datasets/chip_paralle_0710_742episodes` on
  `H200-4061`. Both datasets expose the same metadata and Parquet schema.
- SDK behavior: the three supplied NEDF3 wheels. Wheel APIs take precedence
  over assumptions in application code and must be probed in an isolated Linux
  environment before the recorder or converter is activated.

## Camera Roles

NEDF3 recordings use the R1 contract:

| Role | NEDF3 id | NEDF3 topic | LeRobot feature |
| --- | --- | --- | --- |
| right wrist | `camera_0` | `nmx/hal/camera/camera_0/rgbd` | `observation.images.wrist_image_2` |
| left wrist | `camera_1` | `nmx/hal/camera/camera_1/rgbd` | `observation.images.wrist_image_1` |
| head | `camera_2` | `nmx/hal/camera/camera_2/rgbd` | `observation.images.image` |

`metadata.camera_info` stores the R1 role labels `inhand_right`,
`inhand_left`, and `head`. Physical device names are provenance, not semantic
camera ids. `metadata.main_camera` is `camera_0` (right wrist), matching the
Vive recorder and keeping the alignment timeline stable whether or not the
optional head camera is configured.

The numeric SHM suffix is the global `device_class_id`, not the dataset camera
ID or physical `capture_id`. Map camera-only configuration order: OpenPI uses
left/right/head → camera_1/0/2; R1/Windows collection uses right/left/head →
camera_0/1/2. Preceding non-recorded devices do not shift these roles.

## NEDF3 Topics

The R1 HAL recording contract is:

```text
nmx/hal/camera/camera_0/rgbd
nmx/hal/arm/right/tcp/{cmd,state,vel}
nmx/hal/arm/right/joint/{state,vel}
nmx/hal/arm/right/{eef_wrench,extra}
nmx/hal/hand/right/{cmd,state,extra}
nmx/hal/camera/camera_1/rgbd
nmx/hal/arm/left/tcp/{cmd,state,vel}
nmx/hal/arm/left/joint/{state,vel}
nmx/hal/arm/left/{eef_wrench,extra}
nmx/hal/hand/left/{cmd,state,extra}
nmx/hal/camera/camera_2/rgbd
nmx/hal/head/{cmd,state,joint/state,extra}
```

Camera payloads are protobuf `RGBD` messages whose RGB member contains encoded
JPEG bytes. Arm/head TCP commands and states use `ArmTcpCmd` and `ArmTcpState`;
joint streams use `ArmState`; gripper commands and states use `HandCmd` and
`HandState`. Dataset quaternions are always ordered `wxyz`.

The Flexiv 44D state supplies left/right joint positions, measured TCP poses,
and gripper widths. A 20D `YamHeadRobot` state supplies six joints and the
measured ego TCP at indices `13:20`; the source quaternion there is `xyzw` and
must be converted to `wxyz`.

Joint positions are first-class protobuf streams at
`nmx/hal/arm/{left,right}/joint/state` (7D) and
`nmx/hal/head/joint/state` (6D). The authoritative 16D inference
`RizonRobot_<device_class_id>_control` stream is split into left/right `tcp/cmd` and `hand/cmd`
topics. Auxiliary action-chunk, Q, and value SHMs are not recorded as actions.

## LeRobot v2.1

`observation.state` and `action` are float32 vectors with this exact 23D order:

```text
left(x,y,z,qw,qx,qy,qz,gripper),
right(x,y,z,qw,qx,qy,qz,gripper),
ego(x,y,z,qw,qx,qy,qz)
```

The seven scalar/index columns are followed by five extra columns for each of
`left`, `right`, and `ego`:

```text
observation.state
action
timestamp
frame_index
episode_index
index
task_index
observation.extra.{left,right,ego}.raw_pose
observation.extra.{left,right,ego}.hand_state
observation.extra.{left,right,ego}.raw_hand_state
observation.extra.{left,right,ego}.roi_marker_angle
observation.extra.{left,right,ego}.aligned_timestamp_ns
```

The Parquet schema metadata is:

```text
slam_anomaly_status=missing
width_anomaly_status=missing
```

LeRobot `meta/episodes.jsonl` stores `device_type` as the scalar string
`rollout`. Both `task_description` and the single entry in `tasks` are derived
from the episode's NEDF `metadata.task_name`. That task name is resolved from
the currently selected config's `policy.params.task` whenever the control
panel builds the recording runtime config.

Both `frame_index` and `index` are episode-local and must equal
`0..episode_length-1` in every Parquet file. `episode_index` identifies the
episode across the dataset; `index` must not be rewritten as a dataset-global
frame counter when shards are merged.

The reference datasets use LeRobot `v2.1`, `fps=30`, `target_fps=10.0`, AV1
video, 320x240 wrist images (height x width), and 240x320 head images.
For NEDF3, `action[t]` is the command recorded from the authoritative control
SHM at that timestamp. It is not synthesized from `observation.state[t+1]`.

## Compatibility Rules

- New recording writes NEDF3 only.
- Control Panel and direct inference write `episode_*` directories immediately
  below `policy.infer_rollout.output_dir`; neither adds a `YYYYMMDD` directory.
  Replay may scan existing dated children for backward compatibility only.
- The obsolete UMI-NEDF3 topic layout is not accepted by writer, replay, or
  converter code.
- Camera ids must never be inferred without reading
  `metadata.data_format`.
- A config without a head camera remains valid. NEDF3 does not fabricate a raw
  head stream; the LeRobot compatibility layer may create its declared head
  placeholder without blocking inference.
- A config with Yam must record the measured ego TCP. Identity ego padding is
  permitted only for recordings that genuinely have no Yam state.
- Conversion and replay must consume finalized data read-only and preserve the
  source episode hash.

## QC and conversion lifecycle

`result` describes the recorder outcome (`success`, `failure`, or `incomplete`),
while `status`/applied decisions describe later QC actions. They are separate
fields and must not be inferred from one another. The replay API therefore
supports `GET /api/recordings?result=success` for selecting all successful
recordings in a date/search scope, including a successful episode whose final
status is still incomplete.

Finalizing an episode submits its single-episode conversion in the background;
accepting, rejecting, resetting, or stopping inference does not cancel that
job. A batch export may include only conversion-ready shards and must leave the
source NEDF3 tree unchanged. A failed conversion is terminal for the automatic
queue and keeps both the source episode and its diagnostic receipt for retry or
manual inspection.

## Activation Gate

Before enabling automatic conversion, a Linux probe must verify the supplied
wheel versions and APIs, write and finalize one NEDF3 episode, reopen every
declared topic, convert it, fully decode all three videos, validate Parquet and
metadata, and confirm the source tree hash did not change.
