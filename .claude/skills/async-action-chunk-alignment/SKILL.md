---
name: async-action-chunk-alignment
description: Maintain latency-based action chunk replacement, fractional trajectory resampling, and original action-index reporting. Use when changing drop_first_actions, BaseChunkManager, SmoothChunkManager, ActionExecutor chunk switching, or asynchronous policy latency compensation.
---

# Asynchronous Action Chunk Alignment

## Timing contract

- Align replacement chunks from the timestamp represented by `action[0]` to the time the ActionExecutor receives and dispatches the chunk.
- When an observation timestamp is present, use that chunk's measured latency. Do not substitute a rolling percentile for known latency.
- Use tracked P95 only when a chunk lacks an observation timestamp. If neither timestamp nor history is available, warn and use zero offset rather than inventing latency.
- Keep the first accepted chunk at index zero because no older policy trajectory advanced while it was being inferred.
- `drop_first_actions: auto` is wall-clock alignment: it compensates real inference and transport latency from observation timestamp to chunk arrival. In RoboTwin `execution_clock: sim_time`, do not replace this with `0` merely because dispatch is driven by the simulator clock. The simulator clock controls when a target is sent to SAPIEN; the wall clock still describes how old the replacement trajectory is.
- When a sim-time poll crosses multiple action deadlines, advance the existing chunk manager by every crossed execution step and dispatch only the newest row. A replacement chunk that was already wall-clock aligned consumes its current row once, so latency is not counted twice.

## Fractional alignment

For `offset_steps = latency_seconds * execution_fps`, use:

- `base = floor(offset_steps)`
- `fraction = offset_steps - base`
- output step `j` interpolates source steps `base + j` and `base + j + 1`

Use linear interpolation for continuous vector channels and SciPy SLERP for known EEF quaternion layouts:

- 8D single arm: quaternion `[3:7]`, gripper `7`
- 16D dual arm: quaternions `[3:7]` and `[11:15]`, grippers `7` and `15`

`quaternion_format` selects `xyzw` or `wxyz`. `auto_drop_gripper_mode` selects `linear` or zero-order `hold`.

The policy SHM action index is integer-only. Report `ceil(fractional_offset)` for compatibility while exposing the exact value through `get_original_current_offset()`.

When a chunk manager changes the sampling grid, preserve an explicit offset for
every processed action in original policy-index units. For example, 3x execution
upsampling maps processed offsets to `0, 1/3, 2/3, 1, ...`; do not report the
processed 90 Hz index as if it were the original 30 Hz policy index.

Boundary interpolation may prepend processed samples before original action zero.
Automatic latency alignment must add that processed prefix before slicing, so the
first dispatched action still represents `latency * source_fps` in the original
policy timeline.

## Visualization logging

- `ActionExecutor` owns `chunk_switch`, `pred_chunk_start`, `pred_chunk_end`, `all_actions`, and `aligned_actions`.
- Snapshot both the original policy chunk and `BaseChunkManager.get_current_action_chunk()` immediately after `put()` so later cache mutation cannot change either trace.
- Snapshot `get_current_original_action_offsets()` with the aligned chunk; use those exact offsets for `original_action_offset`.
- `all_actions` is the complete original policy chunk. Its step-zero timestamp is `switch_timestamp - original_start_offset * original_policy_interval`.
- Space `all_actions` with the original policy interval and `aligned_actions` with the execution interval. They are different when smoothing or upsampling changes the sampling grid.
- `aligned_actions` is the exact post-drop/interpolation/smoothing chunk. Anchor its step zero to the actual first dispatch timestamp and label later timestamps `action_executor_extrapolated`.
- `pred_chunk_start` and `pred_chunk_end` bound the original `all_actions` timeline, so the band starts exactly at original action zero.
- Set `chunk_stage=policy_original` or `action_executor_aligned` and include `original_chunk_length`, `aligned_chunk_length`, `original_start_offset`, `original_step_interval_ns`, and `aligned_step_interval_ns`.
- Do not reconstruct these events in `PolicyConnector` by matching floating-point action values against control SHM. Fractional interpolation makes that match inherently unreliable.

## Configuration

```yaml
chunk_manager:
  type: base_manager
  drop_first_actions: auto
  interpolate_auto_drop: true
  quaternion_format: xyzw
  auto_drop_gripper_mode: linear
```

## Smooth-manager bring-up

When smoothing appears to have no effect, verify the runtime selection before
tuning the filter. The startup log must say
`ActionExecutor chunk_manager: type=SmoothChunkManager`; a commented
`type: smooth_chunk_manager` leaves `base_manager` active and bypasses all
smoothing.

Treat `smoothing.window` as a window on the dense, internally upsampled grid.
For example, `window: 13` with `upsample_ratio: 5` spans only about 2.6 original
action intervals and can look unchanged. To prove the path works before tuning
fidelity, use a deliberately aggressive odd window near the policy horizon,
`polyorder: 1`, and `upsample_ratio: 1`. Keep `upsampling.method: null` unless
the executor dispatch rate is also adjusted; final-stage upsampling at an
unchanged dispatch FPS slows the original trajectory.

Boundary blending runs before `drop_first_actions`. To retain `N` visibly
blended actions after a fixed drop of `D`, configure at least
`boundary_blend_steps: D + N`. Verify the active production configuration with
a high-frequency synthetic trajectory and require a large reduction in total
variation while checking that `smooth_gripper: false` preserves gripper
channels in both the filter and cross-chunk boundary blend. Do not apply the
boundary blend to gripper indices `[7]` or `[7, 15]` when that option is false.

## Validation

```bash
RUNTIME_CONFIG=<inference-config-name>
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$RUNTIME_CONFIG" --role policy -- \
  -m pytest -q \
  tests/test_chunk_manager_auto_drop.py \
  tests/test_smooth_chunk_manager.py \
  tests/test_action_executor_aligned_chunk_logging.py
```
