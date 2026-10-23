---
name: lingbot-va-rtc
description: >
  Maintain LingBotVA real-time control (RTC) inference in SuperInference. Use
  when changing policy.params.enable_rtc for LingBotVAPolicy, latency-to-step
  estimation, action_index propagation, websocket RTC payloads, LingBotVA server
  action inpainting, or chunk-prefix/action-manager smoothing in
  policy/lingbot_va_policy.py, consumer/lingbot_va_server.py,
  manager/utils/action_manager.py, manager/utils/rtc_interpolation_manager.py,
  and third_party/nmx_vla/nmx_vla/model/modules/lingbot_va.
---

# LingBotVA RTC

## Core Flow

RTC is config-driven from `policy.params.enable_rtc`.

- `manager/base_policy_runner.py` reads `enable_rtc` and writes `action_index` to policy SHM.
- `manager/utils/action_manager.py` must provide `get_original_current_step()`.
- `manager/utils/rtc_interpolation_manager.py` can smooth chunk boundaries while preserving the original chunk timeline.
- `consumer/policy_connector.py` reads `action_index` only when the policy has `enable_rtc=True`.
- `policy/lingbot_va_policy.py` owns the latency tracker and builds the websocket RTC payload.
- `consumer/lingbot_va_server.py` applies RTC only when both payload fields are present.
- `third_party/nmx_vla/nmx_vla/model/modules/lingbot_va/utils/rtc.py` holds RTC helpers.

When `enable_rtc` is false, do not construct RTC prefix payloads and keep the
normal LingBotVA request/response path unchanged.

## Missing Action Filling

For checkpoints trained with action-history dropout / sanity-check missing action
filling, keep the inference wrapper aligned with the training runtime artifact:

- `consumer/lingbot_va_server.py` must pass `learnable_action_embedding_config`
  through to `load_transformer()` so `missing_action_filling_*` kwargs match the
  checkpoint.
- `zero_action_condition: true` only means the action-history condition values
  are zeroed. Set `action_condition_all_missing=True` only when
  `zero_action_condition_uses_missing_embedding` is true, matching training
  samples whose `learnable_action_embedding_enabled=True`.
- For no-learnable-token checkpoints, keep
  `zero_action_condition_uses_missing_embedding: false` so dropped action history
  uses the normal action embedder on zero values, matching the latest
  `zero_condition_mask` training path.
- Head-view/action dropout `action_channel_slice` is in raw/source action
  channel space. In inference, map that slice through `used_action_channel_ids`
  before updating the action layout, and fail on invalid or out-of-range slices
  just like the training path.
- For all-missing action-history checkpoints whose runtime config has empty
  `dataset_used_canonical_dims` / `gripper_canonical_dims`, keep those lists
  empty in the outer SuperInference config instead of inventing dataset-specific
  masks.

FastWAM drop-head/no-learnable-token audit checklist:

- Prefer checkpoint artifacts (`lingbot_runtime_config.json`,
  `checkpoint_step_*/transformer/config.json`, temporal metadata, norm-stats
  manifest) over filename guesses when checking train/infer parity.
- A no-learnable-token checkpoint can still have
  `learnable_action_embedding_config.enabled=true`; the decisive model contract
  is the transformer config buffers. Empty `missing_action_filling_used_channels`,
  empty `missing_action_filling_gripper_channels`, and
  `missing_action_filling_pad_unused=false` mean no dataset-specific missing
  channel routing is configured.
- If `action_history_condition_dropout_prob=1.0` but
  `zero_action_condition_uses_missing_embedding=false`, inference should pass
  zero action-history values through the normal action encoder and should not set
  `action_condition_all_missing`.
- Per-dataset head-view/action dropout keeps the model in canonical
  `action_dim` space. It filters `used_action_channel_ids`, rebuilds
  `inverse_used_action_channel_ids`, masks dropped canonical channels in noisy
  action sampling, and postprocesses only the surviving used channels.
- MoT `null_action_tokens` and `action_encoder_missing` are resident parameters.
  In cached inference, `null_action_tokens` are not used, and
  `action_encoder_missing` participates only when `action_condition_all_missing`
  or explicit validity/missing-action inputs route action embeddings through the
  missing-action filling path.

## Prefix Semantics

LingBotVA actions are relative pose channels. Do not send the previous raw chunk
directly as RTC prefix in local-frame mode.

Use this rebasing pattern:

1. Convert the previous server action from model-relative channels to absolute robot action.
2. Convert that absolute trajectory back to relative channels using the current reference robot state.
3. Shift by `action_index + 1` so prefix step 0 represents the action after the one currently being dispatched.
4. Pad the tail back to the original chunk length before sending `[C,F,H]` to the server.

Do not run control-side gripper normalization while preparing a model-side RTC prefix.

Important coordinate note: the client may temporarily convert the previous
relative chunk into absolute robot poses so it can re-anchor across two
predictions, but the websocket `rtc_prefix_action` sent back to LingBotVA must
be relative to the current robot pose/reference. This matches checkpoints whose
predicted actions are relative to the current robot pose.

## Latency

`LingBotVAPolicy` tracks inference latency at runtime and converts it to steps using `execution_fps`.

Useful config keys:

```yaml
policy:
  params:
    enable_rtc: false
    execution_fps: ${policy.control_fps}
    rtc_default_delay_steps: 24
    rtc_latency_stat: mean
    rtc_prefix_attention_schedule: exp
    rtc_guidance_full_steps_override: null
    rtc_guidance_decay_steps_override: null
```

Keep `rtc_default_delay_steps` conservative enough for the first RTC request before the tracker has samples.
Set `rtc_guidance_full_steps_override` to force the leading
`guidance_weight=1` segment length, and `rtc_guidance_decay_steps_override` to
force the scheduled decay segment length. Leave either value as `null` to keep
that segment automatic. These lengths are action timesteps and are clamped to
the shifted prefix overlap.

## Observability

`policy/lingbot_va_policy.py` logs RTC client-side stats whenever an RTC prefix
is sent:

- latest and mean policy inference latency,
- mean occupied action timesteps (`mean_latency_s * execution_fps`) and its ceil,
- source overlap timesteps copied from the previous prediction chunk into the
  new RTC prefix (`previous_chunk_len - action_index - 1`),
- delay-adjusted overlap timesteps expected to remain after inference returns
  (`source_overlap_steps - inference_delay`, clamped at zero),
- full action-chunk prefix/guidance weight distribution plus nonzero count,
  sum, min, and max.

`consumer/lingbot_va_server.py` also logs the server-side weight tensor it
actually builds with `action_prefix_weight_tensor()`. The client and server
logs should match for the same `inference_delay`, `prefix_attention_horizon`,
and `prefix_attention_schedule`; if they differ, prefer the server log as the
ground truth for inpainting behavior.

The prefix action starts at `action_index + 1`, so RTC horizons must be based on
`overlap_steps = previous_chunk_len - action_index - 1`. Do not compute
`prefix_attention_horizon` directly from `chunk_length - inference_delay -
action_index`; that is one step too long and lets a padded zero tail step receive
nonzero guidance.

## Action Manager Interpolation

Use `rtc_interpolation_manager` when an RTC-enabled config needs an additional
control-side boundary smoother:

```yaml
action_executor:
  action_executor:
    chunk_manager:
      type: rtc_interpolation_manager
      drop_first_actions: 9
      interpolation_steps: 3
      quaternion_format: xyzw
```

This manager drops stale leading actions like `BaseChunkManager`, then rewrites
only the first `interpolation_steps` actions of the accepted chunk as a fixed
length transition from the last actually dispatched action to the end of that
interpolation window. It does not upsample or extend chunks, so
`get_original_current_step()` still reports the original policy chunk index
(`drop_first_actions + current_step`) for RTC feedback.

For base and smooth chunk managers, `drop_first_actions` also accepts `"auto"`.
Automatic mode measures end-to-end
latency from the observation timestamp carried by each policy chunk to the
action executor, tracks a sliding latency window, and drops
`ceil(p95_latency * execution_fps)` actions. It uses six actions before any
valid latency sample is available. Auto/AutoV2/Joint, NoDelay, and Ensemble
managers retain their own chunk-selection semantics and do not use this option.

Avoid enabling the final upsampling stage with RTC unless the manager also remaps
upsampled executor steps back to original chunk indices. Otherwise the policy
can receive an `action_index` on the interpolated timeline instead of the model
chunk timeline.

## Configurable Action Chunk Smoothing

Use `smooth_chunk_manager` for an optional upsample-filter-downsample stage
followed by an optional execution upsampling stage. At least one stage method
must be configured:

```yaml
action_executor:
  action_executor:
    chunk_manager:
      type: smooth_chunk_manager
      drop_first_actions: 9
      smoothing:
        method: savgol
        upsample_ratio: 2
        window: 13
        polyorder: 3
        mode: interp
        boundary_context_steps: 12
        boundary_blend_steps: 12
        smooth_gripper: true
      upsampling:
        method: null
      quaternion_format: xyzw
```

The smoothing stage supports `savgol` and `moving_average`. Savitzky-Golay mode
prepends up to `boundary_context_steps` actions that were actually dispatched,
upsamples with linear interpolation and quaternion slerp, filters, and samples
back onto the original action timestamps. It therefore preserves the incoming
policy chunk length before `BaseChunkManager` applies `drop_first_actions`.
Moving-average mode keeps the former FPS-resampling behavior and accepts
`source_fps` and `command_fps`; its output length can change when those rates
differ.

The optional `upsampling` stage supports `linear` and `cubic` for non-quaternion
channels and always uses quaternion slerp. Its `ratio` controls the final
executor timeline density. Leave `upsampling.method: null` for RTC: enabling it
changes the cached timeline, so `get_original_current_step()` no longer maps
directly to the policy chunk index. Savitzky-Golay smoothing by itself remains
safe for RTC `action_index` feedback.

For dual-arm Rizon EEF action shape 16, quaternion channels are treated as
`[3:7]` and `[11:15]`; for single-arm shape 8, `[3:7]`. Unknown action
dimensions are filtered as plain vectors.

When cross-chunk seams remain visible, prefer increasing boundary-specific
smoothing before globally enlarging `savgol_window`. `boundary_context_steps`
controls how many already dispatched actions are prepended for the Savitzky-
Golay fit, while `boundary_blend_steps` blends the start of the new chunk from
the previous dispatched trend into the filtered prediction with smoothstep
weights. At 30 Hz, 12 blend steps is about 0.4 s. `smooth_gripper: true`
explicitly smooths gripper channels `[7]` for single-arm shape 8 and `[7, 15]`
for dual-arm shape 16.

For FastWAM configs where the first generated chunk can be unstable, prefer
skipping the whole first executor chunk and starting robot execution from the
second policy chunk:

```yaml
action_executor:
  action_executor:
    chunk_manager:
      type: base_manager
      drop_first_actions: 9
      skip_first_chunk: true
```

Keep `drop_first_actions: 9` aligned with the current RTC full-guidance delay.
`skip_first_chunk: true` makes `BaseChunkManager` ignore the first chunk after
reset while remembering its timestamp, so the executor does not repeatedly
re-accept the skipped chunk while waiting for the second one.

Important tuning detail: Savitzky-Golay smoothing applies `boundary_blend_steps`
before `BaseChunkManager` drops the stale leading actions.
With `drop_first_actions: 9`, a small value such as `boundary_blend_steps: 3`
only blends actions that will be discarded. For config-only boundary A/B tests,
set `boundary_blend_steps >= drop_first_actions + executed_transition_steps`
(for example 15 for six executed blend steps after dropping nine), or implement
an RTC-specific manager that applies boundary blending after the drop.

`PolicyConnector` must also write the real observation timestamp for the first
action chunk, not `time.time_ns()`. The action executor uses the frame timestamp
for fresh/stale chunk checks after reset; changing the first chunk to wall-clock
write time bypasses that guard and makes the first chunk semantically different
from later chunks.

## Server Inpainting

Server-side RTC is best-effort:

- normalize `rtc_prefix_action` with `preprocess_action`,
- build action-time weights with `action_prefix_weight_tensor`,
- after each action scheduler step, blend with the prefix projected to the next scheduler noise level,
- fall back to normal sampling if prefix shape/config validation fails.

The current implementation supports `ones`, `zeros`, `linear`, and `exp` prefix schedules.
When running against an `nmx_vla` branch that does not export RTC helpers from
`nmx_vla.model.modules.lingbot_va.utils` (for example sanity-check branches that
only carry missing-action-filling changes), keep these small RTC inpainting
helpers local to `consumer/lingbot_va_server.py` instead of importing missing
symbols from third_party.

## Validation

Run lightweight checks after changes:

```bash
python -m py_compile \
  policy/lingbot_va_policy.py \
  consumer/lingbot_va_server.py \
  manager/utils/action_manager.py \
  manager/utils/rtc_interpolation_manager.py \
  third_party/nmx_vla/nmx_vla/model/modules/lingbot_va/utils/rtc.py
```

If project dependencies are available, run:

```bash
uv run pytest tests/test_lingbot_va_rtc.py -q
```

If `uv`, `pytest`, or numerical packages are unavailable, record that limitation
and at least run `py_compile` plus `ast.parse` checks.
