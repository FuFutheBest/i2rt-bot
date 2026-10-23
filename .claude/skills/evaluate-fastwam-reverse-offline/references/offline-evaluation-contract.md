# Offline FastWAM Evaluation Contract

This reference defines the data and interpretation contract for the
`evaluate-fastwam-reverse-offline` skill. It is deliberately independent of a
particular robot, camera numbering, gripper port, task, or checkpoint host
path.

## Inputs

- One composed SuperInference config whose policy class is exactly
  `FastWAMPolicy` and whose `policy.params.ckpt_path` resolves to the reviewed
  checkpoint.
- A checkpoint directory accepted by
  `consumer/inference/fastwam_server.py`. The production loader requires
  `config.yaml`, `dataset_statistics.json`, and at least one
  `checkpoints/weights/*.pt` artifact; do not use a params-only fixture for a
  model evaluation.
- One completed run from `TimestampedObservationRecorder` in
  `utils/observation_recorder.py`. The run root contains:

  ```text
  manifest.json
  frames.jsonl              # raw synchronized summary-SHM observations
  frames/*.npz
  inference.jsonl           # exact policy inputs and timestamp metadata
  inference/*.npz
  ```

- The camera and robot names recorded in the manifest/config. There is no
  universal left/right naming scheme; resolve the names from the selected
  config. The evaluator must validate the model's dynamic
  `contract.image_meta`, `state_dim`, and `action_dim` before plotting.

The run should be stopped cleanly. A nonzero `dropped_frames` or
`write_failures` field in the manifest makes historical-frame conclusions
questionable and must be reported in the summary.

## Determinism

Use a fixed seed, fixed `num_inference_steps`, fixed device/dtype, and the same
resolved config/checkpoint for every comparison. Disable model warmup and
predicted-video generation for an action-only audit unless the purpose is
explicitly to compare those paths. Exact bitwise reproducibility still depends
on the accelerator, Torch version, nmx_vla checkout, and numerical kernels;
record those identities rather than claiming universal reproducibility.

The evaluator should call the refactored `FastWAMRuntime.infer()` path (or a
thin helper that delegates to it), so preprocessing, normalizers, temporal
metadata, and action postprocessing remain the same as deployment. Do not call
`FastWAMPolicy` through `PolicyConnector`, because that would attach to policy
SHM and change the safety boundary.

## Outputs

Write to a new diagnostic directory and include:

- `offline_result.npz`: normalized, relative, and (when the contract supports
  it) absolute actions in runtime order plus an audit-only reversed copy,
  state, and timestamps.
- `summary.json`: config/checkpoint identity, recorder provenance, runtime
  metadata, dynamic shapes, finite-value status, latency, and temporal-order
  audit. Include recorder drop/write-failure counts.
- A preprocessed input preview, when available.
- Per-channel and per-arm action plots. Use labels derived from
  `contract.action_layout`; only use left/right Cartesian labels for a verified
  dual-arm layout.
- A 3D trajectory plot only when the action contract declares Cartesian pose
  channels and the quaternion order/relative pose frame are recorded.

Every saved numeric array must be finite. Shape mismatches, missing NPZ keys,
unknown camera mappings, and unresolved Hydra values are hard failures, not
warnings.

## Temporal semantics

Reverse training metadata describes how the training sample was constructed; it
does not by itself define the order that a deployed client should command.
Record and compare at least:

- `reverse_temporal_chunks` from the effective model/runtime metadata;
- `rtc_reverse_temporal_guidance` and `enable_rtc` when RTC is enabled;
- the action layout, horizon, and control frequency;
- the checkpoint and vendored nmx_vla revisions.

The audit may reverse the returned time axis and compare endpoint distances to
the current state. A first predicted pose closer to the anchor is consistent
with chronological output; a final pose closer to the anchor is evidence for a
possible reversal. Neither result proves policy quality or authorizes a robot
run. Mixed-arm recommendations must remain `mixed`; do not collapse them into
a global flip decision.

RTC guidance must obey the server/client temporal metadata contract documented
by [`fastwam-rtc-guidance`](../../fastwam-rtc-guidance/SKILL.md). An offline
plot cannot justify changing either flag.

## Read-only boundary

The capture path may open only observation/data SHM readers and copy their
frames. The evaluation path may load a checkpoint into its own model process or
diagnostic context, but it must not:

- open or write any control/policy SHM;
- instantiate `PolicyConnector` or run `consumer/inference/fastwam_policy_infer.py`;
- start or stop `device_starter.py`, a Control Panel workflow, or a robot
  process;
- write source configs, checkpoints, calibration files, or input recorder runs;
- treat a plot recommendation as a deployment setting.

Use a new output directory with exclusive creation and preserve all source
artifacts. If model memory requires stopping an existing FastWAM Server, route
that process lifecycle through `fastwam-inference-runtime`; an offline audit
itself must not issue lifecycle or hardware commands.
