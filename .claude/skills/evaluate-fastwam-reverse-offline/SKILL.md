---
name: evaluate-fastwam-reverse-offline
description: Run or audit read-only offline evaluation of reverse-temporal FastWAM actions from recorded observations, comparing runtime and time-reversed trajectories without writing control shared memory or moving hardware.
---

# Evaluate Reverse FastWAM Offline

Use this skill when a FastWAM checkpoint may emit actions in an unexpected
temporal order, when a reverse-training experiment needs an offline audit, or
when action plots appear to alternate between arms. Read
[`references/offline-evaluation-contract.md`](references/offline-evaluation-contract.md)
before interpreting a result.

## Current refactor boundary

- The model-side implementation is
  `consumer/inference/fastwam_server.py` (`FastWAMRuntime`). The policy client
  is `consumer/inference/fastwam_policy_infer.py`; the files under
  `consumer/` are compatibility facades only.
- `utils/observation_recorder.py` is the recorder contract. A completed run
  contains `manifest.json`, `frames.jsonl`, `inference.jsonl`, and referenced
  `frames/*.npz` and `inference/*.npz` files.
- The adapted helper entrypoints live under `scripts/analysis/`. Use them only
  when the corresponding script is present; do not recreate the pre-refactor
  evaluator under `consumer/` or add a second recorder format.

## Workflow

1. Select one native Hydra config and one checkpoint. Compose and inspect the
   effective config before loading a model:

   ```bash
   python tests/config/policy_launch_plan.py \
     --config-name <native-config-name> --json
   ```

   Confirm that the resolved policy class is exactly `FastWAMPolicy`, and use
   its `policy.params.camera_devices`, `robot_device`, action controls, and
   checkpoint path. Do not infer device names from a branch name or copy a
   workstation's camera, gripper, port, or FPS values into the audit.

2. Prefer a completed `TimestampedObservationRecorder` run. Check that its
   manifest is finalized, that the requested inference record and all NPZ
   references exist, and that the recorded device names match the selected
   config. A live capture helper may read observation SHM only; pass names
   derived from the config rather than relying on historical defaults.

3. Use the read-only capture helper when observation SHM is available, and
   create a new, non-existing output directory. For example:

   ```bash
   python scripts/analysis/capture_fastwam_offline_observation.py \
     --output-dir <new-observation-dir> \
     --left-shm <camera-a-data-shm> \
     --right-shm <camera-b-data-shm> \
     --robot-shm <robot-data-shm>
   ```

   If a deployment does not expose observation SHM, use an existing recorder
   run. Do not silently mix recorder schemas.

4. Run the adapted action-only evaluator with a fixed seed and inference-step
   count. The evaluator must instantiate `FastWAMRuntime` in a separate model
   process or in-process diagnostic context, never `PolicyConnector`:

   ```bash
   python scripts/analysis/evaluate_fastwam_reverse_offline.py \
     --config-name <native-config-name> \
     --observation-dir <observation-run> \
     --inference-index <index> \
     --seed <fixed-seed> \
     --num-inference-steps <fixed-step-count> \
     --output-dir <new-evaluation-dir>
   ```

   Do not pass a checkpoint override that differs from the reviewed config
   unless the evaluator records that override in its summary and the operator
   explicitly selected it.

5. Inspect the evaluator summary, metadata, finite-value check, and per-arm
   plots. Compare `runtime` with a copied, time-axis-reversed array only for
   diagnosis. Never feed the time-reversed array to a policy client or a robot.

6. Record the effective `reverse_temporal_chunks`, `action_layout`, state/action
   dimensions, checkpoint identity, nmx_vla revision, and RTC metadata with the
   result. A temporal-order heuristic is evidence for further review, not a
   deployment approval.

## Interpretation

- `runtime` means the output returned by `FastWAMRuntime.infer()` after the
  current checkpoint normalizer and relative-to-absolute conversion. It does
  not mean the raw training tensor or a direct vendor call.
- `time reversed` is an audit copy. The training setting and the deployed
  `reverse_temporal_chunks` metadata must both be inspected; never assume that
  a reverse-labelled checkpoint should be flipped at deployment.
- Endpoint distance to the current state is only a heuristic. Discontinuous
  policies, relative-pose frame mistakes, stale observations, or an incorrect
  camera mapping can produce the same symptom.
- Plot each action channel and arm independently. A single flattened trace can
  create apparent two-sample alternation even when the model output is valid.
- The common dual-arm 16-channel Cartesian plot is applicable only when the
  resolved contract reports that layout. Otherwise report the actual dynamic
  `action_dim`/`state_dim` and do not label channels as left/right by position
  alone.

## Safety boundary

Offline evaluation may read existing observation data or observation-only SHM.
It must not open a control SHM, instantiate `PolicyConnector`, start
`device_starter.py`, start a policy client, send actions, clear robot faults,
or alter a config, checkpoint, calibration artifact, or recorder input. Write
results only below a newly created diagnostic directory and preserve the input
run. If a question requires changing temporal flags or validating motion, stop
and route it through `fastwam-inference-runtime` and the operator preflight
workflow instead.
