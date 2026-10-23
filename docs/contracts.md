# Runtime Contracts

These contracts let processes communicate through configuration and shared memory without importing hardware or
model implementations into every layer.

## Device and summary shared memory

`devices/base.py` defines the producer foundation. A configured device publishes payloads with a stable shape and
dtype plus nanosecond timestamps. The device key normally combines class name and `device_class_id`.

`manager/base_device_manager.py` and `manager/synchronized_device_manager.py` read per-device buffers and publish
the summary ring buffer. `consumer/base.py` is the current reader used by consumers. Layout helpers, headers, locks,
and policy output helpers live in `utils/shm_utils.py`.

Contract changes must preserve:

- configured shape and dtype;
- timestamp units and meaning;
- deterministic SHM names;
- ring-buffer index semantics;
- read-only consumer ownership (a reader must not unlink a producer's memory).

## Inference and action execution

`consumer/inference/common/policy_connector.py` orchestrates composed components that build observations, invoke a
registered policy, and write action chunks to policy SHM:

- `consumer/inference/common/observation_builder.py`;
- `consumer/inference/common/policy_runtime.py`;
- `consumer/inference/common/policy_runtime_io.py`;
- `consumer/inference/common/policy_output_writer.py`;
- `consumer/inference/common/episode_controller.py`;
- `consumer/inference/common/rollout_controller.py`;
- `consumer/inference/common/policy_visualization.py`;
- `consumer/policy_history/`.

`consumer/policy_connector.py` remains a compatibility import facade. `manager/base_policy_runner.py` consumes
policy chunks, delegates selection/alignment to `manager/utils/`, and writes robot commands. Action arrays must
retain configured target names, chunk length, per-target dimensions, dtype, observation timestamp, and any
action-index semantics used by the selected manager.

For latency-compensated chunk behavior, use
[`async-action-chunk-alignment`](../.claude/skills/async-action-chunk-alignment/SKILL.md). For FastWAM's
Server/client lifecycle, use
[`fastwam-inference-runtime`](../.claude/skills/fastwam-inference-runtime/SKILL.md); for the RoboTwin simulation
bridge, use [`robotwin-async-inference`](../.claude/skills/robotwin-async-inference/SKILL.md). Those Skills own
implementation procedure; this document does not duplicate it.

## Registries and imports

- `devices.DEVICE_CLASSES` is a lazy mapping. Importing `devices` must not import camera or robot SDKs.
- Camera and robot implementations live in `devices/cameras/` and `devices/robots/`; the old flat modules remain
  facades where present.
- `policy.POLICY_CLASSES` currently uses guarded imports and may contain `None` for unavailable optional policies.
  A key being registered does not prove its backend is installed.
- Manager and chunk-manager names are resolved through `manager.MANAGER_CLASSES` and
  `manager.utils.CHUNK_MANAGERS`.

## Incubator boundary

`incubator/control_panel/` is an optional application, not a stable core package. Core directories
`consumer/`, `devices/`, `manager/`, `policy/`, `utils/`, and `visualizers/` must not import `incubator`.
The boundary is checked by `tests/architecture/test_incubator_boundaries.py`.

The panel may consume core contracts and has its own docs and tests under `incubator/control_panel/`. Its lifecycle,
release, security, and recording behavior are not promises of the root runtime API.

## Validation boundary

Contract tests use fake data, temporary files, and dependency-gated imports. They can validate layouts, registry
references, config composition, parsing, and deterministic action flow. They cannot validate physical timing,
vendor SDK compatibility, camera synchronization, robot motion, gripper calibration, network quality, or safety
systems. Those require supervised validation on the target hardware.
