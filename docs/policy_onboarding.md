# Policy Onboarding

Policies implement model loading and prediction under `policy/`. Canonical inference launchers live in
`consumer/inference/`, while shared orchestration lives in `consumer/inference/common/`.

## Add a policy

1. Inherit from `policy/base.py`.
2. Implement the current load and prediction interface used by the intended consumer.
3. Return a mapping from configured control target to an action chunk with the agreed shape and dtype.
4. Add the class to `policy.POLICY_CLASSES` in `policy/__init__.py`.
5. Add or update a top-level config under `configs/`; keep checkpoint and server addresses configurable.
6. Add hardware-free tests using a deterministic fake model or mocked transport.

`policy/__init__.py` currently guards optional imports and may register unavailable classes as `None`. Avoid adding
heavy imports outside a guarded policy module. A future registry refactor may make policies fully lazy, but that is
not the current contract.

## Connect observations and actions

Validate the selected config across:

- `policy.obs.devices` and the summary-SHM device keys;
- model-specific observation transforms;
- `policy.controls[*]` endpoint (`device`, `device_class_id`, `port: command`), shape, and dtype; raw `shm`
  names are not accepted in policy configuration;
- `policy.chunk_length`, `policy.control_fps`, and action-executor interpolation;
- reset behavior for model state and cached action chunks.

Keep shared logic in `consumer/inference/common/` or the focused history/visualization packages. Preserve
`consumer/policy_connector.py` as the compatibility facade. Canonical model launchers are
`consumer/inference/openpi_policy_infer.py`, `consumer/inference/fastwam_policy_infer.py`, and
`consumer/inference/robotwin_policy_infer.py`. FastWAM additionally has a separate
`consumer/inference/fastwam_server.py` that owns the checkpoint and model; RoboTwin's bridge is launched through
`scripts/robotwin/launch.py`.

## Hardware-free validation

```bash
python -m pytest tests/integration/test_fake_inference_smoke.py -q
python -m pytest tests/consumer tests/policy -q
python -m pytest tests/config/test_registry_references.py -q
```

OmegaConf and Hydra are base dependencies, so their absence is an installation failure. Model-marked tests may
skip with a precise reason when an optional framework such as PyTorch is absent; pure policy/config logic must not
skip. A passing fake-policy smoke test only proves deterministic observation-to-action plumbing; it does not
validate a checkpoint, GPU, remote server, latency budget, action scaling, or robot behavior.

Install shared policy helpers with `uv pip install -r requirements/requirements-policy.txt`. Install OpenPI with
`uv pip install -e third_party/openpi`. FastWAM uses `third_party/nmx_vla/requirements.txt` plus a CUDA-matched
PyTorch build in a separate model environment. RoboTwin uses `requirements/requirements-robotwin.txt` in its dedicated
simulation environment. Register and bind each process environment through the
[`runtime-profile-management`](../.claude/skills/runtime-profile-management/SKILL.md) workflow.

For asynchronous chunk replacement use
[`async-action-chunk-alignment`](../.claude/skills/async-action-chunk-alignment/SKILL.md); for FastWAM lifecycle
use [`fastwam-inference-runtime`](../.claude/skills/fastwam-inference-runtime/SKILL.md), and for simulation use
[`robotwin-async-inference`](../.claude/skills/robotwin-async-inference/SKILL.md). Episode-control changes belong in
[`infer-rollout-episode-control`](../.claude/skills/infer-rollout-episode-control/SKILL.md).
