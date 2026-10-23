---
name: infer-rollout-episode-control
description: Use when adding or auditing infer_rollout episode_control keyboard episode switching in SuperInference configs, especially to ensure policy state, action caches, command shared memory, and reset signals do not leak across episodes.
---

# Infer Rollout Episode Control

Use this skill when adding keyboard episode switching through `policy.infer_rollout.episode_control` or auditing whether an inference config can safely move from one episode to the next.

## Minimal Config Pattern

To enable keyboard episode switching without rollout data storage, keep the rollout saver disabled and enable only the nested episode control:

```yaml
policy:
  infer_rollout:
    enabled: false
    episode_control:
      enabled: true
      success_key: 's'
      failure_key: 'f'
      discard_key: 'd'
      auto_restart: true
      home_wait_time: 6.0
      halt_settle_time: 0.5
      policy_runner_reset_timeout: 1.0
```

`PolicyConnector` reads `episode_control` even when `infer_rollout.enabled` is false. In that mode it does not start the rollout saver, and start/stop recording signals are no-ops.

## Safety Checklist

For safe episode transitions, verify these paths in the current worktree:

- `consumer/policy_connector.py` handles keyboard requests in `_handle_episode_transition`.
- Transition disables inference before reset/home and re-enables it only for `auto_restart`.
- Transition calls `_signal_policy_runner_reset()` and waits for `_wait_for_policy_runner_reset()` before the next episode.
- Transition calls `_reset_policy_episode_state()` so LingBotVA server KV cache and connector-side history state do not cross episodes.
- `manager/base_policy_runner.py` handles `policy_runner_reset_command` by clearing chunk manager state, delay queues, first-action cache, policy SHM, robot control SHMs, and action index.
- The policy runner should unlink `policy_runner_reset_command` only after those caches and SHMs are cleared, so the connector can use file disappearance as reset completion evidence.
- After reset, `BasePolicyRunner.last_reset_timestamp` rejects action chunks inferred from observations at or before the reset timestamp.

## Common Issues

- Control Panel pause/resume/reset commands are delivered through the private
  `SUPERINFERENCE_INFERENCE_CONTROL_DIR` spool. `PolicyConnector` must consume
  that spool from its policy loop and ACK after applying the state change;
  otherwise the UI remains pending even though the command file was written.
  Panel pause/reset handling must signal a halt without waiting indefinitely in
  the policy loop. Standalone reset keeps the ACK marked `reset_pending` while
  the robot is moving Home, and publishes `inference_paused: false` only after
  the robot reset signal is consumed. Runtime readiness must therefore exempt a
  healthy motion device for the bounded Home interval both before ACK
  (`pending_command`) and after ACK (`reset_pending`); checking only the command
  spool can safety-stop the runtime while synchronous Home is still running.

- Control Panel and direct reset use the same fixed Home contract. The panel
  must not generate a random-offset target or attach `home_target` to reset,
  accept, or discard commands. With no explicit payload, `PolicyConnector`
  writes Home index `0`, and `RizonRobot` resolves that index from the selected
  runtime config. The JSON parser remains only for legacy command compatibility.

- If the robot reaches home and then immediately returns to the previous episode's final pose, add a post-home reset barrier:
  - reset the policy runner before homing to stop old actions,
  - home the robot,
  - reset the policy runner again after the home wait,
  - reset policy-side episode state after that post-home reset,
  - only then switch teleoperator/control flag back to policy mode and re-enable inference.
- RizonRobot should clear its own control SHM both before and after `_go_home_with_index()`. The second clear covers commands that arrive while the robot is moving home.
- RizonRobot normally consumes its control SHM through a read-only `DataChannel`.
  `_clear_control_shm()` must open a separate temporary read-write connection;
  trying to mutate `control_shm.buf` directly raises `cannot modify read-only
  memory`. Keep the long-lived robot connection read-only.
- `DataChannel.close()` must release its retained base `memoryview` before
  closing `multiprocessing.shared_memory.SharedMemory`. Arrays returned from
  control reads must own copied bytes, otherwise shutdown can fail with
  `cannot close exported pointers exist`.
- After merging chunk-manager changes, preserve both the current `drop_first_actions` option and the legacy
  `skip_first_n_steps` alias used by older RTC configs. Run `tests/manager/utils/test_action_manager.py` to catch
  duplicated conflict-resolution lines and incorrect original action indices.

## Validation

Run lightweight validation after edits:

```bash
python -m py_compile consumer/policy_connector.py manager/base_policy_runner.py
python - <<'PY'
import ast
from pathlib import Path
for path in ["consumer/policy_connector.py", "manager/base_policy_runner.py"]:
    ast.parse(Path(path).read_text())
print("episode-control code parses")
PY
```

If `uv` or project dependencies are unavailable on the machine, use stdlib checks plus direct inspection of the YAML block.
