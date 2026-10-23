---
name: fastwam-rtc-guidance
description: Maintain FastWAM RTC prefix guidance, temporal alignment, and the websocket payload contract.
---

# FastWAM RTC Guidance

The client caches the previous absolute action chunk in executor command order.
At inference start, `action_index` is the last dispatched original command; the
prefix starts at `action_index + 1`, is rebased into the current robot-relative
frame, and is normalized by the same action normalizer as model output.

For action horizon `H`, control rate `F_c`, and inference rate `F_i`, configured
overlap is approximately `O = H - round(F_c / F_i)`. A fixed delay `D` and
exclusive guide end `G` must satisfy `D < G <= O`; if the complete guide no
longer fits, fall back to ordinary inference. Keep padded prefix steps at zero
weight. The executor drops elapsed commands separately, so “guided” does not
mean “dispatched”.

`rtc_reverse_temporal_guidance` must agree with the Server's
`reverse_temporal_chunks` metadata. For the maintained chronological output
layout, use both flags as `false` and guide model indices `[0, G)`. Do not infer
the layout from a training branch name. The Server and client reject mismatched
RTC metadata at connection time.

The supported schedules are `flat`, `exponential`, and `minimum_jerk`. Preserve
the default exponential behavior for old payloads. `rtc_use_vjp` selects exact
latent-only Jacobian VJP versus the faster identity-Jacobian approximation;
report the active value in metadata and responses, and benchmark before turning
on exact VJP for a robot rollout.

When changing timing or manager behavior, inspect the action-index and chunk
switch traces together. `rtc_blend_manager` is an optional manager; it is not a
reason to duplicate hardware or camera configs. Keep `policy.fps`,
`control_fps`, chunk length, and endpoint settings in one canonical profile
with thin sync/RTC overlays.

Useful checks:

```bash
python -m pytest tests/config -q
python -m pytest incubator/control_panel/tests/test_fastwam_workflow_lifecycle.py -q
```

Hardware-free tests do not establish robot safety or latency. Re-measure
inference-start-to-dispatch delay after changing the model, Torch, endpoint,
or action executor.
