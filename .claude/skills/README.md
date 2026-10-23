# SuperInference Skills

This directory is the repository's operational knowledge base. Each Skill contains
the workflow and invariants for one kind of task; this file is only the index and
routing guide. Read the linked `SKILL.md` before changing the corresponding code or
running a specialized hardware/model workflow.

## Skill Map

| Skill | Use it for | Main ownership |
| --- | --- | --- |
| [`superinference-operator`](superinference-operator/SKILL.md) | Day-to-day operation requests, command selection, safety preflight, runtime lifecycle, robot probes, config/SHM checks, camera/data inspection, and failure routing | Repository-wide operator entrypoint |
| [`runtime-profile-management`](runtime-profile-management/SKILL.md) | Register, verify, bind, resolve, and repair the machine-local Python profiles used by each workflow role | `utils/runtime_profiles.py`, `scripts/diagnostics/runtime_profiles.py`, `.superinference/local/` |
| [`workstation-inference-bootstrap`](workstation-inference-bootstrap/SKILL.md) | Fresh-machine OpenPI inference setup, Flexiv firmware-to-RDK selection, submodule access, source environment creation, and installation blockers | OpenPI workstation bootstrap |
| [`robotwin-async-inference`](robotwin-async-inference/SKILL.md) | RoboTwin direct-drive simulation, ACT and XPolicyLab policies, joint14/eef16 contracts, launch, and timing traces | `configs/policy_infer/robotwin/`, `integrations/robotwin/`, `scripts/robotwin/` |
| [`async-action-chunk-alignment`](async-action-chunk-alignment/SKILL.md) | Asynchronous action replacement, latency-based dropping, interpolation/resampling, smoothing, and original action-index reporting | `manager/utils/`, `manager/base_policy_runner.py`, ActionExecutor traces |
| [`infer-rollout-recording`](infer-rollout-recording/SKILL.md) | NEDF3 rollout recording, ActionStateXYZ, TimestampEvent, action trace semantics, and rollout visualization | `consumer/io/`, logger utilities, recording contracts |
| [`infer-rollout-episode-control`](infer-rollout-episode-control/SKILL.md) | Keyboard episode switching, reset barriers, homing, policy-state clearing, and action-cache isolation | `consumer/inference/common/`, `manager/base_policy_runner.py` |
| [`lingbot-va-rtc`](lingbot-va-rtc/SKILL.md) | LingBotVA real-time control, latency-to-step estimation, prefix guidance, server inpainting, and RTC chunk smoothing | `policy/lingbot_va_policy.py`, `consumer/lingbot_va_server.py`, action managers |
| [`fastwam-inference-runtime`](fastwam-inference-runtime/SKILL.md) | FastWAM, FastWAM-memory, and FasterWAM Server/client lifecycle, checkpoint contracts, health/readiness, and model-runtime troubleshooting | `consumer/inference/fastwam_server.py`, `policy/fastwam_policy.py`, `consumer/inference/fastwam_model_registry.py` |
| [`fastwam-rtc-guidance`](fastwam-rtc-guidance/SKILL.md) | FastWAM RTC prefix timing, temporal layout, guidance schedules, and server/client contract checks | `policy/fastwam_policy.py`, FastWAM runtime payloads, `manager/utils/rtc_blend_manager.py` |
| [`evaluate-fastwam-reverse-offline`](evaluate-fastwam-reverse-offline/SKILL.md) | Read-only reverse-temporal FastWAM evaluation from recorded observations, dynamic action-order audits, and trajectory diagnostics | `consumer/inference/fastwam_server.py`, `utils/observation_recorder.py`, adapted tools under `scripts/analysis/` |
| [`superoffline-comparison`](superoffline-comparison/SKILL.md) | Exactly-two-model real-robot offline comparison, explicit FastWAM/OpenPi/LingBotVA contracts, browser UI, trusted recorder inputs, and verified-profile evaluator dispatch | `utils/superoffline_registry.py`, `scripts/analysis/superoffline.py`, evaluators |
| [`fastwam-video-visualization`](fastwam-video-visualization/SKILL.md) | Optional FastWAM predicted-video transport and dashboard artifacts | `visualizers/fastwam_video_prediction.py`, FastWAM server video path |
| [`lktech-gripper-control`](lktech-gripper-control/SKILL.md) | LKTech calibration, encoder wrap handling, width semantics, clipping, homing, and direct serial ownership | `devices/robots/rizon/gripper.py`, `scripts/calibration/`, `incubator/control_panel/gripper_control.py` |
| [`superinference-ci`](superinference-ci/SKILL.md) | Repository boundaries, test selection, GitLab CI jobs, review checklist, and merge readiness | `tests/`, `.gitlab/`, package boundaries |
| [`submodule-maintenance`](submodule-maintenance/SKILL.md) | Configure remote branches, maintain README submodule status, and operate the scheduled GitLab status-refresh MR workflow | `.gitmodules`, `README.md`, `.gitlab-ci.yml` |

## Routing Order

Start with `superinference-operator` for a user-facing operation. It selects the
narrowest specialized Skill when the request crosses into implementation or a deep
workflow.

Examples:

- “读取当前机器人关节角” -> `devices/flexiv_adapter.py --task read_states`;
  do not start `device_starter.py` for a one-shot read.
- “启动某个配置的设备” -> `device_starter.py --config-name <name>` after config
  and hardware preflight.
- “新机器 clone 后配置 OpenPI 推理环境” ->
  `workstation-inference-bootstrap`; resolve the robot software version before any
  Flexiv RDK installation.
- “查看、绑定或修复某个配置的 Python 环境” -> `runtime-profile-management`.
- “启动或修改 RoboTwin ACT/Pi_05/FastWAM 仿真” -> `robotwin-async-inference`.
- “Bind rollout USB grippers and cameras” -> `scripts/diagnostics/rollout_device_preflight.py`
  with the selected config; it performs operator-guided device binding and
  persists the resulting YAML/calibration mapping.
- “修改 action 对齐或 drop_first_actions” -> `async-action-chunk-alignment`.
- “分析 rollout 字段或时间线” -> `infer-rollout-recording`.
- “增加 episode 成功/失败切换” -> `infer-rollout-episode-control`.
- “修改 LingBotVA RTC” -> `lingbot-va-rtc`.
- “启动或排查 FastWAM、FastWAM-memory 或 FasterWAM Server/client” -> `fastwam-inference-runtime`.
- “修改 FastWAM RTC prefix 或时序对齐” -> `fastwam-rtc-guidance`.
- “离线检查 FastWAM 反向时序或动作曲线” -> `evaluate-fastwam-reverse-offline`.
- “比较两个真实机器人 FastWAM/OpenPi/LingBotVA 模型的 recorder 动作” ->
  `superoffline-comparison`.
- “调试 FastWAM 预测视频或 dashboard” -> `fastwam-video-visualization`.
- “校准 LKTech 夹爪” -> `lktech-gripper-control`.
- “修改模块边界、测试或 CI” -> `superinference-ci`.
- “更新或配置 third_party submodule” -> `submodule-maintenance`.

When two Skills apply, load the router first, then the Skill that owns the changed
contract. For example, an RTC rollout trace change needs both
`lingbot-va-rtc` and `infer-rollout-recording`; an episode reset change needs
`infer-rollout-episode-control` before editing reset-related code.

## Hardware Boundary

The Skills distinguish four risk levels:

- **Read-only:** state probes, config inspection, offline data inspection, and
  hardware-free tests.
- **Energizing:** clearing faults or enabling a robot.
- **Moving:** homing, direct motion, policy control, calibration sweeps, and replay.
- **Artifact-changing:** overwriting calibration, rollout, or dataset files.

Only the first level is automatically safe to execute from an explicit user request.
For the other levels, identify the exact target and command, check the environment,
limits, ownership, and emergency-stop path, then obtain confirmation immediately
before the operation unless the user already authorized that exact action.

## Maintenance Rules

- Keep detailed, fast-changing procedures in the owning Skill rather than duplicating
  them here or in a new README.
- Update the relevant Skill when a command, config contract, safety invariant, or
  troubleshooting workaround changes.
- Keep tool commands in `useful_tools_commands.md`, inference startup commands in
  the root `*_inference_commands.md` files, and prompts in `useful_prompts.md`;
  use Skills for operational decisions and implementation-specific knowledge.
- Validate a changed Skill with the Codex `skill-creator` quick validator.
