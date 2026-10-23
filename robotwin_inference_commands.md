# RoboTwin 推理启动命令

RoboTwin 是仿真流程，不运行真机 `rollout_device_preflight.py`。`scripts/robotwin/launch.py` 会先按所选
配置检查仿真依赖和资产，再统一启动 DeviceStarter 与 policy connector。

可选配置：

- `policy_infer/robotwin/config_infer_robotwin_replay_joint14`
- `policy_infer/robotwin/config_infer_robotwin_act_joint14`
- `policy_infer/robotwin/config_infer_robotwin_pi05_joint14`
- `policy_infer/robotwin/config_infer_robotwin_fastwam_joint14`
- `policy_infer/robotwin/config_infer_robotwin_eef16`，需要额外配置 EEF resolver 和动作文件

## 可选环境检查

`doctor` 只检查所选配置的 profile 绑定、解释器、依赖导入和环境指纹，不启动仿真或模型：

```bash
ROBOTWIN_CONFIG=policy_infer/robotwin/config_infer_robotwin_replay_joint14
python3 scripts/diagnostics/runtime_profiles.py doctor \
  --config-name "$ROBOTWIN_CONFIG"
```

## Replay、ACT 或 EEF16

```bash
ROBOTWIN_CONFIG=policy_infer/robotwin/config_infer_robotwin_replay_joint14
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$ROBOTWIN_CONFIG" --role launcher -- \
  scripts/robotwin/launch.py --config-name "$ROBOTWIN_CONFIG" --gui
```

将变量替换为 ACT 或 EEF16 配置即可切换；`--gui` 可省略。

## Pi_05 或 FastWAM

对应的 `policy-server` profile 和 checkpoint 准备好后，可让 launcher 一并启动 XPolicyLab server：

```bash
ROBOTWIN_CONFIG=policy_infer/robotwin/config_infer_robotwin_pi05_joint14
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$ROBOTWIN_CONFIG" --role launcher -- \
  scripts/robotwin/launch.py --config-name "$ROBOTWIN_CONFIG" \
  --start-xpolicylab-server --gui
```

FastWAM 仿真只需把变量改成 `policy_infer/robotwin/config_infer_robotwin_fastwam_joint14`。如果模型服务已在配置的
host/port 上运行，去掉 `--start-xpolicylab-server`。这里两次配置名的含义也相同：外层选择本机运行环境，
内层选择 launcher 实际组合的 Hydra 配置；只核对命令时可把 `run` 改成 `command --json`。
