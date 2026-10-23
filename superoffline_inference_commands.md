# SuperOffline 离线对比启动命令

**SuperOffline 通过 `runtime_profiles.py` 启动；本机环境由 AI 完成注册和绑定后，无需手工指定或激活 Python 环境。**

所有命令从项目根目录执行。SuperOffline 只读取已完成的 observation recorder 记录，
串行比较 2–4 个真机模型；不启动设备、不连接控制 SHM，也不下发机器人动作。

## Python 环境检查（可选）

```bash
python3 scripts/diagnostics/runtime_profiles.py doctor --role superoffline
```

若页面或某个 evaluator 报 profile 未绑定、未验证或依赖缺失，请让 AI 根据
`.claude/skills/runtime-profile-management/SKILL.md` 和
`.claude/skills/superoffline-comparison/SKILL.md` 完成修复，不要猜测 Python 路径或环境名。

## 启动页面

```bash
python3 scripts/diagnostics/runtime_profiles.py run \
  --role superoffline -- \
  scripts/analysis/superoffline.py --host 127.0.0.1 --port 8767
```

浏览器打开 `http://127.0.0.1:8767`。可选健康检查：

```bash
curl --fail http://127.0.0.1:8767/api/healthz
```

## 页面操作

1. 在 `Observation Input` 选择已完成的录制。
2. 在 `Checkpoints` 填写 FastWAM/OpenPI root，再为每一行选择发现到的 checkpoint；也可直接输入路径。
3. 选择 2–4 个不同模型，设置样本范围和输出模式。
4. 先点击 `Validate input`，通过后点击 `Run inference`。

root 只用于发现 checkpoint。列表显示 `gift_v1`、`iphone` 等实验名，实际提交的仍是
完整 checkpoint 路径。OpenPI 离线比较固定使用 `pi_config_name=default`，并自动绑定
checkpoint 中现有的 `assets/*/norm_stats.json`。

## 输入与输出

- 录制默认位于 `logs/offline_observations/<变体>/run_<时间戳>/`。
- 录制必须包含已完成的 `manifest.json`、`frames.jsonl`、`inference.jsonl` 和引用的 NPZ 文件。
- 比较结果写入新的 `logs/superoffline/<job-id>/`。
- 同一任务一次只加载一个 checkpoint；其他任务排队，避免多个模型同时占用 GPU。

完整契约和排错见
[`.claude/skills/superoffline-comparison/SKILL.md`](.claude/skills/superoffline-comparison/SKILL.md)。
