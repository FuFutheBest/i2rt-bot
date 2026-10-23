# SuperInference 常用 Prompt

下面的模板可以直接复制，只需替换配置文件；生成命令时再选择运行方式。

## 配置推理环境

```text
请根据配置文件 `configs/<config>.yaml` 配置当前机器的 SuperInference 推理环境，并注册到
`.superinference/local/runtime_profiles.json`。

请先读取 AGENTS.md、runtime-profile-management 和该配置对应的 skill。根据配置解析所需角色和依赖，
完成 register、bind、verify、doctor。不要猜环境名或 Python 路径，不要把本机路径写入受 Git 跟踪文件，
不要启动硬件、模型或仿真。配置包含 `policy.infer_rollout` 时，即使该功能默认关闭，也必须注册、绑定并
校验 `recording` 角色及其录制依赖。配置属于 FastWAM、FastWAM RTC、FastWAM Memory 或 FasterWAM 时，
在 `fastwam-server` profile 中运行 `scripts/diagnostics/fastwam_model_cache.py prepare --config-name <config>`。
让工具自动检查标准 Hub 缓存、常见模型目录和有限深度的当前用户目录，为完整的 Wan 模型创建
`.superinference/local/model-cache` 软链接；不要要求用户手工提供模型路径，不要移动或复制模型文件。
没有发现完整本地模型时，报告后保留启动时下载行为。
```

## 配置 SuperOffline 环境

```text
请为当前机器配置 SuperOffline 离线对比环境，并注册到
`.superinference/local/runtime_profiles.json`。需要支持以下模型配置：

- `configs/<model-config-1>.yaml`
- `configs/<model-config-2>.yaml`

请先读取 AGENTS.md、runtime-profile-management、superoffline-comparison 和所选模型对应的 skill。
为页面服务注册并绑定默认 `superoffline` 角色，再根据每个原生配置注册并绑定它的
evaluator 角色（例如 `fastwam-server`、`policy` 或 `policy-server`），完成 register、bind、
verify、doctor。不要假设页面和模型共用同一个 Python，不要猜环境名或 Python 路径，
不要把本机路径写入受 Git 跟踪文件。页面 profile 必须能导入 `numpy`、`plotly` 和
`scripts.analysis.superoffline`；启动后检查 `/plotly.min.js` 返回 200，确保交互式 3D/2D 轨迹可用。
不要启动硬件、模型或仿真，也不要修改 checkpoint。
所选 evaluator 属于 FastWAM、FastWAM Memory 或 FasterWAM 时，在对应 `fastwam-server` profile 中运行
`scripts/diagnostics/fastwam_model_cache.py prepare --config-name <config>`，自动发现并绑定本机已有 Wan 模型；
不要要求用户手工输入模型路径。
```

## 给出运行命令

```text
请根据配置文件 `configs/<config>.yaml` 和本机已绑定的 runtime profiles，给出完整运行命令。

运行方式：[直接启动 / Control Panel / 两者都要]

请先读取 AGENTS.md、runtime-profile-management 和该配置对应的 skill，并使用
`runtime_profiles.py command --json` 解析实际解释器、参数、工作目录和环境变量。说明启动顺序、就绪条件
和停止顺序，但不要实际启动硬件、模型或仿真。
```
