## 变更摘要

<!-- 说明改了什么、为什么现在要改。存在 issue、设计说明或运行记录时请链接。 -->

- 背景或问题：
- 方案与原因：
- 风险与回滚方式：

## 变更类型

- [ ] 缺陷修复
- [ ] 新功能、适配器或运行时能力
- [ ] 重构（预期不改变行为）
- [ ] 配置、依赖或运行时 profile 治理
- [ ] SHM、数据、动作或轨迹契约
- [ ] CI/CD、子模块或仓库维护
- [ ] 仅文档

## 范围与责任

- 影响范围：`devices/` / `manager/` / `consumer/` / `policy/` / `configs/` / `utils/` / `visualizers/` / `incubator/` / `third_party/`
- 原生 Hydra 配置名（如涉及）：
- 影响的运行角色或依赖 profile（如涉及）：
- 负责人：
- 建议审阅者：
- [ ] 这是有明确负责人的短期功能分支。
- [ ] 对新的模块边界、广泛重构、公开 API 或 schema 迁移，已链接简短设计说明。
- [ ] 变更大小适合审阅；纯格式化或大规模删除已拆分为独立 MR。

## 兼容性与风险

- [ ] 现有导入路径、兼容门面、配置键、注册表名称和 CLI 行为保持兼容，或已提供迁移说明。
- [ ] 已检查 SHM/数据/轨迹/动作变更的 shape、dtype、单位、时间戳域与语义、序列化和版本兼容性。
- [ ] 涉及动作执行或 RoboTwin 时序时，已检查执行频率、deadline/lateness、catch-up、chunk 对齐/丢弃/平滑和原始动作索引。
- [ ] 未引入隐式全局状态、泄漏的信号/缓存、主机专属路径、IP、设备名或凭据。
- [ ] 可选模型、硬件和厂商依赖仍为延迟导入，并属于正确的依赖 profile；核心代码不导入 `incubator/`。
- [ ] 涉及硬件时，已说明 reset、cleanup、reconnect、halt/home 以及异常路径的安全性和必要的人工验证。
- [ ] 涉及运行时或推理时，使用 `scripts/diagnostics/runtime_profiles.py` 管理 profile，并记录 `register`/`bind`/`doctor` 及必要时的 `verify` 结果；未提交 `.superinference/local/` 或硬编码解释器路径。
- [ ] 涉及配置移动或新增时，已同步原生配置名、默认值、入口点、测试、profile 绑定和相关 Skill。
- [ ] 涉及子模块时，`.gitmodules` 与子模块清单保持一致；gitlink/pin 仅在人工审阅的 MR 中变更，并同步更新绑定提交和日期。

## 验证与证据

<!-- 填写实际执行的命令、结果或 CI Job 链接。未执行的项目请写明原因、风险和后续验证方式。 -->

```text
<命令>
<结果或 CI Job 链接>
```

- [ ] 已运行或检查对应的 `config_contracts`、`device_contracts`、`shm_contracts`、`entrypoint_help` Job，以及与变更匹配的聚焦测试。
- [ ] 已运行根硬件无关测试：`.venv/bin/python -m pytest tests -q -m "not hardware and not manual"`。
- [ ] 已运行根 Ruff 契约检查：`.venv/bin/python -m ruff check --select E9,F63,F7,F82 --exclude third_party .`。
- [ ] 已对变更文件运行 `pre-commit`，并已执行 `git diff --check`。
- [ ] 如查看了全树 Ruff/格式报告，已将其作为非阻断的历史格式债务处理，且未为无关变更批量格式化仓库或 `third_party/`。
- [ ] 已对变更入口点执行安全的 `--help`、配置规划或 `--dry-run` 检查；未启动机器人、相机或模型服务。
- [ ] 模型、Control Panel、硬件和第三方验证已在对应环境中执行并附上证据，或已明确说明为何不适用。
- [ ] 已区分必需根 CI 与允许失败/手动的可选 Job；未将后者视为根 CI 通过的替代品。

## 配置、依赖与运行环境

- 配置文件或原生配置名变更：
- 依赖 profile、SDK、CUDA wheel、checkpoint 或子模块变更：
- 新增或变更的环境变量、端口、信号或外部服务：
- 运行时 profile 的注册、绑定、`doctor`/`verify` 状态（如涉及）：
- 子模块变更类型（无 / README 状态刷新 / 人工 pin 升级）：
- [ ] 未把硬件 SDK、CUDA wheel、checkpoint 或厂商包加入默认根 CI profile。
- [ ] 未在受 Git 跟踪的文件中加入主机专属 Python 绝对路径、环境名或本地标定数据。
- [ ] 状态刷新 MR 仅更新 README 中的最新远端状态；人工 pin 升级同步 gitlink、`Bound commit` 和 `Bound date`。
- [ ] 涉及定时子模块自动化时，未输出凭据，并继续使用 Group Bot 凭据而非 `CI_JOB_TOKEN` 访问受管子模块。

## 文档与交接

- [ ] 当工作流、不变量、排障规则或架构边界发生变化时，已更新对应 `.claude/skills/` 内容。
- [ ] 已更新受影响的用户文档、配置示例和迁移/回滚说明（如适用）。
- [ ] 延后工作已记录负责人和完成时间（如有）。

## 审阅者检查

- [ ] 变更范围清楚，且属于 `main`；实验性垂直能力保持在 `incubator/` 或明确的可选边界中。
- [ ] 必需 CI Job 通过；可选 Job 已按其依赖和硬件边界正确解读。
- [ ] 契约、安全性、迁移/回滚和测试证据与变更风险相符。
- [ ] 无未解决的审阅讨论。
