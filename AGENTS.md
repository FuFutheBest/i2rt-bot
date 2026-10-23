# Repository Guidelines

## Project Structure

- Runtime packages: `devices/`, `manager/`, `consumer/`, `policy/`, `utils/`, and `visualizers/`.
- Runtime entrypoints: `device_starter.py`, `windows_device_start.py`, and model-specific launchers under
  `consumer/inference/`.
- Hydra configuration source: `configs/`. A native config name is its path below `configs/` without `.yaml`, for
  example `policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi` or
  `policy_infer/robotwin/config_infer_robotwin_replay_joint14`.
- Operator and maintenance tools: `scripts/`, grouped by purpose.
- Optional applications and experimental integrations: `incubator/`.
- Vendor and model integrations: `third_party/`, usually as Git submodules.
- Hardware-free tests: `tests/`. Control Panel tests live separately under `incubator/control_panel/tests/`.

## Setup, Test, And Lint

The root CI profile uses Python 3.10. Model, simulation, recording, and vendor SDK processes may require separate
interpreters; manage those through runtime profiles rather than forcing them into one environment.

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements/requirements-dev.txt

.venv/bin/python -m pytest tests -q -m "not hardware and not manual"
.venv/bin/python -m ruff check --select E9,F63,F7,F82 --exclude third_party .
export PATH="$PWD/.venv/bin:$PATH"
{
  git diff --name-only --diff-filter=ACMR -z
  git ls-files --others --exclude-standard -z
} | xargs -0 -r python -m pre_commit run --files
git diff --check
```

Use narrower pytest paths while developing, then run the full hardware-free root suite. Hardware, model, manual,
incubator, and third-party validation are separate because they have different dependencies and safety boundaries.
The repository still has historical format debt, so do not bulk-format the full tree for an unrelated change.

## Python 运行环境与推理治理（强制）

以下规则适用于 Python 环境安装或修复、推理代码或命令生成，以及实际推理执行任务：

1. **先读取运行环境 Skill**
   - 操作前必须读取 `.claude/skills/runtime-profile-management/SKILL.md`，并同时读取对应推理子系统的
     skill。
   - 使用 `scripts/diagnostics/runtime_profiles.py` 作为环境注册、解析、检查和执行的统一入口。

2. **禁止硬编码或猜测环境**
   - 不得在受 Git 跟踪的代码、配置或文档中写入主机专属 Python 绝对路径。
   - 不得猜测 Conda、uv 或 venv 环境名，也不得新增 `conda run -n <固定名称>`、
     `conda activate <固定名称>` 等依赖本机命名的调用。
   - 主机路径只能作为本地 `register --python` 的输入，或作为明确的一次性兼容覆盖；不得成为仓库
     默认值。

3. **配置或修复环境后必须标定**
   - 安装完成后必须依次执行 `register`、按配置执行 `bind`，然后执行 `doctor`。
   - 环境被重建、解释器改变或核心依赖改变后，必须重新执行 `verify`。
   - `--skip-verify` 仅允许用于修复中的临时状态；未验证或已失效的 profile 不得用于最终命令。

4. **生成推理命令时必须读取本地标定**
   - 用户只要求推理代码或命令时，必须执行 `command --json` 获取解析后的解释器、参数和受管环境，
     再据此回答。
   - 不得直接解析 `.superinference/local/runtime_profiles.json`，不得手工拼接猜测出的 Python 路径。
   - 仅生成命令时不得启动硬件、加载模型或实际执行推理。

5. **实际推理必须先检查再启动**
   - 用户明确要求执行推理时，必须先运行对应配置和角色的 `doctor`，通过后使用 `run --role <role>`
     启动。
   - profile 缺失、失效、依赖导入失败或指纹变化时，必须停止并修复或报告，不得静默回退到名称
     相似的环境。

6. **本机标定不得进入版本控制**
   - `.superinference/local/` 必须保持 Git 忽略，不得提交、跨主机复制或写入凭据。
   - 仓库维护角色、依赖契约和启动逻辑；每台主机独立生成自己的本地标定。

## Coding Style

- Follow the existing package boundaries and compatibility facades; do not move implementation back into old flat
  modules.
- Ruff is the source of truth for formatting and linting. Keep imports sorted and avoid unrelated format churn.
- Prefer descriptive modules and explicit contracts for SHM names, shapes, dtypes, units, and timestamp domains.
- Tests use `test_<feature>.py`. Mark only genuine physical-device tests as `hardware` and operator-only tests as
  `manual`.
- Preserve a dirty worktree. Never revert user changes or submodule state that is unrelated to the task.

## Commit And Merge Requests

- Use short imperative commit subjects and keep each commit focused on one logical change.
- MR descriptions should state the affected runtime/config contract, validation performed, hardware limitations,
  and any submodule change.
- Keep the blocking merge path hardware-free. Run model, hardware, incubator, and third-party jobs only in their
  matching environments.

## Documentation Management

Skills under `.claude/skills/` are the maintained knowledge base for workflows, architecture decisions,
troubleshooting, and safety invariants.

- `README.md`: concise project overview and navigation.
- `useful_tools_commands.md`: concise Flexiv and rollout-device utility reference.
- `*_inference_commands.md`: concise, subsystem-specific inference startup references.
- `useful_prompts.md`: minimal prompt templates for environment setup and launch-command generation.
- `.claude/skills/`: detailed subsystem knowledge and operating procedures.
- `docs/`: only stable contracts and onboarding material that benefits human developers.
- Code comments: implementation details that are easiest to maintain beside the code.

After a meaningful feature, bug fix, performance change, or architecture change, update the owning skill. Do not
create task reports, static test inventories, duplicated command dictionaries, or per-directory README files unless
they are required by packaging or an external tool.
