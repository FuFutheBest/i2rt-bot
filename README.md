# SuperInference

SuperInference 是一个配置驱动的机器人运行时，支持真机数据流、异步策略推理、动作块执行、
rollout 录制和 RoboTwin 仿真。设备进程发布共享内存数据流，设备管理器构建同步观测，
策略消费者发布动作块，策略运行器将命令写入机器人控制共享内存。

## 快速开始

- [常用工具命令](useful_tools_commands.md)：Flexiv 检查、运动控制和 rollout 设备绑定。
- [OpenPI 推理](openpi_inference_commands.md)：真机预检和进程启动。
- [FastWAM 与 RTC 推理](fastwam_inference_commands.md)：真机预检和服务端/客户端启动。
- [FastWAM memory 推理](fastwam_memory_inference_commands.md)：memory 模型预检和服务端/客户端启动。
- [FasterWAM 推理](fasterwam_inference_commands.md)：单次模型预检和服务端/客户端启动。
- [SuperOffline 离线对比](superoffline_inference_commands.md)：两个模型基于真机录制数据的对比。
- [RoboTwin 推理](robotwin_inference_commands.md)：仿真预检和启动命令。
- [常用提示词](useful_prompts.md)：用于环境配置和启动命令生成的可直接使用模板。
- [仓库规则](AGENTS.md)：强制执行的运行环境、安全、编辑、测试和文档规则。
- [Skill 索引](.claude/skills/README.md)：各子系统的详细流程和故障排查知识。

稳定的开发者文档仅包括：

- [配置说明](docs/configs.md)
- [运行时契约](docs/contracts.md)
- [设备接入](docs/device_onboarding.md)
- [策略接入](docs/policy_onboarding.md)
- [测试说明](docs/testing.md)

## 运行模型

```text
设备生产者 -> 单设备 SHM -> 设备管理器 -> 汇总 SHM
                                          |
                                          v
策略适配器 <- 策略消费者 ------------> 策略 SHM
                                          |
                                          v
                                  策略运行器 / 动作执行器
                                          |
                                          v
                                      机器人控制 SHM
```

核心运行时代码位于 `devices/`、`manager/`、`consumer/`、`policy/` 和 `utils/`。
`incubator/control_panel/` 下的控制面板使用相同的 Hydra 配置和运行时契约；它是可选应用，
并非另一套设备栈。

## Start The I2RT Data Collection Web UI

The I2RT DM3 web UI provides the login, task and collection controls, NEDF3 episode recording, discard, and upload
workflow. From the repository root, start it with the project environment:

```bash
source .venv/bin/activate
python -m incubator.nedf3.I2RT_DM3.web --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080> in a browser. The collection page includes a dual-gripper calibration action; it is
enabled only after DM3 login and while the device and NEDF3 saver processes are stopped. The server does not start
devices until the device-start action is requested in the UI. Press `Ctrl+C` in the server terminal to stop the web UI.

For remote access, keep the server bound to loopback and create an SSH tunnel from the client machine:

```bash
ssh -L 8080:127.0.0.1:8080 <user>@<host>
```

## 配置与运行环境

Hydra 只从 `configs/` 读取配置。原生配置名是相对于该目录的路径，并去掉 `.yaml` 后缀：

```text
configs/policy_infer/openpi/example.yaml   -> policy_infer/openpi/example
configs/policy_infer/fastwam/example.yaml  -> policy_infer/fastwam/example
configs/policy_infer/robotwin/example.yaml -> policy_infer/robotwin/example
```

### 配置注意事项

`device_class_id` 是整个 `devices` 列表的零起始下标，包含禁用设备；所有设备类共用
`0, 1, 2, ...` 序列。Python 构造参数、端点和 SHM header 使用同一字段，旧配置字段
`device_id` 会报错。物理相机编号不参与 SHM 命名：

```yaml
# 顶层推理 YAML：preflight 可更新这两个物理入口。
camera:
  left_capture_id: 2   # 打开 /dev/video2，也可填写稳定的 /dev/v4l/by-id/... 链接
  right_capture_id: 4  # 打开 /dev/video4

# 对应 devices YAML：preflight 不得修改 device_class_id。
devices:
  - class: WebCameraOpenPi
    device_class_id: 0
    capture_id: ${camera.left_capture_id}   # SHM: WebCameraOpenPi_0_data
    # state_feedback: ...
  - class: WebCameraOpenPi
    device_class_id: 1
    capture_id: ${camera.right_capture_id}  # SHM: WebCameraOpenPi_1_data
    # state_feedback: ...
  - class: RizonRobot
    device_class_id: 2                     # SHM: RizonRobot_2_data
    # state_feedback / control / robot_sn: ...

# 顶层推理 YAML：0 是左相机的逻辑 ID，与 /dev/video0 无关。
device_manager:
  device_manager:
    master_device_class_id: 0
policy:
  obs:
    devices: [WebCameraOpenPi_0, WebCameraOpenPi_1, RizonRobot_2]
  controls:
    - endpoint: {device: RizonRobot, device_class_id: 2, port: command}
      shape: [16]
      dtype: float64
  observation_recorder:
    superoffline_reviewed_source_profile: rizon-umi3-fisheye-openpi-rgb.v2
```

机器人若位于列表首项，ID 就是 `0`；添加前置设备后需同步编号和端点引用。
preflight 可更新 capture ID、机器人序列号、串口及标定参数，但保留逻辑 ID 和列表顺序。
禁用设备不重新编号；同步主设备必须启用。WebCamera 必须显式配置 `capture_id`，
不会回退到逻辑 ID。更改配置后须停止旧设备进程，再用同一新配置重启整条链路。

`control.source_selection` 用设备端点声明机器人命令来源；省略时默认使用机器人自身的
`command` 流。策略与遥操作共享控制时可配置为：

```yaml
control:
  source_selection:
    mode: dual_source
    sources:
      policy: {device: self, port: command}
      teleoperation: {device: KeyboardTeleoperator, device_class_id: 3, port: state}
    dual_source_selector:
      endpoint: {device: KeyboardTeleoperator, device_class_id: 3, port: control_flag}
```

完整 schema 和校验规则见[配置说明](docs/configs.md)。

同一工作流的启动器、设备、策略客户端、模型服务、录制和转换进程可能需要不同的 Python 环境。
使用 `scripts/diagnostics/runtime_profiles.py` 注册并绑定这些解释器；不得提交本机专属的 Python 路径，
也不得在机器之间复制 `.superinference/local/`。

生成启动命令前先执行 `command --json`。启动模型或硬件进程前先执行 `doctor`，通过后再使用
`run --role <role>` 运行。具体流程见[运行环境 Skill](.claude/skills/runtime-profile-management/SKILL.md)。

## 安全边界

`device_starter.py`、策略消费者、直接运动工具、标定脚本、回放脚本和部分预检操作可能连接或移动硬件。
单元测试和配置组合不能证明机器人固件、SDK 兼容性、标定、时序、工作空间限制、断连处理或急停链路正确。

应将 `device_starter.py` 视为启动器，而不是无害的配置校验器。检查配置时使用无硬件配置测试或启动规划器；
执行任何硬件操作前，必须明确具体配置、机器人序列号、目标、单位和预期操作。

## 开发与测试

根目录 CI 环境使用 Python 3.10。供应商 SDK、模型、OpenPI、RoboTwin 和录制进程可能需要各自经过验证的环境。

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

在逐步清理历史格式问题期间，全仓库 Ruff 和格式报告仍为非阻塞项。不要在无关改动中批量格式化已有代码
或 `third_party/`。

控制面板使用独立测试套件：

```bash
uv pip install --python .venv/bin/python \
  -r requirements/requirements-hardware.txt \
  -r requirements/requirements-scripts.txt
XDG_DATA_HOME=/tmp/superinference-panel-test \
  .venv/bin/python -m pytest incubator/control_panel/tests -q
```

依赖文件按进程环境划分，并非一个通用环境：

- `requirements/requirements.txt`：共享运行时。
- `requirements/requirements-dev.txt`：根目录测试、Ruff 和 pre-commit 支持。
- `requirements/requirements-hardware.txt`：可通过 pip 安装的硬件辅助依赖。
- `requirements/requirements-policy.txt`：轻量策略传输和预处理依赖。
- `requirements/requirements-scripts.txt`：操作、分析和转换工具依赖。
- `requirements/requirements-robotwin.txt`：RoboTwin 专用仿真环境。
- `requirements/requirements-all.txt`：可通过 pip 安装的通用环境，不包含供应商锁定版本的依赖栈。

## 目录结构

```text
configs/                    按子系统组织的 Hydra 运行时配置
requirements/               各进程的 Python 依赖环境
devices/                    设备基类、注册表、相机、机器人和适配器
manager/                    观测管理器、策略运行器和动作块管理器
consumer/                   推理入口、编排、IO 和可视化
policy/                     策略接口和模型适配器
utils/                      共享传输、契约、安全、时序和诊断工具
scripts/                    操作、标定、回放、分析和仿真工具
tests/                      无硬件的根目录契约测试和集成测试
incubator/control_panel/    可选的操作界面和发布工具
third_party/                子模块及供应商/模型集成
.claude/skills/             维护中的操作知识库
```

## Maintenance

The blocking GitLab pipeline is hardware-free. Model, Control Panel, hardware, and third-party checks remain
separate jobs. Review policy and merge checks live in the
[SuperInference CI skill](.claude/skills/superinference-ci/SKILL.md) and the GitLab MR template.

The scheduled submodule job updates only the observed remote-tip columns below. A submodule gitlink change still
requires a manually reviewed commit that updates its bound commit and date.

<!-- SUBMODULE-MAINTENANCE-TABLE:START -->
| Submodule | Tracking branch | Bound commit | Bound date | Latest commit | Latest date |
| --- | --- | --- | --- | --- | --- |
| `third_party/openpi` | `main` | `6b6188790b533177601dd0edfbd16541d1320e00` | 2026-08-18 | `6b6188790b533177601dd0edfbd16541d1320e00` | 2026-08-18 |
| `third_party/i2rt` | `fix/passive-encoder-wrap` | `aceb23a942a61f0c55fa257499b5a64336205315` | 2026-10-09 | `aceb23a942a61f0c55fa257499b5a64336205315` | 2026-10-09 |
| `third_party/lktech_motor_sdk` | `main` | `c7bf45a77cfed682b2b124c4db69df61527fc683` | 2026-09-18 | `c7bf45a77cfed682b2b124c4db69df61527fc683` | 2026-09-18 |
| `third_party/nmx_vla` | `junlin/fasterwam-inference` | `b81a39f8722ecf1a370b83b0d1026e5f6e3346a5` | 2026-09-21 | `b81a39f8722ecf1a370b83b0d1026e5f6e3346a5` | 2026-09-21 |
| `third_party/RoboTwin` | `superinference` | `a5c73d41b0dd5ecd4da92d271466d6655e8293a8` | 2026-09-07 | `6bee183536189bb2890b9077ac82cfd1560324f2` | 2026-09-11 |
<!-- SUBMODULE-MAINTENANCE-TABLE:END -->
