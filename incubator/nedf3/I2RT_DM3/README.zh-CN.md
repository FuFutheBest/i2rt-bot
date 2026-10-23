# I2RT NEDF3 数据采集 WebUI

本目录提供 I2RT 双臂遥操作设备的 DM3 数据采集网页控制台。WebUI 负责 DM3 登录、任务选择、硬件方案绑定、设备和保存进程管理、Episode 控制、实时相机预览以及 NEDF3 Episode 上传。

Web 服务启动时不会打开 CAN、串口或相机设备。只有操作员在网页中完成登录、任务选择并点击“启动设备”后，系统才会启动硬件进程。

## 1. 目录和配置

| 路径 | 用途 |
| --- | --- |
| `incubator/nedf3/I2RT_DM3/web/` | Flask 服务、网页模板、静态资源和 API |
| `incubator/nedf3/I2RT_DM3/controller.py` | DM3 会话、任务、进程和上传的协调器 |
| `incubator/nedf3/I2RT_DM3/process_manager.py` | 启动、监控和停止设备、保存、标定进程 |
| `configs/i2rt_system_profiles.yaml` | WebUI 可选的系统方案清单 |
| `configs/i2rt_d405_yam_collect_async.yaml` | 三个 D405 + YAM 夹爪的设备配置 |
| `configs/i2rt_d405_yam_teleop.yaml` | D405/YAM 的 CAN 通道、端口和控制参数 |
| `configs/i2rt_d405_yam_saver.yaml` | D405/YAM 的 NEDF3 保存参数 |
| `configs/i2rt_nedf3_saver.yaml` | LKTech/鱼眼方案的 NEDF3 保存参数 |
| `.superinference/local/` | 本机运行时标定和硬件绑定，已加入 Git 忽略，不要提交 |

当前系统方案包括：

- `lktech_fisheye_2cam`：LKTech 夹爪 + 左右双鱼眼相机，支持网页夹爪标定。
- `yam_d405_2cam`：YAM 夹爪 + 左右两个 D405。
- `yam_d405_3cam`：YAM 夹爪 + 左右腕部 D405 + 头部 D405。

## 2. Python 环境和依赖

建议使用 Python 3.10 创建设备/WebUI 环境。不要把本机绝对路径写入 Git 跟踪的配置或文档。

### 2.1 创建基础环境

在仓库根目录执行：

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python \
  -r requirements/requirements-hardware.txt
uv pip install --python .venv/bin/python \
  -e third_party/i2rt
```

如果需要运行测试和 pre-commit：

```bash
uv pip install --python .venv/bin/python \
  -r requirements/requirements-dev.txt
```

初始化项目记录的子模块版本：

```bash
git submodule update --init --recursive
```

### 2.2 依赖清单

以下是这套 WebUI、设备进程和 I2RT 驱动使用的包。版本约束以仓库中的 requirements 文件和 `third_party/i2rt/pyproject.toml` 为准。

| 类别 | 包 |
| --- | --- |
| WebUI | `flask`、`werkzeug`（由 Flask 提供） |
| 基础运行时 | `numpy`、`scipy`、`omegaconf`、`hydra-core`、`pyyaml`、`loguru`、`psutil`、`requests`、`pydantic`、`rerun-sdk` |
| 图像和 D405 | `opencv-python`、`pyrealsense2` |
| I2RT 驱动 | `click<8.2.0`、`crcmod-plus`、`dm-env==1.6`、`mcap-ros2-support`、`mink`、`mujoco`、`portal==3.7.3`、`python-can==4.5.0`、`pygame==2.6.1`、`qpsolvers[quadprog]==4.7.1`、`ruckig==0.15.3`、`threadpoolctl==3.6.0`、`tyro`、`viser`、`pyserial` |
| NEDF3 保存和可视化 | `dm3-core==0.2.18`、`nmx-msg==2.2.0`、`rerun-sdk==0.22.1` |
| 测试和开发，可选 | `pytest`、`pytest-mock`、`pre-commit`、`ruff`、`imageio`、`transforms3d`、`matplotlib`、`msgpack`、`websockets`、`ruamel.yaml` |

`requirements/requirements-hardware.txt` 会安装硬件侧通用依赖；I2RT 的依赖由 `uv pip install -e third_party/i2rt` 安装。不要只安装根目录的 `requirements/requirements.txt` 后直接启动 D405，因为其中不包含 `pyrealsense2`、OpenCV 和 Flask。

### 2.3 NEDF3 录制环境

NEDF3 保存进程使用独立的 `recording` runtime profile。当前批准的录制环境是 Python 3.12，并且必须安装：

- `dm3-core==0.2.18`
- `nmx-msg==2.2.0`
- `rerun-sdk==0.22.1`
- `consumer.io.async_data_saver_i2rt_NEDF3` 能导入的仓库代码和其依赖

这两个私有 wheel 不要提交到仓库。将 wheel 放在本机目录后，按运行时管理流程安装、注册、绑定和验证：

```bash
uv python install 3.12
uv venv .superinference/local/i2rt-recording --python 3.12

NEDF_WHEELHOUSE="$PWD/.superinference/local/nedf3-wheelhouse"
incubator/control_panel/release/install_source_recording_profile.sh \
  "$PWD/.superinference/local/i2rt-recording/bin/python" \
  "$NEDF_WHEELHOUSE"

python3 scripts/diagnostics/runtime_profiles.py register i2rt-nedf3-recording \
  --python .superinference/local/i2rt-recording/bin/python \
  --import dm3_core.nedf \
  --import dm3_core.mcap \
  --import nmx_msg \
  --import rerun \
  --import consumer.io.async_data_saver_i2rt_NEDF3

python3 scripts/diagnostics/runtime_profiles.py bind \
  --config-name i2rt_d405_yam_collect_async \
  --role recording \
  --profile i2rt-nedf3-recording

python3 scripts/diagnostics/runtime_profiles.py verify \
  --profile i2rt-nedf3-recording
python3 scripts/diagnostics/runtime_profiles.py doctor \
  --config-name i2rt_d405_yam_collect_async
```

设备环境和录制环境可以使用不同的 NumPy 版本。I2RT 驱动要求 `numpy==2.2.6`；NEDF3 录制 profile 按批准的 wheel 和本机配置管理，不要用未经验证的环境替代它。

## 3. 硬件前置条件

### 3.1 CAN

YAM 配置使用以下 CAN 映射：

| 角色 | 接口 |
| --- | --- |
| 左臂 leader | `can1` |
| 左臂 follower | `can3` |
| 右臂 leader | `can0` |
| 右臂 follower | `can2` |

四个接口必须是实际 SocketCAN 设备、状态为 `UP`、波特率为 1 Mbit/s。启动前只读检查：

```bash
for iface in can0 can1 can2 can3; do
  ip -details link show "$iface"
done
```

WebUI 会使用 `udevadm` 校验适配器身份。将适配器序列号绑定到本机运行时环境，不要写入 Git 跟踪文件：

```bash
export SI_I2RT_CAN_LEADER_LEFT_SERIAL=<can1-ID_SERIAL>
export SI_I2RT_CAN_FOLLOWER_LEFT_SERIAL=<can3-ID_SERIAL>
export SI_I2RT_CAN_LEADER_RIGHT_SERIAL=<can0-ID_SERIAL>
export SI_I2RT_CAN_FOLLOWER_RIGHT_SERIAL=<can2-ID_SERIAL>
export SI_I2RT_YAM_GRIPPER_TYPE=crank_4310
```

如果适配器重新插拔，`canN` 名称可能变化。必须重新确认线缆、序列号、接口和 1 Mbit/s 设置。

### 3.2 D405

三相机方案需要三个不同的 D405 序列号：左腕、右腕和头部。可以用以下命令枚举：

```bash
.venv/bin/python -c \
  'import pyrealsense2 as rs; print([(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()])'
```

旧的命令行启动方式也支持以下环境变量：

```bash
export SI_I2RT_LEFT_RS_SERIAL=<left-wrist-D405-serial>
export SI_I2RT_RIGHT_RS_SERIAL=<right-wrist-D405-serial>
export SI_I2RT_HEAD_RS_SERIAL=<head-D405-serial>
```

推荐在网页的“设备方案”区域点击“重新枚举”，选择设备并点击“保存本机配置”。绑定会写入 `.superinference/local/i2rt_hardware_bindings.json`，不会修改追踪的 YAML。

### 3.3 系统工具和权限

主机需要提供 `ip`、`udevadm`、可用的 SocketCAN 内核接口、RealSense 驱动和正确的 USB/CAN 权限。若系统无法访问 `/dev/video*`、CAN 或串口，应先修复 udev 规则和用户组权限，再启动 WebUI。

## 4. 启动 WebUI

### 4.1 普通本机启动

在仓库根目录执行：

```bash
source .venv/bin/activate
python -m incubator.nedf3.I2RT_DM3.web \
  --host 127.0.0.1 \
  --port 8080
```

浏览器打开 <http://127.0.0.1:8080>。检查服务状态：

```bash
curl http://127.0.0.1:8080/api/health
```

默认只监听回环地址。如果从另一台电脑访问，请使用 SSH 隧道：

```bash
ssh -L 8080:127.0.0.1:8080 <user>@<robot-host>
```

然后在本地浏览器访问 <http://127.0.0.1:8080>。只有在可信网络中并明确需要时，才使用 `--allow-remote` 配合非回环地址。

### 4.2 D405/YAM 三相机启动

D405/YAM 方案要求使用已验证的 `launcher` 和 `recording` profile：

```bash
python3 scripts/diagnostics/runtime_profiles.py doctor \
  --config-name i2rt_d405_yam_collect_async

python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name i2rt_d405_yam_collect_async \
  --role launcher -- \
  -m incubator.nedf3.I2RT_DM3.web \
  --device-config i2rt_d405_yam_collect_async \
  --saver-config configs/i2rt_d405_yam_saver.yaml
```

如果 `doctor` 报告 profile 缺失、未验证或指纹变化，应先修复 runtime profile；不要静默使用名称相似的虚拟环境。

## 5. WebUI 操作流程

### 第一步：登录和选择任务

1. 打开“DM3 登录”页面，输入 DM3 手机号和密码并点击“登录 DM3”。
2. 可按执行日期、页码和每页数量筛选任务，点击“获取任务”。
3. 在任务表中选中一个任务，点击“选择任务”。未选择任务时不能开始保存。

### 第二步：选择和绑定系统方案

1. 打开“数据采集”页面。
2. 在“设备方案”中选择 `lktech_fisheye_2cam`、`yam_d405_2cam` 或 `yam_d405_3cam`。
3. 点击“重新枚举”，确认左右相机、头部相机和夹爪设备与实际硬件一致。
4. 点击“保存本机配置”。设备运行或保存进程运行时不能修改绑定。

### 第三步：标定夹爪（仅 LKTech）

LKTech 方案才显示“标定左右夹爪”。开始标定前必须停止设备和 NEDF3 保存进程。YAM 的 `i2rt_can` 夹爪不使用这个网页标定流程。

### 第四步：启动设备和保存进程

1. 点击“启动设备”，等待设备状态变为“就绪”，并确认两路机器人和相机日志没有错误。
2. 确认相机监看区域出现实时画面。
3. 点击“启动保存”。保存进程只能在设备就绪后启动。
4. 设备、遥操作、保存三项状态都正常后，才开始 Episode。

WebUI 会管理 `device_starter.py` 和 NEDF3 saver 子进程。不要同时在另一个终端启动同一套遥操作或保存程序，否则两个进程可能竞争 CAN、串口或共享内存。

### 第五步：录制 Episode

- 点击“网页开始 Episode”开始录制。
- 操作双臂完成任务。
- 点击同一个按钮结束并保存 Episode。
- 任务失败或不应保留时，点击“网页丢弃 Episode”。

结束采集时必须按以下顺序操作：

1. 先结束或丢弃当前 Episode。
2. 停止 NEDF3 保存进程。
3. 停止设备进程。

WebUI 会阻止在保存进程仍运行时停止设备，以避免写入不完整的 Episode。

### 第六步：上传

“上传与性能监控”页面可以刷新 Episode 扫描、开启自动上传、上传符合条件的 Episode、重试失败项以及查看上传日志。`resource_type` 固定为 `i2rt`。上传前请确认 DM3 登录仍有效并且 Episode 已完成，不要上传 `_inprogress` 目录。

## 6. 数据格式和采样频率

YAM/D405 保存配置使用：

- 机器人状态：90 Hz。
- 夹爪状态：30 Hz。
- 相机：30 Hz。
- JPEG 质量：90。

D405 腕部相机输入为 640 x 360（宽 x 高）。保存前从左右两侧各裁剪 185 像素，得到 270 x 360，再缩放为 480 x 640（宽 x 高）；写入 NEDF3 时对应 HWC 形状 `(640, 480, 3)`。头部相机保存为 640 x 480（宽 x 高），对应 `(480, 640, 3)`。

共享内存中的相机数据为 RGB。`i2rt-nedf3-v2` 保存器在调用 OpenCV JPEG 编码器前显式转换为 BGR，因此 JPEG 解码后保持正确的 RGB 颜色。Rerun 会自动识别并修复旧 `i2rt-nedf3-v1` Episode 中历史遗留的红蓝通道互换；旧文件本身不会被修改。

Episode 通常先写入：

```text
data/<任务名>/episode_0000_inprogress/
```

成功完成后改名为 `episode_0000`。异常中断、缺少必需 topic 或 writer 出错的 Episode 会进入 `quarantine/`，并附带原因文件。

### 使用 Rerun 查看已保存数据

Rerun 工具只读取已经完成的 Episode，不会打开机器人、CAN 或相机硬件。界面同时显示三路相机、leader/follower 左右臂的全部关节曲线以及四路夹爪曲线，并使用 Episode 相对时间保持同步。

通过已验证的 `recording` profile 启动：

```bash
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name i2rt_d405_yam_collect_async \
  --role recording -- \
  scripts/visualization/i2rt_nedf3_rerun.py \
  'data/<任务名>/episode_0000'
```

不提供 Episode 路径时，工具会从 `data/` 下选择最新的已完成 Episode。也可以使用 `--data-dir 'data/<任务名>'` 限制搜索目录。需要生成可传输的 Rerun 文件而不打开窗口时，增加 `--output episode_0000.rrd`。

录制环境固定使用 NumPy 1.26.4，因此 Rerun 使用兼容版本 `rerun-sdk==0.22.1`。不要在该环境中升级到要求 NumPy 2.x 的新版 Rerun。

设备、保存和标定进程全部停止后，也可以在“数据采集”页面点击“选择 Episode 并用 Rerun 打开”。网页会先列出 `data/` 下的任务文件夹和所有已完成 Episode，由操作员选择后再启动 Rerun。

## 7. 常见问题

### `file descriptor cannot be a negative integer (-1)`

这通常表示 CAN socket 已关闭或设备进程正在停止时仍有线程发送消息。停止所有旧的设备/保存进程，确认 `can0` 到 `can3` 状态为 `UP` 且波特率为 1 Mbit/s，再由 WebUI 重新启动设备。不要并行启动第二个 I2RT 进程。

### 等待设备流超时

检查三项：

1. D405 序列号是否唯一且与网页保存的本机绑定一致。
2. `RealSenseCamera_*_data` 共享内存是否已经出现。
3. CAN 接口是否在线、适配器序列号是否匹配、波特率是否为 1 Mbit/s。

### 保存进程无法启动

先执行：

```bash
python3 scripts/diagnostics/runtime_profiles.py doctor \
  --config-name i2rt_d405_yam_collect_async
```

重点检查 `dm3_core.nedf`、`dm3_core.mcap`、`nmx_msg` 和 `consumer.io.async_data_saver_i2rt_NEDF3` 是否能导入，以及录制 profile 是否已经验证。

### 相机预览为空

相机预览读取的是已运行设备进程发布的共享内存，不会单独打开相机。先确认设备状态为“就绪”，再查看设备日志和对应的 `RealSenseCamera_*_data` 或 `WebCameraDevice_*_data` 名称。

## 8. 测试和停止

硬件无关测试：

```bash
.venv/bin/python -m pytest -q \
  tests/incubator/test_i2rt_hardware_bindings.py \
  tests/incubator/test_i2rt_dm3.py \
  tests/incubator/test_i2rt_dm3_web.py
```

停止 WebUI：在运行服务的终端按 `Ctrl+C`。服务退出时会停止它管理的保存、设备、标定和上传进程。硬件测试和实际采集需要真实设备，不应在没有硬件的 CI 环境中执行。
