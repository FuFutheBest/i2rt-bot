# I2RT D405/YAM Data Collection Deployment SOP

本文档用于在新的 I2RT 数采机器上部署双腕 Intel RealSense D405、YAM
`crank_4310` 平行夹爪和 NEDF3 录制系统。

## 1. 硬件和系统要求

每台机器需要：

- Ubuntu Linux x86_64；
- 两台 Intel RealSense D405；
- 四个 SocketCAN 适配器；
- 左右 leader 和左右 follower 机械臂；
- 两个 follower 均使用 `crank_4310` 平行夹爪；
- 可访问 `gitlab.noematrix.cn` 和 `pypi-cloud.noematrix.cn`；
- 已配置访问 SuperInference 的 GitLab SSH key。

安装基础工具：

```bash
sudo apt update
sudo apt install -y \
  git git-lfs curl build-essential pkg-config \
  can-utils v4l-utils usbutils \
  libgl1 libglib2.0-0
```

确认 `uv` 已安装：

```bash
uv --version
```

## 2. 下载代码

```bash
cd "$HOME"
git clone \
  --branch feat/i2rt \
  git@gitlab.noematrix.cn:noematrix/pre-training/SuperInference.git \
  superinference_i2rt
cd "$HOME/superinference_i2rt"
git submodule update --init --recursive
```

确认版本：

```bash
git branch --show-current
git rev-parse HEAD
git submodule status
```

当前部署分支应为 `feat/i2rt`。

## 3. 安装设备环境

设备进程使用 Python 3.10。I2RT 驱动会将 NumPy 固定为 `2.2.6`。

```bash
cd "$HOME/superinference_i2rt"
uv python install 3.10
uv venv .venv --python 3.10
uv pip install \
  --python .venv/bin/python \
  -r requirements/requirements-hardware.txt
uv pip install \
  --python .venv/bin/python \
  -e ./third_party/i2rt
```

检查关键模块：

```bash
.venv/bin/python - <<'PY'
import cv2
import flask
import hydra
import i2rt
import numpy
import omegaconf
import pyrealsense2

print("device environment OK")
print("numpy:", numpy.__version__)
PY
```

## 4. 安装 NEDF3 录制环境

设备和录制环境不能合并。录制端使用 Python 3.12 和 NumPy `1.26.4`。

下载私有 wheel：

```bash
cd "$HOME/superinference_i2rt"
NEDF_WHEELHOUSE="$PWD/.superinference/local/nedf3-wheelhouse"
mkdir -p "$NEDF_WHEELHOUSE"

curl -fL \
  -o "$NEDF_WHEELHOUSE/dm3_core-0.2.18-py3-none-any.whl" \
  'https://pypi-cloud.noematrix.cn/repository/pypi-releases/packages/dm3-core/0.2.18/dm3_core-0.2.18-py3-none-any.whl'

curl -fL \
  -o "$NEDF_WHEELHOUSE/nmx_msg-2.2.0-py3-none-any.whl" \
  'https://pypi-cloud.noematrix.cn/repository/pypi-group/packages/nmx-msg/2.2.0/nmx_msg-2.2.0-py3-none-any.whl'

sha256sum "$NEDF_WHEELHOUSE"/*.whl
```

预期 SHA-256：

```text
dm3-core  1292b6175fea517bbeb1acb4eaa43353151cef41964460b92c75495c9362a8cc
nmx-msg   fe7b674919c5ea744e076f86ae4ad549d5b28e99b58ac1a4a5bbba3d76716584
```

创建并安装录制环境：

```bash
uv python install 3.12
uv venv .superinference/local/i2rt-recording --python 3.12

incubator/control_panel/release/install_source_recording_profile.sh \
  "$PWD/.superinference/local/i2rt-recording/bin/python" \
  "$NEDF_WHEELHOUSE"
```

确认私有包版本：

```bash
.superinference/local/i2rt-recording/bin/python - <<'PY'
from importlib.metadata import version

print("dm3-core:", version("dm3-core"))
print("nmx-msg:", version("nmx-msg"))
PY
```

必须得到 `dm3-core 0.2.18` 和 `nmx-msg 2.2.0`。

## 5. 识别 D405

连接两台 D405 后执行：

```bash
.venv/bin/python - <<'PY'
import pyrealsense2 as rs

for device in rs.context().query_devices():
    print(
        device.get_info(rs.camera_info.name),
        device.get_info(rs.camera_info.serial_number),
    )
PY
```

通过遮挡镜头或逐台拔插，确定左腕和右腕相机序列号。两个序列号必须不同。

## 6. 识别 CAN 适配器

列出接口和硬件序列号：

```bash
ip -brief link | grep can

for iface in can0 can1 can2 can3; do
  echo "===== $iface ====="
  udevadm info -q property \
    -p "/sys/class/net/$iface" |
    grep -E '^(ID_SERIAL|ID_SERIAL_SHORT)='
done
```

通过逐个拔插或硬件标签确认角色：

| 角色 | 配置接口 |
| --- | --- |
| 右 leader | `can0` |
| 左 leader | `can1` |
| 右 follower | `can2` |
| 左 follower | `can3` |

记录每个角色对应的完整 `ID_SERIAL`。重新插拔或更换 USB 端口后必须重新检查。

## 7. 启用 CAN

当前 CANable 不支持 `restart-ms`，不要设置该参数。

```bash
for iface in can0 can1 can2 can3; do
  sudo ip link set dev "$iface" down
  sudo ip link set dev "$iface" type can bitrate 1000000
  sudo ip link set dev "$iface" up
done

for iface in can0 can1 can2 can3; do
  ip -details link show "$iface" | sed -n '1,7p'
done
```

每个接口都应为 `UP`、`LOWER_UP`、`ERROR-ACTIVE`，且 bitrate 为
`1000000`。

## 8. 注册本机 runtime profiles

将尖括号内容替换为本机真实值：

```bash
export LEFT_RS='<左腕D405序列号>'
export RIGHT_RS='<右腕D405序列号>'
export CAN_LEADER_LEFT='<can1完整ID_SERIAL>'
export CAN_FOLLOWER_LEFT='<can3完整ID_SERIAL>'
export CAN_LEADER_RIGHT='<can0完整ID_SERIAL>'
export CAN_FOLLOWER_RIGHT='<can2完整ID_SERIAL>'
```

注册设备环境：

```bash
.venv/bin/python scripts/diagnostics/runtime_profiles.py register \
  i2rt-d405-yam \
  --python .venv/bin/python \
  --cwd "$PWD" \
  --env PYTHONNOUSERSITE=1 \
  --env "SI_I2RT_LEFT_RS_SERIAL=$LEFT_RS" \
  --env "SI_I2RT_RIGHT_RS_SERIAL=$RIGHT_RS" \
  --env SI_I2RT_YAM_GRIPPER_TYPE=crank_4310 \
  --env "SI_I2RT_CAN_LEADER_LEFT_SERIAL=$CAN_LEADER_LEFT" \
  --env "SI_I2RT_CAN_FOLLOWER_LEFT_SERIAL=$CAN_FOLLOWER_LEFT" \
  --env "SI_I2RT_CAN_LEADER_RIGHT_SERIAL=$CAN_LEADER_RIGHT" \
  --env "SI_I2RT_CAN_FOLLOWER_RIGHT_SERIAL=$CAN_FOLLOWER_RIGHT" \
  --import hydra \
  --import omegaconf \
  --import pyrealsense2 \
  --import i2rt \
  --import flask \
  --import cv2
```

绑定设备角色：

```bash
for role in launcher device; do
  .venv/bin/python scripts/diagnostics/runtime_profiles.py bind \
    --config-name i2rt_d405_yam_collect_async \
    --role "$role" \
    --profile i2rt-d405-yam
done
```

注册并绑定录制环境：

```bash
.venv/bin/python scripts/diagnostics/runtime_profiles.py register \
  i2rt-nedf3-recording \
  --python .superinference/local/i2rt-recording/bin/python \
  --cwd "$PWD" \
  --env PYTHONNOUSERSITE=1 \
  --import dm3_core.nedf \
  --import dm3_core.mcap \
  --import nmx_msg \
  --import consumer.io.async_data_saver_i2rt_NEDF3

.venv/bin/python scripts/diagnostics/runtime_profiles.py bind \
  --config-name i2rt_d405_yam_collect_async \
  --role recording \
  --profile i2rt-nedf3-recording
```

验证并检查所有绑定：

```bash
.venv/bin/python scripts/diagnostics/runtime_profiles.py verify \
  --profile i2rt-d405-yam
.venv/bin/python scripts/diagnostics/runtime_profiles.py verify \
  --profile i2rt-nedf3-recording
.venv/bin/python scripts/diagnostics/runtime_profiles.py doctor \
  --config-name i2rt_d405_yam_collect_async
```

所有角色都必须显示 `PASS`。`.superinference/local/` 是本机配置，不得提交到
Git，也不得复制其他机器生成的 `runtime_profiles.json`。

## 9. 启动数采 UI

```bash
cd "$HOME/superinference_i2rt"

.venv/bin/python scripts/diagnostics/runtime_profiles.py doctor \
  --config-name i2rt_d405_yam_collect_async

.venv/bin/python scripts/diagnostics/runtime_profiles.py run \
  --config-name i2rt_d405_yam_collect_async \
  --role launcher -- \
  -m incubator.nedf3.I2RT_DM3.web \
  --host 127.0.0.1 \
  --port 8080 \
  --device-config i2rt_d405_yam_collect_async \
  --saver-config configs/i2rt_d405_yam_saver.yaml
```

本机打开 <http://127.0.0.1:8080>。远程操作时使用 SSH 隧道：

```bash
ssh -L 8080:127.0.0.1:8080 <用户名>@<数采机器IP>
```

然后在操作电脑打开 <http://127.0.0.1:8080>。

## 10. 首次启动验收

启动设备前确认：

1. 人员离开机械臂运动范围，急停可用；
2. 四根 CAN 线角色正确；
3. 两台 D405 左右角色正确；
4. 没有其他进程占用同一组 CAN 设备；
5. DM3 已登录并选择任务。

在 UI 点击“启动设备”后检查进程：

```bash
ps -eo pid,args |
  grep -E 'I2RT_DM3|device_starter|async_data_saver_i2rt_NEDF3' |
  grep -v grep
```

UI 应显示：

- device 和 saver 正常运行；
- 左右腕相机实时显示，画面角色正确；
- 腕部预览接近 30 FPS；
- 头部相机为预期的黑色占位画面；
- leader 和 follower 状态正常。

低速检查以下动作：

- 左 leader 控制左 follower；
- 右 leader 控制右 follower；
- 右腕 J6 旋转方向正确；
- 左右夹爪开合方向和行程正确；
- 停止操作后没有持续运动。

YAM `crank_4310` 不使用 LKTech 网页夹爪标定按钮。

## 11. 首次试采验收

先录制一条约 10 秒的测试数据：

1. 开始 Episode；
2. 缓慢移动左右机械臂；
3. 左右夹爪分别完成一次完整开合；
4. 让两台腕部相机都观察到明显运动；
5. 结束 Episode，并等待写入和检查完成。

默认数据目录为 `data/i2rt_nedf3/`。检查生成文件：

```bash
find data/i2rt_nedf3 -maxdepth 3 -type f | sort | tail -50
```

合格数据应满足：

- Episode 已完成，没有停留在 `_inprogress`；
- 左右腕视频均存在，左右角色正确；
- 主相机是右腕真实相机；
- 头部相机是预期的黑色占位流；
- 左右 arm state 连续；
- 左右 gripper state 都有变化；
- 相机约 30 FPS，机械臂状态约 90 Hz；
- 没有时间戳倒退、大量丢帧或 writer 错误。

## 12. 每次开机流程

```bash
cd "$HOME/superinference_i2rt"
git status --short

for iface in can0 can1 can2 can3; do
  sudo ip link set dev "$iface" down
  sudo ip link set dev "$iface" type can bitrate 1000000
  sudo ip link set dev "$iface" up
done

.venv/bin/python scripts/diagnostics/runtime_profiles.py doctor \
  --config-name i2rt_d405_yam_collect_async
```

之后启动数采 UI。USB 设备重新插拔后，应再次核对 CAN 接口与硬件序列号。
系统会拒绝启动角色错位、未启用或 bitrate 不正确的 CAN 适配器。

## 13. 更新代码

只在 device 和 saver 均停止后更新：

```bash
cd "$HOME/superinference_i2rt"
git status --short
git fetch origin
git switch feat/i2rt
git pull --ff-only origin feat/i2rt
git submodule update --init --recursive
```

环境被重建、解释器变化、核心依赖变化或代码更新后，重新执行：

```bash
.venv/bin/python scripts/diagnostics/runtime_profiles.py verify \
  --profile i2rt-d405-yam
.venv/bin/python scripts/diagnostics/runtime_profiles.py verify \
  --profile i2rt-nedf3-recording
.venv/bin/python scripts/diagnostics/runtime_profiles.py doctor \
  --config-name i2rt_d405_yam_collect_async
```

不要在存在不明本地修改时直接更新，不要使用
`git submodule update --remote`，也不要跨机器复制 `.superinference/local/`。
