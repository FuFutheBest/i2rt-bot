# SuperInference 常用小工具

所有命令都从仓库根目录执行。`<...>` 表示必须替换的值。

- `[只读]`：不发送运动命令。
- `[硬件]`：会连接机器人、相机或串口。
- `[运动]`：可能 enable 或移动机器人、夹爪。
- `[写入]`：会修改配置或校准文件。

下面的命令要求当前 Python 环境已经包含对应硬件 SDK。环境不确定时，使用
[`useful_prompts.md`](useful_prompts.md) 中的“配置推理环境”模板。

## Flexiv Adapter

`devices/flexiv_adapter.py` 支持 Python RDK 和 pybind RDK。默认使用 `python` backend；只有
`flexiv_rdk_ext` 已正确编译并可导入时才使用 `--backend pybind`。

```bash
# [硬件][只读] 验证 RDK 并连接机器人；不会 enable
python devices/flexiv_adapter.py \
  --task test_rdk --robot-sn "<ROBOT_SN>" --backend python

# [硬件][只读] 读取关节角、TCP pose、tool 名称和参数
python devices/flexiv_adapter.py \
  --task read_states --robot-sn "<ROBOT_SN>" --backend python

# [硬件][运动] 关节目标：7 个值，单位 rad
python devices/flexiv_adapter.py \
  --task move_to_joints --robot-sn "<ROBOT_SN>" --backend python \
  --target <J1> <J2> <J3> <J4> <J5> <J6> <J7>

# [硬件][运动] 笛卡尔目标：[x y z qw qx qy qz]，位置单位 m，四元数顺序 wxyz
python devices/flexiv_adapter.py \
  --task move_to_pose --robot-sn "<ROBOT_SN>" --backend python \
  --target <X> <Y> <Z> <QW> <QX> <QY> <QZ>
```

两个运动任务都会尝试清故障、enable、等待 operational 并发送目标。必须显式填写 `--target`，并在运行前
确认机器人序列号、坐标系、运动范围和急停状态。

## Flexiv 双臂装配查看

```bash
# [硬件][只读] 读取 RDK 当前状态，查看双臂装配、法兰和 tool frame
python scripts/visualization/visualize_flexiv_assembly.py \
  "<LEFT_ROBOT_SN>" "<RIGHT_ROBOT_SN>" --gripper-type umi3_camera
```

该工具直接读取 RDK，是独立查看器，不会启动 DeviceManager 或 Control Panel。

## Rollout 设备绑定

```bash
ROLLOUT_CONFIG=policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi

# [硬件][运动][写入] 识别并绑定机器人、ttyUSB 与 capture-capable video 节点
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$ROLLOUT_CONFIG" --role device -- \
  scripts/diagnostics/rollout_device_preflight.py \
  "configs/${ROLLOUT_CONFIG}.yaml"

# [硬件][运动][写入] 绑定设备并重新标定夹爪
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$ROLLOUT_CONFIG" --role device -- \
  scripts/diagnostics/rollout_device_preflight.py \
  "configs/${ROLLOUT_CONFIG}.yaml" --gripper_calibrate
```

必须传入顶层 rollout YAML，不能传 `configs/devices/*.yaml` 或 `configs/devices/infer/*.yaml`。preflight 会把确认后的设备绑定写回配置及其
引用的 devices/calibration 文件；成功后无需手工修改 config，直接继续使用同一个配置文件。即使不加
`--gripper_calibrate`，夹爪识别过程也可能运动。

### 相机物理节点绑定

`WebCameraOpenPi` 的 `device_class_id` 是逻辑 SHM 身份（例如
`WebCameraOpenPi_1_data`），不是 USB 相机的永久编号。USB 插拔后不要把变化的
`/dev/videoN` 写进受 Git 跟踪的配置；为当前 shell 设置稳定的 UVC 节点链接：

```bash
export SUPERINFERENCE_CAMERA_LEFT_CAPTURE_ID='/dev/v4l/by-id/<left-camera>-video-index0'
export SUPERINFERENCE_CAMERA_RIGHT_CAPTURE_ID='/dev/v4l/by-id/<right-camera>-video-index0'
```

也可以使用 `/dev/v4l/by-path/...-video-index0`。FastWAM 和 OpenPI 的顶层配置会把
这两个变量传给 `capture_id`；变量未设置时使用顶层配置声明的物理编号。
不会回退到 `device_class_id`。preflight 可更新物理编号、robot_sn、串口及夹爪参数，
不能修改逻辑 ID 或设备顺序。实际节点名以本机的 `/dev/v4l/by-id` 或 `/dev/v4l/by-path`
目录为准，不要猜测或提交主机专属路径。
