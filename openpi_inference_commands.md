# OpenPI 推理启动命令

**所有进程均可通过 runtime_profiles.py 启动（原有的 Python 环境启动方式仍然可用）；本机环境完成一次注册和绑定后，后续无需指定或激活 Python 环境。**

所有命令从项目根目录执行。可用配置只有：

- 普通推理：`policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi`
- RTC 推理：`policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc`

下面默认使用 RTC；普通推理只需把变量改成第一个配置名。每个新终端都要先设置同一个变量。

## Python 环境检查（可选）

`doctor` 只检查 profile 绑定、解释器、依赖导入和环境指纹，不连接真机，也不启动模型。环境刚配置或依赖有变化时运行：

```bash
OPENPI_CONFIG=policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc
python3 scripts/diagnostics/runtime_profiles.py doctor \
  --config-name "$OPENPI_CONFIG"
```

## 真机预检（带夹爪标定）

```bash
OPENPI_CONFIG=policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$OPENPI_CONFIG" --role device -- \
  scripts/diagnostics/rollout_device_preflight.py \
  "configs/${OPENPI_CONFIG}.yaml" --gripper_calibrate
```

## 真机预检（不带夹爪标定）

```bash
OPENPI_CONFIG=policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$OPENPI_CONFIG" --role device -- \
  scripts/diagnostics/rollout_device_preflight.py \
  "configs/${OPENPI_CONFIG}.yaml"
```

preflight 会扫描并引导确认 Flexiv 机器人、网卡、相机和夹爪，将确认后的序列号、接口、相机 ID、串口及
校准信息写回顶层配置和它引用的设备配置。成功后继续使用同一配置即可，不需要手工改 config。

## 控制面板启动

```bash
OPENPI_CONFIG=policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$OPENPI_CONFIG" --role launcher -- \
  -m incubator.control_panel --config-name "$OPENPI_CONFIG"
```

## 分终端启动

终端 1，设备和动作执行器：

```bash
OPENPI_CONFIG=policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$OPENPI_CONFIG" --role device -- \
  device_starter.py --config-name "$OPENPI_CONFIG"
```

终端 2，OpenPI policy：

```bash
OPENPI_CONFIG=policy_infer/openpi/config_infer_fisheye_umi3_dual_openpi_rtc
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$OPENPI_CONFIG" --role policy -- \
  -m consumer.inference.openpi_policy_infer \
  --config-name "$OPENPI_CONFIG"
```

配置名出现两次是两个不同层次：`--` 前由 `runtime_profiles.py` 选择该配置绑定的本机 Python 环境，
`--` 后由实际进程交给 Hydra 组合运行配置。两处必须保持一致。只生成或核对命令、不启动真机时，把
上面的 `run` 改成 `command --json`。
