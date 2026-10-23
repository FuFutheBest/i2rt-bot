# FastWAM 与 RTC 推理启动命令

**所有进程均可通过 runtime_profiles.py 启动（原有的 Python 环境启动方式仍然可用）；本机环境完成一次注册和绑定后，后续无需指定或激活 Python 环境。**

所有命令从项目根目录执行。本文件覆盖普通 FastWAM 与 RTC：

- 普通推理：`policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam`
- RTC 推理：`policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc`

配置移动后只使用上述完整原生名称；旧的 `fastwam/...` 名称不再有效，本机 runtime profile 也必须按新名称重新绑定。

每个配置名都需要独立绑定本机 profile，不能从另一个 FastWAM 配置继承。checkpoint 与任务文本由各自 YAML 的
`policy.params.ckpt_path` 和 `policy.params.task` 固定维护；runtime profile 只保存解释器和依赖校验信息。

下面默认使用 RTC；普通 FastWAM 只需把变量改成对应配置名。每个新终端都要先设置同一个变量。

## Python 环境检查（可选）

`doctor` 只检查 profile 绑定、解释器、依赖导入和环境指纹，不连接真机，也不启动模型。完整配置检查要求
`launcher`、`planner`、`device`、`fastwam-server`、`policy` 和 `recording` 全部完成绑定；即使 YAML 中
`infer_rollout.enabled=false`，也必须预先注册录制环境：

```bash
FASTWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc
python3 scripts/diagnostics/runtime_profiles.py doctor \
  --config-name "$FASTWAM_CONFIG"
```

只生成或核对命令、不启动真机时，使用 `command --json` 查看本机已解析的解释器和参数：

```bash
python3 scripts/diagnostics/runtime_profiles.py command --json \
  --config-name "$FASTWAM_CONFIG" --role fastwam-server -- \
  consumer/fastwam_server.py --config-name "$FASTWAM_CONFIG"
```

## 真机预检（带夹爪标定）

```bash
FASTWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTWAM_CONFIG" --role device -- \
  scripts/diagnostics/rollout_device_preflight.py \
  "configs/${FASTWAM_CONFIG}.yaml" --gripper_calibrate
```

## 真机预检（不带夹爪标定）

```bash
FASTWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTWAM_CONFIG" --role device -- \
  scripts/diagnostics/rollout_device_preflight.py \
  "configs/${FASTWAM_CONFIG}.yaml"
```

preflight 会扫描并引导确认 Flexiv 机器人、网卡、相机和夹爪，将确认后的序列号、接口、相机 ID、串口及
校准信息写回顶层配置和它引用的设备配置。成功后继续使用同一配置即可，不需要手工改 config。

## 控制面板启动

```bash
FASTWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTWAM_CONFIG" --role launcher -- \
  -m incubator.control_panel --config-name "$FASTWAM_CONFIG"
```

## 分终端启动

终端 1，设备和动作执行器：

```bash
FASTWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTWAM_CONFIG" --role device -- \
  device_starter.py --config-name "$FASTWAM_CONFIG"
```

终端 2，FastWAM 模型服务：

```bash
FASTWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTWAM_CONFIG" --role fastwam-server -- \
  consumer/fastwam_server.py --config-name "$FASTWAM_CONFIG"
```

等待 `curl -fsS http://127.0.0.1:18090/healthz` 成功后，终端 3 启动 policy client：

```bash
FASTWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fastwam_rtc
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTWAM_CONFIG" --role policy -- \
  -m consumer.inference.fastwam_policy_infer \
  --config-name "$FASTWAM_CONFIG"
```

配置名出现两次是两个不同层次：`--` 前由 `runtime_profiles.py` 选择该配置绑定的本机 Python 环境，
`--` 后由实际进程交给 Hydra 组合运行配置。两处必须保持一致。

FastWAM memory 的独立启动命令见
[`fastwam_memory_inference_commands.md`](fastwam_memory_inference_commands.md)；FasterWAM 的独立启动命令见
[`fasterwam_inference_commands.md`](fasterwam_inference_commands.md)。

SuperOffline 的浏览器进程入口及双模型离线对比前置检查见
[`superoffline_inference_commands.md`](superoffline_inference_commands.md)。
