# FasterWAM 推理启动命令

**所有进程均可通过 runtime_profiles.py 启动（原有的 Python 环境启动方式仍然可用）；本机环境完成一次注册和绑定后，后续无需指定或激活 Python 环境。**

所有命令从项目根目录执行。本文件只对应 FasterWAM 配置：

- FasterWAM：`policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam`

配置移动后只使用上述完整原生名称；旧的 `fastwam/...` 名称不再有效，本机 runtime profile 也必须按新名称重新绑定。
该配置是独立的顶层 rollout 配置，直接声明设备 preset 与 camera 绑定，不继承普通 FastWAM 的顶层配置。它使用
one-pass future cache 路径，`policy.params.ckpt_path` 与 `policy.params.task` 由自身 YAML 固定维护；runtime
profile 只保存解释器和依赖校验信息。它也不能继承普通 FastWAM、RTC 或 FastWAM memory 的本机 profile 绑定。

每个新终端都要先设置同一个变量。

## Python 环境检查（可选）

`doctor` 只检查 profile 绑定、解释器、依赖导入和环境指纹，不连接真机，也不启动模型。完整配置检查要求
`launcher`、`planner`、`device`、`fastwam-server`、`policy` 和 `recording` 全部完成绑定；即使 YAML 中
`infer_rollout.enabled=false`，也必须预先注册录制环境：

```bash
FASTERWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam
python3 scripts/diagnostics/runtime_profiles.py doctor \
  --config-name "$FASTERWAM_CONFIG"
```

只生成或核对命令、不启动真机时，使用 `command --json` 查看本机已解析的解释器和参数：

```bash
python3 scripts/diagnostics/runtime_profiles.py command --json \
  --config-name "$FASTERWAM_CONFIG" --role fastwam-server -- \
  consumer/fastwam_server.py --config-name "$FASTERWAM_CONFIG"
```

## 真机预检（带夹爪标定）

```bash
FASTERWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTERWAM_CONFIG" --role device -- \
  scripts/diagnostics/rollout_device_preflight.py \
  "configs/${FASTERWAM_CONFIG}.yaml" --gripper_calibrate
```

## 真机预检（不带夹爪标定）

```bash
FASTERWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTERWAM_CONFIG" --role device -- \
  scripts/diagnostics/rollout_device_preflight.py \
  "configs/${FASTERWAM_CONFIG}.yaml"
```

preflight 会扫描并引导确认 Flexiv 机器人、网卡、相机和夹爪，将确认后的序列号、接口、相机 ID、串口及
校准信息写回顶层配置和它引用的设备配置。机器人候选选择时输入 `s` 或 `skip` 可保留当前 `robot_sn` 和网卡白名单，
并继续执行夹爪和相机预检。成功后继续使用同一配置即可，不需要手工改 config。

## 控制面板启动

```bash
FASTERWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTERWAM_CONFIG" --role launcher -- \
  -m incubator.control_panel --config-name "$FASTERWAM_CONFIG"
```

## 分终端启动

终端 1，设备和动作执行器：

```bash
FASTERWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTERWAM_CONFIG" --role device -- \
  device_starter.py --config-name "$FASTERWAM_CONFIG"
```

终端 2，FastWAM 模型服务：

```bash
FASTERWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTERWAM_CONFIG" --role fastwam-server -- \
  consumer/fastwam_server.py --config-name "$FASTERWAM_CONFIG"
```

等待 `curl -fsS http://127.0.0.1:18090/healthz` 成功后，终端 3 启动 policy client：

```bash
FASTERWAM_CONFIG=policy_infer/fastwam/config_infer_fisheye_umi3_dual_fasterwam
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name "$FASTERWAM_CONFIG" --role policy -- \
  -m consumer.inference.fastwam_policy_infer \
  --config-name "$FASTERWAM_CONFIG"
```

配置名出现两次是两个不同层次：`--` 前由 `runtime_profiles.py` 选择该配置绑定的本机 Python 环境，
`--` 后由实际进程交给 Hydra 组合运行配置。两处必须保持一致。

普通 FastWAM 与 RTC 的启动命令见
[`fastwam_inference_commands.md`](fastwam_inference_commands.md)；FastWAM memory 的独立启动命令见
[`fastwam_memory_inference_commands.md`](fastwam_memory_inference_commands.md)。

SuperOffline 的浏览器进程入口及双模型离线对比前置检查见
[`superoffline_inference_commands.md`](superoffline_inference_commands.md)。
