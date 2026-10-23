# r1_teleop_dm3_upload — DM3 云端上传 (Linux)

`docs/windows_DM3`(Windows tkinter GUI)的 Linux 无界面复刻。复用同一套
跨平台模块,去掉 GUI,改成 CLI,适合采集机/服务器上传。

## 组成

| 文件 | 作用 |
|---|---|
| `dm3_api.py` | DM3 API 客户端:手机号登录、列任务、取 OSS STS 临时令牌 |
| `oss_uploader.py` | 阿里云 OSS(`oss2`)上传 `episode_*` 目录 |
| `upload_state.py` | 本地 `upload.json` 状态(pending/uploading/success/failed) |
| `cli.py` | 命令行入口(无界面) |
| `gui.py` | tkinter 图形界面(对照 Windows GUI) |
| `run_dm3_gui.sh` | 启动 GUI 的脚本 |

## 依赖

```bash
UPLOAD_PYTHON="$(python3 scripts/diagnostics/runtime_profiles.py resolve \
  --config-name quest3_collect_r1 --role upload --require-verified)"
"$UPLOAD_PYTHON" -m pip install requests oss2
# GUI 还需要系统 Tk(Debian/Ubuntu):
sudo apt install python3-tk
```

## 上传的数据布局

`incubator/r1/async_data_saver_NEDF3_r1.py` 现在用纯 UUID 命名 episode(无 `episode_` 前缀):

```
<data_dir>/
  <uuid>/            # 已完成(含 metadata.json) -> 会上传
    metadata.json
    data.mcap
  <uuid>_temp/       # 仍在写入 -> 跳过
```

OSS 对象键:`{sts.dir}/<uuid>/<相对路径>`(`{sts.dir}` 由 DM3 按 `resource_type` 下发),
例如 `<sts.dir>/<uuid>/metadata.json`。

> 上传器靠"目录非 `_temp` 且含 `metadata.json`"识别已完成 episode(不再依赖目录名前缀,旧的 `episode_<uuid>` 也兼容)。

## 用法

凭据优先级:`--phone/--password` > 环境变量 `DM3_PHONE`/`DM3_PASSWORD` > 交互式输入。

```bash
# 上传 saved_data 下所有已完成的 episode_*
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name quest3_collect_r1 --role upload -- \
  -m incubator.r1.r1_teleop_dm3_upload.cli upload saved_data --phone 13800000000

# 只上传单个 episode 目录
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name quest3_collect_r1 --role upload -- \
  -m incubator.r1.r1_teleop_dm3_upload.cli upload saved_data/<uuid>

# 指定 resource_type(默认 teleoperation_r1)
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name quest3_collect_r1 --role upload -- \
  -m incubator.r1.r1_teleop_dm3_upload.cli upload saved_data \
  --resource-type teleoperation_r1

# 列出我的 DM3 任务
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name quest3_collect_r1 --role upload -- \
  -m incubator.r1.r1_teleop_dm3_upload.cli tasks --phone 13800000000

# 用环境变量免交互
export DM3_PHONE=13800000000 DM3_PASSWORD='***'
python3 scripts/diagnostics/runtime_profiles.py run \
  --config-name quest3_collect_r1 --role upload -- \
  -m incubator.r1.r1_teleop_dm3_upload.cli upload saved_data
```

## 图形界面(GUI)

和 Windows 版一样的流程,只是无需命令行:

```bash
incubator/r1/r1_teleop_dm3_upload/run_dm3_gui.sh
```

界面分四块:**1. 登录** → **2. 拉取/选择任务** → **3. 选择本地数据目录、绑定 `upload.json`、上传 OSS** → **日志**。
默认数据目录为 `saved_data`,默认 `resource_type=teleoperation_r1`。

## 说明

- 每次 `upload` 会上传 `data_dir` 下**所有**已完成的 `episode_*`(与 Windows GUI 行为一致,
  不做按 episode 的去重)。重复跑会覆盖 OSS 上同名对象。
- 密码与令牌不落盘;`upload.json` 只记录状态与任务绑定信息。
- `oss2` 仅在真正上传时才需要;只跑 `tasks` 或 `--help` 不需要它。
