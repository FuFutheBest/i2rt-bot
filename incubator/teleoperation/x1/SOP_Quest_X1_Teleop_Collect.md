# Quest3 遥操作 X1 双臂 + NEDF3 数据采集 SOP

本文档记录 **NoeMatrix X1 双臂机器人 + Quest 3** 的低延迟 VR 遥操作与
ego 数据采集（NEDF3/mcap）经实测验证可用的完整流程，含本机调试中踩过的坑。

> 网络 / 自签证书 / Quest 接入（方案 A 路由器 vs 方案 B 热点）与 TJ 机器人完全相同，
> 不在此重复，详见同目录 `SOP_Quest_TJ_Teleop.md` 的 §1、§2、§4。本文只覆盖 **X1 专属**
> 的环境准备、三终端启动、数据落盘与排错。

---

## 0. 环境速查

| 项目 | 值 |
|------|-----|
| 容器 | `robot_test`（host 网络，挂载 `/home/nvidia/nmx_hal → /root`） |
| Python 环境 | `/root/tvenv`（容器内，`include-system-site-packages = true`） |
| 仓库根（容器内） | `/root/SuperInference`（= 宿主 `/home/nvidia/nmx_hal/SuperInference`） |
| Robot ID | `1e783be2dd870ef5` |
| VR 主程序端口 | `8012`（`run_x1_teleop_devices.py` 内的 X1VuerTeleoperator） |
| 视频流端口 | `8080`（`tj_camera_webrtc.py`，WebRTC 推流） |
| 访问 IP（方案 A） | `192.168.0.234` |
| 数据落盘目录 | `/root/SuperInference/saved_data/`（= 宿主 `…/SuperInference/saved_data/`） |

### 设备 / SHM 命名

| 设备类 | 维度 / 类型 | SHM 名 |
|--------|------------|--------|
| `X1Robot` | 30维 float64 `[L_q7,L_tcp7,R_q7,R_tcp7,head2]` | `X1Robot_1_data` |
| `X1VuerTeleoperator` | 18维 float64 `[L_xyz3,L_quat4,L_grip1,R_xyz3,R_quat4,R_grip1,head_pitch1,head_yaw1]` | `X1VuerTeleoperator_0_data` |
| `TjCameraDevice` | 图像 uint8 `[480,640,3]`，id 0=右腕 1=左腕 2=头部 | `TjCameraDevice_{0,1,2}_data` |
| `BaseDeviceManager` | 汇总全部设备 | `device_summary_data` |

---

## 1. 一次性环境准备（首次 / 重建 venv 后才需要）

这些是本机调试中确认必须的修复。**如果环境已修好，可跳过本节直接到 §2。**
快速自检命令（全部成功即可跳过）：

```bash
docker exec -it robot_test bash
source /root/tvenv/bin/activate
cd /root/SuperInference
python -c "import nmxrdk, nmx_nedf_api, rerun, numpy, imageio, h5py, cv2; \
print('numpy', numpy.__version__); print('all import OK')"
python -c "import r1_async_data_saver_NEDF3_x1; print('saver import OK')"
```

期望：`numpy 1.26.4` + 两个 `OK`。若有 `ImportError`，按下表修：

| 症状 | 根因 | 修复 |
|------|------|------|
| `ImportError: cannot import name 'GripperMaxForce' from 'nmx_msg.Gripper_pb2'` | venv 里手装的 `nmx_msg 2.2.0` 盖住了系统匹配 `nmxrdk 2.2.8` 的 `nmx_msg 2.4.4` | `pip uninstall -y nmx_msg`（venv 开了 system-site-packages，会回退到系统 2.4.4） |
| saver 崩在 `pynput ... failed to acquire X connection` | 容器 headless 无 X server，`pynput` 键盘后端导入即失败 | **已在代码层修复**（`r1_async_data_saver_NEDF3_r1.py`、`consumer/data_saver.py` 把 pynput 改为可选导入）。无需键盘，用 Quest 手柄 A/B 录制 |
| `ModuleNotFoundError: imageio / h5py` | 缺依赖 | `pip install imageio h5py` |
| `ModuleNotFoundError: rerun` | 缺依赖 | 见下方 numpy 冲突说明 |
| 装 rerun 后 `numpy` 被顶到 2.x，`nmx_nedf_api` 报错 | 最新 `rerun-sdk` 要求 numpy≥2，而 `nmx_nedf_api`/`nmxrdk` 要求 numpy<2 | 装支持 numpy<2 的旧版：`pip install "numpy==1.26.4" "rerun-sdk==0.23.1"` |

> **可忽略的 pip 警告**（不影响运行）：
> - `nmx-nedf-api requires nmx-msg==2.2.0 but you have 2.4.4` —— 故意用 2.4.4 匹配 nmxrdk，2.4.4 是超集。
> - `opencv-python-headless requires numpy>=2 but you have 1.26.4` —— 元数据 pin 过严，cv2 实测配 numpy 1.26.4 正常。

---

## 2. 清理（每次启动前都做）

```bash
bash /root/cleanup_x1.sh
```

杀残留进程、释放 8012/8080、清 `/dev/shm`，最后打印验证。

> 注：容器内**没有 `fuser`**，`cleanup_x1.sh` 已改为用 `/proc/net/tcp` 解析端口占用 PID
> 来释放，并额外清理 vuer 泄漏的 `multiprocessing-fork` / `resource_tracker` 进程
> （它们不带脚本名，旧版按脚本名 grep 杀不掉，会一直占着 8012）。

---

## 3. 三终端启动流程

每个新终端都先：

```bash
docker exec -it robot_test bash
source /root/tvenv/bin/activate
cd /root/SuperInference
```

> **不要用 `nohup ... &` 把 launcher 丢后台**：它是「前台阻塞 + Ctrl+C 优雅退出」设计，
> 后台跑会看不到子进程报错，且常表现为「起来又秒退」。前台跑日志直观。

### 终端 1 — Launcher（必须最先启动）

拉起 `X1Robot + X1VuerTeleoperator + 3×TjCameraDevice + DeviceManager`，产出 `device_summary_data`。

```bash
python -u incubator/teleoperation/x1/run_x1_teleop_devices.py \
    --arm-side both --fixed-frame --collect \
    --webrtc-src https://192.168.0.234:8080/offer
```

启动成功的标志：

```
X1Robot: homing done / Robot X1Robot_1 server started successfully
TjCameraDevice cam_id=CAMID_0, role=head, available=True      ← 头部相机在线
DeviceManager ... Loaded 5 devices / Created summary SHM
[launch] all processes up (...); press Ctrl+C to stop.
```

> - 默认会真实驱动机器人（`home_on_start`，双臂回初始位、头部归位）。**首次/不确定时**先加
>   `--no-home` 或先 `--arm-side left` 单臂验证，e-stop 在手边。
> - 夹爪/头部默认开启（关闭用 `--no-grippers` / `--no-head`）。
> - 左右腕相机未装物理硬件时会 `probe timed out → black frames`，**属正常**，不影响链路。
> - 反复刷 `[diag] NO HAND_MOVE ... enter VR` 是**正常**的——还没人进 VR、手柄无数据，
>   进 VR 唤醒手柄后即停。

### 终端 2 — WebRTC FPV 推流（端口 8080）

```bash
python3 incubator/teleoperation/x1/tj_camera_webrtc.py --https --shm-name TjCameraDevice_2_data
```

看到 `Running on https://0.0.0.0:8080` 即正常，**保持开着**。

### 终端 3 — NEDF3 数据采集（确认 VR 能操控后再开）

```bash
python -u r1_async_data_saver_NEDF3_x1.py \
    --config configs/quest3_collect_x1.yaml \
    -o saved_data -t data_collection -r X1-nmxrdk
```

启动成功标志：`Connected to summary SHM` + `Waiting for leader robot button press to start episode...`。
那条 `pynput unavailable ... keyboard 'D'-to-discard disabled` 是**预期 warning**，忽略。

---

## 4. Quest3 操作

详见 `SOP_Quest_TJ_Teleop.md` §4。X1 关键点一致：

1. 先放行视频：`https://192.168.0.234:8080/` → 高级/继续 → **Start WebRTC**
2. 再进 VR（**必须带 `?ws=`**）：`https://192.168.0.234:8012?ws=wss://192.168.0.234:8012` → 高级/继续 → **Enter VR**

| 操作 | 按键 |
|------|------|
| 跟随/离合 | 侧握键 grip（按住跟随，松开暂停，再按重捕基准） |
| 末端平移/旋转 | 握离合移动/转动手柄 |
| 夹爪 | 扳机 trigger |
| 头部 | 头显朝向（yaw/pitch） |
| **录制开始/结束** | **右手柄 A 键** |
| **丢弃当前段** | **右手柄 B 键** |

---

## 5. 数据落盘

| 视角 | 路径 |
|------|------|
| 容器内 | `/root/SuperInference/saved_data/` |
| 宿主机 | `/home/nvidia/nmx_hal/SuperInference/saved_data/` |

每段录制是一个 UUID 目录：

```
saved_data/<uuid>/
├── metadata.json                  # 任务/机器人型号/相机映射/device_type
└── data/
    ├── data_<ts>_0.mcap           # NEDF3 数据本体（图像+双臂+夹爪+头部）
    └── mcap_index.json
```

- 带 `_temp` 后缀 = 未正常 finalize 的段（还在录或异常退出），可酌情清理。
- 一段约 30s 的录制 ≈ 77MB，含 22 个话题：双臂 tcp/cmd+state+wrench、双夹爪 cmd+state、
  头部 tcp/joint、3 路相机 rgbd（臂/头 ~59Hz，cmd ~49Hz，相机 ~30fps）。
- 改保存位置用 `-o`，例如 `-o recordings/0623_taskA`。

### 快速校验某段数据（话题消息数 + 时长）

```bash
python - <<'PY'
from mcap.reader import make_reader
from collections import Counter
import glob
p = sorted(glob.glob("saved_data/*/data/*.mcap"))[-1]   # 最新一段
c, t0, t1 = Counter(), None, None
with open(p, "rb") as f:
    for sch, ch, msg in make_reader(f).iter_messages():
        c[ch.topic] += 1
        t0 = t0 or msg.log_time; t1 = msg.log_time
for topic, n in sorted(c.items()):
    print(f"{n:6d}  {topic}")
print("duration(s):", (t1 - t0) / 1e9, " total:", sum(c.values()))
print("file:", p)
PY
```

各话题计数非 0、相机 ~30fps、duration 与录制时长一致即为有效数据。

---

## 6. 排错速查

| 现象 | 原因 | 处理 |
|------|------|------|
| launcher 起来秒退 / `[1]+ Done` | 用了 `nohup &` 后台跑，或某子进程崩溃 | 前台跑；看 `[launch] process '...' exited (exitcode=...)` 定位崩溃源 |
| `GripperMaxForce` 导入失败 → X1Robot 崩 | venv `nmx_msg 2.2.0` 盖住系统 2.4.4 | `pip uninstall -y nmx_msg`（§1） |
| `bind 8012 address already in use` | 上次 vuer 泄漏进程占着端口 | `bash /root/cleanup_x1.sh`（已支持无 fuser 释放端口） |
| saver `pynput ... X connection` | headless 无 X | 已代码层修复为可选导入；拉最新代码即可 |
| saver `ModuleNotFoundError` (imageio/h5py/rerun) | 缺依赖 | 按 §1 安装表补齐 |
| `numpy` 被顶到 2.x，nmx_nedf_api 报错 | rerun 最新版要 numpy≥2 | 钉 `numpy==1.26.4` + `rerun-sdk==0.23.1`（§1） |
| 进 VR 有手柄模型但无视频/手柄无反应 | 8012 地址没带 `?ws=` | 用带 `?ws=wss://…:8012` 的完整地址（§4） |
| 左右腕相机录到黑帧 | 腕部相机未装物理硬件 | 正常；装好后 `available=True` 即出真图 |

---

## 附：一句话回顾

1. 环境四件套：`nmx_msg` 回退 2.4.4、补 `imageio/h5py`、`rerun-sdk==0.23.1` 配 `numpy==1.26.4`、pynput 已改可选。
2. 顺序：`cleanup_x1.sh` → 终端1 launcher（等 `device_summary_data`）→ 终端2 WebRTC → Quest 进 VR 验证 → 终端3 采集。
3. 三个终端都**前台**跑，Ctrl+C 退出；数据在 `saved_data/<uuid>/`，A 键起停、B 键丢弃。
