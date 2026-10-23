# I2RT 控制与采集速览

Python + I2RT SDK + CAN。下文 topic 是 MCAP 文件通道。**

## 1. 整体链路

```mermaid
flowchart TB
    UI["Web 界面<br/>选任务、启动设备、控制录制"] --> START["device_starter.py"]
    START --> TELE["主从臂遥操作<br/>devices/robots/i2rt_teleop/"]
    START --> CAM["左右腕相机"]
    HUMAN["人移动左右主臂<br/>Leader：示教输入"] --> TELE
    TELE -->|"I2RT SDK → CAN"| ROBOT["左右从臂 Follower<br/>操作物体"]
    ROBOT -->|"实际位置反馈"| TELE
    TELE -->|"主臂、从臂状态"| SHM["共享内存 SHM"]
    CAM -->|"RGB 图像"| SHM
    SHM --> SAVER["NEDF3 录制器<br/>consumer/io/async_data_saver_i2rt_NEDF3.py"]
    UI -->|"开始 / 结束 / 丢弃"| SAVER
    SAVER --> FILE["data/任务名/episode_xxxx/<br/>MCAP + JSON"]
```

这批是**人工遥操作示范**：没有模型参与，录制器不负责驱动机器人。

## 2. 机器人怎么跟随？

下图每侧各一套，左右合计 4 个 Worker；上层目标频率 **100 Hz**。

```mermaid
flowchart TB
    L["主臂 + 手柄"] -->|"SDK / CAN 读取"| LW["Leader Worker"]
    LW -->|"get_sample：7D 状态 + 按钮"| C["协调器 coordinator.py"]
    C --> MAP["目标 = 主臂状态 × joint_directions<br/>逐轴 ±1 符号映射"]
    MAP -->|"submit_control_frame"| FW["Follower Worker"]
    FW --> SDK["I2RT SDK<br/>限位、坐标映射、增益、重力补偿"]
    SDK -->|"CAN：位置、速度、kp、kd、前馈力矩"| F["从臂 + 夹爪"]
    F -->|"实际关节与夹爪位置"| FW
    FW -->|"get_sample：7D 状态"| C
    C -. "6D 主臂反馈目标，作用取决于增益" .-> LW
```

| RPC 数据 | 字段 |
|---|---|
| 状态 `get_sample` | `state`、`buttons`、`sampled_at_ns`、`phase`、`fault_reason`、`last_applied_sequence` |
| 命令 `submit_control_frame` | `joint_position`（从臂 7D / 主臂 6D）、`session_id`、`sequence`、`issued_at_ns`、`phase` |

> 主臂记录 ≠ 最终下发命令；从臂反馈 ≠ 目标位置。当前默认双向反馈增益为 0。

## 3. 按钮做什么？

```mermaid
flowchart LR
    BOTH["两侧顶部按钮组合按下"] --> ALIGN["从臂平滑对齐"] --> ACTIVE["开始跟随"]
    ONE["跟随时按任一顶部按钮"] --> HOLD["停止跟随 / 安全保持"]
    BAD["反馈、命令超时或跟随异常"] --> FAULT["故障停止，排查设备"]
    RIGHT["右主臂第二按钮"] --> TOGGLE["开始 / 结束录制"]
    LEFT["左主臂第二按钮"] --> DISCARD["丢弃当前录制"]
```

**跟随开关与录制开关独立；软件安全保持不等于硬件急停。**

## 4. 每份状态是什么？

```mermaid
flowchart TB
    END["单臂 7D<br/>q0…q5：rad<br/>gripper：0 闭 → 1 开"] --> MERGE["左右拼接为 14D float64"]
    MERGE --> A["索引 0–5：左臂关节<br/>索引 6：左夹爪"]
    MERGE --> B["索引 7–12：右臂关节<br/>索引 13：右夹爪"]
    A --> L["主臂一份：I2RTRobot_0_data"]
    B --> L
    A --> F["从臂一份：I2RTRobot_1_data"]
    B --> F
```

## 5. SHM → 文件 topic

```mermaid
flowchart LR
    L["I2RTRobot_0_data<br/>主臂 14D，约 100 Hz"] --> SPLIT["按左右拆分<br/>最新帧采样 + 时间戳去重"]
    F["I2RTRobot_1_data<br/>从臂 14D，约 100 Hz"] --> SPLIT
    SPLIT --> J["4 路 joint<br/>每路 6 个关节位置，90 Hz"]
    SPLIT --> G["4 路 hand_state<br/>每路 1 个开合量，30 Hz"]
    R["RealSenseCamera_3_data<br/>右腕 RGB"] --> C0["camera_0/rgbd"]
    LE["RealSenseCamera_2_data<br/>左腕 RGB"] --> C1["camera_1/rgbd"]
    BLACK["录制器合成黑图"] --> C2["camera_2/rgbd<br/>不是真实头部相机"]
    J --> WRITE["后台写入<br/>Protobuf → MCAP"]
    G --> WRITE
    C0 -->|"JPEG，约 30 FPS"| WRITE
    C1 -->|"JPEG，约 30 FPS"| WRITE
    C2 -->|"30 Hz"| WRITE
```

完整 **12 个 topic**（名称没有前导 `/`）：

| Topic | 数据 |
|---|---|
| `nmx/nedf/metadata` | 任务、设备与契约信息，每 episode 1 条 |
| `nmx/hal/i2rt/leader/left/joint` | 左主臂关节 |
| `nmx/hal/i2rt/leader/right/joint` | 右主臂关节 |
| `nmx/hal/i2rt/follower/left/joint` | 左从臂关节 |
| `nmx/hal/i2rt/follower/right/joint` | 右从臂关节 |
| `nmx/hal/i2rt/leader/left/hand_state` | 左主臂手柄开合 |
| `nmx/hal/i2rt/leader/right/hand_state` | 右主臂手柄开合 |
| `nmx/hal/i2rt/follower/left/hand_state` | 左从臂夹爪反馈 |
| `nmx/hal/i2rt/follower/right/hand_state` | 右从臂夹爪反馈 |
| `nmx/hal/camera/camera_0/rgbd` | 右腕 RGB |
| `nmx/hal/camera/camera_1/rgbd` | 左腕 RGB |
| `nmx/hal/camera/camera_2/rgbd` | 头部黑图占位 |

```text
joint      → ArmState.joints[0…5] → name + pos（rad）
hand_state → HandState.joints[0]  → name="gripper" + pos（0…1）
rgbd       → RGBD.rgb            → JPEG；depth 为空
```

## 6. 你采了哪些？

| 任务 | 示范数 | 总时长 |
|---|---:|---:|
| `put_banana_in_bowl` | 31 | 8.56 分钟 |
| `put_blue_block_in_blue_cup` | 50 | 15.27 分钟 |
| **合计** | **81** | **23.83 分钟** |

```mermaid
flowchart TB
    DATA["这批采集数据"] --> YES["已记录"]
    DATA --> NO["未记录"]
    YES --> STATE["四臂关节位置 + 四路夹爪量"]
    YES --> IMG["两路真实腕部 RGB<br/>共 85,680 帧"]
    YES --> PLACE["头部黑图占位<br/>42,919 帧"]
    NO --> SENSOR["真实头部画面、深度、点云"]
    NO --> DYN["速度、加速度、力矩、电流、温度"]
    NO --> CMD["完整下发命令、CAN 帧<br/>按钮与控制阶段事件"]
    NO --> TCP["独立 TCP 位姿与模型预测动作"]
```

> 统计来自 81 份元数据与统计文件；两个任务各抽查 1 个 MCAP。样例图像为 **640×480 JPEG**，深度为空。统计均未报告丢批或写入错误，但不代表示范全部成功。

## 7. 看数据时只记住三点

- **按时间戳对齐，不按行号**：关节 90 Hz，夹爪/图像约 30 Hz；SHM 时间不代表硬件同步。
- **默认 0 不等于测量值**：消息里的速度、力矩等字段没有写入。
- **配置名不等于历史配置快照**：不要用当前 CAN 编号或旧配置的设备 ID 推断现场接线。

```text
episode_xxxx/
├── metadata.json           任务与设备说明
├── episode_stats.json      时长、消息数、写入错误
└── data/
    ├── mcap_index.json     实际分片索引
    └── data_*.mcap         图像与状态消息
```

源码入口：[协调器](../../devices/robots/i2rt_teleop/coordinator.py) · [硬件端点](../../devices/robots/i2rt_teleop/endpoints.py) · [录制器](../../consumer/io/async_data_saver_i2rt_NEDF3.py) · [topic 契约](../../consumer/io/nedf3_i2rt_contract.py)
