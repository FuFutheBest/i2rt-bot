# 采集数据图解

> 本地数据快照：**I2RT YAM 双臂 · 人工遥操作 · NEDF3 / Protobuf / MCAP · i2rt-nedf3-v2**。
> 全量检查 81 份 metadata、统计与分片数量；两个任务各抽查 `episode_0001` 的 MCAP。未启动硬件。

## 1. 数据规模

```mermaid
flowchart TB
    ALL["81 条示范<br/>23.83 分钟<br/>MCAP 合计约 3.97 GB（十进制）"]
    ALL --> A["put_banana_in_bowl<br/>把桌子上的香蕉放到碗里<br/>31 条 · 8.56 分钟"]
    ALL --> B["put_blue_block_in_blue_cup<br/>把正确的蓝色方块放进蓝色杯子<br/>50 条 · 15.27 分钟"]
    A --> AD["episode_0000 → episode_0030<br/>单条 11.48–24.23 秒<br/>平均 16.57 秒"]
    B --> BD["episode_0000 → episode_0049<br/>单条 14.39–24.80 秒<br/>平均 18.32 秒"]
```

## 2. 这些数据从哪里来？

```mermaid
flowchart TB
    HUMAN["人操作左右主臂 Leader"] --> INPUT["主臂关节角 + 手柄开合<br/>示教输入"]
    INPUT --> CONTROL["Python 遥操作协调器<br/>符号映射、对齐、命令检查"]
    CONTROL -->|"I2RT SDK → CAN"| ROBOT["左右从臂 Follower<br/>执行任务"]
    ROBOT --> FEEDBACK["从臂关节角 + 夹爪开合<br/>实际反馈"]
    INPUT --> L["I2RTRobot_0_data<br/>主臂共享内存"]
    FEEDBACK --> F["I2RTRobot_1_data<br/>从臂共享内存"]
    CAM["两路真实腕部相机"] --> C["RealSenseCamera_3_data：右腕<br/>RealSenseCamera_2_data：左腕"]
    L --> REC["录制器旁路读取 SHM<br/>不负责驱动机器人"]
    F --> REC
    C --> REC
    BLACK["合成 head 黑图"] --> REC
    REC --> MCAP["12 个文件 topic<br/>不是 ROS 2 实时话题"]
```

**主臂状态 ≠ 最终下发命令；从臂状态是反馈，不是目标。** 本批没有模型参与。

## 3. 目录里每个文件做什么？

```text
data/
├── data_collection_guide.md       本文
├── put_banana_in_bowl/
│   └── episode_0000/ ... episode_0030/
└── put_blue_block_in_blue_cup/
    └── episode_0000/ ... episode_0049/
        ├── metadata.json         任务、硬件描述、相机角色、契约
        ├── episode_stats.json    时长、topic 数量、丢批、写入错误
        └── data/
            ├── mcap_index.json   分片文件名、时间范围、topic 列表
            └── data_<ns>_0.mcap  真实消息；当前每条示范一个分片
```

```mermaid
flowchart LR
    META["metadata.json<br/>知道这是什么数据"] --> IDX["data/mcap_index.json<br/>找到实际 MCAP 分片"]
    IDX --> FILE["MCAP<br/>通道 + 内嵌 Protobuf schema + 消息"]
    FILE --> DECODE["按 schema 解码<br/>按 topic 与时间戳使用"]
    STATS["episode_stats.json"] -. "核对数量与写入状态" .-> FILE
```

`metadata.mcap_file_name` 虽写 `data.mcap`，**实际文件以索引为准**；不要硬编码这个文件名。

## 4. 机器人状态：左右、主从、单位

```mermaid
flowchart TB
    L["Leader：人手操作的示教臂"] --> LS["左右各 7D"]
    F["Follower：操作物体的执行臂"] --> FS["左右各 7D"]
    LS --> D["每臂 7D<br/>6 个关节角 rad + 1 个夹爪量"]
    FS --> D
    D --> MERGE["每组主臂 / 从臂各拼成 14D float64"]
    MERGE --> LEFT["[0:6] 左臂 joint0…5<br/>[6] 左夹爪"]
    MERGE --> RIGHT["[7:13] 右臂 joint0…5<br/>[13] 右夹爪"]
    D --> GRIP["夹爪 / 手柄量 ∈ [0,1]<br/>0 闭，1 开<br/>不是米、毫米或电机角度"]
```

14D 是**源 SHM 布局**；写文件后拆成左右关节与夹爪的独立 topic，文件中不是每条消息都带完整 14D。

## 5. 全部 12 个 topic

```mermaid
flowchart LR
    FILE["每条 episode：12 个通道"] --> ARM["4 × ArmState<br/>每条 6 个关节位置<br/>目标 90 Hz"]
    FILE --> HAND["4 × HandState<br/>每条 1 个开合量<br/>目标 30 Hz"]
    FILE --> CAM["3 × RGBD<br/>2 路真 RGB + 1 路黑图<br/>约 30 Hz，depth 空"]
    FILE --> META["1 × Metadata<br/>每条 episode 1 次"]
```

名称**没有前导斜杠**：

| 完整 topic | 来源 / 含义 |
|---|---|
| `nmx/nedf/metadata` | 版本与 episode 元信息 |
| `nmx/hal/i2rt/leader/left/joint` | 左主臂 6 关节 |
| `nmx/hal/i2rt/leader/right/joint` | 右主臂 6 关节 |
| `nmx/hal/i2rt/follower/left/joint` | 左从臂 6 关节 |
| `nmx/hal/i2rt/follower/right/joint` | 右从臂 6 关节 |
| `nmx/hal/i2rt/leader/left/hand_state` | 左主臂手柄开合输入 |
| `nmx/hal/i2rt/leader/right/hand_state` | 右主臂手柄开合输入 |
| `nmx/hal/i2rt/follower/left/hand_state` | 左从臂夹爪开合反馈 |
| `nmx/hal/i2rt/follower/right/hand_state` | 右从臂夹爪开合反馈 |
| `nmx/hal/camera/camera_0/rgbd` | 右腕真实 RGB |
| `nmx/hal/camera/camera_1/rgbd` | 左腕真实 RGB |
| `nmx/hal/camera/camera_2/rgbd` | head 黑图占位 |

### 消息字段

```mermaid
flowchart LR
    A["nmx.msg.ArmState"] --> AJ["joints：6 项"] --> AV["name = joint0…joint5<br/>pos = 关节角 rad"]
    H["nmx.msg.HandState"] --> HJ["joints：1 项"] --> HV["name = gripper<br/>pos = 归一化开合量"]
    M["nmx.msg.Metadata"] --> MV["version = i2rt-nedf3-v2<br/>metadata = JSON 字节"]
    A -. "未写入" .-> MISS["vel / acc / effort"]
    H -. "未写入" .-> TARGET["target_reached"]
```

**Protobuf 默认 0 / false 不代表实测值。** schema 中有某个字段或类型，也不代表采集过相应数据。

## 6. 图像到底是什么？

```mermaid
flowchart TB
    R["右腕 RealSenseCamera_3_data"] --> C0["camera_0<br/>主预览 main_camera<br/>42,855 帧"]
    L["左腕 RealSenseCamera_2_data"] --> C1["camera_1<br/>42,825 帧"]
    BLACK["无真实头部输入"] --> C2["camera_2<br/>42,919 帧合成黑图"]
    C0 --> RGB["RGBD.rgb<br/>640×480，3 通道（样例）<br/>JPEG 有损编码"]
    C1 --> RGB
    C2 --> RGB
    RGB --> EMPTY["RGBD.depth 为空<br/>没有深度、点云"]
```

```text
共享内存：RGB uint8 [H,W,3]
    ↓ RGB → BGR（仅适配 OpenCV 编码接口）
JPEG 编码 → RGBD.rgb.data
    ├── rows / cols / channels：图像尺寸
    ├── type：ENCODED
    └── encoded_format：jpeg

RGBD.depth：rows=cols=channels=0，data 为空
RGBD.depth_scale=1.0：不代表记录了深度
```

当前录制器保留原始尺寸、不裁剪缩放，默认 JPEG quality 90；样例元数据标注图像变换为 `none`。v2 不需要历史 v1 的红蓝通道修复；用 OpenCV 解码得到 BGR 时，再按使用方要求转换 RGB。

## 7. 采样与时间对齐

```mermaid
flowchart LR
    ROB["机器人 SHM<br/>目标 100 Hz"] --> READ["读取最新稳定快照<br/>按源时间戳去重"]
    READ --> J["关节目标 90 Hz"]
    READ --> G["夹爪目标 30 Hz"]
    CAMERA["真实相机约 30 FPS"] --> NEW["发现新时间戳才保存"]
    J --> Q["队列 → 后台写入线程"]
    G --> Q
    NEW --> Q
    Q --> TIME["MCAP log_time = publish_time<br/>使用源 SHM timestamp_ns"]
```

```mermaid
flowchart TB
    CLOCK["两种主机时间"] --> MONO["单调时钟 monotonic_ns<br/>控制命令有效期、反馈超时、采样调度<br/>不能直接与 MCAP 时间相减"]
    CLOCK --> UNIX["Unix epoch 纳秒 time_ns<br/>机器人 SHM 发布、相机主机时间、MCAP"]
    UNIX --> SYNC["各流按时间戳匹配 / 插值<br/>不要按消息行号拼接"]
```

- 同一 14D 快照拆出的左右消息共享时间；机器人时间是**协调器发布时间**，不代表四臂硬件同时采样。
- 相机使用主机 Unix 时间，不是 RealSense 原始设备相对毫秒；图像和机器人没有硬件同步保证。
- 100 → 90 Hz 本身就是选帧；`dropped_batches=0` 不代表全部源帧都保存。
- `duration_s` 来自 episode 开始到结束收尾的主机时间，可能含排空队列/关闭 writer 的耗时；不必等于 MCAP 首末消息时间差。

## 8. 消息数量

| 数据流 | 香蕉任务 | 蓝块任务 | 合计 |
|---|---:|---:|---:|
| 每个独立 joint topic | 46,226 | 82,450 | 128,676 |
| 4 个 joint topic 合计 | 184,904 | 329,800 | 514,704 |
| 每个独立 hand_state topic | 15,416 | 27,503 | 42,919 |
| 4 个 hand_state topic 合计 | 61,664 | 110,012 | 171,676 |
| camera_0 右腕 | 15,400 | 27,455 | 42,855 |
| camera_1 左腕 | 15,384 | 27,441 | 42,825 |
| camera_2 黑图 | 15,416 | 27,503 | 42,919 |
| metadata | 31 | 50 | 81 |

**真实图像 85,680 帧。消息数不是训练样本数；一条 joint 消息有 6 个位置值。**

## 9. 元信息怎么看？

```mermaid
flowchart TB
    M["metadata.json"] --> TASK["task_name / task_description<br/>任务标识与自然语言描述"]
    M --> ROB["collection_mode = teleoperation<br/>robot_vendor_model = I2RT_YAM<br/>robot_type = dual_arm"]
    M --> CAMERA["camera_info：0 右腕 / 1 左腕 / 2 head<br/>main_camera = camera_0"]
    M --> EXTRA["extra"]
    EXTRA --> CONTRACT["contract_version / state_layout<br/>sample_hz：90 / 30 / 30"]
    EXTRA --> SOURCE["hardware_string：源 SHM 名称<br/>synthetic_black_cameras：head"]
    EXTRA --> COLOR["camera_color_space / camera_processing<br/>RGB 与无几何变换说明"]
    EXTRA --> USER["采集者标识、任务执行 ID<br/>分享前脱敏"]
    M --> TRANS["基座到世界、UMI 到 TCP 变换<br/>抽查样例为空，不视为完整标定"]
```

`hardware_config_name=i2rt_dual_bilateral` 是写入器固定描述，**不是完整启动配置快照**；不能据此还原 CAN 接线、夹爪参数或 SDK 版本。

## 10. 录制如何变成一个完成的 episode？

```mermaid
flowchart LR
    START["右侧第二按钮 / 界面开始"] --> TEMP["episode_xxxx_inprogress"]
    TEMP --> WRITE["采样 → 有界队列<br/>JPEG / Protobuf 写入"]
    WRITE --> STOP["结束录制<br/>排空队列、关闭文件、写统计"]
    STOP --> CHECK{"有写入错误<br/>或某 topic 为零？"}
    CHECK -->|"否"| DONE["episode_xxxx<br/>完成目录"]
    CHECK -->|"是"| QUAR["quarantine/<br/>保存错误原因"]
    TEMP -->|"丢弃"| DELETE["删除本次目录"]
    TEMP -. "中断后下次恢复检查" .-> QUAR
```

```text
episode_stats.json
├── contract_version               录制契约
├── started_at_ns / ended_at_ns     Unix 纳秒
├── duration_s                     录制生命周期时长
├── topic_counts                   各 topic 成功写入数量
├── dropped_batches                队列满导致的丢批数（不等于消息数）
└── writer_error                   写入异常；null 表示未报告
```

本批 81 条均有 12 路统计、每条 1 个 MCAP 分片，`dropped_batches=0`、`writer_error=null`。这不证明任务成功、无遮挡、无传感器漏帧；尚未逐帧验收全部 MCAP。

## 11. 使用前的边界

```mermaid
flowchart TB
    DATA["本批示范"] --> YES["可直接读取"]
    YES --> POS["主从臂关节位置<br/>主臂手柄 / 从臂夹爪量<br/>左右腕 RGB、任务描述"]
    DATA --> NO["没有保存"]
    NO --> ABSENT["真实 head、深度、点云<br/>实测速度 / 力矩 / 电流 / 温度<br/>命令帧、CAN 帧、按钮与 phase 时间线<br/>独立 TCP 位姿、模型预测动作"]
    DATA --> DERIVED["需另外定义 / 推导"]
    DERIVED --> TRAIN["统一训练频率、时间偏移、动作标签<br/>夹爪宽度换算、速度差分、正运动学<br/>任务成功标签与质量筛选"]
```

```mermaid
flowchart LR
    READ["按索引读取分片<br/>用内嵌 schema 解码"] --> SELECT["取真实左右腕图像<br/>区分 leader 与 follower"]
    SELECT --> ALIGN["按时间戳对齐<br/>明确延迟与插值"]
    ALIGN --> REVIEW["人工审核示范<br/>确认成功与异常"]
    REVIEW --> EXPORT["再转换训练数据<br/>明确 observation / action 语义"]
```

源码：[topic 契约](../consumer/io/nedf3_i2rt_contract.py) · [录制实现](../consumer/io/async_data_saver_i2rt_NEDF3.py) · [离线读取](../consumer/io/i2rt_nedf3_rerun.py) · [控制链路速览](../docs/guide/i2rt_control_and_dataset_guide.md)
