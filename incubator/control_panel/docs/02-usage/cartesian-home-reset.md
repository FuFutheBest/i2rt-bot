# Direct joint Home reset / 双臂直接关节复位

Rizon 的复位已回退为 Flexiv 原生 joint-position Home：控制面板把配置或候选姿态中的
左右臂 7D jointpos 分别作为单个目标发送，由 RDK 内部完成轨迹规划。

执行流程：

1. 暂停推理动作输出并清空旧 control SHM 数据，不停止 runtime 或 inference server。
2. 将 Rizon 切换到 joint-position 模式。
3. 左右臂各发送一次目标 jointpos，速度和加速度继续使用当前 device config。
4. 轮询真实 joint 状态；两臂均进入 `0.01 rad` 容差后认为复位完成。
5. 恢复 EEF 控制模式、更新实时状态，并继续推理与下一段 Rollout。

当前复位不再生成 Cartesian 直线 TCP waypoint，不再逐点求 IK，也不会使用旧的
Cartesian 路径步长配置。候选 Home 的离线生成仍保留，但候选结果只作为目标
jointpos 使用。

若 12 秒内未到位或模式切换失败，复位返回失败并尝试恢复 EEF 模式；设备 worker、
runtime 和 inference server 保持运行，便于用户排查后重试。
