---
name: superoffline-comparison
description: 维护 SuperOffline 的 2–4 模型真实机器人离线对比、浏览器目录、录制契约和经验证运行环境边界；不用于仿真或未受控离线数据。
---

# SuperOffline 离线模型对比

当新增可比较模型、修改浏览器/API、录制来源或离线执行路径时使用本 Skill。简明启动命令见
[`superoffline_inference_commands.md`](../../../superoffline_inference_commands.md)；本地服务入口为
`scripts/analysis/superoffline.py`，模型目录位于 `utils/superoffline_registry.py`。

## 范围与入口

- 一次请求必须选择 2–4 个互不重复且状态为可用的真实机器人模型。仿真命名空间和仿真适配器永远不能成为比较项。
- 浏览器服务只负责请求校验和任务生命周期。默认执行器在各模型自己的已验证 runtime profile 中串行运行
  只读 evaluator；浏览器进程不导入 policy，不创建控制共享内存，也不下发机器人动作。
- 服务调度器最多同时运行一个比较任务；后续任务保持 `queued`，避免两个 checkpoint 同时占用 GPU。
- 输出只能写入新的 `logs/superoffline/<job-id>/` 目录。一个任务最多同时加载一个 checkpoint。
- 执行页面服务本身不会开始计算；提交任务后才会检查录制内容并串行启动 2–4 个只读 evaluator。

## 模型注册契约

- 使用显式目录，不要扫描 `POLICY_CLASSES`、接受自由输入的配置名，或从 policy 类猜测兼容性。每个选项都必须
  解析到原生配置和 `SuperOfflineModelContract`。
- FastWAM 变体只能来自
  `consumer.inference.fastwam_model_registry.FASTWAM_MODEL_VARIANTS`。变体 ID 和配置名由 registry 管理，
  UI 和 evaluator 不得复制枚举。
- 已审核的可比较目录包括普通 FastWAM、FastWAM memory、FasterWAM、OpenPi UMI3 fisheye，以及已注册的
  LingBotVA UMI3 fisheye/iPhone 配置。它们都声明
  `superoffline.dual_arm_tcp_action_16d.v1`，布局为两臂
  `[tcp_xyz_m, tcp_quaternion_xyzw, gripper_width_m]`。
- Diffusion 可以作为单模型离线诊断，但当前 iPhone-Dahuan rig 缺少已审核的 TCP/UMI 与基座到世界坐标标定包。
  因此它没有 SuperOffline 模型契约，不能绑定 runtime profile 或进入浏览器任务。
- 页面允许任意 2–4 个已注册可用模型的组合，不要求同一模型家族。但同一份录制必须满足所有选中配置各自审核过的来源
  profile；不同相机 rig 共用录制前必须有审核过的输入转换。

## 可信输入与录制契约

共同 preflight 会组合两个精确的原生配置，并在选择运行环境前失败关闭。它验证 `policy.obs.devices` 的
left/right/robot 顺序；FastWAM 和 LingBotVA 还必须与
`policy.params.camera_devices`、`robot_device` 的顺序一致。不要允许 evaluator 从请求中选择设备键。

输入必须来自一个已完成的 `TimestampedObservationRecorder`，并具有 `manifest.json`、`frames.jsonl`、
`inference.jsonl` 和所有被引用的 NPZ 文件。来源必须是真机数据，包含完成的计数与 `stopped_at_ns`，并携带
recorder observation contract、审核来源 profile、来源描述和角色元数据。preflight 会检查索引路径、数组映射、
dtype、shape 和根目录约束，然后才允许模型加载。

来源描述从组合后的真机配置推导，不能由调用者提供。它记录相机的 class/id/shape/dtype/type/rotation、机器人的
class/id/serial/state shape/dtype/layout，以及审核标定文件的摘要。启用 FastWAM observation recorder 时，只有
非 mock 路径可以生成该描述；独立采集命令只接受所选设备的 `<device>_data` SHM 名，不能通过别名选取任意兼容数组。

OpenPi 普通/RTC 与 FastWAM 普通/RTC/memory/FasterWAM 六个原生配置都默认启用
`policy.observation_recorder`，显式设置 30 Hz、512 队列和 `compress: false`。
输出到仓库下的 `logs/offline_observations/<变体>/run_<timestamp>/`，变体目录依次为
`openpi`、`openpi_rtc`、`fastwam`、`fastwam_rtc`、`fastwam_memory`、`fasterwam`。
使用仓库根目录作为 runtime profile 的 cwd；需要其他磁盘时显式覆盖 `output_dir`。
页面递归扫描这些目录，也保留旧 `logs/fastwam_observations` 和旧根目录下的录制发现。
OpenPi 启动器从 `policy.obs.devices` 的左相机、右相机、机器人顺序提取 recorder 设备，并从组合配置生成
来源描述；这些设备参数只传给 recorder，不加入 OpenPi 模型构造参数。配置字段现在为
`superoffline_reviewed_source_profile`，明确只服务于 SuperOffline 数据契约。
新的 OpenPi/FastWAM 配置使用 `rizon-umi3-fisheye-openpi-rgb.v2`，左右腕角色从配置观察顺序取得；
设备键来自连续的 `device_class_id`，不来自 USB capture ID。v2 不固定设备数字后缀，
但仍验证设备类、图像格式、旋转、机器人状态布局及标定摘要。实际机器人序列号记录在 provenance，
不要求与审核模板或另一台机器人相同。不可把不同机器的序列号替换成模板值。
旧 v1 profile、manifest 的 `source_profile` 及描述中的 `device_id` 只在历史数据读取边界兼容，
保留旧摘要和设备键；不可用新 ID 重写已有录制。新 v2 角色必须与 manifest 来源描述一致。
录制随 policy 进程运行，正常退出后才完成 manifest；它独立于 NEDF infer_rollout
的 episode 开关。

录制器在 policy 初始化成功并连接 summary SHM 后才开始。仅启动 FastWAM server
不会录制；模型加载失败也不会产生这一轮 observation 数据。启动日志打印实际绝对目录，
正常关闭时打印帧数、推理输入数、丢帧数和写入失败数。未填写 `output_dir` 时历史默认路径为
`logs/fastwam_observations`，并非自动关闭。排查“没保存”先检查 `enabled`、启动日志和
`frames.jsonl`/`inference.jsonl`；运行中的 manifest 尚无完成计数，页面只列出已结束的记录。
使用正常 Ctrl-C 退出 policy 以排空队列并完成 manifest，不能将强制杀进程后的目录当作完整录制。

这是受信本地运行环境中的配置证明，不是对恶意本地写入者的硬件证明。能够替换同名本地 SHM 或改写录制目录的主体
已经处于受信计算基内。若要防御该威胁，需要设备管理器输出实际序列号/硬件标识的运行时清单，以及受保护的签名或 IPC
权威；不要把当前摘要描述为抵抗本地攻击的证据。

observation binding 摘要只用于完整性校验，不是授权。`read_observation_binding()` 会通过目录重新解析配置，检查
审核过的来源 profile、目标角色、recorder 来源标定和注册的动作标定。即使能重算摘要，手工写入的 binding bundle
也不能自动成为可信输入。

## 动作语义与标定

每个模型契约绑定模型输出到共同 TCP 动作坐标系的转换。binding 必须包含标定 ID、审核文件摘要，以及影响输出的
所有 policy 标量。FastWAM 绑定夹爪最大值、裁剪和归一化；OpenPi 绑定 TCP/UMI 文件、`gripper_open_max` 与
两个夹爪 offset。preflight 验证组合配置后创建 job snapshot，绑定任务的 evaluator 只能使用该 snapshot。相对标定
路径必须相对于仓库根目录解析，不能依赖进程 CWD。

每个 evaluator summary 都要声明动作契约、布局、NPZ 成员和正值 `action_step_interval_ns`。执行器对平移和夹爪
做线性重采样，对四元数做 SLERP，并重采样到较粗的共同时间基。不要直接比较原始 pose 向量范数，因为它混合了米、
四元数和夹爪宽度。

## 运行环境边界

The browser service itself is a separate `superoffline` runtime role. Register a
verified environment that can import `scripts.analysis.superoffline`, `numpy`,
and `plotly`, bind it as the default `superoffline` role, and start the page through
`runtime_profiles.py run`. Do not invoke `python3 scripts/analysis/superoffline.py`
directly: the service rejects an unmanaged entrypoint before loading its offline
dependencies.

The page serves Plotly from the installed Python package at `/plotly.min.js`; it
does not use a CDN. After environment setup, require that endpoint to return 200.
If the trajectory API returns data but the 3D and 2D panels are blank, check this
endpoint and the `plotly` profile import first. The renderer intentionally does
not attempt to synthesize trajectory plots without the browser bundle.

`GET /api/healthz` is a dependency-free liveness endpoint. `GET /api/service-info`
also reports the read-only evaluator catalog and bounded scheduler state without
exposing interpreter paths, checkpoints, or task prompts.

The browser page intentionally follows the FastWAM comparison workspace layout:
`Observation Input`, `Checkpoints`, and `Run Inference` are the three setup
columns, followed by `Job Log`, recent jobs, and comparison results. These are
presentation and workflow parity only; the page submits two to four distinct
verified models to the read-only SuperOffline scheduler and does not start
hardware, a model server, or a simulator. Checkpoint discovery only supplies
local path suggestions. The FastWAM/OpenPi root fields are discovery roots, not
checkpoint selections: each row uses `Choose discovered checkpoint...` or an
explicit path. Every selection must still resolve through the registered
model/config catalog and its runtime profile contract.

任务状态 API 使用真实执行事件，不允许页面伪造固定百分比。job 公开 `stage`、`progress`、
`latest_checkpoint`、`latest_observation`、`heartbeat_at`、有界 `logs` 和 `warnings`。
内置 runner/executor 可选接收 `event_callback` 与 `cancel_event`；旧的注入 runner 仍可只实现
`run(request)`。真实 evaluator 必须使用流式子进程，把 stdout/stderr 同时写入任务日志、终端和
脱敏后的浏览器 Job Log。运行中取消先终止整个 evaluator 进程组，超时后强制清理，不能只把网页
状态改成 cancelled 而留下模型进程。

`Checkpoint Outputs` 只允许展示 evaluator 声明的 `input_preview` 和
`video_prediction`。`action_channels`、`absolute_xyz`、`trajectory_3d` 等静态曲线不得通过遍历
artifact 字典进入 checkpoint 卡片；动作轨迹统一由 canonical action artifact 驱动 Plotly 3D
和左右手二维视图。这样新增诊断文件不会意外改变页面信息结构。
内置 SuperOffline FastWAM evaluator 通过 `--skip-diagnostic-plots` 完全跳过这些未展示 PNG 的生成，
不能只在前端隐藏后继续消耗执行时间和磁盘。

FastWAM prompt conditioning 优先复用预计算 T5 context。evaluator 按格式化 prompt 的 SHA-256 查找
`.superinference/local/fastwam-text-cache/`、显式环境覆盖、checkpoint 训练配置的
`text_embedding_cache_dir`（含 `/mnt/workspace` 的本机映射）、当前 checkout 的历史比较缓存，以及同级
checkout 的历史比较缓存。发现的 payload 必须验证 `context/context_mask` 形状后才会原子导入本机 Git 忽略
缓存；命中时强制 `load_text_encoder=false`，把 context 注入 runtime prompt cache。未命中时允许已配置的
文本编码器生成 context，并在首次推理后自动持久化，后续任务不再加载 UMT5-XXL。不要把发现到的主机路径
写回受 Git 跟踪配置。

执行器维护精确结果缓存。缓存键包含 recorder provenance/artifact digest、每个 observation binding digest、
采样索引、完整模型请求、checkpoint 关键文件身份和全部 metadata。只有 `comparison.json` 及每个 canonical
action artifact 仍完整时才复用；命中返回新的浏览器 job，但引用原完成目录，不创建重复 snapshot，也不启动
doctor 或 evaluator。prompt、checkpoint、录制、标定、采样或推理参数变化必须形成不同键。runner 在每次
执行前仍重新完成一次共同 preflight 和 runtime profile 解析，并把同一份 `PreparedSuperOfflineInput` 交给
内置 executor，避免在同一执行阶段重复校验源录制。

FastWAM checkpoint 的训练 task 按 `datasets.vla_data.override_instruction`、
`datasets.vla_data.task`、`framework.fastwam.task`、顶层 `task` 的顺序读取。它与运行 prompt
归一化后不一致时返回结构化 `task_mismatch` warning：验证仍为 `valid: true`，页面和 Job Log
以红色显示，终端打印 warning，并把 requested/training task 写入结果审计信息。语义不一致本身
不得阻断推理；只有 runtime 确实无法编码请求 prompt 时才作为独立执行错误处理。

Checkpoint 发现展示实验目录名，不把通用的末级 step 目录（例如 `99999` 或
`step_030000`）当作模型名；提交时仍使用完整的叶子 checkpoint 路径。OpenPi 离线 evaluator
固定解析 OpenPi registry 的 `default` 配置，然后扫描所选 checkpoint 的
`assets/*/norm_stats.json` 并显式注入实际 asset；优先 `assets/default`，否则选按名称排序的
第一个可用 asset。这是运行时自动建立的归一化绑定，不会在 checkpoint 中伪造或复制
`norm_stats.json`；若完全没有可读 norm stats，`Validate input` 必须在排队和加载模型前失败。

OpenPi Pi0/Pi05 的模型原生 `action_dim` 可以是 32，而 UMI 双臂业务动作是 16 维。
`PadStatesAndActions` 在模型输入边界补零到原生宽度，`UmiDualArmOutputs` 在输出边界还原为
16 维。SuperOffline 的确定性 noise 必须使用 `policy.pi_config.model.action_dim`，最终 policy 输出
才必须严格为 `[T,16]`；不得因为共同动作契约是 16 维就拒绝原生 32 维模型，也不得手工重复
调用 `PadStatesAndActions`。

The workspace discovery endpoints are read-only: `/api/observations` lists only
finalized timestamped recorder manifests, `/api/observations/samples` reads the
validated inference index, and `/api/checkpoints` reports recognizable local
FastWAM/OpenPi checkpoint layouts for picker suggestions. Completed jobs expose
`/api/jobs/<id>/trajectory`, which reads only the evaluator-declared canonical
action artifact and returns bounded TCP position samples for the SVG result view.
Recorder image previews are generated from the selected finalized inference NPZ
through `/api/observation/preview`; comparison PNG artifacts are served only
under the corresponding completed job output root through
`/api/jobs/<id>/artifact/...`. Neither endpoint accepts an arbitrary file path
for direct download.

Run scope is explicit: `all` evaluates every recorded inference, `sample`
evaluates the selected inference, and `stride` evaluates indices
`[0, N, 2N, ...]` for a positive `sample_stride`. The output mode is also
explicit. `trajectory_only` requests only the comparable action chunk;
`trajectory_video` additionally asks the FastWAM evaluator for its decoded video
prediction. OpenPi and LingBotVA remain action-only and must not synthesize a
placeholder video. FastWAM and OpenPi evaluators receive the selected indices as
one batch, keep the selected checkpoint resident, and write each sample below
`model_<n>/observations/<index>/`; do not move model loading back inside the
per-observation loop.

FastWAM prediction artifacts are served as `video/mp4` and must contain an
H.264/AVC `yuv420p` stream. A valid MP4 container carrying OpenCV `mp4v`
(MPEG-4 Part 2) is not an acceptable browser artifact; Chrome and Edge may report
that no supported format or MIME type was found even though the HTTP MIME type is
correct.

Completed results expose a sample navigator, exact recorder previews, evaluator
images/videos, synchronized video playback, a combined 3D trajectory, separate
left/right 2D projections, arm filtering, and a visualization-only zero-origin
toggle. `GET /api/jobs/<id>/smoothness` aggregates each checkpoint across all
successful chunks. It computes per-arm finite differences only within each
native action chunk, reports translation speed/acceleration/jerk in units per
predicted step, and ranks the unitless roughness
`(RMS(acceleration) + RMS(jerk)) / (RMS(speed) + 0.1 mm)`. Quaternion and gripper
components are excluded from this metric.

内置执行器的角色对应关系如下：

- FastWAM 使用 `evaluate_fastwam_reverse_offline.py`，角色为 `fastwam-server`。
- OpenPi 使用 `evaluate_openpi_offline.py`，角色为 `policy`。
- LingBotVA 使用 `evaluate_lingbotva_offline.py`，角色为 `policy-server`，并由 evaluator 管理临时 loopback server。

实际比较前，先对所有精确配置和对应角色完成 profile 的 register、bind、verify 和 `doctor`。只检查已解析命令时使用
`command --json`；实际执行由 SuperOffline 通过 `run --role <role>` 完成。不要在 profile 注册表中写入 Python 路径、
环境名、checkpoint 或任务文本。

可选替代执行器必须通过 `--executor module:object` 提供对象，而不是函数。对象需要实现
`validate_request(request)` 和 `run(request, contexts)`。`RuntimeProfileComparisonRunner` 始终会在对象的额外验证、
profile 解析和执行前运行共同 recorder/config/action preflight；外部执行器不能绕过它。替代执行器的 `run` 方法同样
必须只读，并且一次最多加载一个 checkpoint。

FastWAM 相关改动还要读取
[`fastwam-inference-runtime`](../fastwam-inference-runtime/SKILL.md)；所有 profile 操作遵循
[`runtime-profile-management`](../runtime-profile-management/SKILL.md)。

## 验证

修改目录、录制契约、动作契约或执行器接口后，至少运行：

```bash
python -m pytest \
  tests/utils/test_superoffline_registry.py \
  tests/utils/test_superoffline_observation.py \
  tests/utils/test_superoffline_action_calibration.py \
  tests/config/test_superoffline_contracts.py \
  tests/utils/test_openpi_checkpoint.py \
  tests/scripts/test_superoffline.py \
  tests/scripts/test_superoffline_fastwam_executor.py \
  tests/scripts/test_fastwam_reverse_offline.py \
  tests/scripts/test_openpi_offline.py \
  tests/scripts/test_lingbotva_offline.py \
  tests/consumer/test_fastwam_policy_recorder_contract.py -q
python -m ruff check utils/superoffline_registry.py \
  utils/superoffline_observation.py utils/superoffline_action_calibration.py \
  utils/superoffline_preflight.py scripts/analysis/superoffline.py \
  scripts/analysis/superoffline_fastwam_executor.py
git diff --check
```

目录、录制来源、动作标定或执行器接口发生变化时，同步更新本 Skill。
