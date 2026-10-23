# Deploy with Codex / 使用 Codex 一键部署

将下面整段 Prompt 交给 Codex，并替换所有尖括号占位符。该 Prompt 与具体版本
无关；不要把其中的附件发现逻辑改成旧 release 的硬编码文件名。

## Release-only requirement / 仅允许 Release 部署

此 Prompt 是 **release-only** 流程。Codex 只能从 `<RELEASE_URL>` 读取 GitHub
Release 附件，完成分片校验、整包校验、解压和归档内 doctor 验证，然后启动解压目录
中的 `start_server.sh`。禁止把源码 checkout、Git clone、Windows/WSL 工作区、旧
conda/venv、系统 site-packages 或 `pip install -e` 当作替代部署来源。

如果 Release 不可访问、附件不完整、SHA256/zstd/manifest/doctor 任一检查失败，必须
停止并报告具体原因；**不得回退到源码部署、在线安装依赖、拼装旧 release，或启动未
通过归档校验的目录**。只有归档内文件可以成为生产服务的代码和运行时来源。

```text
请直接完成部署和只读验收，不要只给我操作步骤。

目标 Linux 主机：<TARGET_HOST，例如 syb@10.128.1.153；若当前终端已在目标机则填 local>
GitHub Release 链接：<RELEASE_URL>
部署根目录：<DEPLOY_ROOT，例如 /home/syb/control_panel>
控制面板端口：<PORT，例如 18088>
持久数据目录：<DATA_ROOT，例如 /home/syb/control_panel/data>

必须满足以下要求：

1. 只在目标原生 Linux 主机操作，不要在 Windows 或 WSL 上构建、解压或测试。
   先确认目标是 Ubuntu 22.04+ x86_64、glibc >= 2.35，并确认部署分区至少有
   25 GB 可用空间。
   本次部署只能使用 `<RELEASE_URL>` 的归档附件；不得 clone 源码、读取其他 checkout
   或使用已有 Python/conda 环境补齐 release 内容。
2. 从 <RELEASE_URL> 解析 GitHub owner/repository 和 release tag。用 GitHub API
   读取 release 元数据、target commit 和完整附件清单，不要猜 tag、commit、
   文件名、分片数量或 SHA256。
3. 检查 `gh auth status --hostname github.com` 以及目标账号对仓库的读取权限。
   如果 GitHub CLI 不存在，使用目标 Linux 的官方包安装方式安装。需要 sudo 时
   让我在目标机终端自行输入密码，不要向我索取、记录或回显密码。
4. 如果私有仓库尚未登录，执行
   `gh auth login --hostname github.com --git-protocol ssh --web`，把一次性授权网址
   和 device code 告诉我，然后暂停等待我在浏览器完成授权。不要让我在聊天中
   发送 GitHub 密码、PAT、token、2FA 或恢复码，也不要读取或输出 gh 凭据文件。
5. 通过 GitHub commits API 将 release tag 本身解析为不可漂移的完整 40 位
   commit SHA，同时记录 release 声明的 `targetCommitish`。如果后者本身就是
   40 位 SHA，必须与 tag SHA 相同；若 release 是 draft、附件不完整或 tag
   无法解析，停止并报告，不要部署。
6. 在 <DEPLOY_ROOT> 下创建版本化的 `downloads/<TAG>`、`releases`、`data`、
   `logs` 和 `run`。不要覆盖旧 release、用户数据、录制、标定或 config 修改。
7. 根据 release 附件清单识别且仅识别一套 Linux x86_64 发布包：
   - 若使用分片，必须有一个 `parts-*.sha256`、一个整包
     `*.tar.zst.sha256`，以及从 00 开始连续编号的 `*.tar.zst.part-*`；
   - 若使用单文件，必须有一个 `*.tar.zst` 和对应的 `*.tar.zst.sha256`；
   - 如果找到零套或多套候选，停止并列出附件，不要自行挑选。
8. 用 `gh release download` 下载该套校验文件和归档/全部分片。分片模式先执行
   `sha256sum -c parts-*.sha256`，再按文件名排序无间隙合并。随后执行整包
   `sha256sum -c *.tar.zst.sha256` 和 `zstd -t`。任一检查失败都立即停止，禁止解压。
   不得改用源码目录、旧归档或在线依赖安装绕过失败的附件校验。
9. 将归档解压到 `releases/` 中的新版本目录，不要复用旧解压目录。检查
   `manifest/release.json`：`git_commit` 必须等于第 5 步解析出的完整 SHA，
   `git_status` 必须为空，release 名称和目标平台必须匹配。再从解压目录执行
   `sha256sum -c manifest/SHA256SUMS` 和 `./bin/superinference-doctor`。
10. 只有全部校验通过后才能原子更新 `<DEPLOY_ROOT>/current` 符号链接。若端口已
    被占用，只能停止 PID 文件、命令行、工作目录都能证明属于该部署根目录的旧
    control-panel 服务；不要杀来源不明的进程，更不要触碰设备或古法推理进程。
11. 使用 release 自带的 `start_server.sh` 后台启动，设置：
    `SUPERINFERENCE_DATA_ROOT=<DATA_ROOT>`、
    `SUPERINFERENCE_CONTROL_PANEL_HOST=0.0.0.0`、
    `SUPERINFERENCE_CONTROL_PANEL_PORT=<PORT>`。
    日志写入 `logs/control-panel-<TAG>.log`，PID 写入
    `run/control-panel-<TAG>.pid`。
    启动命令必须指向新解压目录中的 `start_server.sh`；不得从源码目录或其他环境启动。
12. 不要启动、停止、连接、标定或移动相机、Rizon、Yam、夹爪、runtime、
    inference 或 recording。最多等待 60 秒，只读验证 `/`、`/recordings` 返回
    HTTP 200，`/api/state` 返回有效 JSON，且所有控制面板拥有的设备与工作流均
    为停止/空闲状态。
13. 若新服务启动失败，只终止本次新启动且能够确认归属的 PID，保留下载、解压
    和日志用于排查，不要自动回删旧 release。若原来有健康服务且切换失败，将
    `current` 恢复到原目标。
14. 持续工作到验证完成。最终报告 GitHub 登录账号、release URL/tag/target、
    部署目录、整包 SHA256、manifest 与 doctor 结果、服务 PID、控制面板 URL、
    三个 HTTP 验证结果和日志路径；不得输出任何密码或 token。
```

唯一与版本相关的输入是 `<RELEASE_URL>`；其余占位符描述目标机和部署布局，
不描述 release 内部文件。
