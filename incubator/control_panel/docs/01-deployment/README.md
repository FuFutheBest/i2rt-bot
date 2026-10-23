# Chapter 1: Download, deploy, and start / 下载、部署与启动

Release 是自包含的 Linux x86_64 归档。目标机不需要提前安装 Conda、uv、pip
或 Python 环境，但 NVIDIA 驱动、机械臂网络与许可证、相机/ttyUSB/CAN 的 Linux
权限仍由目标机提供。

任选一种部署方式：

1. [使用 Codex 一键部署](deploy-with-codex.md)：推荐。只替换五个占位符，
   release tag、target commit、归档名、分片数量和校验值均由 Codex 实时发现。
2. [传统手动部署](manual-install.md)：由操作者逐条执行下载、校验、解压和启动
   命令。

## Supported target

- Ubuntu 22.04 or newer, x86_64
- glibc 2.35 or newer
- at least 25 GB free on the deployment filesystem
- GitHub CLI access to `NoematrixTeam/SuperInference` when the repository is
  private
- an unused TCP port, normally `18088`

只在原生 Linux 主机上构建、解压、验证和运行 release。Windows checkout 仅用于
Git 同步、冲突处理和文档备份，不用于 release 运行或运行态验收。

服务启动和打开网页都不会自动启动相机、机械臂、夹爪、runtime、inference 或
recording；硬件只响应操作者的明确操作。
