# SuperInference Control Panel documentation

本目录是控制面板的版本化文档源。构建开箱即用 release 时会原样打包到
`app/incubator/control_panel/docs/`，因此在线 Git 仓库和离线 release 使用同一份说明。

## Operator guide / 操作手册

1. [Download, deploy, and start / 下载、部署与启动](01-deployment/README.md)
   - [Deploy with Codex / 使用 Codex 一键部署](01-deployment/deploy-with-codex.md)
   - [Manual installation / 传统手动部署](01-deployment/manual-install.md)
2. [Usage guide / 使用指南](02-usage/README.md)
   - [Control-panel workflow / 可视化面板与推理 Rollout 流程](02-usage/control-panel.md)
   - [Direct joint Home reset / 双臂直接关节复位](02-usage/cartesian-home-reset.md)
   - [Configuration guide / 修改 SuperInference 与 OpenPI 配置](02-usage/configuration.md)

部署 Prompt 特意使用 `<RELEASE_URL>` 等占位符。每个 release 页面只需要给出
本版本链接并引用目标 commit 中的文档，不再复制一套容易过期的版本专属命令。

## Maintainer references / 开发专题

- [Architecture](ARCHITECTURE.md)
- [Security boundary](SECURITY.md)
- [Gripper ttyUSB permissions](gripper_ttyusb_permissions.md)

策略专属 release 可以补充模型字段，但部署、硬件控制、录制和回放流程仍以本目录的
共同契约为准。
