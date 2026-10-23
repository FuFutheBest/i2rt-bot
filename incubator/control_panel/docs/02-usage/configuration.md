# Configuration guide / 修改 SuperInference 与 OpenPI 配置

SuperInference 启动 YAML 与 OpenPI 模型注册表是两层配置，修改一层不会自动创建
或更新另一层。

## File locations / 文件位置

| 配置层 | 源码 checkout | 已解压 release | 作用 |
| --- | --- | --- | --- |
| SuperInference 顶层 config | `<repo>/configs/<name>.yaml` | `<release>/app/configs/<name>.yaml` | 设备、SHM、Home/init pose、policy 参数、checkpoint、相机角色与录制设置。 |
| SuperInference config group | `<repo>/configs/<group>/...` | `<release>/app/configs/<group>/...` | 被顶层 config 引用的 Hydra 片段。 |
| 内置 OpenPI 注册表 | `<repo>/third_party/openpi/src/openpi/training/config.py` | `<release>/app/third_party/openpi/src/openpi/training/config.py` | 注册 `policy.params.pi_config_name` 指向的 OpenPI 训练/推理配置。 |
| OpenPI client 源码 | `<repo>/third_party/openpi/packages/openpi-client/src` | `<release>/app/third_party/openpi/packages/openpi-client/src` | 与 OpenPI 快照配套的 client/runtime API。 |

如果 YAML 设置了 `policy.openpi_src`，该路径相对于仓库根目录或 release 的
`app/` 解析。此时必须修改它选中的源码树，不能默认修改 `third_party/openpi`。
标准 release 的 policy runtime 会把 `app/third_party/openpi/src` 及配套
`openpi-client` 放在宿主机 Python 包之前。

两层名称必须对齐：

```yaml
policy:
  params:
    pi_config_name: "pi05_example"
    ckpt_dir: "/path/to/checkpoint"
```

`pi05_example` 必须注册在实际生效的 OpenPI `openpi/training/config.py` 中；
checkpoint 还必须包含与该 OpenPI config 对应的 assets 和归一化统计。

## Reload rules / 修改后如何生效

| 修改内容 | 生效方式 |
| --- | --- |
| 服务空闲时新增/删除顶层 YAML | 聚焦或打开 config 下拉框触发 `/api/configs` 重扫，也可刷新页面，然后选择 config。不会自动启动硬件。 |
| 修改尚未选中的 YAML | 保存后选择它；后端在选择时重新从磁盘组合。 |
| 修改当前已选 YAML 或其 Hydra group | 正常停止 inference、runtime、夹爪和受影响设备。切换到另一 config 再切回，或重启控制面板服务。只刷新浏览器**不会**替换后端已组合的 effective config。 |
| 修改 `openpi/training/config.py` | 正常停止 inference/runtime 和硬件，然后重启整个控制面板服务再选择/启动。这会清除 workflow 注册表检查缓存，并保证下一个 policy 进程导入新注册表。 |
| 只改临时 checkpoint 或录制目录 | 使用 UI override，无需修改源文件或重启服务；更换 checkpoint 后需要重启 inference 才会重新加载权重。 |
| 修改 HTML/CSS/JavaScript | 重启服务，再强制刷新浏览器，避免复用旧静态资源。 |

旧 effective config 启动的进程仍在控制硬件时不要修改配置。UI 下拉框只产生运行时
override，不会写回 YAML。

## Editing an extracted release / 直接修改 release

紧急情况下可以直接修改已解压 release，但该安装会从不可变发布物变成本地补丁：

- `manifest/SHA256SUMS` 将不再匹配；
- 同 tag 在另一台机器上无法复现该行为；
- 将 `current` 切到新解压目录会丢失修改；
- `superinference-doctor` 不能替代失败的文件校验。

必须记录本地 diff，并按上表重载。不要重写发布物的 checksum 来伪装“原版未修改”。

## Reproducible release change / 正式修改方式

需要长期保留的 SuperInference YAML 修改，应提交到对应 control-panel 开发分支，
在 Linux 验证后从该 commit 构建新 release。

修改 OpenPI 注册表的正式流程：

1. 在目标 OpenPI checkout/fork 修改并提交。
2. 让 SuperInference YAML 引用已注册的 `policy.params.pi_config_name`；只有确实
   使用非默认源码树时才设置 `policy.openpi_src`。
3. 在 Linux 打包时明确传入该 checkout：

   ```bash
   --submodule-source openpi=/path/to/validated/openpi
   ```

4. 确认 `manifest/release.json` 记录正确的 SuperInference commit 和 OpenPI
   源码快照，执行 release doctor 与全量 checksum，再部署新版本，不在旧 tag 上
   原地打补丁。

这样才能保持 release tag 到 SuperInference YAML、OpenPI registry/client、
运行依赖和 checkpoint 契约之间唯一且可审计的映射。
