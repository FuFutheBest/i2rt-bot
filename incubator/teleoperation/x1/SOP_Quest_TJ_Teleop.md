# Quest3 遥操作 TJ 机械臂 SOP（局域网 + WebRTC 低延迟）

本文档记录经实测验证可用的连接方案：Quest3 与主控机在 **同一局域网**，走
**HTTPS 自签证书 + WebRTC** 实现低延迟图传与遥操作，无需 ngrok。

有两种接入方式，二选一：

- **方案 A（当前在用，推荐）**：普通局域网/路由器。Jetson 有线口 `eth8 = 192.168.0.234`，
  Quest 连同一个路由器的 WiFi，直接访问 `192.168.0.234`。简单、不抢空口。
- **方案 B（备用）**：Jetson 自带 WiFi 热点 `tztek_ax210 = 192.168.88.88`。
  当现场没有可用路由器、或公司 AP 有客户端隔离时用（见 §2、附录的热点专属配置）。

> ⚠️ 下文操作命令默认用**方案 A 的 IP `192.168.0.234`**。若用方案 B（热点），
> 把所有 `192.168.0.234` 换成 `192.168.88.88` 即可，其余步骤相同。

**Quest 浏览器要打开的最终地址（方案 A）**：

```
1) 先放行视频：https://192.168.0.234:8080/         （高级→继续 → Start WebRTC）
2) 再进 VR   ：https://192.168.0.234:8012?ws=wss://192.168.0.234:8012   （高级→继续 → Enter VR）
```

---

## 0. 环境速查（本机已确认的事实）

| 项目 | 值 |
|------|-----|
| 主控机 | NVIDIA Jetson（多网口 + AX210 无线网卡） |
| 容器 | `robot_test`（**host 网络**，挂载 `/home/nvidia/nmx_hal → /root`） |
| Python 环境 | `/root/tvenv`（容器内） |
| 脚本目录 | 容器内 `/root/incubator/teleoperation/x1`（= 宿主 `/home/nvidia/nmx_hal/incubator/teleoperation/x1`） |
| Robot ID | `1e783be2dd870ef5` |
| VR 主程序端口 | `8012`（`tj_controller_teleop.py`，提供 VR 页面） |
| 视频流端口 | `8080`（`tj_camera_webrtc.py`，WebRTC 推流） |
| 证书 | `certs/cert.pem` + `certs/key.pem`（自签，SAN 必须含访问用的 IP） |
| **方案 A 访问 IP（当前）** | `192.168.0.234`（有线口 `eth8`，`192.168.0.0/24` 网段） |
| 方案 B 热点 SSID | `tztek_ax210` / 密码 `12345678` / Jetson 在该网 IP `192.168.88.88` |
| SSH 调试链路 | 走公司 WiFi `wlan0`（不要断，否则失联） |

> 端口都绑在 `0.0.0.0`，所以本机所有 IP（`192.168.0.234` / 热点 `192.168.88.88` / 公司网等）
> 都能访问到，关键是“客户端能不能把包送到 Jetson”。
> 当前证书 SAN 已包含 `192.168.0.234` 和 `192.168.88.88`，两种方案都可直接用。

---

## 1. 方案 B（热点）背景：为什么需要 Jetson 自己的热点

> 如果你用**方案 A（路由器局域网，`192.168.0.234`）**且能正常访问，可**跳过本节和 §2**，
> 直接到 §3 启动服务。本节只在被迫用 Jetson 热点（方案 B）时才需要。

调试中确认过两个拦路问题，方案 B 的配置都是为了绕开它们：

1. **公司无线 AP 客户端隔离**：公司 WiFi（实测是访客网 `Noematrix_Guest`）开启了
   client isolation——放行 ping，却阻断无线设备之间的 TCP。所以 Quest（无线）发往
   Jetson 的 TCP SYN 根本到不了，而走有线的电脑却能连。
   → **对策：让 Quest 连 Jetson 自己的热点 `tztek_ax210`，绕开公司 AP。**

2. **单射频 STA+AP 频道冲突**：AX210 用同一个物理射频既连公司 WiFi（STA, `wlan0`）
   又开热点（AP, `wlan0-ap`）。两者频道不同（曾是 AP=6 / STA=11）时，射频来回切换，
   热点链路严重丢包（实测 50% 丢包、326ms），TCP 建不起来。
   → **对策：把热点频道改成与公司 WiFi 相同，单射频“同频共存”，丢包大幅下降。**

---

## 2. 方案 B（热点）专属配置：Jetson 一次性设置

> 本节仅在使用**方案 B（Jetson 热点）**时需要。方案 A（路由器局域网）无需本节。
> 以下命令在 **Jetson 宿主机** 终端执行（不是容器内）。需要 `sudo`。

### 2.1 确认证书 SAN 包含要用的 IP

```bash
sudo openssl x509 -in /home/nvidia/nmx_hal/incubator/teleoperation/x1/certs/cert.pem \
    -noout -ext subjectAltName
```

输出里必须有你要访问的 IP：方案 A 看 `IP Address:192.168.0.234`，方案 B 看 `IP Address:192.168.88.88`。
**当前证书两者都已包含。** 如果换了网络、出现新 IP 不在里面，重新生成（把要用的 IP 都列进 SAN）：

```bash
cd /home/nvidia/nmx_hal/incubator/teleoperation/x1/certs
sudo cp cert.pem cert.pem.bak; sudo cp key.pem key.pem.bak
SAN="DNS:localhost,IP:192.168.0.234,IP:192.168.88.88,IP:10.128.0.197"   # 按需补上其它要用的 IP
sudo openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout key.pem -out cert.pem -days 825 \
  -subj "/CN=tj-teleop" -addext "subjectAltName=${SAN}"
```

> 注意：自签证书是按 IP 签的。证书里**没有**的 IP，浏览器会直接拒，重签后必须重启服务。

### 2.2 让热点频道与公司 WiFi 频道一致（关键）

1. 查公司 WiFi（STA）当前频道：

```bash
iw dev wlan0 link | grep -E "SSID|freq"
# freq 2412→ch1, 2437→ch6, 2462→ch11 ...（2.4G 常用 1/6/11）
```

2. 把热点 hostapd 频道改成同一个（示例：公司在 ch11）：

```bash
sudo sed -i 's/^channel=.*/channel=11/' /etc/hostapd/hostapd.conf
grep -E "^channel=" /etc/hostapd/hostapd.conf      # 确认
```

3. 重启热点服务使其生效（**只影响热点，不会断 wlan0/SSH**；服务有 ~30s 启动前等待）：

```bash
sudo systemctl restart tztek-jetson-service-net-wifi6-ax210-station-ap.service
sleep 45
sudo iw dev wlan0-ap info | grep -E "channel|ssid"   # 确认 channel 已变
nmcli -t -f NAME,DEVICE,STATE con show --active | grep wlan0   # 确认 SSH 链路还在
```

> ⚠️ 公司 WiFi 的频道可能随时间漂移（AP 自动选频）。如果某天热点又开始丢包，
> 先回到 2.2 重新核对两边频道是否还一致。

### 2.3（可选，彻底消除丢包）SSH 改走有线、断开公司 WiFi

“同频共存”仍会和公司网抢空口（实测残留约 30% ping 丢包，但 TCP 重传后网页/视频可用）。
若想把丢包降到接近 0：

1. 让你的开发电脑接到 Jetson 的有线网段之一（如 `eth1=192.168.1.100` 或 `192.168.11.x`），
   用有线 IP 重新 SSH 进 Jetson。
2. 断开公司 WiFi，把射频完全留给热点：

```bash
sudo nmcli dev disconnect wlan0      # 恢复： sudo nmcli con up Noematrix_Guest
```

此时热点独占射频，链路质量最佳。机器人控制走有线，不受影响。

---

## 3. 启动遥操作服务（容器内，两个终端）

每个新终端都先进容器并激活环境：

```bash
docker exec -it robot_test bash
source /root/tvenv/bin/activate
cd /root/incubator/teleoperation/x1
```

### 终端 A — 相机 WebRTC 推流（端口 8080）

```bash
python3 tj_camera_webrtc.py --https
```

看到 `Running on https://0.0.0.0:8080` 即正常。**这个进程必须一直开着**，否则视频出不来。

### 终端 B — 遥操作主程序（端口 8012）

先做一次单臂 dry-run 验证手柄轴向（不发指令、只打印）：

```bash
python3 tj_controller_teleop.py --arm-side left --dry-run --print-debug \
    --https --webrtc-src https://192.168.0.234:8080/offer
```

确认无误后正式启动（双臂 + 夹爪 + 头部 + WebRTC）：

```bash
python3 tj_controller_teleop.py --arm-side both --enable-grippers --enable-head \
    --https --webrtc-src https://192.168.0.234:8080/offer
```

> - `--webrtc-src` 的 IP 必须用 **Quest 实际访问的那个 IP**（方案 A=`192.168.0.234`，
>   方案 B 热点=`192.168.88.88`），要和 Quest 浏览器地址栏一致，否则视频源跨源会被挡。
> - 默认 `--orientation-mode relative`：握住离合后手柄旋转会带动末端旋转（姿态已做 SLERP 限速平滑）。
>   只想平移、锁死姿态就加 `--lock-orientation`。
> - 启动时会自动执行 `go_home`：双臂 move_joint 回初始位、头部舵机归位（见 `teleop_home.py`，
>   可在其中改初始位姿）。不想回初始位加 `--no-home`。

### 启动后本机自检（可选）

```bash
for p in 8080 8012; do curl -k -s -o /dev/null -w "0.234:$p code=%{http_code}\n" \
  https://192.168.0.234:$p/ --max-time 4; done
# 期望两行都是 code=200
```

---

## 4. Quest3 上的操作

> Quest 上**不需要安装任何程序**，用自带浏览器即可（所有程序都在 Jetson 上跑）。

### 4.1 连接同一局域网

- **方案 A（当前）**：Quest 设置 → WiFi → 连接和 Jetson 同一个**路由器的 WiFi**
  （Jetson 走 `eth8` 有线接在该路由器的 `192.168.0.0/24` 网段）。确认 Quest 拿到 `192.168.0.x` 的 IP。
- **方案 B（热点）**：Quest 设置 → WiFi → 选择 **`tztek_ax210`** → 密码 `12345678`。
  （连上后可能提示“无互联网/受限”，无妨；热点已做 NAT，通常仍能上网，且不影响遥操作。）

### 4.2 放行两个端口的证书（⚠️ 两个端口要各放行一次）

在浏览器地址栏 **手动完整输入带 `https://` 的地址**，依次打开（下面用方案 A 的 `192.168.0.234`；
方案 B 换成 `192.168.88.88`）：

1. `https://192.168.0.234:8080/` → “高级 / Advanced” → “继续前往 192.168.0.234（不安全）”
   → 页面点 **Start WebRTC**，能看到机器人相机画面即 OK。
2. **VR 主页面（⚠️ 必须带 `?ws=` 参数）**：

```
https://192.168.0.234:8012?ws=wss://192.168.0.234:8012
```

   → 同样“高级 → 继续” → 页面加载后点右下角 **Enter VR**。

> ⚠️⚠️ 关键坑：8012 这个地址 **必须带 `?ws=wss://192.168.0.234:8012`**。
> 否则 Vuer 前端的 WebSocket 会连到云端默认地址，而不是你本地的服务器——
> 表现为：能进 VR、能看到场景和手柄模型，但**没有视频、手柄完全没反应**
> （服务器侧 `--print-debug` 一直是 `pos=[0,0,0]`、永远不出现 `[HAND_MOVE]`）。
> 这条 ws 地址要和页面同 IP 同端口。

### 4.3 进入 VR 后的手柄操作

| 操作 | 按键 | 说明 |
|------|------|------|
| 跟随/离合 | **侧握键 grip（食指/中指那个）** | 按住才跟随机械臂；松开暂停；再次按住重新捕捉基准（可重新摆手） |
| 末端平移 | **握住离合移动手柄** | 手柄 x/y/z 位移映射到末端位移 |
| 末端旋转 | **握住离合转动手柄** | 默认 `relative` 模式，手柄转多少末端转多少（已 SLERP 限速）；加 `--lock-orientation` 则锁死不转 |
| 夹爪 | **扳机 trigger** | 松开=张开，扣到底=闭合（需 `--enable-grippers`） |
| 头部转向 | **头显朝向** | 左右转头控制 head yaw；上下点头控制 head pitch（需 `--enable-head`） |

> 注：若头部「上下(pitch)」不动而「左右(yaw)」正常，多半是 pitch 舵机硬件卡死/保护，
> 与软件无关——先给头部断电重启，再用 `tj_head_test.py` 验证两个轴（见该脚本说明）。

---

## 5. 常见问题排查（按现象对号入座）

| 现象 | 原因 | 处理 |
|------|------|------|
| 浏览器“无法连接此网站/无法打开网页” | 服务没起 / TCP 不通 | 查 `ss -tlnp \| grep -E ':8012\|:8080'` 两端口是否在监听；不在就启动对应脚本 |
| ping 通但 TCP 连不上（方案 B） | 公司 AP 客户端隔离 | 改连 Jetson 热点 `tztek_ax210`（见 §4.1）；或改用方案 A 路由器 |
| 热点能连但丢包严重/打不开（方案 B） | 单射频 STA+AP 频道不一致 | 把热点频道改成与公司 WiFi 一致（见 §2.2）；或断 wlan0（见 §2.3） |
| 抓包看到 `GET / HTTP/1.1` 后被服务器 FIN | 浏览器走了明文 **http** 访问 https 端口 | 地址栏**显式输入 `https://`**，别从历史记录点到 `http://` 的旧地址 |
| 证书警告无法继续 / 提示证书无效 | 证书 SAN 不含当前 IP，或缓存了旧证书 | 重签证书含该 IP（§2.1）+ 重启服务；Quest 清除浏览数据后重打完整 `https://` 地址 |
| 页面能进但看不到视频 | 只放行了 8012、没放行 8080；或 `--webrtc-src` IP 不一致 | 单独打开 `https://192.168.0.234:8080/` 放行证书；确认 `--webrtc-src` 用的就是 `192.168.0.234` |
| 进了 VR、能看到手柄模型，但没视频且手柄完全没反应（`--print-debug` 一直 `pos=[0,0,0]`，无 `[HAND_MOVE]`） | 8012 地址没带 `?ws=`，WebSocket 连到云端而非本地 | 用 `https://192.168.0.234:8012?ws=wss://192.168.0.234:8012` 打开（见 §4.2） |
| 末端只平移不旋转 | 带了 `--lock-orientation` | 去掉该参数即用 `relative` 旋转跟随（见 §3 说明） |
| 头部左右(yaw)能动、上下(pitch)不动 | pitch 舵机硬件卡死/过载保护 | 头部断电重启；用 `tj_head_test.py` 验证两轴（非软件问题） |
| 视频卡顿/掉帧（方案 B） | 同频共存仍抢空口（残留丢包） | 走 §2.3 断 wlan0 让射频独占热点；或改用方案 A 路由器 |

### 有用的诊断命令（Jetson 宿主机）

```bash
# 端口监听
ss -tlnp | grep -E ":8012|:8080"
# 热点已连设备 + 信号 + 速率
sudo iw dev wlan0-ap station dump | grep -E "Station|signal|bitrate|connected time"
# 热点上 Quest 的 IP
ip neigh show dev wlan0-ap
# 抓 Quest 到某端口的包（定位卡在哪一步），把 <QUEST_IP> 换成上面查到的（方案 A 抓 eth8，方案 B 抓 wlan0-ap）
sudo timeout 25 tcpdump -ni eth8 "host <QUEST_IP> and tcp port 8080"
# 两端口本机可达性（方案 A）
for p in 8080 8012; do curl -k -s -o /dev/null -w "0.234:$p code=%{http_code}\n" https://192.168.0.234:$p/ --max-time 4; done
```

---

## 6. 关机/收尾

- 主程序终端 `Ctrl+C` 退出（会停止下发，机械臂保持当前位）。
- 相机服务终端 `Ctrl+C`。
- 如果做过 §2.3 断开了公司 WiFi，恢复 SSH/上网：`sudo nmcli con up Noematrix_Guest`。
- 热点频道修改是持久化的（写在 `/etc/hostapd/hostapd.conf`），重启后仍为该频道；
  若公司 WiFi 换了频道需重新核对（§2.2）。

---

## 附：一句话回顾能连上的关键

**方案 A（当前）**：
1. Quest 与 Jetson 同一路由器局域网，访问 Jetson 有线 IP `192.168.0.234`。
2. 证书 SAN 含 `192.168.0.234`，且 Quest 地址栏 **显式输 `https://`**，对 8080/8012 **各放行一次**。
3. 8012 地址 **必须带 `?ws=wss://192.168.0.234:8012`**，否则手柄/视频数据通道连不到本地服务（最隐蔽的坑）。
4. 代码侧 `main_image` 已加 `session.set @ DefaultScene(frameloop="always")`，保证手柄采集逐帧回调持续运行。

**方案 B（热点，备用）** 额外注意：
1. Quest 连 **Jetson 热点 `tztek_ax210`**（绕开公司 AP 隔离）。
2. 热点频道与公司 WiFi **同频**（解决单射频丢包），IP 换成 `192.168.88.88`。

