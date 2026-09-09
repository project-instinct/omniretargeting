# Viser 播放优化与 010 实测（2026-09-09）

010 继续使用 `out/her_c167_010_diagnosis/release_body/final_retargeted.npz`。
实时检查地址为 **http://172.16.0.175:8130/**，缓存播放地址为
**http://172.16.0.175:8131/**。两者均已验证 HTTP 200。

## 方案与复用边界

保留 Viser/mjviser 的模型显示、坐标系、场景和实时交互。没有替换 WebGL 引擎，
没有修改安装在 site-packages 的依赖，也没有改变重定向输出。

实时模式使用单调时钟推进源时间，以绝对刷新截止时间扣除更新耗时；错过的刷新
不形成补发队列。机器人、人体、末端辅助图形放入同一个 Viser `atomic()`，最后
`flush()`。这是 Viser 官方提供的动画同步机制，是软同步约束，不是协议层的
严格事务保证，见 [Viser client handles](https://viser.studio/main/api/handles/client_handles/)。
UI 时间文字与滑块同步降到 10 Hz，精确跳帧立即刷新。隐藏图形不再发送动态数组。
暂停时无变化不重绘，无客户端时冻结播放时钟并停止 FK。重新连接不会追赶无人
观看期间积累的时间。动态阴影默认关闭，`--shadows` 可恢复。

缓存模式复用 **Viser 1.0.30 的官方 StateSerializer、`.viser` 格式，以及该安装
版本自带的完整 frontend**，见 [Viser communication API](https://viser.studio/main/api/auxiliary/infrastructure/)。
场景只创建一次，其后记录变换与人体数组。所有原始源帧时间戳保留，额外显示采样
使用原播放器已有的 MuJoCo 流形插值。没有低通、修改源 fps 或把末帧姿态混入首帧。
末帧停留一个源帧周期后重新播放，因此 010 的最后时间戳是 3.92 s，循环周期 3.96 s。

浏览器以流式 HTTP 下载显示进度，**完整缓存后本地播放**；这不是边下边播、
WebRTC 视频或新的自研流媒体协议。各浏览器独立控制播放。stock Viser 缓存控件
提供相机、时间轴、倍速和暂停；精确源帧输入、接触事件和全部辅助显示开关保留
在实时模式。缓存模式不继承实时模式的初始暂停、起始帧、倍速和关闭循环参数。

这样选择是因为输入是已保存的短动作。标准 WebSocket 本身没有浏览器处理背压，
不能单凭提高发送频率消除积压，见 [MDN WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)。
现有 Viser 会合并尚未发送的重复状态；本次没有另加浏览器 ACK，也没有给已经
进入网络和浏览器的字节队列提供严格上界。缓存模式避免了播放期间持续网络送帧。
长动作的完整缓存准备和浏览器内存随长度增长，长片段可继续使用实时模式。

## 同一 010 的本机对照

旧版为本轮修改前的 `mjviser_viewer.py`，归档于
`out/viewer_performance_20260909/baseline_viewer.py`。旧、新各一个独立进程，
同时接入本机协议客户端，预热 2 秒、测量约 10 秒。发送合法 Viser camera
握手以注册为客户端，接收真实压缩 WebSocket 消息，没有模拟生成数据。

| 指标 | 旧实时模式 | 新实时模式 |
|---|---:|---:|
| 收到机器人姿态包 / 秒 | 43.35 | 59.51 |
| 包间隔 P50 / P95 | 21.28 / 46.99 ms | 16.61 / 23.83 ms |
| 最大包间隔 | 60.57 ms | 59.17 ms |
| 同包包含人体的机器人包 | 328 / 434 | 596 / 596 |
| 动态传输字节 / 秒 | 165,472 | 190,586 |
| 暂停并稳定后的额外包 | 0 | 0 |
| 精确跳到源第 45 帧 | 通过 | 通过 |
| 恢复播放后断开全部客户端，3 秒 CPU 采样 | 22.67% | 0.00% |

CPU 百分比以单核为 100%。新实时模式刷新更多，因此总带宽上升，不能称实时
带宽下降。包间隔仍有长尾，不能称网络或客户端已严格稳定在 60 fps。
这些都是**服务端／本机协议接收指标，不是用户浏览器渲染 FPS**。

## 缓存与发布验收

- 官方缓存文件 4,255,940 字节，解压后 13,122,640 字节；315 个不同时间戳。
- 全部 99 个源时间戳在缓存中精确存在；非零时间没有新增 mesh 创建消息。
- 正式启动缓存准备约 0.76 秒；复制的 frontend 与安装版本逐字节一致。
- HTTP 下载摘要等于 manifest，录制文件用内容哈希命名和 immutable 缓存头。
- 服务仅公开 bundle 资产，不公开模型、源运动或项目目录。
- 8130 正式进程协议复验：约 2 秒收到 119 个机器人姿态更新包。
- 原 NPZ 的 SHA256：`726f091c8a1740a8da9289ee9c10966371fb40de751ec07820e60845d6507963`。

新增 14 项播放器测试覆盖倍速切换前结算旧时钟、暂停／跳帧、不循环结束、
末帧停留、单帧、断开后恢复、超时调度、非整数 fps、HTTP 缓存和资产限制。
真实微型 MuJoCo 模型经 CLI 导出，再独立解码 `.viser`，核对序列位置与原始
源帧数值一致，输入文件逐字节不变。

最终 `pytest -q tests` 为 **281 passed, 4 skipped（17.45 s）**，包含
`tests/test_contact_priority.py`；`git diff --check` 通过。
本会话没有可调用的浏览器控制 runtime，未完成浏览器视觉验收或实际 GPU FPS 测量。
原重定向动作的 solver 抖动不会被播放优化消除。

运行命令见 [README](../README.md#viewing-retargeted-output-with-mjviser)。
原始记录在 `out/viewer_performance_20260909/`：`stream_metrics.json`、
`idle_metrics.json`、`cache_audit.json`、`production_verification.json`、
`tests_final.log`、`production.log`、`launch.json`。只替换了 8130 的 010 播放进程，
并结束本次 8138/8139 测试进程；其他原有播放器保留。

## 后续：用户反馈视角流畅、动作顿挫后的直接姿态更新

用户在第一轮上线后反馈：8130 旋转视角流畅，只有动作一顿一顿。实际连接当时
RTT 约 14 ms，发送队列约 3 KB，服务端仍持续约 59 Hz。这些数据不足以归因于
服务器吞吐，也没有浏览器 FPS 数据可以证明第一轮已经解决显示端卡顿。

检查安装版本前端后发现，`SceneNodeUpdateMessage.batched_positions/wxyzs`
经 `MessageHandler.tsx` 更新 React 管理的 scene props，再由
`mesh/BatchedMeshBase.tsx` 的 `useEffect` 更新实例；而
`SetPositionMessage` / `SetOrientationMessage` 直接写入 mutable pose refs，
在绘制前应用，不需要这条 React props 更新路径。该差异是已确认的实现机制；
尚未通过浏览器 profiling 证明它是用户卡顿的唯一原因。

增加 `_RigidInstanceHandle` 适配器：继续复用原 Viser BatchedGlb 几何、材质、
LOD 和实例缩放，把单个实例的位置/旋转固定为单位变换，通过原生节点位姿移动
整个部件。真实多实例组保留原实现。只接入项目已固定版本 mjviser 的
`_mesh_groups`，没有修改 site-packages、前端构建文件、原模型或源轨迹。

010 共 37 个单实例组适用。在同一 315 时间戳的确定性导出中，机器人动态 props
消息从 24,492 减到 2,512，减少约 **89.7%**；改用 10,990 个位置消息和同样数量
的旋转消息。文件从 4.26 MB 增到 4.53 MB，不能声称总消息数或带宽降低。
新的本机协议测量约 56.75 姿态包/s，P95 间隔 28.32 ms；没有进一步改善服务端
吞吐的结论。此次目标是绕过单实例姿态的 React 更新环节，而非提高发送次数。

实际模型测试重建“外部节点变换 × 内部实例变换”，验证任意旋转、平移、子关节
与 MuJoCo FK 一致；验证重复几何的真批次保持原句柄。CLI 导出测试仍逐帧核对
实际机器人位置和全部源时间戳。完整测试 **282 passed, 4 skipped（16.76 s）**；
最后的历史缓存 URL 兼容修改经 15 项播放器测试通过。缓存服务保留既有哈希 URL
及各自的 ETag，旧浏览器页刷新不会因重新导出而丢失录制文件。

已再次更新 8130/8131。此次 PID 和命令记录于 `launch_direct.json`，日志为
`production_direct.log`；测量见 `direct_stream_metrics.json`、
`direct_message_comparison.json`、`tests_direct.log`。页面刷新后建立新连接才会
取得新场景状态。尚无客户端实测 FPS，不能宣称所有显示顿挫或原轨迹抖动已消失。
