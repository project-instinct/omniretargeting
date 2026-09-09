# HER 010：身体跟随、空翻和场景相交的实测修复

2026-09-09 更新：按用户要求取消固定关节帧间限速，010 第 46 帧伸腿明显改善。
最新输出、8132 播放入口及验收见 [取消固定关节限速](her_010_no_joint_caps.md)。
下文保留 2026-09-08 的历史运行结果和当时配置。

2026-09-08。最新输出为 `out/her_c167_010_diagnosis/release_body/`，播放地址
**http://172.16.0.175:8130/**；8128 是旧结果，8129 是较早的中间实验。
010 的胸部、空翻腿部跟随、脚世界位置误差和可测表面相交明显改善，但部分手臂姿态、
另一段 1530 的手部接触仍有退步，不能称所有动作已彻底修复。

基线为 main 的 `e25e4c08cf3498def903abc3d664705b1ce2f9d9` 加本地已有修改，
不是干净的远端版本。保留原 dirty worktree，没有 reset、切分支、提交或 push。
实际代码 SHA256、展开后的有效配置、原输入、运行日志均随最终输出保存。

## 实际原因和实现位置

| 原因或机制 | 本次实现 |
|---|---|
| 自由根所属 body 与默认骨盆朝向不对应，标准姿态轴不同 | `main.select_mapped_source_base_pose` 从编译模型查唯一 freejoint 的 body，再查 source mapping；`data_sources/smplx.py` 提供独立的 `body_aligned_joint_orientations`，保留原始人体数据。显式 source 选择优先；无可靠校准的 adapter 保留原路径。 |
| 位置锚点与朝向对应部位混在一起 | 两者独立配置。InstinctMJ G1 的 torso 根跟随校准的 Spine3 朝向，位置参考 Pelvis；torso 原点靠近腰部，不应对应人体 Spine3 的世界高度。`core.py` 首帧水平注册使用所选关节点，不使用可能相差数十厘米的 SMPL-X 模型 translation。 |
| 参考校准受场景影响，或把全身翻转误算成各肢体局部姿态误差 | Adapter 用同 shape/gender 的零姿态建立 Z-up 参考；`retargeting.py` 的参考校准与场景隔离，支持最近映射父节点下的相对姿态任务。根旋转不会重复进入相对任务误差/Jacobian。 |
| 位置对应 body 不一定拥有完整旋转自由度 | G1 profile 保持位置映射，姿态映射改用髋、肩、腕末端旋转 body；禁用该 profile 不合适的 Spine1 姿态任务。通用求解器不按机器人名、clip 或帧号选策略。 |
| 绝对姿态步长限制压制快速源动作 | 完整语义轴可用时，以原时间顺序源旋转为参考限制新增旋转误差；每帧只取一次接触任务并复制方向数组。原有关节 0.18 rad 单帧上限和加速度上限保留。不是取消限速，也不是所有失败恢复路径的绝对保证。 |
| 接触与自由运动的优先级不合适 | `balanced` 模式先求并保留可信接触，再联合权衡自由末端、骨段方向和姿态。无语义接触时保留已有加权求解与恢复路径。旧 lexicographic 模式仍可选，其中骨段形态先于不兼容的完整姿态。 |
| 开放扫描面无实体内外证据，旧报告为 0 却实际相交 | `triangle_capsule.py` 独立实现有限胶囊轴/三角面距离；`geometry_query.py` 增加原三角面的 capsule/sphere 相交查询，供 `scene_geometry.py` 的优化和验收复用。覆盖端点/面、线段/边、轴穿面和退化面，不将三角面扩成无限平面。 |
| 轴穿面见证点重合，普通梯度失效 | 对单个三角面与胶囊计算配置空间障碍的最短可靠退出；唯一方向才构造约束，对称/不可靠时保留负 clearance 并报告 uncertain。没有任意选法向或将房间凸包化。 |

复用已有 FK、Jacobian、语义接触和轨迹 refinement。几何梯度在原模型 nv 切空间计算，
原帧求解器使用已有 nq 转换；局部阶段只开放相关关节列。更新后重新 FK，保存前测量
实际碰撞与保真指标。原 SMPL-X、场景和机器人几何尺度没有改动。

## UMR 来源、已有能力和开关

UMR 固定版本 `8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09` 的代码、权重和资产复用许可
未确认，没有搬入项目。完整函数、defaults、batch 调用和许可见 [UMR 审阅](umr_scene_interaction.md)。

| 官方参考 | 本项目对应 |
|---|---|
| `build_robot_object_penetration_cache`、`compute_robot_object_penetration_rows`、`collision_relative_jacobian` | 复用已有固定场景 MuJoCo 凸组件查询，新增开放面有限 capsule/triangle 查询。参考几何、见证点和相对 Jacobian 机制，独立实现。 |
| `solve_sequence`、`select_bidirectional_dp`、batch 双向 warm start | 已有 `local_collision_repair.py`、`local_candidates.py` 保持插值优先；失败窗口才生成双向候选，固定接触计划、统一评分并检查边。本次真实 010 的改善来自原逐帧求解，不能归功于未运行的 DP。 |
| `solve_frame_body_segment_qp`、`solve_clarabel_qp_step` | 复用本项目 QP 和分层接触任务，没有替换整个优化器。线性约束通过不等于非线性零穿透。 |
| `bind_source_slots_with_normals`、`transport_tpose_robot_normals`、`compute_tpose_surface_normal_offsets`、`TemplateSlotKinematicsCache` | 已有可选 `surface_tasks.py` 接收固定绑定并运输机器人参考法向，真实实验未启用。此次骨段相对姿态校准不是 learned correspondence 或稠密表面任务。 |
| `lqr_smooth_qpos_sequence` | 没有替换已有接触感知 refinement，没有新增全段低通。 |

实际默认值：`terrain_sample_seed=0`，`penetration_resolver=hard_constraint`，
`end_effector_priority.hierarchy=true/free_motion_priority=balanced`；局部修复
`enabled/full_geometry/bidirectional_enabled=true`、`max_query_points=null`、`max_substeps=256`。
取消查询点数量截断，没有取消物理偏移限制。自碰撞配置仍为 false；启用时按同一策略覆盖
优化和验收。脚保持 `preserve_world_target=true`，自由手以身体相对形态为主，接触目标保持世界坐标。

可选而默认关闭：完整 source XYZ/rotation 根部预测 warm start、显式虚拟解剖躯干映射、
P2 表面法向增强。预测实验降低部分胸部误差却使腿进入障碍，因此未启用；默认从上一
机器人姿态开始。旧绝对末端朝向步长语义仍可显式选取。
未实现 learned correspondence、自动可靠表面绑定、任意开放扫描体的实体内外推断，
未完成跨机器人真实动作和动力学验证。

## 010：最终保存轨迹独立对照

同一 HER 原发布 99 帧、25 fps、3.92 s；source frame_id 作为 provenance，不补帧或改时序。
独立测量用全部 29 个实际碰撞 capsule/sphere、胶囊轴 1 mm 采样和原场景三角面最近点。
不使用优化器的三角距离实现，轴距离漏差上界 0.5 mm。“表面侵入”是碰撞体与有限
三角面的相交，不是开放扫描房间的实体有符号深度。

| 指标 | 8128 旧结果 | 8130 最终结果 |
|---|---:|---:|
| 最大可测表面侵入 | 63.400 mm | 1.00014 mm |
| 超过 1 mm 的保存帧数 | 16 | 2（数值略超 1 mm） |
| 最差 body / 帧，0 起算 | torso_link / 46 | right_wrist_yaw_link / 45 |
| 胸部世界朝向误差，平均 / 最大 | 31.53° / 59.18° | 12.38° / 33.84° |
| 躯干骨架方向误差，平均 / 最大 | 23.06° / 49.91° | 13.89° / 35.21° |
| 空翻 45–80 帧躯干方向平均误差 | 35.75° | 17.93° |
| 躯干最佳源时间偏移 | −3 帧（滞后 120 ms） | 0 帧 |
| 左脚世界误差，平均 / 最大 | 61.59 / 234.93 mm | 51.14 / 160.31 mm |
| 右脚世界误差，平均 / 最大 | 63.35 / 303.96 mm | 58.39 / 102.64 mm |
| 自由左手世界误差，平均 / 最大 | 152.09 / 384.02 mm | 194.53 / 401.13 mm |
| 自由右手世界误差，平均 / 最大 | 171.45 / 440.75 mm | 192.06 / 574.35 mm |
| 骨盆朝向平均误差 | 22.90° | 24.62° |
| 8 段四肢方向平均误差 | 6.50° | 7.33° |
| 最大关节单帧步长 / 加速度 | 0.18 rad / 50 rad/s² | 0.18 rad / 50 rad/s² |
| 关节 jerk RMS | 797.44 rad/s³ | 834.82 rad/s³ |
| 全流程单次耗时 | 161.59 s | 180.91 s |

躯干骨架方向由骨盆到双肩中点的方向与肩部横轴构建，与胸部关节旋转分别报告。
时间偏移枚举同一空翻区间 −5 至 +5 帧，仅用于诊断，没有改时间戳。
旧相交帧为 40–51、62–64、76；新为 27、45，各持续一个保存帧。
约 1 mm 不是严格零穿透，生产验收仍保留 collision_residual 标记。

空翻左/右大腿平均方向误差从 14.84°/18.13° 降至 9.62°/11.01°，小腿从
11.18°/9.02° 降至 8.49°/8.05°；上臂从 12.13°/10.48° 增至 17.35°/20.31°。
不能仅凭胸部误差称全身都更准，也不能称输出更平滑。自由手世界位置不是高优先级目标，
但其形态退步仍需记录。010 没有语义接触，接触残差和 sticking 滑移为 N/A，不能将
空集合报告的 0 当作无滑移证明。脚仍逐帧跟踪不变世界目标。

## 高度、根轨迹、局部回退与帧间验收

机器人根 Z 为 0.38653–1.08251 m，旧结果 0.37615–1.13809 m。
新原帧求解相对旧结果的根垂直差为 −137.27 至 +115.74 mm，最大水平差 263.58 mm，
最大关节差 1.25908 rad。这是改正锚点并重新求解的整体差异，不能冒称小于 0.2 rad 的
局部修复。没有整体抬升人体、移动场景或缩短连杆。

局部阶段对全部帧、全部 qpos 逐元素 no-op，根 0–6 列完全相同，窗口外和无关关节
完全不变。新增修正的速度、加速度、jerk 均为 0，不产生新增窗口边界不连续。
这不表示重新重定向后的根轨迹与 8128 相同。

| 事件窗口（含端点，0 起算） | 方法 | 实际原因 |
|---|---|---|
| 20–30 | fallback，0 次优化 | 扫描侧别未知，找不到可信固定边界 |
| 30–37 | fallback，0 次优化 | 同上 |
| 40–50 | fallback，0 次优化 | 同上 |
| 43–53 | fallback，0 次优化 | 同上 |

插值和双向求解因 `missing_valid_boundaries` 未启动；优化返回 `uncertain_fixed_boundary`。
回退保留的是本次新逐帧轨迹，不是换回 8128。全轨迹后处理涉及根部、末端或边界偏移
超限的候选也未写入最终轨迹。局部 slack 为 0，碰撞点水平修正 0，垂直为约 1e−16 m
舍入量。原逐帧硬约束与局部 slack 分开，非线性剩余相交以上表实测为准。

生产验收检查 99 个保存姿态和 4,571 个帧间姿态，25,856,236 查询工作量，68.06 s。
缓存复用后局部阶段约 0.044 s，不再查询几何。采用下游同样的标量关节/四元数插值，
自适应运动间隔阈值 10 mm，上限 256，本次无触顶区间。这是离散检查，不是连续无碰撞
证明，不能排除比精度更小的短暂相交。

场景有 2 个闭合组件、16 个开放或不可靠组件；仅可靠凸组件进入 MuJoCo 代理，
最大顶点偏差约 2.02e−9 m，其余保持原三角查询。最终无后端异常、无含糊的
capsule/triangle 退出事件，但所有 99 帧仍有扫描侧别未知区域。最终状态仍为 unresolved，
分别保留 collision_residual 和 uncertain_scene_side，没有宣称可靠闭合实体安全。

最终 qpos 与验收返回数组、local 前后数组逐元素相等，SHA256 为
`19aca6ff3a2ddc5d70cf85873d1f0f30d40495c5a9dc1c6424225d2d5e20463f`。
另一次相同有效策略的 `segment_frames` 输出也逐元素相等。运行代码哈希与当前 Python
文件相符，运行配置等于从当前仓库 profile 展开的配置，没有临时 clip 特例。

## 1530：另一段真实动作的限制

使用 HER 原发布 `c167_img_1530`，143 帧、29.97003 fps，不是旧补帧的 144 帧实验。
相同默认算法和独立几何测量，产物为 `out/her_c167_1530_release_body/`。

| 指标 | 原结果 | 新结果 |
|---|---:|---:|
| 最大表面侵入 / 超过 1 mm 帧数 | 49.99 mm / 55 | 2.03 mm / 7 |
| 躯干骨架方向误差，平均 / 最大 | 11.13° / 29.73° | 13.35° / 45.69° |
| 左脚世界误差，平均 / 最大 | 49.58 / 129.64 mm | 40.70 / 166.93 mm |
| 右脚世界误差，平均 / 最大 | 55.28 / 167.26 mm | 51.92 / 129.08 mm |
| 左手最大接触法向残差 | 144.42 mm | 159.10 mm |
| 关节 jerk RMS | 732.88 rad/s³ | 819.33 rad/s³ |
| 全流程耗时 | 383.31 s | 226.19 s |

新相交帧 43、54、57、76、77、80、88，最长连续 2 帧，最差左腕第 54 帧。
左手保持 14 transition、10 sliding、119 none；没有 sticking，支撑滑移为 N/A。
滑动没有锁成六维固定姿态。最大接触误差在第 72 帧 sliding，置信度约 0.949；
未证明该帧不可达，不能用“预算”或“机器人较短”宣称接触问题已解决。
相交减少与交互/部分姿态退步同时存在，1530 不作为接触质量通过案例。

接触 mask、mode、法向逐元素相同。跨运行重算的 source 位置最大差 4.77e−7 m，
世界接触目标和末端参考最大差 7.15e−7 m，confidence 最大差 1.92e−6；如实记录浮点差，
未宣称这些数组逐位相同。原始源文件数组严格相同，场景 size/mtime 未变，没有有意移动
目标。1530 局部阶段同样完全 no-op，保存的是最终验收过的同一数组。

## 测试和产物

新增测试覆盖任意 source/body 名称、nq != nv、父相对姿态与点 Jacobian 有限差分、
参考姿态零误差、已知快速旋转/反向、readonly 接触输入、可信接触优先、无接触时已有
求解行为、首帧注册、末端完整旋转映射、退化/反向/有限边缘/轴穿面几何。

最终回归日志 `out/her_c167_010_diagnosis/verified_tests.log`：

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests
```

**267 passed, 4 skipped，15.12 s**。包含必需的 `tests/test_contact_priority.py`，此前也已
单独执行该文件通过。4 项跳过为既有 CLI 测试缺少其写死的路径，本次另实际运行两段 HER，
未修改测试资产路径伪装通过。`git diff --check` 通过。
已有插值直接成功、反向候选更好、节点安全而边穿障、接触模式、深箱、凹组件、
缺边界/资产、全部失败、无异常 no-op、不同 fps/重复/重叠窗口和 P2 校准均在回归中。

独立 A/B/C/D 局部合成对照见 [UMR 审阅的实测结果](umr_scene_interaction.md) 和
`out/umr_scene_abcd/`。真实 010/1530 无可信局部窗口边界，不能虚构 C 优于 B；没有启用 D
或证明真实表面增强收益。本次真实 A/B 是旧整段结果与修改后的逐帧求解。

可重新测量已保存输出：

```sh
.venv/bin/python out/her_c167_010_diagnosis/measure.py \
  out/her_c167_010_diagnosis/release_body/final_retargeted.npz
.venv/bin/python out/her_c167_010_diagnosis/body_metrics.py \
  out/her_c167_010_diagnosis/release_body
.venv/bin/python out/her_c167_010_diagnosis/audit_heldout.py \
  out/her_c167_1530_release_body
.venv/bin/python out/her_c167_010_diagnosis/final_report.py
```

`release_body/run.py`、`input_manifest.json`、`source.json`、`robot_profile.json` 记录实际 CLI；
runner 拒绝覆盖最终产物。`code_manifest.json` 固定 Python 文件；`final_report.json` 汇总
数值；`final_retargeted.diagnostics.json` 含有效默认值、查询可信度、事件、后处理拒绝原因；
`baseline.png`、`release_body.png` 为实际碰撞几何离线图。
1530 的 `independent_audit.json`、`contact_frame_errors.json` 保留未解决的接触帧。

8130 已验证 HTTP 200、WebSocket 几何和播放更新，页面有版本与蓝色原人体对照。
无可用浏览器会话，未宣称浏览器视觉验收；已检查独立离线几何渲染。
010 全流程比旧结果多 19.32 s，不能称更快，旧基线碰撞验收存在漏检。
1530 从 383.31 s 降至 226.19 s，也只是本机单次运行对照。
