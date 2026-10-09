# Fast 求解器的耗时与加速边界

本页说明 `--solver fast` 单 clip 墙钟花在哪里、哪些加速已经可用，以及哪些方向已经用真实数据
证伪。重新尝试被证伪的方向之前，先说明它与下面的数据有什么不同。

## 可用的加速（都不改变动作输出）

- `scripts/benchmark_fast.py` 默认并发运行独立验收审计（含逐对 persistence）；`--serial-audits` 可串行对照。
  只有基线 A 和显式 `--source-root` 指定的候选 arm，候选默认不启用任何诊断。
- `scripts/review_scene_metrics.py` 和 `scripts/review_training_collision.py` 的 `--workers`
  默认 `min(8, available_cpus())`；`--workers 1` 保留串行。按帧分块后仍以串行顺序归并，
  输出相同。training 审计的波动计时写到同名 `.timing.json`，审计主 JSON 保持确定。
  不要在验收里加入 GPU/SDF 预筛选：独立审计不能重复求解器近似的漏检。
- fast 只保留主线：精确 CPU 批量几何、自碰撞、短 clip canonical assembly、修复 working set、
  并行修复和窗口调度均直接使用；不提供实验或关闭主线行为的开关。
- 唯一运行诊断开关是 `OMNIRETARGETING_FAST_ENABLE=performance_trace`，只记录计时和调用。
  其他 FAST_ENABLE 名称报错；FAST_DISABLE 已移除。
- 原网格修复对有开放面证据的移动物体启用 `moving_clearance_cache.py`：在物体局部坐标中，
  物体凸包的外侧距离下界减去包含全部 forward 样本及 reverse 查询范围的球半径，仍大于
  activation band 时才跳过整个 geom。缓存按中心位移消耗余量；球包络覆盖任意旋转。
  scale、网格查询、凸包、证据及材料样本身份变化会重新计算。只返回与原 hull-far 分支一致的
  known 身份和空 witness；近场继续 CPU 原三角查询，不改变深度、unknown 或求解 guard。
  `prepare_moving` 只为原生 `SceneQuery` 启用，独立审计及自定义 query 保持原路径。
  8 个基准 clip 各 2 次，NPZ 全字段逐位相同；固定请求重放 254/78/182 跳过 65%/65%/82% 的 geom
  请求，witness/深度/unknown 的 SHA256 相同。omomo_254（whitechair）production 80.8→67.9 s、
  repair 50.7→39.0 s；78/80/485 约快 1–3%；182、5 和静态场景（椅子部件闭合，不触发）持平。
  剩余主耗时是近场开放面查询（whitechair forward material 约 61 ms/次），缓存无法跳过。

在 10 个基准 clip（TRUMANS sit180、wipe300、chair150、fridge957，OMOMO 5、78、80、485、182、254）上，
并发审计让端到端（求解加全部验收）快 1.18–1.40×，最终动作与审计输出和当前实现逐位相同。

## 耗时结构

- 短于 600 帧走整段 LM（`solver.py`）：banded 正规方程不满足 guard 时退回 `shape_conic`
  锥子问题。180 帧时一次锥子问题约 n=6300、m≈50k（约 20k 非负行加 7460 个 SOC），
  Clarabel 22–26 次内点迭代、1.1–1.5 s，是 LM 阶段的主要耗时。
- KKT 分解由时间带状的 LM 正规矩阵（Schur 半带宽约 110）主导，而不是约束行数：只保留
  5–7k 行时每次内点迭代仍约 33 ms（全量约 45 ms）。
- `original_mesh_repair` 占 contact_scene 的 25–50%。多数尝试只降 0.01–0.2 mm：线性化的
  接触/连续性约束与非线性 guard（`CONTACT_DRIFT` 1 mm）不一致，步长常回溯到 1/16–1/64，
  每次回溯都要做一次原几何测量。
- 开放移动扫描物体的 CPU 原几何查询仍是修复瓶颈；静态 GPU 预筛不等于移动查询已加速。
  deadline 覆盖加载到导出，原生查询返回后的检查延迟单列，不把预算当作迭代次数。
- 验收审计里最贵的是 `review_scene_metrics.py`（libigl AABB/winding 已在内部多线程）。

## 已证伪的方向

| 方向 | 结果 |
| - | - |
| 修正量用三次 B 样条表示（`spline_correction`，0.12 s 节点） | 同一批真实锥实例上每次求解慢约 2.5×：每行约束涉及 4 个控制点，约束非零元增至约 4 倍，而 KKT 不由变量数主导；每步下降也变小。整段样条路径（`spline_correction/adaptive/shared_loop`）在 4 个 clip 上全部耗尽每帧 1 s 预算，生产耗时为当前实现的 2.4–7.5×，静态/移动/自碰撞穿透加深到 80–135 mm。实验实现已移至 `out/archive/pre_cleanup_worktree.tar`，不再提供运行开关。 |
| 约束生成/工作集（零步种子、或 [0, 无约束步] 线段种子） | 需要 2–4 轮内点重解，内点法不能热启动，总时间比全量求解长。 |
| 区间预求解删除盒内不可能活跃的行 | 步长盒较宽，只能删除不到 0.3% 的行。 |
| 推测式执行（banded 与 conic 并发、修复回溯候选批量测量） | 结果逐位相同，但与几何 worker 争抢 CPU，墙钟反而变长。 |
| 多线程 KKT（faer） | 无收益。 |
| 冷启动单一协调 AL | 信赖域/filter 控制大步后仍获取不到关键接触；omomo_5 分离 p95 154 mm。不能替代 acquisition。 |
| 碰撞延拓 | 补齐根旋转步盒、弱阶段释放 filter 后，5 段 OMOMO deep/persistent 全部退化，0 段 eligible；omomo_80 腕穿深 27.2→115.2 mm。 |
| 只换初值 | 从旧最终运动起步仍破坏接触，首次退化归因包括根朝向界、soft-push、修正时间项；良好初值不足以解决公式冲突。 |
| acquisition + 协调 AL 混合（al_25–28） | al_25 的 9 个完成片：deep/persistent 30/22→29/28，chatter 53→90，reset invalid 1497→1625，耗时 1823→3031 s；另 2 段绑定异常，不能作为生产替代。 |
| 混合结构只加预算 | 1/3/6 s/帧中，omomo_254 都选 acquisition 初值；3/6 档同为 27 attempts。sit180 三档同为 17 attempts、P=3（旧最终 P=1）；增加预算不弥补修补链收益。 |
| 支撑片推广 | sit180 P=3/chatter=41、wipe300 P=5/chatter=19，仍未达到旧最终的 1/29、2/7；刚体片对应本身可冲突，不能靠加罚权修复。 |
| popsicle 手部契约 | 可视掌面在碰撞 capsule 内 16–42 mm，贴掌面与壳非穿透互斥；当前契约回到原始 mesh，不能跨模型比较后宣称改进。 |

协调 AL 的研究代码与快照留在研究工作区的 `out/restructure_20261008/`，未移入生产。
较早实验的删除前代码见 `out/archive/pre_cleanup_worktree.tar`（未提交实验另存
`out/archive/experimental_untracked.tar`）。结论保留，运行路径只保留主线。

## 剩余杠杆（未实现，都改变求解路径，需要按 AGENTS.md 指标做完整对照）

1. 让修复 QP 的接触/连续性模型与非线性 guard 一致（加曲率余量或二阶校正），减少回溯与原几何
   测量，估计每 clip 省 8–20 s。
2. 按非线性约束预测误差控制 LM 步长并复用模型因子，减少重求锥子问题，估计省 4–18 s。
3. 利用时间带状结构的专用锥内点法：离线探针显示每次迭代最多快 1.5–1.7×，后期 KKT 曾出现
   奇异，数值风险高。

三者单独都达不到 2×；求解器之外的预处理（约 10 s）和导出也占可观比例。

## 保留的环境变量

- `OMNIRETARGETING_FAST_ENABLE`：仅接受 `performance_trace`。
- `OMNIRETARGETING_WORKING_SET_STATS`：进程结束时打印 working-set 统计。
- `OMNIRETARGETING_GPU_GEOMETRY_DEBUG`：静态 GPU 筛选的抽样 CPU 交叉检查。
- `OMNIRETARGETING_CPUS`、`OMNIRETARGETING_WINDOW_NATIVE_THREADS`：工作进程/线程资源预算。
- `OMNIRETARGETING_SDF_CACHE`、`OMNIRETARGETING_WINDING_INCLUDE`：缓存及 native 编译头文件位置。

## 深/持续穿透诊断实验（opt/pen，1d4e2de 起点）

- whitechair 的 acquisition 网格最大深度 8.60 mm，原三角 33.32 mm；旧局部修复并非到期，
  而是被材料点 side vote 消失（frame 16/39）、接触位置冻结、逐 key 历史最小深度及窄关节范围挡住。
- 仅接回已存在的 `certified_outside`，omomo_254 最终 qpos 逐位不变。该例的缺失观察确实未知，
  不能把它改成自由空间；0.001 rad 关节扰动即有 5–48 个原负距离材料点变 unknown。
- 全身原三角 LM 若继续一票否决未知，152.55 s 只动数值量级；直接丢掉未知残差则手部误差
  114→349 mm。两个方向均否决。负无符号距离只作为保守求解残差，不是有符号证书。
- 全轨迹两级弹性 QP（先总碰撞 slack，再原目标）在 148 帧预算内只完成一次有效更新，
  145.35 s 后最大深度 33.32→33.44 mm；不适合生产预算。
- 无约束连续全身投影在 sit180、omomo_78、fridge957 都被已有高频边界拒步；回溯不能改变
  方向朝界外的问题。白椅 pilot 虽 33.32→29.62 mm，但力矩筛查 0→47，不能采用。
  候选改为直接带连续性线性限制的稀疏 QP，并在接受前执行原模型的逆动力学限值筛查。

完整逐对诊断、失败实验与十段前后数值见 `out/opt_pen/REPORT.md`。以上均未改审计、质量阈值、
源动作、场景和物体运动；不依赖 AL 冷启动、碰撞延拓或单纯提高罚权。

继续诊断的证据：全量加入机械线性化而仍用 OSQP，四段中 sit180/fridge957 未动，78 反而新增
persistent；5 s 退出时 primal residual 达 2.36e-3，不能称为物理不可行。相同线性 QP 换直接法后，
sit180 persistent 清零，chatter 29→22、effort screen 11→9；但 957 帧整段 KKT 首轮即约 20 s，
还没得到可行方向。最终方法保留快速候选，仅停滞后刷新一次；长片的直接法只展开当前政策事件
及相邻时间帧，全身关节/根和两方自碰撞 Jacobian 均保留，再对合并轨迹做同样的非线性检查。
不以增加 production 预算或放宽验收解决数值不可行。

### 最终 E：10 段完整生产对照

基线为 `omni_fix_self/out/baseline_1d4e2de/A/normal/repeat_1`；E 的冻结源码和输出在
`out/opt_pen/experiments/final_source`、`E/normal/repeat_1`。每段独立审计各一次，单次 repeat。
下表 D/P 使用未修改的 policy_v1；基线旧 classification 13/11 不等于 policy_v1 18/18。

| clip | policy D/P A→E | production s A→E |
|---|---|---|
| sit180 | 0/1→0/0 | 103.84→141.07 |
| wipe300 | 0/2→0/1 | 190.75→252.97 |
| chair150 | 2/2→2/2 | 70.80→86.25 |
| fridge957 | 3/3→3/3 | 195.15→216.89 |
| omomo_5 | 2/2→2/2 | 76.08→97.35 |
| omomo_78 | 1/0→0/0 | 109.84→116.56 |
| omomo_80 | 2/0→0/0 | 103.28→99.62 |
| omomo_485 | 1/1→0/0 | 138.27→107.40 |
| omomo_182 | 1/1→0/1 | 47.21→99.88 |
| omomo_254 | 6/6→6/5 | 67.41→143.03 |
| 合计 | 18/18→13/14 | 1102.62→1361.02 |

这是部分质量收益，耗时增加 23.43%，不是提速。每段 production 仍低于 1 s/帧；254 在
soft deadline 到期后用预留时间导出，143.03/148 s。原链和原网格局部修复仍消耗预算。
旧 classification 同口径为 13/11→11/9。chatter 合计 52→35，effort screen 238→229；
全部自碰撞最大深度、关节位置/URDF 速度超限和布局翻转为零。完整测试 2220 passed、9 skipped。

没有解决的方向也保留：chair150、omomo_5、fridge957 的全身候选激活新的 native 自接触，
导致三帧机械线性化没预见的力矩跃变（例如 chair frame 85 left_hip_roll 638.71>139 Nm），
精确 screen 拒绝，三段最终 qpos 与基线逐位相同。wipe300/182 在持续性分段平坦区无继续
改善的方向。254 仍有 28.836 mm 手部穿透，必需材料观察失败由 0 增至 120 帧；负无符号
残差不是有符号距离证明。182/78 同类失败分别 43/3 帧，不能用独立审计开放 component 计数
未变来掩盖。物理回放和实际接触力分配仍未验证。

姿态/接触代价不隐藏：sit180 relation p95 0.6320→0.7125 rad、右脚趾法向 p95
12.149→19.855 mm；254 左手法向 max 89.693→99.998 mm；485 右手滑动 max
0.4226→0.5307 m/s。逐 clip、逐碰撞对、最差身体/关节/帧、根补偿和全部缺陷见
[`out/opt_pen/REPORT.md`](../out/opt_pen/REPORT.md)。训练模型 hash matched 不等于物理验收通过。
