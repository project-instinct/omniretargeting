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

## 2026-10-09：近场精确筛选（opt/speed）

基于 `fix/infra-gaps` 的 `1d4e2de`，本轮只优化查询计算，不改源动作、接触/碰撞政策和求解决策规则。
保留三项：

- 闭合组件多于一个时，用已经变换到物体局部坐标的实际材料样本 min/max 盒，排除离该组件
  activation band 之外的整组样本，保守余量覆盖舍入边界。近处保留原逐点 mask 和原查询顺序。
- 原生开放/近闭合 moving repair 的凸包分类，在已有 `ConvexHullScope` 缓存的未决边界点上用
  C++ 提前判定布尔阈值，不生成 point×facet 大矩阵。只返回能证明相同的布尔结果；靠近阈值时
  回退**整个原 boundary batch** 的 NumPy/BLAS，避免缩小批次触发 GEMM/GEMV 末位差异。
  深度、witness、法向、winding、开放面证据仍来自原查询；编译器不可用则保留原路径。
- 无深层 enclosed 样本消费的 winding grid 延迟创建，真正需要时仍按原算法构建；空调用的预热语义保留。

这没有引入新的运行开关。基线已有的精确 geom/object pose 缓存和局部坐标驻留 winding/AABB
不是本轮新增；旧 trace 的完整 frame/pose 几乎不重复，再加同类缓存缺少收益空间。

### 正常计时（每臂两次）

`scripts/benchmark_fast.py` 冻结两臂源码，五段各按 A→B 顺序交替两轮；所有重任务都用
`flock /tmp/omniretargeting_heavy.lock`，显式 `PYTHONPATH=/home/duanxin/Xyk/omni_opt_speed`。
i7-13790F（16 物理核、24 逻辑 CPU）+ RTX 3090；fresh subprocess，保留 OS 缓存，
runner 固定 BLAS/OMP/MKL/NUMEXPR 单线程、torch 4 线程。正常计时关闭 trace/cProfile。
production 是 `timing.production_seconds`（加载到导出）；repair 是 `stage_seconds.original_mesh_repair`。
表中单位 s，均值 [最小, 最大]；两次样本只支持本机实测，不代表跨机器统计结论。

| clip | production A | production B | repair A | repair B | repair A/B |
|---|---:|---:|---:|---:|---:|
| omomo_254 | 67.78 [63.54, 72.03] | 61.75 [61.40, 62.11] | 40.97 [37.66, 44.28] | 35.93 [35.60, 36.27] | 1.140× |
| omomo_78 | 96.33 [95.63, 97.02] | 95.08 [94.93, 95.22] | 38.83 [38.59, 39.07] | 37.69 [37.63, 37.74] | 1.030× |
| omomo_485 | 93.72 [93.42, 94.03] | 93.89 [93.32, 94.45] | 47.49 [47.41, 47.56] | 47.54 [46.89, 48.18] | 0.999× |
| fridge957 | 182.64 [181.59, 183.69] | 182.43 [182.43, 182.43] | 51.93 [51.55, 52.31] | 52.00 [51.91, 52.09] | 0.999× |
| sit180 | 91.58 [91.57, 91.59] | 90.42 [90.07, 90.77] | 29.56 [29.41, 29.71] | 28.17 [28.03, 28.31] | 1.049× |

**整段 repair 2× 目标未达到。** 五段四次最终 qpos 和 NPZ 全字段逐位相同，最大 qpos 差为 0；
各次独立逐对 persistence（含阈值持续时长/占比、unknown、最坏帧）及 reset 核心结果相同。
没有为速度放宽验收；现有深穿透/持续穿透和 unknown 缺陷仍在，物理回放未验证。
完整测试：2223 passed、9 skipped，114.79 s，包含接触与模块行数检查。

### 热点与未采用的实验

先以 `OMNIRETARGETING_FAST_ENABLE=performance_trace` 在冻结基线上剖析：omomo_254
production 75.156 s、repair 39.974 s。另对相同的 2143 次生产请求按 worker 缓存顺序重放；
以下是开启 cProfile 的单进程累计 elapsed（嵌套项重叠），不能当作并行 production 墙钟：

| 热点 | A s | B s |
|---|---:|---:|
| 全部查询（排除 hash/setup） | 55.508 | 48.031 |
| 闭合组件 `_forward_samples` | 8.284 | 1.749 |
| 开放面 `_open_forward_samples` | 23.351 | 22.009 |
| `SceneQuery.query` | 15.104 | 15.391 |
| 凸包分类（含 cache） | 5.054 | 3.437 |
| reverse witnesses | 5.300 | 5.439 |

重放 206368 个 witness、known/penetrating 身份、unknown 计数及全部输出 digest 相同。
此重放候选为 v2；最终版本仅将包围盒舍入余量改为随尺度变化，正常五段计时和完整测试使用最终版本。

整段父/worker profile 的 repair 为 47.063 s，其中父 `geometry.measure` 23.450 s、
初始化 5.602 s、`solve_step` 6.015 s。query 以外仍有显著成本；worker 的 `recv/read` 包括等待，
不应归为计算热点。fridge957 的移动对象全部闭合，上轮 moving 累计 5.9 s 而 self 50.8 s，
因此不能把它的整段修复慢归因于开放移动面。

把整个对象的最近组件 AABB 串行化，150 请求反而 3.275→3.548 s，已撤回。
大组件批量改用 libigl 并行的慢 worker 探针中，AABB 0.928→0.142 s，但一次整 clip pilot 的
repair 35.810→37.077 s；没有观察到整段收益，未合入。这是单次探针，不能推断统计显著性。
剩余热点为开放面原三角查询、证据特征/身份 set 合并、reverse，以及修复父进程的初始化和求解。
继续追求整段 2× 需要覆盖这些部分，不能用局部 kernel 倍数替代整体结果。

完整证据留在本 worktree 的 `out/opt_speed/REPORT.md`、`comparison.json`、`contracts.json`、
两轮 `metrics_repeat_*.md`、冻结两臂和原始日志中；输出目录不提交到 Git。

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
