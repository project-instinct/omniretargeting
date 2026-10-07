# Fast 求解器的耗时与加速边界

本页说明 `--solver fast` 单 clip 墙钟花在哪里、哪些加速已经可用，以及哪些方向已经用真实数据
证伪。重新尝试被证伪的方向之前，先说明它与下面的数据有什么不同。

## 可用的加速（都不改变动作输出）

- `scripts/benchmark_fast.py` 默认并发运行独立验收审计；`--serial-audits` 可串行对照。
  只有基线 A 和显式 `--source-root` 指定的候选 arm，候选默认不启用任何诊断。
- `scripts/review_scene_metrics.py` 和 `scripts/review_training_collision.py` 的 `--workers`
  默认 `min(8, available_cpus())`；`--workers 1` 保留串行。按帧分块后仍以串行顺序归并，
  输出相同。training 审计的波动计时写到同名 `.timing.json`，审计主 JSON 保持确定。
  不要在验收里加入 GPU/SDF 预筛选：独立审计不能重复求解器近似的漏检。
- fast 只保留主线：精确 CPU 批量几何、自碰撞、短 clip canonical assembly、修复 working set、
  并行修复和窗口调度均直接使用；不提供实验或关闭主线行为的开关。
- 唯一运行诊断开关是 `OMNIRETARGETING_FAST_ENABLE=performance_trace`，只记录计时和调用。
  其他 FAST_ENABLE 名称报错；FAST_DISABLE 已移除。

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
- 验收审计里最贵的是 `review_scene_metrics.py`（libigl AABB/winding 已在内部多线程）。

## 已证伪的方向

| 方向 | 结果 |
| - | - |
| 修正量用三次 B 样条表示（`spline_correction`，0.12 s 节点） | 同一批真实锥实例上每次求解慢约 2.5×：每行约束涉及 4 个控制点，约束非零元增至约 4 倍，而 KKT 不由变量数主导；每步下降也变小。整段样条路径（`spline_correction/adaptive/shared_loop`）在 4 个 clip 上全部耗尽每帧 1 s 预算，生产耗时为当前实现的 2.4–7.5×，静态/移动/自碰撞穿透加深到 80–135 mm。实验实现已移至 `out/archive/pre_cleanup_worktree.tar`，不再提供运行开关。 |
| 约束生成/工作集（零步种子、或 [0, 无约束步] 线段种子） | 需要 2–4 轮内点重解，内点法不能热启动，总时间比全量求解长。 |
| 区间预求解删除盒内不可能活跃的行 | 步长盒较宽，只能删除不到 0.3% 的行。 |
| 推测式执行（banded 与 conic 并发、修复回溯候选批量测量） | 结果逐位相同，但与几何 worker 争抢 CPU，墙钟反而变长。 |
| 多线程 KKT（faer） | 无收益。 |

这些实验的删除前代码见 `out/archive/pre_cleanup_worktree.tar`（未提交实验另存
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
