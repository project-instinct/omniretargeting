# 动作质量实验：审计与消融

## 实现前审计（2026-09-11）

基线是任务开始时的**脏工作树**，不是 HEAD 的旧实现；HEAD 为
`0f9d2ebe4ee17d7a6173c739bb3ff3a3a5918d53`。修改前的 Python 包备份于
`/tmp/omniretargeting_motion_quality_baseline/omniretargeting`，用于默认路径对照。

1. CLI `main.main` → `OmniRetargeter.retarget_motion` (`core.py`) →
   `FramewiseMotionMixin._perform_retargeting` (`_core/motion.py`) →
   `GenericInteractionRetargeter.retarget_frame` → `_optimize_configuration` →
   `OptimizationStepMixin._single_optimization_step` (`_solver/step.py`)。
   首帧从 `model.qpos0` 出发，再注册源 root 位置/朝向；后续帧从前一完整求解结果
   warm-start，并按原配置执行 source-root prediction、root 锁定。
   hand-contact overlay 使用同帧 baseline 作初值；无接触且自碰撞检查通过的帧直接复用。
   `_optimize_configuration` 原本还会执行 `_self_collision_initial_guess`。
2. 没有独立的逐帧 bone-direction 初始化。`config_templates/bones_seed_template.yaml`
   是 Bones-Seed **数据源**配置，不是姿态初始化。
3. `_solver/pose.py` 的 `_setup_pose_bones` / `_build_pose_direction_residual`
   已有映射骨段的世界方向项（机器人向量除以参考骨长），属于完整 SQP；其 velocity /
   acceleration 项同样属于完整求解。`_solver/hierarchy.py::solve_hierarchy` 决定优先级。
   `_solver/orientation.py` 在完整 SQP 加入 calibrated body orientation；
   `configure_source_orientation_reference` 每次 motion pass 固定标定一次，使用 source
   bind/rest pose（缺失时首源帧）与机器人 reference pose。profile 声明 reference_pose
   时覆盖位置拟合，固定 twist 约定。`unitree_g1_instinctmj.json` 已声明该姿态。
   `trajectory_refinement.py` 为下游整轨迹 refinement，有独立的末端/手臂姿态与时序任务，
   不执行 interaction-mesh Laplacian，也不是逐帧 seed。
4. `retarget_frame` 对 `[human_joints; terrain_points]` 每帧进行 Delaunay，
   `utils.get_adjacency_list` 建邻接；源端 `calculate_laplacian_coordinates(..., True)`
   每帧一次。机器人端在每次 `_single_optimization_step` 重建 uniform L、`L ⊗ I3`。
5. 当前 uniform 两侧等价：权重只依赖固定邻接。机器人位置与 Jacobian 均按
   `valid_joint_names` 堆叠，静态场景节点的 Jacobian 为零，使用 vertex-major xyz 展平。
6. `utils.calculate_exponential_edge_weights` 已存在但生产路径无调用，且直接 exp/求和
   会在全行下溢时产生 NaN。旧 False 分支不是指数：逐点函数为 `1/(1.5d+epsilon)`，
   矩阵函数为 `1/(d+epsilon)`；此外在机器人姿态上重算距离会与源权重不一致。
   因此保留旧 uniform 路径，复用并稳定化指数 helper，再显式传递冻结源算子。

## 论文依据与边界

依据 [TopoRetarget v2 §3.2–3.4](https://arxiv.org/html/2606.16272v2)：
式 (1) 是相邻单位骨方向的**向量差之差**，并非仅匹配夹角；式 (2) 还有上一初始化的
时序项。式 (5)–(7) 明确在人侧计算指数距离权重并在机器人侧复用。
本次全身实验用局部单位方向残差，不照搬手部时序项或参数，不新增完整求解的 posture loss。
这是本项目适配，不能外推为论文验证的全身方法。

已检查[论文所链作者项目页](https://toporetarget2026.github.io/TopoRetarget/)；
截至本次审计页面未提供可验证的官方算法源码链接。搜索出现的 `0iui0/toporetarget`
明确自称 unofficial，未用其实现细节作为论文依据。原站根路径返回 404，子路径可访问。

## 最小实现

配置只增加两个独立入口，不改机器人 profile：

```json
{
  "retargeting": {
    "end_effector_priority": {
      "pose_initialization": {"enabled": false, "max_iterations": 20},
      "interaction_mesh": {"weighting": "uniform", "kappa": 3.0}
    }
  }
}
```

B 设置 `pose_initialization.enabled=true`；C 设置
`interaction_mesh.weighting="exponential"`；D 同时设置。A 使用上述默认值。
其余既有配置通过原来的 deep merge 保留，不复制一个全套新 profile。

### B：独立的局部骨方向 seed

`_solver/initialization.py::PoseInitializationMixin` 从既有 `pose_bones` 的路径末端
取连续两段，沿编译模型的祖先关系寻找最近的已标定参考 body（一般手臂用躯干，腿用骨盆）。
复用 body-orientation tasks，包括 `parent_relative` 的 parent body；没有第二套 G1 关节名。
没有两段映射、链不连通、缺失参考标定或 source orientation 时，返回原初值并记录原因。
position-only 来源第一版安全回退，不临时从当前姿态猜测固定坐标偏置。

令固定标定为源 `R_s0` 与机器人 `R_r0`，当帧源方向参考为
`R_target = R_s(t) R_s0.T R_r0`，机器人参考为 seed 中参考 body's `R_r(q_init)`。
每段残差是 `R_r.T normalize(p_child-p_parent) - R_target.T normalize(s_child-s_parent)`。
两侧都在由同一次既有 reference-pose 标定对应起来的参考坐标中比较，未直接比较原生轴。
参考 body 与祖先不在变量集合中，因此优化时这些坐标不随肢体变量变化。

使用现有 `_compute_robot_jacobians`，方向 Jacobian 为 `(I-dd.T) J_vector / length`，
再左乘固定参考转置。每段完整三维单位方向保留肘/膝朝向、弯曲角与非退化弯曲平面；
单段绕轴 twist、末端绕轴姿态、伸直时的弯曲平面仍不可观测。
原有 fixed calibration、完整 orientation 项继续负责这些自由度。

SciPy bounded least-squares 只改参考 body 之下、既有优化变量中可动的标量关节，复用既有
机械/安全/任务关节界限。没有 root 更新、人体尺寸重标定、每帧 calibration 或新速度上限。
没有增加时间平滑或追踪 seed 的完整求解 loss。后续帧起点仍是原有 warm-start；
20 次函数评估未收敛、非有限值、错误增加、越界或异常均原样回退。
`last_pose_initialization` / motion pass 的 `last_motion_solve_reports` 记录原因、误差与耗时。
失败诊断是 numerical prefit 的结果，不等价于完整求解失败。

### C：只在源端计算距离算子

输入就是 `retarget_frame(human_joints, ...)` 已有的
`vertices = [human_joints; self.terrain_points]`；它来自
`processed_trajectory[frame, mapped_joint_indices]`。平地实验 scale=1.0，此处不再次缩放。
邻接仍是原有每帧 Delaunay，节点、场景采样与随机种子不变。

复用并稳定化 `utils.calculate_exponential_edge_weights`，新增小型矩阵装配函数
`calculate_source_laplacian`。参数 `kappa=3.0` 的单位为 **1/m**，即 e-fold 距离为
1/3 m；这是待消融的全身实验值，不是论文手部参数，也没有每帧改变的尺度。
每行先减最短边距离，再指数化归一化，避免整行下溢；重复位置合法，孤立节点零行。
有限极端坐标用扩展精度距离计算；拒绝非有限输入、非法索引、重复邻接和自环。
邻接顺序原样对应权重数组。旧 inverse-distance helper 分支保留兼容，实验不走该分支。

同一个只读 `L_source` 显式传给 `_optimize_configuration` 和每次
`_single_optimization_step`，同时用于 `target=L@V_source`、`lap0=L@V_robot` 与
`J_L=(L⊗I3)@J_full`。xyz 按节点展平，场景节点 Jacobian 为零。
uniform 保留旧函数与计算顺序。未改 root/base 锚定、图拓扑、contact 或 collision 策略。

## 运行与审计边界

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/motion_quality_ablation.py \
  --motion test_data/c167_1530_final_v3/smplx_amass.npz \
  --model-directory assets/body_models \
  --robot-config robot_models/unitree_g1/unitree_g1_instinctmj.json \
  --frames 16 --stride 1 --output out/motion_quality
```

默认跑 A/B/C/D；`--variants B` 可只跑一组；`--window label:30` 可明确原始源帧起点。
stride=1 保留 C167 原有约 29.979 Hz；对 120 Hz 数据可以用 stride=4 选取每第四帧。
读取沿用现有 adapter 的 timeline 规则，包括已有的缺失帧插值，未修改 loader、文件或时序。
每个窗口记录原始帧号、源文件/位置 SHA256、实际 package 每文件 SHA256、完整有效配置、
随机种子、原始逐帧与最终轨迹、两阶段指标与逐帧数值状态。
`--source-cache` 是本机可选缓存，校验输入文件哈希及人体模型目录。

本轮运行中检测到另一路工作同时修改 self-collision projection（非本任务改动）。因此
主消融四组统一使用 `out/motion_quality_frozen_final/omniretargeting` 的冻结快照，通过
`OMNI_ABLATION_PACKAGE=out/motion_quality_frozen_final` 加载；结果在
`out/motion_quality_c167_timeline/{A,B,C,D}`。没有关闭该投影或回滚别人的改动。
本任务额外记录投影前的逐帧输出，使其与最终后处理结果明确区分。
因此整管线 A 的准确含义是**冻结快照中的原初始化 + uniform**；不能把整管线变化全部
归因于相对任务开始快照的两个实验。独立 Generic solver 的关闭开关回归另行对照原备份。

四帧 smoke 被主动中止以冻结版本；初选 AMASS 140_02 经源骨架检查主要是贴地动作，
其后处理严重受既有碰撞修复支配，已中止该探索套件并改用更接近直立肢体动作的 C167。
这些未完成探索不参与排名。C167 缓存也检查了 timeline 语义，主运行使用默认 adapter
处理后的 144 帧；原始文件的 143 行未被重写。

已完成结果可重新汇总：

```bash
.venv/bin/python scripts/motion_quality_report.py out/motion_quality_c167_timeline
```

汇总器核对源、帧率、窗口、package SHA256 和移除两个实验开关后的配置完全一致；
缺少任何组的窗口明确标作未完成。`comparison.json` 还记录固定当前源位置、只改变邻接
的图重连目标跳变量，不把不同算子下的数值大小当成动作质量排序。

## 验证结果

### 实际运行

主消融：C167，同一 G1 torso-base MJCF，同一平面和 scale=1.0、采样种子 0、
100 个原有场景样本、完整 SQP 默认 10 次迭代，原有权重/接触/碰撞/后处理均保留。
四窗口在默认 adapter 的时间轴上为 `[95,111)`（抬手特征）、`[51,67)`（屈膝特征）、
`[16,32)`（双手靠近特征）、`[54,70)`（快速肢体特征）。每窗口 16 帧，屈膝与快速窗口
重叠 13 帧，因此不是四个独立动作片段；共 51 个唯一时间点，每组求解 64 帧次。
窗口由源端特征预先选定，不按机器人表现选例，没有专门验证完整挥手、转身或蹲起周期。

以下是两阶段共同的结果：**16 次运行的最终轨迹与最后一个 pass 的投影前 SQP 轨迹
逐元素完全一致**。原后处理实际执行，但候选均由既有接受检查回退（accepted_fraction=0）；
不是关闭后处理，也不能据此宣称后处理改善了本实验结果。

| 组 | 八段肢体平均方向误差 ° | 双脚平均世界位置误差 cm | 初始化总秒 | 完整求解总秒 | 管线总秒 | 初始化回退 / 64 |
|---|---:|---:|---:|---:|---:|---:|
| A | 4.3475 | 6.6023 | 0 | 62.544 | 114.719 | 0 |
| B | 5.8364 | 10.8593 | 1.351 | 133.899 | 163.677 | 36 |
| C | 4.3511 | 6.5938 | 0 | 60.870 | 117.830 | 0 |
| D | 5.3733 | 9.5116 | 1.277 | 107.975 | 152.497 | 41 |

初始化平均每次调用 B 21.1 ms、D 20.0 ms；均是数值拟合，不含完整求解。
总秒为 `retarget_motion`（含标定、建图、完整求解及全部后处理），不含人体模型加载和
离线评价。各组单次测量、相同线程设置，不能据此作显著性或稳定速度提升结论。

| 源窗口 | A 方向 ° | B 方向 ° | C 方向 ° | D 方向 ° | A/B/C/D 关节 jerk 的窗口 p95，rad/s³ |
|---|---:|---:|---:|---:|---|
| 抬手特征 | 2.600 | 4.525 | 2.599 | 3.273 | 2049 / 23116 / 2051 / 11116 |
| 屈膝特征 | 5.113 | 5.862 | 5.120 | 3.800 | 4110 / 15129 / 4063 / 2859 |
| 双手靠近 | 3.340 | 6.451 | 3.340 | 6.314 | 1603 / 30069 / 1775 / 31683 |
| 快速肢体 | 6.338 | 6.508 | 6.344 | 8.105 | 3933 / 7964 / 3942 / 13756 |

原始逐帧及后处理后各骨段、各末端、构型和时序数据均保存在每个案例 JSON；
两阶段表见 `out/motion_quality_c167_timeline/comparison.md`。

- **端点与接触分开**：双手世界 marker 误差也记录在 JSON/表中，但自由手的绝对世界
  位置不是目标，不能据此给方法排名。生产路径在本片段仅用双脚世界跟踪，没有足部接触
  mask，且手部接触为零。接触位置/法向残差、支撑相水平滑移均 **n/a（无有效样本）**，
  不能写成 0 或声称已验证支撑相滑移。对应合成接触兼容性测试另行运行。
- **构型与连续性**：双手靠近窗口的最大弯曲平面单帧转角 A/C 约 10.72°，B/D 约
  70.14°；上述 jerk 也显示 B/D 明显退化。完整 JSON 同时报源平面步长和方向速度误差，
  避免把源动作本来很快误称为抖动。
- **幅度与滞后**：没有统一的幅度缩水；B 抬手/双手靠近的方向幅度比分别 1.23/1.32，
  A 为 1.03/1.02，D 为 1.12/1.28。这种伴随跳变的幅度放大不能当作更忠实。
  ±3 帧的局部滞后估计已保存，但 16 帧冷启动窗口太短，不能可靠判定长期跟随滞后。
- **数值状态与机械限位**：每组 64/64 个完整求解调用得到有限迭代结果，无报告失败的
  凸子问题；存储帧非有限值与机械越界均为 0。B/D 回退均因 20 次函数评估未收敛。
  这不代表运动验收通过。
- **场景与自碰撞分开**：同一平面下确定性碰撞壳采样的最大 scene depth 均为 0，
  无最深 body/frame；这不是连续几何的零穿透证明。既有自碰撞区间验证在 A/C 四窗口
  均 passed；B 四窗口、D 除屈膝外三窗口为 **incomplete**（达到区间细分上限）。
  B 双手靠近区间已采样最大深度约 0.720 mm，D 约 0.974 mm；不能把未知区间当作通过。
  各受影响区间编号在原 JSON 的 `subdivision_limit_intervals` 中，未关闭检查以降低失败数。
- **图跳变**：原 uniform 图确有目标跳变。固定当前 V、只替换邻接后，全部节点的最大
  目标变化在四窗口为 0.945/1.908/0.816/1.908 m（包含远处场景节点）；指数算子的
  相应数值约 0.075/0.093/0.091/0.096 m。它们是不同算子的诊断，不能跨算子当作收益；
  实际方向和足端指标表明 C 在本例的影响极小。本轮没有添加拓扑时间平滑。

### 工程验证

`tests/test_motion_quality_experiments.py` 覆盖：已满足姿态、两段方向下降、弯曲平面、
root/躯干/固定 calibration 不变、机械限位、失败与非有限回退、快速 2 rad 源步进，
源局部原生轴/全局平移不变性；指数归一化/距离次序/重复点/极端距离/孤立点/非法输入/
节点重排；逐点与矩阵一致；生产 SQP 传递同一只读算子并核对真实约束的值和 Jacobian；
浮基有限差分用 `mj_integratePos`，种子标量 Jacobian 同样用切空间扰动。
合成接触集成测试在既有 actuated-only 模式固定基座，四组检验同一个 2 mm 端点要求及
相同 root 锚定，保留 scene/self-collision 开关；不借浮基软姿态容差测试接触精度。

修改前脏工作树备份与关闭实验的 Generic solver：同一三帧、每帧三次 SQP、相同种子、
启用 scene/self-collision，输出逐元素相等，最大 qpos 差为 0。复现脚本和数组保留在
`out/motion_quality_20260911/omni_default_equivalence.py` 与 `default_equivalence.json`。
这项比较不冒充包含另一路自碰撞投影改动的整个 CLI 的旧版本等价性证明。

最终执行 `OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q tests -rs`：
**374 passed, 4 skipped, 4 warnings，21.37 s**。四条 warning 来自既有快速关节动作测试
的 CVXPY inaccurate 提示；四个 skip 都是旧 CLI 集成测试固定要求 `/localhdd/Datasets`，
本机缺少该路径。因此跳过不能算通过；主消融已实际使用 `assets/body_models` 的本地模型完成。
新增/修改的实验模块、运行/汇总脚本和测试通过 Ruff；模块长度测试通过。
消融快照之后另补充了关闭/失败回退时保留输入 dtype 的兼容性测试与处理。
主消融的初值/轨迹均是 float64，此处理没有改变所测数值路径。

## 结论

1. **B：不采用当前每帧预拟合方案作为生产路径。** 合成几何问题能拟合成功，但真实
   短窗中的方向、足端误差和连续性总体退化，且数值预拟合回退比例高。保留关闭的实验
   开关用于复现；不补强 posture loss、不增限速或逐帧挑选来掩盖该结果。
2. **C：保留正确接线的实验能力，默认仍 uniform。** 本轮方向差仅约 +0.0036°，足端
   均值改善约 0.085 mm，没有实质收益证据。更长动作、有真实接触标签的数据和其他
   预先确定的 kappa 可继续实验，但本轮不将它升级为默认。
3. **D：不采用默认组合。** 屈膝窗口有收益，其余窗口退化，不能据单窗口声称组合更优。
   长时序、完整挥手/转身/蹲起周期、真实支撑相滑移、跨机器人稳定收益均尚未验证。
