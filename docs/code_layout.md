# 代码结构

Python 文件保持在 1000 行以内。按职责拆分，保留原有入口和调用签名；
不要通过压缩格式满足行数限制。`tests/test_module_size.py` 检查包、脚本和测试。

| 入口 | 职责 | 实现模块 |
| --- | --- | --- |
| `sequence_retargeting.py` / `retarget_sequence` | 可通过 CLI `--solver sequence` 使用的通用整段求解 | `_sequence/coordinates.py`：模型切空间、机械限位与按秒计算的差分；`tasks.py`：源固定交互图、接触与姿态任务；`_core/sequence.py`：主流程数据衔接。见 [使用说明](sequence_retargeting.md) |
| `_sequence/orientation.py` / `OrientationTargets` | 与根部平移独立的世界/父级相对姿态目标 | `_core/orientation_targets.py` 复用源绑定标定，无其他任务看管的末端（手）按 `UNOBSERVED_LEAF_SIGMA_SCALE` 收紧；[根部自适应说明](adaptive_root_motion.md) 记录论文依据和默认行为 |
| `_sequence/source_pose.py` / `orientation_seed.py` / `source_shape.py` / `pose_guard.py` | 用已标定的源朝向初始化，优先拟合原动作方向；姿态无角度上限，相对固定参考和原动作报告 | 根部跟踪使用源关节对应的机器人刚体；`pose_guard.py` 只限制根部/滑动关节平移，越界候选回退并报告残差 |
| `_sequence/restoration.py` | 恢复非线性姿态与连续性约束的可行性 | 保留固定参考及接触优先级预算；恢复失败则继续回退 |
| `_sequence/windows.py` | 长片段受保护修正步的时间窗口划分与窗口子问题 | 其余帧步长固定为零，所有硬约束与非线性验收不变；相邻轮次窗口边界错开半个窗口。见 [使用说明](sequence_retargeting.md#guarded-steps-numerics-and-runtime) |
| `_fast/pipeline.py` / `solve_fast_sequence` | CLI `--solver fast` 的速度优先整段后端：源形参考 → 接触/场景 → 预算验证与局部重加权 | `objective.py`：打包任务与分块并行评估；`terms.py`：与 `SequenceTasks` 等价的逐帧行；`rows.py`：按刚体 6×6 矩累加法方程；`solver.py`：带状 Levenberg–Marquardt 与精确关节限位/根部平移界；`reference.py`：按源世界/身体相对目标拟合高度、不可变参考的布局检查与姿态/源方向报告；`shape_step.py`：固定包络内的约束方向与非线性可行性恢复；`acquisition_guard.py`：接触获取阶段的支撑/未解释保持法向保护；`contact_guard.py`：不可变接触精度及位置验收；`continuity.py`：逐坐标和根部转动时序验收；`penetration.py`：自由部位穿插修正；`mesh_repair.py`：所有静态/运动物体和自碰撞的原网格有限关节修正，不依赖距离场是否失效；`collision_manifold.py`：与采样密度及查询分块无关的有界多点见证；`repair_step.py`：每组最大深度 epigraph 与最小关节改动子问题；`repair_limits.py`：固定根部、接触锚点、源姿态和逐关节时序预算；`sdf.py`：由原始三角形采样的场景网格缓存；`kinematics.py`：整段正运动学。见 [使用说明](sequence_retargeting.md#fast-backend) |
| `scene_probe_batch.py` | 批量查询原始场景，供无查询预算的离线验收使用 | 保留中心剔除、所有采样点和双向窄相位见证；显式查询预算保留逐次计数 |
| `triangle_candidates.py` / `triangle_capsule.py` | 胶囊／球体对原始有限三角形的保守筛选和批量最近点 | 保留穿面退出向量、歧义及并列见证顺序；见 [算法提速记录](algorithm_runtime_20260924.md) |
| `_sequence/scene_cache.py` / `moving_cache.py` | 原始静态和运动网格的整段查询批处理 | 不减少采样点；材料坐标查询、姿态失效及雅可比有等价性测试 |
| `core.py` / `OmniRetargeter` | 初始化、接触任务准备、重定向和后处理编排 | `_core/motion.py`：逐帧执行；`source_root.py`：源根部参考（躯干朝向估计与根关节选择）；`scaling.py`：可选地形缩放因子与源轨迹缩放；`geometry.py`：机器人几何推断；`feet.py`：足部稳定和局部表面修正；`postprocessing.py`：平滑、关节安全限制和质量报告；`visualization.py`：Matplotlib 预览 |
| `retargeting.py` / `GenericInteractionRetargeter` | 机器人配置、交互网格和逐帧优化循环 | `_solver/step.py`：组装 SQP 问题；`constraints.py`：接触链和根部约束；`hierarchy.py`：分层求解和步长保护；`orientation.py`、`pose.py`、`end_effectors.py`：各类目标；`kinematics.py`：Jacobian 和碰撞约束；`targets.py`：目标变换和关节限制 |
| `trajectory_refinement.py` / `refine_joint_trajectory` | 全轨迹迭代、时间正则化、稀疏最小二乘求解 | `_refinement/effectors.py`：解析末端配置；`tasks.py`：任务线性化；`guard.py`：复用最终验收预算的约束求解与非线性检查；`math.py`：坐标、朝向和差分工具 |
| `trajectory_validation.py` / `TrajectoryValidator` | 轨迹修改共享的 FK、接触和几何验收；原始机器人轨迹为不可变参考 | `trajectory_fidelity.py`：关节修正与连续性预算、接触/摆动/滑移检查及帧间子姿态接触检查 |
| `contact.py` | 场景接触推断和任务数据结构 | `contact_coverage.py`：逐帧表面覆盖与主要接触约束回退；`contact_scene_samples.py`：源表面采样的场景最近点、可靠法向和接触面片；`contact_schedule.py`：滞回、短接触段和间隙处理；`object_contact.py`：原始运动物体的主要端点接触与环境标识 |
| `contact_pass.py` | 显式启用的源侧近距离掠过审查，保留原始候选并生成独立试验任务 | 单一位置误差预算、完整事件运动检查和接触/支撑保护；[实验边界与复现](contact_pass_review.md)；默认主流程尚未启用 |
| `contact_correspondence.py` / `contact_windows.py` | 原始皮肤投影的时序对应、窗口间接触参考一致性 | `contact_scene_normals.py`：可活动空间方向与接触法向；`_sequence/proximity.py`：身体表面间隙；见 [接触连续性](contact_continuity.md) |
| `skin_frames.py` / `data_sources/smplx_feet.py` | 固定皮肤材料点的刚体朝向拟合及 SMPL-X 同体形足底标定 | 保留皮肤原轨迹和形变残差；末端标记与足底使用同一朝向，接触采样包含脚跟和侧缘 |
| `moving_evidence.py` | 将原人体接触的可活动侧证据固定到物体材料坐标 | `moving_scene.py` 对最近开放面片作局部有符号查询；未知区域不外推，修正不能通过丢失证据消除穿透 |
| `data_sources/trumans.py` | 已注册的 TRUMANS 源（`type: trumans`，`motion` 为 `Data_release` 目录，`segment` 必填，可选 `frame_start`/`frame_stop`） | Y-up 参数按 `trumans_review.WORLD_ROTATION` 刚体转到 Z-up（根平移为 `W (j0 + t) - j0`），再走与 OMOMO/AMASS 相同的 `SmplxDataSource` 生成皮肤、足底、标记和关节朝向；关节位置仍用发布的 `human_joints`，重建误差超过 1 mm 报错。房间网格、全部物体和 `scene_enclosed_free_space` 经主流程 CLI 传入 |
| `data_sources/trumans_review.py` | TRUMANS 原始人体、场景和椅子轨迹读取 | 椅子优先读取逐片段 `Object_all/Object_pose`，按 `frame_id` 对齐 XYZ 欧拉角与位置；缺少该记录的增强片段保留自己的汇总变换，禁止套用原片段轨迹。保留来源，不插值或改写观测 |
| `contact_mechanics.py` / `centroidal_demand.py` | 独立的接触合力/合力矩可行性解释 | `contact_temporal.py`：保留歧义的时序交接；`contact_support.py`：任意部位优先级；`support_observation.py`：有界间隙推断与原始观测分开；见 [接口与限制](contact_mechanics.md) |
| `conic_constraints.py` / `endpoint_constraints.py` | 批量稀疏端点、滑移和表面误差约束 | 共享单点及相邻帧预算，避免长序列的表达式展开；保持原非线性验收条件 |
| `whole_body_correction.py` | 固定机器人基准的全身接触/碰撞联合修正与时间正则 | `whole_body_guard.py`：固定末端预算、物体材料坐标帧间目标、实际几何回退；`moving_scene.py`：原始运动物体闭合组件与反向交叉查询 |
| `self_contact.py` | 全身自接触向量、双端 Jacobian 和误差验收；保留旧手—大腿输入兼容 | `full_body_self_contact.py`：源拓扑/测地近邻排除和固定点对；`surface_correspondence.py`：旧 `parts` 输入的区域绑定 |
| `surface_motion.py` / `surface_pose.py` | 三类任务共用的学习表面对应，以及无接触时的位置/法向姿态目标 | `full_body_self_contact.py`：非相邻源段的稳定材料点对；`_core/surfaces.py`：源与求解任务衔接；`_sequence/surface_guard.py`：整段求解中的固定主要末端和接触预算 |
| `surface_contact.py` | 全身表面到场景的接触向量、共享求解/验收与 NPZ 导出 | `surface_learning.py`：固定索引学习、绑定和缓存；`surface_anatomy.py`：关节链对应范围；`surface_observation.py`：材料坐标中的接触观测；`data_sources/smplx_surface.py`：真实全身表面采样 |
| `contact_phase.py` | 统一任意身体部位的支撑、未解释保持、物体控制、跟随和自由角色；支撑优先，几何保持不冒充承重 | `object_control.py`：源物体材料坐标的协同跟随证据；`contact_phase_observation.py`：源净空；`contact_schedule.approach_blend`：源进度交接；见 `docs/contact_phase_priority.md` |
| `local_collision_repair.py` | 局部修复事件、候选验收和诊断 | `collision_window.py`：窗口内关节优化和碰撞评分；`scene_witness_guard.py`：固定表面点的已知几何状态保护；`open_surface_features.py`：开放网格折边可靠性 |
| `staged_contact_repair.py` | 严格整段后端的可选逐接触刚体穿模修正 | `_sequence/contact_hold.py`：其他已建立接触的位移约束；固定根部关节修正后才尝试有退出证据的有界抬升。见 [分阶段修正](staged_contact_repair.md) |
| `self_collision.py` | 碰撞体自碰撞查询、间距约束和独立诊断 | `convex_mesh_distance.py`：完整凸网格支撑映射、最短退出距离及材料见证，不依赖检测间隙；`collision_motion.py`：场景/自碰撞共用运动范围估计；`_refinement/self_collision.py`：轨迹残差与非线性检查 |
| `main.py` | CLI、源配置和主流程 | `source_config.py`：源配置读取、适配器推断和浮动基座朝向帧选择；`motion_visualization.py`：可视化场景、视频导出和播放 |
| `mjviser_viewer.py` | mjviser CLI 和启动 | `_viewer_motion.py`：轨迹读取、源骨架和物体资源；`_viewer_scene.py`：场景、交互回调和录制 |
| `utils.py` | 通用几何和机器人加载工具、历史导入兼容 | `data_sources/smplx_motion.py`：SMPL-X 加载、表面标记和补帧；`motion_interpolation.py`：源适配器共用的位置和旋转插值；`interaction_graph.py`：邻接、源侧边权与 Laplacian |

`_core/` 和 `_solver/` 的 mixin 只划分同一实例上的方法，不创建额外的
机器人状态，也不覆盖初始化。公开类继续支持原来的构造方式、方法访问和
子类覆写。新增通用求解逻辑应留在求解模块；源格式解析属于数据适配器。

原模块定义的函数保留显式重导出，旧的 `core`、`retargeting`、`utils`、
`main` 和 viewer 导入路径继续可用。内部实现直接依赖具体的工具模块，
避免为了复用一个数学函数反向导入整个编排入口。

接触回归测试按目标优先级、场景推断和全轨迹细化分开：

```sh
pytest -q tests/test_contact_priority.py tests/test_contact_inference.py tests/test_trajectory_refinement.py
pytest -q tests
```
- `_fast/limb_scope.py`: source-independent kinematic branch scopes; paired foot
  repair locks the shared ancestor and every non-leg coordinate.

- `anchor_motion.py`: preserves observed scene material targets separately from
  source-following rigid pose fits; incompatible robot geometry cannot displace
  contact goals, and sliding observations remain time varying.
- `pose_relations.py`: morphology-normalized landmark relations and the fixed
  reference posture envelope (whole-body and per-landmark equivalent angles),
  with exact cones for conic steps and tangent limits for repairs. Identical norms
  of several envelopes share one auxiliary, and `LazyRelationFamilies` adds budget
  families only when a solved step violates them. See [pose relations](pose_relations.md).
- `motion_budgets.py`: shared physical defaults, including the windowed per-joint
  temporal budget (`windowed_budget`) used by the continuity guards and repair.
- `scene_geometry.py`: original-triangle signed queries. Nearly closed open components
  decide points the free-side evidence leaves unknown by their completion's winding
  number (`_decisive_winding`, `winding_side`).
- `_fast/repair_restoration.py`: restores nonlinear feasibility within unchanged
  local joint/contact/continuity budgets.
- `_fast/parallel_geometry.py`: frame-parallel original-mesh measurement for the
  repair. Complete measurements equal the serial result; a rejected candidate
  stops all workers at the first rejection found (same decision). See
  [runtime notes](mesh_repair_runtime_20260927.md).
- `moving_reverse_cache.py`: exact-transform original-mesh reverse witnesses;
  Jacobians remain current, with unchanged sample coverage and unknown states.
- `conic_solver.py`: shared conic entry point. `solve_conic_working_set` solves a
  small correction by exact constraint generation (same optimum); the fast
  restoration uses it only with `_fast/shape_step.WORKING_SET_RESTORATION`
  (off by default). Fast-objective values do not depend on the worker chunking.
  See [runtime notes 2026-09-29](runtime_20260929.md).
- `evidence_scope.py`: the open-surface "within the evidence radius" test by
  conservatively classified cells; only boundary cells query the KD tree.
- `_fast/proposals.py`: the damping attempts of one LM iteration; after a
  rejected first attempt the rest are computed in forked processes and taken
  in the sequential order (same step, history and guard counters).
- `source_support_evidence.py`: main-pipeline source net-wrench support evidence
  (SMPL-X segment inertia, friction scenarios, grounded moving objects); feeds
  `contact_phase` roles. See [contact roles](contact_phase_priority.md).
