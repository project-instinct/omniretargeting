# UMR 机制审阅与场景交互增量实现

后续新增的学习式全身表面场景接触映射见
[全身表面场景接触](whole_body_surface_contact.md)。它独立于本文历史 P2 的
少量法向辅助任务，已接入逐帧求解和后处理接触保护。

后续 HER 010/1530 真实动作与当前默认值见
[身体跟随和开放场景实测修复](her_010_scene_and_flip.md)。下文保留各阶段当时的配置、
合成对照与结果；当前查询点上限已取消，默认帧间细分上限为 256，不能把历史默认值
当作最新执行配置。

审阅基线：UMR `8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09`；本地
`main@e25e4c08cf3498def903abc3d664705b1ce2f9d9`，2026-09-08。
工作开始时 core/main/retargeting/defaults/refinement 和两个测试已有未提交修改；
scene_geometry、trajectory_validation、local_collision_repair、相关测试和前次报告
已存在但未跟踪。本次在实际代码上增量修改，未 reset、切分支、提交或 push。
`.omo/`、`assets/` 和机器人模型未修改。

## 许可与来源

检查了指定提交的 GitHub 递归文件树、README、defaults、robot config 和所列脚本。
该版本未发现覆盖项目代码、模型、数据的顶层 LICENSE/COPYING/NOTICE；唯一含
LICENSE 的路径是 `scripts/vendor/three/LICENSE`，不能据此认定其余内容可复用。
README 还明确要求用户自行取得 SMPL-X 模型并接受提供方条款。
结论仅针对审阅提交：复用许可未确认，不将 UMR 代码、权重、机器人或场景资产搬入本项目。
仅下载文本到仓库外 `/tmp/umr-review-8c4db7db` 进行审阅。下列模块是依据机制独立实现。

官方参考：[固定版本代码树](https://github.com/hanyang9/UMR/tree/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09)、
[README](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/README.md)。

## 值得学习什么，以及本项目已有什么

| UMR 官方机制／函数 | 审阅结论与本项目映射 |
|---|---|
| `solve_sequence`、batch `solve_stream_sequence` | 不同时间方向提供不同 warm start，能避开局部 IK 分支依赖。本项目此前只有插值和局部联合优化；新增 `local_candidates.bidirectional_window`，只处理插值失败、有可信两端的局部窗口。 |
| `select_bidirectional_dp`，单序列及 batch 各一份 | 官方比较两个方向的 solver frame cost，加关节差与切换费，没有检查连接碰撞。本项目独立实现最多四状态、统一非线性评分、禁用不合格边、含修正加速度的二阶 DP。 |
| `build_robot_object_penetration_cache`、`compute_robot_object_penetration_rows` | 完整碰撞几何比仅表面点更能覆盖箱角、容纳关系。新增 `geometry_query.RigidGeometryQuery`，经 `ScenePenetrationProbe` 同时进入现有逐帧、全轨迹、局部优化和最终验证路径。 |
| `collision_relative_jacobian` | 距离符号决定 witness 差向量的方向，约束建立在切空间。本项目已有逐帧 `J_v @ T(q)` 转 qpos 参数化，局部 scalar joint 已有 qi/di 配对；新增距离查询沿用原模型 nv Jacobian，仅选允许的 di 列。 |
| `solve_frame_body_segment_qp`、`solve_clarabel_qp_step` | 将位置、法向、接触、碰撞等项线性化，在有限步长内重算。复用本项目 CVXPY/Clarabel 和局部 SQP，不替换已有逐帧／全轨迹优化器。 |
| `TemplateSlotKinematicsCache` | 单次 FK 内缓存点、法向和 Jacobian 有价值。已有验证器缓存碰撞 FK，新可选 `SurfaceKinematics` 自持 FK，qpos 改变即清空缓存。 |
| `bind_source_slots_with_normals` | face_id＋barycentric 固定表面身份。可选 `SurfaceTasks` 接受少量明确绑定；不逐帧最近邻、不使用关节点伪造稠密表面。 |
| `transport_tpose_robot_normals` | 运输机器人参考法向，而不是强迫不同形态的机器人直接匹配人体法向。可选目标为 `R_face(t) @ R_face(ref).T @ n_robot(ref)`。 |
| `compute_tpose_surface_normal_offsets` | 官方该函数同时返回静态差向量与机器人参考法向；tpose_offset 的实际路径使用后者做相对运输，不应理解成逐帧加固定向量。 |
| `sample_segment_slots`、`surface_slot_costs_from_segments` | 按部位选少量点并配权；官方每 slot 赋部位权重，点数变化会影响总量。本项目最多每部位 4 点、总共 16 点，每部位总权重固定。 |
| `mujoco_geom_surface.py` | 区分 visual/mesh/primitive 采样策略。已有本项目碰撞采样覆盖 box 角／棱、胶囊轴向和编译后凸 mesh。新表面法向可引用外观 geom；碰撞验收始终只选碰撞 geom。 |
| `prepare_grail_object_mjcf.py` | 官方优先 exact convex components，否则 CoACD，并按内容摘要缓存分解。当前本项目只自动使用已有闭合凸组件；不自动把扫描场景做一个凸包，也不新增未经误差验证的分解。 |
| `lqr_smooth_qpos_sequence` | 官方为关节轨迹构造有限差分正则系统，之后还会 clamp。这里保留现有接触感知 refinement；新增局部阶段只正则化 Δq，更新后重验。 |

函数链接：
[单序列及碰撞查询](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/scripts/retarget_smpl_to_humanoid_surface_vector.py)、
[batch 路径](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/scripts/retarget_smpl_to_humanoid_surface_vector_batch.py)、
[公共 QP／Jacobian／缓存](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/scripts/smpl_surface_retarget_common.py)、
[表面任务](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/scripts/retarget_body_segment_surface.py)、
[HOI/HSI 表面任务](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/scripts/retarget_body_segment_surface_hoi_hsi.py)、
[几何采样](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/scripts/mujoco_geom_surface.py)、
[物体碰撞准备](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/scripts/prepare_grail_object_mjcf.py)。

## 不能只看参数名：实际 defaults 与调用链

- 官方也有限制。固定版本的 HSI/HOI GRAIL defaults 是 `iters=1`、
  `pose_init_iters=10`，并配置 `max_dq=0.15`、`global_step_size=0.4` 和
  `robot_object_max_pairs=128`；求解函数以有限 `frame_iters` 循环，Clarabel
  使用自身默认停止设置。没有发现等同于本项目“整段 2,000,000 查询点即停止”的规则。
  参考这些有限迭代与缓存机制，不移植官方的碰撞对裁剪或源尺度处理。
  后续离线试跑增加 `max_query_points=null`，只关闭查询点总量上限。
  [官方 GRAIL defaults](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/humanoid_retarget_defaults_hsi_hoi_grail.json)。

- 官方 `humanoid_retarget_defaults.json`：warm start 为 sequential；batch defaults 为
  bidirectional。HSI/HOI standard 为 sequential，GRAIL 为 bidirectional。G1 example
  config 主要覆盖机器人 XML、参考肩肘姿态和关节范围，不能把它等同于 solver defaults。
- HSI/HOI 两套 defaults 实际启用 `robot_object_hard_constraint=true`，margin=0，
  slack=true、slack cost=5000、max pairs=128；soft cost=0 不代表碰撞关闭。
  是否最终有碰撞项仍依赖 object source/cache。脚本在缺 object XML/freejoint/geom 时会
  修改 args 关闭项；距离查询异常会 continue。这些行为没有移植。
- 官方单序列和 streaming batch 都按各自遍历方向维护 `ground_contact_anchor_state`。
  `update_ground_contact_anchors` 从当前 robot slot 的 x/y 新建锚点。本项目两个方向
  复用同一 `SceneContactTask` 和一次锁定的 robot marker offset，不重新推接触或锚点。
- 官方 `smpl_scale=robot_height/human_height` 用于 vertices/joints；物体 motion loader
  也将 positions 乘此比例。`retarget_object_size=original` 只控制 retarget mesh scale，
  不撤销物体轨迹缩放。这里没有移植任何源动作／物体缩放。
- 官方 surface normal 默认 tpose_offset，filter mode 为 lqr。本项目已有语义手脚轴、
  标准姿态关节姿态校准和 contact-aware trajectory refinement，不用表面任务覆盖它们。
- 本项目 CLI 默认 G1 profile 未声明语义接触 effectors；InstinctMJ profile 经 defaults
  合并有接触配置，但还需要 loader markers/hints。日志报告实际 task 和 active stages。
  不能把默认字典中的 enabled 当成实际任务已经运行。
- SMPL-X adapter 已提取 palm/sole markers，内部短期持有 posed vertices；通用 MotionData
  未提供经过验证的标准姿态表面绑定契约。本次 P2 不自动启用，不改源 adapter 的输出或源数据。

## P0：实现边界

`scene_geometry.SceneQuery` 保留原始三角形 AABB／winding 查询，新增完整碰撞查询补充。
闭合凸场景组件按原始世界顶点构建独立 MuJoCo 查询模型；复合凸组件分别保留，不填空隙。
机器人采用原模型的碰撞 primitive 或编译后 mesh 凸表示，不复制 actuator、关节或物体 freejoint。
每个机器人 geom 的查询 body 为 mocap，只由原模型 FK 更新；场景 body 既无 mocap 也无 joint。
查询模型 `nq=nv=njnt=0`。具名 robot geom→query geom/body/mocap 映射编译后逐项核对。
原机器人 qi/di 仍来自原模型，标量关节枚举也修正为按类型而不是跳过前七列。

统一约定为正间隙、负穿透。距离方向由 witness 和符号确定；witness 重合时仅接受
碰撞引擎的有效 contact normal/point。异常或无可靠方向增加 uncertain row 和 failure 日志，
不得被 DP 当作有效节点。包围球只剔除有间隙保证的远对，直接遍历候选对做 geomDistance，
不依赖接触列表覆盖完全容纳情况。接口依据
[MuJoCo 距离和 Jacobian API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)
及本机 MuJoCo 3.10.0 验证。

自动凸组件编译后还原顶点并检查双向最近顶点偏差（上限 1 μm，浮点编译误差）。这是
凸表示顶点核对，不是对任意扫描网格的 Hausdorff 安全证明。非凸／开放组件明确报告
triangle fallback；扫描面的已知侧证据和 uncertainty 继续生效。无场景面、无有效机器人
碰撞 geom、编译失败明确报错。未新增任意外部 collision MJCF 导入器／自动 CoACD 分解。

优化和验收都包含完整可靠碰撞体；验收还保留密集原网格表面检查及帧间检查。
已存在的朝上法向、深穿剔除、box/mesh 覆盖修复经回归测试保留。
自碰撞仍使用原模型已有政策；只在已启用时进入求解和验收，不改过滤参数。

## P1：局部方向候选、评分和验收

新增实际顺序：事件聚合 → 可信边界 → 相关关节插值 → 联合验收；插值失败才运行
正／反两个局部序列。根、窗口端点、窗口外及无关关节由 current/reference 逐元素保留。
所有接触任务按原时间索引读取。`TrajectoryValidator.segment` 只切任务数组视图，
保留已选 marker、不调用滞回或 `_lock_contact_offsets`。两个方向求解同一问题。

候选集合最多四条：current、interpolation、forward、backward。方向只影响 warm start
及邻接 Δq 先验，不机械反转全轨迹，不调用含 forward q_prev/q_prev2 假设的限速投影。
节点按共同 FK 重新计算修正、接触、世界脚参考、可选表面法向和实际碰撞代价。
关节限位、根、接触、末端偏移、几何不确定性或穿透预算不通过即节点不可用，不能用低 loss 抵消。

连接使用和验收器相同的 `mj_differentiatePos`／`mj_integratePos` 插值，检查真实 FK、
碰撞、自碰撞（若启用）、接触以及修正速度。二阶 DP 另外保存前两状态，检查 Δq 加速度。
连续性费针对新增修正的速度，不奖励消除源动作本来的快速运动。两端正常但边穿障的
路径被拒绝；无可用路径交回现有有界窗口优化／扩窗，而不是称为安全。

额外修正了前次验收的无限位 hinge 短弧特判：`mjviser_viewer.render_frame` 实际直接
使用 MuJoCo 的标量关节数值差。因此本次验收也保留 scalar seam 的数值差和显式多圈，
四元数仍按 MuJoCo 流形插值。没有修改输出值去掩盖跨分支运动。

选择后在其附近运行最多两次既有接触感知 SQP，平滑 Δq，端点仍固定；每步重做非线性
检查。不合格更新不采用。最终组装轨迹仍整体复验，保存的 qpos SHA-256 沿用已有输出校验。
`invariants` 对固定坐标、窗口外／无关关节、帧数进行实际逐元素核对，失败即报错。

每事件记录 interpolation rejected、两方向逐帧迭代和 slack、DP 节点／边拒绝原因、选中
states、后续优化、扩窗／回退及 resolved/partial/unresolved。方向候选必须满足原 1 mm
穿透容差才入选；现有受限联合优化可保留预算内改善但未解决的残差，并标 partial。
零 Jacobian 行仍保留 slack 约束，不能静默跳过不可动的碰撞点。

## 开关、预算与 P2

默认启用 `retargeting.local_collision_repair.enabled/full_geometry/bidirectional_enabled`。
`full_geometry=false` 只关闭新增碰撞体后端，保留原三角查询；`bidirectional_enabled=false`
退回原插值＋局部优化路径。`enabled=false` 关闭局部修复，已有保存前验收仍保留。
所有生效默认值在运行诊断中展开，机器人 profile 未加入特例。

现有上限继续为窗口 0.4 s、扩窗 0.1 s、24 窗口、8 次窗口 SQP、256 次 geometry 调用，
2,000,000 查询工作量。新增每方向每内部帧最多 3 次 SQP、最多 4 候选、每窗口 512 次
节点／边评估、选择后最多 2 次 polish。实际还受共享查询／geometry 总预算约束。
所有 collision slack 额外设 0.1 m 绝对上限，且受当前深度＋1 mm 回归上限约束；
这不是放宽最终容差。方向候选线性化目标为零间隙（允许 slack），不对真实接触施加正 margin。
原接触、摆动偏移、关节修正预算保持不变。

P2 是可选 Python API：向 `OmniRetargeter.retarget_motion` 或 `repair_local_collisions`
传入 `surface_tasks=SurfaceTasks(...)`，不传即明确 disabled。构造时要求真实同拓扑表面、
shared metric world 坐标、明确对应的 source/robot reference poses；最多每部位 4 个固定
face/barycentric 与 robot geom/local point/local normal 绑定。支持无名字的 URDF geom id，
但绑定只能用于同一个编译模型，不能按机器人名称误复用。

当前仅法向辅助项（每部位总权重默认 0.05，上限 0.1），位置权重为零：人体表面点随源网格
保留并用于来源记录，没有变成高优先级人体世界位置目标。已有手脚世界接触位置任务不变；
已有语义姿态任务的同 body 表面项跳过，避免重复冲突。退化参考面报错；运动中退化面按
frame/slot 关闭并报告；不对法向做低通。单法向不约束 twist。P2 未宣称实现了完整表面位置任务、
自动人体／机器人语义绑定或标准姿态自动验证，也未导入／训练 learned correspondence。

## 验证与限制

可复现命令（本机使用 `.venv/bin/python -m pytest`）：

```sh
.venv/bin/python -m pytest -q tests/test_contact_priority.py
.venv/bin/python -m pytest -q tests/test_geometry_query.py tests/test_scene_geometry.py
.venv/bin/python -m pytest -q tests/test_local_candidates.py tests/test_local_collision_repair.py tests/test_surface_tasks.py
.venv/bin/python -m pytest -q tests
.venv/bin/python -m tests.local_collision_benchmark --ablations --output out/umr_scene_abcd
```

新增测试覆盖：被大 robot box 完全容纳且位于采样间隙的小障碍、深箱内部、凹组件／复合间隙、
具名映射／固定物体、异常查询、不可靠重合 witness、距离／点／法向 nv 有限差分；插值优先、
反向候选胜出、节点安全但边穿障、正反同一接触 payload/marker、硬节点预算、三帧加速度、
不同 fps、缺边界、重复运行、参考法向零误差、已知旋转、点数权重、缓存失效、P2 可关闭。
原 tests 保留支撑／sliding／transition、脚尖、开放网格、重叠窗口、自碰撞、保存后不再变轨迹的检查。

A/B/C/D 是局部阶段的合成对照，不能冒称重新运行真实 SMPL-X 全流程。A 为同一输入机器人
轨迹原样输出并用统一完整验收测量；B 加完整几何和既有插值／局部优化；C 再加局部双向 DP；
D 再提供合成几何真值表面法向，主要验证机制和兼容性。人体／场景真实配准失败 clip 尚未验证。
默认 G1 和 InstinctMJ profile 及真实机器人碰撞模型被审阅／测试，但 G1 benchmark 运动仍是合成。

帧间方法为 adaptive_discrete，不是 CCD 或连续无碰撞规划：运动位移间距默认 1 cm、最多
64 段／相邻帧，原三角表面采样间距 2 cm；达到上限标 unresolved。可漏掉比离散精度更小的
短暂接触；开放扫描和非凸组件仍受采样覆盖与几何有效性限制。局部方向 SQP 失败不证明无解。
法向翻面／噪声需要真实参考表面验证，当前只屏蔽可测退化，不宣称自动解决了所有扫描拓扑问题。

## 实测结果

| 检查 | 结果 |
|---|---|
| `pytest -q tests/test_contact_priority.py` | 62 passed，1.73 s |
| `pytest -q tests/test_geometry_query.py tests/test_scene_geometry.py` | 18 passed，1.81 s |
| `pytest -q tests/test_local_candidates.py tests/test_local_collision_repair.py` | 34 passed，9.80 s |
| `pytest -q tests/test_surface_tasks.py` | 5 passed，2.33 s |
| `pytest -q tests` | **205 passed, 4 skipped**，12.81 s；既有 H1 姿态测试出现一次 CVXPY inaccurate-solution warning，测试和非线性检查通过 |
| 新增三个模块、三个新增测试、benchmark 及三个扩展模块的 Ruff；`git diff --check` | 通过 |

四项 skipped 为既有真实 CLI 测试，其 SMPL-X 路径仍指向本机不存在的 `/localhdd/Datasets/`。
没有修改资产路径来虚构真实测试完成。Ruff 使用 `uv tool run ruff`（项目 .venv 未安装 Ruff），
没有修改项目依赖。测试根限定 `tests/`，不包含 `cankao/` 的第三方测试。

下面深度为 mm，帧数以 >1 μm 计，**不同于** 1 mm 验收容差。因此 C 有三帧约半毫米穿透
仍报告三帧，resolved 只表示通过所声明容差，不表示零穿透。A/B/C/D 都用相同完整检测测量。

| 合成案例 | A 深度／帧数 | B 深度／帧数 | C 深度／帧数 | D 深度／帧数 | C 方法／状态 |
|---|---:|---:|---:|---:|---|
| 直行＋独立支撑点 | 40.000 / 3 | 0 / 0 | 0 / 0 | 0 / 0 | 插值，resolved |
| 重度箱角 | 60.000 / 3 | 1.033 / 2 | 1.033 / 2 | 1.033 / 2 | DP 无可用路径，局部优化，partial |
| 方向依赖箱角 | 5.858 / 3 | 0 / 0 | 0.505 / 3 | 0.505 / 3 | 反向候选＋DP＋局部优化，resolved |
| 20 mm 摆动预算冲突 | 40.000 / 3 | 20.000 / 3 | 20.000 / 3 | 20.000 / 3 | DP 节点不合格，局部优化，partial |
| sliding | 40.000 / 3 | 0 / 0 | 0 / 0 | 0 / 0 | 插值，resolved |
| 只有帧间穿箱 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 无内部帧，fallback；区间仍穿透约 70 mm |
| 实际 G1 碰撞模型／合成小腿事件 | 10.507 / 3 | 1.015 / 2 | 0.544 / 3 | 0.544 / 3 | 双向候选＋DP＋局部优化，resolved |

新增完整查询测得 G1 初始深度 10.507 mm；前次表面采样报告为 8 mm。差异来自检测覆盖，
没有改变箱体尺寸、碰撞体或容忍深度。对于被机器人 box 完全包围、避开旧采样的小障碍，
旧采样深度为 0，完整几何测试检测深度 >250 mm。零 witness 的对称深穿查询会明确记 uncertain，
不会用任意方向求“安全解”。

方向依赖箱角：B 用更大修正消除穿透；C 允许验收容差内的 0.505 mm 残差，最大关节修正
由 9.894 mm 降到 3.790 mm（这个模型是 slide 关节），碰撞点最大水平修正从 13.358 mm
降到 5.356 mm。C 的帧间最大深度 0.859 mm、P95 保存帧深度 0.505 mm；选择 states 全为
backward。新增 Δq 速度／加速度／jerk 为 0.0568 m/s、0.852 m/s²、12.762 m/s³。
这表明候选提供了不同的保真／碰撞折中，不应宣称 C 在每个指标上支配 B。

G1：C 窗口为 `[1,7]`，选中状态 `[forward, forward, forward, forward, forward, backward,
backward]`，随后两次 bounded polish 迭代；B 扩窗到 `[0,8]` 用局部优化。C 最差 body/frame 为
`left_knee_link@5`，P95=0.539 mm，帧间最大=0.544 mm。右支撑脚接触位置误差与滑移始终为
0；左摆动脚对原世界参考的误差／新增位移为 13.343 mm（B 为 13.421 mm），在原 20 mm 预算内。
接触世界目标、时序、源数据均未修改。这些是合成 FK 参考误差，不是实际人体跟踪误差。

G1 C 最大关节修正为 0.017600 rad（约 1.008°），碰撞点最大水平／垂直修正为
10.142／2.725 mm；足端 marker 对应为 12.945／3.235 mm。新增 Δq 最大速度／加速度／jerk：
B 为 0.265 rad/s、7.962 rad/s²、286.645 rad/s³；C 为 0.370、10.317、523.495。
C 的碰撞更小、窗口更短，但这个合成例的高阶修正抖动较大，不能称为全面更平滑。
根轨迹以及窗口外／无关关节在返回数组上逐元素一致，`invariants` 全部通过。

直行支撑案例的支撑位置／滑移为零；sliding 的法向误差为零，原有切向运动保留。
摆动预算冲突保留 20 mm 保存帧及 24.035 mm 帧间残差，最差 `foot@4`，不突破末端预算。
重度箱角最差 `foot@3`，P95=1.033 mm、帧间=1.054 mm，明确 partial。

slack 单独报告（不是实际深度）：G1 C 两方向峰值约 0.0267 mm，后续优化峰值约 0.0666 mm；
B 优化历史峰值约 7.698 mm。摆动预算冲突中方向／联合优化历史 slack 峰值约 20.577／38.632 mm，
最终真实穿透按上表独立测量。各事件 JSON 还给出联合优化 slack_sum_peak、不可动行计数。
修复成功的最终保存帧／区间没有 unknown；只有帧间穿箱案例的区间 0 因对称深穿 witness 查询
不可靠而记 uncertain，保留并回退。查询失败历史不会因为最终姿态安全而从日志删除。

代表性单次局部耗时（不含机器人加载、SMPL-X 推理、原逐帧求解）：直行 B/C 约 0.08 s；
方向箱角 B/C 约 0.11/0.28 s；重度箱角约 1.0/1.3 s；摆动预算冲突约 3.0/3.8 s；
G1 B/C 约 1.4/0.8 s，D 约 0.9 s。G1 B/C 查询工作量为 174,620/106,331；工作量含三角
查询点和完整 geom 距离对，不能直接与前次仅计采样点的数值比较。运行时间是本机单次观测，
不是统计显著的加速结论；失败窗口可能因新增候选而增加开销。

D 与 C 的主要运动指标一致：本轮合成法向真值已与参考运动一致，只证明校准、缓存、接入和
权重不会无意扰乱结果，**未证明 P2 在真实人体动作上的额外质量收益**。自动 SMPL-X 标准姿态
绑定、更多表面位置任务、全 learned correspondence、任意凹扫描体可靠实体分解、真实失败 clip
和视觉／动力学验收仍未实现或未验证；没有将这些写成已完成能力。

完整结果：`out/umr_scene_abcd/summary.json`、各事件 JSON 与 `before/qpos` NPZ。
每次保存前使用的仍为最后非线性验收过的同一条 qpos；无后续 clamp、混合、滤波或 retiming。
另对 21 个 B/C/D 保存后的 NPZ 重新核对：根轨迹、窗口外和无关关节与各自 before 逐元素一致。

## 后续真实 test_data 复查

上述合成实验之后，使用 `test_data/c167_1530_final_v3` 的 SMPL-X 与原场景运行了 G1。
第一次运行因 200 万查询点上限未完成初检；后续显式设置 `max_query_points=null`、
`max_substeps=128`，完成 144 个保存姿态与 3912 个帧间姿态的检查，22,433,736 个查询点，
约 608.8 秒，没有细分上限中断。这里沿用既有适配器补齐缺失 frame_id=71 的行为；
局部阶段的输入输出帧数、根轨迹、接触计划以及人体/场景文件都保持不变。

真实场景为 67 个开放/不可靠分量，仅有 23 个局部方向证据点。当前不确定性的整帧聚合
过于保守，所有帧都被标记为几何侧不确定，四个左臂窗口缺少可信固定端点；最终均明确
保留原轨迹，没有证明真实修复收益。已有局部有向距离的最大穿透估计为 112.82 mm
（左腕、第 77 帧），31 个保存帧有负距离；帧间最大 114.86 mm。这些值带有扫描面
不确定性，不能当作可靠闭合实体真值或完整安全证明。

本次还增加同一运行内几何缓存共享、动作保真拒绝的前置检查、固定边界不确定时的
明确拒绝原因；没有放宽动作或碰撞验收条件。全部测试为 210 passed、4 skipped，
接触/局部/双向相关测试为 101 passed。完整输入摘要、分阶段运行过程、事件表和同口径
接触误差见 `out/testdata_c167_offline_fullcheck/README.md` 及诊断 JSON。
最后导出的轨迹与已完整检查的 baseline 相同，状态仍为 unresolved。
