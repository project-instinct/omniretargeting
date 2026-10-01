**场景碰撞局部修复：实现与验证记录（2026-09-07）**

起点为 `main@e25e4c0`。开始时仅 `.omo/`、`assets/` 未跟踪；两者保留，未 reset、切分支、提交或 push。未改机器人模型、人体数据、场景文件或自碰撞默认政策。

**实际路径与修改**

- `omniretargeting/scene_geometry.py`：统一 `collision_geom_ids`、`geom_surface_offsets`、`ScenePenetrationProbe`、`measure_scene_penetration`；新增共享 `SceneQuery`，缓存 libigl 三角形 AABB。清除逐帧实现的强制朝上法向和无符号距离深穿剔除。box 覆盖角点、棱边和面；capsule/cylinder 覆盖完整轴向及截面；mesh 使用编译后 MuJoCo 顶点的凸包，保留极值点并采样三角面。
- `omniretargeting/trajectory_validation.py`：同一套 FK、接触、限位、新增关节修正及其时间差分检查；几何按帧／geom、自碰撞按 geom pair 比较。保存帧和帧间子姿态都检查几何；帧间也复核接触位置、必要方向和摆动偏移。未知场景侧、细分上限、查询上限单独报告。
- `omniretargeting/local_collision_repair.py`：按运动链、障碍、时间聚合事件；协调重叠窗口，优先线性关节插值；不裁剪角度来伪造通过。边界不可信、接触切换、超限和帧间碰撞都会阻止直接采用插值。必要时只优化窗口内部的链关节，正则化 Δq、Δq 的速度和加速度；用现有 MuJoCo Jacobian、CVXPY/Clarabel、标量关节限位工具和 `_lock_contact_offsets`。有界碰撞 slack 显式保留，接触和机械预算作为约束，真实 FK／非线性检查决定是否接受步长。一次失败后可在相同接触阶段有限扩窗；总窗口数、迭代和查询均有限额。同侧半空间取自可靠边界的几何方向，不更改场景符号。
- `omniretargeting/retargeting.py`、`trajectory_refinement.py`：两阶段共享查询；旧采样入口保留导入兼容。独立调用的 `lsq_linear` 碰撞项仍是软惩罚；主流程通过 `_refinement/guard.py` 使用 CLARABEL 加入最终验收的累计关节／末端／接触预算和可信场景非退化约束，每次更新再经过真实 FK 与密集帧间几何验收。规范化法向时使用副本，避免写回任务数组。
- `omniretargeting/core.py`：接触任务构造完成后固定查询自由侧证据；打印有效配置和实际开启阶段。`_select_penetration_safe_refinement` 改为联合验收入口；正常调用还传入接触任务、实际 fps 和全局安全配置。取消 75%／50%／25% 整段混合与 `max(10cm, before+regression)` 验收。候选不合格时分局部块回退，每次拼接重新检查邻接运动。
- `omniretargeting/main.py`：保存 `<output>.diagnostics.json`；写 NPZ 前核对最终 qpos 的 SHA-256。`retargeting_defaults.py` 增加可关闭开关，删除旧混合／全局最大深度回归默认字段；机器人 profile 文件未改。
- 新增 `tests/test_scene_geometry.py`、`tests/test_local_collision_repair.py`、`tests/local_collision_benchmark.py`；更新 `test_basic.py` 中旧整段混合契约及轻量编排测试桩，更新 `test_robot_config.py` 的默认配置断言。

最终顺序为：逐帧 SQP → 既有 `_project_actuated_temporal_step` → 基线 joint clamp 和固定足端参考 → 受约束全轨迹 refinement → 可选 smoothing（失败时保留已验证 refinement）→ 最终 joint clamp → 联合验收／局部回退 → 局部插值或受限优化 → 最终复验 → 质量报告 → 校验摘要并保存。固定根主流程跳过修改根位置的 foot stabilization／ground lock／XYZ nudge。局部修复之后没有 clamp、混合、限速或滤波。新增阶段不改根平移／旋转、帧数、fps、接触时间或窗口外关节。原始机器人输出作为固定误差参考，源人体和目标数组不写回。

**查询与接触语义**

`clearance > 0` 表示外侧间隙，`< 0` 表示内部。闭合、绕序一致且非退化的组件通过 winding number 判定内外；一致反向绕序也支持。开放、混合绕序或破损组件只使用固定的局部接触／接近侧证据，可信度为局部证据级别；无证据或证据冲突标记 uncertain，不能默认安全，也不生成任意方向的硬约束。深处内部及不确定的几何不会仅因离最近表面远而剔除。

运动物体按有界物体处理（`SceneQuery(bounded_object=True)`，运动物体查询与物体距离网格共用），另有精确的外侧判定：
记录的物体表面加上封堵其边界环的任意补面，都位于顶点凸包之内（广义绕数分割的定义域，Jacobson 等 2013），
所以凸包外的点在任何距离上都在物体之外，不论附近开放面片的自由侧证据如何投票。证据只在面片附近外推局部侧，
在证据半径边缘会把凸包外的点判成深处内部：omomo_254 的手离白椅子最近表面 13 cm、已在椅子包围盒外，
却被报成 13.4 cm 穿模；omomo_182 同类假阳性 5.4 cm（实际反向样本重叠 2.1 cm）。地形和封闭房间不适用
（地形在最低顶点以下仍是实体，房间的实体在壳外），平面网格没有凸包内部，都不使用这条判定。
凸包外的采样点即使超出证据半径也算已知，修复时手退出物体不会再因“丢失源侧观测”被拒绝；
超出激活距离的凸包外采样点不再做最近三角形查询（到物体的距离不小于其凸包平面余量，不会产生约束行）。

凸包内还有一条否决：证据判为内部、但该组件广义绕数与 0 相差不到 0.01（与闭合组件相同的判定容差）的点，
没有任何闭包包住它，改判为外部。开放扫描的“单面”薄片旁边 |w| 约为 1/2，不会被否决，仍由证据决定；
材料内部 |w| 接近 1，也不受影响。omomo_254 剩余的 10 cm 即属此类：手在白椅子凸包内 0.6 mm、离最近表面
10 cm，证据判为内部，而椅子主体的绕数为 −0.002；同一把椅子表面内侧 0.3–0.9 cm 处绕数为 1.00，3 cm 处为 0
（椅壳很薄）。离表面两格以内由证据决定；更深的点查该组件一次性算好的绕数网格（最长边 64 格，
3.2 万面的椅子约 0.05 s），所在格的 8 个角点都与 0 相差不到 0.01 才否决。逐次调用快速绕数每次约 16 ms，
一次完整修复测量（168 帧）要多 2.7 s。

接触位置使用现有任务的世界目标及锁定 marker。sticking 保护位置及配置要求的方向；sliding／transition 保护表面法向，不锁切向或一律锁整个脚 SE(3)。已达到的锚点采用紧预算；已有接触误差允许向原固定目标改善。自由脚保留世界参考，并同时约束新增机器人末端位移和误差退化。没有分类的纯 world-tracking 脚不会被当成可信自由摆动，局部位移预算收紧到 1 mm；不新增接触时间或移动锚点。旧 profile 缺少源末端任务时，复用已有足端 body／sole marker 解析器，构造仅用于验收的原机器人足端世界轨迹，并使用 1 mm 紧预算；这些参考不导出成人体锚点，也不伪造接触计划。无法解析足端角色的通用调用会明确记录约束不可用，不能据此宣称接触已验证。

默认 CLI profile `robot_models/unitree_g1/unitree_g1.json` 和 `unitree_g1_instinctmj.json` 都会经 `load_robot_config` 合并共用接触默认值；实际接触任务还取决于加载器提供的 marker／接触证据。每次运行日志、诊断 JSON 都记录生效值及 active stages；晚于初始重定向建立的机器人足端保护参考单独记录在 `validation_only_robot_foot_references`，并记录完整 `validation_end_effector_priority`。局部开关为 `retargeting.local_collision_repair.enabled: false`；关闭它仍保留后处理验收。

当前局部默认：窗口最长 0.4 s，扩展预算 0.1 s，关节新增修正 0.2 rad（slide 为 m），已分类自由末端偏移 0.02 m；验证采样间距 0.02 m，帧间几何位移间距 0.01 m，最多 64 段／相邻帧、8 次迭代／事件、24 个修复窗口、256 次几何检查、2,000,000 查询点。接触基准容差复用 5 mm，新增接触／碰撞回归预算 1 mm；Δq 速度／加速度按现有 30 Hz 时间步预算换成物理单位，再按真实 fps 差分。全局 self-collision=true 和更紧的 joint safety fraction 不能被局部配置放宽。

这些值用于限制本轮合成案例的改动和运行量，不是通用安全阈值。20 mm 的摆动预算测试刻意保留约 20 mm 深穿，证明不会为了清空碰撞突破末端预算；0.4 s 与 0.1 s 的语义在 30／60 fps 测试中检查。无碰撞输入是 no-op，插值通过时优化次数为零。

离线检查可以显式配置 `retargeting.local_collision_repair.max_query_points: null`，
允许完整执行点查询，不再因固定查询点总数提前结束。默认仍为 2,000,000；
此开关不改变碰撞容差、几何覆盖、动作修正范围、接触目标、迭代次数、窗口数或
帧间细分上限，也不将开放扫描面的不确定侧变成自由空间。有限上限耗尽时，诊断
现在同时记录实际查询点数、几何调用数；局部初检失败标记 `failed_stage=initial_validation`。

后处理验收和局部修复在同一次运行内共享精确 qpos 的几何缓存，按模型、场景查询对象、
碰撞 geom 集合、采样密度和自碰撞策略隔离。仅在模型与场景不变时复用；每次仍重算
接触残差、时间模式和偏离预算。后处理候选先检查动作保真，已超出预算的候选无需
再运行昂贵的几何查询；这只改变拒绝检查的顺序，不改变采用条件。

局部 SQP 如遇固定窗口端点的几何侧不确定，会明确返回
`uncertain_fixed_boundary`。当前验收包含端点，且局部变量不能移动端点，因此
继续修改内部关节无法让这类窗口通过；仍允许原有规则内的有限扩窗。这个判断
不宣称机器人动作数学上无解，也不将未知几何忽略成无碰撞。

**可复现实验**

```sh
.venv/bin/python -m pytest -q tests/test_contact_priority.py
.venv/bin/python -m pytest -q tests
.venv/bin/python -m tests.local_collision_benchmark --output out/local_collision_ab
```

原实现上的第一批几何回归为 5 failed，复现了深箱内部漏检、底面／斜面方向错误、box 角点漏检。修改后独立接触测试 62 passed；本项目完整测试为 **181 passed, 4 skipped（8.18 s）**。四项跳过及第三方测试收集失败的原因在末尾说明。新模块和新增测试的 Ruff 检查、`git diff --check` 通过。

下列全部为合成运动与合成箱体，帧索引从 0 开始。G1 一行使用实际 profile 解析得到的完整 G1 碰撞模型及 0.3 m 箱体；运动是人为构造的 9 帧小腿事件。左右踝原点是明确指定的测试 marker，右脚使用合成 sticking 计划，左脚世界参考来自输入机器人的 FK；这不代表真实 SMPL-X 接触误差。其他案例使用简单平移关节链，其修正单位为 m。

P95 取每个保存帧的最大 geom 采样深度，包含零深度帧；“穿透帧”计数阈值为 1e-6 m，和 1 mm 接受预算分别记录。因此有很小的残差时，帧数仍可能不为零。

| 案例 | 最大深度 mm（前→后） | P95 mm（前→后） | 穿透帧 | 窗口 | 方法／迭代 | 秒／查询点 |
|---|---:|---:|---:|---|---|---:|
| 线性插值＋独立支撑点 | 40.000 → 0.000 | 33.657 → 0.000 | 3 → 0 | [1, 7] | interpolation / 0 | 0.077 / 2,162 |
| 插值穿箱角 | 60.000 → 1.046 | 60.000 → 1.042 | 3 → 2 | [0, 4] | local_optimization / 8 | 0.836 / 7,094 |
| 20 mm 摆动预算冲突 | 40.000 → 20.000 | 33.657 → 16.527 | 3 → 3 | [0, 8] | local_optimization / 8 | 2.678 / 18,612 |
| 滑动接触 | 40.000 → 0.000 | 33.657 → 0.000 | 3 → 0 | [1, 7] | interpolation / 0 | 0.076 / 2,162 |
| 仅帧间穿透 | 0.000 → 0.000 | 0.000 → 0.000 | 0 → 0 | [0, 1] | fallback / 0 | 0.028 / 1,002 |
| 实际 G1 模型／合成小腿事件 | 8.000 → 1.033 | 8.000 → 1.033 | 3 → 3 | [0, 8] | local_optimization / 8 | 1.200 / 137,480 |

最差部位／帧：线性、滑动、摆动预算案例原为 `foot@4`；线性与滑动修复后无穿透。箱角优化最差为 `foot@1`；摆动预算案例修复后仍为 `foot@4`。G1 前后最差均为 `left_knee_link@3`，3 帧仍有微小残差。仅帧间案例保存帧深度均为零，区间 0→1 实测最大 60 mm；由于没有可修改的内部帧，明确回退，未删除帧或自动降速。

G1 的独立质量量：右脚接触位置误差 0→0 mm，支撑滑移 0→0 m/s；左脚对测试世界参考的误差／新增位移 0→10.239 mm，低于 20 mm 预算。最大关节修正 0.013803 rad（约 0.791°）；新增修正最大速度 0.207049 rad/s、加速度 6.211461 rad/s²、jerk 223.634516 rad/s³。原始合成关节加速度最大 135.000 rad/s²，单独诊断，没有对原轨迹全局低通。

G1 碰撞采样点的最大水平／垂直修正为 7.743／2.040 mm；左踝 marker 的水平／垂直变化为 9.938／2.464 mm。根轨迹保持逐元素相同。该事件先拒绝超出摆动／时间连续性预算的插值，尝试局部优化及有限扩窗，8 次迭代后保留约 1.033 mm 残差，状态仍为 unresolved。

线性＋独立支撑点案例中，支撑点位置误差和滑移均为零；滑动案例的法向误差为零，原切向移动保持，未被锁死。受限摆动案例允许约 20 mm 新增位移，保留约 20 mm 保存帧残差及约 22.69 mm 区间残差。每个事件的尝试窗口、插值拒绝原因、迭代状态、松弛量及回退原因均保存在独立 JSON，不能把最后一次线性子问题失败理解为整个运动数学上无解。

完整诊断在 `out/local_collision_ab/summary.json`、各案例 JSON 和 `effective_configs.json`；NPZ 保存 before／qpos，OBJ 保存对应测试场景。运行时间包含局部查询、候选和验证，不包含模型加载、SMPL-X 推理或完整逐帧重定向；主要开销是密集 FK／场景查询和约束子问题。没有生成 MP4。

**验收边界与未验证项**

优化使用较稀的机器人表面采样，最终验证使用更密的几何采样；最近三角点由 libigl AABB 查询。帧间依据各 geom 祖先链的保守位移上界自适应细分，根四元数使用 MuJoCo 插值；有限位关节不 unwrap，不 clip，无限位关节跨主值角分支时用短弧，但显式多圈角值保留。方法报告为 `adaptive_discrete`，没有使用连续碰撞检测，有限采样不构成连续安全证明。

该检测仍是近似：小于采样间距的薄特征、完全包在机器人内部且避开采样的障碍、未经验证的自交扫描网格等仍可能漏检。闭合组件的可靠性以几何非自交为前提；开放扫描无局部自由侧证据时保守 unresolved。若达到细分／查询预算，保留最后已检查的输出或原轨迹，明确标记预算不足，不将未检查数据称为安全。未做动力学／摩擦／力矩优化，也未做真实长动作或 GUI 视觉验收。

本机存在 AMASS／wall-flip 原始动作及 `assets/body_models` 下的 SMPL-X 模型，但未找到已配对的“短暂绕箱小腿穿透”真实机器人片段，因此未声称真实动作质量改善。既有 4 个完整真实 CLI 测试仍硬编码 `SMPLX_MODEL_DIR=/localhdd/Datasets/`，该路径不存在，故跳过；本轮未改该测试资产布局。项目根目录直接 `pytest -q` 还会误收集 `cankao/human-humanoid-tools` 的第三方测试，因缺少 `hhtools` 出现 9 个 collection errors；本项目 `tests/` 的验证结果单独列出，未把第三方错误藏成通过。

**开放运动物体的局部可活动侧（2026-09-26）**

`moving_evidence.py` 从原始人体接触表提取 `valid`、`active`、置信度至少 0.5、
且位于已观测可活动侧的点。证据固定在原物体材料坐标中，随原物体变换，
不更改源皮肤、物体网格或场景注册，也不授予支撑／控制角色。

`MovingSceneQuery` 对机器人采样点的最近原始开放组件查询这一局部证据。
这补充了反向采样的盲区：机器人可以整体处于开放面片背后，而没有物体采样点
落在机器人内部。只在证据一致、有限三角形／折边可靠时提供完整退出方向。
证据半径使用现有 0.15 m 局部查询尺度，并按物体最大原始尺度换到材料坐标。
机器人采样点本身也必须位于证据的三维邻域内，避免把灯罩背后远处的头部
误判为穿透；超过该局部证据范围仍是未知，不能宣称安全。
未知面片、冲突侧和面片外的投影不会变成无限半空间约束。

闭合组件、局部开放组件和反向交叉的每组深度独立保护，不能用一项较大的
残差掩盖另一项退化。原来已有负间距的材料采样点若在候选里失去局部侧证据，
候选会被拒绝，不能把未知当成清除了穿透。开放组件总数仍完整报告；
局部证据并不把扫描网格认证为闭合实体，也不提供连续无碰撞证明。

`scripts/review_skin_scene.py` 使用同场景、同帧的完整 SMPL-X 皮肤和所有机器人
网格顶点，分别报告地面、闭合物体、已观测开放面片和开放扫描 winding 候选。
最后一项仅用于发现待检查的交叉，不能用作已认证的穿透深度。

局部修正 QP 现在同时使用固定的获取阶段法向预算、已达到的向量／法向／切向
接触精度，以及各关节和根部的连续性预算。通过稀疏选择矩阵固定根部自由度，
不重新定义参考或放宽预算。原非线性检查与回退仍逐候选执行；把这些限制放进
子问题的目的，是在剩余可行方向内继续求解，避免只在事后重复拒绝越界方向。

线性化保留入口已经通过非线性检查的数值舍入量，并始终截断到同一个固定数值容差上限。
这保证已接受状态的零增量仍可行；不会将上一轮候选用作新参考，也不会累计扩大预算。

共面的多个三角形也可能共享开放边界顶点，不能仅凭相邻面数将其当作内部接缝。
开放共面特征要求查询点对有限面片作正交投影；边界之外的切向偏移仍为未知。
该检查独立于面片大小、朝向和绕序，不按穿透深度截断，保留有证据的深层面内投影。

重复候选复用逐物体、逐帧、逐机器人几何体的完全相同材料变换查询。缓存键包含原物体
平移、转动、尺度及机器人几何体的精确变换字节；不量化坐标，雅可比仍按当前状态重算。
缓存有固定容量，新的变换重新查询；深度、退出方向、未知区域与接触保护规则保持一致。

### Isolated lower-limb IK

Foot repair derives the two leg branches from profile-resolved foot endpoints
and their nearest shared ancestor; endpoint names may be explicit or inferred. It permits only
scalar joints strictly below that ancestor; pelvis, waist, floating base and
upper limbs stay fixed during each leg proposal, including on torso-rooted
models. Contact-compensation closure and self-collision partners cannot expand
this scope. This variable restriction does not change evidence-based contact
ranking, source targets, scene registration or the immutable human motion.

Within this scope the collision objective uses a 1 mm residual scale. Local
scalar joint corrections remain bounded by 0.05 rad (about 2.9 degrees) against
the fixed repair entry. Per-body world rotations are reported, not hard posture
limits. The fixed pre-acquisition reference layout, contact and per-joint/time
derivative guards remain in force; posture angles are reported, not bounded. Limits apply to the
cumulative correction, not independently to each iteration. An unreachable
clearance retains its residual; it cannot recruit the upper body or lift the
root. Reports identify each `limb_only` proposal and its fixed anchor/joint set.
This repairs the robot; it does not create or substitute a corrected SMPL-X.


### Material point motion and bounded feasibility restoration

A sticking sample does not establish a locked six-dimensional rigid-body pose.
When a constant material-body placement would leave any frame's original source
orientation envelope, or active samples include sliding, `contact_anchors.py`
uses per-frame rigid placements bounded by that same-frame source orientation.
Each continuous sticking sample run retains one actually observed surface point
in object material coordinates. Sliding observations keep their time variation.
Robot metric offsets, original object transforms/scales, and contact roles remain
unchanged. Morphology-incompatible sample positions retain an explicit fit
residual; they do not stretch the robot or certify exact contact.

Local repair linearizes both fixed source-shape envelopes alongside acquired
contacts, root/joint continuity and original collision witnesses. If a full step
fails a curved nonlinear constraint, a bounded restoration problem searches
within the same local joint scope and original budgets. Its intermediate
infeasible states cannot enlarge temporal slack. Every restored candidate still
requires the original nonlinear contact/shape checks and geometry query; failure
falls back to ordinary backtracking. Every evidence-ranked collision group gets
its own iteration budget, with unresolved and iteration-limited groups reported.

The robot profile boundary attaches `local_repair_group` to resolved foot
endpoints. Generic solver scope construction consumes that field and actual
kinematic ancestry, so inferred names such as `L_Ankle` and explicit names such
as `left_foot` obtain the same upper-body lock. This scope is unrelated to contact
priority. Adapter-declared sample material labels also associate source markers
with renamed endpoints: calibrated sole centers/directions cannot be silently
replaced by an entire region's average merely because the profile renamed it.
Ambiguous material labels do not supply a guessed semantic frame.


Reverse intersection witnesses now have the same exact-transform reuse as
forward material queries. Keys include the original object and robot geometry
poses, object scale, query band and row limit. Every cache miss still checks all
original vertices/triangle centroids in the existing broad phase; stored rows
contain no Jacobians. This does not approximate motion, round transforms, reduce
sampling, or change unknown-region acceptance.


The inferred foot endpoints also enable existing source orientation tracking
when a reliable robot sole plane can be derived and the profile has not
explicitly disabled it. Local up comes from that plane; the in-plane long axis
is signed by the robot's reference heading. Ambiguous geometric calibration is
reported as unavailable. Original adapter-provided sole directions participate
in source-pose fitting **before** freezing the reference; a later repair never
rebases its envelope. Surface-managed contact acquisition keeps its existing
orientation/contact policy. Position diagnostics keep their position target
when direction rows are enabled.


Candidate geometry checks may stop at the first proven regression against the
protected per-frame collision depths. Such a prefix is explicitly incomplete
and can only reject a candidate. Every accepted candidate, initial baseline and
final report still covers the full timeline and original geometry. This avoids
rechecking later frames of an already invalid proposal without weakening any
acceptance condition.


### Surface goals are independent of rigid placement

Rigid-fit output is a robot pose estimate, not a scene-contact target. Contact
rows now retain actually observed scene material points even when the robot
patch has a different width or cannot follow the source orientation. A sticking
run selects one observed material point; sliding retains its per-frame surface
points. Original object rotation, translation and authored scale transport
these targets, while the robot's geometry remains metric and rigid. The rigid
placement and event prepass remain separate diagnostics; their residual cannot
move targets off the surface. This applies to every region, including curved
surfaces and walls, without a ground axis or anatomical priority rule.


The high-level source/contact preparation uses the collision query's existing
solid-scene default (`scene_enclosed_free_space=False`). Previously a missing
setting reached contact observation as `None`, which could flip a closed
surface normal toward slightly penetrating source skin and reject its facing
evidence. Explicit enclosed-room semantics remain supported. Source skin and
its negative contact vectors stay unchanged; robot targets remain on the scene.


### Four-clip regression review (2026-09-26)

The OMOMO 78 / G1, OMOMO 318 / G1, OMOMO 78 / T1, and AMASS 0 / T1
runs are recorded in `out/contact_general_fix_20260926/comparison.json` and
`report.md`, with full original SMPL-X skin comparisons at
`http://localhost:8183/contact-general/`. These are **unresolved outputs**, not
accepted motion examples: all four fast-backend reports retain
`unresolved_source_shape_or_continuity`, and list `SelfContactTasks` and
`SurfacePoseTasks` as skipped. Passing the local guards does not certify the
initial source-shaped reference or whole-motion acceptance.

| Case | Maximum foot depth below floor top | Worst foot / frame |
|---|---:|---|
| OMOMO 78 / G1 | 3.035 cm | right / 83 |
| OMOMO 318 / G1 | 0.159 cm | left / 33 |
| OMOMO 78 / T1 | 4.463 cm | right / 83 |
| AMASS 0 / T1 | 7.940 cm | right / 454 |

Frames are zero based. The floor-top measurement uses the original robot
collision geometry; the reference uses complete original human foot skin.
A thin closed floor's shortest full exit distance is a different quantity and
must not conceal deeper intersections below its top. The report also retains
floating contact frames, including OMOMO 318 / G1 frame 21 where original foot
skin is within 1 cm of the floor but the robot is more than 3 cm above it.
OMOMO 78 / G1 still has a 12.720 cm moving-object intersection at the left hand
in frame 65. The scalar maxima are insufficient to accept these trajectories.

All 88 recorded leg proposals (55 accepted) preserve every non-leg qpos and
non-leg body transform exactly; local floating roots stay unchanged. Original
source positions and object poses/scales remain unchanged. The final AMASS
export separately clamps two left hip-pitch values by at most 6.86e-9 rad to the exact
joint limit; this numerical export change is not attributed to an accepted IK
proposal. Independent endpoint, contact, foot, and derivative measurements use
the actual exported trajectory.

The conservative reach diagnostic relaxes other contacts and temporal limits
while retaining the fixed root and original 0.05 rad local rotation bounds.
For AMASS frame 454, an originally penetrating right-foot material point has
an upper bound of -5.108 cm on achievable height. Thus complete clearance of
that point is impossible within this particular fixed envelope. This does not
prove that the solver minimized the remaining error, or that all intersections
are unavoidable; neither reference rebasing nor automatic budget widening is
permitted. The full test suite passed 1077 tests with 8 skips, including passing
contact-priority and module-size checks. Six profile kinematic-scope checks are
reported separately from the four real motion runs.

### Mesh-repair layout restoration (2026-09-30)

Restoration now keeps the signed margin of the immutable reference layout.
A proposal already across a layout halfspace must return to the legal side;
clamping its negative margin to zero merely prevented further worsening and
could repeat ineffective recoveries. `restore=True` is passed only by
`restore_repair` through `RepairLimits` and `ReferenceShape.repair_bounds`.
Acquisition/reweighting direction and restoration behavior remain unchanged;
the shared acquisition change failed motion acceptance and stays isolated.

This is unconditional repair behavior. The two byte-verified parallel options
(`parallel_shortening`, `parallel_repair_restore`) are enabled by default.
No physical tolerance, reference, locality/continuity constraint or iteration
limit changed. `foot_v18lr` passed all original gates on 78/5/80/485/182/254;
its acquired trajectories are byte-identical to `foot_v16base`, and its final
trajectories are byte-identical to the repair-only replay. The main-workspace
full pytest suite and a complete default CLI run of 78 verify the promotion;
the latter's final qpos is byte-identical to `foot_v18lr/78`. Full timings,
collision-body depths/frames, relation reports, test logs and SHA256 manifests
are linked in `docs/runtime_20260929.md` (final default acceptance subsection).

The unified acquisition fix is not accepted: clip 5 trades away free-hand
fidelity and clip 485 increases the left-hand object penetration. Stopping
by elapsed time/iteration work does not repair either defect. The isolated
strict seed-free-target guard also failed self-collision and performance.

TRUMANS chair retains a pelvis residual of 48.267 mm at output frame 193
(default candidate run `foot_v18default`, wall 689.971 s, repair 260.004 s).
It is unresolved, not a collision-free certificate. The pelvis group rejects
its restored full step for another static witness regression and smaller
steps for frozen contact/normal constraints. The torso-root model's waist
joints can move pelvis; it is incorrect to claim a zero scalar-joint Jacobian.
The original per-witness geometry acceptance rule remains in force pending
a separate six-clip plus chair experiment.

#### TRUMANS 精确恢复统计与全步完整几何检查

修复回放最终 qpos 与 foot_v18default 逐位一致；额外完整几何查询全部在原求解结束后运行，未改变接受轨迹。
底层实际恢复 14 次，全部 restored；其中 2 次一轮、12 次两轮，共 26 轮，没有反复失败重启。
以下恢复耗时来自诊断回放，不与原完整 CLI 各组耗时混加。

| 分组 | 恢复次数 | 恢复内轮数 | 恢复耗时 s |
| --- | ---: | ---: | ---: |
| left_ankle_roll_link | 2 | 4 | 23.555 |
| pelvis | 1 | 2 | 16.079 |
| left_hip_pitch_link | 1 | 2 | 11.718 |
| right_hip_pitch_link | 1 | 2 | 11.107 |
| left_hip_roll_link | 1 | 2 | 12.021 |
| right_wrist_yaw_link | 4 | 6 | 6.890 |
| right_hip_roll_link | 1 | 2 | 13.501 |
| right_wrist_pitch_link | 2 | 4 | 4.889 |
| left_wrist_yaw_link | 1 | 2 | 17.460 |

恢复累计 117.223 s。骨盆等长组主要修正 ContactGuard，第二轮常消除接近 1e-7 m 的残余；未扩大数值或物理容差。

骨盆唯一通过非几何守卫的全步：全局最大深度 48.267074492→48.266891852 mm，下降 0.000182640 mm；总平方深度 1.183955458→1.183227120 m²，下降 0.061517%。

完整查询无失败，但 47 个受保护几何见证退化超限：左踝 geom 21 共 38 帧、右踝 geom 33 共 9 帧。
最先短路报告的是左踝 15 帧 +0.408646 mm；最坏是左踝 220 帧，历史最佳 0.150074→候选 1.040520 mm，加深 0.890446 mm。
这些增量相对逐见证历史最佳深度，不把保护参考改为当前候选。完整列表见 trumans_repair_detail20.json。

结论：当前通用判据已经允许既有穿透在 0.051 mm 余量内变深，同时要求全局平方深度下降。该全步的最大深度收益几乎为零，且多个位置远超原余量，证据不支持为它放宽判据。
本轮未修改几何接受规则，因此没有宣称新规则已通过六段或椅子验收；保持已验收的六段默认和完整椅子结果。
48.267074 mm 的 pelvis@193 残差仍是未解决缺陷。下一种通用方向应把实际发现的双踝几何违例重新加入方向求解，再保持原守卫逐项验收；该方向尚未实现或验证，不能宣称能消除残差或加速。

### 支撑/未解释 hold 的世界连续性（第十轮，2026-09-30）

`_fast/repair_support.py` 在 mesh repair 入口冻结支撑与未解释 hold 末端的
世界速度、加速度向量窗口预算。角色来自源证据，不按身体名称；材质跟踪点和
映射刚体原点分别保护，模板只要触及支撑/hold 帧便包含交接。预算使用同一
`windowed_budget`（7 模板邻域最大值 + 10% 自身 RMS），不跨末端转移；
方向/恢复的锥约束和候选精确检查共用固定预算，数值余量仍为原 2e-7 m。

守卫默认开启，只有消融对照用
`OMNIRETARGETING_FAST_DISABLE=repair_support_continuity` 关闭。叶端朝向
新主源码稳定 20 分钟后，冻结同一快照做六段 A（关闭）/B（默认）完整 CLI
交替验证；六段全过修订后的原质量门。A/B wall 合计 1036.853/1065.023 s，
这是修正连续性合同的实际成本，不冒称提速。获取等 optimize 阶段逐位一致；
主工作区 78 最终 qpos 与 B 逐位一致。54 项定向、1318 项全量测试通过，8 跳过。
全部 SHA256、逐秒负载、每段世界/真实自由目标误差、碰撞体深度/部位/帧、
关系角与最终 TRUMANS 椅子/手腕报告见 `docs/runtime_20260929.md` 第十轮。

未修改源轨迹、接触目标、layout/时间窗口、物理容差、修复进展门或迭代数。
统一获取 lr、fs、root-roundoff 仍隔离；新叶端基线不用于追认旧候选。
相对质量门通过也不证明无穿透，超过 1 cm 的残余及诊断局限必须保留报告。
