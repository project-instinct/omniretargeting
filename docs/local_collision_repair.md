**场景碰撞局部修复：实现与验证记录（2026-09-07）**

起点为 `main@e25e4c0`。开始时仅 `.omo/`、`assets/` 未跟踪；两者保留，未 reset、切分支、提交或 push。未改机器人模型、人体数据、场景文件或自碰撞默认政策。

**实际路径与修改**

- `omniretargeting/scene_geometry.py`：统一 `collision_geom_ids`、`geom_surface_offsets`、`ScenePenetrationProbe`、`measure_scene_penetration`；新增共享 `SceneQuery`，缓存 libigl 三角形 AABB。清除逐帧实现的强制朝上法向和无符号距离深穿剔除。box 覆盖角点、棱边和面；capsule/cylinder 覆盖完整轴向及截面；mesh 使用编译后 MuJoCo 顶点的凸包，保留极值点并采样三角面。
- `omniretargeting/trajectory_validation.py`：同一套 FK、接触、限位、新增关节修正及其时间差分检查；几何按帧／geom、自碰撞按 geom pair 比较。保存帧和帧间子姿态都检查几何；帧间也复核接触位置、必要方向和摆动偏移。未知场景侧、细分上限、查询上限单独报告。
- `omniretargeting/local_collision_repair.py`：按运动链、障碍、时间聚合事件；协调重叠窗口，优先线性关节插值；不裁剪角度来伪造通过。边界不可信、接触切换、超限和帧间碰撞都会阻止直接采用插值。必要时只优化窗口内部的链关节，正则化 Δq、Δq 的速度和加速度；用现有 MuJoCo Jacobian、CVXPY/Clarabel、标量关节限位工具和 `_lock_contact_offsets`。有界碰撞 slack 显式保留，接触和机械预算作为约束，真实 FK／非线性检查决定是否接受步长。一次失败后可在相同接触阶段有限扩窗；总窗口数、迭代和查询均有限额。同侧半空间取自可靠边界的几何方向，不更改场景符号。
- `omniretargeting/retargeting.py`、`trajectory_refinement.py`：两阶段共享查询；旧采样入口保留导入兼容。全轨迹 `lsq_linear` 碰撞项仍明确是软惩罚，新增最终验收承担检查。规范化法向时使用副本，避免写回任务数组。
- `omniretargeting/core.py`：接触任务构造完成后固定查询自由侧证据；打印有效配置和实际开启阶段。`_select_penetration_safe_refinement` 改为联合验收入口；正常调用还传入接触任务、实际 fps 和全局安全配置。取消 75%／50%／25% 整段混合与 `max(10cm, before+regression)` 验收。候选不合格时分局部块回退，每次拼接重新检查邻接运动。
- `omniretargeting/main.py`：保存 `<output>.diagnostics.json`；写 NPZ 前核对最终 qpos 的 SHA-256。`retargeting_defaults.py` 增加可关闭开关，删除旧混合／全局最大深度回归默认字段；机器人 profile 文件未改。
- 新增 `tests/test_scene_geometry.py`、`tests/test_local_collision_repair.py`、`tests/local_collision_benchmark.py`；更新 `test_basic.py` 中旧整段混合契约及轻量编排测试桩，更新 `test_robot_config.py` 的默认配置断言。

最终顺序为：逐帧 SQP → 既有 `_project_actuated_temporal_step` → 全轨迹 refinement → 可选 smoothing → foot stabilization → 最终 joint clamp → 联合验收／局部回退 → 局部插值或受限优化 → 最终复验 → 质量报告 → 校验摘要并保存。局部修复之后没有 clamp、混合、限速或滤波。新增阶段不改根平移／旋转、帧数、fps、接触时间或窗口外关节。原始机器人输出作为固定误差参考，源人体和目标数组不写回。

**查询与接触语义**

`clearance > 0` 表示外侧间隙，`< 0` 表示内部。闭合、绕序一致且非退化的组件通过 winding number 判定内外；一致反向绕序也支持。开放、混合绕序或破损组件只使用固定的局部接触／接近侧证据，可信度为局部证据级别；无证据或证据冲突标记 uncertain，不能默认安全，也不生成任意方向的硬约束。深处内部及不确定的几何不会仅因离最近表面远而剔除。

接触位置使用现有任务的世界目标及锁定 marker。sticking 保护位置及配置要求的方向；sliding／transition 保护表面法向，不锁切向或一律锁整个脚 SE(3)。已达到的锚点采用紧预算；已有接触误差允许向原固定目标改善。自由脚保留世界参考，并同时约束新增机器人末端位移和误差退化。没有分类的纯 world-tracking 脚不会被当成可信自由摆动，局部位移预算收紧到 1 mm；不新增接触时间或移动锚点。旧 profile 缺少源末端任务时，复用已有足端 body／sole marker 解析器，构造仅用于验收的原机器人足端世界轨迹，并使用 1 mm 紧预算；这些参考不导出成人体锚点，也不伪造接触计划。无法解析足端角色的通用调用会明确记录约束不可用，不能据此宣称接触已验证。

实际默认 CLI profile `robot_models/unitree_g1/unitree_g1.json` 没有声明接触效应器，因此不能仅看接触默认字典就声称语义接触或全轨迹 refinement 开启。`unitree_g1_instinctmj.json` 经 `load_robot_config` 合并后具有接触默认值；是否实际启用还取决于加载器是否提供 marker／接触 hint。每次运行日志、诊断 JSON 都记录生效值及 active stages；晚于初始重定向建立的机器人足端保护参考单独记录在 `validation_only_robot_foot_references`，并记录完整 `validation_end_effector_priority`。局部开关为 `retargeting.local_collision_repair.enabled: false`；关闭它仍保留后处理验收。

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
