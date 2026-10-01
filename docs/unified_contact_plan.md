# 统一接触管线计划

状态：实施中（2026-09-24）。当前基线、代码快照与验证记录保存在 `out/unified_contact_20260924/`。
保留现有未提交改动，在 `unified-contact-20260924` 分支实施；不重置、覆盖或先提交其他人的修改。

执行约束：时长过滤只能作用于弱几何候选，不能删除有力学证据的短时蹬离、触地或源显式接触。
所有新锚点和接触优先级受固定的逐关节连续性及累计姿态界约束；无法兼容的约束报告残差。
人体惯性采用可复现的源几何近似，摩擦与导数敏感性显式记录，不将近似可行性当成测量或执行保证。


### 当前验收状态（2026-09-24 更新）

用户已指出试算图片中支撑脚离地、动作走样；该试算已停止，不能视为通过验收或发布结果。
目前发现初始化同时追逐碰撞与全身接触时会选择错误姿态分支，随后累计预算又把错误分支锁住。
正在增加独立的源动作姿态参考求解，并让后续所有接触、几何修正共用这一个固定参考。
验收必须同时对照未修改人体方向与固定机器人参考，不能只凭优化分数、单元测试或相对上一阶段的变化量认定动作相似。
全身接触统计已补入序列日志，避免旧末端约束退出后把“未统计”误报成“零残差”。
坐姿与第二段 TRUMANS 的力学敏感性检查已完成；统一轨迹的视觉、接触、逐关节连续性及穿透验收尚未完成。

后续排查发现了三处推进问题：接触角色在凸求解与非线性验收中的逐帧语义不一致；
碰撞净空被错误加入逐帧接触优先级；姿态边界处小步改善被固定进展阈值忽略。
已修正这些语义，并增加保持原角度、连续性和接触预算的非线性可行性恢复。
源姿态参考建立后直接进入接触求解，删除会先拉开接触的中间弱约束净空阶段。
第二段的诊断试算中脚部法向平均误差约 4 毫米，局部仍有 2–3 厘米残差；
这不代表切向滑移、完整场景穿透或所有数据的最终验收通过。

配置状态：测试片段共用一份算法配置，没有逐动作或逐身体部位调参；
但附录 A 的对外配置合并尚未完成，现有 profile 和默认值中仍有历史参数。
统一累计角度界已经改为单一默认来源，不能据此宣称全部配置已精简。

## 1. 目标

接触只走一条管线：观测 → 时序滤波 → 接触集合 → 锚点 → 角色 → 求解 → 验收。
手、脚和其他身体部位用同一套判据、时序、锚点、角色和预算，代码里不按部位名称分支。

不需要为手脚额外声明任何东西：

- **源侧**：SMPL-X 分段本身就是区域（脚、脚趾、手、臀……），`full_body.labels` 已经给出。
- **机器人侧**：源区域落在机器人哪个刚体，由现有学习对应决定（`surface_anatomy` 按关节链约束，
  人的脚落在 `left_ankle_roll_link`）；接触几何直接取这个刚体的碰撞体。
- **锚定**：每个"机器人刚体 × 被接触表面"按一个刚体放置锚定，源上哪些点在接触只决定这个刚体
  绕哪条边或哪个点转。脚、手、臀规则相同。

手脚本身也不需要声明：它们就是 `joint_mapping` 里没有下游映射的关节（踝、腕）所在的刚体，
连同固定关节焊在上面的子刚体。默认的 `unitree_g1.json` 已映射 `L_Ankle → left_ankle_roll_link`、
`L_Wrist → left_wrist_yaw_link`，而 `left_rubber_hand` 通过固定关节 `left_hand_palm_joint` 焊在腕上，
所以任何 profile 都能直接得到手脚。

手脚与其他部位唯一的区别是：它们是运动链末端，没接触时有末端目标。这个目标由接触状态和净空证据决定：
确认离开所有表面时用身体相对、按形态缩放的目标；净空未知时保持世界目标。

脚几乎一直在接触，承担大部分支撑转换，是这次统一的首要对象。

## 2. 概念

| 概念 | 含义 |
|---|---|
| 源区域 | SMPL-X 分段，如左脚、左脚趾、左手、臀部；直接来自 `full_body.labels`。时序滤波按源区域做，只依赖源和场景 |
| 接触块 | 机器人刚体 × 场景表面 id，锚定按接触块做。同一刚体可同时有多个块（脚跟踩台阶边、前掌踩台阶面） |
| 接触集合 | 每帧所有活动接触块。手脚同时接触是常态 |
| 接触事件 | 接触集合变化的帧，即支撑转换点 |

## 3. 现状：脚的独立逻辑

| 位置 | 现在做什么 | 统一后 |
|---|---|---|
| `core.py:394-476`，`contact.build_world_tracking_tasks` | 脚默认只跟踪源逐帧位置，`contact_mask` 全为 False，没有锚点，源脚滑原样传递 | 由统一接触表中的脚底接触块派生 |
| `contact_phase.py` | 世界跟踪末端借用自身刚体上皮肤点的角色 | 末端角色 = 其接触块角色，删除特例 |
| `data_sources/parc_ms.py:286` | 只有脚有 `source_contact_mask` | 作为任意区域的观测覆盖进入统一表 |
| `data_sources/smplx_surface.py:92`，`surface_pose.py:88-146` | 按关节编号把脚的样本标成 `sample_world_tracking` | 按接触状态决定，删除部位标记 |
| `data_sources/smplx.py:136-290`，`foot_frames.py` | 脚底刚体帧标定、脚底带顶点、`foot_sole_points/centres`（已考虑脚跟到脚尖滚动）；下游主流程未使用 | 接触判定改用全身皮肤的脚、脚趾分段；脚底朝向标定评估后并入通用的刚体朝向目标 |
| `data_sources/smplx_motion.py:194` | 按名字为脚单独建朝向标记 | 同上 |
| `_core/geometry.py:95-190, 390-460` | 按名字含 `foot` 查找脚 body，推断脚底 marker 和支撑点 | 接触几何直接取学习对应所在刚体的碰撞体，删除按名字查找 |
| `_solver/end_effectors.py:320-400` | transition 模式的法向约束、sticking 时脚底朝向跟踪 | 由刚体放置锚定统一提供，不依赖名字 |
| `_solver/end_effectors.py:491` | `name in ("left_foot", "right_foot")` 时保护末端残差 | 按接触块状态 |
| `whole_body_guard.py:37, 44, 195` | 名字含 `foot` 就用 `foot_position_budget` | 按接触块状态和角色选预算 |
| `trajectory_validation.py:416, 558`，`endpoint_constraints.py:30`，`local_candidates.py:67` | 名字含 `foot` 就当世界目标、收紧预算 | 同上 |
| `contact.py:658` | 名字以 `_foot` 结尾时翻转法向 | 平面接触面统一处理 |
| `_core/initialization.py:28` | 初始根高度只对齐到名字以 `_foot` 结尾的末端 | 对齐到最低的支撑接触面 |
| `retargeting_defaults.py:42-52` | 脚 `preserve_world_target=True`、缩放 1.0；手缩放 0.9 | 按接触状态；比例由肢体长度计算 |
| `_core/postprocessing.py:24` | profile 没有任务时，临时为脚造世界参考用于验收 | 统一表对任何 profile 都生成脚接触块后删除 |
| `_core/feet.py`，`foot_stabilization` 配置 | 主流程永远跳过（`core.py:557, 727-741`）；只有 `scripts/retarget_omomo_batch.py:260` 还调用 | 删除 |
| `core.py:926` `_extract_foot_positions` | 无调用 | 删除 |

## 4. 目标数据流

```
源皮肤采样 full_body，按 SMPL-X 分段成区域（无皮肤的源：适配器把末端 contact_samples 转成同一格式）
① 统一观测 observe_contacts：距离、朝向、相对接触面材料速度、表面 id
② 源区域时序：滞回、补间隙、删短段、sticking/sliding          （阈值一套，不按部位）
③ 接触集合与事件
────────────────────── 以上只依赖源和场景 ──────────────────────
④ 学习对应把活动的源点落到机器人刚体上，得到接触块（刚体 × 表面）
⑤ 锚点：每个接触块一个刚体放置，存在被接触表面的材料坐标
   新接触在事件帧联合放置，已锁定的锚点不动；进出接触用 approach_blend
⑥ ContactRoles：每块一个角色；有力学证据时按证据
⑦ 逐帧/整段求解、refinement、guard、局部修复、验收、导出都读这一张表
```

### 刚体放置锚定（以脚为例，手和臀相同）

- 由整段接触区间里的源估计这个刚体在被接触表面上的放置（位置 + 绕表面法向的朝向），
  接触点最多的那些帧权重最大。整段离线求解，所以可以用区间内的全部帧。
- 放置确定了这个刚体表面上每一点在世界里该在哪。每帧只要求当前活动的点（活动源点经学习对应落到的机器人点）
  贴在各自的位置上：脚跟着地时只钉脚跟，脚可以绕脚跟转；全脚掌着地时整只脚固定；蹬地时只钉前掌。
- 不要求每个源接触点都被贴住。G1 脚底约 18.6 cm，人脚约 25 cm，逐点锚定会过约束。
  学习对应的几厘米误差只影响刚体上哪一处被钉住，不会造成滑动。
- 源脚本身穿地或悬空时，锚点仍在被接触表面上，差值作为接触残差报告。

## 5. 分阶段实施

每个阶段单独合入，测试保持通过。新增模块保持在 1000 行以内（`contact.py` 已 922 行）。
配置合并与这些阶段配套进行，见[附录 A](#附录-a配置合并)。

### 阶段 0：准备（不改行为）

- 等当前工作区的未提交改动落地，再开新分支。
- 修 `_core/motion.py:610`：只要有角色表就传帧号。现在没有全身皮肤时，逐帧求解会忽略角色。
- 去掉 `retargeting_defaults.py:379` 的门控（只有声明了 `end_effector_priority.effectors` 才合并接触默认值）。
  手脚改为从 `joint_mapping` 的链末端推出，默认 `unitree_g1.json` 也走完整接触流程；
  `effectors` 里的参数只保留为可选覆盖。
- 基线：c167 场景、OMOMO、TRUMANS 各 1–2 段，保存第 6 节全部指标。

### 阶段 1：统一接触表（只输出诊断）

- 源区域直接用 SMPL-X 分段标签。脚和脚趾分开，脚跟到脚尖的滚动由此直接可见。
- 新模块 `source_contacts.py`，复用 `observe_contacts`、`contact_schedule`、`contact_correspondence`、
  `contact_scene_normals`，锚点逻辑从 `contact.py:774-829` 抽出并改为材料坐标。
- 阈值一套：用基线片段上新旧检测的起止帧一致率确定默认值。
- 验收：手的接触起止帧与旧 `contact.py` 对齐；脚得到站立/摆动/过渡分段，并与 PARC 标签对比；
  臀、膝角色不再逐帧闪烁。

### 阶段 2：脚接入统一接触表

- `core.py:394-476` 的手、脚、物体三路合为一路。脚的 `SceneContactTask` 由脚底接触块派生，字段不变。
- 按第 4 节实现刚体放置锚定，先用于脚。
- 第 3 节表中所有按名字判断脚的地方改为按接触块状态。
- 删除 `build_world_tracking_tasks` 的默认调用和 `_validation_foot_references`。
- 验收：站立脚滑移下降；脚踝逐关节 jerk 不增加；接触残差、穿透不回退；手的指标不变。

### 阶段 3：手接入统一接触表

- 替换 semantic 检测路径；`object_contact.merge_object_contacts` 由材料坐标锚点取代。
- 接触几何直接来自机器人碰撞体：G1 的手是球（`rubber_hand`），自然就是点接触，不需要特判。
- 验收：`pytest -q tests/test_contact_priority.py tests/test_object_contact.py`；手的接触时刻和残差不回退。

### 阶段 4：全身表面接入

- `SurfaceContactTasks` 的 active、modes、targets 从统一表读；学习对应不变。
- 臀、背、膝、前臂同样使用刚体放置锚定，替换现在的逐点、逐帧目标。
- 删除 `_core/geometry.py` 中按名字查找脚 body、推断脚底 marker 的代码。
- 验收：角色切换次数下降；表面接触残差不回退。

### 阶段 5：接触集合事件与联合锚点放置

- 事件帧上，新接触在观测到的接触区域内与已锁定接触联合放置锚点；已锁定的锚点不动。
- 余量只有几厘米（如 G1 脚底在人脚印内前后约 3 cm），只吸收小的尺寸差；剩余冲突按角色让步并报告残差。
- 局部修复窗口以事件帧为边界。
- 验收：多接触片段（推墙、爬起、坐下）的接触残差和支撑脚滑移。

### 阶段 6：力学证据

- SMPL-X 适配器生成 `centroidal_demand` 和 `ContactCandidate`，经 `explain_source_support` 得到
  `source_support_evidence`，先作为配置开关。
- 多接触时由证据决定谁是必要支撑，避免手脚同为 `hold` 时残差平摊、把支撑脚拖滑。

### 阶段 7：自由目标按接触状态，清理

- 应用第 8 节的 AGENTS.md 修改。
- `_solver/targets.py`、`_solver/end_effectors.py:430-460`、`_refinement/*`、`_sequence/tasks.py`：
  确认离开表面 → 缩放目标；净空未知 → 世界目标。`preserve_world_target` 只保留为显式覆盖。
- 缩放比例在校准参考姿态下由肢体长度计算（腿：髋到踝；臂：肩到腕）。
- 删除 `_core/feet.py`、`foot_stabilization` 配置、`sample_world_tracking`、`_extract_foot_positions`，
  更新 `code_layout.md`、`contact_phase_priority.md`、`whole_body_surface_contact.md`。
- 验收：摆动脚步幅、抬脚高度、台阶净空；落地前后脚速度连续；逐关节 jerk 不增加。

## 6. 验收指标

每个阶段与阶段 0 基线对比，按 AGENTS.md 分开报告：

- 末端世界位置误差、接触残差（按接触块）。
- 每个 sticking 接触块的滑移速度，脚单列。
- 时间抖动：根部平移、根部朝向、逐关节逐帧速度/加速度/jerk，支撑交接帧单列。
- 场景穿透：最大深度、部位、帧。
- 相对未修改源和固定修正前参考的累计关节变化与整体朝向变化。
- 接触检测：新旧起止帧偏差、每段角色切换次数。

## 7. 风险与未决问题

- 统一阈值后手的接触时刻可能变化：阶段 1 先只输出诊断对比。
- 锁定站立脚后，源数据的脚滑不再复现；真实滑步相对速度高，会判为 sliding，不锁定。
- 联合放置余量有限，大的尺寸差仍会留下残差。
- 缩放后的摆动脚可能擦台阶边：靠近接触的一段已由 approach_blend 交接到世界锚点，仍需逐段报告净空。
- 待定：自由末端的参考点用链根（髋、肩，推荐）还是骨盆（现状）。
- 默认 profile 启用接触后，没有场景的片段（平地 10 m 盒子）也会跑接触检测，需要确认耗时。
- 工作区存在并行修改（`contact_phase.py` 等），开工前需要先落地。

## 8. AGENTS.md 修改稿

在阶段 7 行为实际改变时应用，替换 Retargeting Priorities 的第 2 条：

```markdown
- Contact uses one whole-body pipeline: detection, temporal filtering, anchoring, roles,
  solving and acceptance treat hands, feet and every other body region alike. Hands and
  feet are the most frequent contact regions, not a separate system; never branch
  contact logic by body part or action name.
- Contact targets of any region stay at the inferred scene/object contact location,
  anchored in the contacted surface's frame; never scale them, make them root-relative,
  or sacrifice contact accuracy for posture.
- Contact regions come from the source body segmentation (e.g. SMPL-X foot, toes, hand,
  pelvis); no per-robot declaration. Each contact is anchored per robot rigid body and
  contacted surface as one rigid placement; the observed human contact samples only
  select which part of that body touches, so heel-to-toe roll and differing body sizes
  do not over-constrain the solve.
- Simultaneous contacts are handled as one contact set per frame. When a contact starts,
  its anchor is placed within the observed contact region jointly with the contacts
  already held, then locked for the contact's duration; held anchors are never moved
  to satisfy later contacts. Remaining conflicts yield by contact role, and residuals
  are reported.
- The target regime follows observed contact state, not body part. An end effector
  observed clear of every surface uses a body-relative, morphology-scaled target whose
  scale is measured from robot/source limb lengths; with unknown clearance it keeps its
  world target. Hand-overs follow source progress, not fixed durations.
```

Validation 增加：

```markdown
- Report stance slip for every sticking contact region, feet included, and swing-foot
  terrain clearance whenever free feet use scaled targets.
```

## 附录 A：配置合并

### A.1 现状

算法参数约 400 个，分两处定义：

- `retargeting_defaults.py`：276 个。其中 `end_effector_priority` 162 个、`trajectory_refinement` 64 个；
  `foot_stabilization` 22 个（对应代码已不运行）；`object_alignment` 5 个（无人读取）。
- 9 个模块各自的 `DEFAULTS`，共 123 个：sequence 29、whole_body_correction 23、local_repair 18、
  surface_contact 16、surface_learning 12、self_contact 10、self_collision 7、surface_pose 6、mink 2。

两套合并机制不一致：`retargeting_defaults` 只在 profile 声明 `effectors` 时合并，模块 `DEFAULTS` 总是合并。

机器人 profile：

- `unitree_g1.json` 的 `retargeting` 段 20 项，除死掉的 `foot_stabilization` 两个值外全部等于默认值。
- `unitree_g1_instinctmj.json` 中真正属于机器人的只有：绑定姿态（29 个关节角）、朝向任务 body、
  手脚 body 和脚底偏移；后两项按正文计划可从 `joint_mapping` 和碰撞体推出。
- 两个 G1 profile 的 `joint_mapping` 不同：一个多 `Spine3`，腕部分别映射 `wrist_yaw` / `wrist_roll`。
- booster_k1、hightorque 的 `solver_max_iter`、`joint_regularization_boost`、顶层 `effectors` 无人读取。

### A.2 问题

1. **死配置**：`foot_stabilization`（默认值和 5 个 profile）、`object_alignment`、booster/hightorque 上述 3 项、
   重复定义的 `terrain_sample_seed`、已不起作用的 `preserve_contact_anchors`。
2. **按部位设参数**：`contact_detection.effectors`（手 3 cm / 2.0 m/s，脚 7 cm / 0.3 m/s）、
   `_hand_defaults` / `_foot_defaults`、`semantic_contact_effectors` / `world_tracking_*`、
   `foot_position_budget` 与 `free_position_budget`、`swing_joint_step_avoidance.effectors`、
   `pose_direction.edge_joint_effectors`。统一接触后大部分消失。
3. **同一物理量多份且数值不同**：

   | 物理量 | 现状 |
   |---|---|
   | 接触检测阈值 | 5 套：contact_detection 0.06、surface_contact 0.03（touch 0.008）、self_contact 0.02、object_contact 一套、self_collision 检测 0.03 |
   | 接触位置 sigma | 6 份：0.025 / 0.02 / 0.012 / 0.008 / 0.015 / 0.001 |
   | 摆动 sigma | 5 份：0.06 / 0.02 / 0.01 / 0.015 / 0.02–0.05 |
   | 平滑（加速度、jerk） | 4 处：逐帧 temporal、refinement、whole_body 按帧计；sequence 按秒计 |
   | 回退预算 | `contact_regression`、`endpoint_regression`、`endpoint_regression_m`、`regression`、`collision_regression`，0.5–2 mm |
   | 优先级容差 | 8 个配置项，另有代码常量 1 mm 和 surface 的 0.01 |
   | 碰撞采样 | 间距 0.005 / 0.01 / 0.015 / 0.03；最大子步 64 / 128 / 256 |
   | 重复项 | `max_joint_correction` 两处；`self_collision_enabled`、`joint_limit_safety_fraction` 顶层与 sequence 各一份 |

4. **开关过多**：约 20 个 `enabled`，以及 `hierarchy`、`free_motion_priority`、`preserve_noncontact_baseline`、
   `local_contact_chain_only` 等实验遗留开关。

### A.3 原则

1. 三层分离：
   - 源配置（YAML）：数据路径、源类型、帧率，不变。
   - 机器人 profile：只放机器人事实（URDF、`joint_mapping`、绑定姿态、朝向 body）。
     默认没有 `retargeting` 段；确需覆盖的值写明理由。
   - 算法参数：单一来源，对所有机器人相同。
2. 按物理量组织，不按模块或部位：一个量一个名字一个值，所有阶段共用；单位用国际单位制，时间导数按秒计。
3. 用户可调的阈值、预算、阶段开关留在配置；迭代数、离散步长等算法内部量改为带注释的模块常量。
4. 开关只保留阶段级：sequence、refinement、local_repair、whole_body、self_collision（以及以后的力学证据）。
5. 所有 `*_config()` 拒绝未知键（与 `contact_phase_config` 一致），拼写错误和死配置直接报错。
6. 生效值继续写入 diagnostics JSON 的 `effective_config`，不需要把默认值抄进 profile。

### A.4 目标形态

机器人 profile（G1 只保留一个）：

```
name, urdf_path
joint_mapping          ~15 项（手脚由链末端推出）
base_orientation
bind_pose              29 个关节角（原 body_orientation.reference_pose）
orientation_bodies     ~6 项（原 body_orientation.tasks）
```

用户可见的算法配置约 40 项：

```
contact     进入/退出距离、贴合距离、法向速度、sticking 速度、最短帧数、最大间隙、锚点漂移  ~8
smoothness  速度/加速度/jerk sigma（按秒）                                              ~3
budgets     接触误差、接触回退、自由末端预算、滑移回退、优先级容差、关节修正上限、累计姿态界  ~8
collision   场景净空、自碰撞净空、采样间距                                              ~3
stages      5 个阶段的 enabled 和迭代次数                                               ~12
learning    学习对应超参（单独保留，决定缓存键）                                         ~6
```

### A.5 步骤

**第 1 步：删死配置（行为零变化，可先于正文阶段 0 进行）**

- 删除 `foot_stabilization`（默认值和 5 个 profile）、`object_alignment`、booster/hightorque 的 3 个无用项、
  重复的 `terrain_sample_seed`。
- 删除 `unitree_g1.json` 的整个 `retargeting` 段。
- 所有 `*_config()` 加未知键检查。
- 验收：测试片段上输出 qpos 与改动前逐位相同；pytest 全部通过。

**第 2 步：去重（行为零变化）**

- 数值完全相同的重复项合为公共量（`max_joint_correction`、jerk 0.01、`self_collision_enabled` 等）。
- 数值不同的同一物理量不动，列清单留到第 4 步。
- 各阶段读公共量；按帧的 sigma 改写为按秒的等价值，数值不变。
- 验收：同第 1 步。

**第 3 步：随正文阶段 0–2**

- 去掉 `effectors` 门控，默认值对所有 profile 生效。
- 手脚 body、`tracking_offset`、`contact_offsets` 由 `joint_mapping` 和碰撞体推出，`effectors` 降为可选覆盖。
- `contact_detection` 按部位的阈值合为一套，数值由正文阶段 1 的新旧一致率决定。
- 删除 `_hand_defaults`、`_foot_defaults`、`semantic_contact_effectors`、`world_tracking_*`，
  以及 `foot_position_budget` / `free_position_budget` 的区分。
- 两个 G1 profile 合为一个，腕部映射定为一个。
- 这一步改变行为（默认 profile 开始走完整接触流程），按正文第 6 节指标验收。

**第 4 步：随正文阶段 3–7**

- 第 2 步清单中的"同一量不同值"逐项在基线片段上比较指标后定值，或写明该阶段需要不同值的理由。
- 按关节名字符串的权重逐项评估：`interaction_mesh.joint_weight_overrides`、
  `pose_direction.edge_weight_overrides`、refinement 的 `joint_temporal_weights`、`joint_posture_targets`、
  `joint_step_avoidance.groups`。能从 URDF 推出的（如按关节限位定步长上限）就推，推不出的移入 G1 profile 并写明理由。
- 迭代数、信赖域等内部量改为模块常量。
- CLI 只保留运行级参数（路径、求解器、可视化）；`--penetration-resolver`、
  `--replace-cylinders-with-capsules` 并入配置。

### A.6 风险

- 第 1、2 步必须零行为变化，用逐位 qpos 比较保证。出现差异说明某个"重复项"被不同路径以不同方式读取，
  该项移到第 4 步处理。
- 许多测试直接设置配置键来测试功能；改为常量后这些测试需改为打补丁。
- 第 4 步的按关节权重是为 G1 动作质量调出的，直接删除会让逐关节平滑度回退，必须逐项看指标。
