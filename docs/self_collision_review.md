2026-09-10 自碰撞调研与现有轨迹复测

2026-09-11：后续实现及验证见 [自碰撞处理](self_collision.md)。下文保留实现前的调研基线。

结论：OmniRetargeting 已有自碰撞距离 Jacobian 和部分局部修复/验收能力，但默认关闭，
全轨迹 refinement 也没有对应项。现有两段 G1 输出的实际碰撞体有厘米级自穿透。
UMR 有可以借鉴的实现，但常用 defaults 同样关闭几何自碰撞；GMR 官方主流程没有接入
其依赖 Mink 提供的避碰约束。不能把任何一个项目的默认输出当作无自碰撞保证。

本轮只做源码调研、只读轨迹测量和方案整理，未改变求解器、默认参数、机器人模型或已有
动作。工作区原有未提交修改保留。这里的数值是问题基线，不是修复后的运动验收。

核对的版本为 UMR `d6bb76123d19afb7c2c1c84162d1af1f142a61ed`（2026-09-10）、
GMR `bb1bbe40774794fceb2a7c579a3464a28e68c844`、
Mink `ab45779fea46933832dee1c240f94103633347a1`。
本地基于 `0f9d2ebe4ee17d7a6173c739bb3ff3a3a5918d53` 加当前工作区修改。
上游文本和 GitHub commit/tree 元数据保存于
`/tmp/omniretargeting-self-collision-review-20260910/`；没有复制上游代码到本项目求解器。

**上游实际做了什么**

| 项目/机制 | 实际行为 | 对本项目的意义 |
|---|---|---|
| UMR 几何自穿透 | 临时增大 MuJoCo geom margin，收集碰撞候选，`mj_geomDistance` 查询有符号距离，两个最近点的相对 Jacobian 进入 QP | 当前逐帧实现已有同类核心数学，不能仅换成 UMR 查询就解决默认关闭/阶段遗漏 |
| UMR 可松弛不等式 | 线性化 `d + J_d Δq >= margin - slack`，slack 非负且有二次代价 | 避免深穿透与小信赖域同时出现时要求一步退出；仍需要限制 slack、迭代重查和最终报告 |
| UMR self-contact map | 对选定源表面点对，在机器人对应点距离小于源点距离时施加单边惩罚 | 可以帮助保持肢体间距；点对距离非 signed solid distance，不能证明全机器人不穿透 |
| GMR 官方主求解器 | `ik_limits` 只加入 `ConfigurationLimit` 和可选 `VelocityLimit`，两级 IK 都使用该列表 | 未接入 `CollisionAvoidanceLimit`，不宜把它当成直接可用的自碰撞修复范例 |
| Mink `CollisionAvoidanceLimit` | 显式构造 geom 对，安全包围体筛选后直接查距离；靠近安全距离时限制继续接近的法向速度 | 可以借鉴提前激活、碰撞对缓存和距离分级；不需要为此替换现有 CVXPY 求解器 |

UMR 的 [common 查询/Jacobian/QP 实现](https://github.com/hanyang9/UMR/blob/d6bb76123d19afb7c2c1c84162d1af1f142a61ed/scripts/smpl_surface_retarget_common.py)
对应 `build_robot_self_penetration_cache`、`compute_robot_self_penetration_rows`、
`collision_relative_jacobian`、`solve_clarabel_qp_step`。
[单序列求解器](https://github.com/hanyang9/UMR/blob/d6bb76123d19afb7c2c1c84162d1af1f142a61ed/scripts/retarget_smpl_to_humanoid_surface_vector.py)
同时包含 self-contact map 和几何自碰撞两条独立路径，
[batch 求解器](https://github.com/hanyang9/UMR/blob/d6bb76123d19afb7c2c1c84162d1af1f142a61ed/scripts/retarget_smpl_to_humanoid_surface_vector_batch.py)
也接入了几何自碰撞行。self-contact map 的单边激活条件是
`robot_distances < source_pair_distances`，不是双边距离保持，也不是有符号防穿透。

UMR defaults 必须分别看：

| 配置 | 几何自穿透 cost | hard_constraint | margin | hard_slack | slack_cost |
|---|---:|---|---:|---|---:|
| [普通 SMPL-X](https://github.com/hanyang9/UMR/blob/d6bb76123d19afb7c2c1c84162d1af1f142a61ed/humanoid_retarget_defaults.json) | 0 | false | -0.001 m | true | 1000 |
| [HSI/HOI standard](https://github.com/hanyang9/UMR/blob/d6bb76123d19afb7c2c1c84162d1af1f142a61ed/humanoid_retarget_defaults_hsi_hoi_standard.json) | 0 | false | -0.001 m | true | 1000 |
| [HSI/HOI GRAIL](https://github.com/hanyang9/UMR/blob/d6bb76123d19afb7c2c1c84162d1af1f142a61ed/humanoid_retarget_defaults_hsi_hoi_grail.json) | 0 | false | -0.001 m | true | 1000 |
| [NR](https://github.com/hanyang9/UMR/blob/d6bb76123d19afb7c2c1c84162d1af1f142a61ed/humanoid_retarget_defaults_nr.json) | 0 | true | -0.001 m | true | 5000 |

前三项 cost=0 且 hard=false，几何自碰撞 cache 不建立。NR 的 margin 表示允许 1 mm
穿透，slack 还能进一步放宽；其 QP 未给此 slack 配有限上界，不能把参数名里的 hard
理解为最终严格无穿透。`collision_threshold=0.1` 是距离查询/激活范围，不是 10 cm
安全间隙，也不是允许穿透 10 cm。是否有 self-contact map 是另一个判断。

[GMR 官方实现](https://github.com/YanjieZe/GMR/blob/bb1bbe40774794fceb2a7c579a3464a28e68c844/general_motion_retargeting/motion_retarget.py)
当前构造函数的 `use_velocity_limit` 默认是 false；开启时才加速度上限。无需将这种
固定关节限速移植回来，本项目已移除它以保留快速源动作。

[Mink 官方实现](https://github.com/kevinzakka/mink/blob/ab45779fea46933832dee1c240f94103633347a1/src/mink/limits/collision_avoidance_limit.py)
默认最小间距 5 mm、检测距离 10 mm、gain=0.85。它过滤同一焊接体、父子体及不匹配的
碰撞掩码。安全距离外限制接近速度；距离已低于下限时，默认约束主要阻止继续靠近，
并不要求每一步主动恢复到安全距离。因此它适合预防，还需要单独的深穿透恢复机制。
本项目也需要核对自身 XML excludes，不能把其他项目的过滤策略直接等同于当前模型。

**本项目的实际缺口**

- [retargeting_defaults.py](../omniretargeting/retargeting_defaults.py) 的
  `self_collision_enabled=False`；两段实测输出的 effective config 与 local config 均为 false。
- [_solver/kinematics.py](../omniretargeting/_solver/kinematics.py) 已有双点相对 Jacobian，
  逐帧约束为 `J_d Δq >= -d - penetration_tolerance`，默认 tolerance=1 mm。
  自碰撞行没有 slack，不能保证深穿透、关节限位、接触与信赖域组合总是可行。
- [_solver/step.py](../omniretargeting/_solver/step.py) 只有在
  `hard_penetration_constraint=True` 时才调用包含自碰撞的函数；
  [_core/motion.py](../omniretargeting/_core/motion.py) 又将该值绑定到
  `penetration_resolver == "hard_constraint"`。因此仅打开 self_collision 开关也未必
  在逐帧阶段生效。`core.py` 的 active_stages.self_collision 目前只显示开关本身。
- [trajectory_refinement.py](../omniretargeting/trajectory_refinement.py) 与
  [_refinement/tasks.py](../omniretargeting/_refinement/tasks.py) 的碰撞项只查询场景，
  没有机器人两体的自碰撞残差。单独修逐帧阶段仍不能保证后续优化不再引入穿透。
- [collision_window.py](../omniretargeting/collision_window.py)、
  [local_candidates.py](../omniretargeting/local_candidates.py)、
  [trajectory_validation.py](../omniretargeting/trajectory_validation.py) 已支持开启后的
  自碰撞局部修复、候选与帧间验收，可以复用。局部约束目前依赖普通 `data.contact`，
  并不统一使用逐帧阶段扩展后的检测 margin；`dist < 0.01` 的条件本身不能让尚未进入
  contact 列表的近距离对被查询。开关关闭时 self_depth 的零数组也不能解释为已测得无碰撞。

**已有输出的只读复测**

使用各输出自身的 `robot_profile.json` 加载 InstinctMJ G1，检查 29 个碰撞
capsule/sphere。编译模型共有 64 个 geom，4 个 exclude，0 个显式 pair；29 个碰撞
geom 的 contype/conaffinity 均为 1/1，问题不是这些碰撞体全部被 mask 关闭。
保留 MuJoCo 原有过滤，对接触候选用 `mj_geomDistance` 复核；两者距离差在本次所有
采样中均为 0。保存的关节顺序与模型逐项核对，读取前后原 NPZ SHA-256 相同。

| 已有输出 | 自穿透 >1 mm 的保存帧 | 保存帧最大深度 | 最严重碰撞体/帧（0 起算） |
|---|---:|---:|---|
| `her_c167_010_diagnosis/core_bind_balanced` | 38 / 99 | 78.618 mm | torso_link ↔ right_wrist_yaw_link，57 |
| `her_c167_1530_release_body` | 50 / 143 | 48.315 mm | torso_link ↔ right_wrist_yaw_link，110 |

010 另有左腕与左大腿约 42.932 mm、右腕与右小腿约 41.397 mm 的相交。
每个相邻帧区间额外查 7 个 MuJoCo 流形插值姿态：010 查 686 个，1530 查 994 个。
010 在第 56→57 帧的 alpha=0.75 处达到 80.669 mm；1530 最大值仍在保存帧。
这只说明保存帧检查可能低估峰值；均匀 8 等分不是连续无碰撞证明。
数值描述碰撞代理几何，不等同于可视网格穿透深度，也未检查被模型明确排除的对。

此前 [010 身体跟随报告](her_010_core_follow.md) 的约 1 mm 是机器人与场景的表面相交，
与这里的机器人自穿透是两个独立指标。场景交互的 10 cm 容忍政策不能自动成为自碰撞
通过标准。本轮未改动作，未重新进行末端误差、接触残差、抖动或场景穿透验收。

复测脚本与逐帧/碰撞对详情位于
[`audit.py`](../out/self_collision_review_20260910/audit.py)、
[`audit.json`](../out/self_collision_review_20260910/audit.json)：

```bash
.venv/bin/python out/self_collision_review_20260910/audit.py \
  out/her_c167_010_diagnosis/core_bind_balanced/final_retargeted.npz \
  out/her_c167_1530_release_body/final_retargeted.npz \
  --output out/self_collision_review_20260910/audit.json
```

**建议的落地顺序（本轮尚未实现）**

1. 将自碰撞检测、约束开关与场景穿透 resolver 解耦。统一几何对、过滤规则、距离/法向
   和配置传递；诊断显式区分未检测、已检测、约束生效，而不是关闭时输出看似通过的零。
   先以当前 010/1530 和模型中立姿态建立可复现基线，再启用默认处理并验证其他 profile。
2. 复用现有 QP 与相对 Jacobian：近距离预防参考 Mink；已有穿透参考 UMR 的松弛形式，
   但增加 slack 上界、非线性重验及分步减小穿透的接受规则。固定浮动基座，调整相关肢体
   的关节，保护可达接触手/脚的世界目标；整体平移或转动机器人不会改变自碰撞相对几何。
3. 为整段 refinement 加入同一套自碰撞行，贯通现有 local repair、DP 连接检查和最终
   帧间检查。正则化新增修正 Δq，不重新引入固定帧间关节限速或延迟快速动作的强滤波。
4. 再考虑 UMR 的源表面点对间距作为辅助姿态项。需要先建立可靠、源无关的表面绑定
   契约；源 SMPL-X 提取留在 adapter。真实自接触可以贴近，不能一律强行撑开较大间隙。
   该项始终不能替代实际机器人碰撞体检查。
5. 用同一源数据/场景/尺度对照，分开报告自碰撞深度及帧数、手脚世界误差、接触残差、
   新增修正的时序抖动和场景穿透；残余 slack、失败窗口及受影响碰撞对必须可追踪。

已有测试验证：`tests/test_contact_priority.py` 全部测试，加 local_collision_repair
的开关/验收用例两项与 local_candidates 的 DP 自碰撞边用例，共 **41 passed**。
这确认当前实现与原有策略一致；其中包含明确要求默认跳过自碰撞的用例，并不是修复证明。
