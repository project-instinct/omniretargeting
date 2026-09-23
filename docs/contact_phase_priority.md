# 接触角色与任务顺序

人走路靠脚撑，撑越靠手撑，爬箱时肚子也可能撑住身体；摸东西时手要跟着，但承重的还是脚。
本模块从源动作观测每一帧“哪里在接触、接触面是否静止”，给每个接触一个角色，
再按角色排定求解顺序。不求接触力、承重比例或关节力矩，也不证明动态稳定。

## 角色从观测读出，不设权重和阈值

所有身体部位走同一套 UMR 全身表面观测（`surface_observation.py`）：距离、相对接触面的速度、
接触的是静态场景还是运动物体。手脚已有的主接触（`contact.py`）在活动期间优先决定该末端的角色。

| 源观测 | 角色 |
|---|---|
| 与静态场景接触，且接触点不滑动（`sticking`） | 支撑：站立脚、蹬墙脚、撑住的手、贴住箱沿的腹部、挂杠的手 |
| 其他接触：滑动、刚接触/将离开（`transition`/`touch`）、运动物体、自接触 | 跟随：摸桌子、手在箱面滑、搬物体 |
| 没有接触，且全部原始采样点确认在所有表面之外 | 自由 |
| 没有接触，但净空证据缺失或有内侧证据 | 自由层，但保留原有末端保护 |

- 不看法向朝向，不看“别处是否还有支撑”，不按部位名称或动作名称分支。
- 世界跟踪的末端（脚）没有主接触分类时，直接取自己刚体上 UMR 采样点的角色。
  OMOMO 的脚就是这样得到站立/过渡/摆动的步态分段，不需要单独的脚接触检测器。
- 某末端的刚体已有世界目标（主接触锚点或世界跟踪）时，这个刚体上的稠密采样点只用于判定角色，
  不再作为第二个目标：学习得到的人体—机器人对应是近似的，锚点更准。
- 运动物体、自接触、稠密滑动点仍按原 UMR 路径进入各自目标。能映射到机器人身体表面的场景滑动/轻触点进入“跟随”层，不被误判成支撑；自接触保持独立任务。
- `physical_support_verified=false`、`force_or_torque_optimization=false` 始终写入诊断。

## 求解顺序：不穿透 > 支撑 > 跟随 > 自由 > 姿态

下层只能在不破坏上层最优值的前提下优化。冲突时谁让步由顺序决定，让多少由几何决定
（机器人够不着的部分），残差照实报告。

源 SMPL-X 本身可能穿进地面或物体。本地 5 段 OMOMO 中，脚/脚趾皮肤最多穿地 1.16 cm，
每段 0–38 帧（`out/contact_roles_20260923/source_penetration.json`）。支撑目标来自源动作，
可能就在表面里面；机器人脚型与人不同，目标在表面外也可能让机器人几何穿进去。
所以不穿透排在所有接触之前：目标在场景内时，机器人停在表面上，差值作为接触残差报告。
不改源数据和接触目标，也不新增参数。

- 逐帧 SQP（`_solver/hierarchy.py`）：末端支撑 → 稠密表面支撑 → 末端跟随 → 稠密表面跟随 → 原有摆动、
  躯干朝向、骨向等各层 → 全身姿态。确认离开表面的世界末端不再被钉在“无表面目标”的基准上，
  但仍留在摆动层，排在姿态之前，不降权。
- 逐帧 SQP 的场景穿透本来就是硬约束（`penetration_resolver: hard_constraint`），各层都在其内求解。
- 整段求解（`_sequence/surface_guard.py`）：场景/自碰撞净空不足 → 末端支撑 → 表面支撑 → 末端与表面跟随，
  其余仍是原有加权整段目标和时间正则。原先碰撞只在加权目标里，严格的接触层可能把几何压进表面；
  现在同一批净空行先求并保持。确认离开表面的世界末端不受相对输入轨迹的固定预算约束。
  求解速度与数值见[整段求解说明](sequence_retargeting.md)：OMOMO 上一轮迭代由 1011 s 降到 62 s，
  最后一个子问题不再停在 Clarabel 迭代上限。
- 后处理（轨迹细化、局部修正、全身修正）不因角色放宽预算：自由末端相对主求解结果的移动
  仍受原预算限制。角色切换帧作为局部修正窗口边界。

旧版的 `free_weight`、`support_weight`、4/10 cm 放松距离、法向朝上 0.35、置信度 0.5、
“需另有支撑候选”、`free_position_budget` 均已删除。

## 过渡：按源动作自身进度交接

手没有接触时的目标跟着身体、按形态缩放；接触时是场景里的固定锚点。两者在接触起始帧硬切会让
目标跳一大步。现在在接触区间外交接（`contact_schedule.approach_blend`）：

- 接触从第 `s` 帧开始、锚点为 `A`。源末端到 `A` 的距离就是它的接触向量长度。
  从上次离开后离 `A` 最远的那一帧起，交接比例
  `α = (d(远) − d(t)) / (d(远) − d(s))`，在起始帧正好为 1。
- 交接目标是“源轨迹整体平移到恰好落在 `A`”的世界参考。源动作快，交接就快；源动作慢，交接就慢。
  没有交接时长或距离阈值。
- 离开接触时对称处理。两次接触之间的间隙在源末端离两个锚点都最远处分开。
- 接触区间内部锚点从第一帧起精确保持，不再像旧版那样在区间内部把目标拉回摆动轨迹。
  源接触起止帧、锚点、源数组都不改。
- 逐帧、整段和轨迹细化使用同一个交接目标；随机器人根部移动的那部分 Jacobian 按 `1−α` 缩放。

整段求解本身有非因果的时间正则，硬切换在整段结果里也会被摊开；交接主要改善逐帧求解。

## 论文依据与取舍

1. **Kanajar、Caldwell、Kormushev，RO-MAN 2017**，
   [Climbing over Large Obstacles with a Humanoid Robot via Multi-Contact Motion Planning](https://kormushev.com/papers/Kanajar_ROMAN-2017.pdf)。
   IV-A 把支撑任务排在姿态之前，相邻阶段保留共同接触。借鉴顺序思想；不重新优化接触位置或接触序列。
2. **UMR，2026**，
   [Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence](https://arxiv.org/html/2609.02134#S3.SS3)。
   III-C 用同一表面对应传递接触向量。本模块复用其逐点观测判定角色，并用末端接触向量长度度量交接进度。
   论文没有按承重角色排序；排序是本项目的扩展。
3. **CoRe，Humanoids 2025**，
   [A Hybrid Approach of Contact-aware Optimization and Learning for Humanoid Robot Motions](https://tmjeong1103.github.io/CoRe/static/pdfs/Humanoids2025__CoRe.pdf)。
   III-C 在运动学层按接触区间约束脚部。借鉴按接触区间施加约束的分工；不照搬其滤波和加速度限制。

## 配置与输出

```json
{"retargeting": {"contact_phase_priority": {"enabled": true}}}
```

只有 `enabled` 一个键；其他键会报错并说明已删除。诊断 `contact_phase_priority` 给出每个末端的
`role_frames`、表面支撑点数和部位、按角色集合划分的 `phases`。NPZ 保存 `contact_role_names`、
`contact_roles`（帧 × 末端）和 `contact_role_surface_support`。

## 验证

- `tests/test_contact_phase.py`：角色时序、静态接触即支撑（与法向和置信度无关）、运动物体/滑动不支撑、
  脚取自身皮肤角色、未知净空保留保护、整段与逐帧中腾空脚让位给腹部支撑、站立脚不被腹部拉走、
  后处理保持原预算、只接受 `enabled`。
- `tests/test_contact_phase_primary.py`：手的支撑或跟随接触都优先于腾空脚；站立脚优先于跟随的手
  （逐帧两种自由层模式）；滑动/过渡为跟随且只约束法向；细化与全身修正保护；
  逐帧起始帧跳变：硬切换一步 12 cm，交接后按源进度推进。
- `tests/test_contact_phase_primary.py::test_non_penetration_outranks_a_support_target_inside_the_scene`：
  支撑目标在墙内 4 cm 时停在墙面（穿透 0），去掉不穿透层则穿进 3.9 cm。
- `tests/test_contact_priority.py`：锚点从起始帧精确保持，交接比例按源进度计算，源数组不变。

OMOMO 实录对照见 [角色对照报告](../out/contact_roles_20260923/RESULTS.md)。
旧阈值策略在 `test_data` 上的对照保留在
[旧报告](../out/contact_phase_testdata_20260922/RESULTS.md)，其数值不代表当前策略。
