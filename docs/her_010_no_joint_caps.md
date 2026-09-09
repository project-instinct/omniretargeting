# HER 010：取消固定关节帧间限速

后续躯干/骨盆校准修正已更新 8132 播放，见 [当前身体跟随结果](her_010_core_follow.md)。
本文保留此前取消限速时的结果与运行信息。

2026-09-09，按用户要求取消固定关节单帧步长、加速度及 jerk 硬上限。
最初通过 `end_effector_priority.temporal.hard_joint_motion_limits_enabled=false`
跳过 SQP 中的对应不等式和求解结束后的时间步投影，并生成下述实测输出。
随后按用户进一步要求，彻底删除该开关、限速参数、不等式、输出投影及 core 中的调用。
旧配置也无法重新开启该机制；下述输出保留原运行配置和哈希，不修改历史产物。
机械关节范围、接触约束、场景约束、单次 SQP 信赖域和源相对根部约束仍按已有配置运行。
不按动作、机器人或帧号特殊处理。

输出：`out/her_c167_010_diagnosis/no_joint_caps/final_retargeted.npz`。
播放：http://172.16.0.175:8132/ ，初始停在第 46 帧（0 起算）。
8130 保留旧的 `release_body`，不是此次输出。

## 原因与第 46 帧

原配置把关节单帧变化限制为 0.18 rad（10.31°），二阶差分限制为 0.08 rad。
源动作在第 43–45 帧快速伸腿，右膝连续两次伸展约 24°、19°。
原机器人右膝此时还在受加速度约束转向，到第 46 帧仍显著屈曲。
此次 A/B 展开配置仅新增上述 false 开关，其余配置相同。

| 第 46 帧指标 | 源动作 | 原输出 | 取消硬限速 |
|---|---:|---:|---:|
| 左膝弯曲角 | 26.63° | 41.48° | 23.79° |
| 右膝弯曲角 | 14.69° | 63.30° | 11.73° |
| 胸部朝向误差 | — | 16.91° | 9.32° |
| 左脚世界位置误差 | — | 55.40 mm | 50.72 mm |
| 右脚世界位置误差 | — | 60.18 mm | 64.61 mm |

膝角由对应髋、膝、踝三点计算，0° 表示伸直，不混用机器人关节零位角。
改善来自重新逐帧求解；最终 qpos 与 framewise、local 前后数组逐元素相同。
没有通过修图、改源数据、改时间轴、后处理抬高或插值伪造改善。
腿部伸展明显改善；躯干和自由手仍有偏差，不能称整个动作已经完全贴合。

## 整段验收（99 帧、25 fps）

| 指标 | 原输出 | 取消硬限速 |
|---|---:|---:|
| 胸部朝向误差，平均 / 最大 | 12.38° / 33.84° | 11.50° / 34.08° |
| 左脚世界误差，平均 / 最大 | 51.14 / 160.31 mm | 50.99 / 112.32 mm |
| 右脚世界误差，平均 / 最大 | 58.39 / 102.64 mm | 57.91 / 89.51 mm |
| 自由左手世界误差，平均 / 最大 | 194.53 / 401.13 mm | 185.44 / 381.77 mm |
| 自由右手世界误差，平均 / 最大 | 192.06 / 574.35 mm | 165.65 / 401.75 mm |
| 最大关节单帧变化 | 0.180 rad | 0.694 rad |
| 最大关节加速度 | 50.0 rad/s² | 379.4 rad/s² |
| 关节 jerk RMS | 834.8 rad/s³ | 1533.0 rad/s³ |
| 最大可测表面相交 | 1.000142 mm | 1.000005 mm |
| 最大相交 body / 帧 | right_wrist_yaw_link / 45 | left_elbow_link / 46 |

此段没有推断到语义接触，接触残差与 sticking 滑移为 N/A。自由手世界位置不是高优先级
目标，上表保留供诊断。jerk 增加包含更快的源运动响应，不能独立区分刻意动作与求解噪声，
不能宣称更平滑或已满足动力学执行要求。

独立表面相交查询沿 29 个实际碰撞胶囊/球的轴以 1 mm 间隔采样，与原场景三角面测距，
轴距离漏差上界 0.5 mm。新输出略超 1 mm 的保存帧为 46、49；不是可靠闭合实体穿透深度。
生产验收还检查 4,373 个帧间姿态，最大约 1.000437 mm，无细分上限触顶。
开放扫描面侧别未知仍涉及全部 99 帧，因此不作连续、闭合实体无碰撞保证。
局部碰撞阶段未改任何帧，新增水平和垂直修正均为 0，不能把新姿态归功于局部修复。
全流程 153.77 s。

原 SMPL-X 文件数组及场景 size/mtime 未变；两次输出中的人体关节、接触 mask/mode、
接触目标和末端世界参考逐元素相同。展开配置和代码哈希随新输出保存。
后处理候选因根部、末端、关节修正和边界偏差被拒绝，未写入最终输出。

## UMR 对照

核对 UMR 固定提交 `8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09`。
在其单序列 `solve_frame_body_segment_qp` 及 GRAIL defaults 路径，没有看到本项目原有的
逐关节相邻帧 0.18 rad 加二阶差分 0.08 rad 硬限制组合，但它也不是完全不限步长：

- 默认 `step_limit_mode=l2`，局部关节每次 QP 增量 L2 范数上限为 0.4。
- `max_dq=0.15` 只在启用 box 模式时形成通用逐关节盒约束；机器人显式
  `dof_max_dq_box` 可另行施加限制，不能仅凭字段存在认定默认生效。
- 默认 `iters=1`，sequential 用前一帧作初值，所以单次迭代上限仍能影响相邻帧跟随。
- 有关节位置范围、速度/加速度软代价和 LQR 后处理，没有照搬其代码或资产。

来源：[GRAIL defaults](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/humanoid_retarget_defaults_hsi_hoi_grail.json)、
[帧求解及序列调用](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/scripts/retarget_smpl_to_humanoid_surface_vector.py)、
[步长约束和 LQR](https://github.com/hanyang9/UMR/blob/8c4db7db7cb1fb998ac5b257e1bb32ad44a50d09/scripts/smpl_surface_retarget_common.py)。

## 验证产物

原始试跑聚焦测试：contact_priority、robot_config、source_orientation_step、body_orientation_reference
共 92 passed；新增正反方向快速关节目标的求解/输出投影测试 4 passed。
数值求解器在合成测试上有 inaccurate 警告，最终姿态、限位及启闭对照断言均通过。

彻底删除限速实现后，移除两项只测试旧投影的用例；快速动作回归改为检查默认配置及
含旧限速字段的配置都能保留正反方向 0.45 rad 的可达目标。
contact_priority、robot_config、basic、local_collision_repair、local_candidates 和
joint_motion_limits 合计 140 passed、4 skipped，5 条求解器 inaccurate 警告。
本轮删除不重新生成已展示的 010 文件；此前该机制已关闭。

明细：`out/her_010_frame46/no_joint_caps_comparison.json`；可复跑脚本
`out/her_010_frame46/compare_caps.py`，独立几何使用已有 `measure.py`。
对照图：`out/her_c167_010_diagnosis/no_joint_caps_frame46.png`。
