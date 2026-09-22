# 代码结构

Python 文件保持在 1000 行以内。按职责拆分，保留原有入口和调用签名；
不要通过压缩格式满足行数限制。`tests/test_module_size.py` 检查包、脚本和测试。

| 入口 | 职责 | 实现模块 |
| --- | --- | --- |
| `sequence_retargeting.py` / `retarget_sequence` | 可通过 CLI `--solver sequence` 使用的通用整段求解 | `_sequence/coordinates.py`：模型切空间、机械限位与按秒计算的差分；`tasks.py`：源固定交互图、接触与姿态任务；`_core/sequence.py`：主流程数据衔接。见 [使用说明](sequence_retargeting.md) |
| `core.py` / `OmniRetargeter` | 初始化、接触任务准备、重定向和后处理编排 | `_core/motion.py`：逐帧执行和源根部参考；`geometry.py`：机器人几何推断；`feet.py`：足部稳定和局部表面修正；`postprocessing.py`：平滑、关节安全限制和质量报告；`visualization.py`：Matplotlib 预览 |
| `retargeting.py` / `GenericInteractionRetargeter` | 机器人配置、交互网格和逐帧优化循环 | `_solver/step.py`：组装 SQP 问题；`constraints.py`：接触链和根部约束；`hierarchy.py`：分层求解和步长保护；`orientation.py`、`pose.py`、`end_effectors.py`：各类目标；`kinematics.py`：Jacobian 和碰撞约束；`targets.py`：目标变换和关节限制 |
| `trajectory_refinement.py` / `refine_joint_trajectory` | 全轨迹迭代、时间正则化、稀疏最小二乘求解 | `_refinement/effectors.py`：解析末端配置；`tasks.py`：任务线性化；`guard.py`：复用最终验收预算的约束求解与非线性检查；`math.py`：坐标、朝向和差分工具 |
| `contact.py` | 场景接触推断和任务数据结构 | `contact_schedule.py`：滞回、短接触段和间隙处理；`object_contact.py`：原始运动物体的主要端点接触与环境标识 |
| `whole_body_correction.py` | 固定机器人基准的全身接触/碰撞联合修正与时间正则 | `whole_body_guard.py`：固定末端预算、物体材料坐标帧间目标、实际几何回退；`moving_scene.py`：原始运动物体闭合组件与反向交叉查询 |
| `self_contact.py` | 保留手与大腿的表面接触向量、共享 Jacobian 和误差验收 | `data_sources/smplx_surface.py`：固定重心采样；`surface_correspondence.py`：参考姿态下绑定实际机器人表面 |
| `surface_contact.py` | 全身表面到场景的接触向量、共享求解/验收与 NPZ 导出 | `surface_learning.py`：固定索引学习、绑定和缓存；`surface_anatomy.py`：关节链对应范围；`surface_observation.py`：材料坐标中的接触观测；`data_sources/smplx_surface.py`：真实全身表面采样 |
| `local_collision_repair.py` | 局部修复事件、候选验收和诊断 | `collision_window.py`：窗口内关节优化和碰撞评分；`scene_witness_guard.py`：固定表面点的已知几何状态保护；`open_surface_features.py`：开放网格折边可靠性 |
| `self_collision.py` | 碰撞体自碰撞查询、间距约束和独立诊断 | `collision_motion.py`：场景/自碰撞共用运动范围估计；`_refinement/self_collision.py`：轨迹残差与非线性检查 |
| `main.py` | CLI、源配置和主流程 | `motion_visualization.py`：可视化场景、视频导出和播放 |
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
