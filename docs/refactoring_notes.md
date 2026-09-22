# 本轮重构与后续边界

本轮围绕工具模块、旋转转换、接触时序和机器人加载做了局部审查；
几何与碰撞模块只检查了职责边界，没有把整个代码库视为已经审完。
算法正确性修复由并行主任务处理。

## 已完成

- 将 `utils.py` 的邻接、源侧边权和 Laplacian 实现迁至
  `interaction_graph.py`，五个旧导入仍显式再导出。图构造、权重和
  节点排序不变。`utils.py` 从本轮开始时的 665 行缩至 508 行。
- 两段手写 axis-angle / quaternion 转换改用已有依赖
  `scipy.spatial.transform.Rotation`；封装仍处理 wxyz 排序、任意前置
  维度、空批次、浮点 dtype 和原先明确的小角度归零阈值。
- 接触评分的 sigmoid 改用已有 `scipy.special.expit`，保留原有
  `[-60, 60]` 输入裁剪；没有更改接触阈值、滞回或分段规则。
- URDF 浮动关节检测交给已经使用的 `xml.etree.ElementTree`。
  合法的单引号和属性空格现在能够识别；注释、其他元素或 Gazebo
  扩展中的同名属性不会被误判为机器人浮动关节。移除无其他调用的
  正则表达式去注释函数。
- 更新 `code_layout.md`，补充旋转边界及真实 MuJoCo 模型加载测试。

旋转转换有两点明确的数值行为修正：旧 `arccos(w)` 会因舍入丢失
大于显式阈值的小旋转，库实现保留这些旋转；零四元数现在抛出
`ValueError`，不再被无效计算静默当作恒等旋转。这两个 helper
目前在仓库内只由 `utils.py` 兼容导出，活动的批量 SLERP 没有改动。

验证：旋转、机器人加载、图权重消融、接触推断和模块大小共
66 个测试通过；旧图函数导出身份检查通过；`git diff --check` 通过。
对 10001 个评分输入的对照中，sigmoid 的 float64 输出一致，float32
最大差为一个机器 epsilon（约 `1.19e-7`）。没有进行速度基准测试。

## 值得继续做，但应分开验证

1. `geometry_query.RigidGeometryQuery.__init__` 同时承担碰撞代理构建、
   编译和几何保真检查。可以拆成模型构建与保真验证助手，并复用两处
   双向 KDTree 顶点距离检查。这里已经使用 MuJoCo 和 SciPy，主要问题
   是职责集中，不是缺少库。
2. `collision_motion.py` 通过 `_refinement.math` 获取通用机器人关节
   坐标，底层碰撞工具依赖了具体优化阶段。可把跨阶段使用的关节索引
   / 坐标工具移至独立的机器人运动学模块，再保留旧导入兼容。
3. `_core.feet` 与 `contact_schedule` 各有连续 True 区间迭代器。
   应直接复用已有实现；不需要为了几行分段代码再引入依赖。

## 不宜机械替换的部分

- 现有 SLERP 支持多个关节与任意时间轴。直接换成每条轨迹一个
  `scipy.spatial.transform.Slerp` 会改变批量接口并增加 Python 循环。
  要替换，先确认等价性和性能，不能仅凭库名判断更好。
- 图中的权重、逐行归一化、孤立点和冻结源侧权重有明确语义。
  不应直接换成通用图 Laplacian 默认配置。
- 场景查询已经使用 libigl 的 AABB / winding-number 和 MuJoCo。
  开放扫描网格的“不确定”状态、碰撞体与视觉体区分仍由项目负责；
  普通 signed-distance API 不能自动替代这些策略。
- 词典序目标保护不等于普通加权 QP。若剖析发现 CVXPY 建模是主要
  开销，可评估直接组装求解器矩阵，但必须保留各层最优值界限和锥约束。

库 API 依据：[SciPy Rotation.from_quat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html)、
[SciPy expit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html)。

## 真实窗口剖析后的优先级

主修复任务提供的 `/tmp/omniretargeting_fix_profile.log` 来自一段真实窗口，
单轮总时间 19.957 秒。以下为累计时间，父子项有包含关系，不能相加：

| 路径 | 累计秒数 | 说明 |
| --- | ---: | --- |
| `TrajectoryValidator.geometry` | 19.499 | 包含保存帧及密集帧间检查 |
| `RigidGeometryQuery._open_surface_rows` | 8.552 | 开放面的精确胶囊查询 |
| `SceneQuery.query` | 7.813 | 点到场景组件的距离及可靠性 |
| `segment_triangle_witnesses` | 5.318 | 已包含在开放面查询时间内 |
| `cvxpy.Problem.solve` | 0.230 | 约占总耗时 1.15% |

这份证据不支持优先更换求解器。应先消除重复几何工作；不能通过
减少帧间样本、忽略开放面或放松 unknown 条件来制造提速。

### 已实施：跳过可证明无效的中心预查询

`ScenePenetrationProbe` 原来每个 pose 先查询所有 geom 中心，再根据
`near` 掩码查询完整表面样本。当场景含有任何一个没有局部定向证据的
开放组件时，`SceneQuery.query` 明确对所有输入设置 `uncertain=True`，
所以所有 geom 必然被保留，中心预查询不能剔除任何工作。

现在在不可变场景初始化时缓存 `has_unsigned_only_components`；只有
这个条件成立时直接保留全部 geom。表面 offsets 本来就包含零偏移中心，
因此中心仍随第二次查询被完整检查。实际样本与顺序、闭合体查询、
开放面胶囊查询和 unknown 传播均不变。有完整局部证据或仅闭合组件时，
仍走原来的中心剔除路径；不会把“存在证据”当作“每个点都可靠”。

`query_work_bound` 保持原来包含中心预查询的上界，没有放宽单次预算
预检。实际计数只减少真正省去的查询。专属回归通过强制恢复旧预查询
分支，与新路径比较所有碰撞字段，并覆盖密集帧间深度及最终验收结论。
13 个专属用例与几何、查询、局部修复及模块大小检查合计 71 项通过。
有限总预算下，省出的查询可能允许检查更多候选；这不代表有限预算的
截断位置也必须相同，单次预估上界仍保持原值。

前述 19.957 秒剖析和已停止的 16 轮整段实测属于优化前已加载的代码。
去重后的 143 帧两轮实验耗时 354.6 秒，详见 `refinement_contract_fix.md`；
没有同预算的端到端计时 A/B。此项等价优化的收益按减少的实际查询次数
验证，不将整个修复流程的耗时归因于这一个改动。

### 其余建议：尚未实施

1. **固定场景数据只解析一次。** `_open_surface_rows` 内层循环反复
   访问 `part.bounds`、`part.triangles_tree`、`part.triangles`；这些属性
   已被 Trimesh 缓存，但每次仍经过缓存有效性检查。剖析中的
   `trimesh.caching.verify` 累计为 0.754 秒。可在几何后端构造时保存
   bounds、三角形数组和现有 R-tree 引用，查询时直接使用。场景必须
   继续保持不可变；保留原 `faces` 映射及排序，不能因树的遍历次序
   改变最近面并列时的选择。该项有证据支持，但收益不应夸大。
2. **精确点退化情形复用现有库。** 球体传入 `start == end`，当前仍
   执行完整的 segment/edge/face 候选枚举。对于所有候选三角形都
   非退化的这种情况，可只调用已有的
   `trimesh.triangles.closest_point` 一次。退化三角形继续使用当前
   边/顶点回退；不能把很短但非零的胶囊轴近似成点。实施前应对面内、
   边、顶点和并列最近点逐项比较 witness 与深度，且先统计该情形频率。
3. **暂不直接替换 segment/triangle 核心。** 现有绑定提供 libigl
   点到网格距离，Trimesh 提供点到三角形最近点；后者已在函数内使用。
   它们没有替代“整条线段到每个有限三角形”的当前接口。仅检查胶囊
   两端会漏掉轴中段穿面；普通射线查询也不能给出胶囊半径导致的接近
   或穿入深度。若以后评估其他库，仍需明确保留轴穿面时的最短退出
   向量、等长多出口的 ambiguous 状态、退化三角形及原始 face ID。

现有 broad phase 已使用库提供的 R-tree，不需要另写一个空间树。
Trimesh 对点最近点及 triangle R-tree 的接口见
[官方 triangles API](https://trimesh.org/trimesh.triangles.html#trimesh.triangles.closest_point)
和 [triangles_tree](https://trimesh.org/trimesh.base.html#trimesh.base.Trimesh.triangles_tree)。
