# OMOMO 空物体帧与 reset

官方 `train/test_diffusion_manip_seq_joints24.p` 的 `obj_scale` 是逐帧尺度。
`OmomoDataSource` 保持正常帧的尺度、平移、旋转和人体参数不变，不能替换成
序列常量。平移为原点且尺度正常的帧有效。

`data_sources/omomo_validity.py` 在 SMPL-X FK、表面采样、接触检测前识别
`obj_scale <= 0` 或非有限值。平移全零且尺度为零也包含在其中。

- 两端有有效帧的缺口，按 `缺帧数 / framerate <= 0.5 s` 判定可补；在两侧
  有效帧之间线性插值尺度、平移和 COM，旋转用 slerp。只写空帧。
- 首尾缺口不外推。更长的缺口也不补。
- 先标出可补的内部缺口，再保留最长连续可用段；长度相同保留最早段。
  只补最终保留段内的缺口，人体时序参数同步切片。静态形状和 rest offsets
  不切片，SMPL-X 表面、足底和朝向从切片后的原始人体参数生成。
- 全段无有效物体帧时明确报错，不生成退化物体。

日志 `OMOMO object gaps:` 和 MotionData 的 `object_gap_report` 记录原序列的
0-based 缺帧、实际补帧、截断范围及保留区间；截断数量分别报告空帧和正常帧。区间 `first/last` 为包含端点，
`retained_stop` 不包含端点。`source_frame_indices` 将输出行映射回原序列；截断
不将物体、人或接触时间线错开，也不填回被丢弃的人体帧。

OMOMO 输出 NPZ 增加：

| 字段 | 含义 |
| --- | --- |
| `object_valid` | `(T,) bool`，原始物体数据有效；补出来的帧为 False |
| `reset_invalid` | `(T,) bool`，源数据导致的强制 reset 排除，即 `~object_valid`；不是完整几何验收 |
| `reset_states` | 导出时空帧 `invalid`，其余 `unknown`，因为独立几何审计尚未完成 |
| `source_frame_indices` | 对应原序列行号；显式重采样后可为小数 |
| `object_gap_report_json` | 不需要 pickle 的 JSON 字符串，记录原始缺口处理 |

独立逐对审计 `review_penetration_persistence.py` 仍计算实际碰撞深度、持续时间和
帧占比；reset 按共享 `quality_policy.reset_states` 合并 `object_valid`，源空帧
优先标 `invalid`，原因写入 `source_object_invalid`。训练碰撞审计也把同一批帧
并入 `failed_frames`，单列 `source_object_invalid_frames`。几何干净或未知均不能
把源空帧变成可用起点。这些标记不改变求解器或几何验收阈值。

显式 `--output-framerate` 重采样时，目标帧涉及的两个源端点必须都有效，才标
`object_valid=True`；原始缺口报告继续指向原序列。源有效性与碰撞/物理就绪是
不同信息，几何 candidate 仍为“未物理验证”。

全量数据统计可复现：

```bash
PYTHONPATH=/home/duanxin/Xyk/omni_opt_omomo .venv/bin/python \
  scripts/scan_omomo_object_frames.py --data-root ~/Datasets/OMOMO \
  --output out/opt_omomo/zero_frames.json
```

扫描只读数据，不加载 SMPL-X、不求解。JSON 包含输入 SHA256、train/test 序列数、
每个受影响序列的缺口和处理计划；完整规则及边界由 `tests/test_omomo_validity.py`
覆盖。
