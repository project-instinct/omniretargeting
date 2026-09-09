# SMPL-X frame IDs 与已发布时间轴

`SmplxDataSource` 的 `frame_id_semantics` 明确区分缺帧索引与来源索引：

- `timeline`（默认）：保留现有行为，在 FK 前按 `frame_ids` 补回可插值的缺帧。
- `provenance`（显式可选）：每一行已经是发布帧率下的输出帧；保留所有动作行、
  dtype、顺序和帧率，不因来源编号有空隙而补帧。原编号复制到
  `MotionData.metadata.original_frame_ids`，不转换成输出时间戳。

例如 HER 的旧输出在 30 fps 推理后发布 25 fps 动作，姿态与 `frame_ids` 同时通过
最近行重采样。此时编号的跳跃是来源映射，不代表发布视频缺了帧。直接 densify
这些编号再按 25 fps 播放，会改变时长。需要先核对发布器的重采样规则、帧数、
帧率、观察覆盖率；不能仅看到编号有缺口就自动选择 `provenance`。

原始 SMPL-X source config 示例：

```json
{
  "type": "smplx",
  "motion": "/path/to/published/smplx_amass.npz",
  "adapter_options": {
    "model_directory": "/path/to/body_models",
    "use_smplx_base_pose": true,
    "frame_id_semantics": "provenance"
  }
}
```

`provenance` 要求有限的正帧率；如存在 `frame_ids`，必须一行对应一个编号。
显式 `person_timeline_sparse=True` 会报错，避免把真实的稀疏观察压缩为连续运动。
该选项不修改源 NPZ、姿态、注册、尺度、接触目标或机器人求解策略；预计算关节
输入不经过原始 SMPL-X 的补帧路径。

2026-09-08 的六段 `HER/results/c167*` 旧发布结果均已逐项核对：原编号恰好等于
发布器的 `rint(linspace(0, inference_count-1, output_count))` 映射，发布行数等于
`source_video_frame_count`，观察覆盖率为 1。仅这些已核对输入的运行配置显式
选择 `provenance`。原编号、源码位置、原质量状态及输入位置记录在
`out/her_c167_20260908/input_manifest.json`；最终输出 sidecar 同目录保留此映射。

验证命令：

```sh
.venv/bin/pytest -q tests/data_sources/test_smplx.py tests/test_contact_priority.py
```

回归覆盖原始发布行不变、默认补帧兼容、无效帧率、稀疏观察冲突和编号长度错误。
