# SMPL-X frame IDs

`SmplxDataSource` has two explicit `frame_id_semantics` modes. Do not infer the
mode from the presence of gaps alone.

| Mode | Meaning |
| --- | --- |
| `timeline` (default) | `frame_ids` addresses rows in the source timeline. Missing IDs represent missing frames and may be interpolated before FK. |
| `provenance` | Every row is already an emitted output frame. IDs are labels only: gaps do not imply missing output frames, and rows, order, dtype, and framerate are preserved. Original labels are copied to `MotionData.metadata.original_frame_ids`. |

Use `provenance` only after checking the publisher's resampling rule, output
frame count, framerate, and observation coverage. Republishing a 30 Hz
inference result at 25 Hz can create sparse labels that are not sparse time.

```json
{
  "type": "smplx",
  "motion": "/path/to/published.npz",
  "adapter_options": {
    "model_directory": "/path/to/body_models",
    "use_smplx_base_pose": true,
    "frame_id_semantics": "provenance"
  }
}
```

`provenance` requires a finite positive framerate. If `frame_ids` exists, it
must contain exactly one ID per row. Combining it with
`person_timeline_sparse=true` is an error because that would compress genuinely
sparse observations into continuous motion.

```sh
.venv/bin/python -m pytest -q tests/data_sources/test_smplx.py tests/test_contact_priority.py
```
