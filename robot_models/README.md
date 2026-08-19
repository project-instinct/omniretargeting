# Robot Model Config Profiles

These JSON files define per-robot configuration for the OmniRetargeting CLI.

Use with:

```bash
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --smplx_model_dir /path/to/smplx/models \
  --smplx_motion /path/to/motion.npz \
  --terrain /path/to/terrain.obj \
  --output /path/to/output.npz
```

Notes:
- Set `urdf_path` in the profile JSON; the CLI does not take a `--urdf` flag.
- If `--mapping` is provided, it overrides `joint_mapping` from the profile.
- Link names in `joint_mapping` must match body names in your URDF/MuJoCo model.
- `unitree_h1/unitree_h1.json` is a starter profile and may need link-name adjustments for your specific URDF variant.
