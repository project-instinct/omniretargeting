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
- Contact, smoothing, and posture defaults are shared by all profiles. A new robot needs its model path, joint mapping, and any required calibration or end-effector geometry. Do not copy old `foot_stabilization` or other unused solver keys into a new profile.
- The G1 profiles use shared solver defaults. Their previous waist/shoulder/elbow/wrist step limits, arm temporal weights, zero-angle arm posture targets, and swing-leg step limits have been removed. Full-sequence TRUMANS solves use the source-shaped pose and correction continuity budgets. Framewise motion quality after this profile change still needs a motion-level comparison.
- The InstinctMJ G1 `body_orientation.reference_pose` is a sparse declared bind pose in radians. Unlisted joints use the robot model's `qpos0`; its listed shoulder yaw values explicitly fix the otherwise unobservable upper-arm twist to zero. Keep the model's home pose and this calibration together when changing the robot asset.
- LAFAN1 BVH motions use `unitree_g1/unitree_g1_lafan1.json`. Its joint names and declared T-pose match the adapter's fixed anatomical reference; hand orientation tasks use the final wrist yaw bodies so all three wrist axes follow the recorded rotations. The SMPL-X calibration in the InstinctMJ profile describes a different reference pose.
