# Robot Model Config Profiles

These JSON files define per-robot configuration for the OmniRetargeting CLI.

Use with:

```bash
.venv/bin/python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1_mesh.json \
  --source-motion /path/to/motion.npz --smplx-model-dir /path/to/smplx/models \
  --terrain /path/to/terrain.obj \
  --output /path/to/output.npz
```

`unitree_g1/unitree_g1_mesh.json` is the default and the collision-model
contract. The other profiles are:

| Profile | Use |
| - | - |
| `unitree_g1/unitree_g1_mesh.json` | default; original-mesh convex-hull collision, same model for solve, audit and training |
| `unitree_g1/unitree_g1_lafan1.json` | LAFAN1 BVH bone names and T-pose |
| `unitree_g1/unitree_g1_instinctmj.json` | InstinctMJ popsicle model (needs `OMNIRETARGETING_ASSETS` or a local InstinctMJ checkout) |
| `unitree_g1/unitree_g1.json` | legacy popsicle URDF; its hand capsules are larger than the palm |
| `unitree_h1/unitree_h1.json`, `booster_k1/booster_k1.json`, `hightorque_mini_pi_plus/hightorque_mini_pi_plus.json` | starter profiles: model and SMPL-X joint mapping only, motion quality not validated |

Notes:
- Set `urdf_path` in the profile JSON; the CLI does not take a `--urdf` flag.
- If `--mapping` is provided, it overrides `joint_mapping` from the profile.
- Link names in `joint_mapping` must match body names in your URDF/MuJoCo model.
- `unitree_h1/unitree_h1.json` is a starter profile and may need link-name adjustments for your specific URDF variant.
- Contact, smoothing, and posture defaults are shared by all profiles. A new robot needs its model path, joint mapping, and any required calibration or end-effector geometry. Do not copy old `foot_stabilization` or other unused solver keys into a new profile.
- The G1 profiles use shared solver defaults. Their previous waist/shoulder/elbow/wrist step limits, arm temporal weights, zero-angle arm posture targets, and swing-leg step limits have been removed. Full-sequence TRUMANS solves use the source-shaped pose and correction continuity budgets. Framewise motion quality after this profile change still needs a motion-level comparison.
- The InstinctMJ G1 `body_orientation.reference_pose` is a sparse declared bind pose in radians. Unlisted joints use the robot model's `qpos0`; its listed shoulder yaw values explicitly fix the otherwise unobservable upper-arm twist to zero. Keep the model's home pose and this calibration together when changing the robot asset.
- `unitree_g1/unitree_g1_mesh.json` is the G1 profile used by the fast-solver benchmarks (`scripts/benchmark_fast.py`). It is the InstinctMJ torso-base profile with `g1_29dof_torsobase_mesh.xml`, whose collision geoms are one convex hull per original visual STL (native CCD disabled). Regenerate the model with `scripts/build_g1_mesh_collision_model.py`; it must reproduce the same SHA-256 unless the collision model is meant to change. Its meshes are the repository `meshes/` copies; InstinctMJ's differing head STL is vendored as `meshes/head_link_torsobase.STL`, so the model loads from a plain checkout.
- LAFAN1 BVH motions use `unitree_g1/unitree_g1_lafan1.json`. Its joint names and declared T-pose match the adapter's fixed anatomical reference; hand orientation tasks use the final wrist yaw bodies so all three wrist axes follow the recorded rotations. The SMPL-X calibration in the InstinctMJ profile describes a different reference pose.
