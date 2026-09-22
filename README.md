# OmniRetargeting

**Generic motion retargeting for any humanoid URDF and terrain mesh.**

This is a re-implementation of the [OmniRetarget](https://arxiv.org/abs/2509.26633) method. OmniRetargeting is a flexible motion retargeting system that can convert human motion trajectories (in SMPLX format) to any humanoid robot operating on any terrain mesh. Unlike specialized retargeting systems, OmniRetargeting automatically adapts to different robot morphologies and terrain types.

## Installation

```bash
pip install omniretargeting
```

Or install from source:

```bash
git clone <repository-url>
cd omniretargeting
pip install -e .
```

For development with testing:

```bash
pip install -e ".[dev,test]"
```

## Quick Start

```python
from omniretargeting import OmniRetargeter
from omniretargeting.utils import load_smplx_trajectory
from pathlib import Path

# Define your inputs
robot_urdf = "path/to/your/robot.urdf"
terrain_mesh = "path/to/your/terrain.obj"

# Load SMPLX trajectory (positions and orientations)
# For raw SMPLX-NG files (e.g., stageii.npz with poses/trans/betas)
smplx_trajectory, smplx_orientations = load_smplx_trajectory(
    Path("path/to/trajectory_stageii.npz"),
    smplx_model_directory="path/to/smplx/models",  # Required for raw files and orientation computation
    gender="neutral",
)

# OR for pre-processed files (.npy or .npz with global_joint_positions)
# smplx_trajectory, smplx_orientations = load_smplx_trajectory(
#     Path("path/to/trajectory.npy")
# )

print(f"Loaded trajectory shape: {smplx_trajectory.shape}")  # (T, J, 3)
if smplx_orientations is not None:
    print(f"Loaded orientations shape: {smplx_orientations.shape}")  # (T, J, 4) - quaternions (wxyz)
else:
    print("Orientations not available for this file format")

# Define joint mapping from SMPLX to robot
joint_mapping = {
    "root": "pelvis",
    "left_hip": "left_hip_pitch",
    "right_hip": "right_hip_pitch",
    "left_knee": "left_knee_pitch",
    "right_knee": "right_knee_pitch",
    # ... add more mappings as needed
}

# Create retargeter
retargeter = OmniRetargeter(
    robot_urdf_path=robot_urdf,
    terrain_mesh_path=terrain_mesh,
    joint_mapping=joint_mapping
)

# Perform retargeting
terrain_scale, retargeted_motion = retargeter.retarget_motion(
    smplx_trajectory,
    enable_terrain_scaling=True,
)

print(f"Terrain scale factor: {terrain_scale}")
print(f"Retargeted motion shape: {retargeted_motion.shape}")  # (T, 7 + DOF)
```

## Input Format

### SMPLX Trajectory

The trajectory must be provided as a numpy array of shape `(T, J, 3)` where:
- **T**: Number of frames
- **J**: Number of joints (typically 22 for body joints)
- **3**: (x, y, z) coordinates in world frame

#### Loading Trajectory Files

OmniRetargeting supports multiple trajectory file formats through the `load_smplx_trajectory` utility:

**1. Pre-processed files (.npy)**:
```python
from omniretargeting.utils import load_smplx_trajectory
# Returns (positions, None) since orientations cannot be computed from positions alone
trajectory, orientations = load_smplx_trajectory(Path("trajectory.npy"))
# orientations will be None
```

**2. Pre-processed files (.npz with 'global_joint_positions')**:
```python
# Returns (positions, orientations) if full_pose data is available
trajectory, orientations = load_smplx_trajectory(
    Path("trajectory.npz"),
    smplx_model_directory="/path/to/smplx/models",  # Needed for parent structure
)
# Looks for 'global_joint_positions' key for positions
# Looks for 'full_pose' and 'root_orient' keys for orientations
```

**3. Raw SMPLX-NG files (stageii.npz)**:

Raw SMPLX-NG files contain SMPLX parameters, not joint positions. Keys include:
- `'gender'`, `'surface_model_type'`, `'mocap_frame_rate'`
- `'trans'`, `'poses'`, `'betas'`
- `'root_orient'`, `'pose_body'`, `'pose_hand'`, `'pose_jaw'`, `'pose_eye'`

To load these, you **must** provide the SMPLX model path:

```python
trajectory, orientations = load_smplx_trajectory(
    Path("HumanEva_S3_Jog_1_stageii.npz"),
    smplx_model_directory="/path/to/smplx/models",  # Required!
    gender="neutral",  # Auto-detected from file if present
)
```

The function **always returns a tuple** `(positions, orientations)`:
1. Load SMPLX parameters from the file
2. Run forward kinematics using the SMPLX model
3. Return joint positions in shape `(T, 22, 3)` and orientations in shape `(T, 22, 4)`

**Orientation Format:**
- Orientations are returned as quaternions in **wxyz** format (scalar-first)
- Shape: `(T, J, 4)` where the last dimension is `[w, x, y, z]`
- Represents the world-frame orientation of each joint
- Will be `None` if orientations cannot be computed (e.g., .npy files with positions only)

### Joint Mapping
A dictionary mapping SMPLX joint names to robot link names:

```python
joint_mapping = {
    "root": "pelvis",           # Root joint
    "left_hip": "left_hip_pitch",
    "right_hip": "right_hip_pitch",
    "left_knee": "left_knee_pitch",
    "right_knee": "right_knee_pitch",
    "left_ankle": "left_ankle_pitch",
    "right_ankle": "right_ankle_pitch",
    # Add more joints as available
}
```

### Terrain Mesh
Supports common mesh formats:
- `.obj` (Wavefront OBJ)
- `.stl` (STL mesh)
- `.ply` (Polygon File Format)
- `.gltf`/`.glb` (glTF)

1. **Optional Terrain Scaling**: The terrain mesh remains unscaled by default. If `--output-scaled-terrain` is provided, OmniRetargeting computes a terrain scale factor, retargets against the scaled terrain mesh, and writes that scaled mesh to disk.

### Robot URDF
Standard URDF format for humanoid robots. The system automatically:
- Detects robot height and dimensions
- Identifies joint limits and types
- Sets up collision detection

## Output Format

The `retarget_motion()` method returns a tuple:

```python
terrain_scale, retargeted_motion = retargeter.retarget_motion(
    smplx_trajectory,
    enable_terrain_scaling=True,
)
```

- **`terrain_scale`**: `1.0` by default, or the computed terrain scaling factor when terrain scaling is enabled
- **`retargeted_motion`**: Numpy array of shape `(T, 7 + DOF)` containing:
  - `[0:3]`: Root position (x, y, z)
  - `[3:7]`: Root quaternion (w, x, y, z)
  - `[7:]`: Joint angles in radians

## Advanced Usage

### Custom Robot Height

```python
retargeter = OmniRetargeter(
    robot_urdf_path=robot_urdf,
    terrain_mesh_path=terrain_mesh,
    joint_mapping=joint_mapping,
    robot_height=1.8  # Override auto-detected height
)
```

### Robot Profile Config (Per-Humanoid)

You can keep one JSON profile per humanoid robot (for example under `robot_models/<robot_name>/`) and feed it to the CLI:

```bash
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --smplx_model_dir /path/to/smplx/models \
  --smplx_motion /path/to/motion.npz \
  --terrain /path/to/terrain.obj \
  --output /path/to/output.npz \
  --output-scaled-terrain /path/to/scaled-terrain.obj
```

`--robot-config` defaults to `robot_models/unitree_g1/unitree_g1.json`. The robot URDF path must be set as `urdf_path` in that JSON profile (the CLI does not accept a separate URDF argument).

Dataset clips can be passed directly; a separate source JSON is unnecessary
for common unambiguous formats:

```bash
python -m omniretargeting.main \
  --source-motion /path/to/parc_ms_clip.pkl \
  --robot-config robot_models/unitree_g1/unitree_g1_instinctmj.json \
  --output /path/to/retargeted.npz
```

The CLI infers `.pkl` as `parc_ms` and `.npz` as `smplx`. For an ambiguous
format such as BVH, add one adapter name, for example
`--source-type lafan1`; source JSON remains available only for real per-source
overrides.

Profile fields can include:
- `joint_mapping`
- `urdf_path`
- `robot_height`
- `smplx_joint_names`
- `height_estimation`
- `base_orientation`
- `retargeting`

Contact, temporal stability, posture preservation, and trajectory refinement
use algorithm-level defaults. A robot profile only describes its own mapping,
end-effector geometry, and genuine model-specific overrides:

```json
{
  "retargeting": {
    "end_effector_priority": {
      "effectors": {
        "left_hand": {"robot_body": "left_hand_link"},
        "right_hand": {"robot_body": "right_hand_link"}
      }
    }
  }
}
```

Declaring interaction effectors automatically enables the shared policy; there
is no preset selector or external algorithm-parameter JSON. Nested
robot-specific objects are merged recursively, while a scalar or list in the
robot profile replaces the algorithm default. The bundled
`unitree_g1_instinctmj.json` uses this compact form.

Robot self-collision avoidance is enabled by default, independently of the
scene penetration resolver. It checks only collision-enabled geoms, respects
model collision filters, and checks the paths between saved poses. Visual-only
geoms are ignored. Set `retargeting.self_collision_enabled` to `false` to opt
out; an old saved profile that explicitly sets it to `false` keeps that choice.
The output diagnostics report self-collision separately from scene penetration.
See [self-collision implementation and validation](docs/self_collision.md).

Palm and sole tracking/contact offsets are inferred from the robot's mesh and
collision geometry when omitted. Profiles only need explicit offsets when the
model has no usable end-effector geometry or intentionally uses a custom tool
point.

Arm joint chains, shoulder/elbow/wrist bodies, and upper-arm/forearm lengths
are likewise inferred from the URDF/MJCF kinematic tree and nominal pose.
Wrist semantic axes come from its roll/pitch joint axes, while axial waist or
torso joints receive an automatic neutral safety window inside their model
limits to prevent whole-body twist drift.

Free-swing targets use the measured robot-root/source-torso registration per
frame. If a source omits the requested upper-spine joint, the highest available
spine joint is selected automatically instead of falling back to an unrelated
array index.

### Validation

```python
# Check if joint mapping is valid
missing_joints = retargeter.validate_joint_mapping()
if missing_joints:
    print(f"Warning: Missing joints: {missing_joints}")

# Get robot information
print(f"Robot DOF: {retargeter.get_robot_dof()}")
print(f"Joint names: {retargeter.get_joint_names()}")
```

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Simple API demonstration and trajectory visualization
- `advanced_usage.py`: Advanced features, validation, and motion comparisons

Run examples with:

```bash
python examples/basic_usage.py
python examples/advanced_usage.py
```

## Running Tests

```bash
pytest tests/
```

## Viewing Retargeted Output with mjviser

Install the package after syncing dependencies, then open any retargeted output:

```bash
omniretargeting-mjviser /path/to/retargeted_motion.npz \
  --robot-config robot_models/unitree_g1/unitree_g1_instinctmj.json \
  --terrain /path/to/scene_tsdf.obj
```

The viewer reads `base_pos_w`, `base_quat_w`, `joint_pos`, and `joint_names`
from `.npz` outputs. If `object_pos_w/object_quat_w` are present, it can also
show an object mesh via `--object-mesh`, or infer common OMOMO meshes from
`~/Datasets/OMOMO/data/captured_objects`.

To restrict both live and cached servers to this machine, add `--host 127.0.0.1`.

The Playback tab supports exact-frame inspection with First/Prev/Next/Last,
direct frame entry, configurable multi-frame jumps, contact-event navigation,
playback speed, looping, and overlay visibility. Keyboard shortcuts are
`Space` (play/pause), `Left`/`Right` (one frame), `PageUp`/`PageDown` (jump),
and `Home`/`End` (first/last frame).

For smooth viewing of a saved motion over a remote connection, also export and
serve a **browser-cached Viser playback**:

```bash
omniretargeting-mjviser /path/to/retargeted_motion.npz \
  --robot-config robot_models/unitree_g1/unitree_g1_instinctmj.json \
  --terrain /path/to/scene.obj --port 8130 \
  --playback-dir out/cached_playback --playback-port 8131
```

Open port **8131** for cached playback. This reuses Viser's official `.viser`
recording format and the frontend installed with the same Viser version. The
browser downloads the recording with a progress indicator, then plays locally;
camera orbit, pause, speed, and timeline controls remain interactive. This is
download-before-play caching, not progressive playback or a WebRTC video stream.
Each browser has its own playback clock. Port **8130** retains the live viewer's
exact source-frame controls and overlay toggles; live clients share controls.

The recording contains the original scene once, all source timestamps, and
MuJoCo-interpolated display samples at `--refresh-hz` (60 by default). It does
not filter or rewrite the motion. The last pose is held for one source frame
before looping, without blending the end into the start. Preparing the complete
cache uses memory proportional to clip length; use live mode for very long clips.
Cached playback uses the stock Viser controls and loops by default; live-only
`--paused`, `--speed`, `--start-frame`, and `--no-loop` do not configure that player.

Live mode uses monotonic deadline scheduling, skips missed display refreshes,
groups robot and source overlays in one Viser atomic update, and updates timeline
text at 10 Hz. It stops rendering when no clients are connected. Singleton robot
mesh instances use Viser's direct node pose
updates instead of changing React-managed batch arrays; genuine batches retain
mjviser's original path, and all geometry, LOD, and transforms are preserved. Shadows
are off by default; `--shadows` restores them. `--playback-stats` logs server
submission rate and processing cost, which are not browser render FPS. Viser
coalesces pending updates, but live mode does not implement browser acknowledgments
or a bound on data already queued in the network/browser.

Use `--playback-dir out/cached_playback --export-only` to export and exit. The
resulting directory can be served by an ordinary HTTP server; it includes the
version-matched frontend and a manifest. No external CDN is needed.

## API Reference

### `OmniRetargeter`

Main class for motion retargeting.

#### Constructor
```python
OmniRetargeter(robot_urdf_path, terrain_mesh_path, joint_mapping, robot_height=None)
```

#### Methods

- `retarget_motion(smplx_trajectory, terrain_coordinates=None)` → `(terrain_scale, retargeted_motion)`
- `get_robot_dof()` → `int`
- `get_joint_names()` → `List[str]`
- `validate_joint_mapping()` → `List[str]`

## Dependencies

- numpy
- torch
- scipy
- matplotlib
- trimesh
- smplx
- mujoco
- viser
- yourdfpy
- cvxpy
- libigl
- open3d
- pyvista

## Architecture

For implementation entry points and module responsibilities, see the
[code layout guide](docs/code_layout.md).

OmniRetargeting adapts the interaction mesh retargeting approach from the holosoma_retargeting project to work with generic robots and terrains:

1. **Terrain Scaling**: Automatically scales terrain mesh to match SMPLX motion scale
2. **Generic Robot Support**: Works with any URDF through automatic model loading and analysis
3. **Interaction Mesh**: Creates tetrahedral mesh from human joints and terrain points
4. **Optimization**: Uses SQP optimization with Laplacian deformation constraints
5. **Collision Avoidance**: Terrain penetration constraints (not yet implemented; see TODO below)
6. **Joint Limits**: Respects robot joint limits during optimization

### Contact-priority hand/foot retargeting

For whole-body scene interaction, raw SMPL-X sources also provide persistent
skin samples. `retargeting.surface_contact.enabled` defaults to true and learns
indexed human/robot reference-surface correspondence, following UMR's contact-map
formulation. Active forearm, torso, pelvis and other surface contacts share the
framewise solver, refinement and final validation; existing hand/foot contacts
retain priority. Source motion, robot size and scene scale remain unchanged.
See [whole-body surface contact](docs/whole_body_surface_contact.md) for configuration,
exported `surface_contact_*` arrays, correspondence diagnostics and limitations.

`retargeting.whole_body_correction.enabled` also defaults to true. The final
coupled correction lets pelvis/waist/leg joints participate alongside hand/foot
compensation, with a fixed floating base and fixed endpoint error budgets.
Prescribed moving-object geometry contributes collision constraints at frames
and between frames; open mesh components remain explicitly unknown. See
[constrained whole-body correction](docs/constrained_whole_body_correction.md)
for the UMR method mapping, configuration and validation coverage.

Robot profiles can enable `retargeting.end_effector_priority`. For raw SMPL-X
motions, the loader samples stable hand and foot surface vertices, detects
scene contact from distance and relative speed, and anchors every contiguous
contact interval to the scene mesh.

At each SQP iteration, a marker residual is linearized as
`r_e(dq) = p_e(q) + J_e(q)dq - p_e*`. The solver then performs three convex
problems in lexicographic order:

1. minimize the normalized active-contact residual norm;
2. constrain that norm to `(1 + relative_tolerance) * optimum + absolute_tolerance`
   and minimize the non-contact hand/foot residual norm;
3. constrain both higher-priority levels and minimize the whole-body
   interaction-mesh, smoothness, orientation, and regularization objective.

Unlike a large finite loss weight, these explicit optimum bounds preserve the
declared priority up to the configured numerical tolerances. The generated
`.npz` additionally stores `contact_names`, `contact_mask`,
`contact_confidence`, `contact_targets_w`, and `end_effector_reference_w` for
downstream robot policy training or evaluation.

Fixed per-frame joint velocity, acceleration, and jerk caps have been removed
so deliberate fast source motion can pass through. Neither the framewise
optimization nor its output applies these caps; old configuration keys cannot
re-enable them.
Mechanical joint ranges, scene/contact constraints, and the per-iteration
optimization trust region remain active. Source-relative base-motion bounds
and soft smoothing are configured separately.

Whole-trajectory refinement in the main pipeline uses the same immutable
framewise reference and budgets as final acceptance. Its CLARABEL subproblem
bounds cumulative joint corrections, added correction velocity/acceleration,
end-effector displacement, contact residuals, and reliable scene penetration.
Each retained update also passes real FK and dense/interframe geometry checks.
These bounds constrain **added corrections**, not the speed of the source motion.
`track_swing: false` disables only free-effector position rows; contact, arm
posture, and orientation rows remain active. Optional smoothing that violates
these budgets leaves the validated refinement intact. Legacy ground-lock/XYZ
foot stabilization is skipped in this fixed-base pipeline; remaining penetration
is handled by local joint repair and reported independently.

The default joint range is the full mechanical range. Core joints no longer
receive automatic extra range reductions or name-based neutral-pose penalties;
explicit profile overrides remain supported. For sources with an authored bind
pose, orientation calibration preserves the model's neutral core relationship
and the declared base reference orientation. A positional fit across different
body proportions therefore cannot bake an incidental waist twist into every
frame's orientation target. First-frame fallback calibration keeps its existing
semantics when the source has no authored bind pose.

## TODO / Limitations

- **Penetration constraint**: The terrain penetration constraint is not yet implemented. The optimization framework has the `_compute_penetration_constraints` method and collision detection scaffolding, but they are currently disabled (`retargeting.py`, lines 520–524) because collision detection setup is incomplete. As a result, retargeted motion may exhibit foot–terrain penetration in some cases.
- **Scaled terrain export**: The current CLI saves retargeted motion to `.npz` but does not persist the scaled terrain mesh to disk. Future work: add an optional argument (for example `--output_scaled_terrain`) to export the scaled terrain mesh used during retargeting.
- Other known TODOs: proper coordinate system alignment (`utils.py`).

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the Apache 2.0 License.

## Citation

This repository is a re-implementation of the OmniRetarget method. If you use this code in your research, please cite the original paper:

```
@article{yang2025omniretarget,
  title={OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction},
  author={Yang, Lujie and Huang, Xiaoyu and Wu, Zhen and Kanazawa, Angjoo and Abbeel, Pieter and Sferrazza, Carmelo and Liu, C. Karen and Duan, Rocky and Shi, Guanya},
  journal={arXiv preprint arXiv:2509.26633},
  year={2025},
  url={https://arxiv.org/abs/2509.26633}
}
```
