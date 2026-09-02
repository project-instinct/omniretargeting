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

The Playback tab supports exact-frame inspection with First/Prev/Next/Last,
direct frame entry, configurable multi-frame jumps, contact-event navigation,
playback speed, looping, and overlay visibility. Keyboard shortcuts are
`Space` (play/pause), `Left`/`Right` (one frame), `PageUp`/`PageDown` (jump),
and `Home`/`End` (first/last frame).

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

OmniRetargeting adapts the interaction mesh retargeting approach from the holosoma_retargeting project to work with generic robots and terrains:

1. **Terrain Scaling**: Automatically scales terrain mesh to match SMPLX motion scale
2. **Generic Robot Support**: Works with any URDF through automatic model loading and analysis
3. **Interaction Mesh**: Creates tetrahedral mesh from human joints and terrain points
4. **Optimization**: Uses SQP optimization with Laplacian deformation constraints
5. **Collision Avoidance**: Terrain penetration constraints (not yet implemented; see TODO below)
6. **Joint Limits**: Respects robot joint limits during optimization

### Contact-priority hand/foot retargeting

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
