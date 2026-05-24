# OmniRetargeting

**Generic motion retargeting for any humanoid URDF and terrain mesh.**

This is a re-implementation of the [OmniRetarget](https://arxiv.org/abs/2509.26633) method. OmniRetargeting is a flexible motion retargeting system that converts ordered human/source target positions to any humanoid robot operating on any terrain mesh. Unlike specialized retargeting systems, OmniRetargeting automatically adapts to different robot morphologies and terrain types.



## Source-Agnostic Architecture

OmniRetargeting uses a source-agnostic architecture that supports multiple motion data formats through a registry system.

### Supported Source Types

- **SMPL-X**: Human body model motion data
- **Custom sources**: Easily add new source adapters

### Using Different Sources

```python
from omniretargeting import OmniRetargeter
from omniretargeting.data_sources import create_data_source

# Create a data source (automatically uses registry)
data_source = create_data_source(
    source_type="smplx",
    motion_file="path/to/motion.npz",
    source_config={"model_directory": "/path/to/models"}
)

# Create retargeter
retargeter = OmniRetargeter(
    robot_urdf_path="robot.urdf",
    terrain_mesh_path="terrain.obj",
    joint_mapping={"Pelvis": "torso_link", ...},
    robot_height=1.6
)

# Retarget motion (batch mode)
scale, robot_motion = retargeter.retarget_motion(data_source)

# Or stream mode for frame-by-frame processing
for robot_frame in retargeter.retarget_stream(data_source):
    # Process each frame
    pass
```

### Adding New Source Adapters

See `docs/ADDING_SOURCE_ADAPTERS.md` for a guide on implementing new source adapters.


## Installation

install from source:

```bash
git clone <https://github.com/project-instinct/omniretargeting>
cd omniretargeting
pip install -e .
```

For development with testing:

```bash
pip install -e ".[dev,test]"
```

## Quick Start

```python
from omniretargeting import OmniRetargeter, load_robot_config
from omniretargeting.data_sources.smplx import SmplxDataSource
from pathlib import Path

# Load a robot profile. The default profiles currently use SMPL-X target names.
cfg = load_robot_config("robot_models/unitree_g1/unitree_g1.json")

# Load source motion as MotionData. SMPL-X is the implemented adapter today.
source = SmplxDataSource(
    motion_file=Path("path/to/motion_stageii.npz"),
    model_directory="path/to/smplx/models",
    gender="neutral",
    target_names_override=cfg.get("source_target_names"),
    betas=cfg.get("smplx_betas"),
)
motion = source.load()

retargeter = OmniRetargeter(
    robot_urdf_path=cfg["urdf_path"],
    terrain_mesh_path="path/to/terrain.obj",
    joint_mapping=cfg["joint_mapping"],
    robot_height=cfg.get("robot_height"),
    source_target_names=motion.target_names,
    height_estimation=cfg.get("height_estimation"),
    base_orientation=cfg.get("base_orientation"),
    retargeting=cfg.get("retargeting"),
    link_offset_config=cfg.get("link_offset_config"),
)

source_to_robot_scale, retargeted_motion = retargeter.retarget_motion(
    motion,
    enable_terrain_scaling=True,
    visualize_trajectory=False,
)

print(f"Source-to-robot scale factor: {source_to_robot_scale}")
print(f"Retargeted motion shape: {retargeted_motion.shape}")  # (T, 7 + DOF)
```

For a ready-to-run setup, omniretargeting ships with robot profiles under
`robot_models/`. These profiles contain robot assets, target-to-link mappings,
height/orientation helpers, link offsets, and retargeting settings.

```python
from omniretargeting import OmniRetargeter, load_robot_config

cfg = load_robot_config("robot_models/unitree_g1/unitree_g1.json")
retargeter = OmniRetargeter(
    robot_urdf_path=cfg["urdf_path"],
    terrain_mesh_path="path/to/terrain.obj",
    joint_mapping=cfg["joint_mapping"],
    robot_height=cfg.get("robot_height"),
    source_target_names=cfg.get("source_target_names"),
    height_estimation=cfg.get("height_estimation"),
    base_orientation=cfg.get("base_orientation"),
    retargeting=cfg.get("retargeting"),
    link_offset_config=cfg.get("link_offset_config"),
)
```

## Input Format

### Source Motion

The retargeting core consumes either a `MotionData` object, a `DataSource`, or a numpy array of target positions with shape `(T, J, 3)`:
- **T**: Number of frames
- **J**: Number of ordered source targets
- **3**: (x, y, z) coordinates in world frame

`MotionData` can also carry `target_names`, optional root orientations/translations, framerate, source height, and source-specific metadata:

```python
from omniretargeting import MotionData

motion = MotionData(
    positions=positions,              # (T, J, 3)
    target_names=["Pelvis", "Head"],  # optional but recommended
    framerate=30.0,
    human_height=1.72,
)
```

### SMPL-X Data Source

SMPL-X is currently the implemented source adapter. It returns `MotionData` through `SmplxDataSource.load()` and can read:

**1. Pre-processed files (.npy)**:
```python
from omniretargeting.data_sources.smplx import SmplxDataSource

motion = SmplxDataSource(motion_file=Path("trajectory.npy")).load()
```

**2. Pre-processed files (.npz with 'global_joint_positions')**:
```python
motion = SmplxDataSource(
    motion_file=Path("trajectory.npz"),
    model_directory="/path/to/smplx/models",
).load()
# Looks for 'global_joint_positions' key for positions
# Looks for 'full_pose' and 'root_orient' keys for orientations
```

**3. Raw SMPL-X-NG files (stageii.npz)**:

Raw SMPL-X-NG files contain SMPL-X parameters, not joint positions. Keys include:
- `'gender'`, `'surface_model_type'`, `'mocap_frame_rate'`
- `'trans'`, `'poses'`, `'betas'`
- `'root_orient'`, `'pose_body'`, `'pose_hand'`, `'pose_jaw'`, `'pose_eye'`

To load these, provide the SMPL-X model path:

```python
motion = SmplxDataSource(
    motion_file=Path("HumanEva_S3_Jog_1_stageii.npz"),
    model_directory="/path/to/smplx/models",
    gender="neutral",
).load()
```

For compatibility, `omniretargeting.utils.load_smplx_trajectory()` still returns `(positions, orientations)`; new code should prefer `SmplxDataSource` and `MotionData`.

### Joint Mapping
A dictionary mapping source target names or IDs (keys) to robot **body/link** names (values) as they appear in the URDF:

```python
joint_mapping = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip_roll_link",
    "R_Hip": "right_hip_roll_link",
    "Spine1": "waist_yaw_link",
    "L_Knee": "left_knee_link",
    "R_Knee": "right_knee_link",
    "L_Ankle": "left_ankle_roll_link",
    "R_Ankle": "right_ankle_roll_link",
    "L_Shoulder": "left_shoulder_roll_link",
    "R_Shoulder": "right_shoulder_roll_link",
    "L_Elbow": "left_elbow_link",
    "R_Elbow": "right_elbow_link",
    "L_Wrist": "left_wrist_yaw_link",
    "R_Wrist": "right_wrist_yaw_link",
}
```

For the current SMPL-X adapter, the default target ordering is:

```
Pelvis, L_Hip, R_Hip, Spine1, L_Knee, R_Knee, Spine2, L_Ankle, R_Ankle,
Spine3, L_Foot, R_Foot, Neck, L_Collar, R_Collar, Head, L_Shoulder,
R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist
```

Pass the corresponding order to `OmniRetargeter(source_target_names=...)`. Any key in `joint_mapping` must be present in `source_target_names`, and each mapped value must match a body name in the robot URDF; unresolved robot body entries are filtered out with a warning at initialization.

### Terrain Mesh
Supports common mesh formats:
- `.obj` (Wavefront OBJ)
- `.stl` (STL mesh)
- `.ply` (Polygon File Format)
- `.gltf`/`.glb` (glTF)

**Optional Terrain Scaling**: the terrain mesh is unscaled by default. If `enable_terrain_scaling=True` is passed to `retarget_motion()` (or `--output-scaled-terrain` is set on the CLI), OmniRetargeting computes a source-to-robot scale factor from the robot/source height ratio and retargets against the scaled source motion and scaled mesh.

### Robot URDF
Standard URDF format for humanoid robots. The system automatically:
- Detects robot height from the default pose (overridable via `robot_height`)
- Reads joint limits and types from the URDF
- Loads visual meshes for (optional) visualization

## Output Format

### `retarget_motion()` return value

```python
source_to_robot_scale, retargeted_motion = retargeter.retarget_motion(
    motion,
    framerate=30.0,
    enable_scene_scaling=True,
)
```

- **`source_to_robot_scale`**: `1.0` by default, or the computed robot/source height ratio when `enable_scene_scaling=True`.
- **`retargeted_motion`**: Numpy array of shape `(T, 7 + DOF)` containing:
  - `[0:3]`: Root position (x, y, z)
  - `[3:7]`: Root quaternion in **wxyz** order (MuJoCo convention)
  - `[7:]`: Joint angles in radians

### CLI `.npz` schema

`python -m omniretargeting.main --output my_motion.npz ...` writes a `.npz`
containing the following keys (the output filename is also normalized to end
with `_retargeted.npz` if it doesn't already):

| Key            | Shape      | Description                                     |
|----------------|------------|-------------------------------------------------|
| `framerate`    | scalar     | Motion framerate (from file or `--framerate`).  |
| `joint_names`  | `(DOF,)`   | Robot joint names (excluding the floating base). |
| `joint_pos`    | `(T, DOF)` | Joint angles in radians.                         |
| `base_pos_w`   | `(T, 3)`   | Root position in world frame.                    |
| `base_quat_w`  | `(T, 4)`   | Root quaternion in world frame (wxyz).           |

If `--output-scaled-terrain` is provided, the scaled terrain mesh used for
retargeting is exported to that path as well. If `--scaled-objects DIR` is
provided and the source adapter exposes an object mesh, the CLI also exports a
scaled object mesh plus per-frame object poses into that directory.

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

### CLI

The CLI is driven by a per-robot JSON profile. The URDF path, joint mapping,
and retargeting settings all come from the profile — the CLI does **not**
accept a separate URDF argument.

#### Recommended: YAML source configs

New workflows should use `--source-config`, which moves source-specific options
into a YAML file instead of requiring many CLI flags.

**SMPL-X example**

```bash
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --source-config config_templates/smplx_template.yaml \
  --terrain /path/to/terrain.obj \
  --output /path/to/output.npz \
  --output-scaled-terrain /path/to/scaled-terrain.obj \
  --framerate 30 \
  --penetration-resolver xyz_nudge
```

**OMOMO / object-interaction example**

```bash
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --source-config config_templates/omomo_floorlamp_example.yaml \
  --terrain /path/to/terrain.obj \
  --output /path/to/output.npz \
  --output-scaled-terrain /path/to/scaled-terrain.obj \
  --scaled-objects /path/to/scaled-objects \
  --save-video /path/to/output.mp4
```

#### Legacy CLI compatibility

The legacy flags still work for existing scripts and tests, but they now emit
DeprecationWarnings and should be migrated to `--source-config` over time.

```bash
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --motion /path/to/motion_stageii.npz \
  --model-dir /path/to/smplx/models \
  --terrain /path/to/terrain.obj \
  --output /path/to/output.npz
```

Main arguments:

| Flag | Default | Description |
|---|---|---|
| `--robot-config` | `robot_models/unitree_g1/unitree_g1.json` | Path to robot profile JSON. |
| `--source-config` | `None` | Recommended YAML source configuration file. See `config_templates/`. |
| `--output` | *(required)* | Output `.npz` path (normalized to end in `_retargeted.npz`). |
| `--terrain` | flat ground | Path to terrain mesh; a default flat terrain is generated if omitted. |
| `--output-scaled-terrain` | `None` | Enables scene scaling and exports the scaled terrain mesh. |
| `--scaled-objects` | `None` | Directory for scaled object mesh exports and per-frame object poses when the source adapter provides them. |
| `--framerate` | auto / 30 | Motion framerate; auto-detected from the source file when possible. |
| `--vis` | off | Launch a MuJoCo viewer on the retargeted motion. |
| `--save-video PATH` | off | Render the retargeted motion to video (requires `imageio[ffmpeg]`, and `MUJOCO_GL=egl`/`osmesa` for headless). |
| `--replace-cylinders-with-capsules` | off | Swap cylinder collision geoms for capsules (IsaacLab/PhysX convention). |
| `--penetration-resolver {hard_constraint,xyz_nudge}` | `xyz_nudge` | Contact handling mode; overrides the value in the profile. |

Legacy source-loading flags:

| Flag | Status | Description |
|---|---|---|
| `--source` | deprecated | Source entry name or source type from the robot profile. |
| `--motion` | deprecated | Legacy path to source motion file. |
| `--source-options` | deprecated | Legacy JSON object with adapter-specific options. |
| `--model-dir` | deprecated | Legacy adapter model directory, e.g. SMPL-X model files. |
| `--smplx_motion` | deprecated alias | Legacy alias for `--motion`. |
| `--smplx_model_dir` | deprecated alias | Legacy alias for `--model-dir`. |

### Robot Profile Config (Per-Humanoid)

Keep one JSON profile per humanoid robot (for example under
`robot_models/<robot_name>/`). Relative `urdf_path` values are resolved against
the profile file's directory.

Current shipped profiles use a flat schema:

- `name` – optional profile name, used in log output
- `urdf_path` – **required**, path to the robot URDF (relative to the profile file)
- `joint_mapping` – **required**, source target name → robot body name
- `robot_height` – optional override for auto-detected robot height
- `source_target_names` / `smplx_joint_names` – optional custom source target ordering
- `height_estimation` – source target names and `head_top_offset` used to estimate source height
- `base_orientation` – source target names used to estimate root orientation (`pelvis`, `left_hip`, `right_hip`, `spine`)
- `link_offset_config` – optional robot-link local offsets for mapped link target points
- `retargeting` – solver settings forwarded to `GenericInteractionRetargeter`:
  - `collision_detection_threshold`
  - `terrain_sample_points`
  - `replace_cylinders_with_capsules`
  - `penetration_resolver`: `"hard_constraint"` or `"xyz_nudge"`
  - `foot_stabilization`: nested block (see `robot_models/unitree_g1/unitree_g1.json`) that controls the post-processing XYZ-nudge pass (`enabled`, `clearance`, `surface_clearance`, `contact_clearance`, `xy_correction_gain`, smoothing windows, wall-contact thresholds, etc.)

`load_robot_config()` also accepts the newer nested profile shape with `robot`, `retargeting.solver`, `active_source`, and `source` entries. The loader normalizes both shapes into the same keys used above.

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

## Running Tests

```bash
pytest tests/
```

## API Reference

### `OmniRetargeter`

Main class for motion retargeting (defined in `omniretargeting/core.py`).

#### Constructor
```python
OmniRetargeter(
    robot_urdf_path,
    terrain_mesh_path,
    joint_mapping,
    robot_height=None,
    source_target_names=None,
    height_estimation=None,
    base_orientation=None,
    retargeting=None,
    link_offset_config=None,
)
```

#### Methods

- `retarget_motion(motion, base_orientations=None, base_translations=None, framerate=None, visualize_trajectory=True, enable_terrain_scaling=False)` → `(source_to_robot_scale, retargeted_motion)`
- `get_robot_dof()` → `int`
- `get_joint_names()` → `List[str]`
- `validate_joint_mapping()` → `List[str]` (robot body names from `joint_mapping` that are missing from the URDF)

### `load_robot_config`

```python
from omniretargeting import load_robot_config
cfg = load_robot_config("robot_models/unitree_g1/unitree_g1.json")
```

Loads a robot profile JSON, resolves `urdf_path` relative to the profile file, and normalizes legacy flat and nested profile fields. Raises if no non-empty mapping is available.

### `SmplxDataSource`

```python
from pathlib import Path
from omniretargeting.data_sources.smplx import SmplxDataSource

motion = SmplxDataSource(
    motion_file=Path("motion_stageii.npz"),
    model_directory="/path/to/smplx/models",
    gender="neutral",
).load()
```

Returns `MotionData`. SMPL-X joint orientations, when available, are stored in `motion.metadata["joint_orientations"]` as wxyz quaternions.

## Dependencies

Declared in `pyproject.toml` / `setup.py`:

- numpy, scipy, matplotlib, tqdm
- torch
- trimesh, smplx, jinja2
- mujoco (≥3.7 for URDF `strippath=false` default)
- viser, yourdfpy, robot_descriptions
- cvxpy, libigl, tyro
- open3d, pyvista

## Architecture

OmniRetargeting adapts the interaction-mesh retargeting approach from the
holosoma_retargeting project to work with generic robots and terrains:

1. **Source-to-Robot Scaling** (optional): Computes the robot/source height ratio and uses it to scale source motion and the terrain mesh before retargeting (enabled by `enable_terrain_scaling=True` or `--output-scaled-terrain`).
2. **Generic Robot Support**: Works with any URDF through automatic model loading, body-name validation, and auto-detected height.
3. **Interaction Mesh**: Builds a tetrahedral interaction mesh from mapped source targets and terrain sample points.
4. **Optimization**: Per-frame SQP optimization with Laplacian-deformation objective, joint limits, and a target base-orientation term for smoothness.
5. **Collision / Penetration Handling**: Two modes selectable via `retargeting.penetration_resolver`:
   - `hard_constraint` – penetration inequalities inside the SQP.
   - `xyz_nudge` – post-optimization foot stabilization that projects probe points out of the terrain and smooths XY drift (see `foot_stabilization` in the profile).
6. **Joint Limits**: Respects robot joint limits throughout.

## Limitations

- **Coordinate-system alignment**: Source adapters should provide positions in
  the project world frame. The current SMPL-X adapter assumes trajectories are
  already in a +Z-up world frame.
- **Foot stabilization tuning**: The `xyz_nudge` resolver is effective on flat
  and mildly uneven terrain but may need per-robot tuning
  (`foot_stabilization` block in the profile) for complex scenes with walls.
- **Object interaction (v1)**:
  - Objects are represented as concatenated point clouds `(T, N, 3)` without per-object identity.
  - No explicit robot-object collision constraints (relies on Laplacian preservation only).
  - Non-convex objects sampled as points may cause issues if penetration constraints are added in future versions (convex hull of points != actual object geometry).
  - No per-object Laplacian weights or metadata tracking.
  - Adapters must provide object points in world frame; core does not handle object pose transformation.

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for the full text.

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
