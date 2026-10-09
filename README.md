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
motion = data_source.load()

# Create retargeter
retargeter = OmniRetargeter(
    robot_urdf_path="robot.urdf",
    terrain_mesh_path="terrain.obj",
    joint_mapping={"Pelvis": "torso_link"},
    robot_height=1.6,
    source_target_names=motion.target_names,
    base_orientation={
        "pelvis": "Pelvis", "left_hip": "L_Hip",
        "right_hip": "R_Hip", "spine": "Spine1",
    },
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
from omniretargeting.data_sources import create_data_source
from pathlib import Path

# Load a robot profile and its selected source entry (SMPL-X for this profile).
cfg = load_robot_config("robot_models/unitree_g1/unitree_g1.json")
source_cfg = cfg["selected_source"]

# The adapter handles source loading, target ordering, and source height.
source = create_data_source(
    source_type=source_cfg["type"],
    motion_file=Path("path/to/motion_stageii.npz"),
    source_config=source_cfg,
    runtime_options={"model_directory": "path/to/smplx/models", "gender": "neutral"},
)
motion = source.load()

retargeter = OmniRetargeter(
    robot_urdf_path=cfg["urdf_path"],
    terrain_mesh_path="path/to/terrain.obj",
    joint_mapping=cfg["joint_mapping"],
    robot_height=cfg.get("robot_height"),
    source_target_names=motion.target_names,
    base_orientation=cfg.get("base_orientation"),
    retargeting=cfg.get("retargeting"),
)

source_to_robot_scale, retargeted_motion = retargeter.retarget_motion(
    motion,
    enable_scene_scaling=True,
    visualize_trajectory=False,
)

print(f"Source-to-robot scale factor: {source_to_robot_scale}")
print(f"Retargeted motion shape: {retargeted_motion.shape}")  # (T, 7 + DOF)
```

For a ready-to-run setup, omniretargeting ships with robot profiles under
`robot_models/`. These profiles contain robot assets, target-to-link mappings,
source adapter options, orientation landmarks, link offsets, and retargeting
settings. Source height comes from `MotionData.source_height`; link offsets are
stored in each `target_mapping` entry as `offset` alongside `robot_link`.

```python
from omniretargeting import OmniRetargeter, load_robot_config
from omniretargeting.data_sources.smplx import DEFAULT_SMPLX_TARGET_NAMES

cfg = load_robot_config("robot_models/unitree_g1/unitree_g1.json")
retargeter = OmniRetargeter(
    robot_urdf_path=cfg["urdf_path"],
    terrain_mesh_path="path/to/terrain.obj",
    joint_mapping=cfg["joint_mapping"],
    robot_height=cfg.get("robot_height"),
    source_target_names=DEFAULT_SMPLX_TARGET_NAMES,
    base_orientation=cfg.get("base_orientation"),
    retargeting=cfg.get("retargeting"),
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

Supplied root poses must be finite, with exact shapes `(T, 3)` and
`(T, 4)`; root quaternions must be unit wxyz. Supplied heights and framerates
must be finite and positive, and the two height aliases must agree. Dense frame
fields such as root poses and object samples must be present on every frame or
absent throughout. Sparse contacts can remain unavailable on individual frames.

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

Pass the corresponding order to `OmniRetargeter(source_target_names=...)`. Any key in `joint_mapping` must be present in `source_target_names`, and each mapped value must match a body name in the robot URDF; invalid robot body entries raise `ValueError` at initialization. Incoming named motion must match the configured target count and order; unnamed position arrays must match the count. Streaming also validates declared `DataSource.target_names` for unnamed frames, without loading the complete stream.

### Terrain Mesh
Supports common mesh formats:
- `.obj` (Wavefront OBJ)
- `.stl` (STL mesh)
- `.ply` (Polygon File Format)
- `.gltf`/`.glb` (glTF)

Automatic scene scaling requires a finite positive `MotionData.source_height`. Adapters return `None` when height landmarks are unavailable; provide a measured height before enabling automatic scaling.

**Optional Scene Scaling**: the scene is unscaled by default. Pass `enable_scene_scaling=True` to `retarget_motion()` or `--enable-scene-scaling` to the CLI to scale source motion, terrain, and objects by the robot/source height ratio. The CLI also exports the scaled scene beside the output motion. Alternatively, `--scale-factor FACTOR` applies a fixed scale without exporting the scene; the two CLI scaling options are mutually exclusive.

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

With `--enable-scene-scaling`, scene exports go into a directory named after
the normalized output stem. For `--output /path/to/output.npz`:

```text
/path/to/
├── output_retargeted.npz
├── output_retargeted.contacts.json  # when contact annotations are available
└── output_retargeted/
    ├── scaled_terrain.obj
    ├── <object_name>.obj            # when the adapter exposes an object mesh
    └── <object_name>_poses.json     # when object poses are available
```

`--save-video PATH` writes to the supplied path. Without automatic scene scaling,
the CLI writes motion and available contact annotations without scene exports.

## Advanced Usage

### Custom Robot Height

```python
retargeter = OmniRetargeter(
    robot_urdf_path=robot_urdf,
    terrain_mesh_path=terrain_mesh,
    joint_mapping=joint_mapping,
    robot_height=1.8,  # Override auto-detected height
    source_target_names=motion.target_names,
    base_orientation=cfg.get("base_orientation"),
)
```

### CLI

The CLI is driven by a per-robot JSON profile. The URDF path, joint mapping,
and retargeting settings all come from the profile — the CLI does **not**
accept a separate URDF argument.

#### YAML source configs

`--source-config` is required. Edit a template in `config_templates/` to set
motion and model paths, adapter options, and optional terrain. For example:

```yaml
type: smplx
motion: /path/to/motion_stageii.npz
model_directory: path/to/smplx/models
terrain: /path/to/terrain.obj
```

Set `terrain` in the YAML; omitting it uses flat ground. Raw SMPL-X parameters
require `model_directory`; processed target-position files do not.

**SMPL-X example**

```bash
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --source-config config_templates/smplx_template.yaml \
  --output /path/to/output.npz \
  --enable-scene-scaling \
  --framerate 30 \
  --penetration-resolver xyz_nudge
```

**OMOMO / object-interaction example**

```bash
python -m omniretargeting.main \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --source-config config_templates/omomo_floorlamp_example.yaml \
  --output /path/to/output.npz \
  --enable-scene-scaling \
  --save-video /path/to/output.mp4
```

**HSOI contacts and implicit detection**

`MotionData` now carries optional `entity_trajectories` and a sparse
`contact_trajectory`: one variable-length contact list per frame. `None` means
unavailable annotations; `[]` means no detected contacts. Each `Contact` names a
human target and scene body, and stores a body-local surface point plus optional
confidence and normal. Entity orientations use `wxyz` quaternions. Scene geometry
and explicit static poses live in `Scene.entities`; moving bodies supply an
`EntityTrajectory`. `MotionFrame` carries `entity_poses`, `contacts`, and the scene.

The geometric detector finds the nearest triangle surface for configured
point/body pairs, checks distance and optional speed **in the body's frame**, and
removes short contact runs. It supports walls and moving objects as well as ground.
Anchors are computed each frame to preserve sliding. Existing available
annotations, including empty lists, survive unless `overwrite: true` is set.
The local `hoi-retarget` implementation provides the binary contact, local-anchor,
and proximity/speed stance precedents; object detection here extends that approach
to arbitrary scene meshes without importing the reference implementation.

On aorua, activate `robot-data` using the installed Miniconda path. This environment
needs `joblib` and `PyYAML` for the OMOMO adapter and CLI; both are declared package
dependencies. Install them in your chosen environment if missing:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robot-data
python -m pip install joblib PyYAML
python -m omniretargeting.main \
  --source-config config_templates/omomo_contacts_example.yaml \
  --output /tmp/omomo_contacts.npz --progress
```

The example uses OMOMO's recorded shaped rest offsets and the existing skeleton
FK helper (`body_position_mode: rest_offsets`), so SMPL-X models and Torch are
optional. `body_position_mode: smplx` retains the existing model-based path.
OMOMO has varying recorded object scales: `object_scale_mode: per_frame` retains
legacy world samples, while `first_frame` explicitly uses one constant scale for
rigid scene geometry, samples, visualization, and contact anchors. Detecting
contacts with the OMOMO object requires constant scale; the example chooses
`first_frame`. Terrain-only detection works with either scale mode. Omitting
`pairs` checks all named targets against the available scene entities; a
varying-scale legacy object has no rigid scene entity.

Tune `contact_detection.pairs`, `distance_threshold`, `velocity_threshold`
(`null` allows distance-only sliding detection), and `min_contact_frames` in YAML.
Thresholds use the input motion's units, before scene scaling. Tune
`contact_edge_weight` separately, or override it with `--contact-edge-weight`.
The default solver weight is zero. Every contact used by the graph needs a source
target mapping to a robot link point; additional `target_mapping` entries in the
source YAML extend or override the robot profile, as shown for G1 toes.

Contact contributions are added to ordinary spatial edges before normalization.
Source coordinates and robot Laplacians use the same frozen normalized graph;
scene samples and inserted anchors have zero Jacobians during each frame solve.
Duplicate records for the same pair/anchor use maximum confidence. Each pair's
distinct active anchors become one confidence-weighted centroid in the graph,
with their mean confidence as the edge contribution. This keeps the number of
anchor residual rows independent of patch density; the original surface anchors
remain in the annotations. Weighting improves interaction relationships;
it does not enforce exact attachment or prevent object penetration.

Output paths are normalized to `*_retargeted.npz`. With annotations, the CLI also
writes `*_retargeted.contacts.json` containing contacts and resolved scene pose
tracks in the output scale. Geometry stays separate. Use `contact_annotations`
in source YAML to read an edited JSON contact trajectory on the original input
timeline; the stored scale is converted back to source coordinates automatically.
For a resampled output, annotations must first be aligned to the input timeline.
Object exports preserve the body's local origin. Rigid scene geometry includes
the constant object scale and exports unit pose scales, matching local contact
coordinates. Legacy varying-scale objects retain their recorded pose scales;
scene scale is baked into exported geometry once in either case.
`MotionData.copy()`, `slice_frames(slice(...))`, `scaled(factor)`,
`transformed(wxyz, translation)`, and `resample(fps)` preserve HSOI fields.
Resampling selects entire nearest-frame contact lists, with earlier frames winning
ties; translations and orientations use linear interpolation and SLERP on the
same time grid. Rigid legacy object samples are regenerated from local samples
and the resampled body pose; world samples without that representation retain
linear interpolation. For a geometry-only entity, existing world samples are
converted into its original body poses, resampled locally, and transformed
by the new poses without dropping samples. Short events can be lost through
downsampling.

Batch jobs accept the same settings through `--source-options`. OMOMO `.p` files
contain many sequences; this command runs the specified `sequence_index` from
each matching file, rather than expanding every sequence in the dataset:

```bash
python -m omniretargeting.batch \
  --source-folder ~/Datasets/OMOMO/data --source-type omomo \
  --file-pattern test_diffusion_manip_seq_joints24.p \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --output-dir /tmp/omomo_contact_batch --max-workers 1 \
  --source-options '{"sequence_index":318,"body_position_mode":"rest_offsets","object_scale_mode":"first_frame","contact_detection":{"pairs":{"floorlamp":["L_Wrist","R_Wrist"]},"distance_threshold":0.10,"velocity_threshold":null,"min_contact_frames":3},"contact_edge_weight":10}'
```

For Python use, construct scene entities and trajectories alongside `MotionData`,
then call `detect_contacts(motion, config)` and attach its returned frame lists
before passing the motion to `OmniRetargeter`. Source body-model details stay in
adapters; the generic solver receives mapped point indices, world anchors, and
explicit edge contributions.

Custom height landmarks can be configured in source YAML for both main and
batch workflows (batch accepts the same block through `--source-options`):

```yaml
height_estimation:
  head_joint: crown
  foot_joints: [sole_l, sole_r]
  head_top_offset: 0.25
```

These names refer to source points, and the offset is in meters. For Python
use, pass the block as `height_estimation` to the adapter constructor or
through `source_config`/`runtime_options` in `create_data_source`. Missing
configured landmarks remain an unavailable measurement; automatic scaling
requires a finite positive `MotionData.source_height`.

#### Migrating older CLI scripts

The main CLI requires `--source-config`; legacy source-loading flags are removed.
Move motion paths, model directories, and terrain paths into the YAML fields
`motion`, `model_directory`, and `terrain`. Use `--enable-scene-scaling` for scene
exports. The batch entry point accepts `--terrain` and JSON `--source-options`,
and writes those settings into each generated source YAML.

Main arguments:

| Flag | Default | Description |
|---|---|---|
| `--robot-config` | `robot_models/unitree_g1/unitree_g1.json` | Path to robot profile JSON. |
| `--source-config` | *(required)* | YAML source configuration file. See `config_templates/`. |
| `--output` | *(required)* | Output `.npz` path (normalized to end in `_retargeted.npz`). |
| `--enable-scene-scaling` | off | Scale motion, terrain, and objects by the robot/source height ratio; export the scene into the normalized output stem directory. |
| `--scale-factor FACTOR` | `None` | Apply a fixed scene scale without scene exports; mutually exclusive with `--enable-scene-scaling`. |
| `--framerate` | auto / 30 | Motion framerate; auto-detected from the source file when possible. |
| `--output-framerate` | `None` | Resample source motion to this framerate before retargeting. |
| `--vis` | off | Launch a MuJoCo viewer on the retargeted motion. |
| `--save-video PATH` | off | Render the retargeted motion to video (requires `imageio[ffmpeg]`, and `MUJOCO_GL=egl`/`osmesa` for headless). |
| `--progress` | off | Show a progress bar while retargeting frames. |
| `--contact-edge-weight` | profile / YAML, otherwise `0` | Override the interaction graph's extra contact weight. |
| `--cprofile PATH` | `None` | Write cProfile statistics for the complete retargeting run. |
| `--penetration-resolver {hard_constraint,hard_constraint_slack,xyz_nudge}` | profile value | Contact handling mode; overrides the value in the profile. Slack mode uses configured values or its documented defaults. |

### Batch Processing

`batch.py` scans a folder of motion files and retargets every file with the same
robot profile and terrain. It writes per-motion YAML source configs, logs
repository status before processing, and runs the first file as a probe job
before processing the rest.

```bash
python -m omniretargeting.batch \
  --source-folder /path/to/motions \
  --source-type smplx \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --output-dir /tmp/batch_output \
  --source-options '{"model_directory":"path/to/smplx/models"}'
```

| Flag | Default | Description |
|---|---|---|
| `--source-folder` | *(required)* | Folder containing motion files to process. |
| `--source-type` | *(required)* | Source type: `smplx`, `lafan1`, `nokov`, or `omomo`. |
| `--robot-config` | *(required)* | Path to robot profile JSON. |
| `--output-dir` | *(required)* | Directory for batch outputs (configs, logs, motions). |
| `--terrain` | `None` | Path to terrain mesh applied to all motions. |
| `--max-workers` | auto | Maximum parallel workers, sized from the probe job and available memory. |
| `--framerate` | auto | Override framerate for all motions. |
| `--source-options` | `None` | JSON adapter options for all motions, including `model_directory` for raw SMPL-X. |
| `--skip-test-job` | off | Skip the initial probe job and process all files directly. |
| `--timeout` | `3600` | Per-file timeout in seconds. |
| `--video` | off | Save per-motion videos beside the motion outputs. |
| `--scale-factor FACTOR` | `None` | Apply one scene scale to all motions; export the shared terrain once when `--terrain` is supplied. |

Output layout under `--output-dir`:

```
output-dir/
├── git_status/              # repository snapshots before processing
│   ├── omniretargeting.status
│   └── omniretargeting.diff  # when changes are present
├── configs/                 # per-motion YAML source configs
│   └── <name>_config.yaml
├── logs/                    # per-motion subprocess stdout/stderr
│   └── <name>.log
├── motions/
│   ├── <name>_retargeted.npz
│   ├── <name>_retargeted.contacts.json  # when annotations are available
│   └── <name>_retargeted.mp4           # with --video
└── terrain/
    └── scaled_terrain.obj   # with --scale-factor and --terrain
```

### Robot Profile Config (Per-Humanoid)

Keep one JSON profile per humanoid robot (for example under
`robot_models/<robot_name>/`). Relative `urdf_path` values are resolved against
the profile file's directory.

Current shipped profiles contain robot fields and a list of source entries:

- `name` – optional profile name, used in log output
- `urdf_path` – **required**, path to the robot URDF (relative to the profile file)
- `source[].target_mapping` – **required** for the selected source, source target
  name → robot body name or `{robot_link, offset}`; normalized to `joint_mapping`
- `robot_height` – optional override for auto-detected robot height
- `source[].target_names` – optional custom source target ordering; use the
  adapter's `motion.target_names` when constructing the retargeter
- `source[].adapter_options` – source loading and body/skeleton options;
  runtime source options override these, which override direct source fields
- `height_estimation` – adapter height measurement controls: source `head_joint`,
  source `foot_joints`, and `head_top_offset` in meters. The CLI uses a top-level
  profile block when the selected source does not define one. Precedence is
  source YAML > `source[].adapter_options` > `source[]` > top-level profile.
  Explicit SMPL-X settings use these landmarks instead of the model height.
- `base_orientation` – source target names used to estimate root orientation (`pelvis`, `left_hip`, `right_hip`, `spine`)
- `retargeting` – solver settings forwarded to `GenericInteractionRetargeter`:
  - `collision_detection_threshold`
  - `terrain_sample_points`
  - `replace_cylinders_with_capsules`
  - `penetration_resolver`: `"hard_constraint"`, `"hard_constraint_slack"`, or `"xyz_nudge"`
  - `penetration_slack`: optional overrides for slack mode (`soft_tolerance`, `hard_bound`, `slack_penalty`); omitting the block still enables slack defaults
  - `penetration_correction`: physical tangent-space correction metric and per-block limits (example below)
  - `solver_diagnostics`: retain detailed contact-row and correction-allocation diagnostics on the inner retargeter's `last_solve_diagnostics` (default `false`)
  - `foot_stabilization`: nested block (see `robot_models/unitree_g1/unitree_g1.json`) that controls the post-processing XYZ-nudge pass (`enabled`, `clearance`, `surface_clearance`, `contact_clearance`, `xy_correction_gain`, smoothing windows, wall-contact thresholds, etc.)

The hard-constraint solvers optimize physical MuJoCo tangent DOFs rather than
the four ambient quaternion coordinates. Their correction allocation can be
configured independently for translation, rotation, and articulated joints:

```json
{
  "retargeting": {
    "penetration_resolver": "hard_constraint_slack",
    "penetration_slack": {
      "soft_tolerance": 0.001,
      "hard_bound": 0.03,
      "slack_penalty": 100000.0
    },
    "penetration_correction": {
      "base_translation_weights": [0.001, 0.001, 1.0],
      "base_rotation_weight": 5.0,
      "joint_weight": 0.001,
      "joint_range_normalization": true,
      "base_translation_step": [0.2, 0.2, 0.05],
      "base_rotation_step": 0.2,
      "joint_step_fraction": 0.1,
      "step_tolerance": 0.00001,
      "feasibility_tolerance": 0.000001,
      "max_backtracks": 6,
      "restoration_penalty": 10000000.0
    }
  }
}
```

The listed values are the defaults when `penetration_correction` is omitted;
`base_translation_step` X/Y and `base_rotation_step` otherwise inherit the
solver's `step_size`. Joint weights are divided by squared joint range when
range normalization is enabled. A matching `joint_regularization_boost.joints`
entry takes the larger weight for that joint. The legacy
`base_position_tracking_weight` takes precedence over X/Y translation weights
when explicit source-root translation is available, and
`base_position_tracking_weight_z` takes precedence over the Z translation
weight in the same case; otherwise Z keeps the independent
`penetration_correction` weight and remains movable.
When a frame starts outside the hard bound by more than one allowed physical
step, the SQP uses heavily penalized restoration variables to make monotone
progress over multiple iterations. A solve is still reported successful only
after nonlinear geometry satisfies the actual hard bound.

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
    base_orientation=None,
    retargeting=None,
)
```

#### Methods

Unknown solver and OMOMO adapter controls raise `ValueError`. Supplied
`penetration_slack` keys and type are checked for every resolver; valid inactive
settings remain available for fallback without activating slack handling.
Public retargeting
raises a `RuntimeError` with the zero-based frame index when numerical recovery
is exhausted; backend exceptions retain their cause and frame context.

- `retarget_motion(motion, base_orientations=None, base_translations=None, framerate=None, visualize_trajectory=True, enable_scene_scaling=False, show_progress=False)` → `(source_to_robot_scale, retargeted_motion)`
- `get_robot_dof()` → `int`
- `get_joint_names()` → `List[str]`
- `validate_joint_mapping()` → `List[str]` (robot body names from `joint_mapping` that are missing from the URDF)

### `load_robot_config`

```python
from omniretargeting import load_robot_config
cfg = load_robot_config("robot_models/unitree_g1/unitree_g1.json")
```

Loads a robot profile JSON, resolves `urdf_path` relative to the profile file, and normalizes legacy flat and nested profile fields. Raises if no non-empty mapping is available.

### Position-array wrappers

`retarget_source_to_robot(..., source_target_names=names, base_orientation=...)`
requires source orientation landmarks (`pelvis`, `left_hip`, `right_hip`, and
`spine`) just like `OmniRetargeter`. Pass the selected profile's orientation
configuration. Both wrappers default to `enable_scene_scaling=False`; to scale,
pass `enable_scene_scaling=True` and an explicit measured `source_height`.
`retarget_smplx_to_robot()` supplies standard SMPL-X target names
and orientation landmarks by default; custom names need an explicit
`base_orientation` override.

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
- clarabel, libigl, tyro
- open3d, pyvista

## Architecture

OmniRetargeting adapts the interaction-mesh retargeting approach from the
holosoma_retargeting project to work with generic robots and terrains:

1. **Source-to-Robot Scaling** (optional): Computes the robot/source height ratio and scales source motion, terrain, and objects before retargeting (enabled by `enable_scene_scaling=True` or `--enable-scene-scaling`).
2. **Generic Robot Support**: Works with any URDF through automatic model loading, body-name validation, and auto-detected height.
3. **Interaction Mesh**: Builds a tetrahedral interaction mesh from mapped source targets and terrain sample points.
4. **Optimization**: Per-frame SQP optimization with Laplacian-deformation objective, joint limits, and a target base-orientation term for smoothness.
5. **Collision / Penetration Handling**: Three modes selectable via `retargeting.penetration_resolver`:
   - `hard_constraint` – penetration inequalities inside the SQP.
   - `hard_constraint_slack` – hard backstop plus penalized soft-tolerance slack inside the SQP.
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
