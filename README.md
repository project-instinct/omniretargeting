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

CLI NPZ output also includes a versioned, per-frame `contact_task_*` interface
for all observed body regions and configured candidates, including knees,
forearms, pelvis and abdomen. It records activation, type/role, importance,
confidence, world targets/normals and scene-object association alongside the
complete `qpos`. Python callers use `retargeter.get_contact_task_output()`.
See the [v1 contact/task contract](docs/contact_task_output.md) for shapes,
unknown/inactive values, candidate configuration and reading examples.

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

### Batch scheduling

Use `scripts/batch_retarget.py` to run independent clips concurrently:

```json
[
  {"name": "clip_a", "args": ["--source-motion", "/path/to/a.pkl", "--output", "/path/to/a.npz"], "memory_gb": 9},
  {"name": "clip_b", "command": ["/path/to/command", "--flag"], "memory_gb": 4, "env": {"MUJOCO_GL": "egl"}}
]
```

```bash
.venv/bin/python scripts/batch_retarget.py clips.json \
  --log-dir runs/batch-logs --max-jobs 4 --cpus-per-job 6
```

Each job is launched as `.venv/bin/python -m omniretargeting.main` with its
`args` (or with the exact `command` list), and receives
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`OMNIRETARGETING_CPUS=<--cpus-per-job>`. The CPU value only shrinks geometry
and evaluation worker pools; scheduling changes job start times, not solver
arguments or results. On the 24-core machine, 3–4 concurrent jobs with 6 CPUs
per job are the recommended starting point.

Before starting a job, the runner keeps both the sum of running `memory_gb`
reservations at or below `--memory-budget-gb` (default 46 GiB) and
`/proc/meminfo` MemAvailable, after those reservations, at or above
`--reserve-gb` (default 10 GiB). Jobs missing `memory_gb` use
`--default-memory-gb` (default 10 GiB). Unless `--no-systemd` is set, each job
is placed in a user systemd scope with `MemoryMax=1.5×memory_gb` and
`MemorySwapMax=0`. Logs and `/usr/bin/time -v` peak RSS are written under
`--log-dir`; the final summary records wall time, queue time, exit codes,
makespan, and failed jobs without retrying them. Use `--largest-first` when
large clips should claim memory before short jobs.

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

The shared contact and smoothing defaults apply to every profile. Declared
effectors supply model-specific body names or contact offsets; there is no
preset selector or external algorithm-parameter JSON. Nested
robot-specific objects are merged recursively, while a scalar or list in the
robot profile replaces the algorithm default. The bundled
`unitree_g1_instinctmj.json` includes calibration and geometry overrides. The
G1 profiles have no joint-specific refinement weights, posture targets, or
fixed per-frame joint-step limits. The TRUMANS full-sequence workflow uses the
source-shaped pose and fixed correction budgets; other workflows inherit the
shared refinement defaults.

For a full TRUMANS clip review using the bundled G1 profile and SMPL-X body
models, change only the data directory, segment, and output directory:

```bash
.venv/bin/python scripts/trumans_review_inputs.py \
  --data /path/to/TRUMANS/Data_release \
  --segment '2023-02-19@22-38-25' \
  --output out/my_clip \
  --solve
```

Use `--robot path/to/robot_profile.json` for a different robot and
`--body-models path/to/body_models` if the body models are stored elsewhere.
An optional `--support-evidence path/to/evidence.npy` supplies measured source
support observations when available. This review keeps the source motion and
scene at their original metric scale.

Window proposals use CPU QOCO, with Clarabel as the fallback; other conic
subproblems use CPU Clarabel.

Robot self-collision avoidance is enabled by default, independently of the
scene penetration resolver. It checks only collision-enabled geoms, respects
model collision filters, and checks the paths between saved poses. Visual-only
geoms are ignored. Set `retargeting.self_collision_enabled` to `false` to opt
out; an old saved profile that explicitly sets it to `false` keeps that choice.
The output diagnostics report self-collision separately from scene penetration.

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

Whole-body interaction is not limited to hands and feet. Surface correspondence,
self-contact, surface posture, and coupled whole-body correction are enabled by
the `retargeting.*` defaults. They preserve source motion and metric scale while
letting the torso, pelvis, legs, and surface patches participate in contact and
collision tasks. Current semantics and diagnostics are summarized in
[retargeting invariants](docs/retargeting_invariants.md); module entry points
are listed in the [code layout guide](docs/code_layout.md).

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

### Fast solver performance trace

Set `OMNIRETARGETING_FAST_ENABLE=performance_trace` with `--solver fast` to add
a `performance` section to the saved diagnostics. It contains one exclusive
timing row for each of `source_pose`, `grid_setup`, `contact_acquisition`,
`reweight`, `free_bodies`, `original_mesh_repair`, and `final_audit`, followed
by counters for banded solves, constraint proposals, recoveries, objective and
Jacobian evaluations, guard checks/rejections, accepted step sizes, failed
candidate time, and the binding frame/contact/joint evidence. Tracing is
diagnostic only and does not alter the accepted trajectory or thresholds.

Proposal phase timings are mutually exclusive: `constraint_row_construction`
(rows and expressions), `canonicalization`, `solver_setup`, `numerical_solve`,
`backtracking`, and `restoration`. Nested phase time is subtracted from its
parent, so restoration excludes its construction and numerical solves. The
legacy `constraint_proposals` and `recoveries` timers are inclusive summaries;
`counter_semantics` marks them as non-additive with the phase timers.
`objective_jacobian_evaluations` is a subset of `objective_evaluations`: a
Jacobian evaluation contributes the **same call and seconds to both**. Never
sum those two counters.

Each outer iteration has an `iteration_performance` record, also retained in
`performance.iterations` and phase/window reports. It records all attempted
`banded_feasible_alpha` values (`null` for an unsolved direction), the actual
`accepted_step_backend` (`null` for a rejected iteration), fallback reasons,
actual proposal/restoration call counts by function, and multiple binding rows.
The existing `step_backend` history field is preserved; the diagnostic field
identifies the accepted attempt even after a rejected conic fallback. Binding
diagnostics inspect later guards without re-running their acceptance predicates.
Tracing disables speculative attempt forks and parallel shortening to count the
calls actually consumed by the solver; execution without tracing is unchanged.

### Original geometry on open scans

A scan made only of open sheets has no signed side, so its shell samples are
unknown. Original-mesh repair then queries the actual triangles with each
collision geom (capsule, sphere, solid cylinder): crossing a sheet is
penetration with an exact exit, a geom wholly behind a sheet is not detected.
Other components or geom types, tied exits and uncertified witnesses keep the
stage `unknown_query`. The repair report lists `query_coverage` per category
and `query_failure_examples` with frame, geom, body and backend.
`scripts/review_scene_metrics.py` audits such scenes with the same exact query.

### Fast production path

Fast retains the production chain: fixed source reference → grid → acquisition →
reweight → free_bodies → original_mesh_repair → final audit/export. Coordinated AL
remains research code. Acquisition uses the legacy guarded LM/conic solver and,
for clips of at least 600 frames, its window scheduler; later stages still freeze
acquired contacts and restrict repair scopes.

The default fast production budget is 1 second per saved frame, measured from
CLI loading through export (`sequence_solver.time_budget_seconds_per_frame`).
Checks at LM/window/repair boundaries stop work and retain the last accepted
motion; reweight/free trials require stage acceptance. Export reserves max(2 s,
5% of the budget). Native calls finish before the next check, and `deadline`
reports any overshoot, incomplete audits and skipped stages. Before expiry the
numerical path is unchanged. Independent audits never trigger another solve.

The G1 collision contract is `unitree_g1_mesh.json` (original STL convex hulls).
Use that profile explicitly with the CLI; benchmark preparation defaults to it.
Output records `collision_model_sha256`/`collision_model_name`; training must use
the same compiled collision model. `review_penetration_persistence.py` reports
physical-pair depth, duration and frame fraction plus reset candidate/unknown/
invalid. Structural model overlaps are measured separately with proposed MJCF
excludes; they are not trajectory defects. The legacy solver's structural-pair
heuristic is unchanged. See [the policy](docs/retargeting_invariants.md).

Fast uses one production path. `OMNIRETARGETING_FAST_ENABLE` accepts only
`performance_trace`; algorithm ablations and `OMNIRETARGETING_FAST_DISABLE`
are no longer available.

Original-mesh repair solves each QP and restoration by exact constraint
generation (`_fast/working_set.py`).
These programs carry 10^5 rows and up to 4·10^4 contact cones, of which under one
percent bind. Each solve starts from the rows and cones near their bound at the
zero step (restorations also drop constraints that cannot reach their bound
inside the trust ball), adds every omitted constraint whose slack at the solved
point falls below 1e-6, and stops when none does: that point is the optimum of
the complete strictly convex program. A certified infeasible working set proves
the complete program infeasible; every other outcome solves the complete
program as before. Acceptance, guards and original-geometry checks are
unchanged; only interior-point roundoff differs, which can move a
path-sensitive clip. `OMNIRETARGETING_WORKING_SET_STATS=1` prints outcome counts.

The canonical assembler builds whole-trajectory proposals and restorations
for clips under 600 frames directly in Clarabel standard form instead of through CVXPY (bit-identical
final poses on nine clips); Original geometry reduction skips exact original
mesh queries only where conservative distance bounds prove the result cannot
reach any consumer (bit-identical); Stationary cost-window convergence ends a pass of short
unguarded steps once the three-step cost window stops decreasing, as guarded
passes already did. The last one removes the idle tail of `free_bodies`
(relative decreases of 1e-6 to 1e-10 for up to 15 iterations) and changes the
path slightly: on ten validation clips every penetration change stayed below
1 mm, in both directions. The main repair cache stores repair limit/contact rows
and witnesses per pose and group; window acquisition borrows idle cores for the
native self-collision and deepest-point queries of busy windows; preprocessing
runs support evidence in parallel and computes the source pose while the scene
loads. All three keep final poses bit-identical.

`scripts/benchmark_fast.py --prepare --run-dir <run> --robot-profile unitree_g1_mesh.json`
freezes a baseline, including the chair2745 long clip;
`--arm <candidate> --source-root <snapshot>` runs a frozen candidate with no
algorithm switches. Independent audits run concurrently by default;
`--serial-audits` selects serial auditing. Collision audit worker counts default
to `min(8, available_cpus())`, with `--workers 1` available for serial checks.
See [fast performance](docs/fast_speed.md) for the remaining diagnostic and
resource environment variables.

Fixed per-frame joint velocity, acceleration, and jerk caps have been removed
so deliberate fast source motion can pass through. Neither the framewise
optimization nor its output applies these caps; old configuration keys cannot
re-enable them.
Mechanical joint ranges, scene/contact constraints, and the per-iteration
optimization trust region remain active. Source-relative base-motion bounds
and soft smoothing are configured separately.

The fast backend uses the same immutable framewise reference for final
continuity acceptance: source-relative high-frequency chatter plus URDF velocity
limits, rather than a reference acceleration/jerk cap. Contact acquisition does
not roll back an otherwise useful sweep for a window-assembly chatter flag;
the final audit still reports and gates chatter. Local repair still has a
small per-step joint locality box and protects supported endpoints against
world slip. Whole-trajectory refinement bounds cumulative correction locality,
end-effector displacement, contact residuals, and reliable scene penetration;
each retained update also passes real FK and dense/interframe geometry checks.
These bounds constrain **locality and contact regression**, not deliberate
source speed.
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
