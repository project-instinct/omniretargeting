# Whole-sequence retargeting

Use the normal data adapter, robot profile and scene pipeline:

```sh
python -m omniretargeting.main --solver sequence \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --source-config source.json --output motion.npz
```

The equivalent profile option is `retargeting.sequence_solver.enabled: true`.
`--solver framewise` explicitly selects the existing framewise backend. The
sequence backend is selectable for any configured source/robot correspondence;
it has no OMOMO clip list, G1 joint count, fixed frame rate, selected bad frames,
or dependency on an upstream checkout or previous output files.

## Algorithm

`sequence_retargeting.retarget_sequence` jointly fits the entire input timeline
with bounded MuJoCo tangent increments and nonlinear objective backtracking.
The robot model supplies scalar coordinates, joint limits and the optional
floating base. All contact/pose/reference terms are evaluated on the same
candidate trajectory as temporal terms; there is no post-hoc low-pass filter.
Velocity, acceleration and jerk regularization use seconds, so changing the
input sample rate does not silently change the intended derivative penalties.
Quaternion signs are aligned before component differences.

Inputs are mapped source positions, immutable root positions, `SceneContactTask`
objects, effector geometry/configuration and optional static/moving environment
points. No SMPL-X extraction occurs in the solver. Pose edges come from supplied
correspondences or mapped robot ancestry. Optional `pose_axes` are linear
combinations of mapped markers, allowing torso axes without hard-coded names.
Core prepares those axes from its existing arm/root correspondences.

Fixed contacts use the exact world anchors in the supplied task. Sliding and
transition modes constrain the scene normal. Free end effectors follow the
configured body-relative morphology target; `preserve_world_target` retains
world targets for feet. One material marker is selected for a contiguous contact
run and carried through final validation/repair. Source positions, contact
schedules, object poses, geometry and timing are not rewritten.

The generic CLI prepares a neutral/calibrated model seed on the source path,
acquires the body/foot configuration with relaxed contact weights, then solves
the original interaction targets across all frames. During initialization,
world-tracked supports retain the side witnessed by the original source and
actual scene normals. This prevents a finite terrain slab from treating a foot
underneath its bottom as a valid initialization. No source or terrain transform
is changed. It does not apply the old hand-contact overlay or
fixed-base trajectory refinement afterwards. Existing bounded local joint repair
and whole-body moving-mesh checks remain in the final pipeline. They preserve
the sequence result's floating base. Collision-free certification and physical
controller tracking are separate from the kinematic solver.

## Configuration and scope

`SEQUENCE_DEFAULTS` contains shared physical normalizations; no per-clip settings
are used. Sticking anchors remain fixed even when transition weights fade; those weights
do not interpolate the anchor away from its original world location. Contact
terms have dominant finite weights, rather than a mathematical
strict-priority guarantee, so final contact/world-position errors must be checked.
Diagnostic reports include all nonlinear iterations, model dimensions, frame
rate, fixed material choices and separate contact/scene/self residuals. Failed
line searches retain the complete previous trajectory, avoiding per-joint splices.

The current implementation supports scalar hinge/slide joints and at most one
free root, including fixed-base models. Ball joints are rejected explicitly.
Matrices are sparse across time but each frame has a local interaction graph;
very long recordings may need additional memory and solve time. Original native
moving-object geometry is checked in the existing final correction stage.

Validation covers synthetic models with 1, 2 and 4 scalar joints, input rates
15–120 fps, 1–3-frame inputs, antipodal root quaternions, and rapid contact motion
without moving its peak frame, cold initialization without recorded seeds,
and feet initialized below a finite terrain slab. A separate H1 19-joint/24 fps FK-marker experiment
checks another real robot model. It is synthetic evidence, not an additional
human recording. OMOMO tests use the same generic API for the three original
clips and the normal CLI for an additional small-box regression recording. Individual measured
results are recorded under `out/omomo_trial_20260921/general_sequence`.
