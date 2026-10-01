# Whole-sequence retargeting

Use the normal data adapter, robot profile and scene pipeline:

```sh
python -m omniretargeting.main --solver sequence \
  --robot-config robot_models/unitree_g1/unitree_g1.json \
  --source-config source.json --output motion.npz
```

The equivalent profile option is `retargeting.sequence_solver.enabled: true`.
`--solver framewise` explicitly selects the existing framewise backend and
`--solver fast` (profile: `sequence_solver.backend: fast`) the
[speed-first backend](#fast-backend). The
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
Root XY is free by default with a soft source-path reference. Root translation
and rotation corrections are regularized relative to the original source, so
deliberate fast travel and back rotations are not smoothed as solver error.

Inputs are mapped source positions, immutable root positions, `SceneContactTask`
objects, effector geometry/configuration and optional static/moving environment
points. No SMPL-X extraction occurs in the solver. Pose edges come from supplied
correspondences or mapped robot ancestry. Optional `pose_axes` are linear
combinations of mapped markers, allowing torso axes without hard-coded names.
Core prepares those axes from its existing arm/root correspondences.
It also transfers calibrated source body orientations (including parent-relative
back/waist frames) into the sequence solver. The source root orientation target
is frozen before bootstrap, not replaced with the bootstrap's fitted posture.

The generic API accepts `root_orientation_targets` as `(T, 4)` wxyz quaternions
and `body_orientation_targets` as robot-frame rotation specifications. When no
root orientations are provided, the API uses the initial root rotations as its
reference. Orientation axis residuals track pitch, roll and heading independently
of root translation. See [adaptive root motion](adaptive_root_motion.md) for the
paper references, configuration, and distinction from footstep planning.

Fixed contacts use the exact world anchors in the supplied task. Sliding and
transition modes constrain the scene normal. Free end effectors follow the
configured body-relative morphology target; `preserve_world_target` retains
world targets for feet. Free-hand targets follow all three translation components
of the current robot root; their Jacobians include this target motion. One material marker is selected for a contiguous contact
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
When dense surface pose or learned whole-body self-contact tasks are supplied,
the solver additionally uses fixed input-trajectory endpoint/contact budgets,
constrained sparse subproblems, and nonlinear acceptance. These budgets prevent
the added tasks from accumulating endpoint regression; they do not certify that
the input contact errors are already small. See [surface objectives](dense_surface_pose_and_self_contact.md).
With [contact roles](contact_phase_priority.md) the constrained subproblem first
keeps scene/self clearance, then endpoint support, surface support and following
contacts, before the weighted objective.

The constrained subproblem is exact but kept small. Per-frame bounds form one
sparse affine map and one vectorized second-order cone. Because frames couple
only through the temporal differences, the least-squares objective is replaced
by its banded Cholesky factor (one row per coordinate instead of one per dense
surface sample). It is then scaled to O(1): the minimizer is unchanged, but at
the raw O(1e7) magnitude Clarabel stopped at its iteration cap with millimetre
bound violations. Dense surface-pose rows are built per frame as one stacked,
pre-weighted block. On OMOMO with learned whole-body surfaces one iteration pair
fell from 1011 s to 62 s, and every subproblem now reports `optimal`
(`out/contact_roles_20260923/BASELINE_PATCH.md`).
Diagnostic reports include all nonlinear iterations, model dimensions, frame
rate, fixed material choices and separate contact/scene/self residuals. Failed
line searches retain the complete previous trajectory, avoiding per-joint splices.

## Guarded steps, numerics and runtime

Guarded passes (a fixed endpoint or correction reference) bound every joint's
per-frame first, second and third differences by the reference. At the first
iterate these rows are active with only round-off slack, so their scale decides
whether a step survives acceptance. Each joint row is expressed in units of its
own budget; this exact positive scaling leaves the feasible set unchanged. With
raw rows, an interior-point accuracy of 1e-6 exceeded the 2e-7 acceptance slack
and forced line-search fractions near 1/8, restoration solves, or no step.

`conic_solver.solve_conic` retries a CLARABEL solve once with 1e-6 targets when
it stalls numerically (numerical error, insufficient progress, or the default
iteration cap). Converged and certified infeasible/unbounded outcomes are not
retried, and an explicit caller iteration/time budget is respected. Every step
is still checked against the unchanged nonlinear bounds.

The two passes (`pose`, `root_continuity`) evaluate the same objective. A pass
that stops on a rejected step, or after two consecutive accepted steps below a
1e-6 relative decrease without a full-tolerance per-frame priority gain, is not
repeated; a global pass that exhausts its budget continues, while windowed
passes count whole sweeps and stop at their budget. A zero step (a failed
subproblem) skips the line search, which could only reproduce the iterate.

Guarded passes of clips longer than `WINDOW_FRAMES` (48) solve the same
linearized problem window by window: frames outside the window keep a zero step,
so only rows whose stencil reaches the window remain. Posture, continuity,
attained-contact, per-frame priority and restoration checks are the same bound
evaluated on the frames a step can move; the cost comparison stays global.
Alternate sweeps shift window boundaries by half a window. Alternate windows
form two groups whose members share no row, bound or priority frame; each group
starts from one iterate, its members are solved in forked worker processes
(`WINDOW_WORKERS`, single-threaded BLAS each) and only their own frames are
merged, so any worker count gives identical results. The unguarded
source-pose pass stays one global solve. On the 1304-frame TRUMANS holdout the
global guarded subproblem (45,640 variables) did not converge in CLARABEL even
after the retry, so contact acquisition never moved; windows keep each conic
problem small enough to solve.

Both are internal module constants, not configuration: they change scheduling
and subproblem size only.

## Fast backend

`--solver fast` keeps the source pipeline, contact inference, task inputs and
the two-stage structure (source-shaped reference, then contacts/scene from that
fixed reference), but replaces the guarded conic cascade by Levenberg–Marquardt
on the banded normal equations of the whole trajectory (`omniretargeting/_fast`).
It adds no configuration: its constants are internal numerical choices and the
existing sigmas, iterations and budgets are reused.

OMOMO object meshes use the recorded per-frame uniform `obj_scale` in addition
to rotation and translation. The fast backend builds a metric-sized object grid
and transforms each frame's points and distances with that scale; it does not
freeze the recorded scale or resize the robot. Contact anchors keep a common
material-frame body origin with fixed metric robot offsets. When object scale
changes, individual points of a rigid robot patch cannot all remain fixed in
object coordinates, so their world-distance fit residual is retained. The review
WebViser also applies the recorded scale. `SelfContactTasks` and
`SurfacePoseTasks` are reported as skipped by the fast backend, rather than
misinterpreted as environment contact tasks.

- Rows. Every per-frame row family of `SequenceTasks.frame_rows` (interaction
  graph, bone axes, orientation targets, effector contact/sliding/free/world
  targets and effector orientation), `SurfaceContactTasks.rows` (including
  pose-normal rows), `SurfaceProximityTasks.rows`, and the root, reference and
  temporal rows of `retarget_sequence` are evaluated for all frames at once
  with the same sigmas; tests compare them with the guarded rows to round-off.
  Points on a body contribute through per-body 6x6 moments, never per-point
  Jacobians. Guarded role levels become role-weighted stiff rows
  (support 4, hold/control 2, follow 1, per contact sigma).
- Normal equations. Per-frame blocks plus third-order temporal rows form one
  symmetric banded system solved by banded Cholesky (single-threaded).
  Per-frame terms are evaluated in forked workers on contiguous frame chunks;
  the worker count changes scheduling only (bit-identical to serial).
- Bounds. Joint limits are exact per-frame boxes; joints resting on a bound
  they would cross are held for the iteration (active set) rather than clipped
  after it. The root translation (3 cm) envelope is an exact projection with a
  radial hold at the boundary; the root orientation has no bound. Posture has
  no angular limit (`pose_relations.py`): all pairs of mapped landmarks and
  effector tips, normalized by their skeleton path length, are compared with the
  immutable pre-contact reference as equivalent rigid-rotation angles, and a soft
  regularizer minimizes that change. The only hard posture rule is the layout:
  clear front/back, left/right and up/down relations between limbs, or a limb and
  the trunk, may not flip. Every nonlinear proposal is checked against it and a
  flip backtracks; the fallback conic step uses its linearized halfspaces and the
  objective a stiff hinge. Relation angles to the reference and to the source,
  and the seed's per-bone source directions, are reported only. A successful
  numerical solve is not motion acceptance.
- Continuity. Fast hard bounds constrain each joint's executed first, second and
  third differences, using the immutable reference's local derivative maximum
  (7 stencils) plus 10% of that joint's RMS. Budgets are per joint and per stencil,
  never shared between joints. Candidate checks and linearized constraints use
  these absolute derivatives; root translation has coordinate checks during
  acquisition and an independent 3D norm audit afterward. Root angular increments
  are checked separately. The default reweight round retains the foot_v8 weights;
  acquisition norm constraints and signal-specific reweighting (`batch2`) remain
  opt-in until the complete six-clip combination passes acceptance.
  Secondary stages preserve acquired continuity, with independent RMS, peak and
  high-frequency acceptance. The guarded sequence backend is unchanged.
  The correction-derivative experiment (`correction_budget`) failed six-clip
  quality acceptance and is disabled by default; see the fifth runtime round.

- Guard-active steps. Once a constrained (conic) step brings continuity or
  acquired-contact bounds to their limit, every banded step crosses them at
  every fraction, and the three conic retries per iteration could all
  overshoot the cost, ending acquisition early (omomo_485: two iterations,
  a hand 6 cm inside the hip). An overshooting conic step is now shortened
  (Armijo backtracking along it) before its attempt is rejected; a fraction
  that decreases the cost but crosses an active nonlinear bound at second
  order is restored onto the guards (`restore_shape_step`, at most two per
  attempt). A shortened step's small decrease counts toward convergence, and
  a guarded pass also ends when its last three accepted iterations together
  lowered the cost by less than three times the convergence threshold (full
  and guard-clipped steps alternate there, so two consecutive slow full steps
  rarely occur: omomo_5 spent 19 iterations on its last 3 %). Restorations
  use exact constraint generation (`shape_step.WORKING_SET_RESTORATION`,
  3-20x faster per solve; omomo_78 acquisition 290 s -> 130 s, same cost).
- World paths. A world-tracked free effector (feet, by adapter semantics)
  follows its source world path as the root follows the source root: the
  first to third derivatives of its offset from that path use the root path
  sigmas (`root_xy_derivative_sigmas`), on stencils whose frames are all
  world-tracked and free (never across a contact target or regime handover).
  A constant morphology offset stays free. As for the root, the path excludes
  its chatter (content below `CHATTER_SCALE_FRAMES`): following the spikes of
  the skin-derived foot target exactly doubled leg joint jerk, while the
  per-frame 2 cm target still holds every recorded position. Before, a planted foot only had its
  2 cm position sigma: on omomo_78 the reference feet chattered at 9 mm/frame²
  in stance while the source ankle moved 0.7 mm/frame², and acquisition kept
  that chatter because its rows are measured against the reference.
- Root path. The root path rows measure the root rotation against the source
  pelvis rotation without its chatter (content below `CHATTER_SCALE_FRAMES`,
  2 frames); the root orientation target still holds every recorded rotation.
  Joints never copied the source's local-rotation noise, but the root copied
  about 1° RMS of pelvis wobble (omomo_78), which the legs then had to absorb
  to keep the feet planted (the waist bounced off its pitch limit).
- Source first. A calibrated orientation seed is refined using source geometry,
  bone directions, body rotations and temporal terms, with root translation
  fixed. Contact or scene targets cannot choose a different action at this
  stage. Next, an affine temporal least-squares solve fits only robot root height
  to the source world and body-relative tracking targets (the feet's world path;
  contact targets are not used, `height_registration.contact_targets_used`).
  Joint angles, root orientation, horizontal path, source trajectory and object
  poses stay unchanged.
- Supports. The registration does not see a seat, so with legs shorter than the
  human's (G1 on a TRUMANS chair) it leaves the base about 8–11 cm inside the
  seat, beyond acquisition's 3 cm root ball (`translation_step_m`) and out of
  reach of any joint. Before the reference is frozen,
  `PenetrationStages.reference_lift` raises it smoothly (0.5 s ramp) by the
  minimal amount reaching the plane through the held contact targets, where at
  least three collision samples of the base sit inside an upward support. Geoms
  carrying a source surface region that the base also carries count with the base
  (the buttock samples bind to G1's hip-pitch housings as well as the pelvis).
  Joint angles, root orientation and the horizontal path are unchanged;
  acquisition then brings the feet back to their world targets. Reported as
  `reference_lift_max_m` / `reference_lift_frames`. The resulting robot reference
  is frozen before contact acquisition; later passes cannot lift or otherwise
  redefine it. Height changes and continuity against the earlier kinematic seed
  are reported separately.
- Morphology. Terminal-chain ratios are measured from immutable bind geometry.
  When a source branch leaves too few mapped points, calibration searches the
  nearest mapped ancestor and independently verifies robot ancestry. Missing
  or incompatible geometry remains unknown, rather than inventing a ratio.
  Source adapters may declare `world_target_joints` (for example the authored
  foot path); this determines target coordinates, never support priority.
- Contacts precede clearance. As in the guarded cascade, scene clearance is a
  secondary objective that must not detach a contact: during acquisition its
  rows use 30 collision sigmas.
- Free bodies last. Freeze the attained contacts once, before any clearance
  pass. Every nonlinear proposal preserves each active endpoint/material
  sample's position and target error within the existing regression allowance
  (at most 1 mm), including unexplained holds and authored sliding paths.
  The root is fixed and each scalar joint retains the acquired per-step
  velocity, acceleration and jerk bounds. Violations backtrack inside the
  same reference instead of resetting it. The final whole-trajectory check
  remains separate. Original-mesh group repair may move the currently repaired
  contact toward its unchanged target; other groups stay anchored and the
  acquisition error budget remains frozen across both correction stages.

- Scene. Static scene and object clearance use grids sampled once from the
  existing original-triangle `SceneQuery` (2 cm scene, 1 cm objects; unreliable
  or out-of-grid nodes are unknown, never free), trilinearly interpolated and
  cached in `~/.cache/omniretargeting/sdf` (`OMNIRETARGETING_SDF_CACHE`).
  Final acceptance still audits the original triangles.
- Original-mesh repair. A separate bounded joint correction runs when a scene,
  moving object or enabled self-collision task is present, regardless of distance
  grid coverage. It uses the existing `MovingSceneQuery`:
  reliable closed-component samples and reverse witnesses from the original
  object's vertices/triangle centroids inside the robot collision hulls. No
  holes are filled and no object is convexified. Static shell penetration and
  robot self-collision are checked alongside these rows. Coverage is reported
  in `object_grid_finite_fraction`; an empty grid's zero proxy cost is not a
  statement that the object is collision-free.
  Static queries are batched up to 1,048,576 points. An unchanged `(frame, geom)`
  world transform reuses only its deepest reliable witness and unknown flag;
  changed transforms are fully re-queried and Jacobians are always recomputed.
  No sample/triangle is dropped, rounded, or replaced with a simplified mesh.
  The root stays bit-identical. The requested foot_v8 baseline rollback retains
  its 0.05 rad joint/body rotation locality (prismatic joints: 3 mm), together
  with the fixed reference layout. Making body rotations report-only
  (`body_rotation_report_only`) remains an opt-in experiment pending six-clip
  acceptance; scalar joint bounds remain in both modes. Per-joint, per-stencil
  velocity, acceleration and jerk magnitudes retain the repair-entry windowed
  budgets (7-stencil maximum plus 10% RMS) and 2e-7 numerical slack; no temporal
  budget is rebased between passes. Source posture and body rotations are reported.
  Other acquired contact points stay within 1 mm. Free endpoints may approach
  their unchanged source-derived targets, but their per-frame target errors
  cannot increase. World targets, body-relative morphology scaling and approach
  blending retain the acquisition semantics; erroneous free positions are not
  frozen as contact anchors. Reports include free-endpoint displacement separately.
  A repaired contact group can approach its unchanged world targets, with
  regression limited by the smaller of its existing tolerance and 1 mm; surface-
  managed contacts use the surface samples, not an inactive legacy endpoint.
  The corrected group is frozen before moving on. Scalar joints on its ancestor
  chain and distal compensation joints are optimized. For self-collision, both
  participants are included; affected contacts on sibling branches bring their
  compensation chains into the same block. The whole block backtracks together;
  unrelated branches remain unchanged. No posture or contact budget is expanded
  by opening these variables. Twelve proposals cover up to six static, moving-object or
  self-collision groups, ordered by the acquired evidence/role policy and then
  depth. Unattempted groups are explicitly reported. Linear constraints reserve
  numerical tolerance, while retaining zero-step feasibility after a different
  chain's accepted roundoff; nonlinear acceptance budgets remain fixed.
  Numerical solver termination only supplies a proposal: backtracking checks
  nonlinear posture/contact limits and fresh original-mesh depths before any
  candidate is retained. Accepted witnesses are reused for Jacobian construction,
  avoiding a duplicate geometry query without reducing samples. An indeterminate
  self-collision normal rejects the candidate; an indeterminate entry is retained
  with `unknown_query`, never certified collision-free. Each acquired per-frame/geom depth is protected against
  regression. `original_mesh_repair` includes before/after depths, affected
  frames, correction sizes, query failures and unresolved open components. This
  saved-frame check is not a full-volume or interframe collision certificate.
  Static-scene proposals retain at most 15 material-point witnesses per geometry:
  the deepest sample plus fixed normal and moment support probes. Their selection
  is independent of sample repetition, ordering and query batching. Each material
  point contributes one reliable exit direction; alternative exits for the same
  point must not be interpreted as simultaneous constraints. All witnesses of a
  collision key share one nonnegative maximum-depth variable, so sampling density
  does not multiply that key's objective weight. The joint regularizer remains
  relative to the immutable repair entry. A flat patch therefore cannot improve
  merely by exchanging its deepest corner for another corner. The bounded
  witness selection is a heuristic: every original sample is still checked for
  nonlinear acceptance, and unknown observations remain unknown. Numerically
  stationary proposals skip redundant line searches; residuals stay reported.
  Entry and final reports separately audit the original source/robot posture
  budgets. Inherited fast-backend excess is not a successful posture certificate,
  even when the repair's non-regression checks pass.
- Not optimized in the main least-squares solve: self-collision, body-relation,
  self-contact and surface-pose
  tasks (listed as `skipped_tasks`), and the framewise post-processing stages
  (`postprocessing_status: skipped_fast_backend`). The original-mesh fallback
  above does include bounded self-collision corrections when it is activated.

Historical measurements before the current universal guards/manifold repair
on TRUMANS (`out/solver_fix_20260925`, unchanged review scripts,
original-triangle penetration audit):

| | holdout `2023-01-16@23-06-03`, 1304 frames | walk `2023-02-19@22-38-25`, 95 frames |
|---|---|---|
| wall time, fast / guarded | 48 s / 5.4 h | 10 s / 12.9 min |
| foot surface normal mean, fast / guarded | 0.78–0.86 / 1.63–2.44 cm | 0.66–0.81 / 0.43–0.50 cm |
| worst per-joint acceleration RMS vs fixed reference | 1.009 | 1.009 |
| pelvis+hips penetration max, fast / guarded | 2.7 / 4.3 cm (29 / 53 frames over 2 cm) | 0 / 0 cm |
| max static / moving penetration, fast / guarded | 8.6 / 5.0 vs 9.3 / 6.1 cm | 6.9 / 0 vs 4.8 / 0 cm |
| max joint change, fast / guarded | 11.5 / 3.5 deg | 10.0 / 11.5 deg |
| mean bone direction error, fast / guarded | 12.3 / 11.2 deg | 11.4 / 11.6 deg |

The holdout reference rose by at most 4.8 cm over 110 sitting frames. On both
clips the free-body stage would have reduced the remaining penetration (wrists
into the table, heels into the floor: holdout 8.2 to 5.1 cm, walk 6.9 to 4.0 cm
on the grid) but added high-frequency ankle/hip motion (walk also RMS and peak),
so the residual is retained and reported. Strict per-frame continuity budgets
are not guaranteed by this backend; use `--solver sequence` when they must hold
exactly.

A runtime-only follow-up on the same pre-extracted inputs and warm SDF cache
reduced the fast pipeline from 47.70 to 37.76 s (holdout) and 9.66 to 7.07 s
(walk). It scopes single-threaded BLAS to the fast solve, transfers frame normal
equations through shared memory, caches weighted temporal operators, removes
explicit sparse zeros, and batches orientation initialization. Iteration counts,
budget counters and correction acceptance decisions are unchanged; maximum body
position differences are below 3.1e-8 m. The timing excludes initial imports/input
loading, source extraction, independent mesh audits and preview generation.

Mesh collision bodies use a support-mapping expansion in `convex_mesh_distance.py`
for self intersections. It uses the full compiled convex hull and closes a bound
on the minimum exit distance, independently of the proximity cutoff; separated
pairs return conservative supporting-plane gaps. It does not make a concave STL
collision model exact. Failed bounds raise an error, rather than reporting zero
penetration. No collision exclusions or mesh decimation are introduced.

Reports and fresh original-mesh audits: `out/efficiency_review_20260925/report.md`.
No additional user settings are required.

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

Cross-model/source validation and current limits are documented in
[the generalization review](../out/generalization_fix_20260926/report.md).
The fast report separates `layout_and_relaxed_chatter_accepted` from
`strict_continuity_accepted`. The latter checks every scalar coordinate and
root translation/rotation difference, and includes initialization versus the
kinematic seed; a supplied reference with no seed audit is not certified.
`layout_continuity_accepted` requires the windowed continuity check plus the
reference layout; posture angles are reported, not bounded. `continuity_vs_acquired` and `contacts_vs_acquired` independently check
secondary corrections; initialization changes are not erased by freezing a
new reference. These flags cover only those named checks;
contact residuals and original-geometry coverage must still be inspected.
