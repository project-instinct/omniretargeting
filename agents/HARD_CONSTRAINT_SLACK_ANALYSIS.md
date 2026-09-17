# `hard_constraint_slack` Penetration-Resolution Analysis

Date: 2026-09-04  
Last updated: 2026-09-07

## Scope

This note analyzes the current SMPL-X-to-robot retargeting pipeline after a
real-collected motion batch showed that selecting
`penetration_resolver: "hard_constraint_slack"` appeared to correct the robot
primarily by moving its floating base along Z. The desired behavior is to let
the solver distribute necessary collision corrections over the robot's usable
generalized coordinates when that produces a better pose.

This is a code-path analysis and implementation plan. The collected SMPL-X
validation cases are listed in
[`agents/computation/ziwen-galaxea-desktop.md`](computation/ziwen-galaxea-desktop.md),
but their numerical correction distribution still needs to be measured before
solver weights are selected.

## Conclusion

The hard-constraint-slack code path is not explicitly restricted to Z. With
the default `q_a_init_idx=-7`, its decision vector contains every `qpos`
coordinate, including floating-base translation, quaternion components, and
all actuated joints. Terrain-contact geometry, solver scaling, and
regularization nevertheless make floating-base Z the cheapest direction in
many frames:

1. On flat terrain, each normal-projected contact Jacobian has a direct `+1`
   coefficient for base Z, while base X/Y have zero contribution.
2. One base-Z increment moves every robot collision geometry away from the
   terrain simultaneously. Joint increments affect only their descendants and
   several joints may need to move together.
3. A single unscaled Euclidean trust region is shared by translation,
   quaternion components, and joints. A one-coordinate base-Z correction uses
   less of that trust region than a coordinated multi-joint correction.
4. Base Z has only the small default regularization weight. Base-orientation
   tracking is much stronger, and the optional base-position tracking cost
   deliberately covers X/Y but excludes Z.

Consequently, widening the existing variable set is not the fix: the actuated
joints are already present. The correction metric and configuration update
must be made physically meaningful so that coordinated articulation can
compete with base-Z translation. The state conversion must cover the whole SQP
at once: mapped-point Jacobians, the bone-direction warm start, pose
regularization, temporal smoothness, joint limits, orientation tracking,
trust-region bounds, and the accepted configuration update all currently
assume additive `qpos` coordinates.

There is also a separate resolver-wiring defect: selecting
`hard_constraint_slack` without a `penetration_slack` dictionary silently
disables slack and produces the plain hard-constraint behavior.

## Current Pipeline

### 1. SMPL-X ingestion

`SmplxDataSource` loads global joint positions and, when
`use_smplx_base_pose` is enabled, also publishes root orientation and root
translation through `MotionData`:

- [`omniretargeting/data_sources/smplx.py`](../omniretargeting/data_sources/smplx.py)

The stock Unitree G1 SMPL-X profile enables this option. Other profiles that do
not enable it discard the explicit root pose even if the raw SMPL-X input
contains one. This is not the direct cause of Z-only collision response, but it
determines whether source-root tracking can provide a base-position reference.

### 2. Resolver selection

`OmniRetargeter.create_stream_state()` maps the configured resolver as follows:

- `hard_constraint`: enable QP penetration constraints without slack.
- `hard_constraint_slack`: enable QP penetration constraints and pass the
  optional `penetration_slack` dictionary.
- `xyz_nudge`: disable QP penetration constraints and use batch-only base
  translation post-processing.

Relevant implementation:

- [`omniretargeting/core.py`](../omniretargeting/core.py)

The current slack activation is indirect. `core.py` passes
`retargeting_config.get("penetration_slack")`; the inner retargeter enables
slack only when that value is not `None`. Therefore this configuration:

```yaml
penetration_resolver: hard_constraint_slack
```

does not enable slack unless it is accompanied by at least:

```yaml
penetration_slack: {}
```

The documented default values are stored by the inner solver, but they are not
used in slack mode when the dictionary is absent. The stock profiles also do
not contain a `penetration_slack` block, so changing only the resolver through
the CLI exposes this defect.

### 3. SQP state parameterization

`GenericInteractionRetargeter._setup_robot_config()` calculates:

```python
start_idx = 7 + q_a_init_idx
q_a_indices = np.arange(start_idx, nq)
```

The default `q_a_init_idx=-7` gives `start_idx=0`, so the optimization includes
the full `qpos` array. There is currently no configuration passed by
`OmniRetargeter` that narrows this set.

The distinction between `qpos` and physical generalized DOFs is important:

- A free base occupies seven `qpos` values: XYZ plus a four-component
  quaternion.
- The free base has six physical tangent DOFs in `qvel`: XYZ plus a
  three-component angular increment.

The current solver operates on the former rather than the latter.

### 4. Penetration constraints

For self-collision, the solver computes the relative point Jacobian
`J_body_a - J_body_b` and projects it onto the contact normal. Rigid base
translation cancels from that relative Jacobian, so self-collision cannot be
resolved by translating the whole robot in Z.

For robot-terrain collision, representative points on every collision geometry
are queried against the terrain mesh. A constraint row is constructed as:

```text
n^T J(q) dqa >= -signed_distance - tolerance
```

In slack mode, each queried pair produces:

```text
phi(q)                 >= -hard_bound
phi(q) + slack         >= -soft_tolerance
0 <= slack             <= hard_bound - soft_tolerance
```

with a quadratic slack penalty. Slack changes feasibility and the permitted
penetration range; it does not change which robot coordinates are available.

For a horizontal surface, `n=[0,0,1]`. The base-translation portion of the
terrain row is therefore `[0,0,1]`. Joint columns can be nonzero, but only for
joints upstream of the sampled body and only when their instantaneous motion
has a component along the surface normal.

### 5. QP objective and update

The per-iteration objective combines:

- interaction-mesh Laplacian error;
- optional bone-direction error;
- coordinate regularization toward zero, with optional per-joint boosts;
- temporal smoothness relative to the preceding frame;
- base-orientation tracking;
- optional X/Y base-position tracking; and
- penetration slack penalty.

The QP applies one Euclidean trust-region norm to all `qpos` increments. It then
adds the optimized increments directly to `qpos` and normalizes the quaternion.
This has two consequences:

1. Meters, radians, and redundant quaternion-component increments share an
   unscaled norm.
2. Quaternion normalization changes the accepted pose after the collision
   constraint was linearized, so the applied rotation is not exactly the
   rotation predicted by the QP.

The optional bone-direction warm initialization uses the same additive update
and quaternion normalization, so changing only the main penetration rows would
leave the solver internally inconsistent.

The outer SQP loop currently checks convergence using objective-value change
before committing the proposed `q_new`. It does not recompute nonlinear
collision distances for the integrated candidate. Therefore a small objective
change can stop the loop without proving that the returned pose satisfies the
configured hard bound.

## Findings

### Finding 1: the observed Z dominance is an optimization-allocation issue

All actuated joints are already represented in the QP. The terrain rows and
solver metric make base Z a universal, low-cost correction direction. This is
especially strong when several terrain-contact rows are active in the same
frame: one Z increment improves all of them, while a joint-space solution may
require multiple coordinated changes.

This also provides a useful diagnostic separation:

- Z-dominant correction for terrain penetration is consistent with the current
  objective and trust region.
- Z-only correction attributed to self-collision means the relevant
  self-collision constraint was not active or was not detected, because rigid
  translation cancels from the self-collision Jacobian.

### Finding 2: `hard_constraint_slack` can silently become `hard_constraint`

The resolver name and numeric-parameter presence are used as two different mode
switches. The outer layer selects slack mode by name, while the inner layer
selects it by `penetration_slack is not None`. These switches disagree when the
parameter block is omitted.

This should be fixed independently of the correction-allocation change because
it directly violates the public resolver semantics.

### Finding 3: the solver uses ambient quaternion coordinates instead of true DOFs

MuJoCo point Jacobians are naturally expressed against `qvel` (`nv`). The
current implementation maps them into `qpos` (`nq`), optimizes four independent
quaternion increments, adds them linearly, and normalizes afterward. Valid
quaternion motion has only three rotational tangent DOFs.

This prevents the solver from treating base rotation and joint articulation in
a consistent physical coordinate system. It can also invalidate a satisfied
linearized constraint after normalization.

### Finding 4: the trust-region metric favors sparse single-coordinate motion

The constraint `||dqa||_2 <= step_size` uses one scale for base translation,
base rotation, and all joint changes. Besides mixing incompatible units, the L2
norm makes coordinated motion across several joints more expensive than a
similar workspace displacement produced by one base coordinate.

The solver needs block or per-DOF scaling, ideally based on physical units and
joint ranges.

### Finding 5: terrain collision samples are inaccurate for cylinders and capsules

The terrain sampler assumes capsule and cylinder axes are local X. MuJoCo
capsules and cylinders use local Z. The sampled endpoints and side points can
therefore lie away from the actual surface represented by MuJoCo.

This can produce incorrect signed distances and contact Jacobians, particularly
for profiles whose collision models rely heavily on cylinders or on cylinders
converted to capsules.

There is already a second primitive sampler in
`OmniRetargeter._sample_geom_points_in_body_frame()` that uses local Z
correctly for capsules and cylinders and also handles ellipsoids. Keeping two
independent samplers has allowed the terrain-constraint and foot-stabilization
paths to disagree. The primitive point construction should be shared; each
caller can then apply the transform it needs.

### Finding 6: convergence does not establish nonlinear feasibility

The QP constraints apply to a first-order model at the current configuration.
After the solver adds an ambient increment and normalizes the quaternion, it
does not run forward kinematics and verify the actual collision distances of
the candidate pose.

The outer loop also checks objective-value convergence before assigning
`q_new` to `q`. A proposed step can therefore be discarded when the objective
change is small, even if that step improves feasibility. Convergence should be
based on the applied tangent step and the nonlinear hard-bound residual, not
only on successive objective values.

### Finding 7: existing tests do not exercise correction distribution

The current tests cover the scalar slack equations, invalid slack bounds, and
configuration propagation. They do not verify:

- that selecting slack mode without overrides enables its defaults;
- which generalized coordinates have support in a real contact row;
- whether an articulated solution can be preferred over base Z;
- whether the nonlinear pose remains within the hard bound after applying the
  SQP update; or
- whether self-collision correction is invariant to rigid translation.

## Fixing Plan

### Phase 1: reproduce and add solver diagnostics

Use cases 1, 3, and 4 from
[`agents/computation/ziwen-galaxea-desktop.md`](computation/ziwen-galaxea-desktop.md)
to retain a small set of representative failure frames, then use case 5 for
batch validation. For each representative frame, run the same initial pose
through no penetration constraints, `hard_constraint`, and
`hard_constraint_slack`, with scene scaling both enabled and disabled where the
case instructions request it.

The no-penetration baseline must disable QP penetration rows and the
`xyz_nudge` post-process; `xyz_nudge` itself is not an unconstrained baseline.

Measure penetration-induced correction as the tangent-space difference from
the no-penetration result to each constrained result. Also retain the
difference from the common initial pose so optimization-path changes are not
mistaken for collision correction.

Record:

- pose difference in MuJoCo tangent coordinates using `mj_differentiatePos`;
- correction norms for base XYZ, base rotation, legs, waist, and arms;
- terrain and self-collision rows separately;
- per-block norms of every active constraint Jacobian;
- hard-bound and soft-tolerance residuals;
- optimized slack values;
- nonlinear signed distances after forward kinematics; and
- solver status, retry count, iterations, and runtime.

Diagnostics should be opt-in and structured; do not add unconditional
per-frame printing to the production path. This phase establishes whether the
failure is primarily objective weighting, weak or incorrect joint Jacobians,
missing collision samples, or a combination. Record the quantitative
acceptance thresholds before weight tuning begins.

### Phase 2: correct resolver semantics

Make the selected resolver the single source of truth.

Recommended minimal behavior:

- When the resolver is `hard_constraint_slack`, pass an empty dictionary if no
  numeric block is configured, so the documented defaults are active.
- When the resolver is not `hard_constraint_slack`, do not enable slack even if
  stale numeric parameters exist.
- Validate that `penetration_slack` is a dictionary, every value is finite,
  `soft_tolerance >= 0`, `hard_bound > soft_tolerance`, and
  `slack_penalty > 0`.

Add a regression test that selects only `hard_constraint_slack` and verifies
that the inner solver has slack enabled with the default tolerance, bound, and
penalty. Keep the existing test that stale slack parameters do not activate
slack for `hard_constraint`, and update the README resolver/configuration list.

This is an independent semantic bug fix and should land before the solver
refactor.

### Phase 3: repair and unify primitive collision sampling

Extract the primitive point construction currently duplicated between
`core.py` and `retargeting.py` into one small local-frame helper, preferably
alongside the existing geometry utilities in `omniretargeting/utils/math.py`:

- sample cylinder/capsule axes along local Z and rings in local XY;
- include the correct capsule poles and cylinder end caps;
- retain sphere and box support and add the existing ellipsoid support;
- let foot stabilization transform geom-local points into the body frame;
- let terrain constraints transform the same geom-local points with the current
  MuJoCo geom world pose.

Add unit tests that apply arbitrary geom rotations and confirm that the
transformed cylinder, capsule, and ellipsoid samples lie on their represented
surfaces. Do not add mesh sampling or row deduplication in this phase unless
Phase 1 shows that mesh geoms or duplicate rows materially affect the reported
failures.

### Phase 4: convert the complete SQP to true generalized increments

Change the SQP decision variable from an ambient `qpos` increment to a tangent
increment in `nv`:

```text
delta_v = [base_translation(3), base_rotation(3), actuated_joint_deltas]
```

Implementation direction:

1. Keep MuJoCo point Jacobians in their native `3 x nv` form.
2. Replace `q_a_indices` with optimized physical DOF indices. Preserve the
   intended `q_a_init_idx` behavior by mapping selected joints through
   `jnt_qposadr` and `jnt_dofadr`; the floating base must be selected as all six
   tangent DOFs or not selected.
3. Remove `_build_transform_qdot_to_qvel_fast()` and all `J_v @ T` conversions.
4. Apply an accepted increment by embedding it in a full `nv` vector and
   calling `mujoco.mj_integratePos()`.
5. Express current-to-reference configuration errors with
   `mujoco.mj_differentiatePos()`. Use a three-dimensional tangent/SO(3)
   residual for base-orientation tracking.
6. Convert the bone-direction warm start, coordinate regularization, temporal
   smoothness, and their objective-value calculations in the same change.
7. Map hinge/slide joint bounds through their `jnt_qposadr` and `jnt_dofadr`
   addresses; floating-base limits come from the step policy rather than fake
   large `qpos` bounds.

This makes "all DOFs" literal and removes the post-solve quaternion
normalization mismatch. Add a finite-difference regression showing that a
native point Jacobian predicts the first-order displacement produced by
`mj_integratePos()`.

This phase should be kept as behavior-preserving as practical. Land the
coordinate-system correction before tuning the allocation policy so geometry,
parameterization, and weighting regressions remain distinguishable.

### Phase 5: introduce a scaled full-state correction metric

Keep all robot DOFs available, but replace the single unscaled metric with
configurable blocks:

```yaml
penetration_correction:
  base_translation_weights: [wx, wy, wz]
  base_rotation_weight: wr
  joint_weight: wj
  joint_range_normalization: true
  base_translation_step: ...
  base_rotation_step: ...
  joint_step_fraction: ...
```

The exact public names can be adjusted to the project's configuration style,
but the semantics should be explicit:

- Base Z receives a strong but finite prior around the retargeted/reference
  base height.
- Base X/Y and rotation receive independent weights.
- Joint increments are normalized by their legal ranges so different joints
  are comparable.
- Translation, rotation, and joint step limits are expressed in meters,
  radians, and fractions of joint range, respectively.
- Do not use one unscaled all-state L2 trust region. Use scaled tangent
  coordinates or separate block/per-DOF limits so a coordinated multi-joint
  correction is not rejected merely because it has support in several
  coordinates.

Use `MotionFrame.root_translation` as the base-position reference when it is
available; otherwise use the configuration at frame entry as the generic
fallback. Keep that reference fixed across all SQP iterations for the frame.
Do not add SMPL-X-specific structures to `GenericInteractionRetargeter`.

The new metric must have defined precedence relative to the existing
`base_position_tracking_weight` and `joint_regularization_boost` settings so
the same state term is not counted twice. Preserve the existing configuration
behavior when the new block is absent, then choose and document the corrected
defaults from the Phase 1 measurements.

Base Z must not be frozen: it remains necessary when no articulated solution
can satisfy the collision constraints.

The initial implementation should keep this metric inside the existing SQP.
A separate post-projection subsystem should only be considered if batch results
show that the interaction objective and penetration response cannot be balanced
in one solve.

### Phase 6: enforce nonlinear acceptance and meaningful convergence

After integrating every proposed tangent update:

1. run `mj_forward()` on the candidate configuration;
2. recompute terrain and self-collision distances, including contacts that were
   not active at the previous linearization point;
3. apply `q_new` before evaluating convergence;
4. require both a small scaled tangent step and a satisfied nonlinear hard
   bound before declaring convergence; and
5. continue SQP iterations when penetration is improving but still infeasible.

If a full step worsens the nonlinear residual, backtrack it before
relinearizing. If the iteration budget is exhausted without a hard-feasible
pose, record an explicit solver failure rather than silently counting the frame
as accepted. This failure path and its batch behavior must be visible in the
diagnostics without changing the returned `qpos` representation.

### Phase 7: regression and batch validation

Add focused tests:

1. **Resolver-default test:** `hard_constraint_slack` without a parameter block
   enables the default slack settings.
2. **Bent-leg terrain test:** a penetrating foot produces nonzero hip, knee,
   and ankle Jacobian columns as well as base-Z support.
3. **Correction-allocation test:** with a strong but finite base-Z weight, the
   solver reduces penetration through articulated joints while keeping base-Z
   motion bounded.
4. **Self-collision test:** rigid translation has zero effect on the relative
   collision row and the solution uses articulated DOFs.
5. **Quaternion/tangent test:** applying the optimized tangent increment
   matches the predicted first-order point displacement.
6. **Primitive-sampling test:** sampled cylinder/capsule points lie on the
   correct MuJoCo local-Z surface.
7. **Nonlinear feasibility test:** after applying an SQP update and running
   forward kinematics, the complete recomputed contact set respects the hard
   bound before convergence is reported.

Run the unit and full suites on `marsbrain`; do not use the local Mac as the
verified test environment. Then run A/B validation on the collected SMPL-X
cases and batch on `ziwen-galaxea-desktop`, using its `kengo` profile without
modifying or copying that profile. Compare:

- maximum penetration depth;
- percentage of frames beyond the soft tolerance;
- hard-bound violations;
- base-Z displacement relative to the source/reference base;
- correction energy by robot DOF group;
- Laplacian/keypoint tracking error;
- temporal joint velocity and acceleration;
- joint-limit violations;
- solver failure rate; and
- runtime per frame.

## Acceptance Criteria

Freeze numerical quality/runtime thresholds after Phase 1 and before parameter
tuning. The fix is successful when:

1. `hard_constraint_slack` always means that slack variables are active,
   regardless of whether numeric overrides are supplied.
2. No accepted frame violates the configured hard penetration bound after
   nonlinear forward kinematics.
3. In representative failure frames where articulation is feasible, collision
   correction is distributed over relevant joints instead of being almost
   entirely base Z.
4. Base Z remains available for genuinely infeasible articulated cases but is
   no longer the universally cheapest response.
5. Interaction/keypoint fidelity and temporal smoothness do not regress enough
   to negate the penetration improvement.
6. The batch completes without a material increase in solver failures; runtime
   impact is measured and reported.

An infeasible frame that is explicitly reported as a solver failure does not
violate criterion 2, but its frequency is counted by criterion 6 and must not
be hidden by returning a best-effort pose as a successful solve.

## Implementation Boundaries

- Do not widen the optimization variable set; the actuated joints are already
  present.
- Do not add source-specific SMPL-X state to the generic math engine.
- Do not add a separate post-projection subsystem in the initial fix.
- Do not add dependencies for this work.
- Defer mesh sampling and constraint-row deduplication unless the measured
  failure frames show that they are necessary.

## Recommended Implementation Order

1. Add diagnostics and reproduce representative frames.
2. Fix slack-mode activation and its regression test.
3. Correct primitive collision sampling.
4. Convert the complete SQP to `nv` tangent increments.
5. Add the block-scaled state prior and step policy.
6. Enforce nonlinear acceptance and convergence.
7. Run synthetic regressions and the real SMPL-X batch comparison.

This order separates the resolver bug and collision-geometry errors from the
larger state-parameterization change, while ensuring that final weight tuning
is performed on correct contact rows and physically meaningful robot DOFs.
