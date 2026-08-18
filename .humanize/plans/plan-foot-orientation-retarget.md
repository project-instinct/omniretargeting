# SMPL-X-Primary Sole Orientation Tracking For Retargeted Humanoid Feet

## Goal Description

Retargeted humanoid feet currently collapse into a toe-vertical (tiptoe) pose that is nearly collinear with the shin, because the optimizer has no per-link orientation objective and the foot carries a single, badly placed positional marker. This plan makes the retargeted sole reproduce the source SMPL-X foot orientation.

The source SMPL-X motion is authoritative. For the reference clip it was derived from the real robot's own motion via a video-based human-mesh-recovery pipeline, so the source foot orientation is by construction kinematically achievable by this robot. There is no human-to-robot morphology gap to absorb, and the observed error is therefore attributable to the solver rather than to feasibility.

Three strictly ordered levels govern the foot:

- **L1 (hard constraint)**: the foot does not penetrate the terrain.
- **L2 (primary objective)**: the sole matches the calibrated SMPL-X source foot frame.
- **L3 (secondary objective)**: during detected contact, the sole normal aligns to the **local terrain surface normal**. Local normal, not world up, because the reference clip is a wall-flip in which the contacted surface is frequently a wall.

When L2 and L3 conflict, L2 wins. This is the inverse of the game-animation foot-IK convention, where surface alignment is the primary goal because no ground-truth pose exists; here a ground-truth pose does exist and it is the SMPL-X source.

Work lands on a clean feature branch cut from `origin/main`. The capability is ported; the local branch's CVXPY solver is not. The local branch serves as design reference and test oracle only.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Solver hygiene and determinism are corrected, each as an independently baselined change.
  - Positive Tests (expected to PASS):
    - Two full runs of the reference clip with the same configured seed produce identical retargeted joint arrays.
    - The SQP loop returns the iterate computed at the step that satisfied the convergence test, not the preceding iterate.
    - The scalar objective used for the stopping decision includes every term present in the QP matrices, verified by a test that adds a synthetic term and asserts both assemblies change together.
    - A solver failure at a linearization point results in a recorded failure or a shrunk-trust-region retry, and the Clarabel status is exposed in diagnostics.
  - Negative Tests (expected to FAIL):
    - A run with an unset seed is rejected by the reproducibility check rather than silently producing different terrain samples.
    - A term added only to the quadratic matrices, without a matching scalar contribution, fails the parity test.
    - A silent re-solve that drops the trust region without recording the fallback fails the diagnostics assertion.
  - AC-1.1: The stopping rule is specified, not left to an objective delta across changing linearizations.
    - Positive: The loop stops when the infinity norm of the actuated step falls below 1e-5, or when the objective re-evaluated at the new configuration by forward kinematics improves by less than 1e-6 relative to its value at the previous configuration; both quantities are computed through the shared residual helper.
    - Negative: A stopping decision based only on comparing linearized predictions from two different linearization points fails review.
  - AC-1.2: Body-shape parameters are resolved to a single landed behavior.
    - Positive: The default SMPL-X source entry consumes the clip's own body-shape parameters; the resolved parameters, the derived source-to-robot scale, the solver iteration limits, the seed, and the feature configuration all appear in diagnostics.
    - Negative: A configuration that silently overrides the clip's sixteen shape parameters with a shorter all-zero vector fails validation.

- AC-2: A frame contract exists on both sides and is unit-testable independently of the solver.
  - Positive Tests (expected to PASS):
    - The source contract maps the SMPL-X toe joint frame to an anatomical sole frame, calibrated once on the shaped template using the thinnest principal axis as sole normal and the ankle-to-toe vector projected into the sole plane as forward, re-orthogonalized.
    - Both calibrated frames are orthonormal with determinant +1.
    - The robot contract is an explicitly configured body-frame-to-sole-frame transform; for the reference robot it reproduces the measured identity rotation.
    - Quaternion ordering, multiplication order, world-versus-body convention, and forward/up sign conventions are asserted by tests rather than assumed.
  - Negative Tests (expected to FAIL):
    - Treating the raw SMPL-X toe joint frame as the sole frame fails, because the two differ by roughly twenty-four degrees, dominated by toe-out yaw.
    - A robot contract inferred from collision geometry without passing degeneracy and confidence checks is rejected.
    - Conflating the joint-frame-to-sole-frame rotation with the elevation of the single ankle-to-toe vector fails the convention test; these are distinct quantities.

- AC-3: The primary SMPL-X orientation objective (L2) exists in the solver and demonstrably drives the feet.
  - Positive Tests (expected to PASS):
    - A soft quadratic orientation residual against the calibrated source sole frame is assembled into the quadratic matrices and the scalar objective through the single shared helper.
    - The body angular Jacobian passes a finite-difference test, including at non-identity floating-base orientations.
    - Enabling the term measurably reduces sole orientation error on the reference clip relative to the post-hygiene baseline.
  - Negative Tests (expected to FAIL):
    - A residual present in the quadratic matrices but absent from the scalar objective fails the parity test of AC-1.
    - A first-order angular linearization asserted to equal the nonlinear residual at arbitrary finite step size fails; the correct assertion is exact value and gradient at zero step with second-order decay of linearization error.
    - A rotation-error formulation that misbehaves near a half-turn fails the near-singularity test.
  - AC-3.1: Root rotation cannot be used to satisfy foot orientation.
    - Positive: The foot orientation residual is formulated root-relative and coupled to the existing world base-orientation target, so rotating the floating base cannot reduce relative ankle orientation error.
    - Negative: Masking floating-base columns out of this one term while other terms continue to co-optimize the root is rejected as an inconsistent linearization.

- AC-4: Terrain conformity (L3) is present, contact-gated, angle-limited, and provably subordinate to L2.
  - Positive Tests (expected to PASS):
    - The secondary term targets the local terrain surface normal at the contact location.
    - The term is gated by detected contact phase and blends to zero during swing and aerial frames.
    - A maximum correction angle bounds how far the term may rotate the foot per solver step.
    - Both residuals are normalized to dimensionless form, and the secondary effective coefficient is at most one tenth of the primary.
    - On the subset of frames where the L2 and L3 target normals disagree by more than fifteen degrees, enabling L3 increases the L2 sole-normal error median by no more than two degrees.
  - Negative Tests (expected to FAIL):
    - An ungated term that rotates airborne feet toward the ground fails the swing-phase assertion.
    - Using world up instead of the local surface normal fails on wall-contact frames.
    - A configuration in which L3 degrades L2 beyond the conflict-frame bound is rejected, including any configuration that merely assumes a smaller scalar weight implies subordination.

- AC-5: The positions-only path degrades explicitly rather than silently.
  - Positive Tests (expected to PASS):
    - When source joint orientations are unavailable, a directional residual on the robot forward axis is used, linearized through the skew-symmetric form of that axis rather than the full angular Jacobian.
    - The directional Jacobian passes its own finite-difference test.
    - Roll is documented and reported as untracked on this path, and a fallback warning is emitted once per sequence.
  - Negative Tests (expected to FAIL):
    - Synthesizing an up vector to fabricate a full orientation target on a positions-only source is rejected.
    - Emitting the fallback warning once per frame fails the log-volume assertion.

- AC-6: Configuration places the fix where the defect actually occurred.
  - Positive Tests (expected to PASS):
    - Orientation settings live under the source and mapping entry, so different sources select different capability levels independently.
    - The generic default is disabled, and the default reference-robot SMPL-X source entry explicitly enables full orientation tracking.
    - A single effective coefficient controls the quadratic term; no redundant weight-and-sigma pair with undefined separate roles.
  - Negative Tests (expected to FAIL):
    - A fix that is implemented but left disabled in the profile that produced the defect fails acceptance.
    - A profile that omits orientation configuration and yet changes behavior fails the legacy-preservation test.

- AC-7: The reference clip meets stratified orientation thresholds against the calibrated source frame.
  - Positive Tests (expected to PASS):
    - The acceptance metric is SO(3) geodesic error between the retargeted sole frame and the calibrated source sole frame, so yaw cannot escape measurement.
    - Thresholds are evaluated on both raw solver output and final post-stabilization output, and both must pass.
    - Contact frames: median geodesic error at most ten degrees, ninetieth percentile at most twenty degrees.
    - Swing and aerial frames: median at most fifteen degrees, ninetieth percentile at most twenty-five degrees.
    - Median geodesic error improves by at least sixty percent relative to the post-hygiene baseline, on both raw and final outputs.
    - Sole-normal error and forward-axis error are reported alongside as diagnostics, decomposed by swing-twist rather than Euler angles.
  - Negative Tests (expected to FAIL):
    - Reporting only sole-normal error, allowing a yaw-only failure to pass, is rejected.
    - Passing on post-stabilization output while raw solver output remains out of threshold is rejected.
    - An unstratified aggregate that hides poor contact-phase behavior behind good aerial-phase numbers is rejected.

- AC-8: Feasibility and non-regression are preserved.
  - Positive Tests (expected to PASS):
    - The landed reference-robot configuration retains the hard non-penetration resolver; slack and post-hoc nudge modes exist only as ablations.
    - Contact-phase penetration on both raw and final outputs has ninety-fifth percentile at most five millimetres and maximum at most twenty millimetres, with contact phase, including wall contact, defined before the criterion is applied.
    - Ankle saturation is reported within 1e-4 radians of a bound and is compared against orientation error, so limits demanded by the source are not counted as defects.
    - Ninetieth-percentile position error of previously mapped bodies worsens by at most ten percent or twenty millimetres, whichever is stricter.
    - Torso-pitch saturation, a separate pre-existing issue, increases by at most five percentage points.
  - Negative Tests (expected to FAIL):
    - Landing a relaxed penetration mode to make orientation numbers look better is rejected.
    - Counting saturation as failure on frames where the source itself demands the limit is rejected.
    - A run that improves foot orientation while degrading hand or pelvis tracking beyond tolerance is rejected.

- AC-9: The secondary corpus is exercised with a pinned, reproducible manifest.
  - Positive Tests (expected to PASS):
    - The manifest is the first one hundred filenames, in lexicographic order, matching the base-motion suffix in the object-interaction corpus, excluding rotation and translation augmentations.
    - The metric on this corpus is forward-axis angular error between the robot toe axis and the source ankle-to-toe direction.
    - Median forward-axis error improves by at least forty percent against the same-seed baseline.
    - Contact-phase penetration ninety-fifth percentile worsens by at most two millimetres, and ankle saturation increases by at most five percentage points.
  - Negative Tests (expected to FAIL):
    - Sampling sequences non-deterministically, or including augmented duplicates of the same base motion, is rejected.
    - Asserting roll accuracy on this corpus is rejected, because roll is unobservable from positions alone.

- AC-10: Instrumentation exists without disturbing the published artifact.
  - Positive Tests (expected to PASS):
    - Diagnostics are written to a separate sidecar file; the existing five-key output schema is preserved exactly.
    - A diagnostic tool reports, per frame and per foot, the calibrated source sole frame, the robot sole frame, the geodesic and component errors, contact phase, terrain clearance and penetration, active-limit flags, per-term residuals, and solver status.
    - With the feature disabled, the five legacy arrays are exactly equal to the baseline under the same seed and the same preceding solver changes.
  - Negative Tests (expected to FAIL):
    - Adding diagnostic keys to the published output by default is rejected.
    - Interpreting feature-off identity as byte equality of the archive is rejected, because archive metadata differs regardless of content.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation lands solver hygiene fixes as separately baselined changes; a source-side anatomical sole-frame calibration and an explicitly configured robot-side sole transform; a root-relative SMPL-X sole orientation residual assembled into both the quadratic matrices and the scalar objective through one shared helper; a contact-gated, angle-limited terrain-normal alignment term at strictly lower priority; a directional fallback for positions-only sources; a derived toe positional marker expressed through the existing body-local offset mechanism; a diagnostic sidecar and a standalone foot-pose diagnostic tool; unit tests for Jacobians, frame validity and near-singular rotation; and stratified acceptance measurement on the reference clip plus a pinned hundred-sequence secondary corpus.

### Lower Bound (Minimum Acceptable Scope)

The implementation lands the solver-hygiene fixes needed to make measurement meaningful (seeded terrain sampling, step acceptance at convergence, objective parity between quadratic matrices and scalar cost, resolved body-shape parameters); the source and robot frame contracts; the root-relative SMPL-X orientation residual enabled in the default reference-robot SMPL-X source entry; the diagnostic tool sufficient to evaluate the acceptance thresholds; and the reference-clip acceptance measurement. The terrain-conformity term, the toe positional marker, and the secondary-corpus sweep may follow, provided the reference-clip thresholds are met and non-regression holds.

### Allowed Choices

- Can use: the existing body-local offset mechanism in the target mapping, which already applies the correct rotated offset and cross-term Jacobian, as the toe marker; the local branch's angular Jacobian construction and template sole-frame calibration as ported design references; the existing penetration resolver modes as ablations; the existing contact-detection machinery on the local branch as a reference for contact gating; a general multi-cone extension of the Clarabel wrapper if per-foot angular rate limits are required; game-animation foot-IK structure, specifically contact-phase blend weighting and a maximum correction angle.
- Cannot use: a new link in the robot description as the toe frame, because fixed massless links are fused away during compilation and do not appear as addressable bodies; a change that alters the actuated joint count, ordering, or names; mandatory inference of the robot sole frame from collision geometry as the generic contract; a relaxed penetration mode in the landed configuration; a synthesized up vector on positions-only sources; the game-animation convention of treating surface alignment as the primary objective; column masking of floating-base terms as the sole remedy for base leakage; byte equality of output archives as a definition of behavioral identity.

> **Note on Deterministic Designs**: If the draft specifies a highly deterministic design with no choices (e.g., "must use JSON format", "must use algorithm X"), then the path boundaries should reflect this narrow constraint. In such cases, upper and lower bounds may converge to the same point, and "Allowed Choices" should explicitly state that the choice is fixed per the draft specification.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Per frame, per foot, one possible path:

```
# --- calibration, computed once ---
R_src_joint_to_sole = calibrate_from_shaped_template(betas)   # sole normal from thinnest
                                                              # principal axis; forward from
                                                              # ankle->toe projected into plane
R_rob_body_to_sole  = configured_transform(robot_profile)      # identity for the reference robot

# --- per frame ---
R_target_world = R_src_joint_world[foot] @ R_src_joint_to_sole
R_current_world = R_body_world[foot_body] @ R_rob_body_to_sole

# root-relative, so rotating the base cannot satisfy this term
R_target_rel  = R_root_world.T @ R_target_world
R_current_rel = R_root_world.T @ R_current_world
e = rotvec(R_target_rel @ R_current_rel.T)

Jr_rel = angular_jacobian(foot_body) - angular_jacobian(root_body)
add_quadratic_residual(w_L2, Jr_rel, e)        # -> matrices AND scalar cost, one helper

# secondary, contact only
if in_contact(frame, foot):
    n_terrain = local_surface_normal(contact_point)
    n_sole    = R_current_world[:, 2]
    e_align   = clip_angle(rotvec_between(n_sole, n_terrain), max_correction_angle)
    add_quadratic_residual(w_L3 * blend(frame), Jr_rel, e_align)   # w_L3 <= w_L2 / 10
```

The toe positional marker needs no model change: the target mapping already accepts a body-local offset and applies it with the correct rotated offset and cross-term Jacobian, which is exactly the toe frame that the reference implementation expresses as a dedicated body.

Measured geometry for the reference robot, to be recomputed once body-shape parameters are resolved: the sole plane sits at local z of -0.035 with the foot spanning x from -0.054 to +0.132; the source toe joint sits inside the foot volume rather than on the sole surface, roughly 0.014 metres above the sole after scaling, and the source ankle roughly 0.058 metres above it. Solving for a self-consistent pair that matches segment length, direction, and both endpoint heights above the sole yields a toe marker near [0.0974, 0, -0.0214] and an ankle anchor near [0, 0, +0.0234]. The reference implementation's independently hand-tuned toe frame sits at [0.1, 0, -0.02], agreeing to within three millimetres.

### Relevant References

- `omniretargeting/retargeting.py` (upstream) - the SQP; contains the objective assembly, the separate scalar-cost recomputation, the Clarabel wrapper with its single second-order cone, the actuated-index construction that includes the floating base, and the convergence break that currently discards the final step.
- `omniretargeting/core.py` (upstream) - target subsetting that drops unmapped source joints, joint-range-midpoint initialization, per-frame warm start, and the translation-only foot stabilization post-process.
- `omniretargeting/utils.py` (upstream) - unseeded terrain surface sampling.
- `omniretargeting/data_sources/smplx.py` (upstream) - emits the toe joints and the full joint orientations that are currently computed and discarded.
- `robot_models/unitree_g1/unitree_g1.json` (upstream) - the source entries; sibling non-SMPL-X sources already declare toe targets with forward offsets, which is the in-file precedent. Also holds the all-zero body-shape override and the reduced solver iteration limits.
- `omniretargeting/retargeting.py`, `omniretargeting/utils.py`, `omniretargeting/contact.py` (local branch) - the angular Jacobian, contact and swing orientation residuals, the angular step cap, the shaped-template sole-frame calibration, and sole-normal versus surface-normal gating. Design reference and oracle.
- `cankao/GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json` and `cankao/GMR/general_motion_retargeting/motion_retarget.py` - the closest external precedent: a full pose task per link with a position cost and an orientation cost, a per-link frame-offset quaternion, the SMPL-X toe joint mapped to a dedicated toe frame, and the highest orientation weight in the configuration assigned to the feet.
- `cankao/GMR/assets/unitree_g1/g1_mocap_29dof.xml` - the dedicated toe frame, declared natively so that it survives compilation.
- `cankao/robot_retargeter/config/robot/g1.yaml` - supporting evidence for per-link rotation weights and for modelling each foot with more than one contact point.
- `scripts/align_retargeted_omomo_ground.py` - existing conventions for loading retargeted output into the model and for foot geometry selection.
- `tests/test_contact_priority.py` (local branch) - fixture style for model-dependent tests with graceful skipping.

## Dependencies and Sequence

### Milestones

1. **Solver hygiene and determinism**: make measurement meaningful. Each item is its own change with its own baseline; this milestone explicitly does change behavior and must not be described as neutral.
   - Phase A: thread an explicit seed into terrain surface sampling and additionally record the sampled points in diagnostic runs, so results survive dependency version drift.
   - Phase B: accept the newly computed configuration before declaring convergence; retain the existing non-finite guard, which correctly refuses a bad step.
   - Phase C: centralize quadratic residual assembly and scalar evaluation in one helper, and route the existing base-position tracking term through it, closing the pre-existing omission where that term reaches the matrices but not the cost.
   - Phase D: replace the objective-delta stopping rule per AC-1.1.
   - Phase E: remove the silent trust-region-free retry; shrink and retry or record a failure, and expose solver status.
   - Phase F: define initialization as one contract, starting from the model's nominal configuration and optionally applying the source's declared pose, then initializing the floating base; land one specified behavior.
   - Phase G: resolve body-shape parameters per AC-1.2.

2. **Baseline and ablation**: a time-boxed measurement spike that informs tuning and scope but does not gate the orientation work.
   - Step 1: seeded baseline on the unmodified branch point.
   - Step 2: post-hygiene baseline, reported separately.
   - Step 3: separate ablations for solver iteration budget, initialization variant, penetration resolver mode, and body-shape parameters. The iteration budget matters because the landed profiles reduce it well below the code defaults, and combined with the discarded-step defect each later frame accepts fewer steps than intended.
   - Step 4: proceed to the frame contract and orientation work regardless of outcome; ablations may retune or rescope, never cancel.

3. **Frame contract**: solver-independent and unit-testable, per AC-2. Depends on the resolved body-shape parameters from Milestone 1 Phase G, because the calibration is computed on the shaped template.

4. **Primary orientation objective**: the root-relative SMPL-X sole residual, per AC-3. Depends on Milestone 3 for its target and on Milestone 1 Phase C for the shared assembly helper. Requires plumbing full joint orientations as a validated, frame-indexed input through the motion data structures, resampling, scaling, and name mapping, rather than passing the whole trajectory array on every frame.

5. **Secondary terrain conformity and positional markers**: per AC-4. Depends on Milestone 4, because subordination can only be demonstrated once the primary term exists. Contact-phase definition, including wall contact, is a prerequisite for both this milestone and the penetration criterion.
   - Step 1: contact-phase detection and the local surface normal at the contact location.
   - Step 2: the gated, angle-limited alignment term with normalized residuals.
   - Step 3: positional marker experiments as separate ablations, toe marker alone first.

6. **Validation**: per AC-7 through AC-10. Unit-level work in this milestone may proceed alongside Milestones 3 and 4; clip-level acceptance depends on Milestones 4 and 5.

The reference clip is the only orientation-bearing motion available, so it is both the development target and the acceptance target; the secondary corpus provides breadth for direction and non-regression but cannot validate roll.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Cut a clean feature branch from the upstream default branch; confirm the defect reproduces there | AC-1 | coding | - |
| task2 | Thread an explicit seed into terrain surface sampling; record sampled points in diagnostic runs | AC-1 | coding | task1 |
| task3 | Accept the computed configuration before the convergence break; retain the non-finite guard | AC-1 | coding | task1 |
| task4 | Centralize quadratic residual assembly and scalar evaluation in one helper; route base-position tracking through it | AC-1, AC-3 | coding | task1 |
| task5 | Replace the stopping rule with step norm plus re-evaluated nonlinear objective | AC-1.1 | coding | task4 |
| task6 | Remove the silent trust-region-free retry; expose solver status and fallback use | AC-1 | coding | task1 |
| task7 | Define and land one initialization contract | AC-1 | coding | task1 |
| task8 | Resolve the body-shape parameter override; record resolved configuration in diagnostics | AC-1.2 | coding | task1 |
| task9 | Build the foot-pose diagnostic tool and the diagnostic sidecar writer | AC-10 | coding | task2, task8 |
| task10 | Record seeded pre-hygiene and post-hygiene baselines | AC-7 | analyze | task9 |
| task11 | Run and report ablations: iteration budget, initialization, penetration mode, body-shape parameters | AC-7, AC-8 | analyze | task10 |
| task12 | Implement the source anatomical sole-frame calibration with asserted conventions | AC-2 | coding | task8 |
| task13 | Implement the configured robot body-to-sole transform contract; optional geometry initializer behind degeneracy checks | AC-2 | coding | task1 |
| task14 | Plumb full joint orientations as validated frame-indexed input through the motion data path | AC-3 | coding | task12 |
| task15 | Port the body angular Jacobian and add its finite-difference test | AC-3 | coding | task4 |
| task16 | Assemble the root-relative primary orientation residual into matrices and scalar objective | AC-3, AC-3.1 | coding | task14, task15 |
| task17 | Add the configuration surface under the source entry; enable full tracking in the default reference-robot SMPL-X source | AC-6 | coding | task16 |
| task18 | Implement the directional positions-only fallback with its own Jacobian test and once-per-sequence warning | AC-5 | coding | task16 |
| task19 | Define contact phase including wall contact; extract the local terrain surface normal at contact | AC-4, AC-8 | coding | task9 |
| task20 | Add the gated, angle-limited terrain alignment term with normalized residuals | AC-4 | coding | task16, task19 |
| task21 | Verify the conflict-frame subordination bound between the primary and secondary terms | AC-4 | analyze | task20 |
| task22 | Recompute the derived toe marker under resolved body-shape parameters; run toe-only, anchor-only and combined ablations | AC-7 | coding | task12, task16 |
| task23 | Add unit tests: frame validity, near-half-turn rotation error, second-order linearization decay | AC-2, AC-3 | coding | task15, task12 |
| task24 | Measure stratified reference-clip acceptance on raw and final outputs | AC-7 | analyze | task20, task22 |
| task25 | Measure non-regression: penetration, saturation versus orientation error, legacy position error, torso saturation, feature-off identity | AC-8, AC-10 | analyze | task24 |
| task26 | Build the pinned secondary-corpus manifest and run the directional sweep | AC-9 | coding | task18 |
| task27 | Optional external oracle: run the reference retargeter on the clip and document its differing assumptions | AC-7 | analyze | task24 |

## Claude-Codex Deliberation

### Agreements

- The landing target is a clean feature branch from the upstream default branch; the capability is ported, the local CVXPY solver is not, and the local branch serves as design reference and oracle.
- The structural diagnosis is code-supported: there is no per-link orientation term anywhere in the objective, only a base-quaternion term; the SMPL-X toe joints are emitted by the data source but dropped by target subsetting; the sole foot marker has a small lever arm on ankle pitch relative to neighbouring joints.
- The measured symptom is reproducible and is a near-constant bias rather than noise.
- The primary orientation target must be the calibrated anatomical sole frame, not the raw source joint frame, and not world-level flatness.
- The angular term and the toe positional marker must remain separately togglable, because adding interaction-mesh vertices perturbs connectivity for every target and destroys attribution.
- Profiles that omit orientation configuration must retain legacy behavior.
- The positions-only corpus must explicitly report roll as unobservable.
- Adding a fixed massless toe link to the robot description is not viable; the existing body-local offset mechanism is the correct expression of the toe marker.
- Solver hygiene defects found during review are real and in scope: the discarded step at convergence, the objective term present in the matrices but absent from the scalar cost, the unseeded terrain sampling, and the silent trust-region-free retry.

### Resolved Disagreements

- **Whether the frame convention was the root cause**: an early hypothesis attributed the defect to a fixed ninety-degree source-to-robot ankle frame offset. Direct measurement refuted it; the ankle frames agree after the existing root axis remap, and the real offset is between the source joint frame and the source anatomical sole frame, dominated by toe-out yaw. Resolution: calibration is necessary but is a minor contributor; it ships together with the orientation objective, never instead of it.
- **Whether the defect could be a playback artifact**: audited and cleared. Joint ordering matches, the quaternion convention is correct, the viewer maps by name, and forward kinematics on the stored arrays reproduces the defect. Resolution: no effort is spent on the rendering path.
- **Ordering of the diagnostic spike**: Claude proposed a gate where a successful initialization or penetration ablation could cancel the orientation work. Codex objected that neither supplies an orientation objective. Resolution: Codex's position adopted; the spike informs tuning and scope only.
- **Deriving the robot sole frame from collision geometry**: Claude proposed making inference mandatory for generality. Codex objected that it is ill-posed for curved soles, mesh-only feet, sparse collision models, and runtime primitive substitution, and noted the reference implementations use explicit transforms. Resolution: Codex's position adopted; an explicitly configured transform is the contract, inference is an optional initializer behind degeneracy checks.
- **Base leakage remedy**: Claude proposed masking floating-base columns out of the orientation term. Codex objected that this is inconsistent when other terms co-optimize the root in the same problem. Resolution: Codex's position adopted; a root-relative residual coupled to the existing world base-orientation target.
- **Default enablement**: Claude proposed defaulting the feature off everywhere. Codex objected that the profile which produced the defect must actually enable it. Resolution: Codex's position adopted; generic default off, default reference-robot SMPL-X source entry explicitly on.
- **Toe frame in the robot description**: Claude proposed adding a dedicated toe link mirroring the external reference implementation. Codex objected that fixed massless links are fused during compilation. Independently confirmed by compile test: the two frame links already declared in the description are absent from the compiled model, and an added toe link with tiny mass and inertia is likewise absent while joint count, ordering and names are unchanged. Resolution: the model change is removed entirely; the existing offset mechanism replaces it. The external reference can declare such a frame only because it ships a native model format rather than a converted description.
- **Strictness of validation mathematics**: Claude proposed asserting agreement between the linearized contribution and the direct residual to a tight tolerance. Codex objected that a first-order angular linearization cannot satisfy this at arbitrary finite step. Resolution: Codex's position adopted; exact value and gradient at zero with second-order decay.
- **Acceptance thresholds**: Codex initially argued the thresholds should be stretch goals rather than hard criteria, given a two-degree-of-freedom ankle with a narrow roll range on a parkour clip. The user then established that the source motion was derived from the real robot's own motion, so the target is achievable by construction. Resolution: hard thresholds retained and tightened for contact frames, stratified by contact versus swing, with the reachability premise recorded as an explicit assumption so that human-capture sources can relax it later.
- **Priority between source fidelity and terrain conformity**: open until the user specified that the sole must follow the source first and the terrain second. Resolution: the three-level ordering in the goal, with a numeric conflict-frame bound so that subordination is verified rather than assumed.
- **Strength of the external reference evidence**: Claude cited two reference repositories as independent precedents for calibrated sole orientation. Codex corrected that in the second repository the dynamically created contact tasks use zero orientation cost, so it supports per-link rotation tasks generally but is not an independent calibrated sole-orientation implementation. Resolution: correction accepted; the first reference remains the genuine precedent.
- **Diagnostics artifact**: Claude proposed additive keys in the published output. Codex preferred a sidecar preserving the legacy schema. Resolution: Codex's position adopted.
- **Terrain auto-level rotation**: flagged as a possible second defect because the pipeline never applies a rotation matrix stored in the source file. Empirical test refuted it; the rotation is already baked into the stored motion, and applying it degrades sole-to-terrain agreement by more than an order of magnitude. Resolution: no change; the field is provenance metadata.

### Convergence Status

- Rounds executed: two.
- Round one produced a substantial required-changes list, all of which were incorporated.
- Round two returned `NOT_CONVERGED` with six blockers, each explicitly characterized as requiring a plan-level commitment rather than further investigation. All six are resolved by specification in this document: the hard penetration resolver is fixed as the landed configuration with numeric thresholds on both raw and final outputs (AC-8); the primary-versus-secondary priority is given normalized residuals, a coefficient ratio, and a numeric conflict-frame degradation bound (AC-4); the body-shape branch is committed to the clip's own parameters (AC-1.2); the orientation acceptance metric is committed to SO(3) geodesic error on both raw and final outputs (AC-7); the secondary-corpus manifest, metric, improvement floor and regression tolerances are made concrete (AC-9); and the stopping rule is fully specified (AC-1.1).
- Final Status: `partially_converged`. The six blocker resolutions were specified but were not re-validated by a third review round, so this plan warrants human review before implementation begins.

## Pending User Decisions

- DEC-1: Ankle positional correspondence — land the toe marker alone, or also correct the ankle anchor.
  - Claude Position: land the toe marker first and treat the ankle-anchor correction as a measured ablation that lands only if it demonstrably helps. Measurement shows the current anchor sits well above where the scaled source ankle should be, so it is genuinely miscalibrated and forms a composite error with any toe offset; but it is derived from one clip's shape and scale.
  - Codex Position: land only the well-supported toe marker and orientation fix, because the ankle value is shape- and scale-specific and should not become a generic robot property without a runtime morphology contract.
  - Tradeoff Summary: correcting both at once is geometrically cleaner but couples two changes and destroys attribution, and hardcodes a clip-derived constant. Correcting only the toe leaves a known miscalibration in place that may cap achievable accuracy.
  - Decision Status: RESOLVED BY DELEGATION — the user asked for the work to proceed without per-item adjudication. Landed decision: toe marker alone in the primary path; ankle-anchor correction run as a separate ablation and landed only on demonstrated improvement, and only if expressed through a morphology and scale contract rather than a hardcoded constant. Reversible on request.

- DEC-2: Validation breadth — accept the single orientation-bearing clip, or acquire a flat-ground source clip first.
  - Claude Position: proceed with the available clip. It is the motion the user reported, it is the only one on disk carrying source orientations, and blocking on data acquisition would stall the fix.
  - Codex Position: acceptance may overfit wall-flip behavior; this is listed as an accepted residual risk rather than a blocker.
  - Tradeoff Summary: proceeding risks tuning to one parkour motion; the secondary corpus provides breadth for direction and non-regression but cannot validate roll or full sole orientation.
  - Decision Status: RESOLVED BY DELEGATION — proceed with the available clip, with the overfitting risk recorded and the stratified contact-versus-swing reporting acting as partial mitigation. Acquiring a flat-ground source clip with body-shape parameters remains the single highest-value addition to this plan and should be revisited before the fix is relied upon beyond this motion.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Verified Facts

Every item below was confirmed against the code or the data during planning, and several correct or supersede statements in the original draft reproduced at the end of this document. Numbers derived from body shape must be recomputed once the body-shape parameter override is resolved.

- The objective contains no per-link orientation term of any kind; the only orientation cost applies to the floating-base quaternion.
- The SMPL-X toe joints are produced by the data source but removed by target subsetting because the default source entry never maps them. Sibling non-SMPL-X sources in the same profile do map toe targets.
- The actuated index set includes all floating-base degrees of freedom, and the base carries a very small regularization weight, so an unconstrained orientation residual would preferentially rotate the base rather than the ankle.
- The Clarabel wrapper supports at most one second-order cone, hardcoded to the leading variables, so per-foot angular rate limits require a general multi-cone extension. The failure path re-solves with no cone at all.
- The scalar objective used for the stopping decision is recomputed separately from the quadratic matrices, and one existing term already reaches the matrices without reaching the cost.
- The convergence break precedes acceptance of the newly computed configuration, so the converged step is discarded and the previous iterate is returned.
- The landed profiles reduce solver iterations far below the code defaults, to a small number on every frame after the first. Combined with the discarded step, each later frame accepts fewer steps still. This was introduced deliberately as a throughput optimization for batch processing and has never been ablated against foot quality.
- The default reference-robot SMPL-X source entry overrides the clip's sixteen body-shape parameters with a shorter all-zero vector, so the solver does not see the subject's real proportions.
- Terrain surface sampling is unseeded, so runs are not reproducible across processes; samples are fixed within a single run.
- The interaction mesh is retetrahedralized every frame, so adding vertices perturbs connectivity and the Laplacian residual for every target, not only the feet.
- The existing foot stabilization post-process modifies base lift and horizontal correction only. It is translation-only and cannot repair foot orientation, which is why position-based clearance checks pass while the sole is grossly misoriented.
- Two massless frame links declared in the robot description are fused away during compilation and are absent from the compiled model. An added fixed toe link behaves the same way, while joint count, ordering, names and limits remain unchanged.
- The rotation matrix stored in the source file as terrain auto-level provenance is already applied to the stored motion. Measured sole-to-terrain agreement is best as stored and degrades by more than an order of magnitude if the rotation is applied again in either direction.
- The source file carries video-derived provenance fields and per-frame contact-alignment records, consistent with the stated origin of the motion.
- Only one orientation-bearing clip exists on disk. The secondary corpus holds joint positions only, with no shape or pose parameters, so calibrated sole orientation cannot be computed there; it contains a few hundred base motions plus rotation and translation augmentations of them.
- The local test suite currently passes.

### Assumptions

- **Source reachability**: the source foot orientation is kinematically achievable by this robot, because the source motion was derived from the robot's own motion. The tightened contact-phase thresholds depend on this. If this plan is later applied to human motion capture, the thresholds must be revisited, because a two-degree-of-freedom ankle with a narrow roll range cannot in general reproduce a human sole pose.
- **Frame consistency**: the source motion and the terrain mesh share one world frame, verified by measuring sole-to-terrain agreement. The secondary terrain-conformity term is well defined only under this assumption.
- **Single-clip acceptance**: thresholds are tuned and verified on one parkour motion. Stratified contact-versus-swing reporting is partial mitigation, not a substitute for breadth.

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# Toe Targets And Ankle Orientation Residual For G1 Feet

## Original Idea

你帮我想想这个怎么办 就是我现在用test——data里面的数据就是重定向出来 就是脚底的不对，就是脚的朝向不对 应该脚的朝向和人物模型一样啊 现在脚尖都是竖着的 为什么这个脚会这样 是哪里的重定向的问题吗

## Primary Direction: Foot orientation objective coverage

### Rationale

Asks whether the SQP solver has any orientation residual on the ankle/foot links at all — i.e. whether the feet are rotationally UNCONSTRAINED and drift, rather than being actively mis-rotated.

### Approach Summary

**Root cause: the ankle/foot links have zero orientation residual in the optimizer, and the SMPL-X toe joints that would constrain them positionally are loaded but thrown away.** The feet are rotationally under-determined and get driven to their joint limits by the penetration constraints.

The evidence chain in the upstream tree (the code that produced `out/c167_scaled_retargeted.npz`):

1. The QP cost is enumerated at `omniretargeting-upstream/omniretargeting/retargeting.py:894-900`: Laplacian interaction-mesh (per-vertex weight `10`), bone-direction prior, joint regularization (`1e-3`), temporal smoothness (`0.2`), **base** orientation (`orientation_weight = 5.0`, line 942), base XY position, penetration slack. There is **no link-orientation term of any kind** — `grep -n "orient"` only ever hits the floating-base quaternion `q[3:7]`. No rotational-Jacobian residual exists anywhere in the upstream package.
2. The only foot participation in the objective is a **single point** per foot: `unitree_g1.json` → `source[smplx].target_mapping.L_Ankle = {robot_link: left_ankle_roll_link, offset: [0,0,0.05]}`. That offset points *upward*, ~3 cm above the ankle pitch axis, not on the sole.
3. SMPL-X `L_Foot`/`R_Foot` (indices 10/11) **are** loaded — `data_sources/smplx.py:29-34` lists them in `DEFAULT_SMPLX_TARGET_NAMES` and `md.positions` is `(143, 22, 3)` — but `core.py:88-97` builds `mapped_source_target_indices` strictly from `joint_mapping.keys()` and `core.py:488` slices by it, so the toe joints never reach the interaction mesh. Sibling sources in the **same JSON file** do map them (`lafan1: LeftToe → left_ankle_roll_link, offset [0.12,0,0]`; `nokov: LeftToeBase, offset [0.15,0,0]`; `bones_seed: LeftToeBase, offset [0.1,0,-0.04]`) — only `smplx` and `omomo` omit them.
4. `smplx_orientations` (T,J,4 wxyz) is computed by the loader but used only for the *root* (`data_sources/smplx.py:90`, `core.py:428-466`); in the local tree it is merely printed (`main.py:539-540`).

Proposed fix, two tiers:

**Tier 1 — config-only (4 JSON lines, 0 LOC).** Add `L_Foot`/`R_Foot` to `source[smplx].target_mapping` (and `omomo`) in `robot_models/unitree_g1/unitree_g1.json`, pointing at `left/right_ankle_roll_link` with offset ≈ `[0.12, 0.0, -0.035]` (toe tip on the sole plane; `-0.035` is where the popsicle sole cylinders actually sit — geoms at `z=-0.025` with radius `0.01`, and the local branch independently converged on `tracking_offset=[0.039,0,-0.035]`). This reuses the existing lafan1/nokov pattern exactly, requires no code change, and adds two mesh vertices whose Laplacian residual sees ankle pitch with a **13.0 cm/rad** lever arm instead of the current 3.2 cm/rad.

**Tier 2 — explicit angular residual (~120 LOC upstream).** Add an optional per-link orientation term driven by `smplx_orientations[:, 10/11]` (the rigid SMPL-X foot frame). Needed because ankle **roll** stays weakly observable positionally (3.5–4.9 cm/rad for any sole point; the foot is only ±3.6 cm wide). Mechanism: world angular Jacobian `Jr @ T` restricted to `q_a_indices`, residual `rotvec(R_target @ R_current^T)`, folded into `P/c` as `w_o * ||Jr_a dqa - e||²`. `Jr @ T` is already computed inside `_compute_robot_jacobians` (`retargeting.py:1211`) for the offset cross-term, so the Jacobian plumbing exists.

### Objective Evidence

- **Quantified symptom, read from the complained-about output.** Ankle joints are **pinned at their URDF limits**: `g1_29dof_popsicle.urdf` gives `ankle_pitch ∈ [-0.87267, 0.5236]`, `ankle_roll ∈ [-0.2618, 0.2618]`. In `out/c167_scaled_retargeted.npz` (143 frames), `left_ankle_pitch` sits at the upper bound in **78/143** frames, `right_ankle_pitch` in **114/143**; `left_ankle_roll` on a bound in 61/143, `right_ankle_roll` in 101/143. A DOF that saturates like this against a `1e-3` regularizer pulling it to zero is being driven by constraints, not by an objective.
- Robot toe axis (`+x` of `ankle_roll_link`) elevation: left median **−55.5°** (range −76.5°…+5.7°), right median **−53.9°**. Sole normal vs world up: median **59.2° / 60.9°**, exceeding 45° in **120/143** and **140/143** frames. That is the "脚尖都是竖着的" the user reports.
- Source SMPL-X `L_Ankle→L_Foot` direction elevation: median **−31.9° / −29.3°**, beyond 45° in only 36/143 and 31/143 frames. Per-frame angle between robot toe axis and source ankle→toe vector: **median 33.2° (left) / 35.8° (right)**, p90 45°, exceeding 30° in 100/143 and 87/143 frames.
- Observability measured on the *only* mesh vertex the ankle DOFs touch (offset `[0,0,0.05]` on `left_ankle_roll_link`, FK-swept +30°): `ankle_pitch` **3.2 cm/rad**, `ankle_roll` **4.9 cm/rad**, vs `knee` **26.5 cm/rad** and `hip_pitch` **59.7 cm/rad** — the ankle is 8–19× less observable than its neighbours. With a toe target at `[0.12,0,-0.035]` the pitch lever becomes **13.0 cm/rad**.
- `omniretargeting-upstream/omniretargeting/retargeting.py:894-900` — the complete objective enumeration (no link-orientation term).
- `omniretargeting-upstream/omniretargeting/retargeting.py:940-958` — the *only* orientation cost, base quaternion, weight `5.0`.
- `omniretargeting-upstream/omniretargeting/retargeting.py:1188-1220` — `_compute_robot_jacobians`; builds `Jr_world = Jr @ T` already, but discards it after the position cross-term.
- `omniretargeting-upstream/robot_models/unitree_g1/unitree_g1.json` — `smplx`/`omomo` map only `L_Ankle`/`R_Ankle`; `lafan1`/`nokov`/`bones_seed` in the same file already map `LeftToe`/`LeftToeBase` to `ankle_roll_link` with a forward offset. **This is the extension point.**
- `omniretargeting-upstream/omniretargeting/data_sources/smplx.py:29-34` — `L_Foot`/`R_Foot` at indices 10/11 already produced by the loader.
- `omniretargeting-upstream/omniretargeting/core.py:88-97, 475-494` — target subsetting that silently drops unmapped joints.
- `omniretargeting-upstream/omniretargeting/core.py:586-690` — `_apply_foot_stabilization` / `_apply_surface_collision_corrections`: a translation-only post-process (base lift + XY nudge). It **cannot** repair foot orientation, which is why the symptom survives it.
- **Working prior art on the local branch `fix/g1-wallflip-foot-tip-standing`:**
  - `omniretargeting/retargeting.py:702-720` — `_body_angular_jacobian(body_id)`: `mj_jacBody` → `@ _build_transform_qdot_to_qvel_fast()` → slice by `q_a_indices`. Directly portable.
  - `omniretargeting/retargeting.py:793-845` — contact-phase foot orientation residual: builds `target_rotation = [forward, lateral, up]`, `rotation_error = Rotation.from_matrix(R_t @ R_c.T).as_rotvec()`, term `(angular_jacobian @ dqa - rotation_error) / orientation_sigma`.
  - `omniretargeting/retargeting.py:876-920` — swing-phase full-orientation and yaw-only ("heading") variants; `:922-945` — hard angular-step cap `||rotvec + Jr dqa|| <= max_orientation_step`.
  - `omniretargeting/utils.py:364-491` — `_extract_smplx_end_effector_markers`: builds the rigid SMPL-X foot frame from LBS joints 10/11, template PCA thinnest axis as sole-up, `template_joints[foot_idx] - template_joints[ankle_idx]` as toe-forward, then rotates per-frame by `world_joint_orientations[:, foot_idx]` — i.e. it consumes exactly the `smplx_orientations` array. Wired at `utils.py:660-680`.
  - `robot_models/unitree_g1/unitree_g1_instinctmj.json` — a fully-tuned config: `left_foot/right_foot` with `track_orientation: true`, `track_swing_orientation: true`, `track_swing_heading: true`, `orientation_sigma: 0.0698` (4°), `max_orientation_step: 0.1745` (10°), `tracking_offset: [0.039, 0, -0.035]`. Note the *default* profile `unitree_g1.json` has no `end_effector_priority` block, and `main.py:20` `DEFAULT_ROBOT_CONFIG_PATH` points at that default — so even on the local branch the machinery is off unless the instinctmj profile is passed.
  - `omniretargeting/contact.py:243-345` — sole-up vs surface-normal gating and contact-orientation anchoring; `tests/test_contact_priority.py` — existing test surface for this machinery.
- **Porting cost.** The local implementation is CVXPY (local `retargeting.py:7`); upstream was rewritten to a hand-assembled sparse QP solved by Clarabel (`_solve_qp_clarabel` at upstream line 140). Tier 2 cannot be copy-pasted — the residual must be re-expressed as `P += 2*w*Jr_aᵀJr_a`, `c += -2*w*Jr_aᵀe` next to the existing `P_diag_extra` block (upstream `retargeting.py:927-978`). ~120 LOC total: `_body_angular_jacobian` (~20), residual assembly (~50), config parse + signature threading through three call sites at lines 584-593 / 669-720 / 793-801 (~30), JSON schema + README (~20). Tier 1 is 4 JSON lines and 0 LOC. Per-frame cost of Tier 2 is 2 extra `mj_jacBody` calls plus a 3×nq_a rank-3 update — negligible against the ~100-vertex Delaunay + Laplacian rebuild per frame.

### Known Risks

- **Tier 1 fixes pitch far better than roll.** Measured roll lever arms: `[0,0,0.05]` → 4.9, `[0.12,0,-0.035]` → 3.5, `[-0.05,0,-0.035]` → 3.5 cm/rad. Adding a toe vertex actually *reduces* roll leverage. Since `ankle_roll` range is only ±15°, residual roll error is bounded, but roll saturation (85/143 frames on the right) will likely persist without Tier 2.
- The SMPL-X toe segment is ~0.145 m (measured median `L_Ankle→L_Foot`) vs the G1 sole forward extent ~0.13 m from the ankle. The Laplacian is scale-normalized via `source_to_robot_scale`, but a mismatched offset will bias pitch systematically — the offset value needs a calibration pass, not a guess.
- Adding vertices changes the Delaunay triangulation and the Laplacian adjacency for *all* joints, so this is not a locally-scoped change: every existing retargeted result for `smplx`/`omomo` sources shifts, and regression baselines in `tests/` will move.
- c167 is a wall-flip/parkour motion on non-flat terrain (`c167_scene_scaled.obj`); some genuinely steep foot poses are correct. The target must be the *source* foot frame, not "flat on the ground" — a naive sole-flattening prior would break the flip.
- The penetration constraints are currently what pins the ankles; adding an orientation objective creates direct competition with a hard constraint. Expect infeasibility or chattering unless the orientation term is soft and the angular-step cap (local `retargeting.py:922-945`) is ported alongside it.
- `smplx_orientations` requires `model_directory` (raw SMPL-X params). Motions loaded from `global_joint_positions` have no joint rotations — Tier 2 needs the `raw_forward = joints[10]-joints[7]` fallback that already exists at local `utils.py:483-490`.
- The local branch is ~86 commits behind `origin/main` and its `unitree_g1.json` uses the older flat `joint_mapping` schema (no per-target offsets); upstream uses nested `source[].target_mapping` with offsets. Any port must target the upstream schema.

## Alternative Directions Considered

### Alt-1: SMPLX-to-URDF frame offset calibration
- Gist: Assume orientation *is* tracked but a fixed rotation between the SMPL-X ankle frame and the G1 ankle-link frame goes uncompensated, producing a constant toe-up error; the fix would be a once-at-init rest-pose calibration of that offset. Direct measurement **refutes the ~90° premise**: after the existing `_SMPLX_ROOT_OFFSET` axis remap, the SMPL-X ankle joint frame and the G1 `*_ankle_roll_link` frame differ by **0.0°**, and the G1 link frame equals its own sole frame to **0.0°** (sole geoms lie on an exactly flat plane at local z = −0.035, toe along +x). The real offset is between the SMPL-X foot *joint* frame and the SMPL-X *anatomical sole* frame: **24.47° (L) / 21.33° (R)**, decomposed as ≈9.6° pitch + 22.3°/−18.8° toe-out yaw + ~2° roll. So calibration is a real, already-written component worth landing, but it is a ~10° contributor to a ~40° error — necessary, not sufficient.
- Objective Evidence:
  - `omniretargeting-upstream/omniretargeting/data_sources/smplx.py:18-26` — `_SMPLX_ROOT_OFFSET = [[0,1,0],[0,0,1],[1,0,0]]`, the only hard-coded axis remap, applied **root-only** at line 89.
  - `omniretargeting-upstream/omniretargeting/main.py:290, 303, 308-311` — `source_orientations` loaded, shape printed, never forwarded to `OmniRetargeter(...)` at line 315-323. This is the actual dead-end.
  - Measured foot-frame error in `out/c167_scaled_retargeted.npz`: **median 54.6° (L) / 52.9° (R)**, min 29°, max 66°; toe elevation source median −11.2°/−12.2° vs robot −55.5°/−53.9° → **~40° of pure pitch error**.
  - `*_ankle_pitch_joint` pinned at the **+0.5236 rad upper limit in 79/143 (L) and 114/143 (R) frames** while −0.873 rad of dorsiflexion remains unused; the median ankle-position residual it is chasing is only 38 mm / 34 mm.
  - `omniretargeting/utils.py:432-478` (local) — the rest-pose foot-frame calibration, with an explicit comment at 432-438 explaining why per-frame PCA was rejected in favour of one-time template calibration.
  - Identity robot-frame assumption verified for **all four robots in the repo** (`unitree_g1` `left_ankle_roll_link`, `unitree_h1` `left_ankle_link`, `booster_k1` `left_foot_link` — all identity xmat at zero pose, sole normal +z, toe +x; `hightorque_mini_pi_plus` has no matching link).
  - Quaternion conventions checked clean on every foot path: `core.py:416` (`scalar_first=True`), `core.py:849-850` (explicit wxyz→xyzw for MuJoCo), `utils.py:469-472`.
  - `omniretargeting/retargeting.py:967-980` (local) — a comment already diagnosing this exact symptom: splitting position/orientation lexicographically "makes the orientation level inherit an already saturated ankle and can leave the toe axis tens of degrees from the captured foot".
- Why not primary: its own headline hypothesis is quantitatively refuted, and the calibration it proposes only becomes a fix when bundled with the missing orientation objective that the primary direction supplies.

### Alt-2: Playback and convention audit
- Gist: Test whether the vertical toes exist in the stored npz at all or are introduced downstream by joint ordering, quaternion convention, or the floating-URDF variant in the mjviser replay path. All four sub-checks came back **clean** and forward kinematics on the raw npz reproduces the vertical toes exactly, so this direction's deliverable is a formal close-out plus a standing `check_foot_orientation.py` diagnostic (~60-80 LOC) modelled on the two existing throwaway scripts.
- Objective Evidence:
  - Joint ordering **clean**: npz `joint_names` is byte-identical, in order, to MuJoCo's scalar-joint list (`scal == jn` → `True`), and `mjviser_viewer.py:129,151` maps **by name** anyway.
  - Quaternion convention **clean**: FK under wxyz gives pelvis tilt 17–22° (upright); under xyzw it gives 110–144° (upside-down). wxyz is unambiguously correct.
  - Defect is real and large: heel sits **+0.148 m (L) / +0.145 m (R)** above the toe on the median frame; sole-normal tilt median 59.2°/60.9°. A frame-convention-independent check — `angle(foot_forward, pelvis_up)` — is exactly **90.0°** in URDF `qpos0` but **median 163.1°/161.0°, max 177.2°/178.2°** in the motion.
  - Kinematic decomposition: median leg pose `hip_pitch −0.116, knee +0.846, ankle_pitch +0.496` sums to ≈70°, and 90° + 70° = 160°, matching the measured 163°/161° to within a degree — unopposed knee bend plus max plantarflexion.
  - `waist_pitch_joint` is at its limit in **94.4%** of frames, a second saturation signal nobody was looking at.
  - Visual-vs-collision **clean**: `mjviser_viewer.py:283-292` hides group 0 when meshes exist (`nmesh=35`); both geom sets are rigidly attached to the same body frame, so no geom choice can make a flat foot render at 163°.
  - Both renderers consume the same array — `main.py:362-364` writes the npz and passes the *same* `retargeted_motion` to the video path at lines 401-417.
  - `omniretargeting-upstream/check_sole.py` (58 LOC) and `check_contact.py` (94 LOC) are the reusable seeds; the latter's `jnt_range` violation scan becomes a saturation-percentage table in ~5 lines.
  - `out/c167_scaled_retargeted.npz` has only 5 keys — no `smplx_joints_w`, `contact_targets_w`, `contact_mask` — yet `mjviser_viewer.py:575-597` supports all those overlays. The single most useful visual for this bug is one `savez` kwarg away.
- Why not primary: it is falsified as a fix site — no playback change alters the geometry — so its value is eliminating a hypothesis and leaving an instrument behind, not solving the problem.

### Alt-3: Sole-contact plane anchoring
- Gist: Sidestep rotation targets entirely by promoting each foot from a 1-point contact anchor to ≥3 sole keypoints (heel, toe-medial, toe-lateral) anchored to an SE(3) frame on the scene, so sole flatness falls out of position geometry. Crucially, the K targets should be generated as `p_run + R_run @ robot_local_offset_k` from the *robot's own* sole offsets rather than from three independently projected SMPL-X keypoints, which would encode the human foot's dimensions and be mutually unsatisfiable on the G1's 186 mm sole. ~300 LOC across `utils.py`, `contact.py`, `retargeting.py`, `trajectory_refinement.py`, and the profile.
- Objective Evidence:
  - `omniretargeting/contact.py:92-156` — `_closest_scene_samples`; line 122 `np.argmin(distances, axis=1)` collapses to **one** point per foot per frame. The single-anchor bottleneck.
  - `omniretargeting/retargeting.py:757-772` — the residual site does `min(candidates, key=...)`, so **even with K offsets configured, exactly one contributes**; line 782-786 emits exactly one half-space constraint.
  - `retargeting.py:228-315` — `_setup_end_effectors` already validates `contact_offsets` as `(K,3)` and its docstring explicitly anticipates this: *"A contact task may expose several surface points (for example heel, toe, and sole corners)"*. The plumbing was designed for K>1; only the selection semantics block it.
  - **Smoking gun in the profile**: in `unitree_g1_instinctmj.json`, `left_hand`/`right_hand` each have **5 `contact_offsets`**; `left_foot`/`right_foot` have **exactly one**, `[0.039,0,-0.035]`, and compensate with `track_orientation: true`. The hands got multi-point treatment; the feet did not.
  - Robot geometry supports it: 7 cylinder collision geoms per foot yield **266 sampled points spanning x ∈ [−0.054, 0.132] (186 mm) × y ∈ [±0.036] (69 mm), 108 within 1 cm of the sole bottom**; a heel/toe triangle has ~93 mm and ~35 mm moment arms — well-conditioned.
  - Measured with those keypoints: left heel **0.137 m above the toe** (128/143 frames > 5 cm), right **0.147 m** (143/143); sole tilt 57.2°/61.2°. Sole keypoint distances to the scene: toes 0.004–0.033 m (in contact), **heel 0.124–0.162 m away every frame** — an error a single centre anchor cannot see.
  - `core.py:1159-1176` `_collect_body_contact_points`, `:1178+` `_sample_geom_points_in_body_frame`, `:1289-1317` `_compute_foot_contact_series` — "which robot points on the sole" is already solved and present **in both trees**, but only used for post-hoc z/xy nudging, never as an optimizer residual.
  - `grep -rn "contact_offsets|SceneContactTask|infer_scene_contact"` over the upstream package returns **zero hits** — this machinery is entirely local WIP, and the complained-about npz was produced with no end-effector constraints of any kind.
- Why not primary: it is the largest surface (~300 LOC across five modules) and none of it reaches the complained-about output without also migrating the run onto the local branch and the instinctmj profile, whereas the primary's Tier 1 lands on the exact upstream config that produced the bug.

### Alt-4: Robot profile config surface
- Gist: Treat the defect as data, not code — the fix is three JSON levers in `unitree_g1.json`: add `L_Foot`/`R_Foot` toe targets to the smplx `target_mapping` (two points on one rigid link fully pin its pitch and yaw), add `"*ankle*pitch*": 0.05` to the existing wildcard-capable `joint_regularization_boost`, and wire `default_joint_positions` into `q_init`. This is the same Tier 1 as the primary, arrived at independently, plus two extra config levers and one 8-LOC code fix.
- Objective Evidence:
  - The smplx `target_mapping` has exactly **14** entries (Pelvis, L/R_Hip, Spine2, L/R_Knee, L/R_Ankle, L/R_Shoulder, L/R_Elbow, L/R_Wrist). This *is* the "14 source targets" from the run log — nothing was dropped by the solver, the config simply never declared toes, head, neck, collars, Spine1/3.
  - Side-by-side in one file: `smplx` has no second foot point, while `lafan1` (`LeftToe`, `[0.12,0,0]`), `nokov` (`LeftToeBase`, `[0.15,0,0]`), and `bones_seed` (`LeftToeBase`, `[0.1,0,-0.04]`) all do.
  - Ankle pitch is nearly unobservable from the declared target: `left_ankle_roll_joint` origin is `[0,0,-0.017558]` from `left_ankle_pitch_link`, a **1.76 cm lever arm**, so its Jacobian column is ~zero and it falls into the null space.
  - **Parsed-but-never-read finding**: `grep -rn "default_joint_positions"` in upstream hits *only* `data_sources/smplx_visualize.py:413-420` and `bvh_visualize.py:94-173` — never `core.py`, `retargeting.py`, or `main.py`. Instead `core.py:320-327` builds `q_init` from **joint-range midpoints**, giving ankle_pitch `(-0.87267+0.5236)/2 = -0.17454` rad, so the profile's declared `T-Pose` ankle of `0.0` never reaches the solver.
  - `retargeting.joint_regularization_boost.joints` already accepts wildcards (`"*shoulder*roll*": 0.2`), consumed at `retargeting.py:491-494` — a pure JSON edit can pull the ankle off its limit.
  - Commit `982ca94 "refactor: fold link_offset_config into target_mapping dict entries"`; `retargeting.py:1193-1215` applies `R_WB @ o_local` with the correct `-skew(o_world) @ Jr` cross-term, so offset targets are exact.
  - `utils.py validate_robot_joint_mapping` already tolerates the dict form and duplicate links (compares a *set* of bodies), so two targets on one link validate cleanly — proven by lafan1/nokov.
  - Stale docs: `README.md:430` still documents the removed top-level `link_offset_config`; `source[]`, `target_mapping` offsets, and `default_joint_positions` are undocumented.
- Why not primary: it is a strict subset of the primary's Tier 1 plus peripheral levers, and it cannot address the ankle-roll saturation that the primary's measured lever-arm analysis shows position targets can never fix.

### Alt-5: Foot pose diagnostic harness
- Gist: Build `scripts/diagnose_foot_pose.py` (~190 LOC) plus `tests/test_foot_pose_diagnostic.py` (~110 LOC) that report, per frame per foot: robot toe axis and sole normal, the same for the SMPL-X source, the error decomposed into pitch/roll/yaw about the source foot's own axes, terrain clearance/penetration, ankle-limit saturation percentages, and a bias-vs-noise verdict. Whole run ≈ 2 s for a 5 s clip (MuJoCo FK 0.75 ms/frame; SMPL-X FK 1.42 s for 143 frames).
- Objective Evidence:
  - Ran it: sole-normal error **median 45.3° (L) / 46.9° (R)**, σ 9.6, max 61.3; >30° in 129/143 and 128/143 frames. Pitch component **+42.3° / +43.5°**, roll +13.9/−12.1, yaw −24.4/+21.8.
  - **Verdict: near-constant bias, not per-frame noise** — pitch error positive on 140/143 frames, 73% of frames within ±15° of the median, lag-1 autocorrelation **0.974**. Roll and yaw are symmetric-opposite between feet, i.e. the SMPL-X template's toe-out splay, a calibratable constant rather than the defect.
  - Sign verified by construction: `+0.3 rad` ankle pitch drives toe elevation from 0° to **−17.2°**, so the IK is *actively* commanding maximum plantarflexion. On the 65 non-saturated left frames pitch error is still **+28.2°** → limits amplify but do not cause it.
  - Robot toe elevation median −55.5° vs shin (knee→ankle) elevation median −57.7° — the foot is **collinear with the shin**, the signature of a purely positional IK with nothing constraining foot pitch.
  - `check_sole.py` on the same file reports **min clearance +0.016 m, zero penetrating frames, 116/143 in contact** — the existing check *passes* while the pose is grossly wrong. Exactly why an orientation metric is the missing instrument.
  - `omniretargeting/utils.py:518` `load_smplx_trajectory(..., return_end_effectors=True)` already returns per-frame world `forward_directions`/`up_directions` for `left_foot`/`right_foot` — the source-side ground truth needs no re-derivation.
  - `scripts/align_retargeted_omomo_ground.py:25-95` — `_model_joint_names`, `_load_qpos`, `_foot_geom_ids`, `_geom_min_z` directly reusable; `:129-139` and `retarget_omomo_batch.py:454-465` are the repo's argparse/`main()` convention.
  - Two bugs found in the seed script: `check_sole.py`'s `p[2] - r - half` assumes a z-aligned capsule, but the G1 foot capsules are `rpy="0 ±1.5708 0"` (x-aligned); and `mj_id2name(...mjOBJ_GEOM...)` returns `None` for every foot geom, so body-id selection is mandatory.
  - `grep -rn "terrain_auto_level"` returns **zero hits** across both packages, so the source npz's `terrain_auto_level_rotation_matrix` is never applied — confirming source and robot orientations are directly comparable in the same world frame.
- Why not primary: it measures rather than fixes, and its central measurements have already been produced here, so the marginal value is regression protection for whichever fix ships rather than the fix itself.

## Synthesis Notes

All six explorations converged independently on the same root cause — no per-link orientation objective plus a single, badly-placed foot target — with four separate teams measuring the ankle-pitch saturation (78/143 L, 114/143 R at the `+0.5236` plantarflexion limit) from different starting hypotheses. Two directions falsified themselves productively and should be folded into the primary rather than pursued: Alt-2 conclusively eliminates playback, ordering, quaternion convention, and the floating-URDF variant as causes, so no effort belongs there; Alt-1 refutes the intuitive "90° frame bug" story but replaces it with a real, measured 24.5°/21.3° SMPL-X joint-frame-vs-sole-frame offset that the primary's Tier 2 must calibrate out or it will chase a phantom ~10° of the error. Note one apparent conflict worth resolving during planning: source toe elevation is reported as −32° by the raw `L_Ankle→L_Foot` joint vector and −11° by the calibrated template sole frame — Alt-5 explains the gap (`L_Foot` is the toe-*base* joint and already points ~26° down in the neutral pose), which means the orientation target must be built from the calibrated frame, not the joint vector. Alt-4's `default_joint_positions`-never-read finding (`core.py:320-327` uses joint-range midpoints, seeding ankle_pitch at −0.175 rad) and its wildcard `joint_regularization_boost` lever are cheap, independent additions that belong in Tier 1 regardless. Alt-5's diagnostic should be built *first* — it turns the fix into a measurable delta and its bias-vs-noise verdict (lag-1 autocorrelation 0.974) is exactly the signal that distinguishes a successful Tier 1 from a successful Tier 2. Alt-3 is the right long-term shape for contact phases specifically, and its key insight — generate multi-point targets from the robot's own sole offsets transported by an anchored SE(3) scene frame, never from independently projected human keypoints — should constrain how Tier 2's contact-phase residual is written even if the full 3-keypoint rework is deferred; its warning that `hierarchy: false` and `weighted_scale: 0.2` in the current profile would out-compete any foot residual applies to Tier 2 verbatim.

--- Original Design Draft End ---
