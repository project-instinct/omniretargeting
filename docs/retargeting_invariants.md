# Retargeting Design (current implementation)

Current mechanisms and known gaps; goals and priorities belong to [`AGENTS.md`](../AGENTS.md).

## Reference and Jitter Budgets
- Before any clearance optimization the pipeline builds a source-shaped robot reference. Later
  stages measure posture and continuity against it, and it stays fixed within a run: rebasing
  it to the latest candidate let repeated passes progressively twist the robot.
- Sole morphology targets use source bone directions and robot segment lengths, relative to
  the robot hip (`foot_targets.py`). Source fitting and acquisition share calibrated offsets.
  Height registration holds joints and root XY fixed, follows observed sole planes, and spreads
  the collision-sole extent correction over 0.5 s; registration and support lift are reported.
- Continuity follows `AGENTS.md`: added high-frequency content is chatter, while a smooth
  low-frequency clearance correction is not. `motion_budgets.chatter_excess` high-passes each
  signal with Gaussian scale `CHATTER_SCALE_FRAMES = 2`, and allows the source-shaped
  reference's local high-frequency amplitude (`CHATTER_WINDOW = 7`), half its RMS, and a
  `2e-3` numerical/motion floor, with a separate `1e-3` excess tolerance for isolated
  sub-threshold samples. Signals cover scalar joints, root translation and root angular
  increments; budgets are not pooled across joints.
- Acquisition candidate guards do not cap acceleration or jerk by the reference derivative and
  do not reject a sweep merely because assembling windows creates a flagged frame: that would
  roll back useful clearance. They retain hard URDF velocity rows when the owner supplies its
  URDF, while final audit reports and gates sustained added chatter.
  Local repair also protects support-endpoint world continuity against slip.
- `scripts/review_joint_smoothness.py` audits the same source-relative chatter, URDF velocity
  limits, and a contact-free inverse-dynamics effort screen. The effort screen reports
  candidates for a contact-force split; it is not yet a final torque certificate.

## Posture

- `pose_relations.py` measures posture on every pair of mapped landmarks and effector tips,
  divided by the skeleton path length between them, so bone directions and relative positions
  share one unit; changes are stated as equivalent rigid-rotation angles.
- The posture term pulls softly toward the fixed reference, scaled by the robot's own
  morphology angle to the source (clip median, at least `POSTURE_SCALE_RAD`). The scale weights
  posture; it does not limit it.
- Layout reads limbs from the landmark tree, not from names. Source landmarks determine the
  legal side wherever both the source and source-shaped reference are clear and agree; pairs
  involving an unobserved effector tip stay soft posture terms. A collapse requires the clear
  opposite side (`LAYOUT_FLIP_MARGIN`) for `LAYOUT_FLIP_WINDOW_FRAMES = 7` consecutive frames.
  Near-zero and brief crossings get a deadband in the linearized rows; a sustained clear flip
  is restored and rejected by exact final acceptance.
- The legacy local pass uses `_fast/repair_limits.LOCAL_ANGLE`. Full-body refinement instead
  uses all tangent coordinates with a fixed 0.4 rad correction neighborhood and the original
  joint/root envelopes. This numerical neighborhood is reported as a possible blocker; it
  does not rebase the source posture reference or freeze individual limb/contact solutions.
- Distal segments that no other task observes (hands) follow their calibrated source
  orientation closely (`UNOBSERVED_LEAF_SIGMA_SCALE`, `_core/orientation_targets.py`), because
  landmark relations cannot see a leaf turning about its own axis. Leaves whose chain carries a
  sole frame (`track_orientation`, the feet) keep the default weight. The rule reads only the
  kinematic tree and resolved effectors, so it needs no per-robot settings.
- Production and independent audits share `SelfCollisionQuery`; audits restore `model_pairs`
  with `use_model_pairs()` and
  measure sampled always-overlapping pairs separately as `model_structural_overlap`. They
  retain per-frame depths and deterministic sample evidence, exclude them from trajectory
  defects/reset invalid, and suggest MJCF `<exclude>` entries without applying them.
  Classification requires 65 determinate samples (default pose plus seed-0 joint-limit
  configurations); failed queries contribute no evidence. The same RNG supplies up to 64
  replacements. Incomplete evidence retains the pair and records an uncertified reason;
  audits treat it as a trajectory pair. Evidence includes unknown/replacement counts and
  sample indices/depths; it is not an analytical proof for every joint angle.

## Contact Pipeline

Geometry detects, mechanics ranks, priority resolves; the same pipeline serves every body,
object and scene.

1. Detect: a contact is where the source body surface (e.g. SMPL-X skin, via its adapter)
   touches a scene or object surface, between the source's own start and end frames.
   Mechanics never adds or removes a contact, so a jump or flip has none in the air.
2. Rank: mechanics orders the detected contacts with one test for every moving body, the human
   and each moving object alike.
   - Supports are members of a minimal feasible support set (`contact_mechanics.explain_contacts`),
     e.g. the hands under a carried box, without which the box would float. Ties over time
     follow `contact_temporal.explain_support_timeline`, preferring a path the robot can hand
     off make-before-break, then the one closest to the source root path.
   - Each moving object gets the same feasibility test on its own wrench demand from its
     prescribed motion, which needs its COM and inertia; its contacts with the body or the
     scene that lie in a minimal feasible set are supports.
   - Object control comes next, evidenced by source skin contacts and prescribed object motion
     (`object_control_evidence.py`). Bilateral control uses a shared object identity and
     material-relative motion, not two hands merely near a surface.
   - Other touches are the `follow` state of `contact_support.py`. Unverified sticking is a
     geometric hold, not proof of support; body part, proximity and duration are not evidence,
     and ambiguous support alternatives and real short impulses are kept.
   - Where mechanics cannot explain a frame, its contacts stay protected and the uncertainty is
     reported.
3. Resolve: when not all contacts fit, the lower-ranked one yields and its gap is reported.
   Contacts are one-sided, pulling a body onto a surface and never into it. The legacy local
   stages freeze achieved accuracy, with a penetrating body yielding by its depth; full-body
   refinement releases these intermediate locks and retains the source contact targets.
   In the local stages a protected point is frozen at its normal distance only while it lies on an actual
   surface (`_fast/surface_presence.py`); a point off every surface, e.g. buried in an object,
   may approach its target but not recede.
4. Hand off: when support shifts between contacts the source holds together (standing up from a
   chair, setting a box on a table), the robot reaches the incoming support before releasing
   the outgoing one, yielding posture first, then the smallest root shift the budgets allow;
   otherwise the support gap is reported (frames, distance, binding constraint).

The fast sole target handover spans `HANDOVER_SECONDS = 0.15` (5 frames at 30 Hz)
before contact and after departure, with cubic smoothstep interpolation. Short swing gaps
connect the two boundary corrections directly. Dense material contacts transfer their mean
rigid displacement to the endpoint at the fixed reference orientation, so approaching a dense
contact does not aim at an unrelated semantic-foot anchor. World anchors and observed contact
intervals are unchanged. Solver support stiffness is extended within existing geometric
contact intervals before an incoming support and after an outgoing support; source role labels
are unchanged. Sliding sole samples have normal-only endpoint and dense-surface residuals.

An unreachable sole may yield to its morphology target only with a conservative geometric
certificate: its contact plane lies beyond an outer sphere containing every joint configuration
and the allowed root translation ball. The radius includes rigid chain lengths, contact sample
offsets, off-origin pivots and bounded slides. This proves impossibility even without angular
limits, hence within the actual limits too. Unsupported unbounded translations never produce
a certificate. The solver's source contact arrays remain intact for auditing the resulting gap.
The certificate is conservative: collision-constrained or joint-direction-constrained failures
inside the sphere remain protected contacts and unresolved gaps, not falsely certified misses.

`_fast/support_handoff.py` checks incoming sole normal gaps at the last source overlap frame
before a static proximal geometric contact releases (support/hold role flapping alone is not
a release). When mechanics identifies a minimal incoming support set, the check uses that set;
otherwise it conservatively checks observed incoming soles. It runs once after solving, reports every failed
handoff, and does not trigger another solve. This geometric check and anticipatory targets do
not yet guarantee a feasible transition when the bounded root/clearance constraints conflict;
such a transition remains a defect, not certified support or permission to unload the seat.

Force-balanced support transitions and the approximate object-support mechanics remain unverified.

## Penetration

- Measured on the robot's collision geoms (`scene_geometry.collision_geom_ids`: `contype` or
  `conaffinity` set), which is what InstinctLab simulates; visual meshes do not count.
- Contact acquisition includes whole-body contact tasks. During it, scene/object clearance and
  self-collision use a quarter of the contact sigma (`ACQUISITION_CLEARANCE_SCALE = 0.25` in
  `_fast/objective.py`), so the deepest sample of a geom outweighs the many contact samples of
  a body. Remaining penetration is then reduced with contacts held.
- Current correction (`_fast/penetration.py`): bounded, iterative local joint updates from
  collision-point Jacobians with temporal regularization, along each sample's shortest reliable
  exit vector (no forced lateral/vertical split; a sample past the medial axis of a thin part
  exits toward its entry side). Corrections are spread over neighbouring frames, starting early
  enough to stay within the budgets, and are kept only if verification finds no added chatter or
  posture excess; otherwise the residual is reported.
- Root lift: when several collision geoms carried by the floating base sit coherently inside
  an upward support (`LIFT_MIN_POINTS = 3`, support normal within 45° of vertical), no joint can
  move them out, so the fixed reference rises by the minimal amount, ramped over
  `LIFT_RAMP_SECONDS = 0.5`, with joint angles unchanged.
- Scene signing (`scene_geometry.py`): open objects are signed near their source free-side
  evidence. A nearly closed component (boundary loops cappable with at most 1% extra area, e.g.
  a scanned box with cracks) also decides points the evidence leaves unknown by its completion's
  winding number, outside the loops' bounding spheres, within a 0.01 winding tolerance. Holes,
  fragments and true open sheets stay evidence-only. A point outside the convex hull of a moving
  object's recorded vertices, or beyond the evidence's reach where the generalized winding
  number is decisively zero, is outside that object (not applied to terrain or enclosed rooms,
  which continue beyond their vertices).
- Unsigned scenes (`_fast/original_rigid_fallback.py`): when every component is an open sheet
  without evidence (or a closed convex part), original-mesh repair replaces the unknown shell
  samples with exact geom/finite-triangle queries (capsule, sphere and solid cylinder,
  `triangle_cylinder.py`). Such a sheet has no solid side: crossing it is penetration, a geom
  wholly behind it is not detected. Any other component, geom type, tied exit or uncertified
  witness stays `unknown_query`, counted by category; it is never zero depth.
- InstinctLab has no terrain-aware spawn penetration detection: its reset height offset is an
  unconditional vertical lift that cannot fix lateral intersections, and physics separation is
  a downstream fallback, not a repaired trajectory.

## Solver Backends

- `--solver fast` always uses its production path; only `performance_trace` is optional timing
  diagnostics. It checks sustained source layout on candidates, verifies source-relative
  chatter and URDF velocity limits after solving, and reports `budgets` (including
  `relation_*`, `layout_flip_*`, `source_relative_chatter`) and `skipped_tasks`.
- `--solver sequence` bounds only root/slide translation and fits source directions first,
  without angular slack.
- Fast has a production deadline from CLI loading to export: default
  `sequence_solver.time_budget_seconds_per_frame=1.0`, with max(2 s, 5%) export reserve.
  LM iterations/attempts, window sweeps, reweight/free phases and repair-group attempts check
  it; GPU service waits also use the remaining budget. Expiry keeps accepted motion; a reweight/free
  iteration needs its original stage guard. Before acquisition, the accepted source/initial
  pose is an explicitly unfinished fallback. Native calls finish before a cooperative check;
  `deadline` reports elapsed time, reserve, first expiry, overshoot and `skipped_stages`.
  Budget-unexpired objectives, iterations, guards and numeric decisions are unchanged.

## Collision contract and independent acceptance

The declared robot contract is `unitree_g1_mesh.json` → `g1_29dof_torsobase_mesh.xml`:
original STL convex hull per link, including the hand. `collision_manifest.py` hashes the
compiled contact excludes too. The mesh MJCF explicitly excludes torso×waist_yaw and both
knee×ankle_roll pairs: all 65 seed-0 samples overlapped, with minimum depths 12.7/17.0/17.0 mm.
These pairs are absent in solving, acceptance and training; other profiles are unchanged.
The manifest identifies the compiled collision model; NPZ stores `collision_model_sha256`
and `collision_model_name`.
`training_match` checks motion/audit/declared training assets, not a running simulator.
Missing hashes are unverified. Training uses the same original scene/object triangle meshes
(e.g. Isaac/PhysX triangle-mesh collision); hull or convex-decomposition proxies are not used. The acquisition SDF is a known approximation, not this audit contract.

`quality_policy.py` and `review_penetration_persistence.py` report each physical pair
(robot geom × static scene/object/self geom), never treating scan fragments as new pairs.
They give Dmax, worst frame and D/F/L/runs at 0.5/2/5/10/20 mm; unknown is not a pass.
Deep means >10 mm. Persistent means >5 mm for >0.3 s or >10% frames; non-support/lateral/self
also fails at >2 mm for >1 s or >25% frames. Verified same-surface normal support residuals
≤5 mm are marked physically unverified; mild brief residuals are reported, not auto-resolved.
Saved depth caches lack per-frame support exit-normal/component certificates: support waivers
remain provisional and cannot pass; reset stays unknown unless another pair proves invalid.
Reset is separately invalid (lateral/self >1 mm or scene/object >5 mm), unknown, or candidate
(geometry only). These are independent audit policies; this port does not alter legacy guards
or stopping thresholds. Audit runs once after solving and never triggers another solve.

The production chain is source_pose → grid → acquisition → reweight → free_bodies →
original_mesh_repair. ≥600 frames retain window acquisition. The legacy local pass supplies an incumbent. A subsequent
`whole_body_projection` releases its acquired-position and per-key minimum locks: original
triangle/reverse/self rows directly enter simultaneous root and all-joint corrections. Source
contact positions remain soft targets; the fixed source reference is never rebased. Physical
collision pairs merge query representations before the unchanged quality policy ranks them.

The added correction has a curvature regularizer, while source-relative high-frequency bounds
are explicit linear QP constraints. An operator-splitting proposal gets at most 5 s per step;
if no admissible progress remains, the same solve refreshes once to a direct QP with the
three-frame inverse-dynamics Jacobian. It does not raise penalty weights. Long-clip direct
steps use temporal policy-event working sets, with a 12-frame halo and merged high-pass
stencils; every time window still contains the whole body. Exact nonlinear layout, continuity, joint projection and
inverse-dynamics effort screening gate acceptance. Existing mechanical violations are reported;
refinement cannot create or enlarge one. The inverse-dynamics screen follows the independent
audit stencil and is not a physical replay certificate. Root movement is bounded by the original
source envelope and reported explicitly, with compensating joints available.

A lost source-side observation first attempts original-geometry outside proof. If it remains
unknown, its negative unsigned triangle distance stays as a conservative optimization residual;
it never supplies a positive-clearance certificate and the unknown remains in diagnostics.
Already computed `refinement_budgets` and the live refinement ledger survive a deadline
that skips final diagnostics. They are not a physical acceptance certificate.
The refinement stops at known policy satisfaction, lack of feasible quality progress after
the single model refresh, or the
production deadline. It does not rerun the independent audits or extend the time budget.
