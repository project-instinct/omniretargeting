# Retargeting Design (current implementation)

How the current pipeline realizes the goal in [`AGENTS.md`](../AGENTS.md), with the constants
and code locations behind it. `AGENTS.md` owns the goal metrics, hard lines and trade-off
order; this file explains the mechanisms. Each mechanism is the current approach, not a
requirement: replace it when the cross-clip metrics show a better result without crossing a
hard line, and update this file with it.

## Reference and Jitter Budgets

- Before any clearance optimization the pipeline builds a source-shaped robot reference. Later
  stages measure posture and continuity against it, and it stays fixed within a run: rebasing
  it to the latest candidate let repeated passes progressively twist the robot.
- Because later stages are bounded relative to the reference, a distorted reference passes
  every later check. Its quality is therefore judged directly against the source (relation
  angles, per-bone source directions of the seed). Building a better reference is one of the
  most leveraged improvements available.
- Calibrated sole frames use the same morphology target in source fitting and acquisition
  (`_fast/foot_targets.py`). Each source hip–knee–ankle segment retains its direction with
  the robot's segment length; the semantic tip offset uses the measured terminal-chain scale.
  The target is relative to the robot hip and its Jacobian includes that moving origin. This
  avoids imposing a human-length leg endpoint as a 2 cm world target during reference fitting.
  Missing morphology calibration retains the existing tracking semantics. The anatomical hip
  is the first articulation below its mapped parent, rather than an intermediate serial joint
  origin displaced by an upstream hip rotation; source arrays and robot profiles stay intact.
- After the joint fit, height registration holds every joint angle and root XY fixed. Only
  observed sole contacts provide world-plane measurements (`height_tracking_tasks`); free
  soles remain morphology-relative. The existing support lift then prevents proximal support
  penetration before the reference is frozen. Registration shifts and subsequent lifts are
  reported separately; this is robot height adaptation, never an edit to a source target.
  The height fit is bounded by the collision sole's extent on observed upward static support
  planes (`bound_sole_registration`), not only by its semantic marker. An ankle tilt can put
  collision points below that marker; the minimal extra height is spread over the existing
  0.5 s lift ramp and reported as `sole_extent_correction_max_m`. Joint angles and XY remain
  unchanged in this registration step.
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
  Support-endpoint
  world continuity remains protected during local repair because its purpose is preventing
  support slip, not general smoothness.
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
- Secondary corrections stay local to the motion they receive (`_fast/repair_limits.LOCAL_ANGLE`,
  `max_joint_correction`). This bounds correction locality, not posture against the source.
- Distal segments that no other task observes (hands) follow their calibrated source
  orientation closely (`UNOBSERVED_LEAF_SIGMA_SCALE`, `_core/orientation_targets.py`), because
  landmark relations cannot see a leaf turning about its own axis. Leaves whose chain carries a
  sole frame (`track_orientation`, the feet) keep the default weight. The rule reads only the
  kinematic tree and resolved effectors, so it needs no per-robot settings.
- Self-collision pairs that overlap in every sampled configuration are model geometry
  ("always in collision", `self_collision.py`) and are left out of checks and reports.

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
   Contacts are one-sided, pulling a body onto a surface and never into it. Achieved contact
   accuracy is frozen for later corrections, but a contact body that penetrates yields by its
   depth. A protected point is frozen at its normal distance only while it lies on an actual
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

Example: a seated source with longer legs than the robot. Seated, the pelvis is the support and
the robot's feet hang (reported gap). Before the weight moves to the feet, the robot slides
forward on the seat until its feet reach the floor, then rises. Sinking the pelvis into the seat
to keep the feet down trades penetration for contact.

`_fast/support_handoff.py` checks incoming sole normal gaps at the last source overlap frame
before a static proximal geometric contact releases (support/hold role flapping alone is not
a release). When mechanics identifies a minimal incoming support set, the check uses that set;
otherwise it conservatively checks observed incoming soles. It runs once after solving, reports every failed
handoff, and does not trigger another solve. This geometric check and anticipatory targets do
not yet guarantee a feasible transition when the bounded root/clearance constraints conflict;
such a transition remains a defect, not certified support or permission to unload the seat.

Open problems: force-balanced make-before-break transitions are not certified. Object support
mechanics is approximate (`object_support_evidence.py`: touching objects form one system,
uniform density, nominal demand, full friction cones, environment first) and only promotes skin
contacts to support.

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
- A residual deeper than 1 cm usually traces to a joint limit, a sustained layout collapse,
  chatter/velocity limit, or correction locality. Penetration held in place only by
  lower-priority terms means the
  penetration weight is too low.
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
- Open problem: no backend has a wall-clock budget yet. The 957-frame TRUMANS clip has taken
  1664–3236 s (`out/speedup_20261001/runs/*957*/timing.json`), above the 1 s/frame target in
  `AGENTS.md`. A budget should return the best accepted result so far and report the rest.

## Useful Diagnostics

Beyond the metrics in `AGENTS.md`, these help locate a problem: horizontal/vertical components
of penetration corrections, relation angles against the fixed reference as well as the source,
and the seed's per-bone source directions.
