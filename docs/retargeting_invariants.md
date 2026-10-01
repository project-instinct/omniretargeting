# Retargeting Invariants (detailed)

Full wording of the retargeting rules, with the constants, thresholds and implementation
pointers behind them. `AGENTS.md` carries the always-loaded summary and the priority order;
read this file before changing contact ranking, budgets, posture relations or penetration repair.
If the two files disagree, fix the discrepancy instead of picking one.

## Retargeting Priorities

- Keep robot and scene at real metric scale (`scale=1.0`) unless explicitly requested otherwise.
  Input SMPL-X motion and scene registration are immutable: never rewrite, rescale, lift,
  or deform the human trajectory to ease the robot solution.
- Rank interactions by evidence and role, never a body-part whitelist: source-dynamics
  support alternatives come first, followed by evidenced coordinated object control.
  Unverified sticking is a geometric hold, not proof of support; ordinary touches
  and separated near passes must not acquire support/control priority. Preserve
  ambiguous support alternatives and real short impulses; duration alone is not evidence.
  Hands, feet, pelvis, back, abdomen, forearms, knees and new regions use the same rule.
  Bilateral object control uses a shared object identity and material-relative motion,
  not two hands being close to a surface or stationary in world coordinates.
  Missing mechanics is not permission to release a possible support for object
  control; keep unexplained holds protected and report the uncertainty.
  The main pipeline derives object-control evidence from the source skin contact
  observations and prescribed object motion (`object_control_evidence.py`).
  A protected support/hold point is frozen at its normal distance only while it lies
  on an actual surface (`_fast/surface_presence.py`); a point off every surface, e.g.
  far from its target or buried in an object, may approach its target but not recede.
- Contact targets stay at inferred
  scene-contact locations in world coordinates; never scale them, make them root-relative,
  or sacrifice contact accuracy for posture. Free hands prefer body-relative,
  morphology-scaled targets; feet retain world targets in both regimes.
- Hands are fine detail: a distal segment that no other task observes follows its
  calibrated source orientation closely (`UNOBSERVED_LEAF_SIGMA_SCALE`,
  `_core/orientation_targets.py`), since landmark relations cannot see a leaf turning
  about its own axis. A leaf whose terminal chain carries a sole frame
  (`track_orientation`, the feet) keeps the default weight. The rule reads only the
  kinematic tree and the resolved effectors; keep solver parameters generic and do not
  add per-robot or per-body orientation weights to profile JSON.
- Preserve source root path, timing, motion direction, and recognizable whole-body pose.
  Deviate only as required by reachability, joint limits, collision avoidance, or incompatible constraints.
  Root path and timing rank below contacts and above motion tracking (free-hand targets,
  source orientations and posture), which is one soft level balanced by weights.
- Smooth only solver chatter; preserve deliberate fast motion and contact timing.
  Avoid filtering that causes lag, over-smoothing, foot sliding, or endpoint drift.
- Suppressing solver-induced jitter is a primary acceptance requirement, covering
  robot global translation, root orientation, joints, and support handoffs.
  Contact acquisition and subsequent collision/posture corrections must preserve
  achieved temporal continuity; a lower geometric error does not justify added chatter.
- Check joint smoothness per joint and per time step, as well as in aggregate.
  A smoother arm must not hide increased ankle, wrist, or other joint chatter.
  Secondary corrections must preserve each joint's acquired velocity,
  acceleration, and jerk budgets, alongside root continuity and the reference layout.
  A budget is windowed, per joint and per stencil: the largest magnitude of the
  reference's own derivative within 7 stencils plus 10% of that joint's RMS derivative
  (`motion_budgets.windowed_budget`, the added-chatter envelope). It is never transferred
  between joints; a stencil-exact budget left no room to correct where the reference is smooth.
- Preserve the source motion's recognizable overall posture, judged against the
  original motion. Posture has no angular limit: do not add fixed angle caps (whole
  body, landmark, scalar joint, root orientation or per-bone direction) deciding how
  far the robot may depart from the source. Prefer the fixed source-shaped reference
  softly and never rebase it to the latest candidate, so repeated passes cannot
  progressively twist the robot. When contact/clearance pulls the pose away from the
  source, report the relation angles, landmark and frames rather than hiding them.
- Measure posture change on morphology-normalized landmark relations
  (`omniretargeting/pose_relations.py`): every pair of mapped landmarks and effector
  tips, divided by the skeleton path length between them, so bone directions and
  relative positions share one unit. State changes as equivalent rigid-rotation angles.
  Posture is in the lowest level, motion tracking, together with free-hand targets and
  source orientations: after non-penetration, evidenced contacts and the source root path,
  minimize relation change to the fixed reference, softly, in units of the robot's
  own morphology angle to the source (clip median, at least `POSTURE_SCALE_RAD`); that
  scale weights posture and never limits it. The only hard posture rule is the layout:
  between different limbs, or a limb and the trunk (limbs are read from the landmark
  tree, not names), relation components clear in the reference's heading frame
  (front/back, left/right, up/down) may approach zero but must not flip; bends within
  one limb are not layout. Reject/backtrack a flipping proposal and continue; never
  move the reference to proceed, and report a binding layout conflict.
  Self-collision pairs overlapping in every sampled configuration are model geometry
  (MoveIt-style "always in collision") and are excluded from checks and reports.
  Secondary corrections (collision repair, whole-body correction, refinement) stay local
  to the motion they receive (`_fast/repair_limits.LOCAL_ANGLE`, `max_joint_correction`);
  that is correction locality, not a posture limit against the source.
- Establish a source-shaped kinematic reference before scene-clearance optimization.
  Never certify a distorted initialization merely because later changes are bounded.
  Final acceptance must compare landmark relations with the immutable source as
  well as changes against the fixed robot reference, and report the
  seed's per-bone source directions. Visible loss of the action's shape or intended
  support is a rejection, even if aggregate costs improve.
- Include whole-body contact tasks during contact acquisition. Keep the source-shaped
  posture/continuity reference fixed from before acquisition, and separately freeze
  achieved contact accuracy for later corrections; never reset the posture reference or
  jitter budgets to the latest candidate. Clearance reduction must yield to the layout and
  jitter budgets. It holds evidence-ordered contacts, including whole-body support, but a
  contact never protects its own body's penetration: that penetration is still reduced.

## Penetration-Correction Policy

- Keep scene and floating base fixed by default. First attempt bounded, iterative local joint
  corrections using collision-point Jacobians and temporal regularization. Use each sample's
  shortest reliable full exit vector, without forcing a lateral/vertical classification.
- Avoid penetration wherever the fixed layout and continuity budgets allow: non-penetration
  precedes contacts (a target inside a surface leaves a reported contact residual, not a
  penetration) and precedes source joint-angle or free-hand fidelity within those budgets.
  During contact acquisition, scene/object clearance and self-collision use a quarter of the
  contact sigma (`ACQUISITION_CLEARANCE_SCALE`): one deepest sample per geom must outweigh the
  many contact samples of a body, never be outvoted by them; remaining penetration is then
  reduced with contacts held, and a contact must yield by its own body's penetration depth.
  Open objects are signed near their source free-side evidence. A nearly closed open
  component (its boundary loops can be capped with at most 1% extra area, e.g. a scanned box
  with cracks) additionally decides points the evidence leaves unknown by its completion's
  winding number, outside the loops' bounding spheres and within the 0.01 winding tolerance
  (`scene_geometry.py`); holes, fragments and true open sheets stay evidence-only. A moving
  object is bounded: a point outside the convex hull of its recorded vertices, or deeper than
  the evidence's reach where the component's generalized winding number is decisively zero,
  is outside it whatever the evidence votes (never for terrain or enclosed rooms, which
  continue beyond their vertices). Preserve
  the target semantics above, including body-relative free hands; avoid endpoint drift, limb
  snapping, chatter, large base translations/rotations, or visible trajectory warping.
- Contacts follow the detect, rank, resolve, hand-off pipeline in `AGENTS.md`: detection is
  geometric (source skin against scene/object surfaces) and mechanics only ranks. Supports are
  members of the source's minimal feasible sets (`contact_mechanics.explain_contacts`); ties
  over time follow `contact_temporal.explain_support_timeline`, preferring a path the robot can
  hand off make-before-break, then the one closest to the source root path. Other contacts are
  the `follow` state of `contact_support.py`. Each moving object gets the same feasibility test
  on its own wrench demand from its prescribed motion, which needs the object's COM and inertia;
  its contacts with the body or the scene that lie in a minimal feasible set are supports.
  Status (2026-10-01): no explicit make-before-break hand-off check exists yet, and object
  support mechanics is not implemented: `object_control.py` uses kinematic co-motion only.
- Consider a small robot-root lift only when many active collision points have coherent upward
  exit directions, local correction is insufficient, and interaction targets, source timing,
  and recognizable motion remain materially unchanged.
- For test data, measure against the actual scene (static, moving objects and self-collision)
  and aim for no penetration. Any residual deeper than 1 cm must be traced to the specific
  binding constraint (joint limit, layout/continuity budget, correction locality) and
  reported with depth, body and frames; no contact, target or posture term justifies it. It is a defect to investigate,
  not an accepted default.
- InstinctLab has no terrain-aware spawn penetration detection; its reset height offset is an
  unconditional vertical lift and cannot directly fix lateral intersections. Physics separation
  or invalid-reset handling is a downstream fallback, not evidence of a repaired trajectory.

Implementation and validation notes: [local collision repair](local_collision_repair.md).

## Validation

- Run focused tests first; contact-aware changes require `pytest -q tests/test_contact_priority.py`.
- For final motion acceptance, report end-effector world-position errors, contact residuals,
  temporal jitter, and scene penetration separately. For penetration corrections, also report
  horizontal/vertical displacement components and before/after maximum depth, body, and frame.
- Also report cumulative joint changes and relation angles (whole body and worst
  landmark) against the unmodified source and the fixed pre-correction robot
  reference; they are judged against the original motion, not fixed angles.
  Distinguish source-authored fast motion from added jitter; verify that successive
  corrections preserve the action's recognizable posture as well as temporal continuity.
- The fast backend (`--solver fast`) enforces the reference layout on every candidate
  and verifies per-joint continuity after solving; report its `budgets` (including
  `relation_*` and `layout_flip_*`) and `skipped_tasks`. `--solver sequence` bounds only
  root/slide translation and fits source directions first, without angular slack.
