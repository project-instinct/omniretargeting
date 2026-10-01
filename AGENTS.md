# OmniRetargeting Agent Instructions

Canonical project instructions for all coding agents; `CLAUDE.md` imports this file.
Precise definitions, constants and thresholds: [retargeting invariants](docs/retargeting_invariants.md).

## Project

Python package for humanoid motion retargeting. Robot profiles: `robot_models/`; tests: `tests/`;
module map: [code layout](docs/code_layout.md).

- Use the project interpreter: `.venv/bin/python`.
- Contact-aware changes: `.venv/bin/python -m pytest -q tests/test_contact_priority.py` (~2 s).
- Full suite: `.venv/bin/python -m pytest -q tests` (~2 min).
- CLI: `.venv/bin/python -m omniretargeting.main --solver {framewise,sequence,mink,fast} ...`
  (arguments in `README.md`).

## Fixed Inputs and Reference

The robot reproduces the human's motion, so the source is ground truth, not a variable.

- SMPL-X motion, scene registration and object motion are immutable. Robot and scene stay at
  metric scale (`scale=1.0`) unless the user asks otherwise. Never rewrite, rescale, lift or
  deform the source to make the robot easier to solve.
- Preserve source root path, timing, motion direction and recognizable whole-body pose; deviate
  only for reachability, joint limits, collision avoidance or incompatible constraints.
- Contact targets stay at the inferred world-frame scene locations: never scaled or
  root-relative. Free hands use body-relative, morphology-scaled targets; feet keep world targets.
- Build a source-shaped robot reference before any clearance optimization, then keep it and its
  per-joint jitter budgets fixed for every later stage. Never rebase either to the latest
  candidate: repeated passes would progressively twist the robot.

## Priority Order

When tasks conflict, the higher item wins.

1. Joint limits and the reference layout: between different limbs, or a limb and the trunk,
   a relation component that is clear in the reference heading frame (front/back, left/right,
   up/down) may approach zero but must not flip. Reject or backtrack a flipping proposal and
   report the conflict; bends within one limb are not layout.
2. Temporal continuity: no solver jitter beyond each joint's windowed velocity/acceleration/jerk
   budget, including root translation, root orientation and support handoffs. A lower geometric
   error never justifies added chatter, and one joint's budget never pays for another's.
3. Non-penetration with the scene, moving objects and the robot itself, for every body
   including bodies in contact. It is the dominant soft term: its weight must exceed the
   combined pull of every lower item on the same body, so no contact, target or posture term
   can hold a body inside a surface.
4. Contacts: supports, then evidenced object control, then other touches (see Contacts).
5. Source root path and timing: deviate only as the items above require, and report it.
6. Motion tracking: free-hand targets, source orientations and posture relations to the fixed
   reference form one soft level, balanced by weights rather than ordered among themselves.

## Contacts

One pipeline for every body, object and scene: geometry detects, mechanics ranks, priority
resolves.

1. Detect: a contact is where the source body surface (e.g. SMPL-X skin, via its adapter)
   touches a scene or object surface, between the source's own start and end frames. Mechanics
   never adds or removes a contact, so a jump or flip simply has none in the air.
2. Rank: mechanics only orders the detected contacts, with the same test for every moving
   body, the source human and each moving object alike. A contact in any minimal feasible
   support set of either is a support, e.g. the hands under a carried box, without which the
   box would float. Object control evidenced by the object's motion
   (`object_control_evidence.py`) comes next; other touches last. Body part, proximity and
   duration are not evidence. Where mechanics cannot explain a frame, keep its contacts
   protected and report the uncertainty.
3. Resolve: when the robot cannot satisfy all contacts, the lower-ranked one yields and its gap
   is reported. Contacts are one-sided, pulling a body onto a surface and never into it:
   penetration is never traded for contact, and many contact samples never outvote one
   penetrating sample. Achieved contacts are held through later corrections, but a contact body
   that penetrates yields by its depth.
4. Hand off: when the support of the human or an object shifts between contacts the source
   holds together (standing up from a chair, setting a box on a table), the robot reaches the
   incoming support before releasing the outgoing one, yielding posture first, then the
   smallest root shift the continuity budgets allow. If that is impossible, report the support
   gap (frames, distance, binding constraint).

Example: a seated source with longer legs than the robot. Seated, the pelvis is the support and
the robot's feet hang (reported gap). Before the weight moves to the feet, the robot slides
forward on the seat until its feet reach the floor, then rises. Sinking the pelvis into the seat
to keep the feet down is wrong.

## Posture

- Measure posture as morphology-normalized landmark relations (`pose_relations.py`), stated as
  rigid-rotation angles, against both the immutable source and the fixed reference.
- Posture is judged against the original motion, so add no fixed angle caps (whole body,
  landmark, joint, root orientation or bone direction). When contact or clearance pulls the pose
  away from the source, report the relation angles, landmark and frames.
  `LOCAL_ANGLE`/`max_joint_correction` bound correction locality, not posture.
- Keep solver weights generic, derived from the kinematic tree and resolved effectors; do not
  add per-robot or per-body weights to profile JSON.

## Penetration Correction and Smoothing

- Keep scene and floating base fixed. Correct with bounded, iterative local joint updates from
  collision-point Jacobians with temporal regularization, along each sample's shortest reliable
  exit vector. Spread each correction over neighbouring frames, starting early enough to stay
  within the continuity budgets, instead of snapping out of a surface in one frame.
- A small root lift is allowed only when many active collision points exit coherently upward,
  local correction is insufficient, and targets, timing and motion stay materially unchanged.
- Aim for zero penetration on the actual scene. A residual deeper than 1 cm is a defect: trace it
  to its binding constraint (joint limit, layout/continuity budget, correction locality) and
  report depth, body and frames. Penetration held only by lower items means its weight is too
  low.
- Smooth only solver chatter. Keep deliberate fast motion and contact timing; avoid lag,
  over-smoothing, foot sliding and endpoint drift.
- InstinctLab's reset height offset is an unconditional lift and its physics separation is a
  downstream fallback; neither is evidence of a repaired trajectory.

## When Blocked

If a check passes only by one of the following, stop and report the binding constraint (rule,
body, frames, magnitude) to the user instead of applying it:

- changing the source, scene, scale, fixed reference or jitter budgets;
- adding angle caps, per-robot/per-body weights or body-part whitelists;
- moving the root beyond the lift rule or the hand-off step above;
- loosening a test or threshold, or accepting a better aggregate while one joint, landmark,
  contact or support gets worse.

## Acceptance Report

Report each item separately:

- end-effector world-position errors and contact residuals;
- jitter per joint and per time step as well as in aggregate, separating source-authored fast
  motion from added chatter;
- penetration: before/after maximum depth, body and frame, plus horizontal/vertical components
  of the correction;
- support: yielded contacts with their gaps, hand-off order, root shifts used and any support
  gap;
- cumulative joint changes and relation angles (whole body, worst landmark) against the source
  and against the fixed reference;
- with `--solver fast`: `budgets` (including `relation_*` and `layout_flip_*`) and `skipped_tasks`.

Visible loss of the action's shape or intended support is a rejection, even if aggregate costs
improve.

## Working Rules

- Preserve unrelated user changes; use small patches and check for existing utilities first.
- Keep maintained Python files at or below 1000 lines, split by responsibility
  (`tests/test_module_size.py`).
- Keep generic solver contracts source-agnostic; SMPL-X extraction belongs in loaders/adapters
  and contact-marker preparation.
- WebViser/WebVisualizer previews use only port `8183`. Check locally on
  `http://127.0.0.1:8183/`; give the user `http://10.19.126.196:8183/<page>/` (router-forwarded),
  never this machine's address or another port.
- Do not push to a remote without explicit user approval.
