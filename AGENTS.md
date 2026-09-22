# OmniRetargeting Agent Instructions

Canonical project instructions for all coding agents; tool-specific files reference this file.

## Project

Python package for humanoid motion retargeting. CLI: `python -m omniretargeting.main`.
Robot profiles: `robot_models/`; tests: `tests/`.
See [code layout](docs/code_layout.md) for public entry points and implementation modules.

## Retargeting Priorities

- Keep robot and scene at real metric scale (`scale=1.0`) unless explicitly requested otherwise.
  Input SMPL-X motion and scene registration are immutable: never rewrite, rescale, lift,
  or deform the human trajectory to ease the robot solution.
- Hands and feet are the primary interaction end effectors. Contact targets stay at inferred
  scene-contact locations in world coordinates; never scale them, make them root-relative,
  or sacrifice contact accuracy for posture. Free hands prefer body-relative,
  morphology-scaled targets; feet retain world targets in both regimes.
- Preserve source root path, timing, motion direction, and recognizable whole-body pose.
  Deviate only as required by reachability, joint limits, collision avoidance, or incompatible constraints.
- Smooth only solver chatter; preserve deliberate fast motion and contact timing.
  Avoid filtering that causes lag, over-smoothing, foot sliding, or endpoint drift.

## Penetration-Correction Policy

- Keep scene and floating base fixed by default. First attempt bounded, iterative local joint
  corrections using collision-point Jacobians and temporal regularization. Use each sample's
  shortest reliable full exit vector, without forcing a lateral/vertical classification.
- Source-motion and reachable end-effector fidelity take priority over penetration reduction.
  Preserve the target semantics above, including body-relative free hands; avoid endpoint drift,
  limb snapping, chatter, large base translations/rotations, or visible trajectory warping.
- Consider a small robot-root lift only when many active collision points have coherent upward
  exit directions, local correction is insufficient, and interaction targets, source timing,
  and recognizable motion remain materially unchanged.
- For test data, measure against the actual scene and aim for penetration below 10 cm;
  zero penetration is not required, and lateral wall penetration is not an automatic rejection.
  Retain unresolved intersections only after bounded joint correction has been attempted and
  further correction would violate reachability, joint limits, interaction targets, motion
  fidelity, or smoothness. Report affected frames.
- InstinctLab has no terrain-aware spawn penetration detection; its reset height offset is an
  unconditional vertical lift and cannot directly fix lateral intersections. Physics separation
  or invalid-reset handling is a downstream fallback, not evidence of a repaired trajectory.

Implementation and validation notes: [local collision repair](docs/local_collision_repair.md).

## Validation

- Run focused tests first; contact-aware changes require `pytest -q tests/test_contact_priority.py`.
- For final motion acceptance, report end-effector world-position errors, contact residuals,
  temporal jitter, and scene penetration separately. For penetration corrections, also report
  horizontal/vertical displacement components and before/after maximum depth, body, and frame.

## Working Rules

- Preserve unrelated user changes; use small patches and check for existing utilities first.
- Keep maintained Python files at or below 1000 lines; split by responsibility, not compressed
  formatting. Enforced by `tests/test_module_size.py`.
- Keep generic solver contracts source-agnostic; SMPL-X extraction belongs in loaders/adapters
  and contact-marker preparation.
- Do not push to a remote without explicit user approval.
