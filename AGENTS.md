# OmniRetargeting Agent Instructions

This is the canonical project-level instruction file for all coding agents.
Tool-specific instruction files should point here instead of duplicating these
rules.

## Project Shape

- Python package for humanoid motion retargeting.
- Main CLI: `python -m omniretargeting.main`
- Core orchestration: `omniretargeting/core.py`
- Framewise solver: `omniretargeting/retargeting.py`
- Contact inference: `omniretargeting/contact.py`
- Whole-trajectory refinement: `omniretargeting/trajectory_refinement.py`
- Robot profiles: `robot_models/`
- Tests: `tests/`

## Retargeting Priorities

- Preserve the robot and scene at their real metric scale. Keep both at
  `scale=1.0` unless the user explicitly requests scaling.
- Treat the input SMPL-X motion and scene registration as immutable source
  data. Do not rewrite, rescale, globally lift, or otherwise deform the human
  trajectory to make the robot solution easier.
- Hands and feet are the primary interaction end effectors. Split the two
  regimes explicitly:
  - **In contact.** Keep the robot target on the inferred scene-contact
    location in world coordinates. Never convert a contact target to a
    root-relative or morphology-scaled position, and never trade contact
    accuracy for posture.
  - **Free (no contact).** The world position of a free-swinging hand is not
    the objective. Prefer a body-relative, morphology-scaled target so the
    limb keeps the same pose relative to the robot's own torso as the actor's
    limb had relative to theirs. A shorter robot will then carry its hand
    below the actor's in world coordinates; that is accepted, because a
    coherent whole-body pose reads better than a hand dragged to a world point
    the body cannot support. Feet keep their world targets in both regimes —
    they carry the support and the ground contact.
- Preserve the source root path, timing, motion direction, and recognizable
  whole-body pose as closely as the robot morphology permits. Accept only the
  deviation required by robot reachability, joint limits, collision avoidance,
  or mutually incompatible constraints.
- Smoothness is for removing solver chatter, not for changing the motion.
  Preserve deliberate fast source movement and contact timing. Do not use
  aggressive low-pass filtering that causes lag, over-smoothing, foot sliding,
  or hand/foot target drift.
- Reduce scene penetration on the robot side with small, local joint-space
  corrections whenever possible. Penetration refinement is subordinate to
  source-motion and reachable end-effector fidelity: do not move the human
  data, introduce a large floating-base translation/rotation, or visibly warp
  the robot trajectory merely to achieve zero penetration.
- Zero penetration is not required for the test data. Measure the final robot
  trajectory against the actual scene and try to keep penetration below 10 cm,
  but do not treat lateral wall penetration as an automatic rejection. Reduce
  it when small robot-side joint corrections are sufficient; if fixing it
  would noticeably damage reachable hand/foot interaction targets, source
  motion fidelity, or smoothness, preserve the motion and report the affected
  frames instead. Prefer a source-faithful trajectory with known penetration
  over a collision-free but visibly incorrect trajectory.
- InstinctLab does not perform terrain-aware spawn penetration detection. Its
  reset height offset is an unconditional vertical lift and cannot directly
  fix lateral intersections. Such frames may still be included in training and
  left to physics separation or invalid-reset handling, but treat that as a
  known downstream fallback rather than evidence that the penetration was
  repaired.

## Penetration-Correction Policy

- Do not force penetration into a brittle binary "lateral" versus "vertical"
  label. Reconstructed triangle normals are noisy, and contacts near a step
  edge naturally contain both horizontal and vertical components. For every
  penetrating sample, estimate the shortest reliable direction that moves the
  point out of the scene and use that full vector as the correction direction.
- Keep the scene and floating base fixed by default, compute the penetrating
  collision point's Jacobian, and move the relevant ancestor joints along that
  exit direction. For a leg crossing a step, this should primarily use hip,
  knee, and ankle motion to move the leg around the step; use the corresponding
  shoulder, elbow, and wrist chain for an arm. The Jacobian naturally selects
  which joints can move the collision point in the requested direction.
- Use bounded, iterative joint corrections with temporal regularization. Keep
  reachable hand/foot world targets at high priority so collision correction
  does not create endpoint drift, limb snapping, or frame-to-frame chatter.
- Use local end-effector/joint motion for every penetration first. Consider a
  small robot-root height correction only when many active collision points
  have a coherent upward exit direction, local correction is insufficient, and
  the lift does not materially change interaction targets, source timing, or
  the recognizable motion. Never modify the stored SMPL-X trajectory or scene
  to perform that lift.
- Report horizontal and vertical components of the achieved correction rather
  than relying on a hard class label, together with before/after maximum depth,
  body, and frame. An unresolved intersection may be retained for InstinctLab
  only after bounded joint-space correction has been attempted and further
  correction would violate reachability, joint limits, interaction targets, or
  motion fidelity.

## Working Rules

- Expect a dirty worktree. Preserve unrelated user changes and use small,
  surgical patches.
- Check for existing utilities before adding new implementations.
- Keep generic solver contracts source-agnostic; SMPL-X-specific extraction
  belongs in loaders/adapters and contact-marker preparation.
- Validate changes with focused tests first. For contact-aware changes, run at
  least `pytest -q tests/test_contact_priority.py`.
- For final motion acceptance, report end-effector world-position errors,
  contact residuals, temporal jitter, and scene penetration separately.
- Do not push to a remote unless the user explicitly approves it.
