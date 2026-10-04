# OmniRetargeting Agent Instructions

Canonical instructions for all coding agents; `CLAUDE.md` imports this file. Definitions,
constants and the current implementation: [retargeting invariants](docs/retargeting_invariants.md).
Documentation policy and index: [docs](docs/README.md).

## Goal

Turn SMPL-X human motion, with its scene and objects, into robot motion that is recognizably
the same action, touches what the human touches, and passes through nothing. The output
trains RL tracking policies in InstinctLab, so it must be physically trackable: penetration,
foot sliding and jitter are what a policy cannot follow or learns as artifacts. Penetration
means the robot's collision geoms (`contype`/`conaffinity` set, `collision_geom_ids`), which is
what the simulator sees; visual meshes do not count.

The job is to improve these metrics; measure them before and after any change that affects
motion output:

| Metric | Good | Audit |
| - | - | - |
| Penetration of robot collision geoms (scene, moving objects, self) | zero; deeper than 1 cm is a defect | `scripts/review_scene_metrics.py` |
| Pose similarity to the source | small limb/relation angles, action recognizable | `scripts/review_source_pose.py` |
| Contacts | at the source's scene locations, no foot sliding | `scripts/review_contact_metrics.py` |
| Jitter | no chatter added beyond the source's own motion | `scripts/review_joint_smoothness.py` |

Compare across clips with `scripts/trumans_crossclip_review.py` and
`scripts/review_crossclip_summary.py`. Better methods are welcome, including replacing solver
stages, when these metrics show the gain.

## Project

- Interpreter: `.venv/bin/python`. Robot profiles: `robot_models/`; module map:
  [code layout](docs/code_layout.md).
- Contact-aware changes: `.venv/bin/python -m pytest -q tests/test_contact_priority.py` (~2 s).
- Full suite: `.venv/bin/python -m pytest -q tests` (~2 min).
- CLI: `.venv/bin/python -m omniretargeting.main --solver {framewise,sequence,mink,fast} ...`
  (arguments in `README.md`).

## Hard Lines

Never cross these, whatever a metric gains:

- The SMPL-X motion, scene registration and object motion are ground truth: never rewrite,
  rescale, lift or deform them. Robot and scene stay at `scale=1.0` unless the user asks.
- Joint limits hold.
- Layout: between different limbs, or a limb and the trunk, a relation component clear in the
  reference heading frame (front/back, left/right, up/down) may approach zero but never flips.
  Bends within one limb are not layout.
- Never pass a check by loosening a test or threshold, or with angle caps, per-robot/per-body
  weights in profile JSON or body-part whitelists.

If a metric cannot improve without crossing a hard line, stop and report the binding
constraint (rule, body, frames, magnitude).

## Trade-offs

When goals conflict, the higher item wins, by the smallest deviation that resolves the
conflict:

1. Temporal continuity: no jitter beyond each joint's windowed velocity/acceleration/jerk
   budget, including root and support hand-offs; one joint's budget never pays for another's.
2. Non-penetration, for every body including bodies in contact: no contact, target or posture
   term may hold a body inside a surface.
3. Contacts: supports, then evidenced object control, then other touches.
4. Source root path and timing.
5. Pose similarity: free-hand targets, source orientations and posture relations, one soft
   level balanced by weights.

A small loss in a lower item is acceptable when a higher one clearly improves; state both
numbers. If a lower item collapses (the action becomes unrecognizable or a support is lost),
the result is rejected even if other metrics improve: look for a better method, and report
only if none works.

## Domain Rules

- Contact targets stay at their world-frame scene locations, never scaled or root-relative.
  Free hands use body-relative, morphology-scaled targets; feet keep world targets.
- Build a source-shaped robot reference before clearance optimization. Within a run, keep it
  and its jitter budgets fixed; never rebase them to the latest candidate, or repeated passes
  progressively twist the robot. Improving how the reference is built is fine.
- Contacts: geometry detects (source body surface touching a scene or object surface);
  mechanics only ranks, with one test for the human and every moving object. A contact in a
  minimal feasible support set is a support (e.g. hands under a carried box); object control
  needs evidence from object motion (`object_control_evidence.py`). Body part, proximity and
  duration are not evidence; contacts mechanics cannot explain stay protected.
- Contacts pull onto a surface, never into it: a contact body that penetrates yields by its
  depth, and a yielded contact reports its gap.
- Hand off make-before-break: reach the incoming support before releasing the outgoing one,
  yielding posture first, then the smallest root shift. Example: a seated robot with shorter
  legs than the source slides forward on the seat until its feet reach the floor, then rises;
  sinking the pelvis into the seat is wrong.
- Shifting or lifting the whole root is the usual cheat for penetration and contacts; it shows
  up as floating feet or root drift. Use it only when joint corrections cannot resolve the
  problem, spread corrections over neighbouring frames instead of snapping, and report any
  root shift.
- Smooth only solver chatter: keep deliberate fast motion and contact timing, with no lag,
  foot sliding or endpoint drift.
- Measure posture as morphology-normalized landmark relations (`pose_relations.py`), judged
  against the source rather than fixed angle limits.
- InstinctLab's reset height offset and physics separation do not count as repairs.

## Reporting

For changes that affect motion output, report each metric before/after on the clips you ran,
with the worst body, joint and frame rather than only averages; separate source fast motion
from added chatter; name every trade-off and every remaining defect with its binding
constraint. With `--solver fast`, include `budgets` (with `relation_*`, `layout_flip_*`) and
`skipped_tasks`. Code-only changes need the tests above.

## Working Rules

- Preserve unrelated user changes; use small patches and reuse existing utilities.
- Keep Python files at or below 1000 lines, split by responsibility
  (`tests/test_module_size.py`).
- Keep generic solver contracts source-agnostic; SMPL-X extraction belongs in loaders/adapters
  and contact-marker preparation.
- WebViser/WebVisualizer previews use only port `8183`: check `http://127.0.0.1:8183/`, give
  the user `http://10.19.126.196:8183/<page>/`.
- Do not push to a remote without explicit user approval.

## Sub-agents (paseo)

- Review and analysis: `codex-glm/glm-5.3` (codex is expensive); `codex/gpt-6-astra` only
  for design problems the GLM models cannot handle.
- Writing code: `codex/gpt-6.1-sol`.
- Simple implementation: `codex-glm/glm-5.3`.
- Running and monitoring jobs: `codex-glm/glm-5.3-flash`.
