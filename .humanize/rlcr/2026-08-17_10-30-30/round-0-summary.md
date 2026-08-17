# Round 0 Summary

## Outcome

The defect is two thirds gone and every other measure improved, but **AC-1's numeric gate is not met**. The contact-state step in base height is 6.35 cm against a 3 cm gate. I am reporting this rather than claiming completion.

## What was implemented

**Base height became measurable** (`scripts/diagnose_foot_pose.py`). The defect escaped the existing metrics because they covered joint angles and not the floating base: joint acceleration was 27.9 against the untouched pipeline's 28.7 while the base was convulsing. The diagnostic now reports base second-difference RMS, largest single-frame step, and the median base height split by whether a contact constraint is active.

**Source skeleton matching** (`omniretargeting/core.py`, `_match_skeleton_to_robot` and helpers). The interaction mesh tracks absolute source joint positions, so with the scene left at captured scale and the robot shorter than the subject, the pelvis target is unreachable: it cannot be at human height while the feet are on the real floor. Rescaling the source skeleton to robot proportions removes the conflict instead of balancing it. Three details decide whether it works:

- The scale is measured from pelvis-to-ankle length, not height. Source 0.861 m against robot 0.751 m gives 0.8724 and an 11.0 cm pelvis descent.
- The rescaling is pinned so the source's contacting sole centre lands on the anchored surface point. Rescaling about the surface point alone leaves the foot where the source had it, and a video-derived source floats several centimetres above its own scan, so the interaction mesh would still ask for a foot the contact task must drag down. This detail alone closed 3.5 cm of the step.
- The anchor is carried across gaps between contacts. Recentring on the pelvis between contacts would reintroduce the pull toward human height that the mechanism removes.

Off unless a profile enables it, and a no-op when the scene has been scaled, since there the whole world was already brought to the robot.

**Contact weight reduced 2000 to 500.** It was only that high to overcome a 20 cm gap that no longer exists.

## Files changed

- `omniretargeting/core.py` - skeleton matching, scale derivation, anchor series, and schema tolerance for mapping entries that carry either a link name or a dict.
- `scripts/diagnose_foot_pose.py` - base height continuity reporting.
- `robot_models/unitree_g1/unitree_g1.json` - `skeleton_matching` enabled, contact weight 500.
- `tests/test_skeleton_matching.py` - new, 11 tests.
- `.gitignore` - run artefacts were drowning every status and diff.

## Measurements

Reference clip, terrain unscaled, fixed terrain seed:

| | step | z jitter | z vel p95 | foot orient | clearance | slide | joint accel p95 |
|---|---|---|---|---|---|---|---|
| untouched pipeline | 2.44 cm | 11.8 mm | 1.15 | 55.96/62.16 | 24.4/23.8 | 2.23 cm | 28.7 |
| before this round | 18.33 cm | 65.3 mm | 4.64 | 0.82/0.68 | 0.23/0.01 | 0.46 cm | 27.9 |
| scaled about anchor | 9.50 cm | 35.0 mm | 2.45 | 0.73/0.57 | 0.07/−0.19 | 0.55 cm | 23.3 |
| pinned to anchor | 6.03 cm | 24.0 mm | 1.62 | 0.72/0.56 | 0.02/−0.24 | 0.36 cm | 21.9 |
| **landed** | **6.35 cm** | **22.5 mm** | **1.62** | **0.72/0.54** | **0.43/0.07** | **0.94 cm** | **21.0** |

Against the acceptance criteria:

- AC-1 base height continuity: **not met.** Step 6.35 cm against a 3 cm gate. Second-difference RMS 22.5 mm against a gate of 1.5x the untouched 11.8 mm, so 17.7 mm; also not met. Largest single step 85 mm.
- AC-2 skeleton at robot proportions, scene untouched: met. Terrain vertices identical to input; pelvis descends 11.0 cm consistent with the measured leg-length difference; contact anchors on the unscaled surface.
- AC-3 foot quality preserved: met with margin. Orientation 0.72/0.54 deg against a 2 deg gate; clearance 0.43/0.07 cm against 1 cm; slide 0.94 cm against 1 cm; ankle pitch saturation zero.
- AC-4 whole-body smoothness: met. Joint acceleration 95th percentile 21.0 against the untouched 28.7. Base velocity 1.62 against 1.15, within the 50 percent allowance.
- AC-5 expressed generally: partly met. The scale is derived, not declared. The four-robot matrix and the scale-invariance metamorphic test are queued, not run.

## Remaining items

- **AC-1's gate.** The residual step barely moves across contact weights of 200, 500 and 2000 (5.30, 5.74, 6.03 cm), so it is not the contact task overshooting and needs a different explanation. Candidates I have not yet separated: the contact task's tangential term still pulling toward a source location the shorter robot cannot occupy; the interpolated anchor drifting during long airborne stretches; the interaction mesh's terrain vertices, which are not rescaled with the skeleton and so still encode the original proportions.
- task10 scale-invariance metamorphic test, task11 four-robot matrix, task12 CLI decoupling: queued behind AC-1.

## Test status

177 passed, 16 skipped. Three pre-existing failures in `tests/test_omomo_integration.py`, all from a hardcoded dataset path `/localhdd/Datasets/OMOMO/` that does not exist on this machine. They fail identically on the branch point and are unrelated to this work.

## BitLesson Delta

Action: add

Two lessons worth recording, both about measurement rather than about the fix.

## Lesson: base-metrics-must-include-floating-base
Lesson ID: BL-20260817-base-metrics-floating-base
Scope: scripts/diagnose_foot_pose.py, motion quality metrics
Problem Description: A base height that convulsed between two levels 18 cm apart passed every smoothness check, because the checks aggregated `joint_pos` only. Joint acceleration read 27.9 against an untouched baseline of 28.7 while the pelvis was visibly jumping.
Root Cause: The floating base is stored separately from the joint angles in the output, and the metric was written over the joint array alone.
Solution: Report base position continuity as its own quantity, and partition it by the state of whichever task is suspected of driving it. Here that is contact state, which made the bistability obvious in one line.
Constraints: A single aggregate over the base is not enough either; the partition by task state is what distinguishes a step from ordinary motion.
Validation Evidence: The step, 18.33 cm before and 6.35 cm after, is invisible in joint acceleration and immediate in the partitioned base median.
Source Rounds: 0

## Lesson: derive-scales-from-the-spanning-segment
Lesson ID: BL-20260817-scale-from-spanning-segment
Scope: omniretargeting/core.py, source-to-robot scaling
Problem Description: The plan asserted the pelvis had to drop 22.3 cm, derived by multiplying the source leg length by the height ratio. The real figure is 11.0 cm. Acting on the plan's number would have overshot by a factor of two.
Root Cause: Height ratio is not leg-length ratio. This robot has proportionally longer legs than the subject relative to its height, so scaling by height understates its reach.
Solution: Measure the segment that actually has to span the gap, pelvis to ankle on both sides, on both source and robot, and take the ratio. Never substitute overall height.
Constraints: Requires the mapping to identify a pelvis and at least one ankle; reports zero and leaves the skeleton alone otherwise rather than guessing.
Validation Evidence: Measured source 0.861 m, robot 0.751 m at nominal configuration, ratio 0.8724, pelvis descent 11.0 cm; matches the observed contact-frame base height.
Source Rounds: 0
