# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

<!-- Add lessons below using the strict template. -->

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
Problem Description: A plan asserted the pelvis had to drop 22.3 cm, derived by multiplying the source leg length by the height ratio. The real figure is 11.0 cm. Acting on the stated number would have overshot by a factor of two.
Root Cause: Height ratio is not leg-length ratio. The robot has proportionally longer legs than the subject relative to its height, so scaling by height understates its reach.
Solution: Measure the segment that actually has to span the gap, pelvis to ankle on both sides, on both source and robot, and take the ratio. Never substitute overall height.
Constraints: Requires the mapping to identify a pelvis and at least one ankle; reports zero and leaves the skeleton alone otherwise rather than guessing.
Validation Evidence: Measured source 0.861 m, robot 0.751 m at nominal configuration, ratio 0.8724, pelvis descent 11.0 cm; matches the observed contact-frame base height.
Source Rounds: 0
