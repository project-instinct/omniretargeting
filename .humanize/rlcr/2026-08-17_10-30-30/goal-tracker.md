# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

On branch `feat/foot-orientation-tracking` the retargeted base height snaps between two levels roughly 18 cm apart, switching ten times across the 143-frame reference clip. The switches coincide exactly with scene-contact interval boundaries. Visually the robot's pelvis convulses up and down.

The cause is a direct conflict between two objectives that are both doing what they were told:

- The interaction-mesh Laplacian tracks the source's absolute joint positions, so it pulls the robot pelvis toward the *human's* pelvis height.
- The scene-contact task pulls the sole onto the real scene surface, which for a robot with shorter legs requires the pelvis to sit lower.

Measured on the reference clip, terrain unscaled (real scale), 63 of 143 frames carry a contact constraint:

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->


Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Base height is continuous across contact state changes.
  - Positive Tests (expected to PASS):
    - The largest single-frame change in base height across the clip is no more than three times the largest single-frame change of the source pelvis height over the same clip.
    - Base height second-difference RMS is within 50 percent of the pre-change pipeline's value, measured with a fixed terrain sample seed.
    - Partitioning frames by whether a contact constraint is active, the difference between the two medians of base height is under three centimetres.
  - Negative Tests (expected to FAIL):
    - A run whose base height separates into two levels more than five centimetres apart by contact state is rejected.
    - A run that achieves continuity by disabling the contact task is rejected, verified by asserting contact clearance is still met.
    - A run that achieves continuity by smoothing the output after the fact, rather than in the objective, is rejected.

- AC-2: The source skeleton the Laplacian tracks is expressed at robot proportions while the scene stays at its captured scale.
  - Positive Tests (expected to PASS):
    - With scene scaling disabled the terrain mesh vertices are bit-identical to the input mesh.
    - The pelvis position target sits a distance below the source pelvis consistent with the pelvis-to-ankle length difference between source and robot, within two centimetres.
    - Contact anchors remain on the unscaled scene surface.
  - Negative Tests (expected to FAIL):
    - Scaling the terrain to fit the robot is rejected; the scene is a real environment the robot will be deployed into.
    - Scaling the motion and terrain together, the existing coupled behaviour, is rejected for this path because it moves the scene.
    - A formulation that leaves the pelvis target at source height and relies on the contact task to pull it down is rejected, since that is the present defect.

- AC-3: Foot quality already achieved is preserved.
  - Positive Tests (expected to PASS):

---

## MUTABLE SECTION

### Active Tasks

| Task | Tag | AC | Status | Note |
|---|---|---|---|---|
| task1 base height continuity in the diagnostic | coding | AC-1 | done | reports second-difference RMS, largest single step, and the median split by contact state |
| task2 record seeded baselines | analyze | AC-1, AC-4 | done | untouched pipeline and pre-change branch state, fixed terrain seed |
| task3 review the conflated scales | analyze | AC-2, AC-5 | done | consulted before implementing; the split adopted is scene / skeleton proportions / absolute placement |
| task4 decouple scene scaling from skeleton scaling | coding | AC-2 | done | new `skeleton_matching` config, independent of `enable_scene_scaling`, no-op when the scene was scaled |
| task5 derive the skeleton scale from segment lengths | coding | AC-5 | done | measured pelvis-to-ankle, source 0.861 m against robot 0.751 m, ratio 0.8724 |
| task6 pin the rescaling to the contact anchor | coding | AC-1, AC-2 | done | maps the source's contacting sole centre onto the anchored surface point, anchor carried across gaps |
| task7 verify continuity | analyze | AC-1 | **not met** | contact-state step 6.35 cm against a 3 cm gate; down from 18.33 cm |
| task8 verify foot quality preserved | analyze | AC-3 | met | all four thresholds met with margin |
| task9 verify whole-body smoothness | analyze | AC-4 | met | joint acceleration better than the untouched pipeline |
| task10 scale-invariance metamorphic test | coding | AC-5 | queued | waiting on task7 reaching its gate |
| task11 four-robot matrix | analyze | AC-5 | queued | waiting on task7 |
| task12 decouple scene scaling from the CLI output path | coding | AC-2 | queued | housekeeping, off the path to this defect |

### Completed Items

- Base height continuity is measurable, and the measurement is in the maintained diagnostic rather than a scratch script.
- Source skeleton scaling exists as its own mechanism, derived rather than declared, and is a no-op unless a profile asks for it.
- The rescaling is pinned to the scene, not to the source's own floating foot position.

### Explicitly Deferred

None. task10 to task12 are queued, not deferred: they are still required by the acceptance criteria and are blocked only on task7.

### Blocking Side Issues

- AC-1 is not met. The contact-state step is 6.35 cm against a 3 cm gate. Two thirds of the original 18.33 cm is gone and every other measure improved, but the remaining step is a real residual conflict, not noise: it is stable across contact weights from 200 to 2000 (5.30, 5.74, 6.03 cm), which says it is not the contact task overshooting.

### Queued Side Issues

- The contact proximity gate is an absolute distance and therefore means different things at different scene scales. It is why an earlier run detected 4 and 15 contact frames instead of 27 and 40.
- Temporal smoothness penalises a mixed-unit quantity with no timestep, so its tuned value has no physical meaning and will not transfer across frame rates.
- Waist pitch still saturates on about ten percent of frames, down from ninety-four.

### Plan Evolution Log

- **Correction to a figure in the plan's own statement of the problem.** The plan says the pelvis must sit about 22.3 cm lower than the source's, derived by multiplying the source leg length by the height ratio. Measuring the robot's pelvis-to-ankle distance directly gives 0.751 m against the source's 0.861 m, so the required drop is 11.0 cm, not 22.3 cm. The robot has proportionally longer legs than the subject relative to its height. The mechanism is unaffected, since it measures the ratio rather than using the plan's figure, but the plan's arithmetic was wrong and AC-2's tolerance refers to it.
- **DEC-1 resolved in favour of carrying the anchor forward**, as the plan's Claude position proposed. Implemented as interpolation between known anchors, which also holds the nearest anchor before the first contact and after the last, so no frame is left unpinned.
