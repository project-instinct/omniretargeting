# Resolve Base Height Bistability When Retargeting Into An Unscaled Real Scene

## Goal Description

On branch `feat/foot-orientation-tracking` the retargeted base height snaps between two levels roughly 18 cm apart, switching ten times across the 143-frame reference clip. The switches coincide exactly with scene-contact interval boundaries. Visually the robot's pelvis convulses up and down.

The cause is a direct conflict between two objectives that are both doing what they were told:

- The interaction-mesh Laplacian tracks the source's absolute joint positions, so it pulls the robot pelvis toward the *human's* pelvis height.
- The scene-contact task pulls the sole onto the real scene surface, which for a robot with shorter legs requires the pelvis to sit lower.

Measured on the reference clip, terrain unscaled (real scale), 63 of 143 frames carry a contact constraint:

| | base_z median | below source pelvis |
|---|---|---|
| contact frames | 89.95 cm | 19.06 cm |
| non-contact frames | 108.28 cm | 0.73 cm |
| difference | 18.33 cm | switches 10 times |

The geometry requires the pelvis to sit about 22.3 cm lower than the source's throughout, because the G1's pelvis-to-ankle length is 0.637 m against the source's 0.861 m. During contact the solver finds nearly that; between contacts the Laplacian returns the pelvis to human height. The two regimes are the defect.

The goal is a base height that is continuous and geometrically consistent for the whole clip, without giving up the foot orientation and contact quality already achieved: foot sole orientation within about one degree of the calibrated source frame, sole clearance on contact frames within a few millimetres, and foot slide within a contact below one centimetre.

## Acceptance Criteria

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
    - Foot sole orientation geodesic error against the calibrated source sole frame: median at most 2 degrees per foot.
    - Sole clearance on source-contact frames: median within 1 cm of the surface, per foot.
    - Foot slide within a contact interval: median at most 1 cm.
    - Ankle pitch saturation stays at zero on both feet.
  - Negative Tests (expected to FAIL):
    - A run that fixes base height while foot orientation error grows past 2 degrees median is rejected.
    - A run whose feet leave the surface by more than 2 cm median on contact frames is rejected.

- AC-4: Whole-body motion quality does not regress against the untouched pipeline.
  - Positive Tests (expected to PASS):
    - Joint acceleration 95th percentile no worse than the untouched pipeline's.
    - Peak joint acceleration no worse than the untouched pipeline's.
    - Base linear velocity 95th percentile within 50 percent of the untouched pipeline's.
    - No non-finite values, and zero joint-limit violations.
  - Negative Tests (expected to FAIL):
    - A run with base velocity spikes above 3 m/s at the 95th percentile is rejected.
    - A run that reports solver failures without recording them is rejected.

- AC-5: The change is expressed generally rather than as a constant for this robot and clip.
  - Positive Tests (expected to PASS):
    - The scale relating source skeleton to robot skeleton is derived from measured segment lengths, not declared per clip.
    - The same configuration runs on the other three humanoids in `robot_models` without per-robot numeric tuning of this mechanism.
    - Rescaling source and scene together by a factor leaves joint angles and contact labels unchanged and base translation scaled by that factor.
  - Negative Tests (expected to FAIL):
    - A hardcoded 22.3 cm pelvis drop is rejected.
    - A mechanism that only works when the source is taller than the robot is rejected.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation separates source-to-robot skeleton scaling from scene scaling, derives the scaling from measured source and robot segment lengths, keeps contact anchoring on the unscaled scene, adds a base-height continuity diagnostic to the foot-pose tool, adds regression tests for continuity and for the scale-invariance metamorphic property, and verifies the result on all four humanoids in the repository.

### Lower Bound (Minimum Acceptable Scope)

The implementation makes the Laplacian's source targets consistent with robot proportions on the unscaled-scene path, so the base height has a single regime, with the derived scaling and a continuity test on the reference clip.

### Allowed Choices

- Can use: a derived source-to-robot skeleton scale applied to the Laplacian targets only; per-segment or per-limb scaling if uniform scaling proves insufficient; a base-height objective informed by contact; reformulating what the pelvis target means; the existing contact machinery as the absolute anchor.
- Cannot use: scaling the terrain on this path; a hardcoded pelvis offset; post-hoc filtering of the output base trajectory as the primary fix; disabling the contact task; weakening foot orientation below the AC-3 thresholds.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The pipeline currently offers one switch that scales the motion and the terrain together. Three quantities are conflated:

1. the scene's scale, which is fixed by reality when the robot will be deployed there;
2. the source skeleton's proportions, which must be brought to the robot's proportions for absolute joint targets to be reachable;
3. the absolute placement of the motion in the scene, which contact should anchor.

One possible split:

```
scene            : never scaled on this path
Laplacian target : source joints rescaled about a per-frame anchor so that
                   segment lengths match the robot's, leaving the anchor where
                   the source put it
anchor           : the contacting foot when one exists, otherwise carried
                   forward from the last contact
```

Rescaling about the contacting foot rather than about the pelvis is what keeps the feet on the real surface while the pelvis drops by the leg-length difference. Between contacts the anchor is carried forward, which is also what gives continuity.

The scale itself is measurable: source pelvis-to-ankle length against robot pelvis-to-ankle length, both already computed elsewhere in this work. On the reference clip those are 0.861 m and 0.637 m.

### Relevant References

- `omniretargeting/core.py` - `retarget_motion` builds the scaled motion data and the scaled terrain from one switch; `_attach_contact_targets` runs before the frame loop and is where the contact anchors are established; `_resolve_source_to_robot_scale` computes the height ratio.
- `omniretargeting/retargeting.py` - the Laplacian objective assembly, and the scene contact and link orientation residuals added on this branch.
- `omniretargeting/scene_contact.py` - contact interval detection and anchoring.
- `omniretargeting/foot_frames.py` - sole derivation from collision geoms, which supplies robot foot geometry.
- `scripts/diagnose_foot_pose.py` - the measurement tool; base height continuity belongs here.
- `main.py` - couples scene scaling to whether a scaled-terrain output path was requested, which is worth decoupling regardless.

## Dependencies and Sequence

### Milestones

1. **Measurement**: add base height continuity and contact-state-partitioned statistics to the diagnostic, and record them for the untouched pipeline and for the current branch state.
   - Phase A: extend the diagnostic.
   - Phase B: record both baselines with a fixed seed.
2. **Decouple the switches**: separate scene scaling from source skeleton scaling, so the unscaled-scene path can still bring source proportions to the robot.
3. **Anchor and rescale**: derive the skeleton scale from segment lengths and rescale the Laplacian targets about a contact-aware anchor.
   - Depends on Milestone 2 for the plumbing and on the existing contact machinery for the anchor.
4. **Validation**: continuity and foot-quality criteria on the reference clip, the scale-invariance metamorphic test, and the four-robot matrix.
   - Depends on Milestones 1 and 3.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Add base height continuity and contact-state-partitioned statistics to the diagnostic | AC-1 | coding | - |
| task2 | Record seeded baselines: untouched pipeline and current branch state | AC-1, AC-4 | analyze | task1 |
| task3 | Review the three conflated scales and confirm the split before implementing | AC-2, AC-5 | analyze | task2 |
| task4 | Decouple scene scaling from source skeleton scaling in the config and the core path | AC-2 | coding | task3 |
| task5 | Derive the source-to-robot skeleton scale from measured segment lengths | AC-5 | coding | task4 |
| task6 | Rescale the Laplacian source targets about a contact-aware anchor, carried forward between contacts | AC-1, AC-2 | coding | task5 |
| task7 | Verify continuity and the single-regime property on the reference clip | AC-1 | analyze | task6 |
| task8 | Verify foot orientation, clearance and slide are preserved | AC-3 | analyze | task6 |
| task9 | Verify whole-body smoothness against the untouched pipeline | AC-4 | analyze | task6 |
| task10 | Add the scale-invariance metamorphic test | AC-5 | coding | task6 |
| task11 | Run the four-robot matrix without per-robot tuning of this mechanism | AC-5 | analyze | task10 |
| task12 | Decouple scene scaling from the scaled-terrain output path in the CLI | AC-2 | coding | task4 |

## Claude-Codex Deliberation

### Agreements

- The bistability is a genuine objective conflict, not numerical noise: the two levels are 18.33 cm apart and the switches align with contact boundaries.
- Contact is the quantity that must survive, because the scene is real and the robot will be deployed in it.
- The scene must not be rescaled on this path.

### Resolved Disagreements

- Whether the earlier contact blend ramp addresses this: it does not. It softens the transition over a few frames but leaves two levels; and measured on foot slide it cost 0.5 cm to 3.2 cm for a tenth of the acceleration. It was disabled for that reason, and the height defect was missed because the smoothness metric covered joint angles only and not the floating base.

### Convergence Status

- Final Status: `partially_converged`

## Pending User Decisions

- DEC-1: Whether between-contact frames should carry the last contact anchor forward or blend toward the source's own placement.
  - Claude Position: carry the last anchor forward, since it is what produces a single continuous regime and matches how the motion actually proceeds between footfalls.
  - Codex Position: not yet consulted on this specific question.
  - Tradeoff Summary: carrying forward risks drift over long airborne stretches; blending toward the source reintroduces a pull toward human height, which is the defect.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Verified Facts

- 63 of 143 frames carry a contact constraint; contact state switches 10 times.
- Base height medians: 89.95 cm with contact active, 108.28 cm without; difference 18.33 cm.
- Source pelvis height median 109.01 cm. Between contacts the robot pelvis is 0.73 cm below it; during contact 19.06 cm below.
- Source pelvis-to-ankle length 0.861 m; the same length on the G1 is 0.637 m, a difference of 22.3 cm.
- Base height second-difference RMS: 11.8 mm for the untouched pipeline, 65.3 mm on the current branch.
- Base linear velocity 95th percentile: 1.15 m/s untouched, 4.64 m/s on the current branch.
- Joint-angle acceleration is not affected in the same way: 95th percentile 28.7 untouched against 27.9 on the current branch, which is why the defect escaped the existing metrics.
