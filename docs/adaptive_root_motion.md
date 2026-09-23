# Root translation for reachability with source back orientation

The robot may move its root horizontally to satisfy fixed interaction targets.
Source motion, scene geometry, object motion, contact anchors, timing and metric
scale remain unchanged. Root location is a soft preference; back orientation is
an independent task, so shorter arms need not be compensated only by bending.

## Research basis

- [Harmanoid / It Takes Two, §4.1](https://arxiv.org/html/2510.10206v1#S4.SS1)
  optimizes relative root translation and rotation to align corresponding
  inter-agent contacts. We borrow the separation of root placement from contact
  geometry, not its human-shape fitting or changes to an intermediate SMPL root.
  Our variables belong to the robot and may vary smoothly along the sequence;
  this is an adaptation of the mechanism, not a reproduction of that pipeline.
- [OmniRetarget, §III](https://arxiv.org/html/2509.26633v1)
  preserves body/object/terrain relations through interaction-mesh Laplacian
  deformation with kinematic constraints. The existing source-fixed graph remains
  the whole-body shape objective alongside explicit contact and orientation tasks.
- [Rouxel et al., Multicontact Motion Retargeting (2022)](https://discovery.ucl.ac.uk/id/eprint/10148414/)
  jointly optimizes whole-body configurations and force equilibrium. It motivates
  allowing the body to participate in reach adaptation; force equilibrium and
  dynamic stability are **not implemented** by the kinematic changes here.

## Objective and target semantics

For robot root translation `r(t)` and immutable source anchor `s(t)`, the sequence
solver penalizes `delta_xy(t) = r_xy(t) - s_xy(t)` and its first, second and third
time derivatives. Derivatives use the actual frame rate. The regularizer acts on
the correction, preserving the timing of fast source movement. Root rotation
tracks the source using three rotation axes; temporal terms act on the relative
rotation error quaternion, rather than on the source's own angular motion.

Calibrated body rotations retain the existing robot bind pose and optional
parent-relative frame conventions. Parent-relative tasks preserve the waist/back
relationship without creating a spurious root-rotation gradient. Position-only
sources retain their semantic torso axes. Bootstrap and final optimization use
the same source orientation targets.

Free-hand positions use `r(t) + scale * (hand_source(t) - s(t))`; an explicitly
configured legacy height offset is still respected by the framewise/refinement
paths. Both sides of moving-target residuals are differentiated. Contacting hands
use the original world contact anchors. Feet keep world targets in both regimes.
No footstep is inserted: if fixed supports prevent reaching, residuals remain
visible instead of moving the anchor or claiming the target was reached.

## Defaults and compatibility

The sequence defaults are:

```json
{
  "lock_root_xy": false,
  "root_xy_sigma_m": 0.25,
  "root_xy_derivative_sigmas": [0.8, 3.6, 108.0],
  "root_orientation_sigma_rad": 0.05
}
```

The sigmas normalize finite soft costs; they are not hard error tolerances or
guarantees. Translation steps remain bounded inside each nonlinear iteration.
Explicit `lock_root_xy: true` still supports locked-path ablations.

Framewise defaults disable `preserve_noncontact_baseline` and
`local_contact_chain_only` so contact solving can use the whole body. The
`base_position_constraint` has `lock_xy: false`, `hard_constraints: false`,
`use_initial_offset: false`, `lock_initial_xy: false`, and `xy_sigma: 0.25`.
`base_orientation_weight: 400` separately encourages source trunk orientation.
Explicit old locks/envelopes remain supported. A soft base cost is no longer
mistakenly applied as a hard XY projection after the solve.

Local collision repair and whole-body post-correction still preserve the
retargeted floating base. These are separate stages from reach adaptation.

## Verification

`tests/test_adaptive_root_motion.py` compares locked and free-root reaching with
two fixed supports, exercises both framewise coordinate modes, checks world and
parent-relative rotation Jacobians, and verifies free-hand translation invariance.
Rapid source travel/rotation tests at 30 and 60 Hz cover peak timing and quaternion
sign changes near 180 degrees. Existing contact, explicit-lock, calibration,
refinement and module-size regressions also apply.

For motion acceptance, inspect world endpoint errors, contact residuals, source
back-angle errors, root XY offset, temporal jitter and actual scene penetration
separately. Finite weighted costs and sampled geometry are not a physical tracking
or continuous collision-free guarantee.

## 2026-09-22 validation

The pre-change code checkpoint is `fdf77ed`. Final regression: **560 passed,
6 skipped** (`pytest -q tests`), including the required contact-priority suite.
Five CVXPY inaccurate-solution warnings remain in existing solver tests. Ruff
checks of the new orientation modules, sequence integration and new tests pass.

A controlled replay of `sub16_largebox_000` (89 frames at original cadence) used
the same source, contacts, fixed material markers, seed, morphology and solver
parameters, changing only `lock_root_xy`. Both variants then underwent the same
bounded whole-body correction against the original moving box. This is a
prepared replay of the sequence API, not a claim that all source adapters or
physical controllers have been evaluated. The replay uses source root rotations
and shoulder/pelvis axes; calibrated body rotations are additionally covered by
the core integration regression.

| Measure after bounded correction | Locked XY | Adaptive XY |
| --- | ---: | ---: |
| Root horizontal offset, mean / max | 0 / 0 cm | 8.83 / 12.59 cm |
| Root absolute vertical offset from source, mean / max | 11.88 / 15.59 cm | 11.67 / 15.28 cm |
| Back longitudinal-axis error, mean / max | 3.38 / 10.17° | 3.35 / 8.35° |
| Shoulder-span axis error, mean / max | 2.90 / 8.14° | 2.74 / 7.76° |
| Root rotation error, mean / max | 5.88 / 15.49° | 5.78 / 15.16° |
| Left foot world error, mean / max | 1.44 / 3.02 cm | 0.88 / 3.10 cm |
| Right foot world error, mean / max | 1.65 / 4.95 cm | 0.77 / 2.38 cm |
| Left hand source-world error, mean / max | 5.14 / 15.19 cm | 11.16 / 19.00 cm |
| Right hand source-world error, mean / max | 5.31 / 19.49 cm | 7.68 / 16.59 cm |
| Right sticking-hand error, max (8 frames) | 2.26 mm | 2.90 mm |
| Right hand contact normal error, max (15 frames) | 5.29 mm | 2.37 mm |
| Scalar-joint acceleration RMS | 6.43 rad/s² | 4.96 rad/s² |
| Scalar-joint jerk RMS | 117.19 rad/s³ | 85.99 rad/s³ |
| Added root XY acceleration RMS | 0 m/s² | 0.86 m/s² |
| Sampled ground penetration | 0 mm | 0 mm |
| Original-box reverse intersection, max | 0 mm | 4.31 mm |

Hand source-world errors are reported separately: the free left hand has no
contact frames, and free hands follow the robot-relative target. Their increased
absolute world error is expected when the body moves, and must not be presented
as improved source-world tracking. Raw acceleration/jerk also include intended
motion; their reduction alone is not proof that only solver chatter was removed.
Back axes measure source/robot geometric directions, not identical human and
robot spine joint angles. Back alignment remains a soft objective.

Adaptive XY initially introduced an 8.16 mm reverse intersection at **frame 32
(zero-based), `right_wrist_yaw_link`**. Three accepted bounded joint corrections
reduced it to 4.31 mm; interframe reverse depth fell from 8.02 to 4.19 mm. Correction
kept the floating root exactly fixed and moved sampled collision points at most
10.06 mm horizontally and 4.54 mm vertically. The sticking-hand maximum increased
from 2.15 to 2.90 mm within the existing 1 mm regression budget. The locked variant
accepted no correction and remains explicitly reported as `no_feasible_update`.

The original box has 3 reliable closed components and 300 unknown/open components.
Closed-component signed-depth checks found no penetration; reverse checks use
41,132 original material samples and report sampled intersection lower bounds.
There are no sampled frames over 10 cm. **Unknown interiors remain unknown**, so
these figures do not certify the whole trajectory collision-free or ready for a
physical controller. No scene convexification or rescaling was used.

Artifacts and the independent evaluation script are in
[`out/adaptive_root_20260922`](../out/adaptive_root_20260922/):
`comparison.json`, `back_axes_after.json`, per-variant `sequence_report.json`,
`correction_report.json`, original solver NPZs, and corrected qpos arrays.
Replay with `scripts/omomo_sequence_trial.py`, using `--lock-root-xy` only for
the locked comparison and separate `--output-dir` paths.
