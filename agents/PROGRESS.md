# OmniRetargeting Progress

## Architecture Summary

- Source-agnostic adapter architecture with per-source `target_mapping` and `base_orientation`
- Data sources: SMPL-X, OMOMO, LAFAN1 (BVH), Nokov (BVH)
- HOI support: `object_points` in `MotionData`/`MotionFrame`, scene scaling, OMOMO adapter
- CLI: required YAML `--source-config`, explicit scene scaling, main/batch entry points
- Visualization: MuJoCo offscreen rendering, object mesh injection, `--save-video`

## Contact Interface Commit Preparation (2026-10-09)

- Committed only home-relative paths in the three OMOMO JSON fixtures first
  (`8818771`); their `body_position_mode` additions remain with the interface.
- Prepared the contact trajectory interface, adapters, solver integration,
  CLI/export fixes, documentation, and regression tests as the second commit.
- Fresh aorua/`robot-data` full suite: **210 passed, 12 skipped**, three
  existing deprecation warnings, in 24.92 seconds. Log:
  `/tmp/omniretargeting-commit-check.log`. JSON parsing and staged whitespace
  checks passed. No pre-commit configuration is present.
- Left agent-rule edits and the latest fallback review outside these commits.

## Remaining README CLI Finding Fixed (2026-10-09)

- Read the complete current `agents/REVIEW.md` and fixed its one remaining
  finding. Per-frame OMOMO scaling stays outside the agreed review scope.
- README main workflows now require source YAML, put terrain in YAML, use
  `--enable-scene-scaling`, and describe actual normalized scene export paths.
  Removed the legacy compatibility claim and corrected argument tables,
  fixed-scale behavior, the batch model-options example, and batch output layout.
- Corrected main CLI output help to describe NPZ output.
- Added a regression that extracts all README main/batch shell commands and
  calls the real entry-point parsers before asset loading or job execution.
  It reproduced four failures before the fix; all five current commands parse.
- Verified on aorua in `robot-data`: public API/documentation, batch, contact CLI,
  and scene-export checks: **30 passed**. Log:
  `/tmp/omniretargeting-readme-cli-check.log`. `git diff --check` passed.
- Marked the review finding resolved while preserving its original reproduction.
  No retargeting or contact behavior changed.

## Review Scope Decision (2026-10-09)

- Per the user's decision, removed per-frame object scaling from the open
  findings in `agents/REVIEW.md`; treat it as OMOMO dataset inaccuracy.
- Retained the P2 README CLI finding for another agent to fix at the user's
  request. No implementation changes or new test runs were made.

## Hazard Re-review (2026-10-09)

- Rechecked `agents/REVIEW.md`, the latest fixes, and related code paths.
  Confirmed all six previous findings and the earlier three remain resolved.
- Rewrote the report within 200 lines with two reproduced pre-existing P2
  hazards: variable-scale object visualization freezes the first-frame scale,
  and README CLI commands still advertise removed options.
- Fresh verification on aorua in `robot-data`: **205 passed, 12 skipped**,
  three existing deprecation warnings, in 25.06 seconds. Independent helper
  and CLI subprocess probes confirmed both hazards; `git diff --check` passed.
- Recorded annotation timeline and mutually exclusive scaling requirements
  as documented boundaries. Package code and tests were not changed.

## Second Review Fixes (2026-10-09)

- Read the complete updated `agents/REVIEW.md` and fixed all six findings.
  Preserved the original reproductions and added their resolution.
- Geometry-only scene entities retain world samples by recovering local
  coordinates from original poses and resampling them in the body frame.
  Static and rotating regressions check sample count and pose consistency.
- OMOMO factory options now use runtime > nested `adapter_options` > direct
  source fields. A loaded-profile regression exercises actual adapter loading
  and runtime overrides of framerate and object scale mode.
- Object exports keep the body origin and apply scene scale once. Rigid
  exports use baked scene geometry and explicit body poses with unit scale;
  legacy varying-scale exports keep dimensionless recorded pose scales.
  Export/reload tests check world vertices and local contact anchor alignment.
- Exposed source orientation configuration on the generic public wrapper and
  added SMPL-X-specific default names/landmarks in the compatibility wrapper.
  Real G1 frame tests exercise both wrappers and custom name overrides;
  invalid-input tests preserve the existing ValueError before name derivation.
- Activated batch subprocesses quote argv with `shlex.join()`. A real
  subprocess regression covers paths and log directories containing spaces.
- Updated all README constructor examples, source/profile option descriptions,
  and wrapper documentation. Signature checks cover five examples; both
  quick-start blocks execute with bundled robot assets and processed positions.
- Focused verification in aorua's installed `robot-data`: **54 passed** in
  3.25 seconds. Log: `/tmp/omniretargeting-review2-focused.log`.
- Final full suite: **205 passed, 12 skipped**, with
  three existing deprecation warnings, in 25.44 seconds. Log:
  `/tmp/omniretargeting-review2-pytest.log`. Licensed SMPL-X asset tests remain
  skipped; public wrappers and processed-position README flows were executed.
- A real activated OMOMO CLI run with nested profile options and spaced paths
  completed **109 finite frames at 45fps**, with 158 contact records and scene
  scale `0.7589267358`. Reloaded exports reproduced source world geometry to
  **8.41e-9 m** maximum error and used the same object poses as annotations.
  Outputs, temporary configs, and log: `/tmp/omniretargeting review2/`.
  `git diff --check` passed.

## Second Review of Current Changes and Codebase (2026-10-09)

- Re-reviewed the working tree against HEAD `8e2c09a`, confirmed the previous
  three findings are resolved, and rewrote `agents/REVIEW.md`.
- Recorded six reproduced P2 findings: resampling drops legacy object samples
  for geometry-only scene entities (current regression); nested OMOMO adapter
  options are ignored; object exports change origins and apply scene scale
  twice; the public retargeting wrapper omits required orientation settings;
  activated batch commands split paths containing spaces; README examples
  use unsupported constructor arguments (five pre-existing codebase issues).
- Fresh verification on aorua in `robot-data`: **188 passed, 12 skipped**, with
  three existing deprecation warnings, in 23.34 seconds. Separate probes
  confirmed all six findings; `git diff --check` passed. Only this report
  and the required log were changed; package code and tests were not edited.

## HSOI Review Fixes (2026-10-09)

- Fixed all three correctness findings in `agents/REVIEW.md`; retained the
  original review and added a resolution section.
- Rigid legacy `object_points` now follow local scene samples transformed by
  the resampled pose, including SLERP rotations. Unrepresented world samples
  retain linear interpolation. Regressions use a rotating OMOMO adapter clip
  in both `first_frame` and `per_frame` scale modes.
- The CLI's OMOMO constant-scale requirement now applies only to explicitly
  requested object-body detection. Terrain-only pairs and omitted pairs work
  with varying legacy object scale; omitted pairs use available scene entities.
- Solver graph construction reduces each source-point/body patch to one
  confidence-weighted centroid with mean confidence. Duplicate observations
  use maximum confidence, zero-confidence anchors are excluded, and original
  surface annotations remain intact. This preserves the raw pair budget and
  avoids multiplying reverse anchor residual rows. The objective regression
  covers one, two, and eight near-identical anchors at weights 10 and 100.
- New regressions reproduced the original hazards. Verification on aorua in
  the installed `robot-data` environment: focused suite **46 passed**; full
  suite **188 passed, 12 skipped**, with three existing deprecated-argument
  warnings, in 22.80 seconds. Log:
  `/tmp/omniretargeting-review-fixes-pytest.log`.

- Real `main.py` checks on prepared OMOMO sequence 318 both completed at 45fps
  with **109 frames**: rigid object/terrain detection (`first_frame`, 158 contact
  records), and terrain-only detection preserving varying object scale
  (`per_frame`, 154 records). Both outputs have finite robot values, unit base
  quaternions, and aligned contact frame counts. Configs, NPZ/contact outputs,
  and logs are under `/tmp/omniretargeting-review-{rigid,terrain}*`.

## Current Changes Review (2026-10-09)

- Reviewed the working-tree changes against HEAD `8e2c09a`, including the new
  contact module, tests, HSOI design, and example configuration.
- Wrote `agents/REVIEW.md` with three P2 findings: rigid object samples diverge
  from interpolated poses after resampling; terrain-only OMOMO contact detection
  is rejected for varying object scale; multiple anchors increase a contact
  pair's solver influence despite its fixed raw edge budget.
- Verified on aorua in `robot-data`: focused suite **39 passed**; full suite
  **181 passed, 12 skipped**, with three existing deprecation warnings, in 22.68
  seconds. Separate in-memory reproductions confirmed the three findings, and
  `git diff --check` passed. Package code and tests were not changed by the review.

## HSOI Environment and Reference Data Follow-up (2026-10-09)

- Installed the missing runtime dependencies directly into aorua's `robot-data`
  conda environment: joblib 1.6.0, PyYAML 6.0.3, and joblib's cloudpickle dependency.
  Verification now runs without `/tmp/omniretargeting-hsoi-deps` or a custom
  `PYTHONPATH`.
- Replaced OMOMO reference-fixture paths and integration-test dataset lookups
  with `~/Datasets/...`. Dataset availability checks and direct joblib loading
  explicitly expand `~`; the adapter already expands its input paths.
  SMPL-X reference paths in the basic tests now also start at `~`.
- Configured OMOMO fixtures and coordinate/data validation tests explicitly to
  use the prepared dataset's `rest_offsets` FK mode. This machine does not have
  SMPL-X body-model assets; tests still check named joints, pelvis trajectories,
  object transforms, synchronized frames, and loading the available object types.
  The fixture loader forwards `body_position_mode`, so a fixture can select the
  existing SMPL-X path when those optional assets are provided.
- Corrected the private aorua computation notes to use `~/miniconda3`,
  `conda activate robot-data`, and a home-relative repository path.
- Full suite in the installed environment: **181 passed, 12 skipped, no failures**,
  with three existing deprecation warnings, in 22.91 seconds. All nine OMOMO
  integration tests pass. Log: `/tmp/omniretargeting-hsoi-installed-pytest.log`.
- Re-ran the real 73-frame OMOMO contact example through `main.py` with ordinary
  environment activation. Robot output and annotations were written to
  `/tmp/omniretargeting-hsoi-installed_retargeted.npz` and the corresponding
  `.contacts.json`; log: `/tmp/omniretargeting-hsoi-installed-main.log`.

## HSOI Contact Implementation (2026-10-09)

- Implemented `Contact`, `EntityPose`, `EntityTrajectory`, `SceneEntity`, and
  `Scene` in `omniretargeting/contacts.py`. Motion data/frame contracts validate
  aligned timelines, unique source names, explicit body poses, contact endpoints,
  finite anchors, optional confidence, normals, and unit wxyz orientations.
- Motion copy, slicing, frame/data-source round-trip, resampling, uniform scaling,
  and world-frame changes carry scene poses and sparse annotations. Resampling
  uses a shared time grid and nearest contact lists, with earlier-frame ties;
  it preserves empty and unavailable frames and supports single-frame clips.
- Added generic mesh contact detection using Open3D nearest triangle queries,
  distance, optional body-relative speed, and minimum per-pair run duration.
  Existing available annotations are retained unless overwrite is requested.
  Sliding anchors remain per-frame; moving objects and arbitrary surfaces use
  the same body-local geometry/pose contract.
- Added optional `contact_edge_weight` (default zero). Explicit mapped contact
  anchors are fixed environment vertices; raw spatial and symmetric contact
  contributions are combined before normalization, and the same graph is reused
  by source coordinates and every robot SQP iteration. Duplicate pair/anchor
  records use maximum confidence; distinct anchors share the pair's weight budget.
  Unmapped source contacts raise a named error when graph weighting is enabled.
- OMOMO supports explicit `body_position_mode: rest_offsets`, reusing the existing
  skeleton FK helper without SMPL-X/Torch. The existing `smplx` mode remains the
  default. `object_scale_mode: first_frame` explicitly bakes one constant object
  scale into rigid geometry; `per_frame` retains legacy world samples and does
  not represent varying scale as a rigid scene trajectory. Original scales are
  retained in metadata. World samples are never transformed twice.
- Added `config_templates/omomo_contacts_example.yaml`, editable JSON annotation
  output beside the normalized NPZ, annotation reload with scale conversion,
  source-YAML target-mapping overrides, and README manual/API examples.
  Batch discovery now recognizes OMOMO `.p`/`.pkl`; `--file-pattern` can select
  sequence archives. Each matching archive runs the configured `sequence_index`;
  batch does not implicitly expand every sequence in an archive.
- Verified directly on aorua in `robot-data`. Actual conda installation is
  `/home/leo/miniconda3`, not the Anaconda path in the computation notes.
  Missing `joblib`, PyYAML, and Black were installed only under
  `/tmp/omniretargeting-hsoi-deps` for verification; no conda environment was edited.
  OMOMO/CLI dependencies are declared in both package manifests and README.
- Full suite: **174 passed, 16 skipped, 3 failed**. The three existing OMOMO
  integration fixture failures reference unavailable `/home/ziwen/Datasets/...`
  paths. All new contact and adapter tests pass, including detection on rotating
  moving bodies, sliding, empty/unavailable frames, changing pairs, articulated
  IDs, discrete resampling, scaled/world-transformed anchors, graph equality,
  and zero environment Jacobians.
- Real prepared OMOMO sequence 318 (`sub17_floorlamp_023`): `main.py` completed
  all **73 frames**, detecting 106 records with the example thresholds (103 toe/
  terrain and 3 wrist/object records). `batch.py` completed **1/1** archive job
  through its subprocess path and wrote 73 robot/contact frames. Separate real
  runs verified automatic scaling plus 15fps resampling (**37 frames**, scale
  `0.7589267358`) and manual scale `0.8` (**73 frames**). Saved entity translations
  and local-anchor scale conversion were independently checked against source
  data. All robot outputs are finite; quaternion norm error is below `7e-16`.
  Output artifacts/logs are under `/tmp/omniretargeting-hsoi-*`.
- Contact thresholds and graph weights remain tuning parameters. This verification
  establishes executable data flow and math consistency, not exact attachment
  quality or object non-penetration guarantees.

## HSOI Interface Simplification (2026-10-09)

- Defined `contact_trajectory` as a temporal list of variable-length contact lists:
  each contact carries its human point, scene body, and local contact location.
- Replaced packed offsets and the availability flag with direct frame lists;
  `[]` means no detected contacts, and `None` means unavailable annotations.
- Removed `evaluation_scope` and exhaustive-pair coverage rules from the design.
- Contact frames report detected records, an available empty result, or unavailable
  annotations; detector candidate selection remains in detector configuration.

## HSOI Interface Design (2026-10-08)

- Documented the proposed human/entity trajectories and sparse per-frame contacts
  in `agents/HSOI_TRAJECTORY.md`; implementation remains pending.
- Contact records carry their own source-point/scene-body endpoints, allowing
  pairs to change each frame without persistent contact or track IDs.
- Consistency review clarified annotation availability, source-point resolution,
  legacy world-space object samples,
  discrete contact resampling, and shared normalized source/robot graph weights.
- Added implementation tasks and acceptance cases to the design document;
  changes remain documentation-only.

## Current Work (2026-09-07)

### `hard_constraint_slack` correction-allocation fix

- Implemented the plan in `agents/HARD_CONSTRAINT_SLACK_ANALYSIS.md`:
  - `hard_constraint_slack` now enables its documented defaults even when
    `penetration_slack` is omitted; stale slack parameters do not activate
    slack for other resolvers, and all slack values are validated.
  - The complete SQP now uses native MuJoCo tangent coordinates (`nv`): point
    Jacobians remain in velocity coordinates, configuration residuals use
    `mj_differentiatePos`, and accepted steps use `mj_integratePos`. Floating
    quaternions therefore stay valid by construction.
  - Replaced the mixed-unit global trust region with physical per-DOF bounds
    and configurable translation, rotation, and joint correction weights.
    Joint weights can be normalized by legal range, and the source root (or
    frame-entry pose) supplies a fixed per-frame base reference.
  - Added nonlinear candidate re-evaluation, feasibility-aware backtracking,
    convergence based on applied scaled step plus the true hard bound, explicit
    failure diagnostics, and heavily penalized restoration variables for poses
    that cannot become feasible in one physical step.
  - Added optional structured solver diagnostics for contact-row Jacobian
    support, slack/restoration values, correction norms by robot block,
    iteration count, backtracks, feasibility, and runtime.
  - Unified terrain and foot-stabilization primitive sampling. MuJoCo cylinders
    and capsules now use local Z, with correct cap centers/poles; spheres,
    boxes, and ellipsoids share the same helper.
  - Removed the unsafe unsigned center prefilter from nonlinear terrain checks.
    Nearby faces remain local constraints, while deep signed constraints are
    retained only for upward-facing support surfaces so non-watertight distant
    walls do not create global half-space obstacles.
- Added regression coverage for resolver defaults and validation, tangent-space
  finite differences, quaternion validity, physical joint limits, bent-leg
  Jacobian support, articulated correction allocation, rigid-translation
  cancellation in self-contact, primitive surfaces, and nonlinear feasibility.
- Updated README documentation for all three penetration resolvers,
  `penetration_correction`, defaults/precedence, restoration, and diagnostics.

### Ziwen data-source correction

- Updated OMOMO runnable configs, integration-test resources, adapter defaults,
  and examples to use the datasets that actually exist on
  `ziwen-galaxea-desktop`:
  - OMOMO: `/home/ziwen/Datasets/OMOMO`
  - SMPL-X models: `/home/ziwen/Datasets/smplx`
- Remote OMOMO integration verification: **9 passed**.

### Verification completed on `ziwen-galaxea-desktop`

- Focused solver suite: **71 passed, 12 skipped**.
- Final full remote suite: **149 passed, 12 skipped, 3 deprecation
  warnings** in 40.71 seconds.
- Real case 1, unscaled: 173 frames, maximum sampled penetration 1.070 mm,
  zero 30 mm hard-bound violations, zero joint-limit violations, quaternion
  norm error below `7e-16`.
- Real case 1, scaled (`0.900355`): maximum penetration 1.020 mm, zero hard
  violations, zero joint-limit violations. A seeded constrained/unconstrained
  A/B showed non-Z-only correction across base translation/rotation, legs,
  waist, and arms; mean base-Z correction was 43.4 mm.
- Real OMOMO case 2, sequence 318, unscaled and scaled (`0.853570`): the scaled
  result's maximum penetration was 1.005 mm with zero hard-bound and joint-limit
  violations.
- Real case 3: 514 frames, maximum penetration 1.002 mm, zero hard-bound and
  joint-limit violations.
- Real case 4: 781 frames, completed through the longer-timeout batch path;
  maximum penetration 29.408 mm, zero 30 mm hard-bound violations, zero
  joint-limit violations, and quaternion norm error below `1.2e-15`.
- EDP batch case 5: `omniretargeting.batch` processed all **9/9** SMPL-X files
  serially (`--max-workers 1`), totaling 4,256 frames in about 97 minutes.
  All nine NPZ outputs were written and all logs were free of tracebacks,
  timeouts, kills, and error markers.
- Independent post-run analysis passed for all nine batch outputs. Maximum
  sampled penetration ranged from 1.002 mm to 29.408 mm; every trajectory had
  zero 30 mm hard-bound violations, zero solver hard violations, zero
  joint-limit violations, and quaternion norm error at or below `1.12e-15`.
- Remote results are under
  `/tmp/omniretargeting-hard-slack-Loy6rD`; ask before deleting them.
- The protected remote `kengo.json` has only been read. No `edp_batch.py`
  exists in the local or remote checkout.

### Verification notes

- Final code/path follow-ups were synchronized to the remote test checkout
  before the full suite.
- The remote environment does not have Black installed and this repository has
  no pre-commit configuration, so no formatter was installed or run. Local
  `git diff --check` passed.
- Remote long-running commands use redirected logs under the result directory;
  this avoids losing test output when the SSH transport disconnects.

## Recent Changes (2026-07-28)

### Base Position Tracking to Fix kengo Base Drift
- Added a configurable base-position tracking cost that penalizes deviation of
  the floating-base x-y origin from the scaled source root translation. This
  directly fixes the base drift seen with distance-weighted Laplacian edges
  (TopoRetarget exponential weighting), where terrain vertices receive low
  weights and no longer anchor the global x-y position.
- New retargeting config key: `base_position_tracking_weight` (default `0.0`,
  i.e. unchanged original behavior). The cost is added to the per-frame QP as
  `w * ||q[:2] - root_translation[:2]||^2` whenever the DataSource provides
  `root_translation` and `w > 0`. Z is intentionally left free so the optimizer
  can still respect robot leg kinematics and terrain constraints.
- Plumb `root_translation` from `MotionFrame` through `core.py::retarget_frame`,
  `retargeting.py::retarget_frame`, `_optimize_configuration`, and
  `_single_optimization_step`.
- kengo.json (desktop) enables the cost with `base_position_tracking_weight: 100.0`.
- Added unit tests for config propagation and default weight.

### Retargeting Speed-Up: Direct CLARABEL QP + Terrain Prefilter (~7x)

Profiled kengo + LAFAN1 (`aiming1_subject1`, simplelab terrain): 245.7 ms/frame,
split ~47% CVXPY+CLARABEL solves (13.2 ms each: 7.4 ms CVXPY canonicalization +
5 ms solver) and ~43% penetration constraints (16 ms per SQP iteration, dominated
by a 308-point trimesh closest-point query). After the change: **34.8 ms/frame
(7.1x)** with numerically identical output (seeded A/B, 60 frames: max |dq| =
5.2e-05, mean 9.8e-07).

- **Direct QP assembly, no CVXPY** (`retargeting.py`): the auxiliary
  `lap_var = lap0 + J_L @ dqa` equality is substituted into the objective, leaving
  a small QP in `dqa` (+ unit slack vars). New `_solve_qp_clarabel()` calls
  CLARABEL directly (same solver, same default settings, same SOC trust region
  and no-SOC fallback). Warm-init QP (`_warm_init_bone_direction`) uses the same
  path. 13.2 ms -> 1.2 ms per solve.
- **Terrain proximity prefilter**: per-iteration, geoms whose center is farther
  than `threshold + bounding_radius` from the terrain are skipped before surface
  sampling (`_geom_bounding_radius`, conservative; mesh/other types fall back to
  `geom_rbound` or never-skip). Emitted rows are provably identical. 16 ms ->
  1.7 ms per iteration.
- **Numeric penetration rows**: `_compute_penetration_constraints` /
  `_penetration_constraint_terms` now return `(hard_rows, slack_rows)` numeric
  tuples instead of CVXPY constraints; tests updated to the new interface.
- **Hoisted per-frame/per-iteration constants**: `L`/`Kron` computed once per
  frame in `retarget_frame`; `Q_diag_modified` precomputed in
  `_setup_robot_config`; qdot->qvel `T` built once per SQP iteration and passed
  down (was rebuilt per point Jacobian, ~5k builds/frame).
- **Robustness**: SQP loop now breaks on non-finite cost (previously a failing
  solve burned all `max_iter` iterations on identical retries); removed the
  `sys._omni_frame_count` debug-print machinery.
- **Known pre-existing issue (not changed)**: `sample_points_on_mesh` is
  unseeded — every run samples different terrain points, so retargeting output
  varies run-to-run by ~cm. Consider a `terrain_sample_seed` config.
- CVXPY is no longer imported by `retargeting.py`; the dependency is kept in
  `pyproject.toml`/`setup.py` for downstream users.
- Test status (desktop, kengo env): 107 passed before and after; the same 9
  failures pre-exist in both (2 mock tests vs `core.py` attributes, 4 tpose
  tests vs Spine1 config requirement, 3 OMOMO tests missing the dataset).

### Code-review follow-up fixes

- Declared `clarabel` explicitly in `pyproject.toml`/`setup.py` (it is imported
  directly; previously only present transitively via cvxpy).
- Removed `cvxpy` from `pyproject.toml`/`setup.py`/README dependency lists —
  nothing in the package or tests imports it anymore (supersedes the
  "kept for downstream users" note in the speed-up entry above).
- `_compute_penetration_constraints`: dropped the now-unused `q` parameter (the
  kinematics-current precondition is documented and enforced by the caller).
- `_optimize_configuration` / `_single_optimization_step`: dropped the unused
  `adj_list` passthrough; `L`/`Kron` are now required positional parameters
  instead of `Optional` (a `None` crashed with an opaque `TypeError`).
- `_solve_qp_clarabel` prints the exception message before returning failure,
  so hand-assembly bugs are distinguishable from genuine solver failures.
- `retarget_frame`: dropped the `sp.issparse(L)` guard on the Laplacian matrix
  (`calculate_laplacian_matrix` always returns dense; the conversion is now
  unconditional). All other guards in the diff were audited and kept: the QP
  try/except, quaternion `1e-12`, and `quat_indices == 4` checks are carried
  over from the pre-change code, the `T is None` fallbacks are genuinely used
  (`_warm_init_bone_direction` calls `_compute_robot_jacobians` without T),
  and the `geom_rbound > 0 else inf` fallback is required for prefilter
  conservatism on mesh geoms.
- tests: the two `retarget_frame` mock tests
  (`..._root_pose_for_frame_zero_init_when_present`,
  `..._falls_back_to_estimated_root_pose_when_absent`) now set
  `retargeter.retargeting_config = {}` explicitly — fixing the 2 pre-existing
  failures noted above. `OmniRetargeter.__new__` bypasses `__init__`, and the
  production invariant is that `retargeting_config` always exists
  (`core.py:450` raises on its absence; the other 7 `__new__` test sites
  already set it).
- tests: `test_tpose_retargeting_alignment` (all 4 robots) fixed test-side —
  it fed a 22-joint SMPLX-layout array but let `source_target_names` fall back
  to the 11 mapped names, so mapped indices grabbed wrong joints and
  base_orientation couldn't find Spine1 (the profiles map the waist to Spine2).
  Now passes `DEFAULT_SMPLX_TARGET_NAMES` (mirroring production, where names
  come from the DataSource) and handles dict-valued target_mapping entries in
  the verification loop. Verified accuracy: 0.10-0.43 m mean joint distance.
- Test status after fixes (desktop, full suite): 113 passed; 3 pre-existing
  failures remain (OMOMO tests, dataset missing at /localhdd/Datasets/OMOMO).

## Recent Changes (2026-07-27)

### TopoRetarget Feature Set (bone-direction prior + penetration slack)
- Implemented the remaining TopoRetarget (arXiv:2606.16272) differences vs OmniRetarget
  (distance-weighted Laplacian and source-graph reuse were already in):
  - **Bone-direction prior** (Eq. 1-2, 8): relative direction of adjacent bones along
    config-defined chains, world-frame adaptation (paper uses wrist-attached frames for
    hands). Two stages: optional warm-init QP per frame (`warm_init`, Eq. 2) and a
    refinement-stage objective term in the main QP. Math helpers:
    `parse_bone_chains` / `compute_bone_direction_targets` /
    `compute_bone_direction_residual_and_jacobian` in `retargeting.py`.
  - **Penetration slack variables** (Eq. 8): hard backstop `phi >= -b` plus soft
    tolerance `phi + s >= -tau`, `0 <= s <= b - tau`, objective `w_s/2 * sum(s^2)`.
    Selected via `penetration_resolver: "hard_constraint_slack"` (a slack variant of
    the hard-constraint resolver; numeric parameters in `penetration_slack`).
    The slack is solved as `s = (b - tau) * s_unit` with
    `s_unit` in [0, 1] — mathematically identical, but avoids the ill-conditioned
    5e4-scale quadratic that stalls CLARABEL (InsufficientProgress) when `s` is
    optimized in meters directly. First-solve `SolverError` now falls back to the
    no-SOC retry instead of failing the frame.
- All paper values configurable via the retargeting config (robot profile
  `retargeting` section), defaults OFF (previous behavior unchanged):
  `bone_direction: {enabled, chains, lambda_warm=1.0, lambda_smooth=2.5, lambda_bone=0.1,
  warm_init=true, warm_init_iters=3}` and
  `penetration_resolver: "hard_constraint" | "hard_constraint_slack" | "xyz_nudge"` with
  `penetration_slack: {soft_tolerance=0.001, hard_bound=0.03, slack_penalty=1e5}`.
- kengo.json (desktop) enables the full set with paper values; chains cover both legs
  and both arms of the LAFAN1 mapping (6 adjacent bone pairs).

### Distance-Dependent Laplacian Edge Weights (TopoRetarget)
- Added optional exponential distance-dependent adjacency weights for the interaction-mesh
  Laplacian, from TopoRetarget (arXiv:2606.16272, Eq. 5): `w_ij ∝ exp(-kappa * d_ij)`,
  row-normalized, computed once on the source configuration per frame and reused for the
  robot-side Laplacian matrix.
- New: `calculate_exponential_edge_weights()` in `utils.py`; optional `edge_weights` argument
  in `calculate_laplacian_coordinates()` / `calculate_laplacian_matrix()`.
- Config (retargeting dict / robot profile `retargeting.solver`):
  `laplacian_edge_weighting: "uniform" | "exponential"` (default `"uniform"`, i.e. weighting
  OFF, original behavior) and `laplacian_distance_decay` (kappa, default 30.0).
- **kappa is an inverse length scale — tune per mesh scale.** The paper's kappa=30 is
  calibrated for dexterous hand-object meshes (cm-scale edges). A/B on kengo + LAFAN1
  (`aiming1_subject1`, scale 0.8476) showed kappa=30 disconnects body vertices from static
  terrain edges (terrain weight share ~1e-12 per row), losing global x-y anchoring
  (base xy dev vs scaled source: mean 0.295 m, max 2.07 m). kappa=3 restores anchoring
  (dev mean 0.059 m, max 0.17 m; uniform: 0.032/0.11 m) while keeping local distance
  emphasis. Rule of thumb: kappa ≈ 1 / characteristic edge length of the interaction mesh.

## Recent Changes (2026-06-22)

### LAFAN1 Batch Retargeting
- Added `omniretargeting/batch.py` — parallel batch processing with memory-based worker sizing
- Removed ground-ensuring (foot Z=0 shift) from LAFAN1 and Nokov data sources

### MuJoCo `mj_collision` FatalError Fix
- **Symptom:** `mujoco.FatalError: mj_narrowphase: collision function returned 9 contacts for geom pair, expected at most 8 from mj_maxContact`
- **Root cause:** `mjMAXCONPAIR=8` is a compile-time constant in MuJoCo. Box-box face-face overlap can produce up to 12 contacts.
- **MuJoCo upstream (v3.9.0):** Not fixed. The limit is still hardcoded.
- **Fix in `retargeting.py`:** `_prefilter_pairs_with_mj_collision()` catches `mujoco.FatalError` and falls back to `_brute_force_candidate_pairs()` (pairwise `mj_geomDistance`, negligible cost).
- **Alternative long-term fix:** Change fist geoms from boxes to capsules/spheres in the URDF.

## Key Files

### Core
- `omniretargeting/core.py` — OmniRetargeter, scene scaling, frame dispatch
- `omniretargeting/retargeting.py` — GenericInteractionRetargeter, collision constraints
- `omniretargeting/main.py` — CLI entry point (YAML + legacy)
- `omniretargeting/batch.py` — parallel batch processing

### Data Sources
- `omniretargeting/data_sources/base.py` — MotionData/MotionFrame containers
- `omniretargeting/data_sources/smplx.py` — SMPL-X adapter
- `omniretargeting/data_sources/omomo.py` — OMOMO object interaction adapter
- `omniretargeting/data_sources/lafan1.py` — LAFAN1 BVH adapter
- `omniretargeting/data_sources/nokov.py` — Nokov BVH adapter

### Robot Configs
- `robot_models/unitree_g1/unitree_g1.json`
- `robot_models/unitree_h1/unitree_h1.json`
- `robot_models/booster_k1/booster_k1.json`
- `robot_models/hightorque_mini_pi_plus/hightorque_mini_pi_plus.json`

### Tests
- `tests/test_basic.py` — CLI and regression coverage
- `tests/test_objects.py` — object-point unit coverage
- `tests/test_omomo_integration.py` — OMOMO integration
- `tests/data_sources/test_lafan1.py` — 15 tests
- `tests/data_sources/test_nokov.py` — 16 tests
- `tests/data_sources/test_smplx.py` — 6 tests

## Remaining Work

1. Validate HOI retargeting quality with real OMOMO end-to-end runs
2. Decide whether to add regression tests for YAML config loading and scaled-object export
3. Complete LAFAN1 batch retargeting and validate output quality
