# OmniRetargeting Progress

Updated: 2026-10-10. This log consolidates completed work and current verification;
Git history retains earlier implementation details.

## Current State

- Source-agnostic `OmniRetargeter` orchestrates adapters, scene scaling,
  batch/streaming, and post-processing. Source parsing, height measurement,
  target mapping, and base orientation belong to DataSource adapters.
- Supported adapters: SMPL-X, OMOMO, LAFAN1 BVH, and Nokov BVH.
- Manual entry points: `python -m omniretargeting.main` and
  `python -m omniretargeting.batch`, using required source YAML configuration.
  Terrain belongs in YAML; automatic scene scaling is explicitly enabled.
- Motion output is NPZ, with optional contact JSON and normalized scene exports.
  MuJoCo offscreen visualization supports object meshes and `--save-video`.
- Recent commits: `8818771` (home-relative OMOMO fixture paths), `dd9438f`
  (contact/scene interface), and `f152892` (strict inputs and fallback fixes).
- The prior scoped review findings and three final failing regressions were
  resolved. Completed `agents/REVIEW.md` was removed at the user's request;
  `98b681d` compacted this log and committed the agent-instruction edits.
  The branch is published as `origin/feat/contact`; PR #12 is open.

## Configured Height Regression (2026-10-10)

- Verified PR #12's P2 finding before production edits: 18 new adapter/CLI cases
  failed because configured landmarks were dropped or rejected; another test
  reproduced SMPL-X model height overriding explicit measurement settings.
- All five adapters now accept `height_estimation` and forward it to the existing
  estimator. Precedence: runtime YAML > nested adapter options > source fields;
  the CLI inherits the top-level profile block when the selected source omits it.
  Explicit SMPL-X settings govern measurement; omitted settings retain defaults.
- Permanent regressions use actual NPZ, OMOMO FK, and BVH loading. Three real CLI
  cases exercise profile/source/YAML settings and verify exported terrain bounds
  at scale 1.35/1.8 = 0.75. All **19 new cases pass** (1.36 s).
- Full aorua/robot-data suite: **319 passed, 12 skipped**, three existing
  deprecation warnings, 35.93 s. Missing licensed SMPL-X assets account for skips.
  Logs: `/tmp/omniretargeting-height-{before,model-before,fixed,full}.log`.
  README documents the settings; `git diff --check` passed.

## Contact and Scene Interface

- `contacts.py` defines `Contact`, `EntityPose`, `EntityTrajectory`, `SceneEntity`,
  and `Scene`. Contacts identify a named source point, scene body, and body-local
  anchor; optional confidence and normals are validated. Body poses use unit
  wxyz quaternions and explicit local geometry origins.
- `contact_trajectory` is a temporal list of variable-length contact lists:
  `[]` means an available empty result; `None` means unavailable annotations.
  Copying, slicing, frame round-trips, scaling, and world transforms retain them.
- Resampling shares a timeline, uses SLERP for rigid orientations and nearest
  contact lists with earlier-frame ties. Rigid object samples follow resampled
  body poses; geometry-only entities retain their original samples and count.
- Generic detection uses nearest mesh triangles, distance, optional body-relative
  speed, and minimum contact duration. Available annotations are retained unless
  overwrite is requested. Moving bodies, terrain, and sliding anchors share the
  same geometry/pose contract. Optional `pairs` restrict candidate searches;
  omitted pairs search available scene entities. Human self-contact is unsupported.
- Detection defaults off and `contact_edge_weight` defaults to zero. The runnable
  example is `config_templates/omomo_contacts_example.yaml`; thresholds and
  weights remain tunable through YAML and the public API.
- Solver contact patches use a confidence-weighted anchor centroid and mean
  confidence. Duplicate records use maximum confidence; zero-confidence anchors
  are excluded. Spatial/contact edges combine before normalization, and source
  and robot reuse the same graph with zero environment Jacobians. Unmapped
  source contacts raise when contact weighting is enabled.
- OMOMO `body_position_mode: rest_offsets` uses skeleton FK without SMPL-X/Torch;
  `smplx` remains the default. `object_scale_mode: first_frame` bakes a constant
  scale into rigid geometry; `per_frame` retains legacy varying-scale samples.
  Terrain-only detection accepts varying scales; explicit object-body detection
  requires a rigid representation. Original scales remain in metadata.
- OMOMO option precedence is runtime > nested `adapter_options` > source fields.
  Exports preserve body origins, apply scene scale once, and retain aligned poses.
  Annotation reload converts local-anchor scale consistently.
- Batch discovers `.p`/`.pkl` archives, processes the configured `sequence_index`
  per archive, and quotes subprocess arguments with `shlex.join()` for spaced paths.
  README constructors and main/batch examples are covered by signature/parser tests.

## Strict Inputs and Solver Failures

- Target count and named order are checked before orientation or extraction.
  Streaming also checks DataSource-declared names as each frame arrives, including
  lazy declarations; unnamed positional inputs remain supported without loading
  the full source. Batch loading preserves frame identity.
- Invalid mapped robot links report the source target and link. Explicit invalid
  feet and enabled stabilization without usable feet raise; omitted feet may infer.
- Supplied root poses require finite exact shapes and unit wxyz orientations;
  framerates and source/human heights must be positive and finite, with aliases
  agreeing. Dense optional fields reject mixed availability; sparse contacts may
  contain `None`.
- Unknown solver, nested, and OMOMO controls raise. Penetration-slack key/type
  checks apply to every resolver at both core and solver boundaries; valid
  inactive settings remain available without activating slack handling.
- Assumed 1.7/1.75 m heights were removed. Missing landmarks yield no measurement;
  automatic scaling requires a measured or explicitly supplied positive height.
  Array wrappers expose `source_height` and default `enable_scene_scaling=False`.
- Backend exceptions retain QP/frame context and causes. Exhausted recovery raises
  before returning a pose or advancing stream state. Successful numerical recovery
  remains supported, and temporary solver settings are restored on failure.

## Previous Verification (aorua, 2026-10-09)

- Installed environment: `~/miniconda3`, `conda activate robot-data`.
  Missing runtime dependencies were installed; tests need no temporary dependency
  directory or custom `PYTHONPATH`. Dataset references use `~/Datasets/...`.
  Consult `agents/computation/` before starting a new verified compute workflow.
- Repository suite: **300 passed, 12 skipped**, three existing deprecation warnings,
  34.49 s. Log: `/tmp/omniretargeting-review-gaps-full.log`.
- Exact follow-up review cases plus strict-input suite: **94 passed**, including
  all three formerly failing cases. Log: `/tmp/omniretargeting-review-gaps-fixed.log`.
- Independent full-suite recheck plus four temporary review cases: **304 passed,
  12 skipped**, three existing warnings, 34.63 s. Log:
  `/tmp/omniretargeting-review-recheck.log`. No scoped finding remained reproducible.
- The 12 skips require unavailable licensed SMPL-X assets. Prepared OMOMO fixture
  tests use explicit `rest_offsets` FK; all nine integration tests pass.
- Real OMOMO sequence 318 (`sub17_floorlamp_023`) completed **109 finite frames at
  45 fps**, with unit base quaternions, 158 contacts, and aligned scene exports.
  Automatic scale: `0.7589267358`; export/reload geometry error: at most `8.41e-9 m`.
  Latest CLI artifacts/log: `/tmp/omniretargeting-strict-contact*`.
- Terrain-only varying-scale detection also completed 109 frames with 154 contacts.
  Earlier main/batch runs verified 73-frame clips, 15 fps resampling, and manual
  scale 0.8; batch completed 1/1 archive job. Artifacts: `/tmp/omniretargeting-hsoi-*`.
- `git diff --check` passed for the implementation. No pre-commit configuration
  is present. This documentation cleanup does not change executable code.

## Solver History and Established Behavior

- September 2026 penetration correction uses native MuJoCo tangent coordinates,
  `mj_differentiatePos` residuals, and `mj_integratePos` updates. Physical per-DOF
  bounds and configurable translation/rotation/joint weights replace the mixed-unit
  trust region. Candidates undergo nonlinear feasibility checks/backtracking;
  optional diagnostics report slack/restoration and correction allocation.
- `hard_constraint_slack` activates its defaults when selected, with soft tolerance,
  a hard penetration backstop, and scaled slack variables. Other resolvers are
  `hard_constraint` and `xyz_nudge`. Terrain/stabilization share primitive sampling;
  deep signed constraints are limited to upward-facing support surfaces.
- September remote verification on `ziwen-galaxea-desktop`: **149 passed, 12 skipped**.
  Real OMOMO cases and a 9/9-file SMPL-X batch (4,256 frames) had no 30 mm hard-bound
  or joint-limit violations; quaternion norm error was at most `1.12e-15`.
  Remote artifacts: `/tmp/omniretargeting-hard-slack-Loy6rD`; ask before deleting.
  Protected remote `kengo.json` was only read; no `edp_batch.py` exists.
- July 2026 added optional bone-direction warm initialization/refinement,
  exponential Laplacian weights reused from the source graph, and base XY tracking.
  Distance decay is an inverse length scale; whole-body meshes need appropriate
  tuning (kappa 30 lost terrain anchoring in the recorded test; kappa 3 improved it).
- Direct CLARABEL QP assembly removed CVXPY, with numeric penetration rows and
  cached frame/iteration constants. Historical seeded benchmark: 245.7 to 34.8
  ms/frame (7.1x); maximum configuration difference `5.2e-5`. Later terrain checks
  removed the unsafe unsigned center prefilter. `clarabel` is declared directly.
- MuJoCo box-pair contact overflow falls back from `mj_collision` to pairwise
  `mj_geomDistance`. LAFAN1/Nokov retain source height without shifting feet to Z=0.

## Key Files and Remaining Work

- Core/math: `omniretargeting/core.py`, `omniretargeting/retargeting.py`.
- Contacts: `omniretargeting/contacts.py`; motion containers/adapters:
  `omniretargeting/data_sources/`.
- Entry points: `omniretargeting/main.py`, `omniretargeting/batch.py`.
- Robot profiles: `robot_models/{unitree_g1,unitree_h1,booster_k1,hightorque_mini_pi_plus}/`.
- Tests: `tests/test_strict_inputs.py`, contact/scene tests, `tests/test_basic.py`,
  object/OMOMO integration tests, and `tests/data_sources/`.
- Next work: assess/tune HOI attachment and retargeting quality on real motions and
  validate broader LAFAN1 batch output quality. Executable data flow is verified;
  exact attachment quality and object non-penetration are not established.
- OMOMO per-frame object visualization/scaling remains excluded by the user's
  dataset-inaccuracy decision. Terrain sampling is unseeded and can vary outputs
  between runs; retain this limitation when comparing quality or performance.
