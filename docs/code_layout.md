# Code layout

This map orients changes; it is not a module freeze. Keep Python files below
1000 lines and split by responsibility rather than by making another mixed
pipeline file. `tests/test_module_size.py` enforces the limit.

| Area | Entry points | Responsibility |
| --- | --- | --- |
| CLI and orchestration | `main.py`, `core.py`, `_core/` | Parse configuration, load sources and robots, select a backend, assemble contact inputs, run retargeting, post-process, and export. |
| Source adapters | `data_sources/` | Translate one input format to the source-neutral `MotionData` contract. Format-specific parsing, SMPL-X extraction, surface sampling, inertia, and frame provenance stay here. |
| Robot model | `robot_config.py`, `robot_models/` | Load robot geometry, joints, collision geoms, limits, and declared correspondence facts. Algorithm policy belongs in solver defaults, not robot profiles. |
| Framewise solver | `retargeting.py`, `_solver/` | Solve one frame with the interaction graph, contact hierarchy, orientation and posture tasks, and bounded SQP steps. |
| Sequence solver | `sequence_retargeting.py`, `_sequence/`, `_core/sequence.py` | Solve the whole trajectory with temporal tasks, source-shape initialization, guarded nonlinear steps, and windowed correction. |
| Fast solver | `_fast/pipeline.py` and neighbouring `_fast/` modules | Build the fixed source-shaped reference, acquire contacts, repair penetration, verify per-joint budgets, and report skipped tasks. |
| Mink solver | `mink_retargeting.py` | Lightweight mink-based alternative used for comparison and debugging. |
| Contact evidence | `source_contacts.py`, `contact_phase.py`, `contact_mechanics.py`, `contact_support.py`, `object_control*.py`, `*_support_evidence.py` | Detect source contacts in adapters, rank supports/control with mechanics, and carry evidence into generic solver tasks. |
| Surfaces and posture | `surface_*.py`, `full_body_self_contact.py`, `pose_relations.py`, `body_relations.py` | Build source/robot surface correspondence, self-contact and posture tasks, and morphology-normalized relation diagnostics. |
| Scene and collision | `scene_geometry.py`, `moving_scene.py`, `geometry_query.py`, `triangle_*.py`, `self_collision*.py` | Query original static/moving geometry and robot collision geoms, including open-surface uncertainty and self-collision. |
| Correction and refinement | `trajectory_refinement.py`, `_refinement/`, `local_collision_repair.py`, `whole_body_correction.py`, `staged_contact_repair.py` | Improve an existing trajectory while preserving contacts, posture locality, continuity, and collision acceptance. |
| Motion budgets and quality | `motion_budgets.py`, `motion_quality.py`, `trajectory_validation.py`, `trajectory_fidelity.py` | Define per-joint temporal budgets and validation/reporting fields used across stages. |
| Review tools | `scripts/review_*.py`, `scripts/trumans_*review*.py` | Measure the four AGENTS metrics and compare clips without changing motion. |

Entry-point compatibility matters more than this table. When moving code, keep
the public call signature or update callers and tests in the same change.
