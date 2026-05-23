# OmniRetargeting Progress

Last Updated: 2026-05-18

## Current Status: HOI/Object Interaction Integration In Progress 🔄

The source-agnostic adapter architecture remains in place, and this branch now extends it toward
human-object interaction support with OMOMO object sampling, object-aware visualization, and dual-mode
CLI loading (new YAML configs plus deprecated legacy CLI compatibility).

## Current Branch Focus

### Object Interaction (HOI) Support
- **Status:** 🔄 Core pieces implemented and validated with focused tests
- `MotionFrame` and `MotionData` now support per-frame `object_points`
- `MotionData` also carries optional `object_mesh` for visualization/export
- `OmniRetargeter.retarget_motion()` supports scene scaling for object points via `enable_scene_scaling`
- `GenericInteractionRetargeter` accepts object points in frame retargeting

### OMOMO Adapter
- **Status:** ✅ Implemented
- **Location:** `omniretargeting/data_sources/omomo.py`
- Loads OMOMO sequences, object meshes, and sampled object point clouds
- Transforms sampled object points into world coordinates per frame
- Exposes metadata such as object name and sequence name

### Visualization / Export
- **Status:** ✅ Implemented
- `temporary_visualization_scene()` can inject object meshes for MuJoCo visualization
- `save_trajectory_video()` and `visualize_trajectory()` accept object meshes
- CLI supports `--scaled-objects` export for scaled meshes and pose trajectories

### CLI Migration
- **Status:** 🔄 YAML mode added, legacy compatibility restored
- `--source-config` YAML loading is supported for new source-driven workflows
- Legacy CLI arguments (`--motion`, `--model-dir`, `--source`, `--source-options`) still work
- Deprecated legacy flags now coexist with YAML mode instead of breaking existing tests

## Recent Changes (2026-05-18)

### HOI Data Flow ✅
- Added `object_points` validation for frame and motion containers
- Added per-frame extraction of object points through `MotionData.iter_frames()`
- Added object-point scaling support in batch retargeting

### OMOMO Integration ✅
- Added OMOMO adapter and integration tests
- Verified object mesh loading and sampled point generation
- Verified OMOMO fixtures load successfully in focused tests

### CLI / Compatibility Fix ✅
- Fixed `omniretargeting/main.py` so YAML source configs do not break legacy CLI users
- Restored compatibility with existing `tests/test_basic.py` main-script integration coverage
- Kept deprecation path for legacy arguments while preserving current workflows

## Test Results

### Sequential Validation on marsbrain (2026-05-18)
- `pytest -q tests/data_sources/test_smplx.py` → **6 passed**
- `pytest -q tests/test_validation.py` → **8 passed**
- `pytest -q tests/test_objects.py` → **15 passed**
- `pytest -q tests/test_omomo_integration.py` → **7 passed**
- `pytest -q tests/test_basic.py::TestUtils tests/test_basic.py::test_load_robot_config_nested_source_profile tests/test_basic.py::TestPackageImport` → **9 passed**
- `pytest -q` G1 real-data main-script cases (`simplelab`, `wallflip`, `prox-sofa`) → **3 passed**
- `pytest -q` H1 real-data main-script cases (`simplelab`, `wallflip`, `prox-sofa`) → **3 passed**
- `pytest -q` Booster K1 real-data main-script cases (`simplelab`, `wallflip`, `prox-sofa`) → **3 passed**
- `pytest -q` Mini Pi Plus real-data main-script cases: `simplelab` → **passed**, `wallflip` → **passed**
- Mini Pi Plus `prox-sofa` was validated manually by running the equivalent CLI command directly on marsbrain, producing `/tmp/mini_pi_plus_prox_sofa_manual_retargeted.npz` and `/tmp/mini_pi_plus_prox_sofa_scaled.obj`
- `python -m py_compile omniretargeting/main.py` → **passed**

### Notes
- No stale `pytest` processes remained on marsbrain after cleanup.
- Some single-case `pytest` node-id invocations for the Mini Pi Plus `prox-sofa` case appeared to hang in the local task wrapper even though the underlying retargeting pipeline completed when run directly.
- The previously stale progress numbers from 2026-05-11 are no longer representative of this branch.

## Remaining Work

1. Validate object interaction retargeting quality with real OMOMO end-to-end runs, not just structure/tests
2. Clean up branch artifacts if confirmed safe (`*.backup`, draft notes)
3. Decide whether to add dedicated regression tests for YAML config loading and scaled-object export
4. Investigate why certain single-case `pytest` node-id runs can hang in the local task wrapper even when the equivalent marsbrain CLI run completes

## Key Files

### Core
- `omniretargeting/core.py` - scene scaling and frame object-point handling
- `omniretargeting/retargeting.py` - interaction mesh construction with environment/object samples
- `omniretargeting/main.py` - YAML + legacy CLI loading, visualization, export

### Data Sources
- `omniretargeting/data_sources/base.py` - motion container object fields and validation
- `omniretargeting/data_sources/omomo.py` - OMOMO object adapter
- `omniretargeting/data_sources/smplx.py` - SMPL-X adapter used by legacy/integration flows

### Tests
- `tests/test_objects.py` - object-point unit coverage
- `tests/test_omomo_integration.py` - OMOMO integration coverage
- `tests/test_basic.py` - CLI and regression coverage

## Conclusion

This branch now has the main HOI plumbing in place: object-aware motion containers, OMOMO adapter support,
object visualization/export, and a CLI that supports the new YAML path without regressing the existing
main-script workflows. The next important step is deeper end-to-end validation of actual retargeted HOI output quality.

## Investigation Note (2026-05-20)

### OMOMO Object Mesh Orientation
- The stale AGENTS.md note about base orientation has been removed; the current architecture note correctly says root orientation comes from motion_data.root_orientations.
- Diagnostics on the OMOMO floorlamp sequence showed the raw object transform convention itself was consistent with a vertical lamp when using the adapter's current point transform (scaled_points at rotation.T plus translation).
- The first render issue was a visualization/export handoff mismatch: build_object_tracks() needed to transpose each object rotation so the rendered mesh used the same world-frame convention as the sampled OMOMO object points.
- The second render issue was in MuJoCo dynamic mesh binding: _dynamic_object_specs() was storing 2 * object_mesh_ids[obj_idx] instead of the actual mesh asset id, so the renderer could attach the object transform to the wrong mesh asset/data id.
- After changing build_object_tracks() to transpose the rotation and _dynamic_object_specs() to preserve the real mesh id, a second marsbrain render test succeeded and wrote updated outputs under /tmp/omniretargeting_20260520_tests/, including floorlamp_orientation_fix_v2.mp4, floorlamp_retargeted_v2_retargeted.npz, and floorlamp_scaled_objects_v2/.
