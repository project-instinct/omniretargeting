"""OmniRetargeter.retarget_motion orchestration and postprocessing penetration guard."""

from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def mock_collision_pipeline(monkeypatch):
    """These four orchestration tests use 3-column fake motions, not qpos.

    Real FK acceptance and save-order contracts are exercised separately in
    test_local_collision_repair.py.
    """
    from omniretargeting import core
    from omniretargeting import self_collision
    monkeypatch.setattr(self_collision, "SelfCollisionQuery", lambda *args, **kwargs: Mock())
    monkeypatch.setattr(self_collision, "summarize_self_collisions",
                        lambda *args, **kwargs: {"status": "mocked", "max_depth": None})
    monkeypatch.setattr(core.SceneQuery, "from_tasks", lambda *args, **kwargs: Mock())
    monkeypatch.setattr(core.OmniRetargeter, "_clamp_final_joint_safety_limits", lambda self, q: q)
    monkeypatch.setattr(core, "select_safe_postprocessing",
                        lambda model, baseline, candidate, *args, **kwargs: (candidate, {}, {}))
    monkeypatch.setattr(core, "repair_local_collisions",
                        lambda model, candidate, *args, **kwargs: (candidate, {"status": "mocked", "events": []}))


def test_retarget_motion_uses_identity_terrain_scale_by_default(mock_collision_pipeline):
    from omniretargeting import OmniRetargeter

    original_terrain_copy = Mock(name="original_terrain_copy")
    scaled_terrain = Mock(name="scaled_terrain")
    processed_trajectory = np.full((2, 22, 3), 7.0, dtype=float)

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = Mock()
    retargeter.terrain_mesh = Mock()
    retargeter.terrain_mesh.copy.return_value = original_terrain_copy
    retargeter._compute_terrain_scale = Mock(return_value=2.5)
    retargeter._scale_terrain_mesh = Mock(return_value=scaled_terrain)
    retargeter._process_smplx_trajectory = Mock(return_value=processed_trajectory)
    retargeter._perform_retargeting = Mock(return_value=np.array([[1.0, 2.0, 3.0]]))
    retargeter._visualize_trajectory = Mock()

    smplx_trajectory = np.ones((2, 22, 3), dtype=float)

    terrain_scale, retargeted_motion = retargeter.retarget_motion(
        smplx_trajectory,
        visualize_trajectory=False,
        enable_terrain_scaling=False,
    )

    assert terrain_scale == 1.0
    assert isinstance(retargeted_motion, np.ndarray)
    retargeter._compute_terrain_scale.assert_not_called()
    retargeter._scale_terrain_mesh.assert_not_called()
    retargeter.terrain_mesh.copy.assert_called_once_with()
    retargeter._process_smplx_trajectory.assert_called_once_with(smplx_trajectory, 1.0)
    retargeter._perform_retargeting.assert_called_once_with(
        processed_trajectory,
        original_terrain_copy,
        base_orientations=None,
        base_translations=None,
    )


def test_retarget_motion_applies_terrain_scale_when_enabled(mock_collision_pipeline):
    from omniretargeting import OmniRetargeter

    scaled_terrain = Mock(name="scaled_terrain")
    processed_trajectory = np.full((2, 22, 3), 9.0, dtype=float)

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = Mock()
    retargeter.terrain_mesh = Mock()
    retargeter._compute_terrain_scale = Mock(return_value=2.5)
    retargeter._scale_terrain_mesh = Mock(return_value=scaled_terrain)
    retargeter._process_smplx_trajectory = Mock(return_value=processed_trajectory)
    retargeter._perform_retargeting = Mock(return_value=np.array([[4.0, 5.0, 6.0]]))
    retargeter._visualize_trajectory = Mock()

    smplx_trajectory = np.ones((2, 22, 3), dtype=float)

    terrain_scale, retargeted_motion = retargeter.retarget_motion(
        smplx_trajectory,
        visualize_trajectory=False,
        enable_terrain_scaling=True,
    )

    assert terrain_scale == 2.5
    assert isinstance(retargeted_motion, np.ndarray)
    retargeter._compute_terrain_scale.assert_called_once_with(smplx_trajectory)
    retargeter._scale_terrain_mesh.assert_called_once_with(2.5)
    retargeter._process_smplx_trajectory.assert_called_once_with(smplx_trajectory, 2.5)
    retargeter._perform_retargeting.assert_called_once_with(
        processed_trajectory,
        scaled_terrain,
        base_orientations=None,
        base_translations=None,
    )


def test_retarget_motion_skips_root_nudges_before_fixed_base_acceptance(mock_collision_pipeline):
    from omniretargeting import OmniRetargeter

    original_terrain_copy = Mock(name="original_terrain_copy")
    processed_trajectory = np.full((2, 22, 3), 7.0, dtype=float)
    raw_motion = np.array([[1.0, 2.0, 3.0]])
    stabilized_motion = np.array([[1.5, 2.5, 3.5]])

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = Mock()
    retargeter.terrain_mesh = Mock()
    retargeter.terrain_mesh.copy.return_value = original_terrain_copy
    retargeter.retargeting_config = {"penetration_resolver": "xyz_nudge"}
    retargeter._compute_terrain_scale = Mock(return_value=2.5)
    retargeter._scale_terrain_mesh = Mock()
    retargeter._process_smplx_trajectory = Mock(return_value=processed_trajectory)
    retargeter._perform_retargeting = Mock(return_value=raw_motion)
    retargeter._apply_foot_stabilization = Mock(return_value=stabilized_motion)
    retargeter._visualize_trajectory = Mock()

    smplx_trajectory = np.ones((2, 22, 3), dtype=float)

    terrain_scale, retargeted_motion = retargeter.retarget_motion(
        smplx_trajectory,
        framerate=60.0,
        visualize_trajectory=False,
        enable_terrain_scaling=False,
    )

    assert terrain_scale == 1.0
    np.testing.assert_array_equal(retargeted_motion, raw_motion)
    retargeter._apply_foot_stabilization.assert_not_called()


def test_retarget_motion_skips_foot_stabilization_for_hard_constraint(mock_collision_pipeline):
    from omniretargeting import OmniRetargeter

    original_terrain_copy = Mock(name="original_terrain_copy")
    processed_trajectory = np.full((2, 22, 3), 7.0, dtype=float)
    raw_motion = np.array([[1.0, 2.0, 3.0]])

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = Mock()
    retargeter.terrain_mesh = Mock()
    retargeter.terrain_mesh.copy.return_value = original_terrain_copy
    retargeter.retargeting_config = {"penetration_resolver": "hard_constraint"}
    retargeter._compute_terrain_scale = Mock(return_value=2.5)
    retargeter._scale_terrain_mesh = Mock()
    retargeter._process_smplx_trajectory = Mock(return_value=processed_trajectory)
    retargeter._perform_retargeting = Mock(return_value=raw_motion)
    retargeter._apply_foot_stabilization = Mock()
    retargeter._visualize_trajectory = Mock()

    smplx_trajectory = np.ones((2, 22, 3), dtype=float)

    terrain_scale, retargeted_motion = retargeter.retarget_motion(
        smplx_trajectory,
        framerate=60.0,
        visualize_trajectory=False,
        enable_terrain_scaling=False,
    )

    assert terrain_scale == 1.0
    np.testing.assert_array_equal(retargeted_motion, raw_motion)
    retargeter._apply_foot_stabilization.assert_not_called()


def test_penetration_guard_rolls_back_local_regression_without_joint_blending():
    from omniretargeting.local_collision_repair import select_safe_postprocessing
    from tests.test_local_collision_repair import setup_motion
    model, baseline, scene = setup_motion(30)
    candidate = baseline.copy()
    candidate[2:5, 8] += 0.01  # Admissible earlier change must survive.
    candidate[14:17, 8] = 0.01  # Separate bad interval enters the box.
    selected, stats, report = select_safe_postprocessing(
        model, baseline, candidate, scene, {}, {}, {}, framerate=30)
    assert stats['max_depth'] <= 0.001
    np.testing.assert_array_equal(selected[2:5], candidate[2:5])
    np.testing.assert_array_equal(selected[14:17], baseline[14:17])
    assert 0 < report['accepted_fraction'] < 1


def test_penetration_guard_can_retain_improved_known_deep_residual():
    from omniretargeting import core
    from tests.test_local_collision_repair import setup_motion
    from omniretargeting.scene_geometry import measure_scene_penetration
    import trimesh
    model, baseline, _ = setup_motion(3)
    scene = trimesh.creation.box(extents=[0.6]*3)
    baseline[:, 7:9] = [0, 0.1]
    candidate = baseline.copy(); candidate[:, 8] += 0.02
    before = measure_scene_penetration(model, baseline, scene)
    after = measure_scene_penetration(model, candidate, scene)
    selected, stats, fraction = core._select_penetration_safe_refinement(
        model, baseline, candidate, scene, {'target_max_depth': 0.1}, before, after)
    assert 0.1 < stats['max_depth'] < before['max_depth']
    assert fraction == 1.0
    np.testing.assert_array_equal(selected, candidate)


def test_fast_result_digest_covers_the_final_clamp_for_cli_save(mock_collision_pipeline):
    from omniretargeting import OmniRetargeter
    from omniretargeting.local_collision_repair import trajectory_digest

    source = np.ones((2, 22, 3))
    candidate = np.array([[1.0, 2.0, 3.0], [1.1, 2.2, 3.3]])
    clamped = candidate.copy()
    clamped[:, 2] = 2.5
    owner = OmniRetargeter.__new__(OmniRetargeter)
    owner.robot_model = Mock()
    owner.terrain_mesh = Mock()
    owner.retargeting_config = {'sequence_solver': {'enabled': True}}
    owner.last_sequence_report = {'backend': 'fast'}
    owner._process_smplx_trajectory = Mock(return_value=source)
    owner._perform_retargeting = Mock(return_value=candidate)
    owner._clamp_final_joint_safety_limits = Mock(return_value=clamped)
    _, motion = owner.retarget_motion(source, visualize_trajectory=False)
    np.testing.assert_array_equal(motion, clamped)
    diagnostics = owner.last_collision_diagnostics
    assert diagnostics['accepted_qpos_sha256'] == trajectory_digest(clamped)
    assert diagnostics['accepted_qpos_sha256'] != trajectory_digest(candidate)
    assert 'audit required separately' in diagnostics['validation_scope']
