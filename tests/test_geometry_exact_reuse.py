"""Certified hull classification and multiple exact material states retain bits."""

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.moving_scene import MovingSceneQuery
from omniretargeting.scene_geometry import HULL_TOLERANCE, SceneQuery
from tests.test_open_object_hull import V_SHEET, EVIDENCE


@pytest.mark.parametrize('mesh', [V_SHEET, trimesh.creation.icosphere(subdivisions=2)])
def test_certified_hull_decisions_equal_original_at_boundaries_and_after_reuse(mesh):
    query = SceneQuery(mesh, bounded_object=True)
    rng = np.random.default_rng(42)
    points = rng.uniform(-1.5, 1.5, (10000, 3))
    normals = query._hull[:, :3]
    centres = -query._hull[:, 3, None] * normals
    boundary = np.concatenate([centres + normals * t for t in
                               (0., HULL_TOLERANCE, .008, .008 - 1e-15, .008 + 1e-15)])
    points = np.vstack([points, boundary])
    for values in (points, points[::-1], points.copy()):
        expected = query.hull_margin(values)[:, None] > [HULL_TOLERANCE, .008]
        np.testing.assert_array_equal(query.hull_classification(values, HULL_TOLERANCE, .008), expected)
    assert query.hull_classification(np.empty((0, 3)), HULL_TOLERANCE).shape == (0, 1)


def _same(first, second):
    assert first.keys() == second.keys()
    for key in first:
        a, b = first[key], second[key]
        if key == 'rows':
            assert len(a) == len(b)
            for x, y in zip(a, b):
                _same(x, y)
        elif isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key
        elif key == 'forward_components':
            for x, y in zip(a, b):
                assert x.tobytes() == y.tobytes()
        else:
            assert a == b, key


def test_trial_does_not_evict_previous_material_or_reverse_state(monkeypatch):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
      <geom type="sphere" size=".05"/></body></worldbody></mujoco>''')
    spec = dict(mesh=V_SHEET, collision_evidence=EVIDENCE)
    query = MovingSceneQuery(model, [spec], 2)
    pose = model.qpos0.copy()
    pose[:3] = [0., 0., .03]
    first = query.query(pose, 0)
    moved = pose.copy()
    moved[0] += .02
    query.query(moved, 0)
    def unexpected(*args, **kwargs):
        raise AssertionError('exact previous transform must remain cached')
    monkeypatch.setattr(query, '_forward_samples', unexpected)
    monkeypatch.setattr('omniretargeting.moving_reverse_cache.reverse_witnesses', unexpected)
    _same(first, query.query(pose, 0))
    # Returned mutable sets must not poison the stored immutable summaries.
    first['open_known_samples'].clear()
    again = query.query(pose, 0)
    assert again['open_known_samples']


def test_cached_summary_preserves_geometry_bits_and_dof_selection():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
      <geom type="box" size=".04 .04 .04"/></body></worldbody></mujoco>''')
    query = MovingSceneQuery(model, [dict(mesh=V_SHEET, collision_evidence=EVIDENCE)], 2)
    pose = model.qpos0.copy()
    pose[2] = .02
    for dofs in ([], [0, 2, 4], None):
        expected = query.query(pose, 0, dofs)
        _same(expected, query.query(pose, 0, dofs))
        query.material_cache.clear()
        query.reverse_cache.clear()
        _same(expected, query.query(pose, 0, dofs))
