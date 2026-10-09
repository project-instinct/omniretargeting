"""Reuse preparation without inheriting query state or changing witnesses."""
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from omniretargeting._fast import repair_geometry_setup
from omniretargeting._fast import repair_static_query
from omniretargeting.self_collision import SelfCollisionQuery
from omniretargeting.scene_geometry import SceneQuery
from tests.test_fast_backend import _objective


def test_acquisition_self_query_keeps_filtering_but_resets_buffers():
    objective, poses, _ = _objective(temporal=True)
    old = objective.self_collision = SelfCollisionQuery(objective.model)
    old.query(poses[0])
    new = repair_geometry_setup.self_query(objective)
    assert new is not old and new.data is not old.data
    assert new.pair_queries == 0
    assert new.pairs.tobytes() == old.pairs.tobytes()
    assert new.structural_pairs == old.structural_pairs
    assert new.model_pairs.tobytes() == old.model_pairs.tobytes()
    assert new.model_pairs is not old.model_pairs
    assert new.structural_evidence == old.structural_evidence
    assert new.structural_evidence is not old.structural_evidence
    assert new.structural_sampling == old.structural_sampling
    assert new.structural_sampling is not old.structural_sampling
    first, second = old.query(poses[0]), new.query(poses[0])
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert a['pair'] == b['pair']
        assert a['clearance'] == b['clearance']
        np.testing.assert_array_equal(a['points'], b['points'])


def test_nondefault_self_config_is_not_substituted():
    objective, _, _ = _objective(temporal=True)
    objective.self_collision = SelfCollisionQuery(objective.model, {'detection_distance': .07})
    new = repair_geometry_setup.self_query(objective)
    assert new.cfg['detection_distance'] != .07


@pytest.mark.parametrize('parts', [1, 2])
def test_resident_moving_components_preserve_all_query_bytes(parts):
    mesh = trimesh.creation.box(extents=[.3, .4, .5])
    if parts == 2:
        other = mesh.copy()
        other.apply_translation([.7, 0, 0])
        mesh = trimesh.util.concatenate([mesh, other])
    scene = SceneQuery(mesh, full_geometry=False, bounded_object=True)
    obj = {'query': scene, 'samples': np.array([[0., 0., 0.]])}
    moving = SimpleNamespace(objects=[obj])
    repair_geometry_setup.prepare_moving(SimpleNamespace(moving=moving))
    assert moving.objects[0] is not obj  # Immutable original provider unaffected.
    assert moving.objects[0]['samples'] is obj['samples']
    for shift in (0., .0001, -.01):
        points = np.random.default_rng(5).uniform(-.5, 1., (2100, 3)) + shift
        for cid in range(parts):
            first = scene.query(points, component_ids=[cid])
            second = moving.objects[0]['query'].query(points, component_ids=[cid])
            assert first.keys() == second.keys()
            for key in first:
                assert first[key].tobytes() == second[key].tobytes(), key


def test_moving_native_unavailable_preserves_original_dispatch(monkeypatch):
    scene = SceneQuery(trimesh.creation.box(), full_geometry=False, bounded_object=True)

    def unavailable(*a):
        raise FileNotFoundError('optional native headers')
    monkeypatch.setattr(repair_static_query, 'serial_scene', unavailable)
    moving = SimpleNamespace(objects=[{'query': scene}])
    repair_geometry_setup.prepare_moving(SimpleNamespace(moving=moving))
    assert moving.objects[0]['query'] is scene


def test_cold_cpu_bounds_keep_complete_original_witnesses(monkeypatch):
    from tests.test_original_geometry_window import _geometry
    from tests.test_original_geometry_batches import _assert_same
    monkeypatch.setenv('OMNIRETARGETING_GPU_GEOMETRY', '0')
    original, poses = _geometry()
    screened, _ = _geometry()
    original._gpu_allowed = screened._gpu_allowed = False
    before = original.measure(poses, jacobians=True)
    repair_geometry_setup.prepare_static_bounds(screened, poses)
    assert screened._repair_static_bounds
    actual = screened.measure(poses, jacobians=True)
    _assert_same(actual, before)
    assert actual['failures'] == before['failures']
    assert actual['depths'] == before['depths']
    assert screened._static_coarse_bounds is not None


def test_open_and_gpu_scenes_do_not_build_cpu_grid():
    for eligible, gpu in [(False, False), (True, True)]:
        geometry = SimpleNamespace(_static_bound_reuse=eligible, _gpu_allowed=gpu)
        repair_geometry_setup.prepare_static_bounds(geometry, None)
        assert not hasattr(geometry, '_static_coarse_bounds')
