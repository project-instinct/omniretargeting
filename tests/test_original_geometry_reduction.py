"""Conservative minima screening must keep original evidence and bytes."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting._fast.original_geometry_cpu import serial_scene
from omniretargeting._fast.original_geometry_reduction import ClosedComponentReduction
from omniretargeting._fast.repair_geometry_setup import prepare_moving, prepare_moving_reduction
from omniretargeting.moving_scene import MovingSceneQuery
from omniretargeting.scene_geometry import SceneQuery
from tests.test_gpu_geometry import assert_identical
from tests.test_moving_scene import _object, _pose, _robot_xml


def _scene(parts=1):
    mesh = trimesh.creation.box(extents=[.14, .16, .18])
    if parts > 1:
        other = mesh.copy()
        other.apply_translation([.23, 0, 0])
        mesh = trimesh.util.concatenate([mesh, other])
    return serial_scene(SceneQuery(mesh, full_geometry=False, bounded_object=True))


def _compare_minima(scene, points, groups, cid=0, band=.015):
    reducer = ClosedComponentReduction(scene, band)
    original = scene.query(points, component_ids=[cid])
    screened = reducer.query(points, groups, cid, band)
    assert original['uncertain'].tobytes() == screened['uncertain'].tobytes()
    for begin, end in groups:
        old = begin + np.argmin(original['clearance'][begin:end])
        new = begin + np.argmin(screened['clearance'][begin:end])
        if original['clearance'][old] < band:
            assert old == new
            for key in original:
                assert original[key][old].tobytes() == screened[key][new].tobytes(), key
        else:
            assert screened['clearance'][new] >= band
    return reducer


@pytest.mark.parametrize('parts', [1, 2])
def test_minima_include_deep_inside_ties_band_and_grid_exterior(parts):
    scene = _scene(parts)
    rng = np.random.default_rng(84)
    points = rng.uniform([-.1, -.1, -.1], [.1, .1, .1], (2000, 3))
    # Duplicate minima and surface zero-distance witnesses retain argmin order.
    points[:4] = [[0, 0, 0], [0, 0, 0], [.07, 0, 0], [.07 + .015, 0, 0]]
    points[800:1000] = [4., 0., 0.]  # Outside grid must use original query.
    reducer = _compare_minima(scene, points, [(0, 4), (4, 800), (800, 1000), (1000, 2000)])
    assert reducer.stats['exact_points'] < reducer.stats['points']
    if parts == 2:
        _compare_minima(scene, points, [(0, 2000)], cid=1)


def test_unsigned_cones_bound_original_distance():
    scene = _scene()
    reducer = ClosedComponentReduction(scene, .02)
    rng = np.random.default_rng(11)
    points = rng.uniform([-.095, -.105, -.115], [.095, .105, .115], (4000, 3))
    exact = scene.query(points)['unsigned_distance']
    lower, negative = reducer.bounds[0]
    assert np.all(lower.query(points) <= exact)
    assert np.all(-negative.query(points) >= exact)


def test_ambiguous_fast_winding_keeps_original_exact_resolution_and_uncertainty():
    scene = _scene()
    v, f, _, _ = scene._component_query_data[0]
    # Even a bad fast approximation must run the original exact/zero-distance
    # rules, rather than infer a reliable sign from the unsigned cones.
    scene._r4_winding_cache[id(v), id(f)] = lambda p: np.full(len(p), .4)
    points = np.array([[0., 0., 0.], [.04, 0., 0.], [.07, 0., 0.], [.08, 0., 0.]])
    reducer = _compare_minima(scene, points, [(0, 2), (2, 4)])
    assert reducer.stats['exact_points'] >= 3


@pytest.mark.parametrize('scale', [.7, 1., 1.3])
@pytest.mark.parametrize('parts', [1, 2])
def test_moving_consumer_all_fields_and_jacobians_are_bit_identical(scale, parts):
    model = mujoco.MjModel.from_xml_string(_robot_xml(arm_pos=(-.12, 0., 0.)))
    mesh = _scene(parts).mesh
    angle = .31
    rot = np.array([[np.cos(angle), -np.sin(angle), 0.],
                    [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    specs = [_object(mesh, [[0., 0., 0.], [.01, .02, 0.]],
                     rotations=np.array([np.eye(3), rot]), scales=[scale, scale])]
    original = MovingSceneQuery(model, specs, 2)
    screened = original.fresh_state()
    prepare_moving(SimpleNamespace(moving=screened))
    prepare_moving_reduction(SimpleNamespace(moving=screened), np.array([_pose(model)] * 2))
    assert 'repair_reduction' in screened.objects[0]
    for frame in [0, 1, 0]:
        for slide in [0., .02, -.1]:
            for dofs in [[], [6]]:
                pose = _pose(model, slide=slide)
                assert_identical(screened.query(pose, frame, dofs), original.query(pose, frame, dofs))


def test_ineligible_sign_method_keeps_full_query():
    scene = _scene()
    scene.enclosed_free_space = True
    reducer = ClosedComponentReduction(scene, .02)
    assert not reducer.bounds
    points = np.array([[0., 0., 0.], [.1, .1, .1]])
    assert_identical(reducer.query(points, [(0, 2)], 0, .02), scene.query(points, component_ids=[0]))


def test_replaced_query_retains_original_callback_dispatch():
    scene = _scene()
    reducer = ClosedComponentReduction(scene, .02)
    original = scene.query
    calls = []

    def custom(points, **kwargs):
        calls.append(len(points))
        result = original(points, **kwargs)
        result['uncertain'][:] = True
        return result

    scene.query = custom
    points = np.array([[0., 0., 0.], [.08, .08, .08], [1., 1., 1.]])
    expected = custom(points, component_ids=[0])
    actual = reducer.query(points, [(0, 3)], 0, .02)
    assert_identical(actual, expected)
    assert calls == [3, 3]


def test_unreachable_entry_avoids_grid_but_later_approach_is_exact():
    model = mujoco.MjModel.from_xml_string(_robot_xml())
    specs = [_object(_scene().mesh, [[10., 0., 0.], [10., 0., 0.]])]
    original = MovingSceneQuery(model, specs, 2)
    screened = original.fresh_state()
    geometry = SimpleNamespace(moving=screened)
    prepare_moving(geometry)
    prepare_moving_reduction(geometry, np.array([_pose(model)] * 2))
    assert 'repair_reduction' not in screened.objects[0]
    candidate = _pose(model, root=(10., 0., 0.), slide=.57)
    assert_identical(screened.query(candidate, 0, []), original.query(candidate, 0, []))
