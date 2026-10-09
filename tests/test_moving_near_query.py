"""Repair nearest-component batching preserves open-surface evidence bytes."""

from types import SimpleNamespace

import numpy as np
import pytest
import mujoco
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting._fast.hull_threshold import HullThresholds
from omniretargeting._fast.repair_geometry_setup import prepare_moving
from omniretargeting.moving_scene import MovingSceneQuery
from tests.test_open_object_hull import V_SHEET, EVIDENCE, _sphere_model, _query


def assert_same(first, second):
    if isinstance(first, np.ndarray):
        assert first.dtype == second.dtype and first.shape == second.shape
        assert first.tobytes() == second.tobytes()
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_same(a, b)
    else:
        assert first == second


def test_native_hull_preserves_all_moving_evidence():
    model = _sphere_model()
    objects = [dict(mesh=V_SHEET, collision_evidence=EVIDENCE)]
    original = MovingSceneQuery(model, objects, 1)
    fast = original.fresh_state()
    original_surface = original.objects[0]['surface']
    prepare_moving(SimpleNamespace(moving=fast))
    surface = fast.objects[0]['surface']
    assert surface is original_surface
    assert isinstance(fast.objects[0]['query']._repair_hull_thresholds, HullThresholds)
    for position in ([0., 0., .5], [.3, 0., .25], [0., 0., -.5], [.6, 0., .6]):
        pose = model.qpos0.copy()
        pose[:3] = position
        assert_same(original.query(pose, 0), fast.query(pose, 0))
    prepare_moving(SimpleNamespace(moving=fast))
    assert fast.objects[0]['surface'] is surface


def test_native_hull_thresholds_and_roundoff_fallback(monkeypatch):
    scene = _query(bounded_object=True)
    native = HullThresholds(scene)
    points = np.random.default_rng(12).uniform(-1., 1., (4000, 3))
    points[:len(V_SHEET.vertices)] = V_SHEET.vertices
    for bands in ((1e-9, .008), (-.008, .1, 0.), (np.inf, -np.inf), ()):
        assert_same(native(points, *bands), scene.hull_classification(points, *bands))
    original, calls = scene.hull_margin, []

    def counted(points):
        calls.append(len(points))
        return original(points)

    monkeypatch.setattr(scene, 'hull_margin', counted)
    point = points[100:101]
    value = original(point)[0]
    bands = (np.nextafter(value, -np.inf), value, np.nextafter(value, np.inf))
    assert native(point, *bands).tolist() == [[True, False, False]]
    assert calls == [1]


@pytest.mark.parametrize('magnitude', [1e-12, 1., 1e12])
def test_native_hull_many_planes_cancellation_and_empty_points(magnitude):
    rng = np.random.default_rng(29)
    planes = rng.normal(size=(257, 4))
    points = rng.normal(size=(1023, 3)) * magnitude
    planes[:, 3] = -(planes[:, :3] @ points[0])
    scene = SimpleNamespace(_hull=planes,
        hull_margin=lambda p: (p @ planes[:, :3].T + planes[:, 3]).max(axis=1))
    native = HullThresholds(scene)
    margins = scene.hull_margin(points)
    bands = (0., 1e-9, .008, margins[0], np.nextafter(margins[0], np.inf))
    assert_same(native(points, *bands), margins[:, None] > bands)
    assert native(np.empty((0, 3)), *bands).shape == (0, len(bands))


def test_optional_hull_backend_unavailable_keeps_original_query(monkeypatch):
    from omniretargeting._fast import hull_threshold

    def unavailable():
        raise FileNotFoundError('compiler unavailable')

    monkeypatch.setattr(hull_threshold, 'library', unavailable)
    moving = MovingSceneQuery(_sphere_model(), [dict(mesh=V_SHEET, collision_evidence=EVIDENCE)], 1)
    prepare_moving(SimpleNamespace(moving=moving))
    assert not hasattr(moving.objects[0]['query'], '_repair_hull_thresholds')


def test_custom_nearest_tree_keeps_its_dispatch():
    model = _sphere_model()
    moving = MovingSceneQuery(model, [dict(mesh=V_SHEET, collision_evidence=EVIDENCE)], 1)
    surface = moving.objects[0]['surface']
    original = surface.tree
    custom = SimpleNamespace(squared_distance=original.squared_distance)
    surface._tree = custom
    prepare_moving(SimpleNamespace(moving=moving))
    assert moving.objects[0]['surface'] is surface
    assert surface.tree is custom


@pytest.mark.parametrize('enclosed,distance', [(False, 1.), (True, 1e-9)])
def test_unused_winding_grid_is_lazy_but_deep_query_matches_prewarmed(enclosed, distance):
    eager, lazy = _query(bounded_object=True), _query(bounded_object=True)
    eager._winding_free(0, np.empty((0, 3)), np.empty(0), np.empty(0, bool))
    points = np.array([[0., 0., .5]])
    args = (0, points, np.array([distance]), np.array([enclosed]))
    assert_same(eager._winding_free(*args), lazy._winding_free(*args))
    assert not lazy._winding_grids
    args = (0, points, np.array([.5]), np.array([True]))
    assert_same(eager._winding_free(*args), lazy._winding_free(*args))
    assert_same(eager._winding_grids, lazy._winding_grids)


@pytest.mark.parametrize('scale', [1e-12, .6, 1., 1.7, 1e12])
def test_component_boxes_preserve_exact_material_samples(scale):
    model = _sphere_model()
    boxes = []
    for x in (0., .2, 1., -1.):
        box = trimesh.creation.box(extents=[.1, .2, .3])
        box.apply_translation([x, 0., 0.])
        boxes.append(box)
    moving = MovingSceneQuery(model, [dict(mesh=trimesh.util.concatenate(boxes))], 1)
    obj = moving.objects[0]
    rotation = Rotation.from_rotvec([.15, -.32, .27]).as_matrix()
    translation = np.array([.02, -.01, .015])
    band = moving.cfg['activation_distance']
    for shift in (0., .058, .058 - 1e-12, .058 + 1e-12, .21):
        moving.data.qpos[:] = model.qpos0
        moving.data.qpos[0] = shift
        mujoco.mj_forward(model, moving.data)
        actual = moving._forward_samples(obj, translation, rotation, scale, band, moving.shells)
        for world, local, components in actual.values():
            expected = {}
            for cid in obj['closed']:
                bounds = obj['query'].components[cid][0].bounds
                gap = np.maximum(np.maximum(bounds[0] - local, local - bounds[1]), 0.)
                ids = np.flatnonzero(np.linalg.norm(gap, axis=1) <= band / scale)
                if len(ids):
                    expected[cid] = (ids, obj['query'].query(local[ids], component_ids=[cid]))
            assert_same(components, expected)
