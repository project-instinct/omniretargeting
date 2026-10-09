"""A moving hull screen must preserve original evidence and reverse witnesses."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.moving_scene import MovingSceneQuery, interpolate_object
from omniretargeting._fast.repair_geometry_setup import prepare_moving


def make_query(frames=3):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
      <geom type="box" size=".12 .03 .03"/></body></worldbody></mujoco>''')
    mesh = trimesh.creation.box(extents=(.5, .5, .5))
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    objects = [dict(mesh=mesh, collision_evidence=dict(points=np.array([[.25, 0, 0]]),
        normals=np.array([[1., 0, 0]])), translations=np.zeros((frames, 3)),
        rotations=np.repeat(np.eye(3)[None], frames, axis=0))]
    return MovingSceneQuery(model, objects, frames)


def pose(query, xyz=(.41, 0, 0), angle=0):
    q = query.model.qpos0.copy()
    q[:3] = xyz
    q[3:7] = Rotation.from_rotvec([0, angle, 0]).as_quat(scalar_first=True)
    return q


def assert_same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b, equal_nan=True)
    else:
        assert a == b


def test_all_fields_match_through_approach_rotation_and_containment():
    cached, original = make_query(), make_query()
    cached._clearance_enabled = True
    for x, angle in [(.41, 0), (.409, .01), (.408, .02), (.41, 2.5),
                     (.33, .7), (.25, 0), (0., 0), (.41, 0)]:
        q = pose(cached, (x, 0, 0), angle)
        assert_same(cached.query(q, 0), original.query(q, 0))
    counts = cached.clearance_cache.counts
    assert counts['motion_bound_hits'] > 0
    assert counts['fresh_bound_hits'] > 0
    assert counts['exact_required'] > 0


def separated(query, q, time=0, band=None):
    query.data.qpos[:] = q
    mujoco.mj_forward(query.model, query.data)
    obj = query.objects[0]
    t, r, s = interpolate_object(obj['surface'], time)
    return query.clearance_cache.separated(query, 0, obj, time, t, r, s,
        query.cfg['activation_distance'] if band is None else band, query.shells)


def test_bound_consumed_by_motion_never_skips_contained_shell():
    query = make_query()
    assert separated(query, pose(query))
    assert separated(query, pose(query, (.409, 0, 0)))
    assert not separated(query, pose(query, (.3, 0, 0)))
    assert not separated(query, pose(query, (0, 0, 0)))
    # Negative interior margin must never be made unsigned.
    assert not separated(query, pose(query, (-.1, 0, 0)))


def test_band_boundary_and_forward_radius_are_conservative():
    query = make_query()
    shell = query.shells[0]
    radius = max(shell['radius'], np.linalg.norm(shell['offsets'], axis=1).max())
    edge = .25 + radius + query.cfg['activation_distance']
    assert not separated(query, pose(query, (edge, 0, 0)))
    assert not separated(query, pose(query, (edge + 1e-10, 0, 0)))
    assert separated(query, pose(query, (edge + 1e-6, 0, 0)))
    assert not separated(query, pose(query, (edge + 1e-6, 0, 0)), band=.02)
    shell['offsets'] = np.vstack([shell['offsets'], [.5, 0., 0.]])
    assert not separated(query, pose(query, (edge + 1e-6, 0, 0)))


@pytest.mark.parametrize('replacement', ['scale', 'hull', 'evidence', 'offsets', 'query'])
def test_replaced_geometry_evidence_and_scale_invalidate(replacement):
    query = make_query()
    assert separated(query, pose(query))
    old = query.clearance_cache.counts['motion_bound_hits']
    obj, shell = query.objects[0], query.shells[0]
    if replacement == 'scale':
        obj['surface'].scales = np.full(3, 1.001)
    elif replacement == 'hull':
        obj['query']._hull = obj['query']._hull.copy()
    elif replacement == 'evidence':
        obj['query'].evidence_points = obj['query'].evidence_points.copy()
    elif replacement == 'offsets':
        shell['offsets'] = shell['offsets'].copy()
    else:
        import copy
        obj['query'] = copy.copy(obj['query'])
    separated(query, pose(query))
    assert query.clearance_cache.counts['motion_bound_hits'] == old


def test_prescribed_object_motion_and_scale_require_recheck():
    query = make_query()
    assert separated(query, pose(query))
    query.objects[0]['surface'].translations = np.tile([.3, 0, 0], (3, 1))
    assert not separated(query, pose(query))
    query.objects[0]['surface'].translations = np.zeros((3, 3))
    query.objects[0]['surface'].scales = np.full(3, 2.)
    assert not separated(query, pose(query))


def test_cache_is_bounded_and_fresh_state_has_no_query_state():
    query = make_query()
    obj = query.objects[0]
    query.data.qpos[:] = pose(query)
    mujoco.mj_forward(query.model, query.data)
    for frame in range(8200):
        query.clearance_cache.separated(query, 0, obj, frame, np.zeros(3),
                                       np.eye(3), 1., .008, query.shells)
    assert len(query.clearance_cache.entries) == 8192
    fresh = query.fresh_state()
    assert not fresh.clearance_cache.entries
    assert not fresh._separated_summaries
    assert fresh.clearance_cache is not query.clearance_cache


def test_only_original_production_query_enables_screen():
    query = make_query()
    assert not query._clearance_enabled
    prepare_moving(SimpleNamespace(moving=query))
    assert query._clearance_enabled
    custom = make_query()
    custom.objects[0]['query'].query = lambda *a, **kw: {}
    prepare_moving(SimpleNamespace(moving=custom))
    assert not custom._clearance_enabled


def test_random_rigid_queries_keep_every_field():
    cached, original = make_query(), make_query()
    cached._clearance_enabled = True
    rng = np.random.default_rng(148)
    for _ in range(30):
        q = pose(cached, rng.uniform(-.45, .45, 3), rng.uniform(-3., 3.))
        assert_same(cached.query(q, 0), original.query(q, 0))


def test_object_relative_frames_rotations_and_scales_keep_every_field():
    cached, original = make_query(), make_query()
    cached._clearance_enabled = True
    for query in (cached, original):
        surface = query.objects[0]['surface']
        surface.translations = np.array([[0, 0, 0], [.25, -.1, .05], [-.2, .1, 0]])
        surface.rotations = Rotation.from_rotvec([[0., 0., a] for a in [0., .8, -1.1]]).as_matrix()
        surface.scales = np.array([1., .8, 1.2])
    for time in [0, .5, 1, 1.5, 2, 0]:
        q = pose(cached)
        assert_same(cached.query(q, time), original.query(q, time))


def test_unsigned_unobserved_or_unbounded_objects_do_not_certify():
    query = make_query()
    obj = query.objects[0]
    obj['observed_open'], obj['nearly_closed'] = [], []
    assert not separated(query, pose(query))
    obj['observed_open'] = [0]
    obj['query']._hull = None
    assert not separated(query, pose(query))
