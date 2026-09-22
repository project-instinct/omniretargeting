"""Batching keeps each original signed material witness and its geometry."""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.moving_scene import MovingSceneQuery
from tests.test_moving_scene import _robot_xml, _box_object, _pose


@pytest.mark.parametrize('scale,angle', [(1., 0.), (1.2, .37)])
def test_signed_queries_are_batched_without_changing_per_geom_witnesses(monkeypatch, scale, angle):
    model = mujoco.MjModel.from_xml_string(_robot_xml(
        root_pos=(.08, 0., 0.), arm_pos=(-.08, 0., 0.)))
    rotation = Rotation.from_euler('z', angle).as_matrix()
    query = MovingSceneQuery(model, [_box_object(
        extents=(.5, .4, .4), rotations=rotation[None], scales=np.array([scale]))], 1)
    obj = query.objects[0]
    original = obj['query'].query
    calls = []

    def counted(points, **kwargs):
        calls.append(len(points))
        return original(points, **kwargs)

    monkeypatch.setattr(obj['query'], 'query', counted)
    result = query.query(_pose(model), 0)
    assert len(calls) == 1  # Two nearby robot geoms share one mesh evaluation.
    assert not result['failures']
    for shell in query.shells:
        gid = shell['geom_id']
        world = (shell['offsets'] @ query.data.geom_xmat[gid].reshape(3, 3).T
                 + query.data.geom_xpos[gid])
        local = world @ rotation / scale
        bounds = obj['query'].components[0][0].bounds
        gap = np.maximum(np.maximum(bounds[0] - local, local - bounds[1]), 0)
        ids = np.flatnonzero(np.linalg.norm(gap, axis=1) <= query.cfg['activation_distance'] / scale)
        expected = original(local[ids], component_ids=[0])
        k = int(np.argmin(expected['clearance']))
        depth = max(0., -expected['clearance'][k] * scale)
        np.testing.assert_allclose(result['forward_depth'][0, gid], depth, atol=1e-12)
        row = next(r for r in result['rows'] if r['kind'] == 'closed_surface' and r['geom_id'] == gid)
        np.testing.assert_allclose(row['point'], world[ids[k]], atol=1e-12)
        np.testing.assert_allclose(row['direction'], rotation @ expected['directions'][k], atol=1e-12)
