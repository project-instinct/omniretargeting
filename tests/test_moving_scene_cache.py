"""Batching retains moving-mesh witnesses, Jacobians and invalidation semantics."""

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting._sequence.moving_cache import MovingForwardQueries
from omniretargeting.moving_scene import MovingSceneQuery


@pytest.mark.parametrize("batch_points", [1, 100000])
def test_moving_batches_equal_direct_queries(batch_points):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="box" size=".1 .2 .1"/>'
        '<body pos="0 .2 0"><joint name="bend" axis="0 0 1"/>'
        '<geom type="capsule" size=".08 .2"/></body></body></worldbody></mujoco>'
    )
    frames = 7
    poses = np.tile(model.qpos0, (frames, 1))
    poses[:, 0] = np.linspace(-0.3, 0.3, frames)
    poses[:, -1] = np.linspace(-0.2, 0.4, frames)
    obj = {
        "mesh": trimesh.creation.box(extents=[0.3, 0.3, 0.3]),
        "rotations": Rotation.from_rotvec(
            np.c_[np.zeros((frames, 2)), np.linspace(0, 1, frames)]
        ).as_matrix(),
        "translations": np.c_[np.linspace(0.3, -0.3, frames), np.zeros((frames, 2))],
        "scales": np.linspace(0.7, 1.2, frames),
    }
    direct = MovingSceneQuery(model, [obj], frames)
    unculled = MovingSceneQuery(model, [obj], frames)
    unculled.objects[0]["radius"] = 1e9
    batched = MovingSceneQuery(model, [obj], frames)
    batched.forward_cache = MovingForwardQueries(batched, batch_points=batch_points)
    batched.prepare_trajectory(poses)
    calls = batched.objects[0]["query"].query_calls
    batched.prepare_trajectory(poses)
    assert batched.objects[0]["query"].query_calls == calls

    def compare(q):
        for frame, pose in enumerate(q):
            expected, actual = direct.query(pose, frame), batched.query(pose, frame)
            without_bounds = unculled.query(pose, frame)
            for key in ("forward_depth", "reverse_depth", "forward_components"):
                np.testing.assert_allclose(expected[key], without_bounds[key], atol=1e-12)
            assert len(expected["rows"]) == len(without_bounds["rows"])
            for a, b in zip(expected["rows"], without_bounds["rows"]):
                for key in ("point", "direction", "clearance", "jacobian"):
                    np.testing.assert_allclose(a[key], b[key], atol=1e-12)
            for key in ("forward_depth", "reverse_depth", "forward_components"):
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)
            assert actual["failures"] == expected["failures"]
            assert len(actual["rows"]) == len(expected["rows"])
            for a, b in zip(actual["rows"], expected["rows"]):
                for key in ("point", "direction", "clearance", "jacobian"):
                    np.testing.assert_allclose(a[key], b[key], atol=1e-12)
                assert a["kind"] == b["kind"]

    compare(poses)
    changed = poses.copy()
    changed[2, 1] += 0.15
    assert batched.forward_cache.get(changed[2], 2, 0) is None
    batched.prepare_trajectory(changed)
    compare(changed)
    assert batched.forward_cache.get(poses[2], 2, 0) is None


def test_reverse_cache_reuses_only_exact_geometry_and_recomputes_jacobians(monkeypatch):
    import omniretargeting.moving_reverse_cache as reverse
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="base">
      <freejoint/><geom type="box" size=".1 .1 .1"/>
      <body pos=".25 0 0"><joint name="bend" axis="0 0 1"/>
        <geom type="box" pos=".08 0 0" size=".1 .1 .1"/></body>
    </body></worldbody></mujoco>''')
    obj = {'mesh': trimesh.creation.box(extents=[.8, .8, .15])}
    query = MovingSceneQuery(model, [obj], 2)
    calls = []
    original = reverse.reverse_witnesses
    def count(*args):
        calls.append(args[1]['geom_id'])
        return original(*args)
    monkeypatch.setattr(reverse, 'reverse_witnesses', count)
    pose = model.qpos0.copy()
    first = query.query(pose, 0, [])
    count_first = len(calls)
    full = query.query(pose, 0)
    assert len(calls) == count_first
    assert all(row['jacobian'].shape == (model.nv,) for row in full['rows'])
    np.testing.assert_array_equal(first['reverse_depth'], full['reverse_depth'])
    pose[-1] = .1
    changed = query.query(pose, 0)
    assert len(calls) == count_first + 1  # The unchanged base keeps its witnesses.
    fresh = MovingSceneQuery(model, [obj], 2).query(pose, 0)
    for key in ['reverse_depth', 'forward_depth']:
        np.testing.assert_array_equal(changed[key], fresh[key])
    for a, b in zip(changed['rows'], fresh['rows']):
        for key in ['point', 'direction', 'clearance', 'jacobian']:
            np.testing.assert_array_equal(a[key], b[key])
    assert len(changed['rows']) == len(fresh['rows'])
    count_changed = len(calls)
    surface = query.objects[0]['surface']
    surface.translations = surface.translations.copy()
    surface.translations[0, 0] += .01
    query.query(pose, 0)
    assert len(calls) == count_changed + 2
