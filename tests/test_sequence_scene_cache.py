"""Static query batching must preserve signed witnesses and invalidate changed poses."""

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting._sequence.scene_cache import SceneTrajectoryQueries
from omniretargeting.scene_geometry import SceneQuery, geom_surface_offsets


@pytest.mark.parametrize("batch_points", [1, 10000])
def test_batched_queries_match_direct_and_track_pose_changes(batch_points):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="box" size=".1 .2 .3"/>'
        "</body></worldbody></mujoco>"
    )
    room = trimesh.creation.box(extents=[2, 2, 2])
    query = SceneQuery(room, enclosed_free_space=True, full_geometry=False)
    points = geom_surface_offsets(model, 0, 0.1)
    cached = SceneTrajectoryQueries(
        model, query, [(0, points)], batch_points=batch_points
    )
    poses = np.tile(model.qpos0, (7, 1))
    poses[:, 0] = np.linspace(-1.5, 1.5, len(poses))
    cached.prepare(poses)
    calls = query.query_calls
    cached.prepare(poses)
    assert query.query_calls == calls
    direct = SceneQuery(room, enclosed_free_space=True, full_geometry=False)
    for frame, pose in enumerate(poses):
        measured = direct.query(points + pose[:3])
        for name, expected in measured.items():
            np.testing.assert_allclose(
                cached.get(pose, frame)[name], expected, atol=1e-12
            )
    changed = poses.copy()
    changed[3, 1] += 2
    assert cached.get(changed[3], 3) is None
    cached.prepare(changed)
    assert query.query_calls == calls + 1
    for name, expected in direct.query(points + changed[3, :3]).items():
        np.testing.assert_allclose(
            cached.get(changed[3], 3)[name], expected, atol=1e-12
        )
    assert cached.get(poses[3], 3) is None
