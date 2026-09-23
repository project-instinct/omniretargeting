import numpy as np
import trimesh

from omniretargeting.contact import infer_scene_contact_tasks


def test_authored_sole_contact_uses_sole_side_when_source_is_below_floor():
    scene = trimesh.Trimesh(
        vertices=np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]]),
        faces=np.array([[0, 2, 1], [0, 3, 2]]),
        process=False,
    )
    sole = np.tile([0., 0., -.03], (4, 1))
    tasks = infer_scene_contact_tasks(
        {"left_foot": {
            "tracking_positions": sole,
            "contact_samples": sole[:, None, :],
            "up_directions": np.tile([0., 0., 1.], (4, 1)),
            "source_contact_mask": np.ones(4, dtype=bool),
        }},
        scene,
        30.,
        {"surface_clearance": .01, "anchor_contacts": False},
    )
    foot = tasks["left_foot"]
    np.testing.assert_allclose(foot.surface_normals, [[0., 0., 1.]] * 4)
    np.testing.assert_allclose(foot.contact_positions[:, 2], .01)
