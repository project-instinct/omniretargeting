from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.contact_anchors import build_rigid_anchors, rigid_placement


def test_rigid_fit_preserves_size_and_bounds_orientation():
    local = np.array([[-0.1, 0, 0], [0.1, 0, 0], [0, 0.1, 0]])
    turn = Rotation.from_euler("z", 0.8).as_matrix()
    target = local @ turn.T * 1.5 + [1, 2, 3]
    fitted, offset, report = rigid_placement(
        local, target, np.ones(3), np.eye(3), rotation_budget_rad=0.2
    )
    assert report["rotation_bound_active"]
    assert np.isclose(Rotation.from_matrix(fitted).magnitude(), 0.2)
    points = local @ fitted.T + offset
    np.testing.assert_allclose(np.linalg.norm(points[1] - points[0]), 0.2)
    assert (
        report["maximum_m"] > 0
    )  # morphology mismatch is reported, not stretched away


def fixture():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="root"><freejoint/>
      <geom name="pad" type="box" size="0.1 0.05 0.01"/>
      <body name="welded" pos="0.2 0 0"><geom name="tip" type="sphere" size="0.01"/></body>
      </body></worldbody></mujoco>""")
    bindings = [
        SimpleNamespace(robot_geom="pad", local_point=np.array([-0.1, 0, -0.01])),
        SimpleNamespace(robot_geom="tip", local_point=np.array([0, 0, -0.01])),
    ]
    motion = SimpleNamespace(
        points=np.zeros((5, 2, 3)),
        reference_qpos=model.qpos0,
        frames=np.tile(np.eye(3), (5, 2, 1, 1)),
        reference_frames=np.tile(np.eye(3), (2, 1, 1)),
    )
    # Heel first, both middle, toe last; the same body stays in contact.
    active = np.array([[1, 0], [1, 1], [1, 1], [1, 1], [0, 1]], bool)
    env = np.tile([[-0.15, 0, 0], [0.25, 0, 0]], (5, 1, 1))
    observation = {
        "active": active,
        "surface_ids": np.zeros((5, 2), int),
        "modes": np.where(active, "sticking", "none"),
        "confidence": active.astype(float),
        "environment": env,
    }
    return model, bindings, motion, observation


def test_welded_geometry_and_changing_samples_share_one_placement():
    model, bindings, motion, obs = fixture()
    targets, report = build_rigid_anchors(
        model,
        bindings,
        [0, 1],
        obs,
        motion,
        scene_present=True,
        rotation_budget_rad=0.2,
    )
    assert len(report["placements"]) == 1
    np.testing.assert_allclose(targets[1:4, 1] - targets[1:4, 0], [[0.4, 0, 0]] * 3)
    np.testing.assert_allclose(targets[:4, 0], np.broadcast_to(targets[1, 0], (4, 3)))
    np.testing.assert_allclose(targets[1:, 1], np.broadcast_to(targets[1, 1], (4, 3)))


def test_moving_anchor_is_fixed_in_material_coordinates():
    model, bindings, motion, obs = fixture()
    rotation = Rotation.from_euler("z", np.linspace(0, 1, 5)[:, None]).as_matrix()
    translation = np.column_stack([np.linspace(0, 0.5, 5), np.zeros((5, 2))])
    obs["environment"] = (
        np.einsum("tni,tji->tnj", obs["environment"], rotation) + translation[:, None]
    )
    motion.frames = rotation[:, None] @ motion.frames
    targets, report = build_rigid_anchors(
        model,
        bindings,
        [0, 1],
        obs,
        motion,
        scene_present=False,
        scene_objects=[{"rotations": rotation, "translations": translation}],
        rotation_budget_rad=0.2,
    )
    local = np.einsum("tni,tij->tnj", targets - translation[:, None], rotation)
    np.testing.assert_allclose(
        local[:4, 0], np.broadcast_to(local[1, 0], (4, 3)), atol=1e-12
    )
    np.testing.assert_allclose(
        local[1:, 1], np.broadcast_to(local[1, 1], (4, 3)), atol=1e-12
    )
    assert report["placements"][0]["locked"]


def test_recorded_scale_changes_keep_robot_patch_rigid_and_body_origin_material_fixed():
    model, bindings, motion, obs = fixture()
    rotation = Rotation.from_euler("z", np.linspace(0, 1, 5)[:, None]).as_matrix()
    translation = np.column_stack([np.linspace(0, 0.5, 5), np.zeros((5, 2))])
    scale = np.array([0.2, 0.3, 0.5, 0.7, 0.9])
    obs["active"][:] = True
    obs["modes"][:] = "sticking"
    obs["confidence"][:] = 1.0
    material = obs["environment"].copy() + [0.5, 0.2, 0.0]
    obs["environment"] = (
        np.einsum("tni,tji->tnj", material * scale[:, None, None], rotation)
        + translation[:, None]
    )
    unchanged = obs["environment"].copy()
    motion.frames = rotation[:, None] @ motion.frames
    targets, report = build_rigid_anchors(
        model,
        bindings,
        [0, 1],
        obs,
        motion,
        scene_present=False,
        scene_objects=[
            {"rotations": rotation, "translations": translation, "scales": scale}
        ],
        rotation_budget_rad=0.2,
    )
    assert len(report["placements"]) == 1
    placement = report["placements"][0]
    assert placement["locked"]
    assert not placement["fit"]["constant_surface_scale"]
    assert placement["fit"]["robot_metric_size_preserved"]
    # Contact goals follow the authored surface scale. The actual robot patch
    # remains rigid; its 30 cm metric geometry is never used to warp targets.
    np.testing.assert_allclose(targets, unchanged, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(targets[:, 1] - targets[:, 0], axis=1), 0.4 * scale, atol=1e-12
    )
    assert report["targets_on_observed_surface"]
    assert report["rigid_fit_target_residual_max_m"] > 0
    np.testing.assert_array_equal(obs["environment"], unchanged)
    np.testing.assert_array_equal(scale, [0.2, 0.3, 0.5, 0.7, 0.9])
    assert placement["fit"]["maximum_m"] > 0.0
