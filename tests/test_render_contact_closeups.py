import numpy as np
import pytest

from scripts.render_contact_closeups import (
    camera_azimuth, frame_statistics, hand_binding_groups, select_frames,
)


def _plan():
    parts = np.array(["left_hand", "right_hand", "left_foot", "right_forearm"])
    return {
        "surface_contact_parts": parts,
        "surface_contact_effector_names": parts,
        "surface_contact_surface_ids": np.array([
            [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
        ]),
        "surface_contact_mask": np.array([
            [True, False, True, False], [True, True, True, True],
            [False, True, True, True],
        ]),
        "surface_contact_environment_w": np.zeros((3, 4, 3)),
        "surface_contact_normals_w": np.tile([0., 0., 1.], (3, 4, 1)),
        "surface_contact_robot_geoms": np.array([7, 8, 9, 10]),
    }


def test_hand_groups_keep_only_object_hand_bindings():
    groups = hand_binding_groups(_plan())
    assert [(g["effector"], g["part"], g["columns"].tolist()) for g in groups] == [
        ("left_hand", "left_hand", [0]),
        ("right_hand", "right_hand", [1]),
        ("right_forearm", "right_forearm", [3]),
    ]


def test_select_frames_combines_start_middle_and_extremes():
    plan = _plan()
    points = np.zeros((3, 4, 3))
    points[0, 0, 2] = -.02
    points[1, 1, 2] = .04
    points[2, 3, 2] = -.03
    selected = select_frames([plan, plan], [points, points], max_frames=8)
    frames = [row["frame"] for row in selected]
    reasons = {row["frame"]: row["reason"] for row in selected}
    assert frames == sorted(set(frames))
    assert reasons[0] == "contact_start"
    assert 1 in frames and 2 in frames
    assert selected[0]["run"] == 0


def test_unmanaged_left_forearm_keeps_its_side_in_rendering():
    plan = _plan()
    plan['surface_contact_parts'][3] = 'left_forearm'
    plan['surface_contact_effector_names'] = np.array(['left_hand', 'right_hand', 'left_foot', ''])
    group = hand_binding_groups(plan)[-1]
    assert group['effector'] == group['part'] == 'left_forearm'
    assert group['columns'].tolist() == [3]


def test_saved_binding_ids_use_robot_before_scene_geoms_are_inserted(tmp_path):
    import mujoco
    import trimesh
    from scripts.render_contact_closeups import _build_model, binding_material_points, material_points
    path = tmp_path / 'robot.xml'
    path.write_text('''<mujoco><worldbody><body name="left_hand" pos=".3 0 0">
      <joint type="slide" axis="1 0 0"/><geom name="hand" type="sphere" size=".02" pos=".2 0 0"/>
      </body></worldbody></mujoco>''')
    robot = mujoco.MjModel.from_xml_path(str(path))
    scene = _build_model(str(path), trimesh.creation.box(), [])
    assert robot.geom('hand').id != scene.geom('hand').id
    plan = dict(surface_contact_robot_geoms=np.array([robot.geom('hand').id]),
                surface_contact_robot_local_points=np.array([[.02, 0., 0.]]))
    motion = dict(qpos=np.array([[.1]]))
    points = binding_material_points(dict(model_path=str(path)), motion, plan)
    np.testing.assert_allclose(points, [[[.62, 0., 0.]]])
    wrong = material_points(scene, motion['qpos'], plan['surface_contact_robot_geoms'],
                            plan['surface_contact_robot_local_points'])
    assert np.linalg.norm(wrong-points) > .1


def test_frame_statistics_and_camera_azimuth():
    plan = _plan()
    points = np.zeros((3, 4, 3))
    points[1, 3, 2] = -.025
    depths = np.zeros((3, 12))
    depths[1, 10] = .004
    stats = frame_statistics(plan, points, np.array([3]), 1, depths)
    assert stats["normal_median_mm"] == pytest.approx(25.0)
    assert stats["normal_worst_mm"] == pytest.approx(25.0)
    assert stats["backside_samples"] == 1
    assert stats["moving_depth_mm"] == pytest.approx(4.0)
    assert camera_azimuth([0.0, 0.0, 1.0]) == -50.0
    assert camera_azimuth([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert camera_azimuth([0.0, 1.0, 0.0]) == pytest.approx(90.0)


def test_object_scale_is_applied_to_copy_without_changing_source():
    import trimesh
    from scripts.render_contact_closeups import scaled_object_mesh
    mesh = trimesh.creation.box(extents=[3., 6., 9.])
    vertices = mesh.vertices.copy()
    scaled = scaled_object_mesh(dict(mesh=mesh, scales=np.full(4, 1./3)))
    np.testing.assert_allclose(scaled.extents, [1., 2., 3.])
    np.testing.assert_array_equal(mesh.vertices, vertices)
