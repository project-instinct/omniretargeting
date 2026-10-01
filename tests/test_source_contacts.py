import numpy as np
import trimesh

from omniretargeting.source_contacts import build_source_contacts
from omniretargeting.surface_contact import surface_contact_config


def skin(points, label="arbitrary_region"):
    # Clockwise face: skin normal faces the floor.
    offsets = np.array([[-0.001, -0.001, 0], [0, 0.002, 0], [0.001, -0.001, 0]])
    return {
        "triangles": np.asarray(points)[:, None, None, :] + offsets,
        "barycentric": np.ones((1, 3)) / 3,
        "labels": [label],
    }


def floor():
    mesh = trimesh.creation.box(extents=[4, 4, 0.2])
    mesh.apply_translation([0, 0, -0.1])
    return mesh


def test_surface_crossing_keeps_declared_free_side_and_raw_skin():
    points = np.array([[0, 0, z] for z in [0.01, 0.005, -0.005, 0.005, 0.01]])
    source = skin(points)
    before = source["triangles"].copy()
    table = build_source_contacts(
        source,
        floor(),
        surface_contact_config(),
        framerate=30,
        scene_enclosed_free_space=False,
    )
    np.testing.assert_allclose(table.observation["normals"][..., 2], 1)
    assert table.active.all()
    assert len(table.events) == 1
    assert table.events[0]["enter"] == [("arbitrary_region", 0)]
    np.testing.assert_array_equal(source["triangles"], before)
    assert not table.points.flags.writeable


def test_brief_mechanical_hypotheses_survive_geometry_duration_filter():
    points = np.array([[0, 0, z] for z in [0.2, 0.2, 0.005, 0.2, 0.2]])
    table = build_source_contacts(
        skin(points, "new_part"),
        floor(),
        surface_contact_config(),
        framerate=30,
        scene_enclosed_free_space=False,
        min_duration_s=0.1,
    )
    assert not table.active.any()
    frames, regions = table.candidates(
        friction=0.5,
        reconstruction_bound_m=0.03,
        normal_speed_m_s=0.2,
        touch_distance_m=0.008,
        normal_opposition=0.2,
        sticking_speed_m_s=0.04,
    )
    assert len(frames[2]) == 1
    assert len(frames[1]) == 0
    assert set(regions.values()) == {("new_part", 0)}


def test_gap_filling_cannot_create_contact_through_real_departure():
    points = np.array([[0, 0, z] for z in [0.005, 0.005, 0.2, 0.005, 0.005]])
    table = build_source_contacts(
        skin(points),
        floor(),
        surface_contact_config(),
        framerate=1,
        max_gap_s=2,
        scene_enclosed_free_space=False,
    )
    assert not table.active[2].any()
    assert [e["frame"] for e in table.events] == [0, 2, 3]


def test_closed_room_normal_does_not_flip_outside_room():
    room = trimesh.creation.box(extents=[2, 2, 2])
    # The lower room wall's free side is +z, even outside it.
    points = np.array([[0, 0, z] for z in [-0.99, -1.005, -0.99]])
    table = build_source_contacts(
        skin(points),
        room,
        surface_contact_config(),
        framerate=5,
        scene_enclosed_free_space=True,
    )
    np.testing.assert_allclose(table.observation["normals"][..., 2], 1)
    assert table.active.all()
