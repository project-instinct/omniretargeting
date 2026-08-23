"""Geometric tests for source-conditioned scene contact regions."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from omniretargeting.affordance import (
    build_contact_region,
    build_seat_affordance,
)


def _step_box() -> trimesh.Trimesh:
    step = trimesh.creation.box(extents=[1.0, 1.0, 0.4])
    step.apply_translation([0.0, 0.0, 0.2])
    return step


@pytest.mark.parametrize(
    ("expected_type", "points", "normals"),
    [
        (
            "top",
            [[-0.15, -0.10, 0.4], [0.0, 0.10, 0.4], [0.15, 0.0, 0.4]],
            [[0.0, 0.0, 1.0]] * 3,
        ),
        (
            "vertical",
            [[0.5, -0.15, 0.12], [0.5, 0.0, 0.20], [0.5, 0.15, 0.28]],
            [[1.0, 0.0, 0.0]] * 3,
        ),
        (
            "ledge",
            [[0.5, -0.12, 0.4], [0.5, 0.0, 0.4], [0.5, 0.12, 0.4]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ),
    ],
)
def test_classifies_top_vertical_and_mixed_ledge_regions(
    expected_type: str, points: list[list[float]], normals: list[list[float]]
) -> None:
    region = build_contact_region(
        _step_box(),
        np.asarray(points),
        np.asarray(normals),
        object_id="step",
        neighborhood_radius=0.30,
    )

    assert region.region_type == expected_type
    assert region.object_id == "step"
    assert region.region_id.startswith(f"step:{expected_type}:")
    assert region.parameters["face_count"] >= 1
    assert region.parameters["polygon_area"] > 0.01
    if expected_type == "top":
        assert region.parameters["normal_vertical_component"] > 0.95
    elif expected_type == "vertical":
        assert region.parameters["normal_horizontal_component"] > 0.95
    else:
        # Edge evidence deliberately retains both components instead of
        # forcing the noisy patch into a horizontal/vertical binary.
        assert region.parameters["normal_vertical_component"] > 0.25
        assert region.parameters["normal_horizontal_component"] > 0.25


def test_containment_and_nearest_projection_use_plane_local_polygon() -> None:
    region = build_contact_region(
        _step_box(),
        np.asarray([[-0.1, 0.0, 0.4], [0.0, 0.1, 0.4], [0.1, -0.1, 0.4]]),
        np.asarray([0.0, 0.0, 1.0]),
        neighborhood_radius=0.28,
    )
    inside = np.array([0.04, 0.03, 0.405])
    outside = np.array([0.85, 0.0, 0.70])

    assert region.contains(inside)
    query = region.query(outside)
    assert not query.inside
    assert query.plane_distance == pytest.approx(0.30)
    assert query.boundary_distance > 0.4
    assert region.contains(query.nearest_point, plane_tolerance=1e-8)
    assert query.nearest_point[2] == pytest.approx(0.4)
    np.testing.assert_allclose(region.project(outside), query.nearest_point)


def test_region_never_leaks_to_a_nearby_disconnected_object() -> None:
    ground = trimesh.creation.box(extents=[4.0, 4.0, 0.08])
    ground.apply_translation([0.0, 0.0, -0.04])
    step = _step_box()
    # Only a 2 cm gap separates this unrelated wall from the step face.
    wall = trimesh.creation.box(extents=[0.04, 1.2, 1.2])
    wall.apply_translation([0.54, 0.0, 0.6])
    scene = trimesh.util.concatenate([ground, step, wall])
    ground_faces = len(ground.faces)
    step_faces = len(step.faces)

    region = build_contact_region(
        scene,
        np.asarray([[0.43, -0.08, 0.4], [0.46, 0.0, 0.4], [0.43, 0.08, 0.4]]),
        np.tile([0.0, 0.0, 1.0], (3, 1)),
        object_id="step-top",
        neighborhood_radius=0.35,
    )

    assert all(
        ground_faces <= face_id < ground_faces + step_faces
        for face_id in region.face_ids
    )
    assert region.parameters["component_face_count"] == step_faces
    # The wall is closer than the patch radius, but projection remains on the
    # selected step component and its fitted top plane.
    projected = region.project([0.56, 0.0, 0.65])
    assert projected[2] == pytest.approx(0.4)


def _chair_scene() -> tuple[trimesh.Trimesh, float]:
    seat_top = 0.62
    seat = trimesh.creation.box(extents=[0.8, 0.8, 0.12])
    seat.apply_translation([0.0, 0.0, seat_top - 0.06])
    back = trimesh.creation.box(extents=[0.10, 0.8, 0.8])
    back.apply_translation([-0.36, 0.0, 0.96])
    legs = []
    for x in (-0.30, 0.30):
        for y in (-0.30, 0.30):
            leg = trimesh.creation.box(extents=[0.07, 0.07, 0.56])
            leg.apply_translation([x, y, 0.28])
            legs.append(leg)
    return trimesh.util.concatenate([seat, back, *legs]), seat_top


def test_seat_affordance_fits_plane_center_forward_and_boundary() -> None:
    chair, seat_top = _chair_scene()
    buttock_run = np.asarray(
        [
            [-0.16, -0.14, seat_top],
            [-0.12, 0.15, seat_top],
            [0.12, -0.15, seat_top],
            [0.16, 0.14, seat_top],
        ]
    )
    region = build_seat_affordance(
        chair,
        buttock_run,
        np.tile([0.0, 0.0, 1.0], (len(buttock_run), 1)),
        object_id="chair-seat",
        neighborhood_radius=0.36,
    )

    assert region.region_type == "seat"
    assert region.parameters["support_type"] == "top"
    assert region.center[2] == pytest.approx(seat_top)
    assert region.normal[2] > 0.99
    assert abs(float(np.dot(region.normal, region.forward))) < 1e-12
    assert region.boundary.shape[1] == 3
    assert np.allclose(region.boundary[:, 2], seat_top)
    assert region.contains([0.0, 0.0, seat_top])
    assert region.contains([0.24, 0.20, seat_top])
    assert not region.contains([0.55, 0.0, seat_top])


def test_object_local_region_can_follow_a_future_object_transform() -> None:
    box = _step_box()
    initial = np.eye(4)
    region = build_contact_region(
        box,
        np.asarray([[-0.1, 0.0, 0.4], [0.1, 0.0, 0.4]]),
        np.asarray([0.0, 0.0, 1.0]),
        object_id="movable-step",
        world_from_object=initial,
        neighborhood_radius=0.25,
    )
    moved = np.eye(4)
    moved[:3, 3] = [1.2, -0.4, 0.3]

    assert region.contains([1.2, -0.4, 0.7], world_from_object=moved)
    moved_projection = region.project([1.8, -0.4, 1.0], world_from_object=moved)
    assert moved_projection[2] == pytest.approx(0.7)
    assert region.object_id == "movable-step"


def test_region_building_never_mutates_mesh_or_caller_arrays() -> None:
    mesh = _step_box()
    points = np.asarray([[-0.1, -0.1, 0.4], [0.1, 0.1, 0.4]])
    normals = np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    vertices_before = mesh.vertices.copy()
    faces_before = mesh.faces.copy()
    points_before = points.copy()
    normals_before = normals.copy()

    build_contact_region(mesh, points, normals, neighborhood_radius=0.25)

    np.testing.assert_array_equal(mesh.vertices, vertices_before)
    np.testing.assert_array_equal(mesh.faces, faces_before)
    np.testing.assert_array_equal(points, points_before)
    np.testing.assert_array_equal(normals, normals_before)
