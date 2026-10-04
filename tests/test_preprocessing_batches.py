"""Bitwise parity against the scalar rotations and sticking-ray construction."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.anchor_motion import source_material_rotations
from omniretargeting.contact_mechanics import ContactCandidate


@pytest.mark.parametrize("samples", [1, 2, 3, 8, 50, 300, 512])
def test_material_rotation_batch_matches_scalar_bits(samples):
    rng = np.random.default_rng(samples)
    orientation = Rotation.random(10 * samples, random_state=rng).as_matrix().reshape(10, samples, 3, 3)
    # Also exercise SciPy's per-matrix orthogonalization, without changing the
    # matrix-conversion backend when the batch is flattened.
    orientation[2, :, 0, 0] += 1e-4
    surface = Rotation.random(10, random_state=rng).as_matrix()
    relative = surface[:, None].transpose(0, 1, 3, 2) @ orientation
    expected = np.asarray([Rotation.from_matrix(frame).mean().as_matrix() for frame in relative])
    observed = source_material_rotations(orientation, surface)
    assert expected.tobytes() == observed.tobytes()


def scalar_rays(candidate, origin, length_scale):
    points = candidate.surface_points
    normals = np.broadcast_to(candidate.normals, points.shape).copy()
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    forces, locations = [], []
    for point, normal in zip(points, normals):
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        first = np.cross(normal, axis)
        first /= np.linalg.norm(first)
        second = np.cross(normal, first)
        angles = np.arange(candidate.friction_sides) * (2 * np.pi / candidate.friction_sides)
        directions = normal + candidate.friction * (
            np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second)
        forces.extend(directions)
        locations.extend([point] * len(directions))
    forces = np.asarray(forces)
    return np.c_[forces, np.cross(np.asarray(locations) - origin, forces) / length_scale]


def test_sticking_rays_batch_keeps_row_order_and_all_bits():
    rng = np.random.default_rng(23)
    for index in range(150):
        count = int(rng.integers(1, 90))
        points, normals = rng.normal(size=(2, count, 3))
        if index % 3 == 0:
            normals[:min(count, 3)] = np.eye(3)[:min(count, 3)]
        candidate = ContactCandidate("patch", points, points, normals,
                                     float(rng.random()), .01,
                                     friction_sides=int(rng.integers(3, 12)))
        origin, length = rng.normal(size=3), float(rng.random() + .1)
        expected = scalar_rays(candidate, origin, length)
        observed = candidate.rays(origin, length)
        assert expected.tobytes() == observed.tobytes()

