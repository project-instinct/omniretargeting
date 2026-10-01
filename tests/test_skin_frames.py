"""Material frames retain skin rotation without PCA axis swaps or flattening."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.skin_frames import material_surface_frames


@pytest.mark.parametrize('scale', [.01, 1., 12.])
def test_fixed_material_fit_preserves_fast_roll_and_arbitrary_world_registration(scale):
    rng = np.random.default_rng(923)
    reference = rng.normal(size=(31, 3)) * [2, 1, 0] * scale
    rotations = Rotation.from_euler('xyz', [[0, 0, 0], [91, 32, -16], [179, -22, 111]], degrees=True).as_matrix()
    frame = Rotation.from_rotvec([.3, -.1, .5]).as_matrix()
    translations = rng.normal(size=(3, 1, 3)) * scale
    world = np.einsum('tij,nj->tni', rotations, reference) + translations
    original = world.copy()
    actual, residual = material_surface_frames(reference, world, frame)
    np.testing.assert_allclose(actual, rotations @ frame, atol=1e-12)
    np.testing.assert_allclose(residual, 0, atol=1e-12)
    np.testing.assert_array_equal(world, original)


def test_surface_deformation_is_reported_without_reselecting_contact_points():
    reference = np.array([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0.]])
    world = reference[None].copy()
    world[0, 3, 2] += .2
    before = world.copy()
    rotation, rms = material_surface_frames(reference, world, np.eye(3))
    assert rms[0] > .01
    assert np.linalg.det(rotation[0]) == pytest.approx(1)
    np.testing.assert_array_equal(before, world)


def test_material_fit_rejects_unobservable_line():
    points = np.column_stack([np.arange(4), np.zeros((4, 2))])
    with pytest.raises(ValueError, match='unobservable'):
        material_surface_frames(points, points[None], np.eye(3))


def test_resampling_preserves_one_semantic_frame_and_material_ids():
    from omniretargeting.data_sources.base import MotionData

    rotation = Rotation.from_euler('xyz', [[10, 20, 30], [80, -20, 55]], degrees=True).as_matrix()
    metadata = {
        'link_orientations': {'left': rotation},
        'foot_sole_fit_rms': {'left': np.array([.001, .003])},
        'end_effector_markers': {'left_foot': {
            'forward_directions': rotation[:, :, 0], 'up_directions': rotation[:, :, 2],
            # The number of vertices can coincidentally equal the frame count.
            'vertex_ids': np.array([8, 31]),
        }},
    }
    motion = MotionData(np.zeros((2, 1, 3)), framerate=30., metadata=metadata)
    resampled = motion.resample(60.)
    frames = resampled.metadata['link_orientations']['left']
    marker = resampled.metadata['end_effector_markers']['left_foot']
    np.testing.assert_allclose(marker['forward_directions'], frames[:, :, 0], atol=1e-12)
    np.testing.assert_allclose(marker['up_directions'], frames[:, :, 2], atol=1e-12)
    np.testing.assert_array_equal(marker['vertex_ids'], [8, 31])
    np.testing.assert_allclose(resampled.metadata['foot_sole_fit_rms']['left'], [.001, .002, .003])
    np.testing.assert_array_equal(motion.metadata['link_orientations']['left'], rotation)
