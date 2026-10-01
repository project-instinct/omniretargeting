"""Skin-rank chatter is suppressed while actual motion and geometry survive."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.contact_correspondence import coherent_sample_indices


@pytest.mark.parametrize("fps", [15, 30, 60, 120])
def test_rapid_slide_is_preserved_while_nearest_sample_rank_chatters(fps):
    t = np.arange(fps + 1) / fps
    projected = np.zeros((len(t), 2, 3))
    projected[:, :, 0] = 1.5 * t[:, None] + np.array([0.0, 0.08])
    samples = projected.copy()
    samples[:, :, 2] = 0.004 + 0.001 * ((np.arange(len(t))[:, None] + [0, 1]) % 2)
    distance = samples[:, :, 2]
    original = samples.copy()
    ids = coherent_sample_indices(distance, projected, samples, max_distance=0.02)
    assert len(np.unique(ids)) == 1
    observed = projected[np.arange(len(t)), ids]
    np.testing.assert_allclose(np.diff(observed[:, 0]) * fps, 1.5)
    np.testing.assert_array_equal(samples, original)
    rotation = Rotation.from_rotvec([0.6, -0.2, 0.8]).as_matrix()
    transformed = coherent_sample_indices(
        distance[:, ::-1],
        projected[:, ::-1] @ rotation.T + [3.0, -2.0, 4.0],
        samples[:, ::-1] @ rotation.T + [3.0, -2.0, 4.0],
        max_distance=0.02,
    )
    assert np.all(transformed == 1 - ids)


def test_correspondence_cannot_keep_an_identity_that_leaves_contact_geometry():
    samples = np.zeros((12, 2, 3))
    samples[:, 1, 0] = 0.2
    samples[:, :, 2] = 0.005
    samples[6:, 0, 2] = 0.5
    projected = samples.copy()
    projected[:, :, 2] = 0
    ids = coherent_sample_indices(
        samples[:, :, 2], projected, samples, max_distance=0.03
    )
    assert np.all(ids[6:] == 1)
