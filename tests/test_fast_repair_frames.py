from types import SimpleNamespace

import numpy as np
from scipy import sparse

from omniretargeting._fast.repair_frames import group_frames, halo_frames, restrict

MODEL = SimpleNamespace(body_weldid=np.arange(6), geom_bodyid=np.arange(6))


def test_mask_has_clip_length_and_halo_on_short_clips():
    for frames, hit in ((20, 0), (20, 19), (5, 2), (300, 150)):
        mask = group_frames({"depths": {(hit, "static", 2): .01}}, MODEL, 2, frames, 15)
        assert mask.shape == (frames,)
        expected = np.zeros(frames, bool)
        expected[max(0, hit - 15):hit + 16] = True
        assert np.array_equal(mask, expected)


def test_mask_follows_either_self_partner_and_ignores_other_groups():
    depths = {(3, "self", 1, 4): .002, (9, "static", 5): .01, (12, "static", 4): 0.}
    mask = group_frames({"depths": depths}, MODEL, 4, 30, 1)
    assert np.flatnonzero(mask).tolist() == [2, 3, 4]
    assert group_frames({"depths": {(1, "static", 3): 0.}}, MODEL, 3, 10, 2) is None


def test_restrict_keeps_whole_frames_of_the_projection():
    width, frames = 29, 12
    selector = sparse.eye(width, format="csc")[:, [0, 4, 7]]
    projection = sparse.kron(sparse.eye(frames), selector, format="csc")
    mask = np.zeros(frames, bool)
    mask[[2, 3, 9]] = True
    restricted = restrict(projection, mask)
    assert restricted.shape == (frames * width, 9)
    step = np.zeros(frames * width)
    step[restricted @ np.arange(1, 10) != 0] = 1
    moved = np.flatnonzero(step.reshape(frames, width).any(axis=1))
    assert moved.tolist() == [2, 3, 9]
    assert restrict(projection, None) is projection


def test_halo_scales_with_framerate():
    assert halo_frames(30.) == 15
    assert halo_frames(4.) == 4
