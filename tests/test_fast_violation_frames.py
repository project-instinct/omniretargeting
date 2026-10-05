"""Each guard's ``violation_frames`` is the per-frame form of its ``accepts``."""

import numpy as np
import pytest

from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.contact_guard import ContactGuard
from omniretargeting._fast.continuity import CorrectionContinuity
from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from tests.test_fast_backend import _problem
from tests.test_fast_contact_guard import _acquired


def _guards(objective, poses):
    objective.effectors[0]["normal"][:] = [0., 0., 1.]
    rotations = _problem()[5]
    return [ReferenceShape(objective, poses), AcquisitionNormalGuard(objective, poses),
            ContactGuard(objective, poses), CorrectionContinuity(objective.coords, poses),
            SequenceContinuityGuard(objective.tasks, poses, rotations, windowed=True)]


@pytest.mark.parametrize("seed", range(4))
def test_violation_frames_are_the_per_frame_form_of_accepts(seed):
    objective, poses = _acquired(surfaces=True, moving=True)
    guards = _guards(objective, poses)
    rng = np.random.default_rng(seed)
    for scale in (1e-4, 1e-3, 1e-2, 1e-1):
        step = scale * rng.normal(size=(len(poses), objective.width))
        step[rng.random(len(poses)) < .7] = 0.
        candidate = objective.project(objective.coords.integrate(poses, step, 1.))
        for guard in guards:
            bad = guard.violation_frames(candidate)
            assert bad.shape == (len(poses),)
            assert bool(bad.any()) == (not guard.accepts(candidate)), type(guard).__name__
            moved = np.flatnonzero(np.abs(step).sum(axis=1) > 0)
            # Chatter is a whole-trajectory high-frequency certificate, so a
            # flagged frame can depend on samples outside a three-frame stencil.
            if bad.any() and not getattr(guard, "chatter", False):
                # A rejected frame moved or lies in a moved frame's stencil.
                assert np.min(np.abs(np.flatnonzero(bad)[:, None] - moved[None]), axis=1).max() <= 3
