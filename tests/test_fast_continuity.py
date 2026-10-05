"""No RMS, quaternion sign or reference change can hide a bad coordinate."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast.continuity import CorrectionContinuity, audit_continuity
from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._fast.solver import optimize
from tests.test_fast_backend import _objective


def test_small_root_rotation_chatter_is_visible_even_within_pose_envelope():
    objective, poses, _ = _objective()
    coords = objective.coords
    fixed = np.repeat(poses[:1], len(poses), axis=0)
    objective.tasks.human = np.repeat(objective.tasks.human[:1], len(poses), axis=0)
    candidate = fixed.copy()
    yaw = np.zeros(len(poses))
    yaw[::2] = .01
    candidate[:, coords.root_q + 3:coords.root_q + 7] = Rotation.from_euler('z', yaw[:, None]).as_quat(scalar_first=True)
    assert ReferenceShape(objective, fixed).accepts(candidate)
    result = audit_continuity(coords, fixed, candidate)
    assert not result['strict_accepted']
    assert not result['relaxed_chatter_accepted']
    assert result['channels']['root_rotation_order_3']['strict_violations']
    assert not CorrectionContinuity(coords, fixed).accepts(candidate)


def test_authored_fast_motion_and_quaternion_signs_are_preserved():
    objective, poses, _ = _objective()
    coords = objective.coords
    identical = poses.copy()
    identical[::2, coords.root_q + 3:coords.root_q + 7] *= -1
    report = audit_continuity(coords, poses, identical)
    assert report['strict_accepted'] and report['relaxed_chatter_accepted']
    assert CorrectionContinuity(coords, poses).accepts(identical)


def test_strict_velocity_increase_is_separate_from_relaxed_chatter():
    objective, fixed, _ = _objective()
    coords = objective.coords
    fixed = np.repeat(fixed[:1], len(fixed), axis=0)
    candidate = fixed.copy()
    candidate[:, coords.qi[0]] += np.arange(len(candidate)) * .001
    report = audit_continuity(coords, fixed, candidate)
    assert not report['strict_accepted']
    assert report['relaxed_chatter_accepted']
    assert report['channels']['joints_order_1']['strict_per_coordinate'][0] == len(candidate) - 1
    assert CorrectionContinuity(coords, fixed, chatter=True).accepts(candidate)
    noisy = fixed.copy()
    noisy[::2, coords.qi[0]] += .015
    assert not CorrectionContinuity(coords, fixed, chatter=True).accepts(noisy)


def test_secondary_solver_keeps_root_and_each_acquired_temporal_bound():
    objective, poses, _ = _objective()
    coords = objective.coords
    guard = CorrectionContinuity(coords, poses)
    objective.candidate_guards = [guard]
    objective.fixed_root = poses[:, coords.root_q:coords.root_q + 7].copy()
    objective.source_shape = ReferenceShape(objective, poses)
    for eff in objective.effectors:
        eff['contact'] = eff['contact'] + [0., 0., .1]
    accepted = []
    motion, _ = optimize(objective, poses, iterations=3,
                          callback=lambda q, _: accepted.append(q))
    for q in [motion, *accepted]:
        np.testing.assert_array_equal(q[:, coords.root_q:coords.root_q + 7], objective.fixed_root)
        assert guard.accepts(q)


def test_pinning_an_already_locked_root_keeps_integer_variable_indices():
    from omniretargeting._fast.solver import _active_bounds, _hold
    from omniretargeting._sequence.coordinates import Coordinates

    objective, poses, _ = _objective()
    objective.coords = Coordinates(objective.model, lock_root_xy=True,
                                   lock_root_z=True, lock_root_orientation=True)
    objective.fixed_root = poses[:, :7].copy()
    width = objective.coords.width
    variables = _active_bounds(objective, poses, np.zeros(len(poses) * width))
    assert np.issubdtype(variables.dtype, np.integer)
    # Exercise the actual array indexing that used to fail for no root columns.
    _hold(np.zeros((width, len(poses) * width)), variables)


@pytest.mark.parametrize("correction", [False, True])
def test_correction_linearizations_include_noncommuting_world_angular_jerk(correction):
    from omniretargeting._fast.continuity import correction_derivatives, derivatives

    objective, poses, _ = _objective()
    coords = objective.coords
    rng = np.random.default_rng(41)
    poses = coords.integrate(poses, rng.normal(size=(len(poses), coords.width)) * .1, 1.)
    guard = CorrectionContinuity(coords, poses, correction=correction)
    fixed = poses.copy()
    poses = coords.integrate(poses, rng.normal(size=(len(poses), coords.width)) * .07, 1.)
    direction = rng.normal(size=(len(poses), coords.width)) * .02
    epsilon = 1e-5
    values = lambda q: correction_derivatives(coords, fixed, q) if correction else derivatives(coords, q)
    plus = values(coords.integrate(poses, direction, epsilon))
    minus = values(coords.integrate(poses, direction, -epsilon))
    for name, (value, matrix) in guard.linearizations(poses).items():
        np.testing.assert_allclose(value, values(poses)[name], atol=1e-12)
        expected = ((plus[name] - minus[name]) / (2 * epsilon)).ravel()
        np.testing.assert_allclose(matrix @ direction.ravel(), expected, atol=2e-8, rtol=2e-6)


def test_windowed_budget_allows_the_local_reference_envelope_only():
    from omniretargeting.motion_budgets import CHATTER_RMS_FRACTION, windowed_budget

    reference = np.zeros((40, 2))
    reference[20, 0] = 1.0  # one large stencil in a smooth reference
    budget = windowed_budget(reference)
    rms = np.sqrt(np.mean(reference**2, axis=0))
    # Within three stencils of the peak the peak bounds; elsewhere 10% of RMS.
    assert budget[17, 0] == budget[23, 0] == 1.0 + CHATTER_RMS_FRACTION * rms[0]
    assert budget[10, 0] == CHATTER_RMS_FRACTION * rms[0]
    np.testing.assert_array_equal(budget[:, 1], 0.0)
