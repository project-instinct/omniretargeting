"""Emergency scale recovery remains guarded; ordinary transfer does not use it."""

import numpy as np
import pytest

from omniretargeting._fast.coarse_transfer import (
    interpolate_correction, scale_transfer_correction as transfer_correction,
)
from omniretargeting._fast.continuity import CorrectionContinuity
from tests.test_fast_legacy_window import problem


@pytest.mark.parametrize('count', [35, 36])
def test_scalar_interpolation_stays_between_knots_and_preserves_authored_frames(count):
    objective, original = problem()
    fixed = original[:count].copy()
    # Authored odd-frame motion must survive interpolation of zero corrections.
    fixed[1::2, 7] += .02
    indices = np.arange(0, count, 2)
    correction = np.resize([0., .03, .03, 0., 0., -.02], len(indices))
    coarse = fixed[indices].copy()
    coarse[:, 7] += correction
    col = objective.coords.scalar_cols[0]
    delta = interpolate_correction(objective.coords, fixed, coarse, indices)
    np.testing.assert_allclose(delta[indices, col], correction, atol=1e-12)
    for knot in range(len(indices) - 1):
        a, b = correction[knot:knot + 2]
        value = delta[indices[knot] + 1, col]
        assert min(a, b) - 1e-12 <= value <= max(a, b) + 1e-12
    assert delta[-1, col] == pytest.approx(correction[-1])
    zero = interpolate_correction(objective.coords, fixed, fixed[indices], indices)
    np.testing.assert_allclose(objective.coords.integrate(fixed, zero, 1.), fixed, atol=1e-12)


@pytest.mark.parametrize('at_upper', [False, True])
def test_outward_correction_at_omitted_reference_limit_is_projected(at_upper):
    objective, fixed = problem()
    objective.source_shape, objective.candidate_guards = None, []
    direction = 1. if at_upper else -1.
    box = objective.upper if at_upper else objective.lower
    box[9, 0] = fixed[9, 7]  # This limit is absent from the coarse knots.
    indices = np.arange(0, len(fixed), 2)
    coarse = fixed[indices].copy()
    coarse[:, 7] += direction * .01
    warm, report = transfer_correction(objective, fixed, coarse, indices)
    assert report['accepted'] and report['global_scale'] == 1.
    assert report['retained_scale'] > .9
    assert warm[9, 7] == box[9, 0]
    np.testing.assert_allclose(warm[:9, 7], fixed[:9, 7] + direction * .01)
    assert report['projection_frames'] == [9]
    assert report['guard_failures'] == []


def test_root_uses_solver_projection_before_exact_certification():
    objective, fixed = problem()
    objective.source_shape, objective.candidate_guards = None, []
    objective.root_budget = .02
    objective.reference = fixed.copy()
    indices = np.arange(0, len(fixed), 2)
    coarse = fixed[indices].copy()
    coarse[:, :3] += [.03, -.04, .02]
    delta = interpolate_correction(objective.coords, fixed, coarse, indices)
    expected = objective.project(objective.coords.integrate(fixed, delta, 1.))
    warm, report = transfer_correction(objective, fixed, coarse, indices)
    assert report['accepted'] and report['global_scale'] == 1.
    np.testing.assert_allclose(warm, expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(warm[:, :3] - fixed[:, :3], axis=1), .02)
    assert report['guard_failures'] == []


def test_each_scale_is_projected_and_clipping_jerk_is_reported_and_rejected(monkeypatch):
    objective, fixed = problem()
    fixed[:, 7] = .003 * np.arange(len(fixed))
    guard = CorrectionContinuity(objective.coords, fixed, correction=False)
    objective.source_shape, objective.candidate_guards = None, [guard]
    objective.upper[:] = fixed[:, objective.coords.qi] + .3
    objective.upper[8:12, 0] = fixed[8:12, 7] + .025
    project, calls = objective.project, []
    def track(poses):
        result = project(poses)
        calls.append(result.copy())
        return result
    monkeypatch.setattr(objective, 'project', track)
    coarse = fixed[::2].copy()
    coarse[:, 7] += .1
    warm, report = transfer_correction(objective, fixed, coarse, np.arange(0, len(fixed), 2),
                                        spread_frames=2)
    assert len(calls) == len(report['attempts'])
    first = report['first_failing_candidate']
    key = 'CorrectionContinuity:0'
    assert first['guard_failure_counts'] == {key: len(first['failures'][key])}
    assert set(range(5, 8)).issubset(first['failures'][key])
    assert 'joint_limits' not in first['failures']
    assert not guard.accepts(calls[0]) and guard.accepts(warm)
    assert report['retained_scale'] == pytest.approx(.25)
    assert all(np.all(q[:, objective.coords.qi] <= objective.upper) for q in calls)
