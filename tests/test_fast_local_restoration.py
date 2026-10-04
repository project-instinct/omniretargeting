"""Localized second-order corrections retain exact boundary/guard acceptance."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.local_restoration import (
    _program, _support, _violating_columns, backtrack_with_restoration, restore_local,
)
from omniretargeting._fast.repair_restoration import _solve_restoration


def test_restricted_program_keeps_cross_boundary_rows_and_entire_cones():
    # Only x1 can change. Fixed x0=3 is already in each residual: x0+x1<=4,
    # and norm([x0, x1-2])<=sqrt(10). The latter requires x1>=1.
    linear = [(sparse.csr_matrix([[1., 1.]]), [-np.inf], [1.])]
    norms = [(sparse.eye(2, format="csr"), np.array([[3., -2.]]), np.array([np.sqrt(10.)]))]
    result, _ = _solve_restoration(_program(
        linear, norms, np.array([1]), np.ones(1), np.array([-5.]), np.array([5.])))
    np.testing.assert_allclose(result, [1.], atol=1e-6)


@pytest.mark.parametrize("norm", [False, True])
def test_unchangeable_violated_constraint_is_infeasible(norm):
    linear, norms = [], []
    if norm:
        norms = [(sparse.eye(2, format="csr"), np.array([[2., 0.]]), np.array([1.]))]
    else:
        linear = [(sparse.csr_matrix([[1., 0.]]), [1.], [np.inf])]
    result, _ = _solve_restoration(_program(
        linear, norms, np.array([1]), np.ones(1), np.array([-5.]), np.array([5.])))
    assert result is None


class _Coordinates:
    width = 2
    root_q = None
    qi = scalar_cols = np.array([0, 1])

    def integrate(self, poses, step, alpha):
        return poses + alpha * step

    def difference(self, origin, candidate):
        return candidate - origin

    def bounds(self, poses, angular, translation):
        return np.full_like(poses, -angular), np.full_like(poses, angular)


class _CurvedGuard:
    def violation_frames(self, poses):
        bad = np.zeros(len(poses), bool)
        bad[20] = poses[20, 0] ** 2 - poses[20, 1] > 1e-7
        return bad

    def accepts(self, poses):
        return not self.violation_frames(poses).any()

    def repair_bounds(self, poses, **kwargs):
        jac = sparse.csr_matrix(([-2 * poses[20, 0], 1.], ([0, 0], [40, 41])),
                                shape=(1, poses.size))
        return [(jac, [poses[20, 0] ** 2 - poses[20, 1]], [np.inf])], []


class _TemporalGuard:
    def __init__(self, frames):
        diff = sparse.diags([-np.ones(frames - 1), np.ones(frames - 1)], [0, 1],
                            shape=(frames - 1, frames), format="csr")
        self.matrix = sparse.kron(diff, sparse.csr_matrix([[0., 1.]]), format="csr")

    def repair_bounds(self, poses):
        values = self.matrix @ poses.ravel()
        return [(self.matrix, -.002 - values, .002 - values)], []

    def violation_frames(self, poses):
        rows = np.abs(self.matrix @ poses.ravel()) > .002 + 1e-8
        return np.r_[rows, False] | np.r_[False, rows]

    def accepts(self, poses):
        return not self.violation_frames(poses).any()


def _problem():
    poses = np.zeros((64, 2))
    shape, temporal = _CurvedGuard(), _TemporalGuard(len(poses))
    objective = SimpleNamespace(coords=_Coordinates(), source_shape=shape,
                                cfg={"angular_step_rad": .2, "translation_step_m": .1},
                                lower=np.array([-1., -1.]), upper=np.array([1., 1.]),
                                project=lambda q: q)
    direction = np.zeros_like(poses)
    direction[:, 0] = .1
    return objective, poses, direction, [shape, temporal]


def test_second_order_error_is_repaired_without_throttling_other_frames():
    objective, poses, direction, guards = _problem()
    candidate, alpha, restored = backtrack_with_restoration(objective, poses, direction, guards)
    assert alpha == 1.
    assert restored
    assert all(guard.accepts(candidate) for guard in guards)
    outside = ~_support(guards[0].violation_frames(poses + direction))
    np.testing.assert_array_equal(candidate[outside], (poses + direction)[outside])
    # Temporal rows force the normal correction to spread before/after frame 20.
    assert candidate[19, 1] > 0 and candidate[21, 1] > 0


def test_a_linear_solver_result_cannot_override_nonlinear_rejection(monkeypatch):
    import omniretargeting._fast.local_restoration as local

    objective, poses, direction, guards = _problem()
    monkeypatch.setattr(local, "_solve_restoration", lambda p: (np.zeros(len(p[1])), "optimal"))
    bad = guards[0].violation_frames(poses + direction)
    assert restore_local(objective, poses, poses + direction, guards, bad) is None
    candidate, alpha, restored = backtrack_with_restoration(objective, poses, direction, guards)
    assert candidate is None and alpha == 0.
    assert not restored


def test_real_guards_keep_reference_budgets_and_fixed_frames():
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    from omniretargeting._fast.continuity import CorrectionContinuity
    from omniretargeting._fast.reference import ReferenceShape
    from tests.test_fast_contact_guard import _acquired

    objective, poses = _acquired()
    shape = objective.source_shape = ReferenceShape(objective, poses)
    objective.effectors[0]["normal"][:] = [0., 0., 1.]
    continuity = CorrectionContinuity(objective.coords, poses)
    guards = [shape, AcquisitionNormalGuard(objective, poses), continuity]
    saved = {k: v.copy() for k, v in continuity.bounds.items()}
    candidate = poses.copy()
    candidate[2, 2] += .002
    bad = np.logical_or.reduce([g.violation_frames(candidate) for g in guards])
    restored = restore_local(objective, poses, candidate, guards, bad)
    assert restored is not None
    assert all(g.accepts(restored) for g in guards)
    np.testing.assert_array_equal(restored[~_support(bad)], candidate[~_support(bad)])
    np.testing.assert_array_equal(shape.seed, poses)
    for name, budget in saved.items():
        np.testing.assert_array_equal(continuity.bounds[name], budget)


def test_conic_restoration_is_opt_in_and_keeps_the_original_fallback(monkeypatch):
    from dataclasses import replace
    from omniretargeting._fast import local_restoration, runtime_options, shape_step, solver

    objective, poses, direction, guards = _problem()
    calls = []
    monkeypatch.setattr(shape_step, "bounded_shape_step", lambda *a: direction.ravel())
    monkeypatch.setattr(shape_step, "restore_shape_step", lambda *a: calls.append("fallback"))
    monkeypatch.setattr(solver, "backtrack", lambda *a: (poses, .25))
    monkeypatch.setattr(local_restoration, "backtrack_with_restoration", lambda *a: (None, 0., False))
    for enabled in (False, True):
        monkeypatch.setattr(runtime_options, "OPTIONS", replace(
            runtime_options.OPTIONS, local_conic_restoration=enabled))
        proposal = solver._conic_proposal(objective, poses, None, None, None, 1., guards)
        assert proposal[2] == (0. if enabled else .25)
    assert calls == ["fallback"]
    monkeypatch.setattr(local_restoration, "backtrack_with_restoration", lambda *a: (poses, .125, True))
    proposal = solver._conic_proposal(objective, poses, None, None, None, 1., guards)
    assert proposal[2] == .125
    assert calls == ["fallback"]  # A successful local solve is not restored again.


def test_projection_that_breaks_a_guard_cannot_be_accepted():
    objective, poses, direction, guards = _problem()
    # Simulates joint/root projection that undoes the normal correction.
    def project(poses):
        result = poses.copy()
        result[20, 1] = -.01
        return result
    objective.project = project
    candidate = poses + direction
    assert restore_local(objective, poses, candidate, guards,
                         guards[0].violation_frames(candidate)) is None


def test_coordinate_halos_follow_only_the_violated_row_and_cone_support():
    frames, width = 64, 4
    linear = [(sparse.csr_matrix(([1., 1.], ([0, 1], [20 * width, 45 * width + 1])),
                                 shape=(2, frames * width)), [-1., 1.], [1., np.inf])]
    norms = [(sparse.csr_matrix(([1., 1.], ([0, 1], [20 * width + 2, 20 * width + 3])),
                                shape=(2, frames * width)), np.array([[2., 0.]]), np.array([1.]))]
    selected = _violating_columns(linear, norms, frames, width)
    assert not selected[:, 0].any()
    np.testing.assert_array_equal(np.flatnonzero(selected[:, 1]), np.arange(37, 54))
    np.testing.assert_array_equal(np.flatnonzero(selected[:, 2]), np.arange(12, 29))
    np.testing.assert_array_equal(selected[:, 2], selected[:, 3])


def test_widespread_violation_can_restore_one_coordinate_without_freeing_others():
    objective, poses, _, _ = _problem()

    class Guard:
        def violation_frames(self, q):
            return q[:, 0] ** 2 > .01 + 1e-7

        def accepts(self, q):
            return not self.violation_frames(q).any()

        def repair_bounds(self, q, **kwargs):
            jac = sparse.csr_matrix((2 * q[:, 0], (np.arange(len(q)), np.arange(len(q)) * 2)),
                                    shape=(len(q), q.size))
            return [(jac, np.full(len(q), -np.inf), .01 - q[:, 0] ** 2)], []

    guard = objective.source_shape = Guard()
    candidate = poses.copy()
    candidate[:, 0] = .101
    candidate[:, 1] = .15  # An unrelated useful proposal must stay unchanged.
    restored = restore_local(objective, poses, candidate, [guard], guard.violation_frames(candidate))
    assert restored is not None and guard.accepts(restored)
    np.testing.assert_array_equal(restored[:, 1], candidate[:, 1])
