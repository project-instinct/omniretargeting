"""Direct cones preserve the complete legacy window model and guard bounds."""

import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.legacy_window_qp import WindowObjective, serial_proposals
from omniretargeting._fast.shape_step import bounded_shape_step, restore_shape_step
from omniretargeting._fast.solver import banded_system
from omniretargeting._fast.window_conic import bounded_step, guard_program, restore_step, step_bounds
from tests.test_fast_legacy_window import problem


@pytest.mark.parametrize('frames', [range(0, 8), range(8, 20), range(28, 36)])
def test_direct_cones_match_window_constraints(frames):
    objective, poses = problem(extra=True)
    window = WindowObjective(objective, poses, frames)
    origin = poses[window.span]
    lo, hi = step_bounds(window, origin)
    flat = cp.Variable(origin.shape[0] * objective.width)
    constraints = [flat >= lo.ravel(), flat <= hi.ravel()]
    for guard in [window.source_shape, *window.candidate_guards]:
        constraints += guard.constraints(origin, flat)
    matrix, offset, linear, sizes = guard_program(window, origin, lo, hi)
    for amplitude in (0., 1e-5, .001, .5):
        step = np.random.default_rng(14).normal(size=flat.size) * amplitude
        flat.value = step
        expected = all(np.max(c.violation(), initial=0.) <= 1e-9 for c in constraints)
        slack, start = offset - matrix @ step, linear
        actual = bool(np.min(slack[:linear], initial=0.) >= -1e-9)
        for size in sizes:
            actual &= np.linalg.norm(slack[start + 1:start + size]) <= slack[start] + 1e-9
            start += size
        assert actual == expected
        assert start == len(offset)


def test_direct_bounded_step_matches_cvxpy_objective_and_feasibility(monkeypatch):
    from omniretargeting._fast import window_qoco
    # Compares against the Clarabel/CVXPY bounded step, not the default QOCO proposal.
    monkeypatch.setattr(window_qoco, 'WINDOW_QOCO_PROPOSALS', False)
    objective, poses = problem(extra=True)
    window = WindowObjective(objective, poses, range(8, 20))
    origin = poses[window.span]
    acc, residual, jac, *_ = window.evaluate(origin)
    gradient = acc.gradient.ravel() + jac.T @ residual
    system = banded_system(window.frames, window.width, acc.hessian, jac)
    diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), 1e-9)
    with serial_proposals():
        old = bounded_shape_step(window, origin, system, gradient, diagonal, .001)
    new = bounded_step(window, origin, system, gradient, diagonal, .001)
    assert old is not None and new is not None
    hessian = sparse.block_diag(acc.hessian) + jac.T @ jac + sparse.diags(.001 * diagonal + 1e-9)

    def value(step):
        return step @ hessian @ step + 2 * gradient @ step

    assert value(new) == pytest.approx(value(old), rel=2e-5, abs=1e-6)
    np.testing.assert_allclose(new, old, atol=2e-5, rtol=1e-3)


def test_direct_restoration_preserves_guards_and_outside_bytes():
    objective, poses = problem()
    window = WindowObjective(objective, poses, range(9, 16))
    origin = poses[window.span]
    step = np.zeros((window.frames, window.width))
    step[:, objective.coords.scalar_cols[0]] = .001
    with serial_proposals():
        expected = restore_shape_step(window, origin, step)
    actual = restore_step(window, origin, step)
    assert expected is not None and actual is not None
    assert window.source_shape.accepts(actual)
    assert all(g.accepts(actual) for g in window.candidate_guards)
    full = window.embed(actual)
    assert full[:9].tobytes() == poses[:9].tobytes()
    assert full[16:].tobytes() == poses[16:].tobytes()
