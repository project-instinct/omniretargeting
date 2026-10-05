"""Exact restriction of full hard domains, including fixed boundary poses."""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from omniretargeting._fast.window_assembly import prepare_window_model
from omniretargeting._fast.qp_assembly import prepare_model
from omniretargeting._fast.solver import banded_system
from omniretargeting._fast.qp_residual import cone_violation


class EmptyPriority:
    def begin_iteration(self, poses, model, *, frames=None):
        self.model = model


def make_objective():
    from tests.test_fast_backend import _objective
    from omniretargeting._fast.reference import ReferenceShape
    from omniretargeting._fast.continuity import CorrectionContinuity
    from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
    objective, poses, (tasks, _, _) = _objective(temporal=True)
    objective.priority_clearance = EmptyPriority()
    objective.source_shape = ReferenceShape(objective, poses)
    objective.candidate_guards = [
        CorrectionContinuity(objective.coords, poses, root_norm=True, chatter=False),
        SequenceContinuityGuard(tasks, poses, objective.root_rotations, chatter=False),
    ]
    return objective, poses


def test_window_hessian_gradient_equal_global_restriction():
    objective, poses = make_objective()
    frames = range(2, 5)
    acc, r, j, _, _ = objective.evaluate(poses)
    system = banded_system(len(poses), objective.width, acc.hessian, j)
    gradient = acc.gradient.ravel() + j.T @ r
    diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), 1e-9)
    full = prepare_model(objective, poses, system, gradient, diagonal)
    local, mandatory = prepare_window_model(objective, poses, frames)
    cols = slice(frames.start * objective.width, frames.stop * objective.width)
    np.testing.assert_allclose(local.gradient, full.gradient[cols], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(local.hessian.toarray(), full.hessian[cols, cols].toarray(), atol=1e-12, rtol=1e-12)
    assert local.origin.shape == poses.shape and local.frames == frames
    assert mandatory.any()


def test_same_feasible_domain_on_nonzero_fixed_neighbours_and_crossing_stencils():
    objective, poses = make_objective()
    # Nonzero current corrections on fixed neighbours must stay in the RHS.
    coords = objective.coords
    delta = np.zeros((len(poses), coords.width))
    delta[:, coords.inverse[coords.root_d]] = .001
    poses = coords.integrate(poses, delta, 1.)
    frames = range(2, 5)
    acc, r, j, _, _ = objective.evaluate(poses)
    system = banded_system(len(poses), coords.width, acc.hessian, j)
    full = prepare_model(objective, poses, system, acc.gradient.ravel() + j.T @ r,
                          np.ones(len(poses) * coords.width))
    local, _ = prepare_window_model(objective, poses, frames)
    rng = np.random.default_rng(22)
    outcomes = set()
    for scale in (0., 1e-8, 1e-5, .001, .1):
        for _ in range(8):
            step = rng.normal(size=local.gradient.size) * scale
            lifted = np.zeros_like(full.gradient)
            lifted[frames.start * coords.width:frames.stop * coords.width] = step
            f = full.program
            w = local.program
            # Row scaling/merging can differ; cone membership is invariant.
            full_ok = cone_violation(f.b - f.A @ lifted, f.nonneg, f.soc) < 1e-10
            window_ok = cone_violation(w.b - w.A @ step, w.nonneg, w.soc) < 1e-10
            assert window_ok == full_ok
            outcomes.add(full_ok)
    assert outcomes == {True, False}
