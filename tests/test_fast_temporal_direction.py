"""The banded temporal proposal remains subordinate to exact guards."""

import numpy as np
from dataclasses import replace

import omniretargeting._fast.runtime_options as runtime_options
import omniretargeting._fast.shape_step as shape_step
import omniretargeting._fast.temporal_direction as temporal_direction
from omniretargeting._fast.solver import _conic_proposal
from omniretargeting._fast.solver import banded_system
from omniretargeting._fast.temporal_direction import (
    TaggedDirection, _temporal_linear_rows, temporal_budget_direction,
)
from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from omniretargeting._fast.objective import FastObjective
from tests.test_fast_backend import _problem


def test_temporal_direction_contains_every_scalar_budget_and_falls_back_cleanly():
    model, cfg, coords, poses, task_set, rotations = _problem()
    objective = FastObjective(
        task_set, cfg, reference=poses, temporal_reference=poses[:, coords.qi],
        root_rotations=rotations, framerate=30.0,
    )
    objective.candidate_guards = [SequenceContinuityGuard(task_set, poses, rotations)]
    matrix, bounds = _temporal_linear_rows(objective, poses)
    assert matrix.shape[0] > objective.frames * objective.coords.scalar_cols.size
    assert np.isfinite(bounds).all()

    acc, residual, temporal, _, _ = objective.evaluate(poses)
    gradient = acc.gradient.ravel() + temporal.T @ residual
    system = banded_system(objective.frames, objective.width, acc.hessian, temporal)
    diagonal = np.diagonal(acc.hessian, axis1=1, axis2=2).ravel()
    diagnostics = {}
    proposal = temporal_budget_direction(
        objective, poses, system, gradient, diagonal, 1e-3, diagnostics=diagnostics)
    if proposal is None:
        assert diagnostics["fallback_reason"] == "temporal_direction_guard_rejected"
    else:
        step, candidate, alpha = proposal
        assert isinstance(step, TaggedDirection)
        assert step.proposal_backend == "banded_temporal"
        assert alpha >= .25
        assert all(guard.accepts(candidate) for guard in objective.candidate_guards)
        assert diagnostics["temporal_rows"] == matrix.shape[0]


def test_conic_proposal_uses_opt_in_banded_temporal_backend(monkeypatch):
    model, cfg, coords, poses, task_set, rotations = _problem()
    objective = FastObjective(
        task_set, cfg, reference=poses, temporal_reference=poses[:, coords.qi],
        root_rotations=rotations, framerate=30.0,
    )
    objective.candidate_guards = [SequenceContinuityGuard(task_set, poses, rotations)]
    acc, residual, temporal, _, _ = objective.evaluate(poses)
    gradient = acc.gradient.ravel() + temporal.T @ residual
    system = banded_system(objective.frames, objective.width, acc.hessian, temporal)
    diagonal = np.diagonal(acc.hessian, axis1=1, axis2=2).ravel()
    monkeypatch.setattr(runtime_options, "OPTIONS", replace(
        runtime_options.OPTIONS, banded_temporal_direction=True))
    cheap = np.zeros(len(gradient)).view(TaggedDirection)
    cheap.proposal_backend = "banded_temporal"
    monkeypatch.setattr(
        temporal_direction, "temporal_budget_direction",
        lambda *args, **kwargs: (cheap, objective.project(poses), 1.0))
    def forbidden(*args, **kwargs):
        raise AssertionError("cheap proposal must precede the full conic solve")
    monkeypatch.setattr(shape_step, "bounded_shape_step", forbidden)
    proposal = _conic_proposal(
        objective, poses, system, gradient, diagonal, 1e-3, objective.candidate_guards)
    assert proposal is not None
    assert getattr(proposal[0], "proposal_backend", None) == "banded_temporal"


def test_every_guard_variant_adds_the_other_frozen_rows():
    model, cfg, coords, poses, task_set, rotations = _problem()
    objective = FastObjective(
        task_set, cfg, reference=poses, temporal_reference=poses[:, coords.qi],
        root_rotations=rotations, framerate=30.0,
    )
    temporal_guard = SequenceContinuityGuard(task_set, poses, rotations)

    class Band:
        """A non-temporal frozen guard: one two-sided linear row per frame."""
        def qp_bounds(self, _poses):
            from scipy import sparse
            matrix = sparse.eye(objective.frames * objective.width, format="csr")[::objective.width]
            return [(matrix, -np.ones(objective.frames), np.ones(objective.frames))], []

        def accepts(self, _poses):
            return True

    objective.candidate_guards = [temporal_guard, Band()]
    temporal, _ = temporal_direction._temporal_rows(objective, poses)
    every, _ = temporal_direction._temporal_rows(objective, poses, every_guard=True)
    assert len(every) == len(temporal) + 1

    class Opaque:
        def accepts(self, _poses):
            return True

    objective.candidate_guards = [temporal_guard, Opaque()]
    assert temporal_direction._temporal_rows(objective, poses, every_guard=True) is None
    acc, residual, jacobian, _, _ = objective.evaluate(poses)
    gradient = acc.gradient.ravel() + jacobian.T @ residual
    system = banded_system(objective.frames, objective.width, acc.hessian, jacobian)
    diagonal = np.diagonal(acc.hessian, axis1=1, axis2=2).ravel()
    diagnostics = {}
    assert temporal_budget_direction(objective, poses, system, gradient, diagonal, 1e-3,
                                     diagnostics=diagnostics, every_guard=True) is None
    assert diagnostics["fallback_reason"] == "guard_without_qp_bounds"


def test_adaptive_schedule_stops_after_one_failed_cheap_direction(monkeypatch):
    model, cfg, coords, poses, task_set, rotations = _problem()
    objective = FastObjective(
        task_set, cfg, reference=poses, temporal_reference=poses[:, coords.qi],
        root_rotations=rotations, framerate=30.0,
    )
    objective.candidate_guards = [SequenceContinuityGuard(task_set, poses, rotations)]
    acc, residual, temporal, _, _ = objective.evaluate(poses)
    gradient = acc.gradient.ravel() + temporal.T @ residual
    system = banded_system(objective.frames, objective.width, acc.hessian, temporal)
    diagonal = np.diagonal(acc.hessian, axis1=1, axis2=2).ravel()
    monkeypatch.setattr(runtime_options, "OPTIONS", replace(
        runtime_options.OPTIONS, banded_temporal_direction=True, adaptive_banded_direction=True))
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        kwargs["diagnostics"]["fallback_reason"] = "temporal_direction_guard_rejected"
        return None

    monkeypatch.setattr(temporal_direction, "temporal_budget_direction", failing)
    monkeypatch.setattr(shape_step, "bounded_shape_step", lambda *args, **kwargs: None)
    for _ in range(3):
        _conic_proposal(objective, poses, system, gradient, diagonal, 1e-3, objective.candidate_guards)
    assert len(calls) == 1
