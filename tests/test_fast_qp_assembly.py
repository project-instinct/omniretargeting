"""Acquisition assembly equivalence and shared scalar/norm priority epigraphs."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.qp_assembly import SoftTerms, assemble_program, prepare_model, solve_model
from omniretargeting._fast.qp_guard_rows import intervals
from omniretargeting._fast.qp_types import IterationModel


def small_model(rows=(), cones=(), *, gradient=-1., lower=-1., upper=1.):
    soft = SoftTerms(1)
    for item in rows:
        soft.add(item)
    for item in cones:
        soft.add(item, cone=True)
    block = intervals(sparse.eye(1), np.array([lower]), np.array([upper]))
    program = assemble_program([block], soft, 1)
    return IterationModel(np.zeros((1, 1)), None, np.array([[lower]]), np.array([[upper]]),
                           np.ones((1, 1)), sparse.eye(1, format='csc'), np.array([gradient]),
                           np.ones(1), max(abs(gradient), 1.), program, soft)


def test_direct_acquisition_matches_cvxpy_with_both_continuity_guards():
    from tests.test_fast_backend import _objective
    from omniretargeting._fast.reference import ReferenceShape
    from omniretargeting._fast.continuity import CorrectionContinuity
    from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
    from omniretargeting._fast.shape_step import bounded_shape_step
    from omniretargeting._fast.solver import banded_system
    objective, poses, (tasks, _, _) = _objective(temporal=True)
    objective.source_shape = ReferenceShape(objective, poses)
    objective.candidate_guards = [CorrectionContinuity(objective.coords, poses, root_norm=True),
                                  SequenceContinuityGuard(tasks, poses, objective.root_rotations)]
    acc, residual, jacobian, _, _ = objective.evaluate(poses)
    gradient = acc.gradient.ravel() + jacobian.T @ residual
    system = banded_system(len(poses), objective.width, acc.hessian, jacobian)
    diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), 1e-9)
    direct = prepare_model(objective, poses, system, gradient, diagonal)
    baseline = bounded_shape_step(objective, poses, system, gradient, diagonal, .001)
    result = solve_model(direct, .001)
    assert result is not None and baseline is not None
    np.testing.assert_allclose(result.step, baseline, atol=2e-5, rtol=1e-4)
    assert direct.tracking_decrease(result.step) == pytest.approx(
        direct.tracking_decrease(baseline), rel=2e-6, abs=1e-7)


def test_shared_witness_slack_is_charged_once():
    key = ('penetration', 0, 1)
    one = (sparse.csr_matrix([[-1.]]), np.array([-.2]), np.array([3.]), [key])
    many = (sparse.csr_matrix([[-1.], [-1.], [-1.]]), np.full(3, -.2), np.full(3, 3.), [key] * 3)
    a, b = small_model([one], gradient=1.), small_model([many], gradient=1.)
    sa, sb = solve_model(a, .001), solve_model(b, .001)
    assert len(b.soft.keys) == 1
    np.testing.assert_allclose(sa.step, sb.step, atol=2e-7)
    assert sa.slacks[key] == pytest.approx(sb.slacks[key], abs=1e-7)


def test_hierarchy_with_positive_unavoidable_penetration_uses_strict_fallback():
    penetration, contact = ('penetration', 0, 1), ('contact', 'support', 0, 1)
    rows = [(sparse.csr_matrix([[-1.], [1.]]), np.array([-.2, 0.]), np.ones(2),
             [penetration, contact])]
    model = small_model(rows, gradient=100., upper=.1)
    result = solve_model(model, .001, priority=True)
    assert result is not None
    assert result.certificate['certified']
    assert result.certificate['method'] == 'lexicographic'
    assert result.step[0] == pytest.approx(.1, abs=1e-6)
    assert result.slacks[penetration] == pytest.approx(.1, abs=1e-6)
    assert result.slacks[contact] == pytest.approx(.1, abs=1e-6)
    assert np.all(np.asarray(result.certificate['coefficients']) >
                  np.asarray(result.certificate['multiplier_bounds']))


def test_certified_level_cache_reuses_only_the_same_program_domain(monkeypatch):
    import omniretargeting._fast.priority_qp as priority
    from omniretargeting._fast.qp_reduction import restrict_model
    from omniretargeting._fast.qp_assembly import quadratic
    key = ('penetration', 0, 1)
    model = small_model([(sparse.csr_matrix([[-1.]]), np.array([-.2]), np.ones(1), [key])],
                        gradient=100., upper=.1)
    first = solve_model(model, .001, priority=True)
    assert first is not None and model.priority_cache is not None
    original = priority.solve_program
    calls = []
    def counted(*args):
        calls.append(args[0])
        return original(*args)
    monkeypatch.setattr(priority, 'solve_program', counted)
    second = solve_model(model, 1., priority=True)
    assert second is not None and second.certificate['reused_level_optima']
    assert len(calls) == 1
    P, c = quadratic(model, .001)
    restricted, P, c = restrict_model(model, P, c, sparse.eye(1), np.zeros(1),
                                      lower=np.array([-.05]), upper=np.array([.05]))
    assert restricted.priority_cache is None
    calls.clear()
    small = priority.solve_priority(restricted, P, c)
    assert small is not None and not small.certificate['reused_level_optima']
    assert len(calls) > 1
    assert small.step[0] == pytest.approx(.05, abs=1e-6)


def test_exact_soft_soc_and_scalar_share_epigraph():
    key = ('contact', 'support', 0, 1)
    cone = (sparse.csr_matrix([[1.], [0.], [0.]]), np.array([[0., 1., 0.]]),
            np.array([.5]), np.array([1.]), [key])
    row = (sparse.csr_matrix([[0.]]), np.array([-.25]), np.array([1.]), [key])
    model = small_model([row], [cone])
    assert model.program.soc == (4,)
    assert len(model.soft.keys) == 1
    assert model.predicted_slacks(np.array([1.]))[key] == pytest.approx(np.sqrt(2) - .5)
    result = solve_model(model, .001, priority=True)
    assert result is not None and result.certificate['certified']
    assert abs(result.step[0]) < .001
    assert result.slacks[key] == pytest.approx(.5, abs=1e-6)


def test_inconsistent_duplicate_weight_is_an_error():
    key = ('penetration', 0, 1)
    model = SoftTerms(1)
    with pytest.raises(ValueError, match='Inconsistent weight'):
        model.add((sparse.eye(2, 1), np.ones(2), np.array([1., 2.]), [key, key]))


def test_priority_begin_precedes_rows_and_exposes_original_model():
    from tests.test_fast_local_restoration import _Coordinates
    events = []
    class Coordinator:
        def begin_iteration(self, poses, model):
            events.append('begin')
            assert model.soft is None
            assert model.step_upper.shape == poses.shape
        def soft_rows(self, poses):
            events.append('rows')
            return []
    poses = np.zeros((2, 2))
    objective = SimpleNamespace(coords=_Coordinates(), cfg=dict(angular_step_rad=.2, translation_step_m=.1),
                                lower=np.full(2, -1), upper=np.ones(2), priority_clearance=Coordinator(),
                                candidate_guards=[], source_shape=None)
    prepare_model(objective, poses, np.ones((1, 4)), np.ones(4), np.ones(4))
    assert events == ['begin', 'rows']


def test_seed_continuity_adapter_adds_no_reference_shape_rows():
    from omniretargeting._fast.qp_guard_rows import _shape
    adapter = SimpleNamespace(continuity_only=True)
    assert _shape(adapter, np.zeros((3, 2)), np.ones((3, 2))) == []
