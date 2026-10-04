"""Only identical failed searches may be skipped; no infeasibility inference."""

from copy import deepcopy

import numpy as np
import pytest

from omniretargeting._fast.acquisition_clearance import _termination
from omniretargeting._fast.qp_assembly import prepare_model
from tests.test_priority_clearance import _priority


def _case():
    guard, _, _ = _priority()
    objective = guard.objective
    objective.priority_clearance = guard
    objective.coords.qi = np.arange(2)
    objective.coords.scalar_cols = np.arange(2)
    objective.coords.bounds = lambda q, a, t: (-np.full_like(q, .12), np.full_like(q, .12))
    objective.coords.root_q = None
    objective.candidate_guards = [guard]
    objective.source_shape = None
    objective.lower, objective.upper = -np.ones(2), np.ones(2)
    objective.cfg = dict(angular_step_rad=.12, translation_step_m=.12)
    model = prepare_model(objective, guard.current, np.ones((1, 2)), np.ones(2), np.ones(2))
    policy = dict(damping=.001, growth=2., backend='clarabel', alphas=(1., .5), tolerance=1e-8)
    record = dict(phase='contact_scene', iteration=0, priority_status='no_certified_accepted_proposal')
    return guard, model, policy, record


def test_same_problem_skip_is_not_a_feasibility_or_convergence_certificate():
    guard, model, policy, record = _case()
    assert guard.repeated_stall(model, policy) is None
    guard.record_stall(model, policy, record)
    hit = guard.repeated_stall(model, deepcopy(policy))
    assert hit['priority_status'] == 'repeated_stalled_problem'
    assert hit['previous_phase'] == 'contact_scene'
    assert not hit['hard_infeasibility_certified'] and not hit['convergence_certified']
    assert guard.epoch == 0 and guard.spent == {}


@pytest.mark.parametrize('change', [
    'gradient', 'hessian', 'diagonal', 'step_upper', 'program_A', 'program_b',
    'soft_weight', 'policy', 'cfg', 'witness', 'depth', 'evidence', 'best_keys',
    'epoch', 'spent', 'guard_identity', 'contact_identity', 'unsupported_policy',
])
def test_any_problem_or_search_change_invalidates_the_previous_stall(change):
    guard, model, policy, record = _case()
    guard.record_stall(model, policy, record)
    assert guard._stalled_search.saved is not None
    if change in ('gradient', 'diagonal', 'step_upper'):
        array = getattr(model, change)
        array.flat[0] = np.nextafter(array.flat[0], np.inf)
    elif change == 'hessian':
        model.hessian.data[0] += .00001
    elif change == 'program_A':
        model.program.A.data[0] += .00001
    elif change == 'program_b':
        model.program.b[0] += .00001
    elif change == 'soft_weight':
        model.soft.weights[0] += .00001
    elif change == 'policy':
        policy['damping'] *= 2
    elif change == 'cfg':
        guard.objective.cfg['angular_step_rad'] += .00001
    elif change == 'witness':
        guard.current_measurement['rows'][0][1]['point'][0] += .00001
    elif change == 'depth':
        guard.current_measurement['depths'][(0, 'static', 0)] += .00001
    elif change == 'evidence':
        guard.current_measurement['evidence_id'] = 'changed'
    elif change == 'best_keys':
        guard.best_keys[(0, 'static', 0)] -= .00001
    elif change == 'epoch':
        guard.epoch += 1
    elif change == 'spent':
        guard.spent[('test', 1)] = .001
    elif change == 'guard_identity':
        guard.objective.candidate_guards.append(object())
    elif change == 'contact_identity':
        guard.contacts.append(object())
    elif change == 'unsupported_policy':
        policy['opaque'] = object()
    assert guard.repeated_stall(model, policy) is None
    assert guard._stalled_search.saved is None


def test_commit_invalidates_and_nonstall_outcomes_are_never_cached():
    guard, model, policy, record = _case()
    guard.record_stall(model, policy, record)
    guard.commit(guard.current.copy())
    assert guard._stalled_search.saved is None
    guard.record_stall(model, policy, dict(record, priority_status='iteration_limit'))
    assert guard.repeated_stall(model, policy) is None


def test_unknown_signature_does_not_skip_a_search():
    guard, model, policy, record = _case()
    policy['unknown'] = object()
    guard.record_stall(model, policy, record)
    assert guard.repeated_stall(model, policy) is None


def test_actual_solver_and_coordinator_skip_the_second_identical_failed_search(monkeypatch):
    from types import SimpleNamespace
    from scipy import sparse
    import omniretargeting._fast.prioritized_solver as solver
    from omniretargeting._fast.solver import optimize, MAX_SHAPE_SOLVES
    guard, _, _, _ = _case()
    objective = guard.objective
    objective.start_workers = lambda: False
    def evaluate(q, **kwargs):
        acc = SimpleNamespace(cost=1., gradient=np.ones((1, 2)), hessian=np.eye(2)[None])
        return acc, np.empty(0), sparse.csr_matrix((0, 2)), {}, None
    objective.evaluate = evaluate
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        return None
    monkeypatch.setattr(solver, 'solve_model', fail)
    _, first = optimize(objective, guard.current, iterations=35, label='contact_scene')
    assert first[-1]['priority_status'] == 'no_certified_accepted_proposal'
    assert len(calls) == MAX_SHAPE_SOLVES
    assert guard._stalled_search.saved is not None
    _, second = optimize(objective, guard.current, iterations=35, label='priority_clearance_completion')
    assert second[-1]['priority_status'] == 'repeated_stalled_problem'
    assert len(calls) == MAX_SHAPE_SOLVES
    assert guard.epoch == 0 and not guard.spent


def test_completion_reports_unresolved_search_instead_of_infeasibility():
    report = dict(status='residual')
    _termination(report, [dict(priority_status='repeated_stalled_problem')])
    assert report['residuals_unresolved'] and report['status'] == 'residual'
    assert not report['hard_infeasibility_certified']
