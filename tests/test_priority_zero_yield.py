"""V1 signs safe progress only, never optimality or permission to yield."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import runtime_options
from omniretargeting._fast.priority_descent import KIND
from omniretargeting._fast.qp_assembly import prepare_model
from tests.test_priority_stall import _case


def _problem(monkeypatch):
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(
        runtime_options.OPTIONS, priority_zero_yield_descent=True))
    guard, old_model, _, _ = _case()
    # q0 clears body1; q1 changes the contact. No additional yield is needed.
    guard.contacts[0].column = 1
    objective = guard.objective
    model = prepare_model(objective, guard.current, old_model.system, old_model.gradient, old_model.diagonal)
    return guard, model


def test_safe_descent_certificate_and_parent_commit_leave_yield_credit_unspent(monkeypatch):
    guard, model = _problem(monkeypatch)
    candidate = guard.current + [[.01, 0.]]
    certificate = guard.certify_zero_yield(candidate, model.predicted_slacks(np.array([.01, 0.])))
    assert certificate['kind'] == KIND and certificate['descent_certified']
    assert not certificate['optimality_certified'] and not certificate['convergence_certified']
    assert certificate['contact_metrics_checked'] == 1
    assert guard.epoch == 0 and not guard.spent and not guard.yields
    guard.certificate = certificate
    guard.commit(candidate)
    assert guard.epoch == 1 and not guard.spent and not guard.yields
    assert len(guard.descent_certificates) == 1
    assert guard.best_depth[0, 1] == pytest.approx(.04)


def test_v1_is_default_off_and_unknown_certificate_kind_is_never_accepted(monkeypatch):
    guard, _ = _problem(monkeypatch)
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, priority_zero_yield_descent=False))
    assert guard.certify_zero_yield(guard.current + [[.01, 0.]]) is None
    guard.certificate = dict(kind='unchecked_weighted', certified=True)
    with pytest.raises(ValueError, match='Unknown'):
        guard.commit(guard.current)


@pytest.mark.parametrize('failure', ['no_progress', 'actual_no_progress', 'unknown', 'contact',
                                    'hard_guard', 'joint_box', 'wrong_slacks'])
def test_no_false_certificate_for_failed_physical_or_model_check(monkeypatch, failure):
    guard, model = _problem(monkeypatch)
    candidate = guard.current + [[.01, 0.]]
    slacks = None
    if failure == 'no_progress':
        candidate = guard.current.copy()
    elif failure == 'actual_no_progress':
        guard.provider.actual_bias = -.01
    elif failure == 'unknown':
        guard.provider.fail = True
    elif failure == 'contact':
        candidate[0, 1] = .0002  # Even inside the old 1 mm cone, v1 forbids it.
    elif failure == 'hard_guard':
        guard.objective.candidate_guards.append(SimpleNamespace(accepts=lambda q: False))
    elif failure == 'joint_box':
        candidate[0, 0] = .13
    elif failure == 'wrong_slacks':
        slacks = model.predicted_slacks(np.array([.01, 0.]))
        slacks[next(iter(slacks))] += .01
    assert guard.certify_zero_yield(candidate, slacks) is None
    assert not guard.spent and guard.epoch == 0


def test_own_body_depth_allowance_does_not_release_v1_contact(monkeypatch):
    guard, _ = _problem(monkeypatch)
    contact = guard.contacts[0]
    original = contact.evaluate
    contact.evaluate = lambda q, **kw: [replace(b, own=np.full_like(b.own, .05))
                                       for b in original(q, **kw)]
    candidate = guard.current + [[.01, .002]]
    assert guard.evaluate_trial(candidate)['valid']  # Allowed only by old own-depth contract.
    assert guard.certify_zero_yield(candidate) is None


@pytest.mark.parametrize('change', ['pose', 'epoch', 'hash', 'version', 'claimed_optimality',
                                   'descent_false', 'contact', 'geometry', 'flag', 'primal'])
def test_commit_rechecks_certificate_and_current_state(monkeypatch, change):
    guard, _ = _problem(monkeypatch)
    candidate = guard.current + [[.01, 0.]]
    guard.certificate = guard.certify_zero_yield(candidate)
    assert guard.certificate
    if change == 'pose':
        candidate[0, 0] += .001
    elif change == 'epoch':
        guard.certificate['epoch'] += 1
    elif change == 'hash':
        guard.certificate['pose_sha256'] = 'invalid'
    elif change == 'version':
        guard.certificate['version'] += 1
    elif change == 'claimed_optimality':
        guard.certificate['optimality_certified'] = True
    elif change == 'descent_false':
        guard.certificate['descent_certified'] = False
    elif change == 'contact':
        guard.contacts[0].bias = .0002
    elif change == 'geometry':
        guard.provider.fail = True
    elif change == 'flag':
        monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, priority_zero_yield_descent=False))
    elif change == 'primal':
        guard.model.program.b[0] = -100.
    with pytest.raises(ValueError):
        guard.commit(candidate)
    assert guard.epoch == 0 and not guard.spent and not guard.yields


def test_existing_spending_history_and_initial_reference_survive_v1_commit_unchanged(monkeypatch):
    guard, _ = _problem(monkeypatch)
    guard.spent[('old_contact', 1)] = .004
    guard.yields.append(dict(contact='old_contact', gap_m=.004, iteration=-1))
    ledger = deepcopy((guard.spent, guard.yields))
    reference, initial = guard.reference.copy(), guard.initial_depth.copy()
    candidate = guard.current + [[.01, 0.]]
    guard.certificate = guard.certify_zero_yield(candidate)
    guard.commit(candidate)
    assert (guard.spent, guard.yields) == ledger
    np.testing.assert_array_equal(guard.reference, reference)
    np.testing.assert_array_equal(guard.initial_depth, initial)
    assert not guard.reference.flags.writeable and not guard.initial_depth.flags.writeable


def test_new_collision_key_cannot_hide_under_decreasing_same_body_maximum(monkeypatch):
    guard, _ = _problem(monkeypatch)
    original = guard.provider.measure
    key = (0, 'moving', 0, 'closed', 0, 0)  # Same geom/body as existing deeper static collision.
    def measure(q, **kwargs):
        result = original(q, **kwargs)
        if q[0, 0] > 0:
            result['depths'][key] = .002
        return result
    guard.provider.measure = measure
    candidate = guard.current + [[.01, 0.]]
    report = guard.evaluate_trial(candidate)['details']
    assert report['penetration_regressions'][0][0] == key
    assert not report['body_regressions']  # A body maximum alone would miss this.
    assert guard.certify_zero_yield(candidate) is None


def test_another_bodys_regression_cannot_be_paid_by_larger_total_progress(monkeypatch):
    guard, _ = _problem(monkeypatch)
    original = guard.provider.measure
    def measure(q, **kwargs):
        result = original(q, **kwargs)
        if q[0, 0] > 0:
            result['depths'][(0, 'static', 1)] = .002
        return result
    guard.provider.measure = measure
    candidate = guard.current + [[.01, 0.]]
    report = guard.evaluate_trial(candidate)['details']
    assert report['actual_reductions'][0] > 0
    assert (0, 2) in report['body_regressions']
    assert guard.certify_zero_yield(candidate) is None


def test_predicted_key_regression_is_rejected_even_if_actual_query_improves(monkeypatch):
    from omniretargeting._fast.descent_step import original_primal_certificate
    guard, _ = _problem(monkeypatch)
    # Two keys share a body; the new row is clear initially but predicts entry.
    key = (0, 'moving', 0, 'closed', 0, 0)
    measured = guard.current_measurement
    measured['rows'].append((key, dict(clearance=0., jacobian=np.array([-1., 0.]))))
    measured['depths'][key] = 0.
    guard.owners[key] = (1,)
    guard.matrix = sparse.vstack([guard.matrix, sparse.csr_matrix([[-1., 0.]])], format='csr')
    guard.phi = np.r_[guard.phi, 0.]
    # The unchanged original program alone passes; independent predicted-key
    # verification must catch this extra witness rather than use aggregate depth.
    candidate = guard.current + [[.01, 0.]]
    assert original_primal_certificate(guard.model, np.array([.01, 0.]))['primal_feasible']
    assert guard.evaluate_trial(candidate)['valid']
    assert guard.certify_zero_yield(candidate) is None


def test_full_original_primal_is_required_even_when_nonlinear_checks_pass(monkeypatch):
    from omniretargeting._fast.descent_step import original_primal_certificate
    guard, old_model = _problem(monkeypatch)
    linear_guard = SimpleNamespace(
        qp_bounds=lambda q: ([(sparse.csr_matrix([[1., 0.]]), np.array([-np.inf]), np.array([.005]))], []),
        accepts=lambda q: True)
    guard.objective.candidate_guards.append(linear_guard)
    model = prepare_model(guard.objective, guard.current, old_model.system,
                          old_model.gradient, old_model.diagonal)
    candidate = guard.current + [[.01, 0.]]
    assert guard.evaluate_trial(candidate)['valid'] and linear_guard.accepts(candidate)
    primal = original_primal_certificate(model, np.array([.01, 0.]))
    assert not primal['primal_feasible'] and primal['box_violation'] == 0
    assert guard.certify_zero_yield(candidate) is None


def test_missing_contact_metric_never_becomes_a_zero_yield_certificate(monkeypatch):
    guard, _ = _problem(monkeypatch)
    guard.contacts[0].evaluate = lambda q, **kwargs: []
    candidate = guard.current + [[.01, 0.]]
    assert guard.certify_zero_yield(candidate) is None


def test_zero_yield_rows_are_individual_channels_with_half_tolerance_reserve(monkeypatch):
    guard, _ = _problem(monkeypatch)
    linear, cones = guard.zero_yield_bounds(guard.current)
    assert not cones and len(linear) == 1
    matrix, lower, upper = linear[0]
    assert np.isneginf(lower).all()
    np.testing.assert_allclose(upper, [5e-8], rtol=0, atol=0)
    assert (matrix @ np.array([.01, .0002]))[0] > upper[0]


def test_full_norm_channel_at_zero_error_is_not_replaced_by_a_zero_gradient(monkeypatch):
    guard, old_model = _problem(monkeypatch)
    contact = guard.contacts[0]
    original = contact.evaluate
    def evaluate(q, **kwargs):
        block = original(q, **kwargs)[0]
        return [replace(block, scalar=False, residual=np.array([[0., q[0, 1], 0.]]),
                        matrix=sparse.csr_matrix([[0., 0.], [0., 1.], [0., 0.]]))]
    contact.evaluate = evaluate
    model = prepare_model(guard.objective, guard.current, old_model.system,
                          old_model.gradient, old_model.diagonal)
    linear, cones = guard.zero_yield_bounds(guard.current)
    assert not linear and len(cones) == 1 and cones[0][0].nnz == 1
    assert guard.certify_zero_yield(guard.current + [[.01, 0.]])
    assert guard.certify_zero_yield(guard.current + [[.01, -.0002]]) is None


def test_one_improving_contact_cannot_hide_a_regressing_point_in_same_shared_key(monkeypatch):
    guard, old_model = _problem(monkeypatch)
    contact = guard.contacts[0]
    original = contact.evaluate
    def evaluate(q, **kwargs):
        block = original(q, **kwargs)[0]
        return [replace(block, residual=np.array([q[0, 1], .02 - q[0, 1]]),
                        matrix=sparse.csr_matrix([[0., 1.], [0., -1.]]),
                        bound=np.array([.001, .001]), own=np.zeros(2),
                        frames=np.zeros(2, int), bodies=np.array([2, 2]), ranks=np.zeros(2, int),
                        identities=(block.identities[0], ('hold', 0, 0, 1, 'normal')))]
    contact.evaluate = evaluate
    guard._current_contacts = guard._contact_values(evaluate(guard.current))
    prepare_model(guard.objective, guard.current, old_model.system,
                  old_model.gradient, old_model.diagonal)
    candidate = guard.current + [[.01, .0002]]
    report = guard.evaluate_trial(candidate)
    assert report['valid'] and report['details']['actual_reductions'][1] > 0
    assert guard.certify_zero_yield(candidate) is None


def test_actual_priority_solver_accepts_zero_yield_descent_despite_tracking_increase(monkeypatch):
    from omniretargeting._fast.solver import optimize
    guard, _ = _problem(monkeypatch)
    objective = guard.objective
    objective.start_workers = lambda: False
    objective.project = lambda q: q
    objective.coords.integrate = lambda q, d, alpha=1.: q + alpha * d.reshape(q.shape)
    def evaluate(q, **kwargs):
        target = np.array([[-1., 0.]])
        acc = SimpleNamespace(cost=float(np.sum((q - target)**2)), gradient=q - target,
                              hessian=np.eye(2)[None])
        return acc, np.empty(0), sparse.csr_matrix((0, 2)), {}, None
    objective.evaluate = evaluate
    candidate, history = optimize(objective, guard.current, iterations=1)
    assert guard.epoch == 1 and not guard.spent and not guard.yields
    assert candidate[0, 0] > .01
    assert history[0]['after'] > history[0]['cost']
    certificate = history[0]['priority_certificate']
    assert certificate['kind'] == KIND and certificate['descent_certified']
    assert not certificate['optimality_certified'] and not certificate['certified']
