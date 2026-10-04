"""The opt-in descent route cannot claim strict optimality or skip full checks."""

from dataclasses import replace

import numpy as np
from scipy import sparse

from omniretargeting._fast.descent_step import original_primal_certificate
from omniretargeting._fast.solver import optimize
from tests.test_fast_priority_solver import Coordinator, problem
from tests.test_fast_qp_assembly import small_model


class DescentCoordinator(Coordinator):
    def certify_zero_yield(self, poses, predicted_slacks=None):
        self.events.append('descent_check')
        delta = poses.ravel() - self.current.ravel()
        check = original_primal_certificate(self.model, delta)
        before, after = self.evaluate_trial(self.current), self.evaluate_trial(poses)
        if not check['primal_feasible'] or before['levels'][0] - after['levels'][0] <= 1e-7:
            return None
        return dict(kind='zero_additional_contact_yield_descent', descent_certified=True,
                    optimality_certified=False, zero_additional_yield=True)

    def commit(self, poses):
        assert self.certificate['descent_certified']
        assert not self.certificate['certified']
        assert not self.certificate['optimality_certified']
        self.commits += 1
        self.current = poses.copy()


def _enable(monkeypatch):
    from omniretargeting._fast import runtime_options, priority_qp
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS,
                        priority_zero_yield_descent=True))
    def forbidden(*args, **kwargs):
        raise AssertionError('Descent must not first spend solves on strict hierarchy')
    monkeypatch.setattr(priority_qp, 'solve_priority', forbidden)


def test_descent_accepts_tracking_increase_without_strict_optimality_claim(monkeypatch):
    _enable(monkeypatch)
    objective, poses = problem()
    objective.priority_clearance = DescentCoordinator(poses)
    output, history = optimize(objective, poses, iterations=1)
    assert output[0, 0] > .19 and objective.priority_clearance.commits == 1
    record = history[-1]
    assert record['after'] > record['cost']
    assert record['step_backend'] == 'priority_zero_yield_descent'
    assert record['priority_certificate']['kind'] == 'zero_additional_contact_yield_descent'
    assert not record['priority_certificate']['optimality_certified']
    assert not record['priority_certificate']['certified']


def test_shortened_descent_is_certified_at_the_executed_trial(monkeypatch):
    _enable(monkeypatch)
    objective, poses = problem(nonlinear_guard=True)
    objective.priority_clearance = DescentCoordinator(poses)
    output, history = optimize(objective, poses, iterations=1)
    assert 0 < output[0, 0] <= .05
    assert history[-1]['alpha'] < 1.
    assert objective.priority_clearance.commits == 1
    assert 'domain' not in history[-1]['priority_certificate']


def test_uncertified_trial_never_commits_or_claims_infeasibility(monkeypatch):
    _enable(monkeypatch)
    objective, poses = problem()
    coordinator = objective.priority_clearance = DescentCoordinator(poses)
    coordinator.certify_zero_yield = lambda *args, **kwargs: None
    output, history = optimize(objective, poses, iterations=1)
    np.testing.assert_array_equal(output, poses)
    assert coordinator.commits == 0
    assert history[-1]['priority_status'] == 'no_certified_accepted_proposal'


def test_original_primal_checks_boxes_and_constant_full_cones():
    from omniretargeting._fast.qp_guard_rows import norms
    from omniretargeting._fast.qp_assembly import assemble_program
    model = small_model()
    assert original_primal_certificate(model, np.array([.5]))['primal_feasible']
    assert not original_primal_certificate(model, np.array([1.01]))['primal_feasible']
    cone = norms(sparse.csr_matrix((3, 1)), np.array([[2., 0., 0.]]), np.array([1.]))
    model.program = assemble_program([cone], model.soft, 1)
    assert not original_primal_certificate(model, np.zeros(1))['primal_feasible']
    assert not original_primal_certificate(model, np.array([np.nan]))['primal_feasible']
