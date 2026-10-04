"""Priority hooks, tracking-increase acceptance, and transactional proposals."""

import multiprocessing
import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.priority_merit import assess, check_trial
from omniretargeting._fast.qp_types import PriorityMerit
from omniretargeting._fast.solver import optimize
from tests.test_fast_qp_globalization import objective_for


class Coordinator:
    key = ('penetration', 0, 1)
    def __init__(self, poses, *, certificate=False):
        self.current = poses.copy()
        self.events = []
        self.commits = 0
        self.owner = os.getpid()
        self.certificate = None
        self.require_certificate = certificate
    def begin_iteration(self, poses, model):
        np.testing.assert_array_equal(poses, self.current)
        self.events.append('begin')
        self.model = model
    def soft_rows(self, poses):
        self.events.append('rows')
        return [(sparse.csr_matrix([[-1.]]), np.array([poses[0, 0] - .2]), np.ones(1), [self.key])]
    def evaluate_trial(self, poses):
        self.events.append('trial')
        depth = max(.2 - poses[0, 0], 0.)
        return dict(valid=True, levels=(depth, 0., 0., 0.), tolerances=(1e-7,) * 4,
                    slacks={self.key: depth} if depth else {},
                    details=dict(requires_priority_certificate=self.require_certificate))
    def commit(self, poses):
        assert os.getpid() == self.owner
        assert self.certificate['certified']
        self.commits += 1
        self.events.append('commit')
        self.current = poses.copy()


def problem(*, nonlinear_guard=False, certificate=False):
    target, poses = np.array([[-1.]]), np.zeros((1, 1))
    guards = []
    if nonlinear_guard:
        class Guard:
            def qp_bounds(self, poses):
                return [], []
            def accepts(self, poses):
                return poses[0, 0] <= .05
        guards.append(Guard())
    objective = objective_for(target, guards, step=.5)
    def evaluate(q, **kwargs):
        acc = SimpleNamespace(cost=float(np.sum((q - target)**2)), gradient=q - target,
                              hessian=np.ones((1, 1, 1)))
        return acc, np.empty(0), sparse.csr_matrix((0, 1)), {}, None
    objective.evaluate = evaluate
    objective.start_workers = lambda: False
    objective.priority_clearance = Coordinator(poses, certificate=certificate)
    return objective, poses


def test_higher_priority_gain_accepts_tracking_cost_increase_and_commits_once():
    objective, poses = problem()
    result, history = optimize(objective, poses, iterations=1)
    coordinator = objective.priority_clearance
    assert coordinator.commits == 1
    assert result[0, 0] > .19999
    assert history[0]['after'] > history[0]['cost']
    assert history[0]['improved_level'] == 0
    assert coordinator.events[:2] == ['begin', 'rows']
    assert coordinator.events[-1] == 'commit'


def test_rejected_trials_do_not_commit_and_shortened_yield_is_recertified():
    objective, poses = problem(nonlinear_guard=True, certificate=True)
    result, history = optimize(objective, poses, iterations=1)
    coordinator = objective.priority_clearance
    assert coordinator.commits == 1
    assert 0 < result[0, 0] <= .05
    assert coordinator.events.count('trial') > 1
    assert history[0]['priority_certificate']['certified']
    assert history[0]['priority_certificate']['domain'] == 'affine_restriction'
    assert history[0]['alpha'] == 1.


def test_forked_trial_evaluation_cannot_commit_parent_ledger():
    if not hasattr(os, 'fork'):
        pytest.skip('fork unavailable')
    objective, poses = problem()
    context = multiprocessing.get_context('fork')
    parent, child = context.Pipe(duplex=False)
    def run():
        accepted = check_trial(objective, np.array([[.1]]), [])
        child.send((accepted, objective.priority_clearance.commits))
        child.close()
    worker = context.Process(target=run)
    worker.start()
    child.close()
    assert parent.recv() == (True, 0)
    parent.close()
    worker.join()
    assert worker.exitcode == 0
    assert objective.priority_clearance.commits == 0
    assert objective.priority_clearance.events == []


def test_invalid_priority_merit_cannot_be_offset_by_tracking_improvement():
    from tests.test_fast_qp_assembly import small_model
    key = ('penetration', 0, 1)
    model = small_model([(sparse.csr_matrix([[-1.]]), np.array([-.2]), np.ones(1), [key])])
    before = PriorityMerit(True, (.2, 0, 0, 0), (1e-7,) * 4, {key: .2})
    after = PriorityMerit(False, (.3, 0, 0, 0), (1e-7,) * 4, {key: .3})
    accepted, _, _ = assess(model, np.array([1.]), before, after, 10., 0., dict(coefficients=[1.] * 4))
    assert not accepted


def test_l1_predicted_and_actual_decreases_use_same_coefficients():
    from tests.test_fast_qp_assembly import small_model
    key = ('penetration', 0, 1)
    model = small_model([(sparse.csr_matrix([[-1.]]), np.array([-.2]), np.ones(1), [key])])
    before = PriorityMerit(True, (.2, 0, 0, 0), (1.,) * 4, {key: .2})
    after = PriorityMerit(True, (.1, 0, 0, 0), (1.,) * 4, {key: .1})
    accepted, ratio, detail = assess(model, np.array([.1]), before, after, 1., .81,
                                     dict(coefficients=[3.] * 4))
    assert accepted and ratio == pytest.approx(1.)
    assert detail['predicted_merit_decrease'] == pytest.approx(.49)
    assert detail['actual_merit_decrease'] == pytest.approx(.49)


def test_stall_policy_uses_entry_damping_and_repeated_failure_skips_solves(monkeypatch):
    import omniretargeting._fast.prioritized_solver as module
    from omniretargeting._fast.solver import INITIAL_DAMPING, MAX_SHAPE_SOLVES
    objective, poses = problem()
    coordinator = objective.priority_clearance
    recorded, checked, solves = [], [], []
    def repeated(model, policy):
        checked.append(policy)
        return None if not recorded else dict(reason='same_exact_model')
    def record(model, policy, value):
        recorded.append((policy, value.copy()))
    coordinator.repeated_stall = repeated
    coordinator.record_stall = record
    def fail(*args, **kwargs):
        solves.append(1)
        return None
    monkeypatch.setattr(module, 'solve_model', fail)
    _, first = optimize(objective, poses, iterations=1)
    assert len(solves) == MAX_SHAPE_SOLVES
    assert recorded[0][0]['initial_damping'] == INITIAL_DAMPING
    assert recorded[0][0]['initial_growth'] == 2.
    assert first[-1]['damping'] > INITIAL_DAMPING
    _, second = optimize(objective, poses, iterations=1)
    assert len(solves) == MAX_SHAPE_SOLVES
    assert second[-1]['priority_status'] == 'repeated_stalled_problem'
    assert len(recorded) == 1 and coordinator.commits == 0
    assert checked[0] == checked[1]


def test_accepted_step_does_not_record_a_stall():
    objective, poses = problem()
    coordinator = objective.priority_clearance
    coordinator.repeated_stall = lambda model, policy: None
    def unexpected(*args):
        raise AssertionError('Only a terminal rejected iteration may be cached')
    coordinator.record_stall = unexpected
    result, _ = optimize(objective, poses, iterations=1)
    assert result[0, 0] > .19 and coordinator.commits == 1
