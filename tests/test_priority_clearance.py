"""Transactional, explicitly funded contact yield under original geometry."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.priority_clearance import PriorityClearance
from omniretargeting._fast.priority_contacts import ContactRows
from omniretargeting._fast.priority_geometry import body_depths


class _Provider:
    def __init__(self):
        self.fail = False
        self.duplicate = False
        self.actual_bias = 0.

    def measure(self, q, **kwargs):
        phi = -.05 + q[0, 0] + self.actual_bias
        row = {'clearance': phi, 'jacobian': np.array([1., 0.]),
               'point': np.zeros(3), 'direction': np.array([1., 0., 0.]), 'geom_id': 0}
        rows = [((0, 'static', 0), row)] * (2 if self.duplicate else 1)
        return {'rows': rows, 'depths': {(0, 'static', 0): max(-phi, 0.)},
                'failures': ['unknown'] if self.fail else [], 'complete': not self.fail}

    def linear_rows(self, measured):
        return sparse.csr_matrix([r['jacobian'] for _, r in measured['rows']]), np.array([
            r['clearance'] for _, r in measured['rows']])


class _Contact:
    def __init__(self, coupled=True):
        self.column = 0 if coupled else 1
        self.bias = 0.

    def evaluate(self, q, *, jacobian=True):
        row = np.zeros((1, 2))
        row[0, self.column] = 1.
        return [ContactRows(sparse.csr_matrix(row) if jacobian else None,
                            np.array([q[0, self.column] + self.bias]), np.array([.001]),
                            np.zeros(1), np.zeros(1, int), np.array([2]), np.zeros(1, int),
                            (('hold', 0, 0, 0, 'normal'),), True)]


def _priority(coupled=True):
    model = SimpleNamespace(nbody=3, geom_bodyid=np.array([1, 2]), body_weldid=np.arange(3))
    objective = SimpleNamespace(model=model, frames=1, width=2,
                                coords=SimpleNamespace(difference=lambda a, b: b - a))
    provider = _Provider()
    guard = PriorityClearance(objective, provider, np.zeros((1, 2)))
    contact = _Contact(coupled)
    guard.contacts.append(contact)
    guard._current_contacts.update(guard._contact_values(contact.evaluate(guard.current)))
    guard.begin_iteration(guard.current, {})
    return guard, provider, contact


def test_cross_body_yield_has_same_step_donor_and_commit_is_transactional():
    guard, _, _ = _priority()
    q = np.array([[.01, 0.]])
    before = guard.best_depth.copy()
    report = guard.evaluate_trial(q)['details']
    assert report['accepted'] and report['requires_priority_certificate']
    assert report['yields'][0]['body'] == 2
    assert report['yields'][0]['donor_body'] == 1
    assert report['yields'][0]['gap_m'] == pytest.approx(.009)
    assert report['yields'][0]['measured_reduction_m'] == pytest.approx(.01)
    assert guard.spent == {} and guard.epoch == 0
    np.testing.assert_array_equal(guard.best_depth, before)
    with pytest.raises(ValueError, match='certified'):
        guard.commit(q)
    guard.certificate = {'certified': True}
    guard.commit(q)
    assert guard.epoch == 1 and len(guard.yields) == 1
    assert sum(guard.spent.values()) == pytest.approx(.009)


def test_tracking_cannot_buy_a_contact_yield_without_penetration_descent():
    guard, _, contact = _priority()
    contact.bias = .004
    report = guard.evaluate_trial(np.zeros((1, 2)))['details']
    assert not report['accepted']
    assert report['reason'] == 'unfunded_contact_yield'
    assert not guard.spent


def test_unrelated_body_cannot_borrow_penetration_credit():
    guard, _, _ = _priority(coupled=False)
    report = guard.evaluate_trial(np.array([[.01, .01]]))['details']
    assert not report['accepted']
    assert report['unfunded_contacts'][0]['available_m'] == 0.


def test_actual_reduction_cannot_be_replaced_by_optimistic_prediction():
    guard, provider, _ = _priority()
    provider.actual_bias = -.009
    report = guard.evaluate_trial(np.array([[.01, 0.]]))['details']
    assert not report['accepted']
    assert report['unfunded_contacts'][0]['available_m'] == pytest.approx(.001)


def test_predicted_reduction_caps_credit_even_when_query_improves_more():
    guard, provider, contact = _priority()
    provider.actual_bias = .009
    contact.bias = .008
    report = guard.evaluate_trial(np.array([[.002, 0.]]))['details']
    assert not report['accepted']
    assert report['unfunded_contacts'][0]['available_m'] == pytest.approx(.002)


def test_penetration_roundoff_recovery_does_not_mint_new_credit():
    guard, _, _ = _priority()
    q = np.array([[.01, 0.]])
    guard.certificate = {'certified': True}
    guard.commit(q)
    guard.begin_iteration(guard.current)
    # A tolerated regression is not permission to spend its recovery again.
    regressed = q - [[.00004, 0.]]
    assert guard.accepts(regressed)
    guard.commit(regressed)
    guard.begin_iteration(guard.current)
    report = guard.evaluate_trial(q)['details']
    assert not report['accepted']
    assert report['unfunded_contacts'][0]['available_m'] == 0.


def test_unknown_or_incomplete_geometry_never_releases_contacts():
    guard, provider, _ = _priority()
    provider.fail = True
    report = guard.evaluate_trial(np.array([[.01, 0.]]))['details']
    assert not report['accepted'] and not report['reliable']
    assert guard.spent == {}


def test_witness_duplicates_share_the_same_penetration_epigraph_key():
    guard, provider, _ = _priority()
    provider.duplicate = True
    guard.begin_iteration(guard.current)
    matrix, upper, weight, keys = guard.soft_rows(guard.current)[0]
    assert matrix.shape == (2, 2)
    assert keys == (('penetration', 0, 1), ('penetration', 0, 1))
    np.testing.assert_array_equal(upper, [-.05, -.05])
    np.testing.assert_array_equal(weight, [1., 1.])


def test_new_collision_rejection_cannot_be_hidden_by_another_depth_decrease():
    guard, provider, _ = _priority()
    original = provider.measure

    def measured(q, **kwargs):
        result = original(q, **kwargs)
        result['depths'][(0, 'moving', 0, 'closed', 0, 1)] = .002
        return result

    provider.measure = measured
    report = guard.evaluate_trial(np.array([[.01, 0.]]))['details']
    assert not report['accepted'] and report['penetration_regressions']


def test_self_collision_depth_belongs_to_both_actual_bodies():
    model = SimpleNamespace(nbody=3, geom_bodyid=np.array([1, 2]), body_weldid=np.arange(3))
    depths, owners = body_depths(model, 1, {'depths': {(0, 'self', 0, 1): .03},
                                          'complete': True, 'failures': []})
    np.testing.assert_array_equal(depths, [[0., .03, .03]])
    assert owners[(0, 'self', 0, 1)] == (1, 2)


def test_speculative_worker_cannot_commit(monkeypatch):
    guard, _, _ = _priority()
    monkeypatch.setattr('omniretargeting._fast.priority_clearance.os.getpid', lambda: -1)
    with pytest.raises(RuntimeError, match='speculative'):
        guard.commit(guard.current)


def test_rejected_proposal_cannot_become_an_iteration_reference():
    guard, _, _ = _priority()
    with pytest.raises(ValueError, match='committed'):
        guard.begin_iteration(np.array([[.01, 0.]]))


def test_merit_record_matches_the_r2_protocol_including_invalid_geometry():
    from omniretargeting._fast.qp_types import PriorityMerit
    guard, provider, _ = _priority()
    record = PriorityMerit.coerce(guard.evaluate_trial(guard.current))
    assert record.valid and record.levels[0] == pytest.approx(.05)
    assert record.slacks[('penetration', 0, 1)] == pytest.approx(.05)
    provider.fail = True
    record = PriorityMerit.coerce(guard.evaluate_trial(np.array([[.01, 0.]])))
    assert not record.valid


def test_contact_norm_at_zero_error_remains_an_exact_soft_cone():
    from omniretargeting._fast.contact_guard import ContactGuard
    from tests.test_fast_contact_guard import _acquired
    objective, poses = _acquired()

    class Empty:
        def measure(self, q, **kwargs):
            return dict(rows=[], depths={}, failures=[], complete=True)

        def linear_rows(self, measured):
            return sparse.csr_matrix((0, len(poses) * objective.width)), np.empty(0)

    coordinator = PriorityClearance(objective, Empty(), poses)
    objective.priority_clearance = coordinator
    guard = ContactGuard(objective, poses)
    assert guard.repair_bounds(poses) == ([], [])
    coordinator.begin_iteration(poses)
    cones = coordinator.soft_cones(poses)
    assert cones and any(matrix.nnz for matrix, *_ in cones)
    matrix, residual, allowed, weights, keys = cones[0]
    np.testing.assert_allclose(residual, 0., atol=1e-15)
    assert matrix.shape[0] == 3 * len(keys)
    assert keys[0][0:2] == ('contact', 'support')
    candidate = poses.copy()
    candidate[:, 2] += .004
    assert not guard.accepts(candidate)
    assert coordinator.evaluate_trial(candidate)['details']['reason'] == 'unfunded_contact_yield'


def test_original_geometry_adapter_includes_root_jacobian_columns(monkeypatch):
    from omniretargeting._fast.priority_geometry import PriorityGeometry
    from tests.test_fast_backend import _objective

    class Plane:
        def query(self, points):
            return dict(clearance=points[:, 2] - .3, uncertain=np.zeros(len(points), bool),
                        directions=np.tile([0., 0., 1.], (len(points), 1)))

    monkeypatch.setenv('OMNIRETARGETING_GPU_GEOMETRY', '0')
    objective, poses, _ = _objective()
    provider = PriorityGeometry(objective, scene=Plane())
    try:
        with pytest.raises(RuntimeError, match='Freeze'):
            provider.measure(poses)
        reference_id = provider.freeze_reference(poses)
        assert provider.freeze_reference(poses.copy()) == reference_id
        changed = poses.copy()
        changed[0, 2] += .001
        with pytest.raises(ValueError, match='rebased'):
            provider.freeze_reference(changed)
        measured = provider.measure(poses, jacobians=True)
        matrix, phi = provider.linear_rows(measured)
        assert matrix.shape[1] == len(poses) * objective.width
        root_z = objective.coords.inverse[objective.coords.root_d + 2]
        static = [(i, key) for i, (key, _) in enumerate(measured['rows']) if key[1] == 'static']
        assert static
        for i, key in static:
            assert matrix[i, key[0] * objective.width + root_z] == pytest.approx(1.)
        np.testing.assert_array_equal(provider.geometry.di, objective.coords.cols)
        assert not provider.geometry.columns.flags.writeable
    finally:
        provider.close()


def test_certified_qp_clears_penetration_despite_opposing_tracking_and_clean_hold():
    from omniretargeting._fast.qp_assembly import prepare_model, solve_model
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
    # Tracking strongly wants negative q (more penetration). The clean body
    # wants q <= 1 mm. Priority must first clear the original 50 mm overlap.
    model = prepare_model(objective, guard.current, np.array([[1e4, 1.]]),
                          np.array([1e4, 0.]), np.ones(2))
    proposal = solve_model(model, .001, priority=True)
    assert proposal is not None and proposal.certificate['certified']
    candidate = guard.current + proposal.step.reshape(1, 2)
    merit = guard.evaluate_trial(candidate)
    assert merit['valid']
    assert merit['levels'][0] < 1e-6
    assert merit['levels'][1] == pytest.approx(.049, abs=1e-6)
    assert model.tracking_decrease(proposal.step) < 0.
    guard.certificate = proposal.certificate
    guard.commit(candidate)
    assert guard.yields[0]['donor_body'] == 1
