"""Fork-net admission, bounded contact proposals and exact contact SOC values."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.priority_clearance import CONTACT_SLACK
from omniretargeting._fast.continuity import ROUND_OFF
from tests.test_priority_window import case, Contact


class Channel(Contact):
    def __init__(self, name='normal', *, own=0., vector=(1., -1.), body=2, norm=False):
        self.name, self.own, self.vector, self.body, self.norm = name, own, vector, body, norm

    def evaluate(self, poses, *, jacobian=True, frames=None):
        frames = range(len(poses)) if frames is None else frames
        block, = super().evaluate(poses, jacobian=jacobian, frames=frames)
        value = poses[list(frames)] @ self.vector
        if self.norm:
            value = np.column_stack((value, np.zeros((len(frames), 2))))
            jac = np.array([self.vector, (0., 0.), (0., 0.)])
        else:
            jac = np.asarray([self.vector])
        return [replace(block, matrix=sparse.kron(sparse.eye(len(frames)), jac, format='csr')
                        if jacobian else None, residual=value, own=np.full(len(frames), self.own),
                        bodies=np.full(len(frames), self.body), scalar=not self.norm,
                        identities=tuple((self.name, 0, f, 0, 'full' if self.norm else 'normal') for f in frames))]


def domain_case(*channels):
    c, provider = case()
    c.contacts[:] = channels
    frames = range(2, 3)
    c.begin_iteration(c.current, frames=frames)
    return c, provider, frames, c._window_state.contact_domain


@pytest.mark.parametrize('channel', [Channel(vector=(0., 1.)), Channel(body=1)])
def test_uncoupled_or_same_body_contact_has_only_original_hard_allowance(channel):
    c, _, frames, domain = domain_case(channel)
    assert not domain.soft_blocks and not c.soft_limits(c.current, frames=frames)
    linear, cones = c.qp_bounds(c.current, frames=frames)
    assert len(linear) == 2 and not cones
    assert linear[1][2][0] == pytest.approx(.001 + CONTACT_SLACK / 2)


def test_donor_in_another_frame_cannot_enable_local_yield():
    c, provider = case()
    c.contacts[:] = [Channel()]
    q = c.current.copy()
    q[2] = .05
    c.commit(q, frames=range(2, 3))
    c.begin_iteration(q, frames=range(2, 3))
    assert c.current_depth[1, 1] > 0 and c.current_depth[2, 1] == 0
    assert not c.soft_limits(q, frames=range(2, 3))
    assert not c.soft_cones(q, frames=range(2, 3))
    assert all(key[0] != 'contact' for row in c.soft_rows(q, frames=range(2, 3)) for key in row[-1])


def test_own_allowance_and_prior_committed_metric_center_domain():
    c, _, frames, domain = domain_case(Channel(own=.002))
    assert domain.bounds[0][0] == .003
    q = c.current.copy()
    q[2, 0] = .004
    report = c.evaluate_trial(q, frames=frames)
    key, = c.soft_limits(c.current, frames=frames)
    assert report['valid'] and report['slacks'][key] == pytest.approx(.001)
    c.commit(q, frames=frames)
    c.begin_iteration(q, frames=frames)
    assert c._window_state.contact_domain.bounds[0][0] == .004


def test_shared_key_max_cap_retains_per_channel_caps_and_per_identity_ledger():
    c, _ = case()
    c.contacts[:] = [Channel('a'), Channel('b', norm=True)]
    c.spent[(('a', 0, 2, 0, 'normal'), 1)] = .04
    c.spent[(('b', 0, 2, 0, 'full'), 1)] = .02
    frames = range(2, 3)
    limits = c.soft_limits(c.current, frames=frames)
    key, = limits
    assert limits[key] == pytest.approx(.03)
    linear, cones = c.qp_bounds(c.current, frames=frames)
    assert linear[1][2][0] == pytest.approx(.011 + CONTACT_SLACK / 2)
    assert cones[0][2][0] == pytest.approx(.031 + CONTACT_SLACK / 2)
    assert c.soft_rows(c.current, frames=frames)[1][-1] == (key,)
    assert c.soft_cones(c.current, frames=frames)[0][-1] == (key,)
    assert c.soft_limits(c.current) == {}  # Legacy API remains unchanged.


def test_exhausted_identity_and_best_depth_zero_cannot_offer_new_yield():
    c, _ = case()
    c.contacts[:] = [Channel()]
    c.spent[(('normal', 0, 2, 0, 'normal'), 1)] = .05
    assert not c.soft_limits(c.current, frames=range(2, 3))
    c, _ = case()
    c.contacts[:] = [Channel()]
    c.best_depth[2, 1] = 0.
    assert not c.soft_limits(c.current, frames=range(2, 3))


def test_depth_cap_is_only_necessary_and_does_not_certify_donor_progress():
    c, _, frames, _ = domain_case(Channel(norm=True))
    q = c.current.copy()
    q[2, 1] = -.01  # Contact gap grows, donor does not clear at all.
    cap, = c.soft_limits(c.current, frames=frames).values()
    assert .009 < cap
    report = c.evaluate_trial(q, frames=frames)['details']
    assert not report['accepted'] and report['reason'] == 'unfunded_contact_yield'
    assert not c.spent and c.epoch == 0


def test_cleared_private_donor_retains_net_funded_yield_in_next_qp_and_parent():
    c, _ = case()
    frames = range(2, 3)
    child = c.fork_window(frames)
    q = child.current.copy()
    q[2] = .05, .001
    child.advance_local(q, frames=frames)
    assert child.current_depth[2, 1] == 0 and not child.spent and not child.yields
    child.begin_iteration(q, frames=frames)
    cap, = child.soft_limits(q, frames=frames).values()
    assert cap == pytest.approx(.05)
    linear, _ = child.qp_bounds(q, frames=frames)
    assert linear[1][2][0] > 0  # Already funded gap remains QP-feasible.
    q[2] = .055, .006
    child.advance_local(q, frames=frames)
    private = child.evaluate_trial(q, frames=frames)['details']
    parent = c.evaluate_trial(q, frames=frames)['details']
    for name in ('accepted', 'penetration_regressions', 'body_regressions',
                 'total_penetration_regression', 'unfunded_contacts', 'hard_failures', 'yields'):
        assert private[name] == parent[name]
    c.commit(q, frames=frames, transactions=child.export_transactions(frames=frames))
    assert c.epoch == 1 and sum(c.spent.values()) == pytest.approx(.048)
    assert c.current_depth[2, 1] == 0


def test_child_uses_fork_baseline_for_keys_bodies_total_and_qp():
    c, _ = case()
    frames = range(2, 4)
    child = c.fork_window(frames)
    q = child.current.copy()
    q[2, 0] = -.0004
    child.advance_local(q, frames=frames)
    linear, _ = child.qp_bounds(q, frames=frames)
    assert linear[0][2][0] == pytest.approx(0.)  # Reserve already consumed.
    q[3, 0] = -.0002
    a = child.evaluate_trial(q, frames=frames)['details']
    b = c.evaluate_trial(q, frames=frames)['details']
    assert not a['accepted'] and not b['accepted']
    assert a['total_penetration_regression'] and b['total_penetration_regression']
    assert a['rejection_details'] == b['rejection_details']


@pytest.mark.parametrize('failure', ['none', 'key_body', 'total', 'hard', 'funding', 'unknown'])
def test_private_and_parent_exact_acceptance_rules_match_without_neighbor_changes(failure):
    c, provider = case()
    frames = range(2, 4)
    child = c.fork_window(frames)
    q = child.current.copy()
    q[2] = .005
    child.advance_local(q, frames=frames)
    c.begin_iteration(c.current, frames=frames)
    child.begin_iteration(child.current, frames=frames)
    if failure == 'key_body':
        q[2] = -.0006, 0.
    elif failure == 'total':
        q[2:4, 0] = -.0003
    elif failure == 'hard':
        q[2, 1] = 1.1
    elif failure == 'funding':
        q[2] = .006, -.06
    elif failure == 'unknown':
        provider.fail = child.provider.fail = True
    else:
        q[2] = .006
    a = child.evaluate_trial(q, frames=frames)['details']
    b = c.evaluate_trial(q, frames=frames)['details']
    for name in ('accepted', 'reliable', 'reason', 'penetration_regressions', 'body_regressions',
                 'total_penetration_regression', 'unfunded_contacts', 'hard_failures', 'yields'):
        assert a.get(name) == b.get(name), name
    assert a['accepted'] is (failure == 'none')
    assert c.epoch == 0 and child.epoch == 1 and not c.spent and not child.spent


def test_hard_and_soft_soc_values_align_and_reuse_contact_values_without_geometry():
    c, provider, frames, _ = domain_case(Channel('scalar'), Channel('norm', norm=True))
    rows, cones = c.qp_bounds(c.current, frames=frames)
    soft_rows, soft_cones = c.soft_rows(c.current, frames=frames), c.soft_cones(c.current, frames=frames)
    q = c.current.copy()
    q[2, 0] = .002
    c.evaluate_trial(q, frames=frames)
    before = len(provider.calls)
    hard_values, hard_norms = c.qp_values(q, frames=frames)
    soft_values, soft_norms = c.soft_values(q, frames=frames)
    sources = c.qp_bound_sources(c.current, frames=frames)
    assert len(provider.calls) == before and hard_values[0] is None and soft_values[0] is None
    assert sources == (['penetration_nonregression', 'contact_cap:normal'], ['contact_cap:full'])
    step = (q[2] - c.current[2]).ravel()
    np.testing.assert_allclose(hard_values[1][1], rows[1][2] - rows[1][0] @ step)
    np.testing.assert_allclose(soft_values[1], soft_rows[1][1] - soft_rows[1][0] @ step)
    np.testing.assert_allclose(hard_norms[0][0], cones[0][1] + (cones[0][0] @ step).reshape(-1, 3))
    np.testing.assert_allclose(soft_norms[0][0], soft_cones[0][1] + (soft_cones[0][0] @ step).reshape(-1, 3))
    np.testing.assert_array_equal(hard_norms[0][1], cones[0][2])


def test_real_root_ball_roundoff_matches_report_and_does_not_move_budget():
    from tests.test_fast_backend import _objective
    from omniretargeting._fast.reference import ReferenceShape
    from omniretargeting._fast.priority_window import hard_failures
    from omniretargeting._fast.priority_rejections import hard_details
    obj, poses, _ = _objective()
    obj.candidate_guards = []
    obj.source_shape = ReferenceShape(obj, poses)
    obj.source_shape.xy = obj.root_budget = .03
    obj.source_shape.acquired_poses = poses.copy()
    c = SimpleNamespace(objective=obj)
    for extra, expected in [(1.4e-16, False), (.5 * ROUND_OFF, False), (1.1 * ROUND_OFF, True)]:
        q = poses.copy()
        q[2, 0] += .03 + extra
        failures = hard_failures(c, q, range(2, 3))
        balls = [name for name in failures if 'root_ball' in name]
        assert bool(balls) is expected
        if expected:
            assert len(balls) == 3
            details = hard_details(c, q, range(2, 3), balls)
            assert len(details) == 3
            assert all(row['tolerance'] == ROUND_OFF and row['excess'] > 0 for row in details)
    assert obj.root_budget == obj.source_shape.xy == .03


def test_real_coordinator_no_donor_hard_contact_keeps_soc_coverage():
    from omniretargeting._fast.window_assembly import prepare_window_model
    from omniretargeting._fast.window_soc import corrected_model

    class CurvedContact(Channel):
        def __init__(self):
            super().__init__(vector=(0., 1.), norm=True)

        def evaluate(self, poses, *, jacobian=True, frames=None):
            block, = super().evaluate(poses, jacobian=jacobian, frames=frames)
            block.residual[:, 1] = 1000 * block.residual[:, 0] ** 2
            return [block]  # The derivative at the prepared origin (zero) is exact.

    c, provider, frames, _ = domain_case(CurvedContact())
    evaluation = (SimpleNamespace(hessian=np.eye(2)[None], gradient=np.zeros((1, 2))),
                  np.empty(0), sparse.csr_matrix((0, 2)), {}, None)
    model, _ = prepare_window_model(c.objective, c.current, frames, evaluation)
    assert not any(key[0] == 'contact' for key in model.soft.keys)
    q = c.current.copy()
    q[2, 1] = .001
    before = len(provider.calls)
    corrected = corrected_model(model, q, np.array([0., .001]))
    assert corrected is not None and len(provider.calls) == before
    np.testing.assert_array_equal(corrected.gradient, model.gradient)
    np.testing.assert_array_equal(corrected.hessian.toarray(), model.hessian.toarray())
    # Original geometry stays unmodified/unknown for SOC; the exact contact
    # norm gets its second-order offset despite having no soft epigraph.
    for old, new in zip(model.program.hard_blocks, corrected.program.hard_blocks):
        if old.soc:
            assert not np.array_equal(old.b, new.b)
        else:
            np.testing.assert_array_equal(old.b, new.b)
