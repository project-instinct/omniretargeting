"""Original positive-clearance preferences cannot masquerade as penetration."""

from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.priority_margins import preferences, margin_rows, margin_cost
from omniretargeting._fast.qp_assembly import SoftTerms
from omniretargeting._fast.qp_types import TRACKING_LEVEL
from tests.test_priority_window import case, Contact


def enabled_case():
    c, provider = case()
    c.objective.cfg.update(scene_clearance_m=.001, self_clearance_m=.002, collision_sigma_m=.001)
    c.objective.clearance_scale = .25
    c.margin_preferences = preferences(c.objective)
    c.owns_clearance = True
    return c, provider


def test_configured_margins_and_generic_weights_use_tracking_slot_only():
    obj = SimpleNamespace(cfg=dict(scene_clearance_m=.0015, self_clearance_m=.0025,
                                    collision_sigma_m=.001), clearance_scale=.5)
    settings = preferences(obj)
    rows = [((2, 'static', 0), {}), ((2, 'moving', 0, 'closed', 0, 1), {}),
            ((3, 'self', 0, 1), {})]
    phi = np.array([-.01, .0003, .001])
    batch, = margin_rows(sparse.eye(3, format='csr'), phi, rows, settings)
    matrix, upper, weight, keys = batch
    np.testing.assert_allclose(upper, phi - [.0015, .0015, .0025], rtol=0, atol=1e-16)
    np.testing.assert_allclose(weight, np.array([.0015, .0015, .0025]) / .0005 ** 2)
    assert all(key[0] == 'clearance_margin' for key in keys)
    soft = SoftTerms(3)
    soft.add(batch)
    assert soft.levels == [TRACKING_LEVEL] * 3
    values = soft.values(np.zeros(3))
    np.testing.assert_array_equal(soft.level_values(values), np.zeros(4))
    assert soft.penalty(values, np.ones(4)) == 0.
    assert soft.tracking_penalty(values) > 0.


def test_duplicate_manifold_witnesses_share_one_maximum_margin_epigraph():
    c, _ = enabled_case()
    key = (2, 'static', 0)
    rows = [(key, dict(clearance=.0004)), (key, dict(clearance=.0007)),
            (key, dict(clearance=.0004))]
    measured = dict(rows=rows)
    values, cost = margin_cost(measured, c.margin_preferences)
    assert len(values) == 1
    assert values[('clearance_margin', *key)] == pytest.approx(.0006)
    assert cost == pytest.approx(.0006 * c.margin_preferences['static'][1])
    batch, = margin_rows(sparse.csr_matrix(np.ones((3, 1))),
                          np.array([row['clearance'] for _, row in rows]), rows, c.margin_preferences)
    soft = SoftTerms(1)
    soft.add(batch)
    assert soft.tracking_penalty(soft.values([0.])) == pytest.approx(cost)


def test_new_candidate_margin_key_contributes_to_full_actual_tracking_cost():
    c, provider = enabled_case()
    frames = range(2, 3)
    c.begin_iteration(c.current, frames=frames)
    prepared_keys = {key for batch in c.soft_rows(c.current, frames=frames) for key in batch[-1]}
    original = provider.measure
    collision = (2, 'moving', 0, 'closed', 0, 1)

    def measure(q, **kwargs):
        result = original(q, **kwargs)
        if q[2, 1] > 0:
            result['rows'].append((collision, dict(clearance=.0002)))
            result['depths'][collision] = 0.
        return result

    provider.measure = measure
    q = c.current.copy()
    q[2] = .01
    merit = c.evaluate_trial(q, frames=frames)
    key = ('clearance_margin', *collision)
    assert key not in prepared_keys and merit['valid']
    assert merit['slacks'][key] == pytest.approx(.0008)
    expected = .041 * c.margin_preferences['static'][1] + .0008 * c.margin_preferences['moving'][1]
    assert merit['details']['tracking_margin_cost'] == pytest.approx(expected)
    assert merit['levels'][0] == pytest.approx(.04)
    assert merit['details']['yields'] == [] and c.spent == {}


def test_margin_improvement_cannot_fund_any_contact_yield():
    from omniretargeting._fast.priority_clearance import PriorityClearance
    old, provider = enabled_case()
    original = provider.measure

    def measure(q, **kwargs):
        value = original(q, **kwargs)
        for key, row in value['rows']:
            row['clearance'] = .0002 + q[key[0], 0]
            value['depths'][key] = max(-row['clearance'], 0.)
        return value

    provider.measure = measure
    c = PriorityClearance(old.objective, provider, old.current)
    c.contacts = [Contact()]
    c.margin_preferences, c.owns_clearance = old.margin_preferences, True
    c.objective.priority_clearance, c.objective.candidate_guards = c, [c]
    frames = range(2, 3)
    before = c.evaluate_trial(c.current, frames=frames)
    q = c.current.copy()
    q[2, 0] = .002
    after = c.evaluate_trial(q, frames=frames)
    assert before['details']['tracking_margin_cost'] > after['details']['tracking_margin_cost']
    assert before['levels'][0] == after['levels'][0] == 0.
    assert not after['valid'] and after['details']['reason'] == 'unfunded_contact_yield'
    assert after['details']['unfunded_contacts'][0]['available_m'] == 0.
    assert not c.spent and not c.yields


def test_geometry_soc_batches_remain_explicitly_disabled():
    c, _ = enabled_case()
    frames = range(2, 3)
    rows = c.soft_rows(c.current, frames=frames)
    values, _ = c.soft_values(c.current, frames=frames)
    assert len(rows) == len(values) == 3  # penetration, margin, scalar contact
    assert values[0] is None and values[1] is None
    assert values[2] is not None


def test_original_diagnostics_report_penetration_and_keep_unknown_non_numeric():
    c, provider = enabled_case()
    stats = c.collision_diagnostics(c.current, frames=range(2, 4))
    assert stats['scene_depth_m'] == pytest.approx(.05)
    assert stats['collision_diagnostics_source'] == 'original_geometry'
    assert stats['original_collision_families']['static']['frame'] == 2
    assert stats['original_collision_families']['static']['bodies'] == [1]
    assert stats['self_depth_m'] is None  # fake provider has no self-query coverage
    provider.fail = True
    unknown = c.collision_diagnostics(c.current, frames=range(2, 4))
    assert unknown['scene_depth_m'] is None and not unknown['original_collision_reliable']
    assert unknown['original_collision_failures']


def test_zero_configured_margin_adds_no_tracking_cost_or_rows():
    c, _ = enabled_case()
    c.objective.cfg.update(scene_clearance_m=0., self_clearance_m=0.)
    settings = preferences(c.objective)
    rows = [((0, 'static', 0), dict(clearance=-.02))]
    assert margin_rows(sparse.eye(1), [-.02], rows, settings) == []
    assert margin_cost(dict(rows=rows), settings) == ({}, 0.)


def test_real_geometry_ownership_requires_coverage_and_supported_margin(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.priority_geometry import PriorityGeometry
    from omniretargeting._fast.priority_clearance import PriorityClearance
    from tests.test_fast_backend import _objective
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, windowed_priority=True))
    obj, poses, _ = _objective()
    provider = PriorityGeometry(obj)
    c = PriorityClearance(obj, provider, poses)
    assert c.owns_clearance
    assert c.clearance_coverage['self_pair_count'] == len(provider.geometry.self_query.pairs)
    obj.cfg['scene_clearance_m'] = .009
    with pytest.raises(ValueError, match='witness band'):
        PriorityClearance(obj, provider, poses)
    obj.cfg['scene_clearance_m'] = .001
    obj.scene_sdf = object()
    with pytest.raises(ValueError, match='no original static'):
        PriorityClearance(obj, provider, poses)


def test_window_margin_proposal_and_actual_cost_agree_in_tiny_linear_geometry():
    from omniretargeting._fast.window_assembly import prepare_window_model, solve_window_model
    from omniretargeting._fast.qp_types import PriorityMerit
    from omniretargeting._fast.window_local import assess_window
    c, _ = enabled_case()
    frames = range(2, 3)
    acc = SimpleNamespace(hessian=np.eye(2)[None], gradient=np.zeros((1, 2)))
    evaluation = acc, np.empty(0), sparse.csr_matrix((0, 2)), {}, None
    model, mandatory = prepare_window_model(c.objective, c.current, frames, evaluation)
    solution, _ = solve_window_model(model, .001, mandatory)
    assert solution is not None
    q = c.current.copy()
    q[2] += solution.step
    before = PriorityMerit.coerce(c.evaluate_trial(c.current, frames=frames))
    after = PriorityMerit.coerce(c.evaluate_trial(q, frames=frames))
    assert after.valid
    actual = before.details['tracking_margin_cost'] - after.details['tracking_margin_cost']
    predicted = model.soft.tracking_penalty(model.predicted_reductions(solution.step))
    assert actual == pytest.approx(predicted)
    accepted, _, report = assess_window(model, solution.step, before, after, 0., float(np.square(solution.step).sum()),
                                         solution.certificate['coefficients'])
    assert accepted
    c.commit(q, frames=frames)
    assert c.current_depth[2].max() < 1e-7
