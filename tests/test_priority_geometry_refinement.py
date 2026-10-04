"""Cached-trial proposal enrichment cannot change contact or funding authority."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import runtime_options
from omniretargeting._fast.original_geometry_relinearize import witness_identity, pose_identity
from omniretargeting._fast.priority_clearance import PriorityClearance
from omniretargeting._fast.priority_window import _digest
from tests.test_priority_window import case, Contact


class Geometry:
    def __init__(self, specs=None):
        self.specs = specs or [('static', 1, 1, .02, (0., -1.))]
        self.measures, self.queries = 0, []
        self.edit_result = None

    @staticmethod
    def key(frame, kind, geom):
        if kind == 'self':
            return frame, 'self', 0, 1
        if kind == 'reverse':
            return frame, 'moving', 0, 'reverse_surface', 0, geom
        return frame, 'static', geom

    def measure(self, q, *, frames=None, jacobians=False):
        self.measures += 1
        frames = range(len(q)) if frames is None else frames
        rows = []
        for f in frames:
            rows.append(((f, 'static', 0), dict(sample_id=0, clearance=q[f, 0] - .05,
                                               origin_phi=-.05, vector=(1., 0.))))
            if q[f, 1] > .02:
                for kind, geom, sample, phi, vector in self.specs:
                    rows.append((self.key(f, kind, geom), dict(sample_id=sample,
                        facet_id=2 if kind == 'reverse' else None,
                        clearance=phi + q[f] @ vector, origin_phi=phi, vector=vector)))
        depths = {}
        for key, row in rows:
            depths[key] = max(depths.get(key, 0.), -row['clearance'])
        return dict(rows=rows, depths=depths, complete=True, failures=[], evidence_id='frozen-A')

    def linear_rows(self, measured, *, frames=None):
        rows = measured['rows']
        frames = range(len(rows)) if frames is None else frames
        matrix = sparse.lil_matrix((len(rows), len(frames) * 2))
        for i, (key, row) in enumerate(rows):
            column = (key[0] - frames.start) * 2
            matrix[i, column:column + 2] = row['vector']
        return matrix.tocsr(), np.array([row['clearance'] for _, row in rows])

    def relinearize_witnesses(self, origin, candidate, measurement, *, frames, keys):
        self.queries.append(measurement)
        rows = [(key, {**row, 'clearance': row['origin_phi'] + origin[key[0]] @ row['vector']})
                for key, row in measurement['rows']]
        matrix, phi = self.linear_rows(dict(rows=rows), frames=frames)
        result = dict(rows=rows, J=matrix, phi=phi, reliable=np.ones(len(rows), bool),
            witness_ids=tuple(witness_identity(key, row) for key, row in rows),
            origin_pose_id=pose_identity(origin, frames), trial_pose_id=pose_identity(candidate, frames),
            evidence_id='frozen-A', frames=tuple(frames), columns=np.arange(2),
            request_complete=True, failures=[], query_scope=dict(requested_witness_count=len(rows)))
        if self.edit_result is not None:
            self.edit_result(result)
        return result


def prepared(monkeypatch, specs=None):
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS,
                       window_geometry_generation=True))
    old, _ = case()
    provider = Geometry(specs)
    c = PriorityClearance(old.objective, provider, old.current)
    c.objective.priority_clearance = c
    c.objective.candidate_guards = [c]
    c.objective.coords.cols = np.arange(2)
    c.contacts = [Contact()]
    c.margin_preferences = {'static': (.001, 10.), 'moving': (.001, 10.), 'self': (.002, 20.)}
    frames = range(2, 4)
    c.begin_iteration(c.current, frames=frames)
    q = c.current.copy()
    q[2] = .01, .03
    c.evaluate_trial(q, frames=frames)
    return c, provider, frames, q


def refine(c, q, frames, *, known=(), budget=256):
    return c.geometry_refinement(c.current, q, frames=frames,
                                 known_witness_ids=frozenset(known), max_new_witnesses=budget)


def test_new_witness_uses_origin_phi_and_j_without_measure_or_funding_mutation(monkeypatch):
    c, provider, frames, q = prepared(monkeypatch)
    state = c._window_state
    matrix, phi = state.matrix.copy(), state.phi.copy()
    prediction = state.prediction(q)[0].copy()
    bounds = [a.copy() for a in state.contact_domain.bounds]
    caps = [a.copy() for a in state.contact_domain.caps]
    measures, reference, epoch = provider.measures, c.reference, c.epoch
    trial = state.trial
    result = refine(c, q, frames)
    assert result['complete'] and not result['unresolved'] and result['trial_cache_hit']
    assert result['origin_pose_id'] == _digest(c.current) and result['trial_pose_id'] == _digest(q)
    assert result['epoch'] == epoch and result['selection']['new'] == 1
    hard, = result['hard_linear']
    np.testing.assert_array_equal(hard[0].toarray(), [[0., 1., 0., 0.]])
    assert hard[2][0] == .02  # Trial phi was -10 mm; current is +20 mm (>8 mm band).
    pen, margin = result['soft_rows']
    assert pen[-1] == (('penetration', 2, 2),)
    assert margin[-1] == (('clearance_margin', 2, 'static', 1),)
    assert margin[1][0] == pytest.approx(.019)
    assert provider.measures == measures and len(provider.queries) == 1
    assert state.trial is trial and c._window_state is state and c.reference is reference
    assert c.epoch == epoch and not c.spent and not c.yields
    np.testing.assert_array_equal(state.matrix.toarray(), matrix.toarray())
    np.testing.assert_array_equal(state.phi, phi)
    np.testing.assert_array_equal(state.prediction(q)[0], prediction)
    for before, after in zip(bounds + caps, state.contact_domain.bounds + state.contact_domain.caps):
        np.testing.assert_array_equal(before, after)


def test_known_ids_and_original_rows_are_not_queried_or_added_twice(monkeypatch):
    c, provider, frames, q = prepared(monkeypatch)
    first = refine(c, q, frames)
    repeated = refine(c, q, frames, known=first['witness_ids'])
    assert not repeated['unresolved'] and not repeated['hard_linear']
    assert repeated['selection']['duplicate'] >= 1 and len(provider.queries) == 1
    q = c.current.copy()
    q[2, 0] = -.001
    c.evaluate_trial(q, frames=frames)
    existing = refine(c, q, frames)  # Caller starts empty: state IDs still count.
    assert not existing['unresolved'] and not existing['hard_linear']
    assert len(provider.queries) == 1


def test_budget_gives_each_failed_key_a_witness_before_extra_same_key_samples(monkeypatch):
    specs = [('static', 1, i, .02, (0., -1.)) for i in (1, 2, 3)]
    specs += [('static', 0, 4, -.05, (1., 0.))]
    c, provider, frames, q = prepared(monkeypatch, specs)
    q[2, 0] = -.001
    c.evaluate_trial(q, frames=frames)
    result = refine(c, q, frames, budget=2)
    assert result['unresolved'] and result['selection']['cap_exhausted'] == 2
    assert {row['key'] for row in result['row_metadata']} == {(2, 'static', 0), (2, 'static', 1)}
    assert len(provider.queries[0]['rows']) == 2
    assert result['selection']['new'] == 2


def test_same_key_new_samples_share_body_and_margin_keys(monkeypatch):
    c, _, frames, q = prepared(monkeypatch, [('static', 1, i, .02, (0., -1.)) for i in (1, 2)])
    result = refine(c, q, frames)
    assert len(result['witness_ids']) == 2 and len(set(result['witness_ids'])) == 2
    assert result['soft_rows'][0][-1] == (('penetration', 2, 2),) * 2
    assert result['soft_rows'][1][-1] == (('clearance_margin', 2, 'static', 1),) * 2


def test_self_pair_uses_both_canonical_body_epigraphs(monkeypatch):
    c, _, frames, q = prepared(monkeypatch, [('self', 1, None, .02, (0., -1.))])
    result = refine(c, q, frames)
    assert result['row_metadata'][0]['owners'] == (1, 2)
    assert result['soft_rows'][0][-1] == (('penetration', 2, 1), ('penetration', 2, 2))


def test_reverse_facet_excluding_origin_is_not_a_fake_hard_row(monkeypatch):
    c, _, frames, q = prepared(monkeypatch, [('reverse', 1, 1, -.04, (0., -1.))])
    result = refine(c, q, frames)
    assert result['unresolved'] and not result['hard_linear'] and not result['witness_ids']
    assert result['failures'][0]['reason'] == 'facet_incompatible_at_origin'
    assert result['failures'][0]['allowed_depth_m'] == 0.


@pytest.mark.parametrize('field', ['origin_pose_id', 'trial_pose_id', 'evidence_id', 'columns'])
def test_stale_relinearization_or_column_map_fails_closed(monkeypatch, field):
    c, provider, frames, q = prepared(monkeypatch)
    provider.edit_result = lambda r: r.update({field: np.array([1, 0]) if field == 'columns' else 'stale'})
    result = refine(c, q, frames)
    assert result['unresolved'] and not result['hard_linear'] and not result['soft_rows']
    assert c.epoch == 0 and not c.spent


def test_unknown_subset_stays_unresolved_but_reliable_rows_are_finite(monkeypatch):
    c, provider, frames, q = prepared(monkeypatch, [('static', 1, i, .02, (0., -1.)) for i in (1, 2)])
    def fail_one(r):
        r['phi'][1] = np.nan
        r['reliable'][1] = False
        r['request_complete'] = False
    provider.edit_result = fail_one
    result = refine(c, q, frames)
    assert result['unresolved'] and not result['request_complete'] and len(result['witness_ids']) == 1
    assert np.isfinite(result['hard_linear'][0][2]).all()


def test_cache_miss_disabled_and_zero_budget_never_query(monkeypatch):
    c, provider, frames, q = prepared(monkeypatch)
    result = refine(c, q + .001, frames)
    assert result['unresolved'] and result['failures'][0]['reason'] == 'trial_cache_miss'
    result = refine(c, q, frames, budget=0)
    assert result['unresolved'] and result['selection']['cap_exhausted'] == 1
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, window_geometry_generation=False))
    result = refine(c, q, frames)
    assert result['failures'][0]['reason'] == 'window_geometry_generation_disabled'
    assert not provider.queries
