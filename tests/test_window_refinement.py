"""Geometry enrichment preserves original domains and exact admission gates."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.qp_assembly import assemble_program
from omniretargeting._fast.qp_guard_rows import norms
from omniretargeting._fast.window_refinement import GeometryRefinement, augment_model, pose_digest
from omniretargeting._fast.window_screen import residuals
from tests.test_fast_qp_assembly import small_model


PEN = ('penetration', 0, 1)
MARGIN = ('clearance_margin', 0, 'static', 1)


def package(model, candidate=None, *, identity=0):
    candidate = model.origin if candidate is None else candidate
    return dict(hard_linear=[(sparse.csr_matrix([[1.]]), np.array([-np.inf]), np.array([.02]))],
                soft_rows=[(sparse.csr_matrix([[1.]]), np.array([.02]), np.ones(1), (PEN,)),
                           (sparse.csr_matrix([[1.]]), np.array([.01]), np.array([2.]), (MARGIN,))],
                witness_ids=((0, 'material', identity),), unresolved=False, epoch=0,
                origin_pose_id=pose_digest(model.origin), trial_pose_id=pose_digest(candidate),
                evidence_id='fixed')


def model_with_cones():
    contact = ('contact', 'support', 0, 1)
    matrix = sparse.csr_matrix([[1.], [0.], [0.]])
    model = small_model([(sparse.csr_matrix([[-1.]]), np.array([.01]), np.ones(1), (PEN,))],
                        [(matrix, np.zeros((1, 3)), np.array([.1]), np.ones(1), (contact,))])
    model.soft.add_limits({contact: .03})
    model.program = assemble_program([*model.program.hard_blocks,
                                      norms(matrix, np.zeros((1, 3)), [.3])], model.soft, 1)
    model.frames = range(1)
    count = len(model.program.b)
    model.window_row_sources = dict(names=[f'old:{i}' for i in range(count)], codes=np.arange(count))
    model.window_working_set = {'old': True}
    model.priority_cache = {'old': True}
    return model


def test_augmented_domain_is_original_intersect_new_rows_with_shared_slack():
    model = model_with_cones()
    old_A, old_b = model.program.A.copy(), model.program.b.copy()
    old_heads = [(a.copy(), b.copy()) for a, b in model.soft.heads]
    mandatory = np.arange(len(old_b)) % 2 == 0
    updated, keep, diagnostic = augment_model(model, mandatory, package(model))
    assert updated.hessian is model.hessian and updated.gradient is model.gradient
    assert updated.origin is model.origin and updated.step_lower is model.step_lower
    assert updated.soft.keys == [*model.soft.keys, MARGIN]  # PEN is shared, not charged twice.
    assert updated.soft.limits == model.soft.limits
    assert updated.window_working_set is None and updated.priority_cache is None
    assert model.window_working_set == {'old': True}
    assert diagnostic['added_slack_keys'] == 1
    source = updated.window_row_sources
    previous_rows = [np.flatnonzero(source['codes'] == source['names'].index(f'old:{i}')).item()
                     for i in range(len(old_b))]
    np.testing.assert_array_equal(updated.program.A[previous_rows, :old_A.shape[1]].toarray(), old_A.toarray())
    np.testing.assert_array_equal(updated.program.A[previous_rows, old_A.shape[1]:].toarray(), 0.)
    np.testing.assert_array_equal(updated.program.b[previous_rows], old_b)
    # Nonnegative epigraph rows are mandatory regardless of the test's arbitrary original mask.
    domain = np.arange(model.program.nonneg - len(model.soft.keys) - len(model.soft.limits),
                       model.program.nonneg)
    expected = mandatory.copy()
    expected[domain] = True
    np.testing.assert_array_equal(keep[previous_rows], expected)
    for a, b in zip(old_heads, model.soft.heads):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(model.program.A.toarray(), old_A.toarray())
    for step in (-.4, -.1, 0., .01, .02, .021, .13, .4):
        values = updated.predicted_slacks([step])
        x = np.r_[step, [values[k] for k in updated.soft.keys]]
        old_ok = all(np.min(r, initial=0.) >= -1e-14 for r in residuals(model.program, x[:old_A.shape[1]]))
        new_ok = all(np.min(r, initial=0.) >= -1e-14 for r in residuals(updated.program, x))
        assert new_ok == (old_ok and step <= .02)


@pytest.mark.parametrize('field,value', [('epoch', 1), ('origin_pose_id', 'stale'),
                                      ('trial_pose_id', 'stale'), ('evidence_id', 'wrong')])
def test_stale_package_cannot_change_model(field, value):
    model = small_model()
    result = package(model)
    result[field] = value
    coordinator = SimpleNamespace(epoch=0, _window_state=SimpleNamespace(measured={'evidence_id': 'fixed'}),
                                  geometry_refinement=lambda *a, **k: result)
    refiner = GeometryRefinement(coordinator, enabled=True)
    updated, _, log = refiner.refine(model, np.ones(len(model.program.b), bool), model.origin)
    assert updated is None and log['unresolved']
    assert log['reason'] == 'invalid_geometry_refinement' and refiner.rounds == 0


@pytest.mark.parametrize('upper', [-1e-16, np.nan])
def test_origin_incompatible_or_unknown_hard_row_does_not_enter_qp(upper):
    model = small_model()
    result = package(model)
    result['hard_linear'][0][2][0] = upper
    with pytest.raises(ValueError, match='origin-incompatible'):
        augment_model(model, np.ones(len(model.program.b), bool), result)


def test_witness_and_request_budgets_are_iteration_wide(monkeypatch):
    from omniretargeting._fast import window_refinement
    monkeypatch.setattr(window_refinement, 'MAX_NEW_WITNESSES', 2)
    model = small_model()
    calls = []
    def hook(origin, candidate, **kwargs):
        calls.append(kwargs)
        return package(model, identity=len(calls))
    refiner = GeometryRefinement(SimpleNamespace(epoch=0, geometry_refinement=hook), enabled=True)
    mask = np.ones(len(model.program.b), bool)
    for _ in range(2):
        model, mask, log = refiner.refine(model, mask, model.origin)
        assert model is not None and log['added']
    updated, _, log = refiner.refine(model, mask, model.origin)
    assert updated is None and log['reason'] == 'geometry_generation_budget'
    assert [c['max_new_witnesses'] for c in calls] == [2, 1]
    assert calls[0]['known_witness_ids'] == frozenset()
    assert len(calls[1]['known_witness_ids']) == 1


def test_no_more_than_three_relinearization_requests_even_without_new_rows():
    model = small_model()
    calls = []
    def hook(*args, **kwargs):
        calls.append(1)
        return dict(hard_linear=[], soft_rows=[], unresolved=False)
    refiner = GeometryRefinement(SimpleNamespace(epoch=0, geometry_refinement=hook), enabled=True)
    mask = np.ones(len(model.program.b), bool)
    for _ in range(7):
        updated, _, _ = refiner.refine(model, mask, model.origin)
        assert updated is None
    assert len(calls) == 3 and refiner.rounds == 0
    assert refiner.trace[-1]['reason'] == 'geometry_generation_budget'


def test_new_geometry_batches_do_not_disable_existing_contact_soc():
    from omniretargeting._fast.window_soc import corrected_model
    model = model_with_cones()
    provider = SimpleNamespace(soft_values=lambda p, **k: ([None], [
        (np.array([[p[0, 0], .5, 0.]]), np.array([.1]))]))
    model.window_soft_sources = ((provider, 0, 1, 0, 1),)
    updated, _, _ = augment_model(model, np.ones(len(model.program.b), bool), package(model))
    corrected = corrected_model(updated, np.array([[.01]]), np.array([.01]))
    assert corrected is not None
    np.testing.assert_array_equal(corrected.soft.cones[0][1], [[0., .5, 0.]])
    for old, new in zip(updated.soft.rows, corrected.soft.rows):
        np.testing.assert_array_equal(old[1], new[1])
    assert corrected.gradient is model.gradient


def emerging_problem():
    """Exact linear geometry oracle with a 15 mm-clear, initially absent key."""
    from tests.test_priority_window import Provider
    from tests.test_window_driver import problem
    from omniretargeting._fast.priority_clearance import PriorityClearance
    from omniretargeting._fast.original_geometry_relinearize import witness_identity, pose_identity
    class Emerging(Provider):
        replay_calls = 0

        def measure(self, poses, *, frames=None, jacobians=False):
            frames = range(len(poses)) if frames is None else frames
            result = super().measure(poses, frames=frames, jacobians=jacobians)
            for _, row in result['rows']:
                row['sample_id'] = 0
            for f in frames:
                key, clearance = (f, 'static', 1), .015 - poses[f, 0]
                result['depths'][key] = max(-clearance, 0.)
                if clearance < .008:
                    result['rows'].append((key, dict(clearance=clearance, sample_id=1)))
            result['evidence_id'] = 'fixed'
            return result

        def linear_rows(self, measured, *, frames):
            rows = measured['rows']
            matrix = sparse.csr_matrix(([1. if k[-1] == 0 else -1. for k, _ in rows],
                                        (np.arange(len(rows)), [(k[0] - frames.start) * 2 for k, _ in rows])),
                                       shape=(len(rows), len(frames) * 2))
            return matrix, np.array([r['clearance'] for _, r in rows])

        def relinearize_witnesses(self, origin, candidate, measured, *, frames, keys=None):
            Emerging.replay_calls += 1
            rows = [(k, dict(r, clearance=.015 - origin[k[0], 0])) for k, r in measured['rows']]
            matrix, phi = self.linear_rows({'rows': rows}, frames=frames)
            return dict(rows=rows, J=matrix, phi=phi, reliable=np.ones(len(rows), bool),
                        witness_ids=tuple(witness_identity(k, r) for k, r in rows),
                        origin_pose_id=pose_identity(origin, frames), trial_pose_id=pose_identity(candidate, frames),
                        evidence_id='fixed', request_complete=True, failures=[], query_scope={}, frames=tuple(frames))
    objective, poses = problem(2)
    contacts = objective.priority_clearance.contacts
    coordinator = PriorityClearance(objective, Emerging(), poses)
    coordinator.contacts = contacts
    coordinator._current_contacts.update(coordinator._contact_values(contacts[0].evaluate(poses)))
    objective.priority_clearance, objective.candidate_guards = coordinator, [coordinator]
    return objective, poses, Emerging


def test_real_driver_r1_hook_adds_missing_geometry_before_escalation(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.window_local import solve_local_window
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, window_geometry_generation=True))
    objective, poses, provider = emerging_problem()
    result = solve_local_window(objective, poses, range(2), 1, .003)
    row = result['trace'][0]
    assert result['changed'] and row['alpha'] == 1.
    assert row['geometry_generations'] == 1 and row['escalations'] == 0
    assert provider.replay_calls == 1
    np.testing.assert_allclose(result['poses'][:, 0], .015, atol=2e-7)
    full_checks = [c for c in row['trial_trace'] if c['alpha'] == 1.]
    assert len(full_checks) == 2 and not full_checks[0]['valid'] and full_checks[1]['valid']
    solves = [s for s in row['qp_trace'] if s['generation'] == 0]
    assert len(solves) == 2 and solves[0]['damping'] == solves[1]['damping']
    assert {s['penetration_multiplier'] for s in solves} == {1.}
    assert objective.priority_clearance.epoch == 0 and not objective.priority_clearance.spent


def test_flag_off_never_invokes_geometry_replay(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.window_local import solve_local_window
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, window_geometry_generation=False))
    objective, poses, provider = emerging_problem()
    result = solve_local_window(objective, poses, range(2), 1, .003)
    assert provider.replay_calls == 0
    assert result['trace'][0]['geometry_generations'] == 0


def test_unknown_new_witness_uses_shortening_without_blind_penalty_escalation(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.window_local import solve_local_window
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, window_geometry_generation=True))
    objective, poses, provider = emerging_problem()
    def unavailable(*args, **kwargs):
        raise RuntimeError('unknown material sample')
    monkeypatch.setattr(provider, 'relinearize_witnesses', unavailable)
    result = solve_local_window(objective, poses, range(2), 1, .003)
    row = result['trace'][0]
    assert row['escalations'] == row['geometry_generations'] == 0
    assert row['alpha'] < 1.
    assert all(t['unresolved'] for t in row['geometry_refinement_trace'])
    assert all(t['penetration_multiplier'] == 1. for t in row['qp_trace'])
    # Full exact checks still reject the too-large steps; the smaller safe
    # step can be accepted without pretending the unavailable row was added.
    assert result['changed'] and (result['poses'][:, 0] <= .0155).all()
