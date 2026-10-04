"""Net parent acceptance, immutable credit, and exact local contact channels."""

from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.priority_clearance import PriorityClearance
from omniretargeting._fast.priority_contacts import ContactRows


class Provider:
    def __init__(self):
        self.calls = []
        self.fail = False

    def fork_private_client(self):
        clone = copy(self)
        clone.calls = []
        return clone

    def measure(self, poses, *, frames=None, jacobians=False):
        frames = range(len(poses)) if frames is None else frames
        self.calls.append(tuple(frames))
        rows = [((f, 'static', 0), dict(clearance=poses[f, 0] - .05)) for f in frames]
        return dict(rows=rows, depths={k: max(-r['clearance'], 0.) for k, r in rows},
                    complete=not self.fail, failures=['unknown'] if self.fail else [])

    def linear_rows(self, measured, *, frames=None):
        frames = range(len(measured['rows'])) if frames is None else frames
        rows = measured['rows']
        mat = sparse.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)),
                               [(key[0] - frames.start) * 2 for key, _ in rows])),
                               shape=(len(rows), len(frames) * 2))
        return mat, np.array([row['clearance'] for _, row in rows])


class Contact:
    def evaluate(self, poses, *, jacobian=True, frames=None):
        frames = range(len(poses)) if frames is None else frames
        part = slice(frames.start, frames.stop)
        n = len(frames)
        return [ContactRows(sparse.kron(sparse.eye(n), [[1., -1.]], format='csr') if jacobian else None,
                            poses[part, 0] - poses[part, 1], np.full(n, .001), np.zeros(n),
                            np.array(frames), np.full(n, 2), np.zeros(n, int),
                            tuple(('contact', 0, f, 0, 'normal') for f in frames), True)]


def case(count=8):
    model = SimpleNamespace(nbody=3, geom_bodyid=np.array([1, 2]), body_weldid=np.arange(3))
    coords = SimpleNamespace(difference=lambda a, b: b - a, qi=np.arange(2), width=2,
                             scalar_cols=np.arange(2), root_q=None,
                             bounds=lambda p, a, t: (-np.full_like(p, a), np.full_like(p, a)))
    objective = SimpleNamespace(model=model, frames=count, width=2, coords=coords,
                                lower=-np.ones(2), upper=np.ones(2), source_shape=None,
                                window_dependencies=lambda f: range(max(0, f.start - 1), min(count, f.stop + 1)),
                                cfg=dict(angular_step_rad=.12, translation_step_m=.12))
    provider = Provider()
    c = PriorityClearance(objective, provider, np.zeros((count, 2)))
    c.contacts.append(Contact())
    c._current_contacts.update(c._contact_values(c.contacts[0].evaluate(c.current)))
    objective.priority_clearance, objective.candidate_guards = c, [c]
    return c, provider


def test_window_bounds_and_prediction_have_local_columns_global_keys():
    c, provider = case()
    frames = range(2, 4)
    c.begin_iteration(c.current, frames=frames)
    linear, _ = c.qp_bounds(c.current, frames=frames)
    assert linear[0][0].shape == (2, 4)
    soft = c.soft_rows(c.current, frames=frames)
    assert soft[0][3] == (('penetration', 2, 1), ('penetration', 3, 1))
    q = c.current.copy()
    q[2:4, 0] = .01
    report = c.evaluate_trial(q, frames=frames)['details']
    assert report['accepted'] and report['actual_reductions'][0] == pytest.approx(.02)
    assert report['predicted_reductions'][0] == pytest.approx(.02)
    assert provider.calls[1:] == [(2, 3), (2, 3)]
    assert c.spent == {} and c.epoch == 0
    c.commit(q, frames=frames)
    assert len(c.spent) == 2 and c.epoch == 1
    assert all(row['verification'] == 'parent_net' for row in c.yields)
    np.testing.assert_array_equal(c.current[[0, 1, 4, 5, 6, 7]], 0.)


def test_original_frozen_allowance_not_last_committed_metric():
    c, _ = case()
    frames = range(2, 3)
    # Both moves stay in the frozen 1 mm contact budget. v1's 1e-7
    # no-regression condition would reject; V2 must keep the original rule.
    for x in (.0004, .0008):
        q = c.current.copy()
        q[2, 0] = x
        assert c.accepts(q, frames=frames)
        c.commit(q, frames=frames)
    assert not c.spent and not c.yields


def test_fork_no_remeasurement_and_yield_then_unyield_cannot_mint_credit():
    c, provider = case()
    frames = range(2, 4)
    before = len(provider.calls)
    child = c.fork_window(frames)
    assert len(provider.calls) == before and child.provider.calls == []
    assert child.reference is c.reference and child.initial_depth is c.initial_depth
    q = child.current.copy()
    q[2:4, 0] = .01
    child.advance_local(q, frames=frames)
    assert not child.spent and not child.yields and not c.spent and c.epoch == 0
    q[2:4] = .02  # More clearance, but the temporary contact gap has closed.
    child.advance_local(q, frames=frames)
    transaction = child.export_transactions(frames=frames)
    assert transaction['local_steps'][0]['temporary_yields'] == 2
    assert transaction['credit_authority'] is False
    c.commit(q, frames=frames, transactions=transaction)
    assert not c.spent and not c.yields and c.epoch == 1
    assert c.current_depth[2, 1] == pytest.approx(.03)


def test_two_disjoint_footprints_commit_independently_from_one_batch():
    c, _ = case(10)
    frames = [range(1, 3), range(7, 9)]
    children = [c.fork_window(f) for f in frames]
    for child, f in zip(children, frames):
        q = child.current.copy()
        q[f.start:f.stop] = .01
        child.advance_local(q, frames=f)
    for child, f in zip(children, frames):
        c.commit(child.current, frames=f, transactions=child.export_transactions(frames=f))
    assert c.epoch == 2
    np.testing.assert_array_equal(c.current[[1, 2, 7, 8]], .01)
    np.testing.assert_array_equal(c.current[[0, 3, 4, 5, 6, 9]], 0.)


def test_stale_boundary_rejects_atomically_without_partial_credit():
    c, _ = case()
    child = c.fork_window(range(2, 4))
    q = child.current.copy()
    q[2:4] = .01
    child.advance_local(q)
    other = c.current.copy()
    other[1] = .01  # A fixed neighbor in the child's declared read footprint.
    c.commit(other, frames=range(1, 2))
    saved = c.current.copy(), c.best_depth.copy(), dict(c.spent), c.epoch
    with pytest.raises(ValueError, match='Stale'):
        c.commit(q, frames=range(2, 4), transactions=child.export_transactions())
    np.testing.assert_array_equal(c.current, saved[0])
    np.testing.assert_array_equal(c.best_depth, saved[1])
    assert (c.spent, c.epoch) == saved[2:]


def test_parent_remeasures_child_and_ignores_forged_credit_claim():
    c, provider = case()
    child = c.fork_window(range(2, 4))
    q = child.current.copy()
    q[2:4, 0] = .01
    child.advance_local(q)
    claim = child.export_transactions()
    claim.update(certified=True, spent={}, available_credit_m=100.)
    provider.fail = True
    with pytest.raises(ValueError, match='unknown original geometry'):
        c.commit(q, frames=range(2, 4), transactions=claim)
    assert c.epoch == 0 and not c.spent
    np.testing.assert_array_equal(c.current, 0.)


def test_parent_checks_immutable_boxes_but_not_per_iteration_trust_box():
    c, _ = case()
    frames = range(2, 3)
    c.objective.cfg['angular_step_rad'] = .001
    q = c.current.copy()
    q[2] = .02  # Accumulation of many steps may exceed a single trust box.
    c.commit(q, frames=frames)
    bad = q.copy()
    bad[2, 1] = 1.01
    with pytest.raises(ValueError, match='hard_guard'):
        c.commit(bad, frames=frames)
    np.testing.assert_array_equal(c.current, q)


def test_subtolerance_individual_regression_may_increase_window_sum():
    c, _ = case()
    q = c.current.copy()
    q[2, 0] -= 1e-8
    report = c.evaluate_trial(q, frames=range(2, 3))['details']
    assert report['accepted'] and not report['total_penetration_regression']


def test_complete_window_qp_proposal_and_nonlinear_parent_commit():
    from omniretargeting._fast.window_assembly import prepare_window_model, solve_window_model
    c, _ = case()
    frames = range(2, 4)
    acc = SimpleNamespace(hessian=np.tile(np.eye(2), (2, 1, 1)), gradient=np.zeros((2, 2)))
    evaluation = acc, np.empty(0), sparse.csr_matrix((0, 4)), {}, None
    model, mandatory = prepare_window_model(c.objective, c.current, frames, evaluation)
    proposal, trace = solve_window_model(model, .001, mandatory)
    assert proposal is not None and not proposal.certificate['optimality_certified']
    q = c.current.copy()
    q[2:4] += proposal.step.reshape(2, 2)
    assert c.evaluate_trial(q, frames=frames)['valid']
    c.commit(q, frames=frames)
    assert c.current_depth[2:4].max() < 1e-7
    assert c.epoch == 1


@pytest.mark.parametrize('kind', ['acquisition', 'contact'])
def test_legacy_generic_guard_assembly_unchanged_with_default_frames(kind, monkeypatch):
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    from omniretargeting._fast.contact_guard import ContactGuard
    from omniretargeting._fast.qp_guard_rows import _generic, intervals, norms
    from tests.test_fast_contact_guard import _acquired
    objective, poses = _acquired(surfaces=True)
    guard = (AcquisitionNormalGuard if kind == 'acquisition' else ContactGuard)(objective, poses)
    linear, cones = guard.repair_bounds(poses)
    expected = [intervals(*item) for item in linear] + [norms(*item) for item in cones]
    actual = _generic(guard, poses)
    assert len(expected) == len(actual)
    for old, new in zip(expected, actual):
        np.testing.assert_array_equal(new.A.toarray(), old.A.toarray())
        np.testing.assert_array_equal(new.b, old.b)
        assert new.soc == old.soc


@pytest.mark.parametrize('kind', ['acquisition', 'contact'])
def test_window_contact_acceptance_matches_original_frozen_channels(kind, monkeypatch):
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    from omniretargeting._fast.contact_guard import ContactGuard
    from omniretargeting._fast.priority_contacts import PriorityContacts
    from tests.test_fast_contact_guard import _acquired
    objective, poses = _acquired()
    for effector in objective.effectors:
        effector['normal'][:] = [0., 0., 1.]
    module = 'omniretargeting._fast.' + ('acquisition_guard' if kind == 'acquisition' else 'contact_guard')
    monkeypatch.setattr(module + '.on_surface', lambda obj, p: np.ones(p.shape[:2], bool))
    monkeypatch.setattr(module + '.body_penetration',
                        lambda obj, p: np.full((len(p), obj.model.nbody), .02))
    guard = (AcquisitionNormalGuard if kind == 'acquisition' else ContactGuard)(objective, poses)
    frames = range(2, 5)
    for dz in [-.0211, -.0011, -.0005, 0., .0005, .0011, .019, .0211]:
        q = poses.copy()
        q[2:5, 2] += dz
        assert guard.accepts(q, frames=frames) == guard.accepts(q)
        np.testing.assert_array_equal(guard.violation_frames(q, frames=frames), guard.violation_frames(q))
    adapter = PriorityContacts(guard, 'frozen')
    blocks = adapter.evaluate(poses, frames=frames)
    assert all(b.matrix.shape[1] == len(frames) * objective.width for b in blocks)
    inward = [b for b in blocks if b.identities[0][-1] == 'normal_inward']
    assert inward
    np.testing.assert_array_equal(inward[0].own, 0. if kind == 'acquisition' else .02)


def test_same_contact_channels_soc_values_match_prepared_order():
    from omniretargeting._fast.contact_guard import ContactGuard
    from tests.test_fast_contact_guard import _acquired
    objective, poses = _acquired(surfaces=True)

    class Empty:
        def measure(self, q, **kwargs):
            return dict(rows=[], depths={}, failures=[], complete=True)

        def linear_rows(self, measured, *, frames=None):
            count = len(poses) if frames is None else len(frames)
            return sparse.csr_matrix((0, count * objective.width)), np.empty(0)

    c = PriorityClearance(objective, Empty(), poses)
    objective.priority_clearance = c
    objective.candidate_guards = [c]
    ContactGuard(objective, poses)
    frames = range(2, 5)
    c.begin_iteration(poses, frames=frames)
    rows, cones = c.soft_rows(poses, frames=frames), c.soft_cones(poses, frames=frames)
    q = poses.copy()
    q[2:5, 2] += .0001
    values, norms = c.soft_values(q, frames=frames)
    delta = objective.coords.difference(poses, q)[2:5].ravel()
    assert len(values) == len(rows) and len(norms) == len(cones)
    for (matrix, upper, *_), actual in zip(rows, values):
        np.testing.assert_allclose(actual, upper - matrix @ delta, atol=1e-14)
    for (matrix, residual, upper, *_), (actual, bound) in zip(cones, norms):
        np.testing.assert_allclose(actual, residual + (matrix @ delta).reshape(residual.shape), atol=1e-14)
        np.testing.assert_array_equal(bound, upper)
    ContactGuard(objective, poses)
    assert c._window_state is None
    assert len(c.soft_cones(poses, frames=frames)) == 2 * len(cones)
    # With no penetrating donors every channel is hard, not an uncapped slack.
    assert not rows and not cones
    hard_rows, hard_cones = c.qp_bounds(poses, frames=frames)
    assert hard_rows and hard_cones


def test_parent_root_balls_are_checked_even_if_shape_accepts_only_layout():
    from omniretargeting._fast.priority_window import hard_failures
    from tests.test_fast_backend import _objective
    from omniretargeting._fast.reference import ReferenceShape
    objective, poses, _ = _objective()
    objective.candidate_guards = []
    objective.source_shape = ReferenceShape(objective, poses)
    c = SimpleNamespace(objective=objective)
    q = poses.copy()
    q[2, 2] += objective.source_shape.xy + .01
    # Uniform root translation changes no reference layout relations.
    assert objective.source_shape.accepts(q)
    assert 'reference_root_ball:0' in hard_failures(c, q, range(2, 3))
    objective.source_shape = None
    objective.root_budget = .001
    assert 'objective_root_ball' in hard_failures(c, q, range(2, 3))


def test_windowed_pipeline_initializes_reference_without_starting_geometry_workers(monkeypatch):
    from contextlib import ExitStack
    from dataclasses import replace
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.priority_geometry import acquisition_coordinator, PriorityGeometry
    from tests.test_fast_backend import _objective
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, windowed_priority=True))
    monkeypatch.setenv('OMNIRETARGETING_GPU_GEOMETRY', '0')
    starts = []
    monkeypatch.setattr(PriorityGeometry, 'start', lambda self: starts.append(self))
    objective, poses, _ = _objective()
    with ExitStack() as resources:
        c = acquisition_coordinator(objective, poses, (), None, resources)
        assert starts == []
        assert objective.priority_clearance is c
        assert c.provider.geometry.reference_observation_id is not None
        np.testing.assert_array_equal(c.reference, poses)
        assert c.current_measurement['depths'] is not None
        assert not c.provider.started


@pytest.mark.parametrize('reject_first_parent', [False, True])
def test_tiny_window_driver_runs_complete_private_and_parent_path(monkeypatch, reject_first_parent):
    from omniretargeting._fast import window_driver
    c, _ = case(8)
    obj = c.objective
    obj.coords.integrate = lambda q, d, a: q + a * d
    obj.project = lambda q: q.copy()

    def evaluate(q, with_jacobian=True, *, frames=None):
        frames = range(len(q)) if frames is None else frames
        part = q[frames.start:frames.stop]
        acc = SimpleNamespace(cost=float(np.square(part).sum()), gradient=part.copy(),
                              hessian=np.tile(np.eye(2), (len(frames), 1, 1)))
        return acc, np.empty(0), sparse.csr_matrix((0, len(frames) * 2)), {}, None

    obj.evaluate = evaluate
    monkeypatch.setattr(window_driver, 'WINDOW_SWEEPS', 1)
    monkeypatch.setattr(window_driver, 'sweep_windows', lambda n, s: [range(0, 4), range(4, 8)])
    monkeypatch.setattr(window_driver, 'fork_status', lambda: (False, 'synthetic_serial'))
    commits = []
    original_commit = c.commit

    def commit(q, **kwargs):
        commits.append(c.epoch)
        if reject_first_parent and len(commits) == 1:
            assert c.epoch == 0 and not c.spent
            np.testing.assert_array_equal(c.current, 0.)
            raise ValueError('Synthetic stale dependency: retry serially')
        return original_commit(q, **kwargs)

    monkeypatch.setattr(c, 'commit', commit)
    result, history = window_driver.optimize_windowed_priority(obj, c.current, 2, None, 'tiny', 1e-5)
    assert len(history) == 2 and all(row['alpha'] > 0 for row in history)
    assert c.epoch == 2 and c.current_depth.max() < 1e-7
    np.testing.assert_array_equal(result, c.current)
    assert all(row['optimality_certified'] is False for row in history)
    assert history[0]['serial_retry'] is reject_first_parent
    assert not history[1]['serial_retry']
    assert commits == ([0, 0, 1] if reject_first_parent else [0, 1])
    assert all(len(row['local_trace']) <= (4 if row['serial_retry'] else 2) for row in history)
