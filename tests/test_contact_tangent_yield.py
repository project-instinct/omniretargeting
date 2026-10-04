"""Own-body clearance credit applies to tangent precision as to other channels."""

import numpy as np
import pytest
from scipy import sparse
from types import SimpleNamespace

from omniretargeting._fast.contact_guard import ContactGuard
from omniretargeting._fast.priority_clearance import PriorityClearance
from tests.test_fast_contact_guard import _acquired


def elbow_case(depth, *, priority=False):
    obj, poses = _acquired()
    arm = obj.model.body('arm').id
    geom = int(np.flatnonzero(obj.model.geom_bodyid == arm)[0])
    for effector in obj.effectors:
        effector['active'][:] = False
    contact = obj.effectors[-1]
    contact['body'] = arm
    contact['active'][:] = True
    contact['surface_only'][:] = True
    contact['normal'][:] = [0., 0., 1.]
    obj.tasks.contact_phase = SimpleNamespace(role=lambda name, f: 'hold', surface=None, surface_roles={})
    contact['contact'] = obj.kin.points(np.full((len(poses), 1), arm), contact['offset'][:, None])[:, 0].copy()
    own = np.zeros((len(poses), obj.model.nbody))
    own[:, obj.model.body_weldid[arm]] = depth
    obj.frozen_original_depth = lambda q: own.copy()
    obj.source_shape = None
    obj.candidate_guards = []
    obj.root_budget = None  # This fixture isolates contact yield, not root locality.
    provider = None
    if priority:
        class Geometry:
            def measure(self, q, *, frames=None, jacobians=False):
                frames = range(len(q)) if frames is None else frames
                rows = [((f, 'static', geom), dict(clearance=q[f, 2] - poses[f, 2] - depth)) for f in frames]
                return dict(rows=rows, depths={k: max(0., -v['clearance']) for k, v in rows},
                            complete=True, failures=[])

            def linear_rows(self, measured, *, frames=None):
                frames = range(len(poses)) if frames is None else frames
                rows = measured['rows']
                columns = [(k[0] - frames.start) * obj.width + obj.coords.inverse[obj.coords.root_d + 2]
                           for k, _ in rows]
                return (sparse.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)), columns)),
                                          shape=(len(rows), len(frames) * obj.width)),
                        np.array([v['clearance'] for _, v in rows]))

        provider = Geometry()
        coordinator = PriorityClearance(obj, provider, poses)
        obj.priority_clearance, obj.candidate_guards = coordinator, [coordinator]
    return obj, poses, ContactGuard(obj, poses), provider


@pytest.mark.parametrize('depth,accepted', [(.05, True), (0., False)])
def test_standalone_tangent_yield_and_qp_agree_with_own_penetration(depth, accepted):
    obj, poses, guard, _ = elbow_case(depth)
    candidate = poses.copy()
    candidate[:, 0] += .003
    candidate[:, 2] += depth
    assert guard.accepts(candidate) is accepted
    assert guard.accepts(candidate, frames=range(2, 5)) is accepted
    linear, cones = guard.repair_bounds(poses)
    step = obj.coords.difference(poses, candidate).ravel()
    feasible = (all(np.all(A @ step >= lo) and np.all(A @ step <= hi) for A, lo, hi in linear)
                and all(np.all(np.linalg.norm(r + (A @ step).reshape(-1, 3), axis=1) <= hi)
                        for A, r, hi in cones))
    assert feasible is accepted
    tangent = cones[1]
    assert np.max(tangent[2]) == pytest.approx(.001 + depth + 5e-8)
    np.testing.assert_array_equal(guard.groups[0].yielded, depth)
    # No renewed allowance after an accepted check; exceeding own depth fails.
    bad = poses.copy()
    bad[:, 0] += depth + .0011
    assert not guard.accepts(bad)


@pytest.mark.parametrize('depth,accepted', [(.05, True), (0., False)])
def test_window_tangent_B_and_parent_acceptance_use_same_frozen_own_allowance(depth, accepted):
    obj, poses, guard, _ = elbow_case(depth, priority=True)
    c, frames = obj.priority_clearance, range(2, 5)
    c.begin_iteration(poses, frames=frames)
    state = c._window_state
    i = next(i for i, b in enumerate(state.blocks) if b.identities[0][-1] == 'tangent')
    block = state.blocks[i]
    np.testing.assert_array_equal(block.own, depth)
    np.testing.assert_allclose(state.contact_domain.bounds[i], block.bound + depth)
    # Same-body yield belongs in B, not in new cross-body epigraph credit.
    assert not c.soft_limits(poses, frames=frames)
    candidate = poses.copy()
    candidate[2:5, 0] += .003
    candidate[2:5, 2] += depth
    report = c.evaluate_trial(candidate, frames=frames)['details']
    assert report['accepted'] is accepted
    if accepted:
        tangent = [r for r in report['yields'] if r['contact'][-1] == 'tangent']
        assert len(tangent) == 3
        assert all(r['authorization'] == 'frozen_own_body' and r['donor_body'] == r['body'] for r in tangent)
        c.commit(candidate, frames=frames)
        assert not c.spent and c.epoch == 1
        assert not c.current_depth[2:5].any()
    else:
        assert any(r['contact'][-1] == 'tangent' for r in report['unfunded_contacts'])
        assert c.epoch == 0 and not c.spent


def test_clean_contact_does_not_inherit_another_bodys_tangent_allowance():
    obj, poses, _, _ = elbow_case(0.)
    other = np.zeros((len(poses), obj.model.nbody))
    other[:, obj.model.body_weldid[obj.model.body('foot').id]] = .05
    obj.frozen_original_depth = lambda q: other.copy()
    guard = ContactGuard(obj, poses)
    assert not guard.groups[0].yielded.any()
    candidate = poses.copy()
    candidate[:, 0] += .003
    assert not guard.accepts(candidate)
