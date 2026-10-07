"""Disjoint CPU proposal footprints retain ordered, exact parent checks."""
from contextlib import closing
from copy import deepcopy
import os

import numpy as np
import pytest

from omniretargeting._fast import mesh_repair, repair_parallel_groups as parallel, repair_proposals
from omniretargeting._fast.repair_parallel_scope import Footprint, certificate, footprint
from omniretargeting._fast.repair_schedule import RepairSchedule
from tests.test_repair_proposals import _problem


def test_default_off_and_worker_limit():
    pass
    assert parallel.MAX_WORKERS == 8


def test_read_context_blocks_other_limbs_and_third_difference_boundaries():
    def scope(frame, column):
        write = np.zeros((20, 2), bool)
        write[frame, column] = True
        read = np.zeros(20, bool)
        read[max(0, frame-3):min(20, frame+4)] = True
        return Footprint(write, read, frozenset([column]))
    a = scope(5, 0)
    assert not a.disjoint(scope(8, 1))  # Different limbs, shared layout/stencil reads.
    assert a.disjoint(scope(9, 0))
    assert a.disjoint(scope(9, 1))
    assert a.disjoint(scope(9, 1)) == scope(9, 1).disjoint(a)


def test_projection_and_descendants_are_the_existing_self_contact_closure(monkeypatch):
    from omniretargeting._fast.repair_frames import restrict
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    limits.movable_body = group
    limits.frame_mask = np.zeros(len(poses), bool)
    limits.frame_mask[5] = True
    scope = footprint(limits, geometry.model, measured)
    projection = restrict(mesh_repair._joint_projection(limits, geometry.model, measured), limits.frame_mask)
    assert set(np.flatnonzero(scope.write)) == set(projection.indices)
    assert np.array_equal(np.flatnonzero(scope.read_frames), np.arange(2, 9))
    assert group in scope.bodies
    assert not scope.write[:, :].sum() == 0


@pytest.mark.parametrize('kind', ['pose', 'anchor', 'target', 'witness', 'protected', 'box', 'reference'])
def test_scoped_certificate_tracks_every_local_and_immutable_dependency(monkeypatch, kind):
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    limits.movable_body = group
    limits.frame_mask = np.zeros(len(poses), bool)
    limits.frame_mask[5] = True
    scope = footprint(limits, geometry.model, measured)
    initial = certificate(limits, poses, measured, protected, None, .01, scope)
    if kind == 'pose':
        poses[5, -1] += 1e-12
    elif kind == 'anchor':
        limits.anchors[0][5, 0, 0] += 1e-12
    elif kind == 'target':
        limits.target_budgets[0][5, 0] += 1e-12
    elif kind == 'witness':
        next(row for key, row in measured['rows'] if key[0] == 5)['clearance'] += 1e-12
    elif kind == 'protected':
        key = next(key for key in protected['depths'] if key[0] == 5)
        protected['depths'][key] += 1e-12
    elif kind == 'box':
        limits.lower[-1, -1] += 1e-12
    else:
        limits.objective.reference = limits.objective.reference.copy()
        limits.objective.reference[-1, -1] += 1e-12
    assert certificate(limits, poses, measured, protected, None, .01, scope) != initial


def test_unrelated_frame_commit_and_anchor_transition_survive(monkeypatch):
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    limits.objective.reference = limits.objective.reference.copy()
    limits.movable_body = group
    limits.frame_mask = np.zeros(len(poses), bool)
    limits.frame_mask[5] = True
    scope = footprint(limits, geometry.model, measured)
    initial = certificate(limits, poses, measured, protected, None, .01, scope)
    poses[-1, -1] += 1e-12
    limits.anchors[0][-1, 0, 0] += 1e-12
    limits.target_budgets[0][-1, 0] += 1e-12
    key = next(key for key in protected['depths'] if key[0] == len(poses)-1)
    protected['depths'][key] += 1e-12
    assert certificate(limits, poses, measured, protected, None, .01, scope) == initial




@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork')
def test_persistent_worker_ordered_stream_and_stale_resolve(monkeypatch):
    from omniretargeting._fast import repair_prediction
    # This test compares the candidate protocol even when the production
    # predictor would consume its header without requesting candidates.
    monkeypatch.setattr(repair_prediction, 'PREDICTIVE_GATING', False)
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    # Exercise the actual worker protocol and certificate with frame-disjoint
    # scopes. The full-motion guard check is still the production check.
    monkeypatch.setattr(parallel, 'halo_frames', lambda fps: 0)
    selected = deepcopy(measured)
    selected['rows'] = [(key, row) for key, row in selected['rows'] if key[0] == 5]
    selected['depths'] = {key: d for key, d in selected['depths'].items() if key[0] == 5}
    schedule = RepairSchedule([group], 12)
    pool = parallel.ParallelGroups(limits, geometry, count=2)
    limits.movable_body = None
    limits.frame_mask = np.zeros(len(poses), bool)
    # Empty head footprint isolates this lookahead protocol test.
    try:
        pool._dispatch(schedule, poses, selected, protected)
        assert group in pool.pending and len(pool.workers) == 1
        pids = [process.pid for process, _ in pool.workers]
        repair_proposals.transition_group(limits, poses, group)
        limits.frame_mask[5] = True
        depth = max(selected['depths'].values())
        remote = pool.take(poses, selected, protected, None, depth)
        serial = repair_proposals.LocalProposal(limits, geometry, poses, selected, protected, None, depth)
        assert (remote.status, remote.stationary) == (serial.status, serial.stationary)
        a, b = {}, {}
        with closing(remote.candidates(a)) as stream:
            actual = list(stream)
        with closing(serial.candidates(b)) as stream:
            expected = [] if serial.stationary else list(stream)
        assert len(actual) == len(expected)
        for (alpha, candidate), (fraction, reference) in zip(actual, expected):
            assert alpha == fraction and candidate.tobytes() == reference.tobytes()
        if not serial.stationary:
            assert a == b
        assert [process.pid for process, _ in pool.workers] == pids
        # Relaunch the same ready group; modify one read dependency before take.
        limits.movable_body = None
        limits.frame_mask[:] = False
        pool._dispatch(schedule, poses, selected, protected)
        repair_proposals.transition_group(limits, poses, group)
        limits.frame_mask[5] = True
        limits.anchors[0][5, 0, 0] += 1e-12
        rerun = pool.take(poses, selected, protected, None, depth)
        assert isinstance(rerun, repair_proposals.LocalProposal)
        assert pool.stats['stale'] == pool.stats['re_solved'] == 1
        assert pool.stats['reused'] == 1
        assert not pool.pending
    finally:
        processes = [process for process, _ in pool.workers]
        pool.close(abort=True)
        assert all(not process.is_alive() for process in processes)


def test_complete_parallel_switch_retains_exact_parent_acceptance(monkeypatch):
    from tests.test_fast_backend import _objective
    from tests.test_fast_mesh_repair import _open_object
    def run(enabled):
        pass
        objective, poses, _ = _objective(temporal=True)
        return mesh_repair.repair_original_meshes(objective, poses, [_open_object(objective, poses)])
    baseline, off = run(False)
    candidate, on = run(True)
    assert baseline.tobytes() == candidate.tobytes()
    assert on['before'] == off['before'] and on['after'] == off['after']
    assert on['limits'] == off['limits']


def test_pre_dispatch_fast_gate_is_cached_and_launches_nothing(monkeypatch):
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    limits.movable_body = group
    fast = {'predicted_decrease_m': 0., 'model': 'reduced_frame_qp'}
    seen = []
    def gates(*args):
        seen.append(True)
        return None, fast
    def forbidden(*args):
        raise AssertionError('Gated head must build neither workers nor a QP')
    for module in (parallel, repair_proposals):
        pass
    monkeypatch.setattr(repair_proposals, 'gate_predictions', gates)
    monkeypatch.setattr(parallel.ParallelGroups, '_dispatch', forbidden)
    monkeypatch.setattr(mesh_repair, '_qp_step', forbidden)
    pool = parallel.ParallelGroups(limits, geometry, count=2)
    pool.prefetch(RepairSchedule([group], 12), poses, measured, protected)
    proposal = pool.take(poses, measured, protected, None, .01)
    assert proposal.status == 'reduced_predicted_small_progress'
    assert seen == [True] and not pool.workers
    assert '_repair_gate_cache' not in vars(limits)


def test_paused_stream_preserves_unrelated_commit_and_parent_check(monkeypatch):
    from types import SimpleNamespace
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    limits.objective.reference = limits.objective.reference.copy()
    limits.movable_body = group
    limits.frame_mask = np.zeros(len(poses), bool)
    limits.frame_mask[5] = True
    scope = footprint(limits, geometry.model, measured)
    candidate = poses.copy()
    column = int(np.flatnonzero(scope.write[5])[0])
    candidate[5, limits.coords.qi[column]] += 1e-10
    packets = [('header', (True, False, 'Solved', None), {}, [], .1),
               ('candidate', (1., candidate), {'worker': 'first'}, [], .2),
               ('done', None, {}, [], .3)]
    class Connection:
        def __init__(self):
            self.received, self.sent = 0, []
        def recv(self):
            self.received += 1
            return packets.pop(0)
        def send(self, message):
            self.sent.append(message)
    connection = Connection()
    stats = dict(wait_seconds=0., useful_cpu_seconds=0., discarded_cpu_seconds=0., discarded=0)
    pool = SimpleNamespace(limits=limits, stats=stats)
    job = parallel.ParallelProposal(pool, (None, connection), None, poses, scope)
    job.original = poses
    job.pause()
    assert connection.received == 2 and job.paused
    actual = poses.copy()
    actual[-1, -1] += 1e-10
    job.current, job.consumed = actual, True
    job.start()
    assert connection.received == 2
    with closing(job.candidates({})) as values:
        result = list(values)
    assert len(result) == 1
    assert result[0][1][5, limits.coords.qi[column]] == candidate[5, limits.coords.qi[column]]
    assert result[0][1][-1, -1] == actual[-1, -1]
    assert connection.sent == ['next'] and connection.received == 3
    assert limits.check(result[0][1])[0]


def test_custom_guard_and_invalid_worker_count(monkeypatch):
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    with pytest.raises(ValueError):
        parallel.ParallelGroups(limits, geometry, count=9)
    class UnknownGuard:
        pass
    limits.objective.candidate_guards = [UnknownGuard()]
    pool = parallel.ParallelGroups(limits, geometry, count=2)
    assert pool.capacity == 0 and 'custom guard' in pool.stats['disabled_reason']
