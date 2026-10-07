"""Speculative repair is ordered replay, not stale-direction acceptance."""

from contextlib import closing
from copy import deepcopy
import os

import numpy as np
import pytest

from omniretargeting._fast import mesh_repair, repair_proposals
from omniretargeting._fast.repair_limits import RepairLimits
from omniretargeting._fast.repair_schedule import RepairSchedule
from tests.test_fast_backend import _objective
from tests.test_fast_mesh_repair import _open_object


def _problem(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    objective, poses, _ = _objective(temporal=True)
    geometry = mesh_repair.OriginalGeometry(objective, [_open_object(objective, poses)], None)
    measured = geometry.measure(poses, jacobians=True)
    limits = RepairLimits(objective, poses)
    limits.freeze_penetration_yield(measured)
    group = int(objective.model.body_weldid[objective.model.body("hand").id])
    return limits, geometry, poses, measured, {"depths": dict(measured["depths"])}, group


def test_peek_does_not_consume_attempts_or_predict_retries():
    schedule = RepairSchedule([1, 2, 3], 12, total_limit=3)
    assert schedule.next() == (1, 0)
    before = deepcopy(vars(schedule))
    assert schedule.peek_first(4) == [2, 3]
    assert vars(schedule) == before
    schedule.retry(1)
    assert schedule.peek_first(4) == [2]  # One remaining slot is reserved by the real retry.
    assert schedule.next() == (1, 1)
    assert schedule.total == 2


@pytest.mark.parametrize("dependency", ["pose", "anchor", "target", "protected", "witness",
                                        "frame_mask", "group", "reference", "box"])
def test_certificate_covers_global_and_mutable_dependencies(monkeypatch, dependency):
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    original = repair_proposals.certificate(limits, poses, measured, protected, None, .01)
    if dependency == "pose":
        poses[-1, -1] = np.nextafter(poses[-1, -1], 1.)
    elif dependency == "anchor":
        limits.anchors[0][-1, 0, 0] += 1e-12
    elif dependency == "target":
        limits.target_budgets[0][-1, 0] += 1e-12
    elif dependency == "protected":
        key = next(iter(protected["depths"]))
        protected["depths"][key] += 1e-12
    elif dependency == "witness":
        measured["rows"][0][1]["direction"][0] += 1e-12
    elif dependency == "frame_mask":
        limits.frame_mask = np.ones(len(poses), bool)
    elif dependency == "group":
        limits.movable_body = group
    elif dependency == "reference":
        limits.objective.reference = limits.objective.reference.copy()
        limits.objective.reference[-1, -1] += 1e-12
    else:
        limits.lower[0, 0] += 1e-12
    assert original != repair_proposals.certificate(limits, poses, measured, protected, None, .01)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork unavailable")
def test_worker_and_serial_candidate_streams_are_byte_equal(monkeypatch):
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    schedule = RepairSchedule([group], 12)
    workers = repair_proposals.RepairProposals(limits, geometry, count=1)
    try:
        assert len(workers.workers) == 1
        workers.prefetch(schedule, poses, measured, protected)
        assert group in workers.pending
        repair_proposals.transition_group(limits, poses, group)
        limits.frame_mask = repair_proposals.group_frames(measured, geometry.model, group, limits.frames,
                                                          repair_proposals.halo_frames(30.))
        depths = [v for k, v in measured["depths"].items() if any(
            geometry.model.body_weldid[geometry.model.geom_bodyid[g]] == group
            for g in (k[2:] if k[1] == "self" else k[-1:]))]
        depth = max(depths)
        remote = workers.take(poses, measured, protected, None, depth)
        assert workers.stats["reused"] == 1
        serial = repair_proposals.LocalProposal(limits, geometry, poses, measured, protected, None, depth)
        assert (remote.status, remote.has_direction, remote.stationary) == (
            serial.status, serial.has_direction, serial.stationary)
        a, b = {}, {}
        with closing(remote.candidates(a)) as remote_values:
            actual = list(remote_values)
        with closing(serial.candidates(b)) as serial_values:
            expected = list(serial_values)
        assert len(actual) == len(expected) and actual
        for (alpha, candidate), (fraction, reference) in zip(actual, expected):
            assert alpha == fraction
            assert candidate.tobytes() == reference.tobytes()
            assert np.array_equal(candidate[:, :7], poses[:, :7])
        assert a == b
    finally:
        workers.close(abort=True)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork unavailable")
def test_stale_anchor_is_recomputed_instead_of_accepted(monkeypatch):
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    workers = repair_proposals.RepairProposals(limits, geometry, count=1)
    schedule = RepairSchedule([group], 12)
    try:
        workers.prefetch(schedule, poses, measured, protected)
        repair_proposals.transition_group(limits, poses, group)
        limits.frame_mask = repair_proposals.group_frames(measured, geometry.model, group, limits.frames,
                                                          repair_proposals.halo_frames(30.))
        limits.anchors[0][-1, 0, 0] += 1e-12
        depth = max(v for k, v in measured["depths"].items() if any(
            geometry.model.body_weldid[geometry.model.geom_bodyid[g]] == group
            for g in (k[2:] if k[1] == "self" else k[-1:])))
        job = workers.take(poses, measured, protected, None, depth)
        assert isinstance(job, repair_proposals.LocalProposal)
        assert workers.stats["stale"] == 1 and workers.stats["reused"] == 0
    finally:
        workers.close(abort=True)


def test_worker_geometry_has_no_query_capability(monkeypatch):
    _, geometry, _, _, _, _ = _problem(monkeypatch)
    facade = repair_proposals.GeometryRows(geometry)
    assert not any(hasattr(facade, name) for name in ("measure", "moving", "scene", "parallel", "data"))




@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork unavailable")
def test_accepted_change_aborts_unconsumed_cpu_jobs_and_refills(monkeypatch):
    import time

    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    pass

    def pending(limits, geometry, connection, inherited):
        connection.recv()
        time.sleep(600)  # No header arrives; parent must not wait to discard it.

    monkeypatch.setattr(repair_proposals, "_serve", pending)
    schedule = RepairSchedule([group], 12)
    workers = repair_proposals.RepairProposals(limits, geometry, count=1)
    try:
        workers.prefetch(schedule, poses, measured, protected)
        old_process, old_connection = workers.pending[group].slot
        assert not workers._prefetch_triage
        workers.note_outcome(False)
        assert workers.pending
        workers.note_outcome(True)
        assert not workers.pending and not workers.workers
        assert not old_process.is_alive() and old_connection.closed
        assert workers.stats["aborted_stale"] == workers.stats["stale"] == 1
        assert workers.stats["discarded"] == 1
        workers.prefetch(schedule, poses, measured, protected)
        assert group in workers.pending and len(workers.workers) == 1
        assert workers.workers[0][0].pid != old_process.pid
    finally:
        workers.close(abort=True)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork unavailable")
def test_automatic_workers_wait_for_consecutive_unchanged_attempts(monkeypatch):
    monkeypatch.setattr(repair_proposals, "MIN_PARALLEL_FRAMES", 0)
    monkeypatch.setattr(repair_proposals, "available_cpus", lambda: 2)
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    schedule = RepairSchedule([group], 12)
    workers = repair_proposals.RepairProposals(limits, geometry)
    try:
        assert workers.capacity == 1 and not workers.workers
        for changed in (True, False, True, False):
            workers.note_outcome(changed)
            workers.prefetch(schedule, poses, measured, protected)
            assert not workers.workers and not workers.pending
        workers.note_outcome(False)
        workers.prefetch(schedule, poses, measured, protected)
        assert len(workers.workers) == 1 and group in workers.pending
        assert workers.stats["launched"] == 1
        workers.note_outcome(True)
        assert workers.stable_attempts == 0
    finally:
        workers.close(abort=True)
