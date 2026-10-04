"""Temporal row caches retain numerical programs and mutable allowances."""

import numpy as np
import pytest

from omniretargeting._fast.repair_continuity_rows import continuity_bounds
from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from tests.test_fast_legacy_window import problem


def _equal(actual, expected):
    for family, reference in zip(actual, expected):
        assert len(family) == len(reference)
        for (a, low, high), (b, lo, hi) in zip(family, reference):
            assert a.shape == b.shape
            assert a.indptr.tobytes() == b.indptr.tobytes()
            assert a.indices.tobytes() == b.indices.tobytes()
            for first, second in ((a.data, b.data), (low, lo), (high, hi)):
                assert first.dtype == second.dtype
                assert first.tobytes() == second.tobytes()


@pytest.mark.parametrize("correction", [False, True])
def test_continuity_rows_match_original_after_joints_root_and_bound_changes(correction):
    from scipy.spatial.transform import Rotation
    objective, poses = problem(extra=True)
    rotations = Rotation.from_quat(poses[:, 3:7], scalar_first=True)
    guard = SequenceContinuityGuard(objective.tasks, poses, rotations, windowed=True,
                                    correction=correction)
    for kind in ("original", "joint", "root", "bound", "original"):
        trial = poses.copy()
        if kind == "joint":
            trial[15, objective.coords.qi[0]] += .003
        elif kind == "root":
            trial[15, :3] += .001
            trial[15, 3:7] = Rotation.from_rotvec([.001, .002, -.001]).as_quat(scalar_first=True)
        elif kind == "bound":
            guard.bounds["joints", 1] = guard.bounds["joints", 1] + .0001
        _equal(continuity_bounds(guard, trial), guard.repair_bounds(trial))
    assert all(len(cache) <= 2 for cache in guard._repair_continuity_cache.values())


def test_cached_native_guard_and_constant_joint_jacobian_reuse():
    objective, poses = problem(extra=True)
    guard = next(g for g in objective.candidate_guards
                 if isinstance(g, SequenceContinuityGuard))
    _equal(continuity_bounds(guard, poses), guard.repair_bounds(poses))
    matrix = next(iter(guard._repair_continuity_cache["joints", 1].values()))
    trial = poses.copy()
    trial[11, objective.coords.qi[0]] += .001
    _equal(continuity_bounds(guard, trial), guard.repair_bounds(trial))
    assert next(iter(guard._repair_continuity_cache["joints", 1].values())) is matrix


def test_custom_continuity_callback_keeps_original_dispatch(monkeypatch):
    objective, _ = problem(extra=True)
    guard = next(g for g in objective.candidate_guards
                 if isinstance(g, SequenceContinuityGuard))
    monkeypatch.setattr(guard, "repair_bounds", lambda *_: ([], []))
    assert continuity_bounds(guard, objective.reference) is None


def test_zero_stride_signature_is_compact_and_detects_underlying_mutation(monkeypatch):
    objective, poses = problem(extra=True)
    guard = next(g for g in objective.candidate_guards
                 if isinstance(g, SequenceContinuityGuard))
    signals = guard.signals(poses)
    values, jac = signals["joints"]
    base = jac[0].copy()
    signals = dict(signals, joints=(values, np.broadcast_to(base, jac.shape)))
    monkeypatch.setattr(guard, "signals", lambda _: signals)
    _equal(continuity_bounds(guard, poses), guard.repair_bounds(poses))
    cache = guard._repair_continuity_cache["joints", 1]
    key = next(iter(cache))
    assert len(key[-1]) == base.nbytes
    base[0, 0] += .001
    _equal(continuity_bounds(guard, poses), guard.repair_bounds(poses))
    assert len(cache) == 2
