"""Actual local QPs and R1 net commits, including process and legacy routing."""

from dataclasses import replace
from types import SimpleNamespace
import os

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.window_driver import optimize_windowed_priority
from omniretargeting._fast.window_local import solve_local_window
from tests.test_priority_window import case


def problem(count=8):
    coordinator, _ = case(count)
    objective = coordinator.objective
    objective.coords.integrate = lambda p, d, alpha: p + alpha * d
    objective.project = lambda p: np.clip(p, -1., 1.)
    objective.stop_workers = lambda: None
    # Opposing tracking cost: exact penetration descent must win.
    def evaluate(poses, with_jacobian=True, *, frames):
        part = poses[frames.start:frames.stop]
        residual = part - np.array([-.1, 0.])
        acc = SimpleNamespace(cost=float(np.sum(residual ** 2)),
                              hessian=np.tile(np.eye(2), (len(frames), 1, 1)), gradient=residual)
        return acc, np.empty(0), sparse.csr_matrix((0, part.size)), {}, None
    objective.evaluate = evaluate
    return objective, coordinator.current.copy()


def test_local_complete_search_cannot_write_parent_ledger():
    objective, poses = problem()
    parent = objective.priority_clearance
    result = solve_local_window(objective, poses, range(2, 4), 8, .003)
    assert result['changed'] and result['trace'][0]['alpha'] == 1.
    assert result['trace'][0]['after'] > result['trace'][0]['cost']
    assert parent.epoch == 0 and not parent.spent and not parent.yields
    np.testing.assert_array_equal(parent.current, poses)
    assert result['transactions']['credit_authority'] is False


def test_driver_commits_net_progress_and_preserves_default_route(monkeypatch):
    from omniretargeting._fast import solver, runtime_options, window_driver
    objective, poses = problem()
    monkeypatch.setattr(window_driver, 'fork_status', lambda: (False, 'test_serial'))
    starts = []
    objective.start_workers = lambda: starts.append('legacy') or False
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, windowed_priority=True))
    solved, history = solver.optimize(objective, poses, iterations=2)
    assert not starts
    assert objective.priority_clearance.epoch >= 1
    assert history[0]['alpha'] == 1.
    assert history[0]['priority_status'] == 'accepted_window'
    assert objective.priority_clearance.current_depth.max() < 1e-7
    np.testing.assert_array_equal(solved, objective.priority_clearance.current)
    assert all(not row['optimality_certified'] for row in history)
    # Off is exactly the existing internal path, including worker lifecycle.
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, windowed_priority=False))
    monkeypatch.setattr(solver, '_optimize', lambda *a: ('legacy-result', []))
    assert solver.optimize(objective, poses, iterations=1)[0] == 'legacy-result'
    assert starts == ['legacy']


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork')
def test_parallel_children_commit_only_in_parent_stable_window_order(monkeypatch):
    from omniretargeting._fast import window_driver
    objective, poses = problem(80)
    parent = objective.priority_clearance
    calls = []
    original = parent.commit
    def commit(q, **kwargs):
        calls.append((os.getpid(), kwargs['frames'].start))
        return original(q, **kwargs)
    parent.commit = commit
    monkeypatch.setattr(window_driver, 'fork_status', lambda: (True, None))
    solved, history = optimize_windowed_priority(objective, poses, 1, None, 'test', .003)
    assert any(row['parallel'] for row in history)
    assert calls and all(pid == os.getpid() for pid, _ in calls)
    assert parent.epoch == sum(row['alpha'] > 0 for row in history)
    assert not any(row['serial_retry'] for row in history)
    assert parent.current_depth.max() < 1e-7
    np.testing.assert_array_equal(solved, parent.current)


def test_parent_rejection_resolves_once_serially(monkeypatch):
    from omniretargeting._fast import window_driver
    objective, poses = problem()
    original = window_driver._verify_and_commit
    calls = []
    def reject_first(*args):
        calls.append(1)
        return (None, {'parent_reason': 'test_stale'}) if len(calls) == 1 else original(*args)
    monkeypatch.setattr(window_driver, '_verify_and_commit', reject_first)
    solved, history = optimize_windowed_priority(objective, poses, 1, None, 'test', .003)
    assert history[0]['serial_retry'] and history[0]['alpha'] == 1.
    assert objective.priority_clearance.epoch > 0


def test_parent_prepares_geometry_before_any_process_or_window(monkeypatch):
    from omniretargeting._fast import window_driver
    objective, poses = problem()
    events = []
    objective.priority_clearance.provider.geometry = SimpleNamespace(
        scene=None, moving=None, self_query=None, prepare_fork_queries=lambda: events.append('prepare'))
    original = window_driver.solve_batch
    def batch(*args):
        events.append('batch')
        assert events[0] == 'prepare'
        return original(*args)
    monkeypatch.setattr(window_driver, 'solve_batch', batch)
    optimize_windowed_priority(objective, poses, 1, None, 'test', .003)
    assert events.count('prepare') == 1


def test_objective_warmup_runs_once_in_parent_before_geometry_and_windows(monkeypatch):
    from omniretargeting._fast import window_driver
    objective, poses = problem(80)
    owner, events = os.getpid(), []
    def prepare():
        assert os.getpid() == owner
        events.append('objective')
        return objective
    objective.prepare_window_evaluation = prepare
    objective.priority_clearance.provider.geometry = SimpleNamespace(
        scene=None, moving=None, self_query=None, prepare_fork_queries=lambda: events.append('geometry'))
    original = window_driver.solve_batch
    def batch(*args):
        assert events[:2] == ['objective', 'geometry']
        events.append('batch')
        return original(*args)
    monkeypatch.setenv('IGL_NUM_THREADS', '1')
    monkeypatch.setattr(window_driver, 'fork_status', lambda: (True, None))
    monkeypatch.setattr(window_driver, 'solve_batch', batch)
    _, history = optimize_windowed_priority(objective, poses, 1, None, 'test', .003)
    assert events.count('objective') == events.count('geometry') == 1
    assert any(row['parallel'] for row in history)
