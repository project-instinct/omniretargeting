"""A failing external progress callback cannot lose committed poses/history."""

import json

import numpy as np
import pytest

from omniretargeting._fast.window_driver import optimize_windowed_priority
from tests.test_window_driver import problem


def test_callback_failure_happens_after_batch_history_ledger_and_poses_are_durable(tmp_path, monkeypatch):
    from omniretargeting._fast import window_driver
    objective, poses = problem(80)
    original = objective.evaluate
    def evaluate(*args, **kwargs):
        acc, r, j, stats, extra = original(*args, **kwargs)
        return acc, r, j, dict(stats, sticking_max_m=.125), extra
    objective.evaluate = evaluate
    path = tmp_path / 'windows.jsonl'
    monkeypatch.setenv('OMNIRETARGETING_WINDOW_TRACE', str(path))
    monkeypatch.setattr(window_driver, 'fork_status', lambda: (True, None))
    def callback(q, record):
        for key in ('phase', 'iteration', 'cost', 'after', 'alpha', 'damping', 'step_backend', 'seconds'):
            assert key in record
        assert record['sticking_max_m'] == .125
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[-1]['event'] == 'batch_complete'
        windows = [row for row in rows if row['event'] == 'window']
        assert windows and windows[-1]['ledger']['epoch'] == objective.priority_clearance.epoch
        checkpoint = np.load(path.with_suffix('.committed.npz'), allow_pickle=False)
        np.testing.assert_array_equal(checkpoint['poses'], objective.priority_clearance.current)
        metadata = json.loads(str(checkpoint['metadata']))
        assert metadata['ledger']['epoch'] == objective.priority_clearance.epoch
        assert 'spent' in metadata['ledger'] and 'yields' in metadata['ledger']
        raise RuntimeError('external callback failed')
    with pytest.raises(RuntimeError, match='external callback failed'):
        optimize_windowed_priority(objective, poses, 1, callback, 'contact_scene', .003)
    assert json.loads(path.read_text().splitlines()[-1])['event'] == 'callback_error'
    assert not list(tmp_path.glob('*.tmp'))


def test_window_geometry_calls_never_drop_frames(monkeypatch):
    from tests.test_priority_window import Provider
    objective, poses = problem(40)
    original = Provider.measure
    def measure(self, p, *, frames=None, **kwargs):
        assert isinstance(frames, range) and 0 < len(frames) <= 32
        return original(self, p, frames=frames, **kwargs)
    monkeypatch.setattr(Provider, 'measure', measure)
    _, history = optimize_windowed_priority(objective, poses, 1, None, 'test', .003)
    assert history and any(row['alpha'] > 0 for row in history)
    for row in history:
        assert all(k in row for k in ('phase', 'iteration', 'cost', 'after', 'alpha', 'damping',
                                     'step_backend', 'seconds'))
        for local in row['local_trace']:
            for trial in local['trial_trace']:
                assert 'details' in trial and 'penetration_multiplier' in trial
            for qp in local['qp_trace']:
                assert qp['variables'] > 0 and qp['rows'] <= qp['original_rows']
                assert all(attempt['construct_seconds'] >= 0 and attempt['solve_seconds'] >= 0
                           for attempt in qp['solver_attempts'])


def test_mixed_objective_stats_survive_local_parent_history_callback_and_jsonl(tmp_path):
    objective, poses = problem()
    original = objective.evaluate
    diagnostics = dict(collision_proxy='skipped', proxy_collision_terms_skipped=True,
                       clearance_diagnostics_owner='priority_clearance_original_geometry',
                       scene_depth_m=None, moving_depth_m=None, self_depth_m=None,
                       collision_diagnostics={'source': 'original_geometry', 'unknown': None})
    def evaluate(*args, **kwargs):
        acc, r, j, stats, extra = original(*args, **kwargs)
        return acc, r, j, dict(stats, **diagnostics), extra
    objective.evaluate = evaluate
    callbacks = []
    from unittest.mock import patch
    path = tmp_path / 'mixed_stats.jsonl'
    with patch.dict('os.environ', OMNIRETARGETING_WINDOW_TRACE=str(path)):
        _, history = optimize_windowed_priority(objective, poses, 1,
                                                lambda _, record: callbacks.append(record), 'test', .003)
    saved = [json.loads(line)['record'] for line in path.read_text().splitlines()
             if json.loads(line)['event'] == 'window']
    assert callbacks and history and saved
    local = [item['stats'] for row in history for item in row['local_trace']]
    assert local
    for record in [*history, *callbacks, *saved, *local]:
        for key, value in diagnostics.items():
            assert record[key] == value
