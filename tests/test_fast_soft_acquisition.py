"""The bulk/projection boundary and parent commit remain transactional."""

import numpy as np

from omniretargeting._fast import soft_acquisition
from tests.test_fast_legacy_window import problem


def test_only_certified_projection_reaches_callback_and_polish(monkeypatch):
    objective, poses = problem()
    private = poses.copy()
    private[10, 7] += .3
    projected = poses.copy()
    projected[:, 7] += 1e-4
    callbacks, seen = [], {}
    monkeypatch.setattr(soft_acquisition, 'run_bulk', lambda *a, **k:
                        (private, [], {'seconds': .2}))
    monkeypatch.setattr(soft_acquisition, 'project_bulk', lambda *a, **k:
                        (projected, {'seconds': .1}))
    monkeypatch.setattr(soft_acquisition, 'commit_verdict', lambda *a, **k: {'accepted': True})

    def polish(obj, seed, **kwargs):
        seen.update(seed=seed.copy(), **kwargs)
        return poses.copy(), []

    monkeypatch.setattr(soft_acquisition, 'optimize_legacy_windows', polish)
    result, history, report = soft_acquisition.optimize_soft_acquisition(objective, poses, iterations=20,
        callback=lambda q, row: callbacks.append(q))
    np.testing.assert_array_equal(seen['seed'], projected)
    assert seen['sweep_iterations'] == (2,)
    assert seen['max_extra_sweeps'] == 0
    assert callbacks and all(np.array_equal(q, projected) for q in callbacks)
    assert not np.array_equal(private, callbacks[0])
    assert not report['guard_failures']


def test_parent_physical_cost_rejection_rolls_back_projection_before_polish(monkeypatch):
    objective, poses = problem()
    candidate = poses.copy()
    candidate[:, 7] += .001
    monkeypatch.setattr(soft_acquisition, 'run_bulk', lambda *a, **k:
                        (candidate, [], {'seconds': .2}))
    monkeypatch.setattr(soft_acquisition, 'project_bulk', lambda *a, **k:
                        (candidate, {'seconds': .1}))
    monkeypatch.setattr(soft_acquisition, 'commit_verdict', lambda *a, **k: {'accepted': False})

    def polish(obj, seed, **kwargs):
        np.testing.assert_array_equal(seed, poses)
        return seed.copy(), []

    monkeypatch.setattr(soft_acquisition, 'optimize_legacy_windows', polish)
    result, _, _ = soft_acquisition.optimize_soft_acquisition(objective, poses, iterations=20)
    np.testing.assert_array_equal(result, poses)
