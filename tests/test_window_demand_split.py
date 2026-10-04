"""Continuation splits conserve allowances and keep crossing constraints."""
from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast import window_demand_split as splits
from omniretargeting._fast.legacy_window import WindowState, _task
from tests.test_fast_legacy_window import scheduling_state


@pytest.mark.parametrize('quota', [2, 3, 4])
@pytest.mark.parametrize('shift', [-7, 7])
def test_splits_cover_core_keep_halos_and_conserve_allowance(monkeypatch, quota, shift):
    monkeypatch.setattr(splits, 'WINDOW_BACKTRACK_SPLITS', True)
    state, components = scheduling_state(converged=False)
    state.record['local_alphas'] = [.25, 1., .125, .25]
    follow = _task(state.task.core, 300, components, ('unfinished',), shift=shift)
    result = splits.split_followups([follow], [state], 300, components, quota, 32)
    assert [task.core for task in result] == [range(80, 112), range(112, 144)]
    assert sum(task.iteration_quota for task in result) == quota
    assert all(0 < task.iteration_quota <= quota for task in result)
    for task in result:
        assert task.core.start - task.frames.start == follow.core.start - follow.frames.start
        assert task.frames.stop - task.core.stop == follow.frames.stop - follow.core.stop
        assert task.footprint.start == task.frames.start - 3
        assert task.footprint.stop == task.frames.stop + 3
        assert 'backtrack_demand_split' in task.reasons
    state.record['window_seconds'] = 1e30
    assert result == splits.split_followups([follow], [state], 300, components, quota, 32)


@pytest.mark.parametrize('reason', ['off', 'retry', 'short', 'quota', 'easy', 'rejected'])
def test_fallbacks_preserve_original_tasks(monkeypatch, reason):
    monkeypatch.setattr(splits, 'WINDOW_BACKTRACK_SPLITS', reason != 'off')
    state, components = scheduling_state(converged=False)
    state.record['local_alphas'] = [.25, .125, .25, .125]
    if reason == 'easy':
        state.record['local_alphas'] = [.5, 1., .5, .25]
    if reason == 'rejected':
        state.record['local_alphas'][-1] = 0.
    task = _task(state.task.core, 300, components, ('unfinished',),
                 retries=(state,) if reason == 'retry' else ())
    plan = [task]
    result = splits.split_followups(plan, [state], 300, components,
                                   1 if reason == 'quota' else 4,
                                   33 if reason == 'short' else 32)
    assert result == plan


def test_real_lm_split_keeps_guards_references_and_outside_bytes(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.legacy_window import _commit, _failures
    from omniretargeting._fast.legacy_window_qp import WindowObjective, solve_local
    from omniretargeting._fast.proposals import guards_of
    from omniretargeting._fast.runtime_options import RuntimeOptions
    from omniretargeting._fast.window_execution import execute
    from omniretargeting._fast.window_schedule import independent_batches
    from tests.test_fast_legacy_window import problem
    monkeypatch.setattr(splits, 'WINDOW_BACKTRACK_SPLITS', True)
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    objective, poses = problem(extra=True)
    components = [objective, *guards_of(objective)[1]]
    task = _task(range(8, 24), len(poses), components, ('test',), halo=3)
    state = WindowState(task, {'local_alphas': [.25] * 4}, np.zeros(len(task.footprint)))
    plan = splits.split_followups([task], [state], len(poses), components, 5, 8)
    reference = objective.reference.tobytes()
    bounds = {k: v.tobytes() for k, v in objective.candidate_guards[0].bounds.items()}
    for part in plan:
        local = WindowObjective(objective, poses, part.frames)
        for amplitude in (0., .001, .4):
            trial = poses[local.span].copy()
            trial[0, 7] += amplitude
            full = local.embed(trial)
            for wrapped, guard in zip(guards_of(local)[1], guards_of(objective)[1]):
                assert wrapped.accepts(trial) == guard.accepts(full)
    consumed = []

    def solve(obj, snapshot, frames, iterations, convergence):
        consumed.append(iterations)
        return solve_local(obj, snapshot, frames, iterations, convergence)

    current, _, _, _ = execute(SimpleNamespace(jobs=[], objective=objective), poses,
        independent_batches(plan), 5, .003,
        lambda current, snapshot, part, result: _commit(objective, current, snapshot, part, result), solve)
    assert sorted(consumed) == [2, 3] and sum(consumed) == 5
    assert not _failures(objective, current)
    assert current[:task.frames.start].tobytes() == poses[:task.frames.start].tobytes()
    assert current[task.frames.stop:].tobytes() == poses[task.frames.stop:].tobytes()
    assert objective.reference.tobytes() == reference
    for key, value in objective.candidate_guards[0].bounds.items():
        assert value.tobytes() == bounds[key]


@pytest.mark.parametrize('allowance', [1, 3])
def test_split_respects_a_preassigned_penetration_priority_allowance(monkeypatch, allowance):
    monkeypatch.setattr(splits, 'WINDOW_BACKTRACK_SPLITS', True)
    state, components = scheduling_state(converged=False)
    state.record['local_alphas'] = [.125] * 4
    task = _task(state.task.core, 300, components, ('unfinished',))
    task.iteration_quota = allowance
    result = splits.split_followups([task], [state], 300, components, 4, 32)
    assert sum(part.iteration_quota for part in result) == allowance
    assert len(result) == (1 if allowance == 1 else 2)
