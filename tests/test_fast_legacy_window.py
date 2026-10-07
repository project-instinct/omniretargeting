"""Legacy windows keep global rows, immutable guards and whole-window commits."""

import os

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.continuity import CorrectionContinuity
from omniretargeting._fast.legacy_window import (
    WindowWorkers, _commit, _failures, optimize_legacy_windows, windows,
)
from omniretargeting._fast.legacy_window_qp import WindowObjective, serial_proposals, solve_local
from omniretargeting._fast.proposals import guards_of
from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._fast.runtime_options import RuntimeOptions
from omniretargeting._fast.window_objective import temporal_row_indices
from omniretargeting._fast.window_schedule import dependency_window, independent_batches
from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from tests.test_fast_backend import _objective


def problem(*, extra=False):
    objective, poses, _ = _objective(temporal=True, extra=extra)
    objective.evaluation_backend = 'cpu'
    objective.source_shape = ReferenceShape(objective, poses)
    objective.source_shape.freeze_reference(poses)
    objective.candidate_guards = [
        CorrectionContinuity(objective.coords, poses, root_norm=True),
        SequenceContinuityGuard(objective.tasks, poses, objective.root_rotations, windowed=True),
        AcquisitionNormalGuard(objective, poses),
    ]
    objective.hinge_guards = [objective.candidate_guards[-1]]
    return objective, poses


@pytest.mark.parametrize('frames', [range(0, 5), range(10, 23), range(31, 36)])
def test_window_model_and_cost_differences_equal_full_restriction(frames):
    objective, poses = problem(extra=True)
    moved = objective.coords.integrate(poses, np.random.default_rng(3).normal(
        scale=.003, size=(len(poses), objective.width)), 1.)
    local = WindowObjective(objective, moved, frames)
    span = local.span
    a, r, j, *_ = objective.evaluate(moved)
    b, s, k, *_ = local.evaluate(moved[span])
    full_h = sparse.block_diag(a.hessian) + j.T @ j
    small_h = sparse.block_diag(b.hessian) + k.T @ k
    columns = slice(frames.start * objective.width, frames.stop * objective.width)
    np.testing.assert_allclose(small_h.toarray(), full_h[columns, columns].toarray(), rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(b.gradient.ravel() + k.T @ s,
                               (a.gradient.ravel() + j.T @ r)[columns], rtol=1e-12, atol=1e-8)
    row_ids = temporal_row_indices(objective, frames)
    assert b.cost == pytest.approx(a.frame_cost[span].sum() + r[row_ids] @ r[row_ids], rel=1e-12)
    trial = moved[span].copy()
    trial[:, 7] += .002
    full_trial = local.embed(trial)
    assert local.evaluate(trial, False)[0].cost - b.cost == pytest.approx(
        objective.evaluate(full_trial, False)[0].cost - a.cost, rel=1e-10, abs=1e-8)


def test_projection_and_restoration_keep_outside_pose_bytes(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.shape_step import restore_shape_step
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    objective, poses = problem()
    frames = range(9, 16)
    local = WindowObjective(objective, poses, frames)
    step = np.zeros((len(frames), objective.width))
    step[:, objective.coords.scalar_cols[0]] = .001
    trial = local.project(objective.coords.integrate(poses[local.span], step, 1.))
    full = local.embed(trial)
    assert full[:frames.start].tobytes() == poses[:frames.start].tobytes()
    assert full[frames.stop:].tobytes() == poses[frames.stop:].tobytes()
    with serial_proposals():
        restored = restore_shape_step(local, poses[local.span], step)
    assert restored is not None
    full = local.embed(restored)
    assert full[:frames.start].tobytes() == poses[:frames.start].tobytes()
    assert full[frames.stop:].tobytes() == poses[frames.stop:].tobytes()
    assert not _failures(objective, full)


def test_guards_match_whole_trajectory_including_crossing_jerk():
    objective, poses = problem(extra=True)
    frames = range(9, 18)
    local = WindowObjective(objective, poses, frames)
    assert not _failures(objective, poses)
    saw_failure = False
    for amplitude in (0., .0001, .01, .5):
        trial = poses[local.span].copy()
        trial[0, 7] += amplitude
        full = local.embed(trial)
        for wrapped, guard in zip(guards_of(local)[1], guards_of(objective)[1]):
            assert wrapped.accepts(trial) == guard.accepts(full)
        saw_failure |= bool(_failures(objective, full))
    assert saw_failure


def test_conic_constraints_keep_fixed_neighbour_stencils():
    import cvxpy as cp
    objective, poses = problem()
    local = WindowObjective(objective, poses, range(9, 18))
    guard = local.candidate_guards[0]
    delta = cp.Variable(local.frames * local.width)
    constraints = guard.constraints(poses[local.span], delta)
    for amplitude in (0., .3):
        step = np.zeros((local.frames, local.width))
        step[0, objective.coords.scalar_cols[0]] = amplitude
        delta.value = step.ravel()
        linear_ok = all(np.max(c.violation(), initial=0.) < 1e-10 for c in constraints)
        full = local.embed(objective.coords.integrate(poses[local.span], step, 1.))
        assert linear_ok == objective.candidate_guards[0].accepts(full)


def test_local_lm_uses_legacy_decrease_and_keeps_references(monkeypatch):
    from omniretargeting._fast import runtime_options
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    objective, poses = problem()
    reference = objective.reference.copy()
    bounds = {k: v.copy() for k, v in objective.candidate_guards[0].bounds.items()}
    result = solve_local(objective, poses, range(8, 20), 2, .003)
    assert result['history'] and any(r['alpha'] > 0 for r in result['history'])
    for row in result['history']:
        if row['alpha']:
            assert row['after'] < row['cost']
    np.testing.assert_array_equal(objective.reference, reference)
    for key, value in bounds.items():
        np.testing.assert_array_equal(objective.candidate_guards[0].bounds[key], value)
    full = WindowObjective(objective, poses, range(8, 20)).embed(result['poses'])
    assert not _failures(objective, full)


def test_coloring_covers_endpoints_and_all_crossing_dependencies():
    # Schedule larger clips without rebuilding any guard budgets.
    for sweep in (0, 1):
        ranges = windows(900, sweep)
        covered = np.zeros(900, bool)
        for frames in ranges:
            covered[frames.start:frames.stop] = True
        assert covered.all()
        components = [type('Dependencies', (), {'window_dependencies': lambda self, f:
                      range(max(0, f.start - 3), min(900, f.stop + 3))})()]
        plan = [dependency_window(w, 900, components) for w in ranges]
        batches = independent_batches(plan)
        assert len(batches) == 2
        for batch in batches:
            for a, b in zip(batch, batch[1:]):
                assert a.footprint.stop <= b.footprint.start


def test_stale_commit_is_rejected_before_mutation():
    objective, poses = problem()
    window = dependency_window(range(9, 18), len(poses), [objective, *guards_of(objective)[1]])
    changed = poses.copy()
    changed[8, 7] += .001
    with pytest.raises(RuntimeError, match='dependency changed'):
        _commit(objective, changed, poses, window, {'poses': poses[9:18]})


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork')
def test_worker_is_persistent_and_matches_serial_lm(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.window_driver import fork_status
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    objective, poses = problem()
    objective.prepare_window_evaluation()
    if not fork_status()[0]:
        pytest.skip('parent thread state does not permit fork')
    window = dependency_window(range(8, 16), len(poses), [objective, *guards_of(objective)[1]])
    expected = solve_local(objective, poses, window.frames, 1, .003)
    workers = WindowWorkers(objective, 1)
    try:
        first = workers.solve(poses, [window], 1, .003)[0]
        second = workers.solve(poses, [window], 1, .003)[0]
        assert first['worker_pid'] == second['worker_pid'] != os.getpid()
        np.testing.assert_array_equal(first['poses'], expected['poses'])
        np.testing.assert_array_equal(second['poses'], expected['poses'])
    finally:
        workers.close()


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork')
def test_dependency_execution_matches_barrier_snapshots_and_real_lm_bytes():
    from omniretargeting._fast.window_execution import execute
    from omniretargeting._fast.window_driver import fork_status
    if not fork_status()[0]:
        pytest.skip('parent thread state does not permit fork')
    variants = []
    for pipelined in (False, True):
        objective, poses = problem(extra=True)
        objective.prepare_window_evaluation()
        components = [objective, *guards_of(objective)[1]]
        batches = [[dependency_window(f, len(poses), components) for f in batch] for batch in (
            [range(1, 7), range(20, 26)], [range(7, 13), range(26, 32)], [range(13, 20)])]
        reads, decisions = {}, {}

        def commit(current, snapshot, task, result):
            key = task.frames.start, task.frames.stop
            reads[key] = snapshot[task.footprint.start:task.footprint.stop].tobytes()
            decisions[key] = [{k: v for k, v in r.items() if k != 'seconds'} for r in result['history']]
            return _commit(objective, current, snapshot, task, result)

        current = poses.copy()
        workers = WindowWorkers(objective, 2)
        try:
            if pipelined:
                current, _, _, _ = execute(workers, current, batches, 2, .003, commit, solve_local)
            else:
                for batch in batches:
                    snapshot = current.copy()
                    results = workers.solve(snapshot, batch, 2, .003)
                    for task, result in zip(batch, results):
                        current, _ = commit(current, snapshot, task, result)
        finally:
            workers.close()
        assert not _failures(objective, current)
        variants.append((current.tobytes(), reads, decisions))
    assert variants[0] == variants[1]


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork')
def test_dependency_execution_starts_ready_next_color_before_slow_unrelated_job(monkeypatch):
    import multiprocessing
    from types import SimpleNamespace
    from omniretargeting._fast import legacy_window
    from omniretargeting._fast.window_driver import fork_status
    from omniretargeting._fast.window_execution import execute, dependencies
    from omniretargeting._fast.window_schedule import Window
    if not fork_status()[0]:
        pytest.skip('parent thread state does not permit fork')
    finished_next_color = multiprocessing.get_context('fork').Event()
    batches = [[Window(range(0, 1), range(0, 1)), Window(range(4, 5), range(4, 5))],
               [Window(range(1, 2), range(0, 2))]]
    assert dependencies(batches) == [frozenset(), frozenset(), frozenset({0})]

    def solve(obj, snapshot, frames, iterations, convergence):
        if frames.start == 4:
            assert finished_next_color.wait(5), 'unnecessary colour barrier'
        elif frames.start == 1:
            assert snapshot[0, 0] == .9
            finished_next_color.set()
        return dict(poses=snapshot[frames.start:frames.stop] * .9, seconds=.001)

    monkeypatch.setattr(legacy_window, 'solve_local', solve)

    def commit(current, snapshot, task, result):
        span = slice(task.footprint.start, task.footprint.stop)
        assert current[span].tobytes() == snapshot[span].tobytes()
        candidate = current.copy()
        candidate[task.frames.start:task.frames.stop] = result['poses']
        return candidate, {}

    workers = WindowWorkers(SimpleNamespace(), 2)
    try:
        result, _, _, _ = execute(workers, np.ones((6, 1)), batches, 1, .003, commit, solve)
    finally:
        workers.close(abort=True)
    assert result[:, 0].tolist() == [.9, .9, 1., 1., .9, 1.]


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork')
def test_same_color_parallel_results_merge_without_shared_rows(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.window_driver import fork_status
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    objective, poses = problem()
    objective.prepare_window_evaluation()
    if not fork_status()[0]:
        pytest.skip('parent thread state does not permit fork')
    parts = [range(8, 16), range(25, 33)]
    components = [objective, *guards_of(objective)[1]]
    batch = [dependency_window(f, len(poses), components) for f in parts]
    assert len(independent_batches(batch)) == 1
    workers = WindowWorkers(objective, 2)
    try:
        results = workers.solve(poses, batch, 1, .003)
    finally:
        workers.close()
    assert len({r['worker_pid'] for r in results}) == 2
    candidate, decrease = poses.copy(), 0.
    for window, result in zip(batch, results):
        candidate, row = _commit(objective, candidate, poses, window, result)
        assert row['accepted']
        decrease += row['cost'] - row['after']
    assert not _failures(objective, candidate)
    full_decrease = (objective.evaluate(poses, False)[0].cost
                     - objective.evaluate(candidate, False)[0].cost)
    assert full_decrease == pytest.approx(decrease, rel=1e-10, abs=1e-8)
    fixed = np.ones(len(poses), bool)
    for frames in parts:
        fixed[frames.start:frames.stop] = False
    assert candidate[fixed].tobytes() == poses[fixed].tobytes()




def test_failed_whole_sweep_rolls_back_and_hides_callback(monkeypatch):
    from omniretargeting._fast import legacy_window, runtime_options
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    monkeypatch.setattr(legacy_window, 'fork_status', lambda: (False, 'test_serial'))
    objective, poses = problem()
    original_check, callbacks = legacy_window._failures, []
    whole_checks = []

    def audit(obj, candidate, frames=None):
        failures = original_check(obj, candidate, frames)
        if frames is None:
            whole_checks.append(candidate.copy())
            if len(whole_checks) == 2:
                return ['simulated_whole_trajectory_failure']
        return failures

    monkeypatch.setattr(legacy_window, '_failures', audit)
    result, history = optimize_legacy_windows(
        objective, poses, iterations=2, callback=lambda *args: callbacks.append(args))
    assert len(whole_checks) == 2 and not np.array_equal(whole_checks[1], poses)
    assert result.tobytes() == poses.tobytes()
    assert not callbacks
    assert history[-1]['sweep_guard_failures'] == ['simulated_whole_trajectory_failure']
    assert all(row['sweep_rolled_back'] and not row['accepted'] and row['alpha'] == 0
               for row in history)




@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork')
def test_child_commit_verdict_equals_parent_and_needs_no_parent_evaluations(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.window_driver import fork_status
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    objective, poses = problem(extra=True)
    objective.prepare_window_evaluation()
    if not fork_status()[0]:
        pytest.skip('parent thread state does not permit fork')
    window = dependency_window(range(8, 16), len(poses), [objective, *guards_of(objective)[1]])
    workers = WindowWorkers(objective, 1)
    try:
        result = workers.solve(poses, [window], 2, .003)[0]
    finally:
        workers.close()
    plain = {k: v for k, v in result.items() if k != 'commit'}
    expected, old = _commit(objective, poses, poses, window, plain)

    def forbidden(*args, **kwargs):
        raise AssertionError('certified commit repeated an objective/guard evaluation')

    monkeypatch.setattr(objective, 'evaluate', forbidden)
    for guard in guards_of(objective)[1]:
        monkeypatch.setattr(guard, 'accepts', forbidden)
    actual, new = _commit(objective, poses, poses, window, result)
    assert actual.tobytes() == expected.tobytes()
    for key in ('cost', 'after', 'candidate_cost', 'accepted', 'parent_rejections'):
        assert new[key] == old[key]
    assert old['commit_evaluated_by'] == 'parent' and new['commit_evaluated_by'] == 'child'
    stale = poses.copy()
    stale[window.footprint.start, 7] += .001
    with pytest.raises(RuntimeError, match='dependency changed'):
        _commit(objective, stale, poses, window, result)


@pytest.mark.parametrize('amplitude', [0., .5])
def test_certified_rejections_equal_parent_verdict(amplitude):
    from omniretargeting._fast.legacy_window_qp import commit_verdict
    objective, poses = problem(extra=True)
    window = dependency_window(range(8, 16), len(poses), [objective, *guards_of(objective)[1]])
    candidate = poses.copy()
    candidate[8, 7] += amplitude
    verdict = commit_verdict(objective, poses, candidate, window.frames)
    part = candidate[8:16]
    a, first = _commit(objective, poses, poses, window, dict(poses=part, commit=verdict))
    b, second = _commit(objective, poses, poses, window, dict(poses=part))
    assert not first['accepted'] and a.tobytes() == b.tobytes() == poses.tobytes()
    for key in ('cost', 'after', 'candidate_cost', 'accepted', 'parent_rejections'):
        assert first[key] == second[key]
    assert bool(amplitude) == ('CorrectionContinuity' in first['parent_rejections'])


@pytest.mark.parametrize('shortened', [False, True])
def test_convergence_observer_preserves_legacy_shortened_rule(monkeypatch, shortened):
    from omniretargeting._fast import runtime_options, solver
    from omniretargeting._fast.legacy_window_qp import observe_convergence
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    original, state = solver._progress_converged, {}
    costs, slow = [1000.], 0
    with observe_convergence(state):
        for cost in (999., 998.):
            expected = original(costs.copy(), cost, slow, .5, True, .003, shortened=shortened)
            actual = solver._progress_converged(costs, cost, slow, .5, True, .003, shortened=shortened)
            assert actual == expected
            slow, done = actual
            assert state['converged'] == done
    assert solver._progress_converged is original
    assert state['converged'] == shortened


def test_convergence_observer_keeps_guarded_three_step_rule(monkeypatch):
    from omniretargeting._fast import runtime_options, solver
    from omniretargeting._fast.legacy_window_qp import observe_convergence
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    costs, slow, state, verdicts = [1000.], 0, {}, []
    with observe_convergence(state):
        for cost in (999., 998., 997.):
            slow, done = solver._progress_converged(costs, cost, slow, .5, True, .003)
            verdicts.append(done)
            assert state['converged'] == done
    assert verdicts == [False, False, True]


def scheduling_state(*, count=300, core=range(80, 144), converged=True, rejected=False, retry=0):
    from types import SimpleNamespace
    from omniretargeting._fast.legacy_window import WindowState, _task
    components = [SimpleNamespace(window_dependencies=lambda f:
                  range(max(0, f.start - 3), min(count, f.stop + 3)))]
    task = _task(core, count, components, ('test',))
    task.retry_count = retry
    row = dict(iteration=0, frames=[task.frames.start, task.frames.stop],
               local_converged=converged, accepted=True,
               local_stop_reason='rejected_iteration' if rejected else 'converged',
               local_iterations=4, window_seconds=4.)
    return WindowState(task, row, np.zeros(len(task.footprint), np.int64)), components


@pytest.mark.parametrize('trigger', ['none', 'quota', 'stale', 'depth', 'tie', 'unknown'])
def test_continuation_selects_only_unfinished_or_deep_footprints(trigger):
    from omniretargeting._fast.legacy_window import continuation_plan
    state, components = scheduling_state(converged=trigger != 'quota')
    revisions, depths = np.zeros(300, np.int64), np.zeros(300)
    # The evidence is in fixed context, outside the movable window.
    edge = state.task.footprint.start
    if trigger == 'stale':
        revisions[edge] = 1
    if trigger in ('depth', 'tie'):
        depths[edge] = .001 + (1e-8 if trigger == 'depth' else 0.)
    if trigger == 'unknown':
        depths[edge] = np.nan
    plan, decisions = continuation_plan([state], revisions, depths, .001, components, 0)
    selected = trigger not in ('none', 'tie')
    assert bool(plan) == decisions[0]['selected'] == selected
    if selected:
        expected = dict(quota='not_converged', stale='convergence_dependency_changed',
                        depth='acquisition_grid_penetration', unknown='grid_depth_unknown')[trigger]
        assert expected in plan[0].reasons
        assert plan[0].frames != state.task.frames
        assert plan[0].frames.start <= state.task.core.start < state.task.core.stop <= plan[0].frames.stop


def test_rejected_iteration_gets_one_enlarged_retry_then_reports_exhaustion():
    from omniretargeting._fast.legacy_window import WindowState, continuation_plan
    state, components = scheduling_state(converged=False, rejected=True)
    revisions, depths = np.zeros(300, np.int64), np.zeros(300)
    plan, _ = continuation_plan([state], revisions, depths, .001, components, 0)
    task = plan[0]
    assert task.halo == 30 and task.retry_count == 1 and task.retry_of == (0,)
    assert task.frames.start == state.task.frames.start - 15
    assert task.frames.stop == state.task.frames.stop + 15
    failed_again = WindowState(task, state.record, revisions[task.footprint.start:task.footprint.stop].copy())
    plan, decisions = continuation_plan([failed_again], revisions, depths, .001, components, 1)
    assert not plan and 'rejected_retry_exhausted' in decisions[0]['reasons']


def test_second_base_sweep_retries_prior_rejection_and_splits_high_iteration_cores():
    from omniretargeting._fast.legacy_window import base_plan, expected_work
    state, components = scheduling_state(count=512, core=range(64, 128), converged=False, rejected=True)
    plan = base_plan(512, 1, components, [state], 4)
    retries = [task for task in plan if task.retry_count]
    assert len(retries) == 1
    assert retries[0].frames.start <= state.task.frames.start - 15
    assert retries[0].frames.stop >= state.task.frames.stop + 15
    records = []
    for start in range(0, 512, 64):
        item, _ = scheduling_state(count=512, core=range(start, start + 64))
        item.record['local_iterations'] = 6 if start in (192, 256) else 1
        records.append(item)
    plan = base_plan(512, 1, components, records, 4)
    split = [task for task in plan if 'predicted_work_split' in task.reasons]
    assert split and all(len(task.core) == 32 for task in split)
    covered = np.zeros(512, bool)
    for task in plan:
        covered[task.core.start:task.core.stop] = True
    assert covered.all()
    rows = [s.record for s in records]
    assert expected_work(range(200, 230), rows, 4) > expected_work(range(20, 50), rows, 4)
    for index, item in enumerate(records):
        item.record['window_seconds'] = 1e9 / (index + 1)
    again = base_plan(512, 1, components, records, 4)
    assert again == plan


def test_grid_continuation_measurement_matches_existing_fields_without_pool(monkeypatch):
    from tests.test_fast_backend import _floor
    from omniretargeting._fast.legacy_window_qp import grid_depths
    objective, poses = problem()
    objective.scene_sdf = _floor(.25)
    objective.objects = [dict(sdf=_floor(.22), rotations=np.tile(np.eye(3), (len(poses), 1, 1)),
                              translations=np.zeros((len(poses), 3)), scales=None)]
    objective.kin.evaluate(poses, False)
    expected = np.zeros(len(poses))
    for _, *field in objective._clearance_fields():
        values = objective._deepest_samples(*field)[0]
        expected = np.maximum(expected, np.max(np.maximum(-values, 0.), axis=1, initial=0.))

    def forbidden(*args, **kwargs):
        raise AssertionError('continuation started a full-trajectory geometry pool')

    monkeypatch.setattr(objective, 'deepest_fields', forbidden)
    actual, families = grid_depths(objective, poses, chunk_size=11)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
    assert actual.max() > .001 and families == ['moving_depth_m', 'scene_depth_m']


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork')
def test_queue_refills_idle_worker_before_slowest_job_finishes(monkeypatch):
    import multiprocessing
    from types import SimpleNamespace
    from omniretargeting._fast import legacy_window
    from omniretargeting._fast.window_driver import fork_status
    objective, poses = problem()
    objective.prepare_window_evaluation()
    if not fork_status()[0]:
        pytest.skip('parent thread state does not permit fork')
    third_dispatched = multiprocessing.get_context('fork').Event()

    def work(obj, snapshot, frames, iterations, convergence):
        if frames.start == 0:
            assert third_dispatched.wait(5.), 'queue waited for slowest job before refilling'
        elif frames.start == 2:
            third_dispatched.set()
        return dict(worker_pid=os.getpid(), seconds=0., frame=frames.start)

    monkeypatch.setattr(legacy_window, 'solve_local', work)
    batch = [SimpleNamespace(frames=range(i, i + 1)) for i in range(4)]
    workers = WindowWorkers(objective, 2)
    try:
        results = workers.solve(poses, batch, 1, .003)
    finally:
        workers.close(abort=True)
    assert [r['frame'] for r in results] == list(range(4))
    assert results[0]['worker_pid'] != results[1]['worker_pid'] == results[2]['worker_pid']




@pytest.mark.parametrize('budget, expected_sweeps, stop', [
    (128, 2, 'extra_sweep_cap'), (5, 2, 'extra_sweep_cap'),
    (4, 1, 'extra_window_iteration_budget'), (0, 0, 'extra_window_iteration_budget'),
])
def test_driver_caps_selective_extra_work(monkeypatch, budget, expected_sweeps, stop):
    from omniretargeting._fast import legacy_window, runtime_options
    from omniretargeting._fast.legacy_window_qp import commit_verdict
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    monkeypatch.setattr(legacy_window, 'fork_status', lambda: (False, 'test_serial'))
    monkeypatch.setattr(legacy_window, 'MAX_EXTRA_WINDOW_ITERATIONS', budget)
    monkeypatch.setattr(legacy_window, 'grid_depths', lambda obj, q: (np.zeros(len(q)), []))
    objective, poses = problem()

    def work(obj, snapshot, frames, iterations, convergence):
        # An unfinished search with no admissible net change. The actual
        # unchanged candidate is still certified and rejected normally.
        return dict(poses=snapshot[frames.start:frames.stop].copy(), history=[],
                    seconds=.001, worker_pid=os.getpid(), converged=False,
                    stop_reason='iteration_quota', commit=commit_verdict(obj, snapshot, snapshot, frames))

    monkeypatch.setattr(legacy_window, 'solve_local', work)
    result, history = optimize_legacy_windows(objective, poses, iterations=35)
    summary = history[-1]['continuation']
    assert summary['extra_sweeps'] == expected_sweeps and summary['stop_reason'] == stop
    assert summary['extra_window_iterations_reserved'] == min(budget, 8)
    assert summary['extra_window_iterations_completed'] == 0
    assert 'wall_budget_seconds' not in summary
    assert len(history) == 2 + expected_sweeps
    assert all('not_converged' in row['continuation_reasons'] for row in history[2:])
    assert all(row['sweep_guard_accepted'] for row in history)
    assert result.tobytes() == poses.tobytes()


def perturbed_clock(monkeypatch, step):
    from types import SimpleNamespace
    from omniretargeting._fast import legacy_window, legacy_window_qp
    ticks = 0.

    def monotonic():
        nonlocal ticks
        ticks += step
        return ticks

    clock = SimpleNamespace(monotonic=monotonic)
    monkeypatch.setattr(legacy_window, 'time', clock)
    monkeypatch.setattr(legacy_window_qp, 'time', clock)


def test_real_lm_driver_is_byte_identical_under_clock_perturbation(monkeypatch):
    from omniretargeting._fast import legacy_window, runtime_options
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    monkeypatch.setattr(legacy_window, 'fork_status', lambda: (False, 'test_serial'))
    monkeypatch.setattr(legacy_window, 'SWEEP_ITERATIONS', (1, 1))
    monkeypatch.setattr(legacy_window, 'EXTRA_ITERATIONS', 1)
    outputs = []
    for clock_step in (.001, 1e6):
        perturbed_clock(monkeypatch, clock_step)
        objective, poses = problem()
        result, history = optimize_legacy_windows(objective, poses, iterations=35)
        assert result.tobytes() != poses.tobytes()
        assert history[-1]['continuation']['extra_sweeps'] == 2
        assert not _failures(objective, result)
        outputs.append((result.tobytes(), [(r['frames'], r['local_alphas']) for r in history]))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('parallel', [False, True])
def test_adaptive_driver_timing_changes_neither_windows_nor_pose_bytes(monkeypatch, parallel):
    """Exercise splitting, queue refill, shifted continuation and a reduced quota."""
    import time
    from types import SimpleNamespace
    from omniretargeting._fast import legacy_window, runtime_options
    from omniretargeting._fast.legacy_window_qp import commit_verdict
    from omniretargeting._fast.window_driver import fork_status
    if parallel and (not hasattr(os, 'fork') or not fork_status()[0]):
        pytest.skip('parent thread state does not permit fork')
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    monkeypatch.setattr(legacy_window, 'fork_status', lambda: (parallel, None if parallel else 'test_serial'))
    monkeypatch.setattr(legacy_window, 'available_cpus', lambda: 4)
    monkeypatch.setattr(legacy_window, 'MAX_EXTRA_WINDOW_ITERATIONS', 60)
    monkeypatch.setattr(legacy_window, 'grid_depths', lambda obj, q: (np.zeros(len(q)), []))
    count, outputs = 512, []
    for variant in (0, 1):
        perturbed_clock(monkeypatch, .001 if variant == 0 else 1e6)
        objective = SimpleNamespace(
            coords=SimpleNamespace(qi=np.array([0])), lower=-10., upper=10.,
            source_shape=None, candidate_guards=[], stop_workers=lambda: None,
            prepare_window_evaluation=lambda: None,
            window_dependencies=lambda f: range(max(0, f.start - 3), min(count, f.stop + 3)),
            evaluate=lambda q, jac, frames: (SimpleNamespace(cost=float(np.sum(q[list(frames)]**2))),))
        poses = np.full((count, 1), 2.)

        def work(obj, snapshot, frames, iterations, convergence):
            # Boundary-sensitive updates make timing-dependent splitting show
            # up in the trajectory, not just in schedule diagnostics.
            heavy = 192 <= (frames.start + frames.stop) // 2 < 320
            slow = heavy if variant == 0 else not heavy
            if parallel:
                time.sleep(.008 if slow else .001)
            used = iterations if heavy else 1
            candidate = snapshot.copy()
            candidate[frames.start:frames.stop] *= (.85 + len(frames) / 10000.)**used
            trace = [dict(alpha=1., damping=.001) for _ in range(used)]
            return dict(poses=candidate[frames.start:frames.stop], history=trace,
                        seconds=1000. if slow else .001, worker_pid=os.getpid(),
                        converged=False, stop_reason='iteration_quota',
                        commit=commit_verdict(obj, snapshot, candidate, frames))

        monkeypatch.setattr(legacy_window, 'solve_local', work)
        result, history = optimize_legacy_windows(objective, poses, iterations=35)
        summary = history[-1]['continuation']
        assert summary['extra_sweeps'] == 2
        assert summary['extra_window_iterations_reserved'] <= 60
        assert summary['extra_window_iterations_completed'] <= summary['extra_window_iterations_reserved']
        assert any('predicted_work_split' in r['continuation_reasons'] for r in history)
        assert any(r.get('batch_schedule', {}).get('iteration_quota', 4) < 4
                   for r in history if r['sweep'] >= 2)
        assert all(a['accepted'] for a in summary['sweep_audits'])
        signature = [(r['core'], r['frames'], r['sweep'], r['color'], r['dispatch_order'],
                      r['expected_work'], r['local_alphas'], r['continuation_reasons']) for r in history]
        outputs.append((result.tobytes(), signature, summary['selection']))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('cpus, expected', [(24, 8), (16, 8), (8, 4), (1, 1)])
def test_worker_ceiling_does_not_depend_on_first_batch(monkeypatch, cpus, expected):
    from contextlib import contextmanager
    from omniretargeting._fast import legacy_window, runtime_options
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    monkeypatch.setattr(legacy_window, 'fork_status', lambda: (True, None))
    monkeypatch.setattr(legacy_window, 'available_cpus', lambda: cpus)
    original, counts = legacy_window._workers, []

    @contextmanager
    def workers(obj, count):
        counts.append(count)
        with original(obj, 0) as serial:
            yield serial

    monkeypatch.setattr(legacy_window, '_workers', workers)
    objective, poses = problem()
    assert len(windows(len(poses), 0)) == 1
    _, history = optimize_legacy_windows(objective, poses, iterations=1)
    assert counts == [expected]
    assert history[0]['worker_count'] == expected
    assert history[-1]['continuation']['worker_ceiling'] == expected


def test_focus_solves_only_windows_with_flagged_frames(monkeypatch):
    from functools import partial
    from omniretargeting._fast import legacy_window, runtime_options
    from omniretargeting._fast.legacy_window_qp import commit_verdict
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    monkeypatch.setattr(legacy_window, 'fork_status', lambda: (False, 'test_serial'))
    monkeypatch.setattr(legacy_window, 'windows', partial(legacy_window.windows, core=12))
    monkeypatch.setattr(legacy_window, '_task', partial(legacy_window._task, halo=2))
    objective, poses = problem()
    solved = []

    def work(obj, snapshot, frames, iterations, convergence):
        solved.append(frames)
        return dict(poses=snapshot[frames.start:frames.stop].copy(), history=[],
                    seconds=.001, worker_pid=os.getpid(), converged=False,
                    stop_reason='iteration_quota', commit=commit_verdict(obj, snapshot, snapshot, frames))

    monkeypatch.setattr(legacy_window, 'solve_local', work)
    focus = np.zeros(len(poses), bool)
    focus[3] = True
    result, history = optimize_legacy_windows(objective, poses, iterations=35, focus=focus)
    assert solved and all(frames.start <= 3 < frames.stop for frames in solved)
    assert {row['sweep'] for row in history} == {0, 1}
    assert result.tobytes() == poses.tobytes()
    solved.clear()
    result, history = optimize_legacy_windows(objective, poses, iterations=35,
                                              focus=np.zeros(len(poses), bool))
    assert not solved and not history and result.tobytes() == poses.tobytes()
