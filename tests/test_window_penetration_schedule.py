"""Current collision evidence controls complete, deterministic allowances."""
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from omniretargeting._fast import window_penetration_schedule as schedule
from omniretargeting._fast.legacy_window import _task, continuation_plan
from tests.test_fast_legacy_window import scheduling_state


def cases(count=4):
    states, tasks = [], []
    for index in range(count):
        state, components = scheduling_state(count=400, core=range(32 + 80 * index, 96 + 80 * index),
                                              converged=False)
        state.record['local_history'] = [dict(cost=100., after=99.999, alpha=1.)]
        states.append(state)
        tasks.append(_task(state.task.core, 400, components, ('not_converged',)))
    return states, tasks


def allocate(states, plan, grid, robot=None, unknown=None, *, budget=64, round=0):
    return schedule.allocate(plan, states, grid,
        np.zeros(400) if robot is None else robot,
        np.zeros(400, bool) if unknown is None else unknown, .001, 4, budget, round)


def test_worst_fixed_context_penetration_gets_budget_before_small_motion_decrease():
    states, tasks = cases()
    grid, robot = np.zeros(400), np.zeros(400)
    grid[tasks[0].footprint.start] = .009  # outside its movable frames
    robot[tasks[1].core.start + 32] = .019
    grid[tasks[2].core.start + 32] = .035
    plan, decisions = allocate(states, tasks, grid, robot, budget=7)
    assert [task.key for task in plan] == [tasks[1].key, tasks[2].key]
    assert [task.iteration_quota for task in plan] == [3, 4]
    assert sum(task.iteration_quota for task in plan) == 7
    assert decisions[0]['reason'] == 'round_allowance_exhausted'
    assert decisions[3]['reason'] == 'motion_only_tiny_decrease'


def test_unknown_never_receives_false_zero_clearance_credit():
    states, tasks = cases(2)
    grid, unknown = np.zeros(400), np.zeros(400, bool)
    grid[tasks[0].core.start + 32] = .2
    unknown[tasks[1].footprint.stop - 1] = True
    grid[tasks[1].core.start + 32] = np.nan
    plan, decisions = allocate(states, tasks, grid, unknown=unknown, budget=4)
    assert plan == [tasks[1]]
    assert decisions[1]['collision_unknown'] and decisions[1]['grid_max_depth_m'] is None


def test_second_round_reserves_only_deepest_half_with_stable_ties():
    states, tasks = cases()
    grid = np.zeros(400)
    for task in tasks:
        grid[task.core.start] = .01
    plan, decisions = allocate(states, tasks, grid, round=1)
    assert plan == tasks[:2]
    assert all(row['reason'] == 'shallower_collision_deferred' for row in decisions[2:])
    for state in states:
        state.record['window_seconds'] = 1e30
    again, _ = allocate(states, tasks, grid, round=1)
    assert again == plan


def test_collision_has_priority_over_meaningful_motion_only_progress():
    states, tasks = cases(2)
    states[0].record['local_history'] = [dict(cost=100., after=80., alpha=1.)]
    grid = np.zeros(400)
    grid[tasks[1].core.start + 32] = .002
    plan, _ = allocate(states, tasks, grid, budget=4)
    assert plan == [tasks[1]]
    plan, _ = allocate(states, tasks, np.zeros(400), budget=4)
    assert plan == [tasks[0]]


def test_converged_self_collision_only_window_is_still_selected():
    state, components = scheduling_state(converged=True)
    robot, unknown = np.zeros(300), np.zeros(300, bool)
    robot[state.task.footprint.start] = .004
    plan, decisions = continuation_plan([state], np.zeros(300, np.int64), np.zeros(300),
        .001, components, 0, self_depths=robot, self_unknown=unknown)
    assert len(plan) == 1 and 'acquisition_self_penetration' in decisions[0]['reasons']
    robot[:] = 0.
    unknown[state.task.footprint.stop - 1] = True
    plan, decisions = continuation_plan([state], np.zeros(300, np.int64), np.zeros(300),
        .001, components, 0, self_depths=robot, self_unknown=unknown)
    assert len(plan) == 1 and 'self_depth_unknown' in decisions[0]['reasons']


def test_self_depth_signal_matches_legacy_witness_bits_for_every_frame():
    from omniretargeting.self_collision import SelfCollisionQuery
    from tests.test_batched_self_collision import _poses
    from tests.test_convex_mesh_distance import mesh_pair
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    poses = _poses(model)
    expected = np.array([max((max(0., -row['clearance']) for row in query.query(q)), default=0.)
                         for q in poses])
    with threadpool_limits(limits=1):
        actual, unknown = schedule.self_depths(SimpleNamespace(self_collision=query), poses, chunk_size=7)
    assert actual.tobytes() == expected.tobytes()
    assert not unknown.any()


def test_self_query_failure_keeps_unknown_flag():
    def measure(poses, *, frames, jacobians):
        return dict(max_depth=np.full((len(frames), 2), np.nan),
                    body_reliable=np.zeros((len(frames), 2), bool))
    obj = SimpleNamespace(self_collision=object(), _window_self_cache=SimpleNamespace(measure=measure))
    depths, unknown = schedule.self_depths(obj, np.zeros((9, 2)), chunk_size=4)
    assert np.all(depths == 0.) and unknown.all()


def test_policy_stays_default_off():
    assert schedule.WINDOW_PENETRATION_CONTINUATION is False
