"""Current collision evidence controls complete, deterministic allowances."""

import numpy as np

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
