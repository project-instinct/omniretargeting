"""Optional smaller continuation solves after repeated nonlinear backtracking.

Selection uses accepted fractions, never elapsed time. Two smaller cores
share their original core's iteration allowance and retain its asymmetric
halos and full dependency certificates. All nonlinear checks remain intact.
"""
from .window_schedule import dependency_window

WINDOW_BACKTRACK_SPLITS = False


def repeated_backtracking(record):
    fractions = record.get('local_alphas', ())
    return (bool(fractions) and 0 < fractions[-1] <= .25
            and sum(0 < value <= .25 for value in fractions) * 2 >= len(fractions))


def split_followups(plan, previous, count, components, quota, minimum):
    if not WINDOW_BACKTRACK_SPLITS or quota < 2:
        return plan
    from .legacy_window import WindowTask
    parents = {state.task.key: state for state in previous}
    result = []
    for task in plan:
        parent = parents.get(task.key)
        allowance = task.iteration_quota if task.iteration_quota is not None else quota
        if (parent is None or allowance < 2 or task.retry_count or len(task.core) < 2 * minimum
                or not repeated_backtracking(parent.record)):
            result.append(task)
            continue
        middle = (task.core.start + task.core.stop) // 2
        left = task.core.start - task.frames.start
        right = task.frames.stop - task.core.stop
        parts = ((range(task.core.start, middle),
                  range(task.frames.start, min(task.frames.stop, middle + right)), allowance // 2),
                 (range(middle, task.core.stop),
                  range(max(task.frames.start, middle - left), task.frames.stop), allowance - allowance // 2))
        for core, frames, allowance in parts:
            result.append(WindowTask(core, dependency_window(frames, count, components),
                (*task.reasons, 'backtrack_demand_split'), task.retry_count, task.retry_of,
                task.halo, iteration_quota=allowance))
    return result
