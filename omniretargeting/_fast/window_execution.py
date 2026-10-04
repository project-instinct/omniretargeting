"""Execute a coloured sweep in the same dependency order, without barriers.

Only disjoint declared footprints may commute. A window waits for every
earlier-colour window whose footprint intersects its own. Thus all bytes it
can read equal the barrier scheduler's snapshot, regardless of completion
timing; the caller still checks those bytes before committing a whole window.
"""

from multiprocessing.connection import wait
import time

from .window_schedule import overlaps


def dependencies(batches):
    tasks = [task for batch in batches for task in batch]
    return [frozenset(j for j in range(i) if overlaps(task.footprint, tasks[j].footprint))
            for i, task in enumerate(tasks)]


def execute(workers, poses, batches, iterations, convergence, commit, solve_local):
    """Return results in the original colour/order; commits may commute only.

    ``commit(current, snapshot, task, result)`` returns the new current poses
    and caller metadata. No mutable objective/guard pack may change in a sweep.
    """
    started = time.monotonic()
    tasks = [task for batch in batches for task in batch]
    prior = dependencies(batches)
    pending, done, active = set(range(len(tasks))), set(), {}
    results, submitted, completed = [None] * len(tasks), {}, {}
    current = poses

    def quota(index):
        assigned = getattr(tasks[index], 'iteration_quota', None)
        return iterations if assigned is None else min(iterations, assigned)

    def ready():
        return next((i for i in sorted(pending) if prior[i] <= done), None)

    def finish(index, snapshot, value):
        nonlocal current
        value.update(queue_seconds=submitted[index],
                     completion_seconds=time.monotonic() - started)
        current, metadata = commit(current, snapshot, tasks[index], value)
        completed[index] = time.monotonic() - started
        results[index] = value, metadata
        done.add(index)

    while pending or active:
        if not workers.jobs:
            index = ready()
            if index is None:
                raise RuntimeError('Cyclic legacy window dependencies')
            pending.remove(index)
            snapshot = current.copy()
            submitted[index] = time.monotonic() - started
            value = solve_local(workers.objective, snapshot, tasks[index].frames, quota(index), convergence)
            finish(index, snapshot, value)
            continue
        for _, connection in workers.jobs:
            if connection in active:
                continue
            index = ready()
            if index is None:
                break
            pending.remove(index)
            snapshot = current.copy()
            submitted[index] = time.monotonic() - started
            connection.send((snapshot, tasks[index].frames, quota(index), convergence))
            active[connection] = index, snapshot
        if not active:
            raise RuntimeError('Cyclic legacy window dependencies')
        for connection in wait(list(active)):
            index, snapshot = active.pop(connection)
            try:
                status, value = connection.recv()
            except EOFError as error:
                raise RuntimeError(f'Legacy window child exited for {tasks[index].frames}') from error
            if status != 'ok':
                raise RuntimeError(f'Legacy window child failed for {tasks[index].frames}:\n{value}')
            finish(index, snapshot, value)

    offset, frontier, groups = 0, 0., []
    for batch in batches:
        indices = range(offset, offset + len(batch))
        begin = min(submitted[i] for i in indices)
        end = max(completed[i] for i in indices)
        next_frontier = max(frontier, end)
        groups.append(dict(active_span_seconds=end - begin,
                           # Non-overlapping wall shares still sum to sweep
                           # time; active spans of different colours overlap.
                           wall_seconds=next_frontier - frontier))
        frontier, offset = next_frontier, offset + len(batch)
    return current, results, groups, dict(
        wall_seconds=time.monotonic() - started,
        completed_worker_seconds=sum(value['seconds'] for value, _ in results),
        cross_color_execution=True)
