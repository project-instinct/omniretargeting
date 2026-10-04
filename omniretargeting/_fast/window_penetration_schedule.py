"""Optional continuation allowances ordered by current collision evidence.

Depths are refreshed at a sweep boundary, including fixed-context frames.
Unknown queries retain priority. Timings never enter selection or allowances.
The second continuation targets the deeper half of remaining collision
windows; motion-only windows with negligible decrease receive no allowance.
All existing nonlinear, contact, layout and continuity checks remain intact.
"""
import math

import numpy as np

from .solver import CONVERGED_DECREASE, CONVERGED_STEPS

WINDOW_PENETRATION_CONTINUATION = False


def self_depths(objective, poses, *, chunk_size=64):
    """Legacy self witnesses at current poses, without pools or CUDA."""
    depths, unknown = np.zeros(len(poses)), np.zeros(len(poses), bool)
    query = getattr(objective, 'self_collision', None)
    if query is None:
        return depths, unknown
    provider = getattr(objective, '_window_self_cache', None)
    if provider is None:
        from .self_collision_window import SelfCollisionWindow
        provider = SelfCollisionWindow(query).prepare()
    for first in range(0, len(poses), chunk_size):
        frames = range(first, min(len(poses), first + chunk_size))
        measured = provider.measure(poses, frames=frames, jacobians=False)
        reliable = np.all(measured['body_reliable'], axis=1)
        values = measured['max_depth']
        depths[first:frames.stop] = np.max(np.where(np.isfinite(values), values, 0.), axis=1)
        unknown[first:frames.stop] = ~reliable
    return depths, unknown


def recent_decrease(record):
    history = record.get('local_history', ())
    if not history:
        return 0.
    count = min(CONVERGED_STEPS, len(history))
    start, stop = history[-count]['cost'], history[-1]['after']
    return max(0., (start - stop) / max(abs(start), 1e-12) / count)


def allocate(plan, previous, grid, self_collision, unknown, activation,
             quota, allowance, extra_index):
    """Reserve allowances for a complete deterministic sweep, not a prefix.

    The existing allowance bounds each round rather than truncating all
    continuation globally. A smaller last allocation is explicit per task.
    A second round spends only on the deepest half of collision windows.
    """
    parents = {state.task.key: state.record for state in previous}
    ranked, diagnostics = [], []
    for task in plan:
        span = slice(task.footprint.start, task.footprint.stop)
        scene = float(np.max(grid[span], initial=0.))
        robot = float(np.max(self_collision[span], initial=0.))
        uncertain = bool(unknown[span].any() or not np.isfinite(scene) or not np.isfinite(robot))
        depth = max(scene, robot)
        score = math.inf if uncertain else depth
        progress = recent_decrease(parents.get(task.key, {}))
        colliding = uncertain or depth > activation
        eligible = colliding or (extra_index == 0 and progress >= CONVERGED_DECREASE)
        row = dict(core=task.key, frames=(task.frames.start, task.frames.stop),
                   footprint=(task.footprint.start, task.footprint.stop),
                   grid_max_depth_m=scene if np.isfinite(scene) else None,
                   self_max_depth_m=robot if np.isfinite(robot) else None,
                   collision_unknown=uncertain, recent_relative_decrease=progress,
                   selected=False, iteration_quota=0,
                   reason='rank_pending' if eligible else 'motion_only_tiny_decrease')
        diagnostics.append(row)
        if eligible:
            ranked.append((not uncertain, not colliding, -score, task.key, task, row))
    ranked.sort(key=lambda item: item[:4])
    if extra_index:
        limit = math.ceil(len(ranked) / 2)
        for item in ranked[limit:]:
            item[-1]['reason'] = 'shallower_collision_deferred'
        ranked = ranked[:limit]
    admitted, remaining = {}, int(allowance)
    for rank, (_, _, _, _, task, row) in enumerate(ranked):
        assigned = min(int(quota), remaining)
        row.update(priority_rank=rank, selected=assigned > 0, iteration_quota=assigned,
                   reason='penetration_priority' if assigned else 'round_allowance_exhausted')
        if assigned:
            task.iteration_quota = assigned
            admitted[task.key] = task
            remaining -= assigned
    # Retain core order for colouring/dependencies. Priority decides allowances,
    # not timing-dependent commit order or the immutable read certificates.
    return [task for task in plan if task.key in admitted], diagnostics
