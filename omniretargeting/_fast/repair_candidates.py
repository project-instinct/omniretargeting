"""Guard-checked repair proposals, with optional restoration after backtracking."""

import numpy as np
from .speculation import GuardedCall, can_fork
from .proposals import guards_of


FRACTIONS = (1., .5, .25, .125, .0625, .03125, .015625)


def candidate_steps(limits, current, step, restore, record, *, lazy=False, fixed_q=None, restore_below=None,
                    parallel_restore=False):
    """Yield candidates for independent original-mesh validation.

    With lazy restoration, ordinary backtracking gets the first opportunity.
    Exhausting the preferred fractions (including geometry rejections) requests
    the expensive restored full step. ``restore_below`` avoids replacing one
    productive full restoration with many tiny ordinary iterations. No guard
    or collision allowance is changed.
    """
    trials = [(alpha, not lazy and alpha == 1.) for alpha in FRACTIONS]
    if lazy:
        index = (len(trials) if restore_below is None else
                 next((i for i, (alpha, _) in enumerate(trials) if alpha < restore_below), len(trials)))
        trials.insert(index, (1., True))
    pending = None
    stats = None
    if parallel_restore:
        stats = getattr(limits.objective, 'parallel_repair_stats', None)
        if stats is None:
            stats = dict(launched=0, consumed=0, discarded=0, wait_seconds=0.)
            limits.objective.parallel_repair_stats = stats
    try:
        for alpha, can_restore in trials:
            candidate = current.copy()
            candidate[:, limits.coords.qi] += alpha * step
            if fixed_q is not None and not np.array_equal(candidate[:, fixed_q], current[:, fixed_q]):
                record['check'] = {'reason': 'joint outside limb scope changed'}
                continue
            valid, check = limits.check(candidate)
            record['check'] = check
            if (parallel_restore and lazy and alpha == 1. and not can_restore and not valid
                    and can_fork()):
                pending = GuardedCall(restore, (candidate,), guards_of(limits.objective)[1])
                stats['launched'] += 1
            if lazy and can_restore and valid:
                # The exact same full step already failed the caller's geometry
                # check; restoring a guard-feasible step would just return it.
                continue
            if not valid and can_restore:
                if pending is None:
                    candidate, restoration = restore(candidate)
                else:
                    job, pending = pending, None
                    stats['consumed'] += 1
                    try:
                        candidate, restoration = job.receive()
                    finally:
                        stats['wait_seconds'] += job.wait_seconds
                record['restoration'] = restoration
                if candidate is not None:
                    valid, check = limits.check(candidate)
                    record['check'] = check
            if not valid or candidate is None:
                record.setdefault('rejected_limits', []).append({'alpha': alpha, **check})
                continue
            yield alpha, candidate
    finally:
        if pending is not None:
            pending.close()
            stats['discarded'] += 1
