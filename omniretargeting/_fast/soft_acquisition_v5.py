"""Soft-v4 final contract with original-world support and sticking rows."""
import time

from .soft_acquisition_v4 import finish_acquisition
from .soft_bulk import run_bulk
from .soft_support_rows import SupportWeights


def optimize_soft_acquisition_v5(objective, poses, *, iterations, callback=None):
    started = time.monotonic()
    weights = SupportWeights()
    # Two small relative decreases end each phase. These caps only bound a
    # failed plateau detection; they do not force a fixed iteration quota.
    bulk, _, balanced = run_bulk(objective, poses, iterations=24, threshold=.03,
        damping_floor=.001, support_weights=weights, source_yield=True, world_recovery=True)
    bulk, _, clearance = run_bulk(objective, bulk, iterations=8, threshold=.01,
        clearance_multiplier=4., initial_damping=1., damping_floor=1., damping_decay=.8,
        self_monotone=False, support_weights=weights, plateau_steps=1, max_attempts=3,
        source_yield=True, world_recovery=True)
    return finish_acquisition(objective, poses, bulk, [balanced, clearance], started, callback,
                              mode='soft_v5_ranked_support_sticking', support_projection=True)
