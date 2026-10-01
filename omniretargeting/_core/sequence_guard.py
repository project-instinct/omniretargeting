"""Keep one sequence budget across optional downstream correction stages."""

import numpy as np

from .._sequence.continuity_guard import SequenceContinuityGuard
from .._sequence.pose_guard import SequencePoseGuard
from .._sequence.surface_guard import SurfaceSequenceGuard
from ..sequence_retargeting import prepare_sequence, sequence_config


def bind_sequence_guard(
    owner, model, accepted, endpoints, correction, *args, config, **kwargs
):
    cfg = sequence_config(config)
    _, _, tasks, rotations = prepare_sequence(model, endpoints, *args, cfg, **kwargs)
    if tasks is None:
        owner._sequence_acceptance_guard = None
        return
    guard = SurfaceSequenceGuard(tasks, endpoints, cfg["endpoint_regression_m"])
    guard.continuity = SequenceContinuityGuard(tasks, correction, rotations)
    guard.pose = SequencePoseGuard(tasks, correction)
    if not guard.accepts(accepted):
        raise ValueError("Sequence output exceeds its fixed contact or correction budgets")
    owner._sequence_acceptance_guard = guard
    owner._sequence_accepted_motion = np.array(accepted, copy=True)


def preserve_sequence_budgets(owner, candidate):
    guard = getattr(owner, "_sequence_acceptance_guard", None)
    if guard is None:
        return candidate, {"status": "not_available"}
    if guard.accepts(candidate):
        return candidate, {"status": "accepted", "candidate_retained": True}
    return owner._sequence_accepted_motion.copy(), {
        "status": "rejected_cumulative_regression",
        "candidate_retained": False,
        "reference": "fixed_sequence_contact_and_correction_references",
    }
