"""Select coherent observed skin points without smoothing their trajectories."""

import numpy as np

from .contact_schedule import _iter_true_runs


def coherent_sample_indices(
    distances, projected, samples, *, max_distance, switch_weight=1.0, valid=None
):
    """Minimum-cost path through actual surface projections in material axes.

    Emission cost is squared skin/surface gap. Transition cost is squared
    projected-point displacement AFTER removing the current skin sample's own
    motion. Thus a deliberate fast slide or object motion is not smoothed out;
    exchanging distant skin identities to shave a millimetre of gap has a cost.
    ``switch_weight`` is the dimensionless ratio of these two observation costs
    (one means equal position uncertainty). Zero reproduces independent nearest
    samples. The existing contact exit distance bounds admissible geometry.
    No point, contact target, object pose or frame is synthesized/interpolated.
    """
    distances = np.asarray(distances, float)
    projected, samples = np.asarray(projected, float), np.asarray(samples, float)
    limit = np.broadcast_to(np.asarray(max_distance, float), distances.shape)
    if (
        distances.ndim != 2
        or not distances.shape[1]
        or projected.shape != (*distances.shape, 3)
        or samples.shape != projected.shape
        or not all(np.isfinite(a).all() for a in (distances, projected, samples))
        or not np.isfinite(switch_weight)
        or switch_weight < 0
        or not np.isfinite(limit).all()
        or np.any(limit <= 0)
    ):
        raise ValueError(
            "Contact correspondence needs finite metric observations and nonnegative cost ratio"
        )
    selected = np.argmin(distances, axis=1)
    if switch_weight == 0 or distances.shape[1] == 1:
        return selected
    allowed = distances <= limit
    if valid is not None:
        mask = np.asarray(valid, bool)
        if mask.shape != distances.shape:
            raise ValueError("Correspondence validity must share sample identities")
        allowed &= mask
    for start, end in _iter_true_runs(allowed.any(axis=1)):
        previous = np.where(allowed[start], distances[start] ** 2, np.inf)
        parents = []
        for t in range(start + 1, end):
            predicted = projected[t] - (samples[t] - samples[t - 1])
            discrepancy = predicted[None] - projected[t - 1, :, None]
            transitions = previous[:, None] + switch_weight * np.sum(
                discrepancy**2, axis=-1
            )
            parent = np.argmin(transitions, axis=0)
            previous = (
                distances[t] ** 2 + transitions[parent, np.arange(distances.shape[1])]
            )
            previous[~allowed[t]] = np.inf
            # Remove a common offset to bound accumulated floating-point cost.
            previous -= previous.min()
            parents.append(parent)
        selected[end - 1] = int(np.argmin(previous))
        for t in range(end - 2, start - 1, -1):
            selected[t] = parents[t - start][selected[t + 1]]
    return selected
