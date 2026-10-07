"""Conservative bounds for the existing linear maximum-depth prediction.

These certify model progress, not nonlinear clearance. Dropping constraints
enlarges the feasible set, so a failed certificate always falls back to the QP.
No source reference, guard, activation or progress threshold is changed.
"""

import numpy as np

from . import repair_prediction
from .repair_frames import restrict


def progress_bound(limits, poses, geometry, measurement, depth):
    """Upper bound on group-maximum reduction over the projected joint box.

    For every witness, maximize its clearance over the allowed coordinates.
    The maximum depth is at least the negative of each such maximum clearance.
    Unrepresented keys retain their depth, exactly as in ``linear_prediction``.
    Include the QP's box tolerance and outward arithmetic padding. Non-selector
    projections, incomplete evidence and nonfinite data cannot certify a skip.
    """
    from .mesh_repair import _joint_projection
    projection = restrict(_joint_projection(limits, geometry.model, measurement), limits.frame_mask)
    if not projection.shape[1]:
        return dict(reason="no_local_dofs", upper_decrease_m=0., movable_coordinates=0)
    if (np.any(np.diff(projection.indptr) != 1) or np.any(projection.data != 1.)
            or len(np.unique(projection.indices)) != projection.shape[1]
            or measurement.get("failures") or not measurement.get("complete", True)):
        return None
    values = poses[:, limits.coords.qi]
    # Zero remains in the interval, matching LinearBounds.set_box's range.
    low = np.minimum(limits.lower - values - 1e-8, 0.)
    high = np.maximum(limits.upper - values + 1e-8, 0.)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or not np.isfinite(depth):
        return None
    selected = np.zeros(limits.frames * limits.width, bool)
    selected[projection.indices] = True
    selected = selected.reshape(limits.frames, limits.width)
    low, high = np.where(selected, low, 0.), np.where(selected, high, 0.)
    from .repair_triage_temporal import tighten
    tightened = tighten(limits, values, low, high)
    if tightened is None:
        return None
    low, high = tightened
    model, body = geometry.model, limits.movable_body

    def relevant(key):
        geoms = key[2:] if key[1] == "self" else key[-1:]
        return body is None or any(model.body_weldid[model.geom_bodyid[g]] == body for g in geoms)

    rows = measurement["rows"]
    keys = [key for key, _ in rows]
    try:
        jac = np.asarray([row.get("jacobian") for _, row in rows], dtype=float).reshape(-1, limits.width)
        phi = np.asarray([row["clearance"] for _, row in rows], dtype=float)
        depths = np.asarray(list(measurement["depths"].values()), dtype=float)
    except (TypeError, ValueError):
        return None
    if (len(jac) != len(rows) or phi.shape != (len(rows),)
            or not np.isfinite(jac).all() or not np.isfinite(phi).all()
            or not np.isfinite(depths).all()):
        return None
    relevant_indices = np.array([i for i, key in enumerate(keys) if relevant(key)], dtype=int)
    represented = {keys[i] for i in relevant_indices}
    lower_depth, binding = 0., None
    frame = np.array([keys[i][0] for i in relevant_indices], dtype=int)
    selected_jac = jac[relevant_indices]
    terms = np.where(selected_jac >= 0., selected_jac * high[frame], selected_jac * low[frame])
    upper_phi = phi[relevant_indices] + terms.sum(axis=1) + 1e-10 * (1. + np.abs(terms).sum(axis=1))
    if len(upper_phi):
        index = int(np.argmin(upper_phi))
        if upper_phi[index] < 0.:
            lower_depth, binding = -float(upper_phi[index]), keys[relevant_indices[index]]
    for key, value in measurement["depths"].items():
        if relevant(key) and key not in represented:
            if not np.isfinite(value):
                return None
            if value > lower_depth:
                lower_depth, binding = float(value), key
    reason = "projected_witness_temporal_box"
    from .mesh_repair import MIN_GROUP_PROGRESS_M
    if float(depth) - lower_depth >= MIN_GROUP_PROGRESS_M:
        from .repair_triage_dual import maximum
        # A single blocking witness suffices. Bound the most promising three
        # rows; uncertain/time-limited LPs retain the cheap box certificate.
        deepest = relevant_indices[np.argsort(phi[relevant_indices], kind='stable')[:3]]
        for index in deepest:
            key, row = rows[index]
            frame = key[0]
            witnesses = [item for item in rows if item[0][0] == frame]
            upper = maximum(jac[index], low[frame], high[frame], witnesses, measurement['depths'])
            if upper is not None and -row['clearance'] - upper > lower_depth:
                lower_depth, binding = -float(row['clearance']) - upper, key
                reason = 'protected_witness_dual'
            if float(depth) - lower_depth < MIN_GROUP_PROGRESS_M:
                break
    return dict(reason=reason, upper_decrease_m=float(depth) - lower_depth,
                lower_predicted_depth_m=lower_depth,
                binding_key=None if binding is None else list(binding),
                movable_coordinates=int(projection.shape[1]))


def certificate(limits, poses, geometry, measurement, depth):
    """Only opt-in, strict certificates under the established predictive gate."""
    if not repair_prediction.PREDICTIVE_GATING:
        return None
    from .mesh_repair import MIN_GROUP_PROGRESS_M
    if MIN_GROUP_PROGRESS_M <= 0:
        return None
    bound = progress_bound(limits, poses, geometry, measurement, depth)
    if bound is None or bound["upper_decrease_m"] >= MIN_GROUP_PROGRESS_M:
        return None
    return {**bound, "threshold_m": MIN_GROUP_PROGRESS_M,
            "entry_depth_m": float(depth), "predicted_decrease_m": bound["upper_decrease_m"],
            "predicted_depth_m": float(depth) - bound["upper_decrease_m"],
            "model": "pre_qp_upper_bound"}
