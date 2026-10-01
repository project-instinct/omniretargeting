"""Joint bounded placement of new anchors; held material anchors are immutable."""

import cvxpy as cp
import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .conic_solver import solve_conic


def footprint_constraints(delta, observed, robot):
    """Erode an observed planar footprint by the active rigid robot footprint.

    Bounds come from measured geometry, not a translation radius. Nonplanar
    patches have no certified translation freedom: their placement stays put.
    This avoids treating the convex interior of a curved surface as a surface.
    """
    observed, robot = np.asarray(observed, float), np.asarray(robot, float)
    center = observed.mean(0)
    _, singular, basis = np.linalg.svd(observed - center, full_matrices=True)
    rank = np.count_nonzero(singular > max(singular[0], 1e-12) * 1e-8)
    if rank == 0 or rank == 3:
        return [delta == 0], {"rank": int(rank), "translation_freedom": False}
    hull = None
    if rank == 2:
        try:
            hull = ConvexHull((observed - center) @ basis[:2].T)
        except QhullError:
            # qhull merges nearly coincident points more coarsely than the
            # rank tolerance above: numerically the footprint is a segment.
            rank = 1
    tangent, normal = basis[:rank].T, basis[rank:].T
    footprint = (observed - center) @ tangent
    local_robot = (robot - center) @ tangent
    if rank == 1:
        a = np.array([[1.], [-1.]])
        b = np.array([footprint.max(), -footprint.min()])
    else:
        a, b = hull.equations[:, :-1], -hull.equations[:, -1]
    allowance = b - (local_robot @ a.T).max(0)
    # Preserve the measured normal gap: anchor placement never invents a
    # different surface or silently moves a target out of its tangent plane.
    return [normal.T @ delta == 0, a @ tangent.T @ delta <= allowance], {
        "rank": int(rank), "translation_freedom": True,
        "minimum_initial_footprint_margin_m": float(allowance.min()),
    }


def reach_upper_bound(model, data, first, second):
    """Kinematic path-length bound, independent of anatomical naming."""
    def ancestors(body):
        result = [int(body)]
        while result[-1]:
            result.append(int(model.body_parentid[result[-1]]))
        return result
    a, b = ancestors(first), ancestors(second)
    common = next(body for body in a if body in b)
    edges = a[:a.index(common)] + b[:b.index(common)]
    bound = 0.
    for child in edges:
        bound += np.linalg.norm(data.xpos[child] - data.xpos[model.body_parentid[child]])
        for joint in range(model.body_jntadr[child], model.body_jntadr[child] + model.body_jntnum[child]):
            if joint >= 0 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE:
                if not model.jnt_limited[joint]:
                    return np.inf
                bound += np.ptp(model.jnt_range[joint])
    return float(bound)


def place_event(new, held, *, reach_bounds):
    """Each entry supplies material pose, footprint, and its world surface pose.

    This is an anchor feasibility prepass, not full joint-limit/pose IK. First
    minimize incompatible reach, then movement within the footprint. Infeasible
    geometry keeps the original anchors and reports the unresolved constraint.
    """
    if not new:
        return [], {"status": "no_new_contacts"}
    delta = cp.Variable((len(new), 3))
    constraints, footprints = [], []
    world = []
    for i, anchor in enumerate(new):
        rows, report = footprint_constraints(delta[i], anchor["observed"], anchor["robot_points"])
        constraints.extend(rows)
        footprints.append(report)
        world.append(anchor["surface_rotation"] @ (anchor["translation"] + delta[i])
                     * anchor["surface_scale"] + anchor["surface_translation"])
    for anchor in held:
        world.append(anchor["surface_rotation"] @ anchor["translation"]
                     * anchor["surface_scale"] + anchor["surface_translation"])
    pairs = [(a, b, bound) for (a, b), bound in reach_bounds.items() if np.isfinite(bound)]
    slack = cp.Variable(len(pairs), nonneg=True) if pairs else None
    for i, (a, b, bound) in enumerate(pairs):
        constraints.append(cp.norm(world[a] - world[b]) <= bound + slack[i])
    if pairs:
        first = cp.Problem(cp.Minimize(cp.sum(slack)), constraints)
        solve_conic(first, solver="CLARABEL")
        if first.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return [np.zeros(3) for _ in new], {"status": "incompatible_footprint", "footprints": footprints}
        # Numerical feasibility allowance only, not a physical reach budget.
        constraints.append(cp.sum(slack) <= max(float(first.value), 0.) + 1e-8)
    final = cp.Problem(cp.Minimize(cp.sum_squares(delta)), constraints)
    solve_conic(final, solver="CLARABEL")
    if final.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return [np.zeros(3) for _ in new], {"status": "incompatible_footprint", "footprints": footprints}
    return np.asarray(delta.value), {"status": "placed", "footprints": footprints,
            "remaining_reach_violation_m": [] if slack is None else np.asarray(slack.value).tolist(),
            "held_anchors_modified": False}


def place_anchor_events(model, data, placements, poses):
    """Visit source event times; each locked placement is optimized only once."""
    reports = []
    events = sorted({p["start"] for p in placements if p["locked"]})
    for frame in events:
        new = [p for p in placements if p["locked"] and p["start"] == frame]
        held = [p for p in placements if p["locked"] and p["start"] < frame < p["stop"]]
        entries = []
        for p in new + held:
            rotation, translation, scale = poses[p["surface"]]
            entries.append(dict(p, surface_rotation=rotation[frame],
                                surface_translation=translation[frame], surface_scale=scale[frame]))
        bounds = {(a, b): reach_upper_bound(model, data, entries[a]["body"], entries[b]["body"])
                  for a in range(len(new)) for b in range(a + 1, len(entries))}
        delta, report = place_event(entries[:len(new)], entries[len(new):], reach_bounds=bounds)
        for p, offset in zip(new, delta):
            p["translation"] = p["translation"] + offset
            p["event_adjustment_m"] = np.asarray(offset)
            p["robot_points"] = p["robot_points"] + offset
        reports.append(dict(report, frame=frame, new_bodies=[p["body"] for p in new]))
    return reports
