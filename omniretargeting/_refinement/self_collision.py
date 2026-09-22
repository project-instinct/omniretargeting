"""Collision-body rows and nonlinear acceptance for joint trajectory updates."""

import numpy as np
from scipy import sparse


def linearize_self_collisions(query, trajectory, dofs):
    rows, cols, values, rhs = [], [], [], []
    sigma = query.cfg["slack_sigma"]
    target = query.cfg["minimum_distance"]
    def append(collision, jac, endpoints):
        if collision["clearance"] >= target:
            return
        jac = jac[dofs] / sigma
        nonzero = np.flatnonzero(np.abs(jac) > 1e-14)
        for frame, weight in endpoints:
            rows.extend([len(rhs)] * len(nonzero))
            cols.extend(frame * len(dofs) + nonzero)
            values.extend(weight * jac[nonzero])
        rhs.append((target - collision["clearance"]) / sigma)

    for frame, q in enumerate(trajectory):
        for collision in query.query(q):
            append(collision, query.jacobian(collision, query.data), [(frame, 1.0)])
    # A clear pair of saved poses can still tunnel during playback. Both
    # endpoint joint vectors are variables in this solve; using only alpha*J
    # (the framewise convention) would omit the first endpoint's escape motion.
    for frame, (first, second) in enumerate(zip(trajectory[:-1], trajectory[1:])):
        for collision in query.segment(first, second, jacobians=True)["rows"]:
            alpha = collision["alpha"]
            if alpha >= 1.0:
                continue
            append(collision, collision["jacobian"] / alpha,
                   [(frame, 1.0 - alpha), (frame + 1, alpha)])
    block = sparse.coo_matrix(
        (values, (rows, cols)), shape=(len(rhs), len(trajectory) * len(dofs))
    ).tocsr()
    return block, np.asarray(rhs)


def guard_self_collision_update(query, before, candidate, qpos_indices):
    """Backtrack one trajectory step together, preserving temporal coherence."""
    old = [query.depths(q) for q in before]
    old_intervals = [query.segment(a, b) for a, b in zip(before[:-1], before[1:])]
    delta = candidate[:, qpos_indices] - before[:, qpos_indices]
    for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        trial = before.copy()
        trial[:, qpos_indices] += alpha * delta
        if not all(query.acceptable(depths, query.depths(q)) for depths, q in zip(old, trial)):
            continue
        valid = True
        for previous, a, b in zip(old_intervals, trial[:-1], trial[1:]):
            current = query.segment(a, b)
            if current["subdivision_limit"] or not query.acceptable(previous["depths"], current["depths"]):
                valid = False
                break
        if valid:
            return trial
    return before.copy()
