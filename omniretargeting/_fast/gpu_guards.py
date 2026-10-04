"""Spatial moment rows for the existing acquisition normal guard.

Hinge decisions, directions and residuals use exactly the reference NumPy
expressions. Compressing each point Jacobian to six body coefficients avoids
the large (frames, points, 3, tangent-width) intermediates.
"""

import numpy as np

from .rows import _point_rows


def acquisition_rows(guard, kin, frames, weight, sigma, acc, native=None):
    scale = np.asarray(weight, dtype=float)[:, None] / sigma
    residuals, jacobians = [], []
    for kind, indices, local, target, normal, active, established, bound, yielded in guard.groups:
        local_part = local if kind == 'surface' else local[frames]
        if kind == 'surface':
            points = kin.geom_points(indices, local_part)
            bodies = guard.objective.model.geom_bodyid[indices]
        else:
            bodies = np.broadcast_to(indices, local_part.shape[:2])
            points = kin.points(bodies, local_part)
        offset = points - target[frames]
        along = np.sum(offset * normal[frames], axis=-1)
        distance = np.linalg.norm(offset, axis=-1)
        held, reach = (active & established)[frames], (active & ~established)[frames]
        out = np.maximum(along - bound[frames] - yielded[frames], 0.)
        inward = np.maximum(-along - bound[frames], 0.)
        far = np.maximum(distance - bound[frames] - yielded[frames], 0.)
        residual = np.where(held, out + inward, np.where(reach, far, 0.))
        direction = np.where(held[..., None], normal[frames] * np.sign(out - inward)[..., None],
                             offset / np.maximum(distance, 1e-12)[..., None])
        direction = np.where((residual > 0)[..., None], direction, 0.)
        # AcquisitionNormalGuard fixes its endpoint/body indices at creation.
        bodies = bodies[0] if bodies.ndim == 2 else bodies
        if native is not None:
            residuals.append(residual * scale)
            if acc.jacobian:
                jacobians.append(native.rows(bodies, points, direction) * scale[..., None])
        else:
            coefficients = _point_rows(direction, points - kin.xpos[:, bodies]) * scale[..., None]
            acc.add_body_rows(residual * scale, coefficients, bodies, kin)
    if native is not None and residuals:
        acc.add(np.concatenate(residuals, axis=1),
                np.concatenate(jacobians, axis=1) if acc.jacobian else None)
