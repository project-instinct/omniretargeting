"""Opt-in signed contact-plane protection for scalar-joint repair limits."""

import numpy as np

from .rows import _block_rows
from .surface_presence import on_surface


def prepare(limits):
    obj = limits.objective
    normals = [eff['normal'][:, None] for eff in obj.effectors] + [s['normals'] for s in obj.surfaces]
    result = []
    for (points, _, budget), normal, target in zip(limits.points, normals, limits.targets):
        normal = np.array(normal, dtype=float, copy=True)
        size = np.linalg.norm(normal, axis=-1)
        valid = np.isfinite(normal).all(axis=-1) & (size >= 1e-12)
        normal = np.divide(normal, size[..., None], out=np.zeros_like(normal), where=valid[..., None])
        held = (budget == .001) & valid & on_surface(obj, points)
        target = np.array(target, copy=True)
        floor = np.minimum(np.sum((points-target)*normal, axis=-1), 0.)
        for value in (normal, target, held, floor):
            value.setflags(write=False)
        result.append((normal, target, held, floor))
    return tuple(result)


def add(limits, index, points, jac, bounds):
    normal, target, held, floor = limits.contact_planes[index]
    if held.any():
        along = np.sum((points-target)*normal, axis=-1)
        derivative = np.einsum('fsi,fsiw->fsw', normal, jac)
        bounds.add(_block_rows(derivative, limits.width)[held.ravel()],
                   (floor-along)[held] - 5e-8, np.inf)


def check(limits, index, points):
    normal, target, held, floor = limits.contact_planes[index]
    excess = floor - np.sum((points-target)*normal, axis=-1)
    bad = held & (~np.isfinite(excess) | (excess > 1e-7))
    if bad.any():
        return {'reason': 'contact into supporting surface',
                'violating_frames': np.flatnonzero(bad.any(axis=1)).tolist(),
                'max_inward_regression_m': float(excess[bad].max())}
    return None
