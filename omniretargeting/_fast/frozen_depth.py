"""Original collider depths frozen at the state owned by each contact guard."""

import numpy as np


def reliable_body_depth(model, frames, measurement):
    if measurement.get('failures') or not measurement.get('complete', True):
        raise ValueError('Contact yield requires a complete, reliable measurement')
    depth = np.zeros((frames, model.nbody))
    for key, value in measurement['depths'].items():
        if key[1] not in ('static', 'moving'):
            continue
        if not np.isfinite(value) or value < 0:
            raise ValueError('Contact yield requires finite, nonnegative penetration')
        body = model.body_weldid[model.geom_bodyid[key[-1]]]
        depth[key[0], body] = max(depth[key[0], body], value)
    depth.setflags(write=False)
    return depth


class FrozenDepth:
    """Explicit freezing only; candidate checks never invoke this provider.

    Each fresh measurement owns an independent source-observation baseline.
    Repair's mandatory sample registry is a different OriginalGeometry object.
    """
    def __init__(self, objective, objects, scene):
        from .mesh_repair import OriginalGeometry
        self.geometry = OriginalGeometry(objective, objects, scene)
        self.cache = {}
        self.audits = []

    def __call__(self, poses):
        poses = np.asarray(poses)
        key = (poses.shape, poses.dtype.str, poses.tobytes())
        if key in self.cache:
            value, self.last_measurement = self.cache[key]
            return value
        geometry = self.geometry
        geometry._moving_evidence_baseline.clear()
        geometry.start_workers(len(poses))
        try:
            measured = geometry.measure(poses)
        finally:
            geometry.stop_workers()
        self.last_measurement = measured
        audit = geometry.summary(measured)
        try:
            value = reliable_body_depth(geometry.model, len(poses), measured)
            audit['yield_reliable'] = True
        except ValueError:
            # Unknown geometry grants no extra freedom, and stays reported.
            value = np.zeros((len(poses), geometry.model.nbody))
            value.setflags(write=False)
            audit['yield_reliable'] = False
        self.audits.append(audit)
        self.cache[key] = value, measured
        return value
