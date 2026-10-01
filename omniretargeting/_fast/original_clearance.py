"""Early original-collider rows for overlaps outside the local repair box."""

import copy

import mujoco
import numpy as np


class OriginalClearance:
    def __init__(self, geometry, geoms):
        self.model = geometry.model
        self.geoms = set(geoms)
        self.scene = geometry.scene
        self.moving = None if geometry.moving is None else copy.copy(geometry.moving)
        if self.moving is not None:
            self.moving.shells = [s for s in self.moving.shells if s['geom_id'] in self.geoms]
            self.moving.data = mujoco.MjData(self.model)
        objective = geometry.objective
        self.static = [(g, objective.shell_local[objective.shell_sample_geom == g]) for g in sorted(self.geoms)]
        self.data = mujoco.MjData(self.model)

    def evaluate(self, poses, first=0):
        """One deepest original row per field/geom; times stay globally aligned."""
        result = {}
        offset = int(self.scene is not None)
        pending, count = [], 0
        limit = 1 << 20
        for frame, pose in enumerate(poses):
            if self.moving is not None:
                measured = self.moving.query(pose, first + frame, [])
                for row in measured['rows']:
                    key = (offset + row['object_id'], frame, row['geom_id'])
                    if key not in result or row['clearance'] < result[key]['clearance']:
                        result[key] = row
            if self.scene is not None:
                self.data.qpos[:] = pose
                mujoco.mj_forward(self.model, self.data)
                for geom, local in self.static:
                    points = local @ self.data.geom_xmat[geom].reshape(3, 3).T + self.data.geom_xpos[geom]
                    cursor = 0
                    while cursor < len(points):
                        group = points[cursor:cursor + limit - count]
                        pending.append((frame, geom, group))
                        cursor += len(group)
                        count += len(group)
                        if count == limit:
                            self._static_batch(result, pending)
                            pending, count = [], 0
        if pending:
            self._static_batch(result, pending)
        return result

    def _static_batch(self, result, pending):
        """Amortize original-mesh query setup, preserving each first minimum."""
        points = np.concatenate([points for _, _, points in pending])
        signed = self.scene.query(points)
        first = 0
        for frame, geom, group in pending:
            last = first + len(group)
            value = signed['clearance'][first:last]
            valid = ~signed['uncertain'][first:last] & np.isfinite(value)
            if valid.any():
                index = first + int(np.argmin(np.where(valid, value, np.inf)))
                clearance = float(signed['clearance'][index])
                key = (0, frame, geom)
                if key not in result or clearance < result[key]['clearance']:
                    result[key] = dict(clearance=clearance, direction=signed['directions'][index],
                                       point=points[index])
            first = last


def replace_deeper_original(field, geoms, measured, clearance, deficit, rows, kin, bodies, target):
    """Replace grid rows only where reliable original geometry is more restrictive.

    The row count and weighting stay per geom. A missing/unknown original
    witness never erases an existing grid or exit-plane constraint.
    """
    from .terms import _point_rows
    columns = {int(g): i for i, g in enumerate(geoms)}
    for (source, frame, geom), witness in measured.items():
        if source != field or geom not in columns:
            continue
        column = columns[geom]
        phi = witness['clearance']
        if not np.isfinite(phi) or phi >= target:
            continue
        if np.isfinite(clearance[frame, column]) and phi >= clearance[frame, column]:
            continue
        deficit[frame, column] = phi - target
        lever = witness['point'] - kin.xpos[frame, bodies[column]]
        rows[frame, column] = _point_rows(witness['direction'], lever)
