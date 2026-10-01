"""Prescribed object collision rows on the same timeline as contact targets."""

import numpy as np

from ..geometry_query import GeometryQueryError
from ..moving_scene import MovingSceneQuery


class PrescribedSceneClearance:
    """Expose original moving meshes to the generic sequence clearance level."""

    def __init__(self, model, objects, frames, *, clearance, query_config=None):
        self.query = MovingSceneQuery(model, objects, frames, query_config)
        self.clearance = float(clearance)
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError("Moving-scene clearance must be finite and nonnegative")
        self.cached = {}

    def prepare_trajectory(self, poses):
        self.query.prepare_trajectory(poses)

    def clearance_rows(self, pose, frame, dofs, stats):
        key = (np.ascontiguousarray(pose).tobytes(), tuple(dofs))
        entry = self.cached.get(frame)
        if entry is None or entry[0] != key:
            entry = (key, self.query.query(pose, frame, dofs))
            self.cached[frame] = entry
        measured = entry[1]
        if measured["failures"]:
            raise GeometryQueryError(
                f"Moving scene query at frame {frame}: {measured['failures']}"
            )
        stats["moving_depth_m"] = max(
            float(measured["forward_depth"].max(initial=0)),
            float(measured["reverse_depth"].max(initial=0)),
        )
        for row in measured["rows"]:
            if row["clearance"] < self.clearance:
                yield (
                    np.atleast_1d(row["clearance"] - self.clearance),
                    row["jacobian"][None],
                )

    def rows(self, pose, frame, dofs):
        # Clearance is handled once, alongside scene/self collision, so it is
        # also included in the hierarchy's clearance level when roles are used.
        return ()
