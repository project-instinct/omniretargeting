"""Allocation-efficient reference-order point Jacobians and normal rows."""

import numpy as np

from .batched_self_collision import native_library, _pointer


class NativePointRows:
    def __init__(self, kin):
        self.library = native_library()
        self.kin = kin
        self.jacp = np.ascontiguousarray(kin.jacp)
        self.jacr = np.ascontiguousarray(kin.jacr)

    def _arrays(self, bodies, points):
        points = np.ascontiguousarray(points, dtype=float)
        ids = np.ascontiguousarray(np.broadcast_to(np.asarray(bodies, np.int64), points.shape[:2]))
        return ids, points

    def jacobians(self, bodies, points):
        ids, points = self._arrays(bodies, points)
        frames, count = ids.shape
        width = self.jacp.shape[-1]
        result = np.empty((frames, count, 3, width))
        self.library.ws3_point_jac(frames, self.jacp.shape[1], count, width,
                                  *(_pointer(a) for a in (ids, points, self.kin.xpos, self.jacp, self.jacr, result)))
        return result

    def rows(self, bodies, points, directions):
        ids, points = self._arrays(bodies, points)
        frames, count = ids.shape
        width = self.jacp.shape[-1]
        result = np.empty((frames, count, width))
        directions = np.ascontiguousarray(directions, dtype=float)
        self.library.ws3_point_rows(frames, self.jacp.shape[1], count, width,
                                   *(_pointer(a) for a in (ids, points, self.kin.xpos, self.jacp, self.jacr, directions, result)))
        return result
