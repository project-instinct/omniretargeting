"""Reuse bit-identical support directions inside a short frame window."""

import numpy as np

from .._sequence.proximity import support_points


class WindowGeometry:
    def __init__(self, objective):
        self.objective = objective
        self.groups = {}
        self.bulk = None
        self.native = None
        from .window_native import library
        from .batched_self_collision import NativeUnavailable
        try:
            library()
            self.native = True
        except NativeUnavailable:
            self.native = False

    def _bulk(self):
        from .runtime_options import OPTIONS
        if not OPTIONS.bulk_cpu_geometry:
            return None
        if self.bulk is None:
            from .bulk_geometry import BulkGeometry, NativeUnavailable
            try:
                self.bulk = BulkGeometry(self.objective)
            except NativeUnavailable:
                self.bulk = False
        return self.bulk or None

    def deepest(self, sdf, rotation, translation, scales=None):
        bulk = self._bulk()
        if bulk is not None:
            return bulk.deepest(sdf, rotation, translation, scales)
        client = getattr(self.objective, '_window_shell_client', None)
        if client is not None and id(sdf) in client.fields:
            return client.deepest(self.objective, sdf, rotation, translation, scales)
        if self.native:
            from .window_native import deepest
            return deepest(self.objective, sdf, rotation, translation, scales)
        obj = self.objective
        return obj.__class__._deepest_samples(obj, sdf, rotation, translation, scales)

    def proximity_points(self, prox):
        obj, kin = self.objective, self.objective.kin
        # The native kernel certifies the original first argmax vertex. Ties
        # and its rounding band fall back to the same NumPy support routine.
        from .window_native import support
        query = support if self.native else support_points
        bulk = self._bulk()
        result = np.empty(prox['normals'].shape)
        for geom in np.unique(prox['geoms']):
            index = np.flatnonzero(prox['geoms'] == geom)
            normals = np.ascontiguousarray(prox['normals'][:, index])
            key = int(geom), normals.shape, normals.tobytes()
            if key not in self.groups:
                count = len(index)
                keys = np.empty((obj.frames * count, 4), np.uint64)
                keys[:, 0] = np.arange(obj.frames).repeat(count)
                keys[:, 1:] = normals.reshape(-1, 3).view(np.uint64)
                _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
                self.groups[key] = first, inverse
            first, inverse = self.groups[key]
            frame, sample = first // len(index), index[first % len(index)]
            rotation = kin.geom_xmat[frame, geom]
            directions = -np.matmul(prox['normals'][frame, sample, None, :], rotation)[:, 0]
            local = (query(obj.model, int(geom), directions) if bulk is None else
                     bulk.support_unique(int(geom), directions, fallback=query))
            point = kin.geom_xpos[frame, geom] + np.matmul(rotation, local[..., None])[..., 0]
            result[:, index] = point[inverse].reshape(obj.frames, len(index), 3)
        return result
