"""Reuse bit-identical support directions inside a short frame window."""

from collections import OrderedDict

import numpy as np

from .._sequence.proximity import support_points


def _array_key(value):
    if value is None:
        return None
    value = np.asarray(value)
    return value.shape, value.dtype.str, value.tobytes()

class WindowGeometry:
    def __init__(self, objective):
        self.objective = objective
        self.groups = {}
        self.results = OrderedDict()
        self.packs = OrderedDict()
        self.diagnostics = {}
        self._shell_refs = None
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
        if self.bulk is None:
            from .bulk_geometry import BulkGeometry, NativeUnavailable
            try:
                self.bulk = BulkGeometry(self.objective)
                self.bulk.window_support_reuse = True
                from .window_query_native import WindowQueries
                try:
                    self.bulk.window_queries = WindowQueries(self.objective)
                except NativeUnavailable:
                    pass
            except NativeUnavailable:
                self.bulk = False
        return self.bulk or None

    def _remember(self, key, query):
        if key in self.results:
            self.results.move_to_end(key)
            self.diagnostics['hits'] = self.diagnostics.get('hits', 0) + 1
            value = self.results[key]
        else:
            self.diagnostics['misses'] = self.diagnostics.get('misses', 0) + 1
            value = query()
            self.results[key] = value
            # Bound large proximity point arrays in persistent window workers.
            while len(self.results) > 12:
                self.results.popitem(last=False)
        # Consumers may annotate clearance arrays; keep owned return buffers.
        return tuple(a.copy() for a in value) if isinstance(value, tuple) else value.copy()

    def deepest(self, sdf, rotation, translation, scales=None):
        obj = self.objective
        from .sdf import GridSDF
        if type(sdf) is not GridSDF:
            return self._deepest(sdf, rotation, translation, scales)
        # SDF grids and robot shells are immutable within acquisition. Include
        # their identities and every mutable field transform/threshold; pack
        # replacements and in-place object-motion changes cannot reuse a result.
        refs = tuple(getattr(obj, name) for name in ('shell_geoms', 'shell_index', 'shell_local', 'shell_radius'))
        if self._shell_refs is None:
            self._shell_refs = refs
        elif any(a is not b for a, b in zip(refs, self._shell_refs)):
            self.results.clear()
            self.bulk = None
            self._shell_refs = refs
        # The immutable shell index can be tens of MB. Do not copy/hash it
        # for every pose or for distant fields with no near geom at all.
        key = ('deepest', sdf, id(sdf.clearance),
               _array_key(getattr(sdf, 'lower', None)), getattr(sdf, 'spacing', None),
               obj.cfg['scene_clearance_m'], _array_key(rotation),
               _array_key(translation), _array_key(scales),
               _array_key(obj.kin.geom_xpos), _array_key(obj.kin.geom_xmat))
        return self._remember(key, lambda: self._deepest(sdf, rotation, translation, scales))

    def _deepest(self, sdf, rotation, translation, scales):
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
        kin = self.objective.kin
        signature = _array_key(prox['geoms']), _array_key(prox['normals'])
        if signature not in self.packs:
            self.packs[signature] = signature
        self.packs.move_to_end(signature)
        while len(self.packs) > 8:
            self.packs.popitem(last=False)
        # Reuse the immutable bytes, rather than retaining another full normal
        # pack (several MB) in every candidate's result key.
        key = ('proximity', self.packs[signature],
               _array_key(kin.geom_xpos), _array_key(kin.geom_xmat))
        return self._remember(key, lambda: self._proximity_points(prox))

    def _proximity_points(self, prox):
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
