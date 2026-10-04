"""Coalesced exact geometry, with independently shaped GPU window reductions.

Only frame-independent certified support selections and shell queries are
coalesced. Each request keeps its own float64 reduction dimensions regardless
of batch membership/order, so coalescing cannot select another GEMM shape.
Native convex distances remain parallel CPU work, one private query per view.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import time
import threading
from types import SimpleNamespace

import numpy as np


class PreparedShell:
    def __init__(self, original, fields):
        self.original, self.fields = original, fields

    def prepare(self, kin):
        self.original.prepare(kin)
        self.position, self.rotation = self.original.position, self.original.rotation

    def deepest(self, sdf, rotation, translation, scales=None):
        return self.fields[id(sdf), id(rotation), id(translation), id(scales)]


class WindowEvaluationBatch:
    def __init__(self, objective, *, device_moments=True, workers=8):
        from .gpu_objective import GPUFrameEvaluator
        self.objective, self.workers = objective, workers
        self.views = OrderedDict()
        self.geometry = GPUFrameEvaluator(objective)
        self.device_moments = device_moments
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.requests = self.batches = 0
        self.seconds = 0.
        self.statistics_lock = threading.Lock()

    def view(self, owner, first, stop):
        key = owner, first, stop
        if key not in self.views:
            from .batched_self_collision import BatchedSelfCollision
            from .gpu_objective import GPUFrameEvaluator
            obj = self.objective.for_frames(slice(first, stop))
            obj._gpu_pid = None
            if obj.self_collision is not None:
                obj.self_collision = obj.self_collision.fresh_state()
                obj._self_batch = BatchedSelfCollision(obj.self_collision, workers=1)
            evaluator = GPUFrameEvaluator(obj)
            evaluator.device_moments = self.device_moments
            self.views[key] = evaluator
        self.views.move_to_end(key)
        while len(self.views) > 64:
            self.views.popitem(last=False)
        return self.views[key]

    @staticmethod
    def prepare(item):
        evaluator, request = item
        _, _, poses, jacobian, _ = request
        evaluator.objective.kin.evaluate(poses, jacobian, native=True)
        return evaluator, request

    def shell_fields(self, items):
        from .gpu_shell import GPUShell
        from .window_tracking import skips_proxy_collision
        prepared = [dict() for _ in items]
        if skips_proxy_collision(self.objective):
            return prepared
        lengths = [len(request[2]) for _, request in items]
        edges = np.r_[0, np.cumsum(lengths)]
        # Shared immutable hull/shell definitions; only transforms are packed.
        kin = SimpleNamespace(**{name: np.concatenate([getattr(e.objective.kin, name) for e, _ in items])
                                  for name in ('geom_xpos', 'geom_xmat')})
        proxy = copy.copy(items[0][0].objective)
        proxy.frames, proxy.kin = int(edges[-1]), kin
        geometry = self.geometry
        original = geometry.objective
        geometry.objective = proxy
        try:
            shell = GPUShell(geometry)
            shell.local = geometry.shell.local
            shell.prepare(kin)
            fields = [list(e.objective._clearance_fields()) for e, _ in items]
            for field_index in range(len(fields[0])):
                group = [values[field_index] for values in fields]
                assert all(row[0] == group[0][0] and row[1] is group[0][1] for row in group)
                sdf = group[0][1]
                def cat(index):
                    return None if group[0][index] is None else np.concatenate([row[index] for row in group])
                result = shell.deepest(geometry.grid(sdf), cat(2), cat(3), cat(4))
                for i, (evaluator, _) in enumerate(items):
                    lo, hi = edges[i:i + 2]
                    clearance, frames, geoms, world, local = result
                    take = (frames >= lo) & (frames < hi)
                    row = group[i]
                    key = id(evaluator.grid(sdf)), id(row[2]), id(row[3]), id(row[4])
                    prepared[i][key] = (clearance[lo:hi], frames[take] - lo,
                                        geoms[take], world[take], local[take])
        finally:
            geometry.objective = original
        return prepared

    def supports(self, items):
        # Geometry queries use the same CPU direction/normalization expressions
        # as GPUFrameEvaluator. The returned local vertices are certified.
        requests, lookup = {}, [dict() for _ in items]
        for i, (evaluator, _) in enumerate(items):
            obj = evaluator.objective
            for prox in obj.proximities:
                for geom in np.unique(prox['geoms']):
                    index = np.flatnonzero(prox['geoms'] == geom)
                    first, _ = evaluator._group(prox, int(geom))
                    frame, sample = first // len(index), index[first % len(index)]
                    directions = -np.matmul(prox['normals'][frame, sample, None, :],
                                            obj.kin.geom_xmat[frame, geom])[:, 0]
                    requests.setdefault(int(geom), []).append((i, directions))
        for geom, group in requests.items():
            offsets = np.r_[0, np.cumsum([len(d) for _, d in group])]
            points = self.geometry.support_unique(geom, np.concatenate([d for _, d in group]))
            for (index, direction), lo, hi in zip(group, offsets[:-1], offsets[1:]):
                lookup[index][geom, direction.shape, direction.tobytes()] = points[lo:hi]
        return lookup

    @staticmethod
    def evaluate(item):
        evaluator, request, fields, supports = item
        _, _, poses, jacobian, weight = request
        shell, support = evaluator.shell, evaluator.support_unique
        evaluator.shell = PreparedShell(shell, fields)
        def selected(geom, directions):
            array = np.asarray(directions, float)
            return supports[geom, array.shape, array.tobytes()]
        evaluator.support_unique = selected
        try:
            measured = evaluator.evaluate(poses, jacobian, weight)
            # The temporal evaluator in the child already owns these points.
            for key in ('root_point', 'root_jacobian', 'effector_point', 'effector_jacobian'):
                measured.pop(key)
            return measured
        finally:
            evaluator.shell, evaluator.support_unique = shell, support

    def submit(self, requests):
        """Coalesce geometry, then release independent frame evaluations.

        A slow convex-distance query must not delay another child's response
        or keep the service from accepting its next objective evaluation.
        """
        items = [(self.view(owner, request[0], request[1]), request) for owner, request in requests]
        items = list(self.executor.map(self.prepare, items))
        fields, supports = self.shell_fields(items), self.supports(items)
        self.requests += len(items)
        self.batches += 1
        def evaluated(item):
            started = time.monotonic()
            try:
                return self.evaluate(item)
            finally:
                with self.statistics_lock:
                    self.seconds += time.monotonic() - started
        return [self.executor.submit(evaluated, (e, r, f, s))
                for (e, r), f, s in zip(items, fields, supports)]

    def run(self, requests):
        return [future.result() for future in self.submit(requests)]

    def close(self):
        self.executor.shutdown()
