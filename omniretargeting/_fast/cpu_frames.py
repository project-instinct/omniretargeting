"""Reference-order CPU rows over small frame batches, with GPU geometry.

The device selects supports and deepest grid samples once for the complete
trajectory. Independent frame batches then retain NumPy's BLAS calls and
reduction order; the dense rows fit in cache rather than crossing processes.
"""

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import copy
import time

import numpy as np

from .acquisition_guard import AcquisitionNormalGuard
from .gpu_guards import acquisition_rows
from .native_point_rows import NativePointRows
from .rows import _Accumulator, _DeferredRows
from .terms import ENVELOPE_SIGMA


class ReferenceFrames:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def proximity_points(self, prox):
        evaluator, obj = self.evaluator, self.evaluator.objective
        result = np.empty(prox['normals'].shape)
        for geom in np.unique(prox['geoms']):
            index = np.flatnonzero(prox['geoms'] == geom)
            first, inverse = evaluator._group(prox, int(geom))
            frame, sample = first // len(index), index[first % len(index)]
            rotation = obj.kin.geom_xmat[frame, geom]
            directions = -np.matmul(prox['normals'][frame, sample, None, :], rotation)[:, 0]
            local = evaluator.support_unique(int(geom), directions)
            points = obj.kin.geom_xpos[frame, geom] + np.matmul(rotation, local[..., None])[..., 0]
            result[:, index] = points[inverse].reshape(obj.frames, len(index), 3)
        return result

    @staticmethod
    def envelope(chunk, acc, stats, native):
        if chunk.temporal_reference is None:
            return
        rows, jac, excess = chunk.relation_envelope.hinge_rows(
            chunk.kin, chunk.envelope_weight, ENVELOPE_SIGMA, acc.jacobian)
        stats['layout_flip_max'] = excess
        acc.add(rows, jac)
        for guard in getattr(chunk, 'hinge_guards', ()):
            if type(guard) is AcquisitionNormalGuard and native is not None:
                acquisition_rows(guard, chunk.kin, chunk.frame_slice,
                                 chunk.envelope_weight, ENVELOPE_SIGMA, acc, native)
            else:
                rows, jac = guard.hinge_rows(chunk.kin, chunk.frame_slice,
                                             chunk.envelope_weight, ENVELOPE_SIGMA, acc.jacobian)
                if rows is not None:
                    acc.add(rows, jac)

    def evaluate(self, poses, with_jacobian, envelope_weight):
        evaluator, obj = self.evaluator, self.evaluator.objective
        timings = {}

        def collisions():
            started = time.perf_counter()
            acc = _Accumulator(obj.frames, obj.width, with_jacobian)
            stats = {}
            obj._self_seed_provider = evaluator.self_seeds
            try:
                obj._self_collision_terms(poses, acc, stats)
            finally:
                del obj._self_seed_provider
            timings['self_collision'] = time.perf_counter() - started
            return acc, stats

        with ThreadPoolExecutor(max_workers=1) as collision_worker:
            collision = collision_worker.submit(collisions)
            started = time.perf_counter()
            points = [self.proximity_points(prox) for prox in obj.proximities]
            timings['support_points'] = time.perf_counter() - started
            fields = {}
            started = time.perf_counter()
            from .window_tracking import skips_proxy_collision
            if not skips_proxy_collision(obj):
                for _, sdf, rotation, translation, scales in obj._clearance_fields():
                    fields[id(sdf)] = evaluator.shell.deepest(evaluator.grid(sdf), rotation, translation, scales)
            timings['deepest_samples'] = time.perf_counter() - started

            def batch(bounds):
                start, stop = bounds
                frame_slice = slice(start, stop)
                kin = copy.copy(obj.kin)
                kin.cache = OrderedDict()
                chunk = obj.for_frames(frame_slice, kinematics=kin)
                chunk.envelope_weight = envelope_weight[frame_slice]
                for name in ('xpos', 'xmat', 'geom_xpos', 'geom_xmat', 'jacp', 'jacr'):
                    if with_jacobian or not name.startswith('jac'):
                        setattr(chunk.kin, name, getattr(obj.kin, name)[frame_slice])
                native = NativePointRows(chunk.kin) if with_jacobian else None
                if native is not None:
                    chunk.kin.point_jacobians = native.jacobians
                prox_points = {id(prox): point[frame_slice]
                               for prox, point in zip(chunk.proximities, points)}
                chunk._proximity_points = lambda prox: prox_points[id(prox)]

                def deepest(sdf, rotation, translation, scales=None):
                    clearance, frames, geoms, world, field = fields[id(sdf)]
                    chosen = (frames >= start) & (frames < stop)
                    return clearance[frame_slice], frames[chosen] - start, geoms[chosen], world[chosen], field[chosen]

                chunk._deepest_samples = deepest
                acc = _Accumulator(chunk.frames, chunk.width, with_jacobian)
                stats = {}
                chunk._mapped_terms(acc, stats)
                chunk._orientation_terms(acc, stats)
                chunk._effector_terms(acc, stats)
                chunk._surface_terms(acc, stats)
                chunk._proximity_terms(acc, stats)
                chunk._clearance_terms(acc, stats, poses[frame_slice])
                chunk._plane_terms(acc, stats)
                later = _DeferredRows(with_jacobian)
                chunk._reference_terms(poses[frame_slice], later, stats)
                self.envelope(chunk, later, stats, native)
                # Release this worker before waiting for collision, so the
                # queued frame batches can prepare their rows concurrently.
                return acc, later, stats

            # Original-mesh overrides own mutable query buffers. Keep their
            # reference single-consumer semantics until separately batched.
            workers = 1 if getattr(obj, 'original_clearance', None) is not None else min(8, max(1, obj.frames // 32))
            count = min(2 * workers, max(1, obj.frames // 32))
            edges = np.linspace(0, obj.frames, count + 1).astype(int)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                started = time.perf_counter()
                parts = list(executor.map(batch, zip(edges[:-1], edges[1:])))
                timings['frame_rows'] = time.perf_counter() - started
            measured, stats = collision.result()
            for (start, stop), (part, later, _) in zip(zip(edges[:-1], edges[1:]), parts):
                part.frame_cost += measured.frame_cost[start:stop]
                if with_jacobian:
                    part.hessian += measured.hessian[start:stop]
                    part.gradient += measured.gradient[start:stop]
                later.insert(part)

        acc = _Accumulator(obj.frames, obj.width, with_jacobian)
        acc.frame_cost = np.concatenate([part.frame_cost for part, _, _ in parts])
        acc.cost = float(acc.frame_cost.sum())
        if with_jacobian:
            acc.hessian = np.concatenate([part.hessian for part, _, _ in parts])
            acc.gradient = np.concatenate([part.gradient for part, _, _ in parts])
        for _, _, measured in parts:
            for key, value in measured.items():
                stats[key] = (stats.get(key, 0) + value if key.endswith('_frames')
                              else max(stats.get(key, 0.), value))
        effector_point, effector_jacobian = obj._effector_points(with_jacobian)
        evaluator.diagnostics['timings'] = timings
        return dict(cost=acc.cost, frame_cost=acc.frame_cost,
                    hessian=acc.hessian if with_jacobian else None,
                    gradient=acc.gradient if with_jacobian else None, stats=stats,
                    root_point=obj._root_points(), root_jacobian=obj._root_jacobian() if with_jacobian else None,
                    effector_point=effector_point, effector_jacobian=effector_jacobian)
