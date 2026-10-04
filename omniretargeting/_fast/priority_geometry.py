"""Bounded acquisition adapter around the original-mesh witness service.

The repair service defaults to scalar-joint columns. Acquisition must include
the floating-base tangent columns as well; no root Jacobian may be dropped.
"""

from collections import OrderedDict
import os

import numpy as np


def acquisition_coordinator(objective, reference, objects, scene, resources):
    """Own the geometry lifetime, keeping window forks free of legacy pools."""
    from .priority_clearance import PriorityClearance
    from .runtime_options import OPTIONS
    if resources is None:
        raise RuntimeError('Priority acquisition requires an owned service lifetime')
    provider = PriorityGeometry(objective, objects, scene)
    resources.callback(provider.close)
    if not OPTIONS.windowed_priority:
        provider.start()
    coordinator = PriorityClearance(objective, provider, reference)
    objective.priority_clearance = coordinator
    return coordinator


class PriorityGeometry:
    def __init__(self, objective, objects=(), scene=None, *, geometry=None):
        if geometry is None:
            from .mesh_repair import OriginalGeometry
            geometry = OriginalGeometry(objective, objects, scene, manifold=True,
                                        columns=objective.coords.cols)
        self.geometry = geometry
        self.owner_pid = os.getpid()
        self.objective = objective
        if not np.array_equal(geometry.columns, objective.coords.cols):
            raise ValueError('Priority geometry requires the full acquisition tangent columns')
        self.cache = OrderedDict()
        self.started = False

    def freeze_reference(self, poses):
        """Freeze original observations before even the initial depth measurement."""
        return self.geometry.freeze_reference_observations(poses)

    def start(self):
        if not self.started:
            self.geometry.start_workers(self.objective.frames)
            self.started = True

    def close(self):
        if self.started:
            self.geometry.stop_workers()
            self.started = False
        self.cache.clear()

    def fork_private_client(self):
        result = PriorityGeometry(self.objective, geometry=self.geometry.fork_private_client())
        result._private = True
        return result

    def measure(self, poses, *, jacobians=False, frames=None):
        if os.getpid() != self.owner_pid:
            if not getattr(self, '_private', False):
                raise RuntimeError('Priority geometry queries require the owning process; forked clients are unsafe')
            self.geometry = self.geometry.fork_private_client()
            self.owner_pid, self.started = os.getpid(), False
            self.cache = OrderedDict()
        poses = np.asarray(poses)
        if frames is not None:
            from .priority_window_contacts import checked_frames
            part = checked_frames(frames, len(poses))
            key = (frames.start, frames.stop, poses.dtype.str, poses[part].tobytes(), jacobians)
            measured = self.cache.pop(key, None)
            cache_hit = measured is not None
            if measured is None:
                measured = self.geometry.measure(poses, frames=frames, jacobians=jacobians)
            scope = dict(measured.get('query_scope', {}))
            scope.update(requested_frames=list(frames), requested_frame_count=len(frames),
                         adapter_cache_hit=cache_hit)
            if cache_hit:
                scope.update(queried_frames=[], queried_frame_count=0, cached_frame_count=len(frames))
            measured = dict(measured, query_scope=scope)
            self.cache[key] = measured
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
            return measured
        key = (poses.shape, poses.dtype.str, poses.tobytes())
        measured = self.cache.pop(key, None)
        if measured is None:
            if self.geometry.reference_observation_id is None:
                raise RuntimeError('Freeze original reference observations before priority queries')
            measured = self.geometry.measure(poses, jacobians=False)
        if jacobians and not measured.get('priority_linearized', False):
            measured = self.geometry.linearize(poses, measured)
            measured['priority_linearized'] = True
        self.cache[key] = measured
        while len(self.cache) > 3:
            self.cache.popitem(last=False)
        return measured

    def linear_rows(self, measured, *, frames=None):
        if frames is not None:
            return self.geometry.linear_rows(measured, frames=frames)
        return self.geometry.linear_rows(measured)

    def relinearize_witnesses(self, origin, candidate, measured, *, frames, keys=None):
        """Selected original witnesses only; never reuse trial world Jacobians."""
        if os.getpid() != self.owner_pid:
            raise RuntimeError('Selected witness queries require an initialized private client')
        from .original_geometry_relinearize import relinearize_witnesses
        return relinearize_witnesses(self.geometry, origin, candidate, measured, frames=frames, keys=keys)

    def summary(self, measured):
        return self.geometry.summary(measured)


def body_depths(model, frames, measured):
    """Reliable maximum depth of each welded body, including self pairs."""
    if (measured.get('failures') or not measured.get('complete', True)
            or not measured.get('reliable', True)
            or not np.asarray(measured.get('frame_complete', True)).all()):
        raise ValueError('Incomplete or unknown original geometry cannot authorize yield')
    result = np.zeros((frames, model.nbody))
    owners = {}
    for key, value in measured['depths'].items():
        if not np.isfinite(value) or value < 0:
            raise ValueError('Original depths must be finite and nonnegative')
        geoms = key[2:] if key[1] == 'self' else key[-1:]
        bodies = tuple(sorted({int(model.body_weldid[model.geom_bodyid[g]]) for g in geoms}))
        owners[key] = bodies
        for body in bodies:
            result[key[0], body] = max(result[key[0], body], value)
    return result, owners


def window_body_depths(model, count, frames, measured):
    """Use R4's already measured maxima, without a second merge/query.

    R4 exposes weld aliases too. The priority sum counts only canonical weld
    bodies, just as the legacy body_depths reducer does.
    """
    if 'max_depth' not in measured:
        return body_depths(model, count, measured)[0]
    value = np.asarray(measured['max_depth'])
    reliable = np.asarray(measured['body_reliable'])
    if (tuple(measured.get('queried_frames', ())) != tuple(frames)
            or value.shape != (len(frames), model.nbody) or reliable.shape != value.shape
            or not reliable.all() or not np.isfinite(value).all() or np.any(value < 0)
            or measured.get('failures') or not measured.get('complete', True)
            or not measured.get('reliable', True)
            or not np.asarray(measured.get('frame_complete', True)).all()):
        raise ValueError('Incomplete or unknown original window maxima cannot authorize yield')
    result = np.zeros((count, model.nbody))
    canonical = np.flatnonzero(np.asarray(model.body_weldid) == np.arange(model.nbody))
    result[np.ix_(np.array(frames), canonical)] = value[:, canonical]
    return result
