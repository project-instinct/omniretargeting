"""Unchanged full-rate certificates composed with the coarse interpolant."""

from collections import OrderedDict
from copy import copy

import cvxpy as cp
import mujoco
import numpy as np
from scipy import sparse

from .coarse_embedding import LinearCorrectionEmbedding
from .frame_kinematics import FrameKinematicsCache
from .guard_signals import prepare
from .kinematics import TrajectoryKinematics


def _constraints(linear, cones, flat):
    result = []
    for matrix, lower, upper in linear:
        lo, hi = np.isfinite(lower), np.isfinite(upper)
        if lo.any():
            result.append(matrix[lo] @ flat >= lower[lo])
        if hi.any():
            result.append(matrix[hi] @ flat <= upper[hi])
    for matrix, residual, allowed in cones:
        if len(allowed):
            value = cp.reshape(matrix @ flat, residual.shape, order='C') + residual
            scale = 1 / np.maximum(allowed, 1e-3)
            result.append(cp.norm(cp.multiply(scale[:, None], value), axis=1) <= allowed * scale)
    return result


class FullRateGuard:
    def __init__(self, embedding, full_guard, delegate=None):
        self.embedding, self.full_guard, self.delegate = embedding, full_guard, delegate
        self.rejections = 0

    def __getattr__(self, name):
        return getattr(self.full_guard if self.delegate is None else self.delegate, name)

    def accepts(self, poses, frames=None):
        if not np.isfinite(poses).all():
            self.rejections += 1
            return False
        full = self.embedding.embed(poses)
        good = (self.full_guard.accepts(full) if frames is None else
                self.full_guard.accepts(full, frames=self.embedding.affected_frames(frames)))
        self.rejections += not good
        return bool(good)

    def violation_frames(self, poses, *, frames=None):
        if not np.isfinite(poses).all():
            return ~np.isfinite(poses).all(axis=1)
        full = self.embedding.embed(poses)
        bad = (self.full_guard.violation_frames(full) if frames is None else
               self.full_guard.violation_frames(full, frames=self.embedding.affected_frames(frames)))
        return self.embedding.map_violations(bad)

    def window_dependencies(self, frames):
        affected = self.embedding.affected_frames(frames)
        needed = self.full_guard.window_dependencies(affected)
        return self.embedding.contributors(needed)

    def qp_bounds(self, poses, *, frames=None):
        frames = range(self.embedding.knots) if frames is None else frames
        full_frames = self.embedding.affected_frames(frames)
        linear, cones = self.full_guard.qp_bounds(self.embedding.embed(poses), frames=full_frames)
        jacobian = self.embedding.jacobian(poses, frames, full_frames)
        return ([(matrix @ jacobian, lo, hi) for matrix, lo, hi in linear],
                [(matrix @ jacobian, residual, allowed) for matrix, residual, allowed in cones])

    def constraints(self, poses, flat, box=None):
        return _constraints(*self.qp_bounds(poses), flat)

    def diagnostics(self, poses):
        return self.full_guard.diagnostics(self.embedding.embed(poses))

    def project(self, poses):
        # Only the sampled objective's normal knot projection is permitted.
        # Odd frames are constrained by rows and certified, never clipped.
        return self.delegate.project(poses)


class FullRateBoxes:
    def __init__(self, embedding, objective):
        self.embedding, self.objective = embedding, objective
        self.rejections = 0

    def _violations(self, full, frames):
        objective, coords = self.objective, self.objective.coords
        span = slice(frames.start, frames.stop)
        value = full[span, coords.qi]
        lower = np.broadcast_to(objective.lower, (len(full), len(coords.qi)))[span]
        upper = np.broadcast_to(objective.upper, (len(full), len(coords.qi)))[span]
        bad = ~np.isfinite(full[span]).all(axis=1) | ((value < lower) | (value > upper)).any(axis=1)
        q = coords.root_q
        if q is not None:
            domains = []
            if objective.root_budget is not None:
                domains.append((objective.reference, objective.root_budget))
            shape = objective.source_shape
            if shape is not None:
                domains.append((shape.seed, shape.xy))
                if shape.acquired_poses is not None:
                    domains.append((shape.acquired_poses, shape.xy))
            for reference, bound in domains:
                bad |= np.linalg.norm(full[span, q:q + 3] - reference[span, q:q + 3], axis=1) > bound + 1e-7
            if objective.fixed_root is not None:
                bad |= (full[span, q:q + 7] != objective.fixed_root[span]).any(axis=1)
        result = np.zeros(len(full), bool)
        result[span] = bad
        return result

    def violation_frames(self, poses, *, frames=None):
        if not np.isfinite(poses).all():
            return ~np.isfinite(poses).all(axis=1)
        affected = range(self.embedding.count) if frames is None else self.embedding.affected_frames(frames)
        return self.embedding.map_violations(self._violations(self.embedding.embed(poses), affected))

    def accepts(self, poses, frames=None):
        good = not self.violation_frames(poses, frames=frames).any()
        self.rejections += not good
        return good

    def window_dependencies(self, frames):
        return self.embedding.contributors(self.embedding.affected_frames(frames))

    def qp_bounds(self, poses, *, frames=None):
        frames = range(self.embedding.knots) if frames is None else frames
        affected = self.embedding.affected_frames(frames)
        full, objective = self.embedding.embed(poses), self.objective
        coords, span = objective.coords, slice(affected.start, affected.stop)
        jacobian = self.embedding.jacobian(poses, frames, affected)
        matrix = sparse.kron(sparse.eye(len(affected)), coords.selector, format='csr') @ jacobian
        lower = np.broadcast_to(objective.lower, (len(full), len(coords.qi)))[span] - full[span, coords.qi]
        upper = np.broadcast_to(objective.upper, (len(full), len(coords.qi)))[span] - full[span, coords.qi]
        linear, cones = [(matrix, lower.ravel(), upper.ravel())], []
        if coords.root_q is not None:
            q, d = coords.root_q, coords.root_d
            if objective.root_budget is not None:
                select = np.zeros((3, coords.width))
                for axis in range(3):
                    column = coords.inverse.get(d + axis)
                    if column is not None:
                        select[axis, column] = 1.
                matrix = sparse.kron(sparse.eye(len(affected)), select, format='csr') @ jacobian
                cones.append((matrix, full[span, q:q + 3] - objective.reference[span, q:q + 3],
                              np.full(len(affected), objective.root_budget)))
            if objective.fixed_root is not None:
                columns = [coords.inverse[axis] for axis in range(d, d + 6) if axis in coords.inverse]
                target = full[span].copy()
                target[:, q:q + 7] = objective.fixed_root[span]
                delta = coords.difference(full[span], target)[:, columns].ravel()
                select = sparse.eye(coords.width, format='csr')[columns]
                matrix = sparse.kron(sparse.eye(len(affected)), select, format='csr') @ jacobian
                linear.append((matrix, delta, delta))
        return linear, cones

    def constraints(self, poses, flat):
        return _constraints(*self.qp_bounds(poses), flat)


def install_full_rate_guards(coarse, full, sampled_shape, sampled_hinges):
    """Own mutable FK/caches; share unchanged frozen full-rate quantities only."""
    embedding = LinearCorrectionEmbedding(full.coords, full.reference, np.arange(0, full.frames, 2))
    owner = copy(full)
    for name in list(owner.__dict__):
        if name.startswith(('_window_', '_gpu_')):
            owner.__dict__.pop(name)
    owner.pool, owner.worker_output = None, None
    owner.kin = TrajectoryKinematics(full.model, full.coords.cols)
    cache = FrameKinematicsCache(full.model, full.coords.cols)
    owner._window_frame_cache = cache
    # Identical knot poses share exact FK with the even-frame objective;
    # full-rate guard rows need to compute only new odd-frame poses.
    coarse._window_frame_cache = cache

    def private(guard):
        result = copy(guard)
        for name in list(result.__dict__):
            if name.startswith('_window_'):
                result.__dict__.pop(name)
        result._window_frame_cache, result.rejections = cache, 0
        if hasattr(result, 'kin'):
            result.kin = TrajectoryKinematics(full.model, full.coords.cols, cache=owner.kin.cache)
        if hasattr(result, 'objective'):
            result.objective = owner
        if hasattr(result, '_signals_cache'):
            result.tasks = copy(guard.tasks)
            result.tasks.data = mujoco.MjData(full.model)
            result._signals_cache = OrderedDict()
            result._joint_matrices = {}
        if hasattr(result, '_signal_kin'):
            result._signal_kin = TrajectoryKinematics(full.model, full.coords.cols, cache=owner.kin.cache)
        if hasattr(result, '_linear_cache'):
            result._linear_cache, result._operators = None, {}
        return result

    owner.source_shape = None if full.source_shape is None else private(full.source_shape)
    owner.candidate_guards = [private(guard) for guard in full.candidate_guards]
    prepare(owner)
    coarse.coarse_embedding = embedding
    coarse.source_shape = (None if owner.source_shape is None else
                           FullRateGuard(embedding, owner.source_shape, sampled_shape))
    coarse.candidate_guards = [FullRateBoxes(embedding, owner),
                              *(FullRateGuard(embedding, guard) for guard in owner.candidate_guards)]
    coarse.hinge_guards = sampled_hinges
