"""Optional physical convex distances for acquisition objective rows.

Repair and audits keep SelfCollisionQuery's original mesh routine. For
separated meshes that routine returns a supporting-plane lower bound;
native GJK returns the Euclidean gap, so their active hinge sets can differ.
The activation band screens physical gaps, then restores the original hinge
and material Jacobian inside the band. Pair eligibility and structural
exclusions come unchanged from the query.
"""

from collections import OrderedDict
from copy import copy
import time

import mujoco
import numpy as np

from ..geometry_query import GeometryQueryError, distance_direction

# Covers numerical contact and roundoff in broad-phase certificates. This
# widens candidate selection, never shrinks the requested activation cutoff.
DISTANCE_MARGIN = 1e-6
CCD_TOLERANCE = 1e-9
# The 13mm p95 band failed replay; an extra acquired pose also ruled out
# the 18mm p99 band. The sampled maximum understatement is 27.79mm; add
# the unchanged 2mm target and round up to 30mm (the query's detection
# distance). Both offline windows retain the control result. No body tuning.
ACTIVATION_DISTANCE = .030


def acquisition_self_collision(query, workers=1, *, window=False):
    """Use the physical activation band only for local window evaluations.

    Whole-trajectory chunks retain the original rows; their native batch
    routine is faster than screening the band one frame at a time.
    """
    from .runtime_options import OPTIONS
    if window and OPTIONS.fast_self_collision:
        return FastSelfCollision(query, workers)
    from .batched_self_collision import BatchedSelfCollision
    return BatchedSelfCollision(query, workers)


class FastSelfCollision:
    """One consumer's native distance buffers and certified LM trial reuse."""

    def __init__(self, query, workers=1, *, activation_distance=ACTIVATION_DISTANCE):
        self.query, self.model = query, query.model
        if not np.isfinite(activation_distance) or activation_distance < 0:
            raise ValueError('activation_distance must be finite and nonnegative')
        self.activation_distance = float(activation_distance)
        self.workers = max(1, int(workers))
        self.seed_provider = None
        self.first, self.second = query.pairs.T.copy()
        # Never alter the robot model's CCD flags, tolerances or geometry.
        self.distance_model = copy(self.model)
        self.distance_model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        self.distance_model.opt.ccd_tolerance = CCD_TOLERANCE
        self.data = mujoco.MjData(self.distance_model)
        self.center = np.zeros((self.model.ngeom, 3))
        self.half = np.full_like(self.center, np.inf)
        self.radius = np.zeros(self.model.ngeom)
        self.mesh = self.model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
        for geom in query.geom_ids:
            if self.mesh[geom]:
                query.mesh_distance.world_vertices(int(geom), query.data)
                vertices = query.mesh_distance.vertices[int(self.model.geom_dataid[geom])]
                lo, hi = vertices.min(0), vertices.max(0)
                self.center[geom], self.half[geom] = (lo + hi) / 2, (hi - lo) / 2
                self.radius[geom] = np.linalg.norm(vertices, axis=1).max(initial=0.)
            elif self.model.geom_rbound[geom] > 0:
                self.radius[geom] = self.model.geom_rbound[geom]
                self.half[geom] = self.radius[geom]
        self.bounded = (np.isfinite(self.half[self.first]).all(axis=1)
                        & np.isfinite(self.half[self.second]).all(axis=1))
        self.previous = None
        self.cache = OrderedDict()
        self.diagnostics = {}
        self.unknown_frames = []
        self.positive_hinges = None

    def __copy__(self):
        result = type(self).__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        result.data = mujoco.MjData(self.distance_model)
        result.previous = None
        result.cache = OrderedDict()
        result.diagnostics = {}
        result.unknown_frames = []
        result.positive_hinges = None
        return result

    def _nearby(self, kin, cutoff):
        position, rotation = kin.geom_xpos, kin.geom_xmat
        shape = len(position), len(self.first)
        certified = np.zeros(shape, bool)
        gap = np.zeros(shape)
        if self.previous is not None and self.previous[0].shape == position.shape:
            old_position, old_rotation, old_gap = self.previous
            travel = np.linalg.norm(position - old_position, axis=-1)
            travel += np.linalg.norm(rotation - old_rotation, axis=(-2, -1)) * self.radius
            # Distance between convex sets is Lipschitz in their maximum
            # material-point displacement. Old AABB gaps are lower bounds on
            # that distance even when the next pose's AABB changes rotation.
            safe = old_gap - travel[:, self.first] - travel[:, self.second]
            slack = 1e-12 + 128 * np.finfo(float).eps * (
                np.abs(old_gap) + travel[:, self.first] + travel[:, self.second] + 1.)
            certified = self.bounded & (safe > cutoff + DISTANCE_MARGIN + slack)
            gap[certified] = safe[certified]
        center = position + np.einsum('fgij,gj->fgi', rotation, self.center)
        # Unbounded primitives always reach the narrow phase. Use zeros here
        # to avoid 0*inf creating NaNs in rotations with exact zero entries.
        half = np.einsum('fgij,gj->fgi', np.abs(rotation),
                         np.where(np.isfinite(self.half), self.half, 0.))
        frames, pairs = np.nonzero(~certified & self.bounded)
        a, b = self.first[pairs], self.second[pairs]
        separation = np.maximum(0., np.abs(center[frames, a] - center[frames, b])
                                - half[frames, a] - half[frames, b])
        gap[frames, pairs] = np.linalg.norm(separation, axis=-1)
        self.previous = position, rotation, gap
        self.diagnostics['bound_reused_pairs'] = int(certified.sum())
        return (~self.bounded | (gap <= cutoff + DISTANCE_MARGIN)) & ~certified

    def _distance(self, a, b, cutoff, batch_fallback=False):
        points = np.zeros(6)
        phi = float(mujoco.mj_geomDistance(
            self.distance_model, self.data, a, b, cutoff + DISTANCE_MARGIN, points))
        self.diagnostics['native_pairs'] += 1
        if np.isfinite(phi) and phi >= cutoff + DISTANCE_MARGIN:
            return None
        length = np.linalg.norm(points[:3] - points[3:])
        reliable = (np.isfinite(phi) and np.isfinite(points).all()
                    and length > DISTANCE_MARGIN
                    and np.isclose(length, abs(phi), rtol=1e-5, atol=1e-7))
        # Native EPA agrees with the original penetration bound on the real
        # and randomized mesh fixtures. Zero or inconsistent witnesses still
        # use the authority, rather than inventing a normal or dropping a row.
        if not reliable:
            self.diagnostics['fallback_pairs'] += 1
            if self.mesh[a] and self.mesh[b]:
                if batch_fallback:
                    return 'mesh_original'
                phi, witness, direction = self.query.mesh_distance.query(a, b, self.data)
            else:
                return 'original'
        else:
            witness, direction = points.reshape(2, 3).copy(), distance_direction(phi, points)
        if phi >= cutoff:
            return None
        return float(phi), witness, direction

    def _witnesses(self, poses, kin, cutoff, batch_fallback=False):
        # Strong immutable FK references prevent recycled IDs matching a
        # different pose. Cutoff changes invalidate exact-pose row reuse.
        key = id(kin.geom_xpos), id(kin.geom_xmat), cutoff, batch_fallback
        previous = self.cache.get(key)
        if previous is not None:
            self.cache.move_to_end(key)
            self.diagnostics['exact_pose_reused'] = True
            return previous[2], previous[3]
        self.diagnostics['exact_pose_reused'] = False
        nearby = self._nearby(kin, cutoff)
        self.diagnostics['candidate_pairs'] = int(nearby.sum())
        per_frame, failures = [], []
        for frame, pose in enumerate(poses):
            rows = []
            selected = np.flatnonzero(nearby[frame])
            if len(selected):
                self.data.qpos[:] = pose
                mujoco.mj_kinematics(self.distance_model, self.data)
            try:
                for pair in selected:
                    a, b = int(self.first[pair]), int(self.second[pair])
                    measured = self._distance(a, b, cutoff, batch_fallback)
                    if measured is None:
                        continue
                    if isinstance(measured, str):
                        if measured == 'original':
                            # Retain the exact primitive contact-normal/error path.
                            rows = [r for r in self.query.query(pose) if r['clearance'] < cutoff]
                            break
                        phi, points, direction = 0., np.zeros((2, 3)), np.zeros(3)
                    else:
                        phi, points, direction = measured
                    rows.append(dict(pair=[a, b], pair_id=int(self.query.pair_ids[pair]),
                                     pair_index=int(pair),
                                     body_ids=[int(self.model.geom_bodyid[a]), int(self.model.geom_bodyid[b])],
                                     clearance=phi, points=points, direction=direction))
                    if isinstance(measured, str):
                        rows[-1]['_original_required'] = True
            except GeometryQueryError:
                rows = []
                failures.append(frame)
            per_frame.append(rows)
        self.cache[key] = kin.geom_xpos, kin.geom_xmat, per_frame, failures
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return per_frame, failures

    def rows(self, poses, kin, with_jacobian, target=None):
        """Same row contract and ordering as BatchedSelfCollision.rows."""
        started = time.perf_counter()
        cutoff = self.query.cfg['detection_distance']
        if target is not None:
            cutoff = min(cutoff, max(target, self.activation_distance))
        self.diagnostics = dict(native_pairs=0, fallback_pairs=0, bound_reused_pairs=0)
        banded = target is not None and cutoff > target
        witnesses, self.unknown_frames = self._witnesses(poses, kin, cutoff, banded)
        per_frame = []
        for pose, rows in zip(poses, witnesses):
            # Return independent rows: callers may annotate or mutate them;
            # cached witness arrays never include a stale Jacobian.
            current = [{**row, 'points': row['points'].copy(),
                        'direction': row['direction'].copy()} for row in rows]
            active = []
            if with_jacobian:
                for row in current:
                    a, b = row['pair']
                    if not (banded and self.mesh[a] and self.mesh[b]) and (
                            target is None or row['clearance'] < target):
                        active.append(row)
            if active:
                self.data.qpos[:] = pose
                mujoco.mj_kinematics(self.distance_model, self.data)
                mujoco.mj_comPos(self.distance_model, self.data)
                for row in active:
                    row['_jacobian'] = self.query.jacobian(row, self.data)
            per_frame.append(current)
        if banded:
            # Native distance screens the band. Inside it, restore the old
            # hinge and material Jacobian. Overlapping mesh faces can have
            # nonunique EPA witnesses: keep the original gradient there too.
            # Ambiguous native distances join this same batch, avoiding a
            # second scalar mesh query and duplicate positive-gap work.
            from .fast_self_planes import BandMeshHinges
            if self.positive_hinges is None:
                self.positive_hinges = BandMeshHinges(self.query)
            if self.positive_hinges.batch is not None:
                self.positive_hinges.batch.workers = self.workers
                self.positive_hinges.batch.cpu_budget = getattr(self, 'cpu_budget', None)
            tick = time.perf_counter()
            hinges, failures, count = self.positive_hinges.rows(
                poses, kin, per_frame, with_jacobian, target, cutoff)
            self.diagnostics['band_plane_seconds'] = time.perf_counter() - tick
            self.diagnostics['band_plane_pairs'] = count
            if self.positive_hinges.batch is not None:
                self.diagnostics['band_diagnostics'] = dict(self.positive_hinges.batch.diagnostics)
            self.unknown_frames = sorted(set(self.unknown_frames) | set(failures))
            for frame, (rows, original) in enumerate(zip(per_frame, hinges)):
                if frame in self.unknown_frames:
                    rows.clear()
                    continue
                by_pair = {r['pair_index']: r for r in original}
                retained = []
                for row in rows:
                    hinge = by_pair.get(row['pair_index'])
                    if hinge is not None:
                        physical = row['clearance']
                        required = row.pop('_original_required', False)
                        if required and hinge['clearance'] >= cutoff:
                            continue
                        row.update({**hinge, 'points': hinge['points'].copy(),
                                    'direction': hinge['direction'].copy()})
                        if not required and physical > 0:
                            row['clearance'] = physical
                        row['hinge_clearance'] = hinge['clearance']
                    elif row.get('_original_required', False):
                        continue
                    retained.append(row)
                rows[:] = retained
        self.diagnostics['seconds'] = time.perf_counter() - started
        self.diagnostics['activation_distance_m'] = cutoff
        self.diagnostics['activation_rows'] = sum(
            target is not None and row['clearance'] >= target
            and row.get('hinge_clearance', row['clearance']) < target
            for rows in per_frame for row in rows)
        return per_frame, len(self.unknown_frames)
