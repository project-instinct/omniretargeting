"""Exact serial self-collision witnesses with global-frame and pair caches.

Prepare in the parent before forking. Children use inherited native libraries,
private MuJoCo buffers, and immutable cached arrays; no threads, pools or CUDA.
Only the original bounding-sphere predicate can certify omitted legacy rows.
Mesh support-plane distances are not Euclidean distances, so tighter physical
OBB/old-witness bounds cannot replace the reference's active-pair decisions.
"""

from collections import OrderedDict
from copy import copy
import ctypes
import os
import time

import mujoco
import numpy as np

from ..geometry_query import GeometryQueryError
from .batched_self_collision import BatchedSelfCollision, NativeUnavailable, _pointer
from .kinematics import TrajectoryKinematics
from .self_collision_window_native import screen_library


def _copy_row(row):
    return {**row, 'pair': list(row['pair']), 'body_ids': list(row['body_ids']),
            'points': row['points'].copy(), 'direction': row['direction'].copy()}


class SelfCollisionWindow:
    """Immutable query semantics; complete pose/pair results cached per frame.

    ``kinematics`` accepts a SimpleNamespace with float64 geom_xpos/xmat,
    aligned to the requested frames (including a sparse missing-frame list).
    ``measure`` takes a range; ``query_frames`` also accepts ordered frame IDs.
    Returned rows/keys/zero depth entries follow OriginalGeometry's self path.
    Positive activation filtering changes rows only, never depth keys.
    """

    def __init__(self, query, *, columns=None, capacity=2048):
        if capacity < 1 or int(capacity) != capacity:
            raise ValueError('capacity must be a positive integer')
        self.query = query.fresh_state()
        self.model = query.model
        self.columns = np.asarray(np.arange(self.model.nv) if columns is None else columns)
        if (self.columns.ndim != 1 or (self.columns.size and self.columns.dtype.kind not in 'iu')
                or np.any((self.columns < 0) | (self.columns >= self.model.nv))
                or len(np.unique(self.columns)) != len(self.columns)):
            raise ValueError('columns must be unique in-range tangent indices')
        self.columns = np.array(self.columns, dtype=np.int64, copy=True)
        self.columns.setflags(write=False)
        self.capacity = int(capacity)
        self.states, self.pose_cache, self.fallback_reuse = OrderedDict(), OrderedDict(), {}
        self.kin = TrajectoryKinematics(self.model, np.empty(0, dtype=int))
        self.prepared = False
        self.owner_pid = os.getpid()
        self.diagnostics = {}

    def prepare(self):
        """Parent-only build/load; single-thread BLAS is a caller prerequisite."""
        if self.prepared:
            return self
        if os.getpid() != self.owner_pid:
            raise RuntimeError('Prepare SelfCollisionWindow in its parent before forking')
        from threadpoolctl import ThreadpoolController
        from .native_mujoco import native_mujoco

        self.blas_controls = tuple(item for item in ThreadpoolController().lib_controllers
                                   if item.user_api == 'blas')
        if any(item.get_num_threads() != 1 for item in self.blas_controls):
            raise RuntimeError('SelfCollisionWindow requires single-thread BLAS before prepare')
        self.native_unavailable = None
        try:
            self.batch = BatchedSelfCollision(self.query, workers=1)
            self.vertex_radius = self.batch.radius.copy()
            # Primitive surfaces use MuJoCo's enclosing radius. Unbounded
            # geoms never receive a sphere certificate and keep exact queries.
            primitive = self.batch.counts == 0
            self.vertex_radius[primitive] = self.model.geom_rbound[primitive]
            self.vertex_radius.setflags(write=False)
            self.screen = screen_library()
            native_mujoco()
        except NativeUnavailable as error:
            self.batch, self.screen = None, None
            self.native_unavailable = str(error)
        self.threshold = float(self.query.cfg['detection_distance'])
        self.blas_checked_pid = os.getpid()
        self.prepared = True
        return self

    def fork_private_client(self):
        """No preparation/compiler/query calls; private mutable query buffers."""
        if not self.prepared:
            raise RuntimeError('Prepare SelfCollisionWindow before making private clients')
        result = copy(self)
        result.query = self.query.fresh_state()
        result.kin = TrajectoryKinematics(self.model, np.empty(0, dtype=int))
        # States are immutable, and replacing an entry never mutates its arrays.
        result.states = OrderedDict(self.states)
        result.pose_cache = OrderedDict(self.pose_cache)
        result.fallback_reuse = {}
        result.diagnostics = {}
        result.owner_pid = os.getpid()
        if self.batch is not None:
            result.batch = copy(self.batch)
            result.batch.query = result.query
            result.batch.previous = None
            result.batch.cache = OrderedDict()
            result.batch.seed_provider = None
            result.batch.diagnostics = {}
        return result

    def _validate(self, poses, frames, activation):
        if not self.prepared:
            raise RuntimeError('Call SelfCollisionWindow.prepare() in the parent first')
        if self.blas_checked_pid != os.getpid():
            # Parent-resolved controllers call inherited C getters directly.
            # Do not rescan/import every loaded library in each window child.
            if any(item.get_num_threads() != 1 for item in self.blas_controls):
                raise RuntimeError('Forked SelfCollisionWindow requires inherited single-thread BLAS')
            self.blas_checked_pid = os.getpid()
        if self.query.cfg['detection_distance'] != self.threshold:
            raise ValueError('SelfCollisionWindow detection band is frozen')
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != self.model.nq:
            raise ValueError('poses must have full (frames, model.nq) shape')
        values = np.asarray(tuple(frames))
        if values.ndim != 1 or (values.size and values.dtype.kind not in 'iu'):
            raise ValueError('frame IDs must be integers')
        ids = np.asarray(values, dtype=np.int64)
        if (np.any((ids < 0) | (ids >= len(poses)))
                or np.any(np.diff(ids) <= 0)):
            raise ValueError('frame IDs must be increasing, unique and in range')
        if not np.isfinite(poses[ids]).all():
            raise ValueError('requested poses must be finite')
        if activation is not None and (not np.isfinite(activation) or not 0 <= activation <= .008):
            raise ValueError('activation_distance must be in [0, 0.008] metres')
        return poses, ids

    def measure(self, poses, *, frames, jacobians=False, activation_distance=None,
                kinematics=None):
        if (not isinstance(frames, range) or frames.step != 1 or frames.start < 0
                or frames.stop < frames.start or frames.stop > len(poses)):
            raise ValueError('frames must be a contiguous range with step=1')
        return self.query_frames(poses, frames, jacobians=jacobians,
                                 activation_distance=activation_distance, kinematics=kinematics)

    def seed_completed(self, poses, measurement, kinematics, *, frames=None):
        """Import only reported, reliable legacy witnesses without any query.

        Geom arrays align to ``frames`` (default measurement.queried_frames,
        otherwise the full trajectory). No pose-level cache is populated:
        unreported nearby pairs remain unmeasured and will be queried. Supplied
        witnesses must belong to these poses; an available pose_id is checked.
        Incomplete/failing measurements grant no cache or separation credit.
        """
        if frames is None:
            frames = measurement.get('queried_frames', range(len(poses)))
        poses, ids = self._validate(poses, frames, None)
        if (not measurement.get('complete', False) or measurement.get('failures')
                or not bool(measurement.get('reliable', True))):
            return dict(seeded_frames=0, seeded_pairs=0, skipped='incomplete_or_unreliable_measurement')
        available = measurement.get('pose_id')
        if available is not None:
            # This helper does not invent or redefine a provider's pose identity.
            raise ValueError('Provider pose_id requires an explicit identity adapter before seeding')
        complete = np.asarray(measurement.get('frame_complete', np.ones(len(ids), bool)))
        if complete.shape != (len(ids),) or not complete.all():
            return dict(seeded_frames=0, seeded_pairs=0, skipped='incomplete_frames')
        position = np.asarray(kinematics.geom_xpos, dtype=np.float64)
        rotation = np.asarray(kinematics.geom_xmat, dtype=np.float64)
        if (position.shape != (len(ids), self.model.ngeom, 3)
                or rotation.shape != (len(ids), self.model.ngeom, 3, 3)
                or not np.isfinite(position).all() or not np.isfinite(rotation).all()):
            raise ValueError('kinematics must be finite geom transforms aligned to seeded frames')
        if self.batch is None:
            return dict(seeded_frames=0, seeded_pairs=0, skipped='native_unavailable')
        by_frame = {int(frame): [] for frame in ids}
        for key, row in measurement['rows']:
            if key[1] != 'self' or key[0] not in by_frame:
                continue
            pair = int(row['pair_index'])
            if (not 0 <= pair < len(self.query.pairs)
                    or tuple(row['pair']) != tuple(self.query.pairs[pair])
                    or key != (key[0], 'self', *row['pair'])
                    or row['pair_id'] != self.query.pair_ids[pair]
                    or row['body_ids'] != list(map(int, self.model.geom_bodyid[self.query.pairs[pair]]))
                    or not np.isfinite(row['clearance']) or row['clearance'] >= self.threshold
                    or np.asarray(row['points']).shape != (2, 3)
                    or np.asarray(row['direction']).shape != (3,)
                    or not np.isfinite(row['points']).all() or not np.isfinite(row['direction']).all()):
                raise ValueError('Seed witness does not match the frozen legacy pair registry')
            by_frame[key[0]].append(row)
        pending, count = {}, 0
        a, b = self.batch.first, self.batch.second
        bound = self.model.geom_rbound
        for local, frame in enumerate(ids):
            witnesses = by_frame[int(frame)]
            if not witnesses:
                continue
            status = np.zeros(len(a), np.uint8)
            phi = np.full(len(a), np.inf)
            points = np.zeros((len(a), 2, 3))
            direction = np.zeros((len(a), 3))
            for row in witnesses:
                pair = row['pair_index']
                if status[pair]:
                    raise ValueError('Seed measurement contains duplicate self-pair witnesses')
                status[pair], phi[pair], points[pair], direction[pair] = (
                    1, row['clearance'], row['points'], row['direction'])
            gap = np.linalg.norm(position[local, a] - position[local, b], axis=1) - bound[a] - bound[b]
            state = (position[local].copy(), rotation[local].copy(), gap, status, phi, points, direction)
            for value in state:
                value.setflags(write=False)
            pending[int(frame)] = state
            count += len(witnesses)
        # Validation above is atomic: an invalid later row cannot seed a prefix.
        for frame, state in pending.items():
            self.states[frame] = state
            self.states.move_to_end(frame)
        while len(self.states) > self.capacity:
            self.states.popitem(last=False)
        return dict(seeded_frames=len(pending), seeded_pairs=count, skipped=None)

    def _screen(self, position, rotation, ids):
        batch = self.batch
        shape = len(ids), len(batch.first)
        old_position, old_rotation = np.zeros_like(position), np.zeros_like(rotation)
        old_gap, old_status = np.zeros(shape), np.zeros(shape, np.uint8)
        valid = np.zeros(len(ids), np.uint8)
        for local, frame in enumerate(ids):
            state = self.states.get(int(frame))
            if state is not None:
                valid[local] = 1
                old_position[local], old_rotation[local], old_gap[local], old_status[local] = state[:4]
        near, reused, certified = (np.zeros(shape, np.uint8) for _ in range(3))
        gap = np.empty(shape)
        arrays = (batch.first, batch.second, np.ascontiguousarray(self.model.geom_rbound), self.vertex_radius,
                  position, rotation, old_position, old_rotation, old_gap, old_status, valid,
                  near, reused, certified, gap)
        self.screen.ws3_window_screen(len(ids), self.model.ngeom, len(batch.first),
            *(_pointer(value) for value in arrays[:4]), self.threshold,
            *(_pointer(value) for value in arrays[4:]))
        ties = np.nonzero(near == 2)
        if len(ties[0]):
            distance = np.linalg.norm(position[ties[0], batch.first[ties[1]]]
                                      - position[ties[0], batch.second[ties[1]]], axis=1)
            bound = self.model.geom_rbound
            near[ties] = distance <= (bound[batch.first[ties[1]]] + bound[batch.second[ties[1]]]
                                     + self.threshold)
            gap[ties] = distance - bound[batch.first[ties[1]]] - bound[batch.second[ties[1]]]
        reused &= near
        self.diagnostics.update(sphere_reused_pairs=int(certified.sum()),
                                sphere_skipped_pairs=int((near == 0).sum()),
                                transform_reused_pairs=int(reused.sum()),
                                sphere_tie_pairs=len(ties[0]))
        return near, reused, gap

    def seed_query_reuse(self, poses, measurement, kinematics, *, legacy_reuse, frames=None):
        """Optionally import EXPLICIT legacy cached no-row query outcomes.

        Separate from seed_completed: row absence supplies no certificate.
        Only SelfCollisionQuery.query(reuse=...) entries whose value is None,
        threshold and BOTH exact transform keys match, certify no legacy row
        at those unchanged placements. Status 3 means known inactive, not a
        measured infinite clearance. No distance/witness is fabricated, and
        moving either geom forces the original narrow phase again.
        """
        if frames is None:
            frames = measurement.get('queried_frames', range(len(poses)))
        poses, ids = self._validate(poses, frames, None)
        staging = copy(self)
        staging.states = OrderedDict(self.states)
        seeded = staging.seed_completed(poses, measurement, kinematics, frames=ids)
        if seeded['skipped'] is not None:
            return {**seeded, 'seeded_inactive_pairs': 0}
        position = np.asarray(kinematics.geom_xpos, dtype=np.float64)
        rotation = np.asarray(kinematics.geom_xmat, dtype=np.float64)
        registry = {tuple(map(int, pair)): index for index, pair in enumerate(self.query.pairs)}
        imported, skipped, inactive_frames = 0, 0, 0
        a, b = self.batch.first, self.batch.second
        bound = self.model.geom_rbound
        for local, frame in enumerate(ids):
            entries = legacy_reuse.get(int(frame), {})
            selected = []
            for key, value in entries.items():
                if value is not None:
                    continue
                if not isinstance(key, tuple) or len(key) != 7:
                    raise ValueError('Expected original SelfCollisionQuery reuse transform keys')
                g1, g2, threshold, p1, r1, p2, r2 = key
                pair = registry.get((g1, g2))
                if (pair is None or threshold != self.threshold
                        or p1 != position[local, g1].tobytes() or r1 != rotation[local, g1].tobytes()
                        or p2 != position[local, g2].tobytes() or r2 != rotation[local, g2].tobytes()):
                    skipped += 1
                    continue
                selected.append(pair)
            if not selected:
                continue
            previous = staging.states.get(int(frame))
            if (previous is not None and previous[0].tobytes() == position[local].tobytes()
                    and previous[1].tobytes() == rotation[local].tobytes()):
                state = tuple(value.copy() for value in previous)
            else:
                gap = np.linalg.norm(position[local, a] - position[local, b], axis=1) - bound[a] - bound[b]
                state = (position[local].copy(), rotation[local].copy(), gap,
                         np.zeros(len(a), np.uint8), np.full(len(a), np.inf),
                         np.zeros((len(a), 2, 3)), np.zeros((len(a), 3)))
            for pair in selected:
                if state[3][pair] == 1 and state[4][pair] < self.threshold:
                    raise ValueError('Explicit inactive reuse conflicts with reported self witness')
                state[3][pair] = 3
                imported += 1
            for value in state:
                value.setflags(write=False)
            staging.states[int(frame)] = state
            staging.states.move_to_end(int(frame))
            inactive_frames += 1
        while len(staging.states) > self.capacity:
            staging.states.popitem(last=False)
        self.states = staging.states
        return {**seeded, 'seeded_inactive_pairs': imported, 'seeded_inactive_frames': inactive_frames,
                'skipped_inactive_keys': skipped}

    def _run_native(self, position, rotation, near, clearance, points, directions, status):
        batch = self.batch
        arrays = (batch.first, batch.second, batch.starts, batch.counts, batch.vertices,
                  rotation, position, near, batch.seed, clearance, points, directions, status)
        batch.library.ws3_pairs(len(position), self.model.ngeom, len(batch.first),
            *(_pointer(value) for value in arrays), ctypes.c_void_p(), ctypes.c_void_p())

    def _native_rows(self, poses, ids, position, rotation):
        near, reused, gap = self._screen(position, rotation, ids)
        shape = near.shape
        clearance = np.full(shape, np.inf)
        points, directions = np.zeros((*shape, 2, 3)), np.zeros((*shape, 3))
        status = np.zeros(shape, np.uint8)
        for local, frame in enumerate(ids):
            if reused[local].any():
                old = self.states[int(frame)]
                mask = reused[local].astype(bool)
                clearance[local, mask], points[local, mask], directions[local, mask] = (
                    value[mask] for value in old[4:7])
                status[local, mask] = old[3][mask]
        active = np.ascontiguousarray(near & ~reused, dtype=np.uint8)
        start = time.perf_counter()
        self._run_native(position, rotation, active, clearance, points, directions, status)
        self.diagnostics.update(native_seconds=time.perf_counter() - start,
                                native_pairs=int((status[active.astype(bool)] == 1).sum()),
                                fallback_pairs=int((status == 2).sum()))
        results = []
        for local, frame in enumerate(ids):
            try:
                if np.any((status[local] == 2) & ((self.batch.counts[self.batch.first] == 0)
                                                | (self.batch.counts[self.batch.second] == 0))):
                    # Primitive/mixed normals require the original full forward.
                    rows = self.query.query(poses[frame], reuse=self.fallback_reuse.setdefault(int(frame), {}))
                    results.append((rows, None))
                    continue
                rows = []
                fallback = np.flatnonzero(status[local] == 2)
                if len(fallback):
                    self.query.data.qpos[:] = poses[frame]
                    mujoco.mj_kinematics(self.model, self.query.data)
                    for pair in fallback:
                        a, b = self.query.pairs[pair]
                        phi, witness, direction = self.query.mesh_distance.query(int(a), int(b), self.query.data)
                        clearance[local, pair], points[local, pair], directions[local, pair] = phi, witness, direction
                        status[local, pair] = 1
                for pair in np.flatnonzero((status[local] == 1) & (clearance[local] < self.threshold)):
                    a, b = map(int, self.query.pairs[pair])
                    rows.append(dict(pair=[a, b], pair_id=int(self.query.pair_ids[pair]), pair_index=int(pair),
                                     body_ids=[int(self.model.geom_bodyid[a]), int(self.model.geom_bodyid[b])],
                                     clearance=float(clearance[local, pair]), points=points[local, pair].copy(),
                                     direction=directions[local, pair].copy()))
                # Never retain a failed frame as a known result or motion bound.
                state = tuple(value[local].copy() for value in
                              (position, rotation, gap, status, clearance, points, directions))
                for value in state:
                    value.setflags(write=False)
                self.states[int(frame)] = state
                self.states.move_to_end(int(frame))
                results.append((rows, None))
            except GeometryQueryError as error:
                results.append(([], f'unknown self frame {frame}: {error}'))
        while len(self.states) > self.capacity:
            self.states.popitem(last=False)
        return results

    def query_frames(self, poses, frames, *, jacobians=False, activation_distance=None,
                     kinematics=None):
        """Self-only OriginalGeometry fields, ordered by global frame/pair ID."""
        started = time.perf_counter()
        poses, ids = self._validate(poses, frames, activation_distance)
        self.diagnostics = dict(native_available=self.batch is not None, native_seconds=0.,
                                native_pairs=0, fallback_pairs=0, transform_reused_pairs=0,
                                sphere_reused_pairs=0, sphere_skipped_pairs=0, sphere_tie_pairs=0,
                                computed_fk_frames=0)
        keys = [(int(frame), poses[frame].tobytes()) for frame in ids]
        missing = [i for i, key in enumerate(keys) if key not in self.pose_cache]
        if kinematics is not None:
            position = np.asarray(kinematics.geom_xpos, dtype=np.float64)
            rotation = np.asarray(kinematics.geom_xmat, dtype=np.float64)
            if (position.shape != (len(ids), self.model.ngeom, 3)
                    or rotation.shape != (len(ids), self.model.ngeom, 3, 3)
                    or not np.isfinite(position).all() or not np.isfinite(rotation).all()):
                raise ValueError('kinematics must be finite geom transforms aligned to requested frames')
        if missing:
            selected = ids[missing]
            if self.batch is None:
                results = []
                for frame in selected:
                    try:
                        rows = self.query.query(poses[frame], reuse=self.fallback_reuse.setdefault(int(frame), {}))
                        results.append((rows, None))
                    except GeometryQueryError as error:
                        results.append(([], f'unknown self frame {frame}: {error}'))
            else:
                if kinematics is None:
                    kin = self.kin.evaluate(poses[selected], with_jacobian=False, native=True)
                    position, rotation = kin.geom_xpos, kin.geom_xmat
                    self.diagnostics['computed_fk_frames'] = len(selected)
                else:
                    position, rotation = position[missing], rotation[missing]
                results = self._native_rows(poses, selected, np.ascontiguousarray(position),
                                            np.ascontiguousarray(rotation))
            # Unknown frames are returned but never stored as complete cache hits.
            failures_by_index = {}
            for index, (rows, failure) in zip(missing, results):
                if failure is None:
                    self.pose_cache[keys[index]] = tuple(_copy_row(row) for row in rows)
                else:
                    failures_by_index[index] = failure
        else:
            failures_by_index = {}
        rows, depths, failures = [], {}, []
        omitted = np.zeros(len(ids), np.int64)
        body_depth = np.zeros((len(ids), self.model.nbody))
        body_reliable = np.ones(body_depth.shape, bool)
        for index, (frame, key) in enumerate(zip(ids, keys)):
            if index in failures_by_index:
                failures.append(failures_by_index[index])
                body_reliable[index] = False
                continue
            witnesses = [_copy_row(row) for row in self.pose_cache[key]]
            self.pose_cache.move_to_end(key)
            for row in witnesses:
                collision = (int(frame), 'self', *row['pair'])
                depth = max(0., -row['clearance'])
                depths[collision] = depth
                for body in row['body_ids']:
                    owner = self.model.body_weldid[body]
                    body_depth[index, owner] = max(body_depth[index, owner], depth)
            if activation_distance is not None:
                original_count = len(witnesses)
                witnesses = [row for row in witnesses if row['clearance'] < activation_distance]
                omitted[index] = original_count - len(witnesses)
            if jacobians and witnesses:
                self.query.data.qpos[:] = poses[frame]
                mujoco.mj_kinematics(self.model, self.query.data)
                mujoco.mj_comPos(self.model, self.query.data)
                if self.batch is None:
                    jac = [self.query.jacobian(row, self.query.data) for row in witnesses]
                else:
                    from .native_collision_rows import collision_jacobians
                    jac = collision_jacobians(self.query, witnesses)
                for row, value in zip(witnesses, jac):
                    row['jacobian'] = value[self.columns].copy()
            rows.extend(((int(frame), 'self', *row['pair']), row) for row in witnesses)
        while len(self.pose_cache) > 3 * self.capacity:
            self.pose_cache.popitem(last=False)
        body_depth, body_reliable = body_depth[:, self.model.body_weldid], body_reliable[:, self.model.body_weldid]
        body_depth[~body_reliable] = np.nan
        self.diagnostics.update(requested_frames=len(ids), computed_frames=len(missing),
                                pose_reused_frames=len(ids) - len(missing),
                                seconds=time.perf_counter() - started)
        return dict(rows=rows, depths=depths, failures=failures, complete=True,
                    queried_frames=tuple(map(int, ids)), frame_complete=np.ones(len(ids), bool),
                    reliable=not failures, max_depth=body_depth, body_reliable=body_reliable,
                    activation_distance=activation_distance, omitted_near_misses=omitted,
                    diagnostics=dict(self.diagnostics))
