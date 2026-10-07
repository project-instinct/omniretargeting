"""Frame-parallel original-geometry measurement with the serial decision.

``OriginalGeometry.measure_frames`` treats frames independently. Each forked
worker owns a fixed interleaved set of frames for the whole repair, so its
exact pose-signature caches stay coherent between candidates. The parent
merges per-frame rows in frame order and cuts at the first rejected frame.
A complete measurement is exactly the serial one.

A candidate is rejected if any frame rejects it, whichever is found first.
Candidates visit recently rejecting frames first, then their most-changed
frames, and the first rejection found stops every worker. The accept/reject decision is the serial one; a
rejected candidate's rows, reported frame and query count are partial
diagnostics.
"""

import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..cpu_budget import available_cpus

WORKERS = max(1, min(16, 2 * available_cpus() // 3))
# Frames differ greatly in cost. Unpinned workers let idle cores serve the
# libigl threads of busy ones; pinning each worker to its own share measured
# slower on a hybrid 8P+8E CPU (67 s vs 57-60 s of omomo_78 G1 measurement).
# Fewer frames per worker cost more in round trips than they save.
MIN_FRAMES_PER_WORKER = 8
# Recently rejecting frames visited first (a visiting order only).
RECENT_REJECTIONS = 32


def merge_frames(parts):
    """Serial ``measure`` output from per-frame parts of any frame partition."""
    rejected = [part["rejection"] for part in parts if part["rejection"] is not None]
    rejection = min(rejected, default=None)
    frames = sorted((item for part in parts for item in part["frames"]), key=lambda item: item[0])
    rows, depths, failures = [], {}, []
    for frame, frame_rows, frame_depths, frame_failures in frames:
        if rejection is not None and frame > rejection:
            break
        rows += frame_rows
        depths.update(frame_depths)
        failures += frame_failures
    result = {"rows": rows, "depths": depths, "failures": failures, "complete": rejection is None}
    if rejection is not None:
        result["rejection_frame"] = rejection
    return result


def _serve(geometry, frames, cores, connection, stop):
    # libigl sizes its thread pools once per process, before this fork. With
    if cores and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cores)
    own = set(frames)
    while True:
        message = connection.recv()
        if message is None:
            break
        if isinstance(message[0], str):
            command, poses = message
            if command == 'prefetch_self' and poses is None:
                poses = geometry._repair_pose_view
            from ..geometry_query import GeometryQueryError
            try:
                if command == "reference_observations":
                    geometry._install_reference_observations(*poses)
                    connection.send(("reference_frozen", geometry.reference_observation_id, 0))
                    continue
                if command == "prefetch_static":
                    from .gpu_geometry import complete_static
                    complete_static(geometry, poses)
                    connection.send(("static_prefetched",
                                     [(u[0], geometry._static_cache[u[0]]) for u in poses],
                                     geometry.gpu_exact_seconds))
                    continue
                for frame in frames:
                    try:
                        geometry.self_query.query(poses[frame], reuse=geometry._self_reuse.setdefault(frame, {}))
                    except GeometryQueryError:
                        pass  # Normal measurement records the same explicit failure.
                connection.send(("prefetched", None, 0))
            except Exception as error:
                connection.send(("error", error, 0))
            continue
        poses, jacobians, protected, order, static_updates, *moving = message
        if poses is None:
            poses = geometry._repair_pose_view
        from .repair_transport import decode_protected
        protected = decode_protected(geometry, protected)
        visit = frames if order is None else [frame for frame in order if frame in own]
        try:
            if static_updates is not None:
                from .gpu_geometry import complete_static
                complete_static(geometry, static_updates)
            part = geometry.measure_frames(poses, visit, jacobians=jacobians,
                                           protected_depths=protected, stop=stop)
            if static_updates is not None:
                part["_static_updates"] = [(u[0], geometry._static_cache[u[0]]) for u in static_updates]
                part["_static_seconds"] = geometry.gpu_exact_seconds
            part["_static_query_coverage"] = dict(getattr(geometry, "static_query_coverage", {}))
            part["_static_query_examples"] = list(getattr(geometry, "static_query_examples", ()))
            count = 0 if geometry.moving is None else geometry.moving.query_count
            reply = ("ok", part, count)
        except Exception as error:  # re-raised by the parent
            reply = ("error", error, 0)
        connection.send(reply)
    connection.close()


class ParallelMeasure:
    def __init__(self, geometry, frames):
        self.frames = int(frames)
        self.geometry = geometry
        self.workers, self.counts = [], []
        self.reference, self.recent = None, []
        self.reference_observation_id = geometry.reference_observation_id
        self.prefetched = False
        self.static_sender, self.static_pending = None, []
        count = min(WORKERS, self.frames // MIN_FRAMES_PER_WORKER)
        if count < 2 or not hasattr(os, "fork"):
            return
        context = multiprocessing.get_context("fork")
        from .repair_transport import prepare
        prepare(geometry, context, self.frames)
        # First rejected frame found by any worker; -1 while none is known.
        self.stop = context.RawValue("i", -1)
        available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        share = max(1, len(available) // count)
        for index in range(count):
            cores = set()
            parent, child = context.Pipe()
            process = context.Process(target=_serve, daemon=True,
                                      args=(geometry, range(index, self.frames, count), cores, child, self.stop))
            process.start()
            child.close()
            self.workers.append((process, parent))
        self.counts = [0] * count

    def sync_reference_observations(self):
        """One explicit reference update; trials never send evidence mutations."""
        if self.prefetched or self.static_sender is not None:
            raise RuntimeError("Freeze observations only while geometry workers are idle")
        geometry, count = self.geometry, len(self.workers)
        for index, (_, connection) in enumerate(self.workers):
            registry = {f: v for f, v in geometry._moving_evidence_baseline.items() if f % count == index}
            failures = {f: v for f, v in geometry.reference_observation_failures.items() if f % count == index}
            connection.send(("reference_observations", (
                registry, failures, geometry._reference_pose_key, geometry.reference_observation_id)))
        error = None
        for _, connection in self.workers:
            status, value, _ = connection.recv()
            if status == "error":
                error = error or value
            elif status != "reference_frozen" or value != geometry.reference_observation_id:
                error = error or RuntimeError("Geometry worker reference registry mismatch")
        if error is not None:
            raise error
        self.reference_observation_id = geometry.reference_observation_id

    def prefetch_self(self, poses):
        """Overlap exact self-pair cache filling with parent GPU scene screening."""
        value = np.asarray(poses)
        if not self.prefetched:
            from .repair_transport import pose_message
            value = pose_message(self.geometry, value)
        for _, connection in self.workers:
            connection.send(("prefetch_self", value))
        self.prefetched = True

    def prefetch_static(self, updates):
        """Overlap exact static queries with parent moving-object screening."""
        count = len(self.workers)
        parts = [[] for _ in self.workers]
        for update in updates:
            parts[update[0][0] % count].append(update)
        self.static_sender = ThreadPoolExecutor(max_workers=count)
        self.static_pending = [self.static_sender.submit(connection.send, ("prefetch_static", part))
                               for (_, connection), part in zip(self.workers, parts)]

    def measure(self, poses, jacobians, protected_depths, *, static_updates=None,
                moving_updates=None, frames=None):
        poses = np.asarray(poses)
        stream = getattr(static_updates, "query_worker", None)
        if self.prefetched and stream is None:
            for _, connection in self.workers:
                status, value, _ = connection.recv()
                if status == "error":
                    raise value
            self.prefetched = False
        prefetched_seconds = []
        if self.static_sender is not None:
            for sent in self.static_pending:
                sent.result()
            self.static_sender.shutdown()
            self.static_sender, self.static_pending = None, []
            error = None
            for _, connection in self.workers:
                status, value, seconds = connection.recv()
                if status == "error":
                    error = error or value
                else:
                    self.geometry._static_cache.update(value)
                    prefetched_seconds.append(seconds)
            if error is not None:
                raise error
            static_updates = None  # Already exact and resident in each worker.
        order = None
        if protected_depths is not None and self.reference is not None:
            # Line-search candidates tend to fail where their predecessors did,
            # else where they moved most since the last complete measurement.
            # Unchanged frames are exact cache hits.
            change = np.abs(poses - self.reference).max(axis=1)
            change[self.recent] = np.inf
            order = np.argsort(-change, kind="stable").tolist()
            first = {frame: rank for rank, frame in enumerate(self.recent)}
            order[:len(first)] = sorted(order[:len(first)], key=first.get)
        if frames is not None:
            requested = set(frames)
            order = list(frames) if order is None else [f for f in order if f in requested]
        self.stop.value = -1
        count = len(self.workers)
        updates = None if static_updates is None else [[] for _ in self.workers]
        if updates is not None and stream is None:
            for update in static_updates:
                updates[update[0][0] % count].append(update)
        moving = None if moving_updates is None else [[] for _ in self.workers]
        if moving is not None:
            for update in moving_updates:
                moving[update[0] % count].append(update)

        pose_payload, protected_payload = poses, protected_depths
        if hasattr(self.geometry, '_repair_pose_view'):
            from .repair_transport import pose_message, protected_message
            # Streaming may deliberately retain an unfinished self prefetch.
            # Its shared snapshot must survive until the acknowledgements are
            # drained below; this exceptional request uses ordinary pickle.
            if not self.prefetched:
                pose_payload = pose_message(self.geometry, poses)
            protected_payload = protected_message(self.geometry, protected_depths)

        def message(index):
            value = (pose_payload, jacobians, protected_payload, order, None if updates is None else updates[index])
            return value if moving is None else (*value, moving[index])
        launch = list(range(count))
        if stream is not None and order is not None:
            rank = {frame: index for index, frame in enumerate(order)}
            launch.sort(key=lambda index: min((rank[f] for f in range(index, self.frames, count)
                                              if f in rank), default=len(rank)))
        if stream is None:
            for index in launch:
                self.workers[index][1].send(message(index))
        else:
            # A worker may still be filling its exact self cache. Pipe writes
            # must not block the parent from screening the next GPU batch.
            with ThreadPoolExecutor(max_workers=count) as sender:
                pending = []
                for index in launch:
                    updates[index] = stream(index, count) if self.stop.value < 0 else None
                    pending.append(sender.submit(self.workers[index][1].send,
                                                 message(index)))
                for sent in pending:
                    sent.result()
            if self.prefetched:
                for _, connection in self.workers:
                    status, value, _ = connection.recv()
                    if status == "error":
                        raise value
                self.prefetched = False
        parts, error, static_seconds = [], None, prefetched_seconds
        for index, (_, connection) in enumerate(self.workers):
            status, value, count = connection.recv()
            if status == "error":
                error = error or value
            else:
                self.geometry._static_cache.update(value.pop("_static_updates", ()))
                static_seconds.append(value.pop("_static_seconds", 0.))
                coverage = value.pop("_static_query_coverage", {})
                for name, count_value in coverage.items():
                    current = self.geometry.static_query_coverage
                    current[name] = current.get(name, 0) + count_value
                from .original_rigid_fallback import record_example
                for example in value.pop("_static_query_examples", []):
                    record_example(self.geometry, example)
                parts.append(value)
                self.counts[index] = count
        if error is not None:
            raise error
        if (static_updates is not None or prefetched_seconds) and hasattr(self.geometry, "gpu_stats"):
            self.geometry.gpu_stats["exact_worker_seconds"] = static_seconds
        found = [part["rejection"] for part in parts if part["rejection"] is not None]
        if not found:
            self.reference = poses.copy()
        self.recent = (sorted(set(found)) + [f for f in self.recent if f not in found])[:RECENT_REJECTIONS]
        return parts

    @property
    def query_count(self):
        return sum(self.counts)

    def close(self):
        if self.static_sender is not None:
            self.static_sender.shutdown()
            self.static_sender = None
            # If parent GPU screening failed, workers can still be sending
            # large static-cache replies. Drain them before closing pipes.
            for _, connection in self.workers:
                try:
                    if self.prefetched:
                        connection.recv()
                    connection.recv()
                except (EOFError, OSError):
                    pass
            self.prefetched = False
        for process, connection in self.workers:
            try:
                connection.send(None)
            except (BrokenPipeError, OSError):
                pass
            connection.close()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self.workers = []
