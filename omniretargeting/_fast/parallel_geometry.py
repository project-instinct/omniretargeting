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

import numpy as np

WORKERS = max(1, min(16, 2 * (os.cpu_count() or 2) // 3))
# Frames differ greatly in cost. Unpinned workers let idle cores serve the
# libigl threads of busy ones; pinning each worker to its own share measured
# slower on a hybrid 8P+8E CPU (67 s vs 57-60 s of omomo_78 G1 measurement).
PIN_WORKERS = False
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
    # ``PIN_WORKERS`` each worker's threads stay on its own cores.
    if cores and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cores)
    own = set(frames)
    while True:
        message = connection.recv()
        if message is None:
            break
        poses, jacobians, protected, order = message
        visit = frames if order is None else [frame for frame in order if frame in own]
        try:
            part = geometry.measure_frames(poses, visit, jacobians=jacobians,
                                           protected_depths=protected, stop=stop)
            count = 0 if geometry.moving is None else geometry.moving.query_count
            reply = ("ok", part, count)
        except Exception as error:  # re-raised by the parent
            reply = ("error", error, 0)
        connection.send(reply)
    connection.close()


class ParallelMeasure:
    def __init__(self, geometry, frames):
        self.frames = int(frames)
        self.workers, self.counts = [], []
        self.reference, self.recent = None, []
        count = min(WORKERS, self.frames // MIN_FRAMES_PER_WORKER)
        if count < 2 or not hasattr(os, "fork"):
            return
        context = multiprocessing.get_context("fork")
        # First rejected frame found by any worker; -1 while none is known.
        self.stop = context.RawValue("i", -1)
        available = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        share = max(1, len(available) // count)
        for index in range(count):
            cores = (set(available[index * share:(index + 1) * share]) or set(available[index::count])
                     if PIN_WORKERS else set())
            parent, child = context.Pipe()
            process = context.Process(target=_serve, daemon=True,
                                      args=(geometry, range(index, self.frames, count), cores, child, self.stop))
            process.start()
            child.close()
            self.workers.append((process, parent))
        self.counts = [0] * count

    def measure(self, poses, jacobians, protected_depths):
        poses = np.asarray(poses)
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
        self.stop.value = -1
        for _, connection in self.workers:
            connection.send((poses, jacobians, protected_depths, order))
        parts, error = [], None
        for index, (_, connection) in enumerate(self.workers):
            status, value, count = connection.recv()
            if status == "error":
                error = error or value
            else:
                parts.append(value)
                self.counts[index] = count
        if error is not None:
            raise error
        found = [part["rejection"] for part in parts if part["rejection"] is not None]
        if not found:
            self.reference = poses.copy()
        self.recent = (sorted(set(found)) + [f for f in self.recent if f not in found])[:RECENT_REJECTIONS]
        return parts

    @property
    def query_count(self):
        return sum(self.counts)

    def close(self):
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
