"""Temporal windows for block-coordinate guarded sequence steps.

A guarded correction step over a long clip is one large, degenerate conic
problem. Frames couple only through short temporal differences, so the same
linearized problem restricted to a window of frames, with every other frame
held fixed, keeps all hard bounds exact while solving far smaller systems.
"""

import os
import pickle
import selectors
import traceback
from itertools import pairwise

import numpy as np


def temporal_windows(frames, size, sweep):
    """Disjoint consecutive windows covering every frame once per sweep.

    Alternate sweeps shift the boundaries by half a window, so no frame stays
    permanently next to a boundary whose neighbour is held fixed.
    """
    shift = (sweep % 2) * (size // 2)
    edges = sorted({0, frames, *range(shift, frames, size)})
    return [range(first, stop) for first, stop in pairwise(edges)]


def window_system(matrix, residual, window, width):
    """Objective rows that a step confined to ``window`` can change.

    ``matrix`` is the full-trajectory Jacobian in CSC form. Rows without a
    coefficient in the window are constant for this step and are dropped;
    the minimizer of the remaining least-squares problem is unchanged.
    """
    columns = matrix[:, window.start * width : window.stop * width].tocsr()
    keep = np.diff(columns.indptr) > 0
    return columns[keep], -np.asarray(residual)[keep]


def independent_groups(windows):
    """Alternate windows: members of a group share no frame stencil or row.

    Consecutive windows are separated by a whole window of the other group,
    which is wider than any temporal difference, so the step of one member
    cannot change a row, bound or per-frame priority of another member.
    """
    return [windows[0::2], windows[1::2]]


def _single_threaded_blas():
    """Workers already run in parallel; nested BLAS pools only oversubscribe.

    Each process otherwise starts one BLAS thread per logical CPU, so a group
    of workers multiplies the thread count and slows every worker down.
    Thread counts change scheduling only, not the computed values.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(1)


def run_forked(jobs, workers):
    """Run independent jobs in forked processes, returning results in order.

    Each child owns a copy-on-write image of the solver state (MuJoCo data,
    caches, guards), so the serial window step runs unchanged without any
    shared mutable state. Only picklable results return to the parent.
    """
    if workers <= 1 or len(jobs) <= 1 or not hasattr(os, "fork"):
        return [job() for job in jobs]
    results, failures = [None] * len(jobs), []
    pending, chunks = list(enumerate(jobs)), {}
    selector = selectors.DefaultSelector()
    try:
        while pending or chunks:
            # Keep every worker busy: start the next job as soon as one ends.
            while pending and len(chunks) < workers:
                index, job = pending.pop(0)
                read, write = os.pipe()
                pid = os.fork()
                if pid == 0:  # child: never return into the parent's control flow
                    os.close(read)
                    try:
                        _single_threaded_blas()
                        payload = pickle.dumps(("ok", job()))
                    # Any exit of the child, even an interrupt, must end here
                    # and never continue into the parent's control flow.
                    except BaseException:  # noqa: BLE001
                        payload = pickle.dumps(("error", traceback.format_exc()))
                    with os.fdopen(write, "wb") as stream:
                        stream.write(payload)
                    os._exit(0)
                os.close(write)
                chunks[index] = []
                selector.register(read, selectors.EVENT_READ, (index, pid))
            for key, _ in selector.select():
                index, pid = key.data
                data = os.read(key.fd, 1 << 16)
                if data:
                    chunks[index].append(data)
                    continue
                selector.unregister(key.fd)
                os.close(key.fd)
                os.waitpid(pid, 0)
                payload = b"".join(chunks.pop(index))
                if not payload:
                    failures.append("A window worker exited without a result")
                    continue
                status, value = pickle.loads(payload)
                if status == "ok":
                    results[index] = value
                else:
                    failures.append(f"A window worker failed:\n{value}")
    finally:
        selector.close()
    if failures:
        raise RuntimeError(failures[0])
    return results
