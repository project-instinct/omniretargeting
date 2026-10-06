"""Fork-process support for independent saved-frame self-collision queries."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import os
import sys

import numpy as np


@dataclass
class _SelfWorkerState:
    template: object
    poses: np.ndarray
    workers: int
    query: object | None = None


_SELF_STATE: _SelfWorkerState | None = None
_THREAD_LIMIT = None


def _frame_result(
    query: object, pose: np.ndarray, *, compact: bool = False
) -> dict[str, object]:
    try:
        rows = query.query(pose)
    except Exception as error:
        return {
            "rows": [],
            "error": f"{type(error).__name__}: {error}",
        }
    if not compact:
        return {"rows": rows}
    return {
        "rows": [
            {
                "pair": [int(gid) for gid in row["pair"]],
                "clearance": float(row["clearance"]),
            }
            for row in rows
        ]
    }


def _query_frame(frame: int) -> tuple[int, dict[str, object]]:
    assert _SELF_STATE is not None and _SELF_STATE.query is not None
    return frame, _frame_result(_SELF_STATE.query, _SELF_STATE.poses[frame], compact=True)


def _worker_initializer() -> None:
    global _THREAD_LIMIT
    from threadpoolctl import threadpool_limits

    assert _SELF_STATE is not None
    _SELF_STATE.query = _SELF_STATE.template.fresh_state()
    threads = max(1, len(os.sched_getaffinity(0)) // _SELF_STATE.workers)
    _THREAD_LIMIT = threadpool_limits(limits=threads)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def query_self_frames(query: object, poses: np.ndarray, workers: int) -> list[dict[str, object]]:
    """Return one raw query result per frame; callers own all aggregation."""
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers == 1:
        return [_frame_result(query, pose) for pose in poses]

    global _SELF_STATE
    _SELF_STATE = _SelfWorkerState(
        template=query, poses=np.asarray(poses), workers=workers
    )
    try:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_worker_initializer,
        ) as executor:
            indexed = list(executor.map(_query_frame, range(len(poses)), chunksize=1))
    finally:
        _SELF_STATE = None
    indexed.sort(key=lambda item: item[0])
    return [result for _frame, result in indexed]
