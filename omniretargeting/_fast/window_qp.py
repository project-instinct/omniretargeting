"""Residual-controlled temporal block solve of one whole-trajectory conic QP.

Crossing rows and cones remain complete. A window iterate is returned only
after a full primal/dual KKT certificate at the full solver's tolerance;
otherwise None requests the caller's full solve. Coordinate-wise stagnation
is not convergence when constraints couple windows.
"""

import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

import cvxpy as cp
import numpy as np
from scipy import sparse

WINDOW_FRAMES = 40
SWEEPS = 8
THREADS = 8
_ACCEPTED = frozenset({"Solved", "AlmostSolved"})
# The process that imported this module; forked children see a different pid.
_MAIN_PID = os.getpid()


def solve_windows(problem, variable, frames, width, *, size=WINDOW_FRAMES, sweeps=SWEEPS):
    """Step for ``variable`` (frame-major, ``width`` per frame), or None to use the full solve."""
    data, _, _ = problem.get_problem_data(cp.CLARABEL)
    dims = data["dims"]
    n = len(data["c"])
    if dims.zero or dims.exp or dims.psd or dims.p3d or dims.pnd or n != frames * width or variable.size != n:
        return None
    return solve_standard(data.get("P"), data["c"], data["A"], data["b"], int(dims.nonneg),
                          np.asarray(dims.soc, dtype=int), np.arange(n) // width, frames,
                          size=size, sweeps=sweeps)


def solve_standard(P, c, A, b, nonneg, soc, frame_of, frames, *, size=WINDOW_FRAMES, sweeps=SWEEPS,
                   settings=None, accepted=_ACCEPTED):
    """Windowed solve of  min ½xᵀPx + cᵀx  s.t.  Ax + s = b,  s ∈ R₊^nonneg × SOCs.

    ``frame_of[i]`` is the frame of variable ``i``; ``settings`` returns fresh
    CLARABEL settings (default tolerances and QDLDL otherwise). ``sweeps`` is a
    maximum, never a convergence criterion. Returns a certified x, or None.
    """
    import clarabel
    from .qp_residual import kkt_residual

    n = len(c)
    A = sparse.csr_matrix(A)
    Acsc = A.tocsc()
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    P = sparse.csr_matrix((n, n)) if P is None else sparse.csr_matrix(P)
    P = (sparse.triu(P) + sparse.triu(P, 1).T).tocsr()
    soc = np.asarray(soc, dtype=int)
    starts = nonneg + np.r_[0, np.cumsum(soc)[:-1]].astype(int)
    cone_of_row = np.full(A.shape[0], -1)
    for k, (start, count) in enumerate(zip(starts, soc)):
        cone_of_row[start:start + count] = k
    frame_of = np.asarray(frame_of)
    if settings is None:
        def settings():
            # The full solve's single-threaded factorization. The default
            # ("auto": faer) starts a process-wide thread pool, and a forked
            # speculative attempt waits forever on that pool's absent threads.
            value = clarabel.DefaultSettings()
            value.verbose = False
            value.direct_solve_method = "qdldl"
            return value

    def window(columns, x):
        touched = np.diff(Acsc[:, columns].tocsr().indptr) > 0
        linear = np.flatnonzero(touched[:nonneg])
        cones = np.unique(cone_of_row[nonneg:][touched[nonneg:]])
        rows = np.r_[linear, *(starts[k] + np.arange(soc[k]) for k in cones)].astype(int)
        fixed = x.copy()
        fixed[columns] = 0.0
        kinds = ([clarabel.NonnegativeConeT(len(linear))] if len(linear) else []) + [
            clarabel.SecondOrderConeT(int(soc[k])) for k in cones]
        solution = clarabel.DefaultSolver(
            sparse.triu(P[columns][:, columns]).tocsc(), c[columns] + P[columns] @ fixed,
            A[rows][:, columns].tocsc(), b[rows] - A[rows] @ fixed, kinds, settings()).solve()
        return columns, np.asarray(solution.x), rows, np.asarray(solution.z), str(solution.status)

    # Speculative attempts run in forked children; they solve their windows
    # one after another rather than start threads after a fork.
    threads = 1 if multiprocessing.parent_process() is not None or os.getpid() != _MAIN_PID else THREADS
    # Coloring by time is safe only when no cone/linear row or Hessian edge
    # couples two windows of the same color. A global repair trust-region
    # cone violates that assumption even if each of its vector rows is local.
    coo = A.tocoo()
    first, last = np.full(A.shape[0], frames), np.full(A.shape[0], -1)
    np.minimum.at(first, coo.row, frame_of[coo.col])
    np.maximum.at(last, coo.row, frame_of[coo.col])
    long_range = np.any(last[:nonneg] - first[:nonneg] >= size)
    for start, count in zip(starts, soc):
        long_range |= last[start:start + count].max() - first[start:start + count].min() >= size
    coo = P.tocoo()
    long_range |= np.any(np.abs(frame_of[coo.row] - frame_of[coo.col]) >= size)
    if long_range:
        threads = 1
    x = np.zeros(n)
    z = np.zeros(len(b))
    stalled = 0
    for sweep in range(sweeps):
        before = x.copy()
        shift = (sweep % 2) * (size // 2)
        edges = sorted({0, frames, *range(shift, frames, size)})
        windows = [np.flatnonzero((frame_of >= first) & (frame_of < stop))
                   for first, stop in zip(edges[:-1], edges[1:])]
        # Groups alternate by window position, so no two members are adjacent.
        for group in ([w for w in windows[0::2] if len(w)], [w for w in windows[1::2] if len(w)]):
            if threads == 1:
                # Generator: each update is installed before solving the next
                # window, which matters when a global cone couples the group.
                results = (window(columns, x) for columns in group)
            else:
                with ThreadPoolExecutor(min(threads, max(1, len(group)))) as pool:
                    results = list(pool.map(lambda columns: window(columns, x), group))
            for columns, values, rows, duals, status in results:
                if status not in accepted or not np.isfinite(values).all():
                    return None
                x[columns] = values
                z[rows] = duals
        residual = kkt_residual(P, c, A, b, nonneg, soc, x, z, settings())
        if residual['certified']:
            return x
        stalled = stalled + 1 if np.max(np.abs(x - before), initial=0.) < 1e-11 else 0
        if stalled >= 2:
            break
    return None
