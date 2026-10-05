"""Shared conic solve entry point with an optional CUDA backend.

The optimization problem, constraints, and acceptance checks are unchanged by
the backend. GPU selection is process-wide so motion clips and robot profiles
do not need solver-specific tuning.
"""

from __future__ import annotations

import os
from functools import lru_cache
from types import SimpleNamespace

from .gpu_conic_runtime import prepare as _prepare_gpu_runtime

_prepare_gpu_runtime()

import cvxpy as cp
import numpy as np
from scipy import sparse

_SUCCESS = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
_CERTIFIED = {cp.INFEASIBLE, cp.UNBOUNDED}
_GPU_ERRORS = (cp.error.SolverError, ImportError, OSError, RuntimeError, ValueError)
# Interior-point targets for one retry after a numerical stall. Degenerate
# budget rows can leave CLARABEL's default 1e-8 targets unreachable although
# its objective has converged; callers still re-check every step nonlinearly.
RETRY_TOLERANCE = 1e-6
# Constraint generation (``solve_conic_working_set``). The first working set
# holds every row/cone within this slack of its bound at the zero step; an
# omitted one joins once its slack at a solved step falls below the margin.
# Both only affect speed: the returned step satisfies every omitted
# constraint with at least the margin to spare.
WORKING_SET_SLACK = 1e-3
WORKING_SET_MARGIN = 1e-6
WORKING_SET_ROUNDS = 16
# Beyond this share of all rows the complete problem is solved instead.
WORKING_SET_FRACTION = 0.5
# Smaller problems stay on CLARABEL with the GPU backend: a CUDA solve has a
# fixed setup cost that a few-variable contact or anchor problem never repays.
GPU_MIN_VARIABLES = 2000


@lru_cache(maxsize=64)
def _soc_descriptor(size):
    """Clarabel copies immutable cone dimensions into each private solver."""
    import clarabel
    return clarabel.SecondOrderConeT(size)


def gpu_enabled() -> bool:
    return os.environ.get("OMNIRETARGETING_CONIC_BACKEND", "clarabel").lower() in {
        "qoco_cuda", "qoco_cuda_strict"
    }


def gpu_strict() -> bool:
    return os.environ.get("OMNIRETARGETING_CONIC_BACKEND", "").lower() == "qoco_cuda_strict"


def solve_conic(problem: cp.Problem, *, cpu_only: bool = False, **cpu_options) -> float | None:
    """Solve with the selected backend, retaining CLARABEL as GPU fallback.

    Set ``OMNIRETARGETING_CONIC_BACKEND=qoco_cuda`` to use QOCO's CUDA linear
    algebra. ``qoco_cuda_strict`` exposes GPU failures during validation.
    The default is the established CLARABEL path. ``cpu_only`` keeps a caller
    on CLARABEL with either backend.
    """
    backend = os.environ.get("OMNIRETARGETING_CONIC_BACKEND", "clarabel").lower()
    if backend not in {"clarabel", "qoco_cuda", "qoco_cuda_strict"}:
        raise ValueError(f"Unknown conic backend: {backend}")
    from ._fast.runtime_options import OPTIONS
    if OPTIONS.gpu_conic:
        if not cpu_only and not {'max_iter', 'time_limit'} & cpu_options.keys():
            from ._fast.gpu_conic import solve_problem
            if solve_problem(problem):
                return problem.value
        return _solve_cpu(problem, cpu_options)
    if backend == "clarabel" or cpu_only or problem.size_metrics.num_scalar_variables < GPU_MIN_VARIABLES:
        return _solve_cpu(problem, cpu_options)

    gpu_options = {
        "solver": cp.QOCO,
        "algebra": "cuda",
        "verbose": cpu_options.get("verbose", False),
    }
    if "max_iter" in cpu_options:
        gpu_options["max_iters"] = cpu_options["max_iter"]
    if "max_iters" in cpu_options:
        gpu_options["max_iters"] = cpu_options["max_iters"]
    try:
        value = problem.solve(**gpu_options)
        if problem.status in _SUCCESS:
            return value
        if backend == "qoco_cuda_strict":
            raise cp.error.SolverError(f"QOCO CUDA status: {problem.status}")
    except _GPU_ERRORS:
        if backend == "qoco_cuda_strict":
            raise
    return _solve_cpu(problem, cpu_options)


def _solve_cpu(problem: cp.Problem, options: dict) -> float | None:
    """CLARABEL solve with one relaxed-accuracy retry after a numerical stall.

    Only a solve that did not converge is retried: successful and certified
    infeasible/unbounded outcomes are returned unchanged, and an explicit
    caller iteration/time budget is respected. The problem is not modified.
    """
    if str(options.get("solver", "")).upper() != cp.CLARABEL:
        return problem.solve(**options)
    try:
        value = problem.solve(**options)
    except cp.error.SolverError:
        pass
    else:
        if problem.status in _SUCCESS or problem.status in _CERTIFIED:
            return value
        if problem.status == cp.USER_LIMIT and {"max_iter", "time_limit"} & set(
            options
        ):
            return value
    return problem.solve(
        **{
            **options,
            "tol_gap_abs": RETRY_TOLERANCE,
            "tol_gap_rel": RETRY_TOLERANCE,
            "tol_feas": RETRY_TOLERANCE,
        }
    )


def solve_conic_working_set(problem: cp.Problem, **cpu_options) -> float | None:
    """``solve_conic`` by exact constraint generation, for small correction steps.

    A restoration moves a nearly admissible candidate by a tiny step, so only
    a few of its tens of thousands of rows can bind. The problem is solved
    with the rows and cones near their bound at the zero step; every omitted
    one is then evaluated at the solved step, and those without
    ``WORKING_SET_MARGIN`` of slack are added before solving again. The final
    step satisfies all omitted constraints strictly, so, with their
    multipliers zero, it is the optimum of the complete problem (unique for a
    strictly convex objective). Any other outcome falls back to
    ``solve_conic`` on the complete problem.

    The small working-set problems stay on the CPU with the GPU backend too
    (a GPU solve costs more than one of them); only the complete-problem
    fallback uses the selected backend.
    """
    if str(cpu_options.get("solver", "")).upper() != cp.CLARABEL:
        return solve_conic(problem, **cpu_options)
    import clarabel

    options = {key: value for key, value in cpu_options.items() if key not in ("solver", "verbose")}
    data, chain, inverse = problem.get_problem_data(cp.CLARABEL, solver_opts=options)
    dims = data["dims"]
    if dims.psd or dims.exp or dims.p3d or dims.pnd:
        return solve_conic(problem, **cpu_options)
    matrix = sparse.csr_matrix(data["A"])
    offset = np.asarray(data["b"], dtype=float)
    linear = np.asarray(data["c"], dtype=float)
    quadratic = (sparse.triu(data["P"]).tocsc() if data.get("P") is not None
                 else sparse.csc_matrix((len(linear), len(linear))))
    zero, nonneg = int(dims.zero), int(dims.nonneg)
    sizes = np.asarray(dims.soc, dtype=int)
    starts = zero + nonneg + np.r_[0, np.cumsum(sizes)[:-1]].astype(int)

    def margins(slack):
        """Slack of each nonnegative row and of each second-order cone."""
        cones = np.empty(len(sizes))
        for size in np.unique(sizes):
            which = np.flatnonzero(sizes == size)
            block = slack[starts[which, None] + np.arange(size)]
            cones[which] = block[:, 0] - np.linalg.norm(block[:, 1:], axis=1)
        return slack[zero:zero + nonneg], cones

    rows_linear, rows_cones = margins(offset)
    keep_linear = rows_linear <= WORKING_SET_SLACK
    keep_cones = rows_cones <= WORKING_SET_SLACK
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    for key, value in options.items():
        setattr(settings, key, value)
    iterations, seconds = 0, 0.0
    for _ in range(WORKING_SET_ROUNDS):
        cones = np.flatnonzero(keep_cones)
        rows = np.concatenate([np.arange(zero), zero + np.flatnonzero(keep_linear)]
                              + [starts[c] + np.arange(sizes[c]) for c in cones]).astype(int)
        if len(rows) > WORKING_SET_FRACTION * matrix.shape[0]:
            break
        kinds = ([clarabel.ZeroConeT(zero)] if zero else []) + (
            [clarabel.NonnegativeConeT(int(keep_linear.sum()))] if keep_linear.any() else [])
        kinds += [_soc_descriptor(int(sizes[c])) for c in cones]
        from ._fast.gpu_conic import dimensions, solve as solve_gpu
        reduced_matrix = matrix[rows].tocsc()
        from ._fast.runtime_options import OPTIONS
        solution = (solve_gpu(quadratic, linear, reduced_matrix, offset[rows],
                              dimensions(kinds), context='restoration_working_set')
                    if OPTIONS.gpu_conic else None)
        if solution is None:
            solution = clarabel.DefaultSolver(quadratic, linear, reduced_matrix, offset[rows],
                                              kinds, settings).solve()
        iterations += solution.iterations
        seconds += solution.solve_time
        if str(solution.status) != "Solved":
            break
        step = np.asarray(solution.x, dtype=float)
        slack = offset - matrix @ step
        now_linear, now_cones = margins(slack)
        add_linear = ~keep_linear & (now_linear < WORKING_SET_MARGIN)
        add_cones = ~keep_cones & (now_cones < WORKING_SET_MARGIN)
        if not add_linear.any() and not add_cones.any():
            duals = np.zeros(matrix.shape[0])
            duals[rows] = np.asarray(solution.z, dtype=float)
            problem.unpack_results(SimpleNamespace(
                status=solution.status, x=step, z=duals, s=slack, obj_val=solution.obj_val,
                solve_time=seconds, iterations=iterations), chain, inverse)
            return problem.value
        keep_linear |= add_linear
        keep_cones |= add_cones
    return solve_conic(problem, **cpu_options)
