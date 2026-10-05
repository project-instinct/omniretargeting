"""Default-off strict QOCO-CUDA proposals with unchanged Clarabel fallback.

The same canonical matrices and cone rows are passed to either solver. Only
SOLVED and finite vectors are returned; all other outcomes request CPU fallback.
Workers never import the CUDA extension or create a CUDA context.
"""
from collections import Counter
import re
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .. import gpu_conic_runtime as runtime
from .runtime_options import OPTIONS

# Final crossover is selected from captured production programs; small sparse
# programs do not amortize cuDSS analysis and GPU setup.
MIN_VARIABLES = 1000
MIN_ROWS = 15000
MIN_NNZ = 250000
MAX_SOC_SIZE = 16
TOLERANCE = 1e-8
_COUNTS = Counter()
_RECORDS = []
_UNAVAILABLE = None
_CONE = re.compile(r'(\w+)\((\d+)\)')


def reset_report():
    _COUNTS.clear()
    _RECORDS.clear()


def report():
    return dict(enabled=OPTIONS.gpu_conic, tolerance=TOLERANCE,
                min_variables=MIN_VARIABLES, min_rows=MIN_ROWS,
                min_nnz=MIN_NNZ,
                max_soc_size=MAX_SOC_SIZE,
                main_process_only=True, unavailable=_UNAVAILABLE or runtime.error(),
                counts=dict(_COUNTS), solves=list(_RECORDS))


def dimensions(cones):
    """Accept only standard order: equality, nonnegative, then SOC blocks."""
    zero = nonnegative = 0
    sizes, previous = [], -1
    for cone in cones:
        match = _CONE.fullmatch(repr(cone))
        if match is None:
            return None
        kind, size = match.groups()
        rank = {'ZeroConeT': 0, 'NonnegativeConeT': 1, 'SecondOrderConeT': 2}.get(kind)
        if rank is None or rank < previous:
            return None
        previous, size = rank, int(size)
        if rank == 0:
            zero += size
        elif rank == 1:
            nonnegative += size
        else:
            sizes.append(size)
    return SimpleNamespace(zero=zero, nonneg=nonnegative, soc=sizes)


def _csc(matrix):
    value = sparse.csc_matrix(matrix, dtype=np.float64)
    if max(value.shape, default=0) > np.iinfo(np.int32).max or value.nnz > np.iinfo(np.int32).max:
        raise ValueError('QOCO CSC index overflow')
    # Do not mutate the caller's canonical CSC buffers.
    value = value.copy()
    value.indices = value.indices.astype(np.int32)
    value.indptr = value.indptr.astype(np.int32)
    return value


def solve(quadratic, linear, matrix, offset, dims, *, context='conic'):
    """Clarabel-compatible result namespace, or None for the original CPU path."""
    global _UNAVAILABLE
    if not OPTIONS.gpu_conic:
        return None
    if not runtime.main_process():
        _COUNTS['worker_cpu'] += 1
        return None
    n, m = matrix.shape[1], matrix.shape[0]
    nnz = matrix.nnz + (quadratic.nnz if quadratic is not None else 0)
    if n < MIN_VARIABLES or m < MIN_ROWS or nnz < MIN_NNZ:
        _COUNTS['small_cpu'] += 1
        return None
    if _UNAVAILABLE or runtime.error():
        _COUNTS['unavailable_cpu'] += 1
        return None
    device_error = runtime.device_error()
    if device_error:
        _UNAVAILABLE = device_error
        _COUNTS['unavailable_cpu'] += 1
        return None
    if dims is None or any(getattr(dims, name, 0) for name in ('psd', 'exp', 'p3d', 'pnd')):
        _COUNTS['unsupported_cpu'] += 1
        return None
    if max(dims.soc, default=0) > MAX_SOC_SIZE:
        # cuDSS expands a large SOC into a dense KKT clique: a captured 2162D
        # trust cone took minutes, versus seconds on the original CPU solver.
        _COUNTS['large_soc_cpu'] += 1
        return None
    started = time.perf_counter()
    row = dict(context=context, n=n, m=m, nnz=int(nnz))
    from .performance_trace import _ACTIVE_TRACE
    trace = _ACTIVE_TRACE.get()
    if trace is not None:
        row['stage'] = trace.current_stage
    try:
        import qoco
        zero = int(dims.zero)
        sizes = np.asarray(dims.soc, dtype=np.int32)
        if zero + int(dims.nonneg) + int(sizes.sum()) != m:
            raise ValueError('Inconsistent canonical cone dimensions')
        P = _csc(sparse.triu(quadratic, format='csc') if quadratic is not None
                 else sparse.csc_matrix((n, n)))
        A = _csc(matrix)
        c, b = np.asarray(linear, float), np.asarray(offset, float)
        solver = qoco.QOCO(algebra='cuda')
        solver.setup(n, m-zero, zero, P, c, _csc(A[:zero]) if zero else None,
                     b[:zero] if zero else None, _csc(A[zero:]), b[zero:],
                     int(dims.nonneg), len(sizes), sizes, verbose=0,
                     max_iters=100, abstol=TOLERANCE, reltol=TOLERANCE)
        result = solver.solve()
        row.update(status=str(result.status), iterations=int(result.iters),
                   primal_residual=float(result.pres), dual_residual=float(result.dres),
                   gap=float(result.gap))
        if result.status != 'QOCO_SOLVED':
            _COUNTS[f'fallback_{result.status}'] += 1
            return None
        x = np.array(result.x, dtype=float, copy=True)
        dual = np.r_[result.y, result.z].astype(float)
        slack = b - A @ x
        if x.shape != (n,) or dual.shape != (m,) or not all(
                np.isfinite(v).all() for v in (x, dual, slack)):
            row['status'] = 'invalid_vectors'
            _COUNTS['fallback_invalid_vectors'] += 1
            return None
        _COUNTS['solved'] += 1
        return SimpleNamespace(x=x, z=dual, s=slack, status='Solved',
                               obj_val=float(result.obj), iterations=int(result.iters),
                               solve_time=time.perf_counter()-started)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, ArithmeticError) as error:
        row.update(status=type(error).__name__, error=str(error))
        _COUNTS[f'fallback_{type(error).__name__}'] += 1
        if isinstance(error, (ImportError, OSError)) or (
                isinstance(error, RuntimeError) and 'not available' in str(error)):
            _UNAVAILABLE = f'{type(error).__name__}: {error}'
        return None
    finally:
        row['seconds'] = time.perf_counter()-started
        _RECORDS.append(row)


def solve_problem(problem):
    """Use the direct adapter contract, including CVXPY's dual reconstruction."""
    import cvxpy as cp
    if not runtime.main_process():
        _COUNTS['worker_cpu'] += 1
        return False
    if problem.size_metrics.num_scalar_variables < MIN_VARIABLES:
        _COUNTS['small_cpu'] += 1
        return False
    data, chain, inverse = problem.get_problem_data(cp.CLARABEL, solver_opts={})
    result = solve(data.get('P'), data['c'], data['A'], data['b'], data['dims'])
    if result is None:
        return False
    try:
        problem.unpack_results(result, chain, inverse)
    except (cp.error.SolverError, ValueError, TypeError) as error:
        _COUNTS['solved'] -= 1
        _COUNTS['fallback_unpack'] += 1
        _RECORDS[-1].update(status='unpack_error', error=str(error))
        return False
    return True
