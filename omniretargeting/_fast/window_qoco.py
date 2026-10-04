"""QOCO interior-point solve of window proposal (bounded-step) programs.

Offline on 72 real window proposals QOCO (1e-4 tolerances) was 2.0x faster than
Clarabel and solved all of them; restoration programs keep Clarabel, where QOCO
reported numerical errors (and an end-to-end trial was slower and worse). The
proposal is still checked by every nonlinear guard and by the objective decrease,
and a status other than solved falls back to the established Clarabel solve.
Default since 2026-10-03 (user decision): 957-frame TRUMANS total 344 -> 307 s,
acquisition 203 -> 146 s; self penetration 0.99 -> 1.70 mm, frames over 1 cm
106 -> 109, sticking +3 mm, world tracking -7 mm, contact-normal -30 mm.
"""

import numpy as np
from scipy import sparse
from collections import Counter
import time

WINDOW_QOCO_PROPOSALS = True
WINDOW_QOCO_RESTORATIONS = False
TOLERANCE = 1e-4
MAX_ITERATIONS = 80
RESTORATION_DIAGNOSTICS = Counter()


def solve_proposal(data, variables):
    """Step of a canonical window program (Clarabel form, no zero cone), or None."""
    import qoco

    dims = data['dims']
    if dims.zero:
        return None
    sizes = np.asarray(dims.soc, dtype=np.int32)
    solver = qoco.QOCO()
    solver.setup(len(data['c']), data['A'].shape[0], 0,
                 sparse.triu(data['P'], format='csc'), np.asarray(data['c'], float), None, None,
                 sparse.csc_matrix(data['A']), np.asarray(data['b'], float),
                 int(dims.nonneg), len(sizes), sizes,
                 verbose=0, max_iters=MAX_ITERATIONS, abstol=TOLERANCE, reltol=TOLERANCE)
    result = solver.solve()
    if str(result.status) != 'QOCO_SOLVED':
        return None
    step = np.asarray(result.x[:variables], dtype=float)
    return step if np.isfinite(step).all() else None


def solve_restoration(data, variables):
    """Optional restoration attempt; every unusable outcome retains Clarabel.

    This changes proposal arithmetic only. The existing restoration loop and
    its nonlinear guard acceptance remain the consumers of a solved step.
    """
    started = time.perf_counter()
    RESTORATION_DIAGNOSTICS['attempts'] += 1
    try:
        step = solve_proposal(data, variables)
        RESTORATION_DIAGNOSTICS['solved' if step is not None else 'fallback'] += 1
        return step
    except (ImportError, RuntimeError, ValueError, ArithmeticError):
        RESTORATION_DIAGNOSTICS['fallback'] += 1
        return None
    finally:
        RESTORATION_DIAGNOSTICS['seconds'] += time.perf_counter() - started
