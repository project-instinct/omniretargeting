"""Cold-start the same native HiGHS LP that SciPy builds for a separator.

Use SciPy's bundled engine, rather than a separately versioned ``highspy``:
the selected separator (not just its optimal value) affects uncertainty.
The public ``linprog`` caller remains the fallback when the native API is
unavailable or does not return an optimum.
"""

import numpy as np
from scipy.sparse import csc_array

try:
    from scipy.optimize._highspy import _core as _highs
except ImportError:
    _highs = None


def _options():
    if _highs is None:
        return None
    options = _highs.HighsOptions()
    # _linprog_highs passes these explicit values; all None-valued options
    # (solver, time/iteration limits, IPM tolerance and edge-weight strategy)
    # retain the bundled engine's defaults, exactly as its wrapper does.
    options.presolve = "on"
    options.highs_debug_level = 0
    options.log_to_console = False
    options.output_flag = False
    options.primal_feasibility_tolerance = 1e-9
    options.dual_feasibility_tolerance = 1e-9
    options.simplex_strategy = int(_highs.simplex_constants.SimplexStrategy.kSimplexStrategyDual)
    return options


_OPTIONS = _options()


def separator(rays, wrench):
    """Return SciPy-identical x, or None to request the public-API fallback."""
    if _OPTIONS is None:
        return None
    rows = len(rays)
    # CSC conversion drops explicit zeros and preserves row/column ordering,
    # just like csc_array(vstack((A_ub, empty_A_eq))) in _linprog_highs.
    matrix = csc_array(-rays)
    lp = _highs.HighsLp()
    lp.num_col_, lp.num_row_ = 6, rows
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = 6, rows
    lp.a_matrix_.format_ = _highs.MatrixFormat.kColwise
    lp.col_cost_ = wrench
    lp.col_lower_, lp.col_upper_ = -np.ones(6), np.ones(6)
    lp.row_lower_ = np.full(rows, -_highs.kHighsInf)
    lp.row_upper_ = np.zeros(rows)
    lp.a_matrix_.start_ = matrix.indptr
    lp.a_matrix_.index_ = matrix.indices
    lp.a_matrix_.value_ = matrix.data
    # A new solver is essential: warm-starting an optimal face can return a
    # different separator even when the objective value is unchanged.
    model = _highs._Highs()
    if model.passOptions(_OPTIONS) == _highs.HighsStatus.kError:
        return None
    if model.passModel(lp) == _highs.HighsStatus.kError:
        return None
    if model.run() == _highs.HighsStatus.kError:
        return None
    if model.getModelStatus() != _highs.HighsModelStatus.kOptimal:
        return None
    return np.array(model.getSolution().col_value)
