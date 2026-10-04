"""Default-off QOCO repair proposals, with unconditional Clarabel fallback.

Only a finite QOCO_SOLVED result may replace the existing conic solve. The
unchanged nonlinear guards and original-mesh checks still decide acceptance.
"""

import numpy as np
from scipy import sparse

QOCO_REPAIR_QP = False
QOCO_RESTORATIONS = False
TOLERANCE = 1e-4
MAX_ITERATIONS = 100


def solve(data):
    """Full canonical solution, or None to request the established solver."""
    try:
        import qoco

        dims = data["dims"]
        # Repair canonical programs contain nonnegative and SOC blocks only.
        if dims.zero or any(getattr(dims, name, 0) for name in ("psd", "exp", "p3d", "pnd")):
            return None
        sizes = np.asarray(dims.soc, dtype=np.int32)
        solver = qoco.QOCO()
        solver.setup(len(data["c"]), data["A"].shape[0], 0,
                     sparse.triu(data["P"], format="csc"), np.asarray(data["c"], float), None, None,
                     sparse.csc_matrix(data["A"]), np.asarray(data["b"], float),
                     int(dims.nonneg), len(sizes), sizes, verbose=0,
                     max_iters=MAX_ITERATIONS, abstol=TOLERANCE, reltol=TOLERANCE)
        result = solver.solve()
        if str(result.status) != "QOCO_SOLVED":
            return None
        x = np.asarray(result.x, dtype=float)
        if x.shape != (len(data["c"]),) or not np.isfinite(x).all():
            return None
        return x
    except (ImportError, RuntimeError, ValueError, ArithmeticError, OSError):
        return None
