"""Mink task assembly with direct Clarabel norm constraints (no CVXPY).

A quadratic IK objective plus Euclidean error budgets is a conic QP. Keeping
the norm cones preserves the sequence guard's metric budgets; replacing them
with independent coordinate bounds would change the admissible error.
"""

import clarabel
import mink
import numpy as np
from mink.limits.limit import Constraint
from scipy import sparse


class Rows(mink.Task):
    def __init__(self, residual, jacobian):
        super().__init__(cost=np.ones(len(residual)))
        self.residual, self.jacobian = residual, jacobian

    def compute_error(self, configuration):
        return self.residual

    def compute_jacobian(self, configuration):
        return self.jacobian


class Linear(mink.Limit):
    def __init__(self, G, h):
        self.G, self.h = G, h

    def compute_qp_inequalities(self, configuration, dt):
        return Constraint(G=self.G, h=self.h)


def solve(configuration, rows, limits, budgets):
    """Solve mink's quadratic objective subject to linear and norm limits.

    ``budgets`` contains (residual, Jacobian, maximum norm). All Jacobians use
    the full MuJoCo tangent coordinates, including locked coordinates.
    """
    qp = mink.build_ik(configuration, [Rows(*rows)], dt=1.0,
                       damping=1e-8, limits=limits)
    matrices, values, cones = [], [], []
    if qp.G is not None:
        matrices.append(sparse.csc_matrix(qp.G))
        values.append(qp.h)
        cones.append(clarabel.NonnegativeConeT(len(qp.h)))
    if qp.A is not None:
        matrices.append(sparse.csc_matrix(qp.A))
        values.append(qp.b)
        cones.append(clarabel.ZeroConeT(len(qp.b)))
    for residual, jacobian, bound in budgets:
        # b - A x = (bound, r + J x) must lie in a second-order cone.
        matrices.append(sparse.csc_matrix(np.vstack([np.zeros(jacobian.shape[1]),
                                                     -jacobian])))
        values.append(np.r_[bound, residual])
        cones.append(clarabel.SecondOrderConeT(len(residual) + 1))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    scale = max(float(np.max(np.abs(qp.P), initial=0)),
                float(np.max(np.abs(qp.q), initial=0)), 1.0)
    solver = clarabel.DefaultSolver(
        sparse.triu(sparse.csc_matrix(qp.P / scale), format="csc"), qp.q / scale,
        sparse.vstack(matrices, format="csc"), np.concatenate(values), cones,
        settings,
    )
    solution = solver.solve()
    if solution.status != clarabel.SolverStatus.Solved:
        return None
    step = np.asarray(solution.x)
    if not np.all(np.isfinite(step)):
        return None
    # Independently check metric feasibility, even for a successful solver status.
    tolerance = 1e-7
    if qp.G is not None and np.any(qp.G @ step > qp.h + tolerance):
        return None
    if qp.A is not None and np.any(np.abs(qp.A @ step - qp.b) > tolerance):
        return None
    if any(np.linalg.norm(r + j @ step) > b + tolerance for r, j, b in budgets):
        return None
    return step
