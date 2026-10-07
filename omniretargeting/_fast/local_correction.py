"""Small minimum-norm SQP corrections, with exact trust-ball screening.

Only constraints involving window variables enter the program. Screening is
independent of the active set: a constraint is removed only when *every* point
in the physical joint-space trust ball satisfies it strictly. A larger retry
therefore rebuilds the screened program from the same complete linearization.
"""

import time

import numpy as np
from scipy import sparse

from .local_qp import ACCEPTED, _settings

# Work limits are deterministic: at most three trust radii and this many IPM
# iterations per solve. Wall-clock time is recorded only by the caller below.
MAX_ITERATIONS = 60
# Corrections are normally much smaller than the trust ball. Keeping their
# objective near unit scale avoids losing the minimum-norm term to gap tolerances.
STEP_SCALE = 1e-4
SMALL_CORRECTION_RADIUS = 2e-4


def _norms(matrix):
    return np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())


class _Problem:
    def __init__(self, bounds, projection):
        self.projection = sparse.csc_matrix(projection)
        self.variables = projection.shape[1]
        p = self.projection
        self.selector = (np.all(np.diff(p.indptr) == 1) and np.all(p.data == 1.)
                         and len(np.unique(p.indices)) == self.variables)
        matrix, self.lower, self.upper = bounds.packed()
        matrix = sparse.csr_matrix(matrix)
        self.matrix = (matrix @ p).tocsr()
        self.matrix.eliminate_zeros()
        self.varying = np.diff(self.matrix.indptr) > 0
        # For a selector ||P d|| == ||d||. For a general P use the original
        # physical row: |a P d| <= ||a|| ||P d||, without assuming orthogonality.
        self.reach = _norms(self.matrix if self.selector else matrix)
        # packed() equilibrates rows. Retain the original units as well so
        # scaling a large bound down cannot conceal a physical violation.
        self.physical = (sparse.vstack(bounds.matrices, format="csr") @ p).tocsr()
        self.physical.eliminate_zeros()
        self.physical_varying = np.diff(self.physical.indptr) > 0
        self.physical_lower = np.concatenate(bounds.lower)
        self.physical_upper = np.concatenate(bounds.upper)
        self.norms = []
        self.constants = int(np.sum(~self.varying & ((self.lower > 0) | (self.upper < 0))))
        for jac, residual, allowed in bounds.cones():
            jac = sparse.csr_matrix(jac)
            projected = (jac @ p).tocsr()
            projected.eliminate_zeros()
            dimension = residual.shape[1]
            varying = np.diff(projected.indptr).reshape(-1, dimension).sum(axis=1) > 0
            self.constants += int(np.sum(~varying & (np.linalg.norm(residual, axis=1) > allowed)))
            reach = _norms(projected if self.selector else jac).reshape(-1, dimension).sum(axis=1)
            self.norms.append((projected, residual, allowed, varying, reach))

    def program(self, radius, screen):
        import clarabel

        upper = self.varying & np.isfinite(self.upper)
        lower = self.varying & np.isfinite(self.lower)
        if screen:
            # Strict inequality keeps tangent rows and violated rows. One
            # outward ulp protects the bound against multiplication roundoff.
            reach = np.nextafter(self.reach * radius, np.inf)
            upper &= self.upper <= reach
            lower &= -self.lower <= reach
        linear = sparse.vstack((self.matrix[upper], -self.matrix[lower]), format="csc")
        scale = min(radius, STEP_SCALE)
        blocks = [linear * scale]
        targets = [np.r_[self.upper[upper], -self.lower[lower]]]
        cones = [clarabel.NonnegativeConeT(linear.shape[0])] if linear.shape[0] else []
        # Solve in x = delta / scale; trust normalization remains independent
        # of this objective scaling, so ||P delta|| <= radius exactly.
        trust = sparse.eye(self.variables, format="csc") if self.selector else self.projection
        if not self.selector:
            trust = trust.tocsr()
            trust = trust[np.diff(trust.indptr) > 0]
        blocks.append(sparse.vstack((sparse.csc_matrix((1, self.variables)),
                                     -trust * (scale / radius)), format="csc"))
        targets.append(np.r_[1., np.zeros(trust.shape[0])])
        cones.append(clarabel.SecondOrderConeT(trust.shape[0] + 1))
        kept_cones = 0
        for jac, residual, allowed, varying, reach in self.norms:
            dimension = residual.shape[1]
            keep = varying.copy()
            if screen:
                slack = allowed - np.linalg.norm(residual, axis=1)
                keep &= slack <= np.nextafter(reach * radius, np.inf)
            count = int(keep.sum())
            if not count:
                continue
            normalization = np.maximum(allowed[keep], 1e-3)
            selected = jac[np.repeat(keep, dimension)]
            selected = (sparse.diags(np.repeat(scale / normalization, dimension)) @ selected).tocoo()
            row = selected.row + selected.row // dimension + 1
            blocks.append(sparse.csc_matrix((-selected.data, (row, selected.col)),
                                            shape=(count * (dimension + 1), self.variables)))
            targets.append(np.c_[allowed[keep] / normalization,
                                  residual[keep] / normalization[:, None]].ravel())
            cones.extend(clarabel.SecondOrderConeT(dimension + 1) for _ in range(count))
            kept_cones += count
        program = (sparse.eye(self.variables, format="csc") * 2,
                   np.zeros(self.variables), sparse.vstack(blocks, format="csc"),
                   np.concatenate(targets), cones)
        return program, dict(rows=int(linear.shape[0]), cones=kept_cones,
                             constant_violations=self.constants)

    def feasible(self, delta, radius, diagnostics=None):
        """Check the complete projected system, excluding its fixed constraints."""
        values = self.matrix @ delta
        keep = self.varying
        linear_error = float(max(np.max(self.lower[keep] - values[keep], initial=0.),
                                 np.max(values[keep] - self.upper[keep], initial=0.)))
        values = self.physical @ delta
        keep = self.physical_varying
        linear_error_absolute = float(max(
            np.max(self.physical_lower[keep] - values[keep], initial=0.),
            np.max(values[keep] - self.physical_upper[keep], initial=0.)))
        cone_error = cone_error_absolute = 0.
        for jac, residual, allowed, varying, _ in self.norms:
            values = (jac @ delta).reshape(residual.shape) + residual
            # Report both the solver's cone normalization and physical units.
            error = (np.linalg.norm(values, axis=1) - allowed) / np.maximum(allowed, 1e-3)
            cone_error = max(cone_error, float(np.max(error[varying], initial=0.)))
            cone_error_absolute = max(cone_error_absolute,
                                      float(np.max((np.linalg.norm(values, axis=1) - allowed)[varying],
                                                   initial=0.)))
        trust_error = max(0., float(np.linalg.norm(self.projection @ delta) - radius))
        if diagnostics is not None:
            diagnostics.update(linear_error=linear_error, linear_error_absolute=linear_error_absolute,
                               cone_error=cone_error,
                               cone_error_absolute=cone_error_absolute, trust_error=trust_error)
        # Cone normalization is for conditioning, not a stricter physical
        # acceptance threshold (a 1 mm cone would otherwise require 10 pm).
        return (linear_error <= 1e-8 and linear_error_absolute <= 1e-8
                and cone_error_absolute <= 1e-9 and trust_error <= 1e-9)


def _solve(problem, radius, *, screen=True):
    import clarabel

    program, info = problem.program(radius, screen)
    settings = _settings()
    settings.direct_solve_method = "qdldl"
    settings.max_threads = 1
    settings.max_iter = MAX_ITERATIONS
    settings.time_limit = float("inf")
    info["max_iter"] = MAX_ITERATIONS
    try:
        result = clarabel.DefaultSolver(*program, settings).solve()
    except RuntimeError as error:
        return None, "NumericalError", dict(info, error=str(error))
    status = str(result.status)
    info["iterations"] = int(result.iterations)
    if status in ACCEPTED and result.x is not None and np.isfinite(result.x).all():
        delta = np.asarray(result.x) * min(radius, STEP_SCALE)
        info["linearized_feasible"] = problem.feasible(delta, radius, info)
        # Status alone is not a feasibility certificate, especially for
        # inexact solves. Screening also requires the physical trust ball.
        if info["linearized_feasible"]:
            return delta, status, info
    return None, status, info


def solve(bounds, projection, radius, *, screen=True):
    """Return window-coordinate delta, status, diagnostics for one trust ball."""
    if projection.shape[1] == 0:
        return None, "no_local_dofs", {}
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Correction radius must be finite and positive")
    return _solve(_Problem(bounds, projection), radius, screen=screen)


def correction(limits, current, trial, projection, rows, cuts, linear, radius, ledger):
    """Linearize at the trial; at most three solves, with fixed iteration limits."""
    from .local_qp import cut_lower, witness_lower
    from .local_repair import CORRECTION_RADIUS, _reserve

    if not projection.shape[1]:
        return None
    qi = limits.coords.qi
    bounds = limits.constraints(trial, projection=projection)
    _reserve(bounds)
    collision, phi, budgets, _ = rows
    offset = (trial[:, qi] - current[:, qi]).ravel()
    bounds.add(collision, witness_lower(phi, budgets, collision @ offset), np.inf)
    for matrix, value, budget, displacement in cuts:
        bounds.add(matrix, cut_lower(matrix, value, budget, displacement), np.inf)
    for matrix, lower, upper in linear:
        bounds.add(matrix, lower, upper)
    problem = _Problem(bounds, projection)
    radii = (SMALL_CORRECTION_RADIUS, CORRECTION_RADIUS, max(radius, CORRECTION_RADIUS))
    for ball in dict.fromkeys(radii):
        tick = time.monotonic()
        delta, status, info = _solve(problem, ball)
        ledger.corrections.append(dict(radius=ball, status=status,
                                       seconds=time.monotonic() - tick, **info))
        if delta is not None:
            return np.asarray(projection @ delta).reshape(limits.frames, limits.width)
    return None
