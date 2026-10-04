"""Infeasible-start primal-dual KKT for the constrained quadratic step.

The physical GN model is quadratic. Scalar guard faces are linear and the
linearized vector guards remain convex quadratic balls. Newton elimination
adds PSD row blocks directly to its band, without changing physical budgets.
Intermediate primal/dual iterates are proposals; exact guards certify motion.
"""

import time
import numpy as np
from scipy import sparse
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.linalg.blas import dsbmv

from .barrier_assembly import scatter
from .constrained_active import ActiveSetFailure

MAX_ITERATIONS = 45


def band_product(ab, vector):
    return dsbmv(len(ab) - 1, 1., ab, vector, lower=1)


class QuadraticDomain:
    """Scaled scalar faces and all component rows of affine vector balls."""
    def __init__(self, families, lower, upper, held):
        from .constrained_step import with_boxes
        from .rows import _merge_identical
        free = np.ones(len(lower))
        free[held] = 0.
        scalar, bounds = [], []
        self.balls = []
        for f in families:
            matrix = f.matrix.multiply(free).tocsr()
            if not f.ball:
                scalar.append(-matrix)
                reserve = np.minimum(1e-7, .5 * np.maximum(f.slack, 0.))
                bounds.append(f.slack - reserve)
            else:
                radius = np.sqrt(np.maximum(f.slack + np.square(f.residual).sum(axis=1), 0.))
                gap = radius - np.linalg.norm(f.residual, axis=1)
                reserve = 1e-8 if f.label[0] == 'RootDomain' else 1e-7
                radius -= np.minimum(reserve, .5 * np.maximum(gap, 0.))
                dim = f.residual.shape[1]
                norm = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel().reshape(-1, dim).sum(axis=1))
                keep = norm > 1e-14
                if np.any((~keep) & (gap < -1e-12)):
                    raise ActiveSetFailure('binding ball has no free tangent column')
                matrix = matrix[np.repeat(keep, dim)]
                if keep.any():
                    self.balls.append((matrix, f.residual[keep], radius[keep], norm[keep], dim))
        matrix = sparse.vstack(scalar, format='csr') if scalar else sparse.csr_matrix((0, len(lower)))
        bounds = np.concatenate(bounds) if bounds else np.empty(0)
        matrix, bounds = with_boxes(matrix, bounds, lower, upper)
        matrix = matrix.multiply(free).tocsr()
        matrix, _, bounds = _merge_identical(matrix, np.full(len(bounds), -np.inf), bounds)
        norm = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
        keep = norm > 1e-14
        if np.any((~keep) & (bounds < -1e-12)):
            raise ActiveSetFailure('binding scalar face has no free tangent column')
        possible = matrix.maximum(0.) @ upper + matrix.minimum(0.) @ lower
        keep &= possible >= bounds - 1e-12
        self.matrix = matrix[keep].multiply((1 / norm[keep])[:, None]).tocsr()
        self.bounds = bounds[keep] / norm[keep]
        self.count = len(self.bounds) + sum(len(r) for _, r, _, _, _ in self.balls)

    def evaluate(self, direction):
        values = [self.matrix @ direction - self.bounds]
        matrices = [self.matrix]
        for matrix, residual, radius, norm, dim in self.balls:
            vector = residual + (matrix @ direction).reshape(-1, dim)
            values.append((np.square(vector).sum(axis=1) - radius**2) / (2 * radius * norm))
            count = len(radius)
            reduce = sparse.csr_matrix(((vector / (radius * norm)[:, None]).ravel(),
                (np.repeat(np.arange(count), dim), np.arange(count * dim))), shape=(count, count * dim))
            matrices.append(reduce @ matrix)
        return np.concatenate(values), sparse.vstack(matrices, format='csr')

    def curvature(self, system, dual):
        offset = len(self.bounds)
        for matrix, residual, radius, norm, dim in self.balls:
            count = len(radius)
            scatter(system, matrix, np.repeat(dual[offset:offset + count] / (radius * norm), dim))
            offset += count


def _fraction(value, change):
    negative = change < 0
    return min(1., float(np.min(-value[negative] / change[negative], initial=np.inf)))


def solve(system, gradient, domain, warm, diagnostics, *, restoration=False):
    """Two RHS solves per Newton band factor, with positive primal/dual slacks."""
    scale_cost = max(float(np.max(np.abs(gradient), initial=0.)), 1.)
    physical = system / scale_cost
    linear = gradient / scale_cost
    started = time.monotonic()
    if not domain.count:
        if not np.any(gradient):
            return np.zeros(len(gradient))
        from .barrier_assembly import equilibrated_solve
        from .solver import _banded_solve
        diagnostics['factorizations'] += 1
        diagnostics['band_solves'] += 1
        return equilibrated_solve(physical, -linear, _banded_solve)
    direction = np.array(warm.get('ip_start', np.zeros(len(gradient))), copy=True)
    value, matrix = domain.evaluate(direction)
    slack = np.maximum(-value, 1e-5)
    cached = warm.get('ip_dual')
    dual = np.full(len(value), 1e-3) if cached is None or len(cached) != len(value) else np.maximum(
        cached * warm.get('ip_cost_scale', scale_cost) / scale_cost, 1e-10)
    trace = []
    status = 'iteration_budget'
    for iteration in range(MAX_ITERATIONS):
        value, matrix = domain.evaluate(direction)
        transpose = matrix.T.tocsr()
        stationarity = band_product(physical, direction) + linear + transpose @ dual
        primal = value + slack
        mu = float(slack @ dual / len(slack))
        excess = float(value.max(initial=0.))
        residual = max(float(np.max(np.abs(stationarity), initial=0.)),
                       float(np.max(np.abs(primal), initial=0.)), mu)
        trace.append(dict(iteration=iteration, excess=excess, residual=residual, mu=mu))
        stationarity_tolerance = 1e-4 if restoration else 1e-7
        central_tolerance = 1e-8 if restoration else 1e-11
        if (float(np.max(np.abs(stationarity), initial=0.)) < stationarity_tolerance and
                float(np.max(np.abs(primal), initial=0.)) < 1e-9 and mu < central_tolerance and excess < 1e-10):
            status = 'converged'
            break
        ab = physical.copy(order='F')
        domain.curvature(ab, dual)
        scatter(ab, matrix, dual / slack)
        scaling = 1 / np.sqrt(ab[0])
        for offset in range(len(ab)):
            length = ab.shape[1] - offset
            ab[offset, :length] *= scaling[offset:] * scaling[:length]
        diagnostics['factorizations'] += 1
        factor = cholesky_banded(ab, lower=True, overwrite_ab=True, check_finite=False)

        def newton(complement):
            rhs = -stationarity + transpose @ ((complement - dual * primal) / slack)
            dx = scaling * cho_solve_banded((factor, True), rhs * scaling, check_finite=False)
            ds = -primal - matrix @ dx
            dl = (-complement - dual * ds) / slack
            diagnostics['band_solves'] += 1
            return dx, ds, dl

        dx, ds, dl = newton(slack * dual)
        ap, ad = _fraction(slack, ds), _fraction(dual, dl)
        predicted_mu = float((slack + ap * ds) @ (dual + ad * dl) / len(slack))
        sigma = min(1., max(0., predicted_mu / mu)**3)
        dx, ds, dl = newton(slack * dual + ds * dl - sigma * mu)
        alpha = min(1., .995 * min(_fraction(slack, ds), _fraction(dual, dl)))
        # The ball Hessian is exact; a residual merit handles its nonlinear
        # primal equations during the numerical solve, before motion trials.
        merit = np.linalg.norm(np.r_[stationarity, primal, slack * dual])
        for _ in range(20):
            x, s, l = direction + alpha * dx, slack + alpha * ds, dual + alpha * dl
            f, j = domain.evaluate(x)
            rd = band_product(physical, x) + linear + j.T @ l
            new_merit = np.linalg.norm(np.r_[rd, f + s, s * l])
            if np.isfinite(new_merit) and new_merit <= merit * (1 - 1e-4 * alpha):
                direction, slack, dual = x, s, l
                break
            alpha *= .5
        else:
            status = 'residual_stall'
            break
    diagnostics.setdefault('ip_trace', []).append(trace)
    diagnostics['ip_iterations'] = diagnostics.get('ip_iterations', 0) + len(trace)
    diagnostics['ip_excess'] = float(domain.evaluate(direction)[0].max(initial=0.))
    diagnostics['ip_seconds'] = diagnostics.get('ip_seconds', 0.) + time.monotonic() - started
    diagnostics.setdefault('ip_statuses', []).append(status)
    warm['ip_dual'] = dual.copy()
    warm['ip_cost_scale'] = scale_cost
    if not np.isfinite(direction).all():
        raise ActiveSetFailure('nonfinite primal-dual KKT direction')
    return direction
