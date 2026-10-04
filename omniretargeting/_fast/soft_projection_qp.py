"""Sparse closest-point QPs/SOCPs; no CVXPY graph construction or budget changes."""

import time

import clarabel
import numpy as np
from scipy import sparse


class ProjectionFailure(RuntimeError):
    pass


def closest(target, matrix, lower, upper, *, balls=(), weights=None, report=None,
            variable_scale=None, quadratic=None, linear=None):
    """Weighted least squares with intervals and intact linearized vector balls."""
    tick = time.monotonic()
    target = np.asarray(target).ravel()
    weights = np.ones(len(target)) if weights is None else np.asarray(weights).ravel()
    matrix = sparse.csr_matrix(matrix)
    lower, upper = np.asarray(lower), np.asarray(upper)
    original_matrix, original_lower, original_upper = matrix, lower, upper
    scale = np.ones(len(target)) if variable_scale is None else np.asarray(variable_scale).ravel()
    if quadratic is not None:
        # Shared root coordinates accumulate curvature from every frame.
        # Equilibrate those against per-frame joints before normalizing P;
        # fixed metre/radian scales leave orders-of-magnitude tiny pivots.
        scale = 1 / np.sqrt(np.maximum(quadratic.diagonal(),1e-18))
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError('Projection scales must be positive and finite')
    variables = sparse.diags(scale, format='csr')
    # Equivalent variable and row units, including one common unit per SOC.
    # Tiny derivative budgets must not share a solver unit with metre/radian
    # domains hundreds of thousands of times larger.
    matrix = matrix @ variables
    if variable_scale is not None:
        row_scale = np.maximum(np.asarray(abs(matrix).max(axis=1).toarray()).ravel(), 1e-12)
        matrix = sparse.diags(1 / row_scale) @ matrix
        lower, upper = lower / row_scale, upper / row_scale
    equality = np.isfinite(lower) & (lower == upper)
    lo, hi = np.isfinite(lower) & ~equality, np.isfinite(upper) & ~equality
    blocks, rhs, cones = [], [], []
    if equality.any():
        blocks.append(matrix[equality])
        rhs.append(upper[equality])
        cones.append(clarabel.ZeroConeT(int(equality.sum())))
    if lo.any() or hi.any():
        blocks.append(sparse.vstack((matrix[hi], -matrix[lo]), format='csr'))
        rhs.append(np.r_[upper[hi], -lower[lo]])
        cones.append(clarabel.NonnegativeConeT(int(hi.sum() + lo.sum())))
    for jacobian, residual, radius in balls:
        residual = np.asarray(residual)
        count, dim = residual.shape
        if not count:
            continue
        if variable_scale is not None:
            units = np.maximum(np.asarray(radius), 1e-7)
            jacobian = sparse.diags(np.repeat(1 / units, dim)) @ jacobian @ variables
            residual, radius = residual / units[:, None], radius / units
        jacobian = sparse.coo_matrix(jacobian)
        rows = jacobian.row // dim * (dim + 1) + 1 + jacobian.row % dim
        blocks.append(sparse.csr_matrix((-jacobian.data, (rows, jacobian.col)),
                                        shape=(count * (dim + 1), len(target))))
        rhs.append(np.column_stack((radius, residual)).ravel())
        cones.extend([clarabel.SecondOrderConeT(dim + 1) for _ in range(count)])
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 120
    # Clarabel's feasibility tolerance is relative to the whole RHS norm.
    # Large regions must still satisfy each original 1e-7/2e-7 guard slack.
    settings.tol_feas = 1e-11
    settings.reduced_tol_feas = 1e-10
    settings.tol_infeas_abs = settings.tol_infeas_rel = 1e-13
    settings.tol_gap_abs = settings.tol_gap_rel = 1e-9
    settings.max_threads = 1
    diagonal = weights * scale**2 if quadratic is None else quadratic.diagonal()*scale**2
    normalization = max(float(diagonal.max()), 1e-30)
    hessian = (sparse.diags(diagonal / normalization, format='csc') if quadratic is None else
               sparse.triu(variables @ quadratic @ variables / normalization, format='csc'))
    gradient = -weights * scale * target if linear is None else scale*np.asarray(linear)
    solver = clarabel.DefaultSolver(hessian, gradient / normalization,
        sparse.vstack(blocks, format='csc'), np.concatenate(rhs), cones, settings)
    result = solver.solve()
    point = scale * np.asarray(result.x)
    values = original_matrix @ point
    excess = float(np.maximum(original_lower-values, values-original_upper).max(initial=0.))
    for jacobian, residual, radius in balls:
        size = np.asarray(residual).shape
        actual = np.asarray(residual) + (jacobian @ point).reshape(size)
        excess = max(excess, float((np.linalg.norm(actual, axis=1)-radius).max(initial=0.)))
    # Tiny cones can produce a numerical-status exit with residuals smaller
    # than the proposal's roundoff reserves. Keep such a private SQP trial;
    # original nonlinear guards, domains and commit still own acceptance.
    # This is a numerical solve criterion, never a physical allowance.
    usable = np.isfinite(point).all() and excess <= 1e-8
    if report is not None:
        report['qp_solves'] = report.get('qp_solves', 0) + 1
        report['qp_seconds'] = report.get('qp_seconds', 0.) + time.monotonic() - tick
        report.setdefault('qp_statuses', []).append(str(result.status))
        report.setdefault('qp_diagnostics', []).append(dict(status=str(result.status),
            iterations=result.iterations, primal_residual=result.r_prim, dual_residual=result.r_dual,
            original_linear_excess=excess, usable_feasible_exit=usable))
    if (str(result.status) not in ('Solved', 'AlmostSolved') and not usable) or not np.isfinite(point).all():
        raise ProjectionFailure(str(result.status))
    return point


def family_program(families, columns, *, screen=False, active=None, curvature=None):
    """Original scalar slacks and vector radii, with half-roundoff reserves.

    Only contacts/layout are screened. Exact checks add newly binding faces
    in the next SQP round; all temporal constraints remain in every program.
    """
    linear, upper, balls = [], [], []
    for family in families:
        geometry = family.label[0] in ('AcquisitionNormalGuard', 'ReferenceShape')
        reserve = 5e-8 if geometry else 1e-7
        if family.label[0] == 'RootDomain':
            reserve = 1e-9
        if family.ball:
            radius = np.sqrt(np.maximum(family.slack + np.square(family.residual).sum(axis=1), 0.))
            gap = radius - np.linalg.norm(family.residual, axis=1)
        else:
            gap = family.slack
        keep = np.ones(len(gap), bool)
        if screen and geometry:
            keep = gap < 2 * reserve
            if active is not None:
                keep |= np.fromiter((family.identity(i) in active for i in range(len(gap))), bool)
                active.update(family.identity(int(i)) for i in np.flatnonzero(keep))
        if not keep.any():
            continue
        margins = np.full(len(gap), reserve)
        if curvature is not None:
            margins += np.fromiter((curvature.get(family.identity(i), 0.) for i in range(len(gap))), float)
            if family.ball:
                margins = np.minimum(margins, .05 * radius)
        if family.ball:
            dim = family.residual.shape[1]
            index = (np.flatnonzero(keep)[:, None] * dim + np.arange(dim)).ravel()
            jac = family.matrix[index][:, columns]
            movable = np.asarray(jac.getnnz(axis=1)).reshape(-1, dim).any(axis=1)
            if np.any(~movable & (gap[keep] < 0)):
                raise ProjectionFailure('binding ball has no free coordinates')
            index = (np.flatnonzero(movable)[:, None] * dim + np.arange(dim)).ravel()
            balls.append((jac[index], family.residual[keep][movable],
                          np.maximum((radius - margins)[keep][movable], 0.)))
        else:
            jac = family.matrix[keep][:, columns]
            movable = np.asarray(jac.getnnz(axis=1)).ravel() > 0
            if np.any(~movable & (gap[keep] < 0)):
                raise ProjectionFailure('binding scalar face has no free coordinates')
            linear.append(-jac[movable])
            upper.append((family.slack - margins)[keep][movable])
    size = len(columns)
    matrix = sparse.vstack(linear, format='csr') if linear else sparse.csr_matrix((0, size))
    bounds = np.concatenate(upper) if upper else np.empty(0)
    return matrix, np.full(len(bounds), -np.inf), bounds, balls
