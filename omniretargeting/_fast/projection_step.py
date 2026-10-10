"""Sparse linear QP for the high-priority continuity of full-body corrections."""
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d

from .continuity import CorrectionContinuity
from . import deadline_state


def constrained_direction(projection, current, hessian, rhs):
    """Optimize in normalized tangents; exact nonlinear guards still follow.

    A line search cannot fix a direction pointing out of an active chatter
    boundary. Put those bounds into the step, rather than repeatedly shrinking
    an infeasible direction or stiffening its penalty.
    """
    p, co = projection, projection.coords
    frames, width = p.frames, p.width
    matrices, lows, highs = [], [], []

    def add(matrix, lower, upper):
        matrix = sparse.csc_matrix(matrix)
        lower, upper = np.broadcast_to(lower, (matrix.shape[0],)), np.broadcast_to(upper, (matrix.shape[0],))
        # Identical row scaling of bounds and matrix, not relaxed tolerances.
        scale = 1. / np.maximum(np.maximum(np.abs(lower), np.abs(upper)), 1e-3)
        scale[~np.isfinite(scale)] = 1.
        scale[scale == 0] = 1.
        matrices.append(sparse.diags(scale) @ matrix)
        lows.append(lower * scale)
        highs.append(upper * scale)

    lo, hi = co.bounds(current, .15, .01)
    lo[:, co.scalar_cols] = np.maximum(lo[:, co.scalar_cols], p.entry[:, co.qi] - .4 - current[:, co.qi])
    hi[:, co.scalar_cols] = np.minimum(hi[:, co.scalar_cols], p.entry[:, co.qi] + .4 - current[:, co.qi])
    add(sparse.eye(frames * width), lo.ravel() / p.flat_scale, hi.ravel() / p.flat_scale)
    parts = {'joints': (current[:, co.qi], sparse.kron(sparse.eye(frames), co.selector))}
    if co.root_q is not None:
        selector = np.zeros((3, width))
        for axis in range(3):
            if co.root_d + axis in co.inverse:
                selector[axis, co.inverse[co.root_d + axis]] = 1.
        parts['root_translation'] = (current[:, co.root_q:co.root_q + 3],
                                     sparse.kron(sparse.eye(frames), selector))
        angular = CorrectionContinuity(co, p.obj.reference, chatter=True)
        parts['root_rotation'] = angular.linearizations(current)['root_rotation_order_1']
    for name, (values, jac) in parts.items():
        length, dimension = values.shape
        kernel = sparse.csr_matrix(np.eye(length) - gaussian_filter1d(np.eye(length), 2., axis=0, mode='nearest'))
        operator = sparse.kron(kernel, sparse.eye(dimension), format='csr')
        residual = operator @ (values - p.reference_signals[name]).ravel()
        cap = p.allowed[name].ravel() + 5e-8
        add(operator @ jac @ p.scale, -cap - residual, cap - residual)
    for guard in p.velocity_guards:
        linear, _ = guard.repair_bounds(current)
        for matrix, lower, upper in linear:
            add(matrix @ p.scale, lower, upper)
    from .projection_mechanics import mechanical_rows
    if p.direct:
        mechanical, lower, upper = mechanical_rows(p.mechanics, co, current)
        if mechanical.shape[0]:
            add(mechanical @ p.scale, lower, upper)
    if not p.direct:
        return _operator_step(p, hessian, rhs, matrices, lows, highs)
    matrix = sparse.vstack(matrices, format='csc')
    lower, upper = np.concatenate(lows), np.concatenate(highs)
    if frames >= 600 and not p.focus.all():
        from .projection_windows import solve_windows
        return solve_windows(p, hessian.tocsc(), rhs, matrix, lower, upper, _direct_step)
    direction, p.step_status = _direct_step(hessian, rhs, matrix, lower, upper)
    return None if direction is None else direction.reshape(frames, width) * p.scales


def _direct_step(hessian, rhs, matrix, lower, upper):
    import clarabel
    state = deadline_state.state()
    seconds = 10. if state is None else min(10., max(.01, state.deadline.remaining()))
    finite_lo, finite_hi = np.isfinite(lower), np.isfinite(upper)
    bound = np.r_[upper[finite_hi], -lower[finite_lo]]
    constraint = sparse.vstack([matrix[finite_hi], -matrix[finite_lo]], format='csc')
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 80
    settings.time_limit = seconds
    settings.tol_feas = 1e-9
    settings.tol_gap_abs = settings.tol_gap_rel = 1e-7
    solver = clarabel.DefaultSolver(sparse.triu(hessian, format='csc'), -rhs,
                                   constraint, bound, [clarabel.NonnegativeConeT(len(bound))], settings)
    solved = solver.solve()
    record = dict(status=str(solved.status), iterations=int(solved.iterations),
                         seconds=float(solved.solve_time), primal_residual=float(solved.r_prim),
                         dual_residual=float(solved.r_dual))
    if str(solved.status) in {'PrimalInfeasible', 'DualInfeasible',
                              'AlmostPrimalInfeasible', 'AlmostDualInfeasible', 'NumericalError'}:
        return None, record
    if solved.x is None or not np.isfinite(solved.x).all():
        return None, record
    return np.asarray(solved.x), record


def _operator_step(projection, hessian, rhs, matrices, lows, highs):
    """Fast proposal; exact guards decide, and a stalled pass refreshes once."""
    import osqp
    state = deadline_state.state()
    seconds = 5. if state is None else min(5., max(.01, state.deadline.remaining()))
    solver = osqp.OSQP()
    solver.setup(P=sparse.triu(hessian, format='csc'), q=-rhs,
                 A=sparse.vstack(matrices, format='csc'), l=np.concatenate(lows), u=np.concatenate(highs),
                 eps_abs=1e-6, eps_rel=1e-6, max_iter=4000, time_limit=seconds,
                 polishing=True, verbose=False)
    solved = solver.solve(raise_error=False)
    projection.step_status = dict(status=solved.info.status, iterations=int(solved.info.iter),
                                  seconds=float(solved.info.run_time),
                                  primal_residual=float(solved.info.prim_res),
                                  dual_residual=float(solved.info.dual_res))
    if 'infeasible' in solved.info.status.lower() or solved.x is None or not np.isfinite(solved.x).all():
        return None
    return solved.x.reshape(projection.frames, projection.width) * projection.scales
