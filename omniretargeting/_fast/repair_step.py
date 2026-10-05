"""Bounded joint correction minimizing each collision group's maximum depth.

Witness multiplicity changes the feasible set, never a collision's weight.
The conic solution remains a proposal for exact nonlinear acceptance.
"""

import numpy as np
from scipy import sparse

from .runtime_options import OPTIONS
from .window_qp import WINDOW_FRAMES, solve_standard


def solve_step(limits, poses, geometry, measurement, baseline, projection, *,
               collision_sigma, local_angle):
    import clarabel

    variables = projection.shape[1]
    if not variables:
        return None, "no_local_dofs"
    constraints = limits.constraints(poses, projection=projection)
    collision, phi = geometry.linear_rows(measurement)
    budgets = np.array([baseline["depths"].get(key, 0.) for key, _ in measurement["rows"]])
    if OPTIONS.abl_repair_ratchet:
        from .repair_ablation import collision_budgets
        budgets = collision_budgets(limits.objective, measurement, baseline)
    # Keep the nonlinear guard's tiny tolerance in reserve. Spending it in
    # the linear subproblem leaves no room for curvature and roundoff.
    # Accepted roundoff on a different fixed chain must leave zero feasible.
    # Final acceptance still uses the original protected depths + REGRESSION.
    constraints.add(collision, -np.maximum(budgets, -phi) - phi, np.inf)
    # One epigraph per collision group: all witnesses constrain its maximum
    # depth. Duplicating samples never duplicates objective weight, and a flat
    # patch cannot improve by exchanging its deepest corner for another one.
    keys = []
    for key, depth in measurement["depths"].items():
        if depth <= 0:
            continue
        if limits.movable_body is not None:
            geoms = key[2:] if key[1] == "self" else key[-1:]
            if not any(geometry.model.body_weldid[geometry.model.geom_bodyid[g]] == limits.movable_body
                       for g in geoms):
                continue
        keys.append(key)
    if not keys:
        return None, "no_active_collision"
    lookup = {key: index for index, key in enumerate(keys)}
    selected = np.array([i for i, (key, _) in enumerate(measurement["rows"]) if key in lookup], dtype=int)
    group = np.array([lookup[measurement["rows"][i][0]] for i in selected], dtype=int)
    joint_variables = variables
    count = len(keys)
    variables += count
    shift = (poses[:, limits.coords.qi] - limits.entry[:, limits.coords.qi]).ravel()
    joint_hessian = projection.T @ projection / local_angle**2
    hessian = sparse.block_diag((joint_hessian, sparse.eye(count) / collision_sigma**2), format="csc")
    gradient = np.r_[projection.T @ shift / local_angle**2, np.zeros(count)]
    scale = 1 / np.sqrt(np.maximum(hessian.diagonal(), 1e-12))
    diagonal = sparse.diags(scale)
    joint_diagonal = sparse.diags(scale[:joint_variables])
    a, lower, upper = constraints.packed()
    upper_mask, lower_mask = np.isfinite(upper), np.isfinite(lower)
    constrained = sparse.hstack((a @ projection, sparse.csc_matrix((a.shape[0], count))), format="csc") @ diagonal
    constrained.eliminate_zeros()
    variable = np.asarray(constrained.getnnz(axis=1)).ravel() > 0
    upper_mask &= variable | (upper < 0)
    lower_mask &= variable | (lower > 0)
    matrix = sparse.vstack([constrained[upper_mask], -constrained[lower_mask]], format="csc")
    rhs = np.r_[upper[upper_mask], -lower[lower_mask]]
    epigraph = sparse.csc_matrix((-np.ones(len(selected)), (np.arange(len(selected)), group)),
                                  shape=(len(selected), count))
    maximum_rows = sparse.hstack((-collision[selected] @ projection, epigraph), format="csc") @ diagonal
    positive_rows = sparse.hstack((sparse.csc_matrix((count, joint_variables)), -sparse.eye(count)),
                                 format="csc") @ diagonal
    blocks, targets = [matrix, maximum_rows, positive_rows], [rhs, phi[selected], np.zeros(count)]
    nonneg = len(rhs) + len(selected) + count
    cones = [clarabel.NonnegativeConeT(nonneg)]
    soc_sizes = []
    for jac, residual, allowed in constraints.cones():
        dimension = residual.shape[1]
        jac = (jac @ projection).tocsr()
        jac.eliminate_zeros()
        variable = np.asarray(jac.getnnz(axis=1)).reshape(-1, dimension).sum(axis=1) > 0
        constant_valid = np.linalg.norm(residual, axis=1) <= allowed + 1e-12
        keep = variable | ~constant_valid
        jac, residual, allowed = jac[np.repeat(keep, dimension)], residual[keep], allowed[keep]
        norm_count, dimension = residual.shape
        if not norm_count:
            continue
        # Normalize every cone by its unchanged geometric budget.
        normalization = np.maximum(allowed, 1e-3)
        jac = (sparse.diags(np.repeat(1 / normalization, dimension)) @ jac @ joint_diagonal).tocoo()
        rows = jac.row + jac.row // dimension + 1
        blocks.append(sparse.csc_matrix((-jac.data, (rows, jac.col)),
                                        shape=(norm_count * (dimension + 1), variables)))
        targets.append(np.c_[allowed / normalization, residual / normalization[:, None]].ravel())
        cones.extend(clarabel.SecondOrderConeT(dimension + 1) for _ in range(norm_count))
        soc_sizes += [dimension + 1] * norm_count
    matrix, rhs = sparse.vstack(blocks, format="csc"), np.concatenate(targets)

    def settings():
        value = clarabel.DefaultSettings()
        value.verbose = False
        value.max_iter = 100
        value.tol_gap_abs = value.tol_gap_rel = value.tol_feas = 1e-9
        return value

    # Numerical termination is only a proposal. Exact nonlinear limits and
    # original geometry below decide acceptance, including after backtracking.
    accepted = {"Solved", "AlmostSolved", "InsufficientProgress", "MaxIterations"}
    quadratic = sparse.triu(diagonal @ hessian @ diagonal, format="csc")
    columns_per_frame = joint_variables // limits.frames
    if (OPTIONS.window_qp and limits.frames >= 2 * WINDOW_FRAMES
            and columns_per_frame * limits.frames == joint_variables):
        # Temporal block coordinates (``window_qp``): joint columns are
        # frame-major; each epigraph belongs to its collision key's frame.
        frame_of = np.r_[np.arange(joint_variables) // columns_per_frame, [key[0] for key in keys]]
        x = solve_standard(quadratic, gradient * scale, matrix, rhs, nonneg, soc_sizes, frame_of,
                           limits.frames, settings=settings, accepted=accepted)
        if x is not None:
            step = projection @ (x[:joint_variables] * scale[:joint_variables])
            return step.reshape(limits.frames, limits.width), "Windowed"
    from . import repair_qoco
    if repair_qoco.QOCO_REPAIR_QP:
        from types import SimpleNamespace
        x = repair_qoco.solve(dict(P=quadratic, c=gradient * scale, A=matrix, b=rhs,
                                  dims=SimpleNamespace(zero=0, nonneg=nonneg, soc=soc_sizes)))
        if x is not None:
            return (projection @ (x[:joint_variables] * scale[:joint_variables])).reshape(limits.frames, limits.width), "QOCO_SOLVED"
    solution = None
    if OPTIONS.repair_working_set:
        # Same optimum on the few rows that can bind (``working_set``).
        from .working_set import solve as solve_working_set
        solution = solve_working_set(quadratic, gradient * scale, matrix, rhs, cones, settings)
    if solution is None and OPTIONS.gpu_conic:
        from .gpu_conic import dimensions, solve as solve_gpu
        solution = solve_gpu(quadratic, gradient * scale, matrix, rhs,
                             dimensions(cones), context='repair_full')
    if solution is None:
        solution = clarabel.DefaultSolver(quadratic, gradient * scale, matrix, rhs, cones, settings()).solve()
    status = str(solution.status)
    if status not in accepted or solution.x is None or not np.isfinite(solution.x).all():
        return None, status
    return (projection @ (np.asarray(solution.x[:joint_variables]) * scale[:joint_variables])).reshape(limits.frames, limits.width), status
