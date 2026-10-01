"""Bounded joint correction minimizing each collision group's maximum depth.

Witness multiplicity changes the feasible set, never a collision's weight.
The conic solution remains a proposal for exact nonlinear acceptance.
"""

import numpy as np
from scipy import sparse


def solve_step(limits, poses, geometry, measurement, baseline, projection, *,
               collision_sigma, local_angle):
    import clarabel

    variables = projection.shape[1]
    if not variables:
        return None, "no_local_dofs"
    constraints = limits.constraints(poses)
    collision, phi = geometry.linear_rows(measurement)
    budgets = np.array([baseline["depths"].get(key, 0.) for key, _ in measurement["rows"]])
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
    cones = [clarabel.NonnegativeConeT(len(rhs) + len(selected) + count)]
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
    matrix, rhs = sparse.vstack(blocks, format="csc"), np.concatenate(targets)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 100
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9
    solver = clarabel.DefaultSolver(sparse.triu(diagonal @ hessian @ diagonal, format="csc"),
                                    gradient * scale, matrix, rhs,
                                    cones, settings)
    solution = solver.solve()
    status = str(solution.status)
    # Numerical termination is only a proposal. Exact nonlinear limits and
    # original geometry below decide acceptance, including after backtracking.
    if status not in {"Solved", "AlmostSolved", "InsufficientProgress", "MaxIterations"} or not np.isfinite(solution.x).all():
        return None, status
    return (projection @ (np.asarray(solution.x[:joint_variables]) * scale[:joint_variables])).reshape(limits.frames, limits.width), status


