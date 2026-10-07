"""One SQP subproblem of the local repair (``local_repair``).

The same program family as ``repair_step.solve_step``: every immutable repair
limit and guard linearized at ``poses`` (``RepairLimits.constraints``),
original-mesh witness rows that keep each collision key within its protected
depth, and one epigraph per penetrating key of the window's bodies. Further
rows: cutting planes linearized at rejected trial poses (witnesses the current
linearization did not contain), exact linear chatter rows, optional elastic
chatter rows with penalized slack, and an optional per-coordinate trust box.
The rows of a guard linearized at a slightly infeasible pose ask the step to
remove that infeasibility, so successive subproblems correct their own
second-order remainders.
"""

import time

import numpy as np
from scipy import sparse

ACCEPTED = {"Solved", "AlmostSolved", "InsufficientProgress", "MaxIterations"}
# Linearized witnesses stay this far inside their key's protected depth (a
# clear witness keeps this much clearance) unless already closer: a curved or
# switching witness then lands inside its budget instead of micrometres past it.
CLEARANCE_MARGIN_M = 3e-4
# Weight of elastic slack (per radian) relative to the unit-scaled step terms.
ELASTIC_WEIGHT = 1e4


def witness_lower(phi, budgets, offset_rows):
    """Lower bound on J d for each witness row (``solve_step`` plus the margin).

    The linearized depth of a witness may reach its key's protected depth
    less the margin, or stay where it is if it is already deeper than that;
    for a clear witness this keeps ``min(margin, clearance)`` of clearance.
    """
    floor = np.maximum(budgets - CLEARANCE_MARGIN_M, -phi)
    return -floor - phi - offset_rows


def cut_lower(matrix, value, budget, displacement):
    """Cutting plane of a witness linearized at a trial ``displacement`` away.

    Its linearized depth must come back to the protected depth less the
    margin, or stay at its first-order depth at the current pose if that is
    already deeper (keeping the zero step feasible).
    """
    shifted = matrix @ displacement
    floor = np.maximum(budget - CLEARANCE_MARGIN_M, -(value - shifted))
    return -floor - value + shifted


def _settings():
    import clarabel

    value = clarabel.DefaultSettings()
    value.verbose = False
    value.max_threads = 1
    # Every step is corrected and checked exactly afterwards; iterations past
    # this cap only polish a solution the correction would move anyway.
    value.max_iter = 60
    value.tol_gap_abs = value.tol_gap_rel = value.tol_feas = 1e-9
    return value


def solve(limits, poses, projection, rows, keys, sigmas, *, local_angle, cuts=(), trust=None,
          offset=None, linear=(), elastic=None, quadratic=None):
    """Projected joint step of one window, or (None, status, info).

    ``rows`` is (matrix, phi, budgets, key_of_row) of the current witnesses
    (Jacobians at the origin of ``offset``); ``offset`` is the flattened joint
    displacement from that origin to ``poses`` (zero when they coincide).
    ``keys`` are the penetrating keys entering the objective with their
    epigraph ``sigmas``. ``cuts`` are (matrix, phi, budget, displacement)
    linearized at earlier trial poses, displacement measured from ``poses``.
    ``linear`` are extra hard (matrix, lower, upper) rows on the flattened
    step; ``elastic`` is one (matrix, lower, upper) block whose violation is
    allowed at a linear slack cost.
    """
    import clarabel

    joint_variables = projection.shape[1]
    if not joint_variables:
        return None, "no_local_dofs", {}
    if not keys and elastic is None:
        return None, "no_active_defect", {}
    started = time.monotonic()
    constraints = limits.constraints(poses, projection=projection)
    built = time.monotonic()
    collision, phi, budgets, key_of_row = rows
    shift_origin = np.zeros(collision.shape[1]) if offset is None else offset
    constraints.add(collision, witness_lower(phi, budgets, collision @ shift_origin), np.inf)
    for matrix, value, budget, displacement in cuts:
        constraints.add(matrix, cut_lower(matrix, value, budget, displacement), np.inf)
    for matrix, lower, upper in linear:
        constraints.add(matrix, lower, upper)
    lookup = {key: index for index, key in enumerate(keys)}
    selected = np.array([i for i, key in enumerate(key_of_row) if key in lookup], dtype=int)
    group = np.array([lookup[key_of_row[i]] for i in selected], dtype=int)
    count = len(keys)
    slack_matrix = None
    slacks = 0
    if elastic is not None:
        slack_matrix = sparse.csr_matrix(elastic[0]) @ projection
        slacks = slack_matrix.shape[0]
    variables = joint_variables + count + slacks
    shift = (poses[:, limits.coords.qi] - limits.entry[:, limits.coords.qi]).ravel()
    joint_hessian = projection.T @ projection / local_angle**2
    joint_gradient = projection.T @ shift / local_angle**2
    # Soft quadratic terms on the projected step (S x + c), e.g. contact slip;
    # ``quadratic`` is called after the limits have evaluated ``poses``.
    extra = None if quadratic is None else quadratic()
    if extra is not None:
        matrix_s, offset_s = extra
        joint_hessian = joint_hessian + (matrix_s.T @ matrix_s)
        joint_gradient = joint_gradient + matrix_s.T @ offset_s
    weights = 1. / np.asarray(sigmas, float)**2 if count else np.zeros(0)
    # Slack is linear in the objective; a tiny curvature keeps it unique.
    hessian = sparse.block_diag((joint_hessian, sparse.diags(weights), sparse.eye(slacks) * 1e-6),
                                format="csc")
    gradient = np.r_[joint_gradient, np.zeros(count), np.full(slacks, ELASTIC_WEIGHT)]
    scale = 1 / np.sqrt(np.maximum(hessian.diagonal(), 1e-12))
    scale[joint_variables + count:] = 1.
    diagonal = sparse.diags(scale)
    joint_diagonal = sparse.diags(scale[:joint_variables])

    def widen(matrix):
        """Joint-step rows embedded in the full variable vector, scaled."""
        return sparse.hstack((matrix, sparse.csc_matrix((matrix.shape[0], count + slacks))),
                             format="csc") @ diagonal

    a, lower, upper = constraints.packed()
    upper_mask, lower_mask = np.isfinite(upper), np.isfinite(lower)
    constrained = widen(a @ projection)
    constrained.eliminate_zeros()
    variable = np.asarray(constrained.getnnz(axis=1)).ravel() > 0
    # A row without window variables keeps its current value whatever the
    # step; one the accepted pose already exceeds within the nonlinear check's
    # tolerance cannot make this window infeasible. Exact acceptance still
    # checks every row.
    constant_violations = int(np.sum(~variable & ((upper < 0) | (lower > 0))))
    upper_mask &= variable
    lower_mask &= variable
    blocks = [sparse.vstack([constrained[upper_mask], -constrained[lower_mask]], format="csc")]
    targets = [np.r_[upper[upper_mask], -lower[lower_mask]]]
    if count:
        epigraph = sparse.csc_matrix((-np.ones(len(selected)), (np.arange(len(selected)), group)),
                                     shape=(len(selected), count))
        current = collision[selected] @ shift_origin
        blocks.append(sparse.hstack((-collision[selected] @ projection, epigraph,
                                     sparse.csc_matrix((len(selected), slacks))), format="csc") @ diagonal)
        targets.append(phi[selected] + current)
        blocks.append(sparse.hstack((sparse.csc_matrix((count, joint_variables)), -sparse.eye(count),
                                     sparse.csc_matrix((count, slacks))), format="csc") @ diagonal)
        targets.append(np.zeros(count))
    if slacks:
        _, low, high = elastic
        identity = sparse.eye(slacks, format="csc")
        rows_up = sparse.hstack((slack_matrix, sparse.csc_matrix((slacks, count)), -identity), format="csc")
        rows_down = sparse.hstack((-slack_matrix, sparse.csc_matrix((slacks, count)), -identity), format="csc")
        blocks += [rows_up @ diagonal, rows_down @ diagonal,
                   sparse.hstack((sparse.csc_matrix((slacks, joint_variables + count)), -identity),
                                 format="csc") @ diagonal]
        targets += [high, -low, np.zeros(slacks)]
    if trust is not None:
        box = widen(sparse.eye(joint_variables, format="csc"))
        blocks += [box, -box]
        targets += [np.full(joint_variables, trust), np.full(joint_variables, trust)]
    nonneg = sum(block.shape[0] for block in blocks)
    cones = [clarabel.NonnegativeConeT(nonneg)]
    for jac, residual, allowed in constraints.cones():
        dimension = residual.shape[1]
        jac = (jac @ projection).tocsr()
        jac.eliminate_zeros()
        variable = np.asarray(jac.getnnz(axis=1)).reshape(-1, dimension).sum(axis=1) > 0
        constant_violations += int(np.sum(~variable & (np.linalg.norm(residual, axis=1) > allowed + 1e-12)))
        keep = variable
        jac, residual, allowed = jac[np.repeat(keep, dimension)], residual[keep], allowed[keep]
        norm_count = residual.shape[0]
        if not norm_count:
            continue
        normalization = np.maximum(allowed, 1e-3)
        jac = (sparse.diags(np.repeat(1 / normalization, dimension)) @ jac @ joint_diagonal).tocoo()
        cone_rows = jac.row + jac.row // dimension + 1
        blocks.append(sparse.csc_matrix((-jac.data, (cone_rows, jac.col)),
                                        shape=(norm_count * (dimension + 1), variables)))
        targets.append(np.c_[allowed / normalization, residual / normalization[:, None]].ravel())
        cones.extend(clarabel.SecondOrderConeT(dimension + 1) for _ in range(norm_count))
    program_matrix, program_rhs = sparse.vstack(blocks, format="csc"), np.concatenate(targets)
    quadratic = sparse.triu(diagonal @ hessian @ diagonal, format="csc")
    assembled = time.monotonic()
    # A complete interior-point solve: on these window programs a working set
    # needs several cold restarts and was slower (out/unified_local_20261006/bench).
    solution = clarabel.DefaultSolver(quadratic, gradient * scale, program_matrix, program_rhs,
                                      cones, _settings()).solve()
    status = str(solution.status)
    info = dict(variables=variables, rows=int(program_matrix.shape[0]), keys=count, slacks=slacks,
                constant_violations=constant_violations,
                cuts=len(cuts), iterations=int(getattr(solution, "iterations", -1)),
                cones=len(cones), constraint_seconds=built - started,
                assembly_seconds=assembled - built, solve_seconds=time.monotonic() - assembled)
    if status not in ACCEPTED or solution.x is None or not np.isfinite(solution.x).all():
        return None, status, info
    x = np.asarray(solution.x) * scale
    step = (projection @ x[:joint_variables]).reshape(limits.frames, limits.width)
    info["predicted_depths"] = x[joint_variables:joint_variables + count]
    info["slack"] = x[joint_variables + count:]
    return step, status, info
