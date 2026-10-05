"""Find a useful LM direction inside the fixed reference layout and guard bounds.

Backtracking an unconstrained direction cannot remove its outward component at
an already active bound. Reuse the guarded solver's local cones for a
fallback direction; the fast solver still checks every proposal nonlinearly.
"""

import cvxpy as cp
import numpy as np
from scipy import sparse

from ..conic_solver import gpu_enabled, solve_conic
from .._sequence.restoration import restore_candidate
from .rows import _merge_identical
from .runtime_options import OPTIONS
from .window_qp import WINDOW_FRAMES, solve_windows


def _guard_constraints(guards, poses, flat):
    """Every guard's constraints; identical linear rows of several guards
    (e.g. two continuity guards on one fixed reference) become one row with
    the intersected interval, scaled by its half-width as each guard scales
    its own rows. The feasible set is unchanged."""
    import inspect

    rows, constraints = [], []
    for guard in guards:
        if "merge_rows" in inspect.signature(guard.constraints).parameters:
            constraints += guard.constraints(poses, flat, merge_rows=rows)
        else:
            constraints += guard.constraints(poses, flat)
    if rows:
        matrix, lower, upper = _merge_identical(
            sparse.vstack([m for m, _, _ in rows], format="csr"),
            np.concatenate([lo for _, lo, _ in rows]), np.concatenate([hi for _, _, hi in rows]))
        center, half = (upper + lower) / 2, (upper - lower) / 2
        residual = sparse.diags(1 / half) @ matrix @ flat - center / half
        constraints += [residual <= 1., residual >= -1.]
    return constraints


def _pin_root(objective, poses, lower, upper):
    coords = objective.coords
    fixed = getattr(objective, 'fixed_root', None)
    if fixed is None:
        return
    target = poses.copy()
    target[:, coords.root_q:coords.root_q + 7] = fixed
    displacement = coords.difference(poses, target)
    columns = [coords.inverse[d] for d in range(coords.root_d, coords.root_d + 6)
               if d in coords.inverse]
    lower[:, columns] = upper[:, columns] = displacement[:, columns]


# Solve restorations by exact constraint generation (same optimum, 3-20x
# faster; ``conic_solver.solve_conic_working_set``). Its interior-point
# solutions differ from the full solve at the 1e-7 tolerance level, which once
# sent acquisition to other local results; with overshooting conic steps
# shortened and restored instead of ending the pass (``solver._shorten_conic``)
# restorations are frequent and acquisition no longer stops on such a path.
WORKING_SET_RESTORATION = True


def _canonical_window(objective):
    from .legacy_window_qp import WindowObjective
    # The assembler reproduces WindowGuard.constraints specifically. Other
    # objectives and optional conic backends keep their established path.
    return isinstance(objective, WindowObjective) and not gpu_enabled() and not OPTIONS.window_qp


class _ShapeRestoration:
    def __init__(self, objective):
        self.objective = objective
        self.shape = objective.source_shape
        self.tasks = self.shape.pose.tasks

    def accepts(self, poses, frames=None):
        return self.shape.accepts(poses) and all(
            guard.accepts(poses) for guard in self.objective.candidate_guards)

    def restoration_constraints(self, poses, flat, lower, upper, priority_bounds, frames):
        coords = self.objective.coords
        lower[:, coords.scalar_cols] = np.maximum(
            lower[:, coords.scalar_cols], self.objective.lower - poses[:, coords.qi])
        upper[:, coords.scalar_cols] = np.minimum(
            upper[:, coords.scalar_cols], self.objective.upper - poses[:, coords.qi])
        _pin_root(self.objective, poses, lower, upper)
        constraints = [flat >= lower.ravel(), flat <= upper.ravel()]
        box = np.maximum(np.abs(lower), np.abs(upper))
        shape = self.shape.constraints(poses, flat, np.where(np.isfinite(box), box, 1e3))
        return constraints + shape + _guard_constraints(self.objective.candidate_guards, poses, flat)


def restore_shape_step(objective, origin, step):
    """Correct curvature error without changing the source reference or bounds."""
    if _canonical_window(objective):
        from .window_canonical import restore_step
        return restore_step(objective, origin, step)
    from .whole_canonical import supported
    if supported(objective):
        from .whole_canonical import restore_step
        return restore_step(objective, origin, step)
    coords = objective.coords
    candidate = objective.project(coords.integrate(origin, step.reshape(len(origin), -1), 1.))
    lower, upper = coords.bounds(origin, objective.cfg["angular_step_rad"],
                                 objective.cfg["translation_step_m"])
    restored = restore_candidate(_ShapeRestoration(objective), origin, candidate,
                                 lower, upper, {}, working_set=WORKING_SET_RESTORATION)
    # The accepted state must be what the next step's projection keeps.
    return None if restored is None else objective.project(restored)


def _qoco_proposal(problem, flat):
    """Whole-trajectory proposal by the window path's QOCO solve, or None.

    Same program, same canonical data and the same 1e-4 QOCO tolerances as
    ``window_qoco`` (default for windows since 2026-10-03). Every nonlinear
    guard and the objective decrease still decide acceptance; any status
    other than solved returns None and the caller keeps Clarabel.
    """
    from .window_qoco import solve_proposal
    data, _, inverse = problem.get_problem_data(cp.CLARABEL)
    offsets = [d.var_offsets for d in inverse if hasattr(d, "var_offsets")]
    if not offsets or offsets[-1].get(flat.id) != 0:
        return None
    data = dict(data)
    for key in ("P", "A"):
        # QOCO's CSC binding takes 32-bit indices; CVXPY may emit 64-bit ones.
        matrix = sparse.csc_matrix(data[key])
        matrix.indices, matrix.indptr = matrix.indices.astype(np.int32), matrix.indptr.astype(np.int32)
        data[key] = matrix
    try:
        return solve_proposal(data, flat.size)
    except (ImportError, RuntimeError, ValueError, TypeError, ArithmeticError):
        return None


def bounded_shape_step(objective, poses, system, gradient, diagonal, damping):
    shape = objective.source_shape
    coords = objective.coords
    count = len(poses) * coords.width
    bands = [system[k, :count - k] for k in range(len(system))]
    lower = sparse.diags(bands, -np.arange(len(bands)), shape=(count, count), format="csc")
    matrix = lower + lower.T - sparse.diags(system[0])
    matrix += sparse.diags(damping * diagonal + 1e-9)
    lo, hi = coords.bounds(poses, objective.cfg["angular_step_rad"],
                           objective.cfg["translation_step_m"])
    _pin_root(objective, poses, lo, hi)
    lo[:, coords.scalar_cols] = np.maximum(lo[:, coords.scalar_cols],
                                          objective.lower - poses[:, coords.qi])
    hi[:, coords.scalar_cols] = np.minimum(hi[:, coords.scalar_cols],
                                          objective.upper - poses[:, coords.qi])
    scale = max(float(np.linalg.norm(gradient, ord=np.inf)), 1.)
    if _canonical_window(objective):
        from .window_canonical import canonical_data, solve_data
        data = canonical_data(objective, poses, lo, hi, matrix * (2 * (1 / scale)),
                              (2 * gradient) * (1 / scale))
        from .window_qoco import WINDOW_QOCO_PROPOSALS, solve_proposal
        if WINDOW_QOCO_PROPOSALS:
            step = solve_proposal(data, count)
            if step is not None:
                return step
        return solve_data(data, count)
    from .whole_canonical import supported
    if supported(objective) and not OPTIONS.whole_qoco_proposals:
        from .whole_canonical import canonical_data
        from .window_canonical import solve_data
        data = canonical_data(objective, poses, lo, hi, matrix * (2 * (1 / scale)),
                              (2 * gradient) * (1 / scale))
        return solve_data(data, count)
    flat = cp.Variable(count)
    constraints = [flat >= lo.ravel(), flat <= hi.ravel()]
    # Both frozen reference layouts (seed and acquired reference) apply;
    # components no step inside this box can flip are omitted.
    box = np.maximum(np.abs(lo), np.abs(hi))
    constraints += shape.constraints(poses, flat, np.where(np.isfinite(box), box, 1e3))
    constraints += _guard_constraints(objective.candidate_guards, poses, flat)
    problem = cp.Problem(cp.Minimize((cp.quad_form(flat, cp.psd_wrap(matrix)) + 2 * gradient @ flat) / scale),
                         constraints)
    if OPTIONS.window_qp and len(poses) >= 2 * WINDOW_FRAMES:
        # Temporal block-coordinate step: every linearized bound stays exact.
        step = solve_windows(problem, flat, len(poses), coords.width)
        if step is not None:
            return step
    if OPTIONS.whole_qoco_proposals:
        step = _qoco_proposal(problem, flat)
        if step is not None:
            return step
    try:
        solve_conic(problem, solver=cp.CLARABEL, direct_solve_method="qdldl")
    except cp.error.SolverError:
        return None
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or flat.value is None:
        return None
    step = np.asarray(flat.value)
    return step if np.isfinite(step).all() else None
