"""Adaptive one-sided guard hinges in the physical banded LM model."""

import numpy as np
from scipy import sparse

from .barrier_assembly import equilibrated_solve, scatter
from .barrier_solver import _equalities
from .solver import _active_bounds, _banded_solve, _hold, DIAGONAL_FLOOR

HINGE_ROUNDS = 7
INITIAL_STIFFNESS = 100.


def step_boxes(objective, poses):
    """Original conic trust boxes intersected with actual scalar boxes."""
    lo, hi = objective.coords.bounds(poses, objective.cfg['angular_step_rad'],
                                    objective.cfg['translation_step_m'])
    cols = objective.coords.scalar_cols
    lo[:, cols] = np.maximum(lo[:, cols], objective.lower - poses[:, objective.coords.qi])
    hi[:, cols] = np.minimum(hi[:, cols], objective.upper - poses[:, objective.coords.qi])
    return lo.ravel(), hi.ravel()


def with_boxes(matrix, bounds, lower, upper):
    size = matrix.shape[1]
    lo, hi = np.flatnonzero(np.isfinite(lower)), np.flatnonzero(np.isfinite(upper))
    lower_matrix = sparse.csr_matrix((-np.ones(len(lo)), (np.arange(len(lo)), lo)), shape=(len(lo), size))
    upper_matrix = sparse.csr_matrix((np.ones(len(hi)), (np.arange(len(hi)), hi)), shape=(len(hi), size))
    return sparse.vstack([matrix, lower_matrix, upper_matrix], format='csr'), np.r_[bounds, -lower[lo], upper[hi]]


def hinge_directions(objective, poses, rows, families, physical, gradient, diagonal,
                     damping, diagnostics):
    """Yield successive directions after adding/escalating violated faces.

    Curvature/row scaling is numerical, derived from the physical diagonal.
    No row weight is a robot/body preference or a change to a hard bound.
    """
    lower, upper = step_boxes(objective, poses)
    held = np.unique(np.r_[_equalities(objective, poses), _active_bounds(objective, poses, gradient)])
    direction = np.zeros(len(gradient))
    weights = None
    active = None
    for round_index in range(HINGE_ROUNDS):
        matrix, bounds, _ = rows.inequalities(families, direction)
        matrix, bounds = with_boxes(matrix, bounds, lower, upper)
        violation = matrix @ direction - bounds
        norm = np.asarray(matrix.power(2) @ (1 / np.maximum(diagonal, DIAGONAL_FLOOR))).ravel()
        if weights is None:
            weights = INITIAL_STIFFNESS / np.maximum(norm, 1e-30)
            active = bounds < 0
        else:
            newly = violation > 0
            weights[active & newly] *= 10.
            # One-sided hinges release a face when the current prediction
            # moves back inside it; retaining it would turn it into a spring.
            active = newly
        system = physical.copy(order='F')
        system[0] += damping * diagonal + DIAGONAL_FLOOR
        rhs = -gradient.copy()
        selected = matrix[active]
        if selected.shape[0]:
            scatter(system, selected, weights[active])
            rhs += selected.T @ (weights[active] * bounds[active])
        _hold(system, held)
        rhs[held] = 0.
        diagnostics['factorizations'] += 1
        diagnostics['hinge_rounds'] += 1
        direction = equilibrated_solve(system, rhs, _banded_solve)
        diagnostics['active_rows'] = int(active.sum())
        diagnostics['max_linear_excess'] = float(np.max(matrix @ direction - bounds, initial=0))
        diagnostics['held_variables'] = len(held)
        yield direction


def active_directions(objective, poses, rows, families, physical, gradient, diagonal,
                      damping, diagnostics, warm):
    """Warm coupled working sets, with direction-dependent cone halfspaces."""
    from .constrained_active import ActiveSetFailure, MAX_WORKING_SETS, normalized_rows, solve_active
    from .constrained_warm import working_set
    lower, upper = step_boxes(objective, poses)
    held = np.unique(np.r_[_equalities(objective, poses), np.flatnonzero(lower == upper)])
    system = physical.copy(order='F')
    system[0] += damping * diagonal + DIAGONAL_FLOOR
    _hold(system, held)
    gradient = gradient.copy()
    gradient[held] = 0.
    if getattr(rows, 'primal_dual', False):
        from .constrained_ip import QuadraticDomain, solve
        from .constrained_soc import shifted
        adjusted = families
        rounds = 4 if getattr(rows, 'second_order', False) else 1
        warm.pop('ip_start', None)
        for index in range(rounds):
            rows._soc_last = index == rounds - 1
            rows._soc_trial = None
            domain = QuadraticDomain(adjusted, lower, upper, held)
            direction = solve(system, gradient, domain, warm, diagnostics)
            yield direction
            if rows._soc_trial is None:
                return
            measured, displacement = rows._soc_trial
            adjusted = shifted(families, measured, displacement, diagnostics)
            if adjusted is None:
                return
            warm['ip_start'] = direction
        return
    free = np.ones(len(gradient))
    free[held] = 0.
    direction = np.zeros(len(gradient))
    first = diagnostics['factorizations']
    for _ in range(3):
        remaining = MAX_WORKING_SETS - (diagnostics['factorizations'] - first)
        if remaining <= 0:
            raise ActiveSetFailure('local banded factorization budget exhausted')
        matrix, bounds, _ = rows.inequalities(families, direction, inner=False)
        matrix, bounds = with_boxes(matrix, bounds, lower, upper)
        matrix = matrix.multiply(free).tocsr()
        if getattr(rows,'merge_faces',False):
            from .rows import _merge_identical
            matrix,_,bounds=_merge_identical(matrix,np.full(len(bounds),-np.inf),bounds)
        # A face unreachable anywhere inside the original trust boxes can be
        # omitted. Box rows themselves remain, making this screen conservative.
        possible = matrix.maximum(0.) @ upper + matrix.minimum(0.) @ lower
        keep = possible >= bounds - 1e-12
        box_rows = 0 if getattr(rows,'merge_faces',False) else int(np.isfinite(lower).sum() + np.isfinite(upper).sum())
        if box_rows:
            keep[-box_rows:] = True
        chosen = np.flatnonzero(keep)
        selected, limits = matrix[chosen], bounds[chosen]
        normal, normalized_limits, valid = normalized_rows(selected, limits, system[0])
        row_scale = np.sqrt(np.asarray(selected.power(2) @ (1 / system[0])).ravel())
        cached = warm.get('projected_dual')
        initial = {} if cached is None or len(cached) != len(bounds) else dict(
            multipliers=cached[chosen] * row_scale)
        if getattr(rows,'use_primal_warm',False):
            initial.update(direction=warm.get('previous_step'),lower=lower,upper=upper)
        else:
            initial['direction']=np.zeros(len(gradient))
        direction, dual = working_set(system, gradient, normal, normalized_limits, initial, diagnostics)
        if getattr(rows,'kkt_refine',False):
            initialized=dict(active=dual>1e-7,multipliers=dual,selective=True)
            direction=solve_active(system,gradient,selected,limits,initialized,diagnostics,limit=64)
        if getattr(rows,'scalar_repair',False):
            from .constrained_scalar import repair
            direction=repair(objective,selected,limits,direction,diagnostics)
        if getattr(rows,'primal_polish',False):
            from .constrained_primal import solve_primal
            direction=solve_primal(system,gradient,selected,limits,direction,diagnostics)
        stored = np.zeros(len(bounds))
        stored[chosen] = np.divide(dual, row_scale, out=np.zeros_like(dual), where=valid)
        warm['projected_dual'] = stored
        diagnostics['screened_rows'] = int(len(bounds) - len(chosen))
        diagnostics['held_variables'] = len(held)
        yield direction
    remaining = MAX_WORKING_SETS - (diagnostics['factorizations'] - first)
    if remaining > 0:
        initialized = dict(active=dual > 1e-7, multipliers=dual)
        yield solve_active(system, gradient, selected, limits, initialized, diagnostics, limit=remaining)
