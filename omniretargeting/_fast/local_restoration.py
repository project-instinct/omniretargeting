"""Second-order conic-step correction on the support of nonlinear violations.

The rest of the proposed trajectory stays fixed. All crossing temporal rows
and entire cones enter the restricted problem; the original nonlinear guards
alone decide acceptance. This is one trajectory's SQP restoration, not a set
of separately retargeted clips.
"""

import numpy as np
from scipy import sparse

from .proposals import ALPHAS
from .repair_limits import LinearBounds
from .repair_restoration import _solve_restoration
from .shape_step import _pin_root

# Scheduling/effort limits only; none is a physical acceptance allowance.
HALO = 8
MAX_SELECTED_FRACTION = .6
MAX_RESTORATIONS = 2
RESTORATION_ITERATIONS = 3


def _support(bad, halo=HALO):
    selected = bad.copy()
    for offset in range(1, min(halo + 1, len(bad))):
        selected[offset:] |= bad[:-offset]
        selected[:-offset] |= bad[offset:]
    return selected


def _program(linear, norms, columns, scales, lower, upper):
    """Direct Clarabel form in normalized selected-column coordinates.

    The correction outside ``columns`` is zero at the current candidate.
    Restrict columns, never rows by frame: boundary stencils stay complete.
    Constant violated rows/cones remain, making impossible windows infeasible.
    """
    import clarabel

    size = len(columns)
    projection = sparse.diags(scales, format="csc")
    bounds = LinearBounds(size)
    bounds.set_box(lower / scales, upper / scales)
    for matrix, lo, hi in linear:
        bounds.add(matrix[:, columns] @ projection, lo, hi)
    for matrix, residual, allowed in norms:
        bounds.norms.append((matrix[:, columns] @ projection, residual, allowed))
    matrix, lo, hi = bounds.packed()
    finite_lo, finite_hi = np.isfinite(lo), np.isfinite(hi)
    blocks = [sparse.vstack([-matrix[finite_lo], matrix[finite_hi]], format="csc")]
    targets = [np.r_[-lo[finite_lo], hi[finite_hi]]]
    cones = [clarabel.NonnegativeConeT(len(targets[0]))]
    for jacobian, residual, allowed in bounds.cones():
        if not len(allowed):
            continue
        dimension = residual.shape[1]
        scale = np.maximum(allowed, 1e-3)
        jacobian = (sparse.diags(np.repeat(1 / scale, dimension)) @ jacobian).tocoo()
        rows = jacobian.row + jacobian.row // dimension + 1
        blocks.append(sparse.csc_matrix((-jacobian.data, (rows, jacobian.col)),
                                       shape=(len(allowed) * (dimension + 1), size)))
        targets.append(np.c_[allowed / scale, residual / scale[:, None]].ravel())
        cones.extend(clarabel.SecondOrderConeT(dimension + 1) for _ in allowed)
    return (2 * sparse.eye(size, format="csc"), np.zeros(size),
            sparse.vstack(blocks, format="csc"), np.concatenate(targets), cones)


def _bounds(objective, candidate, guards):
    """Every guard plus the acquisition-only full root translation balls."""
    linear, norms = [], []
    shape, coords = objective.source_shape, objective.coords
    for guard in guards:
        lo, balls = (guard.repair_bounds(candidate, restore=True) if guard is shape
                     else guard.repair_bounds(candidate))
        linear.extend(lo)
        norms.extend(balls)
    if coords.root_q is not None:
        jacobian = np.zeros((3, coords.width))
        for axis in range(3):
            column = coords.inverse.get(coords.root_d + axis)
            if column is not None:
                jacobian[axis, column] = 1.
        matrix = sparse.kron(sparse.eye(len(candidate)), jacobian, format="csr")
        references = [shape.seed]
        if shape.acquired_poses is not None:
            references.append(shape.acquired_poses)
        q = coords.root_q
        for reference in references:
            norms.append((matrix, candidate[:, q:q + 3] - reference[:, q:q + 3],
                          np.full(len(candidate), shape.xy)))
    return linear, norms


def _violating_columns(linear, norms, frames, width):
    """Kinematic columns of violated rows/cones, with per-coordinate halos.

    A root-rotation stencil may reject many frames but only involve three
    coordinates. Derive support from Jacobians, never body-part labels.
    """
    selected = np.zeros(frames * width, bool)
    for matrix, lo, hi in linear:
        bad = (np.asarray(lo) > 0.) | (np.asarray(hi) < 0.)
        selected[matrix[bad].nonzero()[1]] = True
    for matrix, residual, allowed in norms:
        bad = np.linalg.norm(residual, axis=1) > allowed
        selected[matrix[np.repeat(bad, residual.shape[1])].nonzero()[1]] = True
    return _support(selected.reshape(frames, width))


def _selection(objective, candidate, guards, bad):
    frames, width = len(candidate), objective.coords.width
    selected, prepared = _support(bad), None
    maximum = max(2 * HALO + 1, MAX_SELECTED_FRACTION * frames)
    if selected.sum() <= maximum:
        columns = np.flatnonzero(np.repeat(selected, width))
    else:
        prepared = _bounds(objective, candidate, guards)
        coordinates = _violating_columns(*prepared, frames, width)
        if not coordinates.any() or coordinates.sum() > maximum * width:
            return None
        columns = np.flatnonzero(coordinates)
        selected = coordinates.any(axis=1)
    return columns, selected, prepared


def restore_local(objective, origin, candidate, guards, bad, *, selection=None):
    """Restore a sparse violation cluster, or None for established fallback.

    The selected union is solved jointly: at an infeasible candidate, a
    feasible solution need not exist for one block with its neighbours fixed.
    """
    coords = objective.coords
    selection = _selection(objective, candidate, guards, bad) if selection is None else selection
    if selection is None:
        return None
    columns, selected, prepared = selection
    lo, hi = coords.bounds(origin, objective.cfg["angular_step_rad"],
                           objective.cfg["translation_step_m"])
    scales = np.maximum(np.maximum(abs(lo), abs(hi)).ravel()[columns], 1e-8)
    untouched = np.array(candidate[~selected], copy=True)
    for _ in range(RESTORATION_ITERATIONS):
        displacement = coords.difference(origin, candidate)
        lower, upper = lo - displacement, hi - displacement
        lower[:, coords.scalar_cols] = np.maximum(
            lower[:, coords.scalar_cols], objective.lower - candidate[:, coords.qi])
        upper[:, coords.scalar_cols] = np.minimum(
            upper[:, coords.scalar_cols], objective.upper - candidate[:, coords.qi])
        _pin_root(objective, candidate, lower, upper)
        linear, norms = _bounds(objective, candidate, guards) if prepared is None else prepared
        prepared = None
        program = _program(linear, norms, columns, scales,
                           lower.ravel()[columns], upper.ravel()[columns])
        delta, _ = _solve_restoration(program)
        if delta is None or not np.isfinite(delta).all():
            return None
        step = np.zeros(len(origin) * coords.width)
        step[columns] = delta * scales
        candidate = objective.project(coords.integrate(candidate, step.reshape(len(origin), -1), 1.))
        # Projection must not silently change any fixed frame.
        candidate[~selected] = untouched
        if all(guard.accepts(candidate) for guard in guards):
            actual = coords.difference(origin, candidate)
            if np.all(actual >= lo - 1e-9) and np.all(actual <= hi + 1e-9):
                return candidate
            return None
    return None


def backtrack_with_restoration(objective, poses, direction, guards):
    """Candidate, nominal fraction, and whether local restoration succeeded."""
    capable = all(hasattr(g, "repair_bounds") and hasattr(g, "violation_frames") for g in guards)
    attempts = 0
    for alpha in ALPHAS:
        candidate = objective.project(objective.coords.integrate(
            poses, direction.reshape(len(poses), -1), alpha))
        if all(guard.accepts(candidate) for guard in guards):
            return candidate, alpha, False
        if not capable or alpha < .125 or attempts >= MAX_RESTORATIONS:
            continue
        bad = np.zeros(len(poses), bool)
        for guard in guards:
            bad |= guard.violation_frames(candidate)
        if not bad.any():
            continue
        selection = _selection(objective, candidate, guards, bad)
        if selection is None:
            continue
        attempts += 1
        restored = restore_local(objective, poses, candidate, guards, bad, selection=selection)
        if restored is not None:
            return restored, alpha, True
    return None, 0., False
