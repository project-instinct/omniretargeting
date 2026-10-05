"""Cheap banded direction under every frozen temporal budget.

Before the full conic fallback, solve the same damped LM model inside the same
trust/joint boxes and fixed root, subject only to the temporal budgets of every
``SequenceContinuityGuard`` and ``CorrectionContinuity``: all first, second and
third order per-joint rows with their existing linearization slack, and each
norm budget.
The banded primal-dual KKT of ``constrained_ip`` (the window B route) solves
it over the complete row list, norm budgets as exact balls; no row is chosen
from a log. Ordinary
backtracking against every candidate guard and the objective's descent test
still decide acceptance; any failure returns None and the unchanged conic
fallback runs.
"""

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgError

from .barrier_rows import ball, scalar
from .constrained_active import ActiveSetFailure
from .constrained_ip import QuadraticDomain, solve
from .constrained_step import step_boxes
from .continuity import CorrectionContinuity
from .proposals import ALPHAS, backtrack, guards_of
from .rows import _merge_identical
from .solver import DIAGONAL_FLOOR, _hold
from .._sequence.continuity_guard import SequenceContinuityGuard

# Predicted excess (metres/radians) still sent to the nonlinear guards: their
# acceptance slack exceeds the linearization slack of these rows by 1e-7.
LINEAR_TOLERANCE = 1e-8
MIN_ALPHA = .25


class TaggedDirection(np.ndarray):
    """ndarray carrying only the proposal backend used by history/trace."""


def _temporal_rows(objective, poses, every_guard=False):
    """Scalar intervals and norm cones of every temporal guard, at ``poses``.

    ``every_guard`` also takes the frozen rows of the other guards (contact
    bands and balls, reference layout and root balls); a guard without
    ``qp_bounds`` then makes the direction unavailable (None).
    """
    intervals, cones = [], []
    for guard in guards_of(objective)[1]:
        # A window wrapper embeds the local poses and returns local columns.
        temporal = isinstance(getattr(guard, "guard", guard),
                              (SequenceContinuityGuard, CorrectionContinuity))
        if every_guard and not temporal and not callable(getattr(guard, "qp_bounds", None)):
            return None
        if temporal or every_guard:
            linear, norms = guard.qp_bounds(poses)
            intervals += linear
            cones += [(sparse.csr_matrix(matrix), np.asarray(value), np.asarray(radius))
                      for matrix, value, radius in norms]
    return intervals, cones


def _temporal_linear_rows(objective, poses, intervals=None):
    """One-sided rows ``A d <= h`` of every scalar temporal budget."""
    if intervals is None:
        intervals, _ = _temporal_rows(objective, poses)
    size = len(poses) * objective.coords.width
    if not intervals:
        return sparse.csr_matrix((0, size)), np.empty(0)
    matrix, lower, upper = _merge_identical(
        sparse.vstack([m for m, _, _ in intervals], format="csr"),
        np.concatenate([lo for _, lo, _ in intervals]), np.concatenate([hi for _, _, hi in intervals]))
    finite_low, finite_high = np.isfinite(lower), np.isfinite(upper)
    return (sparse.vstack([-matrix[finite_low], matrix[finite_high]], format="csr"),
            np.r_[-lower[finite_low], upper[finite_high]])


def _cone_excess(cones, direction):
    excess = 0.
    for matrix, value, radius in cones:
        predicted = value + (matrix @ direction).reshape(value.shape)
        excess = max(excess, float(np.max(np.linalg.norm(predicted, axis=1) - radius, initial=0.)))
    return excess


def _band_product(ab, vector):
    """Symmetric lower-banded matrix times ``vector``."""
    result = ab[0] * vector
    for offset in range(1, len(ab)):
        band = ab[offset, :len(vector) - offset]
        result[offset:] += band * vector[:-offset]
        result[:-offset] += band * vector[offset:]
    return result


def temporal_budget_direction(objective, poses, system, gradient, diagonal,
                              damping, *, diagnostics=None, every_guard=False):
    """Return ``(step, candidate, alpha)``, or ``None`` to use the old fallback."""
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(factorizations=0, band_solves=0)
    coordinates = objective.coords
    lower, upper = step_boxes(objective, poses)
    fixed = getattr(objective, "fixed_root", None)
    if fixed is not None and coordinates.root_q is not None:
        target = poses.copy()
        target[:, coordinates.root_q:coordinates.root_q + 7] = fixed
        displacement = coordinates.difference(poses, target).ravel()
        width = coordinates.width
        for d in range(coordinates.root_d, coordinates.root_d + 6):
            column = coordinates.inverse.get(d)
            if column is not None:
                columns = np.arange(len(poses)) * width + column
                lower[columns] = upper[columns] = displacement[columns]
    held = np.flatnonzero(lower == upper)
    if np.any(lower[held] != 0.):
        diagnostics["fallback_reason"] = "temporal_held_displacement"
        return None
    damped = system.copy(order="F")
    damped[0] += damping * diagonal + DIAGONAL_FLOOR
    _hold(damped, held)
    reduced = gradient.copy()
    reduced[held] = 0.

    rows = _temporal_rows(objective, poses, every_guard)
    if rows is None:
        diagnostics["fallback_reason"] = "guard_without_qp_bounds"
        return None
    intervals, cones = rows
    linear, limits = _temporal_linear_rows(objective, poses, intervals)
    diagnostics.update(temporal_rows=int(linear.shape[0]),
                       temporal_cones=int(sum(len(r) for _, _, r in cones)),
                       held_variables=int(len(held)))
    families = []
    for matrix, low, high in intervals:
        matrix = sparse.csr_matrix(matrix)
        finite = np.isfinite(high)
        if finite.any():
            families.append(scalar(high[finite], -matrix[finite], ("Temporal", "upper"), None))
        finite = np.isfinite(low)
        if finite.any():
            families.append(scalar(-low[finite], matrix[finite], ("Temporal", "lower"), None))
    families += [ball(value, radius, matrix, ("Temporal", "norm"), None)
                 for matrix, value, radius in cones]
    try:
        # The B-route primal-dual KKT: exact balls, banded Newton factors.
        domain = QuadraticDomain(families, lower, upper, held)
        direction = solve(damped, reduced, domain, {}, diagnostics)
    except (ArithmeticError, LinAlgError, ValueError, ActiveSetFailure) as error:
        diagnostics.update(fallback_reason="temporal_qp_failed", error=f"{type(error).__name__}: {error}")
        return None
    linear_excess = float(np.max(linear @ direction - limits, initial=0.))
    cone_excess = _cone_excess(cones, direction)
    diagnostics.update(domain_rows=int(domain.count), linear_excess=linear_excess,
                       cone_excess=cone_excess, ip_status=diagnostics.get("ip_statuses", ["unconstrained"])[-1])
    if max(linear_excess, cone_excess) > LINEAR_TOLERANCE:
        diagnostics["fallback_reason"] = "temporal_qp_unconverged"
        return None
    direction = direction.view(TaggedDirection)
    direction.proposal_backend = "banded_guard" if every_guard else "banded_temporal"
    # Fractions below MIN_ALPHA are never taken here: stop the search there.
    candidate, alpha = backtrack(objective, poses, direction, guards_of(objective)[1],
                                 alphas=tuple(a for a in ALPHAS if a >= MIN_ALPHA))
    diagnostics["feasible_alpha"] = float(alpha)
    if candidate is None or alpha < MIN_ALPHA:
        diagnostics["fallback_reason"] = "temporal_direction_guard_rejected"
        if getattr(objective, "performance_trace", None) is not None:
            # Diagnostic only: which guards reject the full step.
            full = objective.project(coordinates.integrate(
                poses, direction.reshape(objective.frames, objective.width), 1.))
            diagnostics["rejecting_guards"] = [
                type(getattr(guard, "guard", guard)).__name__
                for guard in guards_of(objective)[1] if not guard.accepts(full)]
        return None
    # The same Gauss-Newton model decrease and true descent as the LM loop.
    actual = coordinates.difference(poses, candidate).ravel()
    predicted = -2. * float(gradient @ actual) - float(actual @ _band_product(system, actual))
    before = objective.evaluate(poses, False)[0].cost
    after = objective.evaluate(candidate, False)[0].cost
    ratio = (before - after) / predicted if predicted > 0 else -1.
    diagnostics.update(cost_before=float(before), cost_after=float(after), ratio=float(ratio))
    if not (np.isfinite(after) and after < before and ratio > 0):
        diagnostics["fallback_reason"] = "temporal_direction_no_decrease"
        return None
    return direction, candidate, alpha
