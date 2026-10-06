"""Levenberg-Marquardt on the banded normal equations of a whole trajectory.

Frames are ordered consecutively, so per-frame blocks and temporal rows (at
most third differences) give a symmetric banded system. One banded Cholesky
solve replaces the constrained conic subproblems of the guarded solver.
Internal numerical constants below are not user configuration.
"""

import time

import numpy as np
from scipy.linalg import solveh_banded

from .proposals import Attempts, backtrack, guards_of

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - scheduling only
    threadpool_limits = None

# Marquardt damping relative to the Gauss-Newton diagonal (Nielsen update).
INITIAL_DAMPING = 1e-3
MIN_DAMPING, MAX_DAMPING = 1e-9, 1e8
# Diagonal floor keeping the banded Cholesky positive definite.
DIAGONAL_FLOOR = 1e-9
# Two consecutive accepted iterations below this relative cost decrease end
# the pass: further refinement is invisible next to the grid proxies.
CONVERGED_DECREASE = 3e-3
CONVERGED_STEPS = 2
MAX_REJECTIONS = 8
# The constrained fallback direction is retried after damping grows: a shorter
# conic step has less curvature error at nonlinear guards. Each solve is costly.
MAX_SHAPE_SOLVES = 3
# Fractions of an overshooting conic step tried before its attempt is rejected,
# and at most this many of them restored onto the guards (one conic solve each).
CONIC_FRACTIONS = (.5, .25, .125, .0625, .03125)
MAX_CONIC_RESTORATIONS = 2
# Joints within this distance of a limit, with descent pointing outward, are
# held for the iteration (active set) instead of being clipped after a step.
BOUND_TOLERANCE = 1e-9
# Root envelope balls are not boxes: at a boundary the outward radial step
# component is pinned by a block this many times the largest diagonal.
ROOT_HOLD_STIFFNESS = 1e6


def _progress_converged(costs, cost, slow, alpha, guarded, threshold, shortened=False):
    """A shortened feasible step alone is not a stationarity certificate.

    Two small full steps may end a pass. Guard-clipped steps instead use the
    existing three-step cost window, which includes the preceding useful
    step and cannot mistake two curvature corrections for convergence.
    """
    from .runtime_options import OPTIONS
    if not OPTIONS.batch2 and shortened:
        alpha = 1.
    decrease = (costs[-1] - cost) / max(costs[-1], 1e-12)
    slow = slow + 1 if decrease < threshold and alpha == 1. else 0
    costs.append(cost)
    window = CONVERGED_STEPS + 1
    # ``stationary_window``: the same window also ends passes of short
    # unguarded (banded) steps whose cost has stopped moving.
    done = slow >= CONVERGED_STEPS or (
        (guarded or OPTIONS.stationary_window) and len(costs) > window
        and costs[-1 - window] - cost < window * threshold * costs[-1 - window])
    return slow, done


def banded_system(frames, width, hessian_blocks, temporal):
    """Lower banded storage of blockdiag(H_f) + J_t^T J_t."""
    size = frames * width
    extra = None
    if temporal is not None:
        extra = (temporal.T @ temporal).tocsr()
        extra.sum_duplicates()
        extra = extra.tocoo()
    band = width - 1
    if extra is not None and extra.nnz:
        band = max(band, int(np.max(extra.row - extra.col)))
    ab = np.zeros((band + 1, size), order="F")
    rows, cols = np.tril_indices(width)
    offsets = (np.arange(frames) * width)[:, None]
    # Block entries and canonical sparse entries are unique band positions.
    ab[np.broadcast_to(rows - cols, (frames, len(rows))), offsets + cols] = hessian_blocks[:, rows, cols]
    if extra is not None and extra.nnz:
        lower = extra.row >= extra.col
        ab[extra.row[lower] - extra.col[lower], extra.col[lower]] += extra.data[lower]
    return ab


def _banded_solve(ab, rhs):
    """Narrow-band Cholesky; threaded BLAS only adds overhead at this band."""
    if threadpool_limits is None:
        return solveh_banded(ab, rhs, lower=True, check_finite=False, overwrite_ab=True)
    with threadpool_limits(1):
        return solveh_banded(ab, rhs, lower=True, check_finite=False, overwrite_ab=True)


def _active_bounds(objective, poses, gradient):
    """Flat indices of scalar joints resting on a bound they would cross."""
    coords = objective.coords
    frames, width = len(poses), gradient.size // len(poses)
    values = poses[:, coords.qi]
    slope = gradient.reshape(frames, width)[:, coords.scalar_cols]
    held = ((values <= objective.lower + BOUND_TOLERANCE) & (slope > 0)) | (
        (values >= objective.upper - BOUND_TOLERANCE) & (slope < 0)
    )
    frame, joint = np.nonzero(held)
    variables = frame * width + coords.scalar_cols[joint]
    if getattr(objective, "fixed_root", None) is not None:
        columns = [coords.inverse[d] for d in range(coords.root_d, coords.root_d + 6)
                   if d in coords.inverse]
        if columns:
            variables = np.r_[variables, (np.arange(frames)[:, None] * width + columns).ravel()]
    return variables


def _root_holds(objective, poses, gradient):
    """Outward root rows (frames, columns, normals) resting on an envelope ball."""
    coords = objective.coords
    frames, width = len(poses), gradient.size // len(poses)
    holds = []
    for offset, normals in objective.root_boundaries(poses):
        columns = [coords.inverse.get(coords.root_d + offset + k) for k in range(3)]
        if any(column is None for column in columns):
            continue
        slope = np.einsum("fi,fi->f", gradient.reshape(frames, width)[:, columns], normals)
        frame = np.flatnonzero(np.any(normals != 0, axis=1) & (slope < 0))
        holds.append((frame, np.asarray(columns), normals[frame]))
    return holds


def _hold_radial(ab, width, holds):
    """Pin each outward root component with a stiff block n n^T."""
    stiffness = ROOT_HOLD_STIFFNESS * max(float(ab[0].max()), 1.0)
    for frame, columns, normals in holds:
        base = frame * width
        for i in range(3):
            for j in range(i + 1):
                ab[columns[i] - columns[j], base + columns[j]] += stiffness * normals[:, i] * normals[:, j]


def _hold(ab, variables):
    """Decouple held variables in lower banded storage (unit diagonal)."""
    ab[1:, variables] = 0.0
    for offset in range(1, ab.shape[0]):
        shifted = variables[variables >= offset] - offset
        ab[offset, shifted] = 0.0
    ab[0, variables] = 1.0


def _model_decrease(acc, temporal_j, step, gradient):
    """Decrease of the Gauss-Newton model: -2 g.d - d.(J^T J) d."""
    frames, width = acc.gradient.shape
    blocks = step.reshape(frames, width)
    curvature = float(np.einsum("fw,fwv,fv->", blocks, acc.hessian, blocks))
    temporal = temporal_j @ step
    curvature += float(temporal @ temporal)
    decrease = -2.0 * float(gradient @ step) - curvature
    model = getattr(acc, 'joint_model', None)
    return decrease if model is None else decrease + model.correction(step)


def optimize(objective, poses, *, iterations, callback=None, label="fast",
             converged_decrease=CONVERGED_DECREASE):
    """Minimize the objective; returns poses and per-iteration history.

    A pass ends after ``CONVERGED_STEPS`` accepted iterations below the
    relative cost decrease ``converged_decrease``. Correction stages whose
    own rows are a small part of the total cost pass a smaller one.
    """
    from .runtime_options import OPTIONS
    if OPTIONS.windowed_priority and getattr(objective, 'priority_clearance', None) is not None:
        from .window_driver import optimize_windowed_priority
        return optimize_windowed_priority(objective, poses, iterations, callback, label, converged_decrease)
    started = objective.start_workers()
    try:
        return _optimize(objective, poses, iterations, callback, label, converged_decrease)
    finally:
        if started:
            objective.stop_workers()


def _conic_proposal(objective, poses, system, gradient, diagonal, damping, guards, *, model=None):
    """Constrained fallback direction, backtracked or restored; None if unsolved.

    The candidate is None when neither a fraction nor the restored step passes
    the guards; the direction itself is still returned for ``_shorten_conic``.
    """
    from .shape_step import bounded_shape_step, restore_shape_step
    from .runtime_options import OPTIONS

    trace = getattr(objective, "performance_trace", None)
    if model is not None:
        if trace is not None:
            trace.record_call("proposal", "model_qp")
        from .qp_assembly import solve_model
        from .globalization import globalize
        solution = solve_model(model, damping)
        if solution is None:
            return None
        solution, projected, fraction = globalize(
            objective, model, solution, damping, guards,
            soc=OPTIONS.objective_soc, frame_lengths=OPTIONS.frame_step_lengths)
        if fraction < .25:
            restoration_tick = time.monotonic()
            restored = restore_shape_step(objective, poses, solution.step)
            if trace is not None:
                trace.add_operation("recoveries", time.monotonic() - restoration_tick)
                if restored is None:
                    trace.add_failed(time.monotonic() - restoration_tick,
                                     "model_restoration_unsolved")
            if restored is not None and all(guard.accepts(restored) for guard in guards):
                projected, fraction = restored, 1.
        return solution.step, projected, fraction
    # Opt-in schedule: after one failed cheap direction on this objective (one
    # LM solve or window), its binding guards are not temporal; go straight to
    # the conic fallback for the rest of that solve.
    skip = OPTIONS.adaptive_banded_direction and getattr(objective, "_banded_direction_failed", False)
    if (OPTIONS.banded_temporal_direction or OPTIONS.banded_guard_direction) and not skip:
        from .temporal_direction import temporal_budget_direction
        temporal_tick = time.monotonic()
        temporal_diagnostics = {}
        if trace is not None:
            trace.record_call("proposal", "banded_guard" if OPTIONS.banded_guard_direction
                              else "banded_temporal")
        cheap = temporal_budget_direction(
            objective, poses, system, gradient, diagonal, damping,
            diagnostics=temporal_diagnostics, every_guard=OPTIONS.banded_guard_direction)
        if trace is not None:
            seconds = time.monotonic() - temporal_tick
            trace.add_operation("banded_temporal_directions", seconds)
            trace.temporal_direction(temporal_diagnostics, seconds, accepted=cheap is not None)
        if cheap is not None:
            return cheap
        objective._banded_direction_failed = True
        if trace is not None:
            reason = temporal_diagnostics.get("fallback_reason", "temporal_direction_unsolved")
            trace.add_failed(time.monotonic() - temporal_tick, reason)
            trace.fallback(reason)
    if trace is not None:
        trace.record_call("proposal", "shape_conic")
    bounded = bounded_shape_step(objective, poses, system, gradient, diagonal, damping)
    if bounded is None:
        return None
    if getattr(objective, 'joint_witness', None) is not None:
        from .joint_proposal import globalize
        return globalize(objective, poses, bounded, guards)
    locally_restored = False
    if OPTIONS.local_conic_restoration:
        from .local_restoration import backtrack_with_restoration
        restoration_tick = time.monotonic()
        projected, fraction, locally_restored = backtrack_with_restoration(
            objective, poses, bounded, guards)
        if trace is not None:
            trace.add_operation("recoveries", time.monotonic() - restoration_tick)
    else:
        projected, fraction = backtrack(objective, poses, bounded, guards)
    if fraction < .25 and not locally_restored:
        from .proposal_curvature import CONTACT_CURVATURE, restore_curvature
        restored = (restore_curvature(objective, poses, bounded, guards,
                                     (system, diagonal, damping)) if CONTACT_CURVATURE else None)
        if restored is None:
            restoration_tick = time.monotonic()
            restored = restore_shape_step(objective, poses, bounded)
            if trace is not None:
                trace.add_operation("recoveries", time.monotonic() - restoration_tick)
                if restored is None:
                    trace.add_failed(time.monotonic() - restoration_tick,
                                     "shape_restoration_unsolved")
        if restored is not None and all(guard.accepts(restored) for guard in guards):
            projected, fraction = restored, 1.
    return bounded, projected, fraction


def _shorten_conic(objective, poses, step, alpha, acc, temporal_j, gradient, cost, *, metric=None):
    """Shorter fraction of a conic step whose objective decreases, or None.

    The conic fallback solves the damped model under every guard's
    linearization, so its step descends the model and fractions of it stay
    inside those linear bounds. Once guard-active stencils reject every banded
    step, the capped conic retries (each a new solve at larger damping) could
    all overshoot and end the pass at a large residual; along the same step,
    short enough fractions follow the model instead (Armijo backtracking).
    """
    from .shape_step import restore_shape_step
    from .shortening import FractionRestorations
    from .runtime_options import OPTIONS
    from .proposal_curvature import CONTACT_CURVATURE, restore_curvature

    if getattr(objective, 'joint_witness', None) is not None:
        from .joint_proposal import shorten
        return shorten(objective, poses, step, alpha, acc, temporal_j, gradient, cost)
    if not CONTACT_CURVATURE:
        from .window_restoration_parallel import fraction_class
        FractionRestorations = fraction_class(objective, FractionRestorations)
    coords = objective.coords
    trace = getattr(objective, "performance_trace", None)
    guards = guards_of(objective)[1]
    step = np.asarray(step).reshape(objective.frames, objective.width)
    restorations = 0
    with FractionRestorations(restore_shape_step, objective, poses, step, alpha,
                              CONIC_FRACTIONS, MAX_CONIC_RESTORATIONS, guards,
                              enabled=(OPTIONS.parallel_shortening and trace is None
                                       and not CONTACT_CURVATURE)) as pending:
        for fraction in CONIC_FRACTIONS:
            scaled = alpha * fraction
            candidate = objective.project(coords.integrate(poses, step, scaled))
            if not objective.evaluate(candidate, with_jacobian=False)[0].cost < cost:
                continue
            if not all(guard.accepts(candidate) for guard in guards):
                # An active nonlinear bound is crossed at second order by any
                # linearly feasible direction: restore the short step instead.
                if restorations == MAX_CONIC_RESTORATIONS:
                    return None
                restorations += 1
                restoration_tick = time.monotonic()
                candidate = (restore_curvature(objective, poses, step * scaled, guards, metric)
                             if CONTACT_CURVATURE else None)
                if candidate is None:
                    candidate = pending.restore(fraction)
                if trace is not None:
                    trace.add_operation("recoveries", time.monotonic() - restoration_tick)
                    if candidate is None:
                        trace.add_failed(time.monotonic() - restoration_tick,
                                         "short_step_restoration_unsolved")
                if candidate is None or not all(guard.accepts(candidate) for guard in guards):
                    continue
            actual = coords.difference(poses, candidate).ravel()
            predicted = _model_decrease(acc, temporal_j, actual, gradient)
            trial = objective.evaluate(candidate)
            ratio = (cost - trial[0].cost) / predicted if predicted > 0 else -1.0
            if np.isfinite(trial[0].cost) and trial[0].cost < cost and ratio > 0:
                return candidate, scaled, trial, ratio
    return None


def _optimize(objective, poses, iterations, callback, label, converged_decrease):
    trace = getattr(objective, "performance_trace", None)
    if trace is None or not trace.enabled:
        return _optimize_uncounted(
            objective, poses, iterations, callback, label, converged_decrease)

    trace.begin_phase(label)
    saved_evaluate = objective.__dict__.get("evaluate")
    evaluate_was_installed = getattr(objective, "_performance_trace_installed", False)
    if not evaluate_was_installed:
        original_evaluate = objective.evaluate

        def traced_evaluate(*args, **kwargs):
            tick = time.monotonic()
            try:
                return original_evaluate(*args, **kwargs)
            finally:
                with_jacobian = kwargs.get("with_jacobian", args[1] if len(args) > 1 else True)
                trace.add_evaluation(time.monotonic() - tick,
                                     with_jacobian=bool(with_jacobian))

        objective.evaluate = traced_evaluate

    from .performance_trace import explain_guard_rows, timed_phase
    from .proposals import guards_of
    guards = guards_of(objective)[1]
    saved_accepts = [(guard, guard.__dict__.get("accepts")) for guard in guards]
    saved_constraints = []
    for guard in guards:
        for name in ("constraints", "qp_bounds"):
            original = getattr(guard, name, None)
            if callable(original):
                saved_constraints.append((guard, name, guard.__dict__.get(name)))
                setattr(guard, name, timed_phase("constraint_row_construction")(original))
    for guard in guards:
        original_accepts = guard.accepts

        def traced_accepts(*args, _guard=guard, _original=original_accepts, **kwargs):
            tick = time.monotonic()
            accepted = False
            try:
                accepted = bool(_original(*args, **kwargs))
                return accepted
            finally:
                guard_seconds = time.monotonic() - tick
                explanation = None
                if not accepted:
                    with trace.operation("guard_diagnostics"):
                        explanation = explain_guard_rows(objective, _guard, args[0])
                    # Observe later guards without calling their acceptance predicates:
                    # the solver's original all(...) may short-circuit on this one.
                    for other in guards:
                        if other is not _guard:
                            with trace.operation("guard_diagnostics"):
                                explanation.extend(row for row in explain_guard_rows(objective, other, args[0])
                                                   if row.get("excess") is not None and row["excess"] > 0)
                trace.add_guard(_guard, accepted, guard_seconds, explanation)

        guard.accepts = traced_accepts

    automatic_iterations = (getattr(objective, "_soft_bulk", False) or
                            getattr(objective, "priority_clearance", None) is not None)
    iteration_start = len(trace.iterations)

    def traced_callback(candidate, row):
        if automatic_iterations:
            trace.add_alpha(row.get("alpha", 0.))
            trace.finish_iteration(row, row.get("step_backend"))
        if callback is not None:
            callback(candidate, row)

    try:
        with trace.instrument(automatic_iterations=automatic_iterations):
            poses, history = _optimize_uncounted(
                objective, poses, iterations, traced_callback, label, converged_decrease)
            if automatic_iterations and trace._iteration is not None and history:
                trace.finish_iteration(history[-1], None)
        if automatic_iterations:
            for row, diagnostic in zip(history, trace.iterations[iteration_start:]):
                row["iteration_performance"] = diagnostic
        if not history:
            stop_reason = "no_history"
        elif history[-1].get("alpha", 0.0) == 0.0:
            stop_reason = "rejected_iteration"
        elif len(history) < iterations:
            stop_reason = "converged"
        else:
            stop_reason = "iteration_cap"
        phase = trace.finish_phase(stop_reason)
        if phase is not None and history:
            history[-1]["solver_performance"] = phase
        return poses, history
    finally:
        for guard, name, original in saved_constraints:
            if original is None:
                guard.__dict__.pop(name, None)
            else:
                setattr(guard, name, original)
        for guard, original_accepts in saved_accepts:
            if original_accepts is None:
                guard.__dict__.pop("accepts", None)
            else:
                guard.accepts = original_accepts
        if not evaluate_was_installed:
            if saved_evaluate is None:
                objective.__dict__.pop("evaluate", None)
            else:
                objective.evaluate = saved_evaluate


def _optimize_uncounted(objective, poses, iterations, callback, label, converged_decrease):
    if getattr(objective, '_soft_bulk', False):
        from .soft_bulk import optimize_bulk
        return optimize_bulk(objective, poses, iterations, callback, label, converged_decrease)
    from .runtime_options import OPTIONS
    if OPTIONS.windowed_priority and getattr(objective, 'priority_clearance', None) is not None:
        from .window_driver import optimize_windowed_priority
        return optimize_windowed_priority(objective, poses, iterations, callback, label, converged_decrease)
    if getattr(objective, 'priority_clearance', None) is not None:
        from .prioritized_solver import optimize_priority
        return optimize_priority(objective, poses, iterations, callback, label, converged_decrease)
    coords = objective.coords
    joint = getattr(objective, 'joint_witness', None)
    trace = getattr(objective, "performance_trace", None)
    frames, width = objective.frames, objective.width
    # Source fitting stays byte-identical. This experiment starts only after
    # the immutable reference shape has been established.
    active_box = OPTIONS.active_joint_bounds and getattr(objective, 'source_shape', None) is not None

    def band_step(system, diagonal, held, radial, rhs, damping):
        tick = time.monotonic()
        damped = system.copy(order="F")
        damped[0] += damping * diagonal + DIAGONAL_FLOOR
        _hold_radial(damped, width, radial)
        if active_box:
            from .box_step import direction_bounds, solve_box_banded
            lower, upper = direction_bounds(objective, poses, held)
            result = solve_box_banded(damped, rhs, lower, upper, _banded_solve, _hold)
        else:
            _hold(damped, held)
            result = _banded_solve(damped, rhs)
        if trace is not None:
            trace.add_operation("banded_solves", time.monotonic() - tick)
            if result is None:
                trace.add_failed(time.monotonic() - tick, "banded_solve_unsolved")
        return result

    damping, growth = INITIAL_DAMPING, 2.0
    history, slow = [], 0
    start = time.monotonic()
    acc, temporal_r, temporal_j, stats, _ = objective.evaluate(poses)
    cost = acc.cost
    costs = [cost]
    for iteration in range(iterations):
        if joint is not None:
            joint.begin(poses, iteration)
        if trace is not None and trace.enabled:
            trace.begin_iteration(label, iteration)
        gradient = acc.gradient.ravel() + temporal_j.T @ temporal_r
        system = banded_system(frames, width, acc.hessian, temporal_j)
        # Temporal rows can be extremely stiff while allowing smooth common
        # motion. Damping by their diagonal falsely pins that nullspace and
        # makes useful contact updates look converged before damping decays.
        diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), DIAGONAL_FLOOR)
        held = _active_bounds(objective, poses, gradient)
        if active_box:
            # Scalar variables are selected by the actual coupled direction;
            # keep only unconditional root holds from the old selector.
            held = held[~np.isin(held % width, coords.scalar_cols)]
        radial = _root_holds(objective, poses, gradient)
        rhs = -gradient
        rhs[held] = 0.0
        accepted = False
        shape_solves = 0
        step_backend = "banded"
        shortened = None
        shape, _ = guards_of(objective)
        conic_possible = shape is not None and all(
            hasattr(g, 'constraints') for g in objective.candidate_guards)
        model = None
        if conic_possible and (OPTIONS.direct_acquisition_qp or OPTIONS.objective_soc
                               or OPTIONS.frame_step_lengths):
            from .qp_assembly import prepare_model
            from .qp_guard_rows import UnsupportedRows
            try:
                model = prepare_model(objective, poses, system, gradient, diagonal)
            except UnsupportedRows:
                # Unknown external guard: preserve its original constraints.
                # Priority mode never takes this fallback or drops a soft row.
                pass
        # Every rejection below grows the damping the same way.
        dampings, value, factor = [], damping, growth
        for _ in range(MAX_REJECTIONS):
            dampings.append(value)
            value, factor = min(value * factor, MAX_DAMPING), factor * 2
        kwargs = {} if model is None else {'model': model}
        attempts = Attempts(objective, poses, system, gradient, diagonal, held, radial, rhs,
                            dampings, band_step, _conic_proposal, **kwargs)
        for attempt in range(MAX_REJECTIONS):
            if attempt == 1:
                # The first attempt was rejected: compute the rest at once.
                if trace is None and joint is None:
                    attempts.speculate(1, shape_solves, MAX_SHAPE_SOLVES, conic_possible)
            # Nonlinear source-shape checks supplement the scalar boxes. A
            # rejected proposal backtracks inside the same immutable envelope.
            candidate_tick = time.monotonic()
            if trace is not None:
                trace.record_call("proposal", "banded")
            proposal = attempts.banded(attempt)
            if trace is not None:
                trace.banded_alpha(None if proposal is None else float(proposal[2]))
            if proposal is None:
                if trace is not None:
                    trace.add_failed(time.monotonic() - candidate_tick,
                                     "banded_proposal_unsolved")
                    trace.fallback("banded_proposal_unsolved")
                damping, growth = min(damping * growth, MAX_DAMPING), growth * 2
                continue
            step, candidate, alpha = proposal
            conic = False
            trial_damping = damping
            decreased = False
            checked_short = (OPTIONS.accept_short_banded or joint is not None) and candidate is not None and alpha < .25
            if checked_short:
                actual_step = coords.difference(poses, candidate).ravel()
                predicted = _model_decrease(acc, temporal_j, actual_step, gradient)
                trial = objective.evaluate(candidate)
                ratio = (cost - trial[0].cost) / predicted if predicted > 0 else -1.0
                decreased = bool(np.isfinite(trial[0].cost) and trial[0].cost < cost and ratio > 0)
            if (not decreased and conic_possible and shape_solves < MAX_SHAPE_SOLVES
                    and (candidate is None or alpha < .25)):
                shape_solves += 1
                proposal_tick = time.monotonic()
                if trace is not None:
                    trace.fallback("banded_infeasible" if candidate is None else "banded_alpha_below_0.25")
                bounded = attempts.conic(attempt, shape_solves)
                if trace is not None:
                    trace.add_operation("constraint_proposals",
                                        time.monotonic() - proposal_tick)
                    if bounded is None:
                        trace.fallback("conic_proposal_unsolved")
                if bounded is not None:
                    step, candidate, alpha = bounded
                    step_backend = getattr(step, "proposal_backend", "shape_conic")
                    conic = True
                    trial_damping = attempts.conic_damping(attempt)
            if candidate is None and not conic:
                if trace is not None:
                    trace.add_failed(time.monotonic() - candidate_tick,
                                     "constraint_proposal_unsolved")
                damping, growth = min(damping * growth, MAX_DAMPING), growth * 2
                continue
            if candidate is not None and (not checked_short or conic):
                actual_step = coords.difference(poses, candidate).ravel()
                predicted = _model_decrease(acc, temporal_j, actual_step, gradient)
                trial = objective.evaluate(candidate)
                ratio = (cost - trial[0].cost) / predicted if predicted > 0 else -1.0
                decreased = bool(np.isfinite(trial[0].cost) and trial[0].cost < cost and ratio > 0)
            if conic and not decreased:
                shortened = _shorten_conic(objective, poses, step, 1. if candidate is None else alpha,
                                           acc, temporal_j, gradient, cost,
                                           metric=(system, diagonal, trial_damping))
                if shortened is not None:
                    candidate, alpha, trial, ratio = shortened
                    decreased = True
            if decreased:
                if OPTIONS.accept_short_banded:
                    step_backend = "shape_conic" if conic else "banded"
                damping = max(trial_damping * max(1 / 3, 1 - (2 * ratio - 1) ** 3), MIN_DAMPING)
                growth, accepted = 2.0, True
                if trace is not None:
                    trace.add_alpha(alpha)
                    accepted_step_backend = getattr(step, "proposal_backend",
                                                    "shape_conic" if conic else "banded")
                break
            if trace is not None:
                reason = ("conic_shortening_failed" if conic else "objective_no_decrease")
                trace.add_failed(time.monotonic() - candidate_tick, reason)
            damping, growth = min(damping * growth, MAX_DAMPING), growth * 2
        if joint is not None:
            joint.finish_iteration(accepted)
        if not accepted:
            history.append(dict(phase=label, iteration=iteration, cost=cost, after=cost,
                                alpha=0.0, damping=damping, seconds=time.monotonic() - start, **stats))
            if trace is not None:
                trace.finish_iteration(history[-1], None)
            break
        poses = candidate
        acc, temporal_r, temporal_j, stats, _ = trial
        history.append(dict(phase=label, iteration=iteration, cost=cost, after=acc.cost,
                            alpha=alpha, damping=damping, step_backend=step_backend,
                            seconds=time.monotonic() - start, **stats))
        if trace is not None:
            trace.finish_iteration(history[-1], accepted_step_backend)
        if callback is not None:
            callback(poses.copy(), dict(history[-1]))
        cost = acc.cost
        slow, converged = _progress_converged(
            costs, cost, slow, alpha, conic_possible, converged_decrease, shortened=shortened is not None)
        if converged:
            break
    return poses, history
