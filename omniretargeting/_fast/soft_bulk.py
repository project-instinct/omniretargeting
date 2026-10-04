"""Private soft-guard acquisition bulk phase; never a final feasible output.

Whole-trajectory LM uses parallel original frame terms, frozen continuity
penalties, boxes/root domains, full steps and damping only. No guard predicate,
conic solve, restoration, speculative attempt or fraction fan runs here.
"""

from contextlib import contextmanager
import copy
import time

import numpy as np
from scipy.linalg import LinAlgError

from .barrier_assembly import equilibrated_solve, physical_system
from .soft_rows import SoftTemporalRows

MODEL_ROUNDS = 1
MAX_ATTEMPTS = 12
SPEED_GATE_SECONDS = 2.
SPEED_GATE_SAMPLES = 3


class SoftObjective:
    _soft_bulk = True
    source_shape = None
    candidate_guards = ()
    priority_clearance = None

    def __init__(self, objective, poses, report, *, speed_gate=True, support_weights=None,
                 continuity_stiffness=100., source_yield=False, world_recovery=False):
        self.owner = self.global_objective = objective
        self.span = slice(0, objective.frames)
        self.rows = SoftTemporalRows(objective, poses, stiffness=continuity_stiffness)
        self.report, self.speed_gate = report, speed_gate
        self.contacts = None
        self.recovery = None
        if support_weights is not None:
            from .soft_support_rows import SupportRows
            self.contacts = SupportRows(objective, support_weights, yield_targets=source_yield)
            report['support_rows'] = self.contacts.diagnostics
        if world_recovery:
            from .soft_swing_rows import SwingRows
            # Added leg recovery belongs to swing phases. The original world
            # tracking term remains present during support, below contacts.
            self.recovery = SwingRows(objective, include_contacts=False)
            report['world_target_recovery'] = self.recovery.report

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def evaluate(self, poses, with_jacobian=True):
        tick = time.monotonic()
        value = self.owner.evaluate(poses, with_jacobian)
        multiplier = self.report.get('fitting_multiplier', 1.)
        if multiplier != 1.:
            acc, residual, temporal, stats, signals = value
            original = self.physical_cost(value)
            clearance = signals['clearance_cost']
            acc.cost *= multiplier
            acc.frame_cost *= multiplier
            if with_jacobian:
                acc.hessian *= multiplier
                acc.gradient *= multiplier
            signals = dict(signals, guard_cost=signals['guard_cost'] * multiplier,
                priority_extra=((multiplier - 1.) * (original - clearance)
                                + (self.report['clearance_multiplier'] - 1.) * clearance))
            value = (acc, residual * np.sqrt(multiplier),
                     temporal * np.sqrt(multiplier) if temporal is not None else None, stats, signals)
        if self.contacts is not None:
            from scipy import sparse
            contact_tick = time.monotonic()
            residual, matrix, stats = self.contacts.evaluate(poses, with_jacobian)
            acc, original, temporal, original_stats, signals = value
            cost = float(residual @ residual)
            acc.cost += cost
            signals = dict(signals, priority_extra=signals.get('priority_extra', 0.) + cost,
                           support_cost=cost)
            value = (acc, np.r_[original, residual], sparse.vstack([temporal, matrix], format='csr'),
                     dict(original_stats, **stats), signals)
            self.report['support_evaluation_seconds'] = (self.report.get('support_evaluation_seconds', 0.)
                                                         + time.monotonic() - contact_tick)
        if self.recovery is not None:
            from scipy import sparse
            residual, matrix, recovery_stats = self.recovery.evaluate(poses, with_jacobian)
            acc, original, temporal, original_stats, signals = value
            cost = float(residual @ residual)
            acc.cost += cost
            signals = dict(signals, priority_extra=signals.get('priority_extra',0.)+cost,
                           world_recovery_cost=cost)
            value = (acc,np.r_[original,residual],sparse.vstack([temporal,matrix],format='csr'),
                     dict(original_stats,world_target_recovery=recovery_stats),signals)
        self.report['evaluation_seconds'] += time.monotonic() - tick
        self.report['evaluations'] += 1
        key = 'jacobian_evaluations' if with_jacobian else 'value_evaluations'
        self.report[key] = self.report.get(key, 0) + 1
        return value

    @staticmethod
    def physical_cost(evaluation):
        return evaluation[0].cost - evaluation[4]['guard_cost'] - evaluation[4].get('priority_extra', 0.)

    @staticmethod
    def acceptance_cost(evaluation):
        return evaluation[0].cost - evaluation[4]['guard_cost']

    def penalty_lower_bound(self, poses, diagonal):
        """Reject expensive evaluations only when nonnegative penalties suffice.

        This evaluates soft rows, not acceptance predicates. Original C is
        nonnegative, so the bound cannot reject a decreasing merit trial.
        """
        from .rows import _Accumulator
        from .soft_frame_rows import add_layout
        kin = self.owner.window_kinematics(poses, range(len(poses)), with_jacobian=False)
        local = copy.copy(self.owner)
        local.kin = kin
        local.frame_slice = slice(0, len(poses))
        acc = _Accumulator(len(poses), self.width, False)
        add_layout(local, acc, {})
        signals = dict(root_point=local._root_points(), root_jacobian=None)
        penalty = self.rows.accumulate(poses, signals, diagonal)[0]
        self.report['penalty_value_evaluations'] += 1
        return acc.cost + penalty


@contextmanager
def bulk_context(objective, poses, report, *, speed_gate=False, guard_stiffness=1.,
                 clearance_multiplier=1., support_weights=None, continuity_stiffness=100., source_yield=False,
                 world_recovery=False):
    """Restore original semantics/pool even on a numerical failure or exception."""
    facade = SoftObjective(objective, poses, report, speed_gate=speed_gate, support_weights=support_weights,
                           continuity_stiffness=continuity_stiffness, source_yield=source_yield,
                           world_recovery=world_recovery)
    marker = object()
    fields = ('_soft_bulk_signals', '_soft_bulk_layouts', '_soft_guard_stiffness',
              '_soft_clearance_multiplier', 'evaluation_backend')
    saved = {name: getattr(objective, name, marker) for name in fields}
    was_running = getattr(objective, 'pool', None) is not None
    objective.stop_workers()
    try:
        objective._soft_bulk_signals = True
        objective._soft_bulk_layouts = list(objective.source_shape.envelopes)
        objective._soft_guard_stiffness = guard_stiffness
        objective._soft_clearance_multiplier = clearance_multiplier
        objective.evaluation_backend = 'cpu'
        tick = time.monotonic()
        objective.start_workers()
        report.update(pool_start_seconds=time.monotonic() - tick,
                      workers=len(objective.pool._pool) if objective.pool is not None else 1)
        yield facade
    finally:
        objective.stop_workers()
        for name, value in saved.items():
            if value is marker:
                objective.__dict__.pop(name, None)
            else:
                setattr(objective, name, value)
        if was_running:
            objective.start_workers()


def optimize_bulk(objective, poses, iterations, callback, label, threshold):
    from .barrier_solver import _equalities, _radial
    from .solver import (_active_bounds, _banded_solve, _hold, _hold_radial,
                         INITIAL_DAMPING, MIN_DAMPING, MAX_DAMPING, DIAGONAL_FLOOR)
    report = objective.report
    initial_tick = time.monotonic()
    evaluation = objective.evaluate(poses)
    report['initial_evaluation_seconds'] = time.monotonic() - initial_tick
    physical_cost = objective.physical_cost(evaluation)
    acceptance_cost = objective.acceptance_cost(evaluation)
    report['initial_cost'] = physical_cost
    report['initial_stats'] = dict(evaluation[3])
    history, slow, damping = [], 0, report.get('initial_damping', INITIAL_DAMPING)
    report['stop_reason'] = 'bulk_iteration_cap'
    for iteration in range(iterations):
        tick = time.monotonic()
        counters = {key: report[key] for key in ('evaluation_seconds', 'evaluations', 'factorizations')}
        assembly_seconds = solve_seconds = 0.
        if not evaluation[0].jacobian:
            evaluation = objective.evaluate(poses, True)
        assembly_tick = time.monotonic()
        acc, residual, temporal, stats, signals = evaluation
        physical = physical_system(acc, temporal)
        physical_gradient = acc.gradient.ravel() + temporal.T @ residual
        diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), DIAGONAL_FLOOR)
        blocks = list(objective.rows.blocks(poses, signals, True))
        penalty, _, penalty_stats = objective.rows.accumulate(poses, signals, diagonal, blocks=blocks)
        merit = acc.cost + penalty
        assembly_seconds += time.monotonic() - assembly_tick
        accepted = None
        for _ in range(report.get('max_attempts', MAX_ATTEMPTS)):
            direction = np.zeros(len(physical_gradient))
            try:
                for _ in range(MODEL_ROUNDS):
                    model_tick = time.monotonic()
                    system = physical.copy(order='F')
                    _, model_gradient, _ = objective.rows.accumulate(
                        poses, signals, diagonal, system, blocks=blocks, direction=direction)
                    gradient = physical_gradient + model_gradient
                    held = np.unique(np.r_[_active_bounds(objective, poses, gradient),
                                          _equalities(objective, poses)])
                    rhs = -gradient
                    _hold_radial(system, objective.width, _radial(objective, poses, gradient))
                    system[0] += damping * diagonal + DIAGONAL_FLOOR
                    _hold(system, held)
                    rhs[held] = 0.
                    assembly_seconds += time.monotonic() - model_tick
                    report['factorizations'] += 1
                    solve_tick = time.monotonic()
                    direction = equilibrated_solve(system, rhs, _banded_solve)
                    solve_seconds += time.monotonic() - solve_tick
            except LinAlgError:
                report['numerical_failures'] += 1
                damping = min(damping * 5., MAX_DAMPING)
                continue
            raw = objective.coords.integrate(poses, direction.reshape(objective.frames, objective.width), 1.)
            candidate = objective.project(raw)
            trial = objective.evaluate(candidate, False)
            model_tick = time.monotonic()
            trial_penalty, _, trial_penalty_stats = objective.rows.accumulate(
                candidate, trial[4], diagonal)
            assembly_seconds += time.monotonic() - model_tick
            after_merit = trial[0].cost + trial_penalty
            candidate_cost = objective.physical_cost(trial)
            # Private full steps are judged by actual physical progress. Soft
            # guards guide the banded direction; the final projection owns
            # feasibility. A numerical penalty increase cannot veto C decrease.
            candidate_acceptance = objective.acceptance_cost(trial)
            families = ('self_depth_m',) if report.get('self_monotone', False) else ()
            if report.get('clearance_monotone', False):
                families = ('scene_depth_m', 'self_depth_m', 'plane_depth_m', 'moving_depth_m')
            clearance_ok = all(trial[3].get(key, 0.) <= stats.get(key, 0.) for key in families)
            physical_ok = not report.get('physical_descent', False) or candidate_cost < physical_cost
            if np.isfinite(candidate_acceptance) and candidate_acceptance < acceptance_cost and clearance_ok and physical_ok:
                accepted = candidate, trial, trial_penalty_stats, after_merit
                break
            report.setdefault('rejected_trials', []).append(dict(iteration=iteration,
                damping=damping, cost=candidate_cost, previous_cost=physical_cost,
                physical_decrease=candidate_cost<physical_cost,
                physical_descent_required=report.get('physical_descent',False),
                clearance_nonregressing=clearance_ok,
                depths={key: trial[3].get(key, 0.) for key in families}))
            report['merit_rejections'] += 1
            damping = min(damping * 5., MAX_DAMPING)
        seconds = time.monotonic() - tick
        timing = dict(assembly_seconds=assembly_seconds, solve_seconds=solve_seconds,
            evaluation_seconds=report['evaluation_seconds'] - counters['evaluation_seconds'],
            evaluations=report['evaluations'] - counters['evaluations'],
            factorizations=report['factorizations'] - counters['factorizations'])
        if accepted is None:
            report['stop_reason'] = 'physical_cost_stall'
            history.append(dict(phase=label, iteration=iteration, alpha=0., cost=physical_cost,
                after=physical_cost, merit=merit, after_merit=merit, iteration_seconds=seconds, **timing, **stats))
            break
        previous_cost = physical_cost
        previous_acceptance = acceptance_cost
        poses, evaluation, penalty_stats, after_merit = accepted
        physical_cost = objective.physical_cost(evaluation)
        acceptance_cost = objective.acceptance_cost(evaluation)
        damping = max(damping * report.get('damping_decay', .5), report.get('damping_floor', MIN_DAMPING))
        row = dict(phase=label, iteration=iteration, alpha=1., cost=previous_cost, after=physical_cost,
            merit=merit, after_merit=after_merit, damping=damping,
            acceptance='priority_full_step',
            priority_cost=acceptance_cost, clearance_cost=evaluation[4].get('clearance_cost', 0.),
            support_cost=evaluation[4].get('support_cost', 0.),
            iteration_seconds=seconds, step_backend='soft_banded', **timing, **evaluation[3], **penalty_stats)
        history.append(row)
        if callback is not None:
            callback(poses.copy(), dict(row))
        decrease = (previous_acceptance - acceptance_cost) / max(previous_acceptance, 1e-12)
        slow = slow + 1 if decrease < threshold else 0
        if slow >= report.get('plateau_steps', 2):
            report['stop_reason'] = 'priority_cost_plateau'
            report['plateau_iterations'] = iteration + 1
            break
        if objective.speed_gate and len(history) >= SPEED_GATE_SAMPLES:
            median = float(np.median([r['iteration_seconds'] for r in history[-SPEED_GATE_SAMPLES:]]))
            if median > SPEED_GATE_SECONDS:
                report.update(stop_reason='bulk_speed_gate', speed_gate_median_seconds=median)
                break
    report.update(final_cost=physical_cost, final_damping=damping, history=history,
        iteration_seconds=[row['iteration_seconds'] for row in history],
        correction_rms=float(np.sqrt(np.mean(objective.coords.difference(objective.reference, poses)**2))))
    return poses, history


def run_bulk(objective, poses, *, iterations=20, callback=None, threshold=.001,
             speed_gate=False, guard_stiffness=1., initial_damping=None,
             clearance_multiplier=1., damping_floor=None, damping_decay=.5, self_monotone=False,
             support_weights=None, continuity_stiffness=100., fitting_multiplier=1., plateau_steps=2,
             max_attempts=MAX_ATTEMPTS, source_yield=False, world_recovery=False,
             clearance_monotone=False, physical_descent=False):
    """Prototype API; returned poses must be projected/certified before use."""
    from .solver import _optimize
    report = dict(mode='soft_bulk_only', evaluations=0, evaluation_seconds=0., factorizations=0,
                  merit_rejections=0, numerical_failures=0, plateau_iterations=None,
                  exact_guard_calls=0, conic_calls=0, restoration_calls=0)
    report.update(penalty_value_evaluations=0, penalty_rejections=0)
    if not np.isfinite(guard_stiffness) or guard_stiffness <= 0:
        raise ValueError('Soft numerical guard stiffness must be positive')
    report['guard_stiffness'] = guard_stiffness
    if not isinstance(plateau_steps, int) or plateau_steps < 1:
        raise ValueError('Plateau steps must be a positive integer')
    report['plateau_steps'] = plateau_steps
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError('LM attempts must be a positive integer')
    report['max_attempts'] = max_attempts
    if not np.isfinite(fitting_multiplier) or not 0 < fitting_multiplier <= 1.:
        raise ValueError('Private fitting multiplier must be in (0, 1]')
    report['fitting_multiplier'] = fitting_multiplier
    if not np.isfinite(continuity_stiffness) or continuity_stiffness <= 0:
        raise ValueError('Numerical continuity stiffness must be finite and positive')
    report['continuity_stiffness'] = continuity_stiffness
    if not np.isfinite(clearance_multiplier) or clearance_multiplier < 1.:
        raise ValueError('Clearance multiplier must be finite and at least one')
    report['clearance_multiplier'] = clearance_multiplier
    if damping_floor is not None:
        if not np.isfinite(damping_floor) or damping_floor <= 0:
            raise ValueError('Damping floor must be finite and positive')
        report['damping_floor'] = damping_floor
    if not 0 < damping_decay <= 1:
        raise ValueError('Damping decay must be in (0, 1]')
    report['damping_decay'] = damping_decay
    report['self_monotone'] = self_monotone
    report['clearance_monotone'] = clearance_monotone
    report['physical_descent'] = physical_descent
    if initial_damping is not None:
        if not np.isfinite(initial_damping) or initial_damping <= 0:
            raise ValueError('Initial LM damping must be positive and finite')
        report['initial_damping'] = initial_damping
    tick = time.monotonic()
    with bulk_context(objective, poses, report, speed_gate=speed_gate,
                      guard_stiffness=guard_stiffness, clearance_multiplier=clearance_multiplier / fitting_multiplier,
                      support_weights=support_weights, continuity_stiffness=continuity_stiffness,
                      source_yield=source_yield, world_recovery=world_recovery) as facade:
        bulk, history = _optimize(facade, poses.copy(), iterations, callback, 'soft_bulk', threshold)
    report['seconds'] = time.monotonic() - tick
    return bulk, history, report
