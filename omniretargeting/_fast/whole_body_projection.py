"""Full-body original-geometry refinement with smooth, simultaneous corrections.

Source contact targets remain soft. No acquired contact position or historical
per-key depth becomes a constraint. Every step uses both bodies of self pairs,
root tangents and compensating joints, with fixed source layout/continuity.
"""
import time

import numpy as np
from scipy import sparse

from . import deadline_state
from .continuity import temporal_signals
from .deadline import DeadlineExceeded
from .kinematics import TrajectoryKinematics
from .mesh_repair import OriginalGeometry, repair_original_meshes as local_repair
from .repair_geometry_setup import prepare_static_bounds, prepare_moving_reduction
from .rows import _block_rows
from .projection_mechanics import MechanicalScreen
from ..motion_budgets import high_frequency, chatter_budget, CHATTER_EXCESS_TOLERANCE
from ..quality_policy import evaluate, checkpoint_key


def quality(measurement, frames, fps):
    """Physical pairs, merging query representations before applying policy V1.

    No unverified support exception. Unknown surrogate rows remain conservative
    optimization residuals; they never produce a collision-free certificate.
    """
    series = {}
    for key, value in measurement['depths'].items():
        pair = (key[1], key[2], key[-1]) if key[1] == 'moving' else key[1:]
        spec = series.setdefault(pair, dict(kind=key[1], depth=np.zeros(frames)))
        spec['depth'][key[0]] = max(spec['depth'][key[0]], value)
    return evaluate(series, fps)


def rank(measurement, frames, fps):
    result = quality(measurement, frames, fps)
    key = checkpoint_key(result, velocity_excess=0., chatter_excess=0.,
                         support_excess=0., control_excess=0., posture_excess=0., base_cost=0.)
    return key[2:5]


class Projection:
    def __init__(self, objective, entry):
        self.obj, self.entry = objective, entry.copy()
        self.direct = False
        self.focus = np.ones(len(entry), bool)
        self.coords = objective.coords
        self.frames, self.width = len(entry), objective.width
        self.kin = TrajectoryKinematics(objective.model, self.coords.cols)
        self.mechanics = MechanicalScreen(objective, entry)
        self.scales = np.full(self.width, .15)
        if self.coords.root_d is not None:
            self.scales[[self.coords.inverse[d] for d in range(self.coords.root_d, self.coords.root_d + 3)
                         if d in self.coords.inverse]] = .01
        self.flat_scale = np.tile(self.scales, self.frames)
        self.scale = sparse.diags(self.flat_scale)
        # Penalize only the extra correction's curvature, not source motion.
        d = sparse.diags([np.ones(self.frames - 2), -2 * np.ones(self.frames - 2),
                          np.ones(self.frames - 2)], [0, 1, 2],
                         shape=(self.frames - 2, self.frames))
        temporal = sparse.kron(d, sparse.eye(self.width), format='csc') / .015
        self.smooth = temporal.T @ temporal
        self.regularizer = self.smooth + sparse.eye(self.frames * self.width) * .1
        self.reference_signals = temporal_signals(self.coords, objective.reference)
        initial = temporal_signals(self.coords, entry)
        self.allowed = {name: np.maximum(chatter_budget(ref) + CHATTER_EXCESS_TOLERANCE,
                            np.abs(high_frequency(initial[name] - ref)))
                        for name, ref in self.reference_signals.items()}
        self.velocity_guards = [g for g in objective.candidate_guards
                                if type(g).__name__ == 'SequenceContinuityGuard']

    def rejection(self, trial):
        if not self.obj.source_shape.accepts(trial):
            return 'fixed_source_layout_or_root_envelope'
        if not all(g.accepts(trial) for g in self.velocity_guards):
            return 'source_continuity_or_absolute_velocity'
        signals = temporal_signals(self.coords, trial)
        for name, ref in self.reference_signals.items():
            excess = np.abs(high_frequency(signals[name] - ref)) - self.allowed[name]
            if np.any(excess > 2e-7):
                frame, column = np.unravel_index(np.argmax(excess), excess.shape)
                return dict(kind='added_chatter', signal=name, frame=int(frame),
                            coordinate=int(column), excess=float(excess[frame, column]))
        # A numerical neighborhood prevents one refinement from destroying the
        # recognized motion. It is fixed for this solve, never re-centered.
        if np.max(np.abs(trial[:, self.coords.qi] - self.entry[:, self.coords.qi])) > .4:
            return 'correction_trust_region'
        return self.mechanics.rejection(trial)

    def contact_rows(self, poses):
        self.kin.evaluate(poses)
        for surf in self.obj.surfaces:
            active = surf['active']
            if not active.any():
                continue
            points = self.kin.geom_points(surf['geoms'], surf['local'])
            jac = self.kin.point_jacobians(surf['bodies'], points)
            yield points - surf['targets'], jac, active
        for eff in self.obj.effectors:
            active = eff['active'] & ~eff['managed']
            if not active.any():
                continue
            bodies = np.full((self.frames, 1), eff['body'])
            points = self.kin.points(bodies, eff['offset'][:, None])
            jac = self.kin.point_jacobians(bodies, points)
            yield points - eff['contact'][:, None], jac, active[:, None]

    def step(self, current, geometry, measurement):
        rows = [(k, row) for k, row in measurement['rows'] if row['clearance'] < .008]
        jac, phi = geometry.linear_rows(dict(rows=rows))
        matrices, residuals = [jac @ self.scale / .003], [np.minimum(phi + .0015, 0.) / .003]
        for residual, jacobian, active in self.contact_rows(current):
            mask = np.repeat(active.ravel(), 3)
            matrix = _block_rows(jacobian.reshape(self.frames, -1, self.width), self.width)[mask]
            matrices.append(matrix @ self.scale / .012)
            residuals.append(residual[active].ravel() / .012)
        matrix, residual = sparse.vstack(matrices, format='csc'), np.concatenate(residuals)
        correction = self.coords.difference(self.entry, current).ravel() / self.flat_scale
        source_error = self.coords.difference(self.obj.reference, current).ravel() / self.flat_scale
        rhs = -matrix.T @ residual - self.smooth @ correction - .1 * source_error
        from .projection_step import constrained_direction
        return constrained_direction(self, current, matrix.T @ matrix + self.regularizer, rhs)


def refine(objective, poses, objects, scene):
    started = time.monotonic()
    geometry = OriginalGeometry(objective, objects, scene, manifold=True, columns=objective.coords.cols)
    geometry.uncertain_exit_rows = True
    prepare_static_bounds(geometry, poses)
    prepare_moving_reduction(geometry, poses)
    geometry.start_workers(len(poses))
    fixed_root = objective.fixed_root
    objective.fixed_root = None
    report = dict(method='smooth_full_body_original_geometry', root_fixed=False,
                  acquired_contact_hard_constraints=False, historical_pair_minima=False, history=[])
    state = deadline_state.state()
    if state is not None:
        # Keep the live ledger even if final diagnostics are skipped at expiry.
        state.report['whole_body_projection'] = report
    current = poses.copy()
    try:
        measured = geometry.measure(current, jacobians=True)
        report['before'] = geometry.summary(measured)
        projection = Projection(objective, poses)
        for iteration in range(12):
            deadline_state.check('whole_body_projection')
            old_rank = rank(measured, len(poses), objective.framerate)
            if old_rank[0] <= 0 and old_rank[2] <= 0:
                report['stop_reason'] = 'known_quality_satisfied'
                break
            tick = time.monotonic()
            if projection.direct and len(poses) >= 600:
                from .projection_windows import focus_frames
                projection.focus = focus_frames(quality(measured, len(poses), objective.framerate), len(poses))
            step = projection.step(current, geometry, measured)
            record = dict(iteration=iteration, trials=[], rank_before=old_rank, alpha=0.,
                          proposal=projection.step_status)
            if step is None:
                report['stop_reason'] = 'no_numerical_direction'
                report['history'].append(record)
                if not projection.direct:
                    projection.direct = True
                    record['refresh'] = 'direct_full_body_mechanical_model'
                    continue
                break
            for alpha in (1., .5, .25, .125, .0625):
                deadline_state.check('whole_body_projection_trial')
                trial = objective.project(objective.coords.integrate(current, step, alpha))
                reason = projection.rejection(trial)
                if reason:
                    record['trials'].append(dict(alpha=alpha, rejected=reason))
                    continue
                candidate = geometry.measure(trial, jacobians=True)
                candidate_rank = rank(candidate, len(poses), objective.framerate)
                record['trials'].append(dict(alpha=alpha, rank=candidate_rank,
                                             unknown=len(candidate['failures'])))
                # Final-quality aggregate, never a per-key monotonic restriction.
                if candidate_rank < old_rank:
                    current, measured = trial, candidate
                    record['alpha'] = alpha
                    deadline_state.accepted(current, 'original_mesh_repair')
                    break
            record['seconds'] = time.monotonic() - tick
            report['history'].append(record)
            if not record['alpha']:
                if not projection.direct:
                    projection.direct = True
                    record['refresh'] = 'direct_full_body_mechanical_model'
                    continue
                report['stop_reason'] = 'no_feasible_quality_progress'
                break
        else:
            report['stop_reason'] = 'iteration_limit'
        report['after'] = geometry.summary(measured)
    except DeadlineExceeded:
        report['stop_reason'] = 'deadline'
        if 'measured' in locals():
            report['after'] = geometry.summary(measured)
    finally:
        objective.fixed_root = fixed_root
        geometry.stop_workers()
    report['seconds'] = time.monotonic() - started
    report['root_shift_max_m'] = float(np.linalg.norm(current[:, :3] - poses[:, :3], axis=1).max())
    return current, report


def repair_original_meshes(objective, poses, scene_objects=(), scene_query=None):
    """Retain the incumbent legacy solve, then release its intermediate locks."""
    current, report = local_repair(objective, poses, scene_objects, scene_query)
    if len(poses) < 3 or not len(objective.coords.qi):
        return current, report
    deadline_state.accepted(current, 'original_mesh_repair')
    candidate, full = refine(objective, current, scene_objects, scene_query)
    report['local_status'] = report.get('status')
    if not np.array_equal(candidate, current):
        report['status'] = 'corrected_full_body'
    report['whole_body_projection'] = full
    report['after_local'] = report.get('after')
    report['after'] = full.get('after', report.get('after'))
    report['seconds'] += full['seconds']
    return candidate, report
