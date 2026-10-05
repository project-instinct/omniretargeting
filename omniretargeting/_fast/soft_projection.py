"""One final projection of private bulk motion onto immutable acquisition guards."""

import time

import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter1d

from .box_step import direction_bounds
from .constrained_rows import ConstrainedRows
from .legacy_window_qp import WindowObjective, guard_failures
from .soft_projection_qp import ProjectionFailure, closest, family_program

ROOT_ROUNDS = 8
SHAPE_ROUNDS = 3
FEASIBILITY_ROUNDS = 7


def project_joints(objective, bulk, *, report):
    """Intersect both executed-difference families, independently per joint."""
    from .continuity import CorrectionContinuity, ROUND_OFF
    from .._sequence.continuity_guard import SequenceContinuityGuard, ACCEPTANCE_SLACK
    tick = time.monotonic()
    frames, coords = len(bulk), objective.coords
    bounds = {}
    operators = {k: sparse.diags({1: [-1., 1.], 2: [1., -2., 1.], 3: [-1., 3., -3., 1.]}[k],
        np.arange(k + 1), shape=(frames - k, frames), format='csr')
        for k in (1, 2, 3) if frames > k}
    for guard in objective.candidate_guards:
        if isinstance(guard, CorrectionContinuity):
            limits = {k: guard.bounds[f'joints_order_{k}']
                      for k in operators if f'joints_order_{k}' in guard.bounds}
            fixed, tolerance = guard.fixed[:, coords.qi], ROUND_OFF
        elif isinstance(guard, SequenceContinuityGuard):
            limits = {k: guard.bounds['joints', k]
                      for k in operators if ('joints', k) in guard.bounds}
            fixed, tolerance = guard.reference['joints'][0], ACCEPTANCE_SLACK
        else:
            continue
        for order, limit in limits.items():
            center = np.diff(fixed, n=order, axis=0) if guard.correction else np.zeros_like(limit)
            low, high = center - limit - tolerance / 2, center + limit + tolerance / 2
            if order in bounds:
                old_low, old_high = bounds[order]
                low, high = np.maximum(low, old_low), np.minimum(high, old_high)
            bounds[order] = low, high
    lower = np.broadcast_to(objective.lower, (frames, len(coords.qi)))
    upper = np.broadcast_to(objective.upper, lower.shape)
    matrix = sparse.vstack([operators[k] for k in bounds] + [sparse.eye(frames)], format='csr')
    result = bulk.copy()
    for joint, qi in enumerate(coords.qi):
        coupled = report.get('_contact_program')
        if coupled is not None and not coupled.enabled[joint]:
            # These coordinates affect a nonlinear root signal. A second
            # closest-point solve can move them by solver roundoff and lose
            # the already certified root-anchor continuity intersection.
            continue
        lo = np.concatenate([bounds[k][0][:, joint] for k in bounds] + [lower[:, joint]])
        hi = np.concatenate([bounds[k][1][:, joint] for k in bounds] + [upper[:, joint]])
        metric = report.get('_metric')
        weights = None if metric is None else metric[:, coords.scalar_cols[joint]]
        quadratic = linear = None
        if coupled is not None:
            quadratic, linear = coupled.program(joint, result)
            if callable(getattr(coupled,'boxes',None)):
                layout_low,layout_high=coupled.boxes(joint,result)
                lo[-frames:]=np.maximum(lo[-frames:],layout_low)
                hi[-frames:]=np.minimum(hi[-frames:],layout_high)
        value = closest(bulk[:, qi], matrix, lo, hi, weights=weights, report=report,
                        quadratic=quadratic, linear=linear)
        result[:, qi] = np.clip(value, lower[:, joint], upper[:, joint])
    report['joint_seconds'] = time.monotonic() - tick
    return result


def temporal_families(objective, poses, jacobian):
    window = WindowObjective(objective, poses, range(len(poses)))
    rows = ConstrainedRows(window)
    rows.guards = [g for g in rows.guards if g[0] in ('sequence', 'correction')]
    return rows.evaluate(poses, jacobian)


def root_families(objective, poses, jacobian):
    return [f for f in temporal_families(objective, poses, jacobian)
            if f.label[0] == 'RootDomain' or f.label[1] != 'joints']


def certify(objective, poses):
    from .barrier_solver import _domains
    failures = guard_failures(objective, poses, range(len(poses)))
    if not _domains(WindowObjective(objective, poses, range(len(poses))), poses):
        failures.append('root_or_box_domains')
    return failures


def project_roots(objective, poses, target, fixed, *, report):
    tick = time.monotonic()
    coords, width = objective.coords, objective.width
    root_cols = [coords.inverse[d] for d in range(coords.root_d or 0, (coords.root_d or 0) + 6)
                 if coords.root_d is not None and d in coords.inverse]
    if not root_cols:
        report['root_seconds'] = 0.
        return poses
    coupled = False
    curvature = {}
    for iteration in range(ROOT_ROUNDS):
        roots = root_families(objective, poses, True)
        if all(np.all(f.slack >= 0) for f in roots):
            break
        selected = root_cols
        families = roots
        if coupled:
            # The anatomical anchor depends on torso/waist joints as well as
            # the floating root. Retain those exact Jacobian columns and ALL
            # their frozen joint stencils, instead of an infeasible root-only
            # program followed by a destructive reset to the entry path.
            columns = np.unique(np.concatenate([f.matrix.indices % width for f in roots]))
            selected = sorted(set(root_cols) | set(columns))
            families = temporal_families(objective, poses, True)
        indices = (np.arange(len(poses))[:, None] * width + selected).ravel()
        matrix, lo, hi, balls = family_program(families, indices, curvature=curvature)
        from .barrier_solver import _equalities
        window = WindowObjective(objective, poses, range(len(poses)))
        lower, upper = direction_bounds(window, poses, _equalities(window, poses))
        matrix = sparse.vstack((matrix, sparse.eye(len(indices))), format='csr')
        lo, hi = np.r_[lo, lower[indices]], np.r_[hi, upper[indices]]
        desired = coords.difference(poses, target).ravel()[indices]
        scales = np.tile([.03 if c in root_cols[:3] else .1 for c in selected], len(poses))
        try:
            metric = report.get('_metric')
            weights = None if metric is None else metric.ravel()[indices]
            system = report.get('_quadratic')
            quadratic = None if system is None else system[indices][:,indices]
            linear = None if system is None else -system[indices] @ coords.difference(poses,target).ravel()
            step = closest(desired, matrix, lo, hi, balls=balls, report=report,
                           variable_scale=scales, weights=weights, quadratic=quadratic, linear=linear)
        except ProjectionFailure as error:
            report.setdefault('root_failures', []).append(dict(iteration=iteration,
                status=str(error), coupled=coupled))
            if not coupled:
                coupled = True
                continue
            break
        full = np.zeros((len(poses), width))
        full.ravel()[indices] = step
        candidate = objective.project(coords.integrate(poses, full, 1.))
        following = root_families(objective, candidate, False)
        for family, actual in zip(roots, following):
            if family.ball:
                predicted = family.residual + (family.matrix @ full.ravel()).reshape(family.residual.shape)
                error = np.linalg.norm(actual.residual, axis=1) - np.linalg.norm(predicted, axis=1)
            else:
                error = family.slack + family.matrix @ full.ravel() - actual.slack
            for i in np.flatnonzero(error > 0):
                curvature[family.identity(int(i))] = 1.5 * float(error[i])
        poses = candidate
        report['root_rounds'] = iteration + 1
        report.setdefault('root_round_details', []).append(dict(iteration=iteration,
            coupled=coupled, columns_per_frame=len(selected), violated=sum(int((f.slack < 0).sum()) for f in following),
            binding=[dict(label=f.label, slack=float(f.slack.min())) for f in following if np.any(f.slack < 0)]))
    if any(np.any(f.slack < 0) for f in root_families(objective, poses, False)):
        q = coords.root_q
        poses[:, q:q + 7] = fixed[:, q:q + 7]
        report['root_fallback'] = 'immutable_feasible_entry'
    report['root_seconds'] = time.monotonic() - tick
    return poses


def bad_geometry(objective, poses):
    bad = np.zeros(len(poses), bool)
    needed = []
    for guard in [objective.source_shape, *objective.candidate_guards]:
        if type(guard).__name__ in ('ReferenceShape', 'AcquisitionNormalGuard'):
            mask = guard.violation_frames(poses)
            if mask.any():
                needed.append(type(guard).__name__)
                bad |= mask
    return bad, needed


def regions(mask):
    values = np.flatnonzero(mask)
    if not len(values):
        return []
    split = np.flatnonzero(np.diff(values) > 1) + 1
    return [range(int(part[0]), int(part[-1]) + 1) for part in np.split(values, split)]


def feasibility_merit(families):
    """Numerical merit for private SQP trials, never an acceptance budget."""
    paired = {f.label:f.slack for f in families if f.label[-1] in ('upper', 'lower')}
    value = 0.
    for f in families:
        if f.ball:
            radius = np.sqrt(np.maximum(f.slack + np.square(f.residual).sum(axis=1), 0.))
            gap = radius - np.linalg.norm(f.residual, axis=1)
            units = np.maximum(radius, 2e-7)
        else:
            gap = f.slack
            opposite = (*f.label[:-1], 'lower' if f.label[-1] == 'upper' else 'upper')
            units = np.maximum(.5*(gap + paired[opposite]), 2e-7) if opposite in paired else .01
        error = np.maximum(-gap, 0.) / units
        value += float(error @ error)
    return value


def region_direction(window, poses, target, families, columns, curvature, record, report):
    """Working faces include every reached affine layout/contact constraint."""
    from .barrier_solver import _equalities
    lower, upper = direction_bounds(window, poses[window.span], _equalities(window, poses[window.span]))
    desired = window.coords.difference(poses[window.span], target[window.span]).ravel()[columns]
    metric = report.get('_metric')
    weights = None if metric is None else metric[window.span].ravel()[columns]
    quadratic = linear = None
    if report.get('_physical_round'):
        from .soft_projection_physical import model
        quadratic, linear, cost, stats = model(window, poses[window.span], target[window.span], columns, report)
        record['physical_cost_before'] = cost
    scales = np.tile([.03 if c in range(3) else .1 for c in range(window.width)], window.frames)[columns]
    for cutting_round in range(4):
        matrix, lo, hi, balls = family_program(families, columns, screen=True,
            active=record['active'], curvature=curvature)
        matrix = sparse.vstack((matrix, sparse.eye(len(columns))), format='csr')
        lo, hi = np.r_[lo, lower[columns]], np.r_[hi, upper[columns]]
        solve = closest
        if report.get('_physical_round') and report.get('linear_physical_qp'):
            from .soft_projection_linear_qp import solve
        direction = solve(desired, matrix, lo, hi, balls=balls, report=report,
            variable_scale=scales, weights=weights, quadratic=quadratic, linear=linear)
        full = np.zeros(window.frames*window.width)
        full[columns] = direction
        added = 0
        for f in families:
            if f.label[0] not in ('ReferenceShape','AcquisitionNormalGuard'):
                continue
            if f.ball:
                radius = np.sqrt(np.maximum(f.slack+np.square(f.residual).sum(axis=1),0.))
                gap = radius-np.linalg.norm(f.residual+(f.matrix@full).reshape(f.residual.shape),axis=1)
            else:
                gap = f.slack+f.matrix@full
            for i in np.flatnonzero(gap < 5e-8):
                identity = f.identity(int(i))
                if identity not in record['active']:
                    record['active'].add(identity)
                    added += 1
        if not added:
            break
    record['linear_cut_rounds'] = record.get('linear_cut_rounds',0)+cutting_round+1
    return full, matrix.shape[0], sum(len(b[2]) for b in balls)


def project_region(objective, poses, target, frames, *, report):
    record = dict(frames=[frames.start, frames.stop], rounds=[], active=set())
    curvature = {}
    root_released = False
    cleanup_target = None
    for iteration in range(SHAPE_ROUNDS + FEASIBILITY_ROUNDS):
        window = WindowObjective(objective, poses, frames)
        rows = ConstrainedRows(window)
        families = rows.evaluate(poses[window.span], True)
        if all(np.all(f.slack >= 0) for f in families):
            break
        columns = (np.arange(window.frames * window.width) if root_released else
                   (np.arange(window.frames)[:, None] * window.width + window.coords.scalar_cols).ravel())
        report['_physical_round'] = report.get('objective_aware', False) and iteration < SHAPE_ROUNDS
        if report.get('objective_aware') and iteration >= SHAPE_ROUNDS and cleanup_target is None:
            cleanup_target = poses.copy()
        aim = target if cleanup_target is None else cleanup_target
        try:
            full_step, interval_rows, ball_count = region_direction(window, poses, aim,
                families, columns, curvature, record, report)
        except ProjectionFailure as error:
            if not root_released:
                root_released = True
                record['root_release_reason'] = str(error)
                continue
            if curvature:
                curvature.clear()
                record.setdefault('curvature_resets', 0)
                record['curvature_resets'] += 1
                continue
            record['failure'] = str(error)
            break
        before_merit = feasibility_merit(families)
        fraction = 1.
        for _ in range(9):
            candidate = window.project(window.coords.integrate(poses[window.span],
                full_step.reshape(window.frames, window.width), fraction))
            next_families = rows.evaluate(candidate, False)
            after_merit = feasibility_merit(next_families)
            if after_merit < before_merit or after_merit == 0.:
                break
            fraction *= .5
        else:
            record['failure'] = 'nonlinear_feasibility_stall'
            break
        actual_step = window.coords.difference(poses[window.span], candidate).ravel()
        for family, following in zip(families, next_families):
            if len(family.label) > 1 and family.label[1] == 'joints':
                continue
            if family.ball:
                predicted = family.residual + (family.matrix @ actual_step).reshape(family.residual.shape)
                error = np.linalg.norm(following.residual, axis=1) - np.linalg.norm(predicted, axis=1)
            else:
                error = family.slack + family.matrix @ actual_step - following.slack
            for i in np.flatnonzero(error > 0):
                curvature[family.identity(int(i))] = 1.5 * float(error[i])
        poses = window.embed(candidate)
        failures = guard_failures(objective, poses, frames)
        detail = dict(iteration=iteration, failures=failures, variables=len(columns),
            interval_rows=interval_rows, balls=ball_count, alpha=fraction,
            before_feasibility_merit=before_merit, after_feasibility_merit=after_merit)
        if report.get('_physical_round'):
            from .soft_projection_physical import value
            cost, stats = value(window, candidate, report)
            detail.update(physical_cost_before=record.pop('physical_cost_before'),
                          physical_cost_after=cost, physical_stats=stats)
        record['rounds'].append(detail)
        if not failures:
            break
    record['active_faces'] = len(record.pop('active'))
    record['root_released'] = root_released
    remaining = rows.evaluate(poses[window.span], False)
    record['remaining'] = [dict(label=f.label, minimum_slack=float(f.slack.min()),
        identity=f.identity(int(np.argmin(f.slack)))) for f in remaining if np.any(f.slack < 0)]
    return poses, record

def project_regions(objective, poses, target, *, report):
    tick = time.monotonic()
    bad, needed = bad_geometry(objective, poses)
    report.update(geometry_guards=needed, geometry_violation_frames=int(bad.sum()), regions=[])
    plans = regions(maximum_filter1d(bad, size=13, mode='constant'))
    if report.get('parallel_projection'):
        from .soft_projection_parallel import project_parallel
        return project_parallel(objective, poses, target, plans, report=report)
    for frames in plans:
        poses, record = project_region(objective, poses, target, frames, report=report)
        report['regions'].append(record)
    report['geometry_seconds'] = time.monotonic() - tick
    report.pop('_physical_round', None)
    report['geometry_projection'] = 'local_joint_first_with_root_release'
    return poses


def project_bulk(objective, bulk, fixed, *, measure_costs=False, objective_aware=False,
                 physical_majorizer=False, parallel_projection=False, linear_physical_qp=False, physical_first_order=False):
    if not measure_costs:
        return _project_bulk(objective, bulk, fixed, objective_aware=objective_aware,
                             physical_majorizer=physical_majorizer, parallel_projection=parallel_projection, linear_physical_qp=linear_physical_qp, physical_first_order=physical_first_order)
    from .soft_projection_metrics import measurements
    started = time.monotonic()
    with measurements(objective, bulk) as tracker:
        candidate, report = _project_bulk(objective, bulk, fixed, tracker=tracker,
            objective_aware=objective_aware, physical_majorizer=physical_majorizer, parallel_projection=parallel_projection, linear_physical_qp=linear_physical_qp, physical_first_order=physical_first_order)
        report.update(cost_by_guard_family=tracker.history,
                      projection_cost_seconds=tracker.seconds)
    report['seconds'] = time.monotonic() - started
    return candidate, report


def _project_bulk(objective, bulk, fixed, tracker=None, objective_aware=False, physical_majorizer=False, parallel_projection=False, linear_physical_qp=False, physical_first_order=False):
    """Only a fully exact-certified result leaves this projection transaction."""
    started = time.monotonic()
    if certify(objective, fixed):
        raise ProjectionFailure('The immutable entry is not guard feasible')
    report = dict(qp_solves=0, qp_seconds=0., qp_statuses=[], root_rounds=0,
                  objective_aware=objective_aware, physical_majorizer=physical_majorizer, parallel_projection=parallel_projection, linear_physical_qp=linear_physical_qp, physical_first_order=physical_first_order)
    if tracker is not None:
        report['_metric'] = tracker.metric
        report['_quadratic'] = tracker.system
    if np.array_equal(bulk, fixed):
        report.update(seconds=time.monotonic() - started, joint_seconds=0., root_seconds=0.,
            geometry_seconds=0., geometry_guards=[], geometry_violation_frames=0, regions=[],
            exact_guard_failures=[], fixed_to_bulk_rms=0., bulk_to_projected_rms=0.,
            retained_correction_fraction=1., removed_correction_fraction=0.,
            joints_retained_fraction=1., root_retained_fraction=1.)
        report.pop('_metric', None)
        report.pop('_quadratic', None)
        return fixed.copy(), report
    candidate = project_joints(objective, objective.project(bulk), report=report)
    if tracker is not None:
        tracker.record('joint_continuity_and_boxes', ['SequenceContinuityGuard', 'CorrectionContinuity'], candidate)
    candidate = project_roots(objective, candidate, bulk, fixed, report=report)
    if tracker is not None:
        tracker.record('root_continuity_and_domains', ['SequenceContinuityGuard', 'CorrectionContinuity', 'RootDomain'], candidate)
    candidate = project_regions(objective, candidate, bulk, report=report)
    if tracker is not None:
        tracker.record('coupled_geometry_and_continuity', ['AcquisitionNormalGuard', 'ReferenceShape', 'SequenceContinuityGuard', 'CorrectionContinuity'], candidate)
    failures = certify(objective, candidate)
    if failures:
        # A region replaced by fixed may break a crossing stencil. Blend the
        # entire trajectory and certify every original predicate instead.
        report['fallback_failures'] = failures
        tangent = objective.coords.difference(fixed, candidate)
        fraction = 1.
        for _ in range(30):
            fraction *= .5
            trial = objective.project(objective.coords.integrate(fixed, tangent, fraction))
            if not certify(objective, trial):
                candidate = trial
                break
        else:
            fraction, candidate = 0., fixed.copy()
        report['fallback_fraction'] = fraction
        if tracker is not None:
            tracker.record('exact_global_fallback', failures, candidate)
    final = certify(objective, candidate)
    if final:
        raise ProjectionFailure(f'Projection failed exact certification: {final}')
    bulk_delta = objective.coords.difference(fixed, bulk)
    projected_delta = objective.coords.difference(fixed, candidate)
    loss = objective.coords.difference(bulk, candidate)
    report.update(seconds=time.monotonic() - started, exact_guard_failures=final,
        fixed_to_bulk_rms=float(np.sqrt(np.mean(bulk_delta**2))),
        bulk_to_projected_rms=float(np.sqrt(np.mean(loss**2))),
        retained_correction_fraction=float(np.linalg.norm(projected_delta) / max(np.linalg.norm(bulk_delta), 1e-30)),
        removed_correction_fraction=float(np.linalg.norm(loss) / max(np.linalg.norm(bulk_delta), 1e-30)))
    for name, columns in [('joints', objective.coords.scalar_cols),
                          ('root', [i for i in range(objective.width) if i not in objective.coords.scalar_cols])]:
        if len(columns):
            report[name + '_retained_fraction'] = float(np.linalg.norm(projected_delta[:, columns]) /
                max(np.linalg.norm(bulk_delta[:, columns]), 1e-30))
    report['rms_by_signal'] = {}
    groups = [('joints_rad', objective.coords.scalar_cols)]
    if objective.coords.root_d is not None:
        groups += [(name, [objective.coords.inverse[d] for d in range(first, first + 3)
                           if d in objective.coords.inverse])
                   for name, first in [('root_translation_m', objective.coords.root_d),
                                       ('root_rotation_rad', objective.coords.root_d + 3)]]
    for name, columns in groups:
        if len(columns):
            report['rms_by_signal'][name] = dict(fixed_to_bulk=float(np.sqrt(np.mean(bulk_delta[:, columns]**2))),
                bulk_to_projected=float(np.sqrt(np.mean(loss[:, columns]**2))))
    report.pop('_metric', None)
    report.pop('_quadratic', None)
    return candidate, report
