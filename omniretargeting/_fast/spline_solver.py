"""All-frame SQP on immutable-reference cubic correction coefficients."""
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, lsmr

from .barrier_solver import _domains
from .continuity import chatter_reports
from .proposals import ALPHAS, guards_of
from .qp_types import settings
from .solver import INITIAL_DAMPING, MIN_DAMPING, MAX_DAMPING, CONVERGED_DECREASE
from .spline_embedding import CorrectionSpline
from .spline_model import ChatterRows, prepare_spline_model


class _DeadlineReached(RuntimeError):
    pass


def _recordable(value):
    """Keep original tuple keys internally, but make diagnostics JSON-safe."""
    if isinstance(value, dict):
        return {key if isinstance(key, str) else repr(key): _recordable(item)
                for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_recordable(item) for item in value]
    return value


def step_merit(model, step, alpha, before, after, measured):
    """Actual original geometry improves before a tracking tie can be accepted."""
    if not np.isfinite(after):
        return False, -1., 'nonfinite_objective', {}
    from .spline_collision_priority import measurement_collision_layer
    actual = measurement_collision_layer(measured, frame_count=len(model.step_lower))
    if not actual['valid']:
        return False, -1., 'unknown_original_geometry', dict(collision_after_m=None,
                    unknown_reasons=actual['unknown_reasons'])
    layer = getattr(model, 'spline_collision_layer', None)
    if layer is None:
        # A zero-depth origin still needs complete original-frame evidence.
        # The ordinary tracking branch cannot buy a small new penetration
        # simply because the legacy per-key roundoff guard permits it.
        if actual['value'] > settings().tol_gap_abs:
            return False, -1., 'original_geometry_curvature_or_new_witness', dict(
                collision_before_m=0., collision_after_m=actual['value'])
        predicted = model.tracking_decrease(alpha * step)
        ratio = (before - after) / predicted if predicted > 0 else -1.
        return after < before and ratio > 0, ratio, 'nonlinear_cost_curvature', {}
    actual = measurement_collision_layer(measured, expected_keys=layer.keys,
                                         frame_count=len(model.step_lower))
    prediction = layer.prediction(alpha * step)
    record = dict(predicted_collision_reduction_m=prediction['improvement'],
                  collision_before_m=layer.baseline['value'], collision_after_m=actual['value'])
    if not actual['valid']:
        return False, -1., 'unknown_original_geometry', record
    reduction = layer.baseline['value'] - actual['value']
    tolerance = settings()
    numerical = tolerance.tol_gap_abs + tolerance.tol_gap_rel * abs(layer.baseline['value'])
    if reduction > numerical and prediction['improvement'] > 0:
        record['accepted_priority'] = 'collision'
        return True, reduction / prediction['improvement'], '', record
    if reduction >= -numerical and prediction['improvement'] <= numerical:
        predicted = model.tracking_decrease(alpha * step)
        ratio = (before - after) / predicted if predicted > 0 else -1.
        record['accepted_priority'] = 'tracking_at_collision_tie'
        return after < before and ratio > 0, ratio, 'collision_layer_stationary', record
    return False, -1., 'original_geometry_curvature_or_new_witness', record


def projection_domain(objective, poses):
    """Use the existing projection-domain predicate through its window view."""
    view = SimpleNamespace(coords=objective.coords, lower=objective.lower,
        upper=objective.upper, fixed_root=objective.fixed_root,
        global_objective=objective, span=slice(None))
    return _domains(view, poses)


def transition_times(objective):
    """Knots at authored contact switches, without changing any frame/target."""
    masks = [eff['active'] | eff['managed'] for eff in objective.effectors]
    masks += [surface['active'] for surface in objective.surfaces]
    changes = np.zeros(objective.frames, bool)
    for mask in masks:
        changed = np.asarray(mask)[1:] != np.asarray(mask)[:-1]
        changes[1:] |= changed.reshape(objective.frames - 1, -1).any(axis=1)
    frames = np.flatnonzero(changes)
    return frames[(frames > 0) & (frames < objective.frames - 1)] / objective.framerate


class SplineSolve:
    def __init__(self, objective, geometry, *, adaptive=False, deadline=None, identity=False):
        self.objective, self.geometry = objective, geometry
        self.spline = CorrectionSpline(objective.coords, objective.reference, objective.framerate,
            identity=identity, extra_times=() if identity or not adaptive else transition_times(objective))
        self.a = self.spline.coefficients.copy()
        self.chatter = ChatterRows(objective.coords, self.spline.reference)
        self.adaptive, self.deadline = adaptive, deadline
        self.damping = INITIAL_DAMPING
        self.trust_scale = 1.
        self.history, self.refinements, self.failures = [], [], []
        self.timings = dict(evaluation=0., assembly=0., linear_solve=0., conic_solve=0.,
                            trials=0., refinement=0., reweight=0.)
        self.assembly_details = {}
        self.conic_calls = self.accepted = self.recoveries = self.rejected = 0
        self.max_variables = self.max_hessian_nnz = self.max_constraint_nnz = 0
        self.remaining_iterations = int(objective.cfg['iterations']) * 3
        self.exhausted = False

    @property
    def poses(self):
        return self.spline.embed(self.a)

    def remaining(self):
        return np.inf if self.deadline is None else self.deadline - time.monotonic()

    def expired(self):
        if self.remaining() <= 0:
            self.exhausted = True
            return True
        return False

    def _assembled(self, model, tick):
        self.timings['assembly'] += time.monotonic() - tick
        for name, seconds in model.spline_assembly_seconds.items():
            self.assembly_details[name] = self.assembly_details.get(name, 0.) + seconds

    def _solve(self, model):
        """Sparse unconstrained proposal, then existing exact cone working set."""
        if self.expired():
            return None, dict(backend='budget_exhausted')
        measurement = getattr(model, 'spline_geometry', None)
        if measurement is not None:
            from .spline_collision_priority import (build_collision_layer,
                collision_first_proposal, measurement_collision_layer)
            layer_value = measurement_collision_layer(measurement, frame_count=len(model.step_lower))
            if not layer_value['valid']:
                return None, dict(backend='original_geometry_unknown', status='unknown',
                                  unknown_reasons=layer_value['unknown_reasons'], conic_calls=0)
            if layer_value['value'] > 0:
                try:
                    model.spline_collision_layer = build_collision_layer(measurement,
                        self.geometry.provider.linear_rows, model.spline_mapping, model.spline_selected)
                except (KeyError, TypeError, ValueError) as error:
                    return None, dict(backend='original_geometry_unknown', status='unknown',
                                      unknown_reasons=[str(error)], conic_calls=0)
                tick = time.monotonic()
                step, diagnostics = collision_first_proposal(model, model.spline_mapping,
                    model.spline_selected, measurement, self.geometry.provider.linear_rows,
                    self.damping, self.remaining, layer=model.spline_collision_layer)
                self.timings['conic_solve'] += time.monotonic() - tick
                self.conic_calls += diagnostics['conic_calls']
                for name in ('collision_program', 'tracking_program'):
                    size = diagnostics.get(name, {})
                    self.max_variables = max(self.max_variables, size.get('variables', 0))
                    self.max_hessian_nnz = max(self.max_hessian_nnz, size.get('P_nnz', 0))
                    self.max_constraint_nnz = max(self.max_constraint_nnz, size.get('A_nnz', 0))
                return step, diagnostics
        matrix = (model.hessian + sparse.diags(self.damping * model.diagonal + 1e-9)).tocsc()
        if self.expired():
            return None, dict(backend='budget_exhausted')
        tick = time.monotonic()
        step = spsolve(matrix, -model.gradient)
        self.timings['linear_solve'] += time.monotonic() - tick
        if self.expired():
            return None, dict(backend='budget_exhausted', interrupted_after='sparse_solve')
        program = model.program
        slack = program.b - program.A @ step
        good = np.isfinite(step).all() and np.all(slack[:program.nonneg] >= 0)
        cursor = program.nonneg
        for size in program.soc:
            cone = slack[cursor:cursor + size]
            good &= cone[0] >= np.linalg.norm(cone[1:])
            cursor += size
        if good:
            return step, dict(backend='sparse_unconstrained', conic_calls=0)
        if self.remaining() <= 0:
            return None, dict(backend='budget_exhausted')
        tick = time.monotonic()
        P = sparse.triu(2 * matrix / model.scale).tocsc()
        c = 2 * model.gradient / model.scale
        diagnostics = dict(backend='clarabel_working_set')
        conic_calls = 0
        def options():
            nonlocal conic_calls
            if self.expired():
                raise _DeadlineReached
            conic_calls += 1
            value = settings()
            value.time_limit = max(self.remaining(), 1e-6)
            return value
        from .working_set import solve as working_set
        try:
            result = working_set(P, c, program.A, program.b, program.kinds(), options,
                                 diagnostics=diagnostics)
            # A linear subproblem retry, never a clip optimization restart.
            if result is None and self.remaining() > 0:
                import clarabel
                diagnostics['complete_program_fallback'] = True
                result = clarabel.DefaultSolver(P, c, program.A, program.b,
                                                 program.kinds(), options()).solve()
        except _DeadlineReached:
            result = None
            diagnostics['interrupted_after'] = 'working_set_round'
        self.conic_calls += conic_calls
        diagnostics['conic_calls'] = conic_calls
        self.timings['conic_solve'] += time.monotonic() - tick
        diagnostics['status'] = None if result is None else str(result.status)
        if self.expired():
            diagnostics.setdefault('interrupted_after', 'cone_solve')
            return None, diagnostics
        if result is None or str(result.status) not in {'Solved', 'AlmostSolved'}:
            return None, diagnostics
        return np.asarray(result.x), diagnostics

    def refine(self, focus, *, reason='local_basis_capacity'):
        """Local Boehm insertion; the inherited accepted curve is unchanged."""
        if self.expired() or self.spline.identity or self.spline.controls >= self.spline.count + 3:
            return False
        tick = time.monotonic()
        knots = np.unique(self.spline.knot_vector)
        times = self.spline.times[np.flatnonzero(focus)]
        spans = np.unique(np.clip(np.searchsorted(knots, times, side='right') - 1,
                                  0, len(knots) - 2))
        insertion = .5 * (knots[spans] + knots[spans + 1])
        insertion = insertion[(knots[spans + 1] - knots[spans]) > 1 / self.objective.framerate]
        if not len(insertion):
            return False
        # Numerical size budget, never a pose/collision quality tolerance.
        insertion = insertion[:max(0, self.spline.count + 3 - self.spline.controls)]
        before = self.spline.controls
        self.spline, self.a = self.spline.insert_knots(insertion, self.a)
        self.refinements.append(dict(reason=reason, inserted_times=insertion.tolist(),
            focus_frames=np.flatnonzero(focus).tolist(), controls_before=before,
            controls_after=self.spline.controls, reference_rebased=False, inherited_curve=True))
        self.timings['refinement'] += time.monotonic() - tick
        return True

    def capacity_focus(self, evaluation, mapping):
        """Numerical stationarity diagnostic, never a quality acceptance gate."""
        empty = np.zeros(self.spline.count, bool)
        if self.expired():
            return empty, 0.
        acc, residual, temporal, _, _ = evaluation
        gradient = acc.gradient.ravel() + temporal.T @ residual
        jac, value = self.geometry.residual_rows(self.poses)
        if self.expired():
            return empty, 0.
        gradient += jac.T @ value
        if self.objective.fixed_root is not None:
            shaped = gradient.reshape(self.spline.count, self.objective.width)
            frozen = np.ones(self.objective.width, bool)
            frozen[self.objective.coords.scalar_cols] = False
            shaped[:, frozen] = 0.
        fit = lsmr(mapping, gradient, atol=1e-8, btol=1e-8, maxiter=100)[0]
        missing = (gradient - mapping @ fit).reshape(self.spline.count, -1)
        amplitude = np.linalg.norm(missing, axis=1)
        relative = float(np.linalg.norm(missing) / max(np.linalg.norm(gradient), 1e-12))
        focus = self.geometry.focus(self.spline.count)
        # Control allocation follows the local unrepresented gradient. This
        # 1e-3 is a numerical fit diagnostic, not an angular/collision allowance.
        focus |= amplitude > max(float(amplitude.max(initial=0)) * .5, 1e-12)
        return focus if relative > 1e-3 else np.zeros_like(focus), relative

    def optimize(self, iterations, *, label='spline', callback=None, controls=None,
                 converged_decrease=CONVERGED_DECREASE, weight_updater=None):
        costs, slow = [], 0
        columns = (self.objective.coords.scalar_cols if self.objective.fixed_root is not None else None)
        for iteration in range(iterations):
            if self.remaining() <= 0 or self.remaining_iterations <= 0:
                self.exhausted = True
                break
            self.remaining_iterations -= 1
            poses = self.poses
            tick = time.monotonic()
            evaluation = self.objective.evaluate(poses)
            self.timings['evaluation'] += time.monotonic() - tick
            if self.expired():
                break
            tick = time.monotonic()
            model, mapping, selected = prepare_spline_model(self.objective, self.spline, self.a,
                evaluation, chatter=self.chatter, geometry=self.geometry,
                controls=controls, columns=columns, trust_scale=self.trust_scale)
            self._assembled(model, tick)
            if self.expired():
                break
            self.max_variables = max(self.max_variables, model.gradient.size)
            self.max_hessian_nnz = max(self.max_hessian_nnz, model.hessian.nnz)
            self.max_constraint_nnz = max(self.max_constraint_nnz, model.program.A.nnz)
            if not len(selected):
                self.failures.append(dict(phase=label, reason='no_interior_supported_controls'))
                break
            before = evaluation[0].cost
            accepted, reason, diagnostics = None, 'linear_subproblem', {}
            for recovery in range(3):
                if self.remaining() <= 0:
                    break
                step, diagnostics = self._solve(model)
                if diagnostics.get('status') in {'unknown', 'unknown_geometry'}:
                    reason = 'unknown_original_geometry'
                    break
                if step is not None:
                    tick = time.monotonic()
                    direction = np.zeros(self.a.size)
                    direction[selected] = step
                    for alpha in ALPHAS:
                        if self.expired():
                            break
                        trial_a = self.a + alpha * direction.reshape(self.a.shape)
                        candidate = self.spline.embed(trial_a)
                        if not projection_domain(self.objective, candidate):
                            reason = 'projection_domain'
                            continue
                        failed_guard = next((guard for guard in guards_of(self.objective)[1]
                                             if not guard.accepts(candidate)), None)
                        if failed_guard is not None:
                            reason = 'nonlinear_guard_curvature'
                            diagnostics['rejected_guard'] = type(failed_guard).__name__
                            diagnostics['rejected_alpha'] = alpha
                            continue
                        if chatter_reports(self.objective.coords, self.spline.reference, candidate)[1].any():
                            reason = 'nonlinear_rotation_chatter'
                            continue
                        if self.expired():
                            break
                        good, measured, bad = self.geometry.accepts(candidate)
                        if self.expired():
                            break
                        if measured.get('failures') or not measured.get('complete', True):
                            reason = 'unknown_original_geometry'
                            diagnostics['unknown_reasons'] = measured.get('failures', ())
                            break
                        if not good:
                            reason = 'original_geometry_curvature_or_new_witness'
                            if self.geometry.learn(poses, candidate, measured, bad):
                                diagnostics['new_original_witnesses'] = len(bad)
                            continue
                        value = self.objective.evaluate(candidate, False)
                        if self.expired():
                            break
                        after = value[0].cost
                        good, ratio, reason, merit = step_merit(model, step, alpha, before, after, measured)
                        diagnostics['nonlinear_merit'] = merit
                        if good:
                            accepted = trial_a, candidate, measured, value, after, ratio, alpha
                            break
                        if reason == 'unknown_original_geometry':
                            break
                    self.timings['trials'] += time.monotonic() - tick
                if accepted is not None:
                    break
                if reason == 'unknown_original_geometry':
                    break
                if self.expired():
                    break
                self.damping = min(self.damping * 10, MAX_DAMPING)
                self.recoveries += 1
                if reason in {'nonlinear_guard_curvature', 'nonlinear_rotation_chatter',
                              'original_geometry_curvature_or_new_witness', 'nonlinear_cost_curvature'}:
                    self.trust_scale *= .25
                # Newly discovered material witnesses participate in the next
                # model at the accepted origin, with correct origin Jacobians.
                if recovery < 2:
                    tick = time.monotonic()
                    model, mapping, selected = prepare_spline_model(self.objective, self.spline,
                        self.a, evaluation, chatter=self.chatter, geometry=self.geometry,
                        controls=controls, columns=columns, trust_scale=self.trust_scale)
                    self._assembled(model, tick)
            row = dict(phase=label, iteration=iteration, cost=before, after=before, alpha=0.,
                damping=self.damping, trust_scale=self.trust_scale, solver=_recordable(diagnostics), variables=model.gradient.size,
                hessian_nnz=model.hessian.nnz, constraint_nnz=model.program.A.nnz,
                original_frames=self.spline.count, controls=self.spline.controls)
            row['assembly_seconds'] = model.spline_assembly_seconds
            if accepted is None:
                self.rejected += 1
                row['rejected_reason'] = reason
                self.history.append(row)
                if self.expired():
                    row['rejected_reason'] = 'budget_exhausted'
                    break
                if reason == 'unknown_original_geometry':
                    self.failures.append(dict(phase=label, reason=reason))
                    break
                if reason in {'nonlinear_guard_curvature', 'nonlinear_rotation_chatter',
                              'original_geometry_curvature_or_new_witness', 'nonlinear_cost_curvature'}:
                    # Curvature is a trust-region problem, not a certificate
                    # that more controls are needed. Keep the accepted curve.
                    row['recovery'] = 'smaller_increment_trust_region'
                    continue
                # Curvature already received three bounded trust adjustments.
                # Enrich only spans with residual defects; never restart the clip.
                focus, capacity = self.capacity_focus(evaluation, mapping[:, selected])
                row['unrepresented_gradient_fraction'] = capacity
                if self.adaptive and controls is None and focus.any() and self.refine(focus):
                    self.damping = INITIAL_DAMPING
                    continue
                self.failures.append(dict(phase=label, reason=reason, frames=np.flatnonzero(focus).tolist()))
                break
            self.a, candidate, measured, value, after, ratio, alpha = accepted
            self.geometry.commit(measured)
            self.accepted += 1
            if alpha == 1.:
                self.trust_scale = min(1., self.trust_scale * 2.)
            self.damping = max(self.damping * max(1 / 3, 1 - (2 * ratio - 1)**3), MIN_DAMPING)
            row.update(after=after, alpha=alpha, ratio=ratio, **value[3])
            self.history.append(row)
            if callback is not None:
                callback(candidate, row)
            weights_changed = False
            if weight_updater is not None and not self.expired():
                tick = time.monotonic()
                weights_changed = weight_updater(candidate, label)
                self.timings['reweight'] += time.monotonic() - tick
                if weights_changed:
                    row['objective_weights_changed_after_acceptance'] = True
                    row['new_weight_model_required_before_next_step'] = True
                    self.damping = max(self.damping, INITIAL_DAMPING)
            merit = diagnostics.get('nonlinear_merit', {})
            if merit.get('accepted_priority') == 'collision':
                decrease = ((merit['collision_before_m'] - merit['collision_after_m'])
                            / max(merit['collision_before_m'], 1e-12))
            else:
                decrease = (before - after) / max(before, 1e-12)
            slow = slow + 1 if not weights_changed and alpha == 1. and decrease < converged_decrease else 0
            costs.append(after)
            if slow >= 2:
                if self.adaptive and controls is None and self.geometry.focus(len(poses)).any():
                    if self.expired():
                        break
                    # Capacity is diagnosed at the accepted curve, with its
                    # current full-frame gradient and Integrate differential.
                    tick = time.monotonic()
                    evaluation = self.objective.evaluate(self.poses)
                    self.timings['evaluation'] += time.monotonic() - tick
                    if self.expired():
                        break
                    mapping = self.spline.jacobian(self.a)
                    focus, capacity = self.capacity_focus(evaluation, mapping[:, selected])
                    row['unrepresented_gradient_fraction'] = capacity
                    if focus.any() and self.refine(focus):
                        self.damping, slow = INITIAL_DAMPING, 0
                        continue
                break
        return self.poses

    def report(self):
        return dict(original_frames=self.spline.count, controls=self.spline.controls,
            full_rate_variables=self.spline.count * self.objective.width,
            coefficient_variables=self.a.size, max_subproblem_variables=self.max_variables,
            max_hessian_nnz=self.max_hessian_nnz, max_constraint_nnz=self.max_constraint_nnz,
            conic_calls=self.conic_calls, recovery_count=self.recoveries,
            accepted_updates=self.accepted, rejected_iterations=self.rejected,
            timing_seconds=self.timings, assembly_details_seconds=self.assembly_details,
            refinements=self.refinements, defects=self.failures,
            budget_exhausted=self.exhausted or self.remaining() <= 0,
            full_rate_fallback_count=0, full_rate_fallback_seconds=0.,
            reference_rebased=False, targets_sampled=False, final_motion_filtered=False,
            all_original_frames_in_objective_and_constraints=True)
