"""Strict collision-first coefficient proposals, followed by tracking ties.

``collision_first_proposal(model, mapping, selected, measurement, linear_rows,
damping, remaining, *, layer=None)`` returns a selected step and diagnostics.
The caller owns exact nonlinear guards/chatter/non-regression, acceptance and
SplineGeometry.commit; neither an optimum nor a zero prediction proves safety.
The model must already contain *all* original hard rows, including trust/root
bounds and collision non-regression. Its H/g are the caller's tracking model.

The first objective is the sum of unique original collision keys' maximum
depths, in metres. SoftTerms supplies one nonnegative epigraph per key, with
every witness imposing -J delta - u <= phi. There is no tracking regularizer
in this level. Its certified optimum is frozen using only existing Clarabel
numerical tolerances before the ordinary, damped tracking QP is solved.

Use ``build_collision_layer(...).values(alpha * step)`` for backtracking
predictions and ``measurement_collision_layer`` on complete nonlinear
measurements. The latter includes new keys; unknown values are None, not zero.
No FK, geometry measurement, full-frame Hessian, working set or GPU is used.
"""

from copy import copy
from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse

from .priority_qp import _freeze
from .qp_assembly import SoftTerms, assemble_program, quadratic
from .qp_residual import kkt_residual
from .qp_types import ConeProgram, settings


def _key(key):
    if (not isinstance(key, tuple) or len(key) < 3
            or not isinstance(key[0], (int, np.integer)) or key[0] < 0
            or key[1] not in {'static', 'moving', 'self'}):
        raise ValueError(f'Invalid original collision key: {key!r}')
    hash(key)
    return key


def measurement_collision_layer(measurement, *, expected_keys=(), frame_count=None):
    """Reduce all original keys once, without body/self-pair averaging.

    A complete provider query can omit keys outside its activation band. Only
    an explicitly complete, reliable measurement gives those expected keys
    zero contribution. This is a layer value, never a clearance/reset/physics
    certificate. Missing completeness, partial scopes and failures are unknown.
    Nonlinear rows need clearance, but need not have a Jacobian.
    """
    reasons, depths = [], {}
    if not isinstance(measurement, dict):
        return dict(valid=False, status='unknown_geometry', value=None, per_key={},
                    unknown_reasons=['missing_measurement'], collision_free_certificate=False)
    reasons.extend(str(reason) for reason in measurement.get('failures', ()))
    if not measurement.get('complete', False):
        reasons.append('incomplete_geometry')
    if (not measurement.get('reliable', True)
            or not np.asarray(measurement.get('frame_complete', True)).all()):
        reasons.append('unreliable_geometry')
    if frame_count is not None:
        scope = measurement.get('query_scope', {})
        frames = scope.get('requested_frames', measurement.get('queried_frames'))
        if frames is not None and tuple(frames) != tuple(range(frame_count)):
            reasons.append('partial_original_frame_scope')
    if 'depths' not in measurement or 'rows' not in measurement:
        reasons.append('missing_depths_or_witnesses')
    try:
        for key, value in measurement.get('depths', {}).items():
            _key(key)
            if np.ndim(value) or not np.isfinite(value) or value < 0:
                raise ValueError(f'Invalid original depth for {key!r}')
            depths[key] = float(value)
        for key, row in measurement.get('rows', ()):
            _key(key)
            value = row['clearance']
            if np.ndim(value) or not np.isfinite(value):
                raise ValueError(f'Unknown witness clearance for {key!r}')
            depths[key] = max(depths.get(key, 0.), -float(value), 0.)
        for key in expected_keys:
            _key(key)
            if not reasons:
                depths.setdefault(key, 0.)
        if frame_count is not None and any(key[0] >= frame_count for key in depths):
            raise ValueError('Collision key is outside original frame scope')
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(str(error))
    valid = not reasons
    value = float(sum(depths.values())) if valid else None
    if value is not None and not np.isfinite(value):
        reasons.append('nonfinite_collision_layer')
        valid, value = False, None
    return dict(valid=valid,
                status=('known_penetration' if value else 'zero_known_depth') if valid else 'unknown_geometry',
                value=value, per_key=depths, unknown_reasons=reasons,
                collision_free_certificate=False)


@dataclass
class CollisionLinearization:
    """Sparse selected-column pullback and unchanged original key ownership."""
    jacobian: sparse.csr_matrix
    phi: np.ndarray
    soft: SoftTerms
    keys: tuple
    row_keys: tuple
    baseline: dict
    ownership: dict
    source_mapping: sparse.csr_matrix
    selected: np.ndarray
    source_jacobians: tuple

    def matches(self, measurement, mapping, selected):
        """Check a cached linearization without linear_rows or multiplication."""
        mapping = sparse.csr_matrix(mapping)
        source = self.source_mapping
        if (mapping.shape != source.shape or not np.array_equal(selected, self.selected)
                or not np.array_equal(mapping.indptr, source.indptr)
                or not np.array_equal(mapping.indices, source.indices)
                or not np.array_equal(mapping.data, source.data)):
            return False
        rows = measurement['rows']
        baseline = measurement_collision_layer(measurement)
        if (not baseline['valid'] or baseline['per_key'] != self.baseline['per_key']
                or tuple(key for key, _ in rows) != self.row_keys
                or not np.array_equal([row['clearance'] for _, row in rows], self.phi)):
            return False
        return all(np.array_equal(row.get('jacobian'), jacobian)
                   for (_, row), jacobian in zip(rows, self.source_jacobians))

    def values(self, step):
        """Maximum predicted depth of each key, including all its witnesses."""
        step = np.asarray(step, float)
        if step.shape != (self.soft.size,) or not np.isfinite(step).all():
            raise ValueError('Collision prediction needs a finite selected coefficient step')
        values = self.soft.values(step)
        return {key: float(values[('penetration', key)]) for key in self.keys}

    def prediction(self, step):
        after = self.values(step)
        before = self.values(np.zeros(self.soft.size))
        reductions = {key: before[key] - after[key] for key in self.keys}
        return dict(before=float(sum(before.values())), after=float(sum(after.values())),
                    improvement=float(sum(reductions.values())), before_per_key=before,
                    after_per_key=after, per_key_improvement=reductions,
                    collision_free_certificate=False)


def build_collision_layer(measurement, linear_rows, mapping, selected):
    """Pull original tangent witness rows back through the supplied mapping.

    ``mapping`` is the full Integrate differential, including SO(3)'s right
    Jacobian. ``selected`` is exactly the model's coefficient-column order.
    Slicing occurs before multiplication, so locked/local columns cannot leak.
    This calls only the supplied linear_rows assembler, never provider.measure.
    """
    baseline = measurement_collision_layer(measurement)
    if not baseline['valid']:
        raise ValueError('; '.join(baseline['unknown_reasons']))
    mapping = sparse.csr_matrix(mapping)
    selected = np.asarray(selected)
    if (selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0) or np.any(selected >= mapping.shape[1])
            or len(np.unique(selected)) != len(selected)):
        raise ValueError('selected must be unique coefficient columns in model order')
    matrix, phi = linear_rows(measurement)
    matrix, phi = sparse.csr_matrix(matrix), np.asarray(phi, float)
    rows = measurement['rows']
    if matrix.shape != (len(rows), mapping.shape[0]) or phi.shape != (len(rows),):
        raise ValueError('Original collision linear_rows/mapping shapes disagree')
    if (not np.isfinite(matrix.data).all() or not np.isfinite(mapping.data).all()
            or not np.isfinite(phi).all()):
        raise ValueError('Unknown nonfinite collision Jacobian or mapping')
    if not np.array_equal(phi, np.asarray([row['clearance'] for _, row in rows])):
        raise ValueError('linear_rows clearance does not belong to measured witnesses')
    jacobian = (matrix @ mapping[:, selected]).tocsr()
    jacobian.eliminate_zeros()
    if not np.isfinite(jacobian.data).all():
        raise ValueError('Unknown nonfinite collision pullback')
    row_keys = tuple(key for key, _ in rows)
    soft = SoftTerms(len(selected))
    if rows:
        soft.add((-jacobian, phi.copy(), np.ones(len(rows)),
                  tuple(('penetration', key) for key in row_keys)))
    keys = tuple(key[1] for key in soft.keys)
    ownership = {key: dict(epigraph=index, frame=int(key[0]), kind=key[1],
                          geoms=tuple(key[2:] if key[1] == 'self' else key[-1:]), witnesses=[])
                 for index, key in enumerate(keys)}
    for index, key in enumerate(row_keys):
        ownership[key]['witnesses'].append(index)
    zero = soft.values(np.zeros(len(selected)))
    tolerance = settings().tol_feas
    for key, depth in baseline['per_key'].items():
        if depth > zero.get(('penetration', key), 0.) + tolerance:
            raise ValueError(f'Missing deepest measured collision witness for {key!r}')
    return CollisionLinearization(jacobian, phi, soft, keys, row_keys, baseline, ownership,
                                  mapping.copy(), selected.copy(),
                                  tuple(np.asarray(row['jacobian']).copy() for _, row in rows))


def _with_collision_rows(base, layer):
    """Augment the complete program; never reconstruct/drop its hard rows."""
    size, extra = layer.soft.size, len(layer.keys)
    if base.step_size != size or base.A.shape[1] < size:
        raise ValueError('Model program does not use selected coefficient coordinates')
    addition = assemble_program((), layer.soft, size)
    old_extra = base.A.shape[1] - size
    collision = sparse.hstack((addition.A[:, :size],
                              sparse.csc_matrix((addition.A.shape[0], old_extra)),
                              addition.A[:, size:]), format='csc')
    extended = sparse.hstack((base.A, sparse.csc_matrix((base.A.shape[0], extra))), format='csc')
    at = base.nonneg
    matrix = sparse.vstack((extended[:at], collision, extended[at:]), format='csc')
    soft = copy(base.soft) if base.soft is not None else SoftTerms(size)
    soft.heads = [(heads + (heads >= at) * addition.nonneg, indices)
                  for heads, indices in soft.heads]
    return ConeProgram(matrix, np.r_[base.b[:at], addition.b, base.b[at:]],
                       at + addition.nonneg, base.soc, size, base.hard_blocks, soft)


def _dimensions(program, P):
    return dict(variables=int(program.A.shape[1]), rows=int(program.A.shape[0]),
                A_shape=list(program.A.shape), A_nnz=int(program.A.nnz),
                P_shape=list(P.shape), P_nnz=int(sparse.triu(P).nnz),
                nonnegative=int(program.nonneg), soc=list(program.soc))


def _expired(remaining, diagnostics, where):
    value = float(remaining())
    diagnostics['remaining_seconds'] = value
    if np.isnan(value) or value <= 0:
        diagnostics.update(status='budget_exhausted', deadline_reached_at=where,
                           remaining_seconds=value, overrun_seconds=max(0., -value))
        diagnostics['certification']['certified'] = False
        return True
    return False


def _solve_certified(program, P, c, stage, remaining, diagnostics, model):
    """Original settings/retries, with a deadline around *each* Clarabel call."""
    import clarabel

    pending = [(False, False), (True, False)]
    while pending:
        retry, precision = pending.pop(0)
        if _expired(remaining, diagnostics, stage + '_before_prepare'):
            return None
        upper = sparse.triu(P).tocsc()
        vector = np.asarray(c, float)
        options = settings(retry=retry, precision=precision)
        cones = program.kinds()
        attempt = dict(stage=stage, retry=retry, precision=precision,
                       **_dimensions(program, upper))
        if _expired(remaining, diagnostics, stage + '_before_construct'):
            return None
        # Keep the existing numerical settings; only apply the caller's wall
        # budget. Never clamp an expired budget to a new tiny solver call.
        options.time_limit = diagnostics['remaining_seconds']
        tick = time.monotonic()
        try:
            diagnostics['solver_constructions'] += 1
            solver = clarabel.DefaultSolver(upper, vector, program.A, program.b, cones, options)
        except Exception as error:
            attempt.update(status='construction_error', error=f'{type(error).__name__}: {error}',
                           certified=False, construct_seconds=time.monotonic() - tick)
            diagnostics['construction_failures'].append(attempt)
            model.solve_diagnostics.append(attempt)
            if _expired(remaining, diagnostics, stage + '_after_construct'):
                return None
            diagnostics.update(status='solver_failed', failed_stage=stage)
            return None
        attempt['construct_seconds'] = time.monotonic() - tick
        if _expired(remaining, diagnostics, stage + '_before_solve'):
            diagnostics['interrupted_construction'] = attempt
            return None
        diagnostics['conic_calls'] += 1
        diagnostics['clarabel_calls'].append(attempt)
        tick = time.monotonic()
        try:
            result = solver.solve()
        except Exception as error:
            result = None
            attempt.update(status='solve_error', error=f'{type(error).__name__}: {error}')
        attempt['solve_seconds'] = time.monotonic() - tick
        diagnostics['solve_seconds'] += attempt['solve_seconds']
        if result is not None:
            attempt.update(status=str(result.status), iterations=int(result.iterations),
                           primal_residual=float(result.r_prim), dual_residual=float(result.r_dual),
                           primal_cost=float(result.obj_val), dual_cost=float(result.obj_val_dual))
        if _expired(remaining, diagnostics, stage + '_after_solve'):
            attempt.update(certified=False, discarded_after_deadline=True)
            model.solve_diagnostics.append(attempt)
            return None
        if result is None:
            attempt.update(certified=False, reason='solver_exception')
            model.solve_diagnostics.append(attempt)
            diagnostics.update(status='solver_failed', failed_stage=stage)
            return None
        if attempt['status'] in {'Solved', 'AlmostSolved'}:
            certificate = kkt_residual(P, vector, program.A, program.b,
                                       program.nonneg, program.soc, result.x, result.z, settings())
            attempt.update(certificate)
            model.solve_diagnostics.append(attempt)
            if _expired(remaining, diagnostics, stage + '_after_certificate'):
                return None
            if certificate['certified']:
                return result
            # The same refinement used by priority_qp; skip a loose retry of
            # an already solved but uncertified program.
            pending = [] if precision else [(False, True)]
        else:
            attempt.update(certified=False, reason='solver_status')
            model.solve_diagnostics.append(attempt)
            if attempt['status'] in {'PrimalInfeasible', 'DualInfeasible'}:
                diagnostics.update(status='solver_failed', failed_stage=stage)
                return None
            if not pending and not precision:
                pending = [(False, True)]
    diagnostics.update(status='uncertified_subproblem', failed_stage=stage)
    return None


def collision_first_proposal(model, mapping, selected, measurement, linear_rows,
                             damping, remaining, *, layer=None):
    """Solve the collision LP, freeze it, then solve tracking; return no partial step.

    The remaining() callable returns seconds left in the caller's shared wall
    budget. Started solver calls cannot be preempted; overruns are counted and
    discarded. Unknown/no-penetration/deadline/failed branches never claim a
    collision or optimization pass, and have explicit statuses and real counts.
    A prebuilt layer is reusable only for these witnesses, mapping and selected
    columns. Cache validation performs no query, row assembly or pullback.
    """
    diagnostics = dict(backend='clarabel_collision_first', status='not_started',
                       conic_calls=0, solver_constructions=0, solve_seconds=0.,
                       clarabel_calls=[], construction_failures=[], stage_optima={},
                       certification=dict(certified=False, method='collision_first_lexicographic',
                                          subproblem_only=True, collision_free_certificate=False,
                                          nonlinear_acceptance=False, physics=False, reset=False))
    if _expired(remaining, diagnostics, 'proposal_entry'):
        return None, diagnostics
    baseline = measurement_collision_layer(measurement, frame_count=len(model.step_lower))
    diagnostics['measured_layer'] = baseline
    if not baseline['valid']:
        diagnostics.update(status='unknown_geometry', unknown_reasons=baseline['unknown_reasons'])
        return None, diagnostics
    if baseline['value'] == 0:
        diagnostics['status'] = 'no_known_penetration'
        return None, diagnostics
    if model.gradient.size == 0:
        diagnostics['status'] = 'no_selected_coefficients'
        return None, diagnostics
    if selected is None:
        selected = np.arange(mapping.shape[1])
    if len(selected) != model.gradient.size or mapping.shape[0] != model.step_lower.size:
        raise ValueError('Mapping/selected must match the model coefficient order and original frames')
    if not np.isfinite(damping) or damping < 0:
        raise ValueError('damping must be finite and nonnegative')
    if layer is None:
        try:
            layer = build_collision_layer(measurement, linear_rows, mapping, selected)
        except (KeyError, TypeError, ValueError) as error:
            diagnostics.update(status='unknown_geometry', unknown_reasons=[str(error)])
            return None, diagnostics
        diagnostics['reused_collision_layer'] = False
    else:
        if (not isinstance(layer, CollisionLinearization)
                or layer.soft.size != model.gradient.size
                or not layer.matches(measurement, mapping, selected)):
            raise ValueError('Prebuilt collision layer does not match current witnesses/mapping/selected')
        diagnostics['reused_collision_layer'] = True
    program = _with_collision_rows(model.program, layer)
    variables = program.A.shape[1]
    level = np.r_[np.zeros(model.program.A.shape[1]), np.ones(len(layer.keys))]
    zero_P = sparse.csc_matrix((variables, variables))
    diagnostics.update(collision_keys=layer.keys, key_ownership=layer.ownership,
                       coefficient_variables=model.gradient.size, epigraphs=len(layer.keys),
                       collision_program=_dimensions(program, zero_P))
    if _expired(remaining, diagnostics, 'after_collision_assembly'):
        return None, diagnostics
    result = _solve_certified(program, zero_P, level, 'collision', remaining, diagnostics, model)
    if result is None:
        return None, diagnostics
    optimum = max(0., float(level @ result.x),
                  sum(layer.values(np.asarray(result.x)[:model.gradient.size]).values()))
    tolerance = settings()
    upper = optimum + tolerance.tol_gap_abs + tolerance.tol_gap_rel * abs(optimum)
    diagnostics['stage_optima']['collision'] = optimum
    diagnostics.update(collision_upper=upper, collision_stage_prediction=layer.prediction(
        np.asarray(result.x)[:model.gradient.size]))
    if _expired(remaining, diagnostics, 'before_tracking_assembly'):
        return None, diagnostics
    frozen = _freeze(program, level, upper)
    P, c = quadratic(model, damping)
    P = sparse.block_diag((P, sparse.csc_matrix((len(layer.keys), len(layer.keys)))), format='csc')
    c = np.r_[c, np.zeros(len(layer.keys))]
    diagnostics['tracking_program'] = _dimensions(frozen, P)
    result = _solve_certified(frozen, P, c, 'tracking', remaining, diagnostics, model)
    if result is None:
        return None, diagnostics
    primal = np.asarray(result.x, float)
    step = primal[:model.gradient.size].copy()
    prediction = layer.prediction(step)
    actual_epigraph = float(level @ result.x)
    allowed = upper + tolerance.tol_feas
    certified = bool(max(prediction['after'], actual_epigraph) <= allowed)
    diagnostics.update(predicted=prediction,
                       predicted_per_key=prediction['after_per_key'],
                       predicted_improvement=prediction['improvement'],
                       predicted_tracking_decrease=model.tracking_decrease(step))
    diagnostics['stage_optima']['tracking'] = float(.5 * primal @ (P @ primal) + c @ primal)
    diagnostics['certification'].update(certified=certified, collision_optimum=optimum,
                                        collision_upper=upper, collision_actual=prediction['after'],
                                        conic_feasibility_tolerance=tolerance.tol_feas)
    if _expired(remaining, diagnostics, 'proposal_exit'):
        return None, diagnostics
    diagnostics['status'] = 'solved' if certified else 'uncertified_collision_freeze'
    return (step if certified else None), diagnostics
