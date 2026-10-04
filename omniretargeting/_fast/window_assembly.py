"""Assemble a restricted SQP directly from exact, fixed-boundary components."""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .proposals import guards_of
from .qp_assembly import SoftTerms, _providers, assemble_program, quadratic
from .qp_guard_rows import UnsupportedRows, intervals, norms
from .qp_types import IterationModel, QPSolution
from .window_schedule import frame_range
from .window_screen import solve_working_set
from .window_rows import merge_temporal


# Generic level coefficients generate proposals only. Nonlinear per-body and
# contact checks decide acceptance; these numbers assert no finite-weight proof.
LEVEL_WEIGHTS = np.array([1000., 100., 10., 1.])


def _guard_blocks(guard, poses, frames, size):
    provider = getattr(guard, 'qp_bounds', None)
    if provider is None:
        raise UnsupportedRows(f'No window rows for {type(guard).__name__}')
    linear, cones = provider(poses, frames=frames)
    names = getattr(guard, 'qp_bound_sources', None)
    if names is not None:
        linear_names, cone_names = names(poses, frames=frames)
        if len(linear_names) != len(linear) or len(cone_names) != len(cones):
            raise ValueError('Window bound source identities changed')
    else:
        linear_names, cone_names = [None] * len(linear), [None] * len(cones)
    evaluator = getattr(guard, 'qp_values', None)
    cached_pose, cached_value = None, None

    def values(p):
        nonlocal cached_pose, cached_value
        if cached_pose is None or not np.array_equal(p, cached_pose):
            cached_value = evaluator(p, frames=frames)
            cached_pose = np.array(p, copy=True)
            if len(cached_value[0]) != len(linear) or len(cached_value[1]) != len(cones):
                raise ValueError('SOC row identities changed')
        return cached_value

    blocks = []
    for i, (matrix, lo, hi) in enumerate(linear):
        if matrix.shape != (len(lo), size):
            raise ValueError('Window linear columns must be local')
        evaluate = (lambda p, i=i: values(p)[0][i]) if evaluator else None
        blocks.append(intervals(matrix, lo, hi, evaluate))
    for i, (matrix, residual, upper) in enumerate(cones):
        if matrix.shape != (np.size(residual), size):
            raise ValueError('Window cone columns must be local')
        evaluate = (lambda p, i=i: values(p)[1][i]) if evaluator else None
        blocks.append(norms(matrix, residual, upper, evaluate))
    return blocks, [*linear_names, *cone_names]


def _mandatory_guard(guard):
    from .continuity import CorrectionContinuity
    from .._sequence.continuity_guard import SequenceContinuityGuard
    return (isinstance(guard, (CorrectionContinuity, SequenceContinuityGuard))
            or bool(getattr(guard, 'window_mandatory', False)))


def prepare_window_model(objective, poses, frames, evaluation=None):
    """No full Hessian/Jacobian is constructed or sliced by this assembler."""
    from .shape_step import _pin_root
    frames = frame_range(frames, len(poses))
    span, coords = slice(frames.start, frames.stop), objective.coords
    if evaluation is None:
        evaluation = objective.evaluate(poses, frames=frames)
    acc, residual, matrix = evaluation[:3]
    size = len(frames) * coords.width
    if matrix.shape[1] != size or acc.gradient.shape != (len(frames), coords.width):
        raise ValueError('Objective did not return window-local derivatives')
    hessian = (sparse.block_diag(list(acc.hessian), format='csc') + matrix.T @ matrix).tocsc()
    gradient = acc.gradient.ravel() + matrix.T @ residual
    diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), 1e-9)
    lower, upper = coords.bounds(poses[span], objective.cfg['angular_step_rad'],
                                 objective.cfg['translation_step_m'])
    fixed = getattr(objective, 'fixed_root', None)
    _pin_root(SimpleNamespace(coords=coords, fixed_root=None if fixed is None else fixed[span]),
              poses[span], lower, upper)
    joint_lower, joint_upper = np.asarray(objective.lower), np.asarray(objective.upper)
    joint_lower = joint_lower[span] if joint_lower.ndim == 2 else joint_lower
    joint_upper = joint_upper[span] if joint_upper.ndim == 2 else joint_upper
    lower[:, coords.scalar_cols] = np.maximum(lower[:, coords.scalar_cols],
                                              joint_lower - poses[span][:, coords.qi])
    upper[:, coords.scalar_cols] = np.minimum(upper[:, coords.scalar_cols],
                                              joint_upper - poses[span][:, coords.qi])
    origin = np.array(poses, copy=True)
    origin.setflags(write=False)
    model = IterationModel(origin, coords, lower, upper, None, hessian, gradient, diagonal,
                           max(float(np.linalg.norm(gradient, np.inf)), 1.), frames=frames)
    coordinator = objective.priority_clearance
    coordinator.begin_iteration(poses, model, frames=frames)
    soft = SoftTerms(size)
    sources, soft_labels = [], {}
    for provider in _providers(objective):
        from .window_diagnostics import contact_batch_labels
        channels = contact_batch_labels(provider)
        row_start, cone_start = len(soft.rows), len(soft.cones)
        for name, cone in (('soft_rows', False), ('soft_cones', True)):
            function = getattr(provider, name, None)
            if function is not None:
                for item in function(poses, frames=frames):
                    kind, index = ('norm', len(soft.cones)) if cone else ('scalar', len(soft.rows))
                    if id(item[0]) in channels:
                        soft_labels[kind, index] = channels[id(item[0])]
                    soft.add(item, cone=cone)
        sources.append((provider, row_start, len(soft.rows), cone_start, len(soft.cones)))
    for provider, *_ in sources:
        limits = getattr(provider, 'soft_limits', None)
        if limits is not None:
            soft.add_limits(limits(poses, frames=frames))
    blocks = [intervals(sparse.eye(size, format='csr'), lower.ravel(), upper.ravel())]
    labels = {id(blocks[0]): 'step_and_joint_box'}
    mandatory_ids = {id(blocks[0])}
    temporal_linear = []
    for guard in guards_of(objective)[1]:
        additions, source_names = _guard_blocks(guard, poses, frames, size)
        for index, block in enumerate(additions):
            name = ('priority_hard' if guard is coordinator else type(guard).__name__)
            family = 'norm' if block.soc else 'linear'
            if type(guard).__name__ == 'ReferenceShape':
                family = 'root_ball' if block.soc else 'layout'
            labels[id(block)] = f'{source_names[index] or name}:{family}:batch{index}'
        if _mandatory_guard(guard):
            temporal_linear.extend(b for b in additions if not b.soc)
            additions = [b for b in additions if b.soc]
            mandatory_ids.update(map(id, additions))
        blocks.extend(additions)
    combined = merge_temporal(temporal_linear)
    blocks.extend(combined)
    mandatory_ids.update(map(id, combined))
    for block in combined:
        labels[id(block)] = 'continuity:linear:merged'
    # FastObjective.project also enforces this immutable root ball, separately
    # from ReferenceShape's balls. Include it in the proposal domain as well.
    if getattr(objective, 'root_budget', None) is not None and coords.root_q is not None:
        select = np.zeros((3, coords.width))
        for axis in range(3):
            column = coords.inverse.get(coords.root_d + axis)
            if column is not None:
                select[axis, column] = 1.
        root_matrix = sparse.kron(sparse.eye(len(frames)), select, format='csr')
        q = coords.root_q
        def root_value(p):
            return p[span, q:q + 3] - objective.reference[span, q:q + 3], np.full(len(frames), objective.root_budget)
        root_residual, root_bound = root_value(poses)
        blocks.append(norms(root_matrix, root_residual, root_bound, root_value))
        labels[id(blocks[-1])] = 'objective_root_ball'
    model.soft = soft
    model.window_soft_sources = tuple(sources)
    model.program = assemble_program(blocks, soft, size)
    from .window_diagnostics import row_sources
    model.window_row_sources = row_sources(blocks, labels, soft, soft_labels)
    # assemble_program orders all hard scalar rows, then soft scalar rows and
    # slack nonnegativity, then hard and soft complete cones.
    mandatory = np.zeros(len(model.program.b), bool)
    offset = 0
    for block in (b for b in blocks if not b.soc):
        mandatory[offset:offset + len(block.b)] = id(block) in mandatory_ids
        offset += len(block.b)
    if soft.keys:
        mandatory[model.program.nonneg - len(soft.keys) - len(soft.limits):model.program.nonneg] = True
    offset = model.program.nonneg
    for block in (b for b in blocks if b.soc):
        mandatory[offset:offset + len(block.b)] = id(block) in mandatory_ids
        offset += len(block.b)
    return model, mandatory


def solve_window_model(model, damping, mandatory, *, penetration_multiplier=1.):
    P, c = quadratic(model, damping)
    coefficients = LEVEL_WEIGHTS.copy() * model.scale
    coefficients[0] *= penetration_multiplier
    c[model.gradient.size:] += (np.asarray(model.soft.weights) *
                                model.soft.high_coefficients(coefficients) / model.scale)
    from .window_diagnostics import capture
    captured = capture(model, P, c, mandatory)
    result, trace = solve_working_set(model, P, c, mandatory)
    if trace and captured:
        trace[0]['capture'] = captured
    if result is None:
        return None, trace
    x, dual = result
    step = x[:model.gradient.size]
    certificate = dict(kind='window_weighted_proposal', certified=False, optimality_certified=False,
                       coefficients=coefficients.tolist(), full_model_separated=True,
                       predicted_reductions=dict(keys=model.predicted_reductions(step)))
    return QPSolution(step, model.predicted_slacks(step), certificate, x, dual), trace
