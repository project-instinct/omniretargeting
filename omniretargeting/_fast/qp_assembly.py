"""Once-per-iteration acquisition assembly, including shared L1 epigraphs."""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .qp_guard_rows import guard_blocks, norms
from .qp_types import (ConeProgram, IterationModel, QPSolution, TRACKING_LEVEL,
                       priority_level, solve_program, sparse_hessian)


@dataclass
class SoftTerms:
    size: int
    keys: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    levels: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    cones: list = field(default_factory=list)
    heads: list = field(default_factory=list)
    _index: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)

    def add(self, item, *, cone=False):
        if cone:
            matrix, residual, upper, weight, keys = item
            residual = np.asarray(residual, float)
        else:
            matrix, upper, weight, keys = item
        matrix = sparse.csr_matrix(matrix, dtype=float)
        upper, weight = np.asarray(upper, float), np.asarray(weight, float)
        count = len(keys)
        if (upper.shape != (count,) or weight.shape != (count,)
                or matrix.shape != ((3 if cone else 1) * count, self.size)
                or (cone and residual.shape != (count, 3))):
            raise ValueError('Invalid soft row/cone batch shapes')
        if (not np.isfinite(matrix.data).all() or not np.isfinite(upper).all()
                or not np.isfinite(weight).all() or np.any(weight <= 0)
                or (cone and not np.isfinite(residual).all())):
            raise ValueError('Soft rows require finite coefficients and positive weights')
        indices = []
        for key, value in zip(keys, weight):
            level = priority_level(key)
            if key not in self._index:
                self._index[key] = len(self.keys)
                self.keys.append(key)
                self.weights.append(float(value))
                self.levels.append(level)
            index = self._index[key]
            if self.weights[index] != value:
                raise ValueError(f'Inconsistent weight for shared epigraph {key!r}')
            indices.append(index)
        indices = np.asarray(indices, int)
        if cone:
            self.cones.append((matrix, residual, upper, indices))
        else:
            self.rows.append((matrix, upper, indices))

    def values(self, step):
        values = np.zeros(len(self.keys))
        for matrix, upper, indices in self.rows:
            np.maximum.at(values, indices, matrix @ step - upper)
        for matrix, residual, upper, indices in self.cones:
            violation = np.linalg.norm((matrix @ step).reshape(-1, 3) + residual, axis=1) - upper
            np.maximum.at(values, indices, violation)
        return dict(zip(self.keys, values.tolist()))

    def add_limits(self, limits):
        """Intersect hard epigraph uppers without changing shared-key costs."""
        for key, value in limits.items():
            if key not in self._index:
                raise ValueError(f'Soft limit references an unknown key: {key!r}')
            if np.ndim(value) != 0 or not np.isfinite(value) or value < 0:
                raise ValueError('Soft limits require finite nonnegative scalar bounds')
            index = self._index[key]
            self.limits[index] = min(self.limits.get(index, np.inf), float(value))

    def level_values(self, values):
        result = np.zeros(4)
        for key, weight, level in zip(self.keys, self.weights, self.levels):
            # Complete nonlinear geometry records can omit exactly zero keys.
            if level < TRACKING_LEVEL:
                result[level] += weight * values.get(key, 0.)
        return result

    def penalty(self, values, coefficients):
        return float(self.level_values(values) @ coefficients)

    def tracking_penalty(self, values):
        """Original-unit L1 tracking cost; no high-priority coefficient applies."""
        return float(sum(weight * values.get(key, 0.) for key, weight, level in
                         zip(self.keys, self.weights, self.levels) if level == TRACKING_LEVEL))

    def high_coefficients(self, coefficients):
        """Per-key coefficients for only the four high-priority L1 levels."""
        return np.asarray([coefficients[level] if level < TRACKING_LEVEL else 0.
                           for level in self.levels])


def _providers(objective):
    values = [getattr(objective, 'priority_clearance', None), objective]
    values += list(getattr(objective, 'candidate_guards', ()))
    terms = getattr(objective, 'terms', ())
    values += list(terms.values() if isinstance(terms, dict) else terms)
    seen = set()
    for value in values:
        if value is not None and id(value) not in seen:
            seen.add(id(value))
            yield value


def collect_soft(objective, poses, size):
    result = SoftTerms(size)
    for provider in _providers(objective):
        for name, cone in (('soft_rows', False), ('soft_cones', True)):
            function = getattr(provider, name, None)
            if function is not None:
                for item in function(poses):
                    result.add(item, cone=cone)
    return result


def assemble_program(blocks, soft, size):
    linear = [block for block in blocks if not block.soc]
    cones = [block for block in blocks if block.soc]
    extra = len(soft.keys)
    extend = lambda A: sparse.hstack((A, sparse.csr_matrix((A.shape[0], extra))), format='csr')
    matrices, offsets = [extend(v.A) for v in linear], [v.b for v in linear]
    head_groups, count = [], sum(len(v.b) for v in linear)
    for matrix, upper, indices in soft.rows:
        slack = sparse.csr_matrix((-np.ones(len(indices)), (np.arange(len(indices)), indices)),
                                  shape=(len(indices), extra))
        matrices.append(sparse.hstack((matrix, slack), format='csr'))
        offsets.append(upper)
        head_groups.append((count + np.arange(len(indices)), indices))
        count += len(indices)
    if extra:
        # Keep upper rows before slack nonnegativity: existing callers use
        # the last `extra` scalar rows as the mandatory nonnegative domain.
        limits = getattr(soft, 'limits', {})  # Older numerical captures.
        if limits:
            indices = list(limits)
            bound_matrix = sparse.csr_matrix((np.ones(len(indices)),
                              (np.arange(len(indices)), indices)), shape=(len(indices), extra))
            matrices.append(sparse.hstack((sparse.csr_matrix((len(indices), size)), bound_matrix), format='csr'))
            offsets.append(np.asarray(list(limits.values())))
            count += len(indices)
        matrices.append(sparse.hstack((sparse.csr_matrix((extra, size)), -sparse.eye(extra)), format='csr'))
        offsets.append(np.zeros(extra))
        count += extra
    nonneg = count
    sizes = []
    for block in cones:
        matrices.append(extend(block.A))
        offsets.append(block.b)
        sizes.extend(block.soc)
        count += len(block.b)
    for matrix, residual, upper, indices in soft.cones:
        block = norms(matrix, residual, upper)
        slack = sparse.csr_matrix((-np.ones(len(indices)), (4 * np.arange(len(indices)), indices)),
                                  shape=(len(block.b), extra))
        matrices.append(sparse.hstack((block.A, slack), format='csr'))
        offsets.append(block.b)
        head_groups.append((count + 4 * np.arange(len(indices)), indices))
        sizes.extend(block.soc)
        count += len(block.b)
    soft.heads = head_groups
    A = sparse.vstack(matrices, format='csc') if matrices else sparse.csc_matrix((0, size + extra))
    b = np.concatenate(offsets) if offsets else np.empty(0)
    return ConeProgram(A, b, nonneg, tuple(sizes), size, tuple(linear + cones), soft)


def prepare_model(objective, poses, system, gradient, diagonal):
    """Call priority begin hook, then freeze hard/soft rows in the parent before forks."""
    from .shape_step import _pin_root
    coords = objective.coords
    lower, upper = coords.bounds(poses, objective.cfg['angular_step_rad'],
                                 objective.cfg['translation_step_m'])
    _pin_root(objective, poses, lower, upper)
    lower[:, coords.scalar_cols] = np.maximum(lower[:, coords.scalar_cols],
                                             objective.lower - poses[:, coords.qi])
    upper[:, coords.scalar_cols] = np.minimum(upper[:, coords.scalar_cols],
                                             objective.upper - poses[:, coords.qi])
    origin = np.array(poses, copy=True)
    origin.setflags(write=False)
    model = IterationModel(origin, coords, lower, upper, system, sparse_hessian(system),
                           gradient, diagonal, max(float(np.linalg.norm(gradient, np.inf)), 1.))
    coordinator = getattr(objective, 'priority_clearance', None)
    if coordinator is not None:
        coordinator.begin_iteration(poses, model)
    model.soft = collect_soft(objective, poses, gradient.size)
    model.program = assemble_program(guard_blocks(objective, poses, lower, upper), model.soft, gradient.size)
    return model


def quadratic(model, damping):
    matrix = model.hessian + sparse.diags(damping * model.diagonal + 1e-9)
    extra = model.program.A.shape[1] - model.gradient.size
    P = sparse.block_diag((2 * matrix / model.scale, sparse.csc_matrix((extra, extra))), format='csc')
    c = np.r_[2 * model.gradient / model.scale, np.zeros(extra)]
    # A tracking epigraph is part of the tracking model itself. Divide by
    # the same numerical scale as H/g, never multiply by a contact penalty.
    tracking = np.asarray(model.soft.levels) == TRACKING_LEVEL
    c[model.gradient.size:] += np.where(tracking, np.asarray(model.soft.weights), 0.) / model.scale
    return P, c


def solve_model(model, damping, *, priority=False, descent_penalty=None):
    P, c = quadratic(model, damping)
    from .runtime_options import OPTIONS
    if priority and OPTIONS.priority_zero_yield_descent:
        from .descent_step import solve_descent, NORMALIZED_PENALTY
        return solve_descent(model, P, c, penalty=(NORMALIZED_PENALTY if descent_penalty is None
                                                 else descent_penalty))
    if model.soft.keys:
        from .priority_qp import solve_priority
        return solve_priority(model, P, c, strict=priority)
    from .window_qp import WINDOW_FRAMES, solve_standard
    frames, width = model.step_lower.shape
    if OPTIONS.window_qp and not priority and frames >= 2 * WINDOW_FRAMES:
        program = model.program
        step = solve_standard(P, c, program.A, program.b, program.nonneg, program.soc,
                              np.arange(model.gradient.size) // width, frames)
        if step is not None:
            return QPSolution(step)
    result = solve_program(model.program, P, c)
    if result is None:
        return None
    step = np.asarray(result.x)
    certificate = (dict(coefficients=[0.] * 4, multiplier_bounds=[0.] * 4, certified=True,
                        method='no_soft_rows', predicted_reductions=dict(levels=[0.] * 4, keys={}))
                   if priority else {})
    return QPSolution(step, certificate=certificate, primal=step, dual=np.asarray(result.z))
