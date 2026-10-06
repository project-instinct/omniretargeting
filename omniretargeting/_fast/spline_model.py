"""All-frame normal equations and unchanged cones in correction coefficients.

Only the per-frame W x W moments supplied by FastObjective are dense. There
is no (N W) x (N W) Hessian, dense or sparse, before the coefficient assembly.
"""
from functools import lru_cache
import time

import numpy as np
from scipy import sparse

from ..motion_budgets import high_frequency, chatter_budget, CHATTER_EXCESS_TOLERANCE
from .continuity import CorrectionContinuity, temporal_signals
from .qp_assembly import SoftTerms
from .qp_guard_rows import HardBlock, guard_blocks, intervals
from .qp_types import ConeProgram, IterationModel
from .shape_step import _pin_root


def coefficient_program(blocks, size):
    """Exact zero-epigraph assembly without per-block zero-width hstacks."""
    linear = [block for block in blocks if not block.soc]
    cones = [block for block in blocks if block.soc]
    ordered = linear + cones
    soft = SoftTerms(size)
    matrix = (sparse.vstack([block.A for block in ordered], format='csc')
              if ordered else sparse.csc_matrix((0, size)))
    offsets = np.concatenate([block.b for block in ordered]) if ordered else np.empty(0)
    return ConeProgram(matrix, offsets, sum(len(block.b) for block in linear),
                       tuple(size for block in cones for size in block.soc), size,
                       tuple(ordered), soft)


def frame_normal_equations(basis, differential, hessian, gradient):
    """Accumulate local moments directly into sparse coefficient blocks."""
    basis = sparse.csr_matrix(basis)
    count, knots = basis.shape
    width = gradient.shape[1]
    transformed = differential.transpose(0, 2, 1) @ hessian @ differential
    pull = (differential.transpose(0, 2, 1) @ gradient[..., None])[..., 0]
    out_gradient = np.asarray(basis.T @ pull).ravel()
    # Each original frame contributes to at most 16 pairs of cubic controls.
    pairs, frames, products = [], [], []
    for frame in range(count):
        start, stop = basis.indptr[frame:frame + 2]
        index, weight = basis.indices[start:stop], basis.data[start:stop]
        pairs.extend((index[:, None] * knots + index).ravel())
        frames.extend([frame] * len(index)**2)
        products.extend((weight[:, None] * weight).ravel())
    pairs = np.asarray(pairs, int)
    unique, inverse = np.unique(pairs, return_inverse=True)
    blocks = np.zeros((len(unique), width, width))
    # Bound temporary storage even for an identity/full-rate basis.
    for start in range(0, len(pairs), 2048):
        take = slice(start, start + 2048)
        value = transformed[np.asarray(frames[take])] * np.asarray(products[take])[:, None, None]
        np.add.at(blocks, inverse[take], value)
    row = (unique // knots)[:, None, None] * width + np.arange(width)[None, :, None]
    col = (unique % knots)[:, None, None] * width + np.arange(width)[None, None, :]
    out_hessian = sparse.coo_matrix((blocks.ravel(),
        (np.broadcast_to(row, blocks.shape).ravel(), np.broadcast_to(col, blocks.shape).ravel())),
        shape=(knots * width, knots * width)).tocsc()
    out_hessian.eliminate_zeros()
    return out_hessian, out_gradient


@lru_cache(maxsize=8)
def high_frequency_operator(count):
    """Sparse linearization of the actual motion_budgets filter, unchanged."""
    parts = []
    for first in range(0, count, 64):
        columns = np.arange(first, min(count, first + 64))
        impulses = np.zeros((count, len(columns)))
        impulses[columns, np.arange(len(columns))] = 1.
        parts.append(sparse.csr_matrix(high_frequency(impulses)))
    return sparse.hstack(parts, format='csr') if parts else sparse.csr_matrix((0, 0))


class ChatterRows:
    """Linearize existing per-coordinate budgets; no new filter or thresholds."""
    def __init__(self, coords, reference):
        self.coords = coords
        self.reference = temporal_signals(coords, reference)
        self.bounds = {name: chatter_budget(value) + CHATTER_EXCESS_TOLERANCE
                       for name, value in self.reference.items()}
        for value in self.bounds.values():
            value.setflags(write=False)
        self.angular = CorrectionContinuity(coords, reference, chatter=False, correction=False)

    def blocks(self, poses):
        coords, count, width = self.coords, len(poses), self.coords.width
        current = temporal_signals(coords, poses)
        matrices = {'joints': sparse.kron(sparse.eye(count), coords.selector, format='csr')}
        if coords.root_q is not None:
            select = np.zeros((3, width))
            for axis in range(3):
                col = coords.inverse.get(coords.root_d + axis)
                if col is not None:
                    select[axis, col] = 1.
            matrices['root_translation'] = sparse.kron(sparse.eye(count), select, format='csr')
            if count > 1:
                matrices['root_rotation'] = self.angular.linearizations(poses)['root_rotation_order_1'][1]
        blocks = []
        for name, matrix in matrices.items():
            value = current[name]
            if not value.size:
                continue
            operator = high_frequency_operator(len(value))
            residual = high_frequency(value - self.reference[name]).ravel()
            jacobian = sparse.kron(operator, sparse.eye(value.shape[1]), format='csr') @ matrix
            bound = self.bounds[name].ravel()
            blocks.append(intervals(jacobian, -bound - residual, bound - residual))
        return blocks


def prepare_spline_model(objective, spline, coefficients, evaluation, *, chatter=None,
                         geometry=None, controls=None, columns=None, trust_scale=1.):
    """All original objective/guard rows, mapped by d Integrate(ref,Ba)/da."""
    timings, tick = {}, time.monotonic()
    poses = spline.embed(coefficients)
    acc, residual, temporal_j, _, _ = evaluation
    mapping = spline.jacobian(coefficients)
    hessian, gradient = frame_normal_equations(
        spline.basis, spline.tangent_differentials(coefficients), acc.hessian, acc.gradient)
    now = time.monotonic()
    timings['frame_moments_and_mapping'], tick = now - tick, now
    temporal = temporal_j @ mapping
    hessian = (hessian + temporal.T @ temporal).tocsc()
    gradient += np.asarray(temporal.T @ residual).ravel()
    # Project only local objective damping, like the production LM solver.
    frame_diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2), 1e-9)
    diagonal = np.asarray(mapping.power(2).T @ frame_diagonal.ravel()).ravel()
    diagonal = np.maximum(diagonal, 1e-9)
    if not 0 < trust_scale <= 1:
        raise ValueError('Spline increment trust scale must be in (0, 1]')
    lower, upper = objective.coords.bounds(poses, objective.cfg['angular_step_rad'] * trust_scale,
                                          objective.cfg['translation_step_m'] * trust_scale)
    _pin_root(objective, poses, lower, upper)
    scalar = objective.coords.scalar_cols
    lower[:, scalar] = np.maximum(lower[:, scalar], objective.lower - poses[:, objective.coords.qi])
    upper[:, scalar] = np.minimum(upper[:, scalar], objective.upper - poses[:, objective.coords.qi])
    now = time.monotonic()
    timings['temporal_lm_and_bounds'], tick = now - tick, now
    blocks = guard_blocks(objective, poses, lower, upper)
    now = time.monotonic()
    timings['original_guard_rows'], tick = now - tick, now
    if chatter is not None:
        blocks += chatter.blocks(poses)
    now = time.monotonic()
    timings['motion_budget_rows'], tick = now - tick, now
    if geometry is not None:
        blocks += geometry.blocks(poses)
    now = time.monotonic()
    timings['original_geometry_rows'], tick = now - tick, now
    blocks = [HardBlock((block.A @ mapping).tocsr(), block.b, block.soc, block.evaluate)
              for block in blocks]
    now = time.monotonic()
    timings['coefficient_row_mapping'], tick = now - tick, now
    size = coefficients.size
    selected = np.arange(size)
    if controls is not None or columns is not None:
        controls = np.arange(spline.controls) if controls is None else np.asarray(controls)
        columns = np.arange(objective.width) if columns is None else np.asarray(columns)
        selected = (controls[:, None] * objective.width + columns).ravel()
        hessian = hessian[selected][:, selected].tocsc()
        gradient, diagonal = gradient[selected], diagonal[selected]
        blocks = [HardBlock(block.A[:, selected], block.b, block.soc, block.evaluate) for block in blocks]
        size = len(selected)
    now = time.monotonic()
    timings['column_selection'], tick = now - tick, now
    program = coefficient_program(blocks, size)
    timings['program_assembly'] = time.monotonic() - tick
    soft = program.soft
    model = IterationModel(poses, objective.coords, lower, upper, None, hessian, gradient,
                           diagonal, max(float(np.max(np.abs(gradient), initial=0.)), 1.),
                           program=program, soft=soft)
    model.spline_assembly_seconds = timings
    # Geometry has its own higher-priority epigraph level. Tracking cannot
    # purchase penetration reduction with a large source/contact cost decrease.
    if geometry is not None:
        model.spline_mapping, model.spline_selected = mapping, selected
        model.spline_geometry = geometry.linearized
    return model, mapping, selected
