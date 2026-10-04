"""Frozen-objective second-order offsets for whole local rows and norm cones."""

from copy import copy
from dataclasses import replace

import numpy as np

from .qp_assembly import assemble_program
from .second_order_step import defect_program


def corrected_model(model, candidate, step):
    """Keep H, g and every Jacobian; use only aligned nonlinear value APIs.

    Missing same-witness values leave that batch untouched. This is a proposal
    model, never an alternative nonlinear acceptance or an optimum certificate.
    """
    hard = defect_program(model, candidate, step)
    soft = copy(model.soft)
    soft.rows, soft.cones = list(soft.rows), list(soft.cones)
    values = model.predicted_slacks(step)
    slacks = np.array([values[k] for k in soft.keys])
    changed = hard is not None
    for provider, r0, r1, c0, c1 in model.window_soft_sources:
        function = getattr(provider, 'soft_values', None)
        if function is None:
            continue
        linear, cones = function(candidate, frames=model.frames)
        if len(linear) != r1 - r0 or len(cones) != c1 - c0:
            raise ValueError('Soft SOC batch identities changed')
        for index, measured in enumerate(linear, r0):
            if measured is None:
                continue
            A, bound, keys = soft.rows[index]
            measured = np.asarray(measured, float)
            if measured.shape != bound.shape or not np.isfinite(measured).all():
                return None
            bad = -measured > slacks[keys]
            changed |= bool(bad.any())
            soft.rows[index] = A, np.where(bad, measured + A @ step, bound), keys
        for index, measured in enumerate(cones, c0):
            if measured is None:
                continue
            A, residual, bound, keys = soft.cones[index]
            value, upper = map(lambda x: np.asarray(x, float), measured)
            if (value.shape != residual.shape or upper.shape != bound.shape
                    or not np.isfinite(value).all() or not np.isfinite(upper).all()):
                return None
            bad = np.linalg.norm(value, axis=1) > upper + slacks[keys]
            changed |= bool(bad.any())
            offset = value - (A @ step).reshape(-1, 3)
            soft.cones[index] = A, np.where(bad[:, None], offset, residual), np.where(bad, upper, bound), keys
    if not changed:
        return None
    blocks = model.program.hard_blocks if hard is None else hard.hard_blocks
    return replace(model, program=assemble_program(blocks, soft, model.gradient.size), soft=soft,
                   priority_cache=None, solve_diagnostics=[])
