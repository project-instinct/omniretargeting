"""Objective-preserving, localized second-order constraint-offset solves."""

from dataclasses import replace

import numpy as np
from scipy import sparse

from .local_restoration import _support
from .qp_assembly import assemble_program
from .qp_reduction import solve_restricted

SOC_FRACTIONS = (1., .5)
SOC_HALOS = (8, 16, 32)
MAX_SOC_SOLVES = 3
SOC_DENSE_FRACTION = .6


def failing_frames(objective, candidate, guards):
    bad = np.zeros(len(candidate), bool)
    for guard in guards:
        if hasattr(guard, 'violation_frames'):
            bad |= np.asarray(guard.violation_frames(candidate), bool)
        elif not guard.accepts(candidate):
            bad[:] = True
    return bad


def defect_program(model, candidate, step):
    """Freeze original Jacobians; shift only failing scalar rows or whole cones.

    For a scalar g<=0, b_new=-g(x+p)+Jp. For a norm cone the
    complete vector remainder is retained, with the original cone radius.
    """
    blocks, changed = [], False
    for block in model.program.hard_blocks:
        if block.evaluate is None:
            blocks.append(block)
            continue
        value = block.evaluate(candidate)
        if value is None:
            # An explicitly unavailable batch (e.g. changed mesh witnesses)
            # stays unchanged; other aligned hard rows still receive SOC.
            blocks.append(block)
            continue
        measured = np.asarray(value)
        if measured.shape != block.b.shape or not np.isfinite(measured).all():
            return None
        shifted = measured + block.A @ step
        bad = measured < 0 if not block.soc else np.zeros(len(measured), bool)
        if block.soc:
            start = 0
            for size in block.soc:
                vector = measured[start:start + size]
                if np.linalg.norm(vector[1:]) > vector[0]:
                    bad[start:start + size] = True
                start += size
        changed |= bool(bad.any())
        blocks.append(replace(block, b=np.where(bad, shifted, block.b)))
    return assemble_program(blocks, model.soft, model.gradient.size) if changed else None


def corrected_candidates(objective, model, solution, damping, guards, *, priority=False):
    """At most three local solves; expand the halo without dropping boundary rows."""
    from .priority_merit import check_trial
    frames, width = model.step_lower.shape
    solves = 0
    for fraction in SOC_FRACTIONS:
        step = fraction * solution.step
        candidate = objective.project(model.coords.integrate(model.origin, step.reshape(frames, width), 1.))
        if check_trial(objective, candidate, guards):
            continue
        bad = failing_frames(objective, candidate, guards)
        if not bad.any():
            continue  # coordinator rejection, not a hard-guard curvature defect
        program = defect_program(model, candidate, step)
        if program is None:
            continue
        for halo in SOC_HALOS:
            if solves >= MAX_SOC_SOLVES:
                return
            support = _support(bad, halo)
            # Dense full-step failures get a half-step SOC attempt first.
            if fraction == 1. and np.mean(support) > SOC_DENSE_FRACTION:
                break
            columns = np.flatnonzero(np.repeat(support, width))
            transform = sparse.eye(frames * width, format='csc')[:, columns]
            fixed = step.copy()
            fixed[columns] = 0.
            solved = solve_restricted(model, damping, transform, fixed, priority=priority, program=program)
            solves += 1
            if solved is not None:
                corrected = solved[0]
                candidate = objective.project(model.coords.integrate(
                    model.origin, corrected.step.reshape(frames, width), 1.))
                if check_trial(objective, candidate, guards):
                    corrected.certificate['soc'] = dict(halo=halo, frames=int(support.sum()),
                                                         starting_fraction=fraction, solves=solves)
                    yield corrected, candidate, fraction
                    break
            if solves >= MAX_SOC_SOLVES:
                return
