"""Constraint generation for one complete local QP; never discard a violation."""

from dataclasses import replace
import time

import numpy as np

from .qp_types import settings, solve_program
from .window_diagnostics import source_counts
from .window_redundancy import box_redundancy, duplicate_cones, cone_groups
from .window_subproblem import solve_subset

# Screening lookahead only: original feasibility/acceptance tolerances do not
# use this band. A 1 mm band keeps near witnesses through boundary switches.
ACTIVATION_MARGIN = 1e-3
MAX_GENERATIONS = 3


def residuals(program, x):
    """Scalar slack and SOC head-minus-tail-norm, one entry per entire cone."""
    s = program.b - program.A @ x
    sizes, starts = cone_groups(program)
    margin = np.empty(len(sizes))
    for size in np.unique(sizes):
        mask = sizes == size
        values = s[starts[mask, None] + np.arange(size)]
        margin[mask] = values[:, 0] - np.linalg.norm(values[:, 1:], axis=1)
    return s[:program.nonneg], margin


def selected_rows(program, linear, cones):
    sizes, starts = cone_groups(program)
    return np.r_[np.flatnonzero(linear),
                 *(np.arange(starts[k], starts[k] + sizes[k]) for k in np.flatnonzero(cones))].astype(int)


def _subset(program, linear, cones):
    sizes, _ = cone_groups(program)
    rows = selected_rows(program, linear, cones)
    return replace(program, A=program.A[rows].tocsc(), b=program.b[rows],
                   nonneg=int(np.count_nonzero(linear)), soc=tuple(sizes[cones]),
                   hard_blocks=(), soft=None, last_solve={}), rows


def initial_working_set(model, mandatory):
    program = model.program
    mandatory = np.asarray(mandatory, bool)
    if mandatory.shape != program.b.shape:
        raise ValueError('Mandatory mask must index all original rows')
    zero = np.zeros(model.gradient.size)
    values = model.predicted_slacks(zero)
    point = np.r_[zero, [values[key] for key in model.soft.keys]]
    linear_margin, cone_margin = residuals(program, point)
    sizes, starts = cone_groups(program)
    linear = mandatory[:program.nonneg] | (linear_margin <= ACTIVATION_MARGIN)
    cones = cone_margin <= ACTIVATION_MARGIN
    for size in np.unique(sizes):
        mask = sizes == size
        cones[mask] |= mandatory[starts[mask, None] + np.arange(size)].any(axis=1)
    _seed_contact_witnesses(model, linear, cones, linear_margin, cone_margin, starts)
    return linear, cones


def _seed_contact_witnesses(model, linear, cones, scalar_margin, cone_margin, starts):
    """Keep the tightest scalar/norm witness per contact epigraph initially.

    Without any representative, a loose-at-origin contact is initially absent
    and the first proposal can activate thousands of its samples at once.
    Every other witness remains subject to complete original separation.
    """
    head_cone = {int(row): k for k, row in enumerate(starts)}
    best = {}
    for heads, keys in model.soft.heads:
        for head, key in zip(heads, keys):
            if model.soft.keys[key][0] != 'contact':
                continue
            norm = head >= model.program.nonneg
            index = head_cone[int(head)] if norm else int(head)
            margin = cone_margin[index] if norm else scalar_margin[index]
            identity = int(key), norm
            if identity not in best or margin < best[identity][0]:
                best[identity] = margin, index
    for (_, norm), (_, index) in best.items():
        (cones if norm else linear)[index] = True


def violated_rows(program, x):
    """Per-row original-unit checks; complete cones always stay together."""
    scalar, cones = residuals(program, x)
    ax = program.A @ x
    tolerance = settings().tol_feas
    scale = np.maximum(1., np.maximum(abs(program.b), abs(ax)))
    linear_bad = (~np.isfinite(scalar)) | (scalar < -tolerance * scale[:program.nonneg])
    sizes, starts = cone_groups(program)
    cone_bad = ~np.isfinite(cones)
    for size in np.unique(sizes):
        mask = sizes == size
        block_scale = scale[starts[mask, None] + np.arange(size)].max(axis=1)
        cone_bad[mask] |= cones[mask] < -tolerance * block_scale
    return linear_bad, cone_bad


def _working_domain(model, mandatory):
    """Reuse only this immutable program's proof/active set across damping."""
    program = model.program
    base_linear, base_cones = initial_working_set(model, mandatory)
    cache = model.window_working_set
    reused = cache is not None and cache['program'] is program
    if not reused:
        proof_linear, proof_cones, protected = box_redundancy(model)
        representatives = duplicate_cones(program)
        cache = dict(program=program, proof_linear=proof_linear, proof_cones=proof_cones,
                     protected=protected, representatives=representatives,
                     linear=np.zeros_like(base_linear), cones=np.zeros_like(base_cones))
        model.window_working_set = cache
    linear = (base_linear | cache['linear'] | cache['protected']) & ~cache['proof_linear']
    cones = _canonical_cones(base_cones | cache['cones'], cache) & ~cache['proof_cones']
    return cache, linear, cones, reused


def _canonical_cones(mask, cache):
    result = np.zeros_like(mask)
    result[cache['representatives'][mask]] = True
    return result


def _solve_round(model, P, c, linear, cones, generation, precision, reason, trace):
    program = model.program
    tick = time.monotonic()
    work, rows = _subset(program, linear, cones)
    subset_seconds = time.monotonic() - tick
    tick = time.monotonic()
    result = solve_subset(work, P, c, solve_program, precision=precision)
    record = dict(generation=generation, rows=len(rows), cones=len(work.soc),
                  variables=work.A.shape[1], matrix_nonzeros=work.A.nnz,
                  quadratic_nonzeros=P.nnz, original_rows=len(program.b),
                  original_cones=len(program.soc), full_program=bool(linear.all() and cones.all()),
                  sources=source_counts(model, rows), original_sources=source_counts(model),
                  fallback_reason=reason, precision_refinement=precision,
                  screening_seconds=0., subset_seconds=subset_seconds,
                  seconds=time.monotonic() - tick, **work.last_solve)
    trace.append(record)
    if result is None:
        return None, rows, record, None
    x = np.asarray(result.x)
    if x.shape != (program.A.shape[1],) or not np.isfinite(x).all():
        record['invalid_primal'] = True
        return None, rows, record, None
    tick = time.monotonic()
    bad_linear, bad_cones = violated_rows(program, x)
    record['separation_seconds'] = time.monotonic() - tick
    bad_rows = selected_rows(program, bad_linear, bad_cones)
    record.update(violated_linear=int(bad_linear.sum()), violated_cones=int(bad_cones.sum()),
                  violated_sources=source_counts(model, bad_rows),
                  active_violated_linear=int((bad_linear & linear).sum()),
                  active_violated_cones=int((bad_cones & cones).sum()))
    return result, rows, record, (bad_linear, bad_cones)


def solve_working_set(model, P, c, mandatory, *, max_generations=MAX_GENERATIONS):
    """Solve a subset, separate every omitted row, then expand or use full QP.

    Objective and variables remain identical in every round. Failure returns
    None, never a candidate whose omitted hard/soft rows were unchecked.
    """
    program = model.program
    tick = time.monotonic()
    cache, linear, cones, reused = _working_domain(model, mandatory)
    screening_seconds = time.monotonic() - tick
    trace = []
    fallback_reason = None
    # Three generation updates permit four subset solves. A nearly-complete
    # working set must get its final separation attempt before a full fallback.
    for generation in range(max_generations + 2):
        full = fallback_reason is not None or generation > max_generations
        if full:
            fallback_reason = fallback_reason or 'generation_limit'
            linear[:], cones[:] = True, True
        else:
            cache['linear'], cache['cones'] = linear.copy(), cones.copy()
        result, rows, record, bad = _solve_round(model, P, c, linear, cones, generation,
                                                False, fallback_reason, trace)
        if generation == 0:
            record.update(screening_seconds=screening_seconds, reused_working_set=reused,
                          box_proven_linear=int(cache['proof_linear'].sum()),
                          box_proven_cones=int(cache['proof_cones'].sum()),
                          duplicate_cones=int(np.count_nonzero(cache['representatives'] !=
                                                              np.arange(len(cones)))))
        # A primal tolerance failure on rows already present is numerical,
        # not evidence that 30k omitted rows need to be added. Refine the
        # identical program once; retain the original separation threshold.
        if result is None or (bad[0] & linear).any() or (bad[1] & cones).any():
            result, rows, record, bad = _solve_round(model, P, c, linear, cones, generation,
                                                    True, fallback_reason, trace)
        if result is None:
            if full or (linear.all() and cones.all()):
                return None, trace
            fallback_reason = 'subset_solver_failed'
            continue
        x = np.asarray(result.x)
        bad_linear, bad_cones = bad
        if not bad_linear.any() and not bad_cones.any():
            dual = np.zeros(len(program.b))
            dual[rows] = result.z
            return (x, dual), trace
        if (bad_linear & linear).any() or (bad_cones & cones).any():
            if full or (linear.all() and cones.all()):
                return None, trace
            fallback_reason = 'active_row_feasibility'
        else:
            scalar_margin, cone_margin = residuals(program, x)
            # Add every violating AND near-active witness, never just the
            # current worst sample. Keep all earlier rounds/damping attempts.
            cache['proof_linear'][bad_linear] = False
            cache['proof_cones'][cache['representatives'][bad_cones]] = False
            add_linear = (bad_linear | (scalar_margin <= ACTIVATION_MARGIN)) & ~cache['proof_linear']
            add_cones = _canonical_cones(bad_cones | (cone_margin <= ACTIVATION_MARGIN), cache)
            add_cones &= ~cache['proof_cones']
            record.update(added_linear=int((add_linear & ~linear).sum()),
                          added_cones=int((add_cones & ~cones).sum()))
            linear |= add_linear
            cones |= add_cones
            cache['linear'], cache['cones'] = linear.copy(), cones.copy()
    return None, trace
