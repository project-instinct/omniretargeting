"""Interpolate only corrections, then certify with unchanged full-rate guards."""

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from .proposals import guards_of
from .coarse_diagnostics import failure_details
from .coarse_embedding import LinearCorrectionEmbedding


def interpolate_correction(coords, fixed, coarse, indices):
    """The exact linear correction operator used inside all coarse guards."""
    return LinearCorrectionEmbedding(coords, fixed, indices).correction(coarse)


def failure_frames(objective, poses, *, certify=False):
    """Check every guard, plus projection-domain limits, with original slack.

    Certification uses ordinary whole-trajectory accepts once per guard;
    only failed guards need a second evaluation to locate violation frames.
    The optional window screen remains available to diagnostic callers.
    """
    count, coords = len(poses), objective.coords
    bad = ~np.isfinite(poses).all(axis=1)
    if bad.any():
        return {'nonfinite': np.flatnonzero(bad).tolist()}
    result = {}
    values = poses[:, coords.qi]
    bad = ((values < objective.lower) | (values > objective.upper)).any(axis=1)
    if bad.any():
        result['joint_limits'] = np.flatnonzero(bad).tolist()
    q = coords.root_q
    if q is not None:
        # Bounds used by project/conic rows, even though ReferenceShape's
        # accepts method only checks layout. Certification also checks these
        # domains after the same projection used by the solver.
        references = []
        if objective.root_budget is not None:
            references.append((objective.reference, objective.root_budget))
        shape = getattr(objective, 'source_shape', None)
        if shape is not None:
            references.append((shape.seed, shape.xy))
            if shape.acquired_poses is not None:
                references.append((shape.acquired_poses, shape.xy))
        for reference, bound in references:
            bad = np.linalg.norm(poses[:, q:q + 3] - reference[:, q:q + 3], axis=1) > bound + 1e-7
            if bad.any():
                result['root_translation_ball'] = sorted(set(result.get('root_translation_ball', ()))
                                                         | set(np.flatnonzero(bad).tolist()))
        fixed_root = getattr(objective, 'fixed_root', None)
        if fixed_root is not None:
            bad = (poses[:, q:q + 7] != fixed_root).any(axis=1)
            if bad.any():
                result['fixed_root'] = np.flatnonzero(bad).tolist()
    for index, guard in enumerate(guards_of(objective)[1]):
        key = f'{type(guard).__name__}:{index}'
        method = getattr(guard, 'violation_frames', None)
        if certify:
            if guard.accepts(poses):
                continue
            frames = [] if method is None else np.flatnonzero(method(poses)).tolist()
            # An accepts/screen disagreement is always a rejection.
            result[key] = frames or list(range(count))
        elif method is None:
            if not guard.accepts(poses):
                result[key] = list(range(count))
        else:
            frames = np.flatnonzero(method(poses, frames=range(count))).tolist()
            if frames:
                result[key] = frames
    return result


def scale_transfer_correction(objective, fixed, coarse, indices, *, min_retention=.25,
                              local_rounds=12, spread_frames=12):
    """Emergency recovery only; the normal composed-guard path never scales.

    Every scale is integrated and projected before certification. Retention
    measures actual projected tangent energy, so a clipped correction cannot
    masquerade as a fully retained warm start. No guard may be relaxed.
    """
    delta = interpolate_correction(objective.coords, fixed, coarse, indices)
    energy = np.sum(delta**2, axis=1)
    total = float(energy.sum())
    report = dict(retained_scale=0., global_scale=0., scale_min=0., scale_mean=0.,
                  scales=np.zeros(len(fixed)).tolist(), guard_failures=[], attempts=[],
                  first_failing_candidate=None, projection_frames=[],
                  interpolation='linear corrections; world-frame root angular transport',
                  retention_metric='projected correction RMS / interpolated correction RMS',
                  accepted=False, reason='no_coarse_correction')
    if total <= 1e-24:
        return fixed.copy(), report
    accumulated, projected_frames = {}, set()
    last_candidate, last_failures = None, {}

    def evaluate(scales, kind):
        nonlocal last_candidate, last_failures
        raw = objective.coords.integrate(fixed, delta * scales[:, None], 1.)
        candidate = objective.project(raw)
        projected = np.flatnonzero((candidate != raw).any(axis=1)).tolist()
        projected_frames.update(projected)
        failures = failure_frames(objective, candidate, certify=True)
        for guard, frames in failures.items():
            accumulated.setdefault(guard, set()).update(frames)
        report['attempts'].append(dict(kind=kind, scale_min=float(scales.min()),
                                       scale_mean=float(scales.mean()), failures=failures,
                                       projection_frames=projected))
        if failures and report['first_failing_candidate'] is None:
            report['first_failing_candidate'] = dict(
                kind=kind, scale_min=float(scales.min()), scale_mean=float(scales.mean()),
                guard_failure_counts={guard: len(frames) for guard, frames in failures.items()},
                failures=failures, details=failure_details(objective, candidate, failures))
        last_candidate, last_failures = candidate, failures
        actual = objective.coords.difference(fixed, candidate)
        # Pose subtraction adds last-bit noise at an exact retention threshold.
        # Round this dimensionless heuristic only; guard decisions are exact above.
        retained = float(np.round(np.sqrt(np.sum(actual**2) / total), 15))
        return candidate, failures, retained

    best, best_scale, best_retained = fixed.copy(), np.zeros(len(fixed)), 0.
    initial_failures = None
    for alpha in (1., .5, .25, .125, .0625, .03125):
        scales = np.full(len(fixed), alpha)
        candidate, failures, retained = evaluate(scales, 'global')
        if initial_failures is None:
            initial_failures = failures
        if not failures:
            best, best_scale, best_retained = candidate, scales, retained
            report['global_scale'] = alpha
            break
    # Global scaling needs no new temporal transition. Local scaling is only
    # retained after all original derivative stencils and geometric guards pass.
    if initial_failures:
        scales = np.ones(len(fixed))
        failures = initial_failures
        for _ in range(local_rounds):
            bad = np.zeros(len(fixed))
            for frames in failures.values():
                bad[frames] = 1.
            influence = maximum_filter1d(bad, size=2 * spread_frames + 1, mode='nearest')
            influence = gaussian_filter1d(influence, max(1., spread_frames / 2), mode='nearest')
            scales *= 1 - .5 * influence
            candidate, failures, retained = evaluate(scales, 'local')
            if not failures:
                if retained > best_retained:
                    best, best_scale, best_retained = candidate, scales.copy(), retained
                break
    report.update(retained_scale=best_retained, scale_min=float(best_scale.min()),
                  scale_mean=float(best_scale.mean()), scales=best_scale.tolist(),
                  guard_failures=[dict(guard=guard, frames=sorted(frames))
                                  for guard, frames in accumulated.items()],
                  projection_frames=sorted(projected_frames),
                  accepted=best_retained >= min_retention,
                  reason='accepted' if best_retained >= min_retention else 'poor_retention')
    if last_failures:
        report['last_failing_candidate'] = dict(
            guard_failure_counts={guard: len(frames) for guard, frames in last_failures.items()},
            details=failure_details(objective, last_candidate, last_failures))
    # best is one of the exact whole-trajectory candidates certified above;
    # selecting it changes neither poses nor scales and needs no repeat audit.
    return (best if report['accepted'] else fixed.copy()), report


def transfer_correction(objective, fixed, coarse, indices, *, min_retention=.25,
                        local_rounds=12, spread_frames=12, safety_scale=False):
    """Embed the certified coarse solution unchanged; a failed check is a bug.

    No projection, smoothing, or normal retention threshold applies. Existing
    scale recovery remains explicitly available as an emergency safety net;
    the acquisition driver falls back to its normal schedule on any failure.
    """
    embedding = LinearCorrectionEmbedding(objective.coords, fixed, indices)
    warm = embedding.embed(coarse)
    failures = failure_frames(objective, warm, certify=True)
    good = not failures
    report = dict(accepted=good, reason='accepted' if good else 'full_rate_interpolant_failure',
                  interpolation='linear corrections; world-frame root angular transport',
                  projected=False, projection_frames=[], retained_scale=1. if good else 0.,
                  global_scale=1. if good else 0., scale_min=1. if good else 0.,
                  scale_mean=1. if good else 0., scales=np.full(len(fixed), float(good)).tolist(),
                  safety_net_triggered=not good,
                  has_coarse_correction=bool(np.any(embedding.correction(coarse))),
                  guard_failures=[dict(guard=guard, frames=frames) for guard, frames in failures.items()],
                  attempts=[dict(kind='linear_interpolant', scale_min=1., scale_mean=1.,
                                 failures=failures, projection_frames=[])],
                  first_failing_candidate=None)
    if failures:
        report['first_failing_candidate'] = dict(kind='linear_interpolant', scale_min=1., scale_mean=1.,
            guard_failure_counts={name: len(frames) for name, frames in failures.items()},
            failures=failures, details=failure_details(objective, warm, failures))
        report['invariant_failure'] = 'BUG: a coarse guard accepted an infeasible full-rate interpolant'
        if safety_scale:
            result, recovery = scale_transfer_correction(
                objective, fixed, coarse, indices, min_retention=min_retention,
                local_rounds=local_rounds, spread_frames=spread_frames)
            recovery.update(safety_net_triggered=True, invariant_failure=report['invariant_failure'],
                            initial_certification=report)
            return result, recovery
    return (warm if good else fixed.copy()), report
