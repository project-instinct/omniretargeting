"""Opt-in video-estimate filtering before SMPL-X FK; never mutates input arrays.

Third differences estimate independent noise (variance gain 20). A channel is
filtered only when that difference is dominated by alternating white noise
(strongly negative lag-one correlation) above a small floor; coherent rapid
motion, however large its acceleration, does not qualify.
The kernel is a Whittaker smoother (third-difference penalty): its response
1/(1+lam(2 sin(pi f/fps))^6) rolls off at 36 dB/octave without the sidelobes
of a local polynomial, so broadband noise does not leak into acceleration.
GCV selects lam per channel from the data itself. Huber reweighting of the
jerk penalty releases sparse large events (takeoff, landing, impacts) while
small alternating jerk keeps the quadratic penalty. Rotations are smoothed as
matrices and projected back to SO(3) (chordal mean), continuous across +/- pi.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solveh_banded
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

# Channel gate: the third difference must be noise-dominated (robust lag-one
# correlation; i.i.d. noise measures about -0.67) and the noise must exceed a
# floor. Correlation alone cannot separate mocap (OMOMO/TRUMANS -0.02 to -0.63)
# from video noise; the floors do (mocap at most 0.2 mm, 0.27 deg).
NOISE_CORRELATION = -0.5
TRANSLATION_FLOOR_M = 1e-3
ROOT_FLOOR_RAD = np.deg2rad(0.20)
BODY_FLOOR_RAD = np.deg2rad(0.35)
# GCV search range of the half-power cutoff, and Huber constant on jerk.
CUTOFFS_HZ = np.geomspace(2.0, 20.0, 40)
HUBER_C = 2.0
HUBER_ITERATIONS = 6
# Tukey bisquare constant on data residuals (outlier samples get zero weight).
BISQUARE_C = 4.685
ROBUST_ITERATIONS = 4
SPIKE_SAMPLES = 3
# Residual scale floor (m or rotation-matrix units): exact data is never an outlier.
RESIDUAL_FLOOR = 1e-3
# Zero-order-hold detection: exact repeats must cover this share of frames.
HOLD_SHARE = 0.2
_D3 = np.array([-1., 3., -3., 1.])


def noise_sigma(values):
    x = np.asarray(values, float)
    if len(x) < 7:
        return np.zeros(x.shape[1:])
    third = np.diff(x, n=3, axis=0)
    return 1.4826 * np.median(np.abs(third - np.median(third, axis=0)), axis=0) / np.sqrt(20)


def incoherence(values):
    """Robust lag-one correlation distinguishes coherent fast motion from noise."""
    v = np.asarray(values, float)
    if len(v) < 3:
        return 0.0
    centre = np.median(v, axis=0)
    scale = 1.4826 * np.median(np.abs(v-centre), axis=0)
    z = np.clip(v-centre, -4*scale, 4*scale)
    norm = np.sqrt(np.sum(z[:-1]**2)*np.sum(z[1:]**2))
    return float(np.sum(z[:-1]*z[1:])/norm) if norm > 1e-20 else 0.0


def runs(mask):
    bounds = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(bounds[::2], bounds[1::2]))


def repair_steps(values, fps):
    """Spread isolated discontinuities while preserving both sides and endpoints."""
    x = np.array(values, dtype=float, copy=True)
    if len(x) < 7:
        return x, []
    delta = np.diff(x, axis=0)
    local = savgol_filter(delta, min(11, len(delta) if len(delta) % 2 else len(delta)-1), 1, axis=0)
    # Smooth motion, including very fast motion, has comparable neighbour steps.
    residual = delta - local
    floor = max(0.12, 12 * float(np.median(np.linalg.norm(residual, axis=1))))
    candidates = np.flatnonzero(np.linalg.norm(residual, axis=1) > floor)
    # A jump that returns within SPIKE_SAMPLES is an outlier sample, not a
    # step: leave it to the robust data weights instead of spreading it.
    spike = set()
    for a in candidates:
        for b in candidates[(candidates > a) & (candidates <= a + SPIKE_SAMPLES)]:
            if np.linalg.norm(delta[a] + delta[b]) < 0.3 * min(np.linalg.norm(delta[a]), np.linalg.norm(delta[b])):
                spike.update((int(a), int(b)))
    records = []
    for i in candidates:
        if int(i) in spike:
            continue
        lo, hi = max(0, i-4), min(len(delta), i+5)
        neighbours = np.delete(delta[lo:hi], i-lo, axis=0)
        jump = delta[i] - np.median(neighbours, axis=0)
        magnitude = float(np.linalg.norm(jump))
        if magnitude < floor:
            continue
        width = max(0.24, np.sqrt(6 * magnitude / 15.0))
        half = max(3, int(np.ceil(width * fps / 2)))
        start, stop = max(0, i+1-half), min(len(x)-1, i+1+half)
        t = np.clip((np.arange(len(x))-start) / max(stop-start, 1), 0, 1)
        ramp = t*t*t * (10 + t*(-15 + 6*t))
        step = (np.arange(len(x)) > i).astype(float)
        x += (ramp-step)[:, None] * jump
        records.append(dict(frame=int(i+1), from_frame=int(i), amplitude_m=magnitude, vector_m=jump.tolist(),
                            raw_step_m=float(np.linalg.norm(delta[i])), raw_vector_m=delta[i].tolist(),
                            start=int(start), stop=int(stop), transition_s=(stop-start)/fps))
    return x, records


def _system(frames, lam, weights, data):
    """Lower band of diag(data) + lam D3^T diag(weights) D3."""
    ab = np.zeros((4, frames))
    ab[0] = data
    rows = np.arange(frames - 3)
    for k in range(4):
        for l in range(k + 1):
            ab[k - l, rows + l] += lam * weights * _D3[k] * _D3[l]
    return ab


def whittaker(values, lam, weights=None, data=None):
    y = np.asarray(values, float)
    w = np.ones(len(y) - 3) if weights is None else weights
    v = np.ones(len(y)) if data is None else data
    rhs = v[:, None] * y.reshape(len(y), -1)
    return solveh_banded(_system(len(y), lam, w, v), rhs, lower=True).reshape(y.shape)


def cutoff_lambda(cutoff, fps):
    return 1. / (2 * np.sin(np.pi * min(cutoff, 0.49 * fps) / fps)) ** 6


def gcv_cutoff(values, fps, data=None):
    """Generalized cross-validation, circulant trace approximation.

    Zero-weight (outlier) samples are excluded from the residual and from
    the effective number of observations.
    """
    y = np.asarray(values, float).reshape(len(values), -1)
    frames = len(y)
    v = np.ones(frames) if data is None else data
    n = float(v.sum())
    response = (2 * np.sin(np.pi * np.arange(frames) / frames)) ** 6
    best = None
    for cutoff in CUTOFFS_HZ:
        lam = cutoff_lambda(cutoff, fps)
        rss = float(np.sum(v[:, None] * (y - whittaker(y, lam, data=v)) ** 2))
        trace = float(np.sum(1. / (1. + lam * response))) * n / frames
        score = n * rss / max(n - trace, 1e-9) ** 2
        if best is None or score < best[0]:
            best = (score, float(cutoff))
    return best[1]


def robust_smooth(values, fps):
    """Robust GCV Whittaker (Garcia 2010) with Huber-reweighted jerk.

    Both weights are updated in one loop, so a genuine sparse event (takeoff,
    landing) is first released by the jerk weights and then fits its samples;
    only samples that no smooth-with-sparse-jerk path explains get bisquare
    weight zero. GCV is evaluated on the current inliers.
    """
    y = np.asarray(values, float)
    flat = y.reshape(len(y), -1)
    data = np.ones(len(y))
    weights = np.ones(len(y) - 3)
    cutoff = gcv_cutoff(y, fps)
    for iteration in range(ROBUST_ITERATIONS + HUBER_ITERATIONS):
        lam = cutoff_lambda(cutoff, fps)
        x = whittaker(y, lam, weights, data)
        jerk = np.linalg.norm(np.diff(x, 3, axis=0).reshape(len(x) - 3, -1), axis=1)
        weights = np.minimum(1., HUBER_C * (1.4826 * np.median(jerk) + 1e-15) / np.maximum(jerk, 1e-15))
        if iteration >= 1:
            residual = np.linalg.norm(flat - x.reshape(len(x), -1), axis=1)
            u = residual / (BISQUARE_C * max(1.4826 * np.median(residual), RESIDUAL_FLOOR))
            data = np.where(u < 1, (1 - u * u) ** 2, 0.)
        if iteration < ROBUST_ITERATIONS:
            cutoff = gcv_cutoff(y, fps, data)
    x = whittaker(y, cutoff_lambda(cutoff, fps), weights, data)
    return x, dict(cutoff_hz=cutoff, released_frames=int(np.sum(weights < 0.5)),
                   outlier_samples=int(np.sum(data == 0)))


def _project_so3(matrices):
    u, _, vt = np.linalg.svd(matrices)
    d = np.sign(np.linalg.det(u @ vt))
    u[..., :, -1] *= d[..., None]
    return u @ vt


def smooth_rotations(rotvecs, fps, floor, force=False):
    shape = np.asarray(rotvecs).shape
    v = np.asarray(rotvecs, float).reshape(len(rotvecs), -1, 3)
    out = v.copy()
    channels = []
    for j in range(v.shape[1]):
        rot = Rotation.from_rotvec(v[:, j])
        inc = (rot[:-1].inv() * rot[1:]).as_rotvec()
        second = np.diff(inc, n=2, axis=0)
        sigma = float(np.linalg.norm(1.4826 * np.median(
            np.abs(second - np.median(second, axis=0)), axis=0) / np.sqrt(20))) if len(second) else 0.0
        correlation = incoherence(second)
        apply = bool(len(v) >= 7 and (force or (sigma > floor and correlation < NOISE_CORRELATION)))
        record = dict(joint=j, sigma_rad=sigma, floor_rad=float(floor),
                      difference_lag1_correlation=correlation, applied=apply)
        if apply:
            smoothed, fit = robust_smooth(rot.as_matrix().reshape(len(v), 9), fps)
            out[:, j] = Rotation.from_matrix(_project_so3(smoothed.reshape(-1, 3, 3))).as_rotvec()
            record.update(fit)
        channels.append(record)
    return out.reshape(shape), channels


def hold_groups(root, body, sample_ids=None):
    """Sample index per frame when rows repeat a lower-rate estimate, else None.

    Video pipelines often publish an inference rate below the video rate by
    repeating each estimate (zero-order hold). The staircase is not motion:
    each repeat is the same observation, best placed at its run centre.
    """
    frames = len(root)
    if sample_ids is not None and len(sample_ids) == frames:
        ids = np.asarray(sample_ids)
        change, source = np.r_[True, ids[1:] != ids[:-1]], 'sample_ids'
    else:
        flat_root = np.asarray(root).reshape(frames, -1)
        flat_body = np.asarray(body).reshape(frames, -1)
        same = np.all(flat_root[1:] == flat_root[:-1], axis=1) & np.all(flat_body[1:] == flat_body[:-1], axis=1)
        change, source = np.r_[True, ~same], 'exact_repeats'
        # Static poses also repeat exactly; a hold repeats in short regular runs.
        lengths = np.diff(np.r_[np.flatnonzero(change), frames])
        short = (lengths >= 2) & (lengths <= 5)
        if lengths[short].sum() < HOLD_SHARE * frames or lengths[lengths > 5].sum() > 0.1 * frames:
            return None
    group = np.cumsum(change) - 1
    if group[-1] + 1 > (1 - HOLD_SHARE) * frames or group[-1] + 1 < 7:
        return None
    return group, source


def _resample(group, x, r, b, frames):
    """C2 spline of filtered samples back to every published frame."""
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import RotationSpline
    count = np.bincount(group)
    centre = np.bincount(group, weights=np.arange(frames)) / count
    # Frames before the first or after the last run centre lie within half a
    # run: the C2 spline extrapolates them instead of a constant (kinked) end.
    t = np.arange(frames, dtype=float)
    trans = CubicSpline(centre, x, axis=0, bc_type='natural')(t)
    def rotations(values):
        v = np.asarray(values).reshape(len(centre), -1, 3)
        return np.stack([RotationSpline(centre, Rotation.from_rotvec(v[:, j]))(t).as_rotvec()
                         for j in range(v.shape[1])], axis=1)
    return trans, rotations(r).reshape(frames, -1), rotations(b).reshape(frames, -1), centre


def _filter_channels(trans, root, body, fps, report, force=False):
    sigma = float(np.linalg.norm(noise_sigma(trans)))
    x, jumps = repair_steps(trans, fps)
    # A channel needs a persistent noise floor; clean rapid motions do not
    # qualify just because their peak acceleration is large.
    correlation = incoherence(np.diff(trans, n=3, axis=0))
    apply = force or (sigma > TRANSLATION_FLOOR_M and correlation < NOISE_CORRELATION)
    fit = {}
    if apply:
        x, fit = robust_smooth(x, fps)
    elif not jumps:
        x = trans
    report['jumps'] = jumps
    report['channels']['translation'] = dict(sigma_m=sigma, floor_m=TRANSLATION_FLOOR_M,
        difference_lag1_correlation=correlation, applied=apply or bool(jumps), **fit,
        raw_acceleration_rms_m_s2=float(np.sqrt(np.mean(np.sum(np.diff(trans,n=2,axis=0)**2,axis=1)))*fps**2))
    # Select root and every body joint independently, keeping clean channels exact.
    r, rc = smooth_rotations(root, fps, ROOT_FLOOR_RAD, force)
    b, bc = smooth_rotations(body, fps, BODY_FLOOR_RAD, force)
    report['channels'].update(root_orientation=rc, body_rotations=bc)
    if not any(c['applied'] for c in rc):
        r = root
    if not any(c['applied'] for c in bc):
        b = body
    lower = any(bc[j]['applied'] for j in [0,1,3,4,6,7,9,10] if j < len(bc))
    return x, r, b, apply or bool(jumps) or any(c['applied'] for c in rc), apply or bool(jumps) or any(
        c['applied'] for c in rc+bc), lower


def denoise_parameters(trans, root, body, fps, mode='off', sample_ids=None):
    """Return original objects in off mode (the default numerical path).

    ``sample_ids`` optionally maps every row to the estimate it repeats.
    """
    if mode not in {'off', 'auto'}:
        raise ValueError("source_denoise must be 'off' or 'auto'")
    if mode == 'off':
        return trans, root, body, None
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Source denoise requires a finite positive source framerate')
    report = dict(mode='auto', enabled=True, applied=False, method='gcv_whittaker_huber_chordal_SO3',
                  fps=float(fps), source_frames=len(trans), jumps=[], contact_merges=[], foot_lock=[], channels={})
    if len(trans) < 7 or not all(np.isfinite(x).all() for x in (trans, root, body)):
        report['reason'] = 'too_short_or_nonfinite'
        return trans, root, body, report
    holds = hold_groups(root, body, sample_ids)
    if holds is None:
        # Rows mapped to estimator samples are a declared video estimate: every
        # channel goes to the robust smoother and GCV decides how much (possibly
        # almost nothing); isolated per-sample switches are rejected as outliers.
        video = sample_ids is not None and len(sample_ids) == len(trans)
        x, r, b, moved, applied, lower = _filter_channels(trans, root, body, fps, report, force=video)
        if video:
            report['video_estimate'] = dict(source='sample_ids', samples=int(len(trans)))
    else:
        group, source = holds
        first = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
        count = np.bincount(group)
        mean_trans = np.add.reduceat(np.asarray(trans, float), first, axis=0) / count[:, None]
        centre = np.bincount(group, weights=np.arange(len(group))) / count
        rate = fps * (len(centre) - 1) / max(centre[-1] - centre[0], 1e-9)
        # A proven repeated video estimate: every channel is an estimate and GCV
        # decides how much (possibly nothing) to smooth it.
        x, r, b, _, _, _ = _filter_channels(mean_trans, np.asarray(root)[first], np.asarray(body)[first], rate, report, force=True)
        for jump in report['jumps']:
            jump.update(sample=jump['frame'], frame=int(first[jump['frame']]))
        x, r, b, _ = _resample(group, x, r, b, len(trans))
        r, b = r.reshape(np.shape(root)), b.reshape(np.shape(body))
        report['zero_order_hold'] = dict(source=source, samples=int(len(centre)), sample_fps=float(rate),
                                         repeated_frames=int(len(trans) - len(centre)))
        moved = applied = lower = True
    report['support_requested'] = moved or lower
    report['applied'] = applied
    report['reason'] = ('zero_order_hold_resampled' if holds is not None else
                        'estimated_video_noise' if applied else 'noise_below_threshold')
    return x, r, b, report


def debounce_contacts(mask, fps, normal_velocity=None, eligible=None, seconds=0.1):
    """Merge bounded short gaps and reject blips; preserve resolved impacts.

    An impact needs approach/rebound faster than 0.15 m/s, with opposite
    signs on the two sides. Noise alone cannot justify it. Geometry eligibility
    prevents joining across a genuine departure or unknown correspondence.
    """
    out = np.asarray(mask, bool).copy()
    original = out.copy()
    limit = max(1, int(np.ceil(seconds*fps)))
    eligible = np.ones(len(out), bool) if eligible is None else np.asarray(eligible, bool)
    events = []
    velocity = None if normal_velocity is None else np.asarray(normal_velocity)
    for start, stop in runs(~original):
        if start == 0 or stop == len(out) or stop-start >= limit or not eligible[start:stop].all():
            continue
        out[start:stop] = True
        events.append(dict(start=int(start), stop=int(stop), action='merge_gap'))
    for start, stop in runs(out):
        if stop-start >= limit:
            continue
        before, after = max(0, start-limit), min(len(out), stop+limit)
        impact = velocity is not None and np.any(velocity[before:start] < -0.15) and np.any(velocity[stop:after] > 0.15)
        if impact:
            events.append(dict(start=int(start), stop=int(stop), action='preserve_impact'))
        else:
            out[start:stop] = False
            events.append(dict(start=int(start), stop=int(stop), action='remove_blip'))
    return out, events
