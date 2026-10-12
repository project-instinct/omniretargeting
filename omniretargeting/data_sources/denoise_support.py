"""Conservative material-foot stabilization for opt-in SMPL-X estimates."""
import numpy as np
from scipy.signal import savgol_filter

from .source_denoise import debounce_contacts, runs


def support_translation(sole_points, fps, report):
    """Bound root correction to 20 mm and only use slow, low sole material.

    This is a video-estimate heuristic, not a physical support certificate.
    Foot rolls are represented by one fixed material point per interval.
    Source ground/objects are never moved. Boundary corrections ramp smoothly.
    """
    first = next(iter(sole_points.values()))
    frames = len(first)
    corrections = np.zeros((frames, 3))
    weights = np.zeros(frames)
    floor = min(float(np.quantile(p[..., 2], 0.02)) for p in sole_points.values())
    for side, points in sole_points.items():
        velocity = np.gradient(points, 1/fps, axis=0)
        speed = np.linalg.norm(velocity, axis=-1)
        low = points[..., 2] < floor + 0.035
        slow = (speed < 0.16) & low
        mask = slow.any(1)
        mask, merges = debounce_contacts(mask, fps, eligible=low.any(1))
        report['contact_merges'].extend(dict(foot=side, stage='support_estimate', fps=float(fps), frame_space='source', **e) for e in merges)
        for start, stop in runs(mask):
            if (stop-start)/fps < 0.15:
                continue
            score = np.mean(speed[start:stop], axis=0)
            score[~low[start:stop].all(0)] = np.inf
            vertex = int(np.argmin(score))
            if not np.isfinite(score[vertex]):
                continue
            p = points[start:stop, vertex]
            # Only lock a nearly stationary material patch, never a sliding foot.
            anchor = np.median(p, axis=0)
            delta = anchor-p
            if np.max(np.linalg.norm(delta, axis=1)) > 0.025:
                continue
            ramp = min(max(2, int(round(0.08*fps))), (stop-start)//2)
            envelope = np.ones(stop-start)
            edge = np.sin(np.linspace(0, np.pi/2, ramp))**2
            envelope[:ramp] = edge
            envelope[-ramp:] = edge[::-1]
            corrections[start:stop] += envelope[:, None]*delta
            weights[start:stop] += envelope
            report['foot_lock'].append(dict(foot=side, start=int(start), stop=int(stop),
                material_vertex=vertex, anchor_m=anchor.tolist(),
                correction_max_m=float(np.max(np.linalg.norm(delta,axis=1)))))
    corrections /= np.maximum(weights, 1)[:, None]
    if frames >= 7:
        size = max(5, int(round(0.10*fps)) | 1)
        size = min(size, frames if frames%2 else frames-1)
        corrections = savgol_filter(corrections, size, 3, axis=0)
    norm = np.linalg.norm(corrections, axis=1)
    corrections *= np.minimum(1, 0.020 / np.maximum(norm, 1e-12))[:, None]
    report['support_correction_max_m'] = float(np.linalg.norm(corrections, axis=1).max())
    report['support_status'] = 'geometry_heuristic_not_physics_verified'
    return corrections
