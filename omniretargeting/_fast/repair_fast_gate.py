"""Default-off reduced-QP predictor ahead of expensive trajectory repair.

This is a predictive gate, not the sound upper-bound certificate in triage.
Shadow replays run the full attempt to audit every proposed rejection.
Unknown/custom/failed local models always retain the full attempt.
"""
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .runtime_options import OPTIONS
from .repair_gate_frames import local_limits
from .repair_frames import restrict
from .repair_step import solve_step
from .repair_prediction import linear_prediction

RECORD_PREDICTIONS = False
DEEPEST_KEYS = 3
WITNESSES_PER_KEY = 3


def predict(limits, poses, geometry, measurement, protected, depth):
    """Retain the full attempt if a local model is unsupported or fails."""
    try:
        return _predict(limits, poses, geometry, measurement, protected, depth)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, ArithmeticError):
        return None


def _predict(limits, poses, geometry, measurement, protected, depth):
    from .mesh_repair import _joint_projection, COLLISION_SIGMA, LIMB_COLLISION_SIGMA, LOCAL_ANGLE
    from .limb_scope import scope_for_body
    started = time.monotonic()
    if OPTIONS.abl_repair_ratchet or OPTIONS.abl_repair_restore_layout or OPTIONS.abl_contact_regression:
        # Reduced frame views do not carry the experimental global history
        # or directional evidence. Retain the full proposal rather than gate it.
        return None
    local_angle = getattr(limits, 'local_angle', LOCAL_ANGLE)
    if (measurement.get('failures') or not measurement.get('complete', True)
            or OPTIONS.repair_aggregate):
        return None
    projection = restrict(_joint_projection(limits, geometry.model, measurement), limits.frame_mask)
    if (np.any(np.diff(projection.indptr) != 1) or np.any(projection.data != 1.)
            or len(np.unique(projection.indices)) != projection.shape[1]):
        return None
    selected = np.zeros(limits.frames*limits.width, bool)
    selected[projection.indices] = True
    selected = selected.reshape(limits.frames, limits.width)
    model, body = geometry.model, limits.movable_body
    def relevant(key):
        geoms = key[2:] if key[1] == 'self' else key[-1:]
        return body is None or any(model.body_weldid[model.geom_bodyid[g]] == body for g in geoms)
    keys = [k for k,d in measurement['depths'].items() if d > 0 and relevant(k)]
    keys.sort(key=lambda k: -measurement['depths'][k])
    frames = list(dict.fromkeys(key[0] for key in keys[:DEEPEST_KEYS]))
    if not frames or any(not selected[f].any() for f in frames):
        return None
    values = poses[:, limits.coords.qi]
    low = np.where(selected, np.minimum(limits.lower-values-1e-8,0.),0.)
    high = np.where(selected, np.maximum(limits.upper-values+1e-8,0.),0.)
    from .repair_triage_temporal import tighten
    tightened = tighten(limits, values, low, high)
    if tightened is None:
        return None
    low, high = tightened
    predictions = []
    from .repair_proposals import GeometryRows
    sigma = LIMB_COLLISION_SIGMA if scope_for_body(limits, body) is not None else COLLISION_SIGMA
    for frame in frames:
        local = local_limits(limits, frame)
        if local is None:
            return None
        local.lower, local.upper = values[frame:frame+1]+low[frame], values[frame:frame+1]+high[frame]
        groups = {}
        for key, row in measurement['rows']:
            if key[0] == frame:
                groups.setdefault(key, []).append(row)
        rows = []
        for key, witnesses in groups.items():
            order = sorted(witnesses, key=lambda row: row['clearance'])
            rows.extend(((0,*key[1:]), row) for row in order[:WITNESSES_PER_KEY])
        measured = dict(rows=rows, depths={(0,*k[1:]):v for k,v in measurement['depths'].items() if k[0]==frame})
        baseline = dict(depths={(0,*k[1:]):v for k,v in protected['depths'].items() if k[0]==frame})
        facade = GeometryRows(SimpleNamespace(model=model, width=limits.width,
                                             objective=SimpleNamespace(frames=1)))
        selector = sparse.eye(limits.width,format='csc')[:, np.flatnonzero(selected[frame])]
        step, status = solve_step(local, poses[frame:frame+1], facade, measured, baseline, selector,
                                  collision_sigma=sigma, local_angle=local_angle)
        if (step is None or not np.isfinite(step).all()
                or status not in ('Solved','AlmostSolved','QOCO_SOLVED')):
            return None
        prediction = linear_prediction(facade, measured, step, body, depth)
        if prediction is None:
            return None
        predictions.append(dict(frame=frame,status=status,coordinates=int(selector.shape[1]),
                                witnesses=len(rows), **prediction))
    predicted = max(p['predicted_depth_m'] for p in predictions)
    windows = None
    if not should_skip({'predicted_decrease_m': float(depth)-predicted}):
        from .repair_gate_windows import predict as window_prediction
        windows = window_prediction(limits, poses, geometry, measurement, protected, depth,
                                    selected, low, high, frames, sigma, local_angle, WITNESSES_PER_KEY)
        if windows is not None:
            predicted = max(p['predicted_depth_m'] for p in windows)
    return dict(model='reduced_frame_qp', entry_depth_m=float(depth),
                predicted_depth_m=predicted, predicted_decrease_m=float(depth)-predicted,
                frames=predictions, windows=windows, seconds=time.monotonic()-started,
                scope='deepest keys; small local repair/contact/layout QPs and temporal windows; no restoration')


def should_skip(prediction):
    from .mesh_repair import MIN_GROUP_PROGRESS_M
    from . import repair_prediction
    return bool(repair_prediction.PREDICTIVE_GATING and MIN_GROUP_PROGRESS_M > 0
                and prediction is not None and np.isfinite(prediction['predicted_decrease_m'])
                and prediction['predicted_decrease_m'] < MIN_GROUP_PROGRESS_M)
