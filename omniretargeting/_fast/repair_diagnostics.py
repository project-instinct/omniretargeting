"""Residual witnesses and the actual rejected immutable repair constraints."""

import numpy as np


def deep_residuals(model, measurement, threshold=.01):
    """Every over-threshold collision key, with the full reliable exit vector."""
    deepest = {}
    for key,row in measurement['rows']:
        if -row['clearance'] <= threshold:
            continue
        if key not in deepest or row['clearance'] < deepest[key]['clearance']:
            deepest[key]=row
    result=[]
    for key,row in sorted(deepest.items()):
        geoms=key[2:] if key[1]=='self' else key[-1:]
        vector=-row['clearance']*np.asarray(row['direction'])
        result.append(dict(key=list(key),frame=int(key[0]),
                           bodies=[model.body(int(model.geom_bodyid[g])).name for g in geoms],
                           depth_m=float(-row['clearance']),exit_xy_m=float(np.linalg.norm(vector[:2])),
                           exit_z_m=float(vector[2])))
    return result


def guard_failure(guard, poses):
    result={'reason':'immutable acquired contact/continuity','guard':type(guard).__name__}
    if hasattr(guard,'diagnostics'):
        detail=guard.diagnostics(poses)
        if 'groups' in detail:
            detail=dict(detail,groups=[g for g in detail['groups'] if g.get('violating_frames')])
        result['constraint']=detail
    return result
