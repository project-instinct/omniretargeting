"""Small witness-window QPs retain local scalar-joint temporal coupling."""
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .repair_gate_frames import local_limits
from .repair_step import solve_step
from .repair_prediction import linear_prediction

WINDOW_RADIUS = 10


def predict(limits, poses, geometry, measurement, protected, depth, selected, low, high,
            frames, sigma, angle, witnesses_per_key):
    from .repair_proposals import GeometryRows
    bands = []
    for frame in sorted(frames):
        start, stop = max(0, frame-WINDOW_RADIUS), min(limits.frames, frame+WINDOW_RADIUS+1)
        if bands and bands[-1][0] <= frame < bands[-1][1]:
            continue
        bands.append((start, stop))
    predictions = []
    for start, stop in bands:
        local = local_limits(limits, start, stop, temporal=True)
        if local is None:
            return None
        part = slice(start, stop)
        values = poses[part, limits.coords.qi]
        local.lower, local.upper = values+low[part], values+high[part]
        groups = {}
        for key, row in measurement['rows']:
            if start <= key[0] < stop:
                groups.setdefault(key, []).append(row)
        rows = []
        for key, witnesses in groups.items():
            rows.extend(((key[0]-start, *key[1:]), row)
                        for row in sorted(witnesses, key=lambda row: row['clearance'])[:witnesses_per_key])
        measured = dict(rows=rows, depths={(k[0]-start,*k[1:]):v
                        for k,v in measurement['depths'].items() if start <= k[0] < stop})
        baseline = dict(depths={(k[0]-start,*k[1:]):v
                        for k,v in protected['depths'].items() if start <= k[0] < stop})
        count = stop-start
        facade = GeometryRows(SimpleNamespace(model=geometry.model, width=limits.width,
                                             objective=SimpleNamespace(frames=count)))
        selector = sparse.eye(count*limits.width, format='csc')[:, np.flatnonzero(selected[part])]
        step, status = solve_step(local, poses[part], facade, measured, baseline, selector,
                                  collision_sigma=sigma, local_angle=angle)
        if (step is None or not np.isfinite(step).all()
                or status not in ('Solved','AlmostSolved','QOCO_SOLVED')):
            return None
        prediction = linear_prediction(facade, measured, step, limits.movable_body, depth)
        if prediction is None:
            return None
        predictions.append(dict(start=start,stop=stop,status=status,
                                coordinates=int(selector.shape[1]),witnesses=len(rows), **prediction))
    return predictions
