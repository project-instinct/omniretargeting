"""Exact pose-and-witness cache for checked collision Jacobians."""

from collections import OrderedDict

import mujoco
import numpy as np


def _content(value):
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if isinstance(value, dict):
        return tuple((k, _content(v)) for k, v in sorted(value.items()) if k != "jacobian")
    if isinstance(value, (tuple, list)):
        return tuple(_content(v) for v in value)
    return value


def _signature(geometry, key, witness, fast):
    if key[1] != 'self':
        # These are every witness field read by the MuJoCo Jacobian below.
        return key, ('point', witness['geom_id'], _content(witness['point']),
                     _content(witness['direction']))
    from ..self_collision import SelfCollisionQuery
    if getattr(geometry.self_query.jacobian, '__func__', None) is SelfCollisionQuery.jacobian:
        return key, ('self', _content(witness['body_ids']), _content(witness['points']),
                     _content(witness['direction']))
    return key, _content(witness)


def _frame_cache(geometry, frame, pose):
    cache = getattr(geometry, "_repair_jacobian_cache", None)
    if cache is None:
        cache = geometry._repair_jacobian_cache = {}
    poses = cache.setdefault(frame, OrderedDict())
    pose = np.asarray(pose)
    key = (geometry.reference_observation_id, geometry.di.tobytes(), pose.dtype.str, pose.tobytes())
    if key not in poses:
        poses[key] = {}
    poses.move_to_end(key)
    while len(poses) > 3:
        poses.popitem(last=False)
    return poses[key]


def remember(geometry, poses, parts):
    for part in parts:
        for frame, rows, _, _ in part["frames"]:
            if frame == part["rejection"]:
                continue
            cache = None
            for key, row in rows:
                jac = row.get("jacobian")
                if jac is None or np.shape(jac) != (geometry.width,):
                    continue
                if cache is None:
                    cache = _frame_cache(geometry, frame, poses[frame])
                cache[_signature(geometry, key, row, True)] = np.array(jac, copy=True)


def linearize(geometry, poses, measurement):
    rows, frame, ready = [], None, False
    reused = computed = 0
    for key, witness in measurement["rows"]:
        if key[0] != frame:
            frame, ready = key[0], False
            cache = _frame_cache(geometry, frame, poses[frame])
        signature = _signature(geometry, key, witness, True)
        jacobian = cache.get(signature)
        if jacobian is None:
            if not ready:
                geometry.data.qpos[:] = poses[frame]
                mujoco.mj_forward(geometry.model, geometry.data)
                ready = True
            if key[1] == "self":
                jacobian = geometry.self_query.jacobian(witness, geometry.data)[geometry.di]
            else:
                jac = np.zeros((3, geometry.model.nv))
                mujoco.mj_jac(geometry.model, geometry.data, jac, None, witness["point"],
                              int(geometry.model.geom_bodyid[witness["geom_id"]]))
                jacobian = np.asarray(witness["direction"]) @ jac[:, geometry.di]
            cache[signature] = jacobian
            computed += 1
        else:
            reused += 1
        rows.append((key, {**witness, "jacobian": jacobian.copy()}))
    geometry.linearization_stats = dict(reused=reused, computed=computed)
    totals = getattr(geometry, "linearization_totals", None)
    if totals is None:
        totals = geometry.linearization_totals = dict(reused=0, computed=0)
    totals["reused"] += reused
    totals["computed"] += computed
    return {**measurement, "rows": rows}
