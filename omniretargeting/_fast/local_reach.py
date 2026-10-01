"""Conservative displacement certificates for the fixed-root repair box."""

import mujoco
import numpy as np


def geom_reach_bound(model, geom, poses, angle, slide):
    """Upper bound for every collision point under the local scalar box.

    Triangle inequalities bound the distance to every ancestor pivot for all
    intermediate poses. Summing hinge chords and slider travel therefore
    bounds the complete rigid displacement, not only its first derivative.
    Floating-base DOFs contribute zero because local repair fixes the root.
    """
    radius = float(model.geom_rbound[geom]) + float(np.linalg.norm(model.geom_pos[geom]))
    body = int(model.geom_bodyid[geom])
    movement = 0.
    while body:
        joints = range(int(model.body_jntadr[body]),
                       int(model.body_jntadr[body] + model.body_jntnum[body]))
        for joint in joints:
            kind = model.jnt_type[joint]
            if kind == mujoco.mjtJoint.mjJNT_HINGE:
                offset = float(np.linalg.norm(model.jnt_pos[joint]))
                movement += 2 * (radius + offset) * np.sin(angle / 2)
                radius += 2 * offset
            elif kind == mujoco.mjtJoint.mjJNT_SLIDE:
                movement += slide
                address = int(model.jnt_qposadr[joint])
                radius += float(np.max(np.abs(poses[:, address]), initial=0.)) + slide
        radius += float(np.linalg.norm(model.body_pos[body]))
        body = int(model.body_parentid[body])
    return float(movement)


def unreachable_geoms(model, poses, measurement, *, angle, slide, residual=.01, closed_static=False):
    """Colliders with a proved residual exceeding residual inside the local box."""
    maxima, frames = {}, {}
    for key, depth in measurement['depths'].items():
        if key[1] not in ('static', 'moving') or depth <= residual:
            continue
        if key[1] == 'static' and not closed_static:
            continue
        if key[1] == 'moving' and key[3] not in ('closed_surface', 'reverse_surface'):
            continue
        geom = int(key[-1])
        maxima[geom] = max(maxima.get(geom, 0.), depth)
        frames.setdefault(geom, []).append(int(key[0]))
    result = []
    for geom, depth in sorted(maxima.items()):
        reach = geom_reach_bound(model, geom, poses, angle, slide)
        if depth - reach > residual:
            result.append(dict(geom_id=geom, body=model.body(int(model.geom_bodyid[geom])).name,
                               depth_m=float(depth), displacement_upper_bound_m=reach,
                               residual_lower_bound_m=float(depth - reach), frames=sorted(set(frames[geom]))))
    return result
