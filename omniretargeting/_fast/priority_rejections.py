"""Read-only rejection evidence using the original nonlinear guard thresholds."""

import numpy as np


def _worst(guard, family, actual, budget, tolerance, first, order=0, **extra):
    actual, budget = np.broadcast_arrays(np.asarray(actual), np.asarray(budget))
    excess = actual - budget - tolerance
    bad = ~np.isfinite(actual) | (excess > 0)
    if not bad.any():
        return None
    index = np.unravel_index(np.argmax(np.where(~np.isfinite(actual), np.inf, excess)), actual.shape)
    frame = int(first + index[0])
    value, bound, error = actual[index], budget[index], excess[index]
    return dict(guard=guard, family=str(family), frame=frame,
                stencil=list(range(frame, frame + order + 1)), coordinate=list(map(int, index[1:])),
                actual=float(value) if np.isfinite(value) else None,
                budget=float(bound) if np.isfinite(bound) else None, tolerance=float(tolerance),
                excess=float(error) if np.isfinite(error) else None,
                violated_scalar_count=int(bad.sum()), **extra)


def hard_details(coordinator, poses, frames, failures):
    from .priority_window import hard_guards
    from .window_guards import _kind, _correction, _sequence, _shape_kin
    from .continuity import ROUND_OFF
    from .._sequence.continuity_guard import ACCEPTANCE_SLACK, _change_size
    obj, result = coordinator.objective, []
    part, coords = slice(frames.start, frames.stop), coordinator.objective.coords

    def add(*args, **kwargs):
        row = _worst(*args, **kwargs)
        if row is not None:
            result.append(row)
        return row

    if 'joint_box' in failures:
        q = poses[part][:, coords.qi]
        low, high = np.asarray(obj.lower), np.asarray(obj.upper)
        low = low[part] if low.ndim == 2 else low
        high = high[part] if high.ndim == 2 else high
        for side, actual, bound in [('lower', -q, -low), ('upper', q, high)]:
            row = add('joint_box', side, actual, bound, 1e-9, frames.start)
            if row is not None:
                row['qpos_column'] = int(coords.qi[row['coordinate'][0]])
    q = getattr(coords, 'root_q', None)
    if q is not None:
        if 'fixed_root' in failures:
            add('fixed_root', 'qpos', np.abs(poses[part, q:q + 7] - obj.fixed_root[part]), 0., 0., frames.start)
        if 'objective_root_ball' in failures:
            add('objective_root_ball', 'xyz_norm',
                np.linalg.norm(poses[part, q:q + 3] - obj.reference[part, q:q + 3], axis=1),
                obj.root_budget, ROUND_OFF, frames.start)
        shape = getattr(obj, 'source_shape', None)
        if shape is not None and hasattr(shape, 'xy'):
            refs = [shape.seed] + ([] if shape.acquired_poses is None else [shape.acquired_poses])
            for i, reference in enumerate(refs):
                name = 'reference_root_ball:' + str(i)
                if name in failures:
                    add(name, 'xyz_norm', np.linalg.norm(poses[part, q:q + 3] - reference[part, q:q + 3], axis=1),
                        shape.xy, ROUND_OFF, frames.start)
    for guard in hard_guards(coordinator):
        name = type(guard).__name__
        if name not in failures:
            continue
        try:
            kind = _kind(guard)
        except TypeError:
            result.append(dict(guard=name, family='unknown_guard', frames=list(frames),
                               excess=None, diagnostic_unavailable=True))
            continue
        if kind == 'shape':
            kin = _shape_kin(guard, poses, frames, False)
            for i, envelope in enumerate(guard.envelopes):
                envelope = envelope.for_frames(part)
                _, _, (change, _) = envelope.measure(kin)
                value, _ = envelope.layout(change)
                if value is None:
                    continue
                row = add(name, 'layout', np.where(envelope.layout_sign != 0, -value, 0.),
                          0., envelope.SLACK, frames.start, envelope=i)
                if row is not None:
                    pair, axis = row['coordinate']
                    landmarks = envelope.graph.pairs[pair]
                    row.update(bodies=envelope.bodies[landmarks].tolist(),
                               landmarks=[envelope.names[int(k)] for k in landmarks], axis=axis)
            continue
        values = (_correction(guard, poses, frames, False) if kind == 'correction'
                  else _sequence(guard, poses, frames, False))
        for key, (value, _, lo, hi, order) in values.items():
            size = np.abs(value) if kind == 'correction' else _change_size(key[0], value)
            row = add(name, key, size, guard.bounds[key][lo:hi],
                      ROUND_OFF if kind == 'correction' else ACCEPTANCE_SLACK, lo, order)
            joint = (str(key).startswith('joints_') if kind == 'correction' else key[0] == 'joints')
            if row is not None and joint and row['coordinate']:
                axis = row['coordinate'][0]
                guard_coords = guard.coords if kind == 'correction' else guard.tasks.coords
                row['qpos_column'] = int(guard_coords.qi[axis])
                if hasattr(guard_coords, 'di') and hasattr(obj.model, 'dof_jntid'):
                    joint_id = int(obj.model.dof_jntid[guard_coords.di[axis]])
                    row.update(joint=joint_id, bodies=[int(obj.model.jnt_bodyid[joint_id])])
            if kind == 'correction' and key in guard.norm_bounds:
                add(name, key + ':norm', np.linalg.norm(value, axis=1), guard.norm_bounds[key][lo:hi],
                    ROUND_OFF, lo, order)
    return result


def collision_details(coordinator, regressions, body_bad, depth, protection, tolerance):
    result = []
    for key, actual, best in regressions:
        model = coordinator.objective.model
        geoms = key[2:] if key[1] == 'self' else key[-1:]
        bodies = sorted({int(model.body_weldid[model.geom_bodyid[g]]) for g in geoms})
        result.append(dict(guard='original_geometry', family='key_nonregression', key=key,
                           frame=int(key[0]), bodies=bodies, actual_m=actual, budget_m=best,
                           tolerance_m=tolerance, excess_m=actual - best - tolerance))
    for frame, body in body_bad:
        actual, best = float(depth[frame, body]), float(protection[frame, body])
        result.append(dict(guard='original_geometry', family='body_nonregression', frame=frame,
                           bodies=[body], actual_m=actual, budget_m=best, tolerance_m=tolerance,
                           excess_m=actual - best - tolerance))
    return result
