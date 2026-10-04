"""Positive clearance preferences from original witnesses, at tracking level.

These costs never enter penetration depths, contact allowances or donor credit.
One maximum epigraph per original collision key avoids multiplying a preference
by the number of retained witnesses. This is an L1 replacement, not equality of
its objective with the former squared grid/convex-distance residuals.
"""

import numpy as np


def preferences(objective):
    cfg = objective.cfg
    sigma = float(cfg['collision_sigma_m']) * float(objective.clearance_scale)
    result = {}
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Clearance preference sigma must be finite and positive')
    for kind, setting in (('static', 'scene_clearance_m'), ('moving', 'scene_clearance_m'),
                          ('self', 'self_clearance_m')):
        margin = float(cfg[setting])
        if not np.isfinite(margin) or margin < 0:
            raise ValueError('Clearance preferences must be finite and nonnegative')
        # Secant of (deficit/sigma)^2 across [0, margin]. At the surface,
        # weight*margin equals the old squared margin cost. Units are cost/m.
        result[kind] = (margin, margin / sigma ** 2)
    return result


def margin_rows(matrix, phi, rows, settings):
    selected, bounds, weights, keys = [], [], [], []
    for index, (key, _) in enumerate(rows):
        margin, weight = settings.get(key[1], (0., 0.))
        if margin == 0:
            continue
        selected.append(index)
        bounds.append(phi[index] - margin)
        weights.append(weight)
        keys.append(('clearance_margin', *key))
    return ([] if not selected else
            [(-matrix[selected], np.asarray(bounds), np.asarray(weights), tuple(keys))])


def margin_cost(measured, settings):
    """All CURRENT original witnesses, including keys absent from the model."""
    values, weights = {}, {}
    for key, row in measured['rows']:
        margin, weight = settings.get(key[1], (0., 0.))
        if not margin:
            continue
        value = float(row['clearance'])
        if not np.isfinite(value):
            raise ValueError('Unknown original clearance cannot evaluate a margin preference')
        identity = ('clearance_margin', *key)
        values[identity] = max(values.get(identity, 0.), max(margin - value, 0.))
        weights[identity] = weight
    return values, float(sum(weights[key] * value for key, value in values.items()))


def coverage(provider, objective, settings):
    """Validate eligibility/bands before claiming the omitted proxies' duties.

    This certifies pair/field inputs, not continuum collision coverage. Moving
    shell samples intentionally differ from the former grid approximation.
    """
    geometry = getattr(provider, 'geometry', None)
    if geometry is None or geometry.model is not objective.model:
        raise ValueError('Original geometry is required before omitting proxy collision terms')
    source = geometry.objective
    for name in ('shell_geoms', 'shell_sample_geom', 'shell_local'):
        if not np.array_equal(getattr(source, name), getattr(objective, name)):
            raise ValueError('Original static shell coverage differs from the objective')
    proxy_self = getattr(objective, 'self_collision', None)
    if proxy_self is not None and not np.array_equal(proxy_self.pairs, geometry.self_query.pairs):
        raise ValueError('Original self pair exclusions differ from the tracking objective')
    if getattr(objective, 'scene_sdf', None) is not None and geometry.scene is None:
        raise ValueError('A scene grid has no original static-scene query')
    objects = getattr(objective, 'objects', ())
    original_objects = () if geometry.moving is None else geometry.moving.objects
    if len(objects) != len(original_objects):
        raise ValueError('Moving object grid/original geometry counts differ')
    if objects and not np.array_equal(geometry.moving.geom_ids, objective.shell_geoms):
        raise ValueError('Original moving-object robot geom coverage differs')
    from .mesh_repair import WITNESS_CLEARANCE
    bands = dict(static=WITNESS_CLEARANCE, self=geometry.self_query.cfg['detection_distance'])
    if geometry.moving is not None:
        bands['moving'] = geometry.moving.cfg['activation_distance']
    for kind, band in bands.items():
        if settings[kind][0] > band:
            raise ValueError(f'{kind} clearance preference exceeds the original witness band')
    return dict(self_pair_count=len(geometry.self_query.pairs),
                static_scene=geometry.scene is not None, moving_objects=len(original_objects),
                collision_geom_count=len(objective.shell_geoms), witness_bands_m=bands,
                moving_sampling_identical_to_grid=False, continuum_collision_certificate=False)


def collision_diagnostics(coordinator, measured):
    """Only measured depths produce numeric statistics; unknowns stay unknown."""
    geometry = getattr(coordinator.provider, 'geometry', None)
    present = ({'static': geometry.scene is not None, 'moving': geometry.moving is not None,
                'self': geometry.self_query is not None} if geometry is not None else
               {kind: any(key[1] == kind for key in measured['depths']) for kind in ('static', 'moving', 'self')})
    reliable = (not measured.get('failures') and measured.get('complete', True)
                and measured.get('reliable', True)
                and np.asarray(measured.get('frame_complete', True)).all()
                and all(np.isfinite(v) and v >= 0 for v in measured['depths'].values()))
    stats = dict(clearance_diagnostics_owner='priority_clearance_original_geometry',
                 collision_diagnostics_source='original_geometry',
                 original_collision_reliable=bool(reliable),
                 original_collision_failures=list(measured.get('failures', ())),
                 original_collision_families={}, collision_free_certificate=False)
    for kind, name in (('static', 'scene_depth_m'), ('moving', 'moving_depth_m'), ('self', 'self_depth_m')):
        rows = {key: value for key, value in measured['depths'].items() if key[1] == kind}
        key = max(rows, key=rows.get, default=None)
        depth = float(rows.get(key, 0.)) if reliable and present[kind] else None
        stats[name] = depth
        detail = dict(measured=present[kind], reliable=bool(reliable and present[kind]), max_depth_m=depth)
        if key is not None:
            geoms = key[2:] if kind == 'self' else key[-1:]
            model = coordinator.objective.model
            bodies = sorted({int(model.body_weldid[model.geom_bodyid[g]]) for g in geoms})
            detail.update(key=key, frame=int(key[0]), bodies=bodies)
            witnesses = [row for k, row in measured['rows'] if k == key]
            if witnesses and depth is not None:
                row = min(witnesses, key=lambda row: row['clearance'])
                direction = row.get('direction')
                if direction is not None:
                    vector = depth * np.asarray(direction)
                    detail.update(exit_xy_m=float(np.linalg.norm(vector[:2])), exit_z_m=float(vector[2]))
        stats['original_collision_families'][kind] = detail
    return stats
