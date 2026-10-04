"""Bounded proposal enrichment from a cached rejected window trial.

New witnesses never enter the accepted ledger, contact domain, or baseline
funding predictor. Exact nonlinear admission remains the sole authority.
"""

import time

import numpy as np
from scipy import sparse

from .original_geometry_relinearize import witness_identity, pose_identity
from .priority_window import _digest, WINDOW_PENETRATION_STEP_TOLERANCE
from .priority_window_contacts import checked_frames


def _owners(model, key):
    geoms = key[2:] if key[1] == 'self' else key[-1:]
    return tuple(sorted({int(model.body_weldid[model.geom_bodyid[g]]) for g in geoms}))


def _freeze(value):
    return tuple(map(_freeze, value)) if isinstance(value, (tuple, list)) else value


def geometry_refinement(c, origin, candidate, *, frames, known_witness_ids, max_new_witnesses):
    started = time.monotonic()
    checked_frames(frames, len(c.current))
    origin, candidate = np.asarray(origin), np.asarray(candidate)
    result = dict(hard_linear=[], soft_rows=[], witness_ids=(), row_metadata=[],
                  origin_pose_id=_digest(origin), trial_pose_id=_digest(candidate),
                  epoch=c.epoch, evidence_id=None, frames=tuple(frames),
                  request_complete=False, complete=False, unresolved=True, failures=[],
                  query_scope={}, coverage='selected_witnesses_only', funding_modified=False,
                  selection=dict(requested=0, selected=0, new=0, duplicate=0, invalid=0,
                                 cap_exhausted=0, unresolved_keys=[]), trial_cache_hit=False)

    def finish(reason=None):
        if reason is not None:
            result['failures'].append(dict(reason=reason))
            result['unresolved'] = True
        result['complete'] = result['request_complete'] = not result['unresolved']
        result['seconds'] = time.monotonic() - started
        return result

    from .runtime_options import OPTIONS
    if not OPTIONS.window_geometry_generation:
        return finish('window_geometry_generation_disabled')
    if (isinstance(max_new_witnesses, bool) or not isinstance(max_new_witnesses, (int, np.integer))
            or max_new_witnesses < 0):
        return finish('invalid_witness_budget')
    state = getattr(c, '_window_state', None)
    if (state is None or state.frames != frames or state.epoch != c.epoch
            or not np.array_equal(state.origin, origin) or not np.array_equal(c.current, origin)):
        return finish('stale_or_unprepared_origin')
    result['epoch'] = state.epoch
    if (state.trial is None or _digest(state.trial[0]) != result['trial_pose_id']
            or candidate.shape != origin.shape):
        return finish('trial_cache_miss')
    _, report, measured, _, _ = state.trial
    result['trial_cache_hit'] = True
    evidence = state.measured.get('evidence_id')
    result['evidence_id'] = evidence
    if evidence is None or measured.get('evidence_id') != evidence:
        return finish('missing_or_changed_evidence')
    if (not report['reliable'] or measured.get('failures') or not measured.get('complete', True)):
        return finish('unreliable_trial_geometry')
    known = {_freeze(k) for k in known_witness_ids}
    known.update(witness_identity(key, row) for key, row in state.measured['rows'])
    keys = {tuple(row[0]) for row in report['penetration_regressions']}
    bodies = set(map(tuple, report['body_regressions']))
    if report['total_penetration_regression']:
        baseline = state.baseline.coordinator.current_depth
        depth = state.trial[3]
        bodies.update((f, int(b)) for f in frames for b in np.flatnonzero(depth[f] > baseline[f]))
    for key, _ in measured['rows']:
        if any((key[0], b) in bodies for b in _owners(c.objective.model, key)):
            keys.add(tuple(key))
    groups, seen = {}, set()
    for key, row in measured['rows']:
        key = tuple(key)
        if key not in keys:
            continue
        identity = witness_identity(key, row)
        result['selection']['requested'] += 1
        if identity in known or identity in seen:
            result['selection']['duplicate'] += 1
            continue
        seen.add(identity)
        value = float(row['clearance'])
        if not np.isfinite(value):
            result['selection']['invalid'] += 1
            result['failures'].append(dict(reason='nonfinite_trial_witness', witness_id=identity))
            continue
        score = max(0., -value - state.measured['depths'].get(key, 0.))
        groups.setdefault(key, []).append((score, key, row))
    missing_keys = keys - {tuple(key) for key, _ in measured['rows']}
    for key in sorted(missing_keys, key=repr):
        result['failures'].append(dict(reason='missing_trial_rows_for_regressed_key', key=key))
    for group in groups.values():
        group.sort(key=lambda item: (-item[0], repr(witness_identity(item[1], item[2]))))
    ordered = sorted(groups, key=lambda key: (-groups[key][0][0], repr(key)))
    # One new witness per failing key before spending remaining capacity on
    # additional material samples of the same collision key.
    pending = [groups[key][0] for key in ordered]
    pending += sorted((item for key in ordered for item in groups[key][1:]),
                      key=lambda item: (-item[0], repr(witness_identity(item[1], item[2]))))
    selected, omitted = pending[:max_new_witnesses], pending[max_new_witnesses:]
    result['selection'].update(selected=len(selected), cap_exhausted=len(omitted),
                               unresolved_keys=sorted({key for _, key, _ in omitted} | missing_keys, key=repr))
    result['unresolved'] = bool(omitted or result['failures'])
    if not selected:
        return finish()
    view = {**measured, 'rows': [(key, row) for _, key, row in selected]}
    expected_ids = tuple(witness_identity(key, row) for key, row in view['rows'])
    query = getattr(c.provider, 'relinearize_witnesses', None)
    if query is None:
        return finish('relinearize_witnesses_unavailable')
    try:
        replay = query(origin, candidate, view, frames=frames, keys={key for _, key, _ in selected})
    except (ValueError, RuntimeError) as error:
        return finish('relinearization_failed: ' + str(error))
    result['query_scope'] = dict(replay.get('query_scope', {}))
    result['provider_request_complete'] = bool(replay.get('request_complete', False))
    expected_shape = (len(selected), len(frames) * c.objective.width)
    if (c.epoch != state.epoch or not np.array_equal(c.current, origin)
            or replay.get('origin_pose_id') != pose_identity(origin, frames)
            or replay.get('trial_pose_id') != pose_identity(candidate, frames)
            or replay.get('evidence_id') != evidence
            or tuple(map(_freeze, replay.get('witness_ids', ()))) != expected_ids
            or tuple(replay.get('frames', ())) != tuple(frames)):
        return finish('stale_or_misaligned_relinearization')
    matrix = sparse.csr_matrix(replay['J'])
    phi, reliable = np.asarray(replay['phi']), np.asarray(replay['reliable'], bool)
    if (matrix.shape != expected_shape or phi.shape != (len(selected),)
            or reliable.shape != phi.shape or len(replay['rows']) != len(selected)
            or any(tuple(key) != selected[i][1] for i, (key, _) in enumerate(replay['rows']))):
        return finish('invalid_relinearization_shape')
    if hasattr(c.objective.coords, 'cols') and not np.array_equal(replay.get('columns'), c.objective.coords.cols):
        return finish('relinearization_columns_changed')
    result['provider_origin_pose_id'] = replay['origin_pose_id']
    result['failures'].extend(replay.get('failures', ()))
    result['unresolved'] |= bool(result['failures']) or not result['provider_request_complete']
    good, bounds, metadata = [], [], []
    baseline = state.baseline.coordinator.current_measurement['depths']
    for i, (key, row) in enumerate(replay['rows']):
        identity = expected_ids[i]
        if not reliable[i] or not np.isfinite(phi[i]) or not np.isfinite(matrix.getrow(i).data).all():
            result['selection']['invalid'] += 1
            result['failures'].append(dict(reason='unreliable_origin_witness', witness_id=identity))
            result['unresolved'] = True
            continue
        allowed = min(state.measured['depths'].get(key, 0.),
                      baseline.get(key, 0.) + WINDOW_PENETRATION_STEP_TOLERANCE)
        upper = float(phi[i] + allowed)
        if upper < 0.:
            reverse = key[1] == 'moving' and key[3] == 'reverse_surface'
            result['failures'].append(dict(reason='facet_incompatible_at_origin' if reverse else
                                          'origin_incompatible_with_protected_depth', witness_id=identity,
                                          clearance_m=float(phi[i]), allowed_depth_m=float(allowed), excess_m=-upper))
            result['unresolved'] = True
            continue
        good.append(i)
        bounds.append(upper)
        metadata.append(dict(key=key, witness_id=identity, frame=int(key[0]),
                             owners=_owners(c.objective.model, key), clearance_m=float(phi[i]),
                             allowed_depth_m=float(allowed), source='cached_trial_relinearized_at_origin'))
    if good:
        new_matrix, new_phi = matrix[good], phi[good]
        result['hard_linear'] = [(-new_matrix, np.full(len(good), -np.inf), np.asarray(bounds))]
        indices, soft_keys = [], []
        for i, row in enumerate(metadata):
            for owner in row['owners']:
                indices.append(i)
                soft_keys.append(('penetration', row['frame'], owner))
        result['soft_rows'] = [(-new_matrix[indices], new_phi[indices], np.ones(len(indices)), tuple(soft_keys))]
        from .priority_margins import margin_rows
        result['soft_rows'] += margin_rows(new_matrix, new_phi, [replay['rows'][i] for i in good], c.margin_preferences)
    result.update(witness_ids=tuple(expected_ids[i] for i in good), row_metadata=metadata)
    result['selection']['new'] = len(good)
    return finish()
