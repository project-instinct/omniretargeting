"""Per-collider audit summaries without repeating geometry queries."""
import json
from pathlib import Path

import numpy as np


def depth_statistics(depth, labels, threshold):
    from scripts.review_spline_quality import intervals
    depth, labels = np.asarray(depth), np.asarray(labels)
    return dict(mean_m=float(depth.mean()), p95_m=float(np.quantile(depth, .95)),
        max_m=float(depth.max(initial=0)), worst_frame=int(labels[np.argmax(depth)]),
        affected_frames=labels[depth > threshold].tolist(),
        persistent_intervals=intervals(depth > threshold, int(labels[0])),
        threshold_m=float(threshold))


def self_details(pair_depths, model, labels, threshold):
    pairs, bodies = {}, {}
    for pair, depth in pair_depths.items():
        names = [model.body(int(model.geom_bodyid[g])).name for g in pair]
        label = ' | '.join(f'{name} / geom {g}' for name, g in zip(names, pair))
        pairs[label] = dict(depth_statistics(depth, labels, threshold), frame_depths_m=np.asarray(depth).tolist())
        for name in names:
            bodies[name] = np.maximum(bodies.get(name, np.zeros(len(labels))), depth)
    return dict(per_pair=pairs, per_body={name: dict(depth_statistics(value, labels, threshold),
        frame_depths_m=value.tolist())
        for name, value in bodies.items()}, body_aggregation='max over participating collider pairs')


def compare_self(path, baseline):
    """Compare existing native self queries without repeating them."""
    path, baseline = Path(path), Path(baseline)
    if not path.is_file() or not baseline.is_file():
        return dict(status='unverified', reason='self audit absent', collision_free_certificate=False)
    current, original = (json.loads(value.read_text()) for value in (path, baseline))
    fields = ('frame_start', 'input_frames', 'motion_payload_framerate_hz')
    if any(current['inputs'][field] != original['inputs'][field] for field in fields):
        raise ValueError('self comparison requires identical frame/time ranges')
    digest = current.get('model', {}).get('collision_model_sha256')
    if not digest or digest != original.get('model', {}).get('collision_model_sha256'):
        return dict(status='unverified', reason='self collision models not verified identical')
    final, before = current['self_collision'], original['self_collision']
    if any(final.get(field) != before.get(field) for field in ('config', 'geom_ids', 'pair_count')):
        return dict(status='unverified', reason='native self query filters/configuration differ')
    if any(value.get('unknown_query_count', 0) or value.get('errors') for value in (final, before)):
        return dict(status='unknown', reason='native self query errors',
                    final=final, A=before, collision_free_certificate=False)
    from scripts.review_spline_quality import intervals
    count, start = current['inputs']['input_frames'], current['inputs']['frame_start']
    result = dict(status='measured', collision_free_certificate=False,
                  absent_key_semantics='no positive depth reported by complete native queries; not a safety certificate')
    for category in ('per_body', 'per_pair'):
        rows = {}
        for name in sorted(set(final.get(category, {})) | set(before.get(category, {}))):
            a_row, b_row = before.get(category, {}).get(name), final.get(category, {}).get(name)
            threshold = (b_row or a_row)['threshold_m']
            a = np.zeros(count) if a_row is None else np.asarray(a_row.get('frame_depths_m', []))
            b = np.zeros(count) if b_row is None else np.asarray(b_row.get('frame_depths_m', []))
            if a.shape != (count,) or b.shape != (count,):
                return dict(status='unverified', reason='native per-part self depth arrays absent')
            row = depth_statistics(b, np.arange(count) + start, threshold)
            row['A'] = depth_statistics(a, np.arange(count) + start, threshold)
            row['delta'] = {metric: row[metric] - row['A'][metric]
                            for metric in ('mean_m', 'p95_m', 'max_m')}
            row['worse_than_A_intervals'] = intervals(b > a, start)
            rows[name] = row
        result[category] = rows
    return result


def scene_details(path, baseline, model):
    """Consume the already completed independent final scene audit."""
    path = Path(path)
    if not path.exists() or not path.with_suffix('.npz').exists():
        return dict(status='unverified', reason='scene audit or depth arrays absent')
    record = json.loads(path.read_text())
    start, count = int(record['start']), int(record['frames'])
    labels = np.arange(count) + start
    original = json.loads(Path(baseline).read_text()) if baseline and Path(baseline).exists() else None
    before = np.load(Path(baseline).with_suffix('.npz')) if original is not None else None
    if original is not None and (original['start'] != start or original['frames'] != count):
        raise ValueError('scene comparison requires identical frame ranges')
    from omniretargeting.scene_geometry import collision_geom_ids
    from scripts.review_spline_quality import intervals
    ids = collision_geom_ids(model)
    result = dict(status='measured', source=str(path), collision_free_certificate=False,
                  body_aggregation='max over the same body collision geoms', categories={})
    with np.load(path.with_suffix('.npz')) as saved:
        for kind in ('static', 'moving'):
            key = f'{kind}_depths'
            if key not in saved:
                result['categories'][kind] = dict(status='not_present')
                continue
            depths = saved[key]
            if depths.shape != (count, model.ngeom):
                raise ValueError('geometry depths do not match model/frame dimensions')
            parts = {}
            for body in np.unique(model.geom_bodyid[ids]):
                subset = ids[model.geom_bodyid[ids] == body]
                peak = depths[:, subset].max(axis=1)
                # Same positive-depth reporting threshold as review_scene_metrics.
                row = depth_statistics(peak, labels, 1e-6)
                if before is not None and key in before:
                    a = before[key][:, subset].max(axis=1)
                    row['A'] = depth_statistics(a, labels, 1e-6)
                    row['delta'] = {name: row[name] - row['A'][name]
                                    for name in ('mean_m', 'p95_m', 'max_m')}
                    row['worse_than_A_intervals'] = intervals(peak > a, start)
                parts[model.body(int(body)).name] = row
            result['categories'][kind] = dict(per_body=parts,
                unknown_evidence=record.get(kind, {}).get('uncertain_samples'),
                query_failures=record.get(kind, {}).get('failures', []),
                method=record.get(kind, {}).get('method'))
    if before is not None:
        before.close()
    return result
