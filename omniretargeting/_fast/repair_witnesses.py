"""Remember acceptance witnesses and requery them at the current iterate.

Candidate geometry is never mistaken for a current-state constraint. Added
rows share the original collision key/epigraph, so discovering more material
samples cannot multiply that collision's weight.
"""

import copy

import mujoco
import numpy as np

from ..moving_scene import interpolate_object


def identity(key, row):
    if key[1] == 'self':
        return (key, None)
    sample = row.get('sample_id')
    return None if sample is None else (key, int(sample))


class RepairWitnesses:
    def __init__(self, *, enabled=True, max_samples=128, per_key=2):
        self.enabled = enabled
        self.max_samples, self.per_key = max(0, int(max_samples)), max(0, int(per_key))
        self.required = set()
        self.depths = {}

    def observe(self, measured, checked):
        if not self.enabled:
            return 0
        existing = {identity(k, r) for k, r in measured['rows']}
        previous = set(self.required)
        for key, row in checked['rows']:
            sample = identity(key, row)
            if row['clearance'] < 0 and sample is not None and sample not in existing:
                self.depths[sample] = max(self.depths.get(sample, 0.), -float(row['clearance']))
        # Bounded auxiliary QP memory only. OriginalGeometry's mandatory
        # samples, complete acceptance queries and protected depths stay intact.
        selected, counts = set(), {}
        for sample in sorted(self.depths, key=lambda item: (-self.depths[item], item)):
            key = sample[0]
            if len(selected) >= self.max_samples:
                break
            if counts.get(key, 0) >= self.per_key:
                continue
            selected.add(sample)
            counts[key] = counts.get(key, 0) + 1
        self.required = selected
        self.depths = {sample: self.depths[sample] for sample in selected}
        return len(selected - previous)

    def augment(self, geometry, poses, measured):
        if not self.enabled:
            return measured
        existing = {identity(k, r) for k, r in measured['rows']}
        rows = list(measured['rows'])
        # Every request in this call uses the same immutable poses. Forward
        # queries already return all material samples of one geom; retain that
        # exact array until all remembered IDs at the same frame have used it.
        forward_cache = {}
        requested = sorted(self.required - existing)
        static_rows = self._static_rows(geometry, poses, requested)
        for key, sample in requested:
            frame = key[0]
            geometry.data.qpos[:] = poses[frame]
            mujoco.mj_forward(geometry.model, geometry.data)
            row = (static_rows[key, sample] if key[1] == 'static' else
                   self._query(geometry, poses[frame], key, sample, forward_cache))
            if row is not None:
                rows.append((key, row))
        return geometry.linearize(poses, {**measured, 'rows': rows})

    @staticmethod
    def _static_rows(geometry, poses, requested):
        """Batch identical point queries without changing point arithmetic/order.

        libigl's winding binding rebuilds its acceleration structure per call.
        A remembered static point needs the same complete query, but not a
        separate rebuild for every material ID. Unknown results stay absent.
        """
        result, pending, points = {}, [], []
        limit = max(1, int(geometry.SCENE_BATCH_POINTS))

        def flush():
            signed = geometry.scene.query(np.array(points))
            for index, (key, sample) in enumerate(pending):
                row = None
                if not signed['uncertain'][index] and np.isfinite(signed['clearance'][index]):
                    row = dict(geom_id=key[-1], sample_id=sample, point=points[index],
                               direction=signed['directions'][index],
                               clearance=float(signed['clearance'][index]))
                result[key, sample] = row

        frame = None
        for key, sample in requested:
            if key[1] != 'static':
                continue
            if frame != key[0]:
                frame = key[0]
                geometry.data.qpos[:] = poses[frame]
                mujoco.mj_forward(geometry.model, geometry.data)
            geom = key[-1]
            rotation = geometry.data.geom_xmat[geom].reshape(3, 3)
            points.append(geometry.data.geom_xpos[geom] + rotation @ geometry.objective.shell_local[sample])
            pending.append((key, sample))
            if len(points) == limit:
                flush()
                pending, points = [], []
        if points:
            flush()
        return result

    @staticmethod
    def _query(geometry, pose, key, sample, forward_cache=None):
        model, data = geometry.model, geometry.data
        if key[1] == 'self':
            query = copy.copy(geometry.self_query)
            mask = np.array([tuple(pair) == key[2:] for pair in query.pairs])
            query.pairs = query.pairs[mask]
            query.pair_ids = query.pair_ids[mask]
            rows = query.query(pose, distance=np.inf)
            return next((r for r in rows if tuple(r['pair']) == key[2:]), None)
        geom = key[-1]
        center, rotation = data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3)
        if key[1] == 'static':
            point = center + rotation @ geometry.objective.shell_local[sample]
            signed = geometry.scene.query(point[None])
            if signed['uncertain'][0] or not np.isfinite(signed['clearance'][0]):
                return None
            return dict(geom_id=geom, sample_id=sample, point=point,
                        direction=signed['directions'][0], clearance=float(signed['clearance'][0]))
        oid, kind, component = key[2:5]
        owner = geometry.moving
        obj = owner.objects[oid]
        shell = next(s for s in owner.shells if s['geom_id'] == geom)
        translation, object_rotation, scale = interpolate_object(obj['surface'], key[0])
        if kind == 'reverse_surface':
            point = obj['samples'][sample] @ object_rotation.T * scale + translation
            local = (point - center) @ rotation
            values = local @ shell['planes'][:, :3].T + shell['planes'][:, 3]
            facet = int(np.argmax(values))
            direction = -(rotation @ shell['planes'][facet, :3])
            clearance = float(values[facet])
        else:
            # Use exactly the production observation scope (nearest component,
            # evidence radius, winding fallback). A remembered witness cannot
            # extend its former tangent patch into a new region.
            signature = (key[0], oid, geom)
            values = None if forward_cache is None else forward_cache.get(signature)
            if values is None:
                owner.data.qpos[:] = pose
                mujoco.mj_forward(model, owner.data)
                band = owner.cfg['activation_distance']
                values = owner._forward_samples(obj, translation, object_rotation, scale, band, [shell])
                owner._open_forward_samples(obj, values, band / scale)
                if forward_cache is not None:
                    forward_cache[signature] = values
            world, _, components = values[geom]
            entry = components.get(component)
            if entry is None:
                return None
            ids, signed = entry
            matches = np.flatnonzero(ids == sample)
            if not len(matches):
                return None
            index = matches[0]
            if signed['uncertain'][index] or not np.isfinite(signed['clearance'][index]):
                return None
            point = world[sample]
            clearance = float(signed['clearance'][index] * scale)
            direction = object_rotation @ signed['directions'][index]
        return dict(geom_id=geom, object_id=oid, kind=kind, component_id=component,
                    sample_id=sample, point=point, direction=direction, clearance=clearance)
