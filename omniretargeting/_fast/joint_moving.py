"""Multiple original forward witnesses without altering MovingSceneQuery defaults."""
import numpy as np

from ..moving_scene import MovingSceneQuery
from ..moving_sample_summary import summarize
from .collision_manifold import select_collision_manifold


class JointMoving(MovingSceneQuery):
    def _material_samples(self, oid, obj, pose, time, translation, rotation, scale, band, shells):
        samples, missing, signatures = {}, [], {}
        object_pose = translation.tobytes(), rotation.tobytes(), float(scale)
        for shell in shells:
            gid = shell['geom_id']
            key = (oid, float(time), gid, self.data.geom_xpos[gid].tobytes(),
                   self.data.geom_xmat[gid].tobytes(), object_pose, float(band))
            signatures[gid] = key
            entry = self.material_cache.get(key)
            if entry is None:
                missing.append(shell)
            else:
                samples[gid] = entry
                self.material_cache.move_to_end(key)
        if missing:
            fresh = self._forward_samples(obj, translation, rotation, scale, band, missing)
            self._open_forward_samples(obj, fresh, band / scale)
            for gid, values in fresh.items():
                summary = summarize(self, oid, obj, gid, values, rotation, scale, band)
                world, _, components = values
                rows = []
                for cid, entry in components.items():
                    if cid == 'hull':
                        continue
                    ids, signed = entry
                    valid = ~signed['uncertain'] & np.isfinite(signed['clearance'])
                    if cid in obj['closed'] and not valid.all():
                        continue  # Retain the base summary's explicit failure.
                    directions = signed['directions'] @ rotation.T
                    phi = signed['clearance'] * scale
                    selected = select_collision_manifold(world[ids], phi, directions, valid=valid,
                                                         origin=self.data.geom_xpos[gid], activation_distance=band)
                    rows.extend((world[ids[k]], directions[k], float(phi[k]), int(ids[k]),
                                 cid in obj['closed'], cid) for k in selected)
                summary['rows'] = rows
                self.material_cache[signatures[gid]] = samples[gid] = summary
            while len(self.material_cache) > 8192:
                self.material_cache.popitem(last=False)
        return samples
