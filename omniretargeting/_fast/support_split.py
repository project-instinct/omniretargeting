"""Per-source-support normal, tangent and world bounds with measured curvature.

Bounds always belong to the fixed reference. Clearance can spend only the
normal exit evidenced at this owner/surface, never another body's penetration.
"""
import cvxpy as cp
import numpy as np

from .kinematics import TrajectoryKinematics
from .rows import _block_rows

ROUND_OFF = 1e-7  # AcquisitionNormalGuard's unchanged default numerical reserve.


def split_error(offset, normal):
    along = np.sum(offset * normal, axis=-1)
    return along, offset - along[..., None] * normal


def split_bounds(offset, normal, source_depth, exit_depth, slack):
    along, tangent = split_error(offset, normal)
    # Initially infeasible normals may approach the surface. No new inward
    # motion is funded. A source-penetrating owner can spend its verified exit.
    outward = np.maximum(along, 0.) + np.where(
        source_depth > 0, np.maximum(exit_depth + np.minimum(along, 0.), 0.), 0.)
    inward = np.maximum(-along, 0.)
    return inward + slack, outward + slack, np.linalg.norm(tangent, axis=-1) + slack, np.linalg.norm(offset, axis=-1) + slack


class SupportSplit:
    def __init__(self, objective, reference, geometry):
        self.objective, self.geometry = objective, geometry
        self.kin = TrajectoryKinematics(objective.model, objective.coords.cols, cache=objective.kin.cache)
        self.kin.evaluate(reference)
        self.reference = reference.copy()
        self.groups = []
        self.allowance_records = {}
        self.origin = None
        self.models = []
        self.rejected = 0

    def extract(self, kind, indices, local, target, normal, active, slack, name):
        normal = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        policy = self.objective.tasks.contact_phase
        if policy is None:
            return active
        task = None
        if kind == 'endpoint':
            support = np.array([policy.role(name, f) == 'support' for f in range(len(active))])[:, None]
            # Endpoint evidence comes from source material contacts on its weld.
            surfaces = [s for s in self.objective.surfaces if s['task'] is getattr(policy, 'surface', None)]
            depth = np.zeros_like(active, dtype=float)
            sid = np.full(active.shape, -1, int)
            body = self.objective.model.body_weldid[indices[0]]
            for surface in surfaces:
                task = surface['task']
                columns = self.objective.model.body_weldid[surface['bodies']] == body
                observed = np.asarray(policy.surface_roles.get('support', False)) & surface['active']
                source = -np.sum(task.source_vectors * surface['normals'], axis=-1)
                for f in range(len(active)):
                    candidates = np.flatnonzero(columns & observed[f])
                    if len(candidates):
                        j = candidates[np.argmax(source[f, candidates])]
                        # Match this endpoint's target plane and outward direction.
                        if (np.dot(normal[f, 0], surface['normals'][f, j]) >= 1 - ROUND_OFF
                                and abs(np.dot(target[f, 0] - task.environment_points[f, j], normal[f, 0])) <= slack):
                            depth[f, 0] = max(source[f, j], 0.)
                            sid[f, 0] = task.surface_ids[f, j]
        else:
            index = int(name.split('_')[-1])
            task = self.objective.surfaces[index]['task']
            if task is not getattr(policy, 'surface', None):
                return active
            support = np.asarray(policy.surface_roles.get('support', np.zeros_like(active)))
            target = task.environment_points  # Source projection, including moving surfaces.
            depth = np.maximum(-np.sum(task.source_vectors * normal, axis=-1), 0.)
            sid = task.surface_ids
        support = active & support
        if not support.any():
            return active
        points = self.points(kind, indices, local)
        exit_depth = np.zeros_like(depth)
        known = np.zeros_like(active)
        # Query only this owner's own point and projected support target. The
        # original surface, sign evidence and normals must agree at both.
        for surface_id in np.unique(sid[support]):
            if surface_id < 0:
                continue
            frames, samples = np.nonzero(support & (sid == surface_id))
            for f in np.unique(frames):
                samples_f = samples[frames == f]
                query, transform, scale = self.surface_query(int(surface_id), int(f))
                if query is None:
                    continue
                pp = points[f, samples_f]
                yy = target[f, samples_f]
                measured = query.query(transform(np.r_[pp, yy]))
                k = len(pp)
                phi = measured['clearance'][:k] * scale
                reliable = ~measured['uncertain'][:k] & ~measured['uncertain'][k:]
                directions = measured['directions'][:k]
                if surface_id != getattr(task, 'static_surface_id', 0) and self.geometry.moving is not None:
                    from ..moving_scene import interpolate_object
                    oid = surface_id - int(self.geometry.scene is not None)
                    _, rotation, _ = interpolate_object(self.geometry.moving.objects[oid]['surface'], f)
                    directions = directions @ rotation.T
                aligned = np.sum(directions * normal[f, samples_f], axis=1) >= 1 - ROUND_OFF
                projected = pp - phi[:, None] * directions
                plane = np.abs(np.sum((projected - yy) * normal[f, samples_f], axis=1)) <= slack + ROUND_OFF
                same = measured['obstacle_ids'][:k] == measured['obstacle_ids'][k:]
                good = reliable & np.isfinite(phi) & aligned & plane & same
                known[f, samples_f] = good
                exit_depth[f, samples_f] = np.where(good, np.maximum(-phi, 0.), 0.)
        # Use only original witnesses of this owner on this exact support
        # plane; a side wall or a different geom's contact grants no yield.
        model = self.objective.model
        bodies = model.geom_bodyid[indices] if kind == 'surface' else indices
        welds = np.broadcast_to(model.body_weldid[bodies], active.shape)
        for f, sample in zip(*np.nonzero(support & known & (depth > 0))):
            cache = getattr(self.geometry, '_window_frame_cache', {}).get(int(f), {})
            token = (getattr(self.geometry, 'reference_observation_id', None), self.reference[f].tobytes())
            entry = cache.get(token)
            for key, row in (() if entry is None else entry[1]):
                if key[1] not in ('static', 'moving'):
                    continue
                row_sid = 0 if key[1] == 'static' else key[2] + int(self.geometry.scene is not None)
                if (row_sid != sid[f, sample]
                        or model.body_weldid[model.geom_bodyid[row['geom_id']]] != welds[f, sample]):
                    continue
                direction, phi = np.asarray(row['direction']), row['clearance']
                projection = row['point'] - phi * direction
                if (np.dot(direction, normal[f, sample]) >= 1 - ROUND_OFF
                        and abs(np.dot(projection - target[f, sample], normal[f, sample])) <= ROUND_OFF):
                    exit_depth[f, sample] = max(exit_depth[f, sample], -phi)
        bounds = split_bounds(points - target, normal, depth, exit_depth, slack)
        group = dict(kind=kind, indices=np.array(indices), local=np.array(local), target=np.array(target),
                     normal=np.array(normal), active=support, source_depth=depth, sid=np.array(sid),
                     verified=known, bounds=bounds, kappa=np.zeros((*active.shape, 4)), name=name)
        self.groups.append(group)
        for f, sample in zip(*np.nonzero(support & known)):
            token = int(f), int(welds[f, sample]), int(sid[f, sample])
            self.allowance_records.setdefault(token, []).append((normal[f, sample].copy(), target[f, sample].copy()))
        return active & ~support

    def surface_query(self, sid, frame):
        if self.geometry.scene is not None and sid == 0:
            return self.geometry.scene, lambda p: p, 1.
        moving = self.geometry.moving
        oid = sid - int(self.geometry.scene is not None)
        if moving is None or not 0 <= oid < len(moving.objects):
            return None, None, 1.
        from ..moving_scene import interpolate_object
        translation, rotation, scale = interpolate_object(moving.objects[oid]['surface'], frame)
        return moving.objects[oid]['query'], lambda p: (p - translation) @ rotation / scale, scale

    def points(self, kind, indices, local):
        if kind == 'surface':
            return self.kin.geom_points(indices, local)
        return self.kin.points(np.broadcast_to(indices, local.shape[:2]), local)

    def values(self, poses, jacobian=False):
        self.kin.evaluate(poses, with_jacobian=jacobian)
        result = []
        for group in self.groups:
            p = self.points(group['kind'], group['indices'], group['local'])
            offset = p - group['target']
            n, tangent = split_error(offset, group['normal'])
            values = np.stack([-n, n, np.linalg.norm(tangent, axis=-1), np.linalg.norm(offset, axis=-1)], -1)
            jac = None
            if jacobian:
                bodies = self.objective.model.geom_bodyid[group['indices']] if group['kind'] == 'surface' else group['indices']
                jp = self.kin.point_jacobians(bodies, p)
                jn = np.einsum('fsi,fsiw->fsw', group['normal'], jp)
                jt = jp - group['normal'][..., None] * jn[..., None, :]
                jac = (jp, jn, jt, tangent, offset)
            result.append((values, jac))
        return result

    def begin(self, poses):
        self.origin = poses.copy()
        self.models = self.values(poses, True)

    def accepts(self, poses):
        values = self.values(poses)
        good = True
        step = None if self.origin is None else self.objective.coords.difference(self.origin, poses)
        for group, (actual, _), model in zip(self.groups, values, self.models or values):
            bounds = np.stack(group['bounds'], -1)
            active = group['active']
            if step is not None and model[1] is not None:
                jp, jn, jt, tangent, offset = model[1]
                dn = np.einsum('fsw,fw->fs', jn, step)
                dt = tangent + np.einsum('fsiw,fw->fsi', jt, step)
                dp = offset + np.einsum('fsiw,fw->fsi', jp, step)
                pred = np.stack([model[0][..., 0] - dn, model[0][..., 1] + dn,
                                 np.linalg.norm(dt, axis=-1), np.linalg.norm(dp, axis=-1)], -1)
                norm2 = np.sum(step * step, axis=1)[:, None, None]
                calibrated = np.maximum(actual - pred, 0.) / np.maximum(norm2, 1e-20)
                group['kappa'] = np.maximum(group['kappa'], calibrated)
            good &= bool(np.all(np.isfinite(actual[active])) and np.all(actual[active] <= bounds[active] + ROUND_OFF))
        self.rejected += not good
        return good

    def constraints(self, poses, flat):
        data = self.values(poses, True)
        width, frames = self.objective.width, len(poses)
        square = cp.sum(cp.square(cp.reshape(flat, (width, frames), order='F')), axis=0)
        constraints = []
        for group, (values, jac) in zip(self.groups, data):
            active = group['active']
            ff, _ = np.nonzero(active)
            jp, jn, jt, tangent, offset = jac
            nmatrix = _block_rows(jn, width)[active.ravel()]
            bounds = np.stack(group['bounds'], -1)[active] + ROUND_OFF / 2
            curvature = group['kappa'][active]
            constraints += [-nmatrix @ flat + values[..., 0][active] + cp.multiply(curvature[:, 0], square[ff]) <= bounds[:, 0],
                            nmatrix @ flat + values[..., 1][active] + cp.multiply(curvature[:, 1], square[ff]) <= bounds[:, 1]]
            for channel, matrix, residual in ((2, jt, tangent), (3, jp, offset)):
                matrix = _block_rows(matrix.reshape(frames, -1, width), width)[np.repeat(active.ravel(), 3)]
                vector = cp.reshape(matrix @ flat + residual[active].ravel(), (len(ff), 3), order='C')
                constraints.append(cp.norm(vector, axis=1) + cp.multiply(curvature[:, channel], square[ff]) <= bounds[:, channel])
        return constraints

    def allowance(self, key, row):
        if key[1] == 'self':
            return 0.
        frame, model = key[0], self.objective.model
        weld = model.body_weldid[model.geom_bodyid[row['geom_id']]]
        sid = 0 if key[1] == 'static' else key[2] + int(self.geometry.scene is not None)
        direction = np.asarray(row['direction'])
        projected = row['point'] - row['clearance'] * direction
        for n, y in self.allowance_records.get((frame, int(weld), sid), ()):
            if (np.dot(n, direction) >= 1 - ROUND_OFF
                    and abs(np.dot(projected - y, n)) <= ROUND_OFF):
                return .005
        return 0.

    def diagnostics(self, poses):
        worst = {'eN': None, 'eT': None, 'world': None, 'excess': None}
        maxima = dict(eN=0., eT=0., world=0., excess=0.)
        for group, (values, _) in zip(self.groups, self.values(poses)):
            active = group['active']
            bounds = np.stack(group['bounds'], -1)
            for name, val in (('eN', abs(values[..., 1])), ('eT', values[..., 2]),
                              ('world', values[..., 3]), ('excess', (values - bounds).max(axis=-1))):
                f, s = np.unravel_index(np.argmax(np.where(active, val, -np.inf)), active.shape)
                if val[f, s] > maxima[name]:
                    maxima[name] = float(val[f, s])
                    worst[name] = dict(group=group['name'], frame=int(f), sample=int(s), value_m=float(val[f, s]))
        return dict(maxima_m=maxima, worst=worst, accepted=maxima['excess'] <= ROUND_OFF,
                    source_support_points=sum(int(g['active'].sum()) for g in self.groups),
                    verified_points=sum(int((g['active'] & g['verified']).sum()) for g in self.groups))
