"""Protect support/unknown-hold separation during contact acquisition.

A weighted vector error can improve while a contact moves away from its
surface. Freeze its normal allowance at the source-shaped reference, before
object-control objectives can trade it away. This is a bound on regression,
not a claim that an initially imperfect contact has been solved.

Non-penetration precedes contacts: a protected point may move outward by as
much as its own body penetrates the scene or objects at the reference
(``surface_presence.body_penetration``), and no further.

Only points actually on an original surface at the reference keep that normal
allowance (``surface_presence``). A point off every surface, e.g. far from its
target or buried in an object wall, only crosses the extension of the
target's tangent plane. It may approach its target but not recede from it.
"""

import cvxpy as cp
import numpy as np

from .kinematics import TrajectoryKinematics
from .rows import _block_rows
from .surface_presence import body_penetration, on_surface


class AcquisitionNormalGuard:
    def __init__(self, objective, reference):
        self.objective = objective
        self.kin = TrajectoryKinematics(objective.model, objective.coords.cols, cache=objective.kin.cache)
        self.kin.evaluate(reference, with_jacobian=False)
        self.groups, self.names = [], []
        self.penetration = body_penetration(objective, reference)
        policy = objective.tasks.contact_phase
        for eff in objective.effectors:
            protected = ~eff['surface_only'] if policy is None else np.array([
                policy.role(eff['name'], f) in ('support', 'hold')
                for f in range(objective.frames)])
            active = eff['active'] & ~eff['managed'] & protected
            self._add('endpoint', np.array([eff['body']]), eff['offset'][:, None],
                      eff['contact'][:, None], eff['normal'][:, None], active[:, None],
                      min(.001, objective.cfg['endpoint_regression_m']), name=eff['name'])
        for index, surf in enumerate(objective.surfaces):
            roles = getattr(policy, 'surface_roles', {})
            if getattr(policy, 'surface', None) is surf['task']:
                protected = roles.get('support', False) | roles.get('hold', False)
            else:
                protected = surf['sticking']
            self._add('surface', surf['geoms'], surf['local'], surf['targets'],
                      surf['normals'], surf['active'] & protected,
                      min(.001, surf['task'].cfg['regression']), name=f'surface_{index}')

    def _points(self, kind, indices, local):
        if kind == 'surface':
            return self.kin.geom_points(indices, local)
        return self.kin.points(np.broadcast_to(indices, local.shape[:2]), local)

    def _add(self, kind, indices, local, target, normal, active, slack, name=None):
        length = np.linalg.norm(normal, axis=-1)
        active = np.asarray(active, bool) & np.isfinite(length) & (length > 1e-12)
        if not active.any():
            return
        normal = normal / np.maximum(length[..., None], 1e-12)
        points = self._points(kind, indices, local)
        established = active & on_surface(self.objective, points)
        bound = np.where(established, np.abs(np.sum((points - target) * normal, axis=-1)),
                         np.linalg.norm(points - target, axis=-1)) + slack
        model = self.objective.model
        bodies = model.geom_bodyid[indices] if kind == 'surface' else np.broadcast_to(indices, local.shape[:2])
        yielded = self.penetration[np.arange(len(points))[:, None], model.body_weldid[bodies]]
        from .runtime_options import OPTIONS
        if OPTIONS.off_surface_approach_only:
            # An absent surface does not establish a normal hold. A different
            # penetrating point on this body must not authorize recession of
            # a separated/unknown-target point from its unchanged target.
            yielded = np.where(established, yielded, 0.)
        values = [np.array(x, copy=True) for x in (indices, local, target, normal, active, established,
                                                   bound, np.broadcast_to(yielded, bound.shape))]
        for value in values:
            value.setflags(write=False)
        self.groups.append((kind, *values))
        self.names.append(name or kind)

    def accepts(self, poses):
        return self.diagnostics(poses)['accepted']

    def diagnostics(self, poses):
        if not np.isfinite(poses).all():
            return dict(accepted=False, max_excess_m=float('inf'))
        self.kin.evaluate(poses, with_jacobian=False)
        excess, worst = 0., None
        for group_index, (kind, indices, local, target, normal, active, established, bound, yielded) in enumerate(self.groups):
            offset = self._points(kind, indices, local) - target
            along = np.sum(offset * normal, axis=-1)
            gap = np.where(established, np.maximum(along - bound - yielded, -along - bound),
                           np.linalg.norm(offset, axis=-1) - bound - yielded)
            if not np.isfinite(gap[active]).all():
                return dict(accepted=False, max_excess_m=float('inf'))
            maximum = float(np.max(gap[active]))
            if maximum > excess:
                frame, sample = np.unravel_index(np.argmax(np.where(active, gap, -np.inf)), gap.shape)
                worst = dict(group=self.names[group_index], frame=int(frame), sample=int(sample),
                             excess_m=maximum, frozen_penetration_yield_m=float(yielded[frame, sample]))
            excess = max(excess, maximum)
        return dict(accepted=bool(np.isfinite(excess) and excess <= 1e-7),
                    max_excess_m=excess, worst_constraint=worst, fixed_pre_acquisition_reference=True,
                    protected_frame_points=sum(int(g[5].sum()) for g in self.groups),
                    on_surface_frame_points=sum(int(g[6].sum()) for g in self.groups),
                    penetration_yield_max_m=float(max((g[8][g[5]].max(initial=0) for g in self.groups),
                                                      default=0.)))

    def hinge_rows(self, kin, frames, weight, sigma, with_jacobian):
        """Stiff one-sided rows of the same bounds on another trajectory's
        kinematics (``kin`` covers ``frames``), so an unconstrained step
        already respects them; acceptance remains ``accepts``."""
        rows, jacs = [], []
        scale = np.asarray(weight, dtype=float)[:, None] / sigma
        for kind, indices, local, target, normal, active, established, bound, yielded in self.groups:
            local_part = local if kind == 'surface' else local[frames]
            if kind == 'surface':
                points = kin.geom_points(indices, local_part)
                bodies = self.objective.model.geom_bodyid[indices]
            else:
                bodies = np.broadcast_to(indices, local_part.shape[:2])
                points = kin.points(bodies, local_part)
            offset = points - target[frames]
            along = np.sum(offset * normal[frames], axis=-1)
            distance = np.linalg.norm(offset, axis=-1)
            held, reach = (active & established)[frames], (active & ~established)[frames]
            out = np.maximum(along - bound[frames] - yielded[frames], 0.)
            inward = np.maximum(-along - bound[frames], 0.)
            far = np.maximum(distance - bound[frames] - yielded[frames], 0.)
            residual = np.where(held, out + inward, np.where(reach, far, 0.))
            rows.append(residual * scale)
            if with_jacobian:
                jac = kin.point_jacobians(bodies, points)
                direction = np.where(held[..., None], normal[frames] * np.sign(out - inward)[..., None],
                                     offset / np.maximum(distance, 1e-12)[..., None])
                direction = np.where((residual > 0)[..., None], direction, 0.)
                jacs.append(np.einsum('fsi,fsiw->fsw', direction, jac) * scale[..., None])
        if not rows:
            return None, None
        return np.concatenate(rows, axis=1), (np.concatenate(jacs, axis=1) if with_jacobian else None)

    def repair_bounds(self, poses):
        """The same frozen bounds in sparse tangent coordinates: normal rows
        for points on a surface, full-distance cones for the others."""
        self.kin.evaluate(poses)
        linear, norms = [], []
        width = self.objective.width
        for kind, indices, local, target, normal, active, established, bound, yielded in self.groups:
            points = self._points(kind, indices, local)
            bodies = self.objective.model.geom_bodyid[indices] if kind == 'surface' else indices
            jac = self.kin.point_jacobians(bodies, points)
            held = active & established
            if held.any():
                rows = np.einsum('fsi,fsiw->fsw', normal, jac)
                matrix = _block_rows(rows, width)[held.ravel()]
                residual = np.sum((points - target) * normal, axis=-1)[held]
                # Preserve zero feasibility when an accepted pose has spent
                # numerical roundoff; the immutable 1e-7 nonlinear cap is unchanged.
                inward = np.minimum(np.maximum(bound[held] + 5e-8, -residual), bound[held] + 1e-7)
                outward = bound[held] + yielded[held]
                outward = np.minimum(np.maximum(outward + 5e-8, residual), outward + 1e-7)
                linear.append((matrix, -inward - residual, outward - residual))
            reaching = active & ~established
            if reaching.any():
                residual = (points - target)[reaching]
                limit = bound[reaching] + yielded[reaching]
                allowance = np.minimum(np.maximum(limit + 5e-8, np.linalg.norm(residual, axis=1)),
                                       limit + 1e-7)
                matrix = _block_rows(jac.reshape(len(poses), -1, width), width)[np.repeat(reaching.ravel(), 3)]
                norms.append((matrix, residual, allowance))
        return linear, norms

    def constraints(self, poses, flat):
        linear, norms = self.repair_bounds(poses)
        result = []
        for matrix, lower, upper in linear:
            result += [matrix @ flat <= upper, matrix @ flat >= lower]
        for matrix, residual, allowance in norms:
            result.append(cp.SOC(allowance, cp.reshape(matrix @ flat + residual.ravel(),
                                                      (3, len(residual)), order="F"), axis=0))
        return result
