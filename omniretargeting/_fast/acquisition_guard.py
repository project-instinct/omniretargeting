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

import os

import cvxpy as cp
import numpy as np

from .kinematics import TrajectoryKinematics
from .rows import _block_rows
from .surface_presence import body_penetration, on_surface
from .contact_protection import endpoint_protection, surface_protection


# Default endpoint slack is min(1 mm, endpoint_regression_m=1 mm) = 1 mm.
# Surface and self-contact regression defaults are 2 mm, hence also 1 mm here.
# Runtime margins must stay below the smallest configured slack; half the
# frozen bound is only an additional per-sample safety clamp.
HINGE_MARGIN_M = float(os.environ.get('OMNIRETARGETING_FAST_HINGE_MARGIN', '0'))


class AcquisitionNormalGuard:
    @property
    def numerical_slack(self):
        """Only this guard family's numerical reserve; larger opt-in wins."""
        from .runtime_options import OPTIONS
        if OPTIONS.abl_normal_slack_1e5:
            return 1e-5
        return 1e-6 if OPTIONS.abl_normal_slack else 1e-7

    def __init__(self, objective, reference):
        self.objective = objective
        self.priority = getattr(objective, 'priority_clearance', None)
        self.kin = TrajectoryKinematics(objective.model, objective.coords.cols, cache=objective.kin.cache)
        self.kin.evaluate(reference, with_jacobian=False)
        self.groups, self.names = [], []
        self.penetration = body_penetration(objective, reference)
        for eff in objective.effectors:
            protected = endpoint_protection(objective, eff)
            active = eff['active'] & ~eff['managed'] & protected
            self._add('endpoint', np.array([eff['body']]), eff['offset'][:, None],
                      eff['contact'][:, None], eff['normal'][:, None], active[:, None],
                      min(.001, objective.cfg['endpoint_regression_m']), name=eff['name'])
        for index, surf in enumerate(objective.surfaces):
            protected = surface_protection(objective, surf)
            self._add('surface', surf['geoms'], surf['local'], surf['targets'],
                      surf['normals'], surf['active'] & protected,
                      min(.001, surf['task'].cfg['regression']), name=f'surface_{index}')
        if self.priority is not None:
            self.priority.register(self)

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

    def window_dependencies(self, frames):
        return frames

    def qp_bounds(self, poses, *, frames=None):
        if frames is None:
            return self.repair_bounds(poses)
        if self.priority is not None:
            return [], []  # Coordinator owns these soft rows and funded acceptance.
        if self.numerical_slack != 1e-7:
            return self._window_ablation_bounds(poses, frames)
        from .priority_contacts import PriorityContacts
        from .priority_window_contacts import frozen_bounds
        return frozen_bounds(PriorityContacts(self, type(self).__name__), poses, frames)

    def accepts(self, poses, *, frames=None):
        if frames is not None:
            return not self.violation_frames(poses, frames=frames).any()
        if self.priority is not None:
            return self.priority.accepts(poses)
        return self.diagnostics(poses)['accepted']

    def violation_frames(self, poses, *, frames=None):
        if frames is not None:
            if self.priority is not None:
                return self.priority.violation_frames(poses, frames=frames)
            if self.numerical_slack != 1e-7:
                bad = np.zeros(len(poses), bool)
                for block in self._window_ablation_rows(poses, frames, jacobian=False):
                    values = block.values
                    violated = ~np.isfinite(values) | (values > block.bound + block.own + self.numerical_slack)
                    bad[block.frames[violated]] = True
                return bad
            from .priority_contacts import PriorityContacts
            from .priority_window_contacts import frozen_violations
            return frozen_violations(PriorityContacts(self, type(self).__name__), poses, frames)
        """Frames ``accepts`` rejects: a protected point beyond its bound."""
        if self.priority is not None:
            return self.priority.violation_frames(poses)
        self.kin.evaluate(poses, with_jacobian=False)
        bad = np.zeros(len(poses), bool)
        for kind, indices, local, target, normal, active, established, bound, yielded in self.groups:
            offset = self._points(kind, indices, local) - target
            along = np.sum(offset * normal, axis=-1)
            gap = np.where(established, np.maximum(along - bound - yielded, -along - bound),
                           np.linalg.norm(offset, axis=-1) - bound - yielded)
            bad |= np.any(active & ~(gap <= self.numerical_slack), axis=1)
        return bad

    def diagnostics(self, poses):
        if self.priority is not None:
            return self.priority.diagnostics(poses)
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
        return dict(accepted=bool(np.isfinite(excess) and excess <= self.numerical_slack),
                    max_excess_m=excess, worst_constraint=worst,
                    fixed_pre_acquisition_reference=not hasattr(self, '_regression_reference'),
                    bound_reference=getattr(self, '_regression_reference', 'fixed_pre_acquisition'),
                    protected_frame_points=sum(int(g[5].sum()) for g in self.groups),
                    on_surface_frame_points=sum(int(g[6].sum()) for g in self.groups),
                    penetration_yield_max_m=float(max((g[8][g[5]].max(initial=0) for g in self.groups),
                                                      default=0.)))

    def _window_ablation_rows(self, poses, frames, *, jacobian=True):
        from .priority_contacts import PriorityContacts
        return PriorityContacts(self, type(self).__name__).evaluate(
            poses, frames=frames, jacobian=jacobian)

    def _window_ablation_bounds(self, poses, frames):
        # Keep the generic frozen-window kernels (and ContactGuard) unchanged.
        linear, cones = [], []
        slack = self.numerical_slack
        for block in self._window_ablation_rows(poses, frames):
            base = block.bound + block.own
            allowed = np.minimum(np.maximum(base + slack / 2, block.values), base + slack)
            if block.scalar:
                linear.append((block.matrix, np.full(len(base), -np.inf), allowed - block.residual))
            else:
                cones.append((block.matrix, block.residual, allowed))
        return linear, cones

    def hinge_rows(self, kin, frames, weight, sigma, with_jacobian, *, inner_margin=None):
        """Stiff one-sided rows of the same bounds on another trajectory's
        kinematics (``kin`` covers ``frames``), so an unconstrained step
        already respects them; acceptance remains ``accepts``."""
        if self.priority is not None:
            return None, None
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
            margin = np.minimum(HINGE_MARGIN_M if inner_margin is None else inner_margin,
                                .5 * bound[frames])
            out = np.maximum(along - bound[frames] - yielded[frames] + margin, 0.)
            inward = np.maximum(-along - bound[frames] + margin, 0.)
            far = np.maximum(distance - bound[frames] - yielded[frames] + margin, 0.)
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
        if self.priority is not None:
            return [], []
        self.kin.evaluate(poses)
        linear, norms = [], []
        width = self.objective.width
        slack = self.numerical_slack
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
                # numerical roundoff; reserve half this family's nonlinear cap.
                inward = np.minimum(np.maximum(bound[held] + slack / 2, -residual), bound[held] + slack)
                outward = bound[held] + yielded[held]
                outward = np.minimum(np.maximum(outward + slack / 2, residual), outward + slack)
                linear.append((matrix, -inward - residual, outward - residual))
            reaching = active & ~established
            if reaching.any():
                residual = (points - target)[reaching]
                limit = bound[reaching] + yielded[reaching]
                allowance = np.minimum(np.maximum(limit + slack / 2, np.linalg.norm(residual, axis=1)),
                                       limit + slack)
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
