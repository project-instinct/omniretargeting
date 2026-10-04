"""Role-preserving world contact rows for private soft acquisition.

Only mechanics-ranked supports receive stiff position anchors. Existing
material sticking observations also receive weaker relative velocity rows,
including role hand-offs on the same surface. All targets/normals come from
the original contact plan; no episode is rebased to an optimized pose.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .kinematics import cross_columns


@dataclass(frozen=True)
class SupportWeights:
    anchor_sigma: float = .001
    normal_sigma: float = .0005
    support_velocity_sigma: float = .001
    sticking_velocity_sigma: float = .005
    observed_anchor_sigma: float = .05
    observed_normal_sigma: float = .02
    hold_normal_sigma: float = .0005
    hold_velocity_sigma: float = .001
    hold_anchor_sigma: float = .003


def surface_roles(objective, surface):
    """Include observations delegated to their endpoint, by source identity."""
    from ..contact_coverage import surface_effector_samples
    policy = getattr(objective.tasks, 'contact_phase', None)
    roles = np.full(surface['active'].shape, 'free', dtype='U7')
    task = surface['task']
    if policy is None or getattr(policy, 'surface', None) is not task:
        return roles
    for role, mask in policy.surface_roles.items():
        roles[np.asarray(mask)] = role
    for name, spec in objective.tasks.effectors.items():
        if name not in policy.roles:
            continue
        selected = surface_effector_samples(task, name, spec)
        owned = surface['active'][:, selected] & (roles[:, selected] == 'free')
        roles[:, selected] = np.where(owned, policy.roles[name][:, None], roles[:, selected])
    return roles


def _matrix(blocks, frames, width, count):
    """Sparse one-frame rows; direct band assembly consumes these blocks."""
    blocks = np.asarray(blocks).reshape(len(frames), -1, width)
    rows = blocks.shape[1]
    columns = frames[:, None, None] * width + np.arange(width)[None, None, :]
    columns = np.broadcast_to(columns, blocks.shape).reshape(-1, width)
    matrix = sparse.csr_matrix((blocks.ravel(), columns.ravel(),
        np.arange(len(frames) * rows + 1) * width), shape=(len(frames) * rows, count * width))
    matrix.eliminate_zeros()
    return matrix


class SupportRows:
    def __init__(self, objective, weights=None, *, yield_targets=False):
        self.objective, self.weights = objective, weights or SupportWeights()
        if any(not np.isfinite(v) or v <= 0 for v in vars(self.weights).values()):
            raise ValueError('Contact row sigmas must be finite and positive')
        self.groups = []
        policy = getattr(objective.tasks, 'contact_phase', None)
        for surface in objective.surfaces:
            roles = surface_roles(objective, surface)
            # Once geometry declares a material contact, its holding rows
            # cannot disappear with fitting confidence. Original exact guards
            # protected these same active points without confidence weights.
            # Normalize only sampling density, uniformly for every source part.
            density = np.zeros(surface['active'].shape)
            parts = np.asarray([b.part for b in surface['task'].bindings])
            for part in np.unique(parts):
                selected = parts == part
                count = max(int(surface['active'][:, selected].sum(axis=1).max()), 1)
                density[:, selected] = surface['active'][:, selected] / count
            self.groups.append(dict(kind='geom', indices=surface['geoms'], bodies=surface['bodies'],
                local=surface['local'], target=surface['targets'], normal=surface['normals'],
                active=surface['active'], sticking=surface['sticking'], roles=roles,
                weights=density,
                ids=np.asarray(surface['task'].surface_ids)))
        for effector in objective.effectors:
            name = effector['name']
            roles = (policy.roles[name] if policy is not None and name in policy.roles
                     else np.full(objective.frames, 'free'))
            task = objective.tasks.effectors[name]['task']
            mode = np.asarray(task.contact_modes) if task.contact_modes is not None else np.full(objective.frames, 'sticking')
            ids = (np.zeros(objective.frames, int) if task.environment_ids is None
                   else np.asarray(task.environment_ids))
            # Dense surface observations own managed endpoints. Avoid double
            # anchors while retaining uncovered endpoint observations.
            self.groups.append(dict(kind='body', indices=np.array([effector['body']]),
                bodies=np.array([effector['body']]), local=effector['offset'][:, None],
                target=effector['contact'][:, None], normal=effector['normal'][:, None],
                active=effector['active'][:, None], sticking=(mode == 'sticking')[:, None],
                roles=np.asarray(roles)[:, None], weights=np.ones((objective.frames, 1)), ids=ids[:, None]))
        for group in self.groups:
            group['fitting_target'] = group['target']
            if yield_targets:
                from .soft_swing_rows import target_yield
                frames,samples=np.nonzero(group['active'])
                fields=objective._field_index(group['ids'][frames,samples])
                depth=np.zeros(len(frames));normal=np.zeros((len(frames),3))
                for field in np.unique(fields[fields>=0]):
                    selected=fields==field
                    depth[selected],normal[selected]=target_yield(objective,
                        group['target'][frames[selected],samples[selected]],frames[selected],field=int(field))
                fitting=group['target'].copy()
                fitting[frames,samples]+=depth[:,None]*normal
                fitting.setflags(write=False)
                group['fitting_target']=fitting
                group['source_yield_depth']=depth
            active, role = group['active'], group['roles']
            group['anchor'] = np.nonzero(active & group['sticking'] & (role == 'support'))
            # Unexplained holds retain their source-world fitting target too.
            # These rows stay softer than ranked supports; their confidence
            # must not silently erase tangential holding while normals remain.
            group['touch_anchor'] = np.nonzero(active & (role != 'free') &
                                               ~((role == 'support') & group['sticking']))
            group['normal_rows'] = np.nonzero(active & (role != 'free'))
            same = group['ids'][1:] == group['ids'][:-1]
            group['velocity'] = np.nonzero(active[1:] & active[:-1] & group['sticking'][1:]
                                           & group['sticking'][:-1] & same)
        self.diagnostics = dict(support_anchor_points=sum(len(g['anchor'][0]) for g in self.groups),
            contact_normal_points=sum(len(g['normal_rows'][0]) for g in self.groups),
            sticking_velocity_points=sum(len(g['velocity'][0]) for g in self.groups),
            support_source='original_contact_mechanics', targets='original_world_surface_targets',
            holding_weights='sample_density_only_no_confidence_attenuation',
            sigmas=vars(self.weights).copy(), source_normal_yield=yield_targets,
            yielded_source_points=sum(int((g.get('source_yield_depth',np.empty(0))>0).sum()) for g in self.groups),
            source_yield_max_m=max((float(g.get('source_yield_depth',np.zeros(1)).max(initial=0.)) for g in self.groups),default=0.))

    @staticmethod
    def points(kin, group, frames, samples, jacobian):
        indices = group['indices'][samples]
        local = group['local']
        local = local[frames, samples] if local.ndim == 3 else local[samples]
        if group['kind'] == 'geom':
            point = kin.geom_xpos[frames, indices] + np.einsum('nij,nj->ni', kin.geom_xmat[frames, indices], local)
        else:
            point = kin.xpos[frames, indices] + np.einsum('nij,nj->ni', kin.xmat[frames, indices], local)
        derivative = None
        if jacobian:
            body = group['bodies'][samples]
            derivative = kin.jacp[frames, body] - cross_columns(point - kin.xpos[frames, body], kin.jacr[frames, body])
        return point, derivative

    def evaluate(self, poses, jacobian=True):
        obj, cfg = self.objective, self.weights
        kin = obj.window_kinematics(poses, range(len(poses)), with_jacobian=jacobian)
        residuals, matrices = [], []
        stats = dict(support_anchor_max_m=0., observed_contact_normal_max_m=0., sticking_slip_max_mps=0.)
        for group in self.groups:
            for family in ('anchor', 'touch_anchor', 'normal_rows', 'velocity'):
                frames, samples = group[family]
                if not len(frames):
                    continue
                if family == 'velocity':
                    frames = frames + 1
                point, derivative = self.points(kin, group, frames, samples, jacobian)
                error = point - group['fitting_target'][frames, samples]
                raw_error = point - group['target'][frames, samples]
                normal = group['normal'][frames, samples]
                weight = np.sqrt(group['weights'][frames, samples])
                role = group['roles'][frames, samples]
                strength = np.sqrt(np.where(role == 'support', 4., np.where(role == 'follow', 1., 2.)))
                if family in ('anchor', 'touch_anchor'):
                    sigma = (cfg.anchor_sigma if family == 'anchor' else
                             np.where(role=='hold',cfg.hold_anchor_sigma,cfg.observed_anchor_sigma))
                    scale = weight * strength / sigma
                    if family == 'anchor':
                        stats['support_anchor_max_m'] = max(stats['support_anchor_max_m'], float(np.linalg.norm(raw_error, axis=1).max()))
                    residual = error * scale[:, None]
                    matrix = _matrix(derivative * scale[:, None, None], frames, obj.width, obj.frames) if jacobian else None
                elif family == 'normal_rows':
                    # Unexplained holds are protected by the original rank
                    # contract too. Keep them above tracking, without giving
                    # other touches a support label or a stiff anchor.
                    sigma=np.where(role=='support',cfg.normal_sigma,
                                   np.where(role=='hold',cfg.hold_normal_sigma,cfg.observed_normal_sigma))
                    scale = weight * strength / sigma
                    along = np.einsum('ni,ni->n', error, normal)
                    raw_along=np.einsum('ni,ni->n',raw_error,normal)
                    stats['observed_contact_normal_max_m'] = max(stats['observed_contact_normal_max_m'], float(np.abs(raw_along).max()))
                    residual = along * scale
                    matrix = _matrix(np.einsum('ni,nij->nj', normal, derivative)[:, None] * scale[:, None, None],
                                     frames, obj.width, obj.frames) if jacobian else None
                else:
                    previous, previous_jac = self.points(kin, group, frames - 1, samples, jacobian)
                    raw_velocity = (raw_error - (previous - group['target'][frames - 1, samples])) * obj.framerate
                    # Source-depth yield must apply to holding velocity as
                    # well as position. Otherwise this stiff row pulls the
                    # robot along the source's motion inside the surface.
                    # The fitting offset is normal-only and frozen from the
                    # source geometry; tangential world targets are unchanged.
                    velocity = (error - (previous - group['fitting_target'][frames - 1, samples])) * obj.framerate
                    tangent = np.eye(3)[None] - normal[:, :, None] * normal[:, None, :]
                    velocity = np.einsum('nij,nj->ni', tangent, velocity)
                    supported = (role == 'support') | (group['roles'][frames - 1, samples] == 'support')
                    holding = (role == 'hold') | (group['roles'][frames - 1, samples] == 'hold')
                    sigma=np.where(supported,cfg.support_velocity_sigma,
                                   np.where(holding,cfg.hold_velocity_sigma,cfg.sticking_velocity_sigma))
                    scale = weight / sigma
                    stats['sticking_slip_max_mps'] = max(stats['sticking_slip_max_mps'], float(np.linalg.norm(raw_velocity, axis=1).max()))
                    residual = velocity * scale[:, None]
                    matrix = None
                    if jacobian:
                        projector = tangent * (scale * obj.framerate)[:, None, None]
                        matrix = (_matrix(np.matmul(projector, derivative), frames, obj.width, obj.frames)
                                  - _matrix(np.matmul(projector, previous_jac), frames - 1, obj.width, obj.frames))
                residuals.append(residual.ravel())
                if jacobian:
                    matrices.append(matrix)
        residual = np.concatenate(residuals) if residuals else np.empty(0)
        matrix = (sparse.vstack(matrices, format='csr') if matrices
                  else sparse.csr_matrix((len(residual), obj.frames * obj.width)))
        return residual, matrix, stats
