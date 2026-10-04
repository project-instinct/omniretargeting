"""World tracking recovery and source-point normal yield, without truth edits."""
import numpy as np
from scipy import sparse
from .soft_support_rows import SupportRows, _matrix


def target_yield(objective, targets, frames, *, field=None):
    """Each target's own signed depth; tangential coordinates stay unchanged.

    Unknown geometry yields nothing. These are objective rows toward the
    surface, never allowances in a continuity/layout/contact certificate.
    """
    targets = np.asarray(targets)
    frames = np.asarray(frames)
    depth = np.zeros(len(targets))
    normal = np.zeros_like(targets)
    fields = objective._clearance_fields()
    for number, (_, sdf, rotation, translation, scales) in enumerate(fields):
        if field is not None and number != field: continue
        point = targets
        if rotation is not None:
            point = np.einsum('nji,nj->ni', rotation[frames], targets-translation[frames])
        if scales is not None: point = point/scales[frames,None]
        value, direction = sdf.query(point, gradient=True)
        value = np.asarray(value)
        if scales is not None:value=value*scales[frames]
        if rotation is not None:direction=np.einsum('nij,nj->ni',rotation[frames],direction)
        length = np.linalg.norm(direction,axis=1)
        own = np.where(np.isfinite(value), np.maximum(-value,0.),0.)
        selected = (own > depth) & np.isfinite(direction).all(axis=1) & (length > 1e-12)
        depth[selected] = own[selected]
        normal[selected] = direction[selected]/length[selected,None]
    return depth, normal


def chain_columns(objective, body):
    """Articulated ancestors of a world effector; no anatomical whitelist."""
    model, cols = objective.model, []
    while body > 0:
        for joint in range(model.body_jntadr[body], model.body_jntadr[body]+model.body_jntnum[body]):
            qi = model.jnt_qposadr[joint]
            if qi in objective.coords.qi:
                cols.append(objective.coords.inverse[int(model.jnt_dofadr[joint])])
        body = int(model.body_parentid[body])
    return cols


class SwingRows:
    def __init__(self, objective, *, tangent_sigma=.003, normal_sigma=.02, include_contacts=False,
                 contact_tangent_sigma=None):
        sigmas=(tangent_sigma,normal_sigma)+(() if contact_tangent_sigma is None else (contact_tangent_sigma,))
        if any(not np.isfinite(s) or s <= 0 for s in sigmas):
            raise ValueError('Recovery sigmas must be finite and positive')
        self.objective, self.groups = objective, []
        self.movable = np.zeros((objective.frames,objective.width),bool)
        self.report = {}
        for eff in objective.effectors:
            spec=objective.tasks.effectors[eff['name']]
            task=spec['task']
            persistent = spec.get('preserve_world_target',False) or getattr(task,'preserve_world_tracking',False)
            if not persistent:continue
            policy=getattr(objective.tasks,'contact_phase',None)
            roles = None if policy is None else policy.roles.get(eff['name'])
            observed=(np.zeros(objective.frames,bool) if task.contact_mask is None
                      else np.asarray(task.contact_mask,bool).copy())
            if roles is not None:observed &= np.asarray(roles)!='free'
            free = np.ones(objective.frames,bool)
            if not include_contacts:
                free &= ~observed
            frames = np.flatnonzero(free)
            if not len(frames):continue
            original = eff['tracking'][frames]
            depth,normal = target_yield(objective,original,frames)
            normal = np.where((depth>0)[:,None],normal,eff['normal'][frames])
            normal /= np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
            tangent = np.eye(3)[None]-normal[:,:,None]*normal[:,None,:]
            sigma=np.full(len(frames),tangent_sigma)
            if contact_tangent_sigma is not None:sigma[observed[frames]]=contact_tangent_sigma
            projection = tangent/sigma[:,None,None] + normal[:,:,None]*normal[:,None,:]/normal_sigma
            local=np.broadcast_to(spec['tracking_offset'],(objective.frames,1,3))
            group = dict(kind='body', indices=np.array([eff['body']]), bodies=np.array([eff['body']]),
                         local=local, frames=frames, target=original+depth[:,None]*normal,
                         original=original, normal=normal, projection=projection, source_depth=depth,
                         name=eff['name'])
            self.groups.append(group)
            columns = chain_columns(objective,eff['body'])
            # A few adjacent frames let exact difference stencils distribute
            # the correction across a hand-off. Source timing is unchanged.
            nearby = (np.unique(np.clip(frames[:,None]+np.arange(-6,7),0,objective.frames-1))
                      if include_contacts else frames)
            self.movable[np.ix_(nearby,columns)] = True
            self.report[eff['name']] = dict(world_target_frames=len(frames), source_inside_frames=int((depth>0).sum()),
                source_depth_max_m=float(depth.max(initial=0.)), chain_columns=columns)

    def evaluate(self, poses, jacobian=True):
        obj = self.objective
        kin = obj.window_kinematics(poses,range(len(poses)),with_jacobian=jacobian)
        residuals,matrices,stats=[],[],{}
        for group in self.groups:
            frames = group['frames']; samples=np.zeros(len(frames),int)
            point,derivative=SupportRows.points(kin,group,frames,samples,jacobian)
            error=point-group['target']
            residuals.append(np.einsum('nij,nj->ni',group['projection'],error).ravel())
            if jacobian:
                matrices.append(_matrix(np.matmul(group['projection'],derivative),frames,obj.width,obj.frames))
            raw=point-group['original']; distance=np.linalg.norm(raw,axis=1)
            f=int(frames[np.argmax(distance)])
            stats[group['name']]=dict(mean_m=float(distance.mean()),p95_m=float(np.quantile(distance,.95)),
                max_m=float(distance.max()),worst_frame=f,
                yielded_normal_gap_max_m=float(np.max(np.einsum('ni,ni->n',raw,group['normal']),initial=0.)))
        residual=np.concatenate(residuals) if residuals else np.empty(0)
        matrix=sparse.vstack(matrices,format='csr') if matrices else sparse.csr_matrix((len(residual),obj.frames*obj.width))
        return residual,matrix,stats
