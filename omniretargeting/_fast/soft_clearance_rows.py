"""Private recovery constraints on the acquisition geometry's own depths.

Every scene/object shell and eligible original self pair participates. This
does not replace the final original-mesh audit. Limits are frozen at entry
and tightened only after accepted motion; a rejected trial earns no credit.
"""
import time
import numpy as np
from .barrier_rows import scalar
from .batched_self_collision import BatchedSelfCollision
from .bulk_geometry import prepare
from .kinematics import cross_columns
from .soft_support_rows import _matrix


class ClearanceRows:
    def __init__(self, objective, poses, report):
        self.objective, self.report = objective, report
        self.query = (None if objective.self_collision is None else
                      BatchedSelfCollision(objective.self_collision, 12))
        self.limits = {}
        measured, _ = self.measure(poses, False)
        self.commit(measured)

    def commit(self, measured):
        for key, value in measured.items():
            depth = max(-value, 0.)
            self.limits[key] = min(self.limits.get(key, depth), depth)

    def measure(self, poses, jacobian, *, scope=None):
        tick = time.monotonic()
        owner = self.objective
        scope = range(len(poses)) if scope is None else scope
        kin = owner.window_kinematics(poses, scope, with_jacobian=jacobian)
        obj = owner if len(scope)==len(poses) else owner.for_frames(slice(scope.start,scope.stop),kinematics=kin)
        offset=scope.start
        # The shell helpers use objective.kin. Reuse this exact immutable FK.
        previous = obj.kin
        obj.kin = kin
        values, families = {}, []
        try:
            prepare(obj)
            geoms = np.asarray(obj.shell_geoms)
            for field, (_, sdf, rotation, translation, scales) in enumerate(obj._clearance_fields()):
                clearance, frames, samples, points, local = obj._deepest_samples(sdf, rotation, translation, scales)
                keep = np.isfinite(clearance[frames, samples])
                frames, samples, points, local = frames[keep], samples[keep], points[keep], local[keep]
                phi = clearance[frames, samples]
                keys = [('field', int(field), int(f)+offset, int(geoms[g])) for f,g in zip(frames,samples)]
                derivative = None
                if jacobian and len(frames):
                    direction = getattr(sdf, 'value_gradient', sdf.gradient)(local)
                    if rotation is not None:direction=np.einsum('nij,nj->ni',rotation[frames],direction)
                    bodies = obj.model.geom_bodyid[geoms[samples]]
                    jp = kin.jacp[frames,bodies]-cross_columns(points-kin.xpos[frames,bodies],kin.jacr[frames,bodies])
                    derivative = np.einsum('ni,niw->nw',direction,jp)
                self._append(values,families,keys,phi,frames+offset,derivative,('RecoveryClearance','field',field))
            clearance, points = obj._plane_clearance()
            frames, samples = np.nonzero(np.isfinite(clearance))
            keys = [('plane', int(f)+offset, int(geoms[g])) for f,g in zip(frames,samples)]
            derivative = None
            if jacobian and len(frames):
                bodies=obj.model.geom_bodyid[geoms[samples]]
                jp=kin.jacp[frames,bodies]-cross_columns(points[frames,samples]-kin.xpos[frames,bodies],kin.jacr[frames,bodies])
                derivative=np.einsum('ni,niw->nw',obj.plane_normal[frames,samples],jp)
            self._append(values,families,keys,clearance[frames,samples],frames+offset,derivative,('RecoveryClearance','plane'))
            if self.query is not None:
                batch, unknown = self.query.rows(poses[scope.start:scope.stop],kin,jacobian)
                if unknown:raise ValueError('Unknown self geometry cannot certify recovery clearance')
                keys, phi, frames, derivative = [], [], [], []
                for frame, rows in enumerate(batch):
                    for row in rows:
                        keys.append(('self',frame+offset,*map(int,row['pair'])))
                        phi.append(row['clearance']);frames.append(frame+offset)
                        if jacobian:derivative.append(row['_jacobian'][obj.coords.cols])
                self._append(values,families,keys,np.asarray(phi),np.asarray(frames),
                    np.asarray(derivative) if jacobian else None,('RecoveryClearance','self'))
        finally:
            obj.kin = previous
            self.report['clearance_query_seconds'] = self.report.get('clearance_query_seconds',0.)+time.monotonic()-tick
        return values, families

    def _append(self, values, families, keys, phi, frames, derivative, label):
        if not len(keys):return
        values.update(zip(keys,map(float,phi)))
        slack=phi+np.asarray([self.limits.get(key,0.) for key in keys])
        matrix = None if derivative is None else _matrix(derivative,frames,self.objective.width,self.objective.frames)
        families.append(scalar(slack,matrix,label,keys))

    def check(self, values):
        # This tiny absolute allowance only covers floating-point geometry
        # reduction. No existing guard tolerance or physical budget changes.
        return [(key,float(-value-self.limits.get(key,0.))) for key,value in values.items()
                if value < -self.limits.get(key,0.)-1e-10]

    @staticmethod
    def maximum_faces(values, families):
        """Proposal cuts for the already enforced maximum-depth line check.

        Each geometry family keeps its current measured maximum; this does
        not change a guard, reference, or physical allowance. Nonlinear
        evaluations still own acceptance. Unlike freezing every material
        distance, these cuts allow clearance to redistribute below that max.
        """
        from dataclasses import replace
        result=[]
        for family in families:
            phi=np.asarray([values[key] for key in family.locations])
            maximum=max(float(-phi.min()),0.)
            result.append(replace(family,slack=phi+maximum))
        return result
