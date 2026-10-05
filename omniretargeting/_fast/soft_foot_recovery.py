"""Joint-first world-effector recovery with original exact final predicates."""
import time
import numpy as np
from scipy import sparse
from .barrier_assembly import physical_system
from .barrier_solver import _equalities
from .constrained_ip import QuadraticDomain, solve
from .constrained_rows import ConstrainedRows
from .constrained_step import step_boxes
from .legacy_window_qp import WindowObjective
from .soft_acquisition_v4 import final_guards
from .soft_bulk import bulk_context
from .soft_projection import certify
from .soft_support_projection import ContactProjection
from .soft_support_rows import SupportRows, SupportWeights
from .soft_swing_rows import SwingRows
from .solver import _hold


def contact_quality(rows, poses):
    """Per observed body, using fixed masks/targets, including role hand-offs."""
    obj=rows.objective;kin=obj.window_kinematics(poses,range(len(poses)),with_jacobian=False)
    result={}
    for number,g in enumerate(rows.groups):
        frames,samples=np.indices(g['active'].shape)
        point,_=rows.points(kin,g,frames.ravel(),samples.ravel(),False)
        error=point.reshape(*g['active'].shape,3)-g['target']
        normal=np.abs(np.sum(error*g['normal'],axis=-1));vector=np.linalg.norm(error,axis=-1)
        velocity=np.linalg.norm(np.diff(error,axis=0)*obj.framerate,axis=-1)
        moving=g['active'][1:]&g['active'][:-1]&g['sticking'][1:]&g['sticking'][:-1]&(g['ids'][1:]==g['ids'][:-1])
        moving &= (g['roles'][1:]=='support') | (g['roles'][:-1]=='support')
        for body in np.unique(g['bodies']):
            selected=g['bodies']==body
            valid=g['active']&selected[None]&(g['roles']=='support')
            if not valid.any():continue
            values=[]
            for metric in (normal,vector):
                x=metric[valid];values.extend([x.mean(),np.quantile(x,.95),x.max()])
            x=velocity[moving&selected[None]]
            if len(x):values.extend([np.quantile(x,.95),x.max()])
            result[number,int(body)]=np.asarray(values)
    return result


def restore_clearance(objective, trial, clearance, held, diagonal, diagnostics, *, rounds=4):
    """Minimum-change second-order correction of nonlinear clearance faces.

    This private correction includes every original continuity/layout row.
    It returns only a certified point; failed restoration leaves the caller's
    accepted state and the clearance ledger untouched.
    """
    from .constrained_active import ActiveSetFailure
    from .soft_projection import regions
    from scipy.ndimage import maximum_filter1d
    from dataclasses import replace
    measured,_=clearance.measure(trial,False)
    bad=np.zeros(len(trial),bool)
    for key,_ in clearance.check(measured):bad[key[2] if key[0]=='field' else key[1]]=True
    scopes=regions(maximum_filter1d(bad,size=17,mode='constant'))
    records=[]
    for scope in scopes:
      span=slice(scope.start,scope.stop)
      columns=slice(scope.start*objective.width,scope.stop*objective.width)
      local_held=held[(held>=columns.start)&(held<columns.stop)]-columns.start
      metric=diagonal[columns]
      curvature={}
      predicted={}
      for number in range(rounds+1):
        measured, extra=clearance.measure(trial,True,scope=scope)
        for key,value in measured.items():
            if key in predicted:
                error=max(predicted[key]-value,0.)
                curvature[key]=max(curvature.get(key,0.),1.5*error)
        regressions=clearance.check(measured)
        window=WindowObjective(objective,trial,scope)
        rows=ConstrainedRows(window)
        current=trial[span]
        from .legacy_window_qp import guard_failures
        from .barrier_solver import _domains
        failures=guard_failures(window,current)
        if not _domains(window,current):failures.append('root_or_box_domains')
        records.append(dict(frames=[scope.start,scope.stop],round=number,violations=len(regressions),failures=failures,
                            max_excess=max((v for _,v in regressions),default=0.)))
        if not regressions and not failures:break
        if number==rounds:return None,None,records
        lower,upper=step_boxes(window,current)
        system=np.zeros((min(4*objective.width,len(metric)),len(metric)),order='F')
        # This minimum-change model has zero linear term. Normalize its
        # arbitrary objective scale before infeasible-start KKT: otherwise
        # the physical 1e9-scale metric and the initial 1e-3 duals produce
        # vanishing restoration steps. A common scale preserves the minimizer.
        system[0]=metric/max(float(metric.max()),1.)
        _hold(system,local_held)
        try:
            local_extra=[replace(f,matrix=f.matrix[:,columns],
                slack=f.slack-np.asarray([curvature.get(key,0.) for key in f.locations])) for f in extra]
            domain=QuadraticDomain(rows.evaluate(current,True)+local_extra,lower,upper,local_held)
            step=solve(system,np.zeros(len(metric)),domain,{},diagnostics,restoration=True)
            if diagnostics.get('ip_excess',0.)>1e-8:
                records[-1]['failure']='inexact_restoration_qp';break
        except (ActiveSetFailure,np.linalg.LinAlgError) as error:
            records[-1]['failure']=str(error);break
        predicted={key:measured[key]+change for f in extra
                   for key,change in zip(f.locations,f.matrix[:,columns]@step)}
        trial=window.embed(window.project(objective.coords.integrate(current,step.reshape(len(scope),-1),1.)))
    measured,_=clearance.measure(trial,False)
    if clearance.check(measured) or certify(objective,trial):return None,None,records
    return trial,measured,records


def recover(objective, poses, *, iterations=4, include_contacts=True,
            protect_clearance=False, preserve_contacts=True, all_joints=False,
            hold_failed_regions=False, weights=None, clearance_ceiling=False,
            hold_binding_frames=False, contact_tangent_sigma=None, root_last=False, root_damping=10.,
            private_callback=None):
    tick=time.monotonic()
    report=dict(mode='joint_first_world_tracking_recovery', evaluations=0,evaluation_seconds=0.,
                factorizations=0,band_solves=0,history=[],root_path='held',conic_calls=0)
    report['preserve_ranked_support_metrics']=preserve_contacts
    report['protect_individual_clearance']=protect_clearance
    if not np.isfinite(root_damping) or root_damping<0:
        raise ValueError('Root damping must be finite and nonnegative')
    recovery=SwingRows(objective,include_contacts=include_contacts,contact_tangent_sigma=contact_tangent_sigma)
    contacts=SupportRows(objective)
    report['target_yield']=recovery.report
    quality=contact_quality(contacts,poses)
    dependency=ContactProjection(objective,poses,np.ones((len(poses),objective.width)))
    movable=recovery.movable.copy()
    if all_joints:movable[:,objective.coords.scalar_cols]=True
    if root_last:
        # Joint-only recovery has already run. This numerical damping keeps
        # root motion expensive while its ORIGINAL balls and both continuity
        # families remain in the coupled QP and exact line certificate.
        movable[:,:]=True
        report['root_path']='coupled_last_with_original_domains'
        report['root_damping']=root_damping
    else:
        movable[:,np.asarray(objective.coords.scalar_cols)[~dependency.enabled]]=False
    held=np.flatnonzero(~movable.ravel())
    warm={}
    clearance_rows=None
    if protect_clearance or clearance_ceiling:
        from .soft_clearance_rows import ClearanceRows
        clearance_rows=ClearanceRows(objective,poses,report)
        report['clearance_constraints']=('entry_depths_tightened_on_acceptance_all_geoms_and_pairs'
            if protect_clearance else 'proposal_cuts_for_exact_maximum_depth_line_check')
    weights=weights or SupportWeights(support_velocity_sigma=.0006)
    with bulk_context(objective,poses,report,clearance_multiplier=4.,support_weights=weights,
                      source_yield=True) as physical, final_guards(objective):
        evaluation=physical.evaluate(poses)
        report['before_cost']=physical.physical_cost(evaluation)
        report['before_stats']=evaluation[3]
        report['before_world_targets']=recovery.evaluate(poses,False)[2]
        for number in range(iterations):
            acc,residual,temporal,stats,_=evaluation
            rr,rj,world=recovery.evaluate(poses)
            combined=sparse.vstack([temporal,rj],format='csr')
            system=physical_system(acc,combined)
            gradient=acc.gradient.ravel()+temporal.T@residual+rj.T@rr
            system[0]+=.001*np.maximum(system[0],1.)
            if root_last:
                root_columns=np.r_[np.arange(6) if objective.coords.root_q is not None else [],
                    np.asarray(objective.coords.scalar_cols)[~dependency.enabled]].astype(int)
                root_indices=(np.arange(len(poses))[:,None]*objective.width+root_columns).ravel()
                system[0,root_indices]+=root_damping*np.maximum(system[0,root_indices],1.)
            window=WindowObjective(objective,poses,range(len(poses)));rows=ConstrainedRows(window)
            lower,upper=step_boxes(window,poses)
            held=np.unique(np.r_[held,_equalities(window,poses)])
            _hold(system,held);gradient[held]=0.
            families=rows.evaluate(poses,True)
            if clearance_rows is not None and (protect_clearance or clearance_ceiling):
                measured,extra=clearance_rows.measure(poses,True)
                if not protect_clearance:
                    extra=clearance_rows.maximum_faces(measured,extra)
                families.extend(extra)
            domain=QuadraticDomain(families,lower,upper,held)
            direction=solve(system,gradient,domain,warm,report)
            accepted=False
            blocked_scope=None
            for alpha in (1.,.5,.25,.125):
                trial=objective.project(objective.coords.integrate(poses,direction.reshape(len(poses),-1),alpha))
                failures=certify(objective,trial)
                rec=dict(iteration=number,alpha=alpha,failures=failures)
                if not failures:
                    following=contact_quality(contacts,trial)
                    contact_ok=not preserve_contacts or all(np.all(following[key]<=values+1e-9) for key,values in quality.items())
                    rec['contact_acceptance_required']=preserve_contacts
                    rec['contacts_nonregressing']=all(np.all(following[key]<=values+1e-9)
                                                     for key,values in quality.items())
                    if not contact_ok:
                        rec['contact_regressions']=[dict(group=int(key[0]),body=objective.model.body(key[1]).name,
                            baseline=values.tolist(),candidate=following[key].tolist())
                            for key,values in quality.items() if np.any(following[key]>values+1e-9)]
                    if contact_ok:
                        measured=None
                        if clearance_rows is not None and protect_clearance:
                            measured,_=clearance_rows.measure(trial,False)
                            regressions=clearance_rows.check(measured)
                            rec['clearance_regressions']=regressions[:8]
                            if regressions:
                                restored,measured,details=restore_clearance(objective,trial,clearance_rows,
                                    held,np.maximum(system[0],1.),report)
                                rec['clearance_restoration']=details
                                if restored is None:
                                    report['history'].append(rec)
                                    if hold_failed_regions and details:
                                        blocked_scope=details[-1].get('frames')
                                        if blocked_scope is not None:break
                                    continue
                                trial=restored
                                following=contact_quality(contacts,trial)
                                if preserve_contacts and any(np.any(following[key]>values+1e-9)
                                    for key,values in quality.items()):
                                    rec['restored_contacts_nonregressing']=False
                                    report['history'].append(rec)
                                    continue
                        value=physical.evaluate(trial,False);tr,_,target_stats=recovery.evaluate(trial,False)
                        clearance=all(value[3].get(k,0.)<=stats.get(k,0.) for k in
                                      ('scene_depth_m','self_depth_m','plane_depth_m','moving_depth_m'))
                        rec.update(cost=physical.physical_cost(value),stats=value[3],world_targets=target_stats,clearance_nonregressing=clearance)
                        before=physical.acceptance_cost(evaluation)+rr@rr
                        after=physical.acceptance_cost(value)+tr@tr
                        if clearance and after<before:
                            poses=trial;quality=following;evaluation=physical.evaluate(poses);accepted=True
                            if measured is not None:clearance_rows.commit(measured)
                report['history'].append(rec)
                if private_callback is not None:private_callback(trial.copy(),rec)
                if accepted:break
            if not accepted:
                if hold_binding_frames and rec.get('clearance_nonregressing') is False:
                    from .soft_clearance_rows import ClearanceRows
                    probe=ClearanceRows(objective,poses,report)
                    old,_=probe.measure(poses,False);new,_=probe.measure(trial,False)
                    def group(key):return key[:2] if key[0]=='field' else key[:1]
                    limits={}
                    for key,value in old.items():limits[group(key)]=max(limits.get(group(key),0.),-value)
                    regressions=sorted(((max(-value,0.)-limits.get(group(key),0.),key)
                        for key,value in new.items() if -value>limits.get(group(key),0.)+1e-10),reverse=True)
                    frames=[]
                    for excess,key in regressions:
                        frame=key[2] if key[0]=='field' else key[1]
                        if frame not in frames:frames.append(frame)
                        if len(frames)==3:break
                    nearby=np.unique(np.clip(np.asarray(frames)[:,None]+np.arange(-3,4),0,len(poses)-1)).astype(int)
                    pinned=np.arange(objective.width) if root_last else objective.coords.scalar_cols
                    blocked=(nearby[:,None]*objective.width+pinned).ravel()
                    if len(np.setdiff1d(blocked,held)):
                        held=np.unique(np.r_[held,blocked]);warm={}
                        report.setdefault('held_binding_frames',[]).append(dict(iteration=number,
                            frames=frames,stencil_neighbours=nearby.tolist(),violations=regressions[:3]))
                        continue
                if blocked_scope is None:break
                start,stop=blocked_scope
                blocked=(np.arange(start,stop)[:,None]*objective.width+objective.coords.scalar_cols).ravel()
                held=np.unique(np.r_[held,blocked])
                report.setdefault('held_failed_regions',[]).append(dict(iteration=number,
                    frames=[start,stop],reason='exact_nonlinear_clearance_restoration_stall'))
                warm={}
    report.update(seconds=time.monotonic()-tick,after_world_targets=recovery.evaluate(poses,False)[2])
    return poses,report
