"""Feasible active-set KKT polish: add only faces reached by a primal step."""
import numpy as np
from .constrained_active import equality_step, normalized_rows, ActiveSetFailure

MAX_PRIMAL_SETS = 160


def solve_primal(system, gradient, matrix, bounds, initial, diagnostics):
    matrix,bounds,valid=normalized_rows(matrix,bounds,system[0])
    if np.any(bounds[valid]<-1e-10):
        raise ActiveSetFailure('primal initialization has no feasible origin')
    direction=np.asarray(initial).copy()
    value=matrix@direction
    outward=valid & (value>bounds)
    fraction=min(1.,float(np.min(np.maximum(bounds[outward],0.)/value[outward],initial=1.)))
    direction*=fraction
    active=valid & (np.abs(matrix@direction-bounds)<1e-10)
    multipliers=np.zeros(len(bounds))
    diagnostics['primal_initial_fraction']=fraction
    for number in range(MAX_PRIMAL_SETS):
        candidate,dual=equality_step(system,gradient,matrix[active],bounds[active],
                                    multipliers[active],diagnostics)
        diagnostics['working_sets']+=1
        if not np.isfinite(candidate).all() or not np.isfinite(dual).all():
            raise ActiveSetFailure('nonfinite primal KKT')
        change=candidate-direction
        slope=matrix@change
        room=bounds-matrix@direction
        blocking=(~active)&valid&(slope>1e-12)&(slope>room+1e-9)
        if blocking.any():
            ratios=np.full(len(bounds),np.inf)
            ratios[blocking]=np.maximum(room[blocking],0.)/slope[blocking]
            alpha=float(ratios.min())
            direction+=alpha*change
            hit=blocking&(ratios<=alpha+1e-12)
            active[hit]=True
            diagnostics['added_rows']+=int(hit.sum())
            multipliers[:]=0.
            continue
        direction=candidate
        multipliers[:]=0.;multipliers[active]=dual
        drop=active&(multipliers < -1e-8)
        if drop.any():
            # Release the most negative dual first; batch release can undo
            # a feasible warm set and rediscover the same faces repeatedly.
            index=int(np.argmin(np.where(drop,multipliers,np.inf)))
            active[index]=False
            diagnostics['dropped_rows']+=1
            continue
        excess=float((matrix@direction-bounds)[valid].max(initial=0.))
        if excess>1e-8:
            raise ActiveSetFailure('primal KKT residual')
        diagnostics.update(primal_sets=number+1,active_rows=int(active.sum()),
                           max_normalized_excess=excess)
        return direction
    raise ActiveSetFailure('primal working-set budget exhausted')
