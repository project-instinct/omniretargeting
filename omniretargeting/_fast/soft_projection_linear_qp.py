"""Physical projection QP with exact linear checks and supporting SOC cuts."""
import time
import numpy as np
import osqp
from scipy import sparse

from .soft_projection_qp import ProjectionFailure, closest


def ball_faces(balls, point):
    matrices, limits = [], []
    for jac, residual, radius in balls:
        residual = np.asarray(residual)
        value = residual+(jac@point).reshape(residual.shape)
        size = np.linalg.norm(value,axis=1)
        keep = size > 0.
        if not keep.any():
            continue
        unit = value[keep]/size[keep,None]
        dim = value.shape[1]
        indices = (np.flatnonzero(keep)[:,None]*dim+np.arange(dim)).ravel()
        operator = sparse.csr_matrix((unit.ravel(),
            (np.repeat(np.arange(len(unit)),dim),np.arange(unit.size))),shape=(len(unit),unit.size))
        matrices.append(operator@jac[indices])
        limits.append(np.asarray(radius)[keep]-np.einsum('ij,ij->i',unit,residual[keep]))
    return matrices, limits


def solve(target, matrix, lower, upper, *, balls=(), quadratic, linear,
          report, **kwargs):
    """Retain the full GN P; use tangent faces, then check every intact ball.

    All tolerances here are numerical. Any unsuccessful cut program returns
    to the original conic solve; final original nonlinear guards still own
    acceptance. The objective and every unscaled constraint are unchanged.
    """
    tick = time.monotonic()
    matrix = sparse.csr_matrix(matrix)
    lower,upper = np.asarray(lower),np.asarray(upper)
    count = matrix.shape[1]
    scale = 1/np.sqrt(np.maximum(quadratic.diagonal(),1e-18))
    variables = sparse.diags(scale,format='csr')
    hessian = sparse.triu(variables@quadratic@variables,format='csc')
    gradient = scale*np.asarray(linear)
    faces,limits = ball_faces(balls,np.zeros(count))
    try:
        warm = None
        for cutting in range(6):
            program = sparse.vstack([matrix,*faces],format='csr')
            lo = np.r_[lower,np.full(sum(map(len,limits)),-np.inf)]
            hi = np.r_[upper,*limits]
            scaled = program@variables
            units = np.maximum(np.asarray(abs(scaled).max(axis=1).toarray()).ravel(),1e-12)
            solver = osqp.OSQP()
            solver.setup(P=hessian,q=gradient,A=(sparse.diags(1/units)@scaled).tocsc(),
                l=lo/units,u=hi/units,verbose=False,eps_abs=1e-7,eps_rel=1e-8,
                max_iter=4000,polishing=True,check_termination=25)
            if warm is not None:
                solver.warm_start(x=warm)
            result = solver.solve(raise_error=False)
            point = scale*result.x
            values = program@point
            excess = float(np.maximum(lo-values,values-hi).max(initial=0.))
            ball_excess = max((float((np.linalg.norm(res+(jac@point).reshape(res.shape),axis=1)-radius).max(initial=0.))
                for jac,res,radius in balls),default=0.)
            report['linear_qp_solves'] = report.get('linear_qp_solves',0)+1
            report.setdefault('linear_qp_diagnostics',[]).append(dict(status=result.info.status,
                iterations=result.info.iter,linear_excess=excess,ball_excess=ball_excess))
            if not np.isfinite(point).all() or excess > 1e-8:
                raise ProjectionFailure('linear_qp_original_residual')
            if ball_excess <= 1e-8:
                report['linear_qp_seconds'] = report.get('linear_qp_seconds',0.)+time.monotonic()-tick
                return point
            more,bounds = ball_faces(balls,point)
            faces.extend(more);limits.extend(bounds);warm=result.x
        raise ProjectionFailure('linear_qp_ball_cuts')
    except ProjectionFailure as error:
        report.setdefault('linear_qp_fallbacks',[]).append(str(error))
        report['linear_qp_seconds'] = report.get('linear_qp_seconds',0.)+time.monotonic()-tick
        return closest(target,matrix,lower,upper,balls=balls,quadratic=quadratic,
                       linear=linear,report=report,**kwargs)
