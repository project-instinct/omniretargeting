"""Banded scalar stencil projection of an inexact coupled QP direction."""
import ctypes
from functools import lru_cache
import hashlib,os,subprocess,time
from pathlib import Path
import numpy as np
from scipy import sparse
from .rows import _merge_identical
from .constrained_active import ActiveSetFailure


@lru_cache(maxsize=1)
def library():
    source=Path(__file__).with_name('constrained_scalar_native.cpp')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:24]
    cache=Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'omniretargeting/native'
    cache.mkdir(parents=True,exist_ok=True)
    path=cache/f'scalar_{digest}.so'
    if not path.exists():
        temp=path.with_suffix(f'.{os.getpid()}.so')
        try:
            subprocess.run(['g++','-std=c++17','-O3','-ffp-contract=off','-fPIC','-shared',str(source),'-o',str(temp)],
                capture_output=True,check=True,timeout=60)
            temp.replace(path)
        finally:temp.unlink(missing_ok=True)
    lib=ctypes.CDLL(str(path))
    lib.scalar_project.argtypes=[ctypes.c_int64]*3+[ctypes.c_void_p]*8
    lib.scalar_project.restype=ctypes.c_int
    return lib


def project_one(matrix,bounds,target):
    matrix=sparse.csr_matrix(matrix,copy=True)
    matrix.sum_duplicates();matrix.sort_indices()
    matrix,_,bounds=_merge_identical(matrix,np.full(len(bounds),-np.inf),bounds)
    norms=np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
    keep=norms>1e-15
    if np.any((~keep)&(bounds<-1e-12)):
        raise ActiveSetFailure('scalar face has no free column')
    matrix=matrix[keep].multiply((1/norms[keep])[:,None]).tocsr();bounds=bounds[keep]/norms[keep]
    arrays=[np.ascontiguousarray(matrix.indptr,dtype=np.int64),np.ascontiguousarray(matrix.indices,dtype=np.int64),
        np.ascontiguousarray(matrix.data,dtype=float),np.ascontiguousarray(bounds,dtype=float),np.ascontiguousarray(target,dtype=float),
        np.zeros(len(target)),np.zeros(1,dtype=np.int64),np.zeros(1)]
    span=int(max((matrix.indices[matrix.indptr[i+1]-1]-matrix.indices[matrix.indptr[i]]
        for i in range(matrix.shape[0]) if matrix.indptr[i]<matrix.indptr[i+1]),default=0))+1
    try:
        status=library().scalar_project(len(target),len(bounds),span,*[ctypes.c_void_p(a.ctypes.data) for a in arrays])
    except (OSError,subprocess.SubprocessError) as error:
        raise ActiveSetFailure(f'scalar native unavailable: {error}') from error
    if status:raise ActiveSetFailure(f'scalar factor failure {status}')
    point=arrays[5]
    # Scalar boxes/equalities are exact holds, not numerical tolerances.
    lo=np.full(len(target),-np.inf);hi=np.full(len(target),np.inf)
    for row in np.flatnonzero(np.diff(matrix.indptr)==1):
        k=matrix.indptr[row];col=matrix.indices[k];coefficient=matrix.data[k]
        if coefficient>0:hi[col]=min(hi[col],bounds[row]/coefficient)
        else:lo[col]=max(lo[col],bounds[row]/coefficient)
    point[:]=np.clip(point,lo,hi)
    value=matrix@point
    over=value>bounds
    fraction=min(1.,float(np.min(np.maximum(bounds[over],0.)/value[over],initial=1.)))
    if fraction<1.:point*=max(0.,fraction*(1-1e-12))
    if np.max(matrix@point-bounds,initial=0.)>1e-12:
        raise ActiveSetFailure('scalar projection residual')
    return point,int(arrays[6][0]),fraction,float(arrays[7][0])


def repair(objective,matrix,bounds,direction,diagnostics):
    started=time.monotonic();result=direction.copy();width=objective.width
    matrix=sparse.csr_matrix(matrix);counts=np.diff(matrix.indptr)
    # Explicit row reductions avoid assuming an arbitrary Jacobian row order.
    minimum=np.full(len(bounds),width);maximum=np.full(len(bounds),-1)
    for i in np.flatnonzero(counts):
        cols=matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]%width
        minimum[i],maximum[i]=cols.min(),cols.max()
    for col in objective.coords.scalar_cols:
        selected=(minimum==col)&(maximum==col)
        if not selected.any():continue
        columns=np.arange(objective.frames)*width+col
        point,iterations,fraction,excess=project_one(matrix[selected][:,columns],bounds[selected],direction[columns])
        result[columns]=point
        diagnostics['scalar_factorizations']=diagnostics.get('scalar_factorizations',0)+1
        diagnostics['scalar_iterations']=diagnostics.get('scalar_iterations',0)+iterations
        diagnostics.setdefault('scalar_repairs',[]).append(dict(column=int(col),iterations=iterations,
            fraction=fraction,excess=excess,correction_norm=float(np.linalg.norm(point-direction[columns]))))
    diagnostics['scalar_seconds']=diagnostics.get('scalar_seconds',0.)+time.monotonic()-started
    return result
