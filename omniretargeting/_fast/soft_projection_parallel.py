"""Private independent physical projection regions, deterministic parent overlay."""
import multiprocessing
import time

from threadpoolctl import threadpool_limits

from ..cpu_budget import available_cpus
from .window_driver import fork_status

_OWNER = _POSES = _TARGET = _BASE = None


def independent(plans):
    """Merge windows sharing any three-frame stencil, including fixed ends."""
    result = []
    for frames in plans:
        if result and frames.start-result[-1].stop <= 6:
            result[-1] = range(result[-1].start, frames.stop)
        else:
            result.append(frames)
    return result


def _worker(frames):
    from .soft_projection import project_region
    _OWNER.pool = _OWNER.worker_output = None
    _OWNER.evaluation_backend = 'cpu'
    report = {name:value for name,value in _BASE.items() if name in
        ('_metric','_quadratic','objective_aware','physical_majorizer','linear_physical_qp','physical_first_order')}
    poses, record = project_region(_OWNER, _POSES, _TARGET, frames, report=report)
    return poses[frames.start:frames.stop], record, {
        name:value for name,value in report.items() if not name.startswith('_')}


def project_parallel(objective, poses, target, plans, *, report):
    from .soft_projection import project_region
    global _OWNER, _POSES, _TARGET, _BASE
    started = time.monotonic()
    plans = independent(plans)
    running = getattr(objective, 'pool', None) is not None
    objective.stop_workers()
    safe, reason = fork_status()
    workers = min(8, max(1, available_cpus()//2), len(plans))
    report.update(projection_workers=workers if safe else 0, projection_fork_reason=reason)
    try:
        if workers < 2 or not safe:
            for frames in plans:
                poses, record = project_region(objective, poses, target, frames, report=report)
                report['regions'].append(record)
        else:
            objective.prepare_window_evaluation()
            _OWNER, _POSES, _TARGET, _BASE = objective, poses, target, report
            with threadpool_limits(limits=1), multiprocessing.get_context('fork').Pool(workers) as pool:
                outputs = pool.map(_worker, plans, chunksize=1)
            # No proposal observes another job's private iterate. Disjoint
            # stencil supports make this overlay independent of completion order.
            poses = poses.copy()
            for frames, (part, record, diagnostics) in zip(plans, outputs):
                poses[frames.start:frames.stop] = part
                report['regions'].append(record)
                for key in ('qp_solves','qp_seconds','physical_model_seconds',
                            'physical_model_evaluations','physical_value_seconds','physical_value_evaluations',
                            'linear_qp_solves','linear_qp_seconds'):
                    report[key] = report.get(key,0)+diagnostics.get(key,0)
                for key in ('qp_statuses','qp_diagnostics','linear_qp_diagnostics','linear_qp_fallbacks'):
                    report.setdefault(key,[]).extend(diagnostics.get(key,[]))
    finally:
        _OWNER = _POSES = _TARGET = _BASE = None
        if running:
            objective.start_workers()
    report.pop('_physical_round',None)
    report.update(geometry_seconds=time.monotonic()-started,
                  geometry_projection='independent_physical_regions')
    return poses
