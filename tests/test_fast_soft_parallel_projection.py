"""Private parallel projections preserve crossing stencils and frozen data."""
import numpy as np
from threadpoolctl import threadpool_limits

from omniretargeting._fast.soft_projection import project_region
from omniretargeting._fast.soft_projection_parallel import independent, project_parallel
from tests.test_fast_legacy_window import problem


def test_merge_prevents_shared_velocity_acceleration_or_jerk_support():
    plans = independent([range(1, 5), range(11, 14), range(22, 26)])
    assert plans == [range(1, 14), range(22, 26)]
    # Enlarging either support by three frames cannot overlap its neighbour.
    for a, b in zip(plans, plans[1:]):
        assert a.stop+3 < b.start-3


def test_forked_physical_regions_equal_serial_with_nonzero_fixed_neighbours():
    objective, fixed = problem(extra=True)
    desired = fixed.copy()
    plans = [range(4, 12), range(23, 31)]
    for frames in plans:
        desired[frames.start:frames.stop, 7] += .03
    reference = objective.reference.tobytes()
    bounds = [{k:v.tobytes() for k,v in g.bounds.items()}
              for g in objective.candidate_guards if hasattr(g, 'bounds')]
    serial = desired.copy()
    with threadpool_limits(limits=1):
        for frames in plans:
            serial, _ = project_region(objective, serial, desired, frames,
                                       report={'objective_aware':True})
        report = {'objective_aware':True, 'regions':[]}
        parallel = project_parallel(objective, desired, desired, plans, report=report)
    np.testing.assert_allclose(parallel, serial, atol=1e-11, rtol=1e-11)
    outside = np.ones(len(fixed),bool)
    for frames in plans:
        outside[frames.start:frames.stop] = False
    np.testing.assert_array_equal(parallel[outside],desired[outside])
    assert objective.reference.tobytes() == reference
    assert bounds == [{k:v.tobytes() for k,v in g.bounds.items()}
                      for g in objective.candidate_guards if hasattr(g, 'bounds')]
    assert report['physical_model_evaluations'] > 0


def test_unsafe_fork_uses_the_same_serial_regions(monkeypatch):
    from omniretargeting._fast import soft_projection_parallel
    objective, fixed = problem(extra=True)
    monkeypatch.setattr(soft_projection_parallel, 'fork_status',lambda:(False,'test_threads'))
    report = {'regions':[]}
    projected = project_parallel(objective,fixed,fixed,[range(4,12),range(23,31)],report=report)
    np.testing.assert_array_equal(projected,fixed)
    assert report['projection_workers'] == 0
    assert report['projection_fork_reason'] == 'test_threads'
