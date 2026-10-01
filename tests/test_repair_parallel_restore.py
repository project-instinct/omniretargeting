"""Overlap restoration with geometry checks without changing candidate order."""
from contextlib import closing
from types import SimpleNamespace
import os
import numpy as np
import pytest

from omniretargeting._fast.repair_candidates import candidate_steps

pytestmark = pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork speculation')


def test_early_geometry_success_discards_unused_restore_exception():
    def restore(_):
        raise ValueError('unused')
    objective = SimpleNamespace(candidate_guards=[])
    limits = SimpleNamespace(objective=objective, coords=SimpleNamespace(qi=np.array([1])),
        check=lambda q: (q[0, 1] <= .6, {'value': float(q[0, 1])}))
    with closing(candidate_steps(limits, np.zeros((1, 2)), np.ones((1, 1)), restore, {},
                                 lazy=True, parallel_restore=True)) as candidates:
        alpha, q = next(candidates)
        assert alpha == .5 and q[0, 1] == .5
    assert objective.parallel_repair_stats['launched'] == 1
    assert objective.parallel_repair_stats['consumed'] == 0
    assert objective.parallel_repair_stats['discarded'] == 1


def _actual_restoration(parallel):
    from omniretargeting._fast.mesh_repair import OriginalGeometry, _joint_projection, _qp_step
    from omniretargeting._fast.repair_limits import RepairLimits
    from omniretargeting._fast.repair_restoration import restore_repair
    from tests.test_limb_repair_scope import _floor_problem
    objective, poses, plane, _ = _floor_problem(.12)
    limits = RepairLimits(objective, poses)
    limits.movable_body = objective.model.body('tip1').id
    geometry = OriginalGeometry(objective, (), plane, manifold=True)
    measured = geometry.measure(poses, jacobians=True)
    step, _ = _qp_step(limits, poses, geometry, measured, measured)
    projection = _joint_projection(limits, objective.model, measured)
    def restore(candidate):
        return restore_repair(limits, poses, candidate, geometry, measured, measured, projection)
    record = {}; results = []
    with closing(candidate_steps(limits, poses, 1.1 * step, restore, record, lazy=True,
                                 parallel_restore=parallel)) as candidates:
        for alpha, candidate in candidates:
            # Exercise intervening original geometry calls before the fallback.
            geometry.measure(candidate)
            results.append((alpha, candidate.tobytes()))
    return results, record, getattr(objective, 'parallel_repair_stats', None)


def test_real_restoration_preserves_all_candidates_and_full_history():
    serial = _actual_restoration(False)
    parallel = _actual_restoration(True)
    assert parallel[2]['launched'] == parallel[2]['consumed'] == 1
    assert parallel[2]['discarded'] == 0
    assert serial[:2] == parallel[:2]
    assert any(alpha == 1. for alpha, _ in parallel[0])
