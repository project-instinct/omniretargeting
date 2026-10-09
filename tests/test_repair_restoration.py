"""Curved candidate restoration retains every original joint/contact budget."""

import numpy as np

from omniretargeting._fast.mesh_repair import (
    OriginalGeometry,
    _joint_projection,
    _qp_step,
)
from omniretargeting._fast.repair_limits import RepairLimits
from omniretargeting._fast.repair_restoration import restore_repair
from omniretargeting._fast.reference import ReferenceShape
from tests.test_limb_repair_scope import _floor_problem
from tests.test_fast_backend import _objective
from tests.test_repair_guard_rows import violation


def test_source_shape_rows_preserve_both_frozen_references():
    objective, poses, _ = _objective(temporal=True)
    shape = ReferenceShape(objective, poses)
    frozen = poses.copy()
    frozen[:, objective.coords.qi[-1]] += 0.03
    shape.freeze_reference(frozen)
    assert (
        violation(
            shape.repair_bounds(poses), np.zeros(objective.frames * objective.width)
        )
        <= 1e-10
    )
    # A large arm change keeps the layout of both references: no bound.
    turned = poses.copy()
    turned[:, objective.coords.qi[-1]] -= 0.6
    step = objective.coords.difference(poses, turned).ravel()
    assert shape.accepts(turned)
    assert violation(shape.repair_bounds(poses), step) <= 1e-10
    # A sustained cross-limb hand/leg flip violates the current layout contract.
    # Same-leg hip/knee folding is not a layout relation.
    candidate = poses.copy()
    candidate[:, objective.coords.qi[-1]] += 1.2
    step = objective.coords.difference(poses, candidate).ravel()
    assert not shape.accepts(candidate)
    assert violation(shape.repair_bounds(poses), step) > 0.01
    np.testing.assert_array_equal(shape.seed, poses)


def test_invalid_candidate_cannot_expand_its_temporal_linearization_allowance():
    objective, poses, _ = _objective(temporal=True)
    limits = RepairLimits(objective, poses)
    candidate = poses.copy()
    candidate[10, objective.coords.qi[-1]] += 0.03  # smooth local correction
    assert limits.check(candidate)[0]
    assert limits.check(candidate)[1]['temporal_excess'] == []


def test_restoration_fixes_small_nonlinear_overshoot_without_undoing_leg_progress():
    objective, poses, plane, scope = _floor_problem(0.12)
    limits = RepairLimits(objective, poses)
    limits.movable_body = objective.model.body("tip1").id
    geometry = OriginalGeometry(objective, (), plane, manifold=True)
    measured = geometry.measure(poses, jacobians=True)
    step, _ = _qp_step(limits, poses, geometry, measured, measured)
    assert step is not None
    candidate = poses.copy()
    candidate[:, limits.coords.qi] += 1.1 * step
    assert not limits.check(candidate)[0]
    restored, report = restore_repair(
        limits,
        poses,
        candidate,
        geometry,
        measured,
        measured,
        _joint_projection(limits, objective.model, measured),
    )
    assert restored is not None, report
    assert limits.check(restored)[0]
    fixed = np.ones(objective.model.nq, bool)
    fixed[limits.coords.qi[sorted(scope.columns)]] = False
    np.testing.assert_array_equal(restored[:, fixed], poses[:, fixed])
    geom = objective.model.geom("sole1").id
    before = max(v for k, v in measured["depths"].items() if k[1:] == ("static", geom))
    checked = geometry.measure(restored)
    after = max(v for k, v in checked["depths"].items() if k[1:] == ("static", geom))
    assert after < before - 1e-5
    np.testing.assert_array_equal(limits.entry, poses)
