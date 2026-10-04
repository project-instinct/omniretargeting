"""Inner-margin behavior for acquisition guard hinge rows."""

import numpy as np

import omniretargeting._fast.acquisition_guard as acquisition_module
from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from tests.test_fast_contact_guard import _acquired
from tests.test_surface_presence import _ground


def _held_guard():
    objective, poses = _acquired()
    objective.tasks.contact_phase = None
    effector = objective.effectors[0]
    effector["normal"][:] = [0., 0., 1.]
    effector["contact"] -= [.1, 0., 0.]
    _ground(objective, 0.)
    return objective, poses, AcquisitionNormalGuard(objective, poses)


def _rows_without_margin(guard, kin, frames, weight, sigma):
    """The pre-margin row construction, used as an exact oracle."""
    rows, jacs = [], []
    scale = np.asarray(weight, dtype=float)[:, None] / sigma
    for kind, indices, local, target, normal, active, established, bound, yielded in guard.groups:
        local_part = local if kind == "surface" else local[frames]
        if kind == "surface":
            points = kin.geom_points(indices, local_part)
            bodies = guard.objective.model.geom_bodyid[indices]
        else:
            bodies = np.broadcast_to(indices, local_part.shape[:2])
            points = kin.points(bodies, local_part)
        offset = points - target[frames]
        along = np.sum(offset * normal[frames], axis=-1)
        distance = np.linalg.norm(offset, axis=-1)
        held = (active & established)[frames]
        reach = (active & ~established)[frames]
        out = np.maximum(along - bound[frames] - yielded[frames], 0.)
        inward = np.maximum(-along - bound[frames], 0.)
        far = np.maximum(distance - bound[frames] - yielded[frames], 0.)
        residual = np.where(held, out + inward, np.where(reach, far, 0.))
        rows.append(residual * scale)
        jac = kin.point_jacobians(bodies, points)
        direction = np.where(
            held[..., None],
            normal[frames] * np.sign(out - inward)[..., None],
            offset / np.maximum(distance, 1e-12)[..., None],
        )
        direction = np.where((residual > 0)[..., None], direction, 0.)
        jacs.append(np.einsum("fsi,fsiw->fsw", direction, jac) * scale[..., None])
    return np.concatenate(rows, axis=1), np.concatenate(jacs, axis=1)


def test_zero_margin_reproduces_the_previous_rows_bit_for_bit(monkeypatch):
    objective, poses, guard = _held_guard()
    monkeypatch.setattr(acquisition_module, "HINGE_MARGIN_M", 0.)
    moved = poses.copy()
    moved[:, 2] += .002
    frames, weight = slice(None), np.ones(len(poses))
    objective.kin.evaluate(moved, with_jacobian=True)
    rows, jacobian = guard.hinge_rows(objective.kin, frames, weight, 1e-3, True)
    old_rows, old_jacobian = _rows_without_margin(guard, objective.kin, frames, weight, 1e-3)
    np.testing.assert_array_equal(rows, old_rows)
    np.testing.assert_array_equal(jacobian, old_jacobian)


def test_positive_margin_starts_inside_the_frozen_bound(monkeypatch):
    objective, poses, guard = _held_guard()
    margin = 5e-4
    monkeypatch.setattr(acquisition_module, "HINGE_MARGIN_M", margin)
    frames, weight = slice(None), np.ones(len(poses))

    objective.kin.evaluate(poses, with_jacobian=False)
    rows, _ = guard.hinge_rows(objective.kin, frames, weight, 1., False)
    np.testing.assert_array_equal(rows, 0.)

    inside = poses.copy()
    inside[:, 2] += .00075
    objective.kin.evaluate(inside, with_jacobian=False)
    kind, indices, local, target, normal, active, established, bound, yielded = guard.groups[0]
    points = objective.kin.points(np.broadcast_to(indices, local.shape[:2]), local)
    along = np.sum((points - target) * normal, axis=-1)
    assert np.all(along <= bound)
    assert np.any(along > bound - margin)
    rows, _ = guard.hinge_rows(objective.kin, frames, weight, 1., False)
    assert np.all(rows[active] > 0.)


def test_margin_does_not_change_guard_acceptance(monkeypatch):
    objective, poses, guard = _held_guard()
    inside = poses.copy()
    inside[:, 2] += .00075
    outside = poses.copy()
    outside[:, 2] += .0011
    candidates = (poses, inside, outside)
    results = []
    for margin in (0., 2.5e-4, 5e-4):
        monkeypatch.setattr(acquisition_module, "HINGE_MARGIN_M", margin)
        results.append(
            ([guard.accepts(candidate) for candidate in candidates],
             [guard.violation_frames(candidate).copy() for candidate in candidates])
        )
    accepted, violations = results[0]
    assert [accepted for accepted, _ in results] == [[True, True, False]] * 3
    for other_accepted, other_violations in results[1:]:
        np.testing.assert_array_equal(other_accepted, accepted)
        np.testing.assert_array_equal(other_violations, violations)
