"""The opt-in fast experiment bounds corrections; defaults keep legacy semantics."""
from types import SimpleNamespace

import pytest
import cvxpy as cp
from tests.test_fast_budget_modes import select_budget_mode


@pytest.fixture(autouse=True)
def correction_budget_enabled(monkeypatch):
    select_budget_mode(monkeypatch, True)

import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting._fast.continuity import CorrectionContinuity, audit_continuity
from omniretargeting._fast.pipeline import verify_budgets
from omniretargeting._fast.reweight import offending_frames
from omniretargeting._fast.repair_limits import RepairLimits
from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from omniretargeting._sequence.coordinates import Coordinates
from tests.test_fast_backend import _objective
from tests.test_joint_continuity import model_with_sliders
from tests.test_repair_guard_rows import violation


def test_authored_motion_leaves_room_for_a_smooth_correction_in_every_row():
    model = model_with_sliders(2)
    coords = Coordinates(model)
    fixed = np.zeros((30, 2))
    fixed[:, 0] = np.sin(np.arange(30) * .12) * .1
    candidate = fixed * 1.5
    tasks = SimpleNamespace(coords=coords)
    legacy = SequenceContinuityGuard(tasks, fixed, None, windowed=True)
    fast = SequenceContinuityGuard(tasks, fixed, None, windowed=True, correction=True)
    direct = CorrectionContinuity(coords, fixed)
    assert not legacy.accepts(candidate)
    assert fast.accepts(candidate) and direct.accepts(candidate)
    for key in legacy.bounds:
        np.testing.assert_array_equal(fast.bounds[key], legacy.bounds[key])
    flat = cp.Variable(candidate.size)
    flat.value = (candidate - fixed).ravel()
    for guard in (fast, direct):
        assert all(np.max(c.violation()) < 1e-9 for c in guard.constraints(fixed, flat))
        assert violation(guard.repair_bounds(fixed), flat.value) < 1e-9
        assert not guard.accepts(fixed * 2.2)
        noisy = candidate.copy()
        noisy[10, 1] += 1e-4
        assert not guard.accepts(noisy)  # no budget transfer to a quiet joint
    report, chatter, _ = verify_budgets(model, coords, fixed, candidate)
    assert not chatter.any() and report['order_3']['strict_violations']
    assert not any(mask.any() for mask in offending_frames(coords, fixed, candidate))
    audit = audit_continuity(coords, fixed, candidate)
    assert audit['windowed_budget_accepted'] and not audit['strict_accepted']


def test_fast_sequence_rotation_and_anchor_correction_jacobians_and_windows():
    objective, fixed, _ = _objective()
    coords = objective.coords
    rng = np.random.default_rng(94)
    fixed = coords.integrate(fixed, rng.normal(size=(len(fixed), coords.width)) * .15, 1.)
    rotations = Rotation.from_quat(fixed[:, 3:7], scalar_first=True)
    guard = SequenceContinuityGuard(objective.tasks, fixed, rotations,
                                    windowed=True, correction=True)
    poses = coords.integrate(fixed, rng.normal(size=(len(fixed), coords.width)) * .08, 1.)
    direction = rng.normal(size=(len(fixed), coords.width)) * .01
    epsilon = 1e-5
    plus = guard.signals(coords.integrate(poses, direction, epsilon))
    minus = guard.signals(coords.integrate(poses, direction, -epsilon))
    for name, (_, jacobian) in guard.signals(poses).items():
        np.testing.assert_allclose(np.einsum('fij,fj->fi', jacobian, direction),
            (plus[name][0] - minus[name][0]) / (2 * epsilon), atol=1e-9, rtol=1e-5)
    assert guard.accepts(fixed)
    signs = fixed.copy(); signs[::2, 3:7] *= -1
    assert guard.accepts(signs)
    # An edit inside a window must still check differences beginning outside it.
    window = range(10, 14)
    bad = fixed.copy(); bad[10, coords.qi[-1]] += 1.
    assert not guard.accepts(bad, window)
    step = cp.Variable(len(window) * coords.width); step.value = np.zeros(step.shape)
    assert any(np.max(c.violation()) > 1e-6 for c in guard.constraints(bad, step, window))
    # A constant WORLD rotation correction is quiet despite a changing reference.
    moved = fixed.copy()
    moved[:, 3:7] = (Rotation.from_rotvec([.07, -.03, .04]) * rotations).as_quat(scalar_first=True)
    assert guard.accepts(moved)
    assert CorrectionContinuity(coords, fixed).accepts(moved)


def test_repair_budget_uses_frozen_entry_correction_in_rows_and_acceptance():
    objective, fixed, _ = _objective()
    # Isolate temporal acceptance from unrelated contact/posture constraints.
    objective.effectors = []; objective.surfaces = []
    objective.candidate_guards = []
    coords = objective.coords
    fixed = np.repeat(fixed[:1], len(fixed), axis=0)
    fixed[:, coords.qi[0]] += .01 * np.sin(np.arange(len(fixed)) * .2)
    limits = RepairLimits(objective, fixed)
    candidate = fixed.copy()
    candidate[:, coords.qi[0]] += .005 * np.sin(np.arange(len(fixed)) * .2)
    assert limits.check(candidate)[0]
    bounds = limits.constraints(fixed)
    matrix, low, high = bounds.packed()
    step = (candidate[:, coords.qi] - fixed[:, coords.qi]).ravel()
    assert np.all(matrix @ step >= low - 1e-10)
    assert np.all(matrix @ step <= high + 1e-10)
    noisy = candidate.copy(); noisy[10, coords.qi[-1]] += .001
    assert limits.check(noisy)[1]['reason'] == 'per-joint temporal budget'
    # Calling other candidates cannot reset the fixed reference or its budget.
    assert limits.check(candidate)[0]


def test_fixed_reference_is_not_aliased_and_rotation_reweight_is_separate():
    from omniretargeting._fast.reweight import reweight_temporal_signals
    objective, original, _ = _objective(temporal=True)
    coords = objective.coords
    fixed = original.copy()
    direct = CorrectionContinuity(coords, fixed)
    rotations = Rotation.from_quat(fixed[:, 3:7], scalar_first=True)
    sequence = SequenceContinuityGuard(objective.tasks, fixed, rotations,
                                       windowed=True, correction=True)
    fixed[:, coords.qi] += 10.
    assert direct.accepts(original) and sequence.accepts(original)
    candidate = original.copy()
    relative = np.zeros((len(candidate), 3)); relative[12, 2] = .05
    candidate[:, 3:7] = (Rotation.from_rotvec(relative) * rotations).as_quat(scalar_first=True)
    report, chatter, _ = verify_budgets(objective.model, coords, original, candidate)
    assert chatter[12] and report['root_rotation_order_3']['added_chatter_violations']
    weighted = reweight_temporal_signals(objective, original, candidate, 10.)
    assert weighted['root_rotation_reweighted']
    assert not weighted['joint_columns'] and not weighted['root_translation_frames']
    assert all(np.all(w == 1.) for w in objective.joint_temporal_weight)
    assert all(np.all(w == 1.) for w in objective.root_translation_weight)
    assert any(np.any(w == 10.) for w in objective.root_rotation_weight)
