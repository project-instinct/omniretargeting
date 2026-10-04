"""Source initialization cannot gain a new seed budget through fit or lifting."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import runtime_options
from omniretargeting._fast.continuity import audit_continuity
from omniretargeting._fast.initialization_envelope import SeedInitialization, InitializationConflict
from omniretargeting._sequence.coordinates import Coordinates


XML = '''<mujoco><compiler angle="radian"/><worldbody><body name="root" pos="0 0 1">
<freejoint/><geom type="sphere" size=".1" mass="1"/>
<body name="link"><joint name="joint" axis="0 1 0" range="-1 1"/>
<geom type="capsule" fromto="0 0 0 0 0 -.3" size=".04" mass="1"/>
</body></body></worldbody></mujoco>'''


def seed_problem(frames=8, **locks):
    model = mujoco.MjModel.from_xml_string(XML)
    coords = Coordinates(model, **locks)
    seed = np.tile(model.qpos0, (frames, 1))
    return coords, seed, SeedInitialization(coords, seed)


def height_objective(coords, target):
    column = coords.inverse[coords.root_d + 2]
    def evaluate(poses):
        hessian = np.zeros((len(poses), coords.width, coords.width))
        hessian[:, column, column] = 1.
        gradient = np.zeros((len(poses), coords.width))
        gradient[:, column] = poses[:, coords.root_q + 2] - target
        return SimpleNamespace(hessian=hessian, gradient=gradient), np.empty(0), \
            sparse.csr_matrix((0, len(poses) * coords.width)), {}, None
    return SimpleNamespace(coords=coords, evaluate=evaluate)


def test_seed_copy_and_budgets_never_follow_a_later_pose():
    coords, seed, envelope = seed_problem()
    original = envelope.seed.copy()
    bounds = {name: value.copy() for name, value in envelope.guard.bounds.items()}
    seed[:, 7] = .5
    assert not envelope.seed.flags.writeable
    assert not envelope.guard.fixed.flags.writeable
    np.testing.assert_array_equal(envelope.seed, original)
    changed = original.copy()
    changed[:, 7] = .1
    assert envelope.check_normalized(changed, stage='constant_correction')['accepted']
    for name, value in bounds.items():
        np.testing.assert_array_equal(envelope.guard.bounds[name], value)


def test_normalization_that_creates_chatter_is_an_explicit_conflict():
    coords, seed, _ = seed_problem()
    seed[:, 7] = np.arange(8) * .1
    envelope = SeedInitialization(coords, seed)
    coords.upper[:] = .25
    clipped = coords.normalize(seed)
    with pytest.raises(InitializationConflict) as error:
        envelope.check_normalized(clipped, stage='source_prepare')
    report = error.value.report
    assert report['stage'] == 'source_prepare'
    assert not report['reference_rebased']
    assert any(item['coordinate'] == 'joint' and item['channel'] == 'joints_order_2'
               for item in report['failures'])
    np.testing.assert_array_equal(envelope.seed, seed)


def test_seed_audit_stays_absolute_with_correction_experiment(monkeypatch):
    coords, seed, envelope = seed_problem()
    seed[:, 7] = np.arange(8) * .01
    envelope = SeedInitialization(coords, seed)
    trial = seed.copy()
    trial[:, 7] *= 1.05
    from omniretargeting._fast import continuity
    monkeypatch.setattr(continuity, 'OPTIONS', replace(runtime_options.OPTIONS, correction_budget=True))
    report = envelope.audit(trial, stage='fit_vs_seed')
    assert report['budget_signal'] == 'absolute executed derivatives'
    assert report['accepted'] == envelope.guard.accepts(trial)
    assert audit_continuity(coords, seed, trial)['budget_signal'].startswith('correction')


def test_valid_height_registration_keeps_original_spsolve_values():
    coords, seed, envelope = seed_problem()
    objective = height_objective(coords, np.full(8, 1.2))
    from omniretargeting._fast.reference import register_height
    reference, reference_shift = register_height(objective, seed)
    result, shift, report = envelope.bounded_height_registration(objective, seed)
    np.testing.assert_array_equal(result, reference)
    np.testing.assert_array_equal(shift, reference_shift)
    assert report['status'] == 'unconstrained_height_certified'


def test_height_qp_respects_original_zero_derivative_budgets():
    coords, seed, envelope = seed_problem(lock_root_xy=True)
    # Later height registration has a different tangent width from source LM.
    objective = height_objective(Coordinates(coords.model), np.array([1, 1, 1.2, 1, 1, 1, 1, 1]))
    result, shift, report = envelope.bounded_height_registration(objective, seed)
    assert report['constrained'] and report['accepted']
    assert not report['request_audit']['accepted']
    assert envelope.guard.accepts(result)
    assert not envelope.guard.violation_frames(result).any()
    assert np.ptp(shift) < 2e-6
    np.testing.assert_array_equal(result[:, coords.qi], seed[:, coords.qi])
    assert abs(np.mean(shift) - .025) < 1e-5


def test_locked_height_is_unchanged():
    coords, seed, envelope = seed_problem(lock_root_z=True)
    objective = SimpleNamespace(coords=coords)
    result, shift, report = envelope.bounded_height_registration(objective, seed)
    np.testing.assert_array_equal(result, seed)
    np.testing.assert_array_equal(shift, np.zeros(8))
    assert report['status'] == 'height_locked'


def test_locked_root_height_blocks_even_a_constant_lift():
    _, seed, envelope = seed_problem(lock_root_z=True)
    shift, report = envelope.bounded_reference_lift(seed, np.full(8, .03))
    np.testing.assert_array_equal(shift, np.zeros(8))
    assert report['status'] == 'lift_limited_by_locked_height'
    assert report['binding_constraint'] == 'locked root height'
    assert report['unachieved_lift_m'] == [.03] * 8


def test_feasible_constant_lift_is_exact_and_does_not_freeze_reference():
    coords, seed, envelope = seed_problem()
    lift = np.full(8, .03)
    actual, report = envelope.bounded_reference_lift(seed, lift)
    np.testing.assert_array_equal(actual, lift)
    np.testing.assert_array_equal(envelope.seed, seed)
    assert report['status'] == 'requested_lift_certified'
    assert not report['limited_frames']
    assert not report['support_clearance_certified']


def test_requested_lift_cannot_turn_into_an_added_chatter_budget():
    coords, seed, envelope = seed_problem()
    lift = np.zeros(8)
    lift[3] = .08
    actual, report = envelope.bounded_reference_lift(seed, lift)
    assert report['status'] == 'lift_limited_by_seed'
    assert report['limited_frames'] == [3]
    assert report['unachieved_lift_m'][3] > .079
    assert report['request_audit']['failures']
    assert np.all(actual >= 0) and np.all(actual <= lift)
    result = seed.copy()
    result[:, 2] += actual
    assert envelope.guard.accepts(result)
    assert actual.max() <= 2e-7
    np.testing.assert_array_equal(envelope.seed, seed)


@pytest.mark.parametrize('operation', ['height', 'lift'])
def test_infeasible_origin_is_not_a_zero_step_certificate(operation):
    coords, seed, envelope = seed_problem()
    bad = seed.copy()
    bad[3, 7] = .01
    with pytest.raises(InitializationConflict) as error:
        if operation == 'height':
            envelope.bounded_height_registration(height_objective(coords, np.ones(8)), bad)
        else:
            envelope.bounded_reference_lift(bad, np.zeros(8))
    assert any(item['coordinate'] == 'joint' for item in error.value.report['failures'])


def test_nonfinite_initialization_is_rejected():
    _, seed, envelope = seed_problem()
    bad = seed.copy()
    bad[0, 2] = np.nan
    with pytest.raises(InitializationConflict):
        envelope.check_normalized(bad, stage='prepare')


def source_problem():
    from tests.test_fast_backend import _problem
    from omniretargeting._fast.objective import FastObjective
    model, cfg, coords, poses, tasks, rotations = _problem(frames=8)
    objective = FastObjective(tasks, cfg, reference=poses.copy(), temporal_reference=None,
                              root_rotations=rotations, framerate=30.)
    return objective, poses, SeedInitialization(coords, poses)


def test_continuity_adapter_supports_legacy_restoration_and_direct_rows():
    objective, poses, envelope = source_problem()
    envelope.attach_source_objective(objective)
    assert objective.source_shape.continuity_only
    assert objective.candidate_guards == [envelope.guard]
    assert objective.source_shape.constraints(poses, None) == []
    assert objective.source_shape.project(poses) is poses
    from omniretargeting._fast.shape_step import _ShapeRestoration
    restoration = _ShapeRestoration(objective)
    assert restoration.tasks is objective.tasks
    assert restoration.accepts(poses)
    from omniretargeting._fast.qp_assembly import prepare_model
    from omniretargeting._fast.solver import banded_system
    acc, residual, jacobian, _, _ = objective.evaluate(poses)
    gradient = acc.gradient.ravel() + jacobian.T @ residual
    system = banded_system(len(poses), objective.width, acc.hessian, jacobian)
    diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), 1e-9)
    model = prepare_model(objective, poses, system, gradient, diagonal)
    assert model.program.A.shape[0] > len(poses) * objective.width


@pytest.mark.parametrize('direct', [False, True])
def test_source_lm_accepts_only_seed_admissible_poses(monkeypatch, direct):
    objective, poses, envelope = source_problem()
    envelope.attach_source_objective(objective)
    from omniretargeting._fast import proposals, shape_step
    from omniretargeting._fast.solver import optimize
    monkeypatch.setattr(proposals, 'SPECULATE', False)
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS,
                        direct_acquisition_qp=direct, objective_soc=False, frame_step_lengths=False))
    accepted = []
    result, history = optimize(objective, poses.copy(), iterations=2,
                               callback=lambda q, row: accepted.append(q))
    assert history and accepted
    assert history[-1]['after'] < history[0]['cost']
    assert envelope.guard.accepts(result)
    for q in accepted:
        assert envelope.check_normalized(q, stage='fit_vs_seed')['accepted']


@pytest.mark.parametrize('direct', [False, True])
def test_zero_seed_budget_uses_conic_direction_instead_of_stalling(monkeypatch, direct):
    coords, seed, envelope = seed_problem()
    column = coords.scalar_cols[0]
    target = .2 + .06 * np.sin(np.arange(8))
    def evaluate(q, **kwargs):
        delta = q[:, 7] - target
        gradient = np.zeros((8, coords.width))
        gradient[:, column] = delta
        hessian = np.zeros((8, coords.width, coords.width))
        hessian[:, column, column] = 1.
        return SimpleNamespace(cost=float(delta @ delta), gradient=gradient, hessian=hessian), \
            np.empty(0), sparse.csr_matrix((0, 8 * coords.width)), {}, None
    objective = SimpleNamespace(coords=coords, frames=8, width=coords.width,
        reference=seed.copy(), tasks=SimpleNamespace(coords=coords), source_shape=None,
        candidate_guards=[], lower=coords.lower, upper=coords.upper,
        cfg=dict(angular_step_rad=.12, translation_step_m=.03), evaluate=evaluate,
        root_boundaries=lambda poses: [], start_workers=lambda: False,
        project=lambda poses: coords.normalize(poses))
    envelope.attach_source_objective(objective)
    from omniretargeting._fast import proposals
    from omniretargeting._fast.solver import optimize
    monkeypatch.setattr(proposals, 'SPECULATE', False)
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS,
                        direct_acquisition_qp=direct, objective_soc=False, frame_step_lengths=False))
    result, history = optimize(objective, seed.copy(), iterations=2)
    assert len(history) == 2
    assert all(row['step_backend'] == 'shape_conic' for row in history)
    assert history[-1]['after'] < history[0]['cost'] * .1
    assert result[:, 7].mean() > .19
    assert envelope.guard.accepts(result)
    assert np.ptp(result[:, 7]) < 2e-6
