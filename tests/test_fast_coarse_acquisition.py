"""Frozen half-rate inputs and authoritative full-rate transfer certificates."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast import coarse_acquisition as driver
from omniretargeting._fast import legacy_window, runtime_options
from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.coarse_problem import aggregate_bounds, make_coarse_problem
from omniretargeting._fast.coarse_transfer import (
    failure_frames, interpolate_correction, scale_transfer_correction, transfer_correction,
)
from omniretargeting._fast.legacy_window_qp import guard_failures
from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._fast.runtime_options import RuntimeOptions
from omniretargeting._sequence.coordinates import temporal_operators
from tests.test_fast_legacy_window import problem


@pytest.fixture(autouse=True)
def ordinary_options(monkeypatch):
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())


@pytest.mark.parametrize('count', [35, 36])
@pytest.mark.parametrize('order', [1, 2, 3])
def test_binomial_bounds_cover_each_original_stencil(count, order):
    signal = np.random.default_rng(4).normal(size=(count, 3))
    full = np.diff(signal, n=order, axis=0)
    coarse = np.diff(signal[::2], n=order, axis=0)
    bounds = aggregate_bounds(np.abs(full), order, len(signal[::2]))
    assert bounds.shape == coarse.shape and not bounds.flags.writeable
    assert np.all(np.abs(coarse) <= bounds + 1e-12)
    from math import comb
    exact = sum(comb(order, j) * full[2 * np.arange(len(coarse)) + j]
                for j in range(order + 1))
    np.testing.assert_allclose(exact, coarse, atol=1e-12)


def test_sampler_preserves_prepared_weights_policies_geometry_and_guards(monkeypatch):
    from omniretargeting._fast.sdf import GridSDF
    from omniretargeting.self_collision import SelfCollisionQuery
    objective, poses = problem(extra=True)
    count = len(poses)
    surface = objective.surface_tasks[0]
    # Prepared policy and weights deliberately vary at omitted frames. They
    # must be sampled, rather than normalized/inferred on a shorter timeline.
    roles = {'foot': np.where(np.arange(count) % 3, 'hold', 'free')}
    masks = {name: np.zeros(surface.active.shape, bool)
             for name in ('support', 'hold', 'control', 'follow')}
    masks['support'][::3] = surface.active[::3]
    policy = SimpleNamespace(roles=roles, clear={'foot': np.ones(count, bool)},
                             surface_roles=masks, support=masks['support'],
                             follow_surface=masks['follow'], surface=surface)
    objective.tasks.contact_phase = surface.contact_phase = policy
    surface.source_clearances = {'foot': np.arange(count, dtype=float)}
    for spec in objective.tasks.effectors.values():
        spec['task'] = replace(spec['task'], contact_phase=policy)
    objective.tasks.effectors['foot']['support_witness'] = {'clearance': np.arange(count)}
    objective.effectors[0]['role'] = np.arange(count, dtype=float) + 1
    field = GridSDF(np.zeros(3), 1., np.ones((3, 3, 3)), np.zeros((3, 3, 3, 3)))
    objective.objects = [dict(sdf=field, translations=poses[:, :3].copy(),
                             rotations=np.tile(np.eye(3), (count, 1, 1)),
                             scales=np.arange(count, dtype=float) + 1)]
    objective.self_collision = SelfCollisionQuery(objective.model)
    frozen = objective.reference.tobytes()
    bounds = {key: value.tobytes() for key, value in objective.candidate_guards[0].bounds.items()}

    def forbidden(*args, **kwargs):
        raise AssertionError('Recomputed frozen acquisition inputs')
    monkeypatch.setattr(AcquisitionNormalGuard, '__init__', forbidden)
    monkeypatch.setattr(ReferenceShape, '__init__', forbidden)
    coarse, indices = make_coarse_problem(objective)
    assert coarse.frames == 18 and coarse.framerate == 15
    assert coarse.pool is None and coarse.self_collision is not objective.self_collision
    assert coarse.tasks.data is not objective.tasks.data
    assert coarse.objects[0]['sdf'] is field
    for name in ('translations', 'rotations', 'scales'):
        np.testing.assert_array_equal(coarse.objects[0][name], objective.objects[0][name][indices])
    for name in ('targets', 'weights', 'active', 'pose_normal_targets'):
        np.testing.assert_array_equal(getattr(coarse.surface_tasks[0], name), getattr(surface, name)[indices])
    np.testing.assert_array_equal(coarse.surface_tasks[0].source_clearances['foot'], indices)
    assert coarse.surface_tasks[0].contact_phase is coarse.tasks.contact_phase
    assert coarse.tasks.contact_phase.surface is coarse.surface_tasks[0]
    for name, value in roles.items():
        np.testing.assert_array_equal(coarse.tasks.contact_phase.roles[name], value[indices])
    for name, spec in coarse.tasks.effectors.items():
        full = objective.tasks.effectors[name]['task']
        np.testing.assert_array_equal(spec['task'].tracking_positions, full.tracking_positions[indices])
        for actual, original in zip(spec['task'].approach, full.approach):
            np.testing.assert_array_equal(actual, original[indices])
    np.testing.assert_array_equal(coarse.tasks.effectors['foot']['support_witness']['clearance'], indices)
    np.testing.assert_array_equal(coarse.effectors[0]['role'], objective.effectors[0]['role'][indices])
    normal, full_normal = coarse.hinge_guards[-1], objective.candidate_guards[-1]
    for group, full_group in zip(normal.groups, full_normal.groups):
        for value, original in zip(group[3:], full_group[3:]):
            np.testing.assert_array_equal(value, original[indices])
    for envelope, full in zip(coarse.source_shape.envelopes, objective.source_shape.envelopes):
        for name in ('reference', 'whole_scale', 'part_scale', 'soft_scale', 'axes', 'layout_sign'):
            np.testing.assert_array_equal(getattr(envelope, name), getattr(full, name)[indices])
    assert not guard_failures(coarse, poses[indices])
    assert objective.reference.tobytes() == frozen
    assert all(value.tobytes() == bounds[key] for key, value in objective.candidate_guards[0].bounds.items())


def test_coarse_per_frame_terms_equal_full_rate_at_sampled_frames():
    objective, poses = problem(extra=True)
    objective.envelope_weight[::3] *= 7
    coarse, indices = make_coarse_problem(objective)
    moved = objective.coords.integrate(poses, np.random.default_rng(3).normal(
        scale=.002, size=(len(poses), objective.width)), 1.)
    full = objective.evaluate_frames(moved, True, objective.envelope_weight)
    sampled = coarse.evaluate_frames(moved[indices], True, coarse.envelope_weight)
    for name in ('frame_cost', 'hessian', 'gradient', 'root_point', 'root_jacobian'):
        np.testing.assert_allclose(sampled[name], full[name][indices], rtol=1e-12, atol=1e-8)
    np.testing.assert_array_equal(sampled['effector_point'], full['effector_point'][indices])


def test_coarse_temporal_operators_use_half_fps_and_filtered_full_paths():
    objective, _ = problem()
    objective.temporal_weight[1][3] = 10
    coarse, indices = make_coarse_problem(objective)
    expected = temporal_operators(coarse.frames, 15., objective.cfg['joint_derivative_sigmas'])
    for actual, original in zip(coarse.joint_ops, expected):
        np.testing.assert_array_equal(actual.toarray(), original.toarray())
    assert coarse.temporal_weight[1][1] == 10  # stencil consumes omitted frame 3
    for (_, path, masks), (_, original, full_masks) in zip(coarse.world_paths, objective.world_paths):
        np.testing.assert_array_equal(path, original[indices])
        for order, (mask, full_mask) in enumerate(zip(masks, full_masks), 1):
            for i, active in enumerate(mask):
                assert active == full_mask[indices[i]:indices[i + order] - order + 1].all()


@pytest.mark.parametrize('count', [35, 36])
def test_rotation_transport_and_even_tail_hold_world_correction(count):
    objective, poses = problem()
    fixed = poses[:count].copy()
    angles = np.linspace(0, 2., count)
    reference = Rotation.from_rotvec(np.column_stack((angles, .3 * angles, -.2 * angles)))
    fixed[:, 3:7] = reference.as_quat(scalar_first=True)
    indices = np.arange(0, count, 2)
    world_delta = np.tile([.1, -.07, .03], (len(indices), 1))
    corrected = Rotation.from_rotvec(world_delta) * reference[indices]
    coarse = fixed[indices].copy()
    coarse[:, 3:7] = corrected.as_quat(scalar_first=True)
    coarse[:, 7] += .02 + indices * .001
    delta = interpolate_correction(objective.coords, fixed, coarse, indices)
    warm = objective.coords.integrate(fixed, delta, 1.)
    result = Rotation.from_quat(warm[:, 3:7], scalar_first=True) * reference.inv()
    np.testing.assert_allclose(result.as_rotvec(), np.tile(world_delta[0], (count, 1)), atol=1e-12)
    np.testing.assert_allclose(warm[indices, 7], coarse[:, 7], atol=1e-12)
    assert warm[-1, 7] - fixed[-1, 7] == pytest.approx(coarse[-1, 7] - fixed[indices[-1], 7])
    coarse[:, 3:7] *= -1
    np.testing.assert_allclose(interpolate_correction(objective.coords, fixed, coarse, indices), delta, atol=1e-12)


def test_transfer_checks_full_rate_continuity_and_keeps_frozen_bounds():
    objective, fixed = problem(extra=True)
    indices = np.arange(0, len(fixed), 2)
    coarse = fixed[indices].copy()
    coarse[5, 7] += .5
    before = {id(guard): {key: value.tobytes() for key, value in guard.bounds.items()}
              for guard in objective.candidate_guards if hasattr(guard, 'bounds')}
    warm, report = transfer_correction(objective, fixed, coarse, indices, min_retention=.01)
    assert report['guard_failures'] and report['attempts']
    assert any('Continuity' in row['guard'] for row in report['guard_failures'])
    assert not failure_frames(objective, warm, certify=True)
    for guard in objective.candidate_guards:
        if id(guard) in before:
            assert {key: value.tobytes() for key, value in guard.bounds.items()} == before[id(guard)]


def test_projection_keeps_unaffected_frames_and_exact_joint_limits():
    objective, fixed = problem()
    objective.source_shape, objective.candidate_guards = None, []
    objective.upper[:] = fixed[:, objective.coords.qi] + .3
    objective.upper[8:12, 0] = fixed[8:12, 7] + .025
    indices = np.arange(0, len(fixed), 2)
    coarse = fixed[indices].copy()
    coarse[:, 7] += .1
    warm, report = scale_transfer_correction(objective, fixed, coarse, indices, spread_frames=2)
    assert report['accepted'] and report['global_scale'] == 1.
    assert report['retained_scale'] == pytest.approx(np.sqrt((len(fixed) - 4 + 4 * .25**2) / len(fixed)))
    assert report['projection_frames'] == list(range(8, 12))
    np.testing.assert_allclose(warm[8:12, 7], objective.upper[8:12, 0])
    np.testing.assert_allclose(warm[:8, 7], fixed[:8, 7] + .1)
    assert not failure_frames(objective, warm, certify=True)


def test_projection_retention_too_small_returns_unchanged_reference():
    objective, fixed = problem()
    objective.source_shape, objective.candidate_guards = None, []
    objective.upper[:] = fixed[:, objective.coords.qi] + .001
    indices = np.arange(0, len(fixed), 2)
    coarse = fixed[indices].copy()
    coarse[:, 7] += .1
    warm, report = scale_transfer_correction(objective, fixed, coarse, indices)
    assert not report['accepted'] and report['reason'] == 'poor_retention'
    assert report['retained_scale'] < .25
    assert report['retained_scale'] == pytest.approx(.01)
    assert report['projection_frames'] == list(range(len(fixed)))
    assert report['guard_failures'] == [] and report['first_failing_candidate'] is None
    np.testing.assert_array_equal(warm, fixed)


def test_ordinary_accepts_is_authoritative_if_window_screen_disagrees():
    objective, fixed = problem()
    class Reject:
        def violation_frames(self, poses, *, frames=None):
            return np.zeros(len(poses), bool)
        def accepts(self, poses):
            return False
    objective.source_shape, objective.candidate_guards = None, [Reject()]
    assert not failure_frames(objective, fixed)
    assert failure_frames(objective, fixed, certify=True)['Reject:0'] == list(range(len(fixed)))


def test_joint_layout_and_protected_normals_are_checked_even_with_other_failures():
    objective, fixed = problem(extra=True)
    moved = fixed.copy()
    moved[8:16, 7] = 10  # long enough to test sustained layout, not one spike
    failures = failure_frames(objective, moved, certify=True)
    assert 10 in failures['joint_limits']
    assert any('ReferenceShape' in name for name in failures)
    assert any('AcquisitionNormalGuard' in name for name in failures)


def test_real_coarse_local_lm_and_transfer_have_feasible_full_rate_entry(monkeypatch):
    objective, fixed = problem()
    coarse, indices = make_coarse_problem(objective)
    monkeypatch.setattr(legacy_window, 'fork_status', lambda: (False, 'focused test'))
    result, history = legacy_window.optimize_legacy_windows(
        coarse, fixed[indices], iterations=1, sweep_iterations=(1,), max_extra_sweeps=0,
        core_frames=8, halo_frames=2, min_core_frames=4)
    assert history and not guard_failures(coarse, result)
    assert all(row['halo_frames'] == 2 for row in history)
    warm, report = transfer_correction(objective, fixed, result, indices)
    assert not guard_failures(objective, warm)
    assert not failure_frames(objective, warm, certify=True)
    assert 'retained_scale' in report


@pytest.mark.parametrize('sweeps,refine', [((6,), 2), ((6, 4), 2), ((6,), 4)])
def test_driver_routes_by_original_length_and_closes_coarse_before_refine(monkeypatch, sweeps, refine):
    objective, fixed = problem()
    monkeypatch.setattr(driver, 'MIN_WINDOW_FRAMES', 30)
    monkeypatch.setattr(driver, 'COARSE_SWEEPS', sweeps)
    monkeypatch.setattr(driver, 'REFINE_ITERATIONS', refine)
    events, callbacks = [], []
    coarse_stop = []
    make = driver.make_coarse_problem
    def sample(obj):
        coarse, indices = make(obj)
        coarse.stop_workers = lambda: coarse_stop.append('closed')
        return coarse, indices
    def solve(obj, entry, **kwargs):
        events.append((obj.frames, kwargs))
        if obj is objective:
            assert coarse_stop == ['closed']
            kwargs['callback'](entry, {})
        moved = entry.copy()
        if obj is not objective:
            moved[:, 7] += .001
        return moved, [dict(phase=kwargs['label'])]
    monkeypatch.setattr(driver, 'make_coarse_problem', sample)
    monkeypatch.setattr(driver, 'optimize_legacy_windows', solve)
    result, history, report = driver.optimize_coarse_acquisition(
        objective, fixed, iterations=35, callback=lambda q, row: callbacks.append(q.shape))
    assert [event[0] for event in events] == [18, 36]
    assert events[0][1]['sweep_iterations'] == sweeps
    assert events[0][1]['iterations'] == sum(sweeps)
    assert events[0][1]['core_frames'] == 32 and events[0][1]['halo_frames'] == 8
    assert events[1][1]['sweep_iterations'] == (refine,)
    assert events[1][1]['max_extra_sweeps'] == 0
    assert callbacks == [fixed.shape] and history[0]['phase'] == 'contact_scene'
    assert not report['fallback_ran'] and report['retained_scale'] == pytest.approx(1.)
    assert all(report[name] >= 0 for name in ('coarse_seconds', 'transfer_seconds', 'refine_seconds'))
    assert not failure_frames(objective, result, certify=True)


def test_driver_interpolant_bug_runs_normal_safety_schedule_from_original_entry(monkeypatch):
    objective, fixed = problem()
    monkeypatch.setattr(driver, 'MIN_WINDOW_FRAMES', 30)
    calls = []
    def solve(obj, entry, **kwargs):
        calls.append(kwargs)
        np.testing.assert_array_equal(entry, fixed if obj is objective else fixed[::2])
        result = entry.copy()
        if obj is not objective:
            result[5, 7] += .5  # A faulty optimizer bypassing its composed guard.
        return result, []
    monkeypatch.setattr(driver, 'optimize_legacy_windows', solve)
    _, _, report = driver.optimize_coarse_acquisition(objective, fixed, iterations=35)
    assert len(calls) == 2 and calls[0]['max_extra_sweeps'] == 0
    assert calls[1]['iterations'] == 35 and 'sweep_iterations' not in calls[1]
    assert report['fallback_ran'] and report['safety_net_triggered']
    assert report['fallback_reason'] == 'transfer: full_rate_interpolant_failure'
    assert 'BUG' in report['transfer']['invariant_failure']


def test_runtime_switch_defaults_off_and_can_be_enabled(monkeypatch):
    monkeypatch.delenv('OMNIRETARGETING_FAST_ENABLE', raising=False)
    monkeypatch.delenv('OMNIRETARGETING_FAST_DISABLE', raising=False)
    assert not RuntimeOptions.from_environment().coarse_acquisition
    monkeypatch.setenv('OMNIRETARGETING_FAST_ENABLE', 'coarse_acquisition')
    assert RuntimeOptions.from_environment().coarse_acquisition


def test_pipeline_freezes_contacts_after_full_rate_acquisition_and_serializes_report(monkeypatch):
    import json
    from omniretargeting._fast import pipeline
    from tests.test_fast_backend import NAMES, _problem
    model, cfg, coords, fixed, tasks, rotations = _problem(12)
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(coarse_acquisition=True))
    monkeypatch.setattr(pipeline, 'OPTIONS', runtime_options.OPTIONS)
    monkeypatch.setattr(driver, 'COARSE_SWEEPS', (1,))
    monkeypatch.setattr(driver, 'REFINE_ITERATIONS', 1)
    monkeypatch.setattr(driver, 'FORCE_WINDOWS', True)
    monkeypatch.setattr(legacy_window, 'fork_status', lambda: (False, 'focused test'))
    events = []
    solve, acquire, contact_guard = (driver.optimize_legacy_windows,
                                     driver.optimize_coarse_acquisition, pipeline.ContactGuard)
    def windows(obj, poses, **kwargs):
        result = solve(obj, poses, **kwargs)
        events.append((kwargs['label'], len(result[0])))
        return result
    monkeypatch.setattr(driver, 'optimize_legacy_windows', windows)
    def complete(*args, **kwargs):
        result = acquire(*args, **kwargs)
        events.append(('full_acquisition_complete', len(result[0])))
        return result
    monkeypatch.setattr(driver, 'optimize_coarse_acquisition', complete)
    # Both contact guard modes follow full acquisition (refine or fallback).
    def guard(obj, acquired, **kwargs):
        assert ('full_acquisition_complete', len(fixed)) in events
        assert ('contact_scene_coarse', len(fixed[::2])) in events
        return contact_guard(obj, acquired, **kwargs)
    monkeypatch.setattr(pipeline, 'ContactGuard', guard)
    common = dict(body_orientation_targets=(), environment_points=None,
                  root_orientation_targets=rotations.as_quat(scalar_first=True), framerate=30.)
    priority = dict(effectors={name: dict(robot_body=spec['robot_body'],
                                         tracking_offset=spec['tracking_offset'],
                                         contact_offsets=spec['contact_offsets'])
                              for name, spec in tasks.effectors.items()})
    motion, report = pipeline.solve_fast_sequence(
        SimpleNamespace(), model, fixed, tasks.human, NAMES, dict(zip(NAMES, NAMES)),
        tasks.roots, {name: spec['task'] for name, spec in tasks.effectors.items()},
        priority, dict(cfg, iterations=1), common, (), reference_trajectory=fixed)
    assert motion.shape == fixed.shape
    stage = report['stages']['contact_scene']
    assert stage['coarse_acquisition']['enabled']
    assert stage['acquisition_continuity_accepted']
    json.dumps(report)
