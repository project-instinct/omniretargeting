"""Full-rate guard composition, exact odd frames, SO(3) chain rule and windows."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast.coarse_embedding import LinearCorrectionEmbedding
from omniretargeting._fast.coarse_guards import FullRateGuard
from omniretargeting._fast.coarse_problem import make_coarse_problem
from omniretargeting._fast.coarse_transfer import failure_frames, transfer_correction
from omniretargeting._fast.continuity import CorrectionContinuity
from omniretargeting._fast.legacy_window_qp import WindowObjective, guard_failures
from omniretargeting._fast.proposals import guards_of
from tests.test_fast_legacy_window import problem


@pytest.mark.parametrize('count', [35, 36])
def test_linear_operator_midpoints_knots_and_held_tail(count):
    objective, original = problem()
    fixed = original[:count].copy()
    fixed[1::2, 7] += .02  # Authored odd motion remains in the interpolant.
    indices = np.arange(0, count, 2)
    embedding = LinearCorrectionEmbedding(objective.coords, fixed, indices)
    poses = fixed[indices].copy()
    correction = np.linspace(-.02, .03, len(indices))
    poses[:, 7] += correction
    poses[:, :3] += correction[:, None] * [.2, -.1, .3]
    full = embedding.embed(poses)
    np.testing.assert_array_equal(full[indices], poses)
    np.testing.assert_allclose(full[1:indices[-1]:2, 7] - fixed[1:indices[-1]:2, 7],
                               (correction[:-1] + correction[1:]) / 2, atol=1e-12)
    np.testing.assert_allclose(np.asarray(embedding.L.sum(axis=1)).ravel(), 1.)
    assert (embedding.L.getnnz(axis=1) <= 2).all()
    assert full[-1, 7] - fixed[-1, 7] == pytest.approx(correction[-1])
    np.testing.assert_array_equal(embedding.embed(fixed[indices]), fixed)


@pytest.mark.parametrize('near_pi', [False, True])
def test_embedding_tangent_jacobian_matches_noncommuting_rotation_finite_differences(near_pi):
    objective, fixed = problem()
    t = np.linspace(0, 1, len(fixed))
    reference = Rotation.from_euler('xyz', np.c_[.4 * t, -.3 * t**2, .5 * t])
    fixed[:, 3:7] = reference.as_quat(scalar_first=True)
    indices = np.arange(0, len(fixed), 2)
    axes = np.c_[np.ones(len(indices)), .15 * np.sin(t[indices]), .2 * np.cos(t[indices])]
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = (np.pi - .12 if near_pi else .12) + .24 * t[indices]
    poses = fixed[indices].copy()
    poses[:, 3:7] = (Rotation.from_rotvec(axes * angles[:, None])
                      * reference[indices]).as_quat(scalar_first=True)
    embedding = LinearCorrectionEmbedding(objective.coords, fixed, indices)
    frames = range(4, 11)
    affected = embedding.affected_frames(frames)
    step = np.random.default_rng(2).normal(size=(len(poses), objective.width))
    step[:frames.start] = step[frames.stop:] = 0.
    expected = (embedding.jacobian(poses, frames, affected) @ step[frames].ravel()).reshape(
        len(affected), objective.width)
    origin = embedding.embed(poses).copy()
    eps = 1e-6
    plus = embedding.embed(objective.coords.integrate(poses, step, eps)).copy()
    minus = embedding.embed(objective.coords.integrate(poses, step, -eps)).copy()
    observed = (objective.coords.difference(origin, plus) - objective.coords.difference(origin, minus)) / (2 * eps)
    np.testing.assert_allclose(expected, observed[affected], atol=2e-8, rtol=2e-7)


def test_composed_linear_and_cone_rows_match_finite_differences():
    objective, fixed = problem(extra=True)
    coarse, indices = make_coarse_problem(objective)
    rng = np.random.default_rng(5)
    poses = objective.coords.integrate(fixed[indices], rng.normal(
        scale=1e-4, size=(len(indices), objective.width)), 1.)
    frames = range(5, 10)
    direction = rng.normal(size=poses.shape[:-1] + (objective.width,))
    direction[:frames.start] = direction[frames.stop:] = 0.
    eps = 1e-6
    plus = objective.coords.integrate(poses, direction, eps)
    minus = objective.coords.integrate(poses, direction, -eps)
    for guard in guards_of(coarse)[1]:
        linear, cones = guard.qp_bounds(poses, frames=frames)
        left, left_cones = guard.qp_bounds(minus, frames=frames)
        right, right_cones = guard.qp_bounds(plus, frames=frames)
        assert len(linear) == len(left) == len(right)
        for (matrix, lo, hi), (_, lo0, hi0), (_, lo1, hi1) in zip(linear, left, right):
            predicted = matrix @ direction[frames].ravel()
            # Bounds freeze; changing a RHS measures minus its signal change.
            choose_hi = np.isfinite(hi)
            observed = np.empty(len(lo))
            observed[choose_hi] = -(hi1[choose_hi] - hi0[choose_hi]) / (2 * eps)
            observed[~choose_hi] = -(lo1[~choose_hi] - lo0[~choose_hi]) / (2 * eps)
            np.testing.assert_allclose(predicted, observed, atol=3e-6, rtol=3e-5,
                                       err_msg=type(getattr(guard, 'full_guard', guard)).__name__)
        for (matrix, value, _), (_, v0, _), (_, v1, _) in zip(cones, left_cones, right_cones):
            predicted = (matrix @ direction[frames].ravel()).reshape(value.shape)
            np.testing.assert_allclose(predicted, (v1 - v0) / (2 * eps), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('frames', [None, range(0, 4), range(5, 10), range(14, 18)])
def test_random_coarse_acceptance_equals_full_interpolant_and_keeps_frozen_bounds(frames):
    objective, fixed = problem(extra=True)
    before = {id(g): {key: value.tobytes() for key, value in g.bounds.items()}
              for g in objective.candidate_guards if hasattr(g, 'bounds')}
    coarse, indices = make_coarse_problem(objective)
    rng = np.random.default_rng(11)
    wrappers = [coarse.source_shape, *coarse.candidate_guards[1:]]
    originals = guards_of(objective)[1]
    for scale in (0., 1e-6, .001, .02):
        poses = objective.coords.integrate(fixed[indices], rng.normal(
            scale=scale, size=(len(indices), objective.width)), 1.)
        full = coarse.coarse_embedding.embed(poses)
        for wrapper, guard in zip(wrappers, originals):
            if frames is None:
                expected = guard.accepts(full)
                bad = guard.violation_frames(full)
            else:
                affected = coarse.coarse_embedding.affected_frames(frames)
                expected = guard.accepts(full, frames=affected)
                bad = guard.violation_frames(full, frames=affected)
            assert wrapper.accepts(poses, frames=frames) == expected
            np.testing.assert_array_equal(wrapper.violation_frames(poses, frames=frames),
                                           coarse.coarse_embedding.map_violations(bad))
        if frames is None:
            assert (not guard_failures(coarse, poses)) == (not failure_frames(objective, full, certify=True))
    assert all({key: value.tobytes() for key, value in g.bounds.items()} == before[id(g)]
               for g in objective.candidate_guards if id(g) in before)
    for wrapper, guard in zip(wrappers[1:], objective.candidate_guards):
        if hasattr(guard, 'bounds'):
            assert wrapper.full_guard.bounds is guard.bounds


def test_odd_only_normal_constraint_is_present_in_coarse_rows():
    objective, fixed = problem(extra=True)
    full_guard = objective.candidate_guards[-1]
    odd = []
    for group in full_guard.groups:
        row = list(group)
        row[5] = group[5].copy()
        row[5][::2] = False
        odd.append(tuple(row))
    full_guard.groups = odd
    coarse, indices = make_coarse_problem(objective)
    assert not coarse.hinge_guards[-1].groups  # Objective retains even frames only.
    wrapper = coarse.candidate_guards[-1]
    assert wrapper.full_guard.groups
    linear, cones = wrapper.qp_bounds(fixed[indices])
    assert sum(matrix.shape[0] for matrix, _, _ in linear + cones) > 0
    assert wrapper.accepts(fixed[indices])


def test_odd_joint_box_is_an_exact_two_knot_row_without_transfer_projection(monkeypatch):
    objective, fixed = problem()
    objective.source_shape, objective.candidate_guards = None, []
    objective.upper[9, 0] = fixed[9, 7]
    coarse, indices = make_coarse_problem(objective)
    poses = fixed[indices].copy()
    poses[:, 7] += .01
    box = coarse.candidate_guards[0]
    assert not box.accepts(poses)
    assert box.violation_frames(poses)[4:6].all()
    rows, _ = box.qp_bounds(fixed[indices], frames=range(4, 6))
    row = 2 * len(objective.coords.qi)  # affected frames begin at 7; frame 9 is third.
    matrix, _, upper = rows[0]
    columns = objective.coords.scalar_cols[0] + np.arange(2) * objective.width
    np.testing.assert_array_equal(matrix[row, columns].toarray().ravel(), [.5, .5])
    assert upper[row] == 0.
    def forbidden(*args):
        raise AssertionError('Ordinary transfer must never project')
    monkeypatch.setattr(objective, 'project', forbidden)
    warm, report = transfer_correction(objective, fixed, poses, indices)
    assert report['safety_net_triggered'] and report['reason'] == 'full_rate_interpolant_failure'
    np.testing.assert_array_equal(warm, fixed)
    poses[4:6, 7] = fixed[indices[4:6], 7]
    assert box.accepts(poses)
    warm, report = transfer_correction(objective, fixed, poses, indices)
    assert report['accepted'] and not report['safety_net_triggered']
    assert report['retained_scale'] == 1. and report['projection_frames'] == []
    np.testing.assert_array_equal(warm, coarse.coarse_embedding.embed(poses))


def test_window_crossing_stencils_use_current_fixed_coarse_neighbours():
    objective, fixed = problem()
    fixed[:, 7] = .003 * np.arange(len(fixed))
    objective.reference = fixed.copy()
    objective.source_shape = None
    guard = CorrectionContinuity(objective.coords, fixed, correction=False)
    objective.candidate_guards, objective.hinge_guards = [guard], []
    coarse, indices = make_coarse_problem(objective)
    wrapper = coarse.candidate_guards[-1]
    frames = range(5, 8)
    assert coarse.coarse_embedding.affected_frames(frames) == range(9, 16)
    assert wrapper.window_dependencies(frames) == range(3, 10)
    candidate = fixed[indices].copy()
    candidate[5, 7] += .001
    bad = wrapper.violation_frames(candidate, frames=frames)
    assert bad[3:5].any()  # Jerk stencil starts before the changed full span.
    assert not wrapper.accepts(candidate, frames=frames)
    current = fixed[indices].copy()
    current[9, 7] += .007  # Fixed neighbour consumed by a crossing stencil.
    local = WindowObjective(coarse, current, frames)
    local_guard = local.candidate_guards[-1]
    a, b = local_guard.qp_bounds(current[frames]), wrapper.qp_bounds(current, frames=frames)
    for (matrix, lo, hi), (expected, lo0, hi0) in zip(a[0], b[0]):
        np.testing.assert_array_equal(matrix.toarray(), expected.toarray())
        np.testing.assert_array_equal(lo, lo0)
        np.testing.assert_array_equal(hi, hi0)
    empty, _ = wrapper.qp_bounds(fixed[indices], frames=frames)
    assert any(not np.array_equal(lo, lo0) for (_, lo, _), (_, lo0, _) in zip(a[0], empty))


def test_objective_and_full_guard_share_exact_even_pose_fk_cache():
    objective, fixed = problem(extra=True)
    coarse, indices = make_coarse_problem(objective)
    frames = range(5, 10)
    poses = fixed[indices].copy()
    coarse.window_kinematics(poses, frames, True)
    cache = coarse._window_frame_cache
    before = cache.computed_frames
    wrapper = coarse.source_shape
    wrapper.qp_bounds(poses, frames=frames)
    assert wrapper.full_guard._window_frame_cache is cache
    assert cache.computed_frames - before <= len(coarse.coarse_embedding.affected_frames(frames)) - len(frames)
