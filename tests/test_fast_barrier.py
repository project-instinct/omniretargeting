"""Barrier calculus, exact domains, boundary stencils and local transactions."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.barrier_assembly import physical_system, scatter
from omniretargeting._fast.barrier_rows import (
    BarrierRows, accumulate, ball, boundary_fraction, scalar, strictly_interior,
)
from omniretargeting._fast.legacy_window_qp import WindowObjective, solve_local
from omniretargeting._fast.runtime_options import RuntimeOptions
from tests.test_fast_legacy_window import problem


def dense(ab):
    size = ab.shape[1]
    result = np.zeros((size, size))
    for offset in range(len(ab)):
        index = np.arange(size - offset)
        result[index + offset, index] = ab[offset, :size - offset]
        if offset:
            result[index, index + offset] = ab[offset, :size - offset]
    return result


def affine_families(x):
    a = np.array([[1., 2., 0.], [0., -1., .3]])
    j = np.array([[.5, 0., 1.], [0., .7, 0.]])
    return [scalar(3 + a @ x, sparse.csr_matrix(a), ('scalar',), [(0,), (1,)]),
            ball(np.array([[.2, -.4]]) + (j @ x)[None], np.array([2.]),
                 sparse.csr_matrix(j), ('ball',), [(0,)])]


def test_barrier_gradient_finite_difference_and_affine_hessian_psd():
    x, mu = np.array([.1, -.2, .05]), .037
    families = affine_families(x)
    ab = np.zeros((3, 3), order='F')
    gradient = accumulate(families, mu, ab)
    hessian = dense(ab)

    def merit(point):
        return -mu * sum(np.log(f.slack).sum() for f in affine_families(point))

    epsilon = 1e-5
    eye = np.eye(3)
    numeric = np.array([(merit(x + epsilon * e) - merit(x - epsilon * e)) / (2 * epsilon)
                        for e in eye])
    np.testing.assert_allclose(gradient, numeric, rtol=1e-8, atol=1e-10)
    numeric_h = np.column_stack([
        (accumulate(affine_families(x + epsilon * e), mu, np.zeros_like(ab, order='F')) -
         accumulate(affine_families(x - epsilon * e), mu, np.zeros_like(ab, order='F'))) / (2 * epsilon)
        for e in eye])
    np.testing.assert_allclose(hessian, numeric_h, rtol=1e-8, atol=1e-10)
    assert np.linalg.eigvalsh(hessian).min() >= -1e-12


def test_physical_and_barrier_band_equal_dense_without_sparse_normal_products(monkeypatch):
    from types import SimpleNamespace
    rng = np.random.default_rng(32)
    frames, width = 6, 4
    blocks = rng.normal(size=(frames, width, width))
    h = np.einsum('fki,fkj->fij', blocks, blocks)
    rows = np.zeros((9, frames * width))
    for r in range(9):
        start = r % 3
        rows[r, start * width:(start + 4) * width] = rng.normal(size=4 * width)
    matrix = sparse.csr_matrix(rows)
    expected = sparse.block_diag(h).toarray() + rows.T @ rows
    acc = SimpleNamespace(gradient=np.zeros((frames, width)), hessian=h)
    original = sparse.csr_matrix.__matmul__

    def forbid_normal(self, other):
        if sparse.issparse(other):
            raise AssertionError('assembled a sparse normal matrix')
        return original(self, other)

    monkeypatch.setattr(sparse.csr_matrix, '__matmul__', forbid_normal)
    ab = physical_system(acc, matrix)
    np.testing.assert_allclose(dense(ab), expected, atol=1e-12)
    weights = rng.uniform(.01, 2, len(rows))
    scatter(ab, matrix, weights)
    np.testing.assert_allclose(dense(ab), expected + rows.T @ (weights[:, None] * rows), atol=1e-12)


def test_fraction_to_boundary_scalar_and_exact_affine_ball():
    families = affine_families(np.zeros(3))
    step = np.array([-8., -2., 4.])
    alpha = boundary_fraction(families, step)
    assert 0 < alpha < 1
    assert strictly_interior(affine_families(alpha * step))
    assert not strictly_interior(affine_families(alpha / .995 * 1.000001 * step))


@pytest.mark.parametrize('correction', [False, True])
def test_crossing_stencils_equal_full_rows_with_nonzero_fixed_neighbours(correction):
    objective, poses = problem(extra=True)
    for guard in objective.candidate_guards[:2]:
        guard.correction = correction
    moved = poses.copy()
    moved[:, 7:] += .003  # Neighbour signals/corrections are not zero.
    frames = range(9, 18)
    local = WindowObjective(objective, moved, frames)
    full = WindowObjective(objective, moved, range(len(moved)))
    small = BarrierRows(local).evaluate(moved[local.span], True)
    large = BarrierRows(full).evaluate(moved, True)
    columns = slice(frames.start * objective.width, frames.stop * objective.width)
    indexed = {}
    for f in large:
        if 'Continuity' not in str(f.label):
            continue
        dimension = 1 if not f.ball else f.residual.shape[1]
        for i in range(len(f.slack)):
            indexed[f.identity(i)] = (f.slack[i], f.matrix[i * dimension:(i + 1) * dimension, columns])
    crossing = 0
    for f in small:
        if 'Continuity' not in str(f.label):
            continue
        dimension = 1 if not f.ball else f.residual.shape[1]
        for i in range(len(f.slack)):
            slack, jac = indexed[f.identity(i)]
            np.testing.assert_allclose(f.slack[i], slack, atol=1e-16)
            np.testing.assert_allclose(f.matrix[i * dimension:(i + 1) * dimension].toarray(),
                                       jac.toarray(), atol=1e-13)
            crossing += int(f.locations[i][0] < frames.start)
    assert crossing > 0


def test_exact_guard_slacks_and_quaternion_tangent_jacobians():
    objective, poses = problem(extra=True)
    local = WindowObjective(objective, poses, range(7, 20))
    rows = BarrierRows(local)
    part = poses[local.span].copy()
    direction = np.random.default_rng(28).normal(scale=.01, size=(local.frames, local.width))
    families = rows.evaluate(part, True)
    epsilon = 1e-5
    plus = rows.evaluate(local.coords.integrate(part, direction, epsilon), False)
    minus = rows.evaluate(local.coords.integrate(part, direction, -epsilon), False)
    for f, a, b in zip(families, plus, minus):
        derivative = f.matrix @ direction.ravel()
        if f.ball:
            derivative = -2 * np.einsum('ij,ij->i', f.residual, derivative.reshape(f.residual.shape))
        numeric = (a.slack - b.slack) / (2 * epsilon)
        np.testing.assert_allclose(derivative, numeric, atol=2e-9, rtol=2e-5)
    for sign in (-1, 1):
        # Quaternion sign is not an orientation change.
        signed = part.copy()
        signed[::2, 3:7] *= sign
        alternative = rows.evaluate(signed, False)
        for f, other in zip(families, alternative):
            np.testing.assert_allclose(f.slack, other.slack, atol=1e-13)
    for amplitude in (0., 1e-4, .04, .8):
        changed = part.copy()
        changed[0, 7] += amplitude
        values = rows.evaluate(changed, False)
        # Identical immutable joint rows shared by the two temporal guards
        # are deduplicated. The complete domains must still agree exactly.
        assert all(np.all(f.slack >= 0) for f in values) == all(
            wrapped.accepts(changed) for wrapped in [local.source_shape, *local.candidate_guards])


def test_exact_asymmetric_contact_yield_and_layout_predicates():
    objective, poses = problem()
    contact = objective.candidate_guards[-1]
    assert contact.groups
    groups = []
    for kind, indices, offsets, target, normal, active, held, bound, own in contact.groups:
        # Freeze a nonzero outward yield; it never applies to the inward face.
        groups.append((kind, indices, offsets, target, normal, active,
                       active.copy(), bound, np.full_like(own, .03)))
    contact.groups = groups
    local = WindowObjective(objective, poses, range(4, 14))
    rows = BarrierRows(local)
    for shift in (-.1, -.02, 0., .02, .1):
        trial = poses[local.span].copy()
        trial[:, 2] += shift
        values = rows.evaluate(trial, False)
        selected = [f for f in values if f.label[0] == 'AcquisitionNormalGuard']
        assert all(np.all(f.slack >= 0) for f in selected) == local.candidate_guards[-1].accepts(trial)
    held = [f for f in rows.evaluate(poses[local.span], False) if f.label[0] == 'AcquisitionNormalGuard']
    assert any('normal_inward' in str(f.identity(0)) for f in held)
    # Large articulated motion tests the original layout predicate itself.
    for angle in (0., .8, 2.5):
        trial = poses[local.span].copy()
        trial[:, 7] = angle
        layout = [f for f in rows.evaluate(trial, False) if f.label[0] == 'ReferenceShape']
        assert all(np.all(f.slack >= 0) for f in layout) == local.source_shape.accepts(trial)


def test_zero_budget_active_box_fixed_root_and_commit(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.continuity import CorrectionContinuity
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(barrier_acquisition=True))
    objective, poses = problem()
    poses[:, 10] = .2  # A truly stationary joint has only the existing roundoff interior.
    objective.candidate_guards[0] = CorrectionContinuity(objective.coords, poses, root_norm=True)
    objective.lower = np.broadcast_to(objective.lower, (len(poses), len(objective.coords.qi))).copy()
    objective.upper = np.broadcast_to(objective.upper, objective.lower.shape).copy()
    objective.lower[:, -1] = objective.upper[:, -1] = .2
    objective.fixed_root = poses[:, :7].copy()
    reference = objective.reference.tobytes()
    local = WindowObjective(objective, poses, range(8, 20))
    families = BarrierRows(local).evaluate(poses[local.span], False)
    assert strictly_interior(families)
    result = solve_local(objective, poses, local.variable_frames, 2, .003)
    assert result['barrier']['mode'] == 'barrier'
    np.testing.assert_array_equal(result['poses'][:, :7], poses[local.span, :7])
    np.testing.assert_array_equal(result['poses'][:, 10], poses[local.span, 10])
    assert objective.reference.tobytes() == reference
    assert all(row['after'] < row['cost'] for row in result['history'] if row['alpha'])
    assert result['barrier']['escapes'] <= 1
    assert result['commit']['accepted'] == (result['commit']['candidate_cost'] < result['commit']['cost'])


def test_barrier_bypasses_attempts_and_restoration_and_retains_parent_rollback(monkeypatch):
    from omniretargeting._fast import runtime_options, proposals, shape_step
    from omniretargeting._fast.legacy_window import _commit
    from omniretargeting._fast.proposals import guards_of
    from omniretargeting._fast.window_schedule import dependency_window
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(barrier_acquisition=True))

    def forbidden(*args, **kwargs):
        raise AssertionError('barrier ran Attempts/restoration')

    monkeypatch.setattr(proposals, 'Attempts', forbidden)
    monkeypatch.setattr(shape_step, 'restore_shape_step', forbidden)
    objective, poses = problem()
    frames = range(8, 20)
    result = solve_local(objective, poses, frames, 2, .003)
    assert result['barrier']['mode'] == 'barrier'
    assert any(row['alpha'] > 0 for row in result['history'])
    task = dependency_window(frames, len(poses), [objective, *guards_of(objective)[1]])
    committed, verdict = _commit(objective, poses, poses, task, result)
    assert verdict['accepted'] and not np.array_equal(committed, poses)
    stale = poses.copy()
    stale[frames.start - 1, 7] += .01
    with pytest.raises(RuntimeError, match='dependency changed'):
        _commit(objective, stale, poses, task, result)
    invalid = result['poses'].copy()
    invalid[0, 7] = 10.
    rolled, verdict = _commit(objective, poses, poses, task, dict(poses=invalid))
    assert not verdict['accepted'] and rolled.tobytes() == poses.tobytes()


def test_default_off_is_byte_identical_and_does_not_load_barrier(monkeypatch):
    from omniretargeting._fast import runtime_options, barrier_solver
    assert RuntimeOptions().barrier_acquisition is False

    def forbidden(*args, **kwargs):
        raise AssertionError('default-off called barrier')

    monkeypatch.setattr(barrier_solver, 'optimize', forbidden)
    outputs = []
    for options in (RuntimeOptions(), replace(RuntimeOptions(), barrier_acquisition=False)):
        monkeypatch.setattr(runtime_options, 'OPTIONS', options)
        objective, poses = problem()
        result = solve_local(objective, poses, range(8, 20), 1, .003)
        outputs.append((result['poses'].tobytes(), [(r['cost'], r['after'], r['alpha']) for r in result['history']]))
    assert outputs[0] == outputs[1]


def test_no_interior_and_unknown_adapter_fall_back_with_binding_row(monkeypatch):
    from omniretargeting._fast import runtime_options, barrier_solver
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(barrier_acquisition=True))
    objective, poses = problem()
    original = BarrierRows.evaluate

    def boundary(self, poses, jacobian=False):
        rows = original(self, poses, jacobian)
        rows[0].slack[0] = 0.
        return rows

    monkeypatch.setattr(BarrierRows, 'evaluate', boundary)
    result = solve_local(objective, poses, range(8, 20), 1, .003)
    assert result['barrier']['mode'] == 'legacy_fallback'
    assert result['barrier']['binding_row']['slack'] == 0
    assert result['barrier']['binding_row']['identity']
    monkeypatch.setattr(BarrierRows, 'evaluate', original)
    objective.candidate_guards.append(object())
    local = WindowObjective(objective, poses, range(8, 20))
    result, _, _, reason, report = barrier_solver.optimize(local, poses[local.span], 1, .003)
    assert result is None and reason == 'unsupported_barrier'
    assert 'unknown barrier adapter' in report['binding_row']


def test_stall_uses_one_escape_for_entire_solve_and_never_claims_convergence(monkeypatch):
    from omniretargeting._fast import barrier_solver, shape_step
    objective, poses = problem()
    local = WindowObjective(objective, poses, range(8, 20))
    calls = []

    def escape(*args, **kwargs):
        calls.append(1)
        return np.zeros(local.frames * local.width)

    monkeypatch.setattr(barrier_solver, '_trial', lambda *args, **kwargs: None)
    monkeypatch.setattr(shape_step, 'bounded_shape_step', escape)
    result, history, converged, reason, report = barrier_solver.optimize(
        local, poses[local.span], 6, .003)
    assert calls == [1] and report['escapes'] == 1
    assert reason == 'barrier_stall' and not converged and history[-1]['alpha'] == 0
    assert report['binding_row']['identity']
    assert result.tobytes() == poses[local.span].tobytes()


def test_rejected_nonlinear_trials_evaluate_slacks_before_physical_jacobians(monkeypatch):
    from omniretargeting._fast import barrier_solver
    objective, poses = problem()
    local = WindowObjective(objective, poses, range(8, 20))
    rows = BarrierRows(local)
    part = poses[local.span]
    families = rows.evaluate(part, True)
    rows.initialize(families)
    calls = []
    original = rows.evaluate

    def rejected(candidate, jacobian=False):
        calls.append(jacobian)
        values = original(candidate, jacobian)
        values[0].slack[:] = -1.
        return values

    monkeypatch.setattr(rows, 'evaluate', rejected)
    monkeypatch.setattr(local, 'evaluate', lambda *args, **kwargs: pytest.fail('evaluated rejected physical trial'))
    diagnostics = dict(boundary_contractions=0, physical_value_evaluations=0, merit_contractions=0)
    size = local.frames * local.width
    result = barrier_solver._trial(local, rows, part, families, np.zeros(size), 1., 100.,
                                    .001, np.zeros(size), diagnostics)
    assert result is None and calls and not any(calls)
