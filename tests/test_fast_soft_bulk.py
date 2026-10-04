"""Private soft LM: exact derivatives, worker reuse, and no guarded proposals."""

import numpy as np
import pytest

from omniretargeting._fast.runtime_options import RuntimeOptions
from omniretargeting._fast.soft_bulk import bulk_context, run_bulk
from tests.test_fast_barrier import dense
from tests.test_fast_legacy_window import problem


def context(objective, poses):
    return bulk_context(objective, poses, dict(evaluation_seconds=0., evaluations=0.), speed_gate=False)


def test_soft_switch_defaults_off_and_context_restores_byte_equivalent_objective():
    assert not RuntimeOptions().soft_acquisition
    objective, poses = problem(extra=True)
    before = objective.evaluate(poses)
    reference = objective.reference.tobytes()
    guards = objective.candidate_guards
    with pytest.raises(RuntimeError, match='injected'):
        with context(objective, poses):
            assert objective._soft_bulk_signals
            raise RuntimeError('injected')
    after = objective.evaluate(poses)
    assert not hasattr(objective, '_soft_bulk_signals')
    assert objective.pool is None
    assert objective.candidate_guards is guards
    assert objective.reference.tobytes() == reference
    assert before[0].cost == after[0].cost
    assert before[0].gradient.tobytes() == after[0].gradient.tobytes()
    assert before[0].hessian.tobytes() == after[0].hessian.tobytes()
    assert before[1].tobytes() == after[1].tobytes()
    assert before[4] is after[4] is None


@pytest.mark.parametrize('correction', [False, True])
def test_soft_continuity_gradient_fd_psd_and_banded_assembly(correction):
    objective, poses = problem(extra=True)
    for guard in objective.candidate_guards[:2]:
        guard.correction = correction
    rng = np.random.default_rng(26)
    moved = objective.coords.integrate(poses, rng.normal(scale=.012, size=(len(poses), objective.width)), 1.)
    with context(objective, poses) as facade:
        initial = facade.evaluate(poses)
        diagonal = np.maximum(np.diagonal(initial[0].hessian, axis1=1, axis2=2).ravel(), 1.)
        blocks = list(facade.rows.blocks(poses, initial[4], True))
        penalty, _, _ = facade.rows.accumulate(poses, initial[4], diagonal, blocks=blocks)
        assert penalty < 1e-15
        value = facade.evaluate(moved)
        band = np.zeros((4 * objective.width, len(poses) * objective.width), order='F')
        blocks = list(facade.rows.blocks(moved, value[4], True))
        penalty, gradient, _ = facade.rows.accumulate(moved, value[4], diagonal, band, blocks=blocks)
        tangent = rng.normal(scale=.02, size=gradient.shape)

        def merit(alpha):
            candidate = objective.coords.integrate(moved, tangent.reshape(len(poses), -1), alpha)
            signals = facade.evaluate(candidate)[4]
            return facade.rows.accumulate(candidate, signals, diagonal)[0]

        epsilon = 1e-5
        numeric = (merit(epsilon) - merit(-epsilon)) / (2 * epsilon)
        assert 2 * gradient @ tangent == pytest.approx(numeric, rel=3e-5, abs=1e-6)
        assert tangent @ dense(band) @ tangent >= -1e-8
        expected = np.zeros((len(gradient), len(gradient)))
        for label, raw, matrix, allowed, norm in blocks:
            if norm:
                sizes = np.linalg.norm(raw, axis=1)
                derivative = matrix.toarray().reshape(len(raw), raw.shape[1], -1)
                derivative = np.einsum('fi,fiw->fw', raw / np.maximum(sizes[:, None], 1e-30), derivative)
            else:
                sizes = np.abs(raw).ravel()
                derivative = matrix.toarray() * np.sign(raw).ravel()[:, None]
            active = sizes > np.asarray(allowed).ravel()
            weights = facade.rows.weights[label][active]
            derivative = derivative[active]
            expected += derivative.T @ (weights[:, None] * derivative)
        np.testing.assert_allclose(dense(band), expected, rtol=1e-12, atol=1e-7)


def test_root_anchor_reuses_parallel_fk_and_preserves_quaternion_signs(monkeypatch):
    from omniretargeting._fast import objective as module
    monkeypatch.setattr(module, 'MIN_CHUNK_FRAMES', 8)
    monkeypatch.setattr(module, 'WORKERS', 3)
    objective, poses = problem(extra=True)
    with context(objective, poses) as facade:
        assert objective.pool is not None
        value = facade.evaluate(poses)
        blocks = list(facade.rows.blocks(poses, value[4], True))
        guard = next(g for kind, g, _ in facade.rows.guards if kind == 'sequence')
        original = guard.signals(poses)
        for label, residual, _, _, _ in blocks:
            if label[0] == type(guard).__name__:
                np.testing.assert_allclose(residual, np.diff(original[label[1]][0], n=label[2], axis=0), atol=1e-14)
        signed = poses.copy()
        signed[::2, 3:7] *= -1
        alt = list(facade.rows.blocks(signed, facade.evaluate(signed)[4], True))
        for first, second in zip(blocks, alt):
            np.testing.assert_allclose(first[1], second[1], atol=1e-13)
    assert objective.pool is None


def test_extra_guard_merit_does_not_change_reported_physical_cost():
    objective, poses = problem(extra=True)
    moved = poses.copy()
    moved[10:20, 7:] += .5
    original = objective.evaluate(moved)[0].cost
    with context(objective, poses) as facade:
        value = facade.evaluate(moved)
        assert value[4]['guard_cost'] > 0.
        assert facade.physical_cost(value) == pytest.approx(original, rel=1e-12)


def test_bulk_has_only_full_banded_steps_without_guard_conic_or_restoration(monkeypatch):
    from omniretargeting._fast import proposals, shape_step
    objective, poses = problem()
    reference = objective.reference.tobytes()

    def forbidden(*args, **kwargs):
        raise AssertionError('guarded proposal in private bulk')

    for guard in [objective.source_shape, *objective.candidate_guards]:
        monkeypatch.setattr(guard, 'accepts', forbidden)
    monkeypatch.setattr(proposals, 'Attempts', forbidden)
    monkeypatch.setattr(shape_step, 'restore_shape_step', forbidden)
    bulk, history, report = run_bulk(objective, poses, iterations=3, speed_gate=False)
    assert history and all(row['alpha'] == 1. for row in history)
    assert all(row['after'] < row['cost'] for row in history)
    assert report['value_evaluations'] >= len(history)
    assert report['jacobian_evaluations'] == len(history)
    assert report['exact_guard_calls'] == report['conic_calls'] == report['restoration_calls'] == 0
    assert np.all(bulk[:, objective.coords.qi] >= objective.lower)
    assert np.all(bulk[:, objective.coords.qi] <= objective.upper)
    assert objective.reference.tobytes() == reference


def test_zero_boxes_root_domains_and_fixed_roots_stay_exact():
    objective, poses = problem()
    objective.fixed_root = poses[:, :7].copy()
    objective.lower = np.broadcast_to(objective.lower, (len(poses), len(objective.coords.qi))).copy()
    objective.upper = np.broadcast_to(objective.upper, objective.lower.shape).copy()
    objective.lower[:, -1] = objective.upper[:, -1] = poses[:, objective.coords.qi[-1]]
    bulk, _, _ = run_bulk(objective, poses, iterations=2, speed_gate=False)
    np.testing.assert_array_equal(bulk[:, :7], poses[:, :7])
    np.testing.assert_array_equal(bulk[:, objective.coords.qi[-1]], poses[:, objective.coords.qi[-1]])
    objective.fixed_root = None
    objective.root_budget = 0.
    bulk, _, _ = run_bulk(objective, poses, iterations=1, speed_gate=False)
    np.testing.assert_array_equal(bulk[:, :3], poses[:, :3])


def test_rejected_trial_uses_values_and_preserves_entry_linearization(monkeypatch):
    objective, poses = problem()
    evaluate = objective.evaluate
    calls = []

    def reject_first(candidate, with_jacobian=True, **kwargs):
        value = evaluate(candidate, with_jacobian, **kwargs)
        calls.append(with_jacobian)
        if calls == [True, False]:
            value[0].cost += 1e12
        return value

    monkeypatch.setattr(objective, 'evaluate', reject_first)
    _, history, report = run_bulk(objective, poses, iterations=1)
    assert history[0]['after'] < history[0]['cost']
    assert calls[0] and not any(calls[1:])
    assert report['jacobian_evaluations'] == 1
    assert report['value_evaluations'] >= 2
