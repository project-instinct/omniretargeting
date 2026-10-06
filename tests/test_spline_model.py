"""Independent checks for the all-frame spline correction model."""

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.continuity import temporal_signals
from omniretargeting._fast.objective import _Accumulator
from omniretargeting._fast.proposals import guards_of
from omniretargeting._fast.qp_assembly import prepare_model
from omniretargeting._fast.qp_types import sparse_hessian
from omniretargeting._fast.solver import DIAGONAL_FLOOR, banded_system
from omniretargeting._fast.spline_embedding import CorrectionSpline
from omniretargeting._fast.spline_model import (
    ChatterRows,
    frame_normal_equations,
    high_frequency_operator,
    prepare_spline_model,
)
from omniretargeting.motion_budgets import (
    CHATTER_EXCESS_TOLERANCE,
    chatter_budget,
    high_frequency,
)
from tests.test_fast_backend import _objective
from tests.test_fast_legacy_window import problem as legacy_problem
from tests.test_fast_whole_canonical import problem as canonical_problem


def _assert_program_bytes_equal(actual, expected):
    assert actual.A.shape == expected.A.shape
    assert actual.b.shape == expected.b.shape
    np.testing.assert_array_equal(actual.A.toarray(), expected.A.toarray())
    np.testing.assert_array_equal(actual.b, expected.b)
    assert actual.nonneg == expected.nonneg
    assert actual.soc == expected.soc


def _program_feasible(program, step, tolerance=1e-9):
    slack = program.b - program.A @ step
    good = bool(np.all(slack[:program.nonneg] >= -tolerance))
    cursor = program.nonneg
    for size in program.soc:
        cone = slack[cursor:cursor + size]
        good &= bool(cone[0] >= np.linalg.norm(cone[1:]) - tolerance)
        cursor += size
    assert cursor == len(slack)
    return good


@pytest.mark.parametrize("make_problem", [canonical_problem, legacy_problem],
                         ids=["whole-canonical", "legacy-window"])
def test_identity_model_matches_production_banded_and_prepare_model(make_problem):
    """At B=I, a zero spline is exactly the production full-rate model."""
    objective, poses = make_problem()
    objective.evaluation_backend = "cpu"
    spline = CorrectionSpline(objective.coords, objective.reference, objective.framerate,
                              identity=True)
    coefficients = np.zeros_like(spline.coefficients)
    evaluation = objective.evaluate(poses)
    accumulator, temporal_r, temporal_j = evaluation[:3]

    model, mapping, selected = prepare_spline_model(objective, spline, coefficients, evaluation)
    system = banded_system(objective.frames, objective.width,
                           accumulator.hessian, temporal_j)
    gradient = accumulator.gradient.ravel() + temporal_j.T @ temporal_r
    diagonal = np.maximum(
        np.diagonal(accumulator.hessian, axis1=1, axis2=2).ravel(), DIAGONAL_FLOOR)
    production = prepare_model(objective, poses, system, gradient, diagonal)

    expected_hessian = sparse_hessian(system)
    assert mapping.shape == (objective.frames * objective.width, coefficients.size)
    assert selected.shape == (coefficients.size,)
    np.testing.assert_allclose(model.hessian.toarray(), expected_hessian.toarray(),
                               rtol=2e-10, atol=2e-9)
    np.testing.assert_allclose(model.hessian.toarray(), production.hessian.toarray(),
                               rtol=2e-10, atol=2e-9)
    np.testing.assert_array_equal(model.gradient, production.gradient)
    np.testing.assert_array_equal(model.gradient, gradient)
    np.testing.assert_array_equal(model.diagonal, production.diagonal)
    np.testing.assert_array_equal(model.diagonal, diagonal)
    np.testing.assert_array_equal(model.origin, production.origin)
    np.testing.assert_array_equal(model.step_lower, production.step_lower)
    np.testing.assert_array_equal(model.step_upper, production.step_upper)
    assert model.scale == production.scale
    _assert_program_bytes_equal(model.program, production.program)

    assert len(model.program.hard_blocks) == len(production.program.hard_blocks)
    for actual, source in zip(model.program.hard_blocks,
                              production.program.hard_blocks):
        np.testing.assert_array_equal(actual.A.toarray(), source.A.toarray())
        np.testing.assert_array_equal(actual.b, source.b)
        assert actual.soc == source.soc
    assert model.program.soc == production.program.soc


def test_cubic_guard_blocks_are_pulled_back_without_changing_cones():
    """Every original hard row is mapped by J; SOC b and cone sizes are frozen."""
    objective, poses = canonical_problem()
    objective.evaluation_backend = "cpu"
    spline = CorrectionSpline(objective.coords, objective.reference, objective.framerate,
                              spacing=.18)
    rng = np.random.default_rng(23)
    coefficients = rng.normal(scale=.002, size=spline.coefficients.shape)
    moved = spline.embed(coefficients)
    evaluation = objective.evaluate(moved)
    accumulator, temporal_r, temporal_j = evaluation[:3]
    model, mapping, _ = prepare_spline_model(objective, spline, coefficients, evaluation)

    system = banded_system(objective.frames, objective.width,
                           accumulator.hessian, temporal_j)
    gradient = accumulator.gradient.ravel() + temporal_j.T @ temporal_r
    diagonal = np.maximum(
        np.diagonal(accumulator.hessian, axis1=1, axis2=2).ravel(), DIAGONAL_FLOOR)
    production = prepare_model(objective, moved, system, gradient, diagonal)
    np.testing.assert_allclose(mapping.toarray(), spline.jacobian(coefficients).toarray(),
                               rtol=1e-13, atol=1e-14)
    assert len(model.program.hard_blocks) == len(production.program.hard_blocks)
    for actual, source in zip(model.program.hard_blocks,
                              production.program.hard_blocks):
        expected = (source.A @ mapping).toarray()
        np.testing.assert_allclose(actual.A.toarray(), expected, rtol=2e-12,
                                   atol=2e-12)
        np.testing.assert_array_equal(actual.b, source.b)
        assert actual.soc == source.soc
    assert model.program.soc == production.program.soc


def test_nonzero_root_rotation_model_gradient_is_half_the_cost_slope():
    """FastObjective returns J.T r, while its cost is a non-halved squared sum."""
    objective, _, _ = _objective(temporal=True, extra=True)
    objective.evaluation_backend = "cpu"
    spline = CorrectionSpline(objective.coords, objective.reference, objective.framerate,
                              spacing=.20)
    rng = np.random.default_rng(31)
    coefficients = rng.normal(scale=.015, size=spline.coefficients.shape)
    coefficients[:, spline.angular] += np.array([.25, -.15, .20])
    assert np.linalg.norm(spline.correction(coefficients)[:, spline.angular], axis=1).max() > .2
    moved = spline.embed(coefficients)
    evaluation = objective.evaluate(moved)
    model, _, _ = prepare_spline_model(objective, spline, coefficients, evaluation)

    epsilon = 1e-5
    for index in range(3):
        direction = rng.normal(size=coefficients.shape)
        direction /= np.linalg.norm(direction)
        plus = objective.evaluate(spline.embed(coefficients + epsilon * direction),
                                  False)[0].cost
        minus = objective.evaluate(spline.embed(coefficients - epsilon * direction),
                                   False)[0].cost
        finite_difference = (plus - minus) / (2 * epsilon)
        # The objective's gradient convention carries no factor of two.
        analytic_slope = 2 * (model.gradient @ direction.ravel())
        np.testing.assert_allclose(finite_difference, analytic_slope,
                                   rtol=5e-8, atol=max(2e-4, 2e-8 * abs(finite_difference)))
        actual_decrease = evaluation[0].cost - plus
        np.testing.assert_allclose(model.tracking_decrease(epsilon * direction.ravel()),
                                   actual_decrease, rtol=5e-6, atol=5e-6)


def test_frame_normal_equations_match_explicit_jacobian_pullback():
    """The block assembler and a small explicit J.T H J oracle are identical."""
    objective, _, _ = _objective(temporal=True, extra=True)
    objective.evaluation_backend = "cpu"
    spline = CorrectionSpline(objective.coords, objective.reference, objective.framerate,
                              spacing=.16)
    rng = np.random.default_rng(47)
    coefficients = rng.normal(scale=.05, size=spline.coefficients.shape)
    coefficients[:, spline.angular] += np.array([.35, -.25, .30])
    moved = spline.embed(coefficients)
    accumulator = objective.evaluate(moved)[0]
    hessian, gradient = frame_normal_equations(
        spline.basis, spline.tangent_differentials(coefficients),
        accumulator.hessian, accumulator.gradient)
    jacobian = spline.jacobian(coefficients)
    frame_hessian = sparse.block_diag(accumulator.hessian, format="csc")
    expected_hessian = jacobian.T @ frame_hessian @ jacobian
    expected_gradient = np.asarray(jacobian.T @ accumulator.gradient.ravel()).ravel()
    np.testing.assert_allclose(hessian.toarray(), expected_hessian.toarray(),
                               rtol=3e-10, atol=3e-9)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=3e-11, atol=3e-10)

    # Exercise the bounded batch loop with a small synthetic full-rate problem.
    frames, controls, width = 300, 47, objective.width
    columns = np.argsort(rng.random((frames, controls)), axis=1)[:, :4]
    rows = np.repeat(np.arange(frames), 4)
    basis = sparse.csr_matrix((rng.normal(size=(frames, 4)).ravel(),
                               (rows, columns.ravel())), shape=(frames, controls))
    blocks = rng.normal(size=(frames, width, width))
    blocks = .5 * (blocks + blocks.transpose(0, 2, 1))
    blocks += (np.arange(width) + 1.)[None, :, None] * np.eye(width)
    local_gradient = rng.normal(size=(frames, width))
    actual_hessian, actual_gradient = frame_normal_equations(
        basis, np.broadcast_to(np.eye(width), (frames, width, width)).copy(),
        blocks, local_gradient)
    dense_jacobian = sparse.kron(basis, sparse.eye(width), format="csc")
    oracle_hessian = dense_jacobian.T @ sparse.block_diag(blocks, format="csc") @ dense_jacobian
    oracle_gradient = np.asarray(dense_jacobian.T @ local_gradient.ravel()).ravel()
    np.testing.assert_allclose(actual_hessian.toarray(), oracle_hessian.toarray(),
                               rtol=3e-11, atol=3e-10)
    np.testing.assert_allclose(actual_gradient, oracle_gradient, rtol=3e-11,
                               atol=3e-10)
    support = ((basis.T @ basis).astype(bool)).toarray()
    assert np.sum(support) < controls * controls
    assert actual_hessian.nnz <= np.sum(support) * width * width


def test_chatter_rows_use_motion_budgets_and_ignore_quaternion_sign():
    objective, _, _ = _objective(temporal=True)
    objective.evaluation_backend = "cpu"
    spline = CorrectionSpline(objective.coords, objective.reference, objective.framerate,
                              spacing=.20)
    rng = np.random.default_rng(43)
    coefficients = rng.normal(scale=.05, size=spline.coefficients.shape)
    coefficients[:, spline.angular] += np.array([.30, -.20, .25])
    moved = spline.embed(coefficients)
    chatter = ChatterRows(objective.coords, spline.reference)
    blocks = chatter.blocks(moved)
    current = temporal_signals(objective.coords, moved)
    reference = temporal_signals(objective.coords, spline.reference)
    jacobian = spline.jacobian(coefficients)
    assert len(blocks) == len(current)

    epsilon = 1e-6
    for (name, signal), block in zip(current.items(), blocks):
        direction = rng.normal(size=coefficients.shape)
        tangent = np.asarray(jacobian @ direction.ravel())
        plus = temporal_signals(objective.coords,
                                spline.embed(coefficients + epsilon * direction))[name]
        minus = temporal_signals(objective.coords,
                                 spline.embed(coefficients - epsilon * direction))[name]
        finite_difference = (
            high_frequency(plus - reference[name])
            - high_frequency(minus - reference[name])) / (2 * epsilon)
        half = len(block.b) // 2
        np.testing.assert_allclose(np.asarray(block.A[:half] @ tangent),
                                   finite_difference.ravel(), rtol=2e-7, atol=2e-9)
        np.testing.assert_allclose(np.asarray(block.A[half:] @ tangent),
                                   -finite_difference.ravel(), rtol=2e-7, atol=2e-9)
        residual = high_frequency(signal - reference[name]).ravel()
        bound = (chatter_budget(reference[name]) + CHATTER_EXCESS_TOLERANCE).ravel()
        np.testing.assert_array_equal(block.b[:half], bound - residual)
        np.testing.assert_array_equal(block.b[half:], -(-bound - residual))

    scalar = np.arange(19.)[:, None]
    np.testing.assert_allclose(np.asarray(high_frequency_operator(19) @ scalar.ravel()),
                               high_frequency(scalar).ravel(), atol=2e-15)

    signed = spline.reference.copy()
    q = objective.coords.root_q
    signed[::2, q + 3:q + 7] *= -1
    signed_spline = CorrectionSpline(objective.coords, signed, objective.framerate,
                                     spacing=.20)
    signed_moved = signed_spline.embed(coefficients)
    signed_blocks = ChatterRows(objective.coords, signed).blocks(signed_moved)
    for original, signed_block in zip(blocks, signed_blocks):
        np.testing.assert_array_equal(original.A.toarray(), signed_block.A.toarray())
        np.testing.assert_array_equal(original.b, signed_block.b)
        assert original.soc == signed_block.soc


def test_linearized_program_matches_nonlinear_guard_acceptance():
    objective, poses = canonical_problem()
    objective.evaluation_backend = "cpu"
    spline = CorrectionSpline(objective.coords, objective.reference, objective.framerate,
                              spacing=.18)
    coefficients = np.zeros_like(spline.coefficients)
    chatter = ChatterRows(objective.coords, spline.reference)
    model, _, _ = prepare_spline_model(
        objective, spline, coefficients, objective.evaluate(poses), chatter=chatter)
    guards = guards_of(objective)[1]
    rng = np.random.default_rng(71)
    saw_feasible = saw_infeasible = False

    for _ in range(3):
        direction = rng.normal(size=coefficients.shape)
        direction /= np.linalg.norm(direction)
        for amplitude in (0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1):
            step = amplitude * direction.ravel()
            candidate = spline.embed(coefficients + amplitude * direction)
            linear = _program_feasible(model.program, step)
            nonlinear = all(guard.accepts(candidate) for guard in guards)
            assert linear == nonlinear
            saw_feasible |= linear
            saw_infeasible |= not linear
    assert saw_feasible and saw_infeasible


def test_odd_only_contacts_and_every_original_box_row_survive():
    objective, poses, _ = _objective(temporal=True)
    objective.evaluation_backend = "cpu"
    foot = next(effector for effector in objective.effectors if effector["name"] == "foot")
    foot["active"][:] = False
    foot["active"][1::2] = True
    foot["managed"][:] = False
    foot["surface_only"][:] = False

    objective.kin.evaluate(poses, with_jacobian=False)
    accumulator = _Accumulator(objective.frames, objective.width, False)
    objective._effector_terms(accumulator, {})
    odd = foot["active"]
    assert np.flatnonzero(odd).tolist() == list(range(1, objective.frames, 2))
    assert np.median(accumulator.frame_cost[odd]) > 20 * np.median(
        accumulator.frame_cost[~odd])

    spline = CorrectionSpline(objective.coords, objective.reference, objective.framerate,
                              spacing=.24)
    model, mapping, _ = prepare_spline_model(
        objective, spline, spline.coefficients, objective.evaluate(poses))
    assert spline.basis.shape[0] == objective.frames
    assert np.all(mapping.getnnz(axis=1) > 0)
    box = model.program.hard_blocks[0]
    assert box.A.shape == (2 * objective.frames * objective.width,
                           spline.controls * objective.width)
    assert not box.soc
    assert model.step_lower.shape == model.step_upper.shape == (
        objective.frames, objective.width)
