"""Sparse collision rows preserve the scalar SQP halfspaces and slack cost."""

import cvxpy as cp
import numpy as np

from omniretargeting.collision_linear_rows import SceneCollisionRows


def test_sparse_collision_rows_match_scalar_window_problem():
    samples = [
        (np.array([1.0, -0.5]), [(0, 1.0)], -0.04, -0.01, 0.03),
        (np.array([0.2, 1.0]), [(0, 0.4), (1, 0.6)], -0.025, -0.01, 0.015),
        (np.array([-0.8, 0.3]), [(1, 1.0)], 0.005, -0.01, 0.0),
    ]
    step = cp.Variable((2, 2))
    rows = SceneCollisionRows(2, 2)
    for sample in samples:
        rows.append(*sample)
    constraints = [step <= 0.1, step >= -0.1]
    objective = [cp.sum_squares(step)]
    rows.add_to_problem(step, constraints, objective, 0.005)
    sparse_problem = cp.Problem(cp.Minimize(sum(objective)), constraints)
    sparse_problem.solve(solver="CLARABEL")

    scalar_step = cp.Variable((2, 2))
    scalar_slacks = [cp.Variable(nonneg=True) for _ in samples]
    scalar_constraints = [scalar_step <= 0.1, scalar_step >= -0.1]
    for (gradient, weights, clearance, target, cap), slack in zip(
        samples, scalar_slacks
    ):
        increment = sum(weight * scalar_step[frame] for frame, weight in weights)
        scalar_constraints.extend([
            clearance + gradient @ increment >= target - slack,
            slack <= cap,
        ])
    scalar_problem = cp.Problem(
        cp.Minimize(cp.sum_squares(scalar_step)
                    + cp.sum_squares(cp.hstack(scalar_slacks) / 0.005)),
        scalar_constraints,
    )
    scalar_problem.solve(solver="CLARABEL")

    np.testing.assert_allclose(step.value, scalar_step.value, atol=1e-7)
    np.testing.assert_allclose(rows.solved_slacks(),
                               [float(item.value) for item in scalar_slacks], atol=1e-7)
    np.testing.assert_allclose(sparse_problem.value, scalar_problem.value, atol=1e-7)
