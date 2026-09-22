# SPDX-License-Identifier: Apache-2.0
"""Box-constrained iLQR with a measured initial rollout and control slew cost.

Backward recursion adapted from Kevin Zakka's mjbatch examples/g1_flip.py,
revision 2f169146b643adda65e195d8ad13d997abdd2a2e (Apache-2.0).
Changes: explicit settings, mixed cost derivatives, failure checks, and logs.
See LICENSE.mjbatch. This solver does not certify motion/contact acceptance.
"""

from dataclasses import dataclass
from time import perf_counter

import mujoco
import numpy as np


@dataclass(frozen=True)
class SolverSettings:
    iterations: int = 20
    relative_tolerance: float = 1e-5
    initial_regularization: float = 1.0
    max_regularization: float = 1e8


def boxqp(hessian, gradient, lower, upper, seed):
    """Solve a positive definite bound-constrained QP, including its free set."""
    solution = np.clip(seed, lower, upper).copy()
    n = len(gradient)
    index = np.empty(n, dtype=np.int32)
    count = mujoco.mju_boxQP(
        solution,
        np.empty((n, n + 7)),
        index,
        np.ascontiguousarray(hessian),
        np.ascontiguousarray(gradient),
        np.ascontiguousarray(lower),
        np.ascontiguousarray(upper),
    )
    if count < 0 or not np.isfinite(solution).all():
        raise np.linalg.LinAlgError("MuJoCo boxQP failed")
    free = np.zeros(n, dtype=bool)
    free[index[:count]] = True
    return solution, free


def backward(A, B, lx, lxx, lu, luu, lux, lower, upper, regularization):
    """Riccati recursion; lux is d²cost / du dx, including control slew."""
    horizon, nu, nx = len(lu), lu.shape[1], lx.shape[1]
    vx, vxx = lx[-1].copy(), lxx[-1].copy()
    k = np.zeros((horizon + 1, nu))
    gains = np.zeros((horizon, nu, nx))
    for t in reversed(range(horizon)):
        qx = lx[t] + A[t].T @ vx
        qu = lu[t] + B[t].T @ vx
        qxx = lxx[t] + A[t].T @ vxx @ A[t]
        quu = luu[t] + B[t].T @ vxx @ B[t]
        qux = lux[t] + B[t].T @ vxx @ A[t]
        regularized = vxx + regularization * np.eye(nx)
        quu_reg = luu[t] + B[t].T @ regularized @ B[t]
        qux_reg = lux[t] + B[t].T @ regularized @ A[t]
        if not np.isfinite(quu_reg).all():
            raise np.linalg.LinAlgError("Nonfinite control Hessian")
        np.linalg.cholesky(quu_reg)
        k[t], free = boxqp(quu_reg, qu, lower[t], upper[t], k[t + 1])
        gains[t, free] = -np.linalg.solve(quu_reg[np.ix_(free, free)], qux_reg[free])
        K = gains[t]
        vx = qx + K.T @ quu @ k[t] + K.T @ qu + qux.T @ k[t]
        vxx = qxx + K.T @ quu @ K + K.T @ qux + qux.T @ K
        vxx = 0.5 * (vxx + vxx.T)
    return k[:-1], gains


def solve(problem, initial_state, controls, settings=None, progress=None):
    """Improve an actual rollout; never accept an unmeasured first update."""
    settings = SolverSettings() if settings is None else settings
    if settings.iterations < 0 or settings.initial_regularization <= 0:
        raise ValueError("Invalid iLQR settings")
    started = perf_counter()
    controls = np.clip(np.asarray(controls, dtype=float), problem.lo, problem.hi)
    states, candidates, costs = problem.rollout(initial_state, controls)
    if not np.isfinite(costs[0]):
        raise ValueError("Initial rollout is unstable; no valid iLQR seed")
    states, controls, cost = states[:, 0], candidates[:, 0], float(costs[0])
    initial_cost = cost
    history = []
    regularization = settings.initial_regularization
    derivatives = None
    status = "iteration_limit"
    for iteration in range(settings.iterations):
        if derivatives is None:
            derivatives = (
                *problem.linearize(states, controls),
                *problem.expand(states, controls),
            )
        record = {
            "iteration": iteration,
            "before_cost": cost,
            "regularization": regularization,
        }
        accepted = False
        try:
            k, K = backward(
                *derivatives,
                problem.lo - controls,
                problem.hi - controls,
                regularization,
            )
            xs, us, totals = problem.rollout(initial_state, controls, (states, k, K))
            best = int(np.argmin(totals))
            if np.isfinite(totals[best]) and totals[best] < cost:
                previous = cost
                states, controls, cost = xs[:, best], us[:, best], float(totals[best])
                record.update(
                    alpha=float(problem.alphas[best]), decrease=previous - cost
                )
                accepted = True
                regularization = max(regularization / 10, 1e-8)
                derivatives = None
                if previous - cost <= settings.relative_tolerance * max(
                    1.0, abs(previous)
                ):
                    status = "converged"
        except np.linalg.LinAlgError as exc:
            record["backward_failure"] = str(exc)
        if not accepted:
            regularization *= 10
            if regularization > settings.max_regularization:
                status = "regularization_limit"
        record.update(accepted=accepted, cost=cost, seconds=perf_counter() - started)
        history.append(record)
        if progress is not None:
            progress(record)
        if status != "iteration_limit":
            break
    return (
        states,
        controls,
        {
            "status": status,
            "initial_cost": initial_cost,
            "final_cost": cost,
            "seconds": perf_counter() - started,
            "iterations": history,
        },
    )
