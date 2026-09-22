"""Independent checks for the optional physics experiment and its boundaries."""

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.optimize import minimize

from omniretargeting._ilqr.dynamics import BatchedDynamics, StateChart
from omniretargeting._ilqr.model import box_scene_bounds
from omniretargeting._ilqr.solver import SolverSettings, solve
from omniretargeting._ilqr.tracking import ReferenceTracker, TrackingWeights, replay

XML = """
<mujoco><option timestep=".002"/>
<worldbody><body name="torso" pos="0 0 2"><freejoint/>
<geom type="sphere" size=".1" mass="1"/>
<body name="tip" pos="0 0 -.3"><joint name="hinge" axis="0 1 0" range="-2 2"/>
<geom size=".03" mass=".1" pos=".2 0 0"/></body></body></worldbody>
<actuator><motor joint="hinge" ctrllimited="true" ctrlrange="-5 5"/></actuator>
</mujoco>
"""


def test_chart_matches_mujoco_and_antipodal_quaternions():
    model = mujoco.MjModel.from_xml_string(XML)
    chart = StateChart(model)
    state = np.r_[model.qpos0, np.zeros(model.nv + model.nu)]
    delta = np.random.default_rng(42).normal(0, 0.15, chart.nx)
    actual = chart.integrate(state, delta)
    expected = model.qpos0.copy()
    mujoco.mj_integratePos(model, expected, delta[: model.nv], 1)
    np.testing.assert_allclose(actual[: model.nq], expected, atol=1e-14)
    np.testing.assert_allclose(chart.difference(state, actual), delta, atol=1e-14)
    opposite = actual.copy()
    opposite[3:7] *= -1
    np.testing.assert_allclose(chart.difference(actual, opposite), 0, atol=1e-14)


def test_real_scene_adapter_rejects_nonbox_instead_of_using_convex_hull():
    box = trimesh.creation.box([10, 10, 0.1])
    np.testing.assert_allclose(box_scene_bounds(box), box.bounds)
    with pytest.raises(ValueError):
        box_scene_bounds(trimesh.creation.icosphere())
    rotation = trimesh.transformations.rotation_matrix(0.3, [0, 1, 0])
    with pytest.raises(ValueError):
        box_scene_bounds(box.copy().apply_transform(rotation))


class LinearProblem:
    """Independent scalar system with previous control and a convex slew cost."""

    T, nx, nu = 8, 2, 1
    lo, hi, alphas = np.array([-0.3]), np.array([0.3]), 0.5 ** np.arange(9)

    def cost(self, t, x, u):
        return x[..., 0] ** 2 + (
            0
            if t == self.T
            else 0.1 * u[..., 0] ** 2 + 2 * (u[..., 0] - x[..., 1]) ** 2
        )

    def rollout(self, initial, controls, gains=None):
        state = np.tile(initial, (len(self.alphas), 1))
        xs, us, total = [state.copy()], [], np.zeros(len(self.alphas))
        for t in range(self.T):
            u = np.tile(controls[t], (len(self.alphas), 1))
            if gains is not None:
                ref, k, K = gains
                u += self.alphas[:, None] * k[t] + (state - ref[t]) @ K[t].T
            u = np.clip(u, self.lo, self.hi)
            total += self.cost(t, state, u)
            state = np.c_[state[:, 0] + u[:, 0], u[:, 0]]
            xs.append(state.copy())
            us.append(u)
        total += self.cost(self.T, state, np.zeros((len(self.alphas), 1)))
        return np.array(xs), np.array(us), total

    def linearize(self, xs, us):
        return np.tile([[1.0, 0], [0, 0]], (self.T, 1, 1)), np.ones((self.T, 2, 1))

    def expand(self, xs, us):
        lx, lxx = np.zeros_like(xs), np.zeros((self.T + 1, 2, 2))
        lx[:, 0], lxx[:, 0, 0] = 2 * xs[:, 0], 2
        delta = us[:, 0] - xs[:-1, 1]
        lx[:-1, 1], lxx[:-1, 1, 1] = -4 * delta, 4
        lu = (0.2 * us[:, 0] + 4 * delta)[:, None]
        lux = np.zeros((self.T, 1, 2))
        lux[:, 0, 1] = -4
        return lx, lxx, lu, np.full((self.T, 1, 1), 4.2), lux


def test_ilqr_matches_independent_convex_optimization_and_measures_seed():
    p = LinearProblem()
    initial, controls = np.array([1.0, 0.0]), np.zeros((p.T, 1))
    _, _, no_steps = solve(p, initial, controls, SolverSettings(iterations=0))
    assert no_steps["final_cost"] == pytest.approx(9)
    xs, us, report = solve(
        p, initial, controls, SolverSettings(iterations=30, relative_tolerance=1e-10)
    )
    independent = minimize(
        lambda u: p.rollout(initial, u[:, None])[2][0],
        controls[:, 0],
        method="L-BFGS-B",
        bounds=[(-0.3, 0.3)] * p.T,
        options={"ftol": 1e-13, "gtol": 1e-8},
    )
    assert report["final_cost"] == pytest.approx(independent.fun, abs=1e-7)
    assert np.max(abs(us)) <= 0.3
    expected_states = p.rollout(initial, independent.x[:, None])[0][:, 0]
    np.testing.assert_allclose(xs, expected_states, atol=1e-5)
    costs = [report["initial_cost"], *[r["cost"] for r in report["iterations"]]]
    assert np.all(np.diff(costs) <= 0)


def batch_available():
    pytest.importorskip("mjbatch")


def test_batched_derivatives_and_continuous_rollout_match_serial_physics():
    batch_available()
    model = mujoco.MjModel.from_xml_string(XML)
    problem = BatchedDynamics(model, horizon=5, substeps=3, threads=2)
    problem.cost = lambda t, x, u: np.zeros(len(x))
    initial = np.r_[model.qpos0, np.zeros(model.nv + model.nu)]
    controls = np.full((problem.T, model.nu), 0.25)
    xs, _, _ = problem.rollout(initial, controls)
    serial = replay(model, initial, controls, problem.sub)
    np.testing.assert_allclose(xs[:, 0, : model.nq + model.nv], serial, atol=1e-12)
    assert xs[-1, 0, 2] < initial[2]  # the root falls; no hidden root clamp/reset
    A, B = problem.linearize(xs[:, 0], controls)
    rng = np.random.default_rng(6)
    dx, du = rng.normal(size=(problem.T, problem.nx)), rng.normal(size=controls.shape)
    eps = 1e-7
    base = problem.step(xs[:-1, 0], controls)
    perturbed = problem.step(
        problem.integrate(xs[:-1, 0], eps * dx), controls + eps * du
    )
    measured = problem.difference(base, perturbed) / eps
    predicted = np.einsum("tij,tj->ti", A, dx) + np.einsum("tij,tj->ti", B, du)
    np.testing.assert_allclose(measured, predicted, atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(A[:, -1], 0)
    np.testing.assert_array_equal(B[:, -1], 1)


def test_tracking_slew_cross_derivative_and_free_hand_global_invariance():
    batch_available()
    model = mujoco.MjModel.from_xml_string(XML)
    poses = np.tile(model.qpos0, (4, 1))
    effectors = {
        name: {"robot_body": "tip"}
        for name in ("left_hand", "right_hand", "left_foot", "right_foot")
    }
    p = ReferenceTracker(
        model,
        poses,
        np.zeros((4, model.nv)),
        effectors,
        "torso",
        1,
        TrackingWeights(slew_weight=3),
        threads=2,
    )
    states = p.reference.copy()
    us = np.full((3, 1), 0.2)
    _, _, lu, _, lux = p.expand(states, us)
    eps = 1e-5
    shifted = states.copy()
    shifted[:, -1] += eps
    _, _, lu_shifted, _, _ = p.expand(shifted, us)
    np.testing.assert_allclose((lu_shifted - lu) / eps, lux[:, :, -1], atol=1e-9)
    numerical = (
        p.cost(0, states[:1], us[:1] + eps) - p.cost(0, states[:1], us[:1] - eps)
    ) / (2 * eps)
    np.testing.assert_allclose(numerical, lu[0], atol=1e-9)
    shifted = states.copy()
    shifted[:, :3] += [2, -3, 1]
    before, after = p.points(states), p.points(shifted)
    np.testing.assert_allclose(before["left_hand"], after["left_hand"], atol=1e-14)
    np.testing.assert_allclose(
        after["left_foot"] - before["left_foot"], np.tile([2, -3, 1], (4, 1))
    )
