"""Correctness of the foot sole orientation residual.

The residual is a first-order model of how the sole rotates when the actuated
configuration changes.  If that model is wrong the optimizer will confidently
walk in the wrong direction, so these tests check it against forward
kinematics rather than against itself:

* the angular Jacobian must predict the actual rotation change, with the
  prediction error shrinking quadratically as the step shrinks;
* the residual must be blind to floating-base rotation, which is the property
  that stops the solver from satisfying a foot target by rolling the robot.
"""

import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from scipy.spatial.transform import Rotation

from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.utils import load_robot_urdf_with_floating_base

URDF = Path(__file__).resolve().parents[1] / "robot_models" / "unitree_g1" / "g1_29dof_popsicle.urdf"
LEFT_FOOT = "left_ankle_roll_link"
RIGHT_FOOT = "right_ankle_roll_link"


@pytest.fixture(scope="module")
def robot():
    if not URDF.exists():
        pytest.skip(f"{URDF} not available")
    cwd = os.getcwd()
    try:
        os.chdir(URDF.parent)
        model = load_robot_urdf_with_floating_base(Path(URDF.name))
    finally:
        os.chdir(cwd)
    return model, mujoco.MjData(model)


def _retargeter(model, data, enabled=True, weight=10.0):
    """A retargeter carrying only what the orientation task needs."""
    r = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
    r.robot_model = model
    r.robot_data = data
    r.q_a_indices = np.arange(0, model.nq)
    r.nq_a = model.nq
    r._setup_link_orientation(
        {
            "enabled": enabled,
            "weight": weight,
            "effectors": {"left": {"body": LEFT_FOOT}, "right": {"body": RIGHT_FOOT}},
        }
        if enabled
        else None
    )
    return r


def _pose(model, rng, base_rot=None):
    q = np.zeros(model.nq)
    q[3:7] = [1.0, 0.0, 0.0, 0.0]
    if base_rot is not None:
        q[3:7] = base_rot
    q[7:] = rng.uniform(-0.3, 0.3, size=model.nq - 7)
    q[2] = 0.8
    return q


def _sole_world(model, data, q, body_id):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    return data.xmat[body_id].reshape(3, 3).copy()


class TestAngularJacobian:
    def test_prediction_error_decays_quadratically(self, robot):
        """A first-order model cannot be exact at finite steps.

        Asserting equality to a fixed tolerance would either pass a wrong
        Jacobian at a small step or fail a correct one at a large step. The
        signature of a correct linearization is that halving the step quarters
        the leftover error.
        """
        model, data = robot
        r = _retargeter(model, data)
        rng = np.random.default_rng(3)
        body_id = r.link_orientation_specs["left"]["body_id"]

        q = _pose(model, rng)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        base_rot = _sole_world(model, data, q, body_id)
        jac = r._body_angular_jacobian(body_id)

        direction = rng.normal(size=model.nq)
        direction[3:7] = 0.0  # keep the quaternion valid under a plain offset
        direction /= np.linalg.norm(direction)

        errors = []
        steps = [1e-2, 5e-3, 2.5e-3]
        for h in steps:
            dq = h * direction
            actual = _sole_world(model, data, q + dq, body_id)
            predicted_rotvec = jac @ dq
            predicted = Rotation.from_rotvec(predicted_rotvec).as_matrix() @ base_rot
            residual = Rotation.from_matrix(actual @ predicted.T).magnitude()
            errors.append(residual)

        assert errors[0] > 0.0
        for coarse, fine in zip(errors, errors[1:]):
            # Halving the step should quarter the error; allow generous slack
            # so the test tracks the order rather than the constant.
            assert fine < coarse * 0.45

    def test_first_order_term_dominates(self, robot):
        model, data = robot
        r = _retargeter(model, data)
        rng = np.random.default_rng(11)
        body_id = r.link_orientation_specs["right"]["body_id"]

        q = _pose(model, rng)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        base_rot = _sole_world(model, data, q, body_id)
        jac = r._body_angular_jacobian(body_id)

        direction = rng.normal(size=model.nq)
        direction[3:7] = 0.0
        direction /= np.linalg.norm(direction)
        dq = 1e-4 * direction

        actual = _sole_world(model, data, q + dq, body_id)
        actual_rotvec = Rotation.from_matrix(actual @ base_rot.T).as_rotvec()
        predicted_rotvec = jac @ dq

        assert np.linalg.norm(predicted_rotvec) > 1e-9
        np.testing.assert_allclose(predicted_rotvec, actual_rotvec, rtol=1e-3, atol=1e-9)

    def test_holds_at_non_identity_base_orientation(self, robot):
        model, data = robot
        r = _retargeter(model, data)
        rng = np.random.default_rng(5)
        body_id = r.link_orientation_specs["left"]["body_id"]

        tilted = Rotation.from_euler("xyz", [25.0, -40.0, 65.0], degrees=True)
        q = _pose(model, rng, base_rot=tilted.as_quat(scalar_first=True))
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        base_rot = _sole_world(model, data, q, body_id)
        jac = r._body_angular_jacobian(body_id)

        direction = rng.normal(size=model.nq)
        direction[3:7] = 0.0
        direction /= np.linalg.norm(direction)
        dq = 1e-4 * direction

        actual = _sole_world(model, data, q + dq, body_id)
        actual_rotvec = Rotation.from_matrix(actual @ base_rot.T).as_rotvec()
        np.testing.assert_allclose(jac @ dq, actual_rotvec, rtol=1e-3, atol=1e-9)


class TestRootRelativeResidual:
    def test_error_is_zero_when_the_sole_already_matches(self, robot):
        model, data = robot
        r = _retargeter(model, data)
        rng = np.random.default_rng(7)
        q = _pose(model, rng)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)

        targets = {
            side: data.xmat[spec["body_id"]].reshape(3, 3).copy()
            for side, spec in r.link_orientation_specs.items()
        }
        residuals = r._link_orientation_residuals(targets)
        assert len(residuals) == 2
        for _, error in residuals:
            np.testing.assert_allclose(error, np.zeros(3), atol=1e-9)

    def test_rotating_the_whole_robot_leaves_the_error_unchanged(self, robot):
        """The residual must not be satisfiable by turning the base.

        The actuated set includes the floating base and the base is barely
        regularized, so a world-frame foot residual would be cheapest to
        satisfy by rolling the entire robot instead of moving the ankle.
        """
        model, data = robot
        r = _retargeter(model, data)
        rng = np.random.default_rng(13)
        q = _pose(model, rng)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)

        misaligned = {
            side: (
                Rotation.from_euler("y", 20.0, degrees=True).as_matrix()
                @ data.xmat[spec["body_id"]].reshape(3, 3)
            )
            for side, spec in r.link_orientation_specs.items()
        }
        before = {side: err for (side, (_, err)) in zip(
            r.link_orientation_specs, r._link_orientation_residuals(misaligned))}

        # Rotate base and targets together, exactly as a rigid turn of the scene.
        spin = Rotation.from_euler("z", 35.0, degrees=True)
        q_spun = q.copy()
        q_spun[3:7] = (spin * Rotation.from_quat(q[3:7], scalar_first=True)).as_quat(scalar_first=True)
        data.qpos[:] = q_spun
        mujoco.mj_forward(model, data)
        spun_targets = {side: spin.as_matrix() @ rot for side, rot in misaligned.items()}
        after = {side: err for (side, (_, err)) in zip(
            r.link_orientation_specs, r._link_orientation_residuals(spun_targets))}

        for side in before:
            np.testing.assert_allclose(after[side], before[side], atol=1e-8)

    def test_disabled_task_produces_no_residuals(self, robot):
        model, data = robot
        r = _retargeter(model, data, enabled=False)
        assert r._link_orientation_residuals({"left": np.eye(3)}) == []

    def test_missing_body_is_rejected_at_construction(self, robot):
        model, data = robot
        r = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
        r.robot_model = model
        r.robot_data = data
        r.q_a_indices = np.arange(0, model.nq)
        with pytest.raises(ValueError, match="not present in the compiled model"):
            r._setup_link_orientation(
                {"enabled": True, "weight": 1.0, "effectors": {"left": {"body": "left_toe_link"}}}
            )

    def test_body_to_sole_is_applied(self, robot):
        model, data = robot
        rot = Rotation.from_euler("x", 15.0, degrees=True)
        r = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
        r.robot_model = model
        r.robot_data = data
        r.q_a_indices = np.arange(0, model.nq)
        r._setup_link_orientation({
            "enabled": True,
            "weight": 1.0,
            "effectors": {"left": {"body": LEFT_FOOT, "body_to_sole": rot.as_matrix().tolist()}},
        })
        rng = np.random.default_rng(17)
        data.qpos[:] = _pose(model, rng)
        mujoco.mj_forward(model, data)

        body_rot = data.xmat[r.link_orientation_specs["left"]["body_id"]].reshape(3, 3)
        # The sole frame, not the body frame, is what must match the target.
        (_, error), = r._link_orientation_residuals({"left": body_rot @ rot.as_matrix()})
        np.testing.assert_allclose(error, np.zeros(3), atol=1e-9)
