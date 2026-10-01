"""Batched orientation rows retain world and parent-relative derivatives."""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._sequence.orientation import OrientationTargets
from tests.test_fast_backend import FLOATING_XML


@pytest.mark.parametrize("columns", [np.arange(10), np.array([8, 2, 6]), np.array([], dtype=int)])
def test_orientation_rows_match_finite_differences(columns):
    model = mujoco.MjModel.from_xml_string(FLOATING_XML)
    data = mujoco.MjData(model)
    target = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()[None]
    specifications = [
        {"robot_body": "base", "rotations": target, "sigma": 0.7},
        {"robot_body": "thigh", "parent_body": "base", "rotations": target, "sigma": 0.3},
        {"robot_body": "shin", "parent_body": "base", "rotations": target, "sigma": 0.2},
        {"robot_body": "hand", "rotations": target, "sigma": 0.5},
    ]
    targets = OrientationTargets(model, 1, specifications)
    pose = model.qpos0.copy()
    pose[7:] = [0.2, -0.6, 0.1, 0.4]
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    _, jacobians, _ = targets.rows(data, columns, 0)
    derivative = np.vstack(jacobians)

    def residual(q):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        rows = []
        for spec in specifications:
            current = data.xmat[model.body(spec["robot_body"]).id].reshape(3, 3)
            if "parent_body" in spec:
                parent = data.xmat[model.body(spec["parent_body"]).id].reshape(3, 3)
                current = parent.T @ current
            rows.extend(((current - target[0]).T / (np.sqrt(2.0) * spec["sigma"])).ravel())
        return np.asarray(rows)

    for index, column in enumerate(columns):
        velocity = np.zeros(model.nv)
        velocity[column] = 1.0
        before, after = pose.copy(), pose.copy()
        mujoco.mj_integratePos(model, before, velocity, -1e-6)
        mujoco.mj_integratePos(model, after, velocity, 1e-6)
        numerical = (residual(after) - residual(before)) / 2e-6
        np.testing.assert_allclose(derivative[:, index], numerical, rtol=1e-7, atol=1e-8)
    assert derivative.shape == (36, len(columns))


def test_no_orientation_targets_has_no_rows():
    model = mujoco.MjModel.from_xml_string(FLOATING_XML)
    targets = OrientationTargets(model, 1, [])
    assert targets.rows(mujoco.MjData(model), np.arange(model.nv), 0) == ([], [], 0.0)
