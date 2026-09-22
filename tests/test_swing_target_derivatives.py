"""Finite differences for moving free-hand targets and fixed world anchors."""

import cvxpy as cp
import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.retargeting import GenericInteractionRetargeter


@pytest.mark.parametrize("mode", ["relative", "world", "explicit", "contact", "no_root"])
@pytest.mark.parametrize("reach", [None, 0.15, 2.0])
def test_endpoint_residual_derivative_matches_recomputed_target(mode, reach):
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><compiler angle="radian"/><worldbody>
      <body name="root"><freejoint/><geom type="sphere" size=".02"/>
        <body name="hand" pos=".3 .04 .1">
          <joint name="elbow" axis="0 1 0"/>
          <geom type="sphere" size=".02"/>
        </body>
      </body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    data.qpos[:3] = [.1, -.1, .9]
    data.qpos[3:7] = np.array([.95, .1, -.15, .2])
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    data.qpos[7] = .2
    mujoco.mj_forward(model, data)
    name = "left_foot" if mode == "world" else "left_hand"
    spec = dict(robot_body="hand", tracking_offset=[.03, -.02, .01],
                track_swing=True, swing_sigma=1., contact_sigma=1.,
                swing_morphology_scale=.9, preserve_world_target=mode == "world",
                reach_reference_body="root", reach_reference_offset=[.08, 0, .04],
                max_swing_reach=reach)
    config = {"enabled": True, "effectors": {name: spec}}
    if mode == "explicit":
        config["swing_target_z_offset"] = -.2
    retargeter = GenericInteractionRetargeter(
        model, data, trimesh.creation.box(), {"Pelvis": "root"}, 1.,
        self_collision_enabled=False, end_effector_config=config,
    )
    target = dict(contact=mode == "contact", tracking_position=[.5, .2, 1.6],
                  contact_position=[.5, .2, 1.6], surface_normal=[0, 0, 1])
    if mode != "no_root":
        target["source_root_position"] = [0, 0, 1.5]
    targets = {name: target}
    step = cp.Variable(retargeter.nq_a)
    step.value = np.zeros(retargeter.nq_a)
    objective_index = 0 if mode == "contact" else 1

    def residual():
        return retargeter._build_end_effector_objectives(step, targets)[objective_index]

    expression = residual()
    baseline = expression.value.copy()
    original = data.qpos.copy()
    epsilon = 1e-6
    for column, qpos_index in enumerate(retargeter.q_a_indices):
        delta = np.zeros(retargeter.nq_a)
        delta[column] = epsilon
        step.value = delta
        predicted = (expression.value - baseline) / epsilon
        step.value = np.zeros(retargeter.nq_a)
        samples = []
        for sign in (1, -1):
            data.qpos[:] = original
            data.qpos[qpos_index] += sign * epsilon
            data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
            mujoco.mj_forward(model, data)
            samples.append(residual().value.copy())
        actual = (samples[0] - samples[1]) / (2 * epsilon)
        np.testing.assert_allclose(predicted, actual, atol=2e-6,
                                   err_msg=f"qpos column {qpos_index}")
    data.qpos[:] = original
    mujoco.mj_forward(model, data)
