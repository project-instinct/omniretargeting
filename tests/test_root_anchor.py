"""A source root names a mapped anatomical point, not a model's free joint."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._core.sequence import solve_core_sequence
from omniretargeting.sequence_retargeting import prepare_sequence, sequence_config


@pytest.mark.parametrize("calibrated_floating_body", [False, True])
def test_cold_seed_aligns_corresponding_body_for_an_offset_floating_base(
    monkeypatch, calibrated_floating_body
):
    from omniretargeting._core import sequence

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="base"><freejoint/><geom size=".01"/>'
        '<body name="anchor" pos=".1 0 -.3"><geom size=".01"/></body>'
        "</body></worldbody></mujoco>"
    )
    roots = np.array([[1, 2, 0.6], [1.1, 2.1, 0.7]])
    orientations = Rotation.from_rotvec([[0, 0, 0], [0, 0.4, 0.3]]).as_quat(
        scalar_first=True
    )
    original_orientations = orientations.copy()
    calibrated = Rotation.from_rotvec([[0.2, 0, 0], [0, -0.2, 0.4]])
    owner = SimpleNamespace(
        robot_model=model,
        _source_framerate=30,
        _resolve_source_root_index=lambda: 0,
        retargeting_config={"end_effector_priority": {}},
        smplx_joint_names=["source_anchor"],
        valid_joint_mapping={"source_anchor": "anchor"},
    )
    solver = SimpleNamespace(
        robot_reference_qpos=model.qpos0,
        terrain_points=np.empty((0, 3)),
        scene_query=None,
    )
    seen = []

    def capture(model, initial, *args, **kwargs):
        if not seen:
            data = mujoco.MjData(model)
            for f, pose in enumerate(initial):
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                np.testing.assert_allclose(
                    data.xpos[model.body("anchor").id], roots[f], atol=1e-12
                )
                expected = (
                    calibrated.as_quat(scalar_first=True)
                    if calibrated_floating_body
                    else orientations
                )
                np.testing.assert_allclose(
                    kwargs["root_orientation_targets"], expected, atol=1e-12
                )
            assert args[-1]["source_root_robot_body"] == "anchor"
        seen.append(initial.copy())
        return initial, {"contact_offset_choice": {}}

    monkeypatch.setattr(sequence, "retarget_sequence", capture)
    monkeypatch.setattr(
        sequence,
        "sequence_body_orientation_targets",
        lambda *args: (
            [{"robot_body": "base", "rotations": calibrated.as_matrix(), "sigma": 1.0}]
            if calibrated_floating_body
            else ()
        ),
    )
    solve_core_sequence(owner, solver, roots[:, None], {}, orientations, None)
    assert seen
    np.testing.assert_array_equal(orientations, original_orientations)


def test_articulated_root_anchor_has_its_own_position_jacobian():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="base"><freejoint/><geom size=".01"/>'
        '<body name="hinged"><joint axis="0 1 0"/><geom size=".01"/>'
        '<body name="anchor" pos="0 0 -.3"><geom size=".01"/></body>'
        "</body></body></worldbody></mujoco>"
    )
    q = model.qpos0[None].copy()
    source = np.array([[[0, 0, -0.3]]])
    coords, _, tasks, _ = prepare_sequence(
        model,
        q,
        source,
        ["a"],
        {"a": "anchor"},
        source[:, 0],
        {},
        {"source_root_robot_body": "anchor"},
        sequence_config({"self_collision_enabled": False}),
    )
    tasks.data.qpos[:] = q[0]
    mujoco.mj_forward(model, tasks.data)
    point, jac = tasks.root_anchor(0)
    step = np.zeros((1, coords.width))
    step[0, -1] = 1e-6
    moved = coords.integrate(q, step, 1)
    tasks.data.qpos[:] = moved[0]
    mujoco.mj_forward(model, tasks.data)
    actual, _ = tasks.root_anchor(0)
    np.testing.assert_allclose((actual - point) / 1e-6, jac[:, -1], atol=2e-7)
    assert abs(jac[0, -1]) > 0.29
