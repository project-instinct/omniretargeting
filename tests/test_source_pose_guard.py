from types import SimpleNamespace
import mujoco
import numpy as np
from omniretargeting._sequence.pose_guard import SequencePoseGuard
from omniretargeting._sequence.coordinates import Coordinates


def test_source_direction_change_is_reported_not_bounded():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/><worldbody>
    <body name="root"><freejoint/><geom size=".01"/><body name="lever"><joint axis="0 0 1" range="-2 2"/><geom size=".01"/>
    <body name="tip" pos="1 0 0"><geom size=".01"/></body></body></body></worldbody></mujoco>""")
    q = model.qpos0.copy()
    q[-1] = 0.4
    tasks = SimpleNamespace(
        model=model,
        coords=Coordinates(model),
        bodyids=[model.body("root").id, model.body("tip").id],
        effectors={},
        config={"translation_step_m": 0.03},
        human=np.array([[[0, 0, 0], [1, 0, 0]]]),
    )
    guard = SequencePoseGuard(tasks, q[None])
    assert guard.accepts(q[None])
    worse = q.copy()
    worse[-1] += 0.5
    assert guard.accepts(worse[None])
    before = guard.diagnostics(q[None])
    after = guard.diagnostics(worse[None])
    assert abs(before["maximum_source_bone_angle_rad"] - 0.4) < 1e-9
    assert abs(after["maximum_source_bone_angle_rad"] - 0.9) < 1e-9
    assert abs(after["maximum_joint_change_rad"] - 0.5) < 1e-12
    assert after["angular_bound"] is None


def test_root_translation_rows_keep_the_exact_bound_and_jacobian():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/><worldbody>
      <body name="base"><freejoint/><geom size=".02"/>
      <body name="a"><joint axis="0 1 0" range="-2 2"/><geom size=".02"/>
      <body name="tip" pos="0 0 1"><geom size=".02"/></body></body></body>
      </worldbody></mujoco>""")
    coordinates = Coordinates(model)
    reference = model.qpos0[None].copy()
    tasks = SimpleNamespace(
        model=model,
        coords=coordinates,
        bodyids=[model.body("base").id, model.body("tip").id],
        effectors={},
        config={"translation_step_m": 0.03},
        human=np.array([[[0, 0, 0], [0, 0, 1]]]),
    )
    guard = SequencePoseGuard(tasks, reference)
    step = np.zeros((1, coordinates.width))
    step[0, :3] = [0.01, -0.015, 0.005]
    step[0, -1] = 0.04
    current = coordinates.integrate(reference, step, 1)
    baseline = list(guard.rows(current, jacobians=True))
    assert len(baseline) == 1
    for column in (0, 1, 2, coordinates.width - 1):
        direction = np.zeros_like(step)
        direction[0, column] = 1e-6
        shifted = coordinates.integrate(current, direction, 1)
        changed = list(guard.rows(shifted, jacobians=False))
        for (_, value, jac, _), (_, after, _, _) in zip(baseline, changed):
            np.testing.assert_allclose(
                (after - value) / 1e-6, jac[:, column], atol=6e-7
            )
    for shift, accepted in [(0.0299, True), (0.0301, False)]:
        q = reference.copy()
        q[0, 0] += shift
        q[0, -1] = 1.0  # a joint change beyond the former 0.2 rad bound
        assert guard.accepts(q) == accepted
