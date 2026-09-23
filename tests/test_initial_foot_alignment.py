"""First-frame foot geometry can seed a seated robot without moving targets."""

from types import SimpleNamespace

import mujoco
import numpy as np

from omniretargeting._core.initialization import align_initial_base_height_to_feet


def test_initial_alignment_uses_low_world_foot_and_keeps_source_immutable():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><worldbody><body name="base"><freejoint/>
      <geom type="sphere" size=".01" mass="1"/>
      <body name="left_foot" pos="0 .1 -.5">
        <geom type="sphere" size=".01" mass=".1"/>
      </body>
      <body name="right_foot" pos="0 -.1 -.5">
        <geom type="sphere" size=".01" mass=".1"/>
      </body>
    </body></worldbody></mujoco>
    ''')
    q_init = model.qpos0.copy()
    q_init[2] = 0.1
    targets = {
        'left_foot': np.array([0.0, 0.1, 0.0]),
        'right_foot': np.array([0.0, -0.1, 0.3]),
    }
    saved = {name: value.copy() for name, value in targets.items()}
    tasks = {name: SimpleNamespace(frame=lambda _frame, value=value: {
        'contact': False, 'tracking_position': value,
    }) for name, value in targets.items()}
    effectors = {name: {
        'body_id': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name),
        'tracking_offset': np.zeros(3), 'preserve_world_target': True,
    } for name in targets}

    aligned = align_initial_base_height_to_feet(model, q_init, effectors, tasks)

    np.testing.assert_allclose(aligned[2], 0.5)
    np.testing.assert_array_equal(q_init, model.qpos0 + np.array([0, 0, .1, 0, 0, 0, 0]))
    for name in targets:
        np.testing.assert_array_equal(targets[name], saved[name])


def test_initial_alignment_does_not_lower_a_clear_robot():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><worldbody><body name="base"><freejoint/>
      <geom type="sphere" size=".01" mass="1"/>
      <body name="left_foot" pos="0 0 -.5">
        <geom type="sphere" size=".01" mass=".1"/>
      </body>
    </body></worldbody></mujoco>
    ''')
    q_init = model.qpos0.copy()
    q_init[2] = 0.8
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_foot')
    task = SimpleNamespace(frame=lambda _frame: {
        'contact': True, 'contact_position': np.zeros(3),
    })
    aligned = align_initial_base_height_to_feet(
        model, q_init, {'left_foot': {
            'body_id': body_id, 'tracking_offset': np.zeros(3),
            'preserve_world_target': True,
        }}, {'left_foot': task},
    )
    np.testing.assert_array_equal(aligned, q_init)
