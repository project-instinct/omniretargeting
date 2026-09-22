"""A shortened SQP update must not shorten prescribed source translation."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh

from tests.test_self_collision_pipeline import motion_driver
from tests._contact_fixtures import _task


@pytest.mark.parametrize('predict', [False, True])
def test_collision_backtracking_starts_on_locked_path_without_mutating_history(monkeypatch, predict):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="root"><freejoint/><geom size=".1"/></body>
    </worldbody></mujoco>''')
    driver = motion_driver(model)
    driver.retargeting_config = {'end_effector_priority': {
        'source_root_prediction': predict,
        'base_position_constraint': {'enabled': True, 'use_initial_offset': True,
                                    'lock_initial_xy': True, 'lock_xy': True,
                                    'track_xy': True, 'track_z': False},
    }}
    source = np.array([[[0., 0., 1.]], [[.2, -.1, 1.1]], [[.3, -.4, 1.2]]])
    source.setflags(write=False)
    histories = []

    def shortened_step(joints, initial, **kwargs):
        histories.append(None if kwargs['q_last'] is None else kwargs['q_last'].copy())
        proposal = initial.copy()
        proposal[:2] = kwargs['target_base_position'][:2]
        proposal[2] += .04
        # Mimic an atomic collision backtrack. XY must already be feasible.
        return initial + .125 * (proposal - initial)

    fake = SimpleNamespace(retarget_frame=shortened_step)
    monkeypatch.setattr('omniretargeting.retargeting.GenericInteractionRetargeter',
                        lambda *a, **kw: fake)
    task = _task(source[:, 0].copy(), np.zeros(3, bool))
    result = driver._perform_retargeting(source, trimesh.creation.box(),
                                        end_effector_tasks={'left_hand': task})
    np.testing.assert_allclose(result[:, :2], source[:, 0, :2], atol=1e-12)
    for i in range(1, len(source)):
        np.testing.assert_array_equal(histories[i], result[i - 1])
