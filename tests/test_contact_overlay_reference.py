"""Frame-zero overlay warm-start regressions."""

import mujoco
import numpy as np
import trimesh

from omniretargeting._core.motion import FramewiseMotionMixin
from tests._contact_fixtures import _task

FREE_ROOT_XML = """
<mujoco>
  <worldbody>
    <body name="root">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_frame_zero_contact_overlay_keeps_baseline_warm_start_and_root_offset(
    monkeypatch,
):
    """An object-contact overlay must start from its solved baseline pose.

    The baseline has a deliberately different position and orientation from
    the source root.  With an initial base offset enabled, the overlay's
    first target should be the baseline pose (source root plus its calibrated
    offset), so local contact-chain refinement cannot reset the robot to the
    source frame before solving.
    """
    model = mujoco.MjModel.from_xml_string(FREE_ROOT_XML)
    driver = FramewiseMotionMixin()
    driver.robot_model = model
    driver.robot_data = mujoco.MjData(model)
    driver.robot_height = 1.0
    driver.retargeting_config = {
        "self_collision_enabled": False,
        "end_effector_priority": {
            "enabled": True,
            "local_contact_chain_only": True,
            "base_position_constraint": {
                "enabled": True,
                "use_initial_offset": True,
                "lock_initial_xy": True,
                "track_xy": True,
                "lock_xy": True,
                "track_z": True,
            },
        },
    }
    driver.valid_joint_mapping = {"Pelvis": "root"}
    driver.valid_joint_names = ["Pelvis"]
    driver.mapped_joint_indices = [0]
    driver.smplx_joint_indices = {"Pelvis": 0}
    driver.smplx_joint_names = ["Pelvis"]
    driver.base_orientation_config = {"root_position": "Pelvis"}
    driver._resolve_source_root_index = lambda: 0
    driver._estimate_base_orientation_from_joints = lambda *_args: None

    source_root = np.array([1.0, 2.0, 3.0])
    processed = source_root[None, None, :]
    baseline = np.array([[10.0, 20.0, 30.0, 0.70710678, 0.0, 0.70710678, 0.0]])
    processed_before = processed.copy()
    baseline_before = baseline.copy()
    contact = _task(
        np.array([[4.0, 5.0, 6.0]]),
        contact_mask=np.array([True]),
    )
    calls = []

    class FakeRetargeter:
        def __init__(self, *_args, **_kwargs):
            self.last_frame_solve = None
            self.surface_contact_tasks = None
            self.robot_reference_qpos = model.qpos0.copy()

        def retarget_frame(self, _human_joints, q_init, **kwargs):
            calls.append((q_init.copy(), kwargs))
            return q_init.copy()

    monkeypatch.setattr(
        "omniretargeting.retargeting.GenericInteractionRetargeter",
        FakeRetargeter,
    )

    driver._perform_retargeting(
        processed,
        trimesh.creation.box(),
        base_orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
        end_effector_tasks={"left_hand": contact},
        reference_trajectory=baseline,
        overlay_contact_names={"left_hand"},
    )

    assert len(calls) == 1
    q_init, kwargs = calls[0]
    np.testing.assert_allclose(q_init, baseline[0], atol=1e-7)
    np.testing.assert_allclose(kwargs["target_base_position"], baseline[0, :3], atol=1e-7)
    np.testing.assert_array_equal(processed, processed_before)
    np.testing.assert_array_equal(baseline, baseline_before)
