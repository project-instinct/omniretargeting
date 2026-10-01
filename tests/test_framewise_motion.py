"""Framewise execution of _perform_retargeting with a mocked frame solver."""

from unittest.mock import Mock, patch

import numpy as np
import pytest


def test_perform_retargeting_passes_hard_penetration_constraint():
    from omniretargeting import OmniRetargeter
    from unittest.mock import patch

    robot_model = Mock()
    robot_model.nq = 7
    robot_model.njnt = 0
    robot_data = Mock()
    processed_trajectory = np.zeros((0, 22, 3), dtype=float)
    scaled_terrain = Mock()

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = robot_model
    retargeter.robot_data = robot_data
    retargeter.valid_joint_mapping = {"Pelvis": "pelvis"}
    retargeter.robot_height = 1.0
    retargeter.retargeting_config = {
        "collision_detection_threshold": 0.2,
        "terrain_sample_points": 123,
        "replace_cylinders_with_capsules": True,
        "penetration_resolver": "xyz_nudge",
    }
    retargeter.valid_joint_names = ["Pelvis"]
    retargeter.base_orientation_config = {}

    with patch("omniretargeting.retargeting.GenericInteractionRetargeter") as retargeter_cls:
        retargeter_instance = Mock()
        retargeter_cls.return_value = retargeter_instance
        result = retargeter._perform_retargeting(processed_trajectory, scaled_terrain)

    assert result.shape == (0,)
    retargeter_cls.assert_called_once_with(
        robot_model,
        robot_data,
        scaled_terrain,
        {"Pelvis": "pelvis"},
        1.0,
        collision_detection_threshold=0.2,
        terrain_sample_points=123,
        terrain_sample_seed=None,
        valid_joint_names=["Pelvis"],
        replace_cylinders_with_capsules=True,
        hard_penetration_constraint=False,
    )


@pytest.mark.parametrize("lock_initial_xy", [False, True])
def test_animation_style_root_motion_uses_one_calibrated_initial_offset(
    lock_initial_xy,
):
    from omniretargeting import OmniRetargeter

    source_roots = np.asarray(
        [
            [10.0, 20.0, 1.0],
            [11.0, 21.0, 1.1],
            [12.0, 23.0, 1.4],
        ]
    )
    processed_trajectory = source_roots[:, None, :]
    solved_qpos = [
        np.asarray([1.0, 2.0, 0.8, 1.0, 0.0, 0.0, 0.0]),
        np.asarray([2.1, 3.2, 0.9, 1.0, 0.0, 0.0, 0.0]),
        np.asarray([3.2, 5.1, 1.1, 1.0, 0.0, 0.0, 0.0]),
    ]

    robot_model = Mock(nq=7, njnt=0)
    robot_model.qpos0 = np.asarray([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = robot_model
    retargeter.robot_data = Mock()
    retargeter.valid_joint_mapping = {"Pelvis": "pelvis"}
    retargeter.valid_joint_names = ["Pelvis"]
    retargeter.mapped_joint_indices = [0]
    retargeter.smplx_joint_indices = {"Pelvis": 0}
    retargeter.smplx_joint_names = ["Pelvis"]
    retargeter.base_orientation_config = {"root_position": "Pelvis"}
    retargeter.robot_height = 1.0
    retargeter.retargeting_config = {
        "end_effector_priority": {
            "base_position_constraint": {
                "enabled": True,
                "use_initial_offset": True,
                "lock_initial_xy": lock_initial_xy,
                "track_xy": True,
                "lock_xy": True,
                "track_z": False,
            }
        }
    }
    task = Mock()
    task.frame.side_effect = [
        {"contact": False, "contact_mode": "none"} for _ in source_roots
    ]
    base_orientations = np.tile([1.0, 0.0, 0.0, 0.0], (len(source_roots), 1))

    with patch(
        "omniretargeting.retargeting.GenericInteractionRetargeter"
    ) as retargeter_cls:
        solver = Mock()
        solver.retarget_frame.side_effect = solved_qpos
        retargeter_cls.return_value = solver
        result = retargeter._perform_retargeting(
            processed_trajectory,
            Mock(),
            base_orientations=base_orientations,
            end_effector_tasks={"left_foot": task},
        )

    assert np.allclose(result, solved_qpos)
    first_kwargs = solver.retarget_frame.call_args_list[0].kwargs
    second_kwargs = solver.retarget_frame.call_args_list[1].kwargs
    third_kwargs = solver.retarget_frame.call_args_list[2].kwargs
    if lock_initial_xy:
        np.testing.assert_allclose(
            first_kwargs["target_base_position"], source_roots[0]
        )
    else:
        assert "target_base_position" not in first_kwargs
    # The first solve calibrates only the horizontal robot-source offset.
    # Source Z remains unregistered because the root-height solve is free.
    np.testing.assert_allclose(second_kwargs["target_base_position"], [2.0, 3.0, 1.1])
    np.testing.assert_allclose(third_kwargs["target_base_position"], [3.0, 5.0, 1.4])
