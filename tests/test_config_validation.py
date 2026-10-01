import pytest

from omniretargeting.self_collision import self_collision_config
from omniretargeting.self_contact import self_contact_config
from omniretargeting.sequence_retargeting import sequence_config
from omniretargeting.surface_contact import surface_contact_config
from omniretargeting.surface_learning import correspondence_config
from omniretargeting.surface_pose import surface_pose_config
from omniretargeting.trajectory_validation import local_repair_config
from omniretargeting.whole_body_correction import whole_body_config
from omniretargeting.retargeting_defaults import apply_retargeting_defaults


@pytest.mark.parametrize(
    "configure",
    [
        self_collision_config,
        self_contact_config,
        sequence_config,
        surface_contact_config,
        correspondence_config,
        surface_pose_config,
        local_repair_config,
        whole_body_config,
    ],
)
def test_unknown_algorithm_option_fails_early(configure):
    with pytest.raises(ValueError, match="Unknown"):
        configure({"distnace": 0.03})


def test_profile_defaults_are_independent_across_robots_and_calls():
    first = apply_retargeting_defaults({})
    first["end_effector_priority"]["temporal"]["enabled"] = False
    first["surface_contact"]["enabled"] = False
    second = apply_retargeting_defaults(
        {"end_effector_priority": {"effectors": {"left_hand": {"robot_body": "tool"}}}}
    )
    assert second["surface_contact"]["enabled"]
    assert second["end_effector_priority"]["temporal"]["enabled"]
    assert "preserve_world_target" not in second["end_effector_priority"]["effectors"]["left_hand"]
