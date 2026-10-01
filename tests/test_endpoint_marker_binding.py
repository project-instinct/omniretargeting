"""Source marker semantics survive arbitrary robot endpoint names."""

from types import SimpleNamespace
import numpy as np
from omniretargeting._core.contact_inputs import endpoint_marker, unified_contact_inputs
from tests.test_source_contacts import skin, floor


def test_renamed_endpoint_uses_original_calibrated_marker_and_keeps_source():
    points = np.tile([0.0, 0.0, 0.005], (5, 1))
    full = skin(points)
    full.update(
        joint_names=["root", "tip"],
        joint_parents=[-1, 0],
        sample_joint_indices=[1],
        sample_effectors=["material_patch"],
        world_target_joints=["tip"],
    )
    tracking = points + [0.02, 0, 0.01]
    marker = dict(
        tracking_positions=tracking,
        forward_directions=np.tile([1.0, 0, 0], (5, 1)),
        up_directions=np.tile([0.0, 0, 1], (5, 1)),
    )
    original = full["triangles"].copy()
    owner = SimpleNamespace(
        smplx_joint_names=["root", "tip"],
        retargeting_config={
            "end_effector_priority": {
                "effectors": {"arbitrary_endpoint": {"source_joint": "tip"}}
            }
        },
    )
    tasks, _ = unified_contact_inputs(
        owner,
        {"full_body": full},
        np.tile(points[:, None], (1, 2, 1)),
        floor(),
        (),
        {"material_patch": marker},
        30.0,
    )
    task = tasks["arbitrary_endpoint"]
    np.testing.assert_array_equal(task.tracking_positions, tracking)
    np.testing.assert_array_equal(task.forward_directions, marker["forward_directions"])
    np.testing.assert_array_equal(task.up_directions, marker["up_directions"])
    np.testing.assert_array_equal(full["triangles"], original)
    assert task.preserve_world_tracking


def test_ambiguous_material_labels_do_not_invent_a_semantic_frame():
    markers = {"a": {"value": 1}, "b": {"value": 2}}
    assert endpoint_marker(markers, "renamed", ["a", "b"], [0, 1]) == {}
    assert endpoint_marker(markers, "a", ["a", "b"], [0, 1]) is markers["a"]
