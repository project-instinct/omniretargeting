"""Fast reachable source motion must not be capped by legacy settings."""

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.utils import calculate_laplacian_coordinates
from tests.test_body_orientation_reference import XML


@pytest.mark.parametrize("legacy_limits", [False, True])
@pytest.mark.parametrize("desired", [-0.45, 0.45])
def test_fast_joint_target_has_no_fixed_framewise_cap(legacy_limits, desired):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    solver = GenericInteractionRetargeter(
        model, data, trimesh.creation.box(), {"root": "base", "end": "tip"}, 0.5,
        terrain_sample_seed=0,
        end_effector_config={
            "enabled": True,
            "temporal_weight": 0.0,
            "lock_root_orientation_to_source": True,
            "base_position_constraint": {
                "enabled": True, "track_xy": True, "lock_xy": True,
                "track_z": True, "max_z_error": 1e-4, "centering_weight": 0.0,
            },
            "effectors": {
                "marker": {
                    "robot_body": "tip", "preserve_world_target": True,
                    "swing_sigma": 0.01,
                },
            },
            "temporal": {
                "enabled": True,
            },
        },
    )
    if legacy_limits:
        # Old saved profiles cannot silently restore the removed constraints.
        solver.end_effector_config["temporal"].update({
            "hard_joint_motion_limits_enabled": True,
            "max_joint_step": 0.18,
            "max_joint_acceleration_step": 0.08,
            "hard_jerk_enabled": True,
            "max_joint_jerk_step": 0.005,
        })
    solver.smooth_weight = 0.0
    solver.Q_diag[:] = 0.0
    previous = model.qpos0.copy()
    data.qpos[7] = desired
    mujoco.mj_forward(model, data)
    positions = np.array([data.xpos[model.body(n).id] for n in ("base", "tip")])
    adjacency = [[1], [0]]
    laplacian = calculate_laplacian_coordinates(positions, adjacency, uniform_weight=True)
    q = previous.copy()
    for _ in range(8):
        q, cost = solver._single_optimization_step(
            q, laplacian, adjacency, np.empty((0, 3)),
            target_human_joints=positions,
            q_last=previous, q_prev=previous, q_prevprev=previous,
            target_base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            target_base_position=np.zeros(3),
            end_effector_targets={
                "marker": {"contact": False, "tracking_position": positions[1]},
            },
        )
        assert np.isfinite(cost)
    assert q[7] == pytest.approx(desired, abs=0.01)
    assert model.jnt_range[1, 0] <= q[7] <= model.jnt_range[1, 1]
