"""Recorded BVH wrist twists must survive loading and robot calibration."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting._core.orientation_targets import sequence_body_orientation_targets
from omniretargeting.data_sources.lafan1 import (
    _ROTATION_MATRIX,
    Lafan1DataSource,
)
from omniretargeting.data_sources.lafan1_reference import lafan1_reference
from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.robot_config import load_robot_config
from omniretargeting.utils import load_robot_urdf_with_floating_base
from tests.data_sources.test_lafan1 import MINIMAL_BVH


def test_world_rotations_follow_intrinsic_bvh_channels_and_world_transform(tmp_path):
    lines = MINIMAL_BVH.splitlines()
    values = np.zeros(18)
    values[:3] = [0.0, 90.0, 0.0]
    values[3:6] = [37.0, -21.0, 19.0]
    values[6:9] = [13.0, 27.0, -9.0]
    values[9:12] = [-4.0, 42.0, 31.0]
    lines[-3:] = [" ".join(map(str, values))] * 3
    path = tmp_path / "rotated.bvh"
    path.write_text("\n".join(lines))
    motion = Lafan1DataSource(path).load()
    rotations = motion.metadata["joint_orientations"]
    expected = _ROTATION_MATRIX.copy()
    for index in range(3):
        local = Rotation.from_euler(
            "ZXY", values[3 + 3 * index : 6 + 3 * index], degrees=True
        )
        expected = expected @ local.as_matrix()
        actual = Rotation.from_quat(rotations[:, index], scalar_first=True).as_matrix()
        np.testing.assert_allclose(
            actual, np.broadcast_to(expected, actual.shape), atol=3e-7
        )
    np.testing.assert_allclose(np.linalg.norm(rotations, axis=-1), 1.0, atol=1e-7)


def test_leaf_twist_is_observed_even_when_every_joint_position_is_identical(tmp_path):
    lines = MINIMAL_BVH.splitlines()
    frames = []
    for twist in (0.0, 45.0, 90.0):
        values = np.zeros(18)
        values[1] = 90.0
        values[10] = twist  # leaf Head Xrotation; no descendant landmark observes it
        frames.append(" ".join(map(str, values)))
    lines[-3:] = frames
    path = tmp_path / "leaf.bvh"
    path.write_text("\n".join(lines))
    motion = Lafan1DataSource(path).load()
    np.testing.assert_array_equal(motion.positions[0], motion.positions[2])
    quats = motion.metadata["joint_orientations"]
    relative = Rotation.from_quat(
        quats[0, 2], scalar_first=True
    ).inv() * Rotation.from_quat(quats[2, 2], scalar_first=True)
    assert relative.magnitude() == pytest.approx(np.pi / 2, abs=2e-7)
    trimmed = Lafan1DataSource(path, start_frame=1).load()
    np.testing.assert_array_equal(trimmed.metadata["joint_orientations"], quats[1:])
    resampled = motion.resample(motion.framerate * 2)
    angles = Rotation.from_quat(
        quats[0, 2], scalar_first=True
    ).inv() * Rotation.from_quat(
        resampled.metadata["joint_orientations"][:, 2], scalar_first=True
    )
    np.testing.assert_allclose(
        angles.magnitude(), np.deg2rad([0.0, 22.5, 45.0, 67.5, 90.0]), atol=2e-7
    )
    assert "joint_rest_positions" not in motion.metadata


def _standard_reference():
    names = [
        "Hips",
        "Spine",
        "Spine2",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
    ]
    parents = np.array([-1, 0, 1, 2, 3, 4, 5, 2, 7, 8, 9, 0, 11, 12, 0, 14, 15])
    offsets = np.tile([30.0, 0.0, 0.0], (len(names), 1))
    offsets[0] = 0.0
    offsets[11], offsets[14] = [0.0, 0.0, 10.0], [0.0, 0.0, -10.0]
    return names, parents, offsets, lafan1_reference(names, parents, offsets)


def test_anatomical_reference_preserves_lengths_and_mirrors_limbs():
    names, parents, offsets, reference = _standard_reference()
    positions = reference["joint_rest_positions"]
    rotations = Rotation.from_quat(
        reference["joint_rest_orientations"], scalar_first=True
    ).as_matrix()
    for index, parent in enumerate(parents):
        if parent >= 0:
            np.testing.assert_allclose(
                positions[index] - positions[parent],
                rotations[parent] @ offsets[index] / 100.0,
                atol=1e-12,
            )
    mirror = np.diag([1.0, -1.0, 1.0])
    for suffix in ("Arm", "ForeArm", "Hand", "UpLeg", "Leg", "Foot"):
        left, right = (names.index(side + suffix) for side in ("Left", "Right"))
        np.testing.assert_allclose(
            positions[left], mirror @ positions[right], atol=1e-12
        )
    assert positions[names.index("LeftHand"), 1] > positions[names.index("LeftArm"), 1]
    assert positions[names.index("LeftLeg"), 2] < positions[names.index("LeftUpLeg"), 2]
    for side in ("Left", "Right"):
        foot = names.index(side + "Foot")
        np.testing.assert_allclose(rotations[foot, :, 0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotations[foot, :, 1], [0.0, 0.0, 1.0])


def test_reference_retains_static_toe_rig_rotation_but_never_dynamic_toe_motion():
    names, parents, offsets, _ = _standard_reference()
    ankle_ids = [names.index(side + "Foot") for side in ("Left", "Right")]
    names += ["LeftToe", "RightToe"]
    parents = np.r_[parents, ankle_ids]
    offsets = np.vstack([offsets, [17.0, 0.0, 0.0], [17.0, 0.0, 0.0]])
    local = np.tile([1.0, 0.0, 0.0, 0.0], (3, len(names), 1))
    toe_rotation = Rotation.from_euler("z", 0.375)
    local[:, -2:] = toe_rotation.as_quat(scalar_first=True)
    reference = lafan1_reference(names, parents, offsets, local)
    rotations = Rotation.from_quat(
        reference["joint_rest_orientations"], scalar_first=True
    ).as_matrix()
    for ankle, toe in zip(ankle_ids, (len(names) - 2, len(names) - 1)):
        np.testing.assert_allclose(
            rotations[ankle] @ toe_rotation.as_matrix(), rotations[toe], atol=1e-12
        )
        assert (
            reference["joint_rest_positions"][toe, 2]
            < reference["joint_rest_positions"][ankle, 2]
        )
    # The entire file supplies calibration, so a crop cannot redefine it.
    cropped = lafan1_reference(names, parents, offsets, local[1:])
    np.testing.assert_array_equal(
        cropped["joint_rest_orientations"], reference["joint_rest_orientations"]
    )
    local[1, -2:] = Rotation.from_euler("z", 0.7).as_quat(scalar_first=True)
    dynamic = lafan1_reference(names, parents, offsets, local)
    canonical = lafan1_reference(names, parents, offsets)
    np.testing.assert_array_equal(
        dynamic["joint_rest_orientations"], canonical["joint_rest_orientations"]
    )


def test_g1_wrist_targets_include_recorded_twist_and_all_three_axes():
    names, _, _, reference = _standard_reference()
    profile = load_robot_config(
        Path(__file__).resolve().parents[2]
        / "robot_models/unitree_g1/unitree_g1_lafan1.json"
    )
    model = load_robot_urdf_with_floating_base(Path(profile["urdf_path"]))
    cfg = profile["retargeting"]["end_effector_priority"]
    cfg["body_orientation"]["_source_orientations_available"] = True
    solver = GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        trimesh.creation.box(),
        profile["joint_mapping"],
        1.75,
        valid_joint_names=list(profile["joint_mapping"]),
        end_effector_config=cfg,
        terrain_sample_seed=0,
    )
    rest = reference["joint_rest_orientations"]
    solver.configure_source_orientation_reference(
        names,
        rest,
        reference["joint_rest_positions"],
        reference_base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        preserve_core_reference=True,
    )
    source = np.tile(rest, (4, 1, 1))
    hand = names.index("LeftHand")
    twists = np.array(
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, -0.3, 0.0], [0.0, 0.0, 0.2]]
    )
    source[:, hand] = (
        Rotation.from_quat(rest[hand], scalar_first=True) * Rotation.from_rotvec(twists)
    ).as_quat(scalar_first=True)
    targets = {
        t["robot_body"]: t
        for t in sequence_body_orientation_targets(solver, source, None)
    }
    task = targets["left_wrist_yaw_link"]
    assert task["parent_body"] == "left_elbow_link"
    target = Rotation.from_matrix(task["rotations"])
    angles = (target[0].inv() * target).magnitude()
    np.testing.assert_allclose(angles, np.linalg.norm(twists, axis=1), atol=1e-10)
    # A twist-only motion now drives the three physical wrist joints rather
    # than being indistinguishable from a stationary hand.
    from omniretargeting._sequence.coordinates import Coordinates
    from omniretargeting._sequence.orientation import OrientationTargets

    coords = Coordinates(model)
    data = mujoco.MjData(model)
    data.qpos[:] = solver.robot_reference_qpos
    mujoco.mj_forward(model, data)
    rows = OrientationTargets(model, 4, [task]).rows(data, coords.di, 1)
    columns = [
        int(
            np.flatnonzero(
                coords.di
                == model.jnt_dofadr[model.joint("left_wrist_" + axis + "_joint").id]
            )[0]
        )
        for axis in ("roll", "pitch", "yaw")
    ]
    assert np.linalg.norm(rows[0][0]) > 0.1
    assert np.linalg.matrix_rank(rows[1][0][:, columns]) == 3
