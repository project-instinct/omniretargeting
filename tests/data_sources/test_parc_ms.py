"""Tests for the hhtools/PARC MS pickle data-source adapter."""

from __future__ import annotations

import pickle

import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.parc_ms import (
    PARC_MS_FOOT_CONTACT_OFFSET_M,
    PARC_MS_BASE_ORIENTATION,
    PARC_MS_JOINT_MAPPING,
    PARC_MS_LOCAL_TRANSLATIONS,
    PARC_MS_PARENT_INDICES,
    PARC_MS_TARGET_NAMES,
    ParcMsDataSource,
    forward_kinematics_parc_ms,
)
from omniretargeting.main import select_source_base_orientations
from omniretargeting.data_sources.registry import create_data_source


def _identity_quaternions(shape):
    values = np.zeros((*shape, 4), dtype=np.float32)
    values[..., 3] = 1.0
    return values


def _write_ms_pickle(tmp_path, *, root_positions=None, body_contacts=None):
    if root_positions is None:
        root_positions = np.asarray(
            [[1.0, 2.0, 0.9], [1.1, 2.0, 0.9]], dtype=np.float32
        )
    num_frames = len(root_positions)
    if body_contacts is None:
        body_contacts = np.zeros((num_frames, 15), dtype=bool)
    motion = {
        "root_pos": root_positions,
        "root_rot": _identity_quaternions((num_frames,)),
        "joint_rot": _identity_quaternions((num_frames, 14)),
        "body_contacts": np.asarray(body_contacts, dtype=bool),
        "fps": 30,
        "loop_mode": "CLAMP",
    }
    terrain = {
        "hf": np.zeros((2, 2), dtype=np.float32),
        "hf_maxmin": np.zeros((2, 2, 2), dtype=np.float32),
        "min_point": np.zeros(2, dtype=np.float32),
        "dx": 0.1,
    }
    container = {
        "motion_data": pickle.dumps(motion),
        "terrain_data": pickle.dumps(terrain),
        "misc_data": pickle.dumps({"test": True}),
    }
    path = tmp_path / "clip.pkl"
    with path.open("wb") as file:
        pickle.dump(container, file)
    (tmp_path / "clip_terrain.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
    )
    return path


def _rest_positions(root_position):
    positions = np.zeros((15, 3), dtype=np.float32)
    positions[0] = root_position
    for joint_index in range(1, 15):
        positions[joint_index] = (
            positions[PARC_MS_PARENT_INDICES[joint_index]]
            + PARC_MS_LOCAL_TRANSLATIONS[joint_index]
        )
    return positions


def test_identity_forward_kinematics_matches_parc_rest_offsets():
    roots = np.asarray([[1.0, 2.0, 0.9]], dtype=np.float32)
    positions, rotations = forward_kinematics_parc_ms(
        roots,
        _identity_quaternions((1,)),
        _identity_quaternions((1, 14)),
    )

    assert np.allclose(positions[0], _rest_positions(roots[0]))
    assert np.allclose(rotations, _identity_quaternions((1, 15)))


def test_data_source_loads_canonical_targets_and_metadata(tmp_path):
    path = _write_ms_pickle(tmp_path)
    motion = ParcMsDataSource(path).load()

    assert motion.positions.shape == (2, 15, 3)
    assert motion.target_names == list(PARC_MS_TARGET_NAMES)
    assert motion.root_orientations.shape == (2, 4)
    assert np.allclose(motion.root_orientations[:, 0], 1.0)
    assert motion.framerate == 30.0
    assert motion.metadata["terrain_data_present"] is True
    assert motion.metadata["terrain_mesh"] == str(
        (tmp_path / "clip_terrain.obj").resolve()
    )
    assert motion.metadata["body_contacts"].shape == (2, 15)
    assert motion.metadata["torso_mapping"] == "pelvis_shoulders"
    assert motion.metadata["joint_mapping"] == PARC_MS_JOINT_MAPPING
    assert motion.metadata["base_orientation"] == PARC_MS_BASE_ORIENTATION
    assert motion.metadata["joint_rest_orientations"].shape == (15, 4)
    assert motion.metadata["joint_rest_positions"].shape == (15, 3)
    np.testing.assert_allclose(
        motion.metadata["joint_rest_positions"],
        _rest_positions(np.zeros(3, dtype=np.float32)),
    )
    assert np.allclose(
        motion.metadata["joint_rest_orientations"],
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )


def test_source_base_orientation_can_follow_torso_instead_of_pelvis(tmp_path):
    motion = ParcMsDataSource(_write_ms_pickle(tmp_path)).load()
    spine_index = motion.target_names.index("Spine1")
    torso_quaternion = np.asarray([0.9238795, 0.0, 0.3826834, 0.0])
    motion.metadata["joint_orientations"][:, spine_index] = torso_quaternion

    selected, source_joint = select_source_base_orientations(
        motion, {"base_orientation_joint": "Spine1"}
    )

    assert source_joint == "Spine1"
    np.testing.assert_allclose(
        selected, np.tile(torso_quaternion, (len(motion.positions), 1))
    )
    assert not np.allclose(selected, motion.root_orientations)


def test_parc_default_torso_frame_tracks_rest_calibrated_landmarks(tmp_path):
    motion = ParcMsDataSource(_write_ms_pickle(tmp_path)).load()
    reference_positions = np.asarray(
        motion.metadata["joint_rest_positions"], dtype=float
    )
    expected_rotation = Rotation.from_euler(
        "zyx", [37.0, -24.0, 11.0], degrees=True
    )
    rotated_positions = expected_rotation.apply(reference_positions)
    motion.positions[:] = rotated_positions[None, ...] + np.asarray(
        [1.2, -0.4, 0.8]
    )

    selected, source_body = select_source_base_orientations(motion, {})

    assert source_body == "Spine1"
    selected_rotations = Rotation.from_quat(
        selected, scalar_first=True
    ).as_matrix()
    np.testing.assert_allclose(
        selected_rotations,
        np.tile(expected_rotation.as_matrix(), (len(selected), 1, 1)),
        atol=1e-6,
    )


def test_explicit_joint_frame_keeps_legacy_mode_over_parc_default(tmp_path):
    motion = ParcMsDataSource(_write_ms_pickle(tmp_path)).load()
    spine_index = motion.target_names.index("Spine1")
    torso_quaternion = np.asarray([0.9238795, 0.0, 0.3826834, 0.0])
    motion.metadata["joint_orientations"][:, spine_index] = torso_quaternion

    selected, source_body = select_source_base_orientations(
        motion, {"base_orientation_joint": "Spine1"}
    )

    assert source_body == "Spine1"
    np.testing.assert_allclose(
        selected, np.tile(torso_quaternion, (len(selected), 1))
    )


def test_origin_bind_lead_in_is_trimmed_with_all_frame_metadata(tmp_path):
    path = _write_ms_pickle(
        tmp_path,
        root_positions=np.asarray(
            [[0.0, 0.0, 0.9], [1.0, 2.0, 0.9], [1.1, 2.0, 0.9]],
            dtype=np.float32,
        ),
    )
    motion = create_data_source("parc_ms", path).load()

    assert motion.positions.shape[0] == 2
    assert np.allclose(motion.root_translations[0], [1.0, 2.0, 0.9])
    assert motion.metadata["body_contacts"].shape[0] == 2
    assert motion.metadata["trimmed_leading_bind_frames"] == 1


def test_foot_contact_samples_use_parc_ankle_to_sole_offset(tmp_path):
    motion = ParcMsDataSource(_write_ms_pickle(tmp_path)).load()
    markers = motion.metadata["end_effector_markers"]

    for side, ankle_name in (("left", "L_Ankle"), ("right", "R_Ankle")):
        ankle_index = motion.target_names.index(ankle_name)
        ankle = motion.positions[:, ankle_index]
        marker = markers[f"{side}_foot"]
        sole = marker["contact_samples"][:, 0]
        assert np.allclose(marker["tracking_positions"], sole)
        assert np.allclose(ankle[:, :2], sole[:, :2])
        assert np.allclose(
            ankle[:, 2] - sole[:, 2], PARC_MS_FOOT_CONTACT_OFFSET_M
        )


def test_parc_body_contacts_are_exposed_as_end_effector_hints(tmp_path):
    body_contacts = np.zeros((2, 15), dtype=bool)
    body_contacts[0, 11] = True
    body_contacts[1, 14] = True
    motion = ParcMsDataSource(
        _write_ms_pickle(tmp_path, body_contacts=body_contacts)
    ).load()
    markers = motion.metadata["end_effector_markers"]

    np.testing.assert_array_equal(
        markers["right_foot"]["source_contact_mask"], [True, False]
    )
    np.testing.assert_array_equal(
        markers["left_foot"]["source_contact_mask"], [False, True]
    )


def test_resample_keeps_parc_end_effector_markers_aligned(tmp_path):
    motion = ParcMsDataSource(_write_ms_pickle(tmp_path)).load().resample(60.0)

    assert motion.positions.shape[0] == 3
    for name, marker in motion.metadata["end_effector_markers"].items():
        assert marker["tracking_positions"].shape[0] == 3
        assert marker["contact_samples"].shape[0] == 3
        if "foot" in name:
            assert marker["source_contact_mask"].shape == (3,)
            assert marker["source_contact_mask"].dtype == np.bool_
        else:
            assert "source_contact_mask" not in marker
