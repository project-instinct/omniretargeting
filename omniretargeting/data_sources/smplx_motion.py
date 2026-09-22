"""SMPL-X loading, marker extraction and missing-frame interpolation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation


def compute_world_joint_orientations(
    global_orient: np.ndarray,
    full_pose: np.ndarray,
    parents: np.ndarray,
    num_body_joints: int = 22,
) -> np.ndarray:
    """
    Compute world-frame joint orientations from SMPLX pose parameters.

    Args:
        global_orient: Root orientation in axis-angle format, shape (T, 3)
        full_pose: Full pose parameters in axis-angle format, shape (T, J_total, 3)
        parents: Parent indices for kinematic tree, shape (J_total,)
        num_body_joints: Number of body joints to return (default 22)

    Returns:
        Joint orientations as quaternions (wxyz format), shape (T, J, 4)
    """
    num_frames = global_orient.shape[0]
    num_joints = min(full_pose.shape[1], num_body_joints)

    # Output: quaternions in wxyz format
    joint_orientations = np.zeros((num_frames, num_joints, 4))

    for frame_idx in range(num_frames):
        # Store rotations for this frame
        frame_rotations = []

        for joint_idx in range(num_joints):
            if joint_idx == 0:
                # Root joint: use global_orient directly
                rot = Rotation.from_rotvec(global_orient[frame_idx])
            else:
                # Other joints: multiply parent's world rotation by local rotation
                parent_idx = parents[joint_idx]
                if parent_idx >= 0 and parent_idx < len(frame_rotations):
                    parent_rot = frame_rotations[parent_idx]
                    local_rot = Rotation.from_rotvec(full_pose[frame_idx, joint_idx])
                    rot = parent_rot * local_rot
                else:
                    # Fallback: use local rotation as world rotation
                    rot = Rotation.from_rotvec(full_pose[frame_idx, joint_idx])

            frame_rotations.append(rot)
            # Store as quaternion in wxyz format (scalar_first=True)
            joint_orientations[frame_idx, joint_idx] = rot.as_quat(scalar_first=True)

    return joint_orientations


def _farthest_point_sample_indices(points: np.ndarray, max_samples: int) -> np.ndarray:
    """Select deterministic, spatially distributed points from a template part."""
    points = np.asarray(points, dtype=float)
    if len(points) <= max_samples:
        return np.arange(len(points), dtype=int)
    max_samples = max(int(max_samples), 1)
    center = np.mean(points, axis=0)
    selected = [int(np.argmax(np.linalg.norm(points - center, axis=1)))]
    min_squared_distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for _ in range(1, max_samples):
        next_idx = int(np.argmax(min_squared_distance))
        selected.append(next_idx)
        squared_distance = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_squared_distance = np.minimum(min_squared_distance, squared_distance)
    return np.asarray(selected, dtype=int)


def _extract_smplx_end_effector_markers(
    body_model,
    smplx_output,
    max_samples: int = 32,
    world_joint_orientations: Optional[np.ndarray] = None,
    shape_betas: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract coherent hand/foot samples and rigid semantic foot frames."""
    vertices = smplx_output.vertices.detach().cpu().numpy()
    weights = body_model.lbs_weights.detach().cpu().numpy()
    template = body_model.v_template.detach().cpu().numpy()
    full_joints = smplx_output.joints.detach().cpu().numpy()
    shaped_template = template
    if shape_betas is not None:
        beta_values = np.asarray(shape_betas, dtype=float).reshape(-1)
        shapedirs = body_model.shapedirs.detach().cpu().numpy()
        num_shapes = min(len(beta_values), shapedirs.shape[2])
        shaped_template = template + np.einsum(
            "vcn,n->vc",
            shapedirs[:, :, :num_shapes],
            beta_values[:num_shapes],
        )
    joint_regressor = body_model.J_regressor.detach().cpu().numpy()
    template_joints = joint_regressor @ shaped_template

    # SMPL-X LBS joint ids: feet 10/11, left fingers 25:40, right
    # fingers 40:55.  Excluding wrist/ankle weights avoids pulling forearm and
    # shin vertices into the contact surface sets.
    part_joint_ids = {
        "left_hand": list(range(25, 40)),
        "right_hand": list(range(40, 55)),
        "left_foot": [10],
        "right_foot": [11],
    }
    fallback_joint_ids = {
        "left_hand": 20,
        "right_hand": 21,
        "left_foot": 10,
        "right_foot": 11,
    }

    markers: Dict[str, Dict[str, np.ndarray]] = {}
    for name, joint_ids in part_joint_ids.items():
        influence = np.sum(weights[:, joint_ids], axis=1)
        vertex_ids = np.flatnonzero(influence > 0.25)
        if len(vertex_ids) == 0:
            tracking = smplx_output.joints[:, fallback_joint_ids[name], :].detach().cpu().numpy()
            markers[name] = {
                "tracking_positions": tracking,
                "contact_samples": tracking[:, None, :],
            }
            continue

        local_ids = _farthest_point_sample_indices(
            template[vertex_ids],
            max_samples=max_samples,
        )
        sampled_vertex_ids = vertex_ids[local_ids]
        part_vertices = vertices[:, vertex_ids, :]
        markers[name] = {
            "tracking_positions": np.mean(part_vertices, axis=1),
            "contact_samples": vertices[:, sampled_vertex_ids, :],
            "vertex_ids": sampled_vertex_ids,
        }
        if (
            name in ("left_hand", "right_hand")
            and world_joint_orientations is not None
        ):
            # Build a rigid palm frame from the shaped neutral template, then
            # carry it with the SMPL-X wrist rotation.  Position-only hand
            # tracking lets the optimizer rotate the G1 wrist to use the long
            # palm offset as a reach extension, which is accurate at one point
            # but produces visibly twisted hands.  The wrist-to-finger-base
            # direction is the hand's local +x; index-to-pinky supplies a
            # stable in-palm lateral axis and fixes the palm-normal sign.
            if name == "left_hand":
                wrist_idx = 20
                finger_base_ids = [25, 28, 31, 34]
                index_idx, pinky_idx = 25, 31
            else:
                wrist_idx = 21
                finger_base_ids = [40, 43, 46, 49]
                index_idx, pinky_idx = 40, 46

            template_forward = (
                np.mean(template_joints[finger_base_ids], axis=0)
                - template_joints[wrist_idx]
            )
            template_forward /= max(
                float(np.linalg.norm(template_forward)), 1e-10
            )
            template_lateral = (
                template_joints[index_idx] - template_joints[pinky_idx]
            )
            template_lateral -= template_forward * float(
                np.dot(template_forward, template_lateral)
            )
            template_lateral /= max(
                float(np.linalg.norm(template_lateral)), 1e-10
            )
            template_up = np.cross(template_forward, template_lateral)
            template_up /= max(float(np.linalg.norm(template_up)), 1e-10)

            rotations = Rotation.from_quat(
                np.asarray(world_joint_orientations)[:, wrist_idx],
                scalar_first=True,
            ).as_matrix()
            markers[name]["forward_directions"] = np.einsum(
                "tij,j->ti", rotations, template_forward
            )
            markers[name]["up_directions"] = np.einsum(
                "tij,j->ti", rotations, template_up
            )
        if name in ("left_foot", "right_foot"):
            ankle_idx = 7 if name == "left_foot" else 8
            foot_idx = 10 if name == "left_foot" else 11

            # Calibrate one rigid, right-handed foot frame on the shaped
            # neutral template.  Per-frame PCA has no persistent eigenvector
            # identity and can turn skin deformation into apparent foot
            # rotation.  The template's thinnest axis is the sole normal; its
            # sign is fixed toward the ankle.  Ankle-to-foot gives the
            # semantic toe direction, projected into the sole plane because
            # rotation-matrix axes must be orthogonal.
            template_part = shaped_template[vertex_ids]
            centered_template = template_part - np.mean(template_part, axis=0)
            covariance = centered_template.T @ centered_template / max(
                len(template_part), 1
            )
            _, eigenvectors = np.linalg.eigh(covariance)
            template_up = eigenvectors[:, 0]
            toward_ankle = (
                template_joints[ankle_idx] - np.mean(template_part, axis=0)
            )
            if float(np.dot(template_up, toward_ankle)) < 0.0:
                template_up *= -1.0
            template_up /= max(float(np.linalg.norm(template_up)), 1e-10)

            template_forward = (
                template_joints[foot_idx] - template_joints[ankle_idx]
            )
            template_forward -= template_up * float(
                np.dot(template_up, template_forward)
            )
            template_forward /= max(
                float(np.linalg.norm(template_forward)), 1e-10
            )
            template_lateral = np.cross(template_up, template_forward)
            template_lateral /= max(
                float(np.linalg.norm(template_lateral)), 1e-10
            )
            template_forward = np.cross(template_lateral, template_up)

            if world_joint_orientations is not None:
                rotations = Rotation.from_quat(
                    np.asarray(world_joint_orientations)[:, foot_idx],
                    scalar_first=True,
                ).as_matrix()
                markers[name]["forward_directions"] = np.einsum(
                    "tij,j->ti", rotations, template_forward
                )
                markers[name]["up_directions"] = np.einsum(
                    "tij,j->ti", rotations, template_up
                )
            else:
                # This path is retained for direct helper use.  Raw SMPL-X
                # loading supplies joint rotations and therefore uses the
                # rigid-frame path above.
                raw_forward = (
                    full_joints[:, foot_idx] - full_joints[:, ankle_idx]
                )
                raw_forward /= np.maximum(
                    np.linalg.norm(raw_forward, axis=1, keepdims=True), 1e-10
                )
                markers[name]["forward_directions"] = raw_forward
    return markers


def _fallback_end_effector_markers(
    joints: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build point markers when the source file has no SMPL-X body surface."""
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joint trajectory with shape (T, J, 3), got {joints.shape}.")
    indices = {
        "left_hand": 20,
        "right_hand": 21,
        "left_foot": 10 if joints.shape[1] > 10 else 7,
        "right_foot": 11 if joints.shape[1] > 11 else 8,
    }
    markers: Dict[str, Dict[str, np.ndarray]] = {}
    for name, joint_idx in indices.items():
        if joint_idx >= joints.shape[1]:
            continue
        positions = joints[:, joint_idx, :]
        markers[name] = {
            "tracking_positions": positions,
            "contact_samples": positions[:, None, :],
        }
    return markers


def _interpolate_rotvecs(
    start: np.ndarray,
    end: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Interpolate independent axis-angle rotations along their shortest arcs."""
    start = np.asarray(start)
    end = np.asarray(end)
    original_shape = start.shape
    start_rot = Rotation.from_rotvec(start.reshape(-1, 3))
    end_rot = Rotation.from_rotvec(end.reshape(-1, 3))
    delta = start_rot.inv() * end_rot
    interpolated = start_rot * Rotation.from_rotvec(delta.as_rotvec() * fraction)
    return interpolated.as_rotvec().reshape(original_shape)


def _densify_smplx_frame_gaps(
    frame_ids: np.ndarray,
    trans: np.ndarray,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    pose_hand: Optional[np.ndarray] = None,
    *,
    max_gap: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Restore short missing source frames before SMPL-X forward kinematics.

    Positions use linear interpolation and every axis-angle joint uses shortest-
    arc rotation interpolation.  This keeps a missing video frame from being
    played as a double-speed motion sample while leaving complete timelines
    untouched.  Large/non-numeric/disordered gaps are rejected rather than
    guessing across an edit or a different frame-id convention.
    """
    ids = np.asarray(frame_ids)
    if ids.ndim != 1 or len(ids) != len(trans):
        return trans, root_orient, pose_body, pose_hand, ids
    try:
        numeric_ids = np.asarray([int(str(value)) for value in ids], dtype=np.int64)
    except (TypeError, ValueError):
        return trans, root_orient, pose_body, pose_hand, ids

    gaps = np.diff(numeric_ids)
    if len(gaps) == 0 or np.all(gaps == 1):
        return trans, root_orient, pose_body, pose_hand, ids
    if np.any(gaps < 1) or np.any(gaps > max_gap):
        return trans, root_orient, pose_body, pose_hand, ids

    dense_trans = []
    dense_root = []
    dense_body = []
    dense_hand = [] if pose_hand is not None else None
    dense_ids = []
    id_width = max(len(str(value)) for value in ids)

    for frame_idx, gap in enumerate(gaps):
        dense_trans.append(np.asarray(trans[frame_idx]))
        dense_root.append(np.asarray(root_orient[frame_idx]))
        dense_body.append(np.asarray(pose_body[frame_idx]))
        if dense_hand is not None:
            dense_hand.append(np.asarray(pose_hand[frame_idx]))
        dense_ids.append(str(ids[frame_idx]))

        for offset in range(1, int(gap)):
            fraction = offset / float(gap)
            dense_trans.append(
                (1.0 - fraction) * trans[frame_idx]
                + fraction * trans[frame_idx + 1]
            )
            dense_root.append(
                _interpolate_rotvecs(
                    root_orient[frame_idx], root_orient[frame_idx + 1], fraction
                )
            )
            dense_body.append(
                _interpolate_rotvecs(
                    np.asarray(pose_body[frame_idx]).reshape(-1, 3),
                    np.asarray(pose_body[frame_idx + 1]).reshape(-1, 3),
                    fraction,
                ).reshape(np.asarray(pose_body[frame_idx]).shape)
            )
            if dense_hand is not None:
                dense_hand.append(
                    _interpolate_rotvecs(
                        np.asarray(pose_hand[frame_idx]).reshape(-1, 3),
                        np.asarray(pose_hand[frame_idx + 1]).reshape(-1, 3),
                        fraction,
                    ).reshape(np.asarray(pose_hand[frame_idx]).shape)
                )
            dense_ids.append(str(numeric_ids[frame_idx] + offset).zfill(id_width))

    dense_trans.append(np.asarray(trans[-1]))
    dense_root.append(np.asarray(root_orient[-1]))
    dense_body.append(np.asarray(pose_body[-1]))
    if dense_hand is not None:
        dense_hand.append(np.asarray(pose_hand[-1]))
    dense_ids.append(str(ids[-1]))

    return (
        np.asarray(dense_trans, dtype=np.asarray(trans).dtype),
        np.asarray(dense_root, dtype=np.asarray(root_orient).dtype),
        np.asarray(dense_body, dtype=np.asarray(pose_body).dtype),
        None
        if dense_hand is None
        else np.asarray(dense_hand, dtype=np.asarray(pose_hand).dtype),
        np.asarray(dense_ids, dtype=ids.dtype),
    )


def load_smplx_trajectory(
    smplx_file: Path,
    smplx_model_directory: Optional[str] = None,
    gender: str = "neutral",
    return_meta: bool = False,
    return_end_effectors: bool = False,
    end_effector_samples: int = 32,
) -> tuple:
    """
    Load SMPLX trajectory from file.

    Handles:
    - Standard .npy files with pre-computed joint positions
    - Pre-processed .npz files with 'global_joint_positions' key
    - Raw SMPLX-NG .npz files (e.g., stageii.npz) with SMPLX parameters

    For raw SMPLX-NG files, requires smplx_model_directory to perform forward kinematics.

    Args:
        smplx_file: Path to trajectory file (.npy or .npz)
        smplx_model_directory: Path to SMPLX model files (required for raw SMPLX-NG files and orientation computation)
        gender: Gender for SMPLX model ('neutral', 'male', 'female')
        return_meta: If True, also return root orientation and translation.
        return_end_effectors: If True, return semantic hand/foot centers and
            compact SMPL-X surface samples for scene-contact inference.
        end_effector_samples: Maximum number of coherent surface samples per
            end effector.

    Returns:
        Tuple beginning with ``(positions, orientations)``.  ``return_meta``
        appends ``(root_orient, trans)`` and ``return_end_effectors`` appends a
        marker dictionary.

        If return_meta=False (default):
            - positions: Joint positions array of shape (T, J, 3)
            - orientations: Joint orientations as quaternions (wxyz), shape (T, J, 4)
                           Returns None if orientations cannot be computed (e.g., .npy files)

        If return_meta=True:
            - positions: Joint positions array of shape (T, J, 3)
            - orientations: Joint orientations as quaternions (wxyz), shape (T, J, 4)
                           Returns None if orientations cannot be computed
            - root_orient: Root orientation parameters in axis-angle format, shape (T, 3) or None
            - trans: Translation parameters, shape (T, 3) or None
    """
    def pack_result(joints, orientations, root_orient, trans, end_effectors):
        result = [joints, orientations]
        if return_meta:
            result.extend([root_orient, trans])
        if return_end_effectors:
            result.append(end_effectors)
        return tuple(result)

    if smplx_file.suffix == ".npy":
        joints = np.load(smplx_file, allow_pickle=True)
        # Cannot compute orientations from positions only
        print("Warning: Cannot compute orientations from .npy file (positions only). Returning None for orientations.")
        end_effectors = _fallback_end_effector_markers(joints) if return_end_effectors else None
        return pack_result(joints, None, None, None, end_effectors)

    smplx_data = np.load(smplx_file, allow_pickle=True)

    if isinstance(smplx_data, np.lib.npyio.NpzFile) and "human_joints" in smplx_data:
        joints = smplx_data["human_joints"]
        print("Warning: Cannot compute orientations from OMOMO human_joints. Returning None for orientations.")
        end_effectors = _fallback_end_effector_markers(joints) if return_end_effectors else None
        return pack_result(joints, None, None, None, end_effectors)

    if isinstance(smplx_data, np.lib.npyio.NpzFile) and "global_joint_positions" in smplx_data:
        joints = smplx_data["global_joint_positions"]
        root_orient = smplx_data["root_orient"] if "root_orient" in smplx_data else None
        trans = smplx_data["trans"] if "trans" in smplx_data else None

        # Try to compute orientations if we have the necessary data
        orientations = None
        if "full_pose" in smplx_data and smplx_model_directory is not None and root_orient is not None:
            # Load body model to get parent structure
            body_model = smplx.create(
                smplx_model_directory,
                "smplx",
                gender=gender,
                use_pca=False,
            )
            full_pose = smplx_data["full_pose"]
            if isinstance(full_pose, np.ndarray) and len(full_pose.shape) == 2:
                # Reshape from (T, J*3) to (T, J, 3)
                full_pose = full_pose.reshape(full_pose.shape[0], -1, 3)

            orientations = compute_world_joint_orientations(
                root_orient,
                full_pose,
                body_model.parents.cpu().numpy(),
                num_body_joints=22,
            )
        else:
            print("Warning: Cannot compute orientations from .npz file (missing full_pose, root_orient, or model directory). Returning None for orientations.")

        end_effectors = _fallback_end_effector_markers(joints) if return_end_effectors else None
        return pack_result(joints, orientations, root_orient, trans, end_effectors)

    # Raw SMPLX-NG file - need to run forward kinematics
    body_model = smplx.create(
        smplx_model_directory,
        "smplx",
        gender=str(smplx_data.get("gender", "neutral")),
        use_pca=False,
    )

    # Extract and adjust betas
    betas = torch.tensor(smplx_data["betas"]).float().view(1, -1)
    if betas.shape[1] > 10:
        betas = betas[:, :10]

    pose_body = smplx_data["pose_body"]
    root_orient = smplx_data["root_orient"]
    trans = smplx_data["trans"]
    pose_hand = smplx_data["pose_hand"] if "pose_hand" in smplx_data else None
    if "frame_ids" in smplx_data:
        original_num_frames = len(trans)
        trans, root_orient, pose_body, pose_hand, _ = _densify_smplx_frame_gaps(
            smplx_data["frame_ids"],
            trans,
            root_orient,
            pose_body,
            pose_hand,
        )
        if len(trans) != original_num_frames:
            print(
                "Restored "
                f"{len(trans) - original_num_frames} missing SMPL-X timeline frame(s) "
                "from frame_ids."
            )
    num_frames = pose_body.shape[0]
    if pose_hand is not None and pose_hand.ndim == 2 and pose_hand.shape[1] >= 90:
        left_hand_pose = torch.tensor(pose_hand[:, :45]).float()
        right_hand_pose = torch.tensor(pose_hand[:, 45:90]).float()
    else:
        left_hand_pose = torch.zeros(num_frames, 45).float()
        right_hand_pose = torch.zeros(num_frames, 45).float()
    smplx_output = body_model(
        betas=betas,  # Shape parameters (1, 10)
        global_orient=torch.tensor(root_orient).float(),  # (N, 3)
        body_pose=torch.tensor(pose_body).float(),  # (N, 63)
        transl=torch.tensor(trans).float(),  # (N, 3)
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        jaw_pose=torch.zeros(num_frames, 3).float(),  # Pose parameters per frame
        leye_pose=torch.zeros(num_frames, 3).float(),  # Pose parameters per frame
        reye_pose=torch.zeros(num_frames, 3).float(),  # Pose parameters per frame
        expression=torch.zeros(num_frames, 10).float(),  # Expression parameters expanded to num_frames
        return_full_pose=True,
    )

    # Extract joints and their world rotations before building rigid semantic
    # foot frames.
    full_joints = smplx_output.joints.detach().cpu().numpy()
    full_pose = smplx_output.full_pose.detach().cpu().numpy()
    # Reshape from (T, J*3) to (T, J, 3)
    full_pose = full_pose.reshape(num_frames, -1, 3)

    orientations = compute_world_joint_orientations(
        root_orient,
        full_pose,
        body_model.parents.cpu().numpy(),
        num_body_joints=22,
    )
    end_effectors = None
    if return_end_effectors:
        end_effectors = _extract_smplx_end_effector_markers(
            body_model,
            smplx_output,
            max_samples=end_effector_samples,
            world_joint_orientations=orientations,
            shape_betas=betas.detach().cpu().numpy(),
        )

    # Return only body joints (first 22)
    joints = full_joints[:, :22, :]

    return pack_result(joints, orientations, root_orient, trans, end_effectors)
