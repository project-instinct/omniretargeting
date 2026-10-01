"""Source-root orientation estimate and root-joint selection."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class SourceRootReferenceMixin:
    """Source-root orientation and position references.

    Internal methods sharing the owning OmniRetargeter instance; no separate state.
    """

    def _estimate_base_orientation_from_joints(self, joints: np.ndarray, last_quat: np.ndarray | None = None) -> np.ndarray | None:
        """
        Estimate a base orientation from joint positions.

        Uses pelvis/hips/spine to build an approximate body frame:
        - up: pelvis -> spine1
        - right: left_hip -> right_hip
        - forward: right x up
        Returns quaternion in xyzw order.

        Args:
            joints: Joint positions array.
            last_quat: Quaternion from previous frame to ensure continuity (xyzw).
        """
        if joints.shape[0] < 4:
            return None

        pelvis_name = self.base_orientation_config.get("pelvis", "Pelvis")
        left_hip_name = self.base_orientation_config.get("left_hip", "L_Hip")
        right_hip_name = self.base_orientation_config.get("right_hip", "R_Hip")
        spine_name = self.base_orientation_config.get("spine", "Spine1")

        # A configured name that is missing from the source layout used to fall
        # back to a positional index, which silently built the body frame out of
        # whichever joints happened to sit at 0/1/2/3.  This frame is what the
        # whole upper body tracks, so a typo here has to be loud.
        frame_indices = {}
        for key, name in (
            ("pelvis", pelvis_name),
            ("left_hip", left_hip_name),
            ("right_hip", right_hip_name),
            ("spine", spine_name),
        ):
            if name not in self.smplx_joint_indices:
                raise ValueError(
                    f"base_orientation.{key} refers to '{name}', which is not in "
                    f"the source joint layout "
                    f"{list(self.smplx_joint_indices)[:8]}..."
                )
            frame_indices[key] = self.smplx_joint_indices[name]
        pelvis_idx = frame_indices["pelvis"]
        left_hip_idx = frame_indices["left_hip"]
        right_hip_idx = frame_indices["right_hip"]
        spine_idx = frame_indices["spine"]

        max_required_idx = max(pelvis_idx, left_hip_idx, right_hip_idx, spine_idx)
        if joints.shape[0] <= max_required_idx:
            return None

        pelvis = joints[pelvis_idx]
        left_hip = joints[left_hip_idx]
        right_hip = joints[right_hip_idx]
        spine1 = joints[spine_idx]

        up = spine1 - pelvis
        right = right_hip - left_hip

        up_norm = np.linalg.norm(up)
        right_norm = np.linalg.norm(right)
        if up_norm < 1e-8 or right_norm < 1e-8:
            return None

        up = up / up_norm
        right = right / right_norm

        # Build forward using right-hand rule
        forward = np.cross(up, right)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            return None
        forward = forward / forward_norm

        # Re-orthogonalize right to ensure perfect orthogonality
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        # Rotation matrix with body axes in world coordinates.
        # Convention: X=forward, Y=right, Z=up
        rot = np.column_stack([forward, right, up])

        # Verify right-handed
        det = np.linalg.det(rot)
        if det < 0:
            forward = -forward
            right = -right
            rot = np.column_stack([forward, right, up])

        current_quat = Rotation.from_matrix(rot).as_quat() # xyzw

        # Ensure continuity if last quaternion is provided
        if last_quat is not None:
            # Check dot product to see if we need to flip sign
            dot = np.dot(current_quat, last_quat)
            if dot < 0:
                current_quat = -current_quat

        return current_quat


    def _resolve_source_root_index(self) -> int:
        """Resolve the highest available source torso joint for robot-root use."""
        indices = getattr(self, "smplx_joint_indices", {})
        if not indices:
            return 0
        requested = str(
            self.base_orientation_config.get("root_position", "Pelvis")
        )
        if requested in indices:
            return int(indices[requested])

        casefold_indices = {
            str(name).casefold(): int(idx) for name, idx in indices.items()
        }
        requested_lower = requested.casefold()
        if "spine" in requested_lower or any(
            token in requested_lower for token in ("chest", "torso", "neck")
        ):
            candidates = (
                "Spine3",
                "Spine2",
                "Spine1",
                "Chest",
                "Torso",
                "Pelvis",
            )
        else:
            candidates = (requested, "Pelvis", "Spine1")
        for candidate in candidates:
            match = casefold_indices.get(candidate.casefold())
            if match is not None:
                return int(match)
        return 0
