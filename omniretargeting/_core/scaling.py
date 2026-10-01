"""Optional terrain scale estimate and source trajectory scaling."""

from __future__ import annotations

import numpy as np
import trimesh


class TerrainScalingMixin:
    """Terrain scale estimate and source trajectory scaling.

    Internal methods sharing the owning OmniRetargeter instance; no separate state.
    """

    def _compute_terrain_scale(
        self,
        smplx_trajectory: np.ndarray,
    ) -> float:
        """
        Compute the appropriate scaling factor for the terrain.

        This implementation assumes the terrain provided corresponds to the real-world scale
        relative to the robot's size.
        """
        # Default implementation: Scale based on robot vs human height ratio only
        # This assumes the terrain mesh is already at a "human scale" (e.g. captured or designed for humans)
        # and we just need to adjust it for the robot's size.

        # Estimate human height from SMPLX trajectory
        # Height ≈ (head Z - min foot Z) in standing pose
        # Or more robustly: Max height difference in trajectory

        # Use simple heuristic if we don't have betas to calculate exact height
        # Assuming typical human height of 1.7m if estimation fails or is unreliable
        estimated_human_height = 1.7

        if smplx_trajectory is not None and len(smplx_trajectory) > 0:
            # Estimate height from joints
            # SMPLX/SMPL joints: 0: Pelvis, 10/11: Feet, 15: Head (approx)
            # We can find the max vertical extent across all frames

            # Find the frame where the person is most upright (max head height)
            # Assuming Z is up
            head_joint_name = self.height_estimation_config.get("head_joint", "Head")
            foot_joint_names = self.height_estimation_config.get("foot_joints", ["L_Foot", "R_Foot"])
            head_top_offset = float(self.height_estimation_config.get("head_top_offset", 0.12))

            head_idx = self.smplx_joint_indices.get(head_joint_name, 15)
            foot_indices = [
                self.smplx_joint_indices[name]
                for name in foot_joint_names
                if name in self.smplx_joint_indices
            ]
            if not foot_indices:
                foot_indices = [10, 11]  # fallback defaults

            # Check if we have enough joints
            if smplx_trajectory.shape[1] > head_idx and all(idx < smplx_trajectory.shape[1] for idx in foot_indices):
                # Calculate height per frame: Head Z - Average Foot Z
                head_z = smplx_trajectory[:, head_idx, 2]

                # Use min foot Z per frame as ground reference relative to body
                feet_z = np.min(smplx_trajectory[:, foot_indices, 2], axis=1)

                # Height per frame
                heights = head_z - feet_z

                # Use the maximum height observed (standing pose)
                # Add offset for top of head (head joint is in neck/center of head)
                # Approx 10-15cm from head joint to top of head
                estimated_human_height = np.max(heights) + head_top_offset

                # Sanity check: Constrain to reasonable human range [1.4, 2.2]
                estimated_human_height = np.clip(estimated_human_height, 1.4, 2.2)

                print(f"Estimated human height from trajectory: {estimated_human_height:.3f}m")

        scale_factor = self.robot_height / estimated_human_height

        print(f"Computed terrain scale factor: {scale_factor:.4f} (Robot: {self.robot_height}m, Human Est: {estimated_human_height:.3f}m)")

        return float(scale_factor)

    def _scale_terrain_mesh(self, scale_factor: float) -> trimesh.Trimesh:
        """Scale the terrain mesh by the given factor."""
        scaled_mesh = self.terrain_mesh.copy()
        scaled_mesh.apply_scale(scale_factor)
        return scaled_mesh

    def _process_smplx_trajectory(
        self,
        smplx_trajectory: np.ndarray,
        terrain_scale: float
    ) -> np.ndarray:
        """Process SMPLX trajectory for retargeting."""
        # Apply terrain scaling to trajectory coordinates
        scaled_trajectory = smplx_trajectory * terrain_scale

        # Transform coordinate system if needed (SMPLX uses different convention)
        # TODO: Add coordinate system transformation if needed

        return scaled_trajectory
