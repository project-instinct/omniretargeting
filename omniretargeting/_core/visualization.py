"""Matplotlib trajectory preview."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.animation import FuncAnimation


class TrajectoryVisualizationMixin:
    """Matplotlib trajectory preview.

    Internal methods sharing the owning OmniRetargeter instance; no separate state.
    """

    def _visualize_trajectory(self, trajectory: np.ndarray, scaled_terrain: trimesh.Trimesh):
        """
        Visualize the SMPLX trajectory using matplotlib 3D animation with terrain mesh.

        Args:
            trajectory: Processed trajectory of shape (T, J, 3) where T is frames, J is joints
                       Coordinates are assumed to be in +Z up convention (already transformed)
            scaled_terrain: Scaled terrain mesh
        """
        print(f"Visualizing trajectory with shape: {trajectory.shape}")
        print(f"Terrain mesh: {len(scaled_terrain.vertices)} vertices, {len(scaled_terrain.faces)} faces")
        print("Coordinate system: +Z is up")

        num_frames, num_joints, _ = trajectory.shape

        # Create figure and 3D axis
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Compute axis limits based on trajectory AND terrain bounds
        all_points = trajectory.reshape(-1, 3)
        traj_bounds = np.array([all_points.min(axis=0), all_points.max(axis=0)])

        # Get terrain bounds
        terrain_bounds = scaled_terrain.bounds  # Shape: (2, 3) - min and max

        # Combine bounds
        x_min = min(traj_bounds[0, 0], terrain_bounds[0, 0])
        y_min = min(traj_bounds[0, 1], terrain_bounds[0, 1])
        z_min = min(traj_bounds[0, 2], terrain_bounds[0, 2])
        x_max = max(traj_bounds[1, 0], terrain_bounds[1, 0])
        y_max = max(traj_bounds[1, 1], terrain_bounds[1, 1])
        z_max = max(traj_bounds[1, 2], terrain_bounds[1, 2])

        # Add some margin
        margin = 0.2
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        ax.set_xlim([x_min - margin * x_range, x_max + margin * x_range])
        ax.set_ylim([y_min - margin * y_range, y_max + margin * y_range])
        ax.set_zlim([z_min - margin * z_range, z_max + margin * z_range])

        # Set labels (Z is up)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m) - Up')
        ax.set_title('SMPLX Trajectory with Terrain Visualization')

        # Set equal aspect ratio
        max_range = max(x_range, y_range, z_range)
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        ax.set_xlim([mid_x - max_range/2, mid_x + max_range/2])
        ax.set_ylim([mid_y - max_range/2, mid_y + max_range/2])
        ax.set_zlim([mid_z - max_range/2, mid_z + max_range/2])

        # Plot terrain mesh
        print("Rendering terrain mesh...")
        terrain_vertices = scaled_terrain.vertices
        terrain_faces = scaled_terrain.faces

        # Create a simplified mesh for visualization if too complex
        if len(terrain_faces) > 10000:
            print(f"Terrain has {len(terrain_faces)} faces, simplifying for visualization...")
            try:
                simplified_terrain = scaled_terrain.simplify_quadric_decimation(10000)
            except ValueError:
                # Fallback for trimesh/fast_simplification version mismatch
                # where target_count is interpreted as target_reduction
                print("Using direct fast_simplification fallback due to trimesh error...")
                import fast_simplification
                vertices, faces = fast_simplification.simplify(
                    scaled_terrain.vertices,
                    scaled_terrain.faces,
                    target_count=10000
                )
                simplified_terrain = trimesh.Trimesh(vertices=vertices, faces=faces)

            terrain_vertices = simplified_terrain.vertices
            terrain_faces = simplified_terrain.faces
            print(f"Simplified to {len(terrain_faces)} faces")

        # Plot terrain as a triangulated surface
        ax.plot_trisurf(
            terrain_vertices[:, 0],
            terrain_vertices[:, 1],
            terrain_vertices[:, 2],
            triangles=terrain_faces,
            color='gray',
            alpha=0.3,
            edgecolor='none',
            shade=True,
            linewidth=0
        )

        # Initialize scatter plot for joints
        scatter = ax.scatter([], [], [], c='blue', marker='o', s=50, alpha=0.9, edgecolors='black', linewidths=0.5)

        # Add text for frame counter
        frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

        # Define SMPLX skeleton connections (approximate body structure)
        # These are indices in the standard SMPLX joint ordering
        skeleton_connections = [
            # Spine
            (0, 3),   # Pelvis -> Spine1
            (3, 6),   # Spine1 -> Spine2
            (6, 9),   # Spine2 -> Spine3
            (9, 12),  # Spine3 -> Neck
            (12, 15), # Neck -> Head

            # Left leg
            (0, 1),   # Pelvis -> L_Hip
            (1, 4),   # L_Hip -> L_Knee
            (4, 7),   # L_Knee -> L_Ankle
            (7, 10),  # L_Ankle -> L_Foot

            # Right leg
            (0, 2),   # Pelvis -> R_Hip
            (2, 5),   # R_Hip -> R_Knee
            (5, 8),   # R_Knee -> R_Ankle
            (8, 11),  # R_Ankle -> R_Foot

            # Left arm
            (9, 13),  # Spine3 -> L_Collar
            (13, 16), # L_Collar -> L_Shoulder
            (16, 18), # L_Shoulder -> L_Elbow
            (18, 20), # L_Elbow -> L_Wrist

            # Right arm
            (9, 14),  # Spine3 -> R_Collar
            (14, 17), # R_Collar -> R_Shoulder
            (17, 19), # R_Shoulder -> R_Elbow
            (19, 21), # R_Elbow -> R_Wrist
        ]

        # Filter connections to only include joints we have
        valid_connections = [(i, j) for i, j in skeleton_connections if i < num_joints and j < num_joints]

        # Initialize line objects for skeleton
        lines = []
        for _ in valid_connections:
            line, = ax.plot([], [], [], 'r-', linewidth=2.5, alpha=0.8)
            lines.append(line)

        # Set viewing angle for better visibility
        ax.view_init(elev=20, azim=45)

        def init():
            """Initialize animation."""
            scatter._offsets3d = ([], [], [])
            frame_text.set_text('')
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [scatter, frame_text] + lines

        def update(frame_idx):
            """Update animation for each frame."""
            # Get joint positions for current frame
            joints = trajectory[frame_idx]  # Shape: (J, 3)

            # Update scatter plot
            xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]
            scatter._offsets3d = (xs, ys, zs)

            # Update frame counter
            frame_text.set_text(f'Frame: {frame_idx + 1}/{num_frames}')

            # Update skeleton lines
            for line, (i, j) in zip(lines, valid_connections):
                x_data = [joints[i, 0], joints[j, 0]]
                y_data = [joints[i, 1], joints[j, 1]]
                z_data = [joints[i, 2], joints[j, 2]]
                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)

            return [scatter, frame_text] + lines

        # Create animation
        print(f"Creating animation for {num_frames} frames...")
        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            init_func=init,
            interval=33,  # ~30 FPS
            blit=True,
            repeat=True
        )

        print("Displaying animation. Close the window to continue...")
        plt.tight_layout()
        plt.show()
        print("Visualization complete.")
