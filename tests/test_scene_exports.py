"""Exported object geometry, body poses, and contact anchors share coordinates."""

import json

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from omniretargeting import Contact, EntityTrajectory, MotionData, Scene, SceneEntity
from omniretargeting.main import export_scaled_objects


@pytest.mark.parametrize("rigid", [False, True])
@pytest.mark.parametrize("scene_scale", [1.0, 2.0])
def test_export_reload_preserves_world_geometry_and_body_origin(
    tmp_path, rigid, scene_scale
):
    mesh = trimesh.creation.box()
    mesh.apply_translation([2, -1, 0.5])
    rotations = Rotation.from_euler("xyz", [[15, 35, 60], [-20, 50, 120]], degrees=True)
    translations = np.array([[10, -2, 3], [11, 1, 0]], dtype=float)
    scales = np.array([3, 3 if rigid else 2], dtype=float)
    expected = np.stack(
        [
            rotations[t].apply(mesh.vertices * scales[t]) + translations[t]
            for t in range(2)
        ]
    )
    geometry = mesh.copy()
    geometry.apply_scale(scales[0])
    motion = MotionData(
        positions=np.zeros((2, 1, 3)),
        target_names=["palm"],
        object_mesh=mesh,
        object_points=expected,
        metadata={
            "object_name": "box",
            "object_points_body_id": "box",
            "object_centroid_local": np.array(mesh.centroid),
            "object_translations": translations,
            "object_rotations": rotations.as_matrix(),
            "object_scales": scales,
        },
        scene=Scene({"box": SceneEntity(geometry)}) if rigid else None,
        entity_trajectories=(
            {
                "box": EntityTrajectory(
                    translations, rotations.as_quat(scalar_first=True)
                )
            }
            if rigid
            else None
        ),
        contact_trajectory=(
            [[Contact("palm", "box", geometry.vertices[0])] for _ in range(2)]
            if rigid
            else None
        ),
    )
    mesh_path, pose_path = export_scaled_objects(motion, tmp_path, scene_scale, True)
    reloaded = trimesh.load(mesh_path, force="mesh", process=False)
    poses = json.loads(pose_path.read_text())
    for t, pose in enumerate(poses):
        rotation = Rotation.from_matrix(pose["rotation_matrix"])
        actual = rotation.apply(reloaded.vertices * pose["scale"]) + pose["translation"]
        distances, _ = cKDTree(expected[t] * scene_scale).query(actual)
        assert len(actual) == len(expected[t])
        np.testing.assert_allclose(distances, 0, atol=1e-7)
        if rigid:
            assert pose["scale"] == 1
            contact = motion.scaled(scene_scale).contact_trajectory[t][0]
            exported_world = rotation.apply(contact.point_local) + pose["translation"]
            source_world = (
                motion.entity_trajectories["box"]
                .pose(t)
                .to_world(motion.contact_trajectory[t][0].point_local)
            )
            np.testing.assert_allclose(exported_world, source_world * scene_scale)
