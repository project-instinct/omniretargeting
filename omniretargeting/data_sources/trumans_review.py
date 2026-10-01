"""Original TRUMANS geometry for diagnostic retargeting, in metric Z-up.

Keep the recorded translation, hand poses and object transforms. In particular,
the pelvis joint is not the SMPL-X translation parameter, and wrist joints are
not substitutes for the hand skin when observing contacts.
"""

from pathlib import Path

import numpy as np
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation

from .smplx_motion import (
    _extract_smplx_end_effector_markers,
    compute_world_joint_orientations,
)
from .smplx_surface import sample_smplx_surfaces

WORLD_ROTATION = Rotation.from_euler("x", 90, degrees=True).as_matrix()


def _recorded_object_poses(directory, segment, indices):
    """Prefer original per-recording poses over the derived aggregate cache.

    Object_pose stores XYZ Euler angles in radians and Y-up locations. Align
    through frame_id, never through array position or the robot trajectory.
    Augmentations without their own recording keep their aggregate transforms;
    borrowing the unaugmented recording would undo the scene augmentation.
    """
    path = directory / "Object_all" / "Object_pose" / f"{segment}.npy"
    if not path.exists():
        return None, None, directory / "object_mat.npy"
    frame_ids = np.asarray(np.load(directory / "frame_id.npy", mmap_mode="r")[indices])
    if frame_ids.ndim != 1 or not np.issubdtype(frame_ids.dtype, np.integer):
        raise ValueError("TRUMANS object alignment requires integer source frame_id")
    return np.load(path, allow_pickle=True).item(), frame_ids, path


def _object_transform(name, recorded, frame_ids, aggregate, indices, slots, source):
    if recorded is None:
        transform = np.array(aggregate[indices, slots], copy=True)
    else:
        # The chair mesh filename includes a scale already baked into its mesh.
        # Strip only that suffix to address the original object-pose dictionary.
        key = name.split("(", 1)[0]
        if key not in recorded:
            raise ValueError(f"Missing recorded object {key!r} in {source}")
        positions = np.asarray(recorded[key]["location"], dtype=float)
        angles = np.asarray(recorded[key]["rotation"], dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1:] != (3,)
            or angles.shape != positions.shape
            or np.any(frame_ids < 0)
            or np.any(frame_ids >= len(positions))
        ):
            raise ValueError(
                f"Object {name!r} does not cover source frame_id in {source}"
            )
        selected_positions, selected_angles = positions[frame_ids], angles[frame_ids]
        valid = np.isfinite(selected_positions).all(1) & np.isfinite(
            selected_angles
        ).all(1)
        if not valid.all():
            raise ValueError(
                f"Nonfinite recorded object {name!r}, source frames "
                f"{frame_ids[~valid].tolist()} in {source}"
            )
        transform = np.tile(np.eye(4), (len(indices), 1, 1))
        transform[:, :3, :3] = Rotation.from_euler("xyz", selected_angles).as_matrix()
        transform[:, :3, 3] = selected_positions
    valid = np.isfinite(transform).all(axis=(1, 2))
    if not valid.all():
        raise ValueError(
            f"Nonfinite object {name!r}, clip frames {np.flatnonzero(~valid).tolist()} "
            f"in {source}; original per-recording poses are required"
        )
    return transform


def load_scene(data_directory, segment):
    """Return the original static mesh and objects named by release metadata."""
    directory = Path(data_directory)
    indices = np.flatnonzero(np.load(directory / "seg_name.npy") == segment)
    if not len(indices):
        raise ValueError(f"No TRUMANS frames for {segment}")
    flags = np.load(directory / "scene_flag.npy", mmap_mode="r")[indices]
    if len(np.unique(flags)) != 1:
        raise ValueError("A review clip must use one static scene")
    scene_name = str(np.load(directory / "scene_list.npy")[int(flags[0])])
    scene = trimesh.load_mesh(
        directory / "Scene_mesh" / (scene_name + ".obj"), process=False
    )
    world = np.eye(4)
    world[:3, :3] = WORLD_ROTATION
    scene.apply_transform(world)
    names = np.load(directory / "object_list.npy")
    matrices = np.load(directory / "object_mat.npy", mmap_mode="r")
    object_flags = np.asarray(
        np.load(directory / "object_flag.npy", mmap_mode="r")[indices]
    )
    recorded, frame_ids, pose_source = _recorded_object_poses(
        directory, segment, indices
    )
    objects = []
    for object_id in np.flatnonzero(np.any(object_flags >= 0, axis=0)):
        slots = object_flags[:, object_id]
        if np.any(slots < 0) or np.any(slots >= matrices.shape[1]):
            raise ValueError(
                "Object visibility changes require separate observation intervals"
            )
        mesh = trimesh.load_mesh(
            directory
            / "Object_chairs"
            / "Object_mesh"
            / (str(names[object_id]) + ".obj"),
            process=False,
        )
        transform = world @ _object_transform(
            str(names[object_id]),
            recorded,
            frame_ids,
            matrices,
            indices,
            slots,
            pose_source,
        )
        objects.append(
            {
                "name": str(names[object_id]),
                "mesh": mesh,
                "translations": transform[:, :3, 3],
                "rotations": transform[:, :3, :3],
                "pose_source": str(pose_source),
            }
        )
    return indices, scene, objects


def reconstruct_source(
    data_directory, segment, model_directory, *, batch_size=64, whole_body_samples=1024
):
    """Reconstruct skin from recorded parameters without fitting or registration."""
    directory = Path(data_directory)
    indices = np.flatnonzero(np.load(directory / "seg_name.npy") == segment)
    if not len(indices):
        raise ValueError(f"No TRUMANS frames for {segment}")
    arrays = {
        name: np.load(directory / f"{name}.npy", mmap_mode="r")
        for name in (
            "human_orient",
            "human_pose",
            "human_transl",
            "left_hand_pose",
            "right_hand_pose",
        )
    }
    model = smplx.create(
        str(model_directory),
        "smplx",
        gender="male",
        use_pca=False,
        batch_size=batch_size,
    )
    orient = np.asarray(arrays["human_orient"][indices])
    full_pose = np.concatenate(
        (orient[:, None], arrays["human_pose"][indices].reshape(-1, 21, 3)), axis=1
    )
    rotations = compute_world_joint_orientations(
        orient, full_pose, model.parents.cpu().numpy()
    )
    markers, vertices, joints = {}, [], []
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        padded = np.pad(chosen, (0, batch_size - len(chosen)), mode="edge")
        values = {
            key: torch.as_tensor(np.array(arrays[name][padded]), dtype=torch.float32)
            for key, name in (
                ("global_orient", "human_orient"),
                ("body_pose", "human_pose"),
                ("transl", "human_transl"),
                ("left_hand_pose", "left_hand_pose"),
                ("right_hand_pose", "right_hand_pose"),
            )
        }
        values["betas"] = torch.zeros((batch_size, 10))
        for key in ("jaw_pose", "leye_pose", "reye_pose"):
            values[key] = torch.zeros((batch_size, 3))
        values["expression"] = torch.zeros((batch_size, 10))
        with torch.no_grad():
            output = model(**values, return_verts=True)
        batch_rotations = np.pad(
            rotations[start : start + len(chosen)],
            ((0, batch_size - len(chosen)), (0, 0), (0, 0)),
            mode="edge",
        )
        extracted = _extract_smplx_end_effector_markers(
            model, output, world_joint_orientations=batch_rotations
        )
        for name, entry in extracted.items():
            for key, value in entry.items():
                if key != "vertex_ids":
                    markers.setdefault(name, {}).setdefault(key, []).append(
                        value[: len(chosen)] @ WORLD_ROTATION.T
                    )
        vertices.append(output.vertices[: len(chosen)].cpu().numpy() @ WORLD_ROTATION.T)
        joints.append(
            output.joints[: len(chosen), :22].cpu().numpy() @ WORLD_ROTATION.T
        )
    source = np.concatenate(joints)
    recorded = (
        np.asarray(np.load(directory / "human_joints.npy", mmap_mode="r")[indices, :22])
        @ WORLD_ROTATION.T
    )
    error = float(np.linalg.norm(source - recorded, axis=-1).max())
    # The release stores joints with slightly different numeric precision.
    # Report the discrepancy; never fit the reconstructed skin to erase it.
    if error > 1e-3:
        raise ValueError(
            f"SMPL-X reconstruction differs from recorded joints by {error} m"
        )
    # The released joint observations remain the immutable skeleton reference.
    markers = {
        name: {key: np.concatenate(value) for key, value in entry.items()}
        for name, entry in markers.items()
    }
    with torch.no_grad():
        rest = model(betas=torch.zeros((batch_size, 10)), return_verts=True)
    surface = sample_smplx_surfaces(
        model,
        rest,
        np.concatenate(vertices),
        WORLD_ROTATION,
        samples_per_part=64,
        whole_body_samples=whole_body_samples,
    )
    world_rotations = (
        (
            Rotation.from_matrix(WORLD_ROTATION)
            * Rotation.from_quat(rotations.reshape(-1, 4), scalar_first=True)
        )
        .as_quat(scalar_first=True)
        .reshape(rotations.shape)
    )
    from .smplx_inertia import skin_face_regions, source_inertia

    dynamics = source_inertia(
        surface["full_body"],
        recorded,
        world_rotations,
        skin_face_regions(model, len(rotations[0])),
        reference_rotation=WORLD_ROTATION,
        framerate=30.0,
    )
    return {
        "positions": recorded,
        "orientations": world_rotations,
        "markers": markers,
        "surface": surface,
        "indices": indices,
        "reconstruction_error_m": error,
        "source_dynamics": dynamics,
    }
