"""Persistent SMPL-X skin samples for whole-body pose and contact.

Sampling is area weighted (trimesh), once on the shaped rest mesh. Face IDs
and barycentric coordinates survive posing and resampling. These are observed
skin surfaces, never capsules fitted to joint centres.
"""

import numpy as np
import trimesh


def sample_smplx_surfaces(
    model, reference, posed_vertices, canonical_to_world, samples_per_part=256,
    whole_body_samples=2048,
):
    vertices = reference.vertices.detach().cpu().numpy()[0]
    faces = np.asarray(model.faces, dtype=int)
    weights = model.lbs_weights.detach().cpu().numpy()
    parts = {}
    for side, hip, wrist in (("left", 1, 20), ("right", 2, 21)):
        prefix = "L" if side == "left" else "R"
        # Wrist-dominated skin selects the palm/wrist, excluding articulated
        # fingers that have no counterpart on a rigid robot hand.
        for name, joint, end, effector in (
            (f"{side}_hand", wrist, None, f"{side}_hand"),
            (f"{side}_thigh", hip, f"{prefix}_Knee", None),
        ):
            mask = weights[:, joint] > 0.5
            face_ids = np.flatnonzero(np.all(mask[faces], axis=1))
            if not len(face_ids):
                continue
            mesh = trimesh.Trimesh(vertices, faces[face_ids], process=False)
            points, selected = trimesh.sample.sample_surface(
                mesh,
                samples_per_part,
                seed=joint,
            )
            selected_ids = face_ids[selected]
            indices = faces[selected_ids]
            bary = trimesh.triangles.points_to_barycentric(vertices[indices], points)
            motion_triangles = np.asarray(posed_vertices[:, indices], dtype=float)
            parts[name] = {
                "source_joint": f"{prefix}_{'Wrist' if effector else 'Hip'}",
                "source_end_joint": end,
                "effector": effector,
                "face_ids": selected_ids,
                "barycentric": bary,
                "reference_triangles": vertices[indices] @ canonical_to_world.T,
                "reference_bounds": np.array(
                    [
                        (vertices[mask] @ canonical_to_world.T).min(axis=0),
                        (vertices[mask] @ canonical_to_world.T).max(axis=0),
                    ]
                ),
                "triangles": motion_triangles,
            }
    # Whole-body correspondence has its own global, area-weighted indexing.
    # Labels group diagnostics; dominant joints and ancestry constrain the
    # learned correspondence through the caller's existing joint mapping.
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    points, face_ids = trimesh.sample.sample_surface(mesh, whole_body_samples, seed=17)
    indices = faces[face_ids]
    bary = trimesh.triangles.points_to_barycentric(vertices[indices], points)
    joint = np.argmax(np.einsum("nij,ni->nj", weights[indices], bary), axis=1)
    parents = model.parents.detach().cpu().numpy()
    for i in range(len(joint)):
        while joint[i] >= 22:
            joint[i] = parents[joint[i]]
    names = ["pelvis", "left_thigh", "right_thigh", "lower_torso", "left_shin", "right_shin",
             "middle_torso", "left_foot", "right_foot", "upper_torso", "left_toes", "right_toes",
             "neck", "left_clavicle", "right_clavicle", "head", "left_upper_arm", "right_upper_arm",
             "left_forearm", "right_forearm", "left_hand", "right_hand"]
    from .smplx import DEFAULT_SMPLX_TARGET_NAMES
    effectors = {7: "left_foot", 10: "left_foot", 8: "right_foot", 11: "right_foot",
                 20: "left_hand", 21: "right_hand"}
    return {
        "kind": "barycentric_skin",
        "parts": parts,
        "reference_pose": "shaped_rest_z_up",
        "metric_scale": 1.0,
        "full_body": {
            "reference_vertices": vertices @ canonical_to_world.T,
            "faces": faces.copy(),
            "face_ids": face_ids,
            "barycentric": bary,
            "triangles": np.asarray(posed_vertices[:, indices], dtype=np.float32),
            "labels": np.asarray(names)[joint],
            "joint_names": np.asarray(DEFAULT_SMPLX_TARGET_NAMES),
            "joint_parents": parents[:22].copy(),
            "sample_joint_indices": joint.copy(),
            "sample_effectors": np.array([effectors.get(int(i), "") for i in joint]),
            "sample_world_tracking": np.isin(joint, [7, 8, 10, 11]),
            "reference_root_position": (
                reference.joints.detach().cpu().numpy()[0, 0] @ canonical_to_world.T
            ),
            "reference_joint_positions": (
                reference.joints.detach().cpu().numpy()[0, :22] @ canonical_to_world.T
            ),
        },
    }
