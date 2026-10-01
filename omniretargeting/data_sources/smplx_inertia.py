"""Expose an explicit SMPL-X geometric inertia approximation to source dynamics."""

import numpy as np
from scipy.spatial.transform import Rotation

from ..centroidal_demand import estimate_centroidal_demand
from ..mesh_inertia import segment_inertias


def skin_face_regions(model, joint_count):
    weights = model.lbs_weights.detach().cpu().numpy()
    parents = model.parents.detach().cpu().numpy()
    collapsed = np.zeros((len(weights), joint_count))
    for index in range(weights.shape[1]):
        parent = index
        while parent >= joint_count:
            parent = int(parents[parent])
        if parent < 0:
            raise ValueError("SMPL-X skin joint has no observed ancestor")
        collapsed[:, parent] += weights[:, index]
    return np.argmax(collapsed[np.asarray(model.faces)].mean(axis=1), axis=1)


def source_inertia(
    full_body, positions, orientations, face_regions, *, reference_rotation, framerate
):
    """Rigid-segment estimate in source coordinates, using the unmodified skin.

    Relative masses follow capped segment volumes; this is an approximation,
    not an anthropometric measurement. Callers must retain its assumptions and
    derivative disagreement when interpreting contact feasibility.
    """
    properties = segment_inertias(
        full_body["reference_vertices"], full_body["faces"], face_regions
    )
    ids = properties["regions"].astype(int)
    reference_joints = np.asarray(full_body["reference_joint_positions"])[ids]
    orientations = np.asarray(orientations)
    rotation = Rotation.from_quat(
        orientations[:, ids].reshape(-1, 4), scalar_first=True
    ).as_matrix()
    rotation = (
        rotation.reshape(len(orientations), len(ids), 3, 3)
        @ np.asarray(reference_rotation).T
    )
    centers = np.asarray(positions)[:, ids] + np.einsum(
        "tbij,bj->tbi", rotation, properties["centers"] - reference_joints
    )
    demand = estimate_centroidal_demand(
        centers,
        rotation,
        properties["masses"],
        properties["inertias"],
        framerate=framerate,
    )
    demand["inertia_source"] = "capped_smplx_segments_uniform_density"
    return {
        "centers": centers,
        "rotations": rotation,
        "properties": properties,
        "demand": demand,
    }
