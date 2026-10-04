"""Source-only segment dynamics, independent of skin and motion file format."""

import numpy as np
from scipy.spatial.transform import Rotation

from .centroidal_demand import estimate_centroidal_demand
from .mesh_inertia import segment_inertias


def segment_source_inertia(model, positions, orientations, joint_names, *, framerate):
    """Evaluate an adapter-declared rigid-segment model on the original motion.

    Segment COM offsets and COM inertia tensors use their anchor joint's local
    axes. Anchors may repeat when a joint has several outgoing rigid segments.
    No robot model, scene registration, or motion correction enters this path.
    """
    names = list(joint_names)
    anchors = [names.index(name) for name in model["joint_names"]]
    centers_local = np.asarray(model["local_centers"], float)
    positions, orientations = np.asarray(positions), np.asarray(orientations)
    if centers_local.shape != (len(anchors), 3):
        raise ValueError("One local segment COM per source anchor is required")
    rotation = (
        Rotation.from_quat(orientations[:, anchors].reshape(-1, 4), scalar_first=True)
        .as_matrix()
        .reshape(len(positions), len(anchors), 3, 3)
    )
    centers = positions[:, anchors] + np.einsum("tbij,bj->tbi", rotation, centers_local)
    masses = np.asarray(model["masses"], float)
    inertias = np.asarray(model["inertias"], float)
    demand = estimate_centroidal_demand(
        centers, rotation, masses, inertias, framerate=framerate
    )
    demand["inertia_source"] = model.get("method", "adapter_segment_model")
    demand["inertia_assumptions"] = dict(model.get("assumptions", {}))
    return {
        "centers": centers,
        "rotations": rotation,
        "properties": {"masses": masses, "inertias": inertias},
        "demand": demand,
    }


def mesh_source_inertia(
    full_body, positions, orientations, face_regions, *, reference_rotation, framerate
):
    """Uniform-density labelled source-mesh approximation, with capped cuts.

    This is geometric input, not a measured subject inertia. SMPL-X adapters
    extract their own segmentation; this calculation consumes the common mesh
    and rigid-joint schema without importing a source-specific loader.
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
    demand["inertia_source"] = "capped_source_mesh_segments_uniform_density"
    demand["inertia_assumptions"] = dict(properties["assumptions"])
    return {
        "centers": centers,
        "rotations": rotation,
        "properties": properties,
        "demand": demand,
    }
