"""Expose an explicit SMPL-X geometric inertia approximation to source dynamics."""

import numpy as np

from ..source_inertia import mesh_source_inertia


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
    result = mesh_source_inertia(
        full_body,
        positions,
        orientations,
        face_regions,
        reference_rotation=reference_rotation,
        framerate=framerate,
    )
    result["demand"]["inertia_source"] = "capped_smplx_segments_uniform_density"
    return result
