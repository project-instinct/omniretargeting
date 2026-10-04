"""Explicit line-segment inertia approximation for skeleton-only adapters."""

import numpy as np


def skeleton_segment_model(names, parents, offsets):
    """Unit-mass uniform linear density on the source's nonzero BVH bones.

    ``offsets`` are in metres in each parent joint's local axes. A segment COM
    is halfway along its bone, not at the joint origin. The thin-rod COM tensor
    is m/12 * (|v|² I - v vᵀ), including its zero longitudinal spin inertia.
    This is an explicit geometric approximation, not anthropometric evidence;
    it does not synthesize skin or create any scene contacts.
    """
    parents = np.asarray(parents, int)
    offsets = np.asarray(offsets, float)
    if (
        parents.shape != (len(names),)
        or offsets.shape != (len(names), 3)
        or not np.isfinite(offsets).all()
        or np.any(parents >= len(names))
        or np.any(parents < -1)
    ):
        raise ValueError("Skeleton inertia needs finite offsets and valid parents")
    lengths = np.linalg.norm(offsets, axis=1)
    ids = np.flatnonzero((parents >= 0) & (lengths > 1e-10))
    if not len(ids):
        return None
    vectors = offsets[ids]
    mass = lengths[ids] / lengths[ids].sum()
    tensor = (
        (
            lengths[ids, None, None] ** 2 * np.eye(3)
            - vectors[:, :, None] * vectors[:, None, :]
        )
        * mass[:, None, None]
        / 12.0
    )
    return {
        "joint_names": [names[parents[i]] for i in ids],
        "segment_names": [f"{names[parents[i]]}->{names[i]}" for i in ids],
        "local_centers": vectors / 2.0,
        "masses": mass,
        "inertias": tensor,
        "method": "source_skeleton_uniform_linear_density",
        "assumptions": {
            "density": "uniform_source_bone_length",
            "mass_scale": 1.0,
            "segment_geometry": "thin_rods",
            "subject_measurement": False,
            "skin_reconstructed": False,
        },
    }
