"""Rigid frames fitted to persistent material points of a deforming surface."""

import numpy as np


def material_surface_frames(reference_points, world_points, reference_frame):
    """Fit SO(3) with fixed correspondences, without moving surface points.

    Unlike a fresh PCA on each frame, material correspondences retain axis
    identity during rolls and bends. A planar patch is sufficient; a line is
    not. Return the deformation residual separately from the fitted frame.
    """
    reference = np.asarray(reference_points, dtype=float)
    world = np.asarray(world_points, dtype=float)
    frame = np.asarray(reference_frame, dtype=float)
    if reference.ndim != 2 or reference.shape[1:] != (3,) or len(reference) < 3:
        raise ValueError("Reference patch requires at least three 3D points")
    if world.ndim != 3 or world.shape[1:] != reference.shape:
        raise ValueError("World patches must preserve reference point identities")
    if not all(np.isfinite(x).all() for x in (reference, world, frame)):
        raise ValueError("Surface frame inputs must be finite")
    if (frame.shape != (3, 3) or not np.allclose(frame.T @ frame, np.eye(3))
            or not np.isclose(np.linalg.det(frame), 1.0)):
        raise ValueError("Reference frame must be a proper rotation")
    centered = reference - reference.mean(axis=0)
    posed = world - world.mean(axis=1, keepdims=True)
    covariance = np.einsum("ni,tnj->tij", centered, posed)
    left, singular, right = np.linalg.svd(covariance)
    if np.any(singular[:, 1] <= np.maximum(singular[:, 0], 1e-20) * 1e-8):
        raise ValueError("Surface patch is degenerate; rotation is unobservable")
    handedness = np.broadcast_to(np.eye(3), covariance.shape).copy()
    handedness[:, 2, 2] = np.linalg.det(left @ right)
    transport = left @ handedness @ right
    rotations = np.swapaxes(transport, 1, 2) @ frame
    fitted = np.einsum("ni,tij->tnj", centered, transport)
    rms = np.sqrt(np.mean(np.sum((posed - fitted) ** 2, axis=2), axis=1))
    return rotations, rms
