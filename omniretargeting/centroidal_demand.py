"""Estimate source centroidal demand from explicitly supplied segment inertias."""

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


def estimate_centroidal_demand(
    centers,
    rotations,
    masses,
    inertias,
    *,
    framerate,
    origin=(0.0, 0.0, 0.0),
    gravity=(0.0, 0.0, -9.81),
    window=None,
    window_seconds=0.2,
):
    """World COM and angular-momentum rates; input motion is never filtered in place.

    centers: (T,B,3) segment COMs, NOT joint origins. rotations: (T,B,3,3)
    segment-to-world rotations. inertias: (B,3,3), about each segment's COM in
    its local axes. Differentiation uses local cubic fits; the returned window
    disagreement is a noise diagnostic, not a statistical confidence bound.
    """
    centers, rotations = np.asarray(centers, float), np.asarray(rotations, float)
    masses, inertias = np.asarray(masses, float), np.asarray(inertias, float)
    origin, gravity = np.asarray(origin, float), np.asarray(gravity, float)
    if (
        centers.ndim != 3
        or centers.shape[-1] != 3
        or centers.shape[1] == 0
        or rotations.shape != (*centers.shape[:2], 3, 3)
        or masses.shape != (centers.shape[1],)
        or inertias.shape != (len(masses), 3, 3)
        or origin.shape != (3,)
        or gravity.shape != (3,)
        or not all(
            np.isfinite(x).all()
            for x in (centers, rotations, masses, inertias, origin, gravity)
        )
        or np.any(masses <= 0)
        or not np.isfinite(framerate)
        or framerate <= 0
        or not np.allclose(inertias, inertias.swapaxes(-1, -2))
        or np.any(np.linalg.eigvalsh(inertias) < -1e-12)
        or not np.allclose(rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=1e-6)
        or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6)
    ):
        raise ValueError(
            "Finite, consistent segment COMs, rotations, positive masses and inertias are required"
        )
    if window is None:
        if not np.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("Derivative window duration must be finite and positive")
        window = max(5, int(round(window_seconds * framerate)) | 1)
    if (
        not isinstance(window, int)
        or window < 5
        or window % 2 == 0
        or len(centers) < window + 2
    ):
        raise ValueError(
            "An odd derivative window >= 5 needs at least window + 2 frames"
        )
    total_mass = masses.sum()
    com = np.einsum("tbi,b->ti", centers, masses) / total_mass
    # Relative rotations are measured in world axes, not by differentiating Euler angles.
    relative = rotations[1:] @ rotations[:-1].swapaxes(-1, -2)
    midpoint_omega = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
    midpoint_omega = midpoint_omega.reshape(-1, len(masses), 3) * framerate
    omega = np.concatenate(
        (
            midpoint_omega[:1],
            0.5 * (midpoint_omega[:-1] + midpoint_omega[1:]),
            midpoint_omega[-1:],
        )
    )
    world_inertia = rotations @ inertias @ rotations.swapaxes(-1, -2)

    def estimate(width):
        def derivative(values, order):
            return savgol_filter(
                values,
                width,
                3,
                deriv=order,
                delta=1 / framerate,
                axis=0,
                mode="interp",
            )

        velocity = derivative(centers, 1)
        com_velocity = np.einsum("tbi,b->ti", velocity, masses) / total_mass
        spin = np.einsum(
            "tbij,tbj->tbi",
            world_inertia,
            savgol_filter(omega, width, 3, axis=0, mode="interp"),
        )
        angular = (
            np.cross(
                centers - com[:, None],
                masses[None, :, None] * (velocity - com_velocity[:, None]),
            )
            + spin
        ).sum(axis=1)
        force = total_mass * (derivative(com, 2) - gravity)
        torque_com = derivative(angular, 1)
        torque_origin = torque_com + np.cross(com - origin, force)
        return np.c_[force, torque_origin], angular, torque_com

    wrench, angular, torque = estimate(window)
    wider, _, _ = estimate(window + 2)
    return {
        "com": com,
        "angular_momentum_com": angular,
        "torque_com": torque,
        "wrench": wrench,
        "origin": origin.copy(),
        "derivative_disagreement": np.abs(wrench - wider),
        "edge_frames": np.r_[
            np.arange(window // 2), np.arange(len(centers) - window // 2, len(centers))
        ],
        "mass": float(total_mass),
        "derivative_window": window,
        "derivative_window_seconds": window / framerate,
        "inertia_source": "caller_supplied_segment_model",
    }
