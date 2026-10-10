"""Small dual certificates for unsupported circular-cone base wrenches."""

import mujoco
import numpy as np

from .effort_dynamics import FORCE_TOLERANCE, FRICTION


def certify_circular_unsupported(model, data, demand, contacts, base_dofs, friction=FRICTION):
    """Prove nonexistence, rather than relabel a failed primal numerical solve.

    For every contact local force (fn, ft), the dual cone requires
    gn >= mu*|gt|. If a base vector y satisfies all those inequalities and
    y*d is negative, no nonnegative circular-cone forces can balance d.
    The solve has only len(base_dofs) variables regardless of patch density.
    A small positive search margin makes verification conservative; it is not
    slack in force balance, friction or motor limits.
    """
    import clarabel
    from scipy import sparse

    if not contacts or not len(base_dofs):
        return None
    blocks = []
    for contact in contacts:
        normal = np.asarray(contact["normal"], float)
        normal = normal / np.linalg.norm(normal)
        tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
        tangent /= np.linalg.norm(tangent)
        orientation = np.c_[normal, tangent, np.cross(normal, tangent)]
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, contact["point"], contact["body"])
        blocks.append(orientation.T @ jacobian[:, base_dofs])
    mapping = np.asarray(blocks)
    cone_mapping = mapping * np.array([1., friction, friction])[None, :, None]
    margin = 1e-8
    rhs = np.tile([-margin, 0., 0.], len(contacts))
    required = demand[base_dofs]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-10
    settings.max_iter = 200
    try:
        result = clarabel.DefaultSolver(
            sparse.eye(len(base_dofs), format="csc"), required,
            sparse.csc_matrix(-cone_mapping.reshape(-1, len(base_dofs))), rhs,
            [clarabel.SecondOrderConeT(3) for _ in contacts], settings).solve()
    except (ValueError, RuntimeError):
        return None
    if str(result.status) not in {"Solved", "AlmostSolved"}:
        return None
    separator = np.asarray(result.x)
    gradient = mapping @ separator
    slacks = gradient[:, 0] - friction * np.linalg.norm(gradient[:, 1:], axis=1)
    separation = float(required @ separator)
    residual_allowance = FORCE_TOLERANCE * float(np.abs(separator).sum())
    # Check the actual coefficients with no relaxed dual-cone inequality.
    if not np.isfinite(separator).all() or slacks.min() < 0 or separation >= -residual_allowance:
        return None
    return {"circular_base_separator": separator.tolist(),
            "circular_base_separator_dot_demand": separation,
            "circular_base_separator_min_cone_slack": float(slacks.min()),
            "circular_base_separator_residual_allowance": residual_allowance}
