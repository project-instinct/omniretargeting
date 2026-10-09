"""Contact-aware effort feasibility and source-fit-relative diagnostics."""

import mujoco
import numpy as np

from .effort_contacts import CONTACT_GAP_M, frame_contacts
from .effort_dynamics import (
    FRICTION, FORCE_TOLERANCE, allocate, contact_matrix, derivatives, friction_rays, motor_demand,
)


def _allocation(model, data, demand, contacts, base, dofs, limits, *, outer=False, friction=FRICTION):
    matrix = contact_matrix(model, data, contacts, friction, outer=outer)
    result = allocate(demand, matrix, base, dofs, limits)
    return result, matrix


def _within_limits(result, limits):
    return result["status"] == "balanced" and np.all(
        np.abs(result["torque"]) <= limits + FORCE_TOLERANCE
    )


def evaluate_effort(model, poses, fps, joints, limits, evidence=None, *, kinematics=None):
    """No unconstrained root wrench, no smoothing, no arbitrary-force certificate."""
    frames, velocity, acceleration = derivatives(model, poses if kinematics is None else kinematics, fps)
    dofs = model.jnt_dofadr[joints]
    base = np.setdiff1d(np.arange(model.nv), dofs)
    data = mujoco.MjData(model)
    rows, demands, torques = [], [], []
    for index, frame in enumerate(frames):
        demand = motor_demand(model, data, poses[frame], velocity[index], acceleration[index])
        attained, optimistic, geometry = frame_contacts(evidence, model, data, frame)
        result, _ = _allocation(model, data, demand, attained, base, dofs, limits)
        status = "feasible"
        optimistic_result = None
        if not _within_limits(result, limits):
            # Infeasibility of an inner friction polygon does not prove
            # infeasibility of the true circular cone. Also grant every
            # source-correspondence contact before declaring unavoidable load.
            optimistic_result, _ = _allocation(model, data, demand, optimistic, base, dofs, limits, outer=True)
            if optimistic_result["status"] != "balanced":
                status = "unknown_base_balance"
            # If every |tau_j| <= effort_j + eps, maximum utilization cannot
            # exceed 1 + eps/min(effort). This proves overload with a tolerance
            # in force units, rather than multiplying roundoff by motor size.
            elif optimistic_result["utilization"] > 1 + FORCE_TOLERANCE / np.min(limits):
                status = "overload_even_with_source_contacts"
            else:
                status = "unknown_contact_or_cone"
        if geometry["unmodelled_object_contacts"]:
            status = "unknown_object_load"
        if geometry.get("invalid_normals"):
            status = "unknown_contact_geometry"
        if not np.isfinite(limits).all():
            status = "unknown_effort_limits"
        witness = optimistic_result if status == "overload_even_with_source_contacts" else (
            result if result["status"] == "balanced" else optimistic_result
        )
        torque = witness["torque"] if witness and witness["status"] == "balanced" else np.full(len(joints), np.nan)
        # A force distribution is a witness, never a measured load. Unavoidable
        # overload is certified by the relaxed problem, not this witness alone.
        rows.append({
            "frame": int(frame), "status": status,
            "minimum_peak_utilization": result["utilization"],
            "optimistic_outer_utilization": optimistic_result["utilization"] if optimistic_result else None,
            "optimistic_base_residual": optimistic_result.get("base_residual") if optimistic_result else None,
            "base_residual": result.get("base_residual"),
            **geometry,
        })
        witness_contacts = optimistic if status == "overload_even_with_source_contacts" else attained
        if status in {"feasible", "overload_even_with_source_contacts"} and len(witness_contacts):
            coefficients = witness["coefficients"].reshape(len(witness_contacts), -1)
            # Enough evidence to inspect the contact split on any audited frame.
            rows[-1]["loaded_contacts"] = [
                {"geom": contact["geom"], "surface": contact["surface"],
                 "slot": contact["slot"], "part": contact["part"],
                 "source_support": contact["source_support"],
                 "normal_force_N": float(weights.sum()),
                 "force_w_N": (friction_rays(contact["normal"], outer=status != "feasible") @ weights).tolist(),
                 "point_w_m": contact["point"].tolist(),
                 "witness": "attained_inner_cone" if status == "feasible" else "source_outer_relaxation",
                 "correspondence_error_m": contact["correspondence_error_m"]}
                for contact, weights in zip(witness_contacts, coefficients) if weights.sum() > FORCE_TOLERANCE
            ]
        demands.append(demand[dofs])
        torques.append(torque)
    return {
        "frames": frames, "velocity": velocity, "acceleration": acceleration,
        "demand": np.asarray(demands).reshape(-1, len(joints)),
        "torque": np.asarray(torques).reshape(-1, len(joints)), "rows": rows,
    }


def effort_report(model, before, after, fps, joints, effort_limits, evidence=None):
    names = [model.joint(int(joint)).name for joint in joints]
    limits = np.array([effort_limits.get(name, np.inf) for name in names])
    # Unknown limits do not enter linprog as infinities; they cannot pass later.
    solve_limits = np.where(np.isfinite(limits), limits, 1e12)
    final = evaluate_effort(model, after, fps, joints, solve_limits, evidence)
    reference = evaluate_effort(model, before, fps, joints, solve_limits, evidence)
    # Same final configuration and contact geometry, but source-fit derivatives:
    # isolates the additional inertial demand without moving any contact target.
    inherited = evaluate_effort(model, after, fps, joints, solve_limits, evidence, kinematics=before)
    if not np.isfinite(limits).all():
        for evaluation in (final, reference, inherited):
            for row in evaluation["rows"]:
                row["status"] = "unknown_effort_limits"
    valid = np.array([row["status"] == "feasible" for row in final["rows"]], dtype=bool)
    overload = np.array([row["status"] == "overload_even_with_source_contacts" for row in final["rows"]], dtype=bool)
    effort = np.abs(final["torque"])
    # Counts describe joint/frame entries in the minimax witness; the proof
    # pertains to a frame's impossibility of satisfying every motor together.
    over = overload[:, None] & (effort > limits + FORCE_TOLERANCE)
    reference_over = np.array([r["status"] == "overload_even_with_source_contacts" for r in reference["rows"]], dtype=bool)
    inherited_over = np.array([r["status"] == "overload_even_with_source_contacts" for r in inherited["rows"]], dtype=bool)
    attribution = {}
    data = mujoco.MjData(model)
    dofs = model.jnt_dofadr[joints]
    base = np.setdiff1d(np.arange(model.nv), dofs)
    for index in np.flatnonzero(overload):
        frame = int(final["frames"][index])
        demand = motor_demand(model, data, before[frame], reference["velocity"][index], reference["acceleration"][index])
        _, source_contacts, _ = frame_contacts(evidence, model, data, frame)
        source_inner, _ = _allocation(model, data, demand, source_contacts, base, dofs, solve_limits)
        source_feasible = _within_limits(source_inner, solve_limits)
        if not inherited_over[index] and inherited["rows"][index]["status"] == "feasible":
            classification = "added_inertial_demand"
        elif source_feasible and inherited_over[index]:
            classification = "added_pose_under_source_contact_assumption"
        elif reference_over[index]:
            classification = "source_fit_inherited"
        else:
            classification = "source_fit_or_geometry_not_separable"
        demand = motor_demand(model, data, after[frame], final["velocity"][index], final["acceleration"][index])
        _, source_contacts, _ = frame_contacts(evidence, model, data, frame)
        sensitivity = {}
        for friction in (0.3, 0.9):
            allocation, _ = _allocation(model, data, demand, source_contacts, base, dofs, solve_limits, outer=True, friction=friction)
            sensitivity[str(friction)] = {"status": allocation["status"], "minimum_peak_utilization": allocation["utilization"]}
        attribution[index] = {
            "frame": frame, "source_fit_overload": bool(reference_over[index]),
            "source_inner_cone_feasible": bool(source_feasible),
            "source_inner_cone_utilization": source_inner["utilization"],
            "source_derivatives_at_final_pose_overload": bool(inherited_over[index]),
            "classification": classification, "friction_sensitivity_outer_cone": sensitivity,
        }
    by_joint = {}
    for column, name in enumerate(names):
        measured = np.isfinite(effort[:, column]) & (valid | overload)
        worst = np.flatnonzero(measured)
        worst_index = int(worst[np.argmax(effort[worst, column])]) if len(worst) else None
        bad = np.flatnonzero(over[:, column])
        correction_acceleration = final["acceleration"][:, model.jnt_dofadr[joints[column]]] - reference["acceleration"][:, model.jnt_dofadr[joints[column]]]
        by_joint[name] = {
            "limit": effort_limits.get(name),
            "peak": float(effort[worst_index, column]) if worst_index is not None else None,
            "worst_frame": int(final["frames"][worst_index]) if worst_index is not None else None,
            "violations": int(len(bad)), "violation_frames": final["frames"][bad].tolist(),
            "contact_force_split": True,
            "contact_free_demand_peak": float(np.abs(final["demand"][:, column]).max(initial=0)),
            "contact_free_demand_worst_frame": int(final["frames"][np.argmax(np.abs(final["demand"][:, column]))]) if len(final["frames"]) else None,
            "reference_contact_free_demand_peak": float(np.abs(reference["demand"][:, column]).max(initial=0)),
            "reference_acceleration_peak": float(np.abs(reference["acceleration"][:, model.jnt_dofadr[joints[column]]]).max(initial=0)),
            "final_acceleration_peak": float(np.abs(final["acceleration"][:, model.jnt_dofadr[joints[column]]]).max(initial=0)),
            "added_acceleration_peak": float(np.abs(correction_acceleration).max(initial=0)),
            "added_acceleration_worst_frame": int(final["frames"][np.argmax(np.abs(correction_acceleration))]) if len(final["frames"]) else None,
            "attribution": [attribution[index] for index in bad],
        }
    counts = {status: sum(row["status"] == status for row in final["rows"])
              for status in sorted({row["status"] for row in final["rows"]})}
    unknown = [row["frame"] for row in final["rows"] if row["status"].startswith("unknown")]
    endpoint_frames = sorted(set(range(len(after))) - set(final["frames"].tolist()))
    return by_joint, {
        "method": "centered M*qacc + bias - passive + dry_friction - J.T*f; minimax effort LP",
        "friction_coefficient": FRICTION, "friction_cone": "8-ray inner/outer circular-cone bracket",
        "contact_correspondence_gate_m": CONTACT_GAP_M,
        "floating_base_balance": "exact; no fictitious root actuation",
        "source_contacts": "all geometrically observed collision-geom patches; optional contacts may carry load",
        "object_support": "only original-geometry grounded and still objects; other object loads unknown",
        "effort_limit_violations": int(over.sum()),
        "unavoidable_overload_frames": final["frames"][overload].tolist(),
        "unknown_frames": unknown, "unknown_frame_count": len(unknown),
        "unaudited_endpoint_frames": endpoint_frames,
        "missing_effort_limits": [name for name in names if name not in effort_limits],
        "frame_status_counts": counts, "frame_diagnostics": final["rows"],
        "reference_frame_status_counts": {status: sum(row["status"] == status for row in reference["rows"])
                                          for status in sorted({r["status"] for r in reference["rows"]})},
        "source_fit_unavoidable_overload_frames": reference["frames"][reference_over].tolist(),
        "source_derivatives_at_final_pose_overload_frames": inherited["frames"][inherited_over].tolist(),
        "overload_frame_attribution": list(attribution.values()),
        "derivative_method": "raw centered 3-frame stencil; no filtering or retiming; quaternion tangent at middle frame",
        "position_noise_sensitivity": {"assumed_error_rad": 0.001,
                                       "worst_second_difference_error_rad_s2": 4 * 0.001 * fps ** 2,
                                       "measured_noise_level": None},
        "physical_execution_verified": False,
        "interpretation": "Instantaneous feasibility witness conditional on source patch correspondences/friction. Unknown and unaudited frames do not pass. Source-fit robot overload is not proof of human-source overload.",
    }
