"""Contact-aware effort feasibility and source-fit-relative diagnostics."""

import mujoco
import numpy as np

from .effort_contacts import CONTACT_GAP_M, frame_contacts
from .effort_dynamics import (
    FRICTION, FORCE_TOLERANCE, allocate, allocate_circular, balance_residual,
    derivatives, friction_rays, motor_demand,
)
from .effort_geometry import SENSITIVITY_GAP_M
from .effort_force_columns import reduced_contact_matrix
from .effort_certificate import certify_circular_unsupported


def _allocation(model, data, demand, contacts, base, dofs, limits, *, outer=False, friction=FRICTION):
    matrix, indices, count = reduced_contact_matrix(model, data, contacts, friction, outer=outer)
    result = allocate(demand, matrix, base, dofs, limits)
    if result["status"] == "balanced":
        expanded = np.zeros(count)
        expanded[indices] = result["coefficients"]
        result["coefficients"] = expanded
    return result, matrix


def _within_limits(result, limits):
    return result["status"] == "balanced" and np.all(
        np.abs(result["torque"]) <= limits + FORCE_TOLERANCE
    )


def _decision(model, data, demand, contacts, geometry, base, dofs, limits):
    inner, _ = _allocation(model, data, demand, contacts, base, dofs, limits)
    # An attained inner-cone witness already proves feasibility; another force
    # allocation cannot strengthen that certificate.
    outer, matrix = (None, None) if _within_limits(inner, limits) else _allocation(
        model, data, demand, contacts, base, dofs, limits, outer=True)
    witness, kind = inner, "actual_inner_cone"
    if _within_limits(inner, limits):
        status, reason = "feasible", "attained_contact_force_witness"
    elif outer["status"] == "base_infeasible":
        status = "infeasible_unsupported"
        reason = "no_attained_contact" if not contacts else "friction_cone_or_support_wrench"
        missing_support = geometry.get("missing_source_support_pairs", [])
        if not contacts and any(p["gap_m"] < -CONTACT_GAP_M for p in missing_support):
            reason = "penetrating_support_geom"
        elif not contacts and any(p["gap_m"] > CONTACT_GAP_M for p in missing_support):
            reason = "source_support_not_attained"
        witness, kind = outer, "actual_outer_cone"
    elif outer["status"] == "balanced" and outer["utilization"] > 1 + FORCE_TOLERANCE / np.min(limits):
        status, reason = "overload", "actual_outer_cone_effort_lower_bound"
        witness, kind = outer, "actual_outer_cone"
    else:
        circular = allocate_circular(model, data, demand, contacts, base, dofs, limits)
        witness, kind = circular, "actual_circular_cone"
        if circular["status"] != "balanced":
            certificate = certify_circular_unsupported(model, data, demand, contacts, base)
            if certificate:
                status, reason = "infeasible_unsupported", "verified_circular_cone_separating_wrench"
                geometry = {**geometry, **certificate}
            else:
                status, reason = "audit_error_numerical", "contact_force_solver_did_not_certify"
        elif _within_limits(circular, limits):
            status, reason = "feasible", "attained_circular_cone_force_witness"
        elif circular.get("dual_utilization_lower_bound") is not None and circular["dual_utilization_lower_bound"] > 1 + FORCE_TOLERANCE / np.min(limits):
            status, reason = "overload", "circular_cone_dual_effort_lower_bound"
        else:
            # A primal conic solution is an upper bound. Retain the uncertainty
            # if the outer LP lower bound has not proved motor overload.
            status, reason = "audit_error_cone_certificate", "circular_optimum_without_verified_lower_bound"
    if geometry.get("missing_evidence") and len(base):
        status, reason = "unknown_contact_evidence", "source_contact_arrays_missing"
    elif status == "infeasible_unsupported" and not geometry.get("original_triangle_geometry_available"):
        status, reason = "unknown_contact_geometry", "original_triangle_geometry_missing"
    if geometry.get("unmodelled_object_contacts"):
        status, reason = "unknown_object_load", "actual_contact_with_nonfixed_object_missing_mass_inertia"
    if geometry.get("uncertain_near_surface_points") or geometry.get("invalid_normals"):
        status, reason = "unknown_contact_geometry", "near_surface_free_side_or_normal_unknown"
    diagnostic = balance_residual(demand, matrix, base) if status == "infeasible_unsupported" else {}
    if status == "infeasible_unsupported" and "circular_base_separator" in geometry:
        diagnostic.update({k: v for k, v in geometry.items() if k.startswith("circular_base_separator")})
    return status, reason, witness, kind, outer, diagnostic


def evaluate_effort(model, poses, fps, joints, limits, evidence=None, *, kinematics=None, frame_indices=None):
    """No unconstrained root wrench, no smoothing, no arbitrary-force certificate."""
    frames, velocity, acceleration = derivatives(model, poses if kinematics is None else kinematics, fps)
    if frame_indices is not None:
        selected = np.isin(frames, list(frame_indices))
        frames, velocity, acceleration = (value[selected] for value in (frames, velocity, acceleration))
    dofs = model.jnt_dofadr[joints]
    base = np.setdiff1d(np.arange(model.nv), dofs)
    data = mujoco.MjData(model)
    rows, demands, torques = [], [], []
    for index, frame in enumerate(frames):
        demand = motor_demand(model, data, poses[frame], velocity[index], acceleration[index])
        attained, optimistic, geometry = frame_contacts(evidence, model, data, frame)
        status, reason, witness, kind, outer, residual = _decision(
            model, data, demand, attained, geometry, base, dofs, limits)
        optimistic_result, _ = _allocation(model, data, demand, optimistic, base, dofs, limits, outer=True)
        if not np.isfinite(limits).all():
            status = "unknown_effort_limits"
        torque = witness["torque"] if witness and witness["status"] == "balanced" else np.full(len(joints), np.nan)
        # A force distribution is a witness, never a measured load. Unavoidable
        # overload is certified by the relaxed problem, not this witness alone.
        rows.append({
            "frame": int(frame), "status": status, "reason": reason,
            "minimum_peak_utilization": witness["utilization"], "force_witness_kind": kind,
            "optimistic_outer_utilization": optimistic_result["utilization"] if optimistic_result else None,
            "optimistic_base_residual": optimistic_result.get("base_residual") if optimistic_result else None,
            "base_residual": witness.get("base_residual"),
            "source_relaxed_status": optimistic_result["status"],
            **geometry, **residual,
        })
        if evidence and evidence.get("geometry") is not None:
            tolerance_contacts, _, tolerance_geometry = frame_contacts(
                evidence, model, data, frame, tolerance=SENSITIVITY_GAP_M)
            ss, rr, ww, kk, _, _ = _decision(model, data, demand, tolerance_contacts,
                                             tolerance_geometry, base, dofs, limits)
            rows[-1]["tolerance_15mm"] = {"status": ss, "reason": rr,
                                           "minimum_peak_utilization": ww["utilization"],
                                           "force_witness_kind": kk}
        if status in {"feasible", "overload"} and len(attained):
            coefficients = witness["coefficients"].reshape(len(attained), -1)
            forces = witness.get("forces_w")
            # Enough evidence to inspect the contact split on any audited frame.
            rows[-1]["loaded_contacts"] = [
                {"geom": contact["geom"], "surface": contact["surface"],
                 "slot": contact["slot"], "part": contact["part"],
                 "source_support": contact["source_support"],
                 "normal_force_N": float(np.dot(forces[i], contact["normal"])) if forces is not None else float(weights.sum()),
                 "force_w_N": forces[i].tolist() if forces is not None else (friction_rays(contact["normal"], outer=kind == "actual_outer_cone") @ weights).tolist(),
                 "point_w_m": contact["point"].tolist(),
                 "witness": kind,
                 "correspondence_error_m": contact["correspondence_error_m"]}
                for i, (contact, weights) in enumerate(zip(attained, coefficients)) if np.linalg.norm(weights) > FORCE_TOLERANCE
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
    overload = np.array([row["status"] == "overload" for row in final["rows"]], dtype=bool)
    effort = np.abs(final["torque"])
    # Counts describe joint/frame entries in the minimax witness; the proof
    # pertains to a frame's impossibility of satisfying every motor together.
    over = overload[:, None] & (effort > limits + FORCE_TOLERANCE)
    reference_over = np.array([r["status"] == "overload" for r in reference["rows"]], dtype=bool)
    inherited_over = np.array([r["status"] == "overload" for r in inherited["rows"]], dtype=bool)
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
        elif reference["rows"][index]["status"] == "feasible" and inherited_over[index]:
            classification = "added_pose_or_contact_geometry"
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
    audit_errors = [row["frame"] for row in final["rows"] if row["status"].startswith("audit_error")]
    endpoint_frames = sorted(set(range(len(after))) - set(final["frames"].tolist()))
    return by_joint, {
        "method": "centered M*qacc + bias - passive + dry_friction - J.T*f; minimax effort LP",
        "friction_coefficient": FRICTION, "friction_cone": "8-ray bracket; exact circular SOCP resolves ambiguity",
        "actual_geometry_gate_m": CONTACT_GAP_M, "sensitivity_geometry_gate_m": SENSITIVITY_GAP_M,
        "floating_base_balance": "exact; no fictitious root actuation",
        "source_contacts": "all collision geoms queried against original triangles; source correspondences only diagnose lost support",
        "object_support": "only original-geometry grounded and still objects; other object loads unknown",
        "effort_limit_violations": int(over.sum()),
        "unavoidable_overload_frames": final["frames"][overload].tolist(),
        "unknown_frames": unknown, "unknown_frame_count": len(unknown),
        "audit_error_frames": audit_errors,
        "infeasible_frames": [row["frame"] for row in final["rows"] if row["status"].startswith("infeasible")],
        "infeasible_frame_attribution": [
            {"frame": row["frame"], "reason": row["reason"],
             "reference_status": reference["rows"][index]["status"],
             "source_derivatives_at_final_pose_status": inherited["rows"][index]["status"],
             "classification": "added_inertial_demand" if inherited["rows"][index]["status"] == "feasible"
                 else "added_pose_or_contact_geometry" if reference["rows"][index]["status"] == "feasible"
                 else "inherited_source_fit_or_geometry" if reference["rows"][index]["status"] == "infeasible_unsupported"
                 else "not_separable_audit_error" if any(evaluation["rows"][index]["status"].startswith("audit_error")
                                                        for evaluation in (reference, inherited))
                 else "not_separable_missing_evidence"}
            for index, row in enumerate(final["rows"]) if row["status"] == "infeasible_unsupported"
        ],
        "unaudited_endpoint_frames": endpoint_frames,
        "missing_effort_limits": [name for name in names if name not in effort_limits],
        "frame_status_counts": counts, "frame_diagnostics": final["rows"],
        "tolerance_15mm_status_counts": {status: sum(row.get("tolerance_15mm", {}).get("status") == status for row in final["rows"])
                                         for status in sorted({r.get("tolerance_15mm", {}).get("status", "unavailable") for r in final["rows"]})},
        "reference_frame_diagnostics": reference["rows"],
        "source_derivatives_at_final_pose_frame_status_counts": {status: sum(row["status"] == status for row in inherited["rows"])
                                                                for status in sorted({r["status"] for r in inherited["rows"]})},
        "source_derivatives_at_final_pose_frame_diagnostics": [
            {key: row[key] for key in ("frame", "status", "reason")} for row in inherited["rows"]],
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
        "interpretation": "Instantaneous feasibility conditional on original-triangle collision-shell coverage and friction. Unsupported frames are definite failures within that model, not missing-data unknowns. Unknown and unaudited frames do not pass. Source-fit robot overload is not proof of human-source overload.",
    }
