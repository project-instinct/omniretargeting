"""Add independently checked circular-base certificates to numerical audit rows.

This preserves existing feasible/overload witnesses and all input motions.
Run under the same shared heavy lock as the original audit.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from omniretargeting.effort_audit import evaluate_effort
from omniretargeting.effort_certificate import certify_circular_unsupported
from omniretargeting.effort_contacts import contact_evidence, frame_contacts, load_fixed_objects
from omniretargeting.effort_dynamics import FRICTION, balance_residual, derivatives, mechanical_limits, motor_demand
from omniretargeting.effort_force_columns import reduced_contact_matrix
from omniretargeting.effort_geometry import SENSITIVITY_GAP_M, load_geometry
from scripts.review_effort_baseline import CLIPS, ROOT

NUMERICAL = {"unknown_numerical", "unknown_cone_boundary", "audit_error_numerical", "audit_error_cone_certificate"}


def refine(baseline, output, clips=CLIPS):
    changes = []
    for clip in clips:
        path = output / clip / "joint_smoothness.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        dynamics = report["effort_dynamics"]
        final_rows = {r["frame"]: r for r in dynamics["frame_diagnostics"]}
        reference_rows = {r["frame"]: r for r in dynamics["reference_frame_diagnostics"]}
        attributions = {r["frame"]: r for r in dynamics["infeasible_frame_attribution"]}
        counter_rows = {r["frame"]: r for r in dynamics.get("source_derivatives_at_final_pose_frame_diagnostics", [])}
        counter_status_by_frame = {frame: row["status"] for frame, row in counter_rows.items()} or {
            frame: row["source_derivatives_at_final_pose_status"] for frame, row in attributions.items()}
        targets = [(role, frame) for role, rows in (("final", final_rows), ("reference", reference_rows))
                   for frame, row in rows.items() if row["status"] in NUMERICAL
                   or row["reason"] == "circular_cone_or_support_wrench"]
        targets += [("inherited", frame) for frame, status in counter_status_by_frame.items() if status in NUMERICAL]
        if not targets:
            continue
        started = time.perf_counter()
        source = baseline / "A/normal/repeat_1" / clip
        saved = np.load(source / "final_retargeted.npz")
        after, before = saved["qpos"], np.load(source / "fixed_reference.npz")["qpos"]
        fps = report["fps"]
        model_path = ROOT / "robot_models/unitree_g1/g1_29dof_torsobase_mesh.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        _, caps, provenance = mechanical_limits(model_path, model)
        for key in ("dynamics_model_sha256", "collision_model_sha256", "limits_urdf_sha256"):
            assert provenance[key] == report["mechanical_model"][key], key
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        joints = np.array([i for i in range(model.njnt) if model.joint(i).name in caps])
        limits = np.array([caps[model.joint(i).name] for i in joints])
        base = np.setdiff1d(np.arange(model.nv), model.jnt_dofadr[joints])
        evidence = contact_evidence(saved, model, load_fixed_objects(
            source / "prepared.npy", source / "objects.npy", len(after), fps))
        evidence["fixed_objects_15mm"] = load_fixed_objects(
            source / "prepared.npy", source / "objects.npy", len(after), fps, distance=SENSITIVITY_GAP_M)
        evidence["geometry"] = load_geometry(source / "prepared.npy", source / "objects.npy", model, len(after), evidence)
        kinematics = {role: derivatives(model, poses, fps) for role, poses in (("final", after), ("reference", before))}
        changed_frames = set()
        data = mujoco.MjData(model)
        for role, frame in targets:
            frames, velocity, acceleration = kinematics["reference" if role != "final" else "final"]
            index = int(np.flatnonzero(frames == frame)[0])
            poses = before if role == "reference" else after
            demand = motor_demand(model, data, poses[frame], velocity[index], acceleration[index])
            contacts, _, geometry = frame_contacts(evidence, model, data, frame)
            assert not geometry["unmodelled_object_contacts"] and not geometry["uncertain_near_surface_points"]
            certificate = certify_circular_unsupported(model, data, demand, contacts, base)
            if not certificate:
                print("uncertified", clip, role, frame, flush=True)
                status = "audit_error_numerical"
                if role == "inherited":
                    old = counter_status_by_frame[frame]
                    counts = dynamics["source_derivatives_at_final_pose_frame_status_counts"]
                    counts[old] -= 1; counts[status] = counts.get(status, 0) + 1
                    if frame in counter_rows:
                        counter_rows[frame].update(status=status, reason="contact_force_solver_did_not_certify")
                else:
                    row = (final_rows if role == "final" else reference_rows)[frame]
                    row.update(status=status, reason="contact_force_solver_did_not_certify",
                               minimum_peak_utilization=None)
                dynamics.setdefault("uncertified_numerical_refinements", []).append({"role": role, "frame": frame})
                changed_frames.add(frame)
                continue
            matrix, _, _ = reduced_contact_matrix(model, data, contacts, FRICTION, outer=True)
            diagnostic = balance_residual(demand, matrix, base)
            status = "infeasible_unsupported"
            if role == "inherited":
                old = counter_status_by_frame[frame]
                counts = dynamics["source_derivatives_at_final_pose_frame_status_counts"]
                counts[old] -= 1; counts[status] = counts.get(status, 0) + 1
                if frame in counter_rows:
                    counter_rows[frame].update(status=status, reason="verified_circular_cone_separating_wrench")
            else:
                row = (final_rows if role == "final" else reference_rows)[frame]
                assert row["attained_candidates"] == len(contacts)
                row.update(status=status, reason="verified_circular_cone_separating_wrench",
                           minimum_peak_utilization=None, force_witness_kind="circular_base_separation_certificate",
                           base_residual=None, **certificate, **diagnostic)
            changes.append({"clip": clip, "role": role, "frame": frame, **certificate})
            changed_frames.add(frame)
        if not changed_frames:
            continue
        inherited = evaluate_effort(model, after, fps, joints, limits, evidence,
                                    kinematics=before, frame_indices=changed_frames)
        counter = {r["frame"]: r for r in inherited["rows"]}
        for frame in changed_frames:
            if final_rows[frame]["status"] != "infeasible_unsupported":
                continue
            reference_status = reference_rows[frame]["status"]
            counter_status = counter[frame]["status"]
            attributions[frame] = {"frame": frame, "reason": final_rows[frame]["reason"],
                "reference_status": reference_status, "source_derivatives_at_final_pose_status": counter_status,
                "classification": "added_inertial_demand" if counter_status == "feasible"
                    else "added_pose_or_contact_geometry" if reference_status == "feasible"
                    else "inherited_source_fit_or_geometry" if reference_status == "infeasible_unsupported"
                    else "not_separable_audit_error" if any(s.startswith("audit_error") for s in (reference_status, counter_status))
                    else "not_separable_missing_evidence"}
        for key, rows in (("frame_status_counts", final_rows), ("reference_frame_status_counts", reference_rows)):
            dynamics[key] = dict(Counter(r["status"] for r in rows.values()))
        dynamics["unknown_frames"] = sorted(f for f, row in final_rows.items() if row["status"].startswith("unknown"))
        dynamics["unknown_frame_count"] = len(dynamics["unknown_frames"])
        dynamics["audit_error_frames"] = sorted(f for f, row in final_rows.items() if row["status"].startswith("audit_error"))
        dynamics["infeasible_frames"] = sorted(f for f, row in final_rows.items() if row["status"] == "infeasible_unsupported")
        dynamics["infeasible_frame_attribution"] = [attributions[f] for f in sorted(attributions)
                                                  if final_rows[f]["status"] == "infeasible_unsupported"]
        if counter_rows:
            dynamics["source_derivatives_at_final_pose_frame_diagnostics"] = [counter_rows[f] for f in sorted(counter_rows)]
        dynamics.setdefault("numerical_certificate_refinements", []).extend(x for x in changes if x["clip"] == clip)
        dynamics["numerical_certificate_refinement_seconds"] = time.perf_counter() - started
        if dynamics["unknown_frames"] or dynamics["infeasible_frames"] or dynamics["audit_error_frames"]:
            report["all_individual_budgets_pass"] = False
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("refined", clip, sorted(changed_frames), flush=True)
    return changes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", choices=CLIPS, default=CLIPS)
    args = parser.parse_args()
    refine(args.baseline, args.output, args.clips)
