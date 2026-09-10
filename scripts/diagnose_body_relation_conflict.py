"""Small local gradient audit on the SAME scalar arm DOFs and whitened rows.

Reports local competition only, never a reachability or global infeasibility claim.
Uses the solver's existing residual builders, not copies of their objectives.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cvxpy as cp
import mujoco
import numpy as np
import trimesh

from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.trajectory_validation import TrajectoryValidator


def diagnose(base, clip, frame):
    path = base / clip
    profile = json.loads((base / "robot_profile.json").read_text())
    case = json.loads((base / "cases.json").read_text())[clip]
    model = mujoco.MjModel.from_xml_path(profile["urdf_path"])
    scene = trimesh.load(case["scene"], force="mesh")
    context = pickle.load((path / "A/validation_context.pkl").open("rb"))
    reference = pickle.load((path / "A/calibration.pkl").open("rb"))
    relations = pickle.load((path / "C/body_relations.pkl").open("rb"))
    source = np.load(path / "A/source_arrays.npz")
    names = source["names"].tolist()
    mapping = profile["joint_mapping"]
    cfg = context["priority"]
    cfg["body_orientation"]["_source_orientations_available"] = True
    solver = GenericInteractionRetargeter(model, mujoco.MjData(model), scene, mapping, 1.3,
                                         terrain_sample_seed=0, end_effector_config=cfg)
    solver.body_orientation_tasks = reference["tasks"]
    solver._source_orientation_indices = {n: names.index(n) for n in reference["source"]}
    solver._source_orientation_references = reference["source"]
    solver._robot_orientation_references = reference["robot"]
    q_a = np.load(path / "A/final_retargeted.npz")["qpos"]
    v = TrajectoryValidator(model, q_a, SceneQuery(scene), context["tasks"], cfg, context["config"], context["framerate"])
    output = dict(frame=frame, scope="local scalar left-arm DOFs; whitened solver rows; interaction mesh omitted",
                  establishes_reachability=False, variants={})
    joints = cfg["effectors"]["left_hand"]["contact_joint_names"]
    qi = np.array([model.joint(n).qposadr[0] for n in joints])
    di = np.array([model.joint(n).dofadr[0] for n in joints])
    cols = [np.flatnonzero(solver.q_a_indices == i)[0] for i in qi]
    for variant in ("A", "C"):
        q = np.load(path / variant / "final_retargeted.npz")["qpos"][frame]
        solver.robot_data.qpos[:] = q
        mujoco.mj_forward(model, solver.robot_data)
        variable = cp.Variable(solver.nq_a)
        variable.value = np.zeros(solver.nq_a)
        jac, positions, _ = solver._compute_robot_jacobians(q)
        bone = solver._build_pose_direction_residual(variable, positions, jac,
                                                    source["positions"][frame, [names.index(n) for n in solver.valid_joint_names]])
        body = solver._build_body_orientation_residual(variable, source["source_joint_orientations"][frame],
                                                       source["base_orientations"][frame])
        solver._active_contact_offsets = {n: int(offsets[frame]) for n, offsets in v.offset_choice.items()}
        payloads = {n: dict(t.frame(frame), source_root_position=source["positions"][frame, names.index("Spine3")])
                    for n, t in context["tasks"].items()}
        contact = solver._build_end_effector_objectives(variable, payloads, {})[0]
        gradients, groups = {}, {}
        for name, expr, weight in (("bone_direction", bone, cfg.get("pose_direction_weight", 10.)),
                                   ("parent_relative_orientation", body, cfg.get("body_orientation_weight", 1.)),
                                   ("contact_primary", contact, 1.)):
            if expr is None:
                continue
            residual = np.asarray(expr.value).reshape(-1)
            jacobian = expr.grad[variable].toarray().T[:, cols]
            gradients[name] = weight * jacobian.T @ residual
            groups[name] = dict(cost=float(weight * residual @ residual), gradient_norm=float(np.linalg.norm(gradients[name])))
        rows = list(relations.rows(q, frame, di))
        g = sum(weight * jacobian.T @ residual for weight, residual, jacobian in rows)
        groups["body_relations"] = dict(cost=sum(float(w * r @ r) for w, r, _ in rows),
                                        gradient_norm=float(np.linalg.norm(g)),
                                        per_joint_gradient=dict(zip(joints, g.tolist())))
        cosine = {}
        for name, other in gradients.items():
            denominator = np.linalg.norm(g) * np.linalg.norm(other)
            cosine[name] = float(g @ other / denominator) if denominator > 1e-12 else None
        output["variants"][variant] = dict(groups=groups, relation_gradient_cosine=cosine,
            active_contact={n: t.frame(frame)["contact_mode"] for n, t in context["tasks"].items() if t.contact_mask[frame]},
            note="Contact gradient uses common A validation marker, which may differ from framewise first-contact selection")
    destination = path / f"gradient_conflict_frame_{frame}.json"
    destination.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--frame", required=True, type=int)
    args = parser.parse_args()
    diagnose(args.output, args.clip, args.frame)
