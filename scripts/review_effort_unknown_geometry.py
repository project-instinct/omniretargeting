"""Inspect representative missing free-side evidence without changing audits.

Run under the shared heavy lock. All queries retain original triangle geometry.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from omniretargeting.effort_contacts import contact_evidence
from omniretargeting.effort_geometry import load_geometry
from scripts.review_effort_baseline import ROOT


def inspect_geometry(baseline, recomputed, output, *, reuse=False):
    result = json.loads(output.read_text()) if reuse and output.exists() else {}
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robot_models/unitree_g1/g1_29dof_torsobase_mesh.xml"))
    for path in sorted(recomputed.glob("*/joint_smoothness.json")):
        clip = path.parent.name
        if clip in result:
            continue
        report = json.loads(path.read_text())
        unknown = [r for r in report["effort_dynamics"]["frame_diagnostics"]
                   if r["status"] == "unknown_contact_geometry"]
        if not unknown:
            result[clip] = None
            continue
        row = unknown[len(unknown) // 2]
        source = baseline / "A/normal/repeat_1" / clip
        with np.load(source / "final_retargeted.npz") as saved:
            evidence = contact_evidence(saved, model)
            geometry = load_geometry(source / "prepared.npy", source / "objects.npy", model,
                                     report["frames"], evidence)
            data = mujoco.MjData(model)
            data.qpos[:] = saved["qpos"][row["frame"]]
            mujoco.mj_fwdPosition(model, data)
            _, pairs, _ = geometry.frame(data, row["frame"])
        result[clip] = {
            "frame": row["frame"], "unknown_frames": len(unknown),
            "actual_nonfixed_object_surfaces": row["unmodelled_object_surfaces"],
            "uncertain_near_surface_points": row["uncertain_near_surface_points"],
            "pairs": [p for p in pairs if p.get("uncertain_points_5mm", 0)],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--recomputed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect_geometry(args.baseline, args.recomputed, args.output)
