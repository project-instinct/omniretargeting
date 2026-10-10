"""Explain actual nonfixed-object contacts in frozen effort audits.

Run under the shared heavy lock; distances use original object vertices and
static triangles, with the same stillness/grounding rules as the effort audit.
"""

import argparse
import json
from pathlib import Path

import igl
import numpy as np
import trimesh

from omniretargeting.effort_contacts import load_fixed_objects
from omniretargeting.source_support_evidence import _object_still, _object_vertices


def inspect_objects(baseline, recomputed, output, *, reuse=False):
    result = json.loads(output.read_text()) if reuse and output.exists() else {}
    for path in sorted(recomputed.glob("*/joint_smoothness.json")):
        clip = path.parent.name
        if clip in result:
            continue
        audit = json.loads(path.read_text())
        unknown = [r for r in audit["effort_dynamics"]["frame_diagnostics"]
                   if r.get("unmodelled_object_contacts", 0)]
        if not unknown:
            result[clip] = []
            continue
        source = baseline / "A/normal/repeat_1" / clip
        frames, fps = audit["frames"], audit["fps"]
        prepared = np.load(source / "prepared.npy", allow_pickle=True).item()
        objects = np.load(source / "objects.npy", allow_pickle=True).tolist()
        speed = prepared["surface_config"]["sticking_speed"]
        fixed = load_fixed_objects(source / "prepared.npy", source / "objects.npy", frames, fps)
        fixed15 = load_fixed_objects(source / "prepared.npy", source / "objects.npy",
                                     frames, fps, distance=.015)
        scene = trimesh.load_mesh(source / "static_scene.obj", process=False)
        vertices = np.asfortranarray(scene.vertices, dtype=float)
        faces = np.asfortranarray(scene.faces, dtype=np.int64)
        tree = igl.AABB()
        tree.init(vertices, faces)
        rows = []
        for sid in sorted({sid for r in unknown for sid in r["unmodelled_object_surfaces"]}):
            spec = objects[sid - 1]
            selected = [r["frame"] for r in unknown if sid in r["unmodelled_object_surfaces"]]
            still = _object_still(spec, frames, speed, fps)
            sample = selected[len(selected) // 2]
            world = _object_vertices(spec, frames, slice(sample - 1, sample + 2))
            velocity = np.linalg.norm((world[2] - world[0]) * (fps / 2), axis=1)
            distance2 = tree.squared_distance(vertices, faces, np.asfortranarray(world[1]))[0]
            rows.append({
                "surface": sid, "name": spec.get("name", "object"),
                "unknown_contact_frames": selected,
                "primary_unknown_object_load_frames": sum(r["status"] == "unknown_object_load"
                                                          for r in unknown if sid in r["unmodelled_object_surfaces"]),
                "additional_geometry_unknown_frames": sum(r["status"] == "unknown_contact_geometry"
                                                          for r in unknown if sid in r["unmodelled_object_surfaces"]),
                "moving_frames": int(sum(not still[f] for f in selected)),
                "still_but_not_proved_fixed_frames": int(sum(still[f] and not fixed[sid-1, f] for f in selected)),
                "15mm_proved_fixed_frames": int(sum(fixed15[sid-1, f] for f in selected)),
                "representative_frame": sample, "maximum_vertex_speed_m_s": float(velocity.max()),
                "sticking_speed_m_s": float(speed),
                "nearest_vertex_to_static_triangles_mm": float(np.sqrt(distance2.min()) * 1000),
            })
        result[clip] = rows
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--recomputed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect_objects(args.baseline, args.recomputed, args.output)
