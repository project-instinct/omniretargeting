"""Summarize completed ablations without comparing incompatible solver losses.

Usage: .venv/bin/python scripts/motion_quality_report.py out/motion_quality_20260911
Incomplete cases stay explicit; only complete A/B/C/D windows are compared.
"""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omniretargeting.motion_quality import summary
from omniretargeting.robot_config import load_robot_config
from omniretargeting.utils import (
    calculate_laplacian_coordinates,
    calculate_source_laplacian,
    get_adjacency_list,
    sample_points_on_mesh,
)


def fixed_config(value):
    """Remove only the two experimental controls; everything else must match."""
    value = deepcopy(value)
    if isinstance(value, dict):
        value.pop("pose_initialization", None)
        if "interaction_mesh" in value:
            value["interaction_mesh"].pop("weighting", None)
            value["interaction_mesh"].pop("kappa", None)
        return {k: fixed_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fixed_config(v) for v in value]
    return value


def mean_present(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def maximum_present(values):
    values = [x for x in values if x is not None]
    return float(max(values)) if values else None


def compact_metrics(measurements):
    direction = [
        v for k, v in measurements["direction"].items() if k.startswith(("L_", "R_"))
    ]
    effectors = measurements["end_effectors"]
    return {
        "limb_direction_mean_deg": mean_present(
            [v["world_angle_deg"]["mean"] for v in direction]
        ),
        "hand_world_mean_m": mean_present(
            [
                v["source_world_marker_error_m"]["mean"]
                for k, v in effectors.items()
                if "hand" in k
            ]
        ),
        "foot_world_mean_m": mean_present(
            [
                v["source_world_marker_error_m"]["mean"]
                for k, v in effectors.items()
                if "foot" in k
            ]
        ),
        "contact_max_m": maximum_present(
            [v["contact_position_error_m"]["max"] for v in effectors.values()]
        ),
        "support_slip_max_m_per_s": maximum_present(
            [
                v["support_slip_m_per_s"]["max"]
                for k, v in effectors.items()
                if "foot" in k
            ]
        ),
        "bend_plane_jump_max_deg": maximum_present(
            [
                v["bend_plane_step_deg"]["max"]
                for v in measurements["limb_configuration"].values()
            ]
        ),
        "direction_step_error": mean_present(
            [v["direction_step_error"]["mean"] for v in direction]
        ),
        "amplitude_ratio_mean": mean_present([v["amplitude_ratio"] for v in direction]),
        "lag_frames_mean": mean_present([v["lag_frames"] for v in direction]),
        "joint_jerk_p95_rad_per_s3": measurements["joint_jerk_rad_per_s3"]["p95"],
        "mechanical_limit_frames": measurements["frames_outside_mechanical_limits"],
        "max_scene_depth_m": measurements["scene_penetration"]["max_depth"],
        "scene_worst_body": measurements["scene_penetration"]["worst_body"],
        "scene_worst_frame": measurements["scene_penetration"]["worst_frame"],
    }


def source_graph_jumps(npz_path, manifest, config):
    archive = np.load(npz_path)
    profile = load_robot_config(manifest["robot_profile"])
    from omniretargeting.data_sources.smplx import DEFAULT_SMPLX_TARGET_NAMES

    indices = [DEFAULT_SMPLX_TARGET_NAMES.index(n) for n in profile["joint_mapping"]]
    floor = trimesh.load(npz_path.parent / "flat_ground.ply", force="mesh")
    samples = sample_points_on_mesh(
        floor, config["terrain_sample_points"], seed=config["terrain_sample_seed"]
    )
    targets = {"uniform": [], "exponential": []}
    adjacency_previous = None
    rewiring = {"uniform": [], "exponential": []}
    for source in archive["source"][:, indices]:
        vertices = np.vstack([source, samples])
        adjacency = get_adjacency_list(Delaunay(vertices).simplices, len(vertices))
        for weighting, trajectory_targets in targets.items():

            def apply(adj, weighting=weighting, vertices=vertices):
                if weighting == "uniform":
                    return calculate_laplacian_coordinates(vertices, adj)
                return (
                    calculate_source_laplacian(
                        vertices, adj, kappa=manifest["kappa_inverse_m"]
                    )
                    @ vertices
                )

            current = apply(adjacency)
            trajectory_targets.append(current)
            if adjacency_previous is not None:
                # Hold current source positions fixed; isolate adjacency change.
                rewiring[weighting].append(
                    np.linalg.norm(current - apply(adjacency_previous), axis=1)
                )
        adjacency_previous = adjacency
    return {
        key: {
            "target_step_m": summary(np.linalg.norm(np.diff(value, axis=0), axis=2)),
            "graph_change_only_m": summary(rewiring[key]),
        }
        for key, value in targets.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifests = {
        v: json.loads((args.root / v / "manifest.json").read_text()) for v in "ABCD"
    }
    reference = manifests["A"]
    for variant, manifest in manifests.items():
        for key in (
            "source_file_sha256",
            "source_positions_sha256",
            "fps",
            "frames",
            "stride",
            "windows",
            "package_sha256",
        ):
            if manifest[key] != reference[key]:
                raise ValueError(f"{variant}: unmatched experimental input {key}")
    results, incomplete = {}, []
    lines = [
        "# Motion-quality A/B/C/D measurements",
        "",
        (
            "Same frozen code, source, model, seeds, original constraints and postprocessing. "
            "Window labels are source-feature candidates, not verified standing gesture classes. "
            "Hand world error is a diagnostic, not the free-hand optimization target. "
            "Missing contact evidence is n/a. Short-window lag/amplitude estimates are exploratory."
        ),
        "",
    ]
    for label in reference["windows"]:
        paths = {v: args.root / v / f"{label}_{v}.json" for v in "ABCD"}
        if not all(path.exists() for path in paths.values()):
            incomplete.append({label: [v for v, p in paths.items() if not p.exists()]})
            continue
        reports = {v: json.loads(path.read_text()) for v, path in paths.items()}
        cfg = fixed_config(reports["A"]["effective_config"])
        if any(fixed_config(r["effective_config"]) != cfg for r in reports.values()):
            raise ValueError(f"{label}: non-experimental configuration differs")
        results[label] = {}
        for stage in ("framewise", "postprocessed"):
            lines += [
                f"## {label}: {stage}",
                "",
                "| Variant | Limb direction ° | Hand world cm | Foot world cm | Contact max cm | Slip max cm/s | Joint jerk p95 rad/s³ | Scene max cm |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for variant in "ABCD":
                compact = compact_metrics(reports[variant]["measurements"][stage])
                results[label].setdefault(variant, {})[stage] = compact
                values = [
                    compact["limb_direction_mean_deg"],
                    compact["hand_world_mean_m"],
                    compact["foot_world_mean_m"],
                    compact["contact_max_m"],
                    compact["support_slip_max_m_per_s"],
                    compact["joint_jerk_p95_rad_per_s3"],
                    compact["max_scene_depth_m"],
                ]
                values = [
                    v * s if v is not None else None
                    for v, s in zip(values, [1, 100, 100, 100, 100, 1, 100])
                ]
                lines.append(
                    "| "
                    + variant
                    + " | "
                    + " | ".join("n/a" if v is None else f"{v:.3f}" for v in values)
                    + " |"
                )
            lines.append("")
        lines += [
            "| Variant | Init s | Full solve s | Total pipeline s | Solve calls with failed subproblem | Init fallbacks |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for variant, report in reports.items():
            t = report["timing"]
            results[label][variant]["timing"] = t
            lines.append(
                f"| {variant} | {t['initialization_seconds']:.3f} | {t['full_solve_seconds']:.3f} | {t['retarget_motion_seconds']:.3f} | {t['frames_with_failed_subproblem']}/{t['solved_frames_including_overlay']} | {sum(t['fallback_reasons'].values())} |"
            )
        lines.append("")
        results[label]["source_graph_jumps"] = source_graph_jumps(
            paths["A"].with_suffix(".npz"), reference, reports["A"]["effective_config"]
        )
    lines += ["Incomplete cases: " + json.dumps(incomplete), ""]
    output = {
        "completed": results,
        "incomplete": incomplete,
        "evaluator_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[1]
                / "omniretargeting/motion_quality.py"
            ).read_bytes()
        ).hexdigest(),
    }
    (args.root / "comparison.json").write_text(
        json.dumps(output, indent=2, allow_nan=False)
    )
    (args.root / "comparison.md").write_text("\n".join(lines))
    print(
        f"Saved {args.root / 'comparison.md'}; {len(results)} complete windows, {len(incomplete)} incomplete"
    )


if __name__ == "__main__":
    main()
