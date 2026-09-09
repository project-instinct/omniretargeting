"""Reproducible synthetic A/B; never presented as captured human motion.

Run: python -m tests.local_collision_benchmark --output out/local_collision_ab
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import trimesh

from omniretargeting.local_collision_repair import (
    diagnostics_json,
    repair_local_collisions,
)
from omniretargeting.robot_config import load_robot_config
from omniretargeting.scene_geometry import SceneQuery, collision_geom_ids
from omniretargeting.trajectory_validation import (
    TrajectoryValidator,
    local_repair_config,
)
from tests.test_contact_priority import _task
from tests.test_local_collision_repair import setup_motion


def run(output: Path, robot_config: str, ablations=False):
    output.mkdir(parents=True, exist_ok=True)
    results = {}

    def record_variant(
        name, model, q, scene, tasks, priority, config, fps, surface_tasks=None
    ):
        started = perf_counter()
        result, report = repair_local_collisions(
            model,
            q,
            scene,
            tasks,
            priority,
            config,
            framerate=fps,
            surface_tasks=surface_tasks,
        )
        if not config.get("enabled", True):
            # Baseline output is measured by the SAME full-geometry validator;
            # disabling repair never hides detection improvements in A/B.
            v = TrajectoryValidator(
                model,
                q,
                SceneQuery.from_tasks(scene, tasks),
                tasks,
                priority,
                local_repair_config(config, priority=priority),
                fps,
            )
            before = v.geometry(q)
            report.update(
                before=before,
                after=before,
                fidelity=v.fidelity(q),
                reference_fidelity=v.fidelity(q),
                unresolved=bool(before["records"] or before["uncertain_frames"]),
                query_points=v.query.query_points,
                geometry_calls=v.geometry_calls,
                seconds=perf_counter() - started,
            )
        report["data_kind"] = "synthetic motion and box; not captured human motion"
        results[name] = report
        (output / f"{name}.json").write_text(diagnostics_json(report), encoding="utf-8")
        np.savez(output / f"{name}.npz", before=q, qpos=result, framerate=fps)
        scene.export(output / f"{name}.obj")

    def record(name, model, q, scene, tasks, priority, config, fps):
        if not ablations:
            return record_variant(name, model, q, scene, tasks, priority, config, fps)
        variants = {
            "A": {
                "enabled": False,
                "full_geometry": False,
                "bidirectional_enabled": False,
            },
            "B": {"full_geometry": True, "bidirectional_enabled": False},
            "C": {"full_geometry": True, "bidirectional_enabled": True},
            "D": {"full_geometry": True, "bidirectional_enabled": True},
        }
        for variant, flags in variants.items():
            surface = None
            if variant == "D":
                # Geometric truth for synthetic normals, explicitly not a human
                # correspondence. The source triangles follow the input robot
                # FK and exercise transport/weights without inventing SMPL-X data.
                from omniretargeting.surface_tasks import SurfaceBinding, SurfaceTasks

                gid = int(collision_geom_ids(model)[0])
                binding = SurfaceBinding(
                    "synthetic_limb",
                    0,
                    (1 / 3, 1 / 3, 1 / 3),
                    model.geom(gid).name or gid,
                    (0, 0, 0),
                    (0, 0, 1),
                )
                local = np.array([[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0]])
                data = mujoco.MjData(model)
                vertices = []
                for pose in q:
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    vertices.append(
                        local @ data.geom_xmat[gid].reshape(3, 3).T
                        + data.geom_xpos[gid]
                    )
                surface = SurfaceTasks(
                    model,
                    [binding],
                    vertices[0],
                    [[0, 1, 2]],
                    vertices,
                    q[0],
                    corresponding_reference_poses=True,
                )
            record_variant(
                f"{name}_{variant}",
                model,
                q,
                scene,
                tasks,
                priority,
                {**config, **flags},
                fps,
                surface,
            )

    model, q, scene = setup_motion()
    q[3:6, 8] = 0.04
    support = _task(np.tile([0, 0, 1.0], (len(q), 1)), np.ones(len(q), bool))
    support_spec = {
        "effectors": {
            "right_foot": {"robot_body": "other", "tracking_offset": [0, 0, 1]}
        }
    }
    record(
        "linear_support", model, q, scene, {"right_foot": support}, support_spec, {}, 30
    )

    model, q, scene = setup_motion(5)
    q[:, 7:9] = [[-0.12, 0], [0, 0], [0, 0], [0, 0], [0, -0.12]]
    record("corner_optimization", model, q, scene, {}, {}, {}, 15)

    if ablations:
        from tests.test_local_candidates import directional_case

        model, q, scene = directional_case()
        record("directional_corner", model, q, scene, {}, {}, {}, 15)

    model, q, scene = setup_motion()
    q[3:6, 8] = 0.04
    target = np.column_stack([q[:, 7], q[:, 8], np.zeros(len(q))])
    swing = _task(target)
    swing_spec = {
        "effectors": {
            "left_foot": {"robot_body": "foot", "preserve_world_target": True}
        }
    }
    record("bounded_swing", model, q, scene, {"left_foot": swing}, swing_spec, {}, 30)
    sliding = _task(target, np.ones(len(q), bool), ["sliding"] * len(q))
    record("sliding", model, q, scene, {"left_foot": sliding}, swing_spec, {}, 30)

    model, q, scene = setup_motion(2)
    q[:, 8] = 0
    record("interframe_fallback", model, q, scene, {}, {}, {}, 30)

    # Same full collision model used by the contact-aware G1 profile. The box
    # is constructed from its compiled shin geometry; no model scale is changed.
    profile = load_robot_config(robot_config)
    model = mujoco.MjModel.from_xml_path(profile["urdf_path"])
    data = mujoco.MjData(model)
    q = np.tile(model.qpos0, (9, 1))
    q[:, 2] = 0.85
    coordinate = int(model.joint("left_knee_joint").qposadr[0])
    q[:, coordinate] = 0.3
    q[3:6, coordinate] += 0.15
    shin = model.body("left_knee_link").id
    gid = next(g for g in collision_geom_ids(model) if model.geom_bodyid[g] == shin)
    data.qpos[:] = q[4]
    mujoco.mj_forward(model, data)
    radius, half_length = model.geom_size[gid, :2]
    point = data.geom_xpos[gid] + data.geom_xmat[gid].reshape(3, 3) @ [
        -radius,
        0,
        -half_length,
    ]
    scene = trimesh.creation.box(extents=[0.3] * 3)
    scene.apply_translation(point + np.array([-1, 1, 1]) * (0.15 - 0.008))
    tasks, priority = {}, {"effectors": {}}
    for name in ["left_foot", "right_foot"]:
        body = f"{name.split('_')[0]}_ankle_roll_link"
        targets = []
        for frame in q:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            targets.append(data.xpos[model.body(body).id].copy())
        tasks[name] = _task(np.asarray(targets), np.full(len(q), name == "right_foot"))
        priority["effectors"][name] = {
            "robot_body": body,
            "preserve_world_target": True,
        }
    # This diagnostic uses explicit ankle-origin markers and synthetic contact
    # schedules. Log them separately from the loaded profile's merged defaults.
    record("g1_shin", model, q, scene, tasks, priority, {}, 30)
    (output / "effective_configs.json").write_text(
        diagnostics_json(
            {
                "loaded_robot_profile": profile,
                "benchmark_end_effectors": priority,
                "benchmark_local_repair": local_repair_config({}),
                "robot_scale": 1.0,
                "scene_scale": 1.0,
            }
        ),
        encoding="utf-8",
    )
    summary = {}
    for name, report in results.items():
        fields = [
            "max_depth",
            "p95_depth",
            "frames_penetrating",
            "worst_body",
            "worst_frame",
            "interframe_max_depth",
        ]
        summary[name] = {
            "before": {key: report["before"][key] for key in fields},
            "after": {key: report["after"][key] for key in fields},
            "fidelity": report["fidelity"],
            "reference_fidelity": report["reference_fidelity"],
            "invariants": report.get("invariants", {}),
            "unresolved": report["unresolved"],
            "seconds": report["seconds"],
            "query_points": report["query_points"],
            "events": [
                {
                    key: e.get(key)
                    for key in [
                        "window",
                        "method",
                        "reason",
                        "iterations",
                        "horizontal_point_correction",
                        "vertical_point_correction",
                        "status",
                        "bidirectional",
                        "optimization",
                    ]
                }
                for e in report["events"]
            ],
        }
    (output / "summary.json").write_text(diagnostics_json(summary), encoding="utf-8")
    print(diagnostics_json(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("out/local_collision_ab"))
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Compare A/B/C/D using identical validation",
    )
    parser.add_argument(
        "--robot-config", default="robot_models/unitree_g1/unitree_g1_instinctmj.json"
    )
    args = parser.parse_args()
    run(args.output, args.robot_config, args.ablations)
