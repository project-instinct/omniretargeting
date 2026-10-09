import json

from scripts.compare_benchmark_runs import main


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_arm(root, *, deep):
    clip = root / "clip"
    _write(clip / "timing.json", {
        "fast_backend_seconds": 10.0,
        "stage_seconds": {"source_pose": 2.0, "new_stage": 1.0},
    })
    _write(clip / "receipt.json", {"production_process_wall_seconds": 20.0})
    thresholds = {
        "0.002": {"longest_run_frames": 4, "frame_fraction": 0.2},
        "0.005": {"longest_run_frames": 2, "frame_fraction": 0.1},
    }
    _write(clip / "penetration_persistence.json", {
        "inputs": {"frames": 20},
        "verdict": {"policy_v1_defect_pairs": 1},
        "summary": {
            "policy_v1_pair_counts": {
                "deep": 1, "persistent": 0, "support_residual": 1,
                "mild_brief": 0, "negligible": 0,
            },
            "max_depth_by_kind_m": {
                "static": 0.010 if deep else 0.004,
                "moving": 0.0,
                "self": 0.0,
            },
            "union_frames": {
                "over_5mm": [1, 2, 3] if deep else [1],
                "over_1cm": [2] if deep else [],
            },
            "unknown": {"total": 0},
        },
        "reset": {"usable_frame_count": 19, "unusable_frame_count": 1},
        "pairs": [{
            "pair_id": "static:1", "kind": "static",
            "robot_geom": {"body": "left_foot"},
            "other": "static_scene",
            "max_depth_m": 0.010 if deep else 0.004,
            "worst_frame": 2, "thresholds_m": thresholds,
            "policy_v1": ["deep"] if deep else ["support_residual"],
        }],
    })
    _write(clip / "contact_metrics.json", {
        "frames": 20,
        "surface_normal_error_m": {
            "left_foot": {"p95": 0.002 if deep else 0.001, "max": 0.004 if deep else 0.002},
        },
        "surface_sticking_target_relative_slip_m_s": {
            "left_foot": {"p95": 0.2, "max": 0.4},
        },
    })
    _write(clip / "joint_smoothness.json", {
        "frames": 20,
        "velocity_limit_violations": 1,
        "effort_limit_violations": 2,
        "root_source_relative_chatter": {
            "translation": {"violations": 0}, "rotation": {"violations": 0},
        },
        "joints": {
            "left_knee_joint": {
                "source_relative_chatter": {
                    "violations": 3, "max_excess": 0.01, "worst_frame": 4,
                },
                "mechanical_limits": {
                    "velocity": {"peak_per_second": 2.0, "violations": 1, "worst_frame": 3},
                    "effort": {"peak": 8.0, "violations": 2, "worst_frame": 5},
                },
            }
        },
    })
    _write(clip / "source_pose.json", {
        "mean_rad": 0.1, "p95_rad": 0.2,
        "parts": {"L_Hip -> L_Knee": {"max_rad": 0.3, "worst_frame": 7}},
    })
    _write(clip / "fast_report.json", {
        "frames": 20,
        "stages": {"contact_scene": {
            "budgets": {
                "relation_part_angle_max_rad": 0.2,
                "layout_flip_frames": 0,
            },
            "skipped_tasks": ["physics"],
            "deadline": {
                "expired": deep, "budget_s": 20.0,
                "outer_iterations": 2, "inner_iterations": 8,
            },
        }},
    })
    return clip


def test_compare_reports_missing_optional_files_and_worse_metrics(tmp_path):
    before_root = tmp_path / "before"
    after_root = tmp_path / "after"
    _make_arm(before_root, deep=False)
    _make_arm(after_root, deep=True)
    # Required-by-contract persistence exists in this fixture; other audit JSON
    # files are intentionally absent to verify missing-file reporting.
    output = tmp_path / "report.md"
    payload = tmp_path / "report.json"
    main([
        "--before", str(before_root), "--after", str(after_root),
        "--output", str(output), "--json", str(payload),
    ])
    report = output.read_text(encoding="utf-8")
    assert "缺失文件 before：scene_metrics.json, training_collision.json" in report
    assert "缺失文件 after：scene_metrics.json, training_collision.json" in report
    assert "after 全部 defect pair" in report
    assert "left_foot→static_scene" in report
    assert "before 最深 top-3 defect pair" in report
    assert "| left_foot | 0.001 | 0.002 |" in report
    assert "source-relative chatter违例" in report
    assert "deadline" in report
    assert "## 变差指标" in report
    assert "static最大穿透" in report
    saved = json.loads(payload.read_text(encoding="utf-8"))
    assert saved["headlines"]["clip"]["after"]["defect_pairs"] == 1
    assert any(item["metric"] == "static最大穿透" for item in saved["worse"])


def test_explicit_clip_and_missing_directory_are_reported(tmp_path):
    before_root = tmp_path / "before"
    _make_arm(before_root, deep=False)
    after_root = tmp_path / "after"
    output = tmp_path / "nested" / "report.md"
    main([
        "--before", str(before_root), "--after", str(after_root),
        "--clips", "clip", "--output", str(output),
    ])
    report = output.read_text(encoding="utf-8")
    assert "缺失文件 after：" in report
    assert "timing.json" in report.split("缺失文件 after：", 1)[1].splitlines()[0]
    assert "penetration_persistence.json：before=存在；after=缺失" in report
