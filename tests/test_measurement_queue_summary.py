"""Fast synthetic coverage for the measurement-queue reporter."""

import argparse
import json
from pathlib import Path

import numpy as np

import scripts.summarize_measurement_queue as reporter


def _args(run_dir: Path, output_dir: Path, *, review: bool = False):
    return argparse.Namespace(
        run_dir=run_dir, output_dir=output_dir, arms="A,B",
        a_repeats=2, e_repeats=1, run_quality_review=review, candidate_arms="B",
    )


def _timing(stage_value: float = 1.0):
    return {
        "fast_backend_seconds": 10.0, "source_load_seconds": 1.0,
        "wall_seconds": 12.0,
        "stage_seconds": {
            "source_pose": stage_value, "grid_setup": stage_value,
            "contact_acquisition": 3.0, "reweight": 2.0,
            "free_bodies": 1.0, "original_mesh_repair": 4.0,
            "final_audit": stage_value,
        },
    }


def _quality_a():
    return {
        "static": {"max_m": 0.006, "worst_frame": 7, "worst_body": "left_foot", "uncertain_samples": 1},
        "moving": {"max_m": 0.0, "worst_frame": 0, "worst_body": "torso",
                   "unknown_components": 0, "failures": []},
        "self_collision": {"max_m": 0.0, "worst_frame": 0, "worst_body": "torso", "unknown_query_count": 0},
        "reset": {"failed_frames": [3, 4]},
        "parts": {"Pelvis -> L_Hip": {"max_rad": 0.2, "worst_frame": 2},
                  "L_Hip -> L_Knee": {"max_rad": 0.4, "worst_frame": 3}},
        "calibrated_body_orientation": {"pelvis": {"max_rad": 0.1, "worst_frame": 1}},
        "surface_sticking_target_relative_slip_m_s": {
            "left_foot": {"p95": 0.02, "max": 0.03},
            "right_foot": {"p95": 0.04, "max": 0.02},
        },
        "root_xy_offset_m": {"max": 0.01}, "root_z_offset_m": {"max": 0.02},
        "joints": {"left_elbow": {
            "source_relative_chatter": {"violations": 1, "max_excess": 0.001, "worst_frame": 2},
            "mechanical_limits": {"velocity": {"violations": 0, "peak_per_second": 1.0},
                                  "effort": {"violations": 2, "peak": 30.0}},
        }, "right_elbow": {
            "source_relative_chatter": {"violations": 2, "max_excess": 0.002, "worst_frame": 5},
            "mechanical_limits": {"velocity": {"violations": 0, "peak_per_second": 1.0},
                                  "effort": {"violations": 0, "peak": 20.0}},
        }},
        "root_source_relative_chatter": {
            "translation": {"violations": 1}, "rotation": {"violations": 0},
        },
        "velocity_limit_violations": 0, "effort_limit_violations": 2,
    }


def _write_clip(root: Path, arm: str, repeat: int, *, changed=False):
    directory = root / arm / "normal" / f"repeat_{repeat}" / "sit180"
    directory.mkdir(parents=True)
    production = 40.0 if arm == "B" else 80.0 + repeat
    end = 50.0 if arm == "B" else 100.0 + repeat
    (directory / "receipt.json").write_text(json.dumps({
        "arm": arm, "mode": "normal", "repeat": repeat, "clip": "sit180",
        "solver_returncode": 0, "production_process_wall_seconds": production,
        "end_to_end_seconds": end,
        "acceptance": {
            "concurrent": arm == "B", "scene_workers": 2,
            "receipts": [
                {"name": "scene_metrics", "seconds": 4.0, "returncode": 0},
                {"name": "joint_smoothness", "seconds": 1.0, "returncode": 0},
            ],
        },
    }))
    (directory / "timing.json").write_text(json.dumps(_timing()))
    (directory / "fast_report.json").write_text(json.dumps({
        "stages": {"contact_scene": {
            "seconds": 11.0, "budgets": {"relation_whole_angle_max_rad": 0.2},
            "skipped_tasks": ["physics"], "original_mesh_repair": {"seconds": 4.0},
        }},
    }))
    quality = _quality_a()
    if arm != "A":
        quality["static"]["max_m"] = 0.012
    for name, payload in (
        ("scene_metrics.json", quality), ("training_collision.json", quality),
        ("source_pose.json", quality), ("contact_metrics.json", quality),
        ("joint_smoothness.json", quality),
    ):
        (directory / name).write_text(json.dumps(payload))
    qpos = np.zeros((2, 3))
    if changed:
        qpos[1, 0] = 0.25
    np.savez(directory / "final_retargeted.npz", qpos=qpos, metadata=np.asarray("same"))


def _patch_clip_sets(monkeypatch):
    monkeypatch.setattr(reporter, "CLIPS", ["sit180"])
    monkeypatch.setattr(reporter, "SPLINE_EXPERIMENT_CLIPS", ["sit180"])


def test_collects_missing_timing_quality_identity_and_baseline_defects(tmp_path, monkeypatch):
    _patch_clip_sets(monkeypatch)
    run = tmp_path / "measure"
    _write_clip(run, "A", 1)
    _write_clip(run, "A", 2)
    _write_clip(run, "B", 1, changed=True)
    result = reporter.collect_report(_args(run, tmp_path / "report"))
    assert result["expected_normal_runs"] == 3
    assert result["receipt_present_runs"] == 3
    aggregate = result["timing_aggregates"]["sit180/A"]
    assert aggregate["end_to_end_seconds"] == {"mean": 101.5, "min": 101.0, "max": 102.0}
    assert result["timing_aggregates"]["sit180/B"]["ratios"]["A_over_arm_end_to_end_seconds"]["mean"] == 101.0 / 50.0
    assert aggregate["audit_seconds"]["training_collision"] is None
    comparison = next(row for row in result["bitwise_final_comparisons"] if row["arm"] == "B")
    assert comparison["bitwise_final_vs_A_repeat_1"]["all_arrays_bitwise_equal"] is False
    mismatch = comparison["bitwise_final_vs_A_repeat_1"]["mismatched_arrays"]
    assert [row["array"] for row in mismatch] == ["qpos"]
    assert mismatch[0]["max_abs_diff"] == 0.25
    quality = next(row for row in result["quality_rows"] if row["arm"] == "B")
    assert quality["static_penetration"]["max_mm"] == 12.0
    assert quality["joint_chatter_max"]["part"] == "right_elbow"
    assert quality["contact_sticking_worst_p95"]["part"] == "right_foot"
    baseline = next(row for row in result["quality_rows"] if row["arm"] == "A" and row["repeat"] == 1)
    kinds = {(item["kind"], item["severity"]) for item in baseline["baseline_defects"]}
    assert ("static_penetration", "residual_over_5mm") in kinds
    assert ("reset_geometry", "failed") in kinds
    assert ("effort_limit_violations", "violated") in kinds
    text = reporter.markdown(result)
    assert "A 的下列数值是基线缺陷" in text
    assert "missing" in text


def test_quality_reviews_are_report_artifacts_and_part_deltas_are_summarized(tmp_path, monkeypatch):
    _patch_clip_sets(monkeypatch)
    run = tmp_path / "measure"
    _write_clip(run, "A", 1)
    _write_clip(run, "B", 1)

    def fake_review(**kwargs):
        output = kwargs["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "comparisons": {
                "final_vs_original_source": {"metrics": {
                    "relation_angle": {"L_Knee": {"mean": 0.2, "p95": 0.3, "max": 0.4, "worst_frame": 4}},
                }},
            },
            "non_degradation_against_A": {"metrics": {
                "relation_angle": {"L_Knee": {
                    "A": {"mean": 0.1, "p95": 0.1, "max": 0.1},
                    "final": {"mean": 0.2, "p95": 0.3, "max": 0.4, "worst_frame": 4},
                    "delta": {"mean": 0.1, "p95": 0.2, "max": 0.3},
                    "worse_than_A_intervals": [{"start_frame": 3, "end_frame": 4, "frames": 2}],
                }},
            }},
        }))
        assert all(Path(value).is_absolute() for value in kwargs.values())
        return {"returncode": 0, "stdout": "", "stderr": "", "status": "completed"}

    monkeypatch.setattr(reporter, "run_spline_quality_review", fake_review)
    output = tmp_path / "report"
    result = reporter.collect_report(_args(run, output, review=True))
    review = result["spline_quality_review_runs"][0]
    assert review["status"] == "completed"
    assert review["output"].startswith(str(output.resolve()))
    assert not (run / "B" / "normal" / "repeat_1" / "sit180" / "spline_quality.json").exists()
    summary = result["spline_quality_part_worst"][0]
    assert summary["part"] == "L_Knee"
    assert summary["worst_delta_max"] == 0.3
    assert summary["worse_than_A_interval_frames"] == 2


def test_nonregression_uses_noise_and_supports_discovered_candidate_arm(tmp_path, monkeypatch):
    _patch_clip_sets(monkeypatch)
    run = tmp_path / "measure"
    _write_clip(run, "A", 1)
    _write_clip(run, "F1", 1)
    args = argparse.Namespace(
        run_dir=run, output_dir=tmp_path / "report", arms="A",
        a_repeats=1, e_repeats=1, run_quality_review=False, candidate_arms="F1",
    )
    result = reporter.collect_report(args)
    comparison = result["clip_nonregression"]["clips"]["F1/sit180"]
    assert comparison["summary_direction"] == "P2 worse"
    p1 = comparison["priorities"]["1"]
    p2 = comparison["priorities"]["2"]
    assert p1["worse"] == [] and p1["flat_count"] > 0
    static = next(item for item in p2["worse"] if item["metric"] == "static_penetration.max_mm")
    assert static["A"] == 6.0 and static["candidate"] == 12.0
    assert static["noise_tolerance"] == 1e-9
    assert result["expected_normal_runs"] == 2
    assert "F1 / sit180" in reporter.markdown(result)
