"""Lightweight tests for the read-only spline-quality table exporter."""

import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts.export_spline_quality_tables import export_tables


def _stats(value, worst=2):
    return {
        "count": 2, "mean": value, "p95": value, "max": value,
        "worst_frame": worst,
        "abnormal_intervals": [{"start_frame": 1, "end_frame": 1, "frames": 1}],
    }


def _quality():
    comparisons = {}
    for section in (
        "ref_vs_original_source", "final_vs_original_source", "final_vs_fixed_ref",
    ):
        comparisons[section] = {"metrics": {"position_error": {
            "left_wrist": _stats(0.2, worst=7),
            "root": _stats(0.1, worst=3),
        }}}
    left = {
        "A": _stats(0.1), "final": _stats(0.3, worst=7),
        "delta": {"mean": 0.2, "p95": 0.2, "max": 0.2},
        "summary_increased": True, "worse_than_A_frames": 2,
        "worse_than_A_intervals": [{"start_frame": 4, "end_frame": 5, "frames": 2}],
    }
    root = {
        "A": _stats(0.2), "final": _stats(0.15, worst=2),
        "delta": {"mean": -0.05, "p95": -0.05, "max": -0.05},
        "summary_increased": False, "worse_than_A_frames": 0,
        "worse_than_A_intervals": [],
    }
    return {
        "clip": "sit180", "frame_start": 0,
        "comparisons": comparisons,
        "non_degradation_against_A": {
            "metrics": {"relation_angle": {"left_wrist": left, "root": root}},
            "regressions": [{"metric": "relation_angle", "part": "left_wrist", "mean": 0.2}],
        },
        "contact_and_root_per_part": {
            "final_against_A": {"metrics": {"root_gap": {
                "left_foot": {**left, "A": _stats(0.01), "final": _stats(0.04, worst=8),
                              "delta": {"mean": 0.03, "p95": 0.03, "max": 0.03}},
            }}},
            "chatter": {
                "status": "measured",
                "roles": {
                    "final_minus_reference": {
                        "surface:left_foot:binding_1": {
                            "part": "left_foot", "kind": "surface", "unit": "m",
                            "binding_index": 1, "active_frames": 5,
                            "active_violating_frames": 2,
                            "whole_trajectory": {
                                "count": 2, "mean": .01, "p95": .02, "max": .03,
                                "worst_frame": 4, "abnormal_intervals": [],
                                "max_excess": .004, "excess_tolerance": .001,
                                "per_coordinate": [
                                    {"count": 2, "mean": .005, "p95": .006, "max": .007,
                                     "worst_frame": 3, "abnormal_intervals": [], "max_excess": .002},
                                    {"count": 2, "mean": .008, "p95": .009, "max": .010,
                                     "worst_frame": 4, "abnormal_intervals": [], "max_excess": .003},
                                    {"count": 2, "mean": .002, "p95": .003, "max": .004,
                                     "worst_frame": 5, "abnormal_intervals": [], "max_excess": .001},
                                ],
                            },
                            "active_contact": {
                                "count": 2, "mean": .012, "p95": .013, "max": .014,
                                "worst_frame": 6,
                                "abnormal_intervals": [{"start_frame": 5, "end_frame": 6, "frames": 2}],
                            },
                            "active_per_coordinate": [
                                {"count": 2, "mean": .004, "p95": .005, "max": .006,
                                 "worst_frame": 5, "abnormal_intervals": []},
                                {"count": 2, "mean": .007, "p95": .008, "max": .009,
                                 "worst_frame": 6, "abnormal_intervals": []},
                                {"count": 2, "mean": .001, "p95": .002, "max": .003,
                                 "worst_frame": 4, "abnormal_intervals": []},
                            ],
                        },
                    },
                    "baseline_minus_baseline_reference": {
                        "surface:left_foot:binding_1": {
                            "part": "left_foot", "kind": "surface", "unit": "m",
                            "binding_index": 1, "active_frames": 4,
                            "active_violating_frames": 1,
                            "whole_trajectory": {
                                "count": 2, "mean": .005, "p95": .006, "max": .007,
                                "worst_frame": 2, "abnormal_intervals": [],
                                "max_excess": .002, "excess_tolerance": .001,
                                "per_coordinate": [
                                    {"count": 2, "mean": .002, "p95": .003, "max": .004,
                                     "worst_frame": 1, "abnormal_intervals": [], "max_excess": .001},
                                ],
                            },
                            "active_contact": {
                                "count": 2, "mean": .005, "p95": .006, "max": .007,
                                "worst_frame": 2, "abnormal_intervals": [],
                            },
                            "active_per_coordinate": [
                                {"count": 2, "mean": .002, "p95": .003, "max": .004,
                                 "worst_frame": 1, "abnormal_intervals": []},
                            ],
                        },
                    },
                },
                "original_source_audits": {
                    "reference": {
                        "endpoint:hand:offset_0": {
                            "status": "unverified", "part": "hand",
                            "reason": "endpoint source correspondence unverified",
                        },
                    },
                    "final": {
                        "endpoint:hand:offset_0": {
                            "status": "unverified", "part": "hand",
                            "reason": "endpoint source correspondence unverified",
                        },
                        "surface:left_foot:binding_1": {
                            "status": "measured", "part": "left_foot",
                            "report": {
                                "count": 2, "mean": .011, "p95": .012, "max": .013,
                                "worst_frame": 7, "abnormal_intervals": [],
                                "max_excess": .003, "excess_tolerance": .001,
                                "per_coordinate": [
                                    {"count": 2, "mean": .004, "p95": .005, "max": .006,
                                     "worst_frame": 6, "abnormal_intervals": [], "max_excess": .002},
                                ],
                            },
                            "provenance": "surface_contact_environment_w + surface_contact_source_vectors_w",
                        },
                    },
                },
                "final_against_A": {"metrics": {
                    "contact_correction_high_frequency_m": {
                        "surface:left_foot:binding_1/axis_1": {
                            "A": _stats(.006), "final": _stats(.009, worst=6),
                            "delta": {"mean": .003, "p95": .003, "max": .003},
                            "summary_increased": True, "worse_than_A_frames": 1,
                            "worse_than_A_intervals": [{"start_frame": 5, "end_frame": 5, "frames": 1}],
                        },
                    },
                    "contact_correction_chatter_excess_m": {
                        "surface:left_foot:binding_1/axis_1": {
                            "A": _stats(.001), "final": _stats(.003, worst=6),
                            "delta": {"mean": .002, "p95": .002, "max": .002},
                            "summary_increased": True, "worse_than_A_frames": 1,
                            "worse_than_A_intervals": [],
                        },
                    },
                }},
            },
        },
        "certification": {"all_quality_criteria_pass": None},
        "original_geometry_per_body": {"categories": {
            "static": {"per_body": {"left_foot": {
                "mean_m": .002, "p95_m": .002, "max_m": .002, "worst_frame": 4,
                "persistent_intervals": [{"start_frame": 4, "end_frame": 4, "frames": 1}],
                "A": {"mean_m": .001, "p95_m": .001, "max_m": .001},
                "delta": {"mean_m": .001, "p95_m": .001, "max_m": .001},
            }}},
            "moving": {"per_body": {"right_hand": {
                "mean_m": .003, "p95_m": .003, "max_m": .003, "worst_frame": 9,
                "A": {"mean_m": .002, "p95_m": .002, "max_m": .002},
                "delta": {"mean_m": .001, "p95_m": .001, "max_m": .001},
            }}},
        }},
        "chatter": {
            "actual_final_minus_fixed_ref": {
                "joints": {"left_elbow_joint": {
                    "units": "rad", "count": 2, "mean": .01, "p95": .01, "max": .02,
                    "worst_frame": 6, "abnormal_intervals": [],
                    "max_excess": .004, "excess_tolerance": .001, "per_coordinate": [
                        {"count": 2, "mean": .008, "p95": .008, "max": .015,
                         "worst_frame": 5, "abnormal_intervals": [], "max_excess": .003,
                         "excess_tolerance": .001},
                    ],
                }},
                "root": {"floating": {
                    "translation": {"count": 2, "mean": .01, "p95": .01, "max": .02,
                                    "worst_frame": 7, "abnormal_intervals": [], "max_excess": .002},
                    "rotation": {"count": 1, "mean": .01, "p95": .01, "max": .02,
                                 "worst_frame": 8, "abnormal_intervals": [], "max_excess": .002},
                }},
                "production_motion_budgets_audit": {"root_rotation": {
                    "violations": 1, "per_coordinate": [
                        {"count": 2, "mean": .01, "p95": .01, "max": .02,
                         "worst_frame": 6, "abnormal_intervals": [], "max_excess": .004},
                    ],
                }},
            },
            "original_source_audits": {
                "reference": {
                    "landmarks": {"L_Wrist": {"count": 2, "mean": .01, "p95": .01, "max": .02,
                                              "worst_frame": 6, "abnormal_intervals": [], "max_excess": .003}},
                    "calibrated_body_orientation": {"pelvis": {
                        "count": 2, "mean": .01, "p95": .01, "max": .02,
                        "worst_frame": 6, "abnormal_intervals": [], "max_excess": .003}},
                    "root_vs_original_source": {"floating": {
                        "translation": {"count": 2, "mean": .01, "p95": .01, "max": .02,
                                        "worst_frame": 7, "abnormal_intervals": [], "max_excess": .002},
                        "rotation": {"count": 1, "mean": .01, "p95": .01, "max": .02,
                                      "worst_frame": 8, "abnormal_intervals": [], "max_excess": .002},
                    }},
                },
                "final": {
                    "landmarks": {"R_Wrist": {"count": 2, "mean": .01, "p95": .01, "max": .02,
                                              "worst_frame": 6, "abnormal_intervals": [], "max_excess": .003}},
                },
            },
        },
    }


def _training():
    body = lambda value, worst: {
        "mean_m": value, "p95_m": value, "max_m": value,
        "worst_frame": worst,
        "A": {"mean_m": value * .5, "p95_m": value * .5, "max_m": value * .5},
        "delta": {"mean_m": value * .5, "p95_m": value * .5, "max_m": value * .5},
    }
    return {
        "acceptance_status": "unverified",
        "self_collision": {
            "status": "measured", "unknown_query_count": 0,
            "per_body": {"left_hand": body(.001, 6)},
            "per_pair": {"left_hand / geom 1 | torso / geom 2": body(.0012, 6)},
            "frames_over_1mm": [], "frames_over_5mm": [], "frames_over_1cm": [],
        },
        "static_scene": {"status": "measured", "uncertain_samples": 0},
        "moving_objects": {"status": "measured", "unknown_query_count": 0},
    }


def _smoothness():
    return {
        "all_individual_budgets_pass": True,
        "joints": {"left_elbow": {
            "units": "rad",
            "source_relative_chatter": {"max_excess": .01, "worst_frame": 5, "violations": 0},
            "orders": {"1": {"count": 1, "max_increase_per_frame_order": .02, "worst_increase_frame": 4}},
            "mechanical_limits": {"velocity": {"peak_per_second": 1.2, "violations": 0}},
        }},
    }


def _clip(root: Path, arm: str, *, complete=True):
    directory = root / arm / "normal" / "repeat_1" / "sit180"
    directory.mkdir(parents=True)
    (directory / "receipt.json").write_text(json.dumps({
        "arm": arm, "mode": "normal", "repeat": 1, "clip": "sit180",
        "solver_returncode": 0, "end_to_end_seconds": 1.0,
    }))
    (directory / "source.json").write_text(json.dumps({
        "type": "trumans", "frame_start": 50, "frame_stop": 230,
    }))
    if complete:
        (directory / "spline_quality.json").write_text(json.dumps(_quality()))
        (directory / "training_collision.json").write_text(json.dumps(_training()))
        (directory / "joint_smoothness_supplemental.json").write_text(json.dumps(_smoothness()))
    return directory


def _read(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _args(run_dir: Path, output_dir: Path):
    return argparse.Namespace(
        run_dir=run_dir, repeats=1, output_dir=output_dir,
        arms="A,B,D", modes="normal", clips="sit180",
    )


def test_exports_parts_geometry_self_source_offset_and_missing_audits(tmp_path):
    run = tmp_path / "measure"
    _clip(run, "A")
    _clip(run, "B")
    _clip(run, "D", complete=False)
    output = tmp_path / "tables"
    result = export_tables(_args(run, output))
    assert result["clips"] == 3
    assert result["part_rows"] > 0
    rows = _read(output / "quality_parts.csv")
    b_quality = [row for row in rows if row["arm"] == "B"]
    sections = {row["section"] for row in b_quality}
    assert {
        "ref_vs_original_source", "final_vs_original_source", "final_vs_fixed_ref",
        "non_degradation_against_A",
    } <= sections
    for section in (
        "ref_vs_original_source", "final_vs_original_source", "final_vs_fixed_ref",
        "non_degradation_against_A",
    ):
        assert {"left_wrist", "root"} <= {
            row["part"] for row in b_quality
            if row["section"] == section
        }

    relation = [row for row in b_quality if row["section"] == "non_degradation_against_A"]
    left = next(row for row in relation if row["part"] == "left_wrist")
    root = next(row for row in relation if row["part"] == "root")
    assert float(left["delta_max"]) == .2
    assert float(root["delta_max"]) == -.05
    assert left["worse_than_A_intervals_json"] != ""
    markdown = (output / "README.md").read_text()
    relation_markdown_rows = [
        line for line in markdown.splitlines()
        if line.startswith("| B | normal | 1 | sit180 | spline_quality | relation_angle |")
    ]
    assert len(relation_markdown_rows) == 1
    assert "left_wrist" in relation_markdown_rows[0]
    assert "| 0.2 | 0.2 | 0.2 |" in relation_markdown_rows[0]
    assert "root" not in relation_markdown_rows[0]

    static_geometry = next(
        row for row in rows
        if row["audit_kind"] == "original_geometry" and row["section"] == "geometry:static"
        and row["part"] == "left_foot"
    )
    moving_geometry = next(
        row for row in rows
        if row["audit_kind"] == "original_geometry" and row["section"] == "geometry:moving"
        and row["part"] == "right_hand"
    )
    assert static_geometry["A_max"] == "0.001"
    assert static_geometry["delta_max"] == "0.001"
    assert static_geometry["abnormal_frames"] == "1"
    assert moving_geometry["A_max"] == "0.002"
    self_row = next(
        row for row in rows
        if row["audit_kind"] == "self_collision" and row["section"] == "per_body" and row["part"] == "left_hand"
    )
    assert self_row["unit"] == "m"
    assert float(self_row["max"]) == .001
    assert self_row["original_source_frame"] == "56"
    assert self_row["source_frame_start"] == "50"
    quality_row = next(
        row for row in b_quality
        if row["section"] == "final_vs_original_source" and row["part"] == "left_wrist"
    )
    assert quality_row["worst_frame_clip_local"] == "7"
    assert quality_row["original_source_frame"] == "57"
    assert quality_row["unit_conversion"] == "none"
    pair_row = next(
        row for row in rows
        if row["audit_kind"] == "self_collision" and row["section"] == "per_pair"
    )
    assert float(pair_row["max"]) == .0012
    chatter_sections = {row["section"] for row in rows if row["audit_kind"] == "chatter"}
    assert {
        "actual_final_minus_fixed_ref:joints", "actual_final_minus_fixed_ref:root",
        "production_motion_budgets_audit", "source_chatter:reference:landmarks",
        "source_chatter:reference:calibrated_body_orientation",
        "source_chatter:reference:root_vs_original_source",
        "source_chatter:final:landmarks",
    } <= chatter_sections
    assert any(
        row["section"] == "actual_final_minus_fixed_ref:joints"
        and row["part"] == "left_elbow_joint / axis_0"
        and row["metric"] == "chatter_excess" for row in rows
    )
    chatter_axis = next(
        row for row in rows
        if row["audit_kind"] == "chatter" and row["metric"] == "chatter_excess"
        and row["part"] == "left_elbow_joint / axis_0"
    )
    assert chatter_axis["max"] == "0.003"
    assert chatter_axis["excess_tolerance"] == "0.001"
    assert chatter_axis["excess_definition"].startswith("abs(high_frequency")

    contact_chatter = [row for row in rows if row["audit_kind"] == "contact_chatter"]
    assert {
        "role:final_minus_reference:whole_trajectory",
        "role:final_minus_reference:active_contact",
        "role:baseline_minus_baseline_reference:whole_trajectory",
        "role:baseline_minus_baseline_reference:active_contact",
        "final_minus_A",
    } <= {row["section"] for row in contact_chatter}
    final_axis = next(
        row for row in contact_chatter
        if row["section"] == "final_minus_A"
        and row["metric"] == "contact_correction_high_frequency_m"
        and row["part"] == "surface:left_foot:binding_1/axis_1"
    )
    assert final_axis["A_max"] == "0.006"
    assert final_axis["max"] == "0.009"
    assert final_axis["delta_max"] == "0.003"
    assert final_axis["binding_id"] == "surface:left_foot:binding_1"
    assert final_axis["binding_index"] == "1"
    assert final_axis["unit"] == "m"
    active = next(
        row for row in contact_chatter
        if row["section"] == "role:final_minus_reference:active_contact"
        and row["metric"] == "active_contact_high_frequency_m"
        and row["part"] == "surface:left_foot:binding_1"
    )
    assert active["max"] == "0.014"
    assert active["active_frames"] == "5"
    assert active["active_violating_frames"] == "2"
    assert active["abnormal_interval_count"] == "1"
    active_axis = next(
        row for row in contact_chatter
        if row["section"] == "role:final_minus_reference:active_contact"
        and row["metric"] == "active_contact_high_frequency_m"
        and row["part"] == "surface:left_foot:binding_1/axis_1"
    )
    assert active_axis["max"] == "0.009"
    assert active_axis["binding_index"] == "1"
    baseline_role = next(
        row for row in contact_chatter
        if row["section"] == "role:baseline_minus_baseline_reference:whole_trajectory"
        and row["metric"] == "high_frequency_content"
    )
    assert baseline_role["max"] == "0.007"
    source_audit = next(
        row for row in rows
        if row["audit_kind"] == "contact_source_chatter"
        and row["section"] == "role:final"
        and row["part"] == "surface:left_foot:binding_1"
        and row["metric"] == "high_frequency_content"
    )
    assert source_audit["max"] == "0.013"
    assert source_audit["source_status"] == "measured"
    assert "surface_contact_environment_w + surface_contact_source_vectors_w" in (
        source_audit["source_provenance"])

    coverage = _read(output / "coverage.csv")
    d = next(row for row in coverage if row["arm"] == "D")
    assert d["quality_present"] == "False"
    assert d["training_collision_present"] == "False"
    assert d["smoothness_present"] == "False"
    assert d["pass_inferred"] == "False"
    defects = _read(output / "defects.csv")
    assert {
        row["metric"] for row in defects if row["arm"] == "D"
    } >= {"spline_quality", "training_collision", "joint_smoothness"}
    assert any(row["metric"] == "relation_angle" and row["part"] == "left_wrist" for row in defects)
    endpoint_unknown = [
        row for row in defects
        if row["kind"] == "contact_source_chatter" and row["severity"] == "unknown"
        and row["part"] == "endpoint:hand:offset_0"
    ]
    assert {row["metric"] for row in endpoint_unknown} == {"role:reference", "role:final"}
    assert all("unverified" in row["detail"] for row in endpoint_unknown)
    assert "no pass is inferred" in markdown


def test_output_directory_is_exclusive(tmp_path):
    run = tmp_path / "measure"
    _clip(run, "A")
    output = tmp_path / "tables"
    output.mkdir()
    before = output.iterdir()
    with pytest.raises(FileExistsError):
        export_tables(_args(run, output))
    assert list(before) == []


def test_expected_benchmark_clips_without_receipts_stay_in_coverage(tmp_path):
    run = tmp_path / "measure"
    _clip(run, "A")
    output = tmp_path / "tables"
    args = argparse.Namespace(
        run_dir=run, repeats=1, output_dir=output,
        arms="A", modes="normal", clips="",
    )
    result = export_tables(args)
    assert result["clips"] == 10
    coverage = _read(output / "coverage.csv")
    assert sum(row["receipt_present"] == "True" for row in coverage) == 1
    assert sum(row["receipt_present"] == "False" for row in coverage) == 9
    defects = _read(output / "defects.csv")
    assert sum(
        row["kind"] == "receipt" and row["severity"] == "missing"
        for row in defects
    ) == 9


def test_quality_self_comparison_exports_two_bodies_and_pair_delta(tmp_path):
    run = tmp_path / "measure"
    _clip(run, "A")
    _clip(run, "B")
    b = run / "B" / "normal" / "repeat_1" / "sit180"

    def compared(mean, worst):
        return {
            "mean_m": mean, "p95_m": mean, "max_m": mean, "worst_frame": worst,
            "persistent_intervals": [{"start_frame": worst, "end_frame": worst, "frames": 1}],
            "A": {"mean_m": mean - .001, "p95_m": mean - .001, "max_m": mean - .001},
            "delta": {"mean_m": .001, "p95_m": .001, "max_m": .001},
            "worse_than_A_intervals": [{"start_frame": worst + 1, "end_frame": worst + 1, "frames": 1}],
        }

    (b / "spline_quality.json").write_text(json.dumps({
        "clip": "sit180",
        "original_self_collision_per_body": {
            "status": "measured",
            "per_body": {
                "left_hand": compared(.003, 4),
                "torso": compared(.002, 7),
            },
            "per_pair": {
                "left_hand / geom 3 | torso / geom 7": compared(.0035, 5),
            },
        },
    }))
    (b / "training_collision.json").write_text(json.dumps({
        "self_collision": {
            "status": "measured",
            "per_body": {"left_hand": {"mean_m": .003, "p95_m": .003, "max_m": .003, "worst_frame": 4}},
            "per_pair": {"left_hand / geom 3 | torso / geom 7": {
                "mean_m": .0035, "p95_m": .0035, "max_m": .0035, "worst_frame": 5,
            }},
        },
    }))
    (b / "joint_smoothness_supplemental.json").unlink()

    output = tmp_path / "tables"
    args = argparse.Namespace(
        run_dir=run, repeats=1, output_dir=output,
        arms="A,B", modes="normal", clips="sit180",
    )
    result = export_tables(args)
    assert result["clips"] == 2
    rows = _read(output / "quality_parts.csv")
    quality_self = [row for row in rows if row["audit_kind"] == "original_self_collision"]
    bodies = [row for row in quality_self if row["section"] == "per_body"]
    assert {row["part"] for row in bodies} == {"left_hand", "torso"}
    for row in bodies:
        assert row["source_path"] == str(b / "spline_quality.json")
        assert row["unit"] == "m"
        assert row["delta_mean"] == "0.001"
        assert row["delta_p95"] == "0.001"
        assert row["delta_max"] == "0.001"
        assert row["abnormal_interval_count"] == "1"
        assert row["worse_than_A_intervals_json"] != ""
    pair = next(row for row in quality_self if row["section"] == "per_pair")
    assert pair["part"] == "left_hand / geom 3 | torso / geom 7"
    assert pair["A_max"] == "0.0025"
    assert pair["max"] == "0.0035"
    assert pair["delta_max"] == "0.001"

    training_self = [
        row for row in rows
        if row["audit_kind"] == "self_collision" and row["source_path"] == str(b / "training_collision.json")
    ]
    assert {row["section"] for row in training_self} >= {"per_body", "per_pair"}
    assert all(row["A_max"] == "" for row in training_self)

    defects = _read(output / "defects.csv")
    assert any(
        row["kind"] == "quality_self_collision" and row["arm"] == "A"
        and row["metric"] == "original_self_collision_per_body"
        for row in defects
    )
