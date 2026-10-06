"""Exercise real MuJoCo FK and production metrics, including leaf twist."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.review_spline_quality import (
    _continuous_log, _compare, evaluate, intervals, main,
)
from scripts.review_source_pose import evaluate as original_pose_audit


@pytest.fixture
def clip(tmp_path):
    model_path = tmp_path / "robot.xml"
    model_path.write_text("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="base"><freejoint name="floating"/><geom size=".02"/>
    <body name="arm" pos="0 0 1"><joint name="turn" axis="0 0 1" range="-3 3"/>
    <geom size=".02"/><body name="tip" pos="1 0 0">
    <joint name="twist" axis="1 0 0" range="-3 3"/><geom size=".02"/>
    </body></body><body name="other" pos="0 1 0"><geom size=".02"/></body>
    </body></worldbody></mujoco>""")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    frames = 24
    reference = np.tile(model.qpos0, (frames, 1))
    source = np.tile([[0, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 0]], (frames, 1, 1))
    prepared = tmp_path / "prepared.npy"
    np.save(prepared, dict(model_path=str(model_path), positions=source,
        source_names=["base", "arm", "tip", "other"],
        mapping={name: name for name in ["base", "arm", "tip", "other"]},
        roots=np.zeros((frames, 3)),
        root_orientation_targets=reference[:, 3:7],
        body_orientation_targets=[dict(robot_body="tip", parent_body="base",
             rotations=np.tile(np.eye(3), (frames, 1, 1))) ]))

    def save(name, q, start=0):
        path = tmp_path / (name + ".npz")
        np.savez(path, qpos=q, framerate=30., frame_start=start)
        return path

    return dict(prepared=prepared, reference=save("reference", reference),
                final=save("final", reference), baseline=save("A", reference)), reference, save


def test_leaf_twist_is_seen_by_fixed_orientation_not_bone_directions(clip):
    paths, q, save = clip
    twist = q.copy()
    twist[:, -1] = .37
    paths["final"] = save("twisted", twist)
    result = evaluate(**paths)
    original = original_pose_audit(paths["prepared"], paths["final"])
    comparison = result["comparisons"]["final_vs_original_source"]
    assert comparison["existing_source_pose_audit"] == original
    assert comparison["metrics"]["bone_directions_rad"]["arm -> tip"]["max"] == 0
    assert comparison["metrics"]["calibrated_body_orientation_rad"]["tip"]["max"] == pytest.approx(.37)
    fixed = result["comparisons"]["final_vs_fixed_ref"]["metrics"]
    assert fixed["calibrated_body_orientation_rad"]["tip"]["max"] == pytest.approx(.37)
    regressions = result["non_degradation_against_A"]["regressions"]
    assert any(row["part"] == "tip" and row["metric"] == "calibrated_body_orientation_rad" for row in regressions)
    assert result["certification"]["all_quality_criteria_pass"] is None
    json.dumps(result, allow_nan=False)


def test_reference_chatter_audits_original_source_and_does_not_certify(clip):
    paths, q, save = clip
    jitter = q.copy()
    jitter[:, -2] = np.where(np.arange(len(q)) % 2, .2, -.2)
    jitter[:, 0] = np.where(np.arange(len(q)) % 2, .03, -.03)
    paths["reference"] = save("jitter_ref", jitter)
    paths["final"] = save("jitter_final", jitter)
    result = evaluate(**paths)
    ref = result["chatter"]["original_source_audits"]["reference"]
    assert ref["any_source_relative_chatter"]
    assert ref["landmarks"]["tip"]["violating_frames"] > 0
    assert ref["root_vs_original_source"]["floating"]["translation"]["violating_frames"] > 0
    correction = result["chatter"]["actual_final_minus_fixed_ref"]
    assert correction["joints"]["turn"]["violating_frames"] == 0
    assert correction["reference_source_audit_has_chatter"]
    assert not correction["scalar_joint_budget_verified_against_original_source"]


def test_sign_equivalent_quaternions_do_not_introduce_chatter(clip):
    paths, q, save = clip
    signs = q.copy()
    signs[::2, 3:7] *= -1
    paths["final"] = save("quaternion_sign", signs)
    result = evaluate(**paths)
    rotation = result["chatter"]["actual_final_minus_fixed_ref"]["root"]["floating"]["rotation"]
    assert rotation["violating_frames"] == 0
    assert rotation["max"] == 0


def test_rotation_log_is_continuous_through_pi():
    angles = np.linspace(2.8, 3.5, 30)
    vectors = _continuous_log(Rotation.from_rotvec(np.c_[np.zeros((30, 2)), angles]))
    np.testing.assert_allclose(vectors[:, 2], angles, atol=1e-12)


def test_per_part_regression_cannot_hide_in_whole_body_mean():
    baseline = {"angle": {"good": np.ones(10), "bad": np.ones(10)}}
    final = {"angle": {"good": np.zeros(10), "bad": np.full(10, 1.01)}}
    result = _compare(final, baseline, 50)
    assert not result["summary_non_degradation"]
    assert result["metrics"]["angle"]["bad"]["worse_than_A_intervals"] == [
        {"start_frame": 50, "end_frame": 59, "frames": 10}]
    assert result["metrics"]["angle"]["bad"]["delta"]["max"] == pytest.approx(.01)


def test_missing_external_audits_and_contacts_are_unverified(clip):
    paths, _, _ = clip
    result = evaluate(**paths)
    for role in ("reference", "final", "baseline"):
        assert result["contact_and_root_drift"][role]["status"] == "unverified"
        for kind in ("scene", "self_collision", "reset", "physics"):
            assert result["existing_audits"][role][kind]["status"] == "unverified"
    assert not result["certification"]["physics_verified"]


def test_external_unknown_and_failed_reset_remain_reported(clip, tmp_path):
    paths, _, _ = clip
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps({"motion": str(paths["final"]),
                                 "static": {"uncertain_samples": 3, "max_m": .02},
                                 "collision_free_certificate": False}))
    reset = tmp_path / "reset.json"
    reset.write_text(json.dumps({"pass": False, "failed_frames": [1, 2]}))
    result = evaluate(**paths, audits={"final": {"scene": str(scene), "reset": str(reset)}})
    audit = result["existing_audits"]["final"]
    assert audit["scene"]["report"]["static"]["uncertain_samples"] == 3
    assert audit["reset"]["report"]["pass"] is False
    assert audit["scene"]["independent_pass_inferred"] is False


def test_wrong_motion_audit_is_rejected(clip, tmp_path):
    paths, _, _ = clip
    report = tmp_path / "wrong.json"
    report.write_text(json.dumps({"motion": str(paths["reference"])}))
    with pytest.raises(ValueError, match="belongs to another motion"):
        evaluate(**paths, audits={"final": {"scene": str(report)}})


def test_frame_offsets_must_match(clip):
    paths, q, save = clip
    paths["final"] = save("shifted", q, 1)
    with pytest.raises(ValueError, match="identical frames"):
        evaluate(**paths)


def test_interval_endpoints_and_no_existing_output_overwrite(clip, tmp_path, monkeypatch):
    assert intervals([False, True, True, False, True], 12) == [
        {"start_frame": 13, "end_frame": 14, "frames": 2},
        {"start_frame": 16, "end_frame": 16, "frames": 1}]
    paths, _, _ = clip
    output = tmp_path / "quality.json"
    argv = ["review_spline_quality.py", "--output", str(output)]
    for name, path in paths.items():
        argv += ["--" + name, str(path)]
    monkeypatch.setattr("sys.argv", argv)
    main()
    assert json.loads(output.read_text())["frames"] == 24
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main()
