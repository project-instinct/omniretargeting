"""Unit coverage for the independent training collision audit."""

import argparse
import hashlib
import json

import mujoco
import numpy as np
import trimesh

from scripts.review_training_collision import (
    audit,
    consume_scene_metrics,
    interpolation_profile,
    TrainingTrajectory,
    training_trajectory,
)


ROBOT_XML = """<mujoco><worldbody><body name="root"><freejoint/>
<geom name="collision" type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>"""


def _profile():
    return {
        "preprocess": "instinct_mj_motion_interpolate_bilinear",
        "object_preprocess": "instinct_mj_pose_interpolate_bilinear",
        "runtime_sampling": "round_nearest_frame",
        "source_framerate": 30.0,
        "target_framerate": 50.0,
        "provenance": {"source": "InstinctMJ loader", "definition": "motion", "lines": "1-2"},
    }


def test_training_interpolation_follows_instinct_mj_frame_indices(tmp_path):
    model = mujoco.MjModel.from_xml_string(ROBOT_XML)
    payload = {"framerate": 30.0, "training_interpolation": _profile()}
    profile = interpolation_profile(payload)
    poses = np.zeros((5, 7), dtype=float)
    poses[:, 3] = 1.0
    poses[:, 2] = np.arange(5)
    result = training_trajectory(model, poses, 30.0, profile, 10)
    assert profile["status"] == "explicit_configuration"
    assert result.source_times.tolist() == [0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6]
    assert result.labels.tolist() == list(range(10, 17))
    np.testing.assert_allclose(result.qpos[:, 2], result.source_times, atol=1e-12)
    assert interpolation_profile({"framerate": 30.0})["status"] == "unverified"


def _write_reusable_audit(root, model, poses, motion, objects, *, valid=True):
    root.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=[0.2, 0.2, 0.2]).export(root / "static_scene.obj")
    depths = np.full((len(poses), model.ngeom), 0.012, dtype=float)
    static = {
        "max_m": 0.012, "p95_m": 0.012, "mean_m": 0.012, "worst_frame": 0,
        "worst_body": "root", "affected_frames": [0, 1, 2], "uncertain_samples": 0,
        "spacing_m": 0.02, "method": "original triangles; winding containment",
        "collision_free_certificate": False,
    }
    moving = {
        "max_m": 0.012, "p95_m": 0.012, "mean_m": 0.012, "worst_frame": 0,
        "worst_body": "root", "affected_frames": [0, 1, 2], "failures": [],
        "unknown_components": 0, "collision_free_certificate": False,
    }
    record_path = root / "scene_metrics.json"
    record_path.write_text(json.dumps({
        "motion": str(motion), "start": 0, "frames": len(poses),
        "static": static, "moving_object_count": 1, "moving": moving,
    }))
    np.savez(root / "scene_metrics.npz", static_depths=depths, moving_depths=depths)
    args = ["python", "--prepared", str(root / "prepared.npy"), "--motion", str(motion),
            "--objects", str(objects), "--output", str(record_path)]
    if not valid:
        args[args.index("--motion") + 1] = str(root / "other.npz")
    (root / "receipt.json").write_text(json.dumps({
        "acceptance": {"receipts": [{"name": "scene_metrics", "args": args, "seconds": 12.5}]}
    }))
    digest = hashlib.sha256(np.ascontiguousarray(poses).tobytes()).hexdigest()
    (root / "final_retargeted.diagnostics.json").write_text(
        json.dumps({"accepted_qpos_sha256": digest})
    )
    return record_path, static, moving


def _training_case(tmp_path):
    robot = tmp_path / "robot.xml"
    robot.write_text(ROBOT_XML)
    model = mujoco.MjModel.from_xml_path(str(robot))
    poses = np.zeros((3, 7), dtype=float)
    poses[:, 3] = 1.0
    motion = tmp_path / "motion.npz"
    np.savez(
        motion, qpos=poses, framerate=30.0, frame_start=0,
        training_interpolation=_profile(),
        training_reset={
            "candidate_frames": "all", "ensure_link_below_zero_ground": False,
            "motion_start_height_offset": 0.0, "sampling": "all",
            "provenance": {"source": "InstinctMJ reset", "definition": "reset", "lines": "1-2"},
        },
    )
    prepared = tmp_path / "prepared.npy"
    np.save(prepared, {"model_path": str(robot)})
    objects = tmp_path / "objects.npy"
    box = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    box.export(tmp_path / "static_scene.obj")
    np.save(objects, np.array([{
        "mesh": box, "translations": np.zeros((3, 3)),
        "rotations": np.tile(np.eye(3), (3, 1, 1)), "scales": np.ones(3),
    }], dtype=object))
    return model, poses, motion, prepared, objects


def test_scene_metrics_reuse_requires_artifact_ownership(tmp_path):
    model, poses, motion, prepared, objects = _training_case(tmp_path)
    record_path, _static, _moving = _write_reusable_audit(
        tmp_path, model, poses, motion, objects
    )
    trajectory = TrainingTrajectory(
        poses.copy(), np.arange(len(poses), dtype=float), np.arange(len(poses), dtype=int),
        {"status": "input_frames"},
    )
    geom_ids = np.flatnonzero((model.geom_contype | model.geom_conaffinity) != 0)
    static, moving, provenance = consume_scene_metrics(
        record_path, motion, prepared, objects, model, trajectory, geom_ids
    )
    assert provenance["status"] == "consumed"
    assert provenance["motion_qpos_sha256"] == hashlib.sha256(
        np.ascontiguousarray(poses).tobytes()
    ).hexdigest()
    assert static["status"] == "measured" and moving["status"] == "measured"
    np.testing.assert_array_equal(static["frame_depths_max_m"], [0.012] * 3)
    assert static["frames_over_1cm"] == [0, 1, 2]
    bad_path, _static, _moving = _write_reusable_audit(
        tmp_path / "bad", model, poses, motion, objects, valid=False
    )
    bad, bad_moving, bad_provenance = consume_scene_metrics(
        bad_path, motion, prepared, objects, model, trajectory, geom_ids
    )
    assert bad["status"] == bad_moving["status"] == "unknown"
    assert bad_provenance["status"] == "unverified"
    assert "receipt_motion_path" in bad_provenance["reason"]


def test_full_audit_consumes_scene_metrics_and_adds_training_gaps(tmp_path):
    model, poses, motion, prepared, objects = _training_case(tmp_path)
    record_path, static_record, _moving_record = _write_reusable_audit(
        tmp_path, model, poses, motion, objects
    )
    output = tmp_path / "training_collision.json"
    report = audit(argparse.Namespace(
        prepared=prepared, motion=motion, objects=objects,
        scene_metrics=record_path, output=output,
    ))
    assert report["scene_metrics_reuse"]["status"] == "consumed"
    assert report["scene_metrics_reuse"]["receipt_scene_metrics_seconds"] == 12.5
    assert report["timing"]["total_including_prior_scene_seconds"] >= 12.5
    assert report["static_scene"]["max_m"] == static_record["max_m"]
    assert report["training_interpolation"]["status"] == "explicit_configuration"
    assert report["interframe_risk_audit"]["status"] == "risk_regions_measured"
    assert report["self_collision"]["status"] in {"measured", "no_eligible_pairs"}
    assert report["reset"]["physics_validation"]["status"] == "unverified"
    assert report["acceptance_status"] == "unverified"
    assert not report["collision_free_certificate"]


def test_enabled_self_query_does_not_overwrite_unknown_measurement(monkeypatch):
    from scripts import review_training_collision as review
    model = mujoco.MjModel.from_xml_string(ROBOT_XML)
    poses = np.repeat(model.qpos0[None], 3, axis=0)
    trajectory = TrainingTrajectory(poses, np.arange(3), np.arange(3), {})

    class Query:
        cfg = {'penetration_tolerance': .001}

        def use_model_pairs(self):
            return []

        def query(self, _pose):
            raise RuntimeError('synthetic native query failure')

        def describe(self):
            return {'status': 'enabled', 'config': self.cfg}

    monkeypatch.setattr(review, 'SelfCollisionQuery', lambda _model: Query())
    result, _ = review.audit_self(model, trajectory, np.array([0]))
    assert result['status'] == 'unknown'
    assert result['query_status'] == 'enabled'
    assert result['unknown_query_count'] == 3
    assert review._global_status({}, {}, result, {})['geometry_status'] == 'unknown'
