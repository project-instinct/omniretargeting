"""Configured landmarks must reach adapter measurements and CLI scene scaling."""

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pytest
import trimesh
import yaml

import omniretargeting.main as cli
from omniretargeting.data_sources.registry import create_data_source
from omniretargeting.data_sources.smplx import DEFAULT_SMPLX_TARGET_NAMES
from omniretargeting.data_sources.smplx import SmplxDataSource
from omniretargeting.robot_config import load_robot_config
from omniretargeting.utils import create_flat_terrain

HEIGHT_ESTIMATION = {
    "head_joint": "crown",
    "foot_joints": ["sole_l", "sole_r"],
    "head_top_offset": 0.25,
}
NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "crown", "sole_l", "sole_r"]
POSITIONS = np.array(
    [
        [0, 0, 0.9],
        [0, -0.1, 0.85],
        [0, 0.1, 0.85],
        [0, 0, 1.1],
        [0, 0, 1.5],
        [0, -0.1, 0],
        [0, 0.1, -0.05],
    ]
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_positions(path):
    np.savez(path, global_joint_positions=POSITIONS[None], framerate=30)


@pytest.fixture(params=["smplx", "omomo", "lafan1", "nokov", "bones_seed"])
def custom_landmark_source(request, tmp_path):
    source_type = request.param
    options = {}
    if source_type == "smplx":
        path = tmp_path / "motion.npz"
        _write_positions(path)
        options["target_names_override"] = NAMES
    elif source_type == "omomo":
        mesh_dir = tmp_path / "data/captured_objects"
        mesh_dir.mkdir(parents=True)
        trimesh.creation.box().export(mesh_dir / "box_cleaned_simplified.obj")
        offsets = np.zeros((24, 3))
        offsets[15, 2] = 1.5
        offsets[11, 2] = -0.05
        sequence = {
            "seq_name": "sub01_box_000",
            "rest_offsets": offsets,
            "root_orient": np.zeros((1, 3)),
            "pose_body": np.zeros((1, 63)),
            "trans": np.zeros((1, 3)),
            "obj_scale": np.ones(1),
            "obj_rot": np.eye(3)[None],
            "obj_trans": np.zeros((1, 3, 1)),
            "obj_com_pos": np.zeros((1, 3)),
        }
        path = tmp_path / "sequence.p"
        joblib.dump({0: sequence}, path)
        names = list(DEFAULT_SMPLX_TARGET_NAMES)
        names[15], names[10], names[11] = "crown", "sole_l", "sole_r"
        options.update(
            data_root=str(tmp_path),
            body_position_mode="rest_offsets",
            target_names=names,
            n_object_samples=4,
        )
    else:
        # All three BVH adapters use cm/Y-up; a 155 cm head-to-lowest-foot span.
        path = tmp_path / "motion.bvh"
        joints = "".join(
            f"    JOINT {name}\n    {{\n        OFFSET 0 {height} 0\n"
            "        CHANNELS 3 Zrotation Xrotation Yrotation\n    }\n"
            for name, height in [("crown", 100), ("sole_l", -50), ("sole_r", -55)]
        )
        path.write_text(
            "HIERARCHY\nROOT Hips\n{\n    OFFSET 0 0 0\n"
            "    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
            + joints
            + "}\nMOTION\nFrames: 1\nFrame Time: 0.0333333333\n"
            + " ".join(["0"] * 15)
            + "\n"
        )
    return source_type, path, options


@pytest.mark.parametrize("location", ["source", "adapter_options", "runtime"])
def test_configured_height_landmarks_reach_adapter(custom_landmark_source, location):
    source_type, path, options = custom_landmark_source
    source = {**options, "height_estimation": {"head_joint": "missing"}}
    runtime = {}
    if location == "source":
        source["height_estimation"] = HEIGHT_ESTIMATION
    elif location == "adapter_options":
        source["adapter_options"] = {"height_estimation": HEIGHT_ESTIMATION}
    else:
        source["adapter_options"] = {"height_estimation": {"head_joint": "missing"}}
        runtime["height_estimation"] = HEIGHT_ESTIMATION

    motion = create_data_source(source_type, path, source, runtime).load()

    # 1.55 m span + the configured 0.25 m offset; neither default landmarks nor
    # the default 0.12 m offset can produce this measurement.
    assert motion.source_height == pytest.approx(1.8)


def test_explicit_height_settings_take_precedence_over_smplx_model(
    tmp_path, monkeypatch
):
    path = tmp_path / "motion.npz"
    _write_positions(path)
    monkeypatch.setattr(SmplxDataSource, "compute_human_height", lambda self: 1.6)
    options = {"target_names_override": NAMES}
    default_motion = create_data_source("smplx", path, options).load()
    configured_motion = create_data_source(
        "smplx", path, {**options, "height_estimation": HEIGHT_ESTIMATION}
    ).load()

    assert default_motion.source_height == pytest.approx(1.6)
    assert configured_motion.source_height == pytest.approx(1.8)


@pytest.mark.parametrize("location", ["profile", "source", "yaml"])
def test_cli_scales_scene_with_configured_height_landmarks(
    tmp_path, monkeypatch, location
):
    motion_path = tmp_path / "motion.npz"
    _write_positions(motion_path)
    cfg = load_robot_config(REPO_ROOT / "robot_models/unitree_g1/unitree_g1.json")
    profile = {
        "urdf_path": cfg["urdf_path"],
        "robot_height": 1.35,
        "source": [
            {
                "name": "smplx",
                "type": "smplx",
                "target_mapping": {
                    name: cfg["joint_mapping"][name if name != "Spine1" else "Spine2"]
                    for name in NAMES[:4]
                },
                "base_orientation": dict(
                    zip(["pelvis", "left_hip", "right_hip", "spine"], NAMES[:4])
                ),
            }
        ],
        "retargeting": {"penetration_resolver": "xyz_nudge", "solver_max_iter": 1},
        "height_estimation": {"head_joint": "missing"},
    }
    terrain_path = tmp_path / "terrain.obj"
    terrain = create_flat_terrain()
    terrain.export(terrain_path)
    source = {
        "type": "smplx",
        "motion": str(motion_path),
        "terrain": str(terrain_path),
        "target_names_override": NAMES,
    }
    if location == "profile":
        profile["height_estimation"] = HEIGHT_ESTIMATION
    elif location == "source":
        profile["source"][0]["height_estimation"] = HEIGHT_ESTIMATION
    else:
        profile["source"][0]["height_estimation"] = {"head_joint": "missing"}
        source["height_estimation"] = HEIGHT_ESTIMATION

    profile_path = tmp_path / "robot.json"
    profile_path.write_text(json.dumps(profile))
    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(source))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "omniretargeting",
            "--robot-config",
            str(profile_path),
            "--source-config",
            str(source_path),
            "--output",
            str(tmp_path / "result.npz"),
            "--enable-scene-scaling",
        ],
    )

    cli.main()

    with np.load(tmp_path / "result_retargeted.npz") as result:
        assert np.isfinite(result["base_pos_w"]).all()
        assert np.isfinite(result["joint_pos"]).all()
    scaled = trimesh.load(
        tmp_path / "result_retargeted/scaled_terrain.obj", force="mesh"
    )
    # The height measurement is consumed by the actual CLI and math engine:
    # robot_height/source_height = 1.35/1.8 = 0.75.
    np.testing.assert_allclose(scaled.bounds, terrain.bounds * 0.75, atol=1e-7)
