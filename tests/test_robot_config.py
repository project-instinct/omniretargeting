import json
from pathlib import Path

import pytest

from omniretargeting.robot_config import load_robot_config


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_PROFILE_CASES = (
    pytest.param(REPO_ROOT / "robot_models" / "unitree_g1" / "unitree_g1.json", id="g1"),
    pytest.param(REPO_ROOT / "robot_models" / "unitree_h1" / "unitree_h1.json", id="h1"),
)


def test_load_robot_config_resolves_relative_urdf(tmp_path: Path):
    config_path = tmp_path / "robot.json"
    rel_urdf = "robots/test.urdf"
    (tmp_path / "robots").mkdir()

    config = {
        "name": "test_robot",
        "urdf_path": rel_urdf,
        "joint_mapping": {"Pelvis": "pelvis"},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_robot_config(config_path)

    assert loaded["joint_mapping"] == {"Pelvis": "pelvis"}
    assert loaded["urdf_path"] == str((tmp_path / rel_urdf).resolve())


def test_load_robot_config_requires_joint_mapping(tmp_path: Path):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps({"name": "broken"}), encoding="utf-8")

    with pytest.raises(ValueError, match="joint_mapping"):
        load_robot_config(config_path)


def test_load_robot_config_requires_object(tmp_path: Path):
    config_path = tmp_path / "invalid_shape.json"
    config_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_robot_config(config_path)


def test_contact_retargeting_defaults_merge_robot_overrides(tmp_path: Path):
    profile = tmp_path / "robot.json"
    profile.write_text(
        json.dumps(
            {
                "joint_mapping": {"Pelvis": "pelvis"},
                "retargeting": {
                    "end_effector_priority": {
                        "temporal": {"max_joint_step": 0.1},
                        "effectors": {
                            "left_hand": {"robot_body": "test_left_hand"}
                        },
                    }
                },
            }
        )
    )

    loaded = load_robot_config(profile)
    retargeting = loaded["retargeting"]

    assert retargeting["end_effector_priority"]["temporal"]["enabled"] is True
    assert retargeting["end_effector_priority"]["temporal"]["max_joint_step"] == 0.1
    assert (
        retargeting["end_effector_priority"]["effectors"]["left_hand"][
            "robot_body"
        ]
        == "test_left_hand"
    )
    assert (
        retargeting["end_effector_priority"]["effectors"]["left_hand"][
            "preserve_world_target"
        ]
        is False
    )
    assert set(retargeting["end_effector_priority"]["effectors"]) == {"left_hand"}
    assert retargeting["trajectory_refinement"]["enabled"] is True


def test_retargeting_config_requires_object(tmp_path: Path):
    profile = tmp_path / "robot.json"
    profile.write_text(
        json.dumps(
            {
                "joint_mapping": {"Pelvis": "pelvis"},
                "retargeting": [],
            }
        )
    )

    with pytest.raises(ValueError, match="retargeting must be a JSON object"):
        load_robot_config(profile)


def test_compact_instinctmj_profile_expands_algorithm_defaults():
    profile = REPO_ROOT / "robot_models" / "unitree_g1" / "unitree_g1_instinctmj.json"

    loaded = load_robot_config(profile)
    priority = loaded["retargeting"]["end_effector_priority"]

    assert "retargeting_preset" not in loaded
    assert loaded["retargeting"]["joint_limit_safety_fraction"] == 0.8
    assert priority["contact_detection"]["transition_frames"] == 4
    assert priority["effectors"]["left_hand"]["robot_body"] == "left_rubber_hand"
    assert priority["effectors"]["left_hand"]["preserve_world_target"] is False
    assert priority["effectors"]["left_foot"]["preserve_world_target"] is True
    base_position = priority["base_position_constraint"]
    assert base_position["enabled"] is True
    assert base_position["use_initial_offset"] is True
    assert base_position["lock_initial_xy"] is True
    assert base_position["track_xy"] is True
    assert base_position["lock_xy"] is True
    assert base_position["track_z"] is False
    assert loaded["retargeting"]["trajectory_refinement"][
        "contact_position_sigma"
    ] == pytest.approx(0.008)
    penetration = loaded["retargeting"]["trajectory_refinement"][
        "penetration"
    ]
    assert penetration["target_max_depth"] == pytest.approx(0.10)
    assert penetration["max_depth_regression"] == pytest.approx(0.01)
    assert penetration["sigma"] == pytest.approx(0.008)
    assert penetration["depth_priority_start"] == pytest.approx(0.05)
    assert penetration["max_depth_weight"] == pytest.approx(9.0)
    assert penetration["contact_anchor_exclusion_radius"] == pytest.approx(0.0)
    assert "swing_target_z_offset" not in priority


@pytest.mark.parametrize("profile_path", ROBOT_PROFILE_CASES)
def test_bundled_robot_profiles_match_urdf_bodies(profile_path: Path):
    mujoco = pytest.importorskip("mujoco")

    loaded = load_robot_config(profile_path)
    urdf_path = Path(loaded["urdf_path"])

    assert urdf_path.exists()

    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    body_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_idx)
        for body_idx in range(model.nbody)
    }
    missing = sorted(set(loaded["joint_mapping"].values()) - body_names)

    assert missing == []


def _write_profile(tmp_path, urdf_path):
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps({"urdf_path": urdf_path, "joint_mapping": {"Pelvis": "pelvis"}})
    )
    return profile


def test_urdf_path_list_picks_the_first_candidate_that_exists(tmp_path):
    present = tmp_path / "there.urdf"
    present.write_text("<robot name='x'/>")
    profile = _write_profile(
        tmp_path, ["${OMNIRETARGETING_ASSETS_UNSET_IN_TESTS}/x.urdf", "missing.urdf", "there.urdf"]
    )

    loaded = load_robot_config(profile)

    assert Path(loaded["urdf_path"]) == present.resolve()


def test_urdf_path_list_reports_every_candidate_when_none_exist(tmp_path):
    profile = _write_profile(tmp_path, ["a.urdf", "b.urdf"])

    with pytest.raises(FileNotFoundError) as excinfo:
        load_robot_config(profile)

    assert "a.urdf" in str(excinfo.value)
    assert "b.urdf" in str(excinfo.value)


def test_single_urdf_path_still_resolves_without_existing(tmp_path):
    """A lone path keeps the old contract: resolve it, let the loader complain."""
    profile = _write_profile(tmp_path, "not_created_yet.urdf")

    loaded = load_robot_config(profile)

    assert Path(loaded["urdf_path"]) == (tmp_path / "not_created_yet.urdf").resolve()


def test_user_home_is_expanded_in_urdf_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    present = tmp_path / "home.urdf"
    present.write_text("<robot name='x'/>")
    profile = _write_profile(tmp_path, ["~/home.urdf"])

    loaded = load_robot_config(profile)

    assert Path(loaded["urdf_path"]) == present.resolve()
