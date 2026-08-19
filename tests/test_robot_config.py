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
