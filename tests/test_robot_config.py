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


def test_shipped_profiles_have_no_unread_configuration_keys():
    """Every profile key must be read somewhere in the package source.

    A JSON key that no code reads is a silent no-op: it looks like a tuning
    knob, drifts out of sync with the key that is actually live, and sends
    whoever tunes the profile down a dead end.  Scanning the source for the
    quoted key name is deliberately permissive -- it only fails on keys that
    appear nowhere at all.
    """
    import json
    from pathlib import Path

    package_root = Path(__file__).resolve().parents[1] / "omniretargeting"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package_root.rglob("*.py")
    )

    # Sections whose keys are user-chosen names (robot joints, effector names,
    # joint-name substrings) rather than fixed configuration keys.
    dynamic_sections = {
        "joint_mapping",
        "joint_posture_limits",
        "joint_temporal_weights",
        "joint_posture_targets",
        "joint_weight_overrides",
        "effectors",
        "groups",
    }

    profiles = sorted(
        (Path(__file__).resolve().parents[1] / "robot_models").rglob("*.json")
    )
    assert profiles, "no robot profiles found"

    # Legacy profiles for robots the contact-aware pipeline has not been
    # migrated to.  They carry an obsolete effector schema and a
    # ``solver_max_iter`` block that no code has ever read, so their SQP
    # iteration counts are silently the built-in defaults.  Listing them keeps
    # the debt visible instead of letting the guard be weakened for everyone.
    known_legacy_debt = {
        "booster_k1.json:retargeting.solver_max_iter",
        "booster_k1.json:retargeting.joint_regularization_boost",
        "booster_k1.json:retargeting.effectors.left.body",
        "booster_k1.json:retargeting.effectors.left.role",
        "booster_k1.json:retargeting.effectors.right.body",
        "booster_k1.json:retargeting.effectors.right.role",
        "hightorque_mini_pi_plus.json:retargeting.solver_max_iter",
        "hightorque_mini_pi_plus.json:retargeting.joint_regularization_boost",
        "hightorque_mini_pi_plus.json:retargeting.effectors.left.body",
        "hightorque_mini_pi_plus.json:retargeting.effectors.left.role",
        "hightorque_mini_pi_plus.json:retargeting.effectors.right.body",
        "hightorque_mini_pi_plus.json:retargeting.effectors.right.role",
    }

    unread: list[str] = []
    for profile_path in profiles:
        config = json.loads(profile_path.read_text(encoding="utf-8"))

        def visit(node, path: str, inside_dynamic: bool) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if inside_dynamic:
                        # A user-chosen name; its children are still checked.
                        visit(value, f"{path}.{key}", False)
                        continue
                    if f'"{key}"' not in source and f"'{key}'" not in source:
                        unread.append(f"{profile_path.name}:{path}.{key}")
                        continue
                    visit(value, f"{path}.{key}", key in dynamic_sections)
            elif isinstance(node, list):
                for item in node:
                    visit(item, path, inside_dynamic)

        visit(config.get("retargeting", {}), "retargeting", False)

    unexpected = sorted(set(unread) - known_legacy_debt)
    assert not unexpected, (
        "configuration keys no code reads: " + ", ".join(unexpected)
    )


def test_instinctmj_profile_expresses_slide_freedom_with_mode_weight():
    """SLIDE softness must not come from disabling the contact-level bound."""
    profile = json.loads(
        (
            REPO_ROOT
            / "robot_models"
            / "unitree_g1"
            / "unitree_g1_instinctmj.json"
        ).read_text(encoding="utf-8")
    )
    priority = profile["retargeting"]["end_effector_priority"]
    refinement = profile["retargeting"]["trajectory_refinement"]

    touch_weight = float(priority.get("touch_weight", 0.5))
    assert 0.0 < priority["slide_tangential_weight"] < touch_weight
    assert priority["contact_detection"]["stabilize_contact_normal_signs"] is True
    assert refinement["contact_hierarchy"] is True
    assert (
        refinement["contact_priority_absolute_tolerance"]
        <= refinement["support_priority_absolute_tolerance"]
    )
