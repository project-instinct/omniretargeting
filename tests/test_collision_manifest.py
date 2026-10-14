from pathlib import Path

import mujoco
import pytest

from omniretargeting.collision_manifest import (
    collision_manifest,
    collision_manifest_difference,
)
from omniretargeting.robot_config import load_robot_config
from omniretargeting.scene_geometry import collision_geom_ids

ROOT = Path(__file__).resolve().parents[1]
MESH_MODEL = ROOT / "robot_models/unitree_g1/g1_29dof_torsobase_mesh.xml"
INSTINCT_PROFILE = ROOT / "robot_models/unitree_g1/unitree_g1_instinctmj.json"


def load_model(path):
    return mujoco.MjModel.from_xml_path(str(path))


def test_same_model_manifest_and_hash_are_stable():
    first = collision_manifest(load_model(MESH_MODEL), source_path=MESH_MODEL)
    second = collision_manifest(load_model(MESH_MODEL), source_path="copy.xml")

    assert first["sha256"] == second["sha256"]
    assert first["geoms"] == second["geoms"]
    assert first["exclude_pairs"] == second["exclude_pairs"]
    assert first["active_geom_count"] == 35


def test_collision_geometry_changes_change_hash():
    model = load_model(MESH_MODEL)
    before = collision_manifest(model)
    geom_id = int(collision_geom_ids(model)[0])

    model.geom_size[geom_id, 0] += 0.001
    radius_changed = collision_manifest(model)
    assert radius_changed["sha256"] != before["sha256"]

    model.geom_contype[geom_id] = 2
    mask_changed = collision_manifest(model)
    assert mask_changed["sha256"] != radius_changed["sha256"]


def test_structural_contact_excludes_change_hash_without_changing_geoms():
    xml = MESH_MODEL.read_text().replace('meshdir="meshes"', f'meshdir="{MESH_MODEL.parent / "meshes"}"')
    for first, second in [('torso_link', 'waist_yaw_link'),
                          ('left_knee_link', 'left_ankle_roll_link'),
                          ('right_knee_link', 'right_ankle_roll_link')]:
        xml = xml.replace(f'    <exclude body1="{first}" body2="{second}" />\n', '')
    before = collision_manifest(mujoco.MjModel.from_xml_string(xml))
    after = collision_manifest(load_model(MESH_MODEL))
    assert before['geoms'] == after['geoms']
    assert before['sha256'] != after['sha256']
    assert len(before['exclude_pairs']) == 4
    assert len(after['exclude_pairs']) == 7


def test_popsicle_and_mesh_models_have_different_manifest_summaries():
    mesh = collision_manifest(load_model(MESH_MODEL))
    popsicle_path = Path(load_robot_config(INSTINCT_PROFILE)["urdf_path"])
    popsicle = collision_manifest(load_model(popsicle_path), source_path=popsicle_path)
    difference = collision_manifest_difference(mesh, popsicle)

    assert mesh["sha256"] != popsicle["sha256"]
    assert difference["geom_count"] == {
        "first": 35,
        "second": 29,
        "delta": -6,
    }
    assert difference["geom_type_counts"] == {
        "first": {"mjGEOM_MESH": 35},
        "second": {"mjGEOM_CAPSULE": 28, "mjGEOM_SPHERE": 1},
    }
    assert difference["geom_type_differences"] == {
        "mjGEOM_CAPSULE": {"first": 0, "second": 28, "delta": 28},
        "mjGEOM_MESH": {"first": 35, "second": 0, "delta": -35},
        "mjGEOM_SPHERE": {"first": 0, "second": 1, "delta": 1},
    }


def test_mesh_model_build_reproduces_the_committed_model(tmp_path):
    import shutil
    from scripts import build_g1_mesh_collision_model as build

    try:
        source = Path(load_robot_config(build.SOURCE_PROFILE)["urdf_path"])
    except FileNotFoundError:
        pytest.skip("InstinctMJ torso-base G1 model not available")
    if not source.exists():
        pytest.skip("InstinctMJ torso-base G1 model not available")
    shutil.copytree(MESH_MODEL.parent / "meshes", tmp_path / "meshes")
    for _ in range(2):  # a rebuild over its own output is a no-op
        model = build.build(tmp_path)
        assert model.read_text() == MESH_MODEL.read_text()
    for mesh in (tmp_path / "meshes").iterdir():
        assert mesh.read_bytes() == (MESH_MODEL.parent / "meshes" / mesh.name).read_bytes(), mesh.name
