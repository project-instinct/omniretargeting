"""Build the G1 mesh-collision model used by ``robot_models/unitree_g1/unitree_g1_mesh.json``.

Keeps the InstinctMJ torso-base G1 kinematics and replaces every active collision
geom with one convex hull per original, undecimated visual STL. Native CCD is
disabled: MuJoCo 3.10 native CCD returned no witnesses for separated wrist hulls,
where libccd agreed with an independent convex QP.

Usage: .venv/bin/python scripts/build_g1_mesh_collision_model.py [--output-dir DIR]
"""
import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from omniretargeting.robot_config import load_robot_config

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROFILE = ROOT / "robot_models/unitree_g1/unitree_g1_instinctmj.json"
MODEL_NAME = "g1_29dof_torsobase_mesh.xml"
PROFILE_NAME = "unitree_g1_mesh.json"


def build(output_dir):
    source = Path(load_robot_config(SOURCE_PROFILE)["urdf_path"])
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "g1_29dof_torsobase_original_mesh_collision_trial")
    compiler = root.find("compiler")
    compiler.set("meshdir", str((source.parent / compiler.get("meshdir", ".")).resolve()))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("nativeccd", "disable")
    for body in root.find("worldbody").iter("body"):
        for geom in list(body.findall("geom")):
            if int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1")):
                body.remove(geom)
            elif geom.get("type") == "mesh":
                collider = copy.deepcopy(geom)
                collider.set("name", f"mesh_collision_{body.get('name')}_{geom.get('mesh')}")
                collider.set("contype", "1")
                collider.set("conaffinity", "1")
                collider.set("group", "3")
                collider.set("density", "0")
                body.append(collider)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_NAME
    ET.indent(tree, space="  ")
    tree.write(model_path, encoding="unicode")
    profile = json.loads(SOURCE_PROFILE.read_text())
    profile["name"] = "unitree_g1_torso_base_original_mesh_collision_trial"
    profile["urdf_path"] = MODEL_NAME
    (output_dir / PROFILE_NAME).write_text(json.dumps(profile, indent=2) + "\n")
    check_kinematics(source, model_path)
    return model_path


def check_kinematics(source, model_path):
    """Only collision geometry may differ: bodies, joints, inertia and FK are identical."""
    old = mujoco.MjModel.from_xml_path(str(source))
    new = mujoco.MjModel.from_xml_path(str(model_path))
    for attr in ("qpos0", "body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia",
                 "body_ipos", "body_iquat", "jnt_type", "jnt_axis", "jnt_pos", "jnt_range",
                 "jnt_qposadr", "jnt_dofadr", "actuator_trnid"):
        assert np.array_equal(getattr(old, attr), getattr(new, attr)), attr
    old_data, new_data = mujoco.MjData(old), mujoco.MjData(new)
    rng = np.random.default_rng(0)
    low, high = old.jnt_range[1:, 0], old.jnt_range[1:, 1]
    for _ in range(64):
        pose = old.qpos0.copy()
        pose[7:] = rng.uniform(low, high)
        old_data.qpos[:] = new_data.qpos[:] = pose
        mujoco.mj_kinematics(old, old_data)
        mujoco.mj_kinematics(new, new_data)
        assert np.array_equal(old_data.xpos, new_data.xpos) and np.array_equal(old_data.xmat, new_data.xmat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "robot_models/unitree_g1")
    print(build(parser.parse_args().output_dir))
