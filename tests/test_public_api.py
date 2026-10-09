"""Public retargeting wrappers and documented Python/CLI examples."""

import argparse
import ast
import importlib
import inspect
from pathlib import Path
import re
import shlex
import sys

import numpy as np
import pytest

from omniretargeting import OmniRetargeter, load_robot_config
from omniretargeting.data_sources.smplx import (
    DEFAULT_SMPLX_TARGET_NAMES,
    retarget_smplx_to_robot,
)
from omniretargeting.retargeting import retarget_source_to_robot
from omniretargeting.utils import create_flat_terrain

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source_t_pose():
    # A source T-pose in metres, independent of licensed body-model assets.
    return np.array(
        [
            [0, 0, 1],
            [0, -0.1, 0.9],
            [0, 0.1, 0.9],
            [0, 0, 1.2],
            [0, -0.1, 0.5],
            [0, 0.1, 0.5],
            [0, 0, 1.4],
            [0, -0.1, 0.1],
            [0, 0.1, 0.1],
            [0, 0, 1.6],
            [0.05, -0.1, 0.05],
            [0.05, 0.1, 0.05],
            [0, 0, 1.8],
            [0, -0.15, 1.75],
            [0, 0.15, 1.75],
            [0, 0, 1.95],
            [0, -0.3, 1.75],
            [0, 0.3, 1.75],
            [0, -0.55, 1.75],
            [0, 0.55, 1.75],
            [0, -0.75, 1.75],
            [0, 0.75, 1.75],
        ],
        dtype=float,
    )[None]


@pytest.mark.parametrize("wrapper", ["generic", "smplx", "smplx_custom"])
def test_public_wrapper_solves_actual_frame(tmp_path, wrapper, source_t_pose):
    config = load_robot_config(REPO_ROOT / "robot_models/unitree_g1/unitree_g1.json")
    terrain_path = tmp_path / "terrain.obj"
    create_flat_terrain().export(terrain_path)
    positions = source_t_pose
    kwargs = {
        "robot_urdf_path": config["urdf_path"],
        "terrain_mesh_path": terrain_path,
        "robot_height": config.get("robot_height"),
    }
    if wrapper == "smplx":
        scale, result = retarget_smplx_to_robot(
            positions, joint_mapping=config["joint_mapping"], **kwargs
        )
    else:
        names = ["center", "left", "right", "upper"]
        mapping = dict(
            zip(
                names,
                [
                    config["joint_mapping"][name]
                    for name in ["Pelvis", "L_Hip", "R_Hip", "Spine2"]
                ],
            )
        )
        orientation = dict(zip(["pelvis", "left_hip", "right_hip", "spine"], names))
        source = positions[
            :,
            [
                DEFAULT_SMPLX_TARGET_NAMES.index(name)
                for name in ["Pelvis", "L_Hip", "R_Hip", "Spine1"]
            ],
        ]
        if wrapper == "generic":
            scale, result = retarget_source_to_robot(
                source,
                joint_mapping=mapping,
                source_target_names=names,
                base_orientation=orientation,
                **kwargs,
            )
        else:
            scale, result = retarget_smplx_to_robot(
                source,
                joint_mapping=mapping,
                smplx_joint_names=names,
                base_orientation=orientation,
                **kwargs,
            )
    assert np.isfinite(scale) and scale > 0
    assert result.shape[0] == 1 and result.shape[1] > 7
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1, atol=1e-7)


def test_readme_constructor_keywords_match_public_signature():
    text = (REPO_ROOT / "README.md").read_text()
    signature = inspect.signature(OmniRetargeter)
    calls = []
    for block in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
        if "OmniRetargeter(" not in block:
            continue
        for node in ast.walk(ast.parse(block)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "OmniRetargeter"
            ):
                signature.bind_partial(
                    *[None for _ in node.args],
                    **{keyword.arg: None for keyword in node.keywords},
                )
                calls.append(node)
    assert len(calls) == 5


def _readme_cli_commands():
    text = (REPO_ROOT / "README.md").read_text()
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            if line.startswith("python -m omniretargeting."):
                commands.append(shlex.split(line)[2:])
    assert commands
    return commands


@pytest.mark.parametrize("command", _readme_cli_commands())
def test_readme_cli_commands_parse(monkeypatch, command):
    # Run each entry point's actual parser, stopping before asset loading or jobs.
    class ParsedArguments(Exception):
        pass

    parse_args = argparse.ArgumentParser.parse_args
    parsed = []

    def capture_args(parser):
        parsed.append(parse_args(parser))
        raise ParsedArguments

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_args)
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(ParsedArguments):
        importlib.import_module(command[0]).main()
    assert len(parsed) == 1


@pytest.mark.parametrize("positions", [np.zeros(3), np.zeros((0, 22, 3)), []])
def test_smplx_wrapper_rejects_invalid_input_before_default_name_resolution(
    tmp_path, positions
):
    with pytest.raises(ValueError, match="Invalid source position trajectory"):
        retarget_smplx_to_robot(
            positions,
            robot_urdf_path=tmp_path / "unused.urdf",
            terrain_mesh_path=tmp_path / "unused.obj",
            joint_mapping={"Pelvis": "pelvis"},
        )


@pytest.mark.parametrize("example", [0, 1], ids=["quick_start", "profile_setup"])
def test_readme_examples_run_with_bundled_robot_and_processed_motion(
    tmp_path, monkeypatch, source_t_pose, example
):
    text = (REPO_ROOT / "README.md").read_text()
    section = text.split("## Quick Start", 1)[1].split("## Input Format", 1)[0]
    code = re.findall(r"```python\n(.*?)```", section, re.DOTALL)[example]
    motion_path = tmp_path / "positions.npy"
    terrain_path = tmp_path / "terrain.obj"
    np.save(motion_path, source_t_pose)
    create_flat_terrain().export(terrain_path)
    code = code.replace('"path/to/motion_stageii.npz"', repr(str(motion_path)))
    code = code.replace('"path/to/terrain.obj"', repr(str(terrain_path)))
    monkeypatch.chdir(REPO_ROOT)
    namespace = {}
    exec(compile(code, "README.md", "exec"), namespace)
    assert namespace["retargeter"].source_target_names == DEFAULT_SMPLX_TARGET_NAMES
    if example == 0:
        result = namespace["retargeted_motion"]
        assert result.shape[0] == 1 and np.isfinite(result).all()
        np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1, atol=1e-7)
