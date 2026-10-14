"""CLI and source-config defaults shared by every entry point."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from omniretargeting.data_sources.base import MotionData
from omniretargeting.data_sources.registry import create_data_source
from omniretargeting.source_config import absolute_source_paths, load_source_config

ROOT = Path(__file__).resolve().parents[1]


class Stop(Exception):
    pass


def _run_cli(monkeypatch, tmp_path, argv, profile=None):
    """Run ``omniretargeting.main`` up to the solve; return the captured inputs."""
    import omniretargeting.main as cli

    seen = {}
    source = MotionData(positions=np.zeros((3, 1, 3)), target_names=["Pelvis"], framerate=30)

    def create(kind, path, source_config=None, **kwargs):
        seen["source_config"] = dict(source_config)
        return SimpleNamespace(load=lambda: source)

    def owner_factory(**kwargs):
        def retarget_motion(*args, **kw):
            seen["sequence_solver"] = dict(owner.retargeting_config.get("sequence_solver") or {})
            raise Stop
        owner = SimpleNamespace(retargeting_config=kwargs["retargeting"], retarget_motion=retarget_motion,
                                robot_model=None)
        return owner

    monkeypatch.setattr(cli, "create_data_source", create)
    monkeypatch.setattr(cli, "OmniRetargeter", owner_factory)
    monkeypatch.setattr(cli, "select_mapped_source_base_pose", lambda *args, **kwargs: None)
    if profile is not None:
        path = tmp_path / "robot.json"
        path.write_text("{}")
        monkeypatch.setattr(cli, "load_robot_config", lambda _: profile)
        argv = [*argv, "--robot-config", str(path)]
    motion = tmp_path / "clip.npz"
    config = tmp_path / "source.yaml"
    config.write_text(yaml.safe_dump({"type": "parc_ms", "motion": str(motion), "auto_mapped_base_pose": False}))
    argv = [arg.replace("{config}", str(config)).replace("{motion}", str(motion)) for arg in argv]
    monkeypatch.setattr(sys, "argv", ["omniretargeting", *argv, "--output", str(tmp_path / "out.npz")])
    with pytest.raises(Stop):
        cli.main()
    return seen


def _profile(**retargeting):
    return {"urdf_path": "robot.xml", "joint_mapping": {"Pelvis": "root"}, "retargeting": retargeting}


def test_direct_motion_forwards_smplx_model_directory(monkeypatch, tmp_path):
    seen = _run_cli(monkeypatch, tmp_path, ["--source-motion", "{motion}", "--smplx-model-dir", "/models"],
                    profile=_profile())
    assert seen["source_config"]["type"] == "smplx"
    assert seen["source_config"]["model_directory"] == "/models"


def test_solver_defaults_to_fast_unless_profile_or_cli_choose(monkeypatch, tmp_path):
    assert _run_cli(monkeypatch, tmp_path, ["--source-config", "{config}"],
                    profile=_profile())["sequence_solver"] == {"enabled": True, "backend": "fast"}
    chosen = _profile(sequence_solver={"enabled": False})
    assert _run_cli(monkeypatch, tmp_path, ["--source-config", "{config}"],
                    profile=chosen)["sequence_solver"] == {"enabled": False}
    assert _run_cli(monkeypatch, tmp_path, ["--source-config", "{config}", "--solver", "framewise"],
                    profile=_profile())["sequence_solver"] == {"enabled": False}


def test_default_profile_is_the_mesh_collision_contract_from_any_cwd(monkeypatch, tmp_path):
    import omniretargeting.main as cli

    loaded = []
    real = cli.load_robot_config
    monkeypatch.setattr(cli, "load_robot_config", lambda path: loaded.append(Path(path)) or real(path))
    monkeypatch.chdir(tmp_path)
    _run_cli(monkeypatch, tmp_path, ["--source-config", "{config}"])
    assert loaded == [ROOT / "robot_models/unitree_g1/unitree_g1_mesh.json"]


def test_config_paths_resolve_against_the_config_file(tmp_path):
    config = tmp_path / "configs" / "source.yaml"
    config.parent.mkdir()
    config.write_text("type: smplx\nmotion: clip.npz\nmodel_directory: ../models\nterrain: scene.obj\n")
    loaded = load_source_config(config)
    assert loaded["motion"] == str(tmp_path / "configs/clip.npz")
    assert loaded["model_directory"] == str(tmp_path / "models")
    assert loaded["terrain"] == str(tmp_path / "configs/scene.obj")


def test_generated_configs_keep_their_inputs_when_saved_elsewhere(tmp_path, monkeypatch):
    from omniretargeting.batch import write_source_config

    monkeypatch.chdir(tmp_path)
    config = absolute_source_paths({"type": "smplx", "motion": "clip.npz", "model_directory": "/abs", "gender": "x"})
    assert config == {"type": "smplx", "motion": str(tmp_path / "clip.npz"), "model_directory": "/abs", "gender": "x"}
    written = write_source_config(Path("motions/clip.npz"), "smplx", tmp_path / "out",
                                  extra_options={"model_directory": "assets/body_models"})
    reloaded = load_source_config(written)
    assert reloaded["motion"] == str(tmp_path / "motions/clip.npz")
    assert reloaded["model_directory"] == str(tmp_path / "assets/body_models")


def test_omomo_data_root_defaults_to_the_dataset_containing_the_sequence(tmp_path):
    from tests.test_omomo_adapter import _write_sequence

    written, data_root, _, _ = _write_sequence(tmp_path)
    sequence = written.rename(data_root / "data" / "test_seq.p")
    source = create_data_source("omomo", sequence, source_config={"type": "omomo", "motion": str(sequence)})
    assert source.data_root == data_root
    elsewhere = sequence.rename(tmp_path / "elsewhere.p")
    explicit = create_data_source("omomo", elsewhere, source_config={"data_root": str(data_root)})
    assert explicit.data_root == data_root


def test_templates_only_name_keys_their_adapter_reads():
    for template in (ROOT / "config_templates").glob("*.yaml"):
        text = template.read_text()
        assert "unitree_g1.json" not in text and "/localhdd" not in text, template.name
        if not template.name.startswith("omomo"):
            assert "framerate:" not in text, template.name
    assert json.loads((ROOT / "robot_models/unitree_g1/unitree_g1_mesh.json").read_text())["urdf_path"]
