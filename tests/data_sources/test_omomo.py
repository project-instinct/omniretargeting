"""Tests for the OMOMO data-source adapter."""

import numpy as np

from omniretargeting.data_sources import omomo


def test_home_relative_paths_are_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    sequence_file = tmp_path / "Datasets" / "OMOMO" / "data" / "sequence.p"
    data_root = tmp_path / "Datasets" / "OMOMO"
    model_directory = tmp_path / "Datasets" / "smplx"
    sequence_file.parent.mkdir(parents=True)
    sequence_file.touch()
    object_mesh_path = (
        data_root
        / "data"
        / "captured_objects"
        / "floorlamp_cleaned_simplified.obj"
    )
    object_mesh_path.parent.mkdir(parents=True)
    object_mesh_path.touch()
    model_directory.mkdir(parents=True)

    monkeypatch.setattr(
        omomo.joblib,
        "load",
        lambda path: [{"seq_name": "sub17_floorlamp_023"}],
    )
    monkeypatch.setattr(
        omomo.trimesh,
        "load",
        lambda path, force: type("Mesh", (), {"vertices": np.zeros((1, 3))})(),
    )

    source = omomo.OmomoDataSource(
        sequence_file="~/Datasets/OMOMO/data/sequence.p",
        sequence_index=0,
        data_root="~/Datasets/OMOMO",
        model_directory="~/Datasets/smplx",
    )

    assert source.sequence_file == sequence_file
    assert source.data_root == data_root
    assert source.model_directory == str(model_directory)
    assert source._resolve_model_directory() == str(model_directory)
