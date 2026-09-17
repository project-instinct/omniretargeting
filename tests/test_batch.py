"""Tests for the batch entry point's command construction."""

import sys
from pathlib import Path
from types import SimpleNamespace

from omniretargeting.utils import batch_processing
from omniretargeting.utils.batch_processing import (
    _build_command,
    _output_exists,
    export_shared_scaled_terrain,
)


def test_build_command_uniform_scale_skips_per_motion_scene_export(tmp_path):
    cmd = _build_command(
        tmp_path / "c.yaml",
        "robot.json",
        tmp_path,
        "stem",
        save_video=False,
        scale_factor=0.85,
    )

    assert "--scale-factor" in cmd
    assert cmd[cmd.index("--scale-factor") + 1] == "0.85"
    assert "--enable-scene-scaling" not in cmd
    assert "--scaled-objects" not in cmd


def test_build_command_default_disables_scene_scaling(tmp_path):
    cmd = _build_command(
        tmp_path / "c.yaml",
        "robot.json",
        tmp_path,
        "stem",
        save_video=False,
    )

    assert "--enable-scene-scaling" not in cmd
    assert "--scale-factor" not in cmd
    assert "--scaled-objects" not in cmd

    output_path = Path(cmd[cmd.index("--output") + 1])
    assert output_path == tmp_path / "motions" / "stem_retargeted.npz"
    assert not (tmp_path / "motions" / "stem").exists()


def test_build_command_places_video_beside_retargeted_motion(tmp_path):
    cmd = _build_command(
        tmp_path / "c.yaml",
        "robot.json",
        tmp_path,
        "stem",
        save_video=True,
    )

    video_path = Path(cmd[cmd.index("--save-video") + 1])
    assert video_path == tmp_path / "motions" / "stem_retargeted.mp4"


def test_output_exists_uses_flat_motions_directory(tmp_path):
    motion_file = tmp_path / "source" / "stem.npz"
    output_path = tmp_path / "output" / "motions" / "stem_retargeted.npz"
    output_path.parent.mkdir(parents=True)
    output_path.touch()

    assert _output_exists(motion_file, tmp_path / "output")


def test_export_shared_scaled_terrain_uses_terrain_subdirectory(
    monkeypatch, tmp_path
):
    exports = []

    class FakeMesh:
        def apply_scale(self, scale_factor):
            self.scale_factor = scale_factor

        def export(self, path):
            exports.append(path)

    mesh = FakeMesh()
    load_calls = []

    def fake_load(path, force):
        load_calls.append((path, force))
        return mesh

    monkeypatch.setitem(sys.modules, "trimesh", SimpleNamespace(load=fake_load))

    result = export_shared_scaled_terrain("source_terrain.obj", 0.85, tmp_path)

    expected = tmp_path / "terrain" / "scaled_terrain.obj"
    assert load_calls == [("source_terrain.obj", "mesh")]
    assert mesh.scale_factor == 0.85
    assert exports == [expected]
    assert result == expected
    assert expected.parent.is_dir()


def test_process_batch_sequential_progress_counts_each_job_once(
    monkeypatch, capsys, tmp_path
):
    motion_files = [tmp_path / f"motion_{i}.npz" for i in range(3)]

    def successful_result(motion_file):
        return {
            "returncode": 0,
            "elapsed": 0.1,
            "peak_memory_mb": 0,
            "motion_file": str(motion_file),
            "motion_stem": motion_file.stem,
            "log_file": "",
        }

    def fake_test_job(files, *args, **kwargs):
        return successful_result(files[0])

    def fake_run_one_motion(motion_file, *args, **kwargs):
        return successful_result(motion_file)

    monkeypatch.setattr(batch_processing, "_run_test_job", fake_test_job)
    monkeypatch.setattr(batch_processing, "_run_one_motion", fake_run_one_motion)

    results = batch_processing.process_batch(
        motion_files,
        source_type="smplx",
        robot_config_path="robot.json",
        output_dir=tmp_path,
        activation_prefix="",
        max_workers=1,
        save_video=False,
    )

    output = capsys.readouterr().out
    assert len(results) == 3
    assert "[2/3] Processing: motion_1.npz" in output
    assert "[3/3] Processing: motion_2.npz" in output
    assert "[4/3]" not in output
