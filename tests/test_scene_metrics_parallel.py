"""Parallel scene audit preserves the serial acceptance outputs."""

import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def _write_case(directory):
    model_path = directory / "robot.xml"
    model_path.write_text(
        """<mujoco><compiler angle=\"radian\"/><worldbody>
          <body name=\"robot\"><freejoint/>
            <geom name=\"shell\" type=\"sphere\" size=\".08\"/>
          </body>
        </worldbody></mujoco>"""
    )
    static_scene = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
    static_scene.export(directory / "static_scene.obj")
    np.save(
        directory / "prepared.npy",
        {"model_path": str(model_path)},
        allow_pickle=True,
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    poses = np.tile(model.qpos0, (5, 1))
    poses[:, 2] = np.linspace(-0.02, 0.02, len(poses))
    np.savez(
        directory / "motion.npz",
        qpos=poses,
        frame_start=np.asarray(1, dtype=int),
    )

    object_mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    translations = np.zeros((6, 3))
    translations[:, 0] = np.linspace(0.03, -0.03, len(translations))
    objects = [
        {
            "mesh": object_mesh,
            "translations": translations,
            "rotations": np.tile(np.eye(3), (len(translations), 1, 1)),
        }
    ]
    np.save(directory / "objects.npy", objects, allow_pickle=True)


def _run_audit(directory, workers):
    output = directory / f"scene_metrics_w{workers}.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/review_scene_metrics.py"),
            "--prepared",
            str(directory / "prepared.npy"),
            "--motion",
            str(directory / "motion.npz"),
            "--objects",
            str(directory / "objects.npy"),
            "--output",
            str(output),
            "--batch",
            "2",
            "--workers",
            str(workers),
        ],
        check=True,
        timeout=25,
        cwd=ROOT,
    )
    return output


def test_scene_metrics_parallel_output_matches_serial(tmp_path):
    _write_case(tmp_path)
    serial = _run_audit(tmp_path, 1)
    parallel = _run_audit(tmp_path, 2)

    assert json.loads(parallel.read_text()) == json.loads(serial.read_text())
    with np.load(serial.with_suffix(".npz")) as left, np.load(
        parallel.with_suffix(".npz")
    ) as right:
        assert set(left.files) == set(right.files)
        for name in left.files:
            assert np.array_equal(left[name], right[name], equal_nan=True)

    report = json.loads(parallel.read_text())
    assert report["static"]["max_m"] > 0
    assert report["moving"]["max_m"] > 0


def _write_open_scan_case(directory):
    """An evidence-free open panel: the static audit takes the open-scan path."""
    _write_case(directory)
    panel = trimesh.Trimesh(
        vertices=[[-0.3, -0.3, 0.0], [0.3, -0.3, 0.0], [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    panel.export(directory / "static_scene.obj")


def _run_script(script, directory, name, *extra):
    output = directory / name
    result = subprocess.run(
        [sys.executable, str(script), "--prepared", str(directory / "prepared.npy"),
         "--motion", str(directory / "motion.npz"), "--output", str(output), "--batch", "2", *extra],
        timeout=25, cwd=ROOT, capture_output=True, text=True,
    )
    error = None if result.returncode == 0 else result.stderr.strip().splitlines()[-1]
    return (json.loads(output.read_text()) if result.returncode == 0 else None), error


def test_open_scan_audit_matches_head_script(tmp_path):
    _write_open_scan_case(tmp_path)
    head = tmp_path / "review_scene_metrics_head.py"
    source = subprocess.run(["git", "show", "HEAD:scripts/review_scene_metrics.py"],
                            cwd=ROOT, capture_output=True, text=True)
    if source.returncode:
        import pytest
        pytest.skip("HEAD script unavailable")
    head.write_text(source.stdout)
    expected = _run_script(head, tmp_path, "head.json")
    assert expected == _run_script(ROOT / "scripts/review_scene_metrics.py", tmp_path, "w1.json")
    assert expected == _run_script(ROOT / "scripts/review_scene_metrics.py", tmp_path, "w2.json",
                                   "--workers", "2")
    if expected[0] is not None:
        assert "open_scan_backend" in expected[0]["static"]
