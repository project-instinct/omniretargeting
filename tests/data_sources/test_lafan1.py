"""Tests for the LAFAN1 BVH data-source adapter."""

import os
import tempfile

import numpy as np

from omniretargeting.data_sources.lafan1 import Lafan1DataSource, _read_bvh
from omniretargeting.data_sources.registry import create_data_source


MINIMAL_BVH = """HIERARCHY
ROOT Hips
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.0 10.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Head
        {
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {
                OFFSET 0.0 5.0 0.0
            }
        }
    }
    JOINT LeftFoot
    {
        OFFSET -5.0 -50.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
    }
    JOINT RightFoot
    {
        OFFSET 5.0 -50.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
    }
}
MOTION
Frames: 3
Frame Time: 0.0333333
0.0 90.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 90.0 10.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 90.0 20.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
"""


def _write_temp_bvh(content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bvh", delete=False) as f:
        f.write(content)
    return f.name


def test_read_bvh_bone_names():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        bvh = _read_bvh(path)
        assert bvh["names"] == ["Hips", "Spine", "Head", "LeftFoot", "RightFoot"]
    finally:
        os.unlink(path)


def test_read_bvh_parents():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        bvh = _read_bvh(path)
        assert bvh["parents"][0] == -1
        assert bvh["parents"][1] == 0
        assert bvh["parents"][2] == 1
        assert bvh["parents"][3] == 0
        assert bvh["parents"][4] == 0
    finally:
        os.unlink(path)


def test_read_bvh_frametime():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        bvh = _read_bvh(path)
        assert abs(bvh["frametime"] - 0.0333333) < 1e-6
    finally:
        os.unlink(path)


def test_read_bvh_quats_shape():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        bvh = _read_bvh(path)
        assert bvh["quats"].shape == (3, 5, 4)
        norms = np.linalg.norm(bvh["quats"], axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5)
    finally:
        os.unlink(path)


def test_load_positions_shape():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        ds = Lafan1DataSource(motion_file=path)
        md = ds.load()
        assert md.positions.shape == (3, 5, 3)
        assert np.isfinite(md.positions).all()
    finally:
        os.unlink(path)


def test_load_target_names():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        ds = Lafan1DataSource(motion_file=path)
        md = ds.load()
        assert md.target_names == ["Hips", "Spine", "Head", "LeftFoot", "RightFoot"]
    finally:
        os.unlink(path)


def test_load_framerate():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        ds = Lafan1DataSource(motion_file=path)
        md = ds.load()
        assert abs(md.framerate - 30.0) < 0.1
    finally:
        os.unlink(path)


def test_load_root_orientations_shape():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        ds = Lafan1DataSource(motion_file=path)
        md = ds.load()
        assert md.root_orientations.shape == (3, 3)
        assert np.isfinite(md.root_orientations).all()
    finally:
        os.unlink(path)


def test_load_root_translations_shape():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        ds = Lafan1DataSource(motion_file=path)
        md = ds.load()
        assert md.root_translations.shape == (3, 3)
        assert np.isfinite(md.root_translations).all()
    finally:
        os.unlink(path)


def test_load_positions_are_z_up():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        ds = Lafan1DataSource(motion_file=path)
        md = ds.load()
        spine_z = md.positions[0, 1, 2]
        hips_z = md.positions[0, 0, 2]
        assert spine_z > hips_z
    finally:
        os.unlink(path)


def test_load_positions_are_meters():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        ds = Lafan1DataSource(motion_file=path)
        md = ds.load()
        assert md.positions[0, 0, 2] < 2.0
    finally:
        os.unlink(path)


def test_load_height_estimated():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        ds = Lafan1DataSource(motion_file=path)
        md = ds.load()
        assert md.source_height is not None
        assert 1.0 < md.source_height < 2.5
    finally:
        os.unlink(path)


def test_iter_frames():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        ds = Lafan1DataSource(motion_file=path)
        frames = list(ds.iter_frames())
        assert len(frames) == 3
        for f in frames:
            assert f.positions.shape == (5, 3)
    finally:
        os.unlink(path)


def test_registry_integration():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        ds = create_data_source("lafan1", path)
        md = ds.load()
        assert md.positions.shape == (3, 5, 3)
    finally:
        os.unlink(path)


def test_metadata_includes_bone_info():
    path = _write_temp_bvh(MINIMAL_BVH)
    try:
        ds = Lafan1DataSource(motion_file=path)
        md = ds.load()
        assert md.metadata["source_type"] == "lafan1"
        assert md.metadata["bone_names"] == ["Hips", "Spine", "Head", "LeftFoot", "RightFoot"]
        assert len(md.metadata["bone_parents"]) == 5
    finally:
        os.unlink(path)
