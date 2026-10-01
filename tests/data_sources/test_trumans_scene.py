"""Chair pose sources, frame alignment and immutable coordinate conversion."""

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.trumans_review import WORLD_ROTATION, load_scene


def release(tmp_path, *, segment="recording", with_recording=True):
    (tmp_path / "Scene_mesh").mkdir()
    (tmp_path / "Object_chairs/Object_mesh").mkdir(parents=True)
    mesh = trimesh.creation.box(extents=[0.4, 0.6, 0.3])
    mesh.export(tmp_path / "Scene_mesh/room.obj")
    name = "movable_chair_seat_01(1.00,1.00,1.00)"
    mesh.export(tmp_path / f"Object_chairs/Object_mesh/{name}.obj")
    aggregate = np.tile(np.eye(4), (3, 5, 1, 1))
    aggregate[:, 2, :3, 3] = [50, 60, 70]
    aggregate[1, 2] = np.nan
    values = {
        "seg_name": np.array([segment] * 3),
        "scene_flag": np.zeros(3, dtype=int),
        "scene_list": np.array(["room"]),
        "object_list": np.array([name]),
        "object_flag": np.full((3, 1), 2, dtype=int),
        "object_mat": aggregate,
        # The source slice is deliberately not the beginning of the recording.
        "frame_id": np.array([1, 3, 4]),
    }
    for key, value in values.items():
        np.save(tmp_path / f"{key}.npy", value)
    positions = np.arange(18, dtype=float).reshape(6, 3) / 10
    angles = np.arange(18, dtype=float).reshape(6, 3) / 17
    raw = {"movable_chair_seat_01": {"location": positions, "rotation": angles}}
    if with_recording:
        (tmp_path / "Object_all/Object_pose").mkdir(parents=True)
        np.save(tmp_path / f"Object_all/Object_pose/{segment}.npy", raw)
    return values, raw


def test_original_chair_recording_overrides_aggregate_without_filling(tmp_path):
    values, raw = release(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.npy")}
    indices, _, objects = load_scene(tmp_path, "recording")
    assert np.array_equal(indices, np.arange(3))
    assert len(objects) == 1
    obj = objects[0]
    frame_ids = values["frame_id"]
    original = raw["movable_chair_seat_01"]
    np.testing.assert_allclose(
        obj["translations"], original["location"][frame_ids] @ WORLD_ROTATION.T
    )
    np.testing.assert_allclose(
        obj["rotations"],
        WORLD_ROTATION
        @ Rotation.from_euler("xyz", original["rotation"][frame_ids]).as_matrix(),
    )
    assert obj["pose_source"].endswith("Object_pose/recording.npy")
    assert all(p.read_bytes() == payload for p, payload in before.items())


def test_augmentation_keeps_its_own_aggregate_frame(tmp_path):
    values, _ = release(tmp_path, with_recording=False, segment="recording_augment1")
    values["object_mat"][1, 2] = values["object_mat"][0, 2]
    np.save(tmp_path / "object_mat.npy", values["object_mat"])
    # An unaugmented pose file is present but must never be substituted.
    (tmp_path / "Object_all/Object_pose").mkdir(parents=True)
    np.save(tmp_path / "Object_all/Object_pose/recording.npy", {})
    _, _, objects = load_scene(tmp_path, "recording_augment1")
    np.testing.assert_allclose(
        objects[0]["translations"], values["object_mat"][:, 2, :3, 3] @ WORLD_ROTATION.T
    )
    assert objects[0]["pose_source"].endswith("object_mat.npy")


def test_missing_recording_reports_invalid_aggregate_frames(tmp_path):
    release(tmp_path, with_recording=False)
    with pytest.raises(ValueError, match=r"clip frames \[1\].*object_mat"):
        load_scene(tmp_path, "recording")


@pytest.mark.parametrize("failure", ["nonfinite", "out_of_range", "missing_object"])
def test_invalid_original_recording_is_not_silently_replaced(tmp_path, failure):
    _, raw = release(tmp_path)
    pose_path = tmp_path / "Object_all/Object_pose/recording.npy"
    if failure == "nonfinite":
        raw["movable_chair_seat_01"]["rotation"][3, 0] = np.nan
        np.save(pose_path, raw)
    elif failure == "out_of_range":
        np.save(tmp_path / "frame_id.npy", np.array([1, 3, 99]))
    else:
        np.save(pose_path, {})
    with pytest.raises(ValueError):
        load_scene(tmp_path, "recording")
