"""Registered TRUMANS source: rigid Z-up conversion, released joints, scene and objects."""

from pathlib import Path

import numpy as np
import pytest

from omniretargeting.data_sources.registry import create_data_source
from omniretargeting.data_sources.trumans_review import WORLD_ROTATION
from tests.data_sources.test_trumans_scene import release

MODELS = Path(__file__).resolve().parents[2] / "assets/body_models"
pytestmark = pytest.mark.skipif(not (MODELS / "smplx").exists(), reason="SMPL-X model files are not installed")


def _human(tmp_path, frames):
    """Recorded Y-up parameters and the joints the release stores for them."""
    import smplx
    import torch

    rng = np.random.default_rng(4)
    values = {
        "human_orient": rng.normal(scale=0.4, size=(frames, 3)),
        "human_pose": rng.normal(scale=0.15, size=(frames, 63)),
        "human_transl": rng.normal(scale=0.5, size=(frames, 3)),
        "left_hand_pose": rng.normal(scale=0.1, size=(frames, 45)),
        "right_hand_pose": rng.normal(scale=0.1, size=(frames, 45)),
    }
    model = smplx.create(str(MODELS), "smplx", gender="male", use_pca=False, batch_size=frames)
    with torch.no_grad():
        output = model(
            global_orient=torch.tensor(values["human_orient"], dtype=torch.float32),
            body_pose=torch.tensor(values["human_pose"], dtype=torch.float32),
            transl=torch.tensor(values["human_transl"], dtype=torch.float32),
            left_hand_pose=torch.tensor(values["left_hand_pose"], dtype=torch.float32),
            right_hand_pose=torch.tensor(values["right_hand_pose"], dtype=torch.float32),
        )
    values["human_joints"] = output.joints[:, :22].cpu().numpy()
    for key, value in values.items():
        np.save(tmp_path / f"{key}.npy", value)
    return values


def _load(tmp_path, **options):
    config = dict(type="trumans", motion=str(tmp_path), segment="recording", model_directory=str(MODELS), **options)
    return create_data_source("trumans", tmp_path, source_config=config).load()


def test_recorded_parameters_are_rotated_rigidly_into_the_released_joints(tmp_path):
    release(tmp_path)
    human = _human(tmp_path, 3)
    data = _load(tmp_path)
    recorded = human["human_joints"] @ WORLD_ROTATION.T
    np.testing.assert_array_equal(data.positions, recorded)
    # Forward kinematics of the converted root reproduces the released joints.
    assert data.metadata["reconstruction_error_m"] < 1e-5
    np.testing.assert_allclose(data.root_translations, recorded[:, 0])
    assert data.metadata["scene_enclosed_free_space"] is True
    assert len(data.metadata["scene_objects"]) == 1
    assert data.metadata["body_surface_samples"]["full_body"]["face_regions"] is not None


def test_frame_range_slices_the_human_and_every_object_together(tmp_path):
    release(tmp_path)
    human = _human(tmp_path, 3)
    whole = _load(tmp_path)
    part = _load(tmp_path, frame_start=1, frame_stop=3)
    np.testing.assert_array_equal(part.positions, human["human_joints"][1:3] @ WORLD_ROTATION.T)
    np.testing.assert_array_equal(part.metadata["source_rows"], [1, 2])
    for key in ("translations", "rotations"):
        np.testing.assert_array_equal(part.metadata["scene_objects"][0][key],
                                      whole.metadata["scene_objects"][0][key][1:3])


def test_a_segment_is_required(tmp_path):
    with pytest.raises(ValueError, match="segment"):
        create_data_source("trumans", tmp_path, source_config=dict(type="trumans", motion=str(tmp_path)))
