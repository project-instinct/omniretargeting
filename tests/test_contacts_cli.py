"""Contact selection through the main CLI with legacy OMOMO object samples."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest
import yaml

from omniretargeting import MotionData
import omniretargeting.main as cli


@pytest.fixture
def omomo_cli(monkeypatch, tmp_path):
    motion = MotionData(
        positions=np.tile([[[0, 0, 0.01], [0.2, 0, 0.01]]], (3, 1, 1)),
        target_names=["L_Foot", "R_Foot"],
        framerate=30,
        object_points=np.array([[[1, 0, 1]], [[1.2, 0, 1]], [[1.1, 0, 1]]]),
        metadata={
            "object_name": "box",
            "object_points_body_id": "box",
            "object_scales": np.array([1, 1.2, 1.1]),
        },
    )
    monkeypatch.setattr(
        cli, "create_data_source", lambda **kwargs: SimpleNamespace(load=lambda: motion)
    )
    captured = []

    def retarget(motion_data, **kwargs):
        captured.append(motion_data)
        result = np.zeros((len(motion_data.positions), 7))
        result[:, 3] = 1
        return 1.0, result

    monkeypatch.setattr(
        cli,
        "OmniRetargeter",
        lambda **kwargs: SimpleNamespace(
            retarget_motion=retarget, get_joint_names=lambda: []
        ),
    )
    source_config = tmp_path / "source.yaml"
    output = tmp_path / "motion.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "omniretargeting",
            "--source-config",
            str(source_config),
            "--output",
            str(output),
        ],
    )

    def run(pairs):
        detection = {"enabled": True}
        if pairs is not None:
            detection["pairs"] = pairs
        source_config.write_text(
            yaml.safe_dump(
                {
                    "type": "omomo",
                    "motion": "unused.p",
                    "object_scale_mode": "per_frame",
                    "contact_detection": detection,
                }
            )
        )
        cli.main()

    return run, captured, motion, tmp_path


@pytest.mark.parametrize("pairs", [{"terrain": ["L_Foot", "R_Foot"]}, None])
def test_cli_allows_terrain_detection_with_variable_object_scale(omomo_cli, pairs):
    run, captured, original, output_dir = omomo_cli
    run(pairs)
    detected = captured[0]
    assert set(detected.scene.entities) == {"terrain"}
    assert all(len(frame) == 2 for frame in detected.contact_trajectory)
    assert all(
        contact.scene_body_id == "terrain"
        for frame in detected.contact_trajectory
        for contact in frame
    )
    np.testing.assert_array_equal(detected.object_points, original.object_points)
    assert (output_dir / "motion_retargeted.contacts.json").exists()


def test_cli_rejects_requested_variable_scale_object_detection(omomo_cli):
    run, captured, original, output_dir = omomo_cli
    with pytest.raises(ValueError, match="object_scale_mode: first_frame"):
        run({"box": ["L_Foot"]})
    assert not captured
