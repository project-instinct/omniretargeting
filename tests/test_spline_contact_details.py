"""Framewise contact diagnostics on a small real MuJoCo model."""

from copy import deepcopy

import mujoco
import numpy as np
import pytest

from omniretargeting.contact import SceneContactTask
from scripts.spline_contact_details import measure


START, FRAMES, FPS = 4, 7, 10.0
TOTAL = START + FRAMES


def fixture():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body name="pelvis" pos="0 0 1"><freejoint/>
        <geom name="pelvis_geom" type="sphere" size=".04"/>
        <body name="hand" pos=".1 0 0">
          <geom name="hand_geom" type="sphere" size=".02"/>
        </body>
        <body name="foot" pos="0 -.1 0">
          <geom name="foot_a" type="sphere" pos="-.03 0 0" size=".015"/>
          <geom name="foot_b" type="sphere" pos=".03 0 0" size=".015"/>
        </body>
      </body>
    </worldbody></mujoco>
    """)
    poses = np.tile(model.qpos0, (FRAMES, 1))
    poses[:, :3] = np.c_[np.arange(FRAMES) * .01, np.zeros(FRAMES),
                         np.ones(FRAMES)]
    pelvis = poses[:, :3]
    hand_body = pelvis + np.array([.1, 0., 0.])
    foot_body = pelvis + np.array([0., -.1, 0.])
    foot_points = np.stack((foot_body + [-.03, 0., 0.],
                            foot_body + [.03, 0., 0.]), axis=1)

    choices = np.zeros(TOTAL, int)
    choices[START + 4:] = 1
    endpoint_point = hand_body.copy()
    endpoint_point[4:] += np.array([0., .02, 0.])
    endpoint_error = np.array([
        [0., 0., 0.], [.01, 0., .02], [0., .02, 0.], [.03, 0., -.01],
        [0., 0., .01], [-.02, .01, 0.], [.01, 0., 0.]])
    endpoint_gap = np.array([.004, -.003, .002, .005, -.002, .003, -.001])

    def full(values, fill=np.nan):
        values = np.asarray(values, float)
        result = np.full((TOTAL,) + values.shape[1:], fill, float)
        result[START:START + FRAMES] = values
        return result

    normal = np.tile([0., 0., 1.], (TOTAL, 1))
    modes = np.full(TOTAL, "sticking", object)
    modes[START + 6] = "sliding"
    environment_ids = np.zeros(TOTAL, int)
    environment_ids[START + 3:] = 1
    managed = np.zeros(TOTAL, bool)
    managed[START + 2] = True
    task = SceneContactTask(
        name="hand",
        tracking_positions=full(hand_body, 0.),
        contact_positions=full(endpoint_point + endpoint_error),
        surface_points=full(endpoint_point
                            - endpoint_gap[:, None] * normal[START:START+FRAMES]),
        surface_normals=normal,
        distances=np.zeros(TOTAL),
        relative_speeds=np.zeros(TOTAL),
        contact_confidence=np.ones(TOTAL),
        contact_mask=np.ones(TOTAL, bool),
        forward_directions=np.tile([1., 0., 0.], (TOTAL, 1)),
        up_directions=normal,
        contact_modes=modes,
        environment_ids=environment_ids,
        surface_managed=True,
        surface_managed_mask=managed,
    )

    surface_vector = np.array([
        [.01, .02], [.01, .03], [.02, .01], [.03, .02],
        [.01, .04], [.02, .01], [.04, .03]])
    surface_gap = np.array([
        [.003, -.004], [-.005, .002], [.002, -.003], [.004, .001],
        [-.001, .004], [.003, -.002], [-.002, .003]])
    surface_ids = np.zeros((TOTAL, 3), int)
    surface_ids[START + 3:, 1:] = 1
    surface_modes = np.full((TOTAL, 3), "sticking", object)
    inactive_target_column = np.zeros((FRAMES, 1, 3))
    inactive_environment_column = np.zeros((FRAMES, 1, 3))
    inactive_source_column = np.full((FRAMES, 1, 3), [.01, -.02, .003])
    environment_values = np.concatenate((
        inactive_environment_column, foot_points
        - surface_gap[:, :, None] * normal[START:START+FRAMES, None]), axis=1)
    source_values = np.concatenate((
        inactive_source_column, foot_points
        + np.array([.002, -.001, .004])), axis=1)
    surface_mask = np.zeros((TOTAL, 3), bool)
    surface_mask[:, 1:] = True
    inputs = {
        "roots": full(pelvis + [0., 0., 0.], 0.) + np.array([.002, -.001, .003]),
        "tasks": {"hand": task},
        "effectors": {
            "source_root_robot_body": "pelvis",
            "effectors": {"hand": {
                "robot_body": "hand",
                "tracking_offset": [0., 0., 0.],
                "contact_offsets": [[0., 0., 0.], [0., .02, 0.]],
            }},
        },
        "surface": {
            "surface_contact_parts": np.array(["heel", "foot", "foot"], object),
            "surface_contact_mask": surface_mask,
            "surface_contact_robot_geoms": np.array([
                model.geom("foot_a").id, model.geom("foot_a").id,
                model.geom("foot_b").id]),
            "surface_contact_robot_local_points": np.zeros((3, 3)),
            "surface_contact_targets_w": full(
                np.concatenate((inactive_target_column, foot_points
                                + surface_vector[:, :, None]
                                * normal[START:START+FRAMES, None]), axis=1)),
            "surface_contact_normals_w": np.tile(normal[:, None], (1, 3, 1)),
            "surface_contact_environment_w": full(environment_values),
            "surface_contact_source_vectors_w": full(source_values - environment_values),
            "surface_contact_modes": surface_modes,
            "surface_contact_surface_ids": surface_ids,
        },
    }
    return model, poses, {"hand": choices}, inputs


def test_real_fk_contact_frames_managed_slip_and_surface_bindings(monkeypatch):
    model, poses, choices, inputs = fixture()
    calls = []
    original = mujoco.mj_forward

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mujoco, "mj_forward", counted)
    report = measure(inputs, model, poses, choices, fps=FPS, start=START)
    assert calls == [1] * len(poses)
    assert report["frames"] == len(poses) and report["start"] == START
    assert report["framerate"] == FPS

    endpoint_vector = report["metrics"]["endpoint_contact_vector_error_m"]["hand"]
    endpoint_normal = report["metrics"]["endpoint_contact_normal_abs_error_m"]["hand"]
    endpoint_gap = report["metrics"]["endpoint_signed_gap_to_observed_plane_m"]["hand"]
    endpoint_slip = report["metrics"]["endpoint_sticking_target_relative_slip_m_s"]["hand"]
    for values in (endpoint_vector, endpoint_normal, endpoint_gap, endpoint_slip):
        assert values.shape == (len(poses),)
    expected_vector = np.linalg.norm([
        [0., 0., 0.], [.01, 0., .02], [np.nan, np.nan, np.nan],
        [.03, 0., -.01], [0., 0., .01], [-.02, .01, 0.], [.01, 0., 0.]], axis=1)
    np.testing.assert_allclose(endpoint_vector, expected_vector, atol=1e-14)
    np.testing.assert_allclose(endpoint_normal,
                               np.array([0., .02, np.nan, .01, .01, 0., 0.]),
                               atol=1e-14)
    np.testing.assert_allclose(endpoint_gap,
                               np.array([.004, -.003, np.nan, .005, -.002, .003, -.001]),
                               atol=1e-14)
    assert np.isnan(endpoint_vector[2]) and np.isnan(endpoint_normal[2])
    assert np.isnan(endpoint_gap[2])
    np.testing.assert_allclose(endpoint_slip[1], np.linalg.norm([.01, .02]) * FPS)
    assert np.isnan(endpoint_slip[0])
    assert np.isnan(endpoint_slip[2]) and np.isnan(endpoint_slip[3])
    np.testing.assert_allclose(endpoint_slip[4], np.linalg.norm([-.03, 0., .02]) * FPS)
    np.testing.assert_allclose(endpoint_slip[5], np.linalg.norm([-.02, .01, -.01]) * FPS)
    assert np.isnan(endpoint_slip[6])

    surface_vector = report["metrics"]["surface_contact_vector_error_m"]["foot"]
    surface_normal = report["metrics"]["surface_contact_normal_abs_error_m"]["foot"]
    surface_gap = report["metrics"]["surface_signed_gap_to_observed_plane_m"]["foot"]
    surface_slip = report["metrics"]["surface_sticking_target_relative_slip_m_s"]["foot"]
    np.testing.assert_allclose(surface_vector, np.max([
        [.01, .02], [.01, .03], [.02, .01], [.03, .02],
        [.01, .04], [.02, .01], [.04, .03]], axis=1), atol=1e-14)
    np.testing.assert_array_equal(surface_normal, surface_vector)
    np.testing.assert_allclose(surface_gap, [
        -.004, -.005, -.003, .004, .004, .003, .003], atol=1e-14)
    np.testing.assert_array_equal(
        report["binding_selection"]["surface_contact_vector_error_m"]["foot"],
        [2, 2, 1, 1, 2, 1, 1])
    np.testing.assert_array_equal(
        report["binding_selection"]["surface_signed_gap_to_observed_plane_m"]["foot"],
        [2, 1, 2, 1, 2, 1, 2])
    np.testing.assert_allclose(surface_slip,
                               [np.nan, .1, .2, np.nan, .2, .3, .2], atol=1e-14)

    root = report["metrics"]["source_root_anatomical_correspondence_error_m"]["pelvis"]
    np.testing.assert_allclose(root, np.linalg.norm([.002, -.001, .003]), atol=1e-14)
    assert any("not an original-geom collision certificate" in value
               for value in report["provenance"].values())
    assert "managed endpoint frames are excluded" in report["limitations"][1]
    assert report["binding_selection_index"] == (
        "global index into the prepared surface_contact_* arrays")
    assert report["binding_parts"]["foot"] == [1, 2]


def test_surface_slip_ignores_inactive_nan_binding_in_the_same_part():
    model, poses, choices, inputs = fixture()
    inputs = deepcopy(inputs)
    # Global bindings are heel=0, foot=1, foot=2.  Keep only foot binding 1
    # active; an inactive NaN in binding 2 must not erase the valid slip.
    inputs["surface"]["surface_contact_mask"][:, 2] = False
    report = measure(inputs, model, poses, choices, fps=FPS, start=START)
    slip = report["metrics"]["surface_sticking_target_relative_slip_m_s"]["foot"]
    np.testing.assert_allclose(
        slip, [np.nan, 0., .1, np.nan, .2, .1, .2], atol=1e-14)
    assert np.isnan(report["metrics"]["surface_contact_vector_error_m"]["heel"]).all()
    assert report["binding_selection"]["surface_contact_vector_error_m"]["foot"].tolist() == [
        1, 1, 1, 1, 1, 1, 1]


def test_binding_signals_use_fixed_full_frame_fk_without_mixing_bindings(monkeypatch):
    model, poses, choices, inputs = fixture()
    inputs = deepcopy(inputs)
    # Keep endpoint switching from the real fixture and make one foot binding
    # inactive for two frames.  Every signal must still carry continuous FK.
    inputs["surface"]["surface_contact_mask"][START:START + 2, 1] = False
    calls = []
    original = mujoco.mj_forward

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mujoco, "mj_forward", counted)
    report = measure(inputs, model, poses, choices, fps=FPS, start=START)
    assert calls == [1] * len(poses)

    signals = report["binding_signals"]
    assert set(signals) == {"endpoint", "surface"}
    assert len(signals["endpoint"]) == 2
    assert len(signals["surface"]) == 3
    for signal in signals["endpoint"] + signals["surface"]:
        assert signal["world_positions"].shape == (len(poses), 3)
        assert signal["active"].shape == (len(poses),)
        assert signal["active"].dtype == bool
        assert np.isfinite(signal["world_positions"]).all()
        assert signal["units"] == "m"
        assert "fixed prepared" in signal["provenance"]
        assert signal["source_provenance"]

    for signal in signals["endpoint"]:
        assert signal["source_world_positions"] is None
        assert "unverified" in signal["source_provenance"]
        assert "target is not substituted" in signal["source_provenance"]

    endpoint0, endpoint1 = signals["endpoint"]
    assert endpoint0["id"] == "endpoint:hand:offset_0"
    assert endpoint1["id"] == "endpoint:hand:offset_1"
    assert endpoint0["active"].tolist() == [True, True, False, True, False, False, False]
    assert endpoint1["active"].tolist() == [False, False, False, False, True, True, True]
    hand_body = poses[:, :3] + np.array([.1, 0., 0.])
    np.testing.assert_allclose(endpoint0["world_positions"], hand_body, atol=1e-14)
    np.testing.assert_allclose(
        endpoint1["world_positions"], hand_body + np.array([0., .02, 0.]), atol=1e-14)
    assert not np.array_equal(endpoint0["world_positions"], endpoint1["world_positions"])

    heel, foot_a, foot_b = signals["surface"]
    foot_body = poses[:, :3] + np.array([0., -.1, 0.])
    assert heel["active"].tolist() == [False] * len(poses)
    assert foot_a["active"].tolist() == [False, False] + [True] * (len(poses) - 2)
    assert foot_b["active"].tolist() == [True] * len(poses)
    np.testing.assert_allclose(
        heel["world_positions"], foot_body + np.array([-.03, 0., 0.]), atol=1e-14)
    np.testing.assert_allclose(
        foot_a["world_positions"], heel["world_positions"], atol=0)
    np.testing.assert_allclose(
        foot_b["world_positions"], foot_body + np.array([.03, 0., 0.]), atol=1e-14)
    assert foot_a["part"] == foot_b["part"] == "foot"
    assert heel["part"] == "heel"
    assert foot_a["binding_index"] == 1 and foot_b["binding_index"] == 2
    assert not np.array_equal(foot_a["world_positions"], foot_b["world_positions"])
    environment = inputs["surface"]["surface_contact_environment_w"][START:START + FRAMES]
    source_vectors = inputs["surface"]["surface_contact_source_vectors_w"][START:START + FRAMES]
    expected_source = environment + source_vectors
    for index, signal in enumerate(signals["surface"]):
        source = signal["source_world_positions"]
        assert source.shape == (len(poses), 3)
        assert np.isfinite(source).all()
        np.testing.assert_allclose(source, expected_source[:, index], atol=1e-15)
        assert not np.array_equal(source, signal["world_positions"])
        assert "surface_contact_environment_w + surface_contact_source_vectors_w" in (
            signal["source_provenance"])
    assert np.isfinite(heel["source_world_positions"]).all()
    assert np.any(np.linalg.norm(heel["source_world_positions"], axis=1) > 0)


def test_invalid_time_and_pose_inputs_are_rejected():
    model, poses, choices, inputs = fixture()
    with pytest.raises(ValueError, match="fps"):
        measure(inputs, model, poses, choices, fps=0, start=START)
    with pytest.raises(ValueError, match="poses"):
        measure(inputs, model, poses[:, :-1], choices, fps=FPS, start=START)
    with pytest.raises(ValueError, match="choice"):
        measure(inputs, model, poses, {"hand": np.zeros(2, int)}, fps=FPS, start=START)
