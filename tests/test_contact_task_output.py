"""Consumer contract, whole-body evidence and export-only behavior."""

import json
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from omniretargeting import OmniRetargeter
from omniretargeting.contact_task_output import (
    ContactTaskOutput,
    build_contact_task_output,
    contact_task_output_config,
)
from omniretargeting.source_contacts import build_source_contacts
from omniretargeting.surface_contact import surface_contact_config
from tests._contact_fixtures import _task
from tests.test_source_contacts import floor


def table(points, labels, *, scene=None, objects=(), valid=None):
    points = np.asarray(points, float)
    source = {"points": points, "skin_normals": np.broadcast_to([0., 0., -1.], points.shape),
              "labels": labels}
    if valid is not None:
        source["valid"] = valid
    return build_source_contacts(source, floor() if scene is None else scene,
                                 surface_contact_config(), framerate=30,
                                 scene_objects=objects, scene_enclosed_free_space=False)


def slot(output, part, surface=0):
    return np.flatnonzero((output.body_parts == part) & (output.slot_scene_object_ids == surface))[0]


def evidence(frames, decisions, unexplained=()):
    return SimpleNamespace(frames=frames, diagnostics={"unexplained_frames": unexplained},
                           role=lambda part, surface, frame: decisions.get((part, surface, frame)))


def test_whole_body_candidates_are_present_even_when_inactive_or_unknown():
    labels = ["left_knee", "left_forearm", "pelvis", "abdomen", "arbitrary_region"]
    points = np.tile([[0., 0., .005]], (3, len(labels), 1))
    points[:, 3, 2] = .2
    points[:, 4, 2] = np.nan
    source = table(points, labels)
    output = build_contact_task_output(source, frames=3, framerate=30)
    for part in labels[:3]:
        column = slot(output, part)
        assert output.active[:, column].all()
        assert (output.contact_types[:, column] == "touch").all()
        assert (output.roles[:, column] == "hold").all()  # static geometry is not support
        np.testing.assert_allclose(output.target_positions_w[:, column, 2], 0, atol=1e-12)
        np.testing.assert_allclose(output.surface_normals_w[:, column], [[0, 0, 1]] * 3)
    known_free, unknown = slot(output, "abdomen"), slot(output, "arbitrary_region")
    assert output.observed[:, known_free].all() and not output.active[:, known_free].any()
    assert not output.observed[:, unknown].any()
    assert not output.observed[:, slot(output, "right_knee")].any()
    assert np.isnan(output.target_positions_w[:, known_free]).all()
    assert (output.scene_object_ids[:, known_free] == -1).all()
    assert output.importance_weights[:, known_free].sum() == 0
    assert output.confidence[:, known_free].sum() == 0
    np.testing.assert_allclose(output.timestamps_s, np.arange(3) / 30)


def test_support_and_sliding_are_independent_and_control_requires_evidence():
    points = np.array([[[x, 0, .005], [x, .2, .005]] for x in [0, .05, .1]])
    source = table(points, ["pelvis", "abdomen"])
    support = evidence(3, {("pelvis", 0, f): "support" for f in range(3)})
    control = evidence(3, {("abdomen", 0, 1): "control"})
    output = build_contact_task_output(source, frames=3, framerate=30,
                                       support_evidence=support, control_evidence=control)
    pelvis, abdomen = slot(output, "pelvis"), slot(output, "abdomen")
    assert (output.contact_modes[:, pelvis] == "sliding").all()
    assert (output.contact_types[:, pelvis] == "support").all()
    assert (output.importance_weights[:, pelvis] == 1).all()
    assert list(output.contact_types[:, abdomen]) == ["slide", "control", "slide"]
    assert list(output.importance_weights[:, abdomen]) == [.25, .5, .25]
    unresolved = build_contact_task_output(source, frames=3, framerate=30,
                                           support_evidence=support, control_evidence=control,
                                           unresolved_support_frames=[False, True, False])
    assert unresolved.roles[1, abdomen] == "hold"
    assert unresolved.roles[1, pelvis] == "support"
    assert unresolved.contact_modes[1, abdomen] == "sliding"


def test_simultaneous_contacts_and_moving_object_world_targets_are_retained():
    translations = np.array([[.6, 0, .5], [.65, 0, .5], [.7, 0, .5]])
    moving = {"name": "box", "mesh": trimesh.creation.box(extents=[.2, .2, .2]),
              "translations": translations}
    points = np.zeros((3, 2, 3))
    points[:, 0] = [0, 0, .005]
    points[:, 1] = translations + [0, 0, .105]
    source = table(points, ["left_forearm"] * 2, objects=[moving])
    output = build_contact_task_output(source, frames=3, framerate=30)
    static, box = slot(output, "left_forearm", 0), slot(output, "left_forearm", 1)
    assert output.scene_object_names.tolist() == ["scene", "box"]
    assert output.active[:, [static, box]].all()
    assert (output.scene_object_ids[:, box] == 1).all()
    assert (output.contact_modes[:, box] == "sticking").all()  # object-relative motion
    np.testing.assert_allclose(output.target_positions_w[:, box], translations + [0, 0, .1])
    assert len([row for row in output.frame(0) if row["body_part"] == "left_forearm" and row["active"]]) == 2


def test_surface_registry_has_no_static_scene_offset_when_scene_is_empty():
    empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), int))
    moving = {"name": "seat", "mesh": floor()}
    source = table(np.tile([[[0, 0, .005]]], (2, 1, 1)), ["pelvis"], scene=empty, objects=[moving])
    output = build_contact_task_output(source, frames=2, framerate=30)
    assert output.scene_object_names.tolist() == ["seat"]
    assert output.frame(0)[slot(output, "pelvis")]["scene_object"] == "seat"


def test_grouped_parts_choose_a_real_support_witness_before_higher_confidence_touch():
    source = table(np.tile([[[0, 0, .015], [.2, 0, .005]]], (2, 1, 1)),
                   ["lower_torso", "middle_torso"])
    support = evidence(2, {("lower_torso", 0, f): "support" for f in range(2)})
    output = build_contact_task_output(source, frames=2, framerate=30, support_evidence=support,
                                       config={"body_part_regions": {"abdomen": ["lower_torso", "middle_torso"]}})
    abdomen = slot(output, "abdomen")
    assert (output.roles[:, abdomen] == "support").all()
    assert (output.source_sample_indices[:, abdomen] == 0).all()
    np.testing.assert_allclose(output.target_positions_w[:, abdomen, 0], 0)
    np.testing.assert_array_equal(output.confidence[:, abdomen], source.observation["confidence"][:, 0])


def test_empty_evidence_and_endpoint_fallback_use_the_same_schema():
    empty = build_contact_task_output(frames=2, framerate=60)
    assert (empty.slot_scene_object_ids == -1).all()
    assert empty.scene_object_names.shape == (0,)
    assert not empty.active.any() and not empty.observed.any()
    json.dumps(empty.frame(0), allow_nan=False)
    task = _task(np.zeros((2, 3)), contact_mask=[True, False], contact_modes=["sliding", "none"])
    output = build_contact_task_output(frames=2, framerate=60, scene_object_names=["scene"],
                                       primary_tasks={"right_knee": task})
    knee = slot(output, "right_knee")
    assert output.contact_types[0, knee] == "slide"
    assert not output.active[1, knee] and output.observed[1, knee]
    assert (output.source_sample_indices[:, knee] == -1).all()
    json.dumps(output.frame(0), allow_nan=False)


def test_export_round_trip_is_pickle_free_and_preserves_exact_qpos_and_evidence(tmp_path):
    source = table(np.tile([[[0, 0, -.005]]], (2, 1, 1)), ["abdomen"])
    arrays = [source.points, source.valid, source.regions, source.active,
              *[value for value in source.observation.values() if isinstance(value, np.ndarray)]]
    original = [array.tobytes() for array in arrays]
    output = build_contact_task_output(source, frames=2, framerate=30)
    qpos = np.array([[0., -0., .4, 1, 0, 0, 0, .1234567890123]] * 2)
    baseline = qpos.tobytes()
    path = tmp_path / "motion.npz"
    np.savez(path, qpos=qpos, joint_pos=qpos[:, 7:], **output.export())
    with np.load(path, allow_pickle=False) as saved:
        assert saved["qpos"].tobytes() == baseline
        assert saved["contact_task_schema_version"].item() == 1
        restored = ContactTaskOutput(**{field.name: saved[f"contact_task_{field.name}"] for field in fields(output)})
        for key, value in output.export().items():
            np.testing.assert_array_equal(restored.export()[key], value)
    assert [array.tobytes() for array in arrays] == original
    assert qpos.tobytes() == baseline
    for value in output.export().values():
        assert not value.dtype.hasobject
    with pytest.raises(ValueError):
        output.active[:] = False
    # Penetrating human skin remains raw evidence; exported target is the surface.
    assert source.points[0, 0, 2] < 0
    assert output.target_positions_w[0, slot(output, "abdomen"), 2] >= -1e-12


def test_getter_detects_export_only_markers_without_enabling_solver_contacts():
    owner = OmniRetargeter.__new__(OmniRetargeter)
    owner._contact_output_frames = 2
    owner._source_framerate = 30
    owner.retargeting_config = {"end_effector_priority": {"enabled": False}}
    owner._contact_output_source_markers = None
    owner._contact_output_scene = floor()
    owner._scene_objects = ()
    owner._body_surface_samples = None
    owner.last_source_contact_table = None
    owner.last_end_effector_tasks = {}
    markers = {"right_knee": {"contact_samples": np.tile([[[0, 0, .005]]], (2, 1, 1))}}
    before = markers["right_knee"]["contact_samples"].tobytes()
    output = owner.get_contact_task_output(source_markers=markers)
    assert output.active[:, slot(output, "right_knee")].all()
    assert owner.last_source_contact_table is None and owner.last_end_effector_tasks == {}
    assert markers["right_knee"]["contact_samples"].tobytes() == before
    owner._contact_output_source_markers = {}  # next clip carries no evidence
    assert not owner.get_contact_task_output().active.any()


@pytest.mark.parametrize("config", [
    {"candidate_parts": "knee"}, {"body_part_regions": {"abdomen": "lower_torso"}},
    {"importance_weights": {"support": np.nan}}, {"importance_weights": {"follow": -1}},
    {"importance_weights": {"support": 2}}, {"importance_weights": {"typo": .5}},
    {"importance_weights": None}, {"unknown": True}, {"candidate_parts": [""]},
])
def test_bad_configuration_is_rejected(config):
    with pytest.raises((ValueError, TypeError)):
        contact_task_output_config(config)


def test_timeline_and_active_geometry_are_validated():
    source = table(np.tile([[[0, 0, .005]]], (2, 1, 1)), ["pelvis"])
    with pytest.raises(ValueError, match="timeline"):
        build_contact_task_output(source, frames=3, framerate=30)
    with pytest.raises(ValueError, match="framerate"):
        build_contact_task_output(source, frames=2, framerate=0)
    with pytest.raises(ValueError, match="timeline"):
        build_contact_task_output(source, frames=2, framerate=30, support_evidence=evidence(1, {}))
    output = build_contact_task_output(source, frames=2, framerate=30)
    with pytest.raises(ValueError, match="unit"):
        replace(output, surface_normals_w=np.zeros_like(output.surface_normals_w))
    with pytest.raises(ValueError, match="observed"):
        replace(output, observed=np.zeros_like(output.observed))
    with pytest.raises(IndexError):
        output.frame(-1)
