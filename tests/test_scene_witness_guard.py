"""Acceptance regressions for fixed known scene material samples."""

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.scene_witness_guard import preserve_known_scene_samples
from omniretargeting.trajectory_validation import (
    TrajectoryValidator,
    local_repair_config,
)


def slide_sphere_model():
    """A real moving collision shell with one independently sliding joint."""
    return mujoco.MjModel.from_xml_string(
        """<mujoco><compiler angle="radian"/>
        <worldbody><body name="base"><freejoint/>
          <geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>
          <body name="slider"><joint name="x" type="slide" axis="1 0 0"
              range="-1 1"/><joint name="y" type="slide" axis="0 1 0"
              range="-1 1"/><geom name="ball" type="sphere" size=".05"
              mass=".1"/>
          </body>
        </body></worldbody></mujoco>"""
    )


def finite_evidence_wall():
    """An open wall whose center is observed but whose edge is ambiguous."""
    # Face winding gives the wall a +X normal.  Its Z half width is smaller than
    # the sphere, so one geom has both center samples with known side evidence
    # and edge samples with no unique open-surface side.  The wider Y span
    # leaves room for a tangential slide that exits the evidence ball while the
    # probe still considers the geom near enough to sample.
    vertices = [
        [0.0, -0.08, -0.03],
        [0.0, 0.08, -0.03],
        [0.0, 0.08, 0.03],
        [0.0, -0.08, 0.03],
    ]
    return trimesh.Trimesh(
        vertices=vertices,
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )


def validator_for(model, q, *, evidence_radius=0.15):
    query = SceneQuery(
        finite_evidence_wall(),
        evidence_points=[[0.0, 0.0, 0.0]],
        evidence_normals=[[1.0, 0.0, 0.0]],
        evidence_radius=evidence_radius,
        full_geometry=False,
    )
    validator = TrajectoryValidator(
        model,
        q,
        query,
        {},
        {},
        local_repair_config({}),
        30,
    )
    return validator


def one_pose(model, x, y=0.0):
    q = model.qpos0[None].copy()
    q[0, 7] = x
    q[0, 8] = y
    return q


def test_known_sample_leaving_evidence_is_rejected_even_when_geom_stays_unknown():
    model = slide_sphere_model()
    reference = one_pose(model, 0.05)
    validator = validator_for(model, reference, evidence_radius=0.05)
    before = validator.geometry(reference)
    candidate = one_pose(model, 0.05, 0.15)
    after = validator.geometry(candidate)

    # The finite wall edge already makes this geom globally uncertain in both
    # poses.  That aggregate flag must not hide the central known sample that
    # moved beyond the fixed evidence ball.
    assert before["uncertain_frames"] == [0]
    assert after["uncertain_frames"] == [0]
    np.testing.assert_array_equal(before["uncertain_geoms"], after["uncertain_geoms"])
    reasons = preserve_known_scene_samples(validator, candidate, before)
    assert reasons == ["known_scene_sample_became_uncertain"]


def test_known_sample_penetration_regression_is_rejected():
    model = slide_sphere_model()
    reference = one_pose(model, 0.05)  # The leftmost shell sample is tangent.
    validator = validator_for(model, reference)
    before = validator.geometry(reference)
    candidate = one_pose(model, 0.04)  # The same known sample enters the wall.

    reasons = preserve_known_scene_samples(validator, candidate, before)
    assert reasons == ["known_scene_sample_regression"]


def test_harmless_local_motion_is_allowed_while_global_unknown_stays_unresolved():
    model = slide_sphere_model()
    reference = one_pose(model, 0.06)
    validator = validator_for(model, reference)
    before = validator.geometry(reference)
    candidate = one_pose(model, 0.07)
    after = validator.geometry(candidate)

    assert before["uncertain_frames"] == [0]
    assert after["uncertain_frames"] == [0]
    assert preserve_known_scene_samples(validator, candidate, before) == []


def test_component_filter_preserves_closed_negative_with_unrelated_unknown_mesh():
    closed = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    unknown_wall = finite_evidence_wall()
    unknown_wall.apply_translation([10.0, 0.0, 0.0])
    query = SceneQuery(trimesh.util.concatenate([closed, unknown_wall]))
    closed_id = next(
        index for index, (_part, _faces, is_closed, _tree)
        in enumerate(query.components) if is_closed
    )

    all_components = query.query([[0.0, 0.0, 0.0]])
    closed_only = query.query([[0.0, 0.0, 0.0]], component_ids=[closed_id])

    assert all_components["uncertain"][0]
    assert all_components["clearance"][0] == pytest.approx(-0.1)
    assert not closed_only["uncertain"][0]
    assert closed_only["clearance"][0] == pytest.approx(-0.1)


def test_segment_rebuilds_witness_guard_for_segment_reference_timeline():
    model = slide_sphere_model()
    reference = np.vstack([
        one_pose(model, 0.05, 0.15),  # Parent frame: no known samples remain.
        one_pose(model, 0.05, 0.0),   # Segment frame: center samples are known.
        one_pose(model, 0.05, 0.0),
    ])
    validator = validator_for(model, reference, evidence_radius=0.05)
    parent_before = validator.geometry(reference)
    preserve_known_scene_samples(validator, reference.copy(), parent_before)
    assert validator._scene_witness_guard is not None
    assert validator._scene_witness_guard.v is validator

    with validator.segment(1, 2) as view:
        assert view._scene_witness_guard is None
        segment_before = view.geometry(view.reference)
        candidate = one_pose(model, 0.05, 0.15)
        reasons = preserve_known_scene_samples(view, candidate, segment_before)

        assert view._scene_witness_guard is not None
        assert view._scene_witness_guard.v is view
        assert reasons == ["known_scene_sample_became_uncertain"]
