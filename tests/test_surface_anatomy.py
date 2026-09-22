"""Regression tests for semantic domains used by surface correspondence."""

import mujoco
import numpy as np

from omniretargeting.surface_anatomy import surface_domains
from omniretargeting.surface_learning import learn_indexed_points


def test_domain_restricted_learning_keeps_identical_limb_shapes_semantic():
    """A same-shaped forearm and upper arm must follow their domain labels.

    The two source and robot patches are deliberately exchanged in world
    position.  Pure geometry therefore prefers the wrong side/limb, while
    the kinematic domain is the only signal that identifies the intended
    assignment.
    """
    source = np.array(
        [[-1.0, 0.0, 0.0], [-1.0, 0.1, 0.0], [1.0, 0.0, 0.0], [1.0, 0.1, 0.0]]
    )
    # Domain 0 (forearm) is at +x on the robot; domain 1 (upper arm) is at -x.
    robot = np.array(
        [[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.1, 0.0]]
    )
    edges = np.array([[0, 1], [2, 3]])
    predicted, report = learn_indexed_points(
        source,
        robot,
        edges,
        {
            "iterations": 120,
            "learning_rate": 0.01,
            "edge_weight": 1e-6,
            "repulsion_weight": 1e-6,
            "cache_dir": None,
        },
        source_domains=np.array([0, 0, 1, 1]),
        target_domains=np.array([0, 0, 1, 1]),
    )

    np.testing.assert_allclose(predicted[:2, 0], 1.0, atol=0.02)
    np.testing.assert_allclose(predicted[2:, 0], -1.0, atol=0.02)
    assert report["chamfer"] < 1e-5


def test_surface_domains_reroot_at_torso_floating_base_and_keep_anatomy():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <body name="torso"><freejoint/>
            <geom name="torso_geom" type="box" size=".2 .2 .2"/>
            <body name="pelvis">
              <geom name="pelvis_geom" type="box" size=".1 .1 .1"/>
              <body name="left_upper"><geom name="left_geom" type="box" size=".05 .05 .05"/></body>
              <body name="right_upper"><geom name="right_geom" type="box" size=".05 .05 .05"/></body>
            </body>
          </body>
        </worldbody></mujoco>
        """
    )
    source = {
        "full_body": {
            "joint_names": ["Spine3", "Pelvis", "L_Shoulder", "R_Shoulder"],
            "joint_parents": np.array([-1, 0, 1, 1]),
            "face_ids": np.arange(4),
            "sample_joint_indices": np.arange(4),
        }
    }
    mapping = {
        "Spine3": "torso",
        "Pelvis": "pelvis",
        "L_Shoulder": "left_upper",
        "R_Shoulder": "right_upper",
    }

    source_ids, face_ids, report = surface_domains(
        model, source, np.arange(model.ngeom), mapping
    )
    torso = model.body("torso").id
    pelvis = model.body("pelvis").id
    left = model.body("left_upper").id
    right = model.body("right_upper").id
    np.testing.assert_array_equal(source_ids, [torso, pelvis, left, right])
    np.testing.assert_array_equal(face_ids, [torso, pelvis, left, right])
    assert report["status"] == "active"
    assert report["domains_without_surface"] == []


def test_incomplete_source_mapping_inherits_and_reports_uncovered_domain():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody><body name="torso"><freejoint/>
          <geom type="box" size=".2 .2 .2"/>
          <body name="left_upper"><geom type="box" size=".1 .1 .1"/>
            <body name="left_elbow"/>
          </body>
        </body></worldbody></mujoco>
        """
    )
    source = {
        "full_body": {
            "joint_names": ["Spine3", "L_Shoulder", "L_Elbow", "L_Wrist"],
            "joint_parents": np.array([-1, 0, 1, 2]),
            "face_ids": np.arange(4),
            "sample_joint_indices": np.arange(4),
        }
    }
    mapping = {
        "Spine3": "torso",
        "L_Shoulder": "left_upper",
        # This mapped body has no collision surface; its child wrist is absent
        # from the mapping and must inherit the nearest represented ancestor.
        "L_Elbow": "left_elbow",
    }

    source_ids, face_ids, report = surface_domains(
        model, source, np.array([0, 1]), mapping
    )
    assert source_ids[-1] == model.body("left_elbow").id
    assert report["inherited_source_joints"] == {"L_Wrist": "L_Elbow"}
    assert report["domains_without_surface"] == ["left_elbow"]
    np.testing.assert_array_equal(face_ids, [model.body("torso").id, model.body("left_upper").id])

