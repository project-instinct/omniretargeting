"""Leg-only IK must not borrow a torso joint, even with an inverted root tree."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting._fast.limb_scope import paired_limb_scope
from omniretargeting._fast.mesh_repair import _joint_projection, repair_original_meshes
from omniretargeting._fast.objective import FastObjective
from omniretargeting._fast.repair_limits import LOCAL_ANGLE
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.contact import SceneContactTask
from omniretargeting.sequence_retargeting import prepare_sequence, sequence_config


XML = """<mujoco><compiler angle="radian"/><worldbody>
 <body name="root" pos="0 0 .51"><freejoint/><geom type="sphere" size=".02"/>
  <body name="trunk"><joint name="waist" axis="0 1 0"/>
   <geom type="sphere" size=".025"/>
   <body name="a" pos="0 .15 0"><joint name="ja" axis="0 1 0"/>
    <geom type="capsule" fromto="0 0 0 0 0 -.2" size=".015"/>
    <body name="b" pos="0 0 -.2"><joint name="jb" axis="0 1 0"/>
     <geom type="capsule" fromto="0 0 0 0 0 -.2" size=".014"/>
     <body name="tip1" pos="0 0 -.2"><joint name="jc" axis="0 1 0"/>
      <geom name="sole1" type="box" size=".04 .025 .015"/>
     </body></body></body>
   <body name="c" pos="0 -.15 0"><joint name="jd" axis="0 1 0"/>
    <geom type="capsule" fromto="0 0 0 0 0 -.2" size=".015"/>
    <body name="d" pos="0 0 -.2"><joint name="je" axis="0 1 0"/>
     <geom type="capsule" fromto="0 0 0 0 0 -.2" size=".014"/>
     <body name="tip2" pos="0 0 -.2"><joint name="jf" axis="0 1 0"/>
      <geom name="sole2" type="box" size=".04 .025 .015"/>
     </body></body></body>
   <body name="upper" pos="0 0 .2"><joint name="jg" axis="0 1 0"/>
    <geom name="upper_geom" type="sphere" size=".015"/>
   </body>
  </body></body></worldbody></mujoco>"""


@pytest.mark.parametrize("angle", [0.0, 0.7])
def test_scope_locks_shared_waist_and_upper_body_with_contact_compensation(angle):
    model = mujoco.MjModel.from_xml_string(XML)
    di = np.arange(6, model.nv)
    scope = paired_limb_scope(model, [model.body(n).id for n in ("tip1", "tip2")], di)
    limits = SimpleNamespace(
        movable_body=model.body("tip1").id,
        width=len(di),
        frames=2,
        limb_scopes=[scope],
        coords=SimpleNamespace(di=di),
        points=[(None, np.array([model.body("upper").id]), np.ones((2, 1)))],
    )
    # A contact-connected arm and actual collision partner cannot defeat scope.
    measured = {
        "depths": {
            (0, "self", model.geom("sole1").id, model.geom("upper_geom").id): 0.01
        }
    }
    projection = _joint_projection(limits, model, measured)
    step = (projection @ np.full(projection.shape[1], 0.02)).reshape(2, len(di))
    qi = model.jnt_qposadr[model.dof_jntid[di]]
    before = model.qpos0.copy()
    before[3:7] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    after = before.copy()
    after[qi] += step[0]
    upper = [model.body(n).id for n in ("root", "trunk", "upper")]
    data = mujoco.MjData(model)
    data.qpos[:] = before
    mujoco.mj_forward(model, data)
    position, rotation = data.xpos[upper].copy(), data.xmat[upper].copy()
    data.qpos[:] = after
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.xpos[upper], position)
    np.testing.assert_array_equal(data.xmat[upper], rotation)
    np.testing.assert_array_equal(step[:, [0, -1]], 0)
    assert np.any(step[:, 1:-1])


def _floor_problem(depth):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    poses = np.tile(model.qpos0, (5, 1))
    poses[:, 8:11] = [0.35, -0.7, 0.35]
    poses[:, 11:14] = [0.35, -0.7, 0.35]
    names = ["root", "trunk", "a", "b", "tip1", "c", "d", "tip2", "upper"]
    data.qpos[:] = poses[0]
    mujoco.mj_forward(model, data)
    points = np.array([data.xpos[model.body(n).id].copy() for n in names])
    height = data.geom_xpos[model.geom("sole1").id, 2] - 0.015 + depth
    positions = np.tile(points, (len(poses), 1, 1))
    cfg = sequence_config({"self_collision_enabled": False})
    contacts, effectors = {}, {}
    for name, body in [("left_foot", "tip1"), ("right_foot", "tip2")]:
        tracking = positions[:, names.index(body)]
        target = tracking.copy()
        target[:, 2] = height
        contacts[name] = SceneContactTask(
            name=name,
            tracking_positions=tracking,
            contact_positions=target,
            surface_points=target,
            surface_normals=np.tile([0.0, 0.0, 1.0], (len(poses), 1)),
            distances=np.full(len(poses), -depth),
            relative_speeds=np.zeros(len(poses)),
            contact_confidence=np.ones(len(poses)),
            contact_mask=np.ones(len(poses), bool),
        )
        effectors[name] = {
            "robot_body": body,
            "tracking_offset": [0, 0, 0],
            "contact_offsets": [[0, 0, -0.015]],
        }
    coords, poses, tasks, rotations = prepare_sequence(
        model,
        poses,
        positions,
        names,
        {n: n for n in names},
        positions[:, 0],
        contacts,
        {"effectors": effectors},
        cfg,
    )
    objective = FastObjective(
        tasks,
        cfg,
        reference=poses,
        temporal_reference=poses[:, coords.qi],
        root_rotations=rotations,
        framerate=30.0,
    )
    scope = paired_limb_scope(
        model, [model.body(n).id for n in ("tip1", "tip2")], coords.di
    )
    floor = trimesh.creation.box(extents=[4, 4, 2])
    floor.apply_translation([0, 0, height - 1])
    return objective, poses, SceneQuery(floor), scope


@pytest.mark.parametrize("depth", [0.004, 0.12])
def test_bounded_ik_reduces_foot_penetration_and_keeps_impossible_residual(depth):
    objective, poses, plane, scope = _floor_problem(depth)
    result, report = repair_original_meshes(objective, poses, scene_query=plane)
    before = report["before"]["static"]["max_depth_m"]
    after = report["after"]["static"]["max_depth_m"]
    assert before == pytest.approx(depth, abs=1e-8)
    assert after < before - 1e-5
    qi = objective.coords.qi
    fixed = np.ones(objective.model.nq, bool)
    fixed[qi[sorted(scope.columns)]] = False
    np.testing.assert_array_equal(result[:, fixed], poses[:, fixed])
    assert np.abs(result - poses).max() <= LOCAL_ANGLE + 1e-8
    assert report["limits"]["root_unchanged"]
    assert max(report["limits"]["temporal_excess"]) <= 2e-7
    assert all(record["limb_only"] for record in report["history"])
    if depth > 0.1:
        assert after > 0.05
    else:
        assert after < 0.001


def test_endpoint_ancestry_cannot_create_an_unsafe_scope():
    model = mujoco.MjModel.from_xml_string(XML)
    with pytest.raises(ValueError, match="separate branches"):
        paired_limb_scope(
            model, [model.body("a").id, model.body("tip1").id], np.arange(6, model.nv)
        )


def test_solver_proposal_cannot_bypass_upper_body_lock(monkeypatch):
    import omniretargeting._fast.mesh_repair as repair

    objective, poses, plane, _ = _floor_problem(0.004)
    # A numerically finite solver proposal with a waist coordinate must be
    # rejected even when its step is below every angular magnitude bound.
    monkeypatch.setattr(
        repair,
        "_qp_step",
        lambda *args: (
            np.full((len(poses), len(objective.coords.qi)), 1e-6),
            "AlmostSolved",
        ),
    )
    result, report = repair_original_meshes(objective, poses, scene_query=plane)
    np.testing.assert_array_equal(result, poses)
    assert report["status"] == "retained"
    assert all(
        record["check"]["reason"] == "joint outside limb scope changed"
        for record in report["history"]
    )


def test_profile_groups_survive_arbitrary_effector_names():
    from omniretargeting._fast.limb_scope import endpoint_limb_scopes

    objective, poses, plane, expected = _floor_problem(0.004)
    for k, effector in enumerate(objective.effectors):
        old = effector["name"]
        spec = objective.tasks.effectors.pop(old)
        spec["local_repair_group"] = "paired_branches"
        effector["name"] = f"endpoint_{k}"
        objective.tasks.effectors[effector["name"]] = spec
    assert endpoint_limb_scopes(objective) == (expected,)
    result, report = repair_original_meshes(objective, poses, scene_query=plane)
    fixed = np.ones(objective.model.nq, bool)
    fixed[objective.coords.qi[sorted(expected.columns)]] = False
    np.testing.assert_array_equal(result[:, fixed], poses[:, fixed])
    assert report["after"]["static"]["max_depth_m"] < 0.001
    assert all(row["limb_only"] for row in report["history"])


@pytest.mark.parametrize("distal_joint", [False, True])
def test_robot_profile_assigns_scope_without_semantic_endpoint_names(distal_joint):
    from omniretargeting.core import OmniRetargeter

    owner = OmniRetargeter.__new__(OmniRetargeter)
    xml = XML
    if distal_joint:
        xml = xml.replace(
            '<geom name="sole1" type="box" size=".04 .025 .015"/>',
            '<geom name="sole1" type="box" size=".04 .025 .015"/>'
            '<body name="extra_tip"><joint/><geom type="sphere" size=".01"/></body>',
        )
    owner.robot_model = mujoco.MjModel.from_xml_string(xml)
    owner.valid_joint_mapping = {
        "source_a": "tip1",
        "source_b": "tip2",
        "source_c": "upper",
    }
    owner.retargeting_config = {
        "foot_stabilization": {"body_names": {"left": "tip1", "right": "tip2"}}
    }
    owner._infer_end_effector_geometry()
    endpoints = owner.retargeting_config["end_effector_priority"]["effectors"]
    assert endpoints["source_a"]["local_repair_group"] == "lower_limbs"
    assert endpoints["source_b"]["local_repair_group"] == "lower_limbs"
    assert "local_repair_group" not in endpoints["source_c"]
