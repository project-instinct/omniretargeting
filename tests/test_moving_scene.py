"""Real MuJoCo coverage for moving-scene signed and reverse witnesses."""

from copy import deepcopy

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.moving_scene import MovingSceneQuery


def _robot_xml(*, arm_pos=(-0.6, 0.0, 0.0), root_pos=(2.0, 2.0, 2.0), arm_size=(0.12, 0.12, 0.12)):
    return f"""
    <mujoco><compiler angle="radian"/><worldbody>
      <body name="root"><freejoint/>
        <geom name="root_geom" type="box" pos="{root_pos[0]} {root_pos[1]} {root_pos[2]}"
              size=".10 .10 .10" mass="1"/>
        <body name="arm"><joint name="slide" type="slide" axis="1 0 0"/>
          <geom name="arm_geom" type="box" pos="{arm_pos[0]} {arm_pos[1]} {arm_pos[2]}"
                size="{arm_size[0]} {arm_size[1]} {arm_size[2]}"/>
        </body>
      </body>
    </worldbody></mujoco>
    """


def _object(mesh, translations, *, rotations=None, scales=None):
    translations = np.asarray(translations, dtype=float)
    frames = len(translations)
    if rotations is None:
        rotations = np.tile(np.eye(3), (frames, 1, 1))
    if scales is None:
        scales = np.ones(frames, dtype=float)
    return {
        "mesh": mesh,
        "translations": np.asarray(translations, dtype=float),
        "rotations": np.asarray(rotations, dtype=float),
        "scales": np.asarray(scales, dtype=float),
    }


def _box_object(center=(0.0, 0.0, 0.0), frames=1, extents=(0.4, 0.4, 0.4),
                rotations=None, scales=None):
    mesh = trimesh.creation.box(extents=extents)
    return _object(mesh, np.tile(center, (frames, 1)), rotations=rotations, scales=scales)


def _pose(model, *, slide=0.0, root=(0.0, 0.0, 0.0)):
    pose = np.asarray(model.qpos0, dtype=float).copy()
    pose[:3] = root
    pose[3:7] = [1.0, 0.0, 0.0, 0.0]
    pose[int(np.asarray(model.joint("slide").qposadr).item())] = slide
    return pose


def _query(model, objects, pose, time=0.0, dofs=None):
    return MovingSceneQuery(model, objects, frames=len(objects[0]["translations"])).query(
        np.asarray(pose, dtype=float), time, dofs=dofs
    )


def _rows(result, *, kind=None, geom_id=None):
    rows = list(result["rows"])
    if kind is not None:
        rows = [row for row in rows if row["kind"] == kind]
    if geom_id is not None:
        rows = [row for row in rows if int(row["geom_id"]) == int(geom_id)]
    assert rows, (kind, geom_id, result)
    return rows


def _same_sample(rows, sample_id):
    same = [row for row in rows if row["sample_id"] == sample_id]
    return same[0] if same else rows[0]


def test_closed_surface_jacobian_tracks_robot_motion_under_object_transform():
    model = mujoco.MjModel.from_xml_string(_robot_xml())
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    objects = [_box_object(
        center=(-0.36, 0.13, 0.0),
        rotations=np.asarray([rotation]),
        scales=np.asarray([1.2]),
    )]
    slide_qpos = int(np.asarray(model.joint("slide").qposadr).item())
    slide_nv = model.joint("slide").dofadr[0]
    pose = _pose(model)
    geom_id = model.geom("arm_geom").id
    result = _query(model, objects, pose, dofs=[slide_nv])
    base = _rows(result, kind="closed_surface", geom_id=geom_id)[0]
    eps = 1e-5
    plus = _rows(_query(model, objects, pose + np.eye(model.nq)[slide_qpos] * eps,
                        dofs=[slide_nv]), kind="closed_surface", geom_id=geom_id)
    minus = _rows(_query(model, objects, pose - np.eye(model.nq)[slide_qpos] * eps,
                         dofs=[slide_nv]), kind="closed_surface", geom_id=geom_id)
    fd = (_same_sample(plus, base["sample_id"])["clearance"]
          - _same_sample(minus, base["sample_id"])["clearance"]) / (2 * eps)
    assert len(base["jacobian"]) == 1
    assert np.isfinite(base["jacobian"]).all()
    np.testing.assert_allclose(base["jacobian"][0], fd, atol=3e-3, rtol=3e-3)
    assert np.isfinite(base["point"]).all()


def test_reverse_open_surface_jacobian_is_nonzero_for_articulated_box():
    model = mujoco.MjModel.from_xml_string(_robot_xml(arm_pos=(0.0, 0.0, 0.0), arm_size=(0.5, 0.5, 0.5)))
    triangle = trimesh.Trimesh(
        vertices=np.array([[-0.12, -0.08, 0.0], [0.12, -0.08, 0.0], [0.0, 0.12, 0.0]]),
        faces=np.array([[0, 1, 2]]), process=False,
    )
    objects = [_object(triangle, [[0.0, 0.0, 0.0]])]
    slide_qpos = int(np.asarray(model.joint("slide").qposadr).item())
    slide_nv = model.joint("slide").dofadr[0]
    pose = _pose(model)
    geom_id = model.geom("arm_geom").id
    result = _query(model, objects, pose, dofs=[slide_nv])
    candidates = _rows(result, kind="reverse_surface", geom_id=geom_id)
    base = max(candidates, key=lambda row: abs(float(row["jacobian"][0])))
    eps = 1e-5
    plus_pose = pose + np.eye(model.nq)[slide_qpos] * eps
    minus_pose = pose - np.eye(model.nq)[slide_qpos] * eps
    plus = _rows(_query(model, objects, plus_pose, dofs=[slide_nv]),
                 kind="reverse_surface", geom_id=geom_id)
    minus = _rows(_query(model, objects, minus_pose, dofs=[slide_nv]),
                  kind="reverse_surface", geom_id=geom_id)
    fd = (_same_sample(plus, base["sample_id"])["clearance"]
          - _same_sample(minus, base["sample_id"])["clearance"]) / (2 * eps)
    assert len(base["jacobian"]) == 1
    assert abs(base["jacobian"][0]) > 0.5
    np.testing.assert_allclose(base["jacobian"][0], fd, atol=3e-3, rtol=3e-3)
    assert result["unknown_components"] >= 0
    assert isinstance(result["failures"], list)


def test_floating_root_geom_has_zero_selected_articulated_jacobian():
    model = mujoco.MjModel.from_xml_string(_robot_xml(root_pos=(0.0, 0.0, 0.0)))
    objects = [_box_object(center=(0.0, 0.0, 0.0), extents=(0.5, 0.5, 0.5))]
    slide_nv = model.joint("slide").dofadr[0]
    result = _query(model, objects, _pose(model), dofs=[slide_nv])
    root = _rows(result, geom_id=model.geom("root_geom").id)
    assert any(abs(float(row["jacobian"][0])) < 1e-10 for row in root)


def test_interval_steps_catches_moving_object_mid_collision_and_budget():
    model = mujoco.MjModel.from_xml_string(_robot_xml(arm_pos=(0.0, 0.0, 0.0), arm_size=(0.1, 0.1, 0.1)))
    objects = [_box_object(center=(-0.3, 0.0, 0.0), frames=2, extents=(0.2, 0.2, 0.2))]
    objects[0]["translations"][1] = [0.3, 0.0, 0.0]
    query = MovingSceneQuery(model, objects, frames=2)
    q0 = _pose(model, slide=0.0)
    q1 = _pose(model, slide=0.0)
    steps = query.interval_steps(q0, q1, frame=0, spacing=0.01, max_substeps=64)
    assert isinstance(steps, (int, np.integer))
    assert steps > 1
    endpoint0 = query.query(q0, 0.0)
    endpoint1 = query.query(q1, 1.0)
    midpoint = query.query(q0, 0.5)
    arm_id = model.geom("arm_geom").id
    endpoint0_rows = [row for row in endpoint0["rows"] if int(row["geom_id"]) == arm_id]
    endpoint1_rows = [row for row in endpoint1["rows"] if int(row["geom_id"]) == arm_id]
    midpoint_rows = [row for row in midpoint["rows"] if int(row["geom_id"]) == arm_id]
    assert all(row["clearance"] >= -1e-8 for row in endpoint0_rows)
    assert all(row["clearance"] >= -1e-8 for row in endpoint1_rows)
    assert any(row["clearance"] < -1e-6 for row in midpoint_rows)
    with pytest.raises(Exception, match="budget|substeps"):
        query.interval_steps(q0, q1, frame=0, spacing=0.01, max_substeps=2)


def test_disconnected_components_keep_known_gap_empty_and_inputs_immutable():
    model = mujoco.MjModel.from_xml_string(_robot_xml(arm_pos=(0.0, 0.0, 0.0), arm_size=(0.1, 0.1, 0.1)))
    left = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    right = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    left.apply_translation([-0.55, 0.0, 0.0])
    right.apply_translation([0.55, 0.0, 0.0])
    mesh = trimesh.util.concatenate([left, right])
    objects = [_box_object(frames=2)]
    objects[0]["mesh"] = mesh
    objects[0]["translations"][0] = [0.0, 0.0, 0.0]
    objects[0]["translations"][1] = [0.0, 0.0, 0.0]
    before = deepcopy(objects)
    query = MovingSceneQuery(model, objects, frames=2)
    result = query.query(_pose(model), 0.0)
    arm_id = model.geom("arm_geom").id
    arm_rows = [row for row in result["rows"] if int(row["geom_id"]) == arm_id]
    assert all(row["clearance"] >= -1e-8 for row in arm_rows)
    assert result["forward_depth"].shape[0] == 1
    assert np.max(result["forward_depth"]) <= 1e-8
    query.interval_steps(_pose(model), _pose(model), frame=0, spacing=0.1, max_substeps=64)
    np.testing.assert_array_equal(objects[0]["mesh"].vertices, before[0]["mesh"].vertices)
    np.testing.assert_array_equal(objects[0]["mesh"].faces, before[0]["mesh"].faces)
    np.testing.assert_array_equal(objects[0]["translations"], before[0]["translations"])
    np.testing.assert_array_equal(objects[0]["rotations"], before[0]["rotations"])
    np.testing.assert_array_equal(objects[0]["scales"], before[0]["scales"])
