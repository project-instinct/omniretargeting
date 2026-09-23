"""Reach adaptation should move the body while retaining back pose and supports."""

from copy import deepcopy
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting._sequence.coordinates import Coordinates
from omniretargeting._sequence.orientation import OrientationTargets
from omniretargeting._sequence.tasks import SequenceTasks
from omniretargeting.contact import build_world_tracking_tasks
from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.retargeting_defaults import CONTACT_RETARGETING_DEFAULTS
from omniretargeting.sequence_retargeting import SEQUENCE_DEFAULTS, retarget_sequence


def reach_case(frames=5):
    # Translating hip joints make independent fixed-foot compensation observable
    # without tying this solver contract to one humanoid's leg proportions.
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><compiler angle="radian"/><worldbody>
        <body name="chest" pos="0 0 1"><freejoint/><geom size=".03"/>
          <body name="hand" pos=".35 0 .2"><geom size=".02"/></body>
          <body name="left_foot" pos="0 .15 -.8">
            <joint type="slide" axis="1 0 0" range="-.4 .4"/>
            <joint type="slide" axis="0 1 0" range="-.4 .4"/>
            <joint type="slide" axis="0 0 1" range="-.4 .4"/>
            <geom size=".02"/></body>
          <body name="right_foot" pos="0 -.15 -.8">
            <joint type="slide" axis="1 0 0" range="-.4 .4"/>
            <joint type="slide" axis="0 1 0" range="-.4 .4"/>
            <joint type="slide" axis="0 0 1" range="-.4 .4"/>
            <geom size=".02"/></body>
        </body></worldbody></mujoco>""")
    seed = np.tile(model.qpos0, (frames, 1))
    seed[:, 3:7] = Rotation.from_euler("xyz", [0.1, 0.2, -0.15]).as_quat(
        scalar_first=True
    )
    data = mujoco.MjData(model)
    data.qpos[:] = seed[0]
    mujoco.mj_forward(model, data)
    names = ["chest", "hand", "left_foot", "right_foot"]
    source = np.tile(
        np.array([data.xpos[model.body(n).id] for n in names]), (frames, 1, 1)
    )
    markers = {
        name: {"tracking_positions": source[:, i].copy()}
        for i, name in enumerate(names)
        if i
    }
    markers["hand"]["tracking_positions"] += [0.18, 0.06, 0]
    tasks = build_world_tracking_tasks(markers)
    tasks = {
        name: replace(
            task,
            contact_mask=np.ones(frames, bool),
            contact_modes=np.full(frames, "sticking"),
        )
        for name, task in tasks.items()
    }
    priority = {
        "effectors": {
            name: {"robot_body": name, "preserve_world_target": name != "hand"}
            for name in tasks
        }
    }
    return model, seed, source, names, tasks, priority


def solve_reach(locked):
    model, seed, source, names, tasks, priority = reach_case()
    solved, report = retarget_sequence(
        model,
        seed,
        source,
        names,
        dict(zip(names, names)),
        source[:, 0],
        tasks,
        priority,
        framerate=30,
        root_orientation_targets=seed[:, 3:7],
        config={
            "lock_root_xy": locked,
            "self_collision_enabled": False,
            "iterations": 20,
        },
    )
    return model, seed, source, tasks, solved, report


def test_reach_moves_xy_while_preserving_back_angles_and_both_feet():
    model, seed, source, tasks, solved, report = solve_reach(False)
    _, _, _, _, locked, locked_report = solve_reach(True)
    np.testing.assert_allclose(
        solved[:, :2] - seed[:, :2], [0.18, 0.06] * np.ones((len(seed), 2)), atol=0.01
    )
    assert report["final_residuals"]["body_orientation_max_rad"] < np.deg2rad(2)
    assert report["final_residuals"]["sticking_max_m"] < 0.001
    assert locked_report["final_residuals"]["body_orientation_max_rad"] > 0.1
    assert np.array_equal(locked[:, :2], seed[:, :2])
    data = mujoco.MjData(model)
    for frame, pose in enumerate(solved):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for name in ("left_foot", "right_foot"):
            assert (
                np.linalg.norm(
                    data.xpos[model.body(name).id]
                    - tasks[name].contact_positions[frame]
                )
                < 0.001
            )
    np.testing.assert_allclose(source[:, 0], seed[:, :3])
    assert np.max(np.abs(np.diff(solved[:, :2], n=2, axis=0))) < 1e-5


@pytest.mark.parametrize("coordinates", ["tangent", "qpos"])
def test_framewise_soft_root_reference_is_not_clipped_after_contact_solve(coordinates):
    model, seed, source, names, tasks, priority = reach_case(1)
    config = deepcopy(CONTACT_RETARGETING_DEFAULTS["end_effector_priority"])
    config.update(priority)
    config["interaction_mesh"] = {"laplacian_weight": 1.0}
    config["pose_direction"] = {"enabled": False}
    if coordinates == "tangent":
        config["step_coordinates"] = coordinates
    terrain = trimesh.creation.box(extents=[3, 3, 0.1])
    terrain.apply_translation([0, 0, -1])
    solver = GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        terrain,
        dict(zip(names, names)),
        1.0,
        end_effector_config=config,
        self_collision_enabled=False,
        terrain_sample_seed=0,
    )
    targets = {
        name: dict(task.frame(0), source_root_position=source[0, 0])
        for name, task in tasks.items()
    }
    solved = solver.retarget_frame(
        source[0],
        seed[0],
        target_base_position=source[0, 0],
        target_base_orientation=seed[0, 3:7],
        end_effector_targets=targets,
        max_iter=25,
    )
    assert np.linalg.norm(solved[:2] - seed[0, :2]) > 0.1
    data = mujoco.MjData(model)
    data.qpos[:] = solved
    mujoco.mj_forward(model, data)
    for name, task in tasks.items():
        assert (
            np.linalg.norm(data.xpos[model.body(name).id] - task.contact_positions[0])
            < 0.005
        )
    assert (
        Rotation.from_quat(seed[0, 3:7], scalar_first=True).inv()
        * Rotation.from_quat(solved[3:7], scalar_first=True)
    ).magnitude() < 0.02


@pytest.mark.parametrize("world", [False, True])
def test_sequence_free_hand_translates_with_root_but_world_foot_does_not(world):
    model, seed, source, names, tasks, priority = reach_case(1)
    task = replace(tasks["hand"], contact_mask=np.zeros(1, bool))
    priority = {
        "effectors": {"hand": {"robot_body": "hand", "preserve_world_target": world}}
    }
    coords = Coordinates(model)
    rows = SequenceTasks(
        model,
        coords,
        seed,
        source,
        names,
        dict(zip(names, names)),
        source[:, 0],
        {"hand": task},
        priority,
        {**SEQUENCE_DEFAULTS, "self_collision_enabled": False},
    )
    before = rows.frame_rows(seed[0], 0)[0][-3:]
    moved = seed[0].copy()
    moved[:3] += [0.3, -0.2, 0.1]
    after = rows.frame_rows(moved, 0)[0][-3:]
    scale = SEQUENCE_DEFAULTS[
        "world_tracking_sigma_m" if world else "free_tracking_sigma_m"
    ]
    np.testing.assert_allclose(
        (after - before) * scale, [0.3, -0.2, 0.1] if world else 0, atol=1e-12
    )
    _residual, jac, _ = rows.frame_rows(moved, 0)
    for axis in range(3):
        step = np.zeros((1, coords.width))
        step[0, coords.inverse[axis]] = 1e-6
        plus = rows.frame_rows(coords.integrate(moved[None], step, 1)[0], 0)[0]
        minus = rows.frame_rows(coords.integrate(moved[None], step, -1)[0], 0)[0]
        np.testing.assert_allclose(
            (plus - minus) / 2e-6,
            jac[:, coords.inverse[axis]].toarray().ravel(),
            atol=1e-7,
        )


@pytest.mark.parametrize("relative", [False, True])
def test_orientation_axes_jacobian_and_root_translation_invariance(relative):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="root"><freejoint/><geom size=".1"/>
        <body name="back"><joint axis="0 1 0"/><geom size=".1"/></body>
      </body></worldbody></mujoco>""")
    coords, data = Coordinates(model), mujoco.MjData(model)
    pose = model.qpos0.copy()
    pose[3:7] = Rotation.from_rotvec([0.5, -0.4, 2.8]).as_quat(scalar_first=True)
    pose[7] = 0.3
    targets = OrientationTargets(
        model,
        1,
        [
            {
                "robot_body": "back",
                "parent_body": "root" if relative else None,
                "rotations": Rotation.from_rotvec([[0.1, 0.4, -0.2]]).as_matrix(),
                "sigma": 0.05,
            }
        ],
    )

    def evaluate(q):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        return targets.rows(data, coords.cols, 0)

    _, matrices, _ = evaluate(pose)
    jac = matrices[0]
    np.testing.assert_allclose(jac[:, :3], 0, atol=1e-12)
    if relative:
        np.testing.assert_allclose(jac[:, :6], 0, atol=1e-12)
    for column in range(coords.width):
        step = np.zeros((1, coords.width))
        step[0, column] = 1e-6
        plus = evaluate(coords.integrate(pose[None], step, 1)[0])[0][0]
        minus = evaluate(coords.integrate(pose[None], step, -1)[0])[0][0]
        np.testing.assert_allclose((plus - minus) / 2e-6, jac[:, column], atol=1e-7)


@pytest.mark.parametrize("fps", [30, 60])
def test_adaptation_preserves_fast_source_path_and_back_rotation(fps):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="chest"><freejoint/><geom size=".03"/>
        <body name="hand" pos=".3 0 .2"><geom size=".02"/></body>
      </body></worldbody></mujoco>""")
    times = np.arange(int(0.6 * fps) + 1) / fps
    pulse = np.exp(-(((times - 0.3) / 0.05) ** 2))
    desired = np.tile(model.qpos0, (len(times), 1))
    desired[:, 0] = 0.4 * pulse
    desired[:, 2] = 1.0
    angles = np.column_stack([0.3 * pulse, 0.4 * pulse, np.deg2rad(170) + 0.6 * pulse])
    desired[:, 3:7] = Rotation.from_euler("xyz", angles).as_quat(scalar_first=True)
    data = mujoco.MjData(model)
    source = []
    for pose in desired:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        source.append(data.xpos[[model.body("chest").id, model.body("hand").id]].copy())
    source = np.array(source)
    target = source[:, 1] + [0.16, -0.04, 0]
    task = build_world_tracking_tasks({"hand": {"tracking_positions": target}})["hand"]
    task = replace(
        task,
        contact_mask=np.ones(len(times), bool),
        contact_modes=np.full(len(times), "sticking"),
    )
    seed = desired.copy()
    seed[:, :2] += (
        np.array([0.16, -0.04]) + 0.02 * (-1.0) ** np.arange(len(times))[:, None]
    )
    # Source quaternion signs carry no physical meaning, including near pi.
    orientations = desired[:, 3:7].copy()
    orientations[1::2] *= -1
    result, report = retarget_sequence(
        model,
        seed,
        source,
        ["chest", "hand"],
        {"chest": "chest", "hand": "hand"},
        source[:, 0],
        {"hand": task},
        {"effectors": {"hand": {"robot_body": "hand"}}},
        framerate=fps,
        root_orientation_targets=orientations,
        config={"self_collision_enabled": False, "iterations": 20},
    )
    assert np.argmax(result[:, 0]) == np.argmax(pulse)
    assert report["final_residuals"]["root_orientation_max_rad"] < 0.01
    assert report["final_residuals"]["sticking_max_m"] < 0.001
    offset = result[:, :2] - desired[:, :2]
    assert np.max(np.linalg.norm(np.diff(offset, n=2, axis=0), axis=1)) < 0.001
    np.testing.assert_array_equal(orientations[1::2], -desired[1::2, 3:7])


@pytest.mark.parametrize("relative", [False, True])
def test_sequence_receives_existing_calibrated_back_orientation(relative):
    from omniretargeting._core.orientation_targets import (
        sequence_body_orientation_targets,
    )
    from tests.test_body_orientation_reference import make_retargeter

    retargeter, names, mapping = make_retargeter(
        trimesh.creation.box(), relative=relative
    )
    retargeter.configure_source_orientation_reference(
        names, np.tile([1.0, 0, 0, 0], (len(names), 1))
    )
    model, data = retargeter.robot_model, retargeter.robot_data
    root = Rotation.from_euler("xyz", [0.4, -0.3, 0.7])
    desired = model.qpos0.copy()
    desired[3:7] = root.as_quat(scalar_first=True)
    desired[7] = 0.4
    data.qpos[:] = desired
    mujoco.mj_forward(model, data)
    positions = np.array([[data.xpos[model.body(mapping[n]).id].copy() for n in names]])
    orientations = np.tile(root.as_quat(scalar_first=True), (1, len(names), 1))
    orientations[:, 1:3] = (root * Rotation.from_rotvec([0, 0.4, 0])).as_quat(
        scalar_first=True
    )
    frozen = orientations.copy()
    targets = sequence_body_orientation_targets(
        retargeter, orientations, desired[None, 3:7]
    )
    seed = desired[None].copy()
    seed[:, 7] = -0.3
    result, report = retarget_sequence(
        model,
        seed,
        positions,
        names,
        mapping,
        positions[:, 0],
        {},
        {},
        framerate=30,
        root_orientation_targets=desired[None, 3:7],
        body_orientation_targets=targets,
        config={"self_collision_enabled": False, "iterations": 20},
    )
    assert abs(result[0, 7] - 0.4) < 0.01
    assert report["final_residuals"]["body_orientation_max_rad"] < 0.01
    np.testing.assert_array_equal(orientations, frozen)
