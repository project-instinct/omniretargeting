"""Regression coverage for surface-only framewise contact overlays."""

import cvxpy as cp
import mujoco
import numpy as np
import trimesh

from omniretargeting._core.motion import FramewiseMotionMixin
from omniretargeting._solver.constraints import build_step_constraints
from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.surface_contact import SurfaceContactTasks
from omniretargeting.surface_tasks import SurfaceBinding
from tests._contact_fixtures import _task

OVERLAY_XML = """
<mujoco><compiler angle="radian"/><worldbody>
  <body name="root"><freejoint/><geom type="sphere" size=".03" mass="1"/>
    <body name="left"><joint name="left_hinge" type="hinge" axis="0 1 0"/>
      <geom name="left_geom" type="capsule" fromto="0 0 0 0.3 0 0" size=".03"/>
    </body>
  </body>
</worldbody></mujoco>
"""


def _motion_driver(surface, model, coupled=True):
    driver = FramewiseMotionMixin()
    driver.robot_model = model
    driver.robot_data = mujoco.MjData(model)
    driver.robot_height = 1.0
    driver.retargeting_config = {
        "self_collision_enabled": False,
        "end_effector_priority": {"enabled": True},
    }
    driver.valid_joint_mapping = {"Pelvis": "root", "L": "left"}
    driver.valid_joint_names = ["Pelvis", "L"]
    driver.mapped_joint_indices = [0, 1]
    driver.smplx_joint_indices = {"Pelvis": 0, "L": 1}
    driver.smplx_joint_names = ["Pelvis", "L"]
    driver.base_orientation_config = {"root_position": "Pelvis"}
    driver._resolve_source_root_index = lambda: 0
    driver._estimate_base_orientation_from_joints = lambda *_args: None
    driver._surface_contacts_built = True
    driver.last_surface_contact_tasks = surface
    driver._coupled_self_collision_projection = coupled
    return driver


def _overlay_surface(frames=3, active_frame=1):
    active = np.zeros((frames, 1), dtype=bool)
    active[active_frame, 0] = True
    # Match the production SurfaceContactTasks contract used by the motion
    # driver before it enters the per-frame loop.
    return type(
        "SurfaceTasks",
        (),
        {"active": active, "targets": np.zeros((frames, 1, 3), dtype=float)},
    )()


def test_surface_only_overlay_solves_from_baseline_and_skips_next_inactive_frame(
    monkeypatch,
):
    model = mujoco.MjModel.from_xml_string(OVERLAY_XML)
    driver = _motion_driver(_overlay_surface(), model)
    source = np.zeros((3, 2, 3), dtype=float)
    source[:, 1, 0] = 0.3
    baseline = np.tile(model.qpos0, (3, 1))
    baseline[:, 3] = 1.0
    baseline[:, 7] = [0.05, 0.10, 0.15]
    primary = _task(np.zeros((3, 3)), contact_mask=np.zeros(3, dtype=bool))
    calls = []

    class FakeRetargeter:
        def __init__(self, *_args, **_kwargs):
            self.last_frame_solve = None
            self.surface_contact_tasks = None
            self.robot_reference_qpos = model.qpos0.copy()
            self.self_collision_enabled = False

        def retarget_frame(self, _human_joints, q_init, **kwargs):
            calls.append({"q_init": q_init.copy(), "overlay": self._surface_overlay_frame,
                          "kwargs": kwargs})
            result = q_init.copy()
            result[7] += 0.125
            return result

    monkeypatch.setattr(
        "omniretargeting.retargeting.GenericInteractionRetargeter",
        FakeRetargeter,
    )
    result = driver._perform_retargeting(
        source,
        trimesh.creation.box(),
        base_orientations=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        end_effector_tasks={"left_hand": primary},
        reference_trajectory=baseline,
        overlay_contact_names={"left_hand"},
    )

    assert len(calls) == 1
    assert calls[0]["overlay"] == 1
    np.testing.assert_array_equal(calls[0]["q_init"], baseline[1])
    assert calls[0]["kwargs"]["relation_frame"] == 1
    np.testing.assert_array_equal(result[0], baseline[0])
    np.testing.assert_allclose(result[1, 7], baseline[1, 7] + 0.125)
    np.testing.assert_array_equal(result[2], baseline[2])


def test_primary_contact_overlay_keeps_surface_overlay_marker_clear(monkeypatch):
    model = mujoco.MjModel.from_xml_string(OVERLAY_XML)
    driver = _motion_driver(_overlay_surface(frames=1, active_frame=0), model, coupled=False)
    source = np.zeros((1, 2, 3), dtype=float)
    baseline = np.tile(model.qpos0, (1, 1))
    baseline[:, 3] = 1.0
    primary = _task(np.zeros((1, 3)), contact_mask=np.ones(1, dtype=bool))
    calls = []

    class FakeRetargeter:
        def __init__(self, *_args, **_kwargs):
            self.last_frame_solve = None
            self.surface_contact_tasks = None
            self.robot_reference_qpos = model.qpos0.copy()
            self.self_collision_enabled = False

        def retarget_frame(self, _human_joints, q_init, **kwargs):
            calls.append((self._surface_overlay_frame, kwargs))
            return q_init.copy()

    monkeypatch.setattr(
        "omniretargeting.retargeting.GenericInteractionRetargeter",
        FakeRetargeter,
    )
    driver._perform_retargeting(
        source,
        trimesh.creation.box(),
        base_orientations=np.array([[1.0, 0.0, 0.0, 0.0]]),
        end_effector_tasks={"left_hand": primary},
        reference_trajectory=baseline,
        overlay_contact_names={"left_hand"},
    )
    assert len(calls) == 1
    assert calls[0][0] is None


def test_surface_overlay_constraints_allow_only_active_articulated_ancestors():
    model = mujoco.MjModel.from_xml_string(OVERLAY_XML.replace(
        "</body>\n</worldbody>",
        "<body name=\"right\"><joint name=\"right_hinge\" type=\"hinge\" axis=\"0 1 0\"/>"
        "<geom name=\"right_geom\" type=\"capsule\" fromto=\"0 0 0 -0.3 0 0\" size=\".03\"/></body>"
        "</body>\n</worldbody>"
    ))
    solver = GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        trimesh.creation.box(),
        {"L": "left", "R": "right"},
        1.0,
        terrain_sample_points=1,
        hard_penetration_constraint=False,
        self_collision_enabled=False,
        valid_joint_names=["L", "R"],
        end_effector_config={},
    )
    frames = 1
    tasks = SurfaceContactTasks(
        model,
        [SurfaceBinding("left", 0, (1, 0, 0), "left_geom", (0, 0, 0), (0, 0, 1))],
        np.zeros((frames, 1, 3)),
        np.zeros((frames, 1, 3)),
        np.ones((frames, 1), dtype=bool),
        np.tile([0.0, 0.0, 1.0], (frames, 1, 1)),
    )
    solver.surface_contact_tasks = tasks
    solver._surface_overlay_frame = 0
    q = model.qpos0.copy()
    mujoco.mj_forward(model, solver.robot_data)
    dqa = cp.Variable(len(solver.q_a_indices))
    constraints = []
    build_step_constraints(
        solver,
        constraints=constraints,
        dqa=dqa,
        end_effector_targets={},
        previous_human_joints=None,
        previous_previous_human_joints=None,
        previous_target_base_orientation=None,
        q=q,
        q_a_current=q[solver.q_a_indices],
        q_last=None,
        q_prev=None,
        target_base_orientation=None,
        target_base_position=None,
        target_human_joints=None,
    )
    problem = cp.Problem(
        cp.Maximize(dqa[int(np.flatnonzero(solver.q_a_indices == model.joint("left_hinge").qposadr)[0])]),
        constraints + [dqa >= -1.0, dqa <= 1.0],
    )
    problem.solve(solver="CLARABEL")
    assert problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    left_qpos = model.joint("left_hinge").qposadr
    for qa, qpos in enumerate(solver.q_a_indices):
        if qpos == left_qpos:
            assert dqa.value[qa] > 0.9
        else:
            assert abs(dqa.value[qa]) < 1e-8
