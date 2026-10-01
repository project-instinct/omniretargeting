"""Sequence passes stop on rejected or converged steps without repeated solves."""

from types import SimpleNamespace

import mujoco
import numpy as np

import omniretargeting.sequence_retargeting as sequence_module
from omniretargeting.contact import build_world_tracking_tasks
from omniretargeting.sequence_retargeting import (
    CONVERGED_RELATIVE_DECREASE,
    retarget_sequence,
)


def free_root_model():
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="base" pos="0 0 1"><freejoint/>'
        '<geom size=".1"/><body name="arm" pos="0 0 .2">'
        '<joint type="hinge" axis="0 1 0" range="-1 1"/>'
        '<geom size=".05" pos="0 0 .1"/><site name="tip" pos="0 0 .2"/>'
        "</body></body></worldbody></mujoco>"
    )


def source_motion(model, frames=12, amplitude=0.4):
    data = mujoco.MjData(model)
    poses = np.tile(model.qpos0, (frames, 1))
    poses[:, 7] = amplitude * np.sin(np.linspace(0, np.pi, frames))
    positions = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        positions.append(
            [data.xpos[model.body("base").id].copy(), data.site_xpos[0].copy()]
        )
    return poses, np.asarray(positions)


def solve(model, seed, positions, iterations, callback=None):
    tasks = build_world_tracking_tasks({"tip": {"tracking_positions": positions[:, 1]}})
    priority = {
        "effectors": {"tip": {"robot_body": "arm", "preserve_world_target": True}}
    }
    return retarget_sequence(
        model,
        seed,
        positions,
        ["root", "tip"],
        {"root": "base", "tip": "arm"},
        positions[:, 0],
        tasks,
        priority,
        framerate=30,
        config={"self_collision_enabled": False, "iterations": iterations},
        callback=callback,
    )


class ZeroStepGuard:
    """A constrained subproblem that fails and therefore proposes no motion."""

    pose = continuity = None
    preserve_contacts = False
    failures = 0

    def __init__(self):
        self.solves = 0

    def solve(self, matrix, rhs, lower, upper, poses):
        self.solves += 1
        self.failures += 1
        return SimpleNamespace(x=np.zeros(lower.size), status=0)

    def accepts(self, poses):
        raise AssertionError("a zero step must not enter the line search")


def test_failed_step_stops_without_repeating_the_identical_pass(monkeypatch):
    model = free_root_model()
    _, positions = source_motion(model)
    seed = np.tile(model.qpos0, (len(positions), 1))
    guard = ZeroStepGuard()
    monkeypatch.setattr(sequence_module, "priority_guard", lambda *args: guard)
    evaluations = []
    solved, report = solve(
        model, seed, positions, 5, callback=lambda q, r: evaluations.append(r)
    )
    assert guard.solves == 1
    assert len(report["history"]) == len(evaluations) == 1
    assert report["history"][0]["alpha"] == 0.0
    assert report["history"][0]["linear_solver_status"] == 0
    assert np.array_equal(solved, seed)


def test_converged_pass_stops_before_the_repeated_pass():
    model = free_root_model()
    _, positions = source_motion(model)
    seed = np.tile(model.qpos0, (len(positions), 1))
    solved, report = solve(model, seed, positions, 40)
    history = report["history"]
    assert len(history) < 40
    assert {row["phase"] for row in history} == {"pose"}
    last = history[-1]
    assert last["alpha"] == 0.0 or (
        last["cost"] - last["after"] <= CONVERGED_RELATIVE_DECREASE * last["cost"]
    )
    assert np.all(np.isfinite(solved))


def test_budget_limited_pass_still_continues_in_the_second_pass():
    model = free_root_model()
    _, positions = source_motion(model)
    seed = np.tile(model.qpos0, (len(positions), 1))
    _, report = solve(model, seed, positions, 1)
    phases = [row["phase"] for row in report["history"]]
    assert phases[0] == "pose" and report["history"][0]["alpha"] > 0
    assert (
        report["history"][0]["cost"] - report["history"][0]["after"]
        > CONVERGED_RELATIVE_DECREASE * report["history"][0]["cost"]
    )
    assert phases == ["pose", "root_continuity"]
