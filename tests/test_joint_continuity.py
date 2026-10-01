"""Smoother total motion cannot hide chatter transferred to another joint."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from omniretargeting._sequence.coordinates import Coordinates
from omniretargeting.sequence_retargeting import retarget_sequence


def model_with_sliders(count):
    bodies = "".join(
        f'<body name="link_{i}" pos="0 0 .2"><joint type="slide" axis="1 0 0" range="-1 1"/><geom size=".02" mass="1"/>'
        for i in range(count)
    )
    return mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>" + bodies + "</body>" * count + "</worldbody></mujoco>"
    )


def test_equal_vector_acceleration_cannot_move_chatter_to_a_quiet_joint():
    model = model_with_sliders(2)
    reference = np.zeros((9, 2))
    reference[:, 0] = 0.01 * (-1) ** np.arange(9)
    moved = reference[:, ::-1].copy()
    np.testing.assert_array_equal(
        np.linalg.norm(np.diff(moved, n=2, axis=0), axis=1),
        np.linalg.norm(np.diff(reference, n=2, axis=0), axis=1),
    )
    guard = SequenceContinuityGuard(
        SimpleNamespace(coords=Coordinates(model)), reference, None
    )
    assert guard.accepts(reference)
    assert not guard.accepts(moved)


def test_budget_scaled_rows_keep_the_exact_joint_feasible_set():
    import cvxpy as cp

    model = model_with_sliders(3)
    frames = 12
    reference = np.zeros((frames, 3))
    reference[:, 0] = 0.01 * (-1) ** np.arange(frames)  # large budgets
    reference[:, 1] = 1e-9 * np.arange(frames)  # budgets of round-off size
    guard = SequenceContinuityGuard(
        SimpleNamespace(coords=Coordinates(model)), reference, None
    )
    flat = cp.Variable(frames * 3)
    constraints = guard.constraints(reference, flat)
    rng = np.random.default_rng(3)
    for scale in (1e-9, 1e-7, 1e-4, 1e-2):
        step = scale * rng.standard_normal((frames, 3))
        flat.value = step.ravel()
        linearized = all(np.all(c.violation() <= 0) for c in constraints)
        expected = all(
            np.all(
                np.abs(np.diff(reference + step, n=order, axis=0))
                <= np.abs(np.diff(reference, n=order, axis=0)) + 1e-7
            )
            for order in (1, 2, 3)
        )
        assert linearized == expected
    flat.value = np.zeros(frames * 3)
    assert all(np.all(c.violation() <= 0) for c in constraints)


@pytest.mark.parametrize("count,fps", [(2, 30), (4, 60)])
def test_optimizer_reduces_chatter_without_spending_another_joints_budget(count, fps):
    model = model_with_sliders(count)
    frames = 15
    reference = np.zeros((frames, count))
    reference[:, 0] = 0.01 * (-1) ** np.arange(frames)
    desired = np.zeros_like(reference)
    desired[:, 1] = reference[:, 0]

    class HostileSecondary:
        def rows(self, pose, frame, dofs):
            yield 1e8, pose - desired[frame], np.eye(count)

    source = np.zeros((frames, 1, 3))
    source[:, 0, 2] = 0.2
    result, _ = retarget_sequence(
        model,
        reference,
        source,
        ["source"],
        {"source": "link_0"},
        source[:, 0],
        {},
        {},
        framerate=fps,
        config={"iterations": 3, "self_collision_enabled": False},
        extra_tasks=(HostileSecondary(),),
        endpoint_reference_trajectory=reference,
    )
    for order in (1, 2, 3):
        assert np.all(
            np.abs(np.diff(result, n=order, axis=0))
            <= np.abs(np.diff(reference, n=order, axis=0)) + 2e-7
        )
    assert np.max(np.abs(np.diff(result[:, 0], n=2))) < 0.5 * np.max(
        np.abs(np.diff(reference[:, 0], n=2))
    )
