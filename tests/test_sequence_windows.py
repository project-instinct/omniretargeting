"""Window-restricted guarded steps keep the exact full-trajectory bounds."""

from itertools import pairwise
from types import SimpleNamespace

import cvxpy as cp
import mujoco
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import lsq_linear

from omniretargeting import sequence_retargeting
from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from omniretargeting._sequence.coordinates import Coordinates
from omniretargeting._sequence.windows import (
    independent_groups,
    run_forked,
    temporal_windows,
    window_system,
)
from omniretargeting.sequence_retargeting import retarget_sequence


def model_with_sliders(count):
    bodies = "".join(
        f'<body name="link_{i}" pos="0 0 .2"><joint type="slide" axis="1 0 0" '
        f'range="-1 1"/><geom size=".02" mass="1"/>'
        for i in range(count)
    )
    return mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>" + bodies + "</body>" * count + "</worldbody></mujoco>"
    )


@pytest.mark.parametrize("frames,size", [(10, 8), (97, 32), (200, 96)])
def test_windows_cover_every_frame_once_and_shift_between_sweeps(frames, size):
    for sweep in (0, 1):
        windows = temporal_windows(frames, size, sweep)
        covered = np.concatenate([np.arange(w.start, w.stop) for w in windows])
        np.testing.assert_array_equal(covered, np.arange(frames))
        assert all(0 < len(w) <= size for w in windows)
    first = {w.start for w in temporal_windows(frames, size, 0)}
    second = {w.start for w in temporal_windows(frames, size, 1)}
    assert first != second


def test_group_members_are_separated_by_a_whole_window():
    windows = temporal_windows(300, 32, 1)
    for group in independent_groups(windows):
        for first, second in pairwise(group):
            assert second.start - first.stop >= min(len(w) for w in windows[1:-1])


def test_forked_jobs_return_in_order_and_report_failures():
    state = {"value": 3}

    def job(k):
        state["value"] += k  # a child's mutation must not reach the parent
        return k * state["value"]

    from functools import partial

    assert run_forked([partial(job, k) for k in range(5)], 3) == [
        k * (3 + k) for k in range(5)
    ]
    assert state["value"] == 3

    # Results larger than a pipe buffer, more jobs than workers.
    large = run_forked([partial(np.full, 2**17, float(k)) for k in range(7)], 3)
    assert [float(a[0]) for a in large] == list(range(7))
    assert all(a.shape == (2**17,) for a in large)

    def broken():
        raise ValueError("window exploded")

    with pytest.raises(RuntimeError, match="window exploded"):
        run_forked([broken, partial(job, 1), partial(job, 2)], 2)


def test_window_system_matches_full_problem_with_fixed_other_frames():
    rng = np.random.default_rng(5)
    frames, width = 12, 3
    matrix = sparse.random(80, frames * width, density=0.15, random_state=rng)
    matrix = sparse.vstack([matrix, sparse.eye(frames * width)], format="csr")
    residual = rng.normal(size=matrix.shape[0])
    window = range(4, 9)
    local, rhs = window_system(matrix.tocsc(), residual, window, width)
    columns = slice(window.start * width, window.stop * width)
    expected = lsq_linear(matrix[:, columns].toarray(), -residual, tol=1e-12).x
    actual = lsq_linear(local.toarray(), rhs, tol=1e-12).x
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    assert local.shape[0] < matrix.shape[0]


def test_windowed_continuity_bounds_equal_full_bounds_with_fixed_frames():
    model = model_with_sliders(2)
    frames = 20
    reference = np.zeros((frames, 2))
    reference[:, 0] = 0.02 * np.sin(np.arange(frames))
    reference[:, 1] = 1e-9 * np.arange(frames)
    guard = SequenceContinuityGuard(
        SimpleNamespace(coords=Coordinates(model)), reference, None
    )
    window = range(6, 13)
    full_flat = cp.Variable(frames * 2)
    local_flat = cp.Variable(len(window) * 2)
    full = guard.constraints(reference, full_flat)
    local = guard.constraints(reference, local_flat, window)
    rng = np.random.default_rng(11)
    for scale in (1e-9, 1e-6, 1e-3, 1e-1):
        step = scale * rng.standard_normal((len(window), 2))
        padded = np.zeros((frames, 2))
        padded[window.start : window.stop] = step
        full_flat.value, local_flat.value = padded.ravel(), step.ravel()
        full_ok = all(np.all(c.violation() <= 1e-12) for c in full)
        local_ok = all(np.all(c.violation() <= 1e-12) for c in local)
        assert full_ok == local_ok
        moved = reference + padded
        assert guard.accepts(moved) == guard.accepts(moved, window)


@pytest.mark.parametrize("window_frames", [8, 16])
def test_windowed_guarded_solve_keeps_every_joint_budget(window_frames, monkeypatch):
    model = model_with_sliders(2)
    frames = 40
    reference = np.zeros((frames, 2))
    reference[:, 0] = 0.01 * (-1) ** np.arange(frames)
    desired = np.zeros_like(reference)
    desired[:, 1] = reference[:, 0]

    class HostileSecondary:
        def rows(self, pose, frame, dofs):
            yield 1e8, pose - desired[frame], np.eye(2)

    source = np.zeros((frames, 1, 3))
    source[:, 0, 2] = 0.2
    results = {}
    for size, workers in ((window_frames, 1), (window_frames, 4), (1000, 1)):
        monkeypatch.setattr(sequence_retargeting, "WINDOW_FRAMES", size)
        monkeypatch.setattr(sequence_retargeting, "WINDOW_WORKERS", workers)
        result, report = retarget_sequence(
            model,
            reference,
            source,
            ["source"],
            {"source": "link_0"},
            source[:, 0],
            {},
            {},
            framerate=30,
            config={
                "iterations": 3,
                "self_collision_enabled": False,
            },
            extra_tasks=(HostileSecondary(),),
            endpoint_reference_trajectory=reference,
        )
        if size == window_frames and workers > 1:
            # Parallel groups are exactly the serial groups.
            np.testing.assert_array_equal(result, results[size])
        results[size] = result
        for order in (1, 2, 3):
            assert np.all(
                np.abs(np.diff(result, n=order, axis=0))
                <= np.abs(np.diff(reference, n=order, axis=0)) + 2e-7
            )
        if size == window_frames:
            windows = {tuple(row["window"]) for row in report["history"]}
            assert len(windows) > 1
    windowed_chatter = np.max(np.abs(np.diff(results[window_frames][:, 0], n=2)))
    assert windowed_chatter < 0.5 * np.max(np.abs(np.diff(reference[:, 0], n=2)))
