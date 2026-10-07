import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import local_qp, local_rounds, local_windows
from omniretargeting._fast.local_chatter import KERNEL_RADIUS
from omniretargeting._fast.local_repair import Ledger, State, _Context
from omniretargeting._fast.local_rounds import _fork_map, run_wave, waves
from omniretargeting._fast.local_windows import ClusterLimits
from omniretargeting._fast.repair_limits import CONTACT_DRIFT


def _window(first, last=None, *, frames=80, body=1, kind="penetration", columns=(0, 1)):
    mask = np.zeros(frames, bool)
    mask[first:first + 1 if last is None else last] = True
    return dict(kind=kind, bodies=[body] if kind == "penetration" else [], block=first,
                mask=mask, columns=list(columns))


def _indices(result, windows):
    return [[next(i for i, original in enumerate(windows) if w is original) for w in wave]
            for wave in result]


def test_nonconflicting_windows_share_wave():
    windows = [_window(0), _window(19), _window(60, kind="chatter")]
    assert _indices(waves(windows), windows) == [[0, 1, 2]]


@pytest.mark.parametrize("distance", [0, 1, 2 * KERNEL_RADIUS + 2])
def test_conflicts_preserve_input_order(distance):
    windows = [_window(30), _window(30 - distance, kind="chatter")]
    assert _indices(waves(windows), windows) == [[0], [1]]


def test_three_chained_conflicts_stay_in_order():
    windows = [_window(0), _window(18), _window(36)]
    # First and third do not conflict, but the second precedes the third.
    assert _indices(waves(windows), windows) == [[0], [1], [2]]


def test_sparse_masks_use_actual_frames_and_wave_pairs_are_independent():
    windows = [_window(0), _window(30), _window(10), _window(70)]
    windows[0]["mask"][60] = True
    result = waves(windows)
    assert _indices(result, windows) == [[0, 1], [2, 3]]
    for wave in result:
        for i, a in enumerate(wave):
            for b in wave[i + 1:]:
                distance = np.abs(np.flatnonzero(a["mask"])[:, None] - np.flatnonzero(b["mask"]))
                assert distance.min() > 2 * KERNEL_RADIUS + 2
    assert waves([]) == []


@pytest.fixture
def fork_enabled(monkeypatch):
    if not hasattr(os, "fork"):
        pytest.skip("fork unavailable")
    monkeypatch.setattr(local_rounds.threading, "active_count", lambda: 1)
    monkeypatch.setattr(local_rounds, "available_cpus", lambda: 6)
    real_fork = os.fork
    children = set()

    def fork():
        pid = real_fork()
        if pid:
            children.add(pid)
        return pid

    monkeypatch.setattr(os, "fork", fork)
    return children


def _no_children(children):
    # Other suite tests retain a GPU geometry service. Check every worker we
    # own, including zombies, without waiting on or reaping that service.
    assert children, "the test must exercise real fork workers"
    for pid in children:
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)


def test_fork_map_matches_serial_map(fork_enabled):
    items = list(range(5))
    serial = _fork_map(lambda x: {"value": np.arange(x + 3) ** 2}, items, parallel=False)
    forked = _fork_map(lambda x: {"value": np.arange(x + 3) ** 2}, items, parallel=True)
    assert all(np.array_equal(a["value"], b["value"]) for a, b in zip(serial, forked))
    _no_children(fork_enabled)


def test_fork_map_worker_error_reaps_even_large_payload(fork_enabled):
    def fail(x):
        if x == 1:
            raise ValueError("boom")
        return np.ones(300000)

    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(RuntimeError, match="boom"):
        _fork_map(fail, [0, 1, 2], parallel=True)
    assert len(os.listdir("/proc/self/fd")) == before
    _no_children(fork_enabled)


@pytest.mark.parametrize("code", [0, 3])
def test_fork_map_empty_or_nonzero_exit_reaps_children(fork_enabled, code):
    def fail(x):
        if x == 1:
            os._exit(code)
        return np.ones(300000)

    pattern = "empty payload" if code == 0 else "exited with status 3"
    with pytest.raises(RuntimeError, match=pattern):
        _fork_map(fail, [0, 1, 2], parallel=True)
    _no_children(fork_enabled)


@pytest.mark.parametrize("operation", ["pipe", "fork", "read", "deserialize"])
def test_fork_map_parent_errors_close_fds_and_reap(fork_enabled, monkeypatch, operation):
    before = len(os.listdir("/proc/self/fd"))
    if operation == "deserialize":
        def fail(data):
            raise ValueError("bad pickle")
        monkeypatch.setattr(local_rounds.pickle, "loads", fail)
    else:
        original = getattr(os, operation)
        calls = 0

        def fail(*args):
            nonlocal calls
            calls += 1
            if calls == (3 if operation in ("pipe", "fork") else 1):
                raise OSError("injected failure")
            return original(*args)

        monkeypatch.setattr(os, operation, fail)
    with pytest.raises(RuntimeError, match="bad pickle|injected failure"):
        _fork_map(lambda x: np.ones(300000), [0, 1, 2], parallel=True)
    assert len(os.listdir("/proc/self/fd")) == before
    _no_children(fork_enabled)


@pytest.mark.parametrize("cpus", [1, 2, 4, 6])
def test_fork_map_batches_at_cpu_cap_and_keeps_order(fork_enabled, monkeypatch, cpus):
    monkeypatch.setattr(local_rounds, "available_cpus", lambda: cpus)
    real_fork, real_wait = os.fork, os.waitpid
    pending, peak, launches = set(), 0, 0

    def fork():
        nonlocal peak, launches
        pid = real_fork()
        if pid:
            launches += 1
            pending.add(pid)
            peak = max(peak, len(pending))
        return pid

    def wait(pid, options):
        result = real_wait(pid, options)
        pending.discard(pid)
        return result

    monkeypatch.setattr(os, "fork", fork)
    monkeypatch.setattr(os, "waitpid", wait)

    def compute(x):
        from threadpoolctl import threadpool_info
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
        time.sleep((5 - x) * .002)
        return x * x

    serial = _fork_map(lambda x: x * x, list(range(6)), parallel=False)
    assert _fork_map(compute, list(range(6)), parallel=True) == serial
    assert launches == 6 and peak == max(1, cpus // 2) and not pending
    _no_children(fork_enabled)


def test_fork_map_reaps_owned_workers_with_an_unrelated_live_child(fork_enabled):
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
                             stdin=subprocess.PIPE)
    try:
        assert _fork_map(lambda x: x * x, [1, 2, 3], parallel=True) == [1, 4, 9]
        _no_children(fork_enabled)
        assert os.waitpid(child.pid, os.WNOHANG) == (0, 0)
    finally:
        child.stdin.close()
        child.wait(timeout=5)


def _cluster(frames=80, width=2):
    # Bypass heavy construction, but exercise the actual production methods.
    limits = ClusterLimits.__new__(ClusterLimits)
    limits.frames, limits.width = frames, width
    limits.coords = SimpleNamespace(qi=np.arange(width))
    limits.objective = SimpleNamespace(model=SimpleNamespace(body_weldid=np.array([0, 1, 1, 2])))
    limits.permission, limits.movable_body = {}, None
    limits.points = np.zeros((frames, 4, 3))
    limits.budget = np.full((frames, 4), CONTACT_DRIFT)
    limits.budget[:, 2] = CONTACT_DRIFT * 2
    limits.anchors = [np.zeros_like(limits.points)]
    limits.targets = [np.zeros_like(limits.points)]
    limits.target_budgets = [np.full_like(limits.budget, 10.)]
    limits.contact_tolerances = [.001]

    def evaluate(poses, jacobians):
        limits.points[..., 0] = poses[:, :1]

    limits.kin = SimpleNamespace(evaluate=evaluate)
    limits.point_groups = lambda: [(limits.points, np.arange(4), limits.budget)]
    return limits


def test_cluster_grant_correction_mask_release_are_per_frame():
    limits = _cluster()
    windows = [_window(1, 3, body=1), _window(40, 42, body=2), _window(60, body=1)]
    limits.grant(windows)
    expected = np.zeros_like(limits.budget, bool)
    expected[[1, 2, 60], 1] = True
    expected[[40, 41], 3] = True
    acquired = limits.correction_mask(np.arange(4), limits.budget)
    np.testing.assert_array_equal(acquired, expected)
    poses = np.full((limits.frames, 2), .02)
    limits.release(poses)
    np.testing.assert_array_equal(limits.anchors[0][expected, 0], .02)
    assert not limits.anchors[0][~expected].any()
    np.testing.assert_allclose(limits.target_budgets[0][expected], .021)
    np.testing.assert_array_equal(limits.target_budgets[0][~expected], 10.)
    assert not limits.permission
    assert not limits.correction_mask(np.arange(4), limits.budget).any()
    limits.release(poses)  # Idempotent; no global permission leaks into the next wave.
    limits.grant([_window(10, body=2)])
    assert not limits.permission.get(1, np.zeros(limits.frames, bool)).any()
    assert limits.permission[2].sum() == 1


def _gate_context(before, after, depth_before, depth_after, kind):
    frames = 30
    current = np.zeros((frames, 2))
    current[10] = before
    limits = _cluster(frames)
    model = SimpleNamespace(body=lambda b: SimpleNamespace(name=f"body{b}"),
                            body_weldid=np.array([0, 1]), geom_bodyid=np.array([1]))
    key = (10, "static", 0)

    def measure(poses, **kwargs):
        changed = not np.array_equal(poses[10], before)
        return dict(depths={key: depth_after if changed else depth_before}, failures=[], rows=[])

    geometry = SimpleNamespace(measure=measure, linearize=lambda pose, measured: measured)
    chatter = SimpleNamespace(stencil_frames=lambda mask: np.flatnonzero(mask), excess=lambda pose: pose)
    ctx = _Context(SimpleNamespace(model=model), limits, geometry, chatter, Ledger(), [])
    state = State(current, measure(current), {key: depth_before})
    window = _window(10, frames=frames, kind=kind)
    return ctx, state, window


@pytest.mark.parametrize("kind,before,after,depth_before,depth_after,accepted", [
    ("penetration", [1., 0.], [.5, .1], .01, .005, False),
    ("chatter", [1., 0.], [.5, .1], .01, .01, False),
    ("chatter", [1., 0.], [.5, 0.], .01, .010000001, False),
    ("chatter", [1., 0.], [.5, 0.], .01, .01, True),
    ("penetration", [1., 0.], [.5, 0.], .01, .005, True),
])
def test_wave_acceptance_is_elementwise_and_chatter_cannot_increase_geometry(
        monkeypatch, kind, before, after, depth_before, depth_after, accepted):
    ctx, state, window = _gate_context(before, after, depth_before, depth_after, kind)

    def proposal(ctx, run, state):
        action = "stop" if run.iteration else "trial"
        return dict(action=action, record=dict(alpha=0.), counts={}, seconds={},
                    trial=np.array([after]), step_max=1.)

    monkeypatch.setattr(local_rounds, "propose", proposal)
    run_wave(ctx, [window], state, parallel=False)
    assert ctx.ledger.counts["accepted"] == int(accepted)
    assert ctx.ledger.counts["geometry_measures"] == 1
    assert ctx.ledger.counts["geometry_remeasures"] == int(accepted)
    assert ctx.ledger.seconds["round_seconds"] > 0
    np.testing.assert_array_equal(state.current[10], after if accepted else before)
    assert not ctx.limits.permission


@pytest.mark.parametrize("error", [RuntimeError, ValueError, np.linalg.LinAlgError])
def test_propose_numeric_errors_stop_window_and_record_elapsed(monkeypatch, error):
    ctx, state, window = _gate_context([1., 0.], [.5, 0.], .01, .01, "chatter")
    run = local_rounds.WindowRun(ctx, window)
    ctx.chatter.rows = lambda *args, **kwargs: None
    monkeypatch.setattr(local_windows, "witness_rows", lambda *args: (
        sparse.csc_matrix((0, 60)), np.zeros(0), []))

    def solve(*args, **kwargs):
        raise error("numeric failure")

    monkeypatch.setattr(local_qp, "solve", solve)
    original = state.current.copy()
    result = local_rounds.propose(ctx, run, state)
    assert result["action"] == "stop" and result["record"]["stop_reason"] == "qp_error"
    assert result["record"]["elapsed"] >= 0
    assert result["counts"]["qp_solves"] == result["counts"]["qp_failures"] == 1
    assert result["seconds"]["qp"] >= 0
    np.testing.assert_array_equal(state.current, original)


def test_wave_releases_permissions_if_proposal_fails(monkeypatch):
    ctx, state, window = _gate_context([1., 0.], [.5, 0.], .01, .01, "chatter")

    def fail(*args):
        raise RuntimeError("worker failure")

    monkeypatch.setattr(local_rounds, "propose", fail)
    with pytest.raises(RuntimeError, match="worker failure"):
        run_wave(ctx, [window], state, parallel=False)
    assert not ctx.limits.permission
    assert ctx.ledger.seconds["round_seconds"] > 0


def test_sibling_permissions_survive_two_rounds_and_serial_matches_fork(fork_enabled, monkeypatch):
    windows = [_window(2, body=1), _window(40, body=2)]

    def proposal(ctx, run, state):
        expected = np.zeros_like(ctx.limits.budget, bool)
        expected[2, 1] = expected[40, 3] = True
        np.testing.assert_array_equal(
            ctx.limits.correction_mask(np.arange(4), ctx.limits.budget), expected)
        if run.iteration == 2:
            return dict(action="stop", record=dict(alpha=0.), counts={}, seconds={})
        trial = state.current[run.frames].copy()
        trial[:, 0] += .004
        return dict(action="trial", record=dict(alpha=0.), counts={}, seconds={},
                    trial=trial, step_max=.004)

    monkeypatch.setattr(local_rounds, "propose", proposal)

    def run(parallel):
        limits = _cluster()
        model = SimpleNamespace(body=lambda b: SimpleNamespace(name=f"body{b}"),
                                body_weldid=limits.objective.model.body_weldid,
                                geom_bodyid=np.array([1, 3]))

        def measure(poses, **kwargs):
            return dict(depths={(2, "static", 0): .01 - poses[2, 0],
                               (40, "static", 1): .01 - poses[40, 0]}, failures=[], rows=[])

        geometry = SimpleNamespace(measure=measure, linearize=lambda p, m: m)
        chatter = SimpleNamespace(stencil_frames=lambda mask: np.flatnonzero(mask),
                                  excess=lambda pose: np.zeros_like(pose))
        ctx = _Context(SimpleNamespace(model=model), limits, geometry, chatter, Ledger(), [])
        current = np.zeros((80, 2))
        measured = measure(current)
        state = State(current, measured, dict(measured["depths"]))
        run_wave(ctx, windows, state, parallel=parallel)
        assert ctx.ledger.counts["rounds"] == 3
        assert ctx.ledger.counts["accepted"] == 4
        assert ctx.ledger.counts["geometry_measures"] == 2
        assert ctx.ledger.counts["geometry_remeasures"] == 2
        assert not limits.permission
        np.testing.assert_allclose(state.current[[2, 40], 0], .008)
        return state, limits, ctx.ledger.counts

    serial, serial_limits, serial_counts = run(False)
    forked, fork_limits, fork_counts = run(True)
    np.testing.assert_array_equal(serial.current, forked.current)
    assert serial.protected == forked.protected
    np.testing.assert_array_equal(serial_limits.anchors, fork_limits.anchors)
    np.testing.assert_array_equal(serial_limits.target_budgets, fork_limits.target_budgets)
    assert serial_counts == fork_counts
    _no_children(fork_enabled)
