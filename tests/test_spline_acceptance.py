"""Read-only acceptance checks for the experimental spline correction path."""

from contextlib import ExitStack
from types import SimpleNamespace
import time

import mujoco
import numpy as np
import pytest
import trimesh
from scipy import sparse

from omniretargeting._fast.runtime_options import RuntimeOptions
from omniretargeting._fast.spline_embedding import CorrectionSpline
from omniretargeting._fast.spline_geometry import SplineGeometry
from omniretargeting._fast.spline_model import prepare_spline_model
from omniretargeting._fast.spline_pipeline import local_controls, local_windows
from omniretargeting._fast.spline_solver import SplineSolve
from omniretargeting._fast.continuity import temporal_signals
from omniretargeting.motion_budgets import high_frequency
from omniretargeting.scene_geometry import collision_geom_ids
from tests.test_fast_backend import _objective
from tests.test_fast_whole_canonical import problem as canonical_problem
from tests.test_original_geometry_batches import _problem as small_geometry_problem
from tests.test_original_geometry_batches import _scene


@pytest.fixture(autouse=True)
def cpu_geometry(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")


def _small_geometry(root_x=None):
    objective, poses = small_geometry_problem()
    if root_x is not None:
        poses[:, 0] = root_x
    model = objective.model
    objective.coords = SimpleNamespace(
        qi=np.array([7, 8]), di=np.arange(model.nv), cols=np.arange(model.nv),
        width=model.nv, root_q=0, root_d=0,
        inverse={column: column for column in range(model.nv)})
    objective.reference = poses.copy()
    objects = [dict(mesh=trimesh.creation.box(extents=[.22, .18, .2]),
                    translations=np.tile([.21, 0, .53], (len(poses), 1)))]
    return objective, poses, objects


def _witness_identity(key, row):
    return repr((tuple(key), row.get("sample_id"), row.get("facet_id")))


def test_spline_geometry_keeps_scene_moving_self_rows_and_collision_columns():
    objective, poses, objects = _small_geometry()
    with ExitStack() as resources:
        geometry = SplineGeometry(objective, objects, _scene("closed"), resources)
        measured = geometry.measure(poses, jacobians=True)
        assert measured["complete"] and not measured["failures"]
        assert {key[1] for key, _ in measured["rows"]} == {"static", "moving", "self"}
        assert np.array_equal(geometry.provider.geometry.columns, objective.coords.cols)
        assert geometry.provider.geometry.width == objective.coords.width
        for _, row in measured["rows"]:
            assert np.isfinite(row["clearance"])
            assert np.isfinite(row["jacobian"]).all()


def test_spline_geometry_uses_prepared_collision_geoms_not_a_new_filter():
    class Floor:
        def query(self, points):
            return dict(clearance=points[:, 2] - .35,
                        directions=np.tile([0., 0., 1.], (len(points), 1)),
                        uncertain=np.zeros(len(points), bool))

    objective, poses, _ = _objective()
    expected = set(map(int, collision_geom_ids(objective.model)))
    assert set(map(int, objective.shell_geoms)) == expected
    with ExitStack() as resources:
        geometry = SplineGeometry(objective, (), Floor(), resources)
        assert set(map(int, geometry.provider.geometry.objective.shell_geoms)) == expected
        assert set(map(int, geometry.provider.geometry.self_query.geom_ids)) == expected
        assert np.array_equal(geometry.provider.geometry.columns, objective.coords.cols)


@pytest.mark.parametrize("kind,root_x", [("static", .42), ("moving", .35)])
def test_rejected_new_witnesses_are_relearned_at_the_accepted_origin(kind, root_x):
    objective, origin, objects = _small_geometry(root_x=1.)
    scene = _scene("closed") if kind == "static" else None
    if kind == "moving":
        objects = objects
    else:
        objects = []
    candidate = origin.copy()
    candidate[:, 0] = root_x

    with ExitStack() as resources:
        geometry = SplineGeometry(objective, objects, scene, resources)
        before = geometry.blocks(origin)
        known = {_witness_identity(key, row) for key, row in geometry.current["rows"]}
        accepted, measured, bad = geometry.accepts(candidate)
        assert not accepted and bad
        assert all(key[1] == kind for key in bad)

        added = geometry.learn(origin, candidate, measured, bad)
        assert added > 0 and not geometry.witness_failures
        new_memories = [(key, row) for (_, key, row) in geometry.witness_memory.values()
                        if _witness_identity(key, row) not in known]
        assert len(new_memories) == added
        assert {key[1] for key, _ in new_memories} == {kind}

        after = geometry.blocks(origin)
        assert after[0].A.shape[0] == before[0].A.shape[0] + added
        rows = geometry.current["rows"] + geometry._remembered_rows(origin)
        assert len(rows) == after[0].A.shape[0]
        assembled = {_witness_identity(key, row) for key, row in rows}
        assert all(_witness_identity(key, row) in assembled for key, row in new_memories)

        if kind == "static":
            _, key, row = next(value for value in geometry.witness_memory.values()
                               if value[1][1] == "static")
            data = mujoco.MjData(objective.model)
            data.qpos[:] = origin[key[0]]
            mujoco.mj_forward(objective.model, data)
            geom = int(key[-1])
            point = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ objective.shell_local[row["sample_id"]]
            exact = scene.query(point[None])
            assert row["point"] == pytest.approx(point, abs=1e-13)
            assert row["clearance"] == pytest.approx(float(exact["clearance"][0]), abs=1e-13)
            np.testing.assert_allclose(row["direction"], exact["directions"][0], atol=1e-13)

            tangent = np.zeros(objective.model.nv)
            tangent[0] = 1.
            epsilon = 1e-6
            finite = []
            for sign in (-1, 1):
                data.qpos[:] = origin[key[0]]
                mujoco.mj_integratePos(objective.model, data.qpos, tangent, sign * epsilon)
                mujoco.mj_forward(objective.model, data)
                moved = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ objective.shell_local[row["sample_id"]]
                finite.append(float(scene.query(moved[None])["clearance"][0]))
            derivative = (finite[1] - finite[0]) / (2 * epsilon)
            np.testing.assert_allclose(row["jacobian"] @ tangent, derivative, atol=2e-7)
            index = next(index for index, (row_key, row_value) in enumerate(rows)
                         if _witness_identity(row_key, row_value)
                         == _witness_identity(key, row))
            assembled_tangent = np.zeros(objective.frames * objective.model.nv)
            assembled_tangent[key[0] * objective.model.nv:
                             (key[0] + 1) * objective.model.nv] = tangent
            np.testing.assert_allclose(np.asarray(after[0].A[index] @ assembled_tangent),
                                       -derivative, atol=2e-7)


@pytest.mark.parametrize('evidence', [
    dict(failures=['unknown_query'], complete=False),
    dict(failures=[], complete=False),
    dict(failures=[], complete=True, reliable=False),
    dict(failures=[], complete=True, query_scope=dict(requested_frames=[0])),
])
def test_unknown_geometry_is_counted_and_reported_but_not_turned_into_a_certificate(monkeypatch, evidence):
    objective, poses, objects = _small_geometry()
    scene = _scene("closed")
    with ExitStack() as resources:
        geometry = SplineGeometry(objective, objects, scene, resources)
        geometry.blocks(poses)
        original = scene.query

        def unknown(points):
            value = original(points)
            value["uncertain"][:] = True
            return value

        trial = poses.copy()
        trial[:, 7] += .001
        measured = dict(rows=[], depths={}, **evidence)
        monkeypatch.setattr(geometry, "measure", lambda q, jacobians=False: measured)
        accepted, measured, bad = geometry.accepts(trial)
        # This API returns the native per-key regression decision; the full
        # measurement remains unknown, and the SQP merit rejects it separately.
        assert accepted and not bad and measured is not None
        assert geometry.unknown_trials == 1 and geometry.rejections == 0
        report = geometry.report()
        assert report["unknown_trials"] == 1
        assert report["collision_free_certificate"] is False


def _fake_program():
    return SimpleNamespace(
        A=sparse.csc_matrix(np.array([[1., 0.]])), b=np.array([0.]),
        nonneg=1, soc=(), kinds=lambda: [])


def _solver_shell(deadline):
    state = SplineSolve.__new__(SplineSolve)
    state.damping = 1e-3
    state.deadline = deadline
    state.timings = dict(linear_solve=0., conic_solve=0.)
    state.conic_calls = 0
    return state


def test_cone_fallback_and_option_construction_are_counted(monkeypatch):
    model = SimpleNamespace(hessian=sparse.eye(2, format="csc"), gradient=np.array([-1., 0.]),
                            diagonal=np.ones(2), scale=1., program=_fake_program())
    state = _solver_shell(time.monotonic() + 10.)
    import omniretargeting._fast.working_set as working_set
    calls = []

    def solve(*args, **kwargs):
        calls.append("working_set")
        args[5]()
        return SimpleNamespace(x=np.zeros(2), status="Solved", iterations=1)

    monkeypatch.setattr(working_set, "solve", solve)
    step, diagnostics = state._solve(model)
    assert step is not None and calls == ["working_set"]
    assert diagnostics["conic_calls"] == 1 and state.conic_calls == 1

    state = _solver_shell(time.monotonic() + 10.)
    calls.clear()

    def fallback(*args, **kwargs):
        calls.append("working_set")
        args[5]()
        return None

    monkeypatch.setattr(working_set, "solve", fallback)
    import clarabel
    constructed = []

    class Solver:
        def __init__(self, *args):
            constructed.append(args)

        def solve(self):
            return SimpleNamespace(x=np.zeros(2), status="Solved", iterations=1)

    monkeypatch.setattr(clarabel, "DefaultSolver", Solver)
    step, diagnostics = state._solve(model)
    assert step is not None
    assert calls == ["working_set"] and len(constructed) == 1
    assert diagnostics["complete_program_fallback"] is True
    assert diagnostics["conic_calls"] == 2 and state.conic_calls == 2


def test_expired_budget_before_an_infeasible_conic_fallback_does_not_call_clarabel(monkeypatch):
    model = SimpleNamespace(hessian=sparse.eye(2, format="csc"), gradient=np.array([-1., 0.]),
                            diagonal=np.ones(2), scale=1., program=_fake_program())
    state = _solver_shell(time.monotonic() - 1.)
    import omniretargeting._fast.working_set as working_set
    monkeypatch.setattr(working_set, "solve", lambda *args, **kwargs: pytest.fail(
        "expired budget must not construct a conic fallback"))
    step, diagnostics = state._solve(model)
    assert step is None
    assert diagnostics["backend"] == "budget_exhausted"
    assert state.conic_calls == 0


def test_expired_budget_does_not_start_even_a_feasible_sparse_solve(monkeypatch):
    model = SimpleNamespace(hessian=sparse.eye(2, format="csc"), gradient=np.zeros(2),
                            diagonal=np.ones(2), scale=1., program=_fake_program())
    state = _solver_shell(time.monotonic() - 1.)
    monkeypatch.setattr("omniretargeting._fast.spline_solver.spsolve",
                        lambda *args: pytest.fail("no solve after expiry"))
    step, diagnostics = state._solve(model)
    assert step is None and diagnostics["backend"] == "budget_exhausted"
    assert state.exhausted


def test_sparse_solve_crossing_deadline_discards_the_unvalidated_step(monkeypatch):
    model = SimpleNamespace(hessian=sparse.eye(2, format="csc"), gradient=np.zeros(2),
                            diagonal=np.ones(2), scale=1., program=_fake_program())
    state = _solver_shell(time.monotonic() + 10.)

    def solve(*args):
        state.deadline = time.monotonic() - 1.
        return np.zeros(2)

    monkeypatch.setattr("omniretargeting._fast.spline_solver.spsolve", solve)
    step, diagnostics = state._solve(model)
    assert step is None and diagnostics["interrupted_after"] == "sparse_solve"
    assert state.exhausted and state.conic_calls == 0


def test_working_set_does_not_start_another_round_after_deadline(monkeypatch):
    model = SimpleNamespace(hessian=sparse.eye(2, format="csc"), gradient=np.array([-1., 0.]),
                            diagonal=np.ones(2), scale=1., program=_fake_program())
    state = _solver_shell(time.monotonic() + 10.)
    import omniretargeting._fast.working_set as working_set

    def rounds(P, c, A, b, cones, options, **kwargs):
        options()
        state.deadline = time.monotonic() - 1.
        options()
        pytest.fail('second expired round must not start')

    monkeypatch.setattr(working_set, 'solve', rounds)
    step, diagnostics = state._solve(model)
    assert step is None and state.exhausted
    assert diagnostics['interrupted_after'] == 'working_set_round'
    assert diagnostics['conic_calls'] == state.conic_calls == 1


def test_fixed_root_pins_steps_without_rebasing_the_spline_reference():
    objective, poses, _ = _objective(temporal=True)
    coords = objective.coords
    q = coords.root_q
    objective.fixed_root = poses[:, q:q + 7].copy()
    fixed = objective.fixed_root.copy()
    spline = CorrectionSpline(coords, objective.reference, objective.framerate, identity=True)
    rng = np.random.default_rng(57)
    coefficients = rng.normal(scale=.03, size=spline.coefficients.shape)
    moved = spline.embed(coefficients)
    target = moved.copy()
    target[:, q:q + 7] = fixed
    model, _, selected = prepare_spline_model(
        objective, spline, coefficients, objective.evaluate(moved),
        columns=coords.scalar_cols)

    assert model.gradient.size == objective.frames * len(coords.scalar_cols)
    assert np.array_equal(spline.reference, objective.reference)
    assert np.array_equal(objective.fixed_root, fixed)
    active_root = [coords.inverse[dimension] for dimension in range(coords.root_d, coords.root_d + 6)
                   if dimension in coords.inverse]
    expected = coords.difference(moved, target)[:, active_root]
    np.testing.assert_array_equal(model.step_lower[:, active_root], expected)
    np.testing.assert_array_equal(model.step_upper[:, active_root], expected)
    assert objective.root_budget == objective.cfg["translation_step_m"]


def test_local_control_window_preserves_exterior_poses_and_c2_closure():
    objective, poses = canonical_problem()
    objective.surfaces = []
    coords = objective.coords
    reference = np.tile(objective.reference, (4, 1))
    spline = CorrectionSpline(coords, reference, objective.framerate, spacing=.24)
    focus = np.zeros(spline.count, bool)
    focus[0] = True
    start, stop = local_windows(spline, focus)[0]
    controls = local_controls(spline, start, stop)
    assert len(controls) and start == 0 and stop < spline.count

    rng = np.random.default_rng(61)
    coefficients = rng.normal(scale=.05, size=spline.coefficients.shape)
    changed = coefficients.copy()
    changed[controls] += rng.normal(scale=.1, size=(len(controls), coords.width))
    before, after = spline.embed(coefficients), spline.embed(changed)
    outside = np.arange(stop, spline.count)
    np.testing.assert_array_equal(before[outside], after[outside])
    affected = spline.affected_frames(controls)
    assert affected.size and np.all((affected >= start) & (affected < stop))

    left, right = spline.times[start], spline.times[stop - 1]
    exterior = np.linspace(right, spline.end_time, 53)
    for derivative in range(3):
        np.testing.assert_array_equal(spline.correction(coefficients, exterior, derivative),
                                      spline.correction(changed, exterior, derivative))

    before_signals = temporal_signals(coords, before)
    after_signals = temporal_signals(coords, after)
    for name in before_signals:
        difference = after_signals[name] - before_signals[name]
        filtered = high_frequency(difference)
        np.testing.assert_allclose(filtered[stop:], 0., atol=2e-6)
        for order in (1, 2, 3):
            finite = np.diff(difference, n=order, axis=0)
            assert not np.any(np.abs(finite[stop:]) > 2e-12)


def test_spline_path_remains_explicitly_opt_in_and_composition_is_rejected(monkeypatch):
    assert RuntimeOptions().spline_correction is False
    assert RuntimeOptions().spline_adaptive is False
    assert RuntimeOptions().spline_shared_loop is False
    monkeypatch.setenv("OMNIRETARGETING_FAST_ENABLE", "spline_correction,spline_adaptive")
    enabled = RuntimeOptions.from_environment()
    assert enabled.spline_correction and enabled.spline_adaptive
    for value in ("spline_adaptive", "spline_shared_loop", "spline_correction,gpu_conic"):
        monkeypatch.setenv("OMNIRETARGETING_FAST_ENABLE", value)
        with pytest.raises(ValueError):
            RuntimeOptions.from_environment()
