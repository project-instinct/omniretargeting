"""Local repair algebra and screened correction, without clip dependencies."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from omniretargeting import motion_budgets as mb
from omniretargeting._fast.local_chatter import CHATTER_RESERVE, JointChatter, high_pass_matrix
from omniretargeting._fast.local_correction import (
    MAX_ITERATIONS, SMALL_CORRECTION_RADIUS, _Problem, correction, solve,
)
from omniretargeting._fast.local_qp import CLEARANCE_MARGIN_M, cut_lower, witness_lower
from omniretargeting._fast.local_repair import CORRECTION_RESERVE, Ledger
from omniretargeting._fast.repair_limits import LinearBounds
from omniretargeting.moving_sample_summary import summarize


@pytest.mark.parametrize("frames", [0, 1, 2, 7, 19, 61])
def test_high_pass_matches_audit_including_boundary_frames(frames):
    rng = np.random.default_rng(frames)
    signal = rng.normal(size=(frames, 4))
    assert_allclose(high_pass_matrix(frames) @ signal, mb.high_frequency(signal), atol=2e-15)
    assert_allclose(high_pass_matrix(frames) @ np.ones(frames), 0., atol=2e-15)


@pytest.mark.parametrize("elastic", [False, True])
def test_chatter_rows_match_reserved_audit_and_existing_excess(elastic):
    rng = np.random.default_rng(85)
    qi = np.array([1, 3, 4])
    reference = rng.normal(scale=.01, size=(31, 5))
    chatter = JointChatter(SimpleNamespace(qi=qi), reference)
    candidate = reference + rng.normal(scale=.04, size=reference.shape)
    frames, columns = np.array([0, 1, 7, 15, 29, 30]), np.array([0, 2])
    matrix, lower, upper = chatter.rows(candidate, frames, columns, elastic=elastic)
    content = chatter.content(candidate)[np.ix_(frames, columns)]
    audit = mb.chatter_excess(reference[:, qi], candidate[:, qi]) - mb.CHATTER_EXCESS_TOLERANCE
    assert_allclose(chatter.excess(candidate), audit, atol=1e-15)
    reserved = chatter.limit[np.ix_(frames, columns)] - CHATTER_RESERVE
    cap = reserved if elastic else np.maximum(reserved, np.abs(content))
    assert_allclose(lower.reshape(cap.shape), -cap - content)
    assert_allclose(upper.reshape(cap.shape), cap - content)
    if not elastic:
        assert np.all(lower <= 0) and np.all(upper >= 0)
    for scale in [0., .001, .05]:
        step = rng.normal(scale=scale, size=(31, 3))
        trial = candidate.copy()
        trial[:, qi] += step
        row_value = np.asarray(matrix @ step.ravel()).reshape(cap.shape)
        assert_allclose(row_value + content, chatter.content(trial)[np.ix_(frames, columns)], atol=2e-15)
        rows_pass = (row_value >= lower.reshape(cap.shape)) & (row_value <= upper.reshape(cap.shape))
        audit_value = mb.chatter_excess(reference[:, qi], trial[:, qi])[np.ix_(frames, columns)]
        # Rows reserve 0.2 mrad inside the audit tolerance; nonelastic rows
        # additionally allow pre-existing excess, exactly as documented.
        allowance = cap - mb.chatter_budget(reference[:, qi])[np.ix_(frames, columns)]
        assert np.array_equal(rows_pass, audit_value <= allowance + 2e-15)
        if elastic and np.all(rows_pass):
            assert np.all(audit_value <= mb.CHATTER_EXCESS_TOLERANCE)
    # Returning all the way to the source satisfies both rows and the audit.
    step = (reference[:, qi] - candidate[:, qi]).ravel()
    assert np.all(matrix @ step >= lower - 1e-14)
    assert np.all(matrix @ step <= upper + 1e-14)


def test_chatter_reserve_excludes_only_the_audits_interior_margin():
    reference = np.zeros((19, 1))
    chatter = JointChatter(SimpleNamespace(qi=np.array([0])), reference)
    pulse = np.zeros((19, 1)); pulse[9] = 1.
    amplitude = np.max(np.abs(mb.high_frequency(pulse)))
    matrix, lower, upper = chatter.rows(reference, np.arange(19), [0], elastic=True)
    for margin, expected in [(2 * CHATTER_RESERVE, True), (CHATTER_RESERVE / 2, False)]:
        trial = pulse * (chatter.limit[9, 0] - margin) / amplitude
        assert np.max(chatter.excess(trial)) < 0
        values = matrix @ trial.ravel()
        assert bool(np.all((values >= lower) & (values <= upper))) == expected


def test_witness_lower_zero_feasible_and_clearance_margin():
    phi = np.array([.002, .0001, -.001, -.005])
    budgets = np.array([0., 0., .003, .005])
    lower = witness_lower(phi, budgets, np.zeros(4))
    assert np.all(lower <= 0.)
    assert_allclose(phi + lower, -np.maximum(budgets - CLEARANCE_MARGIN_M, -phi))
    assert_allclose(phi[:2] + lower[:2], [CLEARANCE_MARGIN_M, .0001])
    offset = np.array([.0002, -.0001, .001, -.001])
    assert_allclose(witness_lower(phi, budgets, offset) + offset, lower)


def test_cut_lower_zero_feasible_and_margin_at_shifted_origin():
    matrix = sparse.csr_matrix([[1., 2.], [-2., 1.], [1., 0.]])
    displacement = np.array([.001, -.0002])
    value = np.array([.002, -.002, -.005])
    budget = np.array([0., .003, .005])
    shifted_value = value - matrix @ displacement
    lower = cut_lower(matrix, value, budget, displacement)
    assert np.all(lower <= 1e-18)
    assert_allclose(lower, witness_lower(shifted_value, budget, np.zeros(3)), atol=1e-18)
    assert_allclose(shifted_value + lower,
                    -np.maximum(budget - CLEARANCE_MARGIN_M, -shifted_value))


def _synthetic():
    bounds = LinearBounds(3)
    bounds.add(sparse.eye(3), [-.15, -.2, -2.], [.2, .2, 2.])
    bounds.add([[0., 1., 0.]], .02, np.inf)
    bounds.add([[.1, .2, 0.]], -2., 2.)
    # First cone binds; second stays strictly inactive throughout the ball;
    # third is fixed and violated, so it must be ignored.
    bounds.norms.append((sparse.csr_matrix([[1., 0., 0.], [0., 1., 0.],
                                          [1., 0., 0.], [0., 1., 0.],
                                          [0., 0., 1.], [0., 0., 0.]]),
                         np.array([[.6, 0.], [.01, 0.], [2., 0.]]), np.array([.5, 2., 1.])))
    return bounds, sparse.eye(3, format="csc")[:, :2]


def test_correction_satisfies_complete_constraints_and_screening_preserves_optimum():
    bounds, projection = _synthetic()
    delta, status, info = solve(bounds, projection, .3)
    full, full_status, full_info = solve(bounds, projection, .3, screen=False)
    assert status == full_status == "Solved"
    assert delta is not None and full is not None
    assert_allclose(delta, full, atol=2e-8)
    assert_allclose(delta, [np.sqrt(.5**2 - .02**2) - .6, .02], atol=2e-8)
    assert _Problem(bounds, projection).feasible(delta, .3)
    assert info["rows"] < full_info["rows"]
    assert info["cones"] == 1 < full_info["cones"]
    assert info["constant_violations"] == 1


@pytest.mark.parametrize("screen", [False, True])
def test_constant_violated_rows_and_cones_do_not_make_correction_infeasible(screen):
    bounds = LinearBounds(2)
    bounds.add([[1., 0.], [0., 1.], [0., 0.]], [1e-5, .1, -np.inf], [np.inf, .2, -1.])
    bounds.norms.append((sparse.csr_matrix([[0., 1.], [0., 0.]]),
                         np.array([[2., 0.]]), np.array([1.])))
    delta, status, info = solve(bounds, sparse.csc_matrix([[1.], [0.]]), .002, screen=screen)
    assert status == "Solved"
    assert info["constant_violations"] == 3
    assert_allclose(delta, [1e-5], atol=2e-8)


def test_general_projection_uses_physical_radius_for_safe_screening():
    bounds = LinearBounds(2)
    bounds.add([[1., 0.], [0., 1.]], [.02, -.2], [.04, .2])
    projection = sparse.csc_matrix([[20., 0.], [0., .1]])
    screened, status, _ = solve(bounds, projection, .03)
    full, full_status, _ = solve(bounds, projection, .03, screen=False)
    assert status == full_status == "Solved"
    assert_allclose(screened, full, atol=1e-8)
    assert np.linalg.norm(projection @ screened) <= .03 + 1e-9
    assert (projection @ screened)[0] >= .02 - 1e-8


def test_micrometre_correction_with_binding_cone_matches_full_program():
    bounds = LinearBounds(2)
    bounds.add(sparse.eye(2), [-1e-3, 2e-6], [1e-3, 1e-3])
    bounds.norms.append((sparse.eye(2, format="csr"), np.array([[1e-3 + 3e-6, 0.]]),
                         np.array([1e-3])))
    projection = sparse.eye(2, format="csc")
    delta, status, _ = solve(bounds, projection, 2e-3)
    full, full_status, _ = solve(bounds, projection, 2e-3, screen=False)
    assert status in {"Solved", "AlmostSolved"} and full_status in {"Solved", "AlmostSolved"}
    assert_allclose(delta, full, atol=2e-8)
    assert_allclose(delta, [np.sqrt(1e-6 - 4e-12) - .001003, 2e-6], atol=2e-8)
    assert np.linalg.norm(delta + [.001003, 0.]) <= .001 + 1e-9
    assert delta[1] >= 2e-6 - 1e-9


def test_radius_screening_keeps_tangent_and_violated_constraints():
    bounds = LinearBounds(2)
    bounds.add([[1., 0.], [0., 1.]], [-.002, 1e-5], [.002, .1])
    problem = _Problem(bounds, sparse.eye(2, format="csc"))
    _, small = problem.program(.002, True)
    _, big = problem.program(.2, True)
    assert small["rows"] == 3 and big["rows"] == 4


def test_correction_retries_larger_radius_and_reserves_violated_rows():
    class Limits:
        coords = SimpleNamespace(qi=np.array([0]))
        frames = width = 1

        def constraints(self, trial, *, projection):
            bounds = LinearBounds(1)
            bounds.add([[1.]], .003, np.inf)
            return bounds

    ledger = Ledger()
    rows = (sparse.csr_matrix((0, 1)), np.zeros(0), np.zeros(0), [])
    result = correction(Limits(), np.zeros((1, 1)), np.zeros((1, 1)),
                        sparse.eye(1, format="csc"), rows, [], [], .01, ledger)
    assert result is not None
    assert_allclose(result, [[.003 + CORRECTION_RESERVE]], atol=1e-8)
    assert [c["radius"] for c in ledger.corrections] == [SMALL_CORRECTION_RADIUS, .002, .01]
    assert [c["status"] for c in ledger.corrections] == ["PrimalInfeasible", "PrimalInfeasible", "Solved"]


def test_solve_never_reads_wall_clock_and_handles_empty_projection(monkeypatch):
    from omniretargeting._fast import local_correction

    def forbidden_clock():
        raise AssertionError("solve must not read the wall clock")

    monkeypatch.setattr(local_correction.time, "monotonic", forbidden_clock)
    bounds, projection = _synthetic()
    delta, status, info = solve(bounds, projection, .3)
    assert delta is not None and status == "Solved"
    assert info["max_iter"] == MAX_ITERATIONS
    assert info["iterations"] <= MAX_ITERATIONS
    delta, status, _ = solve(bounds, projection[:, :0], .3)
    assert delta is None and status == "no_local_dofs"


def test_numerical_retry_preserves_trial_witness_cut_and_exact_linear_rows(monkeypatch):
    import clarabel
    from omniretargeting._fast import local_correction

    current = np.zeros((1, 2))
    trial = np.array([[-2e-6, 0.]])
    projection = sparse.eye(2, format="csc")

    class Limits:
        coords = SimpleNamespace(qi=np.arange(2))
        frames, width = 1, 2
        calls = 0

        def constraints(self, poses, *, projection):
            assert poses is trial
            assert projection.shape == (2, 2)
            self.calls += 1
            bounds = LinearBounds(2)
            bounds.add(sparse.eye(2), -.01, .01)
            return bounds

    original = clarabel.DefaultSolver
    calls = []

    def solver(*args):
        settings = args[-1]
        assert settings.max_iter == MAX_ITERATIONS
        assert settings.max_threads == 1 and settings.time_limit == np.inf
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("synthetic numerical failure")
        return original(*args)

    monkeypatch.setattr(clarabel, "DefaultSolver", solver)
    limits = Limits()
    rows = (sparse.csr_matrix([[1., 0.]]), np.array([-.005]), np.array([.005]), ["key"])
    cuts = [(sparse.csr_matrix([[0., 1.]]), np.array([-.005]), np.array([.005]), np.zeros(2))]
    linear = [(sparse.csr_matrix([[0., 1.]]), np.array([3e-6]), np.array([1e-4]))]
    # An arbitrarily slow clock changes only recorded seconds. A minimal
    # ledger works too, as used by local_rounds' independent window workers.
    for seconds in ([0., .1, .2, .3], [0., 1e6, 2e6, 3e6]):
        calls.clear()
        clock = iter(seconds)
        monkeypatch.setattr(local_correction.time, "monotonic", lambda: next(clock))
        ledger = SimpleNamespace(corrections=[])
        result = correction(limits, current, trial, projection, rows, cuts, linear, .01, ledger)
        assert result is not None
        assert_allclose(result, [[2e-6, 3e-6]], atol=2e-8)
        assert len(calls) == 2
        assert [entry["status"] for entry in ledger.corrections] == ["NumericalError", "Solved"]
        assert [entry["radius"] for entry in ledger.corrections] == [SMALL_CORRECTION_RADIUS, .002]
    assert limits.calls == 2


@pytest.mark.parametrize("closed", [False, True])
def test_optional_witness_band_only_adds_rows_without_changing_depths(closed):
    owner = SimpleNamespace(sample_keys={})
    world = np.arange(12, dtype=float).reshape(4, 3)
    signed = dict(uncertain=np.zeros(4, bool), clearance=np.array([-.005, -.004, -.002, .02]),
                  directions=np.tile([0., 0., 1.], (4, 1)))
    values = (world, None, {0: (np.arange(4), signed)})
    args = (0, {"closed": {0} if closed else set()}, 7, values, np.eye(3), 1., .01)
    original = summarize(owner, *args)
    assert len(original["rows"]) == 1
    assert original["depths"] == [(0, closed, .005)]
    point, direction, clearance, identity, is_closed, cid = original["rows"][0]
    assert_allclose(point, world[0]); assert_allclose(direction, [0., 0., 1.])
    assert (clearance, identity, is_closed, cid) == (-.005, 0, closed, 0)
    owner.witness_band = None
    disabled = summarize(owner, *args)
    assert len(disabled["rows"]) == 1
    for old, new in zip(original["rows"][0], disabled["rows"][0]):
        assert_allclose(old, new)
    owner.witness_band = .003
    expanded = summarize(owner, *args)
    assert len(expanded["rows"]) == 3
    for key in original.keys() - {"rows"}:
        assert expanded[key] == original[key] == disabled[key]
    assert [row[3] for row in expanded["rows"]] == [0, 1, 2]


@pytest.mark.parametrize("status", ["Solved", "AlmostSolved", "MaxIterations", "InsufficientProgress"])
@pytest.mark.parametrize("kind,tolerance", [("linear", 1e-8), ("cone", 1e-9), ("trust", 1e-9)])
@pytest.mark.parametrize("factor,accepted", [(0.5, True), (2., False)])
def test_correction_status_requires_physical_residuals(monkeypatch, status, kind, tolerance, factor, accepted):
    import clarabel

    radius = .002
    bounds = LinearBounds(1)
    bounds.add(sparse.csr_matrix((0, 1)), -np.inf, np.inf)
    if kind == "linear":
        bounds.add([[1.]], -np.inf, 0.)
        step = tolerance * factor
    elif kind == "cone":
        bounds.norms.append((sparse.csr_matrix([[1.]]), np.array([[.001]]), np.array([.001])))
        step = tolerance * factor
    else:
        step = radius + tolerance * factor
    result = SimpleNamespace(status=status, iterations=MAX_ITERATIONS, x=[step / 1e-4])
    monkeypatch.setattr(clarabel, "DefaultSolver", lambda *args: SimpleNamespace(solve=lambda: result))
    delta, returned_status, info = solve(bounds, sparse.eye(1, format="csc"), radius)
    assert returned_status == status
    assert info["linearized_feasible"] == accepted
    assert (delta is not None) == accepted
    residual_key = {"linear": "linear_error", "cone": "cone_error_absolute", "trust": "trust_error"}[kind]
    assert_allclose(info[residual_key], tolerance * factor, atol=1e-18)


@pytest.mark.parametrize("always_infeasible", [False, True])
def test_inexact_correction_retries_or_returns_none(monkeypatch, always_infeasible):
    import clarabel

    class Limits:
        coords = SimpleNamespace(qi=np.array([0]))
        frames = width = 1

        def constraints(self, trial, *, projection):
            bounds = LinearBounds(1)
            bounds.add([[1.]], 3e-6, np.inf)
            return bounds

    calls = []

    def solver(*args):
        calls.append(args)
        # Finite, accepted status, but the first candidate violates the row.
        step = 0. if always_infeasible or len(calls) == 1 else 3e-6 + CORRECTION_RESERVE
        result = SimpleNamespace(status="MaxIterations", iterations=MAX_ITERATIONS, x=[step / 1e-4])
        return SimpleNamespace(solve=lambda: result)

    monkeypatch.setattr(clarabel, "DefaultSolver", solver)
    ledger = SimpleNamespace(corrections=[])
    rows = (sparse.csr_matrix((0, 1)), np.zeros(0), np.zeros(0), [])
    result = correction(Limits(), np.zeros((1, 1)), np.zeros((1, 1)),
                        sparse.eye(1, format="csc"), rows, [], [], .01, ledger)
    radii = [SMALL_CORRECTION_RADIUS, .002, .01] if always_infeasible else [SMALL_CORRECTION_RADIUS, .002]
    assert [entry["radius"] for entry in ledger.corrections] == radii
    assert [entry["linearized_feasible"] for entry in ledger.corrections] == (
        [False, False, False] if always_infeasible else [False, True])
    if always_infeasible:
        assert result is None
    else:
        assert_allclose(result, [[3e-6 + CORRECTION_RESERVE]])


def test_tiny_radius_screens_most_linear_rows():
    bounds = LinearBounds(20)
    bounds.add(sparse.eye(20), -.001, .001)
    bounds.add(sparse.csr_matrix(([1.], ([0], [0])), shape=(1, 20)), 3e-6, np.inf)
    problem = _Problem(bounds, sparse.eye(20, format="csc"))
    _, small = problem.program(SMALL_CORRECTION_RADIUS, True)
    _, larger = problem.program(.002, True)
    assert small["rows"] == 1
    assert larger["rows"] == 40  # Duplicate coefficient rows merge their bounds.


def test_correction_checks_linear_rows_in_original_units_after_equilibration():
    bounds = LinearBounds(1)
    bounds.add([[1.]], -2., 2.)
    problem = _Problem(bounds, sparse.eye(1, format="csc"))
    info = {}
    assert not problem.feasible(np.array([2. + 1.5e-8]), 3., info)
    assert info["linear_error"] < 1e-8
    assert info["linear_error_absolute"] > 1e-8


@pytest.fixture
def pipeline_patch():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "out/unified_local_20261006/harness/apply_pipeline_patch.py"
    spec = importlib.util.spec_from_file_location("apply_pipeline_patch_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, script, module


def _copy_pipeline_pair(tmp_path, root, source="baseline"):
    import shutil

    origin = (root / "out/unified_local_20261006/ab/baseline_source/omniretargeting/_fast"
              if source == "baseline" else root / "omniretargeting/_fast")
    fast = tmp_path / "omniretargeting/_fast"
    fast.mkdir(parents=True)
    paths = [fast / name for name in ("pipeline.py", "runtime_options.py")]
    for path in paths:
        shutil.copy2(origin / path.name, path)
    return fast, paths


@pytest.mark.parametrize("source", ["baseline", "workspace"])
def test_pipeline_patch_on_copies_preserves_reweight_and_is_idempotent(tmp_path, pipeline_patch, monkeypatch, source):
    import runpy
    import subprocess
    import sys

    root, script, module = pipeline_patch
    fast, paths = _copy_pipeline_pair(tmp_path, root, source)
    original_pipeline = paths[0].read_text()
    completed = subprocess.run([sys.executable, str(script), str(fast)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    patched = paths[0].read_text()
    assert patched.count("from .repair_sqp import repair_original_meshes_sqp\n") == 1
    assert patched.count("repair = repair_original_meshes if OPTIONS.legacy_repair else repair_original_meshes_sqp") == 1
    assert patched.count("for round_index in range(REWEIGHT_ROUNDS):") == 1
    assert "replaced_by" not in patched
    # The complete remainder, including other sessions' changes, is identical.
    for old, new in module.PIPELINE_REPLACEMENTS:
        patched = patched.replace(new, old)
    assert patched == original_pipeline
    monkeypatch.setenv("OMNIRETARGETING_FAST_ENABLE", "legacy_repair,performance_trace")
    options = runpy.run_path(str(paths[1]))["OPTIONS"]
    assert options.legacy_repair and options.performance_trace
    monkeypatch.setenv("OMNIRETARGETING_FAST_ENABLE", "")
    assert not runpy.run_path(str(paths[1]))["OPTIONS"].legacy_repair
    contents = [path.read_bytes() for path in paths]
    timestamps = [path.stat().st_mtime_ns for path in paths]
    repeated = subprocess.run([sys.executable, str(script), str(fast)], capture_output=True, text=True)
    assert repeated.returncode == 0, repeated.stderr
    assert "already fully applied" in repeated.stdout
    assert [path.read_bytes() for path in paths] == contents
    assert [path.stat().st_mtime_ns for path in paths] == timestamps
    assert set(fast.iterdir()) == set(paths)


@pytest.mark.parametrize("anchor", [0, 1, 2])
@pytest.mark.parametrize("invalid", ["missing", "duplicate"])
def test_pipeline_patch_validates_all_options_before_writing(tmp_path, pipeline_patch, anchor, invalid):
    root, _, module = pipeline_patch
    fast, paths = _copy_pipeline_pair(tmp_path, root)
    old, _ = module.OPTIONS_REPLACEMENTS[anchor]
    original = paths[1].read_text()
    paths[1].write_text(original.replace(old, "") if invalid == "missing" else original + old)
    contents = [path.read_bytes() for path in paths]
    with pytest.raises(SystemExit, match="anchor"):
        module.main(fast)
    assert [path.read_bytes() for path in paths] == contents
    assert set(fast.iterdir()) == set(paths)


@pytest.mark.parametrize("partial", ["pipeline", "options", "import", "field", "old_reweight"])
def test_pipeline_patch_rejects_partial_pairs_without_writes(tmp_path, pipeline_patch, partial):
    root, _, module = pipeline_patch
    fast, paths = _copy_pipeline_pair(tmp_path, root)
    if partial in {"pipeline", "import", "old_reweight"}:
        source, _ = module.prepare(paths[0].read_text(), module.PIPELINE_REPLACEMENTS)
        if partial == "import":
            old, new = module.PIPELINE_REPLACEMENTS[1]
            source = source.replace(new, old)
        paths[0].write_text(source)
    if partial in {"options", "field", "old_reweight"}:
        replacements = module.OPTIONS_REPLACEMENTS[:1] if partial == "field" else module.OPTIONS_REPLACEMENTS
        source, _ = module.prepare(paths[1].read_text(), replacements)
        paths[1].write_text(source)
    if partial == "old_reweight":
        paths[0].write_text(paths[0].read_text().replace(
            "range(REWEIGHT_ROUNDS):", "range(REWEIGHT_ROUNDS if OPTIONS.legacy_repair else 0):"))
    contents = [path.read_bytes() for path in paths]
    with pytest.raises(SystemExit, match="partial application"):
        module.main(fast)
    assert [path.read_bytes() for path in paths] == contents
    assert set(fast.iterdir()) == set(paths)


def test_options_comment_does_not_count_as_applied_field(tmp_path, pipeline_patch):
    root, _, module = pipeline_patch
    fast, paths = _copy_pipeline_pair(tmp_path, root)
    paths[1].write_text(paths[1].read_text() + "\n# legacy_repair is planned\n")
    module.main(fast)
    assert "    legacy_repair: bool = False\n" in paths[1].read_text()


def test_pipeline_patch_rolls_back_if_second_rename_fails(tmp_path, pipeline_patch, monkeypatch):
    root, _, module = pipeline_patch
    fast, paths = _copy_pipeline_pair(tmp_path, root)
    contents = [path.read_bytes() for path in paths]
    real_replace = module.os.replace

    def replace(source, destination):
        if destination == paths[1]:
            raise OSError("synthetic second rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", replace)
    with pytest.raises(OSError, match="second rename"):
        module.main(fast)
    assert [path.read_bytes() for path in paths] == contents
    assert set(fast.iterdir()) == set(paths)
