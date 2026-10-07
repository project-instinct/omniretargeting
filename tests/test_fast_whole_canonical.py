"""Whole acquisition must retain solver-input bytes and nonlinear decisions."""


import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import shape_step, whole_canonical
from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.continuity import CorrectionContinuity
from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._fast.window_canonical import solve_data
from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from omniretargeting.conic_solver import solve_conic, solve_conic_working_set
from tests.test_fast_backend import _objective
from tests.test_fast_window_canonical import assert_program_equal


def problem(acquired=False, legacy=False):
    obj, poses, _ = _objective(temporal=True, extra=True)
    obj.source_shape = ReferenceShape(obj, poses)
    if acquired:
        reference = obj.coords.integrate(poses, np.random.default_rng(5).normal(
            scale=.001, size=(len(poses), obj.width)), 1.)
        obj.source_shape.freeze_reference(reference)
    sequence = SequenceContinuityGuard(obj.tasks, poses, obj.root_rotations,
                                       chatter=True, enforce_chatter=False)
    # Match the whole scalar-joint velocity rows, with several merged orders.
    sequence.bounds = {('joints', order): np.full((len(poses) - order, len(obj.coords.qi)), .1)
                       for order in (1, 2, 3)}
    obj.candidate_guards = [AcquisitionNormalGuard(obj, poses), sequence,
                           CorrectionContinuity(obj.coords, poses, root_norm=legacy,
                                                chatter=not legacy, enforce_chatter=False)]
    return obj, poses


@pytest.mark.parametrize('amplitude', [0., .001, .05])
@pytest.mark.parametrize('acquired,legacy,fixed', [(False, False, False),
                                                 (True, True, False), (True, False, True)])
def test_whole_coefficients_and_solutions_are_bitwise_equal(amplitude, acquired, legacy, fixed):
    obj, origin = problem(acquired, legacy)
    if fixed:
        obj.fixed_root = origin[:, :7].copy()
    poses = obj.coords.integrate(origin, np.random.default_rng(11).normal(
        scale=amplitude, size=(len(origin), obj.width)), 1.)
    lo, hi = obj.coords.bounds(poses, obj.cfg['angular_step_rad'], obj.cfg['translation_step_m'])
    shape_step._pin_root(obj, poses, lo, hi)
    lo[:, obj.coords.scalar_cols] = np.maximum(lo[:, obj.coords.scalar_cols], obj.lower - poses[:, obj.coords.qi])
    hi[:, obj.coords.scalar_cols] = np.minimum(hi[:, obj.coords.scalar_cols], obj.upper - poses[:, obj.coords.qi])
    count = lo.size
    rng = np.random.default_rng(12)
    # Include temporal Hessian bands, rather than a diagonal-only objective.
    matrix = sparse.diags([rng.uniform(2., 3., count), np.full(count-1, -.2),
                           np.full(count-1, -.2)], [0, -1, 1], format='csc')
    gradient = rng.normal(size=count)
    scale = max(float(np.linalg.norm(gradient, ord=np.inf)), 1.)
    flat = cp.Variable(count)
    constraints = [flat >= lo.ravel(), flat <= hi.ravel()]
    box = np.maximum(np.abs(lo), np.abs(hi))
    constraints += obj.source_shape.constraints(poses, flat, box)
    constraints += shape_step._guard_constraints(obj.candidate_guards, poses, flat)
    objective = cp.Minimize((cp.quad_form(flat, cp.psd_wrap(matrix)) + 2 * gradient @ flat) / scale)
    reference = cp.Problem(objective, constraints)
    expected, _, _ = reference.get_problem_data(cp.CLARABEL)
    actual = whole_canonical.canonical_data(obj, poses, lo, hi, matrix * (2 * (1 / scale)),
                                            (2 * gradient) * (1 / scale))
    assert_program_equal(expected, actual)
    for working_set, solve in ((False, solve_conic), (True, solve_conic_working_set)):
        reference = cp.Problem(objective, constraints)
        try:
            solve(reference, solver=cp.CLARABEL, direct_solve_method='qdldl')
        except cp.error.SolverError:
            expected_step = None
        else:
            expected_step = flat.value if reference.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) else None
        step = solve_data(actual, count, working_set=working_set)
        if expected_step is None:
            assert step is None
        else:
            assert step is not None
            assert step.tobytes() == expected_step.tobytes()




def test_unknown_guards_and_non_scalar_sequence_rows_keep_cvxpy(monkeypatch):
    pass
    obj, poses = problem()
    assert whole_canonical.supported(obj)
    from omniretargeting._fast.guard_signals import prepare
    prepare(obj)
    assert whole_canonical.supported(obj)
    obj.candidate_guards.append(object())
    assert not whole_canonical.supported(obj)
    obj.candidate_guards.pop()
    obj.candidate_guards[1].bounds['root_offset', 1] = np.ones(len(poses)-1)
    assert not whole_canonical.supported(obj)


def test_selected_layout_rows_retain_explicit_zeros_and_order():
    from omniretargeting.pose_relations import _block_matrix
    rng = np.random.default_rng(8)
    blocks = rng.normal(size=(7, 19, 13))
    blocks[rng.uniform(size=blocks.shape) < .4] = 0.
    for mask in (np.zeros(7 * 19, bool), np.ones(7 * 19, bool),
                 rng.uniform(size=7 * 19) < .15):
        expected = _block_matrix(blocks, 13)[mask]
        actual = _block_matrix(blocks, 13, mask)
        assert actual.shape == expected.shape
        assert actual.data.tobytes() == expected.data.tobytes()
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.indptr, expected.indptr)


def test_long_clips_keep_the_established_canonical_path(monkeypatch):
    pass
    obj, _ = problem()
    obj.source_shape.seed = np.zeros((600, obj.model.nq))
    assert not whole_canonical.supported(obj)
