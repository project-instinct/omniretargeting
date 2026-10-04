"""Fundable-contact proposal caps preserve per-channel scalar/SOC domains."""

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.qp_assembly import assemble_program, quadratic
from omniretargeting._fast.qp_guard_rows import intervals, norms
from omniretargeting._fast.window_assembly import prepare_window_model
from omniretargeting._fast.window_screen import solve_working_set, violated_rows
from tests.test_fast_qp_assembly import small_model
from tests.test_window_driver import problem


def test_shared_upper_uses_max_cap_and_keeps_each_channel_hard_cap():
    key = ('contact', 'support', 0, 1)
    matrix = sparse.csr_matrix([[1.], [0.], [0.]])
    model = small_model([(sparse.csr_matrix([[0.]]), np.array([.5]), np.ones(1), [key])],
                        [(matrix, np.zeros((1, 3)), np.array([.1]), np.ones(1), [key])])
    model.soft.add_limits({key: .4})  # max(.1, .4), not min.
    boxes = intervals(sparse.eye(1), [-1.], [1.])
    scalar_cap = intervals(sparse.csr_matrix([[0.]]), [-np.inf], [.6])
    norm_cap = norms(matrix, np.zeros((1, 3)), [.5])
    model.program = assemble_program([boxes, scalar_cap, norm_cap], model.soft, 1)
    P, c = quadratic(model, .001)
    c[-1] = .1
    result, _ = solve_working_set(model, P, c, np.ones(len(model.program.b), bool))
    assert result is not None and result[0][0] == pytest.approx(.5, abs=1e-7)
    assert result[0][-1] == pytest.approx(.4, abs=1e-7)
    assert not any(mask.any() for mask in violated_rows(model.program, result[0]))
    # Tightening a single physical channel must bind even if the shared
    # epigraph would otherwise admit a larger displacement.
    model.program = assemble_program([boxes, scalar_cap, norms(matrix, np.zeros((1, 3)), [.2])], model.soft, 1)
    result, _ = solve_working_set(model, P, c, np.ones(len(model.program.b), bool))
    assert result[0][0] == pytest.approx(.2, abs=1e-7)


def test_epigraph_upper_is_mandatory_and_labelled_in_window_assembly():
    objective, poses = problem()
    key = ('contact', 'support', 0, 2)
    objective.priority_clearance.soft_limits = lambda p, *, frames: {key: .01}
    model, mandatory = prepare_window_model(objective, poses, range(0, 4))
    index = model.soft._index[key]
    assert model.soft.limits[index] == .01
    source = model.window_row_sources
    cap_rows = source['codes'] == source['names'].index('epigraph_upper_limits')
    assert cap_rows.sum() == 1 and mandatory[cap_rows].all()
    np.testing.assert_array_equal(model.program.b[cap_rows], [.01])
    assert model.program.A[cap_rows, model.gradient.size + index].toarray().item() == 1.


@pytest.mark.parametrize('bound', [-.1, np.inf, np.nan, [1.]])
def test_invalid_cap_fails_closed(bound):
    key = ('contact', 'support', 0, 1)
    model = small_model([(sparse.csr_matrix([[1.]]), np.array([0.]), np.ones(1), [key])])
    with pytest.raises(ValueError, match='finite nonnegative'):
        model.soft.add_limits({key: bound})


def test_unknown_key_rejected_and_empty_limits_leave_original_assembly_unchanged():
    model = small_model()
    original = model.program
    model.soft.add_limits({})
    rebuilt = assemble_program(original.hard_blocks, model.soft, 1)
    np.testing.assert_array_equal(original.A.toarray(), rebuilt.A.toarray())
    np.testing.assert_array_equal(original.b, rebuilt.b)
    with pytest.raises(ValueError, match='unknown key'):
        model.soft.add_limits({('contact', 'support', 0, 1): .1})
