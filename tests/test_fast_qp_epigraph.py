"""Unused soft variables can be removed only with a complete original-QP lift."""

from dataclasses import replace

import numpy as np
from scipy import sparse

from omniretargeting._fast.priority_qp import _freeze
from omniretargeting._fast.qp_assembly import quadratic
from omniretargeting._fast.qp_epigraph import eliminate_unused_epigraphs
from omniretargeting._fast.qp_residual import certified_solution
from omniretargeting._fast.qp_types import solve_program
from tests.test_fast_qp_assembly import small_model


def _case():
    key = ('contact', 'follow', 0, 1)
    rows = [(sparse.csr_matrix([[-1.], [1.]]), np.array([-.4, .1]), np.ones(2), [key] * 2)]
    cones = [(sparse.csr_matrix([[1.], [0.], [0.]]), np.array([[0., .2, 0.]]),
              np.array([.1]), np.ones(1), [key])]
    model = small_model(rows, cones)
    P, c = quadratic(model, .001)
    return model.program, P, c


def test_unused_shared_scalar_soc_epigraph_lifts_full_original_certificate():
    program, P, c = _case()
    reduction = eliminate_unused_epigraphs(program, P, c)
    assert reduction is not None and len(reduction.removed) == 1
    assert reduction.program.soc == () and reduction.program.A.shape == (2, 1)
    cols = reduction.columns
    result = solve_program(reduction.program, P[cols][:, cols], c[cols])
    lifted = reduction.lift(result)
    assert certified_solution(program, P, c, lifted)
    assert lifted.x[1] >= max(program.soft.values(lifted.x[:1]).values())
    np.testing.assert_array_equal(lifted.z[np.setdiff1d(np.arange(len(lifted.z)), reduction.rows)], 0.)


def test_freeze_or_objective_use_prevents_epigraph_elimination():
    program, P, c = _case()
    row = np.array([0., 1.])
    frozen = _freeze(program, row, .5)
    assert eliminate_unused_epigraphs(frozen, P, c) is None
    assert eliminate_unused_epigraphs(program, P, c + row) is None
    assert eliminate_unused_epigraphs(program, P + sparse.diags(row), c) is None
    # Off-diagonal quadratic incidence is also a use, even at zero diagonal.
    assert eliminate_unused_epigraphs(program, sparse.csc_matrix([[0., 1.], [0., 0.]]), c) is None


def test_unrecognized_hard_or_soc_tail_incidence_prevents_elimination():
    program, P, c = _case()
    for at in (0, program.nonneg + 1):
        A = program.A.tolil()
        A[at, 1] = -1.
        assert eliminate_unused_epigraphs(replace(program, A=A.tocsc()), P, c) is None


def test_freeze_reindexes_soft_cone_heads_without_mutating_previous_program():
    program, P, c = _case()
    previous = [head.copy() for head, _ in program.soft.heads]
    frozen = _freeze(program, np.array([1., 0.]), .9)
    for old, (new, _) in zip(previous, frozen.soft.heads):
        np.testing.assert_array_equal(new, old + (old >= program.nonneg))
    for old, (current, _) in zip(previous, program.soft.heads):
        np.testing.assert_array_equal(old, current)
    reduction = eliminate_unused_epigraphs(frozen, P, c)
    assert reduction is not None
    assert reduction.program.A.shape == (3, 1)
