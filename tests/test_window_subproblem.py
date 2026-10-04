"""Unused slack elimination preserves the exact screened primal/dual model."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.qp_types import solve_program
from omniretargeting._fast.qp_assembly import quadratic
from omniretargeting._fast.window_subproblem import solve_subset
from omniretargeting._fast.window_screen import _subset
from tests.test_fast_qp_assembly import small_model


def problem():
    key = ('contact', 'support', 0, 1)
    model = small_model([(sparse.csr_matrix([[1.]]), np.array([-.1]), np.ones(1), [key])])
    P, c = quadratic(model, .001)
    c[-1] = 10.
    program, _ = _subset(model.program, np.array([True, True, False, True]), np.zeros(0, bool))
    return program, P, c


def test_unused_slack_primal_dual_lift_satisfies_original_kkt():
    program, P, c = problem()
    full = solve_program(program, P, c, precision=True)
    reduced = solve_subset(program, P, c, solve_program, precision=True)
    assert program.last_solve['eliminated_free_epigraphs'] == 1
    np.testing.assert_allclose(reduced.x, full.x, atol=1e-6)
    assert reduced.x[-1] == 0. and reduced.z[-1] == c[-1]
    assert np.max(abs(P @ reduced.x + c + program.A.T @ reduced.z)) < 1e-8
    assert np.min(program.b - program.A @ reduced.x) > -1e-8
    assert abs((program.b - program.A @ reduced.x) @ reduced.z) < 1e-8


@pytest.mark.parametrize('mutation', ['witness', 'quadratic', 'negative_cost', 'other_bound', 'coupled_row'])
def test_unknown_or_coupled_slack_use_is_never_eliminated(mutation):
    program, P, c = problem()
    matrix = program.A.tolil()
    if mutation == 'witness':
        matrix[0, -1] = -1.
    elif mutation == 'quadratic':
        P = P + sparse.diags([0., 1.])
    elif mutation == 'negative_cost':
        c[-1] = -1.
    elif mutation == 'other_bound':
        program.b[-1] = .1
    else:
        matrix[-1, 0] = .1
    program = replace(program, A=matrix.tocsc())
    def solve(work, P, c, **kwargs):
        assert work.A.shape == program.A.shape
        return SimpleNamespace(x=np.zeros(work.A.shape[1]), z=np.zeros(len(work.b)))
    solve_subset(program, P, c, solve)
    assert program.last_solve['eliminated_free_epigraphs'] == 0
