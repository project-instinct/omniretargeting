"""Every omitted scalar/SOC witness is separated against the full local QP."""

import numpy as np
from scipy import sparse

from omniretargeting._fast.qp_assembly import SoftTerms, assemble_program, quadratic
from omniretargeting._fast.qp_guard_rows import intervals, norms
from omniretargeting._fast.qp_types import solve_program
from omniretargeting._fast.window_screen import solve_working_set, initial_working_set, violated_rows
from tests.test_fast_qp_assembly import small_model


def test_omitted_scalar_constraint_is_added_and_matches_full_objective():
    model = small_model(gradient=-2.)
    model.program = assemble_program([intervals(sparse.eye(1), [-1.], [.2])], model.soft, 1)
    mandatory = np.zeros(len(model.program.b), bool)
    assert not initial_working_set(model, mandatory)[0].any()
    P, c = quadratic(model, .001)
    result, trace = solve_working_set(model, P, c, mandatory)
    full = solve_program(model.program, P, c)
    assert len(trace) == 2 and trace[0]['violated_linear'] == 1
    np.testing.assert_allclose(result[0], full.x, atol=1e-7)


def test_crossing_norm_is_added_as_a_complete_cone_with_fixed_residual():
    model = small_model(gradient=-2.)
    block = norms(sparse.csr_matrix([[1.], [0.]]), np.array([[0., .3]]), [.5])
    model.program = assemble_program([block], model.soft, 1)
    P, c = quadratic(model, .001)
    result, trace = solve_working_set(model, P, c, np.zeros(3, bool))
    assert len(trace) == 2 and trace[0]['violated_cones'] == 1
    assert trace[1]['rows'] == 3 and trace[1]['cones'] == 1
    np.testing.assert_allclose(result[0], [.4], atol=1e-7)


def test_shared_epigraph_worst_witness_switch_keeps_the_full_soft_cost():
    key = ('penetration', 0, 1)
    model = small_model([(sparse.csr_matrix([[1.], [-1.]]), np.array([-.1, .5]),
                          np.ones(2), [key, key])], gradient=2.)
    P, c = quadratic(model, .001)
    c[-1] = 10.
    result, trace = solve_working_set(model, P, c, np.zeros(len(model.program.b), bool))
    full = solve_program(model.program, P, c)
    assert len(trace) > 1
    np.testing.assert_allclose(result[0], full.x, atol=1e-6)
    assert not any(v.any() for v in violated_rows(model.program, result[0]))


def test_impossible_constant_norm_is_never_omitted_or_returned_feasible():
    model = small_model()
    model.program = assemble_program([norms(sparse.csr_matrix((2, 1)),
                                           np.array([[2., 0.]]), [1.])], model.soft, 1)
    P, c = quadratic(model, .001)
    mandatory = np.zeros(3, bool)
    assert initial_working_set(model, mandatory)[1].all()
    result, trace = solve_working_set(model, P, c, mandatory)
    assert result is None


def test_mandatory_rows_are_retained_even_far_from_activity():
    model = small_model()
    mandatory = np.ones(len(model.program.b), bool)
    assert initial_working_set(model, mandatory)[0].all()


def test_precision_refines_the_same_subset_instead_of_loading_all_rows(monkeypatch):
    from types import SimpleNamespace
    from omniretargeting._fast import window_screen
    model = small_model(gradient=-2.)
    model.program = assemble_program([intervals(sparse.csr_matrix([[1.], [1.]]),
                                               [-np.inf, -np.inf], [.2, 10.])], model.soft, 1)
    P, c = quadratic(model, .001)
    calls = []
    def solve(program, P, c, *, precision=False):
        calls.append((len(program.b), precision))
        return SimpleNamespace(x=np.array([.2 + (0. if precision else 2e-8)]),
                               z=np.zeros(len(program.b)))
    monkeypatch.setattr(window_screen, 'solve_program', solve)
    result, trace = solve_working_set(model, P, c, np.array([True, False]))
    assert result is not None and calls == [(1, False), (1, True)]
    assert trace[1]['precision_refinement'] and not any(row['full_program'] for row in trace)


def test_candidate_near_rows_and_damping_active_set_are_retained(monkeypatch):
    from omniretargeting._fast import window_screen
    from types import SimpleNamespace
    from dataclasses import replace
    model = small_model(gradient=-2.)
    model.program = assemble_program([intervals(sparse.csr_matrix([[1.], [1.], [1.]]),
                                               [-np.inf] * 3, [.2, .30005, 10.])], model.soft, 1)
    P, c = quadratic(model, .001)
    def solve(program, P, c, *, precision=False):
        point = .3 if not len(program.b) else .2
        return SimpleNamespace(x=np.array([point]), z=np.zeros(len(program.b)))
    monkeypatch.setattr(window_screen, 'solve_program', solve)
    mandatory = np.zeros(len(model.program.b), bool)
    _, first = solve_working_set(model, P, c, mandatory)
    assert first[0]['added_linear'] == 2 and first[1]['rows'] == 2
    _, second = solve_working_set(model, 2 * P, c, mandatory)
    assert len(second) == 1 and second[0]['rows'] == 2 and second[0]['reused_working_set']
    changed = replace(model, program=replace(model.program, b=model.program.b + .01))
    _, third = solve_working_set(changed, P, c, mandatory)
    assert not third[0]['reused_working_set'] and third[0]['rows'] == 0


def test_loose_contact_keeps_one_tightest_representative_not_every_sample():
    key = ('contact', 'support', 0, 1)
    model = small_model([(sparse.csr_matrix([[1.], [1.], [1.]]), np.array([.3, .2, .4]),
                          np.ones(3), [key] * 3)])
    linear, _ = initial_working_set(model, np.zeros(len(model.program.b), bool))
    np.testing.assert_array_equal(linear[2:5], [False, True, False])


def test_final_generation_adds_last_witness_before_full_fallback(monkeypatch):
    from omniretargeting._fast import window_screen
    from types import SimpleNamespace
    model = small_model()
    model.program = assemble_program([intervals(sparse.eye(1), [-np.inf], [.2])], model.soft, 1)
    P, c = quadratic(model, .001)
    def solve(program, P, c, **kwargs):
        return SimpleNamespace(x=np.array([.2 if len(program.b) else .3]),
                               z=np.zeros(len(program.b)))
    monkeypatch.setattr(window_screen, 'solve_program', solve)
    result, trace = solve_working_set(model, P, c, np.zeros(1, bool), max_generations=1)
    assert result is not None and len(trace) == 2
    assert trace[-1]['fallback_reason'] is None
