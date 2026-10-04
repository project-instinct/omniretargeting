"""Screening removes only proved redundant domains or identical whole norms."""

import numpy as np
from scipy import sparse

from omniretargeting._fast.qp_assembly import assemble_program, quadratic
from omniretargeting._fast.qp_guard_rows import intervals, norms
from omniretargeting._fast.window_redundancy import box_redundancy, duplicate_cones
from omniretargeting._fast.window_screen import solve_working_set, violated_rows
from tests.test_fast_qp_assembly import small_model


def with_blocks(model, *blocks):
    box = intervals(sparse.eye(1), model.step_lower.ravel(), model.step_upper.ravel())
    model.program = assemble_program([box, *blocks], model.soft, 1)
    return model


def test_scalar_box_proof_retains_defining_box_and_roundoff_boundary():
    block = intervals(sparse.csr_matrix([[1.], [1.], [-1.]]), [-np.inf] * 3,
                      [2., np.nextafter(1., np.inf), 2.])
    model = with_blocks(small_model(), block)
    linear, cones, protected = box_redundancy(model)
    np.testing.assert_array_equal(linear, [False, False, True, False, True])
    assert protected[:2].all() and not cones.any()


def test_unenforced_or_nonfinite_metadata_box_cannot_prove_redundancy():
    model = small_model()
    model.program = assemble_program([norms(sparse.csr_matrix([[1.], [0.]]),
                                           np.array([[0., .3]]), [2.])], model.soft, 1)
    assert not any(mask.any() for mask in box_redundancy(model))
    model = small_model(lower=-np.inf)
    assert not any(mask.any() for mask in box_redundancy(model))


def test_soft_rows_and_norms_prove_only_with_nonnegative_slack_and_full_tail_box():
    key = ('contact', 'support', 0, 1)
    model = small_model([(sparse.csr_matrix([[1.]]), np.array([2.]), np.ones(1), [key])],
                        [(sparse.csr_matrix([[1.], [0.], [0.]]), np.array([[0., .3, 0.]]),
                          np.array([2.]), np.ones(1), [key])])
    linear, cones, protected = box_redundancy(model)
    assert linear[2] and cones[0] and protected[-1]
    matrix = model.program.A.tolil()
    matrix[-1, -1] = 1.  # Slack in a norm tail is unbounded: no proof.
    model.program.A = matrix.tocsc()
    assert not box_redundancy(model)[1][0]


def test_box_proof_matches_dense_corners_and_shared_slack_samples():
    key = ('penetration', 0, 1)
    model = small_model([(sparse.csr_matrix([[2.], [-3.], [1.]]), np.array([3., 4., .9]),
                          np.ones(3), [key] * 3)])
    linear, _, _ = box_redundancy(model)
    assert linear[2:4].all() and not linear[4]
    for step in np.linspace(-1., 1., 21):
        for slack in (0., .1, 100.):
            residual = model.program.b - model.program.A @ np.array([step, slack])
            assert (residual[:model.program.nonneg][linear] > 0.).all()


def test_duplicate_norm_uses_complete_affine_data_and_tightest_bound():
    first = norms(sparse.csr_matrix([[1.], [0.]]), np.array([[.2, .3]]), [.5])
    loose = norms(sparse.csr_matrix([[1.], [0.]]), np.array([[.2, .3]]), [.8])
    near = norms(sparse.csr_matrix([[1.], [0.]]),
                 np.array([[np.nextafter(.2, np.inf), .3]]), [.5])
    model = with_blocks(small_model(gradient=-2.), loose, first, near)
    np.testing.assert_array_equal(duplicate_cones(model.program), [1, 1, 2])
    P, c = quadratic(model, .001)
    result, trace = solve_working_set(model, P, c, np.zeros(len(model.program.b), bool))
    assert result is not None and trace[0]['duplicate_cones'] == 1
    assert not any(mask.any() for mask in violated_rows(model.program, result[0]))
    assert all(record['cones'] <= 2 for record in trace)


def test_shared_epigraph_does_not_make_different_sample_norms_duplicates():
    key = ('contact', 'support', 0, 1)
    def cone(residual, key):
        return sparse.csr_matrix([[1.], [0.], [0.]]), np.array([residual]), np.ones(1), np.ones(1), [key]
    model = small_model(cones=[cone([.1, .2, 0.], key), cone([.1, .2, 0.], key),
                               cone([.1, .3, 0.], key), cone([.1, .2, 0.], ('contact', 'follow', 0, 1))])
    np.testing.assert_array_equal(duplicate_cones(model.program), [0, 0, 2, 3])
