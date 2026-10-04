"""SOC changes offsets only; original objective and witness identity survive."""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from omniretargeting._fast.qp_assembly import assemble_program
from omniretargeting._fast.qp_guard_rows import intervals
from omniretargeting._fast.window_soc import corrected_model
from tests.test_fast_qp_assembly import small_model


def test_soft_norm_defect_keeps_jacobian_and_tracking_model():
    key = ('contact', 'support', 0, 1)
    cone = (sparse.csr_matrix([[1.], [0.], [0.]]), np.zeros((1, 3)),
            np.array([.5]), np.ones(1), [key])
    model = small_model(cones=[cone])
    provider = SimpleNamespace(soft_values=lambda p, **k: ([], [(np.array([[p[0, 0], p[0, 0] ** 2, 0.]]), np.array([.5]))]))
    model.window_soft_sources = ((provider, 0, 0, 0, 1),)
    model.frames = range(1)
    original = model.soft.cones[0][1].copy()
    corrected = corrected_model(model, np.array([[.5]]), np.array([.5]))
    assert corrected is not None
    assert corrected.hessian is model.hessian and corrected.gradient is model.gradient
    assert corrected.soft.cones[0][0] is model.soft.cones[0][0]
    np.testing.assert_array_equal(corrected.soft.cones[0][1], [[0., .25, 0.]])
    np.testing.assert_array_equal(model.soft.cones[0][1], original)


def test_scalar_defect_and_missing_geometry_batch_are_not_confused():
    key = ('penetration', 0, 1)
    model = small_model(rows=[(sparse.csr_matrix([[1.]]), np.array([1.]), np.ones(1), [key])])
    provider = SimpleNamespace(soft_values=lambda p, **k: ([None], []))
    model.window_soft_sources = ((provider, 0, 1, 0, 0),)
    model.frames = range(1)
    assert corrected_model(model, np.array([[.1]]), np.array([.1])) is None
    block = intervals(sparse.csr_matrix([[1.]]), [-np.inf], [1.],
                      lambda p: (np.array([-np.inf]), np.array([1. - p[0, 0] ** 2])))
    model.program = assemble_program([block], model.soft, 1)
    corrected = corrected_model(model, np.array([[2.]]), np.array([2.]))
    assert corrected is not None
    # At trial2, g=3 and Jp=2: shifted b=-1, original objective retained.
    np.testing.assert_array_equal(corrected.program.hard_blocks[0].b, [-1.])
    np.testing.assert_array_equal(corrected.soft.rows[0][1], [1.])


def test_missing_hard_geometry_batch_does_not_disable_no_donor_contact_soc():
    from omniretargeting._fast.window_assembly import _guard_blocks
    model = small_model()
    geometry = (sparse.csr_matrix([[1.]]), np.array([-np.inf]), np.array([2.]))
    contact = (sparse.csr_matrix([[1.], [0.], [0.]]), np.zeros((1, 3)), np.array([.5]))
    calls = []
    def values(p, *, frames):
        calls.append(1)
        x = p[0, 0]
        return [None], [(np.array([[x, x * x, 0.]]), np.array([.5]))]
    provider = SimpleNamespace(qp_bounds=lambda p, **k: ([geometry], [contact]), qp_values=values)
    blocks, _ = _guard_blocks(provider, np.zeros((1, 1)), range(1), 1)
    model.program = assemble_program(blocks, model.soft, 1)
    corrected = corrected_model(model, np.array([[.5]]), np.array([.5]))
    assert corrected is not None and len(calls) == 1
    np.testing.assert_array_equal(corrected.program.hard_blocks[0].b, blocks[0].b)
    np.testing.assert_array_equal(corrected.program.hard_blocks[1].b, [.5, 0., .25, 0.])
    assert corrected.gradient is model.gradient and corrected.hessian is model.hessian
