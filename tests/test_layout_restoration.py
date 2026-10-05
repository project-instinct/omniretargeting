"""A sustained clear-layout collapse must return to the source side."""
import cvxpy as cp
import numpy as np
from omniretargeting.pose_relations import RelationEnvelope, heading_axes


class Points:
    def __init__(self, positions):
        self.positions = positions

    def points(self, bodies, offsets):
        return self.positions

    def point_jacobians(self, bodies, points):
        jac = np.zeros((len(self.positions), 4, 3, 1))
        jac[:, 2, 0, 0] = 1.
        return jac


def example():
    one = np.array([[0, 0, 1.], [0, 0, 1.5], [.3, 0, 1.4], [0, -.3, 1.2]])
    reference = np.repeat(one[None], 8, axis=0)
    envelope = RelationEnvelope([0, 1, 2, 3], np.zeros((4, 3)), [-1, 0, 1, 1],
                                reference, axes=heading_axes(np.repeat(np.eye(3)[None], 8, axis=0)))
    return envelope, reference


def test_restoration_returns_outside_proposal_to_unchanged_layout():
    envelope, reference = example()
    outside = reference.copy()
    outside[:, 2, 0] = -.4
    assert not envelope.accepts(Points(outside))
    matrix, lower, upper = envelope.linear_bounds(Points(outside), 1, restore=True)[0]
    delta = cp.Variable(8)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(delta)), [matrix @ delta >= lower])
    problem.solve(solver=cp.CLARABEL, tol_feas=1e-11, tol_gap_abs=1e-11)
    fixed = outside.copy()
    fixed[:, 2, 0] += delta.value
    assert envelope.accepts(Points(fixed))
    assert np.max(np.abs(fixed[6:, 2, 0])) < 1e-6
    assert np.array_equal(envelope.reference, envelope.graph.relations(reference))


def test_legal_proposal_keeps_rows_but_signed_restoration_is_stricter():
    envelope, reference = example()
    a = envelope.linear_bounds(Points(reference), 1)[0]
    b = envelope.linear_bounds(Points(reference), 1, restore=True)[0]
    np.testing.assert_array_equal(a[0].toarray(), b[0].toarray())
    assert np.all(b[1] >= a[1])
    np.testing.assert_array_equal(a[2], b[2])


def test_ambiguous_crossing_keeps_a_zero_step_feasible():
    envelope, reference = example()
    outside = reference.copy()
    outside[:, 2, 0] = -.05
    assert envelope.accepts(Points(outside))
    _, lower, _ = envelope.linear_bounds(Points(outside), 1)[0]
    assert np.all(lower <= 0.)
