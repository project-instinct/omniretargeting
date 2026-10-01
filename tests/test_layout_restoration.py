"""An outside-layout proposal must return to the fixed legal side."""
import cvxpy as cp
import numpy as np
from omniretargeting.pose_relations import RelationEnvelope, heading_axes


class Points:
    def __init__(self, positions):
        self.positions = positions

    def points(self, bodies, offsets):
        return self.positions

    def point_jacobians(self, bodies, points):
        jac = np.zeros((1, 4, 3, 1))
        jac[0, 2, 0, 0] = 1.
        return jac


def example():
    reference = np.array([[[0, 0, 1.], [0, 0, 1.5], [.3, 0, 1.4], [0, -.3, 1.2]]])
    envelope = RelationEnvelope([0, 1, 2, 3], np.zeros((4, 3)), [-1, 0, 1, 1],
                                reference, axes=heading_axes(np.eye(3)[None]))
    return envelope, reference


def test_restoration_returns_outside_proposal_to_unchanged_layout():
    envelope, reference = example()
    outside = reference.copy()
    outside[0, 2, 0] = -.05
    assert not envelope.accepts(Points(outside))
    matrix, lower, upper = envelope.linear_bounds(Points(outside), 1, restore=True)[0]
    delta = cp.Variable(1)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(delta)), [matrix @ delta >= lower])
    problem.solve(solver=cp.CLARABEL, tol_feas=1e-11, tol_gap_abs=1e-11)
    fixed = outside.copy()
    fixed[0, 2, 0] += delta.value[0]
    assert envelope.accepts(Points(fixed))
    assert abs(fixed[0, 2, 0]) < 1e-6
    assert np.array_equal(envelope.reference, envelope.graph.relations(reference))


def test_legal_proposal_keeps_identical_rows_and_budgets():
    envelope, reference = example()
    a = envelope.linear_bounds(Points(reference), 1)[0]
    b = envelope.linear_bounds(Points(reference), 1, restore=True)[0]
    np.testing.assert_array_equal(a[0].toarray(), b[0].toarray())
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


def test_original_direction_mode_remains_unchanged_for_outside_proposal():
    envelope, reference = example()
    outside = reference.copy()
    outside[0, 2, 0] = -.05
    _, old, _ = envelope.linear_bounds(Points(outside), 1)[0]
    _, restored, _ = envelope.linear_bounds(Points(outside), 1, restore=True)[0]
    assert np.all(old <= 0.)
    assert restored.max() > 0.
