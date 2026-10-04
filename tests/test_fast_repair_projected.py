"""Projected assembly must retain the original solver's numerical program."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.repair_limits import LinearBounds, RepairLimits
from omniretargeting._fast.repair_projected import ProjectedBounds
from omniretargeting._fast.repair_restoration import _restoration_program
from tests.test_fast_backend import _objective


def assert_program_equal(full, projected, selector):
    movement = np.zeros(selector.shape[0])
    a = _restoration_program(full, selector, movement, .1)
    b = _restoration_program(projected, selector, movement, .1)
    for i in (0, 2):
        x, y = a[i].toarray(), b[i].toarray()
        assert x.shape == y.shape
        np.testing.assert_array_equal(x, y)
    for i in (1, 3):
        np.testing.assert_array_equal(a[i], b[i])
    assert [str(c) for c in a[4]] == [str(c) for c in b[4]]


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("perturb", [0., .001, .06])
def test_projected_limits_equal_full_with_contacts_and_crossing_support(restore, perturb):
    objective, poses, _ = _objective(temporal=True, extra=True)
    frames, width = len(poses), len(objective.coords.qi)
    objective.tasks.contact_phase = SimpleNamespace(
        roles={e["name"]: np.where(np.arange(frames) < 12, "support", "follow")
               for e in objective.effectors}, surface=None)
    # Support stencils cross both ends of the movable frame interval.
    limits = RepairLimits(objective, poses)
    assert limits.support_continuity.groups
    selected = np.arange(frames * width).reshape(frames, width)[8:13, :2].ravel()
    projection = sparse.eye(frames * width, format="csc")[:, selected]
    candidate = poses.copy()
    candidate[10, objective.coords.qi[0]] += perturb
    # A violation outside the variable domain must remain infeasible.
    candidate[25, objective.coords.qi[-1]] += perturb
    full = limits.constraints(candidate, restore_layout=restore)
    projected = limits.constraints(candidate, restore_layout=restore, projection=projection)
    assert_program_equal(full, projected, projection)


def test_complete_vectors_constant_violations_and_duplicate_linear_rows():
    frames, width = 7, 3
    selector = sparse.eye(frames * width, format="csc")[:, [3, 6, 12]]
    bounds = [LinearBounds(frames * width),
              ProjectedBounds(frames * width, selector, frames, width)]
    rng = np.random.default_rng(51)
    jac = rng.normal(size=(frames, 2, 3, width))
    residual = rng.normal(size=(frames, 2, 3)) * .1
    residual[6, 1] = [0., 0., 2.]  # Violated constant outside selected frames.
    jac[1, 0, 1:] = 0.  # Keep the complete vector, including constant components.
    temporal = sparse.diags([1., -3., 3., -1.], [0, 3, 6, 9],
                            shape=(12, frames * width), format="csr")
    for b in bounds:
        b.set_box(np.full(frames * width, -.1), np.full(frames * width, .1))
        b.norm(jac, residual, .2)
        b.add(temporal, -.001, .001)
        b.add(temporal, -.002, .0005)
        b.add(sparse.csr_matrix((1, frames * width)), 1., np.inf)
    assert_program_equal(*bounds, selector)


def test_nonselector_projection_keeps_general_fallback():
    selector = sparse.csc_matrix([[.5], [1.], [0.]])
    full, projected = LinearBounds(3), ProjectedBounds(3, selector, 1, 3)
    for bounds in (full, projected):
        bounds.set_box(np.full(3, -.1), np.full(3, .1))
        bounds.norm(np.eye(3)[None, None], np.ones((1, 1, 3)), 1.)
    assert_program_equal(full, projected, selector)


@pytest.mark.parametrize("constant_violation", [False, True])
def test_linear_early_filter_preserves_duplicate_intersections_and_scaling(constant_violation):
    selector = sparse.eye(6, format="csc")[:, [2, 3]]
    full, projected = LinearBounds(6), ProjectedBounds(6, selector, 2, 3)
    for bounds in (full, projected):
        bounds.set_box(np.full(6, -.1), np.full(6, .1))
        fixed = sparse.csr_matrix([[1., 0., 0., 0., 0., 0.]])
        varying = sparse.csr_matrix([[0., 1., 1., 0., 0., 0.]])
        bounds.add(fixed, -.05, .04)
        bounds.add(fixed, .03 if constant_violation else -.03, np.inf)
        bounds.add(varying, -.07, .06)
        bounds.add(varying, -.04, .03)
    assert_program_equal(full, projected, selector)
    from tests.test_repair_canonical import assert_canonical_equal
    assert_canonical_equal(projected, selector, np.zeros(6), .1)


def test_qp_canonical_program_bytes_equal_full_constraint_assembly(monkeypatch):
    from omniretargeting._fast import repair_step
    from tests.test_repair_proposals import _problem
    limits, geometry, poses, measured, protected, group = _problem(monkeypatch)
    limits.movable_body = group
    size = limits.frames * limits.width
    selected = np.arange(size).reshape(limits.frames, limits.width)[8:14, :2].ravel()
    projection = sparse.eye(size, format="csc")[:, selected]
    programs = []

    class Capture:
        def __init__(self, p, q, a, b, cones, settings):
            programs.append((p, q, a, b, [str(cone) for cone in cones]))
            self.variables = len(q)

        def solve(self):
            return SimpleNamespace(status="Solved", x=np.zeros(self.variables))
    import clarabel
    monkeypatch.setattr(clarabel, "DefaultSolver", Capture)
    original = limits.constraints
    for full in (False, True):
        if full:
            monkeypatch.setattr(limits, "constraints", lambda poses, **_: original(poses))
        repair_step.solve_step(limits, poses, geometry, measured, protected, projection,
                               collision_sigma=.001, local_angle=.05)
    assert len(programs) == 2
    for index in (0, 2):
        actual, expected = (program[index] for program in programs)
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        assert actual.data.tobytes() == expected.data.tobytes()
    for index in (1, 3):
        assert programs[0][index].tobytes() == programs[1][index].tobytes()
    assert programs[0][4] == programs[1][4]


def test_fixed_probe_collision_does_not_change_duplicate_merging():
    probe = np.random.default_rng(0).uniform(1., 2., 6)
    fixed = sparse.csr_matrix(([probe[2] / probe[0]], ([0], [0])), shape=(1, 6))
    varying = sparse.csr_matrix(([1.], ([0], [2])), shape=(1, 6))
    assert (fixed @ probe)[0] == (varying @ probe)[0]
    selector = sparse.eye(6, format="csc")[:, [2]]
    full, projected = LinearBounds(6), ProjectedBounds(6, selector, 2, 3)
    for bounds in (full, projected):
        bounds.set_box(np.full(6, -.1), np.full(6, .1))
        bounds.add(varying, -.01, .02)
        bounds.add(fixed, -.01, .01)
        bounds.add(varying, -.005, .008)
    assert_program_equal(full, projected, selector)
