import clarabel
import numpy as np
from scipy import sparse

from omniretargeting._fast import working_set


def _settings():
    value = clarabel.DefaultSettings()
    value.verbose = False
    return value


def _program(seed, trust=None):
    rng = np.random.default_rng(seed)
    n, rows, cones = 40, 400, 300
    quadratic = sparse.triu(sparse.eye(n) * 2.0, format="csc")
    linear = rng.normal(size=n)
    linear_rows = rng.normal(size=(rows, n))
    # Mostly slack rows; a few near their bound at the zero step.
    linear_rhs = np.where(rng.random(rows) < 0.05, rng.random(rows) * 1e-4, 1 + 5 * rng.random(rows))
    blocks, rhs, kinds = [sparse.csc_matrix(linear_rows)], [linear_rhs], [clarabel.NonnegativeConeT(rows)]
    if trust is not None:
        blocks.append(sparse.vstack([sparse.csc_matrix((1, n)), -sparse.eye(n)], format="csc"))
        rhs.append(np.r_[trust, np.zeros(n)])
        kinds.append(clarabel.SecondOrderConeT(n + 1))
    for _ in range(cones):
        jac = rng.normal(size=(3, n)) * 0.1
        residual = rng.normal(size=3) * 0.1
        allowed = np.linalg.norm(residual) + rng.choice([1e-5, 2.0])
        blocks.append(sparse.csc_matrix(np.vstack([np.zeros((1, n)), -jac])))
        rhs.append(np.r_[allowed, residual])
        kinds.append(clarabel.SecondOrderConeT(4))
    return quadratic, linear, sparse.vstack(blocks, format="csc"), np.concatenate(rhs), kinds


def test_working_set_matches_complete_program():
    for seed in range(4):
        program = _program(seed)
        full = clarabel.DefaultSolver(*program, _settings()).solve()
        reduced = working_set.solve(*program, _settings)
        assert str(full.status) == "Solved"
        assert reduced is not None and reduced.rows < program[2].shape[0]
        np.testing.assert_allclose(reduced.x, np.asarray(full.x), atol=1e-6)


def test_safe_screening_inside_trust_ball():
    for seed in range(4):
        program = _program(seed, trust=0.05)
        full = clarabel.DefaultSolver(*program, _settings()).solve()
        reduced = working_set.solve(*program, _settings, radius=0.05)
        assert reduced is not None
        np.testing.assert_allclose(reduced.x, np.asarray(full.x), atol=1e-6)


def test_infeasible_program_falls_back():
    n = 3
    quadratic = sparse.triu(sparse.eye(n), format="csc")
    matrix = sparse.csc_matrix(np.vstack([np.eye(n)[:1], -np.eye(n)[:1]]))
    offset = np.array([-1.0, -1.0])  # x0 <= -1 and x0 >= 1
    assert working_set.solve(quadratic, np.zeros(n), matrix, offset,
                             [clarabel.NonnegativeConeT(2)], _settings) is None
