"""The QOCO proposal solve agrees with Clarabel on a canonical window program."""
from types import SimpleNamespace

import clarabel
import numpy as np
from scipy import sparse

from omniretargeting._fast import window_qoco


def _program(seed=4, n=12):
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(n, n))
    P = sparse.csc_matrix(root @ root.T + n * np.eye(n))
    c = rng.normal(size=n)
    box = sparse.vstack([-sparse.eye(n), sparse.eye(n)], format='csc')
    rows, targets, sizes = [box], [np.full(2 * n, .3)], []
    for k in range(3):  # || J x + r || <= t  as  (t, -(J x + r)) in SOC(4)
        J = sparse.csc_matrix(rng.normal(size=(3, n)) * .2)
        rows.append(sparse.vstack([sparse.csc_matrix((1, n)), -J], format='csc'))
        targets.append(np.r_[.5, rng.normal(size=3) * .05])
        sizes.append(4)
    A = sparse.vstack(rows, format='csc')
    b = np.concatenate(targets)
    return dict(P=P, c=c, A=A, b=b, dims=SimpleNamespace(zero=0, nonneg=2 * n, soc=sizes)), n


def test_qoco_proposal_matches_clarabel():
    data, n = _program()
    cones = [clarabel.NonnegativeConeT(data['dims'].nonneg)] + [clarabel.SecondOrderConeT(s) for s in data['dims'].soc]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    ref = clarabel.DefaultSolver(sparse.triu(data['P'], format='csc'), data['c'], data['A'], data['b'],
                                 cones, settings).solve()
    step = window_qoco.solve_proposal(data, n)
    assert step is not None
    np.testing.assert_allclose(step, np.asarray(ref.x), atol=5e-3)


def test_qoco_proposal_rejects_zero_cones():
    data, n = _program()
    data['dims'] = SimpleNamespace(zero=1, nonneg=data['dims'].nonneg - 1, soc=data['dims'].soc)
    assert window_qoco.solve_proposal(data, n) is None
