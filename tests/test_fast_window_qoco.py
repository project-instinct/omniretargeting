"""The opt-in QOCO proposal solve agrees with Clarabel on a canonical window program."""
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


def test_qoco_proposals_default_on_restorations_off_and_zero_cones_rejected():
    assert window_qoco.WINDOW_QOCO_PROPOSALS is True
    assert window_qoco.WINDOW_QOCO_RESTORATIONS is False
    data, n = _program()
    data['dims'] = SimpleNamespace(zero=1, nonneg=data['dims'].nonneg - 1, soc=data['dims'].soc)
    assert window_qoco.solve_proposal(data, n) is None


def test_restoration_rejects_every_non_solved_status_and_nonfinite_step(monkeypatch):
    import qoco

    data, n = _program()
    for status in ('QOCO_NUMERICAL_ERROR', 'QOCO_SOLVED_INACCURATE', 'QOCO_MAX_ITER',
                   'QOCO_PRIMAL_INFEASIBLE', 'QOCO_DUAL_INFEASIBLE'):
        result = SimpleNamespace(status=status, x=np.zeros(n))
        monkeypatch.setattr(qoco, 'QOCO', lambda: SimpleNamespace(setup=lambda *a, **kw: None,
                                                                solve=lambda: result))
        assert window_qoco.solve_restoration(data, n) is None
    result = SimpleNamespace(status='QOCO_SOLVED', x=np.full(n, np.nan))
    assert window_qoco.solve_restoration(data, n) is None


def test_restoration_setup_failure_requests_clarabel_fallback(monkeypatch):
    def fail(*args):
        raise RuntimeError('numerical setup failure')

    monkeypatch.setattr(window_qoco, 'solve_proposal', fail)
    data, n = _program()
    assert window_qoco.solve_restoration(data, n) is None


def test_canonical_restoration_fallback_preserves_default_pose_bytes(monkeypatch):
    from omniretargeting._fast.legacy_window_qp import WindowObjective, serial_proposals
    from omniretargeting._fast.window_canonical import restore_step
    from tests.test_fast_legacy_window import problem

    obj, poses = problem(extra=True)
    window = WindowObjective(obj, poses, range(9, 18))
    origin = poses[window.span]
    step = np.random.default_rng(721).normal(scale=.1, size=(window.frames, window.width))
    with serial_proposals():
        reference = restore_step(window, origin, step)
        monkeypatch.setattr(window_qoco, 'WINDOW_QOCO_RESTORATIONS', True)
        calls = []
        def unusable(data, count):
            calls.append(count)
            return None
        monkeypatch.setattr(window_qoco, 'solve_restoration', unusable)
        actual = restore_step(window, origin, step)
    assert calls, 'fixture must exercise a restoration solve'
    assert (actual is None) == (reference is None)
    if actual is not None:
        assert actual.tobytes() == reference.tobytes()
