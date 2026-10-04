"""Repair solver experiments fall back without changing the Clarabel path."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import repair_qoco, repair_canonical
from tests.test_fast_repair_step import _solve


def _program():
    return dict(P=sparse.eye(1, format="csc"), c=np.array([-1.]),
                A=sparse.csc_matrix([[1.], [-1.]]), b=np.array([.3, .3]),
                dims=SimpleNamespace(zero=0, nonneg=2, soc=[], psd=[], exp=0, p3d=[], pnd=[]))


def test_qoco_is_off_by_default_and_solves_simple_box(monkeypatch):
    assert repair_qoco.QOCO_REPAIR_QP is False
    assert repair_qoco.QOCO_RESTORATIONS is False
    pytest.importorskip("qoco")
    monkeypatch.setattr(repair_qoco, "TOLERANCE", 1e-9)
    np.testing.assert_allclose(repair_qoco.solve(_program()), [.3], atol=1e-7)


@pytest.mark.parametrize("status", ["QOCO_SOLVED_INACCURATE", "QOCO_NUMERICAL_ERROR",
                                  "QOCO_MAX_ITER", "QOCO_PRIMAL_INFEASIBLE", "QOCO_DUAL_INFEASIBLE"])
def test_every_non_solved_status_requests_fallback(monkeypatch, status):
    qoco = pytest.importorskip("qoco")
    result = SimpleNamespace(status=status, x=np.zeros(1))
    monkeypatch.setattr(qoco, "QOCO", lambda: SimpleNamespace(setup=lambda *a, **kw: None, solve=lambda: result))
    assert repair_qoco.solve(_program()) is None


@pytest.mark.parametrize("value", [np.array([np.nan]), np.array([np.inf]), np.zeros(2)])
def test_bad_solution_requests_fallback(monkeypatch, value):
    qoco = pytest.importorskip("qoco")
    monkeypatch.setattr(qoco, "QOCO", lambda: SimpleNamespace(
        setup=lambda *a, **kw: None, solve=lambda: SimpleNamespace(status="QOCO_SOLVED", x=value)))
    assert repair_qoco.solve(_program()) is None


def test_setup_exception_and_unsupported_cone_request_fallback(monkeypatch):
    qoco = pytest.importorskip("qoco")

    def fail():
        raise RuntimeError("failed setup")

    monkeypatch.setattr(qoco, "QOCO", fail)
    assert repair_qoco.solve(_program()) is None
    data = _program()
    data["dims"].zero = 1
    assert repair_qoco.solve(data) is None


def test_qp_fallback_is_byte_identical_and_solved_step_is_guarded_proposal(monkeypatch):
    baseline, status = _solve([[1., 1.], [1., -1.]], [-.01, -.01], cap=.001)
    monkeypatch.setattr(repair_qoco, "QOCO_REPAIR_QP", True)
    monkeypatch.setattr(repair_qoco, "solve", lambda _: None)
    actual, actual_status = _solve([[1., 1.], [1., -1.]], [-.01, -.01], cap=.001)
    assert actual.tobytes() == baseline.tobytes() and actual_status == status


def test_restoration_fallback_is_byte_identical(monkeypatch):
    data = _program()
    baseline, status = repair_canonical.solve_data(data, 1)
    monkeypatch.setattr(repair_qoco, "QOCO_RESTORATIONS", True)
    monkeypatch.setattr(repair_qoco, "solve", lambda _: None)
    actual, actual_status = repair_canonical.solve_data(data, 1)
    assert actual.tobytes() == baseline.tobytes() and actual_status == status
