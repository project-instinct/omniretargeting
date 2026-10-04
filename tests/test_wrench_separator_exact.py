"""The native shortcut must preserve the separator, not merely its objective."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

from omniretargeting.contact_mechanics import ContactCandidate, explain_contacts
from omniretargeting.wrench_feasibility import DualWrenchCone
import omniretargeting.wrench_feasibility as feasibility
import omniretargeting.wrench_separator as native


def scipy_separator(rays, wrench):
    return linprog(
        wrench, A_ub=-rays, b_ub=np.zeros(len(rays)), bounds=[(-1.0, 1.0)] * 6,
        method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                 "dual_feasibility_tolerance": 1e-9},
    ).x


@pytest.mark.parametrize("seed", range(12))
def test_native_separator_preserves_all_solution_bits(seed):
    if native._OPTIONS is None:
        pytest.skip("SciPy bundled native API unavailable")
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(3, 3))
    contacts = [ContactCandidate(str(i), p[None], p[None], [0, 0, 1], 0.6, 0)
                for i, p in enumerate(points)]
    rays = DualWrenchCone(contacts).rays
    # Degenerate objectives/faces, repeated rows, and exact zeros exercise
    # pivot selection and CSC zero elimination as well as ordinary demands.
    for matrix in (rays, np.r_[rays, rays], rays[:, :] * [1, 1, 1, 0, 0, 0]):
        for wrench in (np.zeros(6), np.array([0, 0, 100, 0, 0, 0]), rng.normal(size=6)):
            actual = native.separator(matrix, wrench)
            assert actual is not None
            assert actual.tobytes() == scipy_separator(matrix, wrench).tobytes()


@pytest.mark.parametrize("failure", ["options", "model", "run", "status", "unavailable"])
def test_nonoptimal_native_solve_uses_scipy_fallback(monkeypatch, failure):
    if native._OPTIONS is None:
        pytest.skip("SciPy bundled native API unavailable")
    core = native._highs

    class FailedModel:
        def passOptions(self, options):
            return core.HighsStatus.kError if failure == "options" else core.HighsStatus.kOk

        def passModel(self, lp):
            return core.HighsStatus.kError if failure == "model" else core.HighsStatus.kOk

        def run(self):
            return core.HighsStatus.kError if failure == "run" else core.HighsStatus.kOk

        def getModelStatus(self):
            return core.HighsModelStatus.kInfeasible

    if failure == "unavailable":
        monkeypatch.setattr(native, "_OPTIONS", None)
    else:
        monkeypatch.setattr(native, "_highs", SimpleNamespace(
            HighsLp=core.HighsLp, MatrixFormat=core.MatrixFormat, kHighsInf=core.kHighsInf,
            HighsStatus=core.HighsStatus, HighsModelStatus=core.HighsModelStatus,
            _Highs=FailedModel))
    calls = []

    def fallback(*args, **kwargs):
        calls.append(kwargs)
        return linprog(*args, **kwargs)

    monkeypatch.setattr(feasibility, "linprog", fallback)
    p = np.zeros((1, 3))
    cone = DualWrenchCone([ContactCandidate("point", p, p, [0, 0, 1], 0.5, 0)])
    wrench = np.array([60, 0, 100, 0, 0, 0])
    assert cone._separator(wrench).tobytes() == scipy_separator(cone.rays, wrench).tobytes()
    assert len(calls) == 1


def test_exact_geometry_cache_keeps_subset_specific_exclusions(monkeypatch):
    p = np.zeros((1, 3))
    contacts = [ContactCandidate(name, p, p, [0, 0, 1], 0.5, 0) for name in ("a", "b")]
    contacts.append(ContactCandidate("gap", p + [0, 0, 1], p, [0, 0, 1], 0.5, 0))
    original = DualWrenchCone.check
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(self.rays.tobytes())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DualWrenchCone, "check", counted)
    result = explain_contacts(contacts, [0, 0, 100, 0, 0, 0], method="dual")
    assert result["roles"] == {"a": "optional_contact", "b": "optional_contact",
                               "gap": "geometry_rejected"}
    assert result["minimal_feasible_sets"] == [["a"], ["b"]]
    assert len(calls) == len(set(calls)) == 3  # empty, one patch, two patches
    assert len(result["hypotheses"]) > len(calls)
    for hypothesis in result["hypotheses"]:
        assert hypothesis["excluded_geometry"] == (["gap"] if "gap" in hypothesis["contacts"] else [])
