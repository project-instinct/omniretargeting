"""Built-in guard failure reports remain identical; custom methods still run."""

import pickle

import numpy as np
import pytest

from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.contact_guard import ContactGuard
from omniretargeting._fast.repair_diagnostics import guard_failure
from omniretargeting._fast.repair_guard_checks import check_guard
from tests.test_fast_legacy_window import problem


@pytest.mark.parametrize("kind", [ContactGuard, AcquisitionNormalGuard])
def test_guard_acceptance_and_full_failure_diagnostics_are_identical(kind):
    objective, poses = problem(extra=True)
    guard = kind(objective, poses)
    rejected = False
    for amplitude in (0., .0001, .1, .7, np.nan):
        trial = poses.copy()
        trial[13, objective.coords.qi[0]] += amplitude
        accepted = guard.accepts(trial)
        detail = None if accepted else guard_failure(guard, trial)
        actual, report = check_guard(guard, trial)
        assert actual == accepted
        assert pickle.dumps(report, protocol=5) == pickle.dumps(detail, protocol=5)
        rejected |= not actual
    assert rejected


def test_custom_acceptance_and_diagnostic_overrides_are_called(monkeypatch):
    objective, poses = problem(extra=True)
    guard = ContactGuard(objective, poses)
    calls = []
    monkeypatch.setattr(guard, "accepts", lambda *_: calls.append("accepts") or False)
    monkeypatch.setattr(guard, "diagnostics", lambda *_: calls.append("diagnostics") or {"custom": 7})
    actual, report = check_guard(guard, poses)
    assert not actual and report["constraint"] == {"custom": 7}
    assert calls == ["accepts", "diagnostics"]


def test_contact_coordinator_acceptance_keeps_original_dispatch():
    from types import SimpleNamespace
    objective, poses = problem(extra=True)
    guard = ContactGuard(objective, poses)
    calls = []
    guard.priority = SimpleNamespace(accepts=lambda *_: calls.append("accepts") or False,
                                     diagnostics=lambda *_: calls.append("diagnostics") or {"priority": 1})
    actual, report = check_guard(guard, poses)
    assert not actual and report["constraint"] == {"priority": 1}
    assert calls == ["accepts", "diagnostics"]
