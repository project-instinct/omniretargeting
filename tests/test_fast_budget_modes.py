"""The rejected correction experiment is explicit and cannot leak into defaults."""
from dataclasses import replace
import pytest
from omniretargeting._fast import runtime_options


def select_budget_mode(monkeypatch, enabled):
    from omniretargeting._fast import continuity, pipeline, repair_limits, repair_temporal, reweight
    options = replace(runtime_options.OPTIONS, correction_budget=enabled)
    for module in (runtime_options, continuity, pipeline, repair_limits, repair_temporal, reweight):
        monkeypatch.setattr(module, 'OPTIONS', options)


def test_default_and_explicit_environment(monkeypatch):
    monkeypatch.delenv('OMNIRETARGETING_FAST_ENABLE', raising=False)
    monkeypatch.delenv('OMNIRETARGETING_FAST_DISABLE', raising=False)
    assert not runtime_options.RuntimeOptions.from_environment().correction_budget
    monkeypatch.setenv('OMNIRETARGETING_FAST_ENABLE', 'correction_budget')
    assert runtime_options.RuntimeOptions.from_environment().correction_budget


@pytest.mark.parametrize('enabled', [False, True])
def test_audit_rows_and_repair_switch_together(monkeypatch, enabled):
    import cvxpy as cp
    import numpy as np
    from omniretargeting._fast.continuity import CorrectionContinuity, audit_continuity
    from omniretargeting._fast.pipeline import verify_budgets
    from omniretargeting._fast.reweight import offending_frames
    from omniretargeting._fast.repair_limits import RepairLimits
    from tests.test_fast_backend import _objective
    select_budget_mode(monkeypatch, enabled)
    objective, reference, _ = _objective()
    objective.effectors = []; objective.surfaces = []; objective.candidate_guards = []
    coords = objective.coords
    reference = np.repeat(reference[:1], len(reference), axis=0)
    reference[:, coords.qi[0]] += .01 * np.sin(np.arange(len(reference)) * .2)
    candidate = reference.copy()
    candidate[:, coords.qi[0]] += .005 * np.sin(np.arange(len(reference)) * .2)
    direct = CorrectionContinuity(coords, reference)
    assert direct.correction == enabled
    assert direct.accepts(candidate) == enabled
    step = cp.Variable(len(candidate) * coords.width); step.value = np.zeros(step.shape)
    assert all(np.max(c.violation()) < 1e-10 for c in direct.constraints(candidate, step)) == enabled
    assert audit_continuity(coords, reference, candidate)['windowed_budget_accepted'] == enabled
    assert (not verify_budgets(objective.model, coords, reference, candidate)[1].any()) == enabled
    assert (not any(x.any() for x in offending_frames(coords, reference, candidate))) == enabled
    limits = RepairLimits(objective, reference)
    assert limits.check(candidate)[0] == enabled
    assert direct.accepts(reference)
