"""Passing acquisition must not hide failed or unaudited initialization."""

from omniretargeting._fast.pipeline import _continuity_layers


def test_initialization_failure_survives_acquisition_success():
    bad = dict(windowed_budget_accepted=False, strict_accepted=False, relaxed_chatter_accepted=False)
    good = dict.fromkeys(bad, True)
    result = _continuity_layers(bad, good, good, True)
    assert result['acquisition_guard_accepted']
    assert result['final_vs_fixed_reference']['windowed_budget_accepted']
    assert not result['source_initialization_vs_kinematic_seed']['windowed_budget_accepted']
    assert not result['source_initialization_vs_kinematic_seed']['relaxed_chatter_accepted']


def test_reused_reference_without_initial_audit_is_unknown_not_green():
    good = dict(windowed_budget_accepted=True, strict_accepted=True, relaxed_chatter_accepted=True)
    result = _continuity_layers(None, good, good, True)
    initial = result['source_initialization_vs_kinematic_seed']
    assert not initial['audited']
    assert initial['windowed_budget_accepted'] is None
