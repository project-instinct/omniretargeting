"""Only timing diagnostics are configurable on the fast production path."""
import pytest
from omniretargeting._fast.runtime_options import RuntimeOptions
from omniretargeting._fast.repair_schedule import RepairSchedule


def test_default_has_no_diagnostics(monkeypatch):
    monkeypatch.delenv('OMNIRETARGETING_FAST_ENABLE', raising=False)
    assert RuntimeOptions.from_environment().describe() == {'performance_trace': False}


def test_performance_trace_compatibility(monkeypatch):
    monkeypatch.setenv('OMNIRETARGETING_FAST_ENABLE', 'performance_trace')
    assert RuntimeOptions.from_environment().performance_trace


def test_removed_algorithm_switch_is_error(monkeypatch):
    monkeypatch.setenv('OMNIRETARGETING_FAST_ENABLE', 'spline_correction')
    with pytest.raises(ValueError, match='spline_correction'):
        RuntimeOptions.from_environment()


def test_retry_schedule_has_original_total_attempt_budget():
    groups = list(range(7))
    schedule = RepairSchedule(groups, 12)
    attempts = 0
    while (item := schedule.next()) is not None:
        group, _ = item
        attempts += 1
        schedule.retry(group)
    assert attempts == 12 * len(groups)
