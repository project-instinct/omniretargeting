"""Ablation switches are independent and observable; disabled work stays absent."""
import numpy as np
import pytest
from omniretargeting._fast.runtime_options import RuntimeOptions
from omniretargeting._fast.repair_schedule import RepairSchedule
from omniretargeting._fast.repair_witnesses import RepairWitnesses


def test_default_enables_only_validated_options(monkeypatch):
    monkeypatch.delenv('OMNIRETARGETING_FAST_ENABLE', raising=False)
    monkeypatch.delenv('OMNIRETARGETING_FAST_DISABLE', raising=False)
    enabled = {name for name, value in RuntimeOptions.from_environment().describe().items() if value}
    assert enabled == {
        'parallel_shortening', 'parallel_repair_restore', 'repair_support_continuity',
        'fast_self_collision', 'bulk_cpu_geometry', 'repair_triage',
        'repair_fast_gate', 'repair_parallel_groups', 'whole_canonical',
        'repair_working_set', 'repair_geometry_reduction', 'stationary_window',
        'repair_main_cache',
    }


def test_independent_switches(monkeypatch):
    for name in RuntimeOptions.__dataclass_fields__:
        monkeypatch.setenv('OMNIRETARGETING_FAST_DISABLE', name)
        flags = RuntimeOptions.from_environment().describe()
        assert flags[name] is False
        defaults = RuntimeOptions().describe()
        assert all(value == defaults[key] for key, value in flags.items() if key != name)


def test_repair_main_cache_uses_runtime_switch(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.repair_frame_norm import enabled
    monkeypatch.delenv('OMNIRETARGETING_FAST_ENABLE', raising=False)
    monkeypatch.delenv('OMNIRETARGETING_FAST_DISABLE', raising=False)
    monkeypatch.setenv('OMNIRETARGETING_REPAIR_MAIN', '0')
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions.from_environment())
    assert enabled()
    monkeypatch.setenv('OMNIRETARGETING_FAST_DISABLE', 'repair_main_cache')
    monkeypatch.setenv('OMNIRETARGETING_REPAIR_MAIN', '1')
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions.from_environment())
    assert not enabled()


def test_unknown_switch_is_error(monkeypatch):
    monkeypatch.setenv('OMNIRETARGETING_FAST_DISABLE', 'typo')
    with pytest.raises(ValueError, match='typo'):
        RuntimeOptions.from_environment()


def test_no_deferred_wakeup():
    schedule = RepairSchedule([3, 4], 12, defer_enabled=False)
    assert schedule.next() == (3, 0)
    schedule.defer(3, np.zeros((2, 5)), [1], 'small_progress')
    schedule.changed(np.ones((2, 5)))
    assert schedule.next() == (4, 0)
    assert schedule.next() is None


def test_disabled_witness_does_no_queries():
    witnesses = RepairWitnesses(enabled=False)
    measured = {'rows': []}
    assert witnesses.observe(measured, {'rows': [((), {'clearance': -1})]}) == 0
    assert witnesses.augment(None, None, measured) is measured


def test_batch2_switch_restores_legacy_shortened_convergence(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.solver import _progress_converged
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(batch2=False))
    costs = [100.]
    slow, done = _progress_converged(costs, 99.99, 1, .25, True, .001, shortened=True)
    assert slow == 2 and done
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(batch2=True))
    slow, done = _progress_converged([100.], 99.99, 1, .25, True, .001, shortened=True)
    assert slow == 0 and not done


def test_retry_schedule_has_original_total_attempt_budget():
    groups = list(range(7))
    schedule = RepairSchedule(groups, 12)
    attempts = 0
    while (item := schedule.next()) is not None:
        group, _ = item
        attempts += 1
        schedule.retry(group)
    assert attempts == 12 * len(groups)


def test_explicit_enable_and_conflict(monkeypatch):
    monkeypatch.setenv('OMNIRETARGETING_FAST_ENABLE', 'off_surface_approach_only')
    monkeypatch.delenv('OMNIRETARGETING_FAST_DISABLE', raising=False)
    assert RuntimeOptions.from_environment().off_surface_approach_only
    monkeypatch.setenv('OMNIRETARGETING_FAST_DISABLE', 'off_surface_approach_only')
    with pytest.raises(ValueError, match='Conflicting'):
        RuntimeOptions.from_environment()
