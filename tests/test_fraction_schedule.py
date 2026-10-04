"""Deferred fractions retain the original serial consumer's numerical state."""
from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast.fraction_schedule import DeferredFractionRestorations, restorations_type
from omniretargeting._fast.shortening import FractionRestorations


class Guard:
    rejections = 0


def restore(objective, origin, step):
    for guard in objective.guards:
        guard.rejections += int(round(float(step[0, 0]) * 16))
    if objective.fail and float(step[0, 0]) == .5:
        raise ValueError('restoration failed')
    return origin + step


def context(cls, *, fail=False):
    objective = SimpleNamespace(guards=[Guard()], fail=fail)
    return cls(restore, objective, np.zeros((5, 2)), np.ones((5, 2)), 1.,
               (.5, .25, .125), 2, objective.guards, enabled=True)


@pytest.mark.parametrize('fractions', [(), (.5,), (.25,), (.5, .25), (.125,)])
def test_consumer_outputs_and_counters_equal_reference(fractions):
    reference, fast = context(FractionRestorations), context(DeferredFractionRestorations)
    with reference, fast:
        for fraction in fractions:
            old, new = reference.restore(fraction), fast.restore(fraction)
            assert old.tobytes() == new.tobytes()
            assert reference.objective.guards[0].rejections == fast.objective.guards[0].rejections
    assert reference.objective.guards[0].rejections == fast.objective.guards[0].rejections
    assert not fast.jobs
    if fractions == (.5,):
        assert reference.stats['launched'] == 2
        assert fast.stats['launched'] == 1
        assert fast.stats['discarded'] == 0


def test_exception_at_consumed_fraction_has_original_counter_semantics():
    for cls in (FractionRestorations, DeferredFractionRestorations):
        pending = context(cls, fail=True)
        with pending:
            with pytest.raises(ValueError, match='restoration failed'):
                pending.restore(.5)
        assert pending.objective.guards[0].rejections == 8
        assert not pending.jobs


def test_disabled_fork_retains_serial_path():
    pending = context(DeferredFractionRestorations)
    pending.enabled = False
    with pending:
        assert pending.restore(.25).tobytes() == np.full((5, 2), .25).tobytes()
    assert pending.stats['launched'] == 0


def test_runtime_switch_preserves_reference(monkeypatch):
    from omniretargeting._fast import runtime_options
    monkeypatch.setattr(runtime_options, 'OPTIONS', SimpleNamespace(lazy_fraction_restorations=False))
    assert restorations_type() is FractionRestorations
    monkeypatch.setattr(runtime_options, 'OPTIONS', SimpleNamespace(lazy_fraction_restorations=True))
    assert restorations_type() is DeferredFractionRestorations


def test_complete_lm_history_with_consumed_restorations(monkeypatch):
    from dataclasses import replace
    from omniretargeting._fast import runtime_options, shortening
    from tests.test_fast_proposals import _solve

    monkeypatch.setattr(runtime_options, 'OPTIONS',
                        replace(runtime_options.OPTIONS, parallel_shortening=True))
    reference = _solve(monkeypatch, False)
    calls = []
    original = DeferredFractionRestorations.restore
    def counted(self, fraction):
        calls.append(fraction)
        return original(self, fraction)
    monkeypatch.setattr(DeferredFractionRestorations, 'restore', counted)
    monkeypatch.setattr(shortening, 'FractionRestorations', DeferredFractionRestorations)
    fast = _solve(monkeypatch, False)
    assert calls, 'must consume restored fractions'
    assert fast[0].tobytes() == reference[0].tobytes()
    assert fast[1] == reference[1]
    assert fast[2] == reference[2]
