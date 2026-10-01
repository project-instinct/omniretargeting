import numpy as np
import pytest
from omniretargeting._fast.frozen_depth import reliable_body_depth
from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.contact_guard import ContactGuard
from omniretargeting._fast.repair_limits import RepairLimits
from tests.test_repair_penetration_yield import _problem,_baseline,_move


def test_all_guards_freeze_own_state_depth_and_never_query_candidates():
    objective,entry=_problem()
    objective.tasks.contact_phase=None
    calls=[]
    def provider(poses):
        calls.append(poses.copy())
        depth=.002 if np.array_equal(poses,entry) else .004
        return reliable_body_depth(objective.model,len(poses),_baseline(objective,depth))
    objective.frozen_original_depth=provider
    acquired=_move(objective,entry,.001)
    acquisition=AcquisitionNormalGuard(objective,entry)
    contact=ContactGuard(objective,acquired)
    limits=RepairLimits(objective,acquired)
    limits.freeze_penetration_yield(_baseline(objective,.006))
    body=objective.model.body('wet').id
    assert acquisition.penetration[:,body].max()==.002
    assert contact.penetration[:,body].max()==.004
    assert limits.penetration_yield[:,body].max()==.006
    bad=_move(objective,entry,.01)
    acquisition.accepts(bad);contact.accepts(bad);limits.check(bad)
    assert len(calls)==2
    assert acquisition.penetration[:,body].max()==.002
    assert not contact.penetration.flags.writeable
    assert not limits.penetration_yield.flags.writeable
    with pytest.raises(RuntimeError):limits.freeze_penetration_yield(_baseline(objective,1.))


def test_unknown_or_partial_measurement_cannot_grant_yield():
    objective,entry=_problem()
    for updates in [dict(failures=['unknown']),dict(complete=False)]:
        measured={**_baseline(objective),**updates}
        with pytest.raises(ValueError):reliable_body_depth(objective.model,len(entry),measured)


def test_original_depth_cache_restores_matching_measurement_with_each_state():
    from omniretargeting._fast.frozen_depth import FrozenDepth
    from tests.test_repair_new_witnesses import Plane
    objective,entry=_problem()
    provider=FrozenDepth(objective,[],Plane())
    first=provider(entry); first_measurement=provider.last_measurement
    moved=_move(objective,entry,.001)
    second=provider(moved)
    assert provider.last_measurement is not first_measurement
    assert not np.array_equal(first,second)
    assert provider(entry) is first
    assert provider.last_measurement is first_measurement
    assert len(provider.audits)==2
