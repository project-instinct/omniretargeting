"""Only reliable, complete collision measurements may grant penetration yield."""
import pytest
from omniretargeting._fast.collision_depths import reliable_body_depth
from tests.test_repair_penetration_yield import _problem, _baseline

def test_unknown_or_partial_measurement_cannot_grant_yield():
    objective,entry=_problem()
    for updates in [dict(failures=['unknown']),dict(complete=False)]:
        measured={**_baseline(objective),**updates}
        with pytest.raises(ValueError):reliable_body_depth(objective.model,len(entry),measured)
