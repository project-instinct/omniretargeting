import numpy as np

from omniretargeting._fast.repair_limits import RepairLimits
from omniretargeting._fast.reference import ReferenceShape
from tests.test_fast_backend import _objective
from tests.test_repair_penetration_yield import _linear_accepts


def test_world_body_rotation_is_reported_but_fixed_relation_and_joint_limits_remain(monkeypatch):
    from dataclasses import replace
    from omniretargeting._fast import repair_limits
    monkeypatch.setattr(repair_limits, 'OPTIONS',
                        replace(repair_limits.OPTIONS, body_rotation_report_only=True))
    objective,poses,_=_objective(temporal=True)
    poses=np.tile(poses[0],(len(poses),1))
    objective.effectors=[];objective.surfaces=[]
    objective.source_shape=ReferenceShape(objective,poses)
    limits=RepairLimits(objective,poses)
    candidate=poses.copy()
    for name in ('hip','knee','ankle'):
        candidate[:,objective.model.joint(name).qposadr[0]]+=.025
    accepted,report=limits.check(candidate)
    assert accepted,report
    assert report['max_body_change_rad']>.05
    assert _linear_accepts(limits,poses,candidate)
    assert objective.source_shape.accepts(candidate)
    candidate[:,objective.model.joint('hip').qposadr[0]]+=.026
    assert not limits.check(candidate)[0]
