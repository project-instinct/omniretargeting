"""Secondary per-joint RMS/peak limits cannot be hidden by other joints."""
import numpy as np
import pytest
from tests.test_fast_budget_modes import select_budget_mode
from scipy import sparse

from omniretargeting._refinement.math import _difference_operator
from omniretargeting._fast.repair_limits import LinearBounds
from omniretargeting._fast.repair_temporal import RepairTemporal


@pytest.mark.parametrize("correction", [False, True])
def test_each_joint_rms_and_peak_enter_both_cones_and_acceptance(monkeypatch, correction):
    select_budget_mode(monkeypatch, correction)
    time=np.arange(40)
    entry=np.sin(time[:,None]*.2)*np.array([[1.,.01]])
    operators=[(sparse.kron(_difference_operator(len(time),o),sparse.eye(2),format='csr'),None)
               for o in (1,2,3)]
    guard=RepairTemporal(entry,operators,2)
    candidate=entry*np.array([[.8,2.1 if correction else 1.1]])
    assert np.linalg.norm(np.diff(candidate,n=3,axis=0))<np.linalg.norm(np.diff(entry,n=3,axis=0))
    accepted,excess=guard.check(candidate)
    assert not accepted and all(r['joint_column']==1 for r in excess)
    bounds=LinearBounds(entry.size);guard.add(bounds,entry)
    step=(candidate-entry).ravel()
    assert any(np.any(np.linalg.norm(r+(a@step).reshape(r.shape),axis=1)>b+1e-12)
               for a,r,b in bounds.norms)
    # A constant local correction introduces no derivatives.
    assert guard.check(entry+.025)[0]
    assert guard.check(entry*1.01)[0]
    assert guard.check(entry*1.03)[0] == correction
    assert not guard.check(entry*2.03)[0]
