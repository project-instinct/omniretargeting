"""A collision introduced during acquisition is checked before contact freeze."""
from types import SimpleNamespace
import numpy as np
from omniretargeting._fast.acquisition_clearance import complete_original_clearance


def test_late_local_box_conflict_gets_rows_without_repeated_unchanged_passes(monkeypatch):
    import omniretargeting._fast.acquisition_clearance as module
    selected=[];calls=[]
    objective=SimpleNamespace(model=object(),cfg={'iterations':12},
                              stop_workers=lambda:calls.append('stop'),
                              start_workers=lambda:calls.append('start'))
    geometry=SimpleNamespace(_static_bound_reuse=False)
    class Provider:
        def __init__(self):self.geometry=geometry;self.last_measurement=None
        def __call__(self,poses):self.last_measurement={'depths':{},'pose':poses.copy()}
    provider=Provider()
    # No initial conflict; a newly acquired pose contains a collider that
    # cannot exit in the fixed-root box. Re-observing it must not loop forever.
    monkeypatch.setattr(module,'unreachable_geoms',lambda *a,**k:[dict(geom_id=4,residual_lower_bound_m=.02)])
    monkeypatch.setattr(module,'OriginalClearance',lambda g,ids:selected.append(set(ids)))
    poses=np.zeros((3,8));reference=poses.copy()
    def solve(obj,q,**kwargs):
        assert kwargs['label']=='contact_original_meshes'
        calls.append('solve');return q.copy(),[dict(phase=kwargs['label'])]
    result,history,conflicts=complete_original_clearance(objective,poses,provider,[],solve)
    assert calls==['stop','start','solve'] and selected==[{4}]
    assert len(history)==len(conflicts)==1
    np.testing.assert_array_equal(result,reference)
    np.testing.assert_array_equal(poses,reference)
    assert conflicts[0]['detected']=='before_contact_freeze'
