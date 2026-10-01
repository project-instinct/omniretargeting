import numpy as np

from omniretargeting._fast.repair_diagnostics import deep_residuals, guard_failure
from tests.test_repair_penetration_yield import _problem


def test_all_deep_keys_report_actual_bodies_frames_and_signed_exit_components():
    objective, _ = _problem()
    model = objective.model
    geom = model.geom('wet_geom').id
    other = next(g for g in range(model.ngeom) if g != geom)
    key = (3, 'moving', 0, 'reverse_surface', -1, geom)
    self_key = (5, 'self', other, geom)
    measured = dict(rows=[
        (key, dict(clearance=-.011, direction=np.array([1., 0., 0.]))),
        (key, dict(clearance=-.02, direction=np.array([.6, 0., -.8]))),
        ((4, 'static', geom), dict(clearance=-.01, direction=np.array([0., 0., 1.]))),
        (self_key, dict(clearance=-.012, direction=np.array([0., 1., 0.]))),
    ])
    result = deep_residuals(model, measured)
    assert len(result) == 2
    moving = next(r for r in result if r['key'] == list(key))
    assert moving['frame'] == 3 and moving['depth_m'] == .02
    assert moving['bodies'] == [model.body(int(model.geom_bodyid[geom])).name]
    np.testing.assert_allclose([moving['exit_xy_m'], moving['exit_z_m']], [.012, -.016])
    collision = next(r for r in result if r['key'] == list(self_key))
    assert collision['bodies'] == [model.body(int(model.geom_bodyid[g])).name for g in (other, geom)]


def test_rejected_guard_report_keeps_violating_frames_without_mutating_guard_data():
    class Guard:
        def __init__(self):
            self.report = dict(accepted=False, groups=[dict(name='safe', violating_frames=[]),
                                                       dict(name='held', violating_frames=[7])])
        def diagnostics(self, poses):
            return self.report
    guard = Guard()
    result = guard_failure(guard, np.zeros((8, 1)))
    assert result['guard'] == 'Guard'
    assert result['constraint']['groups'] == [dict(name='held', violating_frames=[7])]
    assert len(guard.report['groups']) == 2
