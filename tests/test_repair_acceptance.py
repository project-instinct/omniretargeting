"""Frozen-depth acceptance and default repair output preserve the original contracts."""
from types import SimpleNamespace
import numpy as np
import pytest
from omniretargeting._fast import repair_acceptance

@pytest.mark.parametrize('depth', [0., 1e-6, .0004, .01, np.nan])
def test_candidate_preserves_legacy_predicate(depth):
    from omniretargeting._fast.mesh_repair import _regressed
    key = (0, 'static', 0)
    protected = {'depths': {key: .0003}}
    checked = dict(depths={key: depth}, failures=[], complete=True)
    calls = []

    def measure(candidate, **kwargs):
        calls.append(kwargs)
        return checked

    output = repair_acceptance.measure_candidate(None, SimpleNamespace(measure=measure), None,
        None, protected, None, .001, frames=None)
    assert calls == [dict(protected_depths=protected['depths'], frames=None)]
    assert output[0] is checked
    assert output[1] == ([key] if _regressed(depth, .0003) else [])
    assert output[3] == (not (depth**2 >= .001 - 1e-10))

def test_cpu_fallback_repair_matches_frozen_qpos_bytes(monkeypatch):
    from omniretargeting._fast.mesh_repair import repair_original_meshes
    from omniretargeting._fast import gpu_geometry
    from tests.test_limb_repair_scope import _floor_problem
    monkeypatch.setattr(gpu_geometry, 'initialize_static', lambda geometry: False)
    obj, poses, scene, _ = _floor_problem(.12)
    actual, report = repair_original_meshes(obj, poses, scene_query=scene)
    # Captured before adding any ablation implementation, in this same
    # five-frame unit fixture. Hex literals preserve every float64 bit.
    row = [0., 0., float.fromhex('0x1.051eb851eb852p-1'), 1., 0., 0., 0., 0.,
           float.fromhex('0x1.666663bbf377fp-2'), -float.fromhex('0x1.80000055e09a7p-1'),
           float.fromhex('0x1.999763fd90c54p-2'), float.fromhex('0x1.66666b6fd0732p-2'),
           -float.fromhex('0x1.7ffffdf0c74b9p-1'), float.fromhex('0x1.99455d8642662p-2'), 0.]
    expected = np.tile(row, (5, 1))
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert [r['alpha'] for r in report['history']] == [1., 0., 1., 0., 1., 0., 1., 0.]
    assert report['local_angle_budget_rad'] == .05
