"""Priority registration never promotes a soft contact into a frozen hold."""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.contact_guard import ContactGuard
from omniretargeting._fast.priority_clearance import PriorityClearance
from omniretargeting._fast.contact_protection import endpoint_protection, surface_protection
from tests.test_fast_contact_guard import _acquired


def coordinator(obj, poses):
    class Empty:
        def measure(self, q, **kwargs):
            return dict(rows=[], depths={}, complete=True, failures=[])

        def linear_rows(self, measured, *, frames=None):
            count = len(poses) if frames is None else len(frames)
            return sparse.csr_matrix((0, count * obj.width)), np.empty(0)

    c = PriorityClearance(obj, Empty(), poses)
    obj.priority_clearance, obj.candidate_guards = c, [c]
    return c


def test_priority_endpoint_protection_matches_original_support_hold_scope():
    obj, poses = _acquired()
    pattern = ['support', 'hold', 'control', 'follow', 'touch', 'free', None, 'hold']
    roles = (pattern * (len(poses) // len(pattern) + 1))[:len(poses)]
    obj.tasks.contact_phase = SimpleNamespace(role=lambda name, f: roles[f], surface=None, surface_roles={})
    active = obj.effectors[0]['active'].copy()
    expected = np.isin(roles, ['support', 'hold'])
    legacy = AcquisitionNormalGuard(obj, poses)
    c = coordinator(obj, poses)
    acquisition = AcquisitionNormalGuard(obj, poses)
    attained = ContactGuard(obj, poses)
    np.testing.assert_array_equal(legacy.groups[0][5][:, 0], expected)
    np.testing.assert_array_equal(acquisition.groups[0][5], legacy.groups[0][5])
    np.testing.assert_array_equal(attained.groups[0].active[:, 0], expected)
    np.testing.assert_array_equal(obj.effectors[0]['active'], active)
    # A clean lower-role contact can regress without penetration funding; the
    # source contact still exists as an objective term with unchanged targets.
    for frame in (2, 3, 4):
        q = poses.copy()
        q[frame, 0] += .003
        assert c.accepts(q, frames=range(frame, frame + 1))
        assert not c.soft_limits(c.current, frames=range(frame, frame + 1))
    q = poses.copy()
    q[0, 0] += .003
    assert not c.accepts(q, frames=range(0, 1))  # Original support remains protected.


def test_priority_surface_protection_uses_existing_policy_without_sticking_override():
    obj, poses = _acquired(surfaces=True)
    for eff in obj.effectors:
        eff['active'][:] = False
    surf = obj.surfaces[0]
    shape = surf['active'].shape
    support, hold = np.zeros(shape, bool), np.zeros(shape, bool)
    support[0], hold[1] = True, True
    surf['sticking'][:] = True  # Explicit roles still take precedence, as before.
    obj.tasks.contact_phase = SimpleNamespace(surface=surf['task'],
        surface_roles={'support': support, 'hold': hold, 'follow': ~(support | hold)},
        role=lambda name, frame: 'follow')
    before = {k: surf[k].copy() for k in ('active', 'targets', 'sticking')}
    legacy = AcquisitionNormalGuard(obj, poses)
    c = coordinator(obj, poses)
    acquisition, attained = AcquisitionNormalGuard(obj, poses), ContactGuard(obj, poses)
    np.testing.assert_array_equal(acquisition.groups[0][5], legacy.groups[0][5])
    np.testing.assert_array_equal(attained.groups[0].active, support | hold)
    for key, value in before.items():
        np.testing.assert_array_equal(surf[key], value)
    # Every registered metric belongs to the old protection mask.
    blocks = [b for adapter in c.contacts for b in adapter.evaluate(poses, frames=range(len(poses)))]
    assert blocks and all(set(b.frames).issubset({0, 1}) for b in blocks)


def test_no_policy_fallback_and_unassociated_surface_sticking_are_unchanged():
    obj, poses = _acquired(surfaces=True)
    obj.tasks.contact_phase = None
    eff = obj.effectors[0]
    eff['surface_only'][::2] = True
    np.testing.assert_array_equal(endpoint_protection(obj, eff), ~eff['surface_only'])
    surf = obj.surfaces[0]
    np.testing.assert_array_equal(surface_protection(obj, surf), surf['sticking'])
    legacy = AcquisitionNormalGuard(obj, poses)
    coordinator(obj, poses)
    priority = AcquisitionNormalGuard(obj, poses)
    assert len(legacy.groups) == len(priority.groups)
    for a, b in zip(legacy.groups, priority.groups):
        np.testing.assert_array_equal(a[5], b[5])
