"""Net parent acceptance, immutable credit, and exact local contact channels."""

from copy import copy

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.priority_contacts import ContactRows


class Provider:
    def __init__(self):
        self.calls = []
        self.fail = False

    def fork_private_client(self):
        clone = copy(self)
        clone.calls = []
        return clone

    def measure(self, poses, *, frames=None, jacobians=False):
        frames = range(len(poses)) if frames is None else frames
        self.calls.append(tuple(frames))
        rows = [((f, 'static', 0), dict(clearance=poses[f, 0] - .05)) for f in frames]
        return dict(rows=rows, depths={k: max(-r['clearance'], 0.) for k, r in rows},
                    complete=not self.fail, failures=['unknown'] if self.fail else [])

    def linear_rows(self, measured, *, frames=None):
        frames = range(len(measured['rows'])) if frames is None else frames
        rows = measured['rows']
        mat = sparse.csr_matrix((np.ones(len(rows)), (np.arange(len(rows)),
                               [(key[0] - frames.start) * 2 for key, _ in rows])),
                               shape=(len(rows), len(frames) * 2))
        return mat, np.array([row['clearance'] for _, row in rows])


class Contact:
    def evaluate(self, poses, *, jacobian=True, frames=None):
        frames = range(len(poses)) if frames is None else frames
        part = slice(frames.start, frames.stop)
        n = len(frames)
        return [ContactRows(sparse.kron(sparse.eye(n), [[1., -1.]], format='csr') if jacobian else None,
                            poses[part, 0] - poses[part, 1], np.full(n, .001), np.zeros(n),
                            np.array(frames), np.full(n, 2), np.zeros(n, int),
                            tuple(('contact', 0, f, 0, 'normal') for f in frames), True)]
























@pytest.mark.parametrize('kind', ['acquisition', 'contact'])
def test_window_contact_acceptance_matches_original_frozen_channels(kind, monkeypatch):
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    from omniretargeting._fast.contact_guard import ContactGuard
    from omniretargeting._fast.priority_contacts import PriorityContacts
    from tests.test_fast_contact_guard import _acquired
    objective, poses = _acquired()
    for effector in objective.effectors:
        effector['normal'][:] = [0., 0., 1.]
    module = 'omniretargeting._fast.' + ('acquisition_guard' if kind == 'acquisition' else 'contact_guard')
    monkeypatch.setattr(module + '.on_surface', lambda obj, p: np.ones(p.shape[:2], bool))
    monkeypatch.setattr(module + '.body_penetration',
                        lambda obj, p: np.full((len(p), obj.model.nbody), .02))
    guard = (AcquisitionNormalGuard if kind == 'acquisition' else ContactGuard)(objective, poses)
    frames = range(2, 5)
    for dz in [-.0211, -.0011, -.0005, 0., .0005, .0011, .019, .0211]:
        q = poses.copy()
        q[2:5, 2] += dz
        assert guard.accepts(q, frames=frames) == guard.accepts(q)
        np.testing.assert_array_equal(guard.violation_frames(q, frames=frames), guard.violation_frames(q))
    adapter = PriorityContacts(guard, 'frozen')
    blocks = adapter.evaluate(poses, frames=frames)
    assert all(b.matrix.shape[1] == len(frames) * objective.width for b in blocks)
    inward = [b for b in blocks if b.identities[0][-1] == 'normal_inward']
    assert inward
    np.testing.assert_array_equal(inward[0].own, 0. if kind == 'acquisition' else .02)
