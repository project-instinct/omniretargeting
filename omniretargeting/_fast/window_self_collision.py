"""Adapt the existing exact global-frame self cache to window objective rows."""

import os

from .batched_self_collision import BatchedSelfCollision

# Exact frame witnesses shared across acquisition window views.
WINDOW_SELF_FRAME_CACHE = True


class WindowSelfCollision:
    """One view of a worker-owned query; Jacobians always use current qpos."""

    workers = 1

    def __init__(self, provider, query):
        self.provider, self.query = provider, query
        self.owner_pid = os.getpid()
        self.fallback = None
        self.diagnostics = {}

    def bind(self, poses, frames):
        self.poses, self.frames = poses, frames

    def rows(self, poses, kin, with_jacobian, target=None):
        provider = self.provider
        if provider.query.cfg['detection_distance'] != self.query.cfg['detection_distance']:
            # The provider freezes its detection band. Dynamic callers retain
            # the existing threshold-aware batch query instead of stale rows.
            if self.fallback is None:
                self.fallback = BatchedSelfCollision(self.query, workers=1)
            result = self.fallback.rows(poses, kin, with_jacobian, target)
            self.diagnostics = dict(self.fallback.diagnostics)
            return result
        if self.owner_pid != os.getpid():
            raise RuntimeError('Window self query must belong to the current worker')
        # Positive activation is a filtering optimization in this API. For
        # other targets query the full original band, then filter below.
        activation = target if target is not None and 0 <= target <= .008 else None
        measured = provider.query_frames(self.poses, self.frames,
                                         jacobians=with_jacobian,
                                         activation_distance=activation, kinematics=kin)
        per_frame = [[] for _ in poses]
        for (frame, *_), row in measured['rows']:
            if target is not None and row['clearance'] >= target:
                continue
            if with_jacobian:
                row['_jacobian'] = row.pop('jacobian')
            per_frame[frame - self.frames.start].append(row)
        self.diagnostics = measured['diagnostics']
        return per_frame, len(measured['failures'])
