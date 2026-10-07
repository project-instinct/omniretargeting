"""Reuse exact evaluated MuJoCo body origins in the temporal guard.

The original guard forwards each frame merely to obtain its root body origin
and Jacobian. Those are already in TrajectoryKinematics. All quaternion,
signal, budget, acceptance and constraint expressions remain inherited,
including the per-frame violation check.
"""

import copy

import numpy as np

from .._sequence.continuity_guard import SequenceContinuityGuard
from .._sequence.tasks import SequenceTasks
from .kinematics import TrajectoryKinematics


class CachedSequenceContinuityGuard(SequenceContinuityGuard):
    def signals(self, poses, span=None):
        first, stop = (0, len(poses)) if span is None else span
        part = np.asarray(poses)[first:stop]
        key = first, stop, part.shape, part.dtype.str, part.tobytes()
        if key in self._signals_cache:
            self._signals_cache.move_to_end(key)
            return self._signals_cache[key]
        if self.tasks.coords.root_q is None or self.tasks.root_body is None:
            return super().signals(poses, span)
        # A private task view suppresses only the redundant root FK loop.
        # It never mutates the original tasks or their model/data buffers.
        view = copy.copy(self)
        view.tasks = copy.copy(self.tasks)
        view.tasks.root_body = None
        signals = SequenceContinuityGuard.signals(view, poses, span)
        kin = self._signal_kin.evaluate(part)
        body = self.tasks.root_body
        point = kin.xpos[:, body] - self.tasks.roots[first:stop]
        if self.correction:
            point -= self.reference['root_anchor_offset'][0][first:stop]
        anchor = point, kin.jacp[:, body]
        # Match the original insertion order, including its cone order.
        result = {}
        for name, value in signals.items():
            if name == 'orientation_offset':
                result['root_anchor_offset'] = anchor
            result[name] = value
        self._signals_cache[key] = result
        return result


# Diagnostics identify the physical certificate, which has not changed.
# Keep the distinct qualified name for normal pickle reconstruction.
CachedSequenceContinuityGuard.__name__ = SequenceContinuityGuard.__name__


def prepare(objective):
    """Opt in existing temporal guard instances without changing their API."""
    for guard in objective.candidate_guards:
        if type(guard) is SequenceContinuityGuard and type(guard.tasks) is SequenceTasks:
            guard.__class__ = CachedSequenceContinuityGuard
            guard._signal_kin = TrajectoryKinematics(
                objective.model, objective.coords.cols, cache=objective.kin.cache)
