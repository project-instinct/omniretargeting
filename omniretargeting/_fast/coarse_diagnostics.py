"""Small exact guard margins for diagnosing a rejected transfer without reruns."""

import numpy as np

from .continuity import CorrectionContinuity, ROUND_OFF, correction_derivatives, derivatives
from .proposals import guards_of
from .._sequence.continuity_guard import ACCEPTANCE_SLACK, SequenceContinuityGuard, _change_size


def failure_details(objective, poses, failures):
    """Worst violating stencil per temporal channel, using frozen bounds/slack."""
    result = {}
    coords = objective.coords
    joint_names = {int(address): objective.model.joint(index).name
                   for index, address in enumerate(objective.model.jnt_qposadr)}
    for index, guard in enumerate(guards_of(objective)[1]):
        key = f'{type(guard).__name__}:{index}'
        if key not in failures:
            continue
        channels = []
        if isinstance(guard, CorrectionContinuity):
            values = (correction_derivatives(coords, guard.fixed, poses) if guard.correction
                      else derivatives(coords, poses))
            channels = [(name, np.abs(values[name]), bound, ROUND_OFF)
                        for name, bound in guard.bounds.items()]
            channels += [(name + '_norm', np.linalg.norm(values[name], axis=1), bound, ROUND_OFF)
                         for name, bound in guard.norm_bounds.items()]
        elif isinstance(guard, SequenceContinuityGuard):
            signals = guard.signals(poses)
            channels = [(f'{name}_order_{order}',
                         _change_size(name, np.diff(signals[name][0], n=order, axis=0)),
                         bound, ACCEPTANCE_SLACK) for (name, order), bound in guard.bounds.items()]
        if channels:
            rows = []
            for name, size, bound, slack in channels:
                excess = size - bound - slack
                if not excess.size or not np.any(excess > 0):
                    continue
                where = np.unravel_index(np.argmax(excess), excess.shape)
                row = dict(channel=name, stencil_start=int(where[0]),
                           value=float(size[where]), frozen_bound=float(bound[where]),
                           slack=slack, excess=float(excess[where]))
                if len(where) > 1:
                    row['coordinate'] = int(where[1])
                    if name.startswith('joints_'):
                        row['joint'] = joint_names.get(int(coords.qi[where[1]]))
                rows.append(row)
            result[key] = rows
        elif type(guard).__name__ == 'AcquisitionNormalGuard':
            result[key] = guard.diagnostics(poses)
    return result
