"""Reweight only offending signals and every stencil touching their frames."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..motion_budgets import windowed_budget
from .runtime_options import OPTIONS
from .continuity import ROUND_OFF, derivatives, correction_derivatives


def offending_frames(coords, fixed, motion):
    """Same selected joint/root signals as the independent final audit."""
    signal = motion - fixed if OPTIONS.correction_budget else motion
    frames = len(motion)
    joint = np.zeros((frames, len(coords.qi)), bool)
    root = np.zeros(frames, bool)
    for order in (2, 3):
        if frames <= order:
            continue
        before = np.diff(fixed[:, coords.qi], n=order, axis=0)
        after = np.diff(signal[:, coords.qi], n=order, axis=0)
        bad = np.abs(after) > windowed_budget(before) + ROUND_OFF
        for start, column in zip(*np.nonzero(bad)):
            joint[start:start + order + 1, column] = True
        if coords.root_q is not None:
            q = coords.root_q
            old = np.linalg.norm(np.diff(fixed[:, q:q + 3], n=order, axis=0), axis=1)
            new = np.linalg.norm(np.diff(signal[:, q:q + 3], n=order, axis=0), axis=1)
            for start in np.flatnonzero(new > windowed_budget(old) + ROUND_OFF):
                root[start:start + order + 1] = True
    return joint, root


def reweight_temporal_signals(objective, fixed, motion, factor):
    joint, root = offending_frames(objective.coords, fixed, motion)
    rotation = np.zeros(len(motion), bool)
    before = derivatives(objective.coords, fixed)
    delta = correction_derivatives(objective.coords, fixed, motion)
    for order in (2, 3):
        name = f'root_rotation_order_{order}'
        if not OPTIONS.correction_budget or name not in before:
            continue
        bad = np.abs(delta[name]) > windowed_budget(before[name]) + ROUND_OFF
        for start in np.flatnonzero(bad.any(axis=1)):
            rotation[start:start + order + 1] = True
    for index in range(len(objective.temporal_weight)):
        order = index + 1
        if len(motion) <= order:
            continue
        # Includes the stencils starting before and ending after each marked
        # interval. No window edge silently loses velocity/acceleration/jerk.
        joint_touched = sliding_window_view(joint, order + 1, axis=0).any(axis=-1)
        root_touched = sliding_window_view(root, order + 1).any(axis=-1)
        objective.joint_temporal_weight[index][joint_touched] *= factor
        objective.root_translation_weight[index][root_touched] *= factor
        rotation_touched = sliding_window_view(rotation, order + 1).any(axis=-1)
        objective.root_rotation_weight[index][rotation_touched] *= factor
    return {'joint_columns': np.flatnonzero(joint.any(axis=0)).tolist(),
            'joint_frame_coordinates': int(joint.sum()),
            'root_translation_frames': np.flatnonzero(root).tolist(),
            'root_rotation_reweighted': bool(rotation.any()), 'world_paths_reweighted': False}
