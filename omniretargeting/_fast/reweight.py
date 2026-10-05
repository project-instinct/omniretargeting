"""Reweight only offending signals and every stencil touching their frames."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..motion_budgets import chatter_excess
from .runtime_options import OPTIONS
from .continuity import temporal_signals
from ..motion_budgets import CHATTER_EXCESS_TOLERANCE


def offending_frames(coords, fixed, motion):
    """Same selected joint/root signals as the independent final audit."""
    frames = len(motion)
    joint = np.zeros((frames, len(coords.qi)), bool)
    root = np.zeros(frames, bool)
    joint[:] = chatter_excess(fixed[:, coords.qi], motion[:, coords.qi]) > CHATTER_EXCESS_TOLERANCE
    if coords.root_q is not None:
        q = coords.root_q
        root[:] = np.any(
            chatter_excess(fixed[:, q:q + 3], motion[:, q:q + 3]) > CHATTER_EXCESS_TOLERANCE,
            axis=1,
        )
    return joint, root


def reweight_temporal_signals(objective, fixed, motion, factor):
    joint, root = offending_frames(objective.coords, fixed, motion)
    rotation = np.zeros(len(motion), bool)
    before, after = temporal_signals(objective.coords, fixed), temporal_signals(objective.coords, motion)
    if "root_rotation" in before:
        rotation[1:] = np.any(
            chatter_excess(before["root_rotation"], after["root_rotation"]) > CHATTER_EXCESS_TOLERANCE,
            axis=1,
        )
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
