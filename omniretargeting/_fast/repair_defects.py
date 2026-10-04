"""Opt-in repair scheduling for remaining scene defects and active self pairs.

The 5 mm scene band chooses work, not acceptance: original geometry still
protects every key and detects any new penetration. Self pairs retain the
existing repair activation. An entry-activated window remains movable until
its group stops, including after the pair falls below activation.
"""

SCENE_WORK_DEPTH_M = .005


def halo(framerate):
    """Retain the established temporal ramp domain around active defects."""
    from .repair_frames import halo_frames
    return halo_frames(framerate)


def active(key, depth):
    from .mesh_repair import _activation_depth
    return depth > (_activation_depth() if key[1] == 'self' else SCENE_WORK_DEPTH_M)


def relevant(model, group, key):
    geoms = key[2:] if key[1] == 'self' else key[-1:]
    return group is None or any(model.body_weldid[model.geom_bodyid[g]] == group for g in geoms)


def has_defect(measured, model, group):
    return any(active(key, depth) and relevant(model, group, key)
               for key, depth in measured['depths'].items())


def frames(limits, measured, model, group, *, padding=None):
    from .repair_frames import group_frames, halo_frames
    from .runtime_options import OPTIONS
    if OPTIONS.repair_defect_windows:
        masks = getattr(limits, '_repair_defect_masks', None)
        if masks is not None and group in masks:
            return masks[group]
    if OPTIONS.repair_defect_windows:
        padding = halo(limits.objective.framerate)
    elif padding is None:
        padding = halo_frames(limits.objective.framerate)
    return group_frames(measured, model, group, limits.frames, padding)
