"""Movable frame domain for the established collision-group repair."""

def frames(limits, measured, model, group, *, padding=None):
    from .repair_frames import group_frames, halo_frames
    if padding is None:
        padding = halo_frames(limits.objective.framerate)
    return group_frames(measured, model, group, limits.frames, padding)
