"""Exact per-pose MuJoCo caches for independently evaluated frame windows."""

from collections import OrderedDict

import numpy as np

from .kinematics import TrajectoryKinematics


def frame_range(frames, count):
    if not isinstance(frames, range) or frames.step != 1:
        raise ValueError('frames must be a contiguous range with step=1')
    if not 0 <= frames.start < frames.stop <= count:
        raise ValueError('frames must be a nonempty range inside the trajectory')
    return frames


class FrameKinematicsCache:
    """Bounded cache keyed by the exact bytes of one pose, never a full clip.

    Values are immutable. A returned kinematic view owns its array assembly,
    so evaluating a second window cannot overwrite a retained first view.
    """

    def __init__(self, model, columns, capacity=2048):
        self.worker = TrajectoryKinematics(model, columns)
        self.entries = OrderedDict()
        self.capacity = capacity
        self.computed_frames = 0

    def evaluate(self, poses, frames, with_jacobian=True):
        frames = frame_range(frames, len(poses))
        part = np.asarray(poses)[frames.start:frames.stop]
        keys = [(pose.dtype.str, pose.shape, pose.tobytes()) for pose in part]
        missing = {}
        for index, key in enumerate(keys):
            previous = self.entries.get(key)
            if previous is None or (with_jacobian and 'jacp' not in previous):
                missing.setdefault(key, index)
        names = ['xpos', 'xmat', 'geom_xpos', 'geom_xmat']
        if with_jacobian:
            names += ['jacp', 'jacr']
        if missing:
            indices = list(missing.values())
            kin = self.worker.evaluate(part[indices], with_jacobian, native=True)
            self.computed_frames += len(indices)
            for index, key in enumerate(missing):
                # Copy one frame: a slice must not retain a whole trial batch.
                values = {name: getattr(kin, name)[index].copy() for name in names}
                for value in values.values():
                    value.setflags(write=False)
                self.entries[key] = values
        kin = object.__new__(TrajectoryKinematics)
        kin.model, kin.columns = self.worker.model, self.worker.columns
        kin.data = self.worker.data
        kin.with_jacobian = with_jacobian
        kin.cache = OrderedDict()
        kin.global_frames = frames
        for name in names:
            value = np.stack([self.entries[key][name] for key in keys])
            value.setflags(write=False)
            setattr(kin, name, value)
        key = (kin.columns.tobytes(), part.shape, part.dtype.str, part.tobytes())
        kin.cache[key] = {name: getattr(kin, name) for name in names}
        for key in keys:
            self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return kin


def window_kinematics(owner, poses, frames, with_jacobian=True):
    cache = getattr(owner, '_window_frame_cache', None)
    if cache is None:
        owner._window_frame_cache = cache = FrameKinematicsCache(owner.model, owner.coords.cols)
    return cache.evaluate(poses, frames, with_jacobian)
