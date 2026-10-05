"""Synchronous geometry IPC with immutable request snapshots in shared CPU RAM."""

from dataclasses import dataclass
import pickle

import numpy as np


@dataclass(frozen=True)
class ProtectedBytes:
    value: bytes | int


def prepare(geometry, context, frames):
    # Created before forking. The parent writes only between fully drained
    # requests; workers reconstruct a private dictionary before querying.
    geometry._repair_pose_storage = context.RawArray('d', frames * geometry.model.nq)
    geometry._repair_pose_view = np.frombuffer(geometry._repair_pose_storage, dtype=np.float64).reshape(
        frames, geometry.model.nq)
    size = max(1 << 20, frames * geometry.model.ngeom * 128)
    geometry._repair_depth_storage = context.RawArray('b', size)


def pose_message(geometry, poses):
    target = getattr(geometry, '_repair_pose_view', None)
    if target is None or poses.dtype != np.float64 or poses.shape != target.shape:
        return poses
    np.copyto(target, poses)
    return None


def protected_message(geometry, depths):
    if depths is None or type(depths) is not dict:
        return depths
    # Custom reducers keep the original per-connection dispatch.
    integers = (int, np.int32, np.int64)
    numbers = (*integers, float, np.float32, np.float64)
    if not all(type(key) is tuple and all(type(item) in (*integers, str) for item in key)
               and type(value) in numbers for key, value in depths.items()):
        return depths
    value = pickle.dumps(depths, protocol=pickle.HIGHEST_PROTOCOL)
    storage = geometry._repair_depth_storage
    if len(value) > len(storage):
        return ProtectedBytes(value)
    np.frombuffer(storage, dtype=np.uint8)[:len(value)] = np.frombuffer(value, dtype=np.uint8)
    return ProtectedBytes(len(value))


def decode_protected(geometry, value):
    if not isinstance(value, ProtectedBytes):
        return value
    if isinstance(value.value, bytes):
        return pickle.loads(value.value)
    return pickle.loads(memoryview(geometry._repair_depth_storage).cast('B')[:value.value])
