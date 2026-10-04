"""Conservative read/write footprints for ordered parallel mesh repair.

The write set comes from the existing projection, including positive self
partners and the contact compensation closure. Frame-local layout constraints
can read another limb, so every body at those frames is a dependency. Third
differences extend that read set by three frames. Unknown/global contracts are
handled by the coordinator's serial fallback.
"""

from dataclasses import dataclass
import pickle

import numpy as np

from .repair_frames import restrict


@dataclass
class Footprint:
    write: np.ndarray
    read_frames: np.ndarray
    bodies: frozenset

    def disjoint(self, other):
        return not (np.any(self.write.any(axis=1) & other.read_frames)
                    or np.any(other.write.any(axis=1) & self.read_frames))


def footprint(limits, model, measured):
    from .mesh_repair import _joint_projection
    projection = restrict(_joint_projection(limits, model, measured), limits.frame_mask)
    write = np.zeros(limits.frames * limits.width, bool)
    write[projection.indices] = True
    write = write.reshape(limits.frames, limits.width)
    frames = write.any(axis=1)
    read = frames.copy()
    for offset in range(1, 4):
        read[offset:] |= frames[:-offset]
        read[:-offset] |= frames[offset:]
    joint_bodies = set(model.jnt_bodyid[model.dof_jntid[limits.coords.di[write.any(axis=0)]]])
    affected = set()
    for body in range(model.nbody):
        ancestor = body
        while ancestor:
            if ancestor in joint_bodies:
                affected.add(body)
                break
            ancestor = int(model.body_parentid[ancestor])
    return Footprint(write, read, frozenset(affected))


def certificate(limits, poses, measured, protected, fixed_q, depth, scope):
    """Exact scoped bytes, including mutable contacts and witness/protection rows.

Outside this scope the step is zero and the known constraints are constants.
They do not change the projected QP. Parent checks still test the complete
rebased trajectory and all original geometry, in the original commit order.
Immutable budgets/reference are covered in full; they are never reconstructed.
"""
    read = scope.read_frames
    state = (limits.movable_body, limits.frame_mask,
             [value[read] for value in limits.anchors],
             [value[read] for value in limits.target_budgets],
             [(key, row) for key, row in measured['rows'] if read[key[0]]],
             [(key, value) for key, value in measured['depths'].items() if read[key[0]]],
             [(key, value) for key, value in protected['depths'].items() if read[key[0]]],
             measured.get('failures', ()), measured.get('complete', True), fixed_q, depth)
    return (poses.dtype.str, poses.shape, poses[read].tobytes(),
            scope.write.tobytes(), read.tobytes(), scope.bodies,
            pickle.dumps(state, protocol=5), limits.entry.tobytes(),
            limits.objective.reference.tobytes(), limits.penetration_yield.tobytes(),
            limits.lower.tobytes(), limits.upper.tobytes(),
            tuple(value.tobytes() for value in limits.tracking_budgets))
