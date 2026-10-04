"""Contact epigraphs for the opt-in original-geometry priority contract.

The geometry, targets and reference errors belong to the original guards.
This adapter changes their level, never their data. Norms remain norms even
at zero error: replacing them by a zero norm gradient would lose the bound.
"""

from dataclasses import dataclass

import numpy as np

from .rows import _block_rows


@dataclass
class ContactRows:
    matrix: object
    residual: np.ndarray
    bound: np.ndarray
    own: np.ndarray
    frames: np.ndarray
    bodies: np.ndarray
    ranks: np.ndarray
    identities: tuple
    scalar: bool

    @property
    def values(self):
        return (np.maximum(self.residual, 0.) if self.scalar
                else np.linalg.norm(self.residual, axis=1))

    @property
    def keys(self):
        # One maximum slack per resolved role/body/frame. Duplicating skin
        # samples or checking several error channels never multiplies weight.
        return tuple(('contact', ('support', 'control', 'follow')[int(r)], int(f), int(b))
                     for r, f, b in zip(self.ranks, self.frames, self.bodies))


def _rank(role):
    return {'support': 0, 'hold': 0, 'control': 1, 'follow': 2}.get(role, 0)


class PriorityContacts:
    def __init__(self, guard, name):
        self.guard, self.name = guard, name
        self.objective = guard.objective
        self.model = self.objective.model
        self.width = self.objective.width
        # Evidence ranks freeze with targets; subsequent task mutations do
        # not demote an attained support or an unresolved hold.
        self.ranks = []
        policy = getattr(self.objective.tasks, 'contact_phase', None)
        for index, group in enumerate(guard.groups):
            if isinstance(group, tuple):
                kind, _, _, _, _, active, *_ = group
                name = guard.names[index]
            else:
                kind, active, name = group.kind, group.active, group.name
            rank = np.zeros(active.shape, dtype=int)
            if policy is not None and kind == 'endpoint':
                for frame in range(len(active)):
                    rank[frame] = _rank(policy.role(name, frame))
            elif policy is not None and kind == 'surface':
                number = int(name.rsplit('_', 1)[-1])
                surface = self.objective.surfaces[number]
                if getattr(policy, 'surface', None) is surface['task']:
                    for role, mask in getattr(policy, 'surface_roles', {}).items():
                        rank[np.broadcast_to(mask, active.shape)] = _rank(role)
            rank.setflags(write=False)
            self.ranks.append(rank)

    def evaluate(self, poses, *, jacobian=True, frames=None):
        if frames is not None:
            from .priority_window_contacts import evaluate_contacts
            return evaluate_contacts(self, poses, frames, jacobian=jacobian)
        guard, kin = self.guard, self.guard.kin
        kin.evaluate(poses, with_jacobian=jacobian)
        result = []
        for number, group in enumerate(guard.groups):
            acquired = not isinstance(group, tuple)
            if acquired:
                kind, indices, local = group.kind, group.indices, group.local
                target, normal, active = group.targets, group.normals, group.active
                held, own = group.established, group.yielded
                points = guard._points(group)
            else:
                kind, indices, local, target, normal, active, held, bound, own = group
                points = guard._points(kind, indices, local)
            bodies = self.model.geom_bodyid[indices] if kind == 'surface' else indices
            bodies = np.broadcast_to(self.model.body_weldid[bodies], active.shape)
            jac = (kin.point_jacobians(self.model.geom_bodyid[indices] if kind == 'surface' else indices,
                                       points) if jacobian else None)
            error = points - target
            along = np.sum(error * normal, axis=-1)
            normal_jac = None if jac is None else np.einsum('fsi,fsiw->fsw', normal, jac)

            def add(channel, residual, derivative, limit, mask, scalar=False, allowance=own):
                if not mask.any():
                    return
                frame, sample = np.nonzero(mask)
                matrix = None
                if derivative is not None:
                    if scalar:
                        matrix = _block_rows(derivative, self.width)[mask.ravel()]
                    else:
                        rows = derivative.reshape(len(poses), -1, self.width)
                        matrix = _block_rows(rows, self.width)[np.repeat(mask.ravel(), 3)]
                result.append(ContactRows(
                    matrix, residual[mask], np.broadcast_to(limit, mask.shape)[mask].copy(),
                    np.broadcast_to(allowance, mask.shape)[mask].copy(), frame, bodies[mask],
                    self.ranks[number][mask],
                    tuple((self.name, number, int(f), int(s), channel) for f, s in zip(frame, sample)),
                    scalar))

            if acquired:
                add('normal', along, normal_jac, group.acquired[..., 1] + group.tolerance, held, True)
                add('full', error, jac, group.acquired[..., 0] + group.tolerance, active)
                projected = held & group.projected
                tangent = error - along[..., None] * normal
                tangent_jac = None if jac is None else jac - normal[..., None] * normal_jac[..., None, :]
                add('tangent', tangent, tangent_jac, group.acquired[..., 2] + group.tolerance,
                    projected)
                if guard.preserve_positions:
                    add('anchor', points - group.positions, jac, group.tolerance, held)
            else:
                add('normal', along, normal_jac, bound, held, True)
                add('full', error, jac, bound, active & ~held)
        return result
