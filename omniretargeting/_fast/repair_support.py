"""Freeze acquired world continuity of source-supported endpoints for repair.

Support and unresolved geometric holds share protection, without treating a
hold as mechanically verified support. The point and its mapped body origin
have separate vector budgets. A stencil touching support includes the handoff;
its allowance is the entry trajectory's own seven-stencil windowed budget.
No endpoint names, source motion classifier or mutable pose reference is used.
"""
import numpy as np
from scipy import sparse

from .._refinement.math import _difference_operator
from ..motion_budgets import windowed_budget
from .rows import _block_rows


class SupportContinuity:
    def __init__(self, objective, kin, *, slack, enabled=True):
        self.groups = []
        self.frames = objective.frames
        self.width = len(kin.columns)
        self.slack = slack
        if not enabled:
            return
        policy = getattr(objective.tasks, "contact_phase", None)
        if policy is None:
            return
        body_masks = {}
        points = []
        for eff in objective.effectors:
            role = getattr(policy, "roles", {}).get(eff["name"])
            if role is None:
                continue
            active = np.isin(role, ("support", "hold"))
            body = eff["body"]
            body_masks.setdefault(body, np.zeros(self.frames, bool))[:] |= active
            # A material tracking point avoids changes of contact-marker choice
            # being mistaken for motion of one physical endpoint.
            offset = objective.tasks.effectors[eff["name"]]["tracking_offset"]
            if np.any(offset):
                points.append((eff["name"], body, np.asarray(offset), active))
        for surf in objective.surfaces:
            if surf["task"] is not getattr(policy, "surface", None):
                continue
            roles = getattr(policy, "surface_roles", {})
            active = np.zeros_like(surf["active"], bool)
            for role in ("support", "hold"):
                active |= roles.get(role, False)
            for index, body in enumerate(surf["bodies"]):
                body_masks.setdefault(int(body), np.zeros(self.frames, bool))[:] |= active[:, index]
        points += [(objective.model.body(body).name, body, np.zeros(3), mask)
                   for body, mask in body_masks.items()]
        for name, body, offset, active in points:
            if not active.any():
                continue
            before = kin.points(np.array([body]), offset[None])[:, 0]
            for order in (1, 2):
                if self.frames <= order:
                    continue
                mask = np.lib.stride_tricks.sliding_window_view(active, order + 1).any(axis=1)
                selected = np.flatnonzero(mask)
                diff = _difference_operator(self.frames, order)
                op = sparse.kron(diff[selected], sparse.eye(3), format="csr")
                limit = windowed_budget(np.linalg.norm(np.diff(before, n=order, axis=0), axis=1))[mask]
                limit.setflags(write=False)
                self.groups.append((name, body, offset.copy(), order, selected, op, limit))

    def add(self, bounds, kin):
        for name, body, offset, order, selected, op, limit in self.groups:
            point = kin.points(np.array([body]), offset[None])
            jac = kin.point_jacobians(np.array([body]), point)[:, 0]
            matrix = op @ _block_rows(jac, self.width)
            value = (op @ point.ravel()).reshape(-1, 3)
            # Leave nonlinear curvature room; restoration cannot turn an
            # over-budget candidate into a larger permanent allowance.
            allowed = np.minimum(limit + self.slack,
                                 np.maximum(limit + self.slack / 4, np.linalg.norm(value, axis=1)))
            bounds.norms.append((matrix, value, allowed))

    def check(self, kin):
        worst = 0.
        for name, body, offset, order, selected, op, limit in self.groups:
            point = kin.points(np.array([body]), offset[None])
            value = np.linalg.norm((op @ point.ravel()).reshape(-1, 3), axis=1)
            delta = value - limit
            worst = max(worst, float(delta.max(initial=0.)))
            if np.any(delta > self.slack):
                i = int(np.argmax(delta))
                return False, dict(reason="support endpoint temporal budget", point=name,
                                   order=order, stencil=int(selected[i]),
                                   actual_m=float(value[i]), allowed_m=float(limit[i]),
                                   excess_m=float(delta[i]))
        return True, dict(support_endpoint_temporal_excess_m=worst,
                          support_endpoint_temporal_groups=len(self.groups))
