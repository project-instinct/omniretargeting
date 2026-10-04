"""Necessary contact-funding bounds for window QP proposals.

Every constant belongs to one committed/fork baseline. Eligibility and depth
caps do not prove donor progress: the exact net nonlinear check remains final.
Credit is per (contact identity, donor), never a shared all-contact pool.
"""

from dataclasses import replace

import numpy as np

from .priority_clearance import CONTACT_SLACK


def columns(matrix):
    return set(map(int, matrix.indices[np.abs(matrix.data) > 1e-12]))


class WindowContactDomain:
    def __init__(self, state):
        baseline = state.baseline
        base = baseline.coordinator
        contact_baseline = ({identity: (block, i, value) for block in baseline.blocks
                             for i, (identity, value) in enumerate(zip(block.identities, block.values))}
                            if baseline is state else {})
        donor_columns = {}
        rows = baseline.measured['rows'] if baseline is state else ()
        for i, (key, _) in enumerate(rows):
            support = columns(baseline.matrix.getrow(i))
            for body in baseline.owners[key]:
                donor_columns.setdefault((key[0], body), set()).update(support)
        self.bounds, self.caps, self.hard_upper, self.selected = [], [], [], []
        self.soft_blocks, self.limits = [], {}
        for number, block in enumerate(state.blocks):
            bound, caps = np.empty(len(block.bound)), np.zeros(len(block.bound))
            # The net funding domain is constant throughout a private solve.
            # Reuse it instead of recomputing every baseline support set.
            identities = block.identities if baseline is state else ()
            if baseline is not state:
                if block.identities != baseline.blocks[number].identities:
                    raise ValueError('Private contact channels changed identity')
                bound = baseline.contact_domain.bounds[number]
                caps = baseline.contact_domain.caps[number]
            for i, identity in enumerate(identities):
                original, j, original_value = contact_baseline[identity]
                bound[i] = max(float(original_value),
                               float(original.bound[j] + original.own[j]))
                start, width = (j, 1) if original.scalar else (3 * j, 3)
                support = columns(original.matrix[start:start + width])
                frame, body = int(original.frames[j]), int(original.bodies[j])
                for donor in np.flatnonzero(base.current_depth[frame] > 0.):
                    donor = int(donor)
                    if donor == body or not support.intersection(donor_columns.get((frame, donor), ())):
                        continue
                    available = min(base.current_depth[frame, donor], base.best_depth[frame, donor],
                                    base.initial_depth[frame, donor] - base.spent.get((identity, donor), 0.))
                    caps[i] = max(caps[i], float(available))
            selected = caps > 0.
            bound.setflags(write=False)
            caps.setflags(write=False)
            hard = bound + caps
            upper = np.minimum(np.maximum(hard + CONTACT_SLACK / 2, block.values), hard + CONTACT_SLACK)
            self.bounds.append(bound)
            self.caps.append(caps)
            self.hard_upper.append(upper)
            self.selected.append(selected)
            if not selected.any():
                continue
            mask = selected if block.scalar else np.repeat(selected, 3)
            soft = replace(block, matrix=block.matrix[mask], residual=block.residual[selected],
                           bound=bound[selected], own=np.zeros(selected.sum()),
                           frames=block.frames[selected], bodies=block.bodies[selected],
                           ranks=block.ranks[selected],
                           identities=tuple(k for k, keep in zip(block.identities, selected) if keep))
            self.soft_blocks.append(soft)
            for key, cap in zip(soft.keys, caps[selected]):
                self.limits[key] = max(self.limits.get(key, 0.), float(cap))

    def soft_rows(self):
        return [(b.matrix, b.bound - b.residual, np.ones(len(b.bound)), b.keys)
                for b in self.soft_blocks if b.scalar]

    def soft_cones(self):
        return [(b.matrix, b.residual.copy(), b.bound.copy(), np.ones(len(b.bound)), b.keys)
                for b in self.soft_blocks if not b.scalar]

    def hard_bounds(self, blocks):
        linear, cones = [], []
        for b, upper in zip(blocks, self.hard_upper):
            if b.scalar:
                linear.append((b.matrix, np.full(len(upper), -np.inf), upper - b.residual))
            else:
                cones.append((b.matrix, b.residual.copy(), upper.copy()))
        return linear, cones

    def soft_values(self, blocks):
        linear, cones = [], []
        for b, bound, selected in zip(blocks, self.bounds, self.selected):
            if not selected.any():
                continue
            if b.scalar:
                linear.append(bound[selected] - b.residual[selected])
            else:
                cones.append((b.residual[selected].copy(), bound[selected].copy()))
        return linear, cones

    def values(self, blocks):
        """Actual slack uses this iteration's frozen B, including every channel."""
        values = {}
        for b, bound, selected in zip(blocks, self.bounds, self.selected):
            for key, value, keep in zip(b.keys, np.maximum(b.values - bound, 0.), selected):
                if keep:
                    values[key] = max(values.get(key, 0.), float(value))
        return values

    def levels(self, blocks):
        result = np.zeros(3)
        ranks = dict(support=0, control=1, follow=2)
        for key, value in self.values(blocks).items():
            result[ranks[key[1]]] += value
        return result.tolist()
