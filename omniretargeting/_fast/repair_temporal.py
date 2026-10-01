"""Optional aggregate temporal check, disabled by default.

Absolute executed derivatives are the default signal. The separate experimental
correction_budget switch changes this check to frozen-entry corrections too.
"""

import numpy as np
from .runtime_options import OPTIONS

TOLERANCE = .02
ROUND_OFF = 2e-7


class RepairTemporal:
    def __init__(self, entry, operators, width):
        self.groups = []
        self.entry = np.array(entry, copy=True)
        self.width = width
        for order, (op, _) in enumerate(operators, 1):
            if order < 2:
                continue
            value = (op @ entry.ravel()).reshape(-1, width)
            rms = np.sqrt(np.mean(value**2, axis=0))
            peak = np.abs(value).max(axis=0)
            self.groups.append((order, op, rms * (1 + TOLERANCE), peak * (1 + TOLERANCE)))

    def add(self, bounds, values):
        for _, op, rms, peak in self.groups:
            current = (op @ (values - self.entry if OPTIONS.correction_budget else values).ravel()).reshape(-1, self.width)
            count = len(current)
            # Each joint owns a separate temporal norm cone; one joint's
            # quiet motion cannot hide extra chatter on another joint.
            ordering = np.arange(op.shape[0]).reshape(-1, self.width).T.ravel()
            norms = np.linalg.norm(current, axis=0)
            allowed = np.sqrt(count) * (rms + ROUND_OFF / 2)
            allowed = np.minimum(np.maximum(allowed, norms), np.sqrt(count) * (rms + ROUND_OFF))
            bounds.norms.append((op[ordering], current.T, allowed))
            slack = np.minimum(ROUND_OFF, np.maximum(ROUND_OFF / 2, np.abs(current) - peak))
            bounds.add(op, (-peak - slack - current).ravel(), (peak + slack - current).ravel())

    def check(self, values):
        excess = []
        for order, op, rms, peak in self.groups:
            current = (op @ (values - self.entry if OPTIONS.correction_budget else values).ravel()).reshape(-1, self.width)
            for name, delta in [('rms', np.sqrt(np.mean(current**2, axis=0)) - rms),
                                ('peak', np.abs(current).max(axis=0) - peak)]:
                if delta.max(initial=0) > ROUND_OFF:
                    joint = int(np.argmax(delta))
                    excess.append(dict(order=order, metric=name, joint_column=joint,
                                       excess=float(delta[joint]), slack=ROUND_OFF))
        return not excess, excess
