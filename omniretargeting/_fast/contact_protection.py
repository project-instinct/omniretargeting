"""Original acquisition protection masks, independent of priority-solver mode."""

import numpy as np


def endpoint_protection(objective, effector):
    policy = objective.tasks.contact_phase
    if policy is None:
        return ~np.asarray(effector['surface_only'], bool)
    return np.array([policy.role(effector['name'], f) in ('support', 'hold')
                     for f in range(objective.frames)], dtype=bool)


def surface_protection(objective, surface):
    policy = objective.tasks.contact_phase
    if getattr(policy, 'surface', None) is surface['task']:
        roles = getattr(policy, 'surface_roles', {})
        return np.asarray(roles.get('support', False)) | np.asarray(roles.get('hold', False))
    return np.asarray(surface['sticking'], bool)
