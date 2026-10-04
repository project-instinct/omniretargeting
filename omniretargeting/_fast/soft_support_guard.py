"""Immutable acquisition protection restricted to mechanics-ranked supports."""
import copy
import numpy as np
from .acquisition_guard import AcquisitionNormalGuard
from .kinematics import TrajectoryKinematics
from .soft_support_rows import surface_roles


def support_guard(objective, frozen):
    guard = copy.copy(frozen)
    guard.kin = TrajectoryKinematics(objective.model, objective.coords.cols, cache=objective.kin.cache)
    guard.groups, guard.names = [], []
    policy = getattr(objective.tasks, 'contact_phase', None)
    for name, group in zip(frozen.names, frozen.groups):
        if group[0] == 'surface':
            index = int(name.rsplit('_', 1)[1])
            ranked = surface_roles(objective, objective.surfaces[index]) == 'support'
        else:
            roles = None if policy is None else policy.roles.get(name)
            ranked = np.zeros(group[5].shape, bool) if roles is None else (np.asarray(roles) == 'support')[:, None]
        active = group[5] & ranked
        if not active.any():
            continue
        active.setflags(write=False)
        values = list(group); values[5] = active
        guard.groups.append(tuple(values)); guard.names.append(name)
    guard._support_only = True
    return guard


def project_supports(objective, poses, entry, *, objective_aware=False):
    from .soft_acquisition_v4 import final_guards
    from .soft_projection import project_bulk
    frozen = next(g for g in objective.candidate_guards if isinstance(g, AcquisitionNormalGuard))
    selected = support_guard(objective, frozen)
    with final_guards(objective):
        objective.candidate_guards = [*objective.candidate_guards, selected]
        result, report = project_bulk(objective, poses, entry, measure_costs=True,
                                      objective_aware=objective_aware)
    report['guard_scope'] = 'original_mechanics_ranked_support_samples_only'
    report['support_points'] = sum(int(g[5].sum()) for g in selected.groups)
    report['support_approach_balls'] = sum(int((g[5]&~g[6]).sum()) for g in selected.groups)
    return result, report
