"""Opt-in repair ratchet: bounded shallow support tradeoffs, exact geometry.

No policy/normal evidence means no relaxation. Keep the legacy predicate
available to other consumers; only original-mesh repair opts into this rule.
"""

import numpy as np

from .runtime_options import OPTIONS

SHALLOW_INCREASE_M = .0005
NORMAL_ALIGNMENT = 1 - 1e-6


def support_normal(objective, key, row):
    """Same welded support body and an aligned, finite source-contact normal.

    A geom name or gravity direction is never support evidence. A reverse
    witness may qualify only if its actual robot exit direction agrees too.
    """
    if key[1] == 'self':
        return False
    policy = getattr(objective.tasks, 'contact_phase', None)
    if policy is None:
        return False
    frame, geom = key[0], key[-1]
    model = objective.model
    weld = model.body_weldid
    body = weld[model.geom_bodyid[geom]]
    direction = np.asarray(row.get('direction', ()), float)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        return False
    size = np.linalg.norm(direction)
    if size < 1e-12:
        return False
    direction = direction / size

    def aligned(normal):
        normal = np.asarray(normal, float)
        size = np.linalg.norm(normal)
        return (normal.shape == (3,) and np.isfinite(normal).all()
                and size >= 1e-12 and direction @ (normal / size) >= NORMAL_ALIGNMENT)

    for eff in objective.effectors:
        if (weld[eff['body']] == body and eff['active'][frame]
                and not eff['managed'][frame] and policy.role(eff['name'], frame) == 'support'
                and aligned(eff['normal'][frame])):
            return True
    for surf in objective.surfaces:
        if getattr(policy, 'surface', None) is not surf['task']:
            continue
        mask = getattr(policy, 'surface_roles', {}).get('support')
        if mask is None:
            continue
        selected = np.asarray(mask)[frame] & surf['active'][frame] & (weld[surf['bodies']] == body)
        if any(aligned(normal) for normal in surf['normals'][frame, selected]):
            return True
    return False


def depth_cap(protected, normal_support):
    from .mesh_repair import _activation_depth, REGRESSION, PENETRATING_REGRESSION_M
    activation = _activation_depth()
    if not normal_support:
        # No new/deeper self or lateral penetration beyond rounding.
        return protected + REGRESSION
    if protected == 0:
        return activation
    if protected <= activation:
        return min(activation, protected + SHALLOW_INCREASE_M)
    # Only shallow keys get the experimental 0.5 mm allowance.
    return protected + REGRESSION + PENETRATING_REGRESSION_M


def collision_budgets(objective, measurement, baseline):
    """Per-witness proposal caps; the nonlinear predicate checks them again."""
    return np.array([depth_cap(baseline['depths'].get(key, 0.),
                               support_normal(objective, key, row))
                     for key, row in measurement['rows']])


def _side_depths(objective, measurement):
    depths = {}
    for key, row in measurement['rows']:
        if not support_normal(objective, key, row):
            depths[key] = max(depths.get(key, 0.), max(0., -row['clearance']))
    return depths


def regressions(objective, measured, checked, protected):
    """Check key caps AND side witnesses hidden by a geom's support minimum."""
    from .mesh_repair import REGRESSION
    by_key = {}
    for key, row in checked['rows']:
        if row['clearance'] < 0:
            by_key[key] = by_key.get(key, True) and support_normal(objective, key, row)
    before_side, after_side = _side_depths(objective, measured), _side_depths(objective, checked)
    bad = []
    for key, value in checked['depths'].items():
        cap = depth_cap(protected['depths'].get(key, 0.), by_key.get(key, False))
        if (not np.isfinite(value) or value > cap
                or after_side.get(key, 0.) > before_side.get(key, 0.) + REGRESSION):
            bad.append(key)
    return bad


def group_max(measurement, model, group):
    def relevant(key):
        geoms = key[2:] if key[1] == 'self' else key[-1:]
        return group is None or any(model.body_weldid[model.geom_bodyid[g]] == group for g in geoms)
    return max((value for key, value in measurement['depths'].items() if relevant(key)), default=0.)


def measure_candidate(objective, geometry, candidate, measured, protected, group, before, *, frames):
    """Legacy call/order/arithmetic when disabled; complete evidence when on."""
    from .mesh_repair import _regressed
    checked = geometry.measure(candidate,
        protected_depths=None if OPTIONS.abl_repair_ratchet else protected['depths'], frames=frames)
    bad = (regressions(objective, measured, checked, protected) if OPTIONS.abl_repair_ratchet else
           [key for key, value in checked['depths'].items()
            if _regressed(value, protected['depths'].get(key, 0.))])
    after = sum(value**2 for value in checked['depths'].values())
    descent = (group_max(checked, objective.model, group) < group_max(measured, objective.model, group)
               if OPTIONS.abl_repair_ratchet else not (after >= before - 1e-10))
    return checked, bad, after, descent
