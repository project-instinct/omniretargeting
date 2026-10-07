"""Geometric make-before-break verification after acquisition, never a re-solve."""
import numpy as np

from ..contact_schedule import _iter_true_runs


def supported_contact_runs(active, support):
    """Role flapping within one observed contact is not a physical release."""
    result = np.zeros_like(active, dtype=bool)
    for start, stop in _iter_true_runs(active):
        if np.any(support[start:stop]):
            result[start:stop] = True
    return result


def handoff_gaps(outgoing, incoming, gaps, tolerance=.005):
    """Inspect the final shared-contact frame before each outgoing release."""
    records = []
    for start, stop in _iter_true_runs(outgoing):
        if stop == len(outgoing):
            continue
        shared = np.flatnonzero(np.any(incoming[start:stop], axis=1)) + start
        if not len(shared):
            continue
        frame = int(shared[-1])
        columns = np.flatnonzero(incoming[frame])
        values = np.asarray(gaps)[frame, columns]
        records.append(dict(release_frame=int(stop), last_overlap_frame=frame,
            incoming_columns=columns.tolist(), normal_gaps_m=values.tolist(),
            acquired_before_release=bool(np.all(np.isfinite(values) & (values <= tolerance)))))
    return records


def audit_support_handoffs(objective, motion):
    """Check observed static proximal supports handing over to sole contacts.

    Contact evidence is read from the original tasks, including unreachable
    contacts whose optimization rows yielded. A failed overlap remains a
    defect and never rewrites the source schedule or claims force balance.
    """
    model, kin = objective.model, objective.kin
    kin.evaluate(motion, with_jacobian=False)
    soles = [e for e in objective.effectors if 'origin_body' in e]
    frames = len(motion)
    if not soles:
        return dict(status='no_calibrated_soles', events=[])
    ancestors = set()
    for eff in soles:
        body = eff['origin_body']
        while body:
            ancestors.add(int(model.body_weldid[body]))
            body = int(model.body_parentid[body])
    outgoing_active = np.zeros(frames, bool)
    outgoing_support = np.zeros(frames, bool)
    incoming = np.zeros((frames, len(soles)), bool)
    gaps = np.full(incoming.shape, np.inf)
    policy = objective.tasks.contact_phase
    for surf in objective.surfaces:
        task = surf['task']
        active = np.asarray(task.active, bool)
        static = np.asarray(task.surface_ids) == 0
        normals = np.asarray(surf['normals'])
        upward = normals[..., 2] >= np.cos(np.pi / 4)
        roles = getattr(policy, 'surface_roles', {}) if getattr(policy, 'surface', None) is task else {}
        support = np.asarray(roles.get('support', np.zeros_like(active)))
        proximal = np.isin(model.body_weldid[surf['bodies']], list(ancestors))
        present = active & static & upward & proximal[None]
        outgoing_active |= present.any(1)
        outgoing_support |= (present & support).any(1)
        points = kin.geom_points(surf['geoms'], surf['local'])
        normal_error = np.abs(np.sum((points - surf['targets']) * normals, axis=-1))
        for column, eff in enumerate(soles):
            on_sole = model.body_weldid[surf['bodies']] == model.body_weldid[eff['body']]
            mask = active & static & upward & on_sole[None]
            count = mask.sum(1)
            incoming[:, column] |= count > 0
            value = np.sum(np.where(mask, normal_error, 0.), axis=1) / np.maximum(count, 1)
            gaps[count > 0, column] = value[count > 0]
    for column, eff in enumerate(soles):
        task = objective.tasks.effectors[eff['name']]['task']
        active = task.contact_mask & ~task.managed_frames & (task.surface_normals[:, 2] >= np.cos(np.pi / 4))
        points = kin.points(np.full((frames, 1), eff['body']), eff['offset'][:, None])[:, 0]
        gap = np.abs(np.sum((points - task.contact_positions) * task.surface_normals, axis=1))
        gaps[active, column] = gap[active]
        incoming[:, column] |= active
    if policy is not None:
        required = np.stack([np.asarray(policy.roles[eff['name']]) == 'support'
                             for eff in soles], axis=1)
        explained = required.any(1)
        incoming[explained] &= required[explained]
    outgoing = supported_contact_runs(outgoing_active, outgoing_support)
    events = handoff_gaps(outgoing, incoming, gaps)
    return dict(status='geometry_only', physical_support_verified=False,
        sole_names=[e['name'] for e in soles], events=events,
        failed_handoffs=sum(not e['acquired_before_release'] for e in events),
        unresolved_reason='joint/root/clearance-constrained acquisition gap; no automatic re-solve')
