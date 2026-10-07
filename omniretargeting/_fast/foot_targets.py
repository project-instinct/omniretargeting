"""Morphology-aware sole tracking and continuous world-contact handovers.

These are solver targets, never edits to the source contact evidence.  Reach
certificates use a conservative outer bound, so an IK failure cannot unload a
support. Targets inside that bound remain contacts, even if difficult to solve.
"""
import numpy as np

from ..contact_schedule import _iter_true_runs

HANDOVER_SECONDS = 0.15


def anatomical_mapping(model, mapping, priority):
    """Use the proximal articulation centre for each measured sole chain.

    A serial hip's intermediate roll origin is displaced by its pitch angle;
    it is not the origin of the anatomical thigh. Walk to the first joint below
    its mapped parent instead of encoding a particular robot joint name.
    """
    result = dict(mapping)
    mapped = {model.body(body).id: name for name, body in mapping.items()}
    for spec in priority.get('effectors', {}).values():
        if not morphology_sole(spec):
            continue
        body = model.body(spec['robot_body']).id
        chain = []
        while body:
            if body in mapped:
                chain.append(body)
            body = int(model.body_parentid[body])
        if len(chain) < 4:
            continue
        hip, parent = chain[2:4]
        proximal = hip
        while int(model.body_parentid[proximal]) != parent:
            proximal = int(model.body_parentid[proximal])
        result[mapped[hip]] = model.body(proximal).name
    return result


def pack_sole_morphology(objective):
    """Reconstruct the free leg using segment lengths about its robot hip.

    Each source segment keeps its world direction; robot lengths replace
    source lengths. A single torso-centred scale incorrectly scales hip width
    and torso-to-pelvis height along with the leg, leaving incompatible IK goals.
    """
    import mujoco
    model, tasks = objective.model, objective.tasks
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mapped = {int(body): index for index, body in enumerate(tasks.bodyids)}
    for eff in objective.effectors:
        spec = tasks.effectors[eff['name']]
        if not morphology_sole(spec):
            continue
        body, chain = eff['body'], []
        while body:
            if body in mapped:
                chain.append(body)
            body = int(model.body_parentid[body])
        if len(chain) < 3:
            continue
        ankle, knee, hip = chain[:3]
        human = np.asarray(tasks.human)
        relative = eff['scale'] * (eff['tracking'] - human[:, mapped[ankle]])
        for first, last in ((hip, knee), (knee, ankle)):
            segment = human[:, mapped[last]] - human[:, mapped[first]]
            length = np.linalg.norm(data.xpos[last] - data.xpos[first])
            relative += segment * (length / np.maximum(np.linalg.norm(segment, axis=1), 1e-12))[:, None]
        eff['origin_body'] = hip
        eff['relative_target'] = relative


def morphology_sole(spec):
    """Sole frames with measured morphology; do not guess a missing scale."""
    return bool(spec.get('track_orientation', False) and spec.get('_scale_calibrated', False))


def smoothstep(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def handover(mask, tracking, anchors, radius):
    """Keep anchors on contact frames; extend their displacement into free time.

    Endpoint continuity comes from the cubic smoothstep. Short
    free gaps interpolate the two boundary corrections directly, avoiding a
    competing max-of-ramps switch in the middle of a swing.
    """
    mask = np.asarray(mask, bool)
    tracking, anchors = np.asarray(tracking), np.asarray(anchors)
    share = mask.astype(float)
    target = tracking.copy()
    target[mask] = anchors[mask]
    radius = max(1, int(radius))
    for start, stop in _iter_true_runs(~mask):
        before, after = start > 0, stop < len(mask)
        ids = np.arange(start, stop)
        if before and after and stop - start < 2 * radius:
            alpha = smoothstep((ids - start + 1) / (stop - start + 1))
            first = anchors[start - 1] - tracking[start - 1]
            last = anchors[stop] - tracking[stop]
            target[ids] += (1 - alpha[:, None]) * first + alpha[:, None] * last
            share[ids] = 1.
            continue
        if before:
            window = ids[ids < start + radius]
            share[window] = smoothstep(1 - (window - start + 1) / (radius + 1))
            target[window] += anchors[start - 1] - tracking[start - 1]
        if after:
            window = ids[ids >= stop - radius]
            share[window] = smoothstep(1 - (stop - window) / (radius + 1))
            target[window] += anchors[stop] - tracking[stop]
    return share, target


def role_envelope(weights, active, radius):
    """Raise priority before and retain it after a supported contact interval.

    This smooths solver stiffness only; source support/hold labels stay intact.
    The extension stops at any loss of geometric contact.
    """
    result = np.asarray(weights, float).copy()
    for start, stop in _iter_true_runs(np.asarray(active, bool) & (result > 0)):
        original = result[start:stop].copy()
        for shift in range(1, radius + 1):
            strength = smoothstep(1 - shift / (radius + 1))
            if shift < len(original):
                result[start + shift:stop] = np.maximum(
                    result[start + shift:stop], 1 + strength * (original[:-shift] - 1))
                result[start:stop - shift] = np.maximum(
                    result[start:stop - shift], 1 + strength * (original[shift:] - 1))
    return result


def root_reach_radius(model, body, offset):
    """Outer sphere valid for every joint angle, including all legal angles.

    Sum rigid translations to the free body; slides add their full bounded
    excursion. Off-origin hinge pivots add a conservative diameter. Unsupported
    unbounded translations return infinity, never an unreachable certificate.
    """
    import mujoco
    radius = float(np.linalg.norm(offset))
    while body:
        joints = range(int(model.body_jntadr[body]),
                       int(model.body_jntadr[body] + model.body_jntnum[body]))
        if any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in joints):
            return body, radius
        radius += float(np.linalg.norm(model.body_pos[body]))
        for joint in joints:
            if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE:
                if not model.jnt_limited[joint]:
                    return body, np.inf
                radius += float(np.max(np.abs(model.jnt_range[joint])))
            else:
                radius += 2 * float(np.linalg.norm(model.jnt_pos[joint]))
        body = int(model.body_parentid[body])
    return 0, np.inf


def prepare_sole_handover(objective):
    """Unify endpoint/dense-contact approach targets at a fixed reference pose."""
    kin = objective.kin
    kin.evaluate(objective.reference, with_jacobian=False)
    radius = max(2, int(np.ceil(objective.framerate * HANDOVER_SECONDS)))
    report = {}
    for eff in objective.effectors:
        spec = objective.tasks.effectors[eff['name']]
        if not morphology_sole(spec):
            continue
        observed = eff['active'] | eff['managed']
        anchors = eff['contact'].copy()
        model = objective.model
        offsets = np.vstack([spec['tracking_offset'], spec['contact_offsets']])
        root, reach = root_reach_radius(model, eff['body'], offsets[np.argmax(np.linalg.norm(offsets, axis=1))])
        for surf in objective.surfaces:
            selected = model.body_weldid[surf['bodies']] == model.body_weldid[eff['body']]
            for column in np.flatnonzero(selected):
                geom = surf['geoms'][column]
                # Triangle inequality is conservative even for arbitrary geom
                # rotations and local sample orientations.
                offset_radius = np.linalg.norm(model.geom_pos[geom]) + np.linalg.norm(surf['local'][column])
                _, bound = root_reach_radius(model, int(surf['bodies'][column]), [offset_radius, 0, 0])
                reach = max(reach, bound)
        budget = objective.root_budget
        # The root sphere is enforced by FastObjective.project and the conic
        # bounds. A target plane outside it is unreachable even if all angular
        # limits were removed, hence also within the actual joint limits.
        distance = np.abs(np.sum((anchors - kin.xpos[:, root]) * eff['normal'], axis=1))
        for surf in objective.surfaces:
            selected = model.body_weldid[surf['bodies']] == model.body_weldid[eff['body']]
            mask = surf['active'] & selected[None]
            gaps = np.abs(np.sum((surf['targets'] - kin.xpos[:, root, None]) * surf['normals'], axis=-1))
            nearest = np.min(np.where(mask, gaps, np.inf), axis=1)
            distance = np.minimum(distance, nearest)
        excess = np.maximum(0., distance - reach - (np.inf if budget is None else budget))
        unreachable = observed & (excess > .001)
        eff['active'] = eff['active'] & ~unreachable
        eff['managed'] = eff['managed'] & ~unreachable
        # Dense material points define a rigid foot placement. Transfer their
        # mean displacement to the tracking point using the fixed orientation.
        tracked = kin.points(np.full((objective.frames, 1), eff['body']),
                             spec['contact_offsets'][objective.tasks.choices[eff['name']]][:, None])[:, 0]
        for surf in objective.surfaces:
            selected = objective.model.body_weldid[surf['bodies']] == objective.model.body_weldid[eff['body']]
            mask = surf['active'] & selected[None]
            count = mask.sum(1)
            points = kin.geom_points(surf['geoms'], surf['local'])
            shift = np.sum(np.where(mask[..., None], surf['targets'] - points, 0.), axis=1)
            use = eff['managed'] & (count > 0)
            anchors[use] = tracked[use] + shift[use] / count[use, None]
            if unreachable.any():
                # Packed solver data is private; keep task/evidence arrays
                # untouched for the independent contact audit and gap report.
                keep = ~(unreachable[:, None] & selected[None])
                surf['active'] = surf['active'] & keep
                for key in ('scale', 'priority', 'normal_scale'):
                    surf[key] = surf[key] * keep
            # A sliding material sample has a normal constraint; its tangent
            # motion belongs to source morphology, never a frozen skin anchor.
            surf.setdefault('normal_only', np.zeros_like(surf['active']))
            surf['normal_only'] |= selected[None] & ~surf['sticking']
            for column in np.flatnonzero(selected):
                surf['priority'][:, column] = np.sqrt(role_envelope(
                    surf['priority'][:, column] ** 2, surf['active'][:, column], radius))
        active = eff['active'] | eff['managed']
        eff['share'], eff['approach'] = handover(active, eff['tracking'], anchors, radius)
        contact_offset = spec['contact_offsets'][objective.tasks.choices[eff['name']]]
        eff['offset'] = np.where(active[:, None], eff['offset'],
            (1 - eff['share'][:, None]) * spec['tracking_offset'] + eff['share'][:, None] * contact_offset)
        eff['role'] = np.sqrt(role_envelope(eff['role'] ** 2, active, radius))
        report[eff['name']] = dict(handover_frames=radius, root_reach_radius_m=reach,
            unreachable_frames=np.flatnonzero(unreachable).tolist(),
            certified_gap_max_m=float(excess[observed].max(initial=0.)),
            root_bound_m=budget, reach_status='outer_sphere_certificate',
            uncertified_ik_failure_does_not_release_contact=True, source_targets_unchanged=True)
    return report
