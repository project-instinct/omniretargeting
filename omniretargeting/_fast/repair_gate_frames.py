"""Private frame and window views of the repair's immutable budgets.

Window views retain interior scalar-joint temporal stencils. Crossing boundary
and support rows are omitted; the full attempt checks every row before commit.
"""
from copy import copy
from types import SimpleNamespace

import numpy as np

from .kinematics import TrajectoryKinematics


def local_limits(limits, frame, stop=None, *, temporal=False):
    """Slice original cones/layout/contact budgets and optional joint rows."""
    from .acquisition_guard import AcquisitionNormalGuard
    from .contact_guard import ContactGuard
    from .reference import ReferenceShape
    from .continuity import CorrectionContinuity
    from .guard_signals import CachedSequenceContinuityGuard
    from .._sequence.continuity_guard import SequenceContinuityGuard
    from .repair_contact_rows import _slice

    original = limits.objective
    part = slice(frame, frame+1 if stop is None else stop)
    objective = copy(original)
    objective.frames = part.stop-part.start
    objective.tasks = copy(original.tasks)
    objective.tasks.roots = original.tasks.roots[part]
    objective.effectors = []
    for eff in original.effectors:
        local = dict(eff)
        for name in ('offset', 'active', 'managed', 'world', 'tracking', 'approach', 'share', 'contact'):
            local[name] = eff[name][part]
        if (isinstance(eff['scale'], np.ndarray) and eff['scale'].ndim
                and eff['scale'].shape[0] == limits.frames):
            local['scale'] = eff['scale'][part]
        objective.effectors.append(local)
    objective.surfaces = [{**surf, 'active': surf['active'][part], 'targets': surf['targets'][part]}
                          for surf in original.surfaces]
    objective.reference = original.reference[part]
    objective.candidate_guards = []
    temporal_types = (CorrectionContinuity, SequenceContinuityGuard, CachedSequenceContinuityGuard)
    for guard in getattr(original, 'candidate_guards', ()):
        if type(guard) in temporal_types:
            if temporal:
                from .repair_gate_temporal import joint_guard
                objective.candidate_guards.append(joint_guard(guard, part, type(guard) is not CorrectionContinuity))
            continue
        if type(guard) not in (ContactGuard, AcquisitionNormalGuard):
            return None
        local = _slice(guard, part)
        local.objective = objective
        local.__dict__.pop('_repair_contact_cache', None)
        objective.candidate_guards.append(local)
    shape = getattr(original, 'source_shape', None)
    if shape is not None:
        if type(shape) is not ReferenceShape:
            return None
        local = copy(shape)
        local.__dict__.pop('_repair_layout_cache', None)
        local.envelopes = [e.for_frames(part) for e in shape.envelopes]
        local.kin = TrajectoryKinematics(original.model, shape.kin.columns)
        objective.source_shape = local
    result = copy(limits)
    result.objective, result.frames = objective, objective.frames
    result.parent_frame = frame
    result.kin = TrajectoryKinematics(original.model, limits.coords.di)
    for name in ('entry', 'entry_rotations', 'lower', 'upper', 'penetration_yield'):
        setattr(result, name, getattr(limits, name)[part])
    for name in ('anchors', 'original_anchors', 'targets', 'target_budgets', 'tracking_budgets'):
        setattr(result, name, [value[part] for value in getattr(limits, name)])
    if getattr(limits, 'ablate_contacts', False):
        result.contact_planes = tuple(tuple(value[part] for value in plane)
                                      for plane in limits.contact_planes)
    from .repair_gate_temporal import differences
    result.differences = differences(limits, part) if temporal else []
    result.support_continuity = SimpleNamespace(add=lambda *args: None)
    result.frame_mask = None
    return result
