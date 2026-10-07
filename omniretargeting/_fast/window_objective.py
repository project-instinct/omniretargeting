"""Restrict the original objective to a window, including boundary stencils.

Only per-frame rows inside the variable window contribute. Every temporal
row whose original stencil intersects it contributes, with the current
fixed neighbours in its residual and only window columns in its Jacobian.
"""

from collections import OrderedDict
import copy
import os

import numpy as np

from .frame_kinematics import frame_range
from .rows import _Accumulator


def prepare_window_evaluation(objective):
    """Warm optional native libraries and configure BLAS in the parent.

    Call before R2 forks window children. No CUDA import or worker is needed.
    """
    from threadpoolctl import threadpool_limits
    from .batched_self_collision import NativeUnavailable, native_library
    from .fast_self_collision import acquisition_self_collision
    from .window_native import library
    threadpool_limits(limits=1)
    for build in (library, native_library):
        try:
            build()
        except NativeUnavailable:
            pass
    from .bulk_geometry import native_library as bulk_library
    try:
        bulk_library()
    except NativeUnavailable:
        pass
    from .window_query_native import prepare
    try:
        prepare(objective)
    except NativeUnavailable:
        pass
    if objective.self_collision is not None and objective.evaluation_backend != 'legacy':
        try:
            objective._window_collision_template = acquisition_self_collision(objective.self_collision, workers=1, window=True)
        except NativeUnavailable:
            objective._window_collision_template = None
        objective.__dict__.pop('_window_self_cache', None)
    return objective


def window_dependencies(objective, frames):
    frames = frame_range(frames, objective.frames)
    return range(max(0, frames.start - 3), min(objective.frames, frames.stop + 3))


def _span(frames, count, order):
    return max(0, frames.start - order), min(count - order, frames.stop)


def temporal_row_indices(objective, frames):
    """Original global temporal-row IDs, including rows with zero weights."""
    indices, offset = [], 0

    def add(ops, size):
        nonlocal offset
        for order, op in enumerate(ops, 1):
            lo, hi = _span(frames, objective.frames, order)
            indices.append(np.arange(offset + lo * size, offset + hi * size))
            offset += op.shape[0] * size

    add(objective.joint_ops, len(objective.coords.qi))
    for _ in objective.world_paths:
        add(objective.path_ops, 3)
    if objective.coords.root_q is not None:
        paths = 1 + int(objective.continuity_root is not None or objective.temporal_reference is not None)
        for _ in range(paths):
            add(objective.xy_ops, 2)
            add(objective.height_ops, 1)
            add(objective.angular_ops, 4)
    return np.concatenate(indices)


def _views(objective, frames):
    key = frames.start, frames.stop
    cache = getattr(objective, '_window_views', None)
    if cache is None:
        objective._window_views = cache = OrderedDict()
    if key not in cache:
        context = window_dependencies(objective, frames)
        local = objective.for_frames(slice(frames.start, frames.stop))
        from .window_geometry import WindowGeometry
        local._window_geometry = WindowGeometry(local)
        temporal = copy.copy(objective)
        temporal.frames = len(context)
        temporal._temporal_cache = {}
        temporal.__dict__.pop('_temporal_signature', None)
        temporal.__dict__.pop('_root_temporal_operators', None)
        for name in ('joint_ops', 'path_ops', 'xy_ops', 'height_ops', 'angular_ops'):
            if hasattr(objective, name):
                ops = []
                for order, op in enumerate(getattr(objective, name), 1):
                    lo, hi = _span(frames, objective.frames, order)
                    ops.append(op[lo:hi, context.start:context.stop])
                setattr(temporal, name, ops)
        from scipy import sparse
        temporal.joint_matrices = [sparse.kron(op, objective.coords.selector, format='csr')
                                   for op in temporal.joint_ops]
        cache[key] = local, temporal, context
    cache.move_to_end(key)
    # Small bounded views; immutable full packs remain owned by the parent.
    while len(cache) > 8:
        cache.popitem(last=False)
    local, temporal, context = cache[key]
    # Refresh every mutable scalar/pack from the parent. Static views share
    # arrays; replacement of a pack is also reflected, without stale weights.
    collision = getattr(local, '_self_batch', None)
    geometry = local._window_geometry
    refreshed = objective.for_frames(slice(frames.start, frames.stop), kinematics=local.kin)
    local.__dict__.update(refreshed.__dict__)
    local._window_geometry = geometry
    local._proximity_points = geometry.proximity_points
    local._deepest_samples = geometry.deepest
    if collision is not None:
        local._self_batch = collision
    span = slice(context.start, context.stop)
    for name in ('cfg', 'correction_temporal_scale', 'angular_cols'):
        if hasattr(objective, name):
            setattr(temporal, name, getattr(objective, name))
    for name in ('reference', 'temporal_reference', 'continuity_joints', 'root_reference',
                 'root_path_rotations'):
        value = getattr(objective, name, None)
        setattr(temporal, name, None if value is None else value[span])
    temporal.tasks = copy.copy(objective.tasks)
    temporal.tasks.roots = np.asarray(objective.tasks.roots)[span]
    temporal.continuity_root = (None if objective.continuity_root is None else
                               tuple(value[span] for value in objective.continuity_root))
    for name in ('temporal_weight', 'joint_temporal_weight', 'root_translation_weight',
                 'root_rotation_weight'):
        weights = []
        for order, weight in enumerate(getattr(objective, name), 1):
            lo, hi = _span(frames, objective.frames, order)
            weights.append(weight[lo:hi])
        setattr(temporal, name, weights)
    temporal.world_paths = [(index, target[span], [mask[slice(*_span(frames, objective.frames, order))]
                            for order, mask in enumerate(masks, 1)])
                            for index, target, masks in objective.world_paths]
    return local, temporal, context


def evaluate_window(objective, poses, with_jacobian, frames):
    frames = frame_range(frames, objective.frames)
    if len(poses) != objective.frames:
        raise ValueError('window evaluation requires the current full poses')
    local, temporal, context = _views(objective, frames)
    local._window_cpu_budget = getattr(objective, '_window_cpu_budget', None)
    # FrameKinematicsCache caches individual frames, but assembles new arrays
    # on every call. Keep the last exact window assemblies as well: collision
    # witnesses are keyed by their immutable geom-array identities. Geometry
    # does not depend on whether body Jacobians were requested.
    assemblies = getattr(local, '_window_assemblies', None)
    if assemblies is None:
        local._window_assemblies = assemblies = OrderedDict()
    context_poses = np.asarray(poses)[context.start:context.stop]
    pose_key = (objective.coords.cols.tobytes(), context_poses.shape,
                context_poses.dtype.str, context_poses.tobytes())
    previous = assemblies.get(pose_key)
    if previous is not None and (not with_jacobian or previous.with_jacobian):
        kin = previous
    else:
        kin = objective.window_kinematics(poses, context, with_jacobian)
        if previous is not None:
            kin._window_slices = getattr(previous, '_window_slices', {}).copy()
            for name in ('xpos', 'xmat', 'geom_xpos', 'geom_xmat'):
                old, new = getattr(previous, name), getattr(kin, name)
                if old.tobytes() == new.tobytes():
                    setattr(kin, name, old)
        assemblies[pose_key] = kin
    assemblies.move_to_end(pose_key)
    while len(assemblies) > 4:
        assemblies.popitem(last=False)
    relative = slice(frames.start - context.start, frames.stop - context.start)
    local.kin = copy.copy(kin)
    local.kin.cache = OrderedDict()
    names = ['xpos', 'xmat', 'geom_xpos', 'geom_xmat'] + (['jacp', 'jacr'] if with_jacobian else [])
    # Retain the slice objects too, including across value -> Jacobian upgrades.
    slices = getattr(kin, '_window_slices', None)
    if slices is None:
        kin._window_slices = slices = {}
    for name in names:
        array = getattr(kin, name)
        cached = slices.get(name)
        if cached is None or cached[0] is not array:
            slices[name] = array, array[relative]
        setattr(local.kin, name, slices[name][1])
    part = np.asarray(poses)[frames.start:frames.stop]
    key = (local.kin.columns.tobytes(), part.shape, part.dtype.str, part.tobytes())
    local.kin.cache[key] = {name: getattr(local.kin, name) for name in names}
    from .window_self_collision import WindowSelfCollision
    provider = getattr(objective, '_window_self_cache', None)
    if provider is not None and provider.owner_pid != os.getpid():
        # Also support ordinary forked callers outside WindowWorkers. Replace
        # inherited mutable query buffers without preparation or CUDA work.
        objective._window_self_cache = provider = provider.fork_private_client()
    if (provider is not None
            and local.self_collision is not None and local.evaluation_backend != 'legacy'):
        collision = getattr(local, '_self_batch', None)
        if not isinstance(collision, WindowSelfCollision) or collision.provider is not provider:
            local._self_batch = WindowSelfCollision(provider, local.self_collision)
        local._self_batch.bind(poses, frames)
    elif isinstance(getattr(local, '_self_batch', None), WindowSelfCollision):
        local.__dict__.pop('_self_batch')
    if (local.self_collision is not None
            and not hasattr(local, '_self_batch') and local.evaluation_backend != 'legacy'):
        from .batched_self_collision import NativeUnavailable
        template = getattr(objective, '_window_collision_template', None)
        if template is not None:
            local._self_batch = copy.copy(template)
            local._self_batch.cache = OrderedDict()
            local._self_batch.diagnostics = {}
        else:
            try:
                from .fast_self_collision import acquisition_self_collision
                local._self_batch = acquisition_self_collision(local.self_collision, workers=1, window=True)
            except NativeUnavailable:
                local._self_batch = None
    collision = getattr(local, '_self_batch', None)
    if collision is not None:
        from .window_cpu_budget import native_workers
        collision.cpu_budget = getattr(objective, '_window_cpu_budget', None)
        collision.workers = native_workers() if collision.cpu_budget is not None else 1
    service = getattr(objective, '_window_eval_client', None)
    weight = objective.envelope_weight[frames.start:frames.stop]
    measured = (local.evaluate_frames(part, with_jacobian, weight) if service is None else
                service.evaluate(part, frames, with_jacobian, weight))
    # Boundary FK only: no per-frame residual family runs on fixed neighbours.
    temporal.kin = kin
    temporal.effectors = [{**eff, 'offset': eff['offset'][context.start:context.stop]}
                          for eff in objective.effectors]
    root_point = kin.xpos[:, objective.tasks.root_body].copy() if objective.tasks.root_body is not None else temporal.tasks.roots
    root_jac = temporal._root_jacobian() if with_jacobian else None
    paths = temporal._effector_points(with_jacobian)
    residual, matrix = temporal._temporal_terms(np.asarray(poses)[context.start:context.stop],
                                               with_jacobian, root_point, root_jac, paths)
    if with_jacobian:
        matrix = matrix[:, relative.start * objective.width:relative.stop * objective.width]
    acc = _Accumulator(len(frames), objective.width, with_jacobian)
    acc.frame_cost = measured['frame_cost']
    acc.cost = float(acc.frame_cost.sum()) + float(residual @ residual)
    if with_jacobian:
        acc.hessian, acc.gradient = measured['hessian'], measured['gradient']
    stats = measured['stats']
    objective.stats = stats
    return acc, residual, matrix, stats, None
