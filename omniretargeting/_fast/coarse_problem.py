"""Half-rate views of prepared acquisition inputs, never a new source fit.

The objective uses prepared even-frame inputs and half-rate operators.
Hard certificates use unchanged full-rate quantities on the same linear
correction interpolant used for transfer, including omitted odd frames.
"""

from collections import OrderedDict
from copy import copy
from dataclasses import fields, replace
from math import comb

import mujoco
import numpy as np
from scipy import sparse

from .._sequence.continuity_guard import SequenceContinuityGuard
from .._sequence.coordinates import temporal_operators
from ..motion_budgets import windowed_budget
from .acquisition_guard import AcquisitionNormalGuard
from .continuity import CorrectionContinuity, derivatives
from .kinematics import TrajectoryKinematics


def aggregate_bounds(bounds, order, count):
    """Triangle bound for D_2^k = (1 + shift)^k D_1^k at even frames."""
    bounds = np.asarray(bounds)
    starts = 2 * np.arange(max(0, count - order))
    value = np.zeros((len(starts), *bounds.shape[1:]), dtype=bounds.dtype)
    for offset in range(order + 1):
        value += comb(order, offset) * bounds[starts + offset]
    value.setflags(write=False)
    return value


def _clear_caches(value):
    for name in list(value.__dict__):
        if name.startswith(('_window_', '_gpu_', '_self_', '_temporal_', '_root_temporal_')):
            value.__dict__.pop(name)


class _Sampler:
    def __init__(self, objective, indices):
        self.objective, self.indices = objective, indices
        self.tasks, self.surfaces, self.policies = {}, {}, {}

    def take(self, value, *, frozen=False):
        if value is None:
            return None
        # scipy Rotation has its own indexing and no mutable ndarray view.
        from scipy.spatial.transform import Rotation
        if isinstance(value, Rotation):
            return value[self.indices]
        result = np.array(value)[self.indices].copy()
        if frozen:
            result.setflags(write=False)
        return result

    def surface(self, task):
        if id(task) not in self.surfaces:
            result = self.surfaces[id(task)] = copy(task)
            for name in ('environment_points', 'source_vectors', 'active', 'normals', 'confidence',
                         'modes', 'surface_ids', 'targets', 'pose_normal_targets', 'pose_normal_valid',
                         'weights'):
                setattr(result, name, self.take(getattr(task, name), frozen=True))
            result.cfg = dict(task.cfg)
            if hasattr(task, 'source_clearances'):
                result.source_clearances = {name: self.take(value, frozen=True)
                                            for name, value in task.source_clearances.items()}
            # SurfaceKinematics buffers are not used by the packed objective,
            # but its task view must not own the full task's mutable FK state.
            from ..surface_tasks import SurfaceKinematics
            result.kinematics = SurfaceKinematics(task.model, task.bindings)
            if getattr(task, 'contact_phase', None) is not None:
                result.contact_phase = self.policy(task.contact_phase)
        return self.surfaces[id(task)]

    def policy(self, policy):
        if policy is None:
            return None
        if id(policy) not in self.policies:
            result = self.policies[id(policy)] = copy(policy)
            result.roles = {name: self.take(value, frozen=True) for name, value in policy.roles.items()}
            result.clear = {name: self.take(value, frozen=True) for name, value in policy.clear.items()}
            result.surface_roles = {name: self.take(value, frozen=True)
                                    for name, value in policy.surface_roles.items()}
            result.support = result.surface_roles['support']
            result.follow_surface = result.surface_roles['follow']
            result.surface = None if policy.surface is None else self.surface(policy.surface)
        return self.policies[id(policy)]

    def contact(self, task):
        count = self.objective.frames
        values = {field.name: self.take(value, frozen=True) for field in fields(task)
                  if isinstance(value := getattr(task, field.name), np.ndarray)
                  and value.ndim and len(value) == count}
        values['approach_reference'] = tuple(self.take(value, frozen=True) for value in task.approach)
        values['contact_phase'] = self.policy(task.contact_phase)
        return replace(task, **values)

    def task_set(self, tasks):
        if id(tasks) not in self.tasks:
            result = self.tasks[id(tasks)] = copy(tasks)
            result.frames = len(self.indices)
            result.config = dict(tasks.config)
            result.data = mujoco.MjData(tasks.model)
            result.human = self.take(tasks.human, frozen=True)
            result.roots = self.take(tasks.roots, frozen=True)
            result.operators = [tasks.operators[i].copy() for i in self.indices]
            result.orientations = copy(tasks.orientations)
            result.orientations.targets = [(body, parent, self.take(target, frozen=True), sigma)
                                           for body, parent, target, sigma in tasks.orientations.targets]
            result.effectors = {name: dict(spec, task=self.contact(spec['task']))
                               for name, spec in tasks.effectors.items()}
            for spec in result.effectors.values():
                if 'support_witness' in spec:
                    spec['support_witness'] = {name: self.take(value, frozen=True)
                                               for name, value in spec['support_witness'].items()}
            result.choices = {name: self.take(value, frozen=True) for name, value in tasks.choices.items()}
            result.extra_tasks = tuple(self.surface(task) for task in tasks.extra_tasks)
            result.contact_phase = self.policy(tasks.contact_phase)
            result.scene_cache = None
            query = getattr(tasks, 'self_query', None)
            result.self_query = None if query is None else query.fresh_state()
        return self.tasks[id(tasks)]

    def shape(self, shape, kin):
        if shape is None:
            return None
        result = copy(shape)
        _clear_caches(result)
        for name in ('seed', 'rotations', 'source', 'seed_source_angles', 'seed_source_errors',
                     'acquired_poses'):
            setattr(result, name, self.take(getattr(shape, name), frozen=True))
        result.tasks = self.task_set(shape.tasks)
        result.kin = TrajectoryKinematics(kin.model, kin.columns, cache=kin.cache)
        result.envelopes = [envelope.for_frames(self.indices) for envelope in shape.envelopes]
        result.pose = copy(shape.pose)
        result.pose.tasks = result.tasks
        result.pose.data = mujoco.MjData(kin.model)
        for name in ('reference', 'rotations', 'direction_targets', 'direction_valid'):
            setattr(result.pose, name, self.take(getattr(shape.pose, name), frozen=True))
        result.rejections = 0
        return result

    def guard(self, guard, objective):
        result = copy(guard)
        _clear_caches(result)
        result.rejections = 0
        if isinstance(guard, AcquisitionNormalGuard):
            result.objective, result.priority = objective, None
            result.kin = TrajectoryKinematics(objective.model, objective.coords.cols, cache=objective.kin.cache)
            result.penetration = self.take(guard.penetration, frozen=True)
            result.names, result.groups = [], []
            for name, (kind, indices, local, *arrays) in zip(guard.names, guard.groups, strict=True):
                sampled = tuple(self.take(value, frozen=True) for value in arrays)
                # A protected touch can occur only on omitted odd frames.
                # The full-rate guard retains it; coarse diagnostics require
                # every retained group to contain an active point.
                if not sampled[2].any():
                    continue
                result.names.append(name)
                result.groups.append((kind, indices,
                    local if kind == 'surface' else self.take(local, frozen=True),
                    *sampled))
        elif isinstance(guard, SequenceContinuityGuard):
            result.tasks = self.task_set(guard.tasks)
            result.rotations = self.take(guard.rotations)
            result.fixed_rotations = self.take(guard.fixed_rotations)
            result.reference = {name: (self.take(value, frozen=True), self.take(jac, frozen=True))
                                for name, (value, jac) in guard.reference.items()}
            result.bounds = {(name, order): aggregate_bounds(bound, order, objective.frames)
                             for (name, order), bound in guard.bounds.items()}
            result._signals_cache, result._joint_matrices = OrderedDict(), {}
            if hasattr(result, '_signal_kin'):
                result._signal_kin = TrajectoryKinematics(objective.model, objective.coords.cols,
                                                         cache=objective.kin.cache)
        elif isinstance(guard, CorrectionContinuity):
            result.fixed = self.take(guard.fixed, frozen=True)
            result._operators, result._linear_cache = {}, None
            angular = derivatives(objective.coords, result.fixed)
            result.bounds = {}
            for name, bound in guard.bounds.items():
                order = int(name[-1])
                if objective.frames <= order:
                    continue
                # World rotation logs of corrections are linear signals in
                # time. Executed angular increments are noncommutative, so
                # use a coarse reference screen only for that signal family.
                if name.startswith('root_rotation_') and not guard.correction:
                    value = windowed_budget(angular[name])
                    value.setflags(write=False)
                    result.bounds[name] = value
                else:
                    result.bounds[name] = aggregate_bounds(bound, order, objective.frames)
            result.norm_bounds = {name: aggregate_bounds(bound, int(name[-1]), objective.frames)
                                  for name, bound in guard.norm_bounds.items()}
        else:
            raise ValueError(f'Unsupported coarse guard: {type(guard).__name__}')
        return result


def _temporal_weights(values, order, count):
    """Retain the stiffest weight on each contributing original stencil."""
    return np.stack([np.max(values[2 * i:2 * i + order + 1], axis=0)
                     for i in range(count - order)])


def make_coarse_problem(objective):
    """Own sampled mutable state; share only immutable model/scene geometry."""
    for name in ('priority_clearance', 'original_clearance', 'frozen_original_depth'):
        if getattr(objective, name, None) is not None:
            raise ValueError(f'Coarse acquisition does not adapt {name}')
    indices = np.arange(0, objective.frames, 2)
    if len(indices) < 4:
        raise ValueError('Coarse acquisition needs at least four sampled frames')
    sampler = _Sampler(objective, indices)
    result = copy(objective)
    _clear_caches(result)
    result.pool, result.worker_output = None, None
    result.__dict__.pop('_pool_pid', None)
    result.__dict__.pop('frame_slice', None)
    result.frames, result.framerate = len(indices), objective.framerate / 2
    result.cfg = dict(objective.cfg)
    result.evaluation_backend, result.stats = 'cpu', {}
    result.kin = TrajectoryKinematics(objective.model, objective.coords.cols)
    result.tasks = sampler.task_set(objective.tasks)
    for name in ('interaction', 'reference', 'temporal_reference', 'reference_rotations',
                 'root_reference', 'root_rotations', 'root_path_rotations', 'plane_normal', 'plane_offset',
                 'plane_field', 'plane_centre', 'plane_radius', 'envelope_weight', 'fixed_root',
                 'continuity_joints'):
        if hasattr(objective, name):
            setattr(result, name, sampler.take(getattr(objective, name)))
    for name in ('lower', 'upper'):
        bound = getattr(objective, name)
        setattr(result, name, sampler.take(bound) if np.ndim(bound) == 2 else bound.copy())
    if objective.continuity_root is not None:
        result.continuity_root = tuple(sampler.take(value) for value in objective.continuity_root)
    result.effectors = [{key: sampler.take(value) if key in objective.PER_FRAME_EFFECTOR else value
                        for key, value in effector.items()} for effector in objective.effectors]
    for target, original in zip(result.effectors, objective.effectors):
        target['orientation'] = [(local, sampler.take(direction), sampler.take(valid))
                                 for local, direction, valid in original['orientation']]
    result.surface_tasks = [sampler.surface(task) for task in objective.surface_tasks]
    result.surfaces = [{key: sampler.take(value) if key in objective.PER_FRAME_SURFACE else value
                       for key, value in surface.items()} for surface in objective.surfaces]
    for surface in result.surfaces:
        surface['task'] = sampler.surface(surface['task'])
    result.proximities = [{key: sampler.take(value) if key in objective.PER_FRAME_PROXIMITY else value
                          for key, value in proximity.items()} for proximity in objective.proximities]
    result.objects = [dict(obj, translations=sampler.take(obj['translations']),
                           rotations=sampler.take(obj['rotations']), scales=sampler.take(obj.get('scales')))
                      for obj in objective.objects]
    result.relation_envelope = (None if objective.relation_envelope is None else
                               objective.relation_envelope.for_frames(indices))
    for name, sigmas in (('joint_ops', 'joint_derivative_sigmas'), ('path_ops', 'root_xy_derivative_sigmas'),
                         ('xy_ops', 'root_xy_derivative_sigmas'), ('height_ops', 'root_height_derivative_sigmas'),
                         ('angular_ops', 'root_quaternion_derivative_sigmas')):
        if hasattr(objective, name):
            setattr(result, name, temporal_operators(result.frames, result.framerate, result.cfg[sigmas]))
    result.joint_matrices = [sparse.kron(op, objective.coords.selector, format='csr')
                             for op in result.joint_ops]
    for name in ('temporal_weight', 'joint_temporal_weight', 'root_translation_weight', 'root_rotation_weight'):
        setattr(result, name, [_temporal_weights(value, order, result.frames)
                              for order, value in enumerate(getattr(objective, name), 1)])
    result.world_paths = []
    for effector, path, masks in objective.world_paths:
        coarse_masks = [np.array([mask[a:b - order + 1].all()
                                 for a, b in zip(indices[:-order], indices[order:])])
                        for order, mask in enumerate(masks, 1)]
        result.world_paths.append((effector, sampler.take(path), coarse_masks))
    result.self_collision = (None if objective.self_collision is None else
                             objective.self_collision.fresh_state())
    sampled_shape = sampler.shape(objective.source_shape, result.kin)
    sampled_hinges = [sampler.guard(guard, result) for guard in getattr(objective, 'hinge_guards', ())]
    from .coarse_guards import install_full_rate_guards
    install_full_rate_guards(result, objective, sampled_shape, sampled_hinges)
    return result, indices
