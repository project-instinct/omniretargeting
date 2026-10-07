"""Immutable source-shape reference shared across fast acquisition/correction.

The orientation fit selects the robot's kinematic branch. Contact objectives
may translate the robot vertically for morphology, but cannot redefine its
reference pose or root path by starting another stage.
"""
from dataclasses import replace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from .._sequence.pose_guard import SequencePoseGuard
from ..conic_constraints import LocalNormBounds
from ..pose_relations import RelationEnvelope, layout_constraints, source_relation_angles
from .kinematics import TrajectoryKinematics


def kinematic_tracking_tasks(tasks, priority):
    """Source-shaped morphology registration, before interaction acquisition.

    Permanent world tracking semantics come from the adapter/profile. A
    temporary contact/unknown-clearance world regime must not lift the entire
    robot to reach an object while defining the initial kinematic reference.
    All world targets and body-relative free targets retain their source data.
    """
    result = {}
    for name, task in tasks.items():
        spec = priority.get("effectors", {}).get(name, {})
        world = spec.get("preserve_world_target", task.preserve_world_tracking)
        # A calibrated sole follows morphology while free, just like a hand.
        # World support is acquired only after the kinematic branch is fixed.
        if spec.get("track_orientation", False) and spec.get("_scale_calibrated", False):
            world = False
        frames = len(task.contact_mask)
        result[name] = replace(
            task, contact_mask=np.zeros(frames, bool), contact_modes=np.full(frames, "none"),
            surface_managed=False, contact_phase=None,
            swing_world_mask=np.full(frames, world, bool),
            approach_reference=(np.zeros(frames), task.tracking_positions.copy()),
        )
    return result


def register_height(objective, poses):
    """Fit robot height to all existing targets with angles/path held fixed.

    Without clearance rows, translation-only residuals are affine. Solve the
    small temporal least-squares system exactly, using the existing role
    weights and sigmas. This cannot privilege one named body or invent support
    evidence, and it does not rescale or translate any human/scene target.
    """
    coords = objective.coords
    if coords.root_d is None or coords.root_d + 2 not in coords.inverse:
        return np.array(poses, copy=True), np.zeros(len(poses))
    column = coords.inverse[coords.root_d + 2]
    acc, residual, jacobian, _, _ = objective.evaluate(poses)
    indices = np.arange(len(poses)) * coords.width + column
    temporal = jacobian[:, indices]
    matrix = sparse.diags(acc.hessian[:, column, column]) + temporal.T @ temporal
    gradient = acc.gradient[:, column] + temporal.T @ residual
    # A vanishing column contains no geometric information: retain the seed.
    matrix += sparse.eye(len(poses)) * 1e-9
    shift = spsolve(matrix.tocsc(), -gradient)
    if not np.isfinite(shift).all():
        raise ValueError("Robot height registration returned a nonfinite shift")
    result = np.array(poses, copy=True)
    result[:, coords.root_q + 2] += shift
    return result, shift


def height_tracking_tasks(tasks, priority):
    """Register height to observed sole planes with joint angles held fixed.

    Free soles retain the same morphology targets as the pose fit. Active
    source soles supply only a vertical registration measurement. A later
    source-supported seat lift prevents this measurement sinking the pelvis;
    it cannot bend a leg or select a different IK branch.
    """
    from .foot_targets import morphology_sole
    result = kinematic_tracking_tasks(tasks, priority)
    for name, task in tasks.items():
        if morphology_sole(priority.get('effectors', {}).get(name, {})):
            result[name] = replace(result[name], contact_mask=task.contact_mask.copy(),
                contact_modes=np.where(task.contact_mask, 'sliding', 'none'),
                surface_managed_mask=None)
    return result


def bound_sole_registration(objective, poses):
    """Do not register a rigid sole through an observed upward support plane.

    Height registration uses a semantic point, which need not be the lowest
    collision point when the source tilts its ankle. Bound that affine fit by
    the collision sole's extent, spreading only the needed height correction.
    """
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    from ..scene_geometry import collision_geom_ids, geom_surface_offsets
    from .foot_targets import morphology_sole
    from .penetration import LIFT_RAMP_SECONDS, UPWARD_COS
    q = objective.coords.root_q
    if q is None:
        return poses, np.zeros(len(poses))
    model, kin = objective.model, objective.kin
    kin.evaluate(poses, with_jacobian=False)
    lower = np.zeros(len(poses))
    geoms = collision_geom_ids(model)
    for spec in objective.tasks.effectors.values():
        if not morphology_sole(spec):
            continue
        task = spec['task']
        normals = np.asarray(task.surface_normals)
        active = task.contact_mask & (normals[:, 2] >= UPWARD_COS)
        if task.environment_ids is not None:
            active &= np.asarray(task.environment_ids) == 0
        if not active.any():
            continue
        on_sole = model.body_weldid[model.geom_bodyid[geoms]] == model.body_weldid[spec['body_id']]
        for geom in np.asarray(geoms)[on_sole]:
            offsets = geom_surface_offsets(model, int(geom), objective.cfg['collision_spacing_m'])
            points = kin.geom_xpos[:, geom, None] + np.einsum('fij,sj->fsi', kin.geom_xmat[:, geom], offsets)
            gap = np.min(np.sum((points - task.surface_points[:, None]) * normals[:, None], axis=-1), axis=1)
            needed = (objective.cfg['scene_clearance_m'] - gap) / np.maximum(normals[:, 2], 1e-12)
            lower = np.maximum(lower, np.where(active, needed, 0.))
    window = max(3, round(LIFT_RAMP_SECONDS * objective.framerate) | 1)
    shift = uniform_filter1d(maximum_filter1d(lower, window, mode='nearest'), window, mode='nearest')
    result = poses.copy(); result[:, q + 2] += shift
    return result, shift


class ReferenceShape:
    """Fixed-reference posture preference on normalized landmark relations.

    Source-clear layout may cross zero briefly, but a sustained clear opposite
    side is collapse (``pose_relations``); no angle limits how far the whole
    body or one landmark changes. Relation angles to the fixed reference and to
    the immutable source, and the per-bone source directions of the seed, are
    reported so a distorted initialization or result is judged against the
    original motion.
    """

    def __init__(self, objective, seed):
        self.coords = objective.coords
        self.seed = np.array(seed, copy=True)
        self.seed.setflags(write=False)
        self.bodies = objective.envelope_bodies
        self.kin = TrajectoryKinematics(objective.model, self.coords.cols, cache=objective.kin.cache)
        self.kin.evaluate(self.seed, with_jacobian=False)
        self.rotations = self.kin.xmat[:, self.bodies].copy()
        self.envelopes = [RelationEnvelope.from_tasks(objective.tasks, self.kin)]
        self.source = np.asarray(objective.tasks.human)
        self.seed_source_angles = source_relation_angles(self.envelopes[0], self.kin, self.source)
        self.pose = SequencePoseGuard(objective.tasks, seed)
        self.tasks = objective.tasks
        self.xy = objective.cfg['translation_step_m']
        self.rejections = 0
        self.seed_source_errors = self.source_errors()
        self.acquired_poses = None

    def freeze_reference(self, poses):
        """Add the acquired reference once; retain the original seed bounds."""
        if self.acquired_poses is not None:
            raise RuntimeError("The acquired source reference is already frozen")
        self.acquired_poses = np.array(poses, copy=True)
        if not np.array_equal(poses, self.seed):
            self.kin.evaluate(poses, with_jacobian=False)
            self.envelopes.append(RelationEnvelope.from_tasks(self.tasks, self.kin))

    def source_errors(self):
        """Per-bone direction angles to the source on the evaluated ``kin``."""
        errors = np.zeros(self.pose.direction_valid.shape)
        for i, (a, b) in enumerate(self.pose.edges):
            direction = self.kin.xpos[:, self.pose.tasks.bodyids[b]] - self.kin.xpos[:, self.pose.tasks.bodyids[a]]
            length = np.linalg.norm(direction, axis=1)
            direction = direction / np.maximum(length[:, None], 1e-12)
            cosine = np.einsum("fk,fk->f", direction, self.pose.direction_targets[:, i])
            errors[:, i] = np.arccos(np.clip(cosine, -1.0, 1.0))
            errors[length < 1e-10, i] = np.pi
        return np.where(self.pose.direction_valid, errors, 0.0)

    def window_dependencies(self, frames):
        from .window_guards import guard_dependencies
        return guard_dependencies(self, frames, len(self.seed))

    def qp_bounds(self, poses, *, frames=None):
        from .window_guards import guard_qp_bounds
        return guard_qp_bounds(self, poses, frames=range(len(poses)) if frames is None else frames)

    def qp_values(self, poses, *, frames=None):
        from .window_guards import guard_qp_values
        linear, cones = guard_qp_values(self, poses, frames=range(len(poses)) if frames is None else frames)
        return ([(lo, hi) for _, lo, hi in linear],
                [(residual, upper) for _, residual, upper in cones])

    def accepts(self, poses, frames=None):
        if frames is not None:
            from .window_guards import guard_accepts
            good = guard_accepts(self, poses, frames=frames)
            self.rejections += not good
            return good
        if not np.isfinite(poses).all():
            self.rejections += 1
            return False
        self.kin.evaluate(poses, with_jacobian=False)
        good = all(envelope.accepts(self.kin) for envelope in self.envelopes)
        if not good:
            self.rejections += 1
        return bool(good)

    def violation_frames(self, poses, *, frames=None):
        if frames is not None:
            from .window_guards import guard_violation_frames
            return guard_violation_frames(self, poses, frames=frames)
        self.kin.evaluate(poses, with_jacobian=False)
        bad = np.zeros(len(poses), bool)
        for envelope in self.envelopes:
            bad |= envelope.violations(self.kin)
        return bad

    def constraints(self, poses, flat, box=None):
        """Root translation balls and linearized layout rows for a conic step.

        ``box`` bounds the step per coordinate; layout components it cannot
        flip's allowed deadband are omitted (``RelationEnvelope.linear_bounds``).
        """
        self.kin.evaluate(poses)
        coords = self.coords
        constraints = []
        if coords.root_q is not None:
            rows = LocalNormBounds(len(poses), coords.width)
            jac = np.zeros((3, coords.width))
            for axis in range(3):
                column = coords.inverse.get(coords.root_d + axis)
                if column is not None:
                    jac[axis, column] = 1.0
            q = coords.root_q
            for reference in [self.seed] + ([] if self.acquired_poses is None else [self.acquired_poses]):
                for f in range(len(poses)):
                    rows.add(poses[f, q:q + 3] - reference[f, q:q + 3], self.xy, (f, jac))
            constraints += rows.constraints(flat)
        bounds = [envelope.linear_bounds(self.kin, coords.width, box) for envelope in self.envelopes]
        return constraints + layout_constraints(bounds, flat)

    def repair_bounds(self, poses, *, restore=False):
        self.kin.evaluate(poses)
        linear = []
        for envelope in self.envelopes:
            linear += envelope.linear_bounds(self.kin, self.coords.width, restore=restore)
        return linear, []

    def project(self, poses):
        """Root XY stays on its original path; vertical registration is free.

        Joint limits are handled by the objective's scalar boxes. The layout
        is checked nonlinearly after this.
        """
        q = self.coords.root_q
        if q is not None:
            delta = poses[:, q:q + 2] - self.seed[:, q:q + 2]
            scale = np.minimum(1., self.xy / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))
            poses[:, q:q + 2] = self.seed[:, q:q + 2] + delta * scale[:, None]
        return poses

    def diagnostics(self, poses):
        self.kin.evaluate(poses, with_jacobian=False)
        relation = self.envelopes[0].summary(self.kin)
        bad = np.zeros(len(poses), bool)
        for envelope in self.envelopes:
            bad |= envelope.violations(self.kin)
        source_bad = self.envelopes[0].source_flips(self.kin) > self.envelopes[0].SLACK
        relative = self.kin.xmat[:, self.bodies] @ self.rotations.transpose(0, 1, 3, 2)
        rotation = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
        source_angles = source_relation_angles(self.envelopes[0], self.kin, self.source)
        bones = self.source_errors()
        seed_frame, seed_bone = np.unravel_index(np.argmax(self.seed_source_errors), self.seed_source_errors.shape)
        return {
            'fixed_seed': True, 'source_modified': False,
            'standard': 'normalized landmark relations, equivalent rotation angles',
            'posture_limit': 'none; soft preference for the fixed reference, judged against the source',
            'relation_envelope': relation,
            'body_rotation_change_max_rad': float(rotation.max(initial=0)),
            'source_relation_angle_seed_max_rad': float(self.seed_source_angles.max(initial=0)),
            'source_relation_angle_max_rad': float(source_angles.max(initial=0)),
            'source_relation_angle_growth_max_rad': float(
                (source_angles - self.seed_source_angles).max(initial=0)),
            # The seed's per-bone source directions are reported, never certified.
            'seed_source_bone_angle_max_rad': float(self.seed_source_errors.max(initial=0)),
            'seed_source_bone_angle_frame': int(seed_frame) if self.seed_source_errors.size else 0,
            'seed_source_bone_angle_edge': ([self.tasks.names[i] for i in self.pose.edges[seed_bone]]
                                            if self.seed_source_errors.size else None),
            'source_bone_angle_max_rad': float(bones.max(initial=0)),
            'fixed_reference_layout_preserved': not bool(bad.any()),
            'source_layout_preserved': not bool(source_bad.any()),
            'layout_preserved': not bool((bad | source_bad).any()),
            'layout_flip_frames': np.flatnonzero(bad | source_bad).tolist(),
            'rejected_candidates': self.rejections,
        }
