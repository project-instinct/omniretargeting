"""Vectorized sequence objective with the rows and sigmas of SequenceTasks.

Every per-frame row family of ``SequenceTasks.frame_rows`` and the root,
reference and temporal rows of ``retarget_sequence`` are evaluated for all
frames at once. Per-frame normal equations ``J_f^T J_f`` are accumulated
directly; only the few temporal rows that couple frames stay sparse.
Scene clearance uses a precomputed grid (``sdf.GridSDF``) instead of
original-triangle queries; the original triangles are audited afterwards.
"""

import copy
from contextlib import contextmanager
import itertools
import multiprocessing
import os
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from ..cpu_budget import available_cpus
from .._sequence.coordinates import temporal_operators
from .._solver.targets import _world_target_regime
from ..contact import _is_surface_only_contact_mode
from ..motion_budgets import CHATTER_SCALE_FRAMES
from ..pose_relations import RelationEnvelope
from ..scene_geometry import collision_geom_ids, geom_surface_offsets
from .kinematics import TrajectoryKinematics
from .penetration import PenetrationStages
from .rows import (  # noqa: F401 - re-exported
    _Accumulator,
    _block_rows,
    _point_rows,
    _unit,
)
from .terms import (  # noqa: F401 - re-exported
    ENVELOPE_SIGMA,
    NEAR_MARGIN,
    FrameTerms,
)

ROLE_WEIGHTS = {"support": 4.0, "hold": 2.0, "control": 2.0, "follow": 1.0}
# Per-frame terms are independent: frames are split across forked workers.
# The worker count changes scheduling only, never the computed values.
WORKERS = max(1, min(12, available_cpus() // 2))
# Shorter chunks cost more in process round trips than they save. Frames
# near contacts and scene surfaces cost far more than free ones, so the
# trajectory is split into more chunks than workers and scheduled dynamically.
MIN_CHUNK_FRAMES = 7
CHUNKS_PER_WORKER = 2
# Shorter trajectories measure their clearance fields serially without a pool.
TEMPORARY_DEEPEST_FRAMES = 120
# Small windows amortize device transfers poorly; keep their reference pool.
MIN_ACCELERATED_FRAMES = 64
_WORKER = {}


def _owned_pool(objective):
    """A pool's task/result threads exist only in the process that made it."""
    pool = getattr(objective, 'pool', None)
    if pool is None:
        return None
    owner = getattr(objective, '_pool_pid', None)
    if owner is None:
        # Also cover an objective whose pool predates the ownership marker.
        workers = getattr(pool, '_pool', ())
        owner = getattr(workers[0], '_parent_pid', os.getpid()) if workers else os.getpid()
    return pool if owner == os.getpid() else None


def _worker_init():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(1)


def _worker_evaluate(task):
    start, stop, poses, with_jacobian, envelope_weight = task
    chunks = _WORKER.setdefault("chunks", {})
    chunk = chunks.get((start, stop))
    if chunk is None:
        chunk = chunks[start, stop] = _WORKER["objective"].for_frames(slice(start, stop))
    result = chunk.evaluate_frames(poses, with_jacobian, envelope_weight)
    # Each task owns a disjoint frame range. Only small diagnostics travel
    # through the pipe; the dense normal equations stay in shared memory.
    for name, output in _WORKER["output"].items():
        value = result.pop(name)
        if value is not None:
            output[start:stop] = value
    return result


@contextmanager
def _temporary_pool(objective, workers):
    """Forked workers inheriting ``objective``; the parent's registry is restored."""
    saved = dict(_WORKER)
    _WORKER.clear()
    _WORKER["objective"] = objective
    try:
        pool = multiprocessing.get_context("fork").Pool(workers, _worker_init)
    finally:
        _WORKER.clear()
        _WORKER.update(saved)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()


def _worker_deepest(task):
    start, stop, poses = task
    chunks = _WORKER.setdefault("chunks", {})
    chunk = chunks.get((start, stop))
    if chunk is None:
        chunk = chunks[start, stop] = _WORKER["objective"].for_frames(slice(start, stop))
    from .bulk_geometry import prepare
    prepare(chunk)
    chunk.kin.evaluate(poses, with_jacobian=False)
    return [chunk._deepest_samples(*field[1:]) for field in chunk._clearance_fields()]


# Clearance/self-collision sigma relative to the contact sigma during acquisition.
ACQUISITION_CLEARANCE_SCALE = 0.25


def _without_chatter(rotations):
    """Source root rotations without their content below the chatter scale.

    Only the root path rows measure against this path; the root orientation
    target keeps every recorded source rotation.
    """
    quat = rotations.as_quat()
    flips = np.cumsum(np.r_[False, np.einsum("ij,ij->i", quat[1:], quat[:-1]) < 0]) % 2 == 1
    quat[flips] *= -1
    smooth = gaussian_filter1d(quat, CHATTER_SCALE_FRAMES, axis=0, mode="nearest")
    return Rotation.from_quat(smooth / np.linalg.norm(smooth, axis=1, keepdims=True))


class FastObjective(FrameTerms, PenetrationStages):
    """All rows of one sequence stage as vectorized per-frame blocks."""

    def __init__(
        self,
        task_set,
        cfg,
        *,
        reference,
        temporal_reference,
        root_rotations,
        framerate,
        scene_sdf=None,
        objects=(),
        surface_tasks=(),
        proximity_tasks=(),
        role_policy=None,
    ):
        self.tasks, self.cfg = task_set, cfg
        self.model, self.coords = task_set.model, task_set.coords
        self.frames, self.width = task_set.frames, task_set.coords.width
        self.kin = TrajectoryKinematics(self.model, self.coords.cols)
        self.reference = np.asarray(reference, dtype=float)
        self.temporal_reference = temporal_reference
        self.root_rotations = root_rotations
        # Root path rows follow the source root path, as in the guarded
        # solver, but not its rotation chatter: joints never copy the source's
        # local-rotation noise, and a root that copied the pelvis noise made
        # stance feet or the joints holding them chatter instead.
        self.root_path_rotations = None if root_rotations is None else _without_chatter(root_rotations)
        self.scene_sdf, self.objects = scene_sdf, list(objects)
        self.surface_tasks = list(surface_tasks)
        self._pack_interaction()
        self._pack_effectors(role_policy)
        self._pack_surfaces(role_policy)
        self._pack_proximity(proximity_tasks)
        self._pack_shells()
        coords = self.coords
        self.selector = coords.selector.toarray()
        self.joint_ops = temporal_operators(
            self.frames, framerate, cfg["joint_derivative_sigmas"]
        )
        # Multipliers of the temporal rows (one per stencil start) and of the
        # per-frame envelope hinges; the verification step raises them locally.
        self.temporal_weight = [np.ones(op.shape[0]) for op in self.joint_ops]
        self.joint_temporal_weight = [np.ones((op.shape[0], len(coords.qi))) for op in self.joint_ops]
        self.root_translation_weight = [np.ones(op.shape[0]) for op in self.joint_ops]
        self.root_rotation_weight = [np.ones(op.shape[0]) for op in self.joint_ops]
        self.envelope_weight = np.ones(self.frames)
        # Extra stiffness of the joint rows only: with a fixed reference they
        # measure the correction, whereas root rows follow the source path.
        self.correction_temporal_scale = 1.0
        self.joint_matrices = [
            sparse.kron(op, coords.selector, format="csr") for op in self.joint_ops
        ]
        if coords.root_q is not None:
            self.xy_ops = temporal_operators(
                self.frames, framerate, cfg["root_xy_derivative_sigmas"]
            )
            self.height_ops = temporal_operators(
                self.frames, framerate, cfg["root_height_derivative_sigmas"]
            )
            self.angular_ops = temporal_operators(
                self.frames, framerate, cfg["root_quaternion_derivative_sigmas"]
            )
            self.angular_cols = [coords.inverse.get(coords.root_d + d) for d in (3, 4, 5)]
            self.quat_slice = slice(coords.root_q + 3, coords.root_q + 7)
        self._pack_world_paths(framerate)
        # Bodies whose cumulative rotation the envelope bounds, as in the
        # guarded SequencePoseGuard: mapped, end-effector and collision bodies.
        bodies = set(task_set.bodyids) | {spec["body_id"] for spec in task_set.effectors.values()}
        bodies |= {int(b) for b in self.model.geom_bodyid[collision_geom_ids(self.model)]}
        self.envelope_bodies = np.array(sorted({int(self.model.body_weldid[b]) for b in bodies} - {0}))
        self.kin.evaluate(self.reference, with_jacobian=False)
        self.reference_rotations = self.kin.xmat[:, self.envelope_bodies].copy()
        # Posture envelope of the fixed reference (``pose_relations``).
        self.relation_envelope = (RelationEnvelope.from_tasks(task_set, self.kin)
                                  if temporal_reference is not None else None)
        # Non-penetration precedes contacts while contacts are acquired: one
        # deepest shell sample per geom opposes dozens of contact samples per
        # body, so at the contact sigma contacts pulled wrists into the box.
        # Clearance and self-collision rows use a quarter of it (omomo_78 G1:
        # box 3.8 -> 1.9 cm, floor 2.8 -> 0.8 cm; a tenth worsened self-collision
        # and hands). The free-body stage and repair continue at the full sigma.
        self.clearance_scale = ACQUISITION_CLEARANCE_SCALE
        # Self-collision rows once a fixed reference exists (contact stages).
        self.self_collision = None
        if temporal_reference is not None and cfg.get("self_collision_enabled", False):
            from ..self_collision import SelfCollisionQuery

            self.self_collision = SelfCollisionQuery(self.model)
        self.framerate = float(framerate)
        # Penetration stages (``PenetrationStages``): plane rows per (frame,
        # geom), their sigma scale, and the motion whose continuity the
        # temporal rows preserve (None: the fixed reference).
        shape = (self.frames, len(self.shell_geoms))
        self.plane_normal = np.zeros((*shape, 3))
        self.plane_offset = np.zeros(shape)
        self.plane_field = np.full(shape, -1)
        self.plane_centre = np.zeros((*shape, 3))
        self.plane_radius = np.full(shape, np.inf)
        self.plane_scale = 1.0
        self.continuity_joints = None
        self.continuity_root = None
        # Per-frame boxes: joint limits and, once a fixed reference exists, the
        # translation bound of slide joints. Hinge joints and the root rotation
        # have no posture limit; the root translation keeps the guard's norm
        # bound (``project``).
        count = len(coords.qi)
        self.lower = np.broadcast_to(coords.lower, (self.frames, count)).copy()
        self.upper = np.broadcast_to(coords.upper, (self.frames, count)).copy()
        self.root_budget = None
        if temporal_reference is not None:
            model = self.model
            slide = model.jnt_type[model.dof_jntid[coords.di]] == mujoco.mjtJoint.mjJNT_SLIDE
            budget = np.where(slide, cfg["translation_step_m"], np.inf)
            self.lower = np.maximum(self.lower, self.reference[:, coords.qi] - budget)
            self.upper = np.minimum(self.upper, self.reference[:, coords.qi] + budget)
            if coords.root_q is not None:
                self.root_budget = cfg["translation_step_m"]
        if coords.root_q is not None:
            self.root_reference = self._root_points()
        self.stats = {}
        self.source_shape = None
        self.candidate_guards = []
        self.fixed_root = None
        # Exact accelerated rows passed the full LM/trajectory comparison.
        # Keep the reference pool available through the explicit legacy switch.
        self.evaluation_backend = os.environ.get('OMNIRETARGETING_FAST_OBJECTIVE', 'auto')
        if self.evaluation_backend not in ('legacy', 'cpu', 'gpu', 'auto'):
            raise ValueError('OMNIRETARGETING_FAST_OBJECTIVE must be legacy, cpu, gpu or auto')

    # ----- packing of immutable task data -------------------------------------
    def _pack_interaction(self):
        ops = self.tasks.operators
        masks = [np.any(op != 0, axis=1) for op in ops]
        rows = max(int(m.sum()) for m in masks)
        n3 = ops[0].shape[1]
        self.interaction = np.zeros((self.frames, rows, n3))
        for f, (op, mask) in enumerate(zip(ops, masks)):
            self.interaction[f, : mask.sum()] = op[mask]

    def _pack_effectors(self, role_policy):
        self.effectors = []
        frames = self.frames
        for name, spec in self.tasks.effectors.items():
            data = [spec["task"].frame(f) for f in range(frames)]
            active = np.array([bool(d["contact"]) for d in data])
            managed = np.array([bool(d.get("surface_managed", False)) for d in data])
            surface_only = np.array(
                [a and _is_surface_only_contact_mode(d["contact_mode"]) for a, d in zip(active, data)]
            )
            world = np.array([_world_target_regime(spec, d) for d in data])
            share = np.array([float(d.get("approach_weight", 0.0)) for d in data])
            choice = self.tasks.choices[name]
            contact_offset = spec["contact_offsets"][choice]
            offset = np.where(
                active[:, None],
                contact_offset,
                (1 - share)[:, None] * spec["tracking_offset"] + share[:, None] * contact_offset,
            )
            normals = np.array(
                [d.get("surface_normal", [0, 0, 1]) if a else [0, 0, 1] for a, d in zip(active, data)],
                dtype=float,
            )
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            # Orientation rows (``track_orientation``), except on frames whose
            # contact is constrained by its rigid surface block.
            orientation = []
            if spec.get("track_orientation", False):
                for local_key, target_key, default in (
                    ("orientation_local_up", "up_direction", [0, 0, 1]),
                    ("orientation_local_forward", "forward_direction", [1, 0, 0]),
                ):
                    present = np.array([d.get(target_key) is not None for d in data])
                    target = np.array(
                        [d[target_key] if ok else np.zeros(3) for d, ok in zip(data, present)],
                        dtype=float,
                    )
                    valid = present & (np.linalg.norm(target, axis=1) >= 1e-8) & ~(active & managed)
                    orientation.append(
                        (np.asarray(spec.get(local_key, default), dtype=float), _unit(target), valid)
                    )
            role = np.ones(frames)
            if role_policy is not None:
                role = np.array(
                    [ROLE_WEIGHTS.get(role_policy.role(name, f), 1.0) for f in range(frames)]
                )
            self.effectors.append(
                {
                    "name": name,
                    "body": spec["body_id"],
                    "active": active & ~managed,
                    "managed": active & managed,
                    "surface_only": surface_only,
                    "world": world,
                    "share": share,
                    "offset": offset,
                    "contact": np.array([d.get("contact_position", np.zeros(3)) for d in data], dtype=float),
                    "tracking": np.array([d["tracking_position"] for d in data], dtype=float),
                    "approach": np.array([d.get("approach_position", np.zeros(3)) for d in data], dtype=float),
                    "normal": normals,
                    "scale": float(spec.get("swing_morphology_scale", 1.0)),
                    "role": np.sqrt(role),
                    "orientation": orientation,
                }
            )

    def _pack_world_paths(self, framerate):
        """World-tracked free effectors follow their source world path as the
        root follows its source path: the same derivative sigmas bound the
        offset from that path (``root_xy_derivative_sigmas``). Without them a
        stance foot only had its 2 cm position sigma and absorbed the source
        root's rotation noise as chatter while the source foot stayed planted.
        As for the root, the path excludes its chatter (content below
        ``CHATTER_SCALE_FRAMES``): copying the spikes of a skin-derived foot
        target doubled the leg jerk. The per-frame target keeps every value.
        Rows cover stencils whose frames are all world-tracked and free, so
        contact targets and regime handovers are never differentiated.
        """
        self.path_ops = temporal_operators(self.frames, framerate, self.cfg["root_xy_derivative_sigmas"])
        self.world_paths = []
        for index, eff in enumerate(self.effectors):
            tracked = eff["world"] & ~(eff["active"] | eff["managed"])
            masks = [np.lib.stride_tricks.sliding_window_view(tracked, order + 1).all(axis=1)
                     for order in range(1, len(self.path_ops) + 1)]
            if any(mask.any() for mask in masks):
                target = eff["tracking"] + eff["share"][:, None] * (eff["approach"] - eff["tracking"])
                path = gaussian_filter1d(target, CHATTER_SCALE_FRAMES, axis=0, mode="nearest")
                self.world_paths.append((index, path, masks))

    def _pack_surfaces(self, role_policy=None):
        self.surfaces = []
        for task in self.surface_tasks:
            # Role levels of the guarded cascade become stiff rows: every part's
            # samples share one endpoint-contact strength, scaled by role.
            priority = np.zeros(np.shape(task.weights))
            roles = getattr(role_policy, "surface_roles", None)
            if roles is not None and getattr(role_policy, "surface", None) is task:
                for role, mask in roles.items():
                    priority = np.where(mask, ROLE_WEIGHTS.get(role, 1.0), priority)
            normalized = np.asarray(task.weights, dtype=float) / task.cfg["weight"]
            kin = task.kinematics
            geoms = np.asarray(kin._geoms, dtype=int)
            # Relative reference-pose normal rows of ``SurfaceContactTasks.rows``.
            normal_scale = np.zeros(np.shape(task.weights))
            pose_normals = np.zeros((*np.shape(task.weights), 3))
            if getattr(task, "pose_normal_targets", None) is not None:
                pose_normals = np.asarray(task.pose_normal_targets, dtype=float)
                normal_scale = np.sqrt(
                    task.cfg["normal_weight"] * np.asarray(task.weights, dtype=float)
                    * (np.asarray(task.active) & np.asarray(task.pose_normal_valid))
                ) / task.cfg["normal_sigma"]
            self.surfaces.append(
                {
                    "task": task,
                    "geoms": geoms,
                    "bodies": self.model.geom_bodyid[geoms],
                    "local": np.asarray(kin._points, dtype=float),
                    "local_normals": np.asarray(kin._normals, dtype=float),
                    "pose_normals": pose_normals,
                    "normal_scale": normal_scale,
                    "hold": np.full(np.shape(task.weights), np.inf),
                    "scale": np.sqrt(np.asarray(task.weights, dtype=float)) / task.cfg["sigma"],
                    "targets": np.asarray(task.targets, dtype=float),
                    "active": np.asarray(task.active, dtype=bool),
                    "sticking": np.asarray(task.modes) == "sticking",
                    # Unit normals: the guarded projector is n n^T / (n . n).
                    "normals": _unit(np.asarray(task.normals, dtype=float)),
                    "priority": np.sqrt(priority * normalized) / self.cfg["contact_sigma_m"],
                }
            )

    def _pack_proximity(self, tasks):
        """``SurfaceProximityTasks`` rows: support-extent gaps along normals."""
        self.proximities = []
        for task in tasks:
            geoms = np.asarray(task.geoms, dtype=int)
            self.proximities.append(
                {
                    "geoms": geoms,
                    "bodies": self.model.geom_bodyid[geoms],
                    "environment": np.asarray(task.environment, dtype=float),
                    "normals": np.asarray(task.normals, dtype=float),
                    "gaps": np.asarray(task.support_gaps, dtype=float),
                    "scale": np.sqrt(np.asarray(task.weights, dtype=float)),
                }
            )

    def _pack_shells(self):
        self.shell_geoms = collision_geom_ids(self.model)
        offsets = [
            geom_surface_offsets(self.model, int(g), self.cfg["collision_spacing_m"])
            for g in self.shell_geoms
        ]
        self.shell_sample_geom = np.concatenate(
            [np.full(len(o), g) for g, o in zip(self.shell_geoms, offsets)]
        ).astype(int)
        self.shell_local = np.concatenate(offsets)
        self.shell_radius = np.array([np.linalg.norm(o, axis=1).max() for o in offsets])
        width = max(len(o) for o in offsets)
        self.shell_index = np.full((len(offsets), width), -1)
        start = 0
        for k, o in enumerate(offsets):
            self.shell_index[k, : len(o)] = np.arange(start, start + len(o))
            start += len(o)

    def hold_contacts(self, poses, clearance_scale=1.0):
        """Hold attained contacts; set the clearance sigma scale.

        Every prioritized surface sample may regress by the surface task's
        ``regression`` tolerance from its error at ``poses``, as the guarded
        per-frame priority bounds allow. Restart workers after calling this.
        """
        self.kin.evaluate(poses, with_jacobian=False)
        for surf in self.surfaces:
            error = self.kin.geom_points(surf["geoms"], surf["local"]) - surf["targets"]
            size = np.where(
                surf["sticking"],
                np.linalg.norm(error, axis=-1),
                np.abs(np.einsum("fsi,fsi->fs", error, surf["normals"])),
            )
            surf["hold"] = np.where(
                surf["priority"] > 0, size + surf["task"].cfg["regression"], np.inf
            )
        self.clearance_scale = clearance_scale

    # ----- feasible set ----------------------------------------------------
    def _root_changes(self, poses):
        """Root translation shift and relative rotation vector (local frame)."""
        q = self.coords.root_q
        shift = poses[:, q : q + 3] - self.reference[:, q : q + 3]
        relative = Rotation.from_quat(self.reference[:, q + 3 : q + 7], scalar_first=True).inv() * (
            Rotation.from_quat(poses[:, q + 3 : q + 7], scalar_first=True)
        )
        return shift, relative.as_rotvec()

    def project(self, poses):
        """Clip joints to their boxes and the root translation onto its ball."""
        poses = poses.copy()
        poses[:, self.coords.qi] = np.clip(poses[:, self.coords.qi], self.lower, self.upper)
        if self.root_budget is not None:
            q = self.coords.root_q
            shift, _ = self._root_changes(poses)
            size = np.linalg.norm(shift, axis=1)
            over = size > self.root_budget
            poses[over, q : q + 3] = self.reference[over, q : q + 3] + shift[over] * (
                self.root_budget / size[over]
            )[:, None]
        if self.source_shape is not None:
            poses = self.source_shape.project(poses)
        if self.fixed_root is not None:
            q = self.coords.root_q
            poses[:, q:q + 7] = self.fixed_root
        return poses

    def root_boundaries(self, poses):
        """Outward unit normals (frames, 3) of the root translation block on its
        bound; translation dofs move the root in world axes. The root rotation
        has no bound."""
        if self.root_budget is None:
            return []
        shift, _ = self._root_changes(poses)
        size = np.linalg.norm(shift, axis=1)
        normal = shift / np.maximum(size, 1e-12)[:, None]
        return [(0, np.where((size >= self.root_budget - 1e-9)[:, None], normal, 0.0))]

    # ----- helpers ----------------------------------------------------------
    def _root_points(self):
        body = self.tasks.root_body
        if body is None:
            return np.asarray(self.tasks.roots)
        return self.kin.xpos[:, body].copy()

    def _root_jacobian(self):
        body = self.tasks.root_body
        if body is None:
            return np.zeros((self.frames, 3, self.width))
        return self.kin.jacp[:, body]

    def _effector_points(self, with_jacobian):
        """Effector points (frames, effectors, 3) and optionally Jacobians."""
        count = len(self.effectors)
        points = np.zeros((self.frames, count, 3))
        jacobians = np.zeros((self.frames, count, 3, self.width)) if with_jacobian else None
        for index, eff in enumerate(self.effectors):
            body = np.full((self.frames, 1), eff["body"])
            points[:, index] = self.kin.points(body, eff["offset"][:, None])[:, 0]
            if with_jacobian:
                jacobians[:, index] = self.kin.point_jacobians(body, points[:, index, None])[:, 0]
        return points, jacobians

    # ----- evaluation -------------------------------------------------------
    # ----- chunked parallel evaluation -----------------------------------
    PER_FRAME_EFFECTOR = ("active", "managed", "surface_only", "world", "share", "offset",
                          "contact", "tracking", "approach", "normal", "role")
    PER_FRAME_SURFACE = (
        "scale", "targets", "active", "sticking", "normals", "priority", "pose_normals", "normal_scale",
        "hold",
    )
    PER_FRAME_PROXIMITY = ("environment", "normals", "gaps", "scale")

    def for_frames(self, frames, *, kinematics=None):
        """Shallow copy restricted to a contiguous slice of frames."""
        chunk = copy.copy(self)
        # Device and collision caches belong to one trajectory, never a slice.
        for name in ('_gpu_frames', '_self_batch'):
            chunk.__dict__.pop(name, None)
        if '_bulk_geometry' in chunk.__dict__:
            for name in ('_bulk_geometry', '_proximity_support', '_deepest_samples'):
                chunk.__dict__.pop(name, None)
        chunk.frames = frames.stop - frames.start
        chunk.kin = (TrajectoryKinematics(self.model, self.coords.cols)
                     if kinematics is None else kinematics)
        chunk.interaction = self.interaction[frames]
        tasks = self.tasks
        chunk.tasks = SimpleNamespace(
            bodyids=tasks.bodyids, human=np.asarray(tasks.human)[frames], axes=tasks.axes,
            orientations=SimpleNamespace(
                targets=[(b, p, r[frames], s) for b, p, r, s in tasks.orientations.targets]
            ),
            roots=np.asarray(tasks.roots)[frames], root_body=tasks.root_body,
        )
        chunk.effectors = [
            {k: (v[frames] if k in self.PER_FRAME_EFFECTOR else v) for k, v in e.items()}
            for e in self.effectors
        ]
        for effector in chunk.effectors:
            effector["orientation"] = [
                (local, target[frames], valid[frames]) for local, target, valid in effector["orientation"]
            ]
        chunk.surfaces = [
            {k: (v[frames] if k in self.PER_FRAME_SURFACE else v) for k, v in s.items()}
            for s in self.surfaces
        ]
        chunk.proximities = [
            {k: (v[frames] if k in self.PER_FRAME_PROXIMITY else v) for k, v in s.items()}
            for s in self.proximities
        ]
        chunk.objects = [
            dict(o, translations=o["translations"][frames], rotations=o["rotations"][frames],
                 scales=None if o.get("scales") is None else o["scales"][frames])
            for o in self.objects
        ]
        chunk.reference = self.reference[frames]
        chunk.plane_normal = self.plane_normal[frames]
        chunk.plane_offset = self.plane_offset[frames]
        chunk.plane_field = self.plane_field[frames]
        chunk.plane_centre = self.plane_centre[frames]
        chunk.plane_radius = self.plane_radius[frames]
        chunk.reference_rotations = self.reference_rotations[frames]
        chunk.frame_slice = frames
        if self.relation_envelope is not None:
            chunk.relation_envelope = self.relation_envelope.for_frames(frames)
        if self.coords.root_q is not None:
            chunk.root_reference = self.root_reference[frames]
            chunk.root_rotations = self.root_rotations[frames]
        return chunk

    def evaluate_frames(self, poses, with_jacobian, envelope_weight):
        """Per-frame terms of this (chunk) objective, plus root kinematics."""
        from .bulk_geometry import prepare
        prepare(self)
        self.envelope_weight = envelope_weight
        self.kin.evaluate(poses, with_jacobian)
        acc = _Accumulator(self.frames, self.width, with_jacobian)
        stats = {}
        self._mapped_terms(acc, stats)
        self._orientation_terms(acc, stats)
        self._effector_terms(acc, stats)
        self._surface_terms(acc, stats)
        self._proximity_terms(acc, stats)
        if getattr(self, '_soft_bulk_signals', False):
            from .soft_frame_rows import add_clearance
            add_clearance(self, poses, acc, stats)
        else:
            self._clearance_terms(acc, stats, poses)
            self._plane_terms(acc, stats)
            self._self_collision_terms(poses, acc, stats)
        self._reference_terms(poses, acc, stats)
        self._envelope_terms(poses, acc, stats)
        effector_point, effector_jacobian = self._effector_points(with_jacobian)
        result = {
            "cost": acc.cost,
            "frame_cost": acc.frame_cost,
            "hessian": acc.hessian if with_jacobian else None,
            "gradient": acc.gradient if with_jacobian else None,
            "stats": stats,
            "root_point": self._root_points(),
            "root_jacobian": self._root_jacobian() if with_jacobian else None,
            "effector_point": effector_point,
            "effector_jacobian": effector_jacobian,
        }
        if getattr(self, '_soft_bulk_signals', False):
            result['soft_guard_cost'] = getattr(acc, 'soft_guard_cost', 0.)
            result['soft_clearance_cost'] = acc.soft_clearance_cost
            result['soft_priority_extra'] = acc.soft_priority_extra
        return result

    def start_workers(self):
        """Fork workers inheriting this objective; False if already running."""
        if getattr(self, 'pool', None) is not None and _owned_pool(self) is None:
            self.pool, self.worker_output = None, None
        from .guard_acceleration import enabled
        if enabled():
            from .guard_signals import prepare
            prepare(self)
        if self._gpu_evaluator() is not None:
            if getattr(self, '_gpu_workers_active', False):
                return False
            self._gpu_workers_active = True
            return True
        if getattr(self, "pool", None) is not None:
            return False
        from .bulk_geometry import prepare, worker_schedule
        prepare(self)  # Build once before fork; children inherit the library.
        workers, chunks_per_worker = worker_schedule(WORKERS, CHUNKS_PER_WORKER)
        self.chunks = min(chunks_per_worker * workers, self.frames // MIN_CHUNK_FRAMES)
        if self.chunks > 1 and hasattr(os, "fork") and not multiprocessing.current_process().daemon:
            context = multiprocessing.get_context("fork")
            shapes = {
                "hessian": (self.frames, self.width, self.width),
                "gradient": (self.frames, self.width),
                "root_point": (self.frames, 3),
                "root_jacobian": (self.frames, 3, self.width),
                "effector_point": (self.frames, len(self.effectors), 3),
                "effector_jacobian": (self.frames, len(self.effectors), 3, self.width),
            }
            self.worker_output = {
                name: np.frombuffer(context.RawArray("d", int(np.prod(shape)))).reshape(shape)
                for name, shape in shapes.items()
            }
            _WORKER.clear()
            _WORKER["objective"] = self
            _WORKER["output"] = self.worker_output
            self._pool_pid = os.getpid()
            self.pool = context.Pool(min(workers, self.chunks), _worker_init)
        return True

    def stop_workers(self):
        self._gpu_workers_active = False
        if getattr(self, 'pool', None) is not None and _owned_pool(self) is None:
            self.pool, self.worker_output = None, None
            return
        if getattr(self, "pool", None) is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
            self.worker_output = None
            if _WORKER.get("objective") is self:
                _WORKER.clear()

    def deepest_fields(self, poses):
        """``_deepest_samples`` of every clearance field over the trajectory.

        With workers running, frame chunks are measured in parallel and joined
        in frame order: each (frame, geom) pair is independent of the others.
        ``self.kin`` is left evaluated at ``poses`` either way.
        """
        from .bulk_geometry import prepare
        prepare(self)
        pool = _owned_pool(self)
        chunks = min(CHUNKS_PER_WORKER * WORKERS, self.frames // MIN_CHUNK_FRAMES)
        if pool is None and (chunks < 2 or self.frames < TEMPORARY_DEEPEST_FRAMES or not hasattr(os, "fork")
                             or multiprocessing.current_process().daemon):
            self.kin.evaluate(poses, with_jacobian=False)
            return [self._deepest_samples(*field[1:]) for field in self._clearance_fields()]
        edges = np.linspace(0, self.frames, (self.chunks if pool is not None else chunks) + 1).astype(int)
        bounds = list(itertools.pairwise(edges))
        tasks = [(a, b, poses[a:b]) for a, b in bounds]
        if pool is None:
            # No evaluation pool (e.g. the GPU evaluator): the same chunked
            # measurement in temporary forked workers.
            with _temporary_pool(self, min(WORKERS, chunks)) as temporary:
                parts = temporary.map(_worker_deepest, tasks, chunksize=1)
        else:
            parts = pool.map(_worker_deepest, tasks, chunksize=1)
        fields = []
        for field in range(len(parts[0])):
            pieces = [part[field] for part in parts]
            fields.append((np.concatenate([p[0] for p in pieces]),
                           np.concatenate([p[1] + a for p, (a, _) in zip(pieces, bounds)]),
                           np.concatenate([p[2] for p in pieces]),
                           np.concatenate([p[3] for p in pieces]),
                           np.concatenate([p[4] for p in pieces])))
        self.kin.evaluate(poses, with_jacobian=False)
        return fields

    def prepare_window_evaluation(self):
        """Prepare single-threaded CPU window kernels before forking."""
        from .window_objective import prepare_window_evaluation
        return prepare_window_evaluation(self)

    def window_kinematics(self, poses, frames, with_jacobian=True):
        """Exact local FK, with immutable arrays and global body/geom IDs."""
        from .frame_kinematics import window_kinematics
        kin = window_kinematics(self, poses, frames, with_jacobian)
        guards = list(self.candidate_guards)
        if self.source_shape is not None:
            guards.append(self.source_shape)
        for guard in guards:
            guard._window_frame_cache = self._window_frame_cache
        return kin

    def window_dependencies(self, frames):
        """Fixed frames that any original temporal stencil can depend on."""
        from .window_objective import window_dependencies
        return window_dependencies(self, frames)

    def evaluate(self, poses, with_jacobian=True, *, frames=None):
        """Cost, per-frame normal equations and sparse temporal rows."""
        if frames is not None:
            from .window_objective import evaluate_window
            return evaluate_window(self, poses, with_jacobian, frames)
        pool = _owned_pool(self)
        gpu = self._gpu_evaluator()
        if gpu is not None:
            parts = [gpu.evaluate(poses, with_jacobian, self.envelope_weight)]
            pool = None
        elif pool is None:
            parts = [self.evaluate_frames(poses, with_jacobian, self.envelope_weight)]
        else:
            edges = np.linspace(0, self.frames, self.chunks + 1).astype(int)
            tasks = [
                (a, b, poses[a:b], with_jacobian, self.envelope_weight[a:b])
                for a, b in itertools.pairwise(edges)
            ]
            parts = pool.map(_worker_evaluate, tasks, chunksize=1)

        def collect(name):
            if pool is None:
                return parts[0][name]
            # The caller may retain a linearization across rejected trials.
            # Return owned arrays so subsequent evaluations cannot overwrite it.
            return self.worker_output[name].copy()

        acc = _Accumulator(self.frames, self.width, False)
        acc.jacobian = with_jacobian
        # Frame costs in frame order: the sum does not depend on the chunking.
        acc.frame_cost = np.concatenate([part["frame_cost"] for part in parts])
        acc.cost = float(acc.frame_cost.sum())
        if with_jacobian:
            acc.hessian = collect("hessian")
            acc.gradient = collect("gradient")
        stats = {}
        for part in parts:
            for key, value in part["stats"].items():
                # Frame counts add up over chunks; every other statistic is a maximum.
                stats[key] = (stats.get(key, 0) + value if key.endswith("_frames")
                              else max(stats.get(key, 0.0), value))
        root_point = collect("root_point")
        root_jac = collect("root_jacobian") if with_jacobian else None
        paths = (collect("effector_point"), collect("effector_jacobian") if with_jacobian else None)
        temporal_r, temporal_j = self._temporal_terms(poses, with_jacobian, root_point, root_jac, paths)
        acc.cost += float(temporal_r @ temporal_r)
        from .window_tracking import collision_stats
        stats = collision_stats(self, poses, stats)
        self.stats = stats
        signals = None
        if getattr(self, '_soft_bulk_signals', False):
            signals = dict(root_point=root_point, root_jacobian=root_jac,
                           guard_cost=sum(part.get('soft_guard_cost', 0.) for part in parts),
                           clearance_cost=sum(part.get('soft_clearance_cost', 0.) for part in parts),
                           priority_extra=sum(part.get('soft_priority_extra', 0.) for part in parts))
        return acc, temporal_r, temporal_j, stats, signals

    def _gpu_evaluator(self):
        """CUDA belongs to the caller process; inherited workers keep CPU FK."""
        backend = getattr(self, 'evaluation_backend', 'legacy')
        from .runtime_options import OPTIONS
        # The opt-in CPU kernels own auto dispatch; an explicit GPU request
        # still selects CUDA. Otherwise later stages bypass the bulk hooks.
        if backend == 'auto' and OPTIONS.bulk_cpu_geometry:
            return None
        if backend not in ('gpu', 'auto'):
            return None
        if backend == 'auto' and (self.temporal_reference is None or self.frames < MIN_ACCELERATED_FRAMES):
            return None
        if getattr(self, '_gpu_pid', os.getpid()) != os.getpid():
            return None
        if not hasattr(self, '_gpu_frames'):
            self._gpu_pid = os.getpid()
            self._gpu_frames = None
            try:
                import torch
            except ImportError:
                return None
            if torch.cuda.is_available():
                try:
                    from .batched_self_collision import native_library, NativeUnavailable
                    native_library()
                    from .gpu_objective import GPUFrameEvaluator
                    self._gpu_frames = GPUFrameEvaluator(self)
                except (ImportError, NativeUnavailable):
                    # Unsupported hardware/dependencies retain the reference
                    # pool. Numerical/runtime errors are never hidden here.
                    pass
        return self._gpu_frames

    def _temporal_terms(self, poses, with_jacobian, root_point, root_jac, paths=None):
        coords = self.coords
        motion = poses[:, coords.qi]
        if self.continuity_joints is not None:
            motion = motion - self.continuity_joints
        elif self.temporal_reference is not None:
            motion = motion - self.temporal_reference
        residuals, matrices = [], []
        weights = self.temporal_weight
        for op, weight, channels in zip(self.joint_ops, weights, self.joint_temporal_weight):
            rows = ((weight * self.correction_temporal_scale)[:, None] * channels).ravel()
            residuals.append(rows * (op @ motion).ravel())
        if with_jacobian:
            matrices.extend(self._temporal_matrices(self.correction_temporal_scale))
        if paths is not None:
            rows, blocks = self._world_path_rows(*paths, with_jacobian)
            residuals += rows
            matrices += blocks

        if coords.root_q is not None:
            rotations = Rotation.from_quat(poses[:, self.quat_slice], scalar_first=True)
            # Root path rows follow the source root path (see __init__).
            paths = [(np.asarray(self.tasks.roots), self.root_path_rotations, 1.0)]
            if self.continuity_root is not None:
                paths.append((*self.continuity_root, self.correction_temporal_scale))
            elif self.temporal_reference is not None:
                # Root corrections against the fixed reference stay as smooth
                # as the joint corrections: translation and orientation too.
                reference = Rotation.from_quat(self.reference[:, self.quat_slice], scalar_first=True)
                paths.append((self.root_reference, reference, self.correction_temporal_scale))
            for positions, frames_rotation, scale in paths:
                rows, blocks = self._root_path_rows(
                    rotations, positions, frames_rotation, scale, with_jacobian, root_point, root_jac
                )
                residuals += rows
                matrices += blocks
        residual = np.concatenate(residuals)
        return residual, (sparse.vstack(matrices, format="csr") if with_jacobian else None)

    def _temporal_matrices(self, scale, *, root=False):
        """Reuse weighted difference operators until verification changes weights.

        Kinematic Jacobians are still recomputed at every pose. Only the
        pose-independent left factors are cached, with in-place weight edits
        included in the key.
        """
        signature = tuple(weight.tobytes() for weights in (self.temporal_weight, self.joint_temporal_weight,
                           self.root_translation_weight, self.root_rotation_weight) for weight in weights)
        if signature != getattr(self, "_temporal_signature", None):
            self._temporal_signature = signature
            self._temporal_cache = {}
        key = (root, float(scale))
        if key not in self._temporal_cache:
            if root:
                if not hasattr(self, "_root_temporal_operators"):
                    self._root_temporal_operators = [
                        (sparse.kron(op, sparse.eye(size), format="csr") if size > 1 else op, size, i, channel)
                        for ops, size, channel in ((self.xy_ops, 2, 'translation'), (self.height_ops, 1, 'translation'),
                                                   (self.angular_ops, 4, 'rotation'))
                        for i, op in enumerate(ops)
                    ]
                operators = self._root_temporal_operators
            else:
                operators = [(matrix, len(self.coords.qi), i, 'joints') for i, matrix in enumerate(self.joint_matrices)]
            matrices = []
            for matrix, size, i, channel in operators:
                weight = self.temporal_weight[i] * scale
                if channel == 'joints':
                    rows = (weight[:, None] * self.joint_temporal_weight[i]).ravel()
                else:
                    channels = self.root_translation_weight if channel == 'translation' else self.root_rotation_weight
                    rows = np.repeat(weight * channels[i], size)
                matrices.append(sparse.diags(rows) @ matrix)
            self._temporal_cache[key] = matrices
        return self._temporal_cache[key]

    def _world_path_rows(self, points, jacobians, with_jacobian):
        """Derivative rows of world-tracked effector offsets from their paths."""
        residuals, matrices = [], []
        for index, target, masks in self.world_paths:
            offset = points[:, index] - target
            block = _block_rows(jacobians[:, index], self.width) if with_jacobian else None
            for op, weight, mask in zip(self.path_ops, self.temporal_weight, masks):
                rows = np.repeat(weight * mask, 3)
                residuals.append(rows * (op @ offset).ravel())
                if with_jacobian:
                    left = sparse.diags(rows) @ sparse.kron(op, sparse.eye(3), format="csr")
                    matrices.append(left @ block)
        return residuals, matrices

    def _root_path_rows(self, rotations, positions, path_rotations, scale, with_jacobian, root_point, root_jac):
        """Derivative rows of root offsets from a path (positions, rotations)."""
        weights = [w * scale * c for w, c in zip(self.temporal_weight, self.root_translation_weight)]
        residuals = [np.repeat(w, 2) * (op @ (root_point[:, :2] - positions[:, :2])).ravel() for op, w in zip(self.xy_ops, weights)]
        residuals += [w * (op @ (root_point[:, 2] - positions[:, 2])) for op, w in zip(self.height_ops, weights)]
        relative = (path_rotations.inv() * rotations).as_quat(scalar_first=True, canonical=True)
        weights = [w * scale * c for w, c in zip(self.temporal_weight, self.root_rotation_weight)]
        residuals += [np.repeat(w, 4) * (op @ relative).ravel() for op, w in zip(self.angular_ops, weights)]
        matrices = []
        if with_jacobian:
            xy = _block_rows(root_jac[:, :2], self.width)
            z = _block_rows(root_jac[:, 2:3], self.width)
            w, x, y, zq = relative.T
            basis = 0.5 * np.stack(
                [
                    np.stack([-x, -y, -zq], -1),
                    np.stack([w, -zq, y], -1),
                    np.stack([zq, w, -x], -1),
                    np.stack([-y, x, w], -1),
                ],
                1,
            )
            blocks = np.zeros((self.frames, 4, self.width))
            if not self.cfg["lock_root_orientation"]:
                for axis, column in enumerate(self.angular_cols):
                    if column is not None:
                        blocks[:, :, column] = basis[:, :, axis]
            angular = _block_rows(blocks, self.width)
            blocks = ([xy] * len(self.xy_ops) + [z] * len(self.height_ops)
                      + [angular] * len(self.angular_ops))
            matrices += [left @ block for left, block in zip(self._temporal_matrices(scale, root=True), blocks)]
        return residuals, matrices
