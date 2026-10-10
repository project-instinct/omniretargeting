"""Small fixed-root joint corrections using original moving-mesh witnesses.

Open meshes need no invented inside/outside: original object samples inside
robot collision hulls supply reverse intersection rows. Unknown regions stay
reported, and zero sampled depth is never a collision-free certificate.
"""

import os
import time
import hashlib
from contextlib import closing
from types import MappingProxyType

import mujoco
import numpy as np
from scipy import sparse

from ..contact_phase import CONTACT_ROLES
from ..geometry_query import GeometryQueryError
from ..moving_scene import MovingSceneQuery
from .original_rigid_fallback import (
    WITNESS_CLEARANCE, enabled as rigid_fallback_enabled,
    query_summary, resolve_unknown_static, rigid_robot_data,
)
from .collision_manifold import select_collision_manifold
from .repair_limits import LOCAL_ANGLE, RepairLimits
from .limb_scope import scope_for_body
from .repair_step import solve_step
from .repair_schedule import RepairSchedule
from .repair_frames import halo_frames, restrict
from .repair_restoration import restore_repair as restore_repair
from .repair_candidates import candidate_steps as candidate_steps
from .repair_frames import group_frames as group_frames
from .original_rigid_fallback import record_example as record_example
from .repair_acceptance import measure_candidate

ITERATIONS = 12
# Small progress ends shallow groups. Deep groups defer until relevant joints
# change or an acceptance query discovers a new constraint at the current pose.
MIN_GROUP_PROGRESS_M = 1e-4
DEEP_RESIDUAL_M = .01
# Rounding allowance of the rigid displacement bound (metres).
BOUND_SLACK = 1e-9
COLLISION_SIGMA = 0.01
LIMB_COLLISION_SIGMA = 0.001
REGRESSION = 1e-6
# A key that already penetrates may exceed its best achieved depth by this
# much (0 disables it), so a step deepening one frame by micrometres while
# lowering the group's worst depth is not rejected. Keys without penetration
# keep ``REGRESSION``: no new penetration beyond it.
PENETRATING_REGRESSION_M = 5e-5
# Witnesses kept per collision key when ``OriginalGeometry.witness_band`` is
# set (local repair): every reliable sample within the band of the deepest.
WITNESS_BAND_CAP = 48


def _activation_depth():
    # A sub-millimetre geom can exceed the external 0.5 mm regression gate.
    # This changes group eligibility, not geometry evidence or guard slack.
    return .001


def _regressed(value, protected):
    """Deeper than the best achieved depth by more than the allowance."""
    allowance = REGRESSION + (PENETRATING_REGRESSION_M if protected > 0 else 0.)
    return value > protected + allowance


def _group_rank(objective, body):
    """Use the acquisition policy's evidence; never infer support from a name."""
    policy = getattr(objective.tasks, "contact_phase", None)
    if policy is None:
        return len(CONTACT_ROLES)
    rank = len(CONTACT_ROLES)
    weld = objective.model.body_weldid
    for effector in objective.effectors:
        if weld[effector["body"]] != body:
            continue
        for frame in np.flatnonzero(effector["active"] | effector["managed"]):
            role = policy.role(effector["name"], int(frame))
            if role in CONTACT_ROLES:
                rank = min(rank, CONTACT_ROLES.index(role))
    for surface in objective.surfaces:
        if surface["task"] is not getattr(policy, "surface", None):
            continue
        columns = weld[surface["bodies"]] == body
        for index, role in enumerate(CONTACT_ROLES):
            mask = getattr(policy, "surface_roles", {}).get(role)
            if mask is not None and np.any(np.asarray(mask)[:, columns]):
                rank = min(rank, index)
    return rank


class OriginalGeometry:
    # Amortize closed-mesh winding construction without retaining all trajectory
    # samples. Query arrays are bounded; a very dense single frame is chunked.
    SCENE_BATCH_POINTS = 1 << 20

    def __init__(self, objective, scene_objects, scene_query, *, manifold=False,
                 columns=None, reference_poses=None):
        from .deadline_state import check
        check("repair_geometry:start")
        self.objective, self.model = objective, objective.model
        self.frames = objective.frames
        selected = np.asarray(objective.coords.di if columns is None else columns)
        if (selected.ndim != 1 or (selected.size and not np.issubdtype(selected.dtype, np.integer))
                or np.any(selected < 0) or np.any(selected >= self.model.nv)
                or len(np.unique(selected)) != len(selected)):
            raise ValueError("Geometry columns must be unique integer MuJoCo tangent columns")
        self.columns = np.array(selected, dtype=int, copy=True)
        self.columns.setflags(write=False)
        # ``di`` remains the legacy repair/consumer alias, not qpos indices.
        self.qi, self.di = objective.coords.qi, self.columns
        self.width = len(self.columns)
        self._moving_source = tuple(id(spec) for spec in scene_objects)
        provider = None
        reusable = getattr(provider, "geometry", None)
        reuse = False
        if reuse:
            self.moving = None if reusable.moving is None else reusable.moving.fresh_state()
            self.self_query = reusable.self_query.fresh_state()
        else:
            self.moving = MovingSceneQuery(self.model, scene_objects, objective.frames) if scene_objects else None
            from .repair_geometry_setup import self_query
            self.self_query = self_query(objective)
        check("repair_geometry:model_queries")
        # Per-frame exact pair results of the last query (see ``query``'s reuse).
        self._self_reuse = {}
        self._window_frame_cache = {}
        self._rigid_static_backend = None
        self._rigid_static_cache = {}
        self.static_query_coverage = {
            "sample_unknown": 0, "rigid_resolved": 0, "rigid_unknown": 0,
            "rigid_rows": 0,
        }
        self.static_query_examples = []
        self._query_owner = os.getpid()
        self.scene = scene_query
        self.data = mujoco.MjData(self.model)
        # Scene geometry/evidence and objective shell samples are immutable for
        # this object's lifetime. Cache only exact transforms and final witnesses.
        self._static_cache = {}
        self._moving_evidence_baseline = {}
        self._reference_observations_frozen = False
        self._reference_pose_key = None
        self.reference_observation_id = None
        self.reference_observation_failures = MappingProxyType({})
        self.manifold = manifold
        # Signed distance to closed, winding-signed components is 1-Lipschitz
        # and certain away from the surface. A geom whose exact clearance
        # exceeds the witness band by more than its rigid sample displacement
        # therefore yields the same summary (no row, known) without a query.
        # Open or evidence-dependent components are always requeried.
        scene = scene_query
        self._static_bound_reuse = bool(
            scene is not None and getattr(scene, "closed_sign_method", None) == "winding"
            and getattr(scene, "components", None)
            and all(part[2] for part in scene.components)
            and getattr(scene, "has_unsigned_only_components", True) is False)
        self._static_radii = {}
        if self._static_bound_reuse:
            ids = objective.shell_sample_geom
            norms = np.linalg.norm(objective.shell_local, axis=1)
            self._static_radii = {int(g): float(norms[ids == g].max(initial=0.)) for g in objective.shell_geoms}
        if self.scene is not None:
            from .repair_static_query import prepare
            prepare(self)
        check("repair_geometry:static_preparation")
        from .repair_geometry_setup import prepare_moving
        if reuse:
            # Prepare the immutable provider, then share its object list.
            # The provider's original dictionaries are never mutated.
            prepare_moving(reusable)
            self.moving = None if reusable.moving is None else reusable.moving.fresh_state()
        else:
            prepare_moving(self)
        check("repair_geometry:moving_preparation")
        self.static_reused = 0
        self._window_parent_mode = bool(False)
        # CUDA belongs to a spawned service; neither the solver process nor
        # its forked measurement workers initialise a CUDA context.
        self._gpu_owner = os.getpid()
        self._gpu_allowed = True
        if self._gpu_allowed:
            try:
                from .gpu_geometry import initialize_static
                self._gpu_allowed = initialize_static(self)
            except (ImportError, RuntimeError, ValueError, OSError, EOFError) as error:
                self._gpu_allowed = False
                self.gpu_unavailable = str(error)
        check("repair_geometry:gpu_preparation")
        if reference_poses is not None:
            self.freeze_reference_observations(reference_poses)

    def _install_reference_observations(self, registry, failures, pose_key, evidence_id):
        """Install immutable reference requirements in the parent or a worker."""
        self._moving_evidence_baseline = MappingProxyType(
            {int(frame): frozenset(samples) for frame, samples in registry.items()})
        self.reference_observation_failures = MappingProxyType(
            {int(frame): tuple(items) for frame, items in failures.items()})
        self._reference_pose_key = pose_key
        self.reference_observation_id = evidence_id
        self._reference_observations_frozen = True

    def freeze_reference_observations(self, reference_poses):
        """Freeze source-side requirements once, before accepted/trial queries.

        Legacy repair instances retain first-measurement seeding unless this
        opt-in method (or ``reference_poses``) is used. Calling again with the
        same exact reference is harmless; a different reference is rejected.
        Existing idle workers receive the same registry; restarted workers
        inherit it. No candidate can seed missing entries or rebase evidence.
        """
        poses = np.asarray(reference_poses, dtype=np.float64)
        if poses.shape != (self.frames, self.model.nq) or not np.isfinite(poses).all():
            raise ValueError("Reference poses must be finite (frames, model.nq)")
        pose_key = (poses.shape, poses.dtype.str, poses.tobytes())
        if self._reference_observations_frozen:
            if pose_key != self._reference_pose_key:
                raise ValueError("Original geometry reference observations cannot be rebased")
            return self.reference_observation_id
        parallel = getattr(self, "parallel", None)
        if parallel is not None and (parallel.prefetched or parallel.static_sender is not None):
            raise RuntimeError("Freeze observations only while geometry workers are idle")
        registry, failures = {}, {}
        for frame, pose in enumerate(poses):
            measured = {} if self.moving is None else self.moving.query(pose, frame, [])
            registry[frame] = frozenset(measured.get("open_penetrating_samples", ()))
            if measured.get("failures"):
                failures[frame] = tuple(
                    f"unknown reference moving frame {frame}: {item}" for item in measured["failures"])
        digest = hashlib.sha256(pose_key[2])
        digest.update(repr([(frame, sorted(samples)) for frame, samples in registry.items()]).encode())
        digest.update(repr(sorted(failures.items())).encode())
        evidence_id = digest.hexdigest()
        self._install_reference_observations(registry, failures, pose_key, evidence_id)
        if parallel is not None:
            try:
                parallel.sync_reference_observations()
            except Exception:
                self.stop_workers()  # No partly synchronized worker pool survives.
                raise
        return evidence_id

    def _prepare_gpu_static(self, poses, *, frames=None):
        if not self._gpu_allowed or os.getpid() != self._gpu_owner:
            return None
        try:
            from .gpu_geometry import prepare_static
            return prepare_static(self, poses, frames=frames)
        except GeometryQueryError:
            raise
        except (ImportError, RuntimeError, ValueError, OSError, EOFError) as error:
            # Missing Warp/CUDA and unsupported geometry retain the exact CPU
            # path. Debug assertions and original GeometryQueryError propagate.
            self._gpu_allowed = False
            self.gpu_unavailable = str(error)
            return None

    def _prepare_gpu_moving(self, poses, *, frames=None):
        return None

    def _static_frames(self, poses, frames=None, *, _gpu_ready=False):
        """Batch changed geoms; reuse exact-transform witnesses, never Jacobians."""
        objective, data = self.objective, self.data
        ids = objective.shell_sample_geom
        groups = [(int(geom), np.flatnonzero(ids == geom)) for geom in objective.shell_geoms]
        limit = max(1, int(self.SCENE_BATCH_POINTS))
        summaries_by_frame, updates, pending, count = [], [], [], 0
        for frame in range(len(poses)) if frames is None else frames:
            pose = poses[frame]
            data.qpos[:] = pose
            mujoco.mj_forward(self.model, data)
            summaries = []
            for geom, selected in groups:
                # Byte comparison intentionally distinguishes even signed zero;
                # no coordinate rounding or approximate cache acceptance.
                signature = (data.geom_xpos[geom].tobytes(), data.geom_xmat[geom].tobytes())
                cached = self._static_cache.get((frame, geom))
                if cached is not None and cached[0] == signature:
                    summary = cached[1]
                    if summary["unknown"]:
                        updates.append(((frame, geom), signature, summary,
                                        data.geom_xpos[geom].copy(), data.geom_xmat[geom].copy(),
                                        rigid_robot_data(data)))
                elif cached is not None and self._static_bound_reuse and self._clear_after_motion(cached, geom):
                    # The cache keeps the exactly queried transform and bound.
                    summary = cached[1]
                    self.static_reused += 1
                else:
                    summary = {"row": None, "unknown": False, "clearance": np.inf,
                               "rows": [], "origin": data.geom_xpos[geom].copy(), "_screen_band": .02}
                    if (getattr(self, "_fork_queries_prepared", False)
                            or getattr(self, "_repair_static_bounds", False)):
                        from .original_geometry_cpu import screened_static_samples
                        from .original_geometry_bounds import far_geom
                        selected = (np.empty(0, int) if far_geom(self, geom, summary) else
                                    screened_static_samples(self, geom, selected, cached, summary))
                    updates.append(((frame, geom), signature, summary,
                                    data.geom_xpos[geom].copy(), data.geom_xmat[geom].copy(),
                                        rigid_robot_data(data)))
                    first = 0
                    while first < len(selected):
                        indices = selected[first:first + limit - count]
                        points = data.geom_xpos[ids[indices]] + np.einsum(
                            "nij,nj->ni", data.geom_xmat[ids[indices]].reshape(-1, 3, 3),
                            objective.shell_local[indices])
                        first += len(indices)
                        if (getattr(self, "_fork_queries_prepared", False)
                                or getattr(self, "_repair_static_bounds", False)):
                            from .original_geometry_bounds import screen_points
                            points, indices = screen_points(self, summary, points, indices)
                        if len(indices):
                            pending.append((summary, points, indices))
                        count += len(indices)
                        if count == limit:
                            self._query_static_batch(pending)
                            pending, count = [], 0
                summaries.append((geom, summary))
            summaries_by_frame.append(summaries)
        if pending:
            self._query_static_batch(pending)
        # Never commit a partially queried summary if SceneQuery raises.
        for key, signature, summary, position, rotation, robot_data in updates:
            if summary["unknown"] and rigid_fallback_enabled(self.scene):
                resolve_unknown_static(self, key[0], key[1], summary, signature,
                                       robot_data, WITNESS_CLEARANCE)
            if summary.pop("_screen_sort", False):
                order = np.argsort(summary["_screen_ids"], kind="stable")
                summary["_screen_ids"] = summary["_screen_ids"][order]
                summary["_screen_phi"] = summary["_screen_phi"][order]
            summary.pop("_reuse_sign_ids", None)
            summary.pop("_reuse_sign_phi", None)
            self._static_cache[key] = (signature, summary, position, rotation)
        self._static_cache = {key: value for key, value in self._static_cache.items()
                              if key[0] < len(poses)}
        yield from summaries_by_frame

    def _clear_after_motion(self, cached, geom):
        """Exact queried clearance minus the geom's largest sample displacement."""
        summary = cached[1]
        if (summary["unknown"] or summary["row"] is not None or summary["rows"]
                or not np.isfinite(summary["clearance"])):
            return False
        data = self.data
        shift = np.linalg.norm(data.geom_xpos[geom] - cached[2])
        turn = np.linalg.norm(data.geom_xmat[geom].reshape(3, 3) - cached[3].reshape(3, 3), 2)
        moved = shift + turn * self._static_radii[geom]
        return summary["clearance"] - moved >= WITNESS_CLEARANCE + BOUND_SLACK

    def _query_static_batch(self, pending):
        """Reduce every original sample to the same first minimum and unknown bit.

        Input and SceneQuery arrays contain at most SCENE_BATCH_POINTS points;
        only O(frames * geoms) small summaries survive the query.
        """
        points = np.concatenate([points for _, points, _ in pending])
        if hasattr(self.scene, "query_with_sign_hints"):
            hints = []
            for summary, part_points, ids in pending:
                previous = summary.get("_reuse_sign_phi")
                hints.append(np.full(len(ids), np.nan) if previous is None else
                             previous[np.searchsorted(summary["_reuse_sign_ids"], ids)])
            measured = self.scene.query_with_sign_hints(points, np.concatenate(hints))
        else:
            from .repair_static_query import query_scene
            measured = query_scene(self, points)
        first = 0
        for summary, group_points, ids in pending:
            last = first + len(group_points)
            value = measured["clearance"][first:last]
            valid = np.isfinite(value) & ~measured["uncertain"][first:last]
            # A material-sample mask may omit only points proven beyond the
            # witness band after their next rigid displacement. Unknown points
            # always remain candidates; the mask contains no reported values.
            if self._static_bound_reuse:
                band = summary.get("_screen_band", .02)
                keep = ids[(value < band) | ~valid]
                summary["_screen_ids"] = np.r_[summary.get("_screen_ids", np.empty(0, dtype=int)), keep]
                kept_phi = np.where(valid, value, np.nan)[(value < band) | ~valid]
                summary["_screen_phi"] = np.r_[summary.get("_screen_phi", np.empty(0)), kept_phi]
            summary["unknown"] |= not valid.all()
            if valid.any():
                index = first + int(np.argmin(np.where(valid, value, np.inf)))
                phi = float(measured["clearance"][index])
                # Strict comparison preserves the first sample on exact ties,
                # including when one geometry straddles a query boundary.
                if phi < summary["clearance"]:
                    summary["clearance"] = phi
                    summary["row"] = (None if phi >= 0.008 else {
                        "clearance": phi, "sample_id": int(ids[index - first]),
                        "direction": measured["directions"][index].copy(),
                        "point": points[index].copy()})
            if self.manifold:
                # Merge fixed support probes, rather than averaging normals or
                # keeping only one corner. At most 15 witnesses survive per geom.
                old = summary["rows"]
                directions = measured["directions"][first:last]
                reliable = (valid & np.isfinite(directions).all(axis=1)
                            & np.any(directions != 0, axis=1))
                summary["unknown"] |= bool(np.any(valid & (value < .008) & ~reliable))
                eligible = reliable & (value < .008)
                p = np.concatenate([np.array([r["point"] for r in old]).reshape(-1, 3), group_points[eligible]])
                d = np.concatenate([np.array([r["direction"] for r in old]).reshape(-1, 3), directions[eligible]])
                v = np.r_[[r["clearance"] for r in old], value[eligible]]
                samples = np.r_[[r["sample_id"] for r in old], ids[eligible]].astype(int)
                selected = select_collision_manifold(p, v, d, origin=summary["origin"])
                band = getattr(self, "witness_band", None)
                if band is not None and len(v):
                    # Local repair models every sample that a small step can
                    # make the deepest, not only the support probes.
                    near = np.flatnonzero(v <= v.min() + band)
                    near = near[np.argsort(v[near], kind="stable")[:WITNESS_BAND_CAP]]
                    selected = np.union1d(selected, near)
                    selected = selected[np.unique(samples[selected], return_index=True)[1]]
                summary["rows"] = [{"point": p[k].copy(), "direction": d[k].copy(),
                                    "sample_id": int(samples[k]), "clearance": float(v[k])} for k in selected]
            first = last

    def start_workers(self, frames):
        """Fork frame-parallel measurement workers (see ``parallel_geometry``)."""
        if getattr(self, "_fork_cpu_only", False):
            raise RuntimeError("Private window geometry must not start worker pools")
        if os.getpid() != self._query_owner:
            raise RuntimeError("Use fork_private_client before starting child geometry workers")
        from .parallel_geometry import ParallelMeasure

        parallel = ParallelMeasure(self, frames)
        self.parallel = parallel if parallel.workers else None

    def stop_workers(self):
        if os.getpid() != self._query_owner:
            raise RuntimeError("A child must not close inherited geometry workers")
        if getattr(self, "parallel", None) is not None:
            self.parallel.close()
            self.parallel = None

    def fork_private_client(self):
        """Independent query buffers/caches, immutable inherited reference evidence."""
        from .original_geometry_window import fork_private_client
        return fork_private_client(self)

    def prepare_fork_queries(self):
        """Parent-only, idempotent preparation of resident serial CPU queries."""
        from .original_geometry_cpu import prepare_fork_queries
        return prepare_fork_queries(self)

    def reevaluate_witnesses(self, poses, measurement, *, frames=None):
        """Exact nonlinear clearances aligned with the frozen witness rows."""
        from .original_geometry_witnesses import reevaluate_witnesses
        return reevaluate_witnesses(self, poses, measurement, frames=frames)

    def relinearize_witnesses(self, origin_poses, trial_poses, trial_measurement, *, frames, keys=None):
        """Origin rows for selected trial identities, without an activation filter."""
        from .original_geometry_relinearize import relinearize_witnesses
        return relinearize_witnesses(self, origin_poses, trial_poses, trial_measurement,
                                    frames=frames, keys=keys)

    def max_depth_per_body(self, poses, frames):
        """Exact cached acceptance depths, without Jacobian assembly."""
        measured = self.measure(poses, frames=frames)
        names = ("max_depth", "body_reliable", "queried_frames", "frame_complete",
                 "reliable", "complete", "depths", "failures", "evidence_id")
        return {name: measured[name] for name in names}

    def measure(self, poses, *, frames=None, jacobians=False, protected_depths=None,
                activation_distance=None):
        """Full acceptance queries; an already rejected candidate may stop early.

        Frames are measured independently (in forked workers once started) and
        merged in frame order, cut at the first frame the candidate is rejected.
        """
        from .parallel_geometry import merge_frames
        from .original_geometry_window import measure_window, remember_frames

        if os.getpid() != self._query_owner:
            raise RuntimeError("Use fork_private_client before querying inherited geometry")
        if frames is not None or activation_distance is not None:
            return measure_window(self, poses, frames, jacobians, protected_depths,
                                  activation_distance)

        parallel = getattr(self, "parallel", None)
        if (parallel is not None and len(poses) == parallel.frames
                and self._gpu_allowed and protected_depths is None):
            parallel.prefetch_self(poses)
        static_updates = self._prepare_gpu_static(poses)
        moving_updates = self._prepare_gpu_moving(poses)
        if parallel is not None and len(poses) == parallel.frames:
            parts = parallel.measure(poses, jacobians, protected_depths,
                                     static_updates=static_updates, moving_updates=moving_updates)
            if self.moving is not None:
                self.moving.query_count = parallel.query_count
        else:
            parts = [self.measure_frames(poses, range(len(poses)), jacobians=jacobians,
                                         protected_depths=protected_depths)]
        remember_frames(self, poses, parts)
        return merge_frames(parts)

    def measure_frames(self, poses, frames, *, jacobians=False, protected_depths=None, stop=None):
        """Per-frame rows, depths and failures of ``frames``, in that order.

        With ``protected_depths`` (a candidate), stop after the first frame
        that has a failure or a protected-depth regression. ``stop`` is the
        parallel workers' shared rejection flag (see ``parallel_geometry``).
        """
        results, rejection = [], None
        static_frames = None if self.scene is None else self._static_frames(poses, frames)

        def rejects(depths, failures):
            return protected_depths is not None and bool(failures or any(
                _regressed(value, protected_depths.get(key, 0.)) for key, value in depths.items()))

        from .deadline_state import check
        for frame in frames:
            check("repair_measure_frame")
            if stop is not None and stop.value >= 0:
                break
            pose = poses[frame]
            rows, depths, failures = [], {}, []
            if self.moving is not None:
                measured = self.moving.query(pose, frame, self.di if jacobians else [])
                failures.extend(measured["failures"])
                if self._reference_observations_frozen:
                    required = self._moving_evidence_baseline.get(frame, frozenset())
                    if frame not in self._moving_evidence_baseline:
                        failures.append(f"missing frozen source-side registry at frame {frame}")
                    failures.extend(self.reference_observation_failures.get(frame, ()))
                else:
                    required = self._moving_evidence_baseline.setdefault(
                        frame, frozenset(measured.get("open_penetrating_samples", ())))
                missing = required.difference(measured.get("open_known_samples", ()))
                certify = getattr(self.moving, "certified_outside", None)
                outside = set() if certify is None else certify(pose, frame, missing)
                if not missing.issubset(outside):
                    failures.append(f"lost original source-side observation at frame {frame}")
                    if getattr(self, "uncertain_exit_rows", False):
                        from .uncertain_exit import material_rows
                        measured["rows"].extend(material_rows(
                            self.moving, pose, frame, missing - outside,
                            self.di if jacobians else []))
                for row in measured["rows"]:
                    # Closed, observed-open and reverse witnesses protect
                    # different overlaps. One larger depth cannot conceal a
                    # regression of another representation or component.
                    key = (frame, "moving", row["object_id"], row["kind"],
                           row.get("component_id", -1), row["geom_id"])
                    self._record(rows, depths, key, row)
                if rejects(depths, failures):
                    # Already rejected: the frame's self/static queries cannot
                    # change that; its rows stay partial diagnostics.
                    results.append((frame, rows, depths, failures))
                    rejection = frame
                    if stop is not None:
                        stop.value = frame
                    break
            try:
                prepared_self = getattr(self, "_fork_self_frames", {}).get(frame)
                if prepared_self is not None and prepared_self[0] == np.asarray(pose).tobytes():
                    self_rows = [dict(row) for row in prepared_self[1]]
                    if len(prepared_self) > 2:
                        failures.extend(prepared_self[2])
                    if jacobians:
                        self.self_query.data.qpos[:] = pose
                        mujoco.mj_forward(self.model, self.self_query.data)
                else:
                    self_rows = self.self_query.query(pose, reuse=self._self_reuse.setdefault(frame, {}))
            except GeometryQueryError as error:
                # Indeterminate geometry cannot certify either the entry or a
                # candidate. Keep the failure visible and reject that candidate.
                failures.append(f"unknown self frame {frame}: {error}")
                self_rows = []
            for row in self_rows:
                if jacobians:
                    row["jacobian"] = self.self_query.jacobian(row, self.self_query.data)[self.di]
                key = (frame, "self", *row["pair"])
                self._record(rows, depths, key, row)
            if static_frames is not None:
                data = self.data
                summaries = next(static_frames)
                if jacobians:
                    # The batch prepass leaves data at its last pose; each witness
                    # Jacobian still belongs to this exact frame.
                    data.qpos[:] = pose
                    mujoco.mj_forward(self.model, data)
                for geom, summary in summaries:
                    if summary["unknown"]:
                        failures.append(f"unknown static frame {frame} geom {geom}")
                        failures.extend(f"rigid backend frame {frame} geom {geom}: {value}"
                                        for value in summary.get("query_failures", ()))
                    witnesses = summary["rows"] if self.manifold else ([] if summary["row"] is None else [summary["row"]])
                    for witness in witnesses:
                        row = {"clearance": witness["clearance"], "direction": witness["direction"].copy(),
                               "point": witness["point"].copy(), "geom_id": geom, "sample_id": witness["sample_id"]}
                        if jacobians:
                            jp = np.zeros((3, self.model.nv))
                            mujoco.mj_jac(self.model, data, jp, None, row["point"], int(self.model.geom_bodyid[geom]))
                            row["jacobian"] = row["direction"] @ jp[:, self.di]
                        self._record(rows, depths, (frame, "static", int(geom)), row)
            results.append((frame, rows, depths, failures))
            if rejects(depths, failures):
                rejection = frame
                if stop is not None:
                    stop.value = frame
                break
        return {"frames": results, "rejection": rejection}

    def linearize(self, poses, measurement):
        """Reuse exactly checked witnesses; only their joint Jacobians change."""
        from .repair_linearize import linearize
        return linearize(self, poses, measurement)

    @staticmethod
    def _record(rows, depths, key, row):
        depth = max(0., -row["clearance"])
        depths[key] = max(depths.get(key, 0.), depth)
        rows.append((key, row))

    def summary(self, measurement):
        result = {}
        for kind in ("static", "moving", "self"):
            subset = {k: v for k, v in measurement["depths"].items() if k[1] == kind}
            key = max(subset, key=subset.get, default=None)
            value = 0. if key is None else subset[key]
            result[kind] = {"max_depth_m": value,
                            "affected_frames": sorted({k[0] for k, v in subset.items() if v > 0.001})}
            if key is not None:
                geoms = key[2:] if kind == "self" else key[-1:]
                result[kind].update(frame=int(key[0]), bodies=[self.model.body(int(self.model.geom_bodyid[g])).name for g in geoms])
                witness = min((row for row_key, row in measurement["rows"] if row_key == key),
                              key=lambda row: row["clearance"])
                exit_vector = value * np.asarray(witness["direction"])
                result[kind].update(exit_xy_m=float(np.linalg.norm(exit_vector[:2])),
                                    exit_z_m=float(exit_vector[2]))
        from .repair_diagnostics import deep_residuals
        result["residuals_over_1cm"] = deep_residuals(self.model, measurement)
        result["query_failures"] = len(measurement["failures"])
        if self.scene is not None:
            result.update(query_summary(self))
        result["unknown_object_components"] = 0 if self.moving is None else self.moving.unknown_components
        result["collision_free_certificate"] = False
        if self.moving is not None:
            result["moving_query"] = self.moving.describe()
        return result

    def linear_rows(self, measurement, *, frames=None):
        rows = measurement["rows"]
        start, count = 0, self.objective.frames
        if frames is not None:
            from .original_geometry_window import validate_frames
            frames = validate_frames(self, frames)
            start, count = frames.start, len(frames)
            rows = [(key, row) for key, row in rows if key[0] in frames]
        frame = np.array([key[0] - start for key, _ in rows], dtype=int)
        jac = np.array([row["jacobian"] for _, row in rows]).reshape(len(rows), self.width)
        matrix = sparse.csr_matrix((jac.ravel(),
                                   (np.repeat(np.arange(len(rows)), self.width),
                                    (frame[:, None] * self.width + np.arange(self.width)).ravel())),
                                  shape=(len(rows), count * self.width))
        phi = np.array([row["clearance"] for _, row in rows])
        return matrix, phi


def _chain_columns(limits, model, body):
    """Affected ancestors plus distal joints that can compensate their motion."""
    ancestors = set()
    parent = body
    while parent > 0:
        ancestors.add(parent)
        parent = int(model.body_parentid[parent])
    joints = model.jnt_bodyid[model.dof_jntid[limits.coords.di]]
    if not np.isin(joints, list(ancestors)).any():
        return set()  # A floating-base collision cannot invent local base DOFs.
    relevant = set(ancestors)
    for descendant in range(1, model.nbody):
        parent = descendant
        while parent > 0 and parent != body:
            parent = int(model.body_parentid[parent])
        if parent == body:
            relevant.add(descendant)
    return set(np.flatnonzero(np.isin(joints, list(relevant))))


def _joint_projection(limits, model, measurement=None):
    """Couple collision partners and the joints preserving affected contacts.

    This changes the variable set only. Root, contact, source-shape, cumulative
    pose and temporal bounds are still supplied by the same immutable limits.
    """
    if limits.movable_body is None:
        columns = set(range(limits.width))
    else:
        bodies = {limits.movable_body}
        if measurement is not None:
            for key, depth in measurement["depths"].items():
                if key[1] != "self" or depth <= _activation_depth():
                    continue
                pair = {int(model.body_weldid[model.geom_bodyid[g]]) for g in key[2:]}
                if limits.movable_body in pair:
                    bodies.update(pair)
        columns = set()
        for body in bodies:
            columns.update(_chain_columns(limits, model, body))
        if measurement is not None:
            # Shared ancestors can move contacts on a different branch. Add
            # their compensating chain until the contact-connected set closes.
            chains = []
            joints = model.jnt_bodyid[model.dof_jntid[limits.coords.di]]
            for _, contact_bodies, budget in limits.points:
                for index, body in enumerate(contact_bodies):
                    if not np.isfinite(budget[:, index]).any():
                        continue
                    ancestors = set()
                    while body > 0:
                        ancestors.add(int(body))
                        body = model.body_parentid[body]
                    chains.append(set(np.flatnonzero(np.isin(joints, list(ancestors)))))
            previous = -1
            while previous != len(columns):
                previous = len(columns)
                for chain in chains:
                    if columns & chain:
                        columns.update(chain)
    scope = scope_for_body(limits, limits.movable_body)
    if scope is not None:
        # Contact closure and self-collision partners cannot recruit the waist
        # or upper limbs to repair a foot. The common ancestor remains fixed.
        columns.intersection_update(scope.columns)
    selector = sparse.eye(limits.width, format="csc")[:, sorted(columns)]
    return sparse.kron(sparse.eye(limits.frames), selector, format="csc")


def _qp_step(limits, poses, geometry, measurement, baseline):
    return solve_step(limits, poses, geometry, measurement, baseline,
                      restrict(_joint_projection(limits, geometry.model, measurement),
                               getattr(limits, "frame_mask", None)),
                      collision_sigma=(LIMB_COLLISION_SIGMA if scope_for_body(limits, limits.movable_body)
                                       is not None else COLLISION_SIGMA),
                      local_angle=getattr(limits, 'local_angle', LOCAL_ANGLE))


def _repair_groups(objective, measurement, activation=None):
    """Include floor/object/self collisions, ranked by contact evidence."""
    groups = {}
    model = objective.model
    activation = _activation_depth() if activation is None else activation
    for key, depth in measurement["depths"].items():
        if depth < activation:
            continue
        geoms = key[2:] if key[1] == "self" else key[-1:]
        for geom in geoms:
            body = int(model.body_weldid[model.geom_bodyid[geom]])
            groups[body] = max(groups.get(body, 0.), depth)
    return sorted(groups, key=lambda body: (_group_rank(objective, body), -groups[body]))


def repair_original_meshes(objective, poses, scene_objects=(), scene_query=None):
    """Retain only small corrections satisfying immutable nonlinear limits."""
    started = time.monotonic()
    entry = np.array(poses, copy=True)
    if not len(objective.coords.qi):
        return entry, {"status": "no_joint_dofs", "seconds": time.monotonic() - started}
    limits = RepairLimits(objective, entry)
    geometry = OriginalGeometry(objective, scene_objects, scene_query, manifold=True)
    from .repair_geometry_setup import prepare_static_bounds, prepare_moving_reduction
    from .deadline_state import check
    check("repair_static_bounds")
    prepare_static_bounds(geometry, entry)
    check("repair_moving_reduction")
    prepare_moving_reduction(geometry, entry)
    check("repair_workers")
    # Frame-parallel measurement returns the serial rows; see parallel_geometry.
    geometry.start_workers(len(entry))
    try:
        return _repair(objective, entry, limits, geometry, started)
    finally:
        proposals = getattr(geometry, "_repair_proposals", None)
        if proposals is not None:
            proposals.close(abort=True)
        geometry.stop_workers()


def _repair(objective, entry, limits, geometry, started):
    from .deadline_state import check, accepted
    check("repair_initial_geometry")
    baseline = geometry.measure(entry, jacobians=True)
    current, measured = entry.copy(), baseline
    protected = {"depths": dict(baseline["depths"])}
    model = objective.model
    ordered = _repair_groups(objective, baseline)
    groups = ordered or [None]
    report = {"before": geometry.summary(baseline), "history": [],
              "entry_limits": limits.check(entry)[1],
              "local_angle_budget_rad": getattr(limits, 'local_angle', LOCAL_ANGLE), "root_fixed": True,
              "limb_scopes": [{"fixed_anchor": model.body(scope.anchor).name,
                               "movable_joints": [model.joint(int(model.dof_jntid[limits.coords.di[c]])).name
                                                  for c in sorted(scope.columns)]}
                              for scope in limits.limb_scopes],
              "scheduled_groups": [model.body(body).name for body in ordered],
              "deferred_groups": [], "iterations_per_group": ITERATIONS,
              "scope": "saved-frame original triangles/reverse samples; no collision-free certificate",
              "collision_objective": "per_group_maximum_depth", "static_manifold_max_witnesses": 15}
    if baseline["failures"]:
        report.update(status="unknown_query", after=report["before"], seconds=time.monotonic() - started)
        return current, report
    limits.freeze_penetration_yield(baseline)
    original_group_count = max(1, len(_repair_groups(objective, baseline, activation=.001)))
    total_limit = ITERATIONS * original_group_count
    report['total_attempt_budget'] = total_limit
    report['activation_depth_m'] = _activation_depth()
    schedule = RepairSchedule(groups, ITERATIONS, total_limit=total_limit)
    from .repair_proposals import RepairProposals, transition_group
    from .repair_parallel_groups import ParallelGroups as RepairProposals
    proposals = geometry._repair_proposals = RepairProposals(limits, geometry)
    iteration = -1
    while (scheduled := schedule.next()) is not None:
        check("repair_group_attempt")
        group, group_iteration = scheduled
        iteration += 1
        # Preserve baseline history numbering, including unused attempts
        # after an earlier group stopped. Scheduling changes are opt-in.
        iteration = groups.index(group) * ITERATIONS + group_iteration
        transition_group(limits, current, group)
        columns = limits.coords.qi[np.unique(_joint_projection(limits, model, measured).nonzero()[0] % limits.width)]
        scope = scope_for_body(limits, group)
        fixed_q = np.ones(current.shape[1], dtype=bool)
        if scope is not None:
            fixed_q[limits.coords.qi[sorted(scope.columns)]] = False
        def group_depths():
            for key, value in measured["depths"].items():
                geoms = key[2:] if key[1] == "self" else key[-1:]
                if group is None or any(model.body_weldid[model.geom_bodyid[g]] == group for g in geoms):
                    yield value
        entry_depth = max(group_depths(), default=0.)
        if entry_depth <= _activation_depth():
            continue
        tick = time.monotonic()
        # Only the group's penetrating frames and their halo move.
        from .repair_defects import frames as attempt_frames
        frame_mask = limits.frame_mask = attempt_frames(limits, measured, model, group,
                                                        padding=halo_frames(objective.framerate))
        subproblem = measured
        proposals.prefetch(schedule, current, measured, protected)
        proposal = proposals.take(current, subproblem, protected,
                                  fixed_q if scope is not None else None, entry_depth)
        status = proposal.status
        record = {"iteration": iteration, "group_iteration": group_iteration, "started": tick,
                  "solver_status": status, "alpha": 0.0,
                  "contact_group": None if group is None else model.body(group).name,
                  "limb_only": scope is not None,
                  "collision_sigma_m": LIMB_COLLISION_SIGMA if scope is not None else COLLISION_SIGMA}
        report["history"].append(record)
        prediction = getattr(proposal, "prediction", None)
        if prediction is not None:
            record["model_progress"] = dict(prediction)
        if not proposal.has_direction:
            proposal.close()
            proposals.note_outcome(False)
            continue
        if proposal.stationary:
            proposal.close()
            proposals.note_outcome(False)
            record["stationary"] = True
            record["stop_reason"] = "stationary_proposal"
            continue
        from .repair_prediction import should_gate
        if should_gate(prediction, MIN_GROUP_PROGRESS_M):
            proposal.close()
            proposals.note_outcome(False)
            record["model_progress"]["realized_decrease_m"] = 0.
            record["stop_reason"] = "predicted_small_progress"
            continue
        # A pair and its contact compensators form one coupled proposal.
        # Backtracking branches independently can destroy that compensation.
        before = sum(value**2 for value in measured["depths"].values())
        first_geometry_candidate = None
        new_witnesses = 0
        with closing(proposal.candidates(record)) as candidates:
            for alpha, candidate in candidates:
                check("repair_trial")
                if first_geometry_candidate is None:
                    first_geometry_candidate = candidate.copy()
                # Unmoved frames reuse their exact per-pose cache entries.
                checked, regressions, after, descent = measure_candidate(
                    objective, geometry, candidate, measured, protected, group, before,
                    frames=None if frame_mask is None else range(len(candidate)))
                if not checked.get("complete", True) or checked["failures"] or regressions or not descent:
                    if not checked.get("complete", True):
                        record["rejected_geometry_prefix_frame"] = checked.get("rejection_frame")
                    record["query_failures"] = checked["failures"][:3]
                    record["collision_regressions"] = len(regressions)
                    record["regression_examples"] = [{
                        "key": list(key),
                        "limit_m": protected["depths"].get(key, 0.),
                        "actual_m": checked["depths"][key],
                        "increase": checked["depths"][key] - protected["depths"].get(key, 0.),
                    } for key in regressions[:3]]
                    continue
                if prediction is not None:
                    from .repair_prediction import linear_prediction
                    candidate_prediction = linear_prediction(
                        geometry, subproblem, candidate[:, limits.coords.qi] - current[:, limits.coords.qi],
                        group, entry_depth)
                    if candidate_prediction is not None:
                        record["model_progress"]["predicted_candidate_decrease_m"] = candidate_prediction["predicted_decrease_m"]
                current = candidate
                accepted(current, "original_mesh_repair")
                measured = geometry.linearize(current, checked)
                for key in protected["depths"].keys() | measured["depths"].keys():
                    protected["depths"][key] = min(protected["depths"].get(key, 0.), measured["depths"].get(key, 0.))
                record.update(alpha=alpha, before=before, after=after)
                break
        proposals.note_outcome(bool(record["alpha"]))
        if prediction is not None:
            record["model_progress"]["realized_decrease_m"] = entry_depth - max(group_depths(), default=0.)
            record["model_progress"]["accepted_alpha"] = record["alpha"]
        small = (MIN_GROUP_PROGRESS_M > 0
                 and entry_depth - max(group_depths(), default=0.) < MIN_GROUP_PROGRESS_M)
        if not record["alpha"] or small:
            reason = "no_admissible_descent" if not record["alpha"] else "small_progress"
            record["stop_reason"] = reason
        else:
            schedule.retry(group)
        record["new_witnesses"] = new_witnesses
        if group_iteration == ITERATIONS - 1:
            record["stop_reason"] = "iteration_limit"
    proposals.close()
    report["proposal_workers"] = proposals.stats
    report["geometry_cache"] = dict(frames=getattr(geometry, "window_totals", {}),
                                     jacobians=getattr(geometry, "linearization_totals", {}))
    if getattr(geometry, '_repair_static_bounds', False):
        report['geometry_cache']['static_bounds'] = geometry.static_bounds_stats
    report["deferred_groups"] = []
    report["remembered_witnesses"] = 0
    # Wall time of each proposal: until the next one (or the end) started.
    ends = [r["started"] for r in report["history"][1:]] + [time.monotonic()]
    for record, end in zip(report["history"], ends):
        record["seconds"] = end - record.pop("started")
    report["iteration_limited_groups"] = [r["contact_group"] for r in report["history"]
                                           if r.get("stop_reason") == "iteration_limit"]
    certified = [r for r in report["history"]
                 if r.get("model_progress", {}).get("model") == "pre_qp_upper_bound"]
    report["triage"] = {"enabled": True, "pre_qp_skips": len(certified),
                        "groups": [{"body": r["contact_group"],
                                    "certificate": r["model_progress"]} for r in certified]}
    gated = [r for r in report["history"]
             if r["solver_status"] == "reduced_predicted_small_progress"]
    report["fast_gate"] = {"enabled": True, "pre_qp_skips": len(gated),
                           "iterations": [r["iteration"] for r in gated]}
    report.update(status="corrected" if np.any(current != entry) else "retained",
                  after=geometry.summary(measured), limits=limits.check(current)[1],
                  seconds=time.monotonic() - started)
    return current, report
