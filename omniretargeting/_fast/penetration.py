"""Penetration handling of the fast backend: supports first, then free bodies.

1. Supports (before contact acquisition). Where collision geoms rigidly
   attached to the floating base, or carrying a source surface region the base
   also carries, sit coherently inside an upward support (a seat, with several
   collision points), no joint can move them out alone:
   the fixed reference rises smoothly by the minimal amount, measured along
   the support normal to the plane through the contact targets and only for
   samples inside the actual object and over the contact patch (a thin seat
   is never crossed). Joint angles, and so the source-shaped posture, are
   unchanged; acquisition then keeps the feet and every other contact on
   their targets within its envelope (AGENTS: small root lift).
2. Free bodies (after acquisition, every contact held). Remaining collision
   geoms move out along their shortest exit; a sample past the medial axis
   of a thin part (whose shortest exit points through it) exits towards the
   side it entered from. The correction is kept only if verification finds
   no added chatter or posture excess; otherwise the residual is reported.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.spatial.transform import Rotation

# Support normals at most this far from vertical may lift the root (cos 45 deg).
UPWARD_COS = np.cos(np.pi / 4)
# "Many active collision points": penetrating samples on upward supports.
LIFT_MIN_POINTS = 3
# The reference lift ramps over this time, never switching on in one frame;
# the lifted reference defines the later continuity budgets.
LIFT_RAMP_SECONDS = 0.5
# Ray marching towards the entry side of a thin part.
EXIT_STEPS = 24
EXIT_MAX_M = 0.3


def _runs(mask):
    """(start, stop) of contiguous True runs of a 1-D mask."""
    padded = np.r_[False, mask, False].astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2], edges[1::2]))


def _worker_plane_clearance(span):
    from .objective import _WORKER
    return _WORKER["objective"]._plane_clearance(span)[0][span[0]:span[1]]


def _chunked_plane_clearance(objective):
    """``_plane_clearance()[0]``; long trajectories in temporary forked workers."""
    import multiprocessing
    import os
    from .objective import (CHUNKS_PER_WORKER, MIN_CHUNK_FRAMES, TEMPORARY_DEEPEST_FRAMES,
                            WORKERS, _temporary_pool)
    from .bulk_geometry import prepare
    prepare(objective)  # Warm grid kernels before temporary clearance workers.
    chunks = min(CHUNKS_PER_WORKER * WORKERS, objective.frames // MIN_CHUNK_FRAMES)
    if (chunks < 2 or objective.frames < TEMPORARY_DEEPEST_FRAMES or not hasattr(os, "fork")
            or multiprocessing.current_process().daemon):
        return objective._plane_clearance()[0]
    edges = np.linspace(0, objective.frames, chunks + 1).astype(int)
    spans = list(zip(edges[:-1], edges[1:]))
    with _temporary_pool(objective, min(WORKERS, chunks)) as pool:
        parts = pool.map(_worker_plane_clearance, spans, chunksize=1)
    return np.concatenate(parts)


class PenetrationStages:
    """Stage preparation and plane rows; mixed into ``FastObjective``."""

    # ----- shared -----------------------------------------------------------
    def _field_index(self, surface_ids):
        """Observation surface id (0 scene, k object k-1) -> clearance field."""
        ids = np.asarray(surface_ids)
        first = 1 if self.scene_sdf is not None else 0
        field = np.where(ids == 0, 0 if self.scene_sdf is not None else -1, first + ids - 1)
        return np.where((ids >= 0) & (field < first + len(self.objects)), field, -1)

    def _shell_points(self, frames, geoms):
        """World shell samples (n, width, 3) of (frame, geom index) pairs."""
        gid = np.asarray(self.shell_geoms)[geoms]
        samples = self.shell_index[geoms]
        valid = samples >= 0
        local = self.shell_local[np.where(valid, samples, 0)]
        rotation = self.kin.geom_xmat[frames, gid]
        points = self.kin.geom_xpos[frames, gid][:, None] + np.einsum("nij,nsj->nsi", rotation, local)
        return points, valid

    def _inside(self, field, frames, points):
        """SDF value of world points (n, width, 3) in clearance field ``field``."""
        _, sdf, rotation, translation, scales = self._clearance_fields()[field]
        if rotation is not None:
            points = np.einsum("nji,nsj->nsi", rotation[frames], points - translation[frames][:, None])
        if scales is not None:
            points = points / scales[frames, None, None]
        from .bulk_geometry import prepare
        from .sdf import GridSDF
        prepare(self)
        helper = getattr(self, '_bulk_geometry', None)
        values = (helper.query(sdf, points) if helper is not None and type(sdf) is GridSDF
                  else sdf.query(points, gradient=False))
        return values if scales is None else values * scales[frames, None]

    def _plane_clearance(self, span=None):
        """Height above its plane of the lowest inside sample, per plane pair.

        Returns (frames, geoms) clearance (+inf without a plane or without an
        inside sample) and the chosen world points. With ``span`` (start,
        stop) only those frames are measured; every pair is independent.
        """
        clearance = np.full(self.plane_field.shape, np.inf)
        chosen = np.zeros((*self.plane_field.shape, 3))
        target = self.cfg["scene_clearance_m"]
        for field in np.unique(self.plane_field[self.plane_field >= 0]):
            frames, geoms = np.nonzero(self.plane_field == field)
            if span is not None:
                keep = (frames >= span[0]) & (frames < span[1])
                frames, geoms = frames[keep], geoms[keep]
                if not len(frames):
                    continue
            points, valid = self._shell_points(frames, geoms)
            value = self._inside(int(field), frames, points)
            normal = self.plane_normal[frames, geoms]
            height = np.einsum("nsi,ni->ns", points, normal) - self.plane_offset[frames, geoms][:, None]
            inside = valid & np.isfinite(value) & (value < target) & self._in_patch(frames, geoms, points)
            height = np.where(inside, height, np.inf)
            pick = np.argmin(height, axis=1)
            row = np.arange(len(frames))
            clearance[frames, geoms] = height[row, pick]
            chosen[frames, geoms] = points[row, pick]
        return clearance, chosen

    def _in_patch(self, frames, geoms, points):
        """Samples whose footprint on the plane lies over the contact patch."""
        normal = self.plane_normal[frames, geoms]
        offset = points - self.plane_centre[frames, geoms][:, None]
        lateral = offset - np.einsum("nsi,ni->ns", offset, normal)[..., None] * normal[:, None]
        return np.linalg.norm(lateral, axis=-1) <= self.plane_radius[frames, geoms][:, None]

    def _plane_terms(self, acc, stats):
        """Plane rows: deficit of the lowest inside sample along the plane normal."""
        if not (self.plane_field >= 0).any():
            return
        target = self.cfg["scene_clearance_m"]
        sigma = self.cfg["collision_sigma_m"] * self.plane_scale
        clearance, chosen = self._plane_clearance()
        finite = np.isfinite(clearance)
        deficit = np.where(finite, np.minimum(clearance - target, 0.0), 0.0)
        bodies = self.model.geom_bodyid[np.asarray(self.shell_geoms)]
        lever = chosen - self.kin.xpos[:, bodies]
        rows = np.concatenate([self.plane_normal, np.cross(lever, self.plane_normal)], -1)
        acc.add_body_rows(deficit / sigma, rows * ((deficit < 0) / sigma)[..., None], bodies, self.kin)
        if finite.any():
            stats["plane_depth_m"] = max(
                stats.get("plane_depth_m", 0.0), float(-clearance[finite].min(initial=0.0))
            )

    def anchor_continuity(self, poses):
        """Temporal rows now measure changes of ``poses``: a later stage may
        only add a smooth correction to the continuity already achieved."""
        self.continuity_joints = poses[:, self.coords.qi].copy()
        if self.coords.root_q is not None:
            self.kin.evaluate(poses, with_jacobian=False)
            self.continuity_root = (
                self._root_points(),
                Rotation.from_quat(poses[:, self.quat_slice], scalar_first=True),
            )

    def _clear_planes(self):
        self.plane_normal[:] = 0.0
        self.plane_offset[:] = 0.0
        self.plane_field[:] = -1
        self.plane_centre[:] = 0.0
        self.plane_radius[:] = np.inf

    def penetration_summary(self, poses):
        """Grid depth per clearance field: maximum and frames deeper than 2 cm."""
        summary = {}
        for (key, *_), (clearance, *_) in zip(self._clearance_fields(), self.deepest_fields(poses)):
            depth = np.where(np.isfinite(clearance), np.maximum(-clearance, 0.0), 0.0).max(1)
            entry = summary.setdefault(key, {"max_m": 0.0, "frames_over_2cm": 0})
            entry["max_m"] = max(entry["max_m"], float(depth.max(initial=0.0)))
            entry["frames_over_2cm"] = max(entry["frames_over_2cm"], int((depth > 0.02).sum()))
        return summary

    # ----- 1. supports: lift of the fixed reference ----------------------------
    def build_support_planes(self):
        """Planes through the prioritized contact targets of each collision geom.

        One plane per (frame, geom): the unit mean contact normal through the
        mean target height, limited to the patch the targets span plus the
        contact distance and shell spacing.
        """
        self._clear_planes()
        count = np.zeros(self.plane_field.shape, int)
        geom_column = {int(g): k for k, g in enumerate(self.shell_geoms)}
        for surf in self.surfaces:
            fields = self._field_index(surf["task"].surface_ids)
            held = surf["active"] & (surf["priority"] > 0)
            geoms = np.asarray(surf["geoms"])
            reach = surf["task"].cfg["distance"] + self.cfg["collision_spacing_m"]
            for geom in np.unique(geoms):
                column = geom_column.get(int(geom))
                if column is None:
                    continue
                on_geom = geoms == geom
                for field in np.unique(fields[:, on_geom][held[:, on_geom]]):
                    if field < 0:
                        continue
                    select = held[:, on_geom] & (fields[:, on_geom] == field)
                    frames = np.flatnonzero(select.sum(1) > count[:, column])
                    if not len(frames):
                        continue
                    pick = select[frames]
                    weight = pick / pick.sum(1, keepdims=True)
                    normal = (surf["normals"][frames][:, on_geom] * pick[..., None]).sum(1)
                    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
                    targets = surf["targets"][frames][:, on_geom]
                    centre = np.einsum("fs,fsi->fi", weight, targets)
                    offset = targets - centre[:, None]
                    lateral = offset - np.einsum("fsi,fi->fs", offset, normal)[..., None] * normal[:, None]
                    spread = np.where(pick, np.linalg.norm(lateral, axis=-1), 0.0).max(1)
                    self.plane_normal[frames, column] = normal
                    self.plane_offset[frames, column] = np.einsum("fi,fi->f", centre, normal)
                    self.plane_field[frames, column] = field
                    self.plane_centre[frames, column] = centre
                    self.plane_radius[frames, column] = spread + reach
                    count[frames, column] = pick.sum(1)
        return geom_column

    def support_depth(self, poses):
        """Depth below its support plane of each geom's lowest inside sample."""
        self.kin.evaluate(poses, with_jacobian=False)
        clearance = _chunked_plane_clearance(self)
        target = self.cfg["scene_clearance_m"]
        return np.where(np.isfinite(clearance), np.maximum(target - clearance, 0.0), 0.0)

    def reference_lift(self):
        """Smooth vertical lift taking the floating base out of its supports.

        Computed on the fixed reference before contact acquisition; zero
        where the base does not penetrate an upward support with several
        collision points. Ramps over ``LIFT_RAMP_SECONDS``.
        """
        self.build_support_planes()
        lift = self._lift_allowance(self.support_depth(self.reference))
        self._clear_planes()
        return lift

    def apply_reference_lift(self, lift):
        """Raise the fixed reference (in place) and its root rows by ``lift``."""
        q = self.coords.root_q
        self.reference[:, q + 2] += lift
        self.root_reference = self.root_reference.copy()
        self.root_reference[:, 2] += lift

    def _lift_allowance(self, depth):
        """Vertical lift where several collision points of the floating base
        penetrate upward supports together.

        Only geoms rigidly attached to the floating base count, together with
        geoms carrying a source surface region that the base also carries
        (``_base_region_geoms``). Every other geom has joints between it and
        the base, so a local joint correction can move it without lifting the
        whole robot.
        """
        root = self.tasks.root_body
        if root is None:
            return np.zeros(self.frames)
        weld = self.model.body_weldid
        shell = np.asarray(self.shell_geoms)
        on_base = weld[self.model.geom_bodyid[shell]] == weld[root]
        on_base |= np.isin(shell, self._base_region_geoms(shell[on_base]))
        upward = (self.plane_normal[..., 2] >= UPWARD_COS) & on_base[None]
        penetrating = np.zeros(self.frames, int)
        target = self.cfg["scene_clearance_m"]
        for field in np.unique(self.plane_field[self.plane_field >= 0]):
            frames, geoms = np.nonzero((self.plane_field == field) & upward & (depth > 0))
            if not len(frames):
                continue
            points, valid = self._shell_points(frames, geoms)
            value = self._inside(int(field), frames, points)
            height = np.einsum("nsi,ni->ns", points, self.plane_normal[frames, geoms])
            below = valid & np.isfinite(value) & (value < target)
            below &= height < self.plane_offset[frames, geoms][:, None] + target
            below &= self._in_patch(frames, geoms, points)
            np.add.at(penetrating, frames, below.sum(1))
        lift = np.where(upward, depth * self.plane_normal[..., 2], 0.0).max(1, initial=0.0)
        lift = np.where(penetrating >= LIFT_MIN_POINTS, lift + target, 0.0)
        window = max(3, round(LIFT_RAMP_SECONDS * self.framerate) | 1)
        return uniform_filter1d(maximum_filter1d(lift, window, mode="nearest"), window, mode="nearest")

    def _base_region_geoms(self, base_geoms):
        """Geoms carrying a source surface region that the base also carries.

        One region's samples form one contact patch on the support (the seat
        under the buttocks). Where the robot spreads that patch over the base
        and its first links (the hip housings), those links rest on the same
        support: rotating them out would swing the limbs and their own
        contacts, so the patch rises with the base.
        """
        shared = set()
        for surf in self.surfaces:
            parts = np.array([binding.part for binding in surf["task"].bindings])
            geoms = np.asarray(surf["geoms"])
            for part in np.unique(parts[np.isin(geoms, base_geoms)]):
                shared.update(geoms[parts == part].tolist())
        return sorted(shared)

    # ----- 2. free bodies ---------------------------------------------------
    def prepare_free_stage(self, poses):
        """Exit planes for thin parts, full clearance, every contact held."""
        self._clear_planes()
        deepest = self.deepest_fields(poses)
        for field, (_, sdf, rotation, translation, scales) in enumerate(self._clearance_fields()):
            clearance, frames, geoms, _, field_points = deepest[field]
            if not len(frames):
                continue
            gradient = sdf.gradient(field_points)
            if rotation is not None:
                gradient = np.einsum("nij,nj->ni", rotation[frames], gradient)
            direction = np.zeros((*clearance.shape, 3))
            direction[frames, geoms] = gradient
            self._entry_planes(field, clearance, direction)
        self.clearance_scale = 1.0
        self.plane_scale = 1.0
        self.anchor_continuity(poses)
        self.hold_contacts(poses, clearance_scale=1.0)

    def _entry_planes(self, field, clearance, direction):
        """Planes at the entry-side exit for samples past a thin part's medial axis.

        Within each penetration episode of a geom, the exit direction of its
        shallowest frame gives the side the geom entered from.
        """
        target = self.cfg["scene_clearance_m"]
        for column in range(clearance.shape[1]):
            deficit = np.isfinite(clearance[:, column]) & (clearance[:, column] < target)
            deficit &= self.plane_field[:, column] < 0
            for start, stop in _runs(deficit):
                span = np.arange(start, stop)
                entry = direction[span[np.argmax(clearance[span, column])], column]
                if np.linalg.norm(entry) < 1e-9:
                    continue
                crossed = span[np.einsum("fi,i->f", direction[span, column], entry) < 0]
                if len(crossed):
                    self._exit_plane(field, crossed, column, entry)

    def _exit_plane(self, field, frames, column, entry):
        """Ray-march the lowest sample along ``entry`` to its exit surface."""
        geoms = np.full(len(frames), column)
        points, valid = self._shell_points(frames, geoms)
        height = np.where(valid, np.einsum("nsi,i->ns", points, entry), np.inf)
        start = points[np.arange(len(frames)), np.argmin(height, axis=1)]
        travel = np.zeros(len(frames))
        target = self.cfg["scene_clearance_m"]
        _, sdf, _, _, scales = self._clearance_fields()[field]
        spacing = sdf.spacing if scales is None else sdf.spacing * scales[frames]
        for _ in range(EXIT_STEPS + 1):
            value = self._inside(field, frames, (start + travel[:, None] * entry)[:, None])[:, 0]
            inside = np.isfinite(value) & (value < target)
            if not inside.any():
                break
            travel = np.where(inside, travel + np.maximum(target - value, 0.25 * spacing), travel)
        done = ~inside & (travel <= EXIT_MAX_M)
        exit_height = np.einsum("ni,i->n", start + travel[:, None] * entry, entry)
        frames = frames[done]
        self.plane_normal[frames, column] = entry
        self.plane_offset[frames, column] = exit_height[done]
        self.plane_field[frames, column] = field
        self.plane_radius[frames, column] = np.inf
