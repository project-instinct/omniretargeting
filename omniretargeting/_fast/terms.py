"""Per-frame residual rows of the fast objective, vectorized over frames.

Each method adds one row family of ``SequenceTasks.frame_rows`` (or of the
sequence reference/envelope rows) to a ``rows._Accumulator`` for all frames of
the (chunk) objective at once; ``FastObjective`` owns the packed task data.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .._sequence.proximity import support_points
from ..geometry_query import GeometryQueryError
from .kinematics import cross_columns
from .rows import _point_rows, _unit

# Grid interpolation slack (metres) of the geom-centre broad phase.
NEAR_MARGIN = 0.03
# Stiffness of the hinge that holds an attained contact within its regression
# tolerance while clearance is reduced (metres).
CONTACT_HOLD_SIGMA = 1e-4
# Stiffness of the posture-envelope hinge (relation units, see
# ``pose_relations``): inside the envelope the rows vanish; beyond it they
# dominate every task, so contacts yield and report residuals.
ENVELOPE_SIGMA = 1e-3


class FrameTerms:
    """Row families evaluated on ``self.kin`` for ``self.frames`` frames."""

    def _mapped_terms(self, acc, stats):
        kin, tasks, cfg = self.kin, self.tasks, self.cfg
        bodies = np.asarray(tasks.bodyids)
        points = kin.xpos[:, bodies]
        human = np.asarray(tasks.human)
        difference = (points - human).reshape(self.frames, -1)
        sigma = cfg["interaction_sigma_m"]
        residual = np.matmul(self.interaction, difference[:, :, None])[:, :, 0] / sigma
        jac = joint = None
        if acc.jacobian:
            joint = kin.jacp[:, bodies]
            jac = np.matmul(self.interaction, joint.reshape(self.frames, -1, self.width)) / sigma
        acc.add(residual, jac)
        for weights, sigma in tasks.axes:
            weights = np.asarray(weights, dtype=float)
            used = np.flatnonzero(weights)
            vector = np.tensordot(points[:, used], weights[used], axes=(1, 0))
            target = np.tensordot(human[:, used], weights[used], axes=(1, 0))
            norm = np.linalg.norm(vector, axis=1)
            target_norm = np.linalg.norm(target, axis=1)
            valid = np.minimum(norm, target_norm) >= 1e-8
            safe = np.where(valid, norm, 1.0)
            direction = vector / safe[:, None]
            residual = np.where(
                valid[:, None], direction - target / np.where(valid, target_norm, 1.0)[:, None], 0
            ) / sigma
            jac = None
            if acc.jacobian:
                derivative = np.tensordot(joint[:, used], weights[used], axes=(1, 0))
                along = np.matmul(direction[:, None, :], derivative)
                jac = (derivative - direction[:, :, None] * along) / (safe * sigma)[:, None, None]
                jac[~valid] = 0
            acc.add(residual, jac)

    def _orientation_terms(self, acc, stats):
        kin, worst = self.kin, 0.0
        for body, parent, targets, sigma in self.tasks.orientations.targets:
            current = kin.xmat[:, body]
            angular = kin.jacr[:, body] if acc.jacobian else None
            if parent is not None:
                parent_rotation = kin.xmat[:, parent]
                current = np.matmul(parent_rotation.transpose(0, 2, 1), current)
                if acc.jacobian:
                    angular = np.matmul(
                        parent_rotation.transpose(0, 2, 1), angular - kin.jacr[:, parent]
                    )
            scale = np.sqrt(2.0) * sigma
            residual = (current - targets).transpose(0, 2, 1).reshape(self.frames, 9) / scale
            jac = None
            if acc.jacobian:
                jac = np.concatenate(
                    [-cross_columns(current[:, :, a], angular) for a in range(3)], axis=1
                ) / scale
            acc.add(residual, jac)
            angle = Rotation.from_matrix(np.matmul(targets, current.transpose(0, 2, 1))).magnitude()
            worst = max(worst, float(angle.max(initial=0)))
        stats["body_orientation_max_rad"] = worst

    def _effector_terms(self, acc, stats):
        kin, cfg = self.kin, self.cfg
        root_point = self._root_points()
        root_jac = self._root_jacobian() if acc.jacobian else None
        roots = np.asarray(self.tasks.roots)
        sticking = normal_max = world_max = free_max = 0.0
        for eff in self.effectors:
            body = np.full((self.frames, 1), eff["body"])
            point = kin.points(body, eff["offset"][:, None])[:, 0]
            jac = kin.point_jacobians(body, point[:, None])[:, 0] if acc.jacobian else None
            active, free = eff["active"], ~(eff["active"] | eff["managed"])
            surface = active & eff["surface_only"]
            full = active & ~eff["surface_only"]
            role = eff["role"][:, None]
            error = point - eff["contact"]
            normal = eff["normal"]
            normal_error = np.einsum("fi,fi->f", normal, error)
            tangent = error - normal_error[:, None] * normal
            rows = np.zeros((self.frames, 4))
            rows[full, :3] = error[full] / cfg["contact_sigma_m"]
            rows[surface, 0] = normal_error[surface] / cfg["contact_normal_sigma_m"]
            rows[surface, 1:] = tangent[surface] / cfg["contact_tangent_sigma_m"]
            # Free targets: world, or around the robot root with morphology
            # scale; approach hands the target over to the contact anchor.
            target = np.where(
                eff["world"][:, None],
                eff["tracking"],
                root_point + eff["scale"] * (eff["tracking"] - roots),
            )
            target = target + eff["share"][:, None] * (eff["approach"] - target)
            free_share = 1.0 - eff["share"]
            sigma = np.where(eff["world"], cfg["world_tracking_sigma_m"], cfg["free_tracking_sigma_m"])
            rows[free, :3] = ((point - target) / sigma[:, None])[free]
            jac_rows = None
            if acc.jacobian:
                jac_rows = np.zeros((self.frames, 4, self.width))
                jac_rows[full, :3] = jac[full] / cfg["contact_sigma_m"]
                n_jac = np.einsum("fi,fiw->fw", normal, jac)
                jac_rows[surface, 0] = n_jac[surface] / cfg["contact_normal_sigma_m"]
                t_jac = jac - normal[:, :, None] * n_jac[:, None, :]
                jac_rows[surface, 1:] = t_jac[surface] / cfg["contact_tangent_sigma_m"]
                moving = np.where(eff["world"], 0.0, free_share)[:, None, None] * root_jac
                jac_rows[free, :3] = ((jac - moving) / sigma[:, None, None])[free]
                jac_rows *= role[:, :, None]
            acc.add(rows * role, jac_rows)
            for local, orientation_target, valid in eff["orientation"]:
                direction = np.matmul(kin.xmat[:, eff["body"]], local)
                # d(R v) = omega x (R v): rows [0, d x e_i].
                turn = np.cross(direction[:, None, :], np.eye(3))
                weight = (valid / cfg["orientation_sigma"])[:, None]
                acc.add_body_rows(
                    (direction - orientation_target) * weight,
                    np.concatenate([np.zeros_like(turn), turn], -1) * weight[..., None],
                    np.full(3, eff["body"]), kin,
                )
            if full.any():
                sticking = max(sticking, float(np.linalg.norm(error[full], axis=1).max()))
            if surface.any():
                normal_max = max(normal_max, float(np.abs(normal_error[surface]).max()))
            distance = np.linalg.norm(point - target, axis=1)
            for mask, key in ((free & eff["world"], "w"), (free & ~eff["world"], "f")):
                if mask.any():
                    if key == "w":
                        world_max = max(world_max, float(distance[mask].max()))
                    else:
                        free_max = max(free_max, float(distance[mask].max()))
        stats.update(
            sticking_max_m=sticking,
            contact_normal_max_m=normal_max,
            world_tracking_max_m=world_max,
            free_tracking_max_m=free_max,
        )

    def _surface_terms(self, acc, stats):
        kin = self.kin
        units = np.eye(3)
        for surf in self.surfaces:
            points = kin.geom_points(surf["geoms"], surf["local"])
            error = points - surf["targets"]
            lever = points - kin.xpos[:, surf["bodies"]]
            # Three axis rows per sample: [e_i, r x e_i].
            axes = _point_rows(np.broadcast_to(units, (*lever.shape[:2], 3, 3)), lever[..., None, :])
            bodies = np.repeat(surf["bodies"], 3)
            scale = surf["scale"][..., None]
            acc.add_body_rows(
                (error * scale).reshape(self.frames, -1),
                (axes * scale[..., None]).reshape(self.frames, -1, 6),
                bodies, kin,
            )
            # Priority rows: full vector for sticking samples, normal distance
            # for sliding/touch samples, as in the guarded role levels.
            normal = surf["normals"]
            along = np.einsum("fsi,fsi->fs", error, normal)
            sticking = surf["sticking"]
            normal_rows = np.zeros_like(axes)
            normal_rows[..., 0, :] = _point_rows(normal, lever)
            normal_error = np.zeros_like(error)
            normal_error[..., 0] = along
            priority = surf["priority"][..., None]
            acc.add_body_rows(
                (np.where(sticking[..., None], error, normal_error) * priority).reshape(self.frames, -1),
                (np.where(sticking[..., None, None], axes, normal_rows) * priority[..., None]).reshape(
                    self.frames, -1, 6
                ),
                bodies, kin,
            )
            if np.isfinite(surf["hold"]).any():
                # Attained contacts may not regress beyond their hold bound.
                size = np.where(sticking, np.linalg.norm(error, axis=-1), np.abs(along))
                excess = np.maximum(size - surf["hold"], 0.0)
                direction = np.where(
                    sticking[..., None],
                    error / np.maximum(size, 1e-12)[..., None],
                    np.sign(along)[..., None] * normal,
                )
                active_hold = (excess > 0)[..., None] / CONTACT_HOLD_SIGMA
                acc.add_body_rows(
                    excess / CONTACT_HOLD_SIGMA,
                    _point_rows(direction, lever) * active_hold,
                    surf["bodies"], kin,
                )
            if surf["normal_scale"].any():
                rotation = kin.geom_xmat[:, surf["geoms"]]
                normal_w = _unit(np.matmul(rotation, surf["local_normals"][..., None])[..., 0])
                # d n = omega x n: rows [0, n x e_i].
                turn = np.cross(normal_w[..., None, :], units)
                rows = np.concatenate([np.zeros_like(turn), turn], -1)
                weight = surf["normal_scale"][..., None]
                acc.add_body_rows(
                    ((normal_w - surf["pose_normals"]) * weight).reshape(self.frames, -1),
                    (rows * weight[..., None]).reshape(self.frames, -1, 6),
                    bodies, kin,
                )
            active = surf["active"]
            if active.any():
                stick = active & sticking
                other = active & ~sticking
                if stick.any():
                    stats["sticking_max_m"] = max(
                        stats.get("sticking_max_m", 0.0),
                        float(np.linalg.norm(error[stick], axis=1).max()),
                    )
                if other.any():
                    stats["contact_normal_max_m"] = max(
                        stats.get("contact_normal_max_m", 0.0), float(np.abs(along[other]).max())
                    )

    def _proximity_terms(self, acc, stats):
        """Source gaps stay gaps: the geom's support extent along each observed
        normal keeps the source skin-to-surface gap (``SurfaceProximityTasks``).
        """
        kin = self.kin
        for prox in self.proximities:
            geoms = prox["geoms"]
            normal = prox["normals"]
            accelerated_points = getattr(self, '_proximity_points', None)
            if accelerated_points is not None:
                points = accelerated_points(prox)
            else:
                rotation = kin.geom_xmat[:, geoms]
                direction = -np.matmul(normal[..., None, :], rotation)[..., 0, :]
                local = np.zeros_like(direction)
                for geom in np.unique(geoms):
                    index = np.flatnonzero(geoms == geom)
                    directions = direction[:, index].reshape(-1, 3)
                    accelerated = getattr(self, '_proximity_support', None)
                    support = (support_points(self.model, int(geom), directions)
                               if accelerated is None else accelerated(prox, int(geom), directions))
                    local[:, index] = support.reshape(self.frames, len(index), 3)
                points = kin.geom_xpos[:, geoms] + np.matmul(rotation, local[..., None])[..., 0]
            error = np.einsum("fki,fki->fk", points - prox["environment"], normal) - prox["gaps"]
            scale = prox["scale"]
            rows = _point_rows(normal, points - kin.xpos[:, prox["bodies"]]) * scale[..., None]
            acc.add_body_rows(error * scale, rows, prox["bodies"], kin)
            active = scale > 0
            if active.any():
                stats["proximity_gap_error_max_m"] = max(
                    stats.get("proximity_gap_error_max_m", 0.0), float(np.abs(error[active]).max())
                )

    def _clearance_fields(self):
        """(stat key, grid, per-frame rotations, translations) of each field."""
        fields = []
        if self.scene_sdf is not None:
            fields.append(("scene_depth_m", self.scene_sdf, None, None, None))
        for obj in self.objects:
            fields.append(("moving_depth_m", obj["sdf"], obj["rotations"], obj["translations"], obj.get("scales")))
        return fields

    def _deepest_samples(self, sdf, rotation, translation, scales=None):
        """Deepest shell sample of every (frame, geom) near a clearance field.

        A geom whose centre is farther from the scene than its bounding radius
        plus a margin cannot have a sample below the clearance target, so only
        near (frame, geom) pairs query their surface samples. Returns the
        clearance (frames, geoms; +inf when far or unknown), the near pair
        indices, and their deepest world and field points.
        """
        kin, target = self.kin, self.cfg["scene_clearance_m"]
        geoms = np.asarray(self.shell_geoms)

        def to_field(points, frame):
            if rotation is None:
                return points
            local = np.einsum("nji,nsj->nsi", rotation[frame], points - translation[frame][:, None])
            return local if scales is None else local / scales[frame, None, None]

        frame_all = np.arange(self.frames).repeat(len(geoms))
        centres = kin.geom_xpos[:, geoms].reshape(-1, 1, 3)
        centre_value = sdf.query(
            to_field(centres, frame_all)[:, 0].reshape(self.frames, len(geoms), 3), gradient=False
        )
        if scales is not None:
            centre_value = centre_value * scales[:, None]
        near = np.isfinite(centre_value) & (centre_value < self.shell_radius + target + NEAR_MARGIN)
        clearance = np.full((self.frames, len(geoms)), np.inf)
        frame_index, geom_index = np.nonzero(near)
        if not len(frame_index):
            empty = np.zeros((0, 3))
            return clearance, frame_index, geom_index, empty, empty
        # Geoms have very different sample counts: query each geom's own
        # samples, not the padding of the widest shell (same values and
        # first-minimum choice as the padded rows).
        deepest = np.full(len(frame_index), np.inf)
        world, field = np.zeros((len(frame_index), 3)), np.zeros((len(frame_index), 3))
        for geom in np.unique(geom_index):
            pairs = np.flatnonzero(geom_index == geom)
            samples = self.shell_index[geom]
            samples = samples[samples >= 0]
            local = self.shell_local[np.broadcast_to(samples, (len(pairs), len(samples)))]
            frame, gid = frame_index[pairs], geoms[geom]
            points = kin.geom_xpos[frame, gid][:, None] + np.einsum(
                "nij,nsj->nsi", kin.geom_xmat[frame, gid], local)
            field_points = to_field(points, frame)
            value = sdf.query(field_points, gradient=False)
            if scales is not None:
                value = value * scales[frame, None]
            value = np.where(np.isfinite(value), value, np.inf)
            choice = np.argmin(value, axis=1)
            row = np.arange(len(pairs))
            deepest[pairs] = value[row, choice]
            world[pairs], field[pairs] = points[row, choice], field_points[row, choice]
        clearance[frame_index, geom_index] = deepest
        return clearance, frame_index, geom_index, world, field

    def _clearance_terms(self, acc, stats, poses=None):
        """Scene and object clearance of the deepest shell sample per geom.

        Non-penetration precedes contacts while they are acquired
        (``clearance_scale``: a quarter of the contact sigma); what remains is
        reduced afterwards with contacts held, and contacts on a penetrating
        body may yield by its depth (``surface_presence.body_penetration``).
        """
        from .window_tracking import skips_proxy_collision
        if skips_proxy_collision(self):
            return
        if self.scene_sdf is None and not self.objects:
            return
        kin, cfg = self.kin, self.cfg
        geoms = np.asarray(self.shell_geoms)
        bodies = self.model.geom_bodyid[geoms]
        target = cfg["scene_clearance_m"]
        sigma = cfg["collision_sigma_m"] * self.clearance_scale
        original = getattr(self, "original_clearance", None)
        measured = {} if original is None else original.evaluate(poses, getattr(self, "frame_slice", slice(0, self.frames)).start)
        for field, (key, sdf, rotation, translation, scales) in enumerate(self._clearance_fields()):
            clearance, frame_index, geom_index, points, field_points = self._deepest_samples(
                sdf, rotation, translation, scales
            )
            finite = np.isfinite(clearance)
            if finite.any():
                stats[key] = max(stats.get(key, 0.0), float(-clearance[finite].min(initial=0.0)))
            # Pairs with a support or exit plane use plane rows instead.
            finite &= self.plane_field != field
            deficit = np.where(finite, np.minimum(clearance - target, 0.0), 0.0)
            rows = np.zeros((self.frames, len(geoms), 6))
            if acc.jacobian and len(frame_index):
                lever = points - kin.xpos[frame_index, bodies[geom_index]]
                # Derivative of the same interpolated clearance as the residual
                # (an analytic field's gradient already is).
                gradient = getattr(sdf, "value_gradient", sdf.gradient)(field_points)
                # d[s * phi(R.T (p-t) / s)]/dp = R * grad(phi): no scale factor.
                if rotation is not None:
                    gradient = np.einsum("nij,nj->ni", rotation[frame_index], gradient)
                rows[frame_index, geom_index] = _point_rows(gradient, lever)
            if measured:
                from .original_clearance import replace_deeper_original
                replace_deeper_original(field, geoms, measured, clearance, deficit, rows, kin, bodies, target)
                stats[key] = max(stats.get(key, 0.), float(-deficit.min(initial=0.) - target))
            acc.add_body_rows(deficit / sigma, rows * ((deficit < 0) / sigma)[..., None], bodies, kin)

    def _self_collision_terms(self, poses, acc, stats):
        """Hinge rows on self-collision pairs closer than ``self_clearance_m``.

        Like scene clearance, self-penetration precedes contacts while they
        are acquired (``clearance_scale``). The query is the one of the
        original-mesh repair: every eligible pair. The optional acquisition
        accelerator screens a physical band before the original mesh hinge;
        repair stays authoritative.
        """
        from .window_tracking import skips_proxy_collision
        if skips_proxy_collision(self):
            return
        query = getattr(self, "self_collision", None)
        if query is None:
            return
        target = self.cfg["self_clearance_m"]
        sigma = self.cfg["collision_sigma_m"] * self.clearance_scale
        cols = self.coords.cols
        per_frame, deepest = [], 0.0
        unknown = 0
        batch = None
        if getattr(self, 'evaluation_backend', 'legacy') != 'legacy':
            from .batched_self_collision import BatchedSelfCollision, NativeUnavailable
            if not hasattr(self, '_self_batch'):
                import multiprocessing
                workers = 1 if multiprocessing.current_process().daemon else 12
                try:
                    from .fast_self_collision import acquisition_self_collision
                    self._self_batch = acquisition_self_collision(query, workers)
                except NativeUnavailable:
                    self._self_batch = None
            if self._self_batch is not None:
                if hasattr(self, '_self_seed_provider'):
                    self._self_batch.seed_provider = self._self_seed_provider
                batch, unknown = self._self_batch.rows(poses, self.kin, acc.jacobian, target)
        for frame, pose in enumerate(poses):
            try:
                measured = query.query(pose) if batch is None else batch[frame]
                rows = [row for row in measured
                        if row.get('hinge_clearance', row['clearance']) < target]
            except GeometryQueryError:
                # Indeterminate here: no row; the original-mesh audit reports it.
                unknown += 1
                rows = []
            per_frame.append([(row.get('hinge_clearance', row['clearance']) - target,
                               (row['_jacobian'] if batch is not None else query.jacobian(row, query.data))[cols]
                               if acc.jacobian else None)
                              for row in rows])
            deepest = max(deepest, max((-row["clearance"] for row in rows), default=0.0))
        stats["self_depth_m"] = max(stats.get("self_depth_m", 0.0), deepest)
        if unknown:
            stats["self_unknown_frames"] = stats.get("self_unknown_frames", 0) + unknown
        # Row counts differ between frames: accumulate each frame's own rows,
        # never rows padded to the widest frame of this chunk.
        for frame, rows in enumerate(per_frame):
            if not rows:
                continue
            residual = np.array([value for value, _ in rows]) / sigma
            jac = np.array([row_jac for _, row_jac in rows]) / sigma if acc.jacobian else None
            acc.add_frame_rows(frame, residual, jac)

    def _reference_terms(self, poses, acc, stats):
        cfg, coords = self.cfg, self.coords
        residual = (poses[:, coords.qi] - self.reference[:, coords.qi]) / cfg["reference_sigma"]
        jac = None
        if acc.jacobian:
            jac = np.broadcast_to(self.selector / cfg["reference_sigma"], (self.frames, *self.selector.shape))
        acc.add(residual, jac)
        if coords.root_q is None:
            return
        roots = np.asarray(self.tasks.roots)
        root_point = self._root_points()
        residual = np.concatenate(
            [
                (root_point[:, :2] - roots[:, :2]) / cfg["root_xy_sigma_m"],
                (root_point[:, 2:] - self.root_reference[:, 2:]) / cfg["root_height_sigma_m"],
            ],
            axis=1,
        )
        jac = None
        if acc.jacobian:
            root_jac = self._root_jacobian()
            jac = np.concatenate(
                [root_jac[:, :2] / cfg["root_xy_sigma_m"], root_jac[:, 2:] / cfg["root_height_sigma_m"]],
                axis=1,
            )
        acc.add(residual, jac)
        stats["root_xy_offset_max_m"] = float(
            np.linalg.norm(root_point[:, :2] - roots[:, :2], axis=1).max()
        )
        relative = self.root_rotations.inv() * Rotation.from_quat(
            poses[:, self.quat_slice], scalar_first=True
        )
        stats["root_orientation_max_rad"] = float(relative.magnitude().max())

    def _envelope_terms(self, poses, acc, stats):
        """Soft posture rows and layout hinges on the fixed reference.

        Relation changes cost softly (``pose_relations``); only a flip of a
        clear layout component meets a stiff hinge. Only used once a fixed
        reference exists.
        """
        if self.temporal_reference is None:
            return
        # Joint limits and the root translation are exact (``lower``/``upper``,
        # ``project``); layout components are not separable per coordinate.
        rows, jac, excess = self.relation_envelope.hinge_rows(
            self.kin, self.envelope_weight, ENVELOPE_SIGMA, acc.jacobian)
        stats["layout_flip_max"] = excess
        acc.add(rows, jac)
        # Guards that bound candidates also shape the step (same stiffness).
        frames = getattr(self, "frame_slice", slice(0, self.frames))
        for guard in getattr(self, "hinge_guards", ()):
            rows, jac = guard.hinge_rows(self.kin, frames, self.envelope_weight, ENVELOPE_SIGMA, acc.jacobian)
            if rows is not None:
                acc.add(rows, jac)
        if getattr(self, '_soft_bulk_layouts', ()):
            from .soft_frame_rows import add_layout
            add_layout(self, acc, stats)
