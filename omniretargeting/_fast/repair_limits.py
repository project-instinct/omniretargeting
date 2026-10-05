"""Immutable, local acceptance limits for original-mesh penetration repair.

These limits supplement the fast backend's fixed pre-acquisition reference.
They never rebase it: the layout of ``source_shape`` (``pose_relations``)
applies through its guard rows and exact check. Only scalar joints move.
"""

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

from .._refinement.math import _difference_operator
from ..motion_budgets import windowed_budget
from .kinematics import TrajectoryKinematics, skew
from .runtime_options import OPTIONS
from .limb_scope import endpoint_limb_scopes
from .rows import _block_rows, _merge_identical
from .repair_temporal import RepairTemporal
from .repair_support import SupportContinuity

# Locality of one secondary correction relative to its entry motion: the
# fixed-root joint box, the reach certificate (``local_reach``) and the step
# scale of the repair QP. Not a posture limit against the source motion.
LOCAL_ANGLE = 0.05
CONTACT_DRIFT = 0.001
FREE_DRIFT = 0.003
NUMERICAL_SLACK = 2e-7


class LinearBounds:
    def __init__(self, size):
        self.size = size
        self.matrices, self.lower, self.upper = [], [], []
        self.norms = []
        self.box = None

    def set_box(self, lower, upper):
        """Joint box rows. Every proposal lies in the box (or leaves a
        coordinate at zero), so a row or cone that no point of the box can
        violate is redundant and omitted exactly by ``packed``/``cones``."""
        self.box = (len(self.matrices), np.minimum(lower, 0.), np.maximum(upper, 0.))
        self.add(sparse.eye(self.size), lower, upper)

    def _range(self, matrix):
        """Exact interval of ``matrix @ step`` over the box."""
        _, low, high = self.box
        positive, negative = matrix.maximum(0), matrix.minimum(0)
        return positive @ low + negative @ high, positive @ high + negative @ low

    def norm(self, jacobian, residual, allowed):
        """Exact norm of a linearized vector, not its tangent halfspace.

        As in the acquisition guards, half of ``check``'s 1e-7 tolerance stays
        in reserve for curvature and roundoff; a pose that already spent it
        remains feasible. The nonlinear cap itself is unchanged.
        """
        shape, width = residual.shape, jacobian.shape[-1]
        jacobian = jacobian.reshape(shape[0], -1, width)
        residual = residual.reshape(-1, shape[-1])
        base = np.broadcast_to(allowed, shape[:-1]).ravel()
        allowed = np.minimum(np.maximum(base + 5e-8, np.linalg.norm(residual, axis=1)), base + 1e-7)
        self.norms.append((_block_rows(jacobian, width), residual, allowed))

    def add(self, matrix, lower, upper):
        matrix = sparse.csr_matrix(matrix)
        self.matrices.append(matrix)
        self.lower.append(np.broadcast_to(lower, (matrix.shape[0],)).copy())
        self.upper.append(np.broadcast_to(upper, (matrix.shape[0],)).copy())

    def packed(self):
        matrix = sparse.vstack(self.matrices, format="csr")
        lower, upper = np.concatenate(self.lower), np.concatenate(self.upper)
        if self.box is not None:
            bottom, top = self._range(matrix)
            needed = (bottom < lower) | (top > upper)
            first = sum(m.shape[0] for m in self.matrices[:self.box[0]])
            needed[first:first + self.size] = True
            matrix, lower, upper = matrix[needed], lower[needed], upper[needed]
        matrix, lower, upper = _merge_identical(matrix, lower, upper)
        matrix = matrix.tocsc()
        # Exact row equilibration, including tiny per-joint derivative budgets.
        scale = 1 / np.maximum(np.maximum(np.abs(lower), np.abs(upper)), 1e-4)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        return sparse.diags(scale) @ matrix, lower * scale, upper * scale

    def cones(self):
        """Norm bounds some point of the box could violate."""
        for jacobian, residual, allowed in self.norms:
            if self.box is not None:
                bottom, top = self._range(jacobian)
                dimension = residual.shape[1]
                flat = residual.ravel()
                worst = np.maximum(np.abs(flat + bottom), np.abs(flat + top)).reshape(-1, dimension)
                needed = np.linalg.norm(worst, axis=1) > allowed
                jacobian = sparse.csr_matrix(jacobian)[np.repeat(needed, dimension)]
                residual, allowed = residual[needed], allowed[needed]
            yield jacobian, residual, allowed


class RepairLimits:
    def __init__(self, objective, entry):
        self.objective, self.coords = objective, objective.coords
        self.entry = np.array(entry, copy=True)
        self.local_angle = .10 if OPTIONS.abl_repair_local_angle else LOCAL_ANGLE
        self.ablate_contacts = OPTIONS.abl_contact_regression
        self.frames, self.width = len(entry), len(self.coords.qi)
        self.kin = TrajectoryKinematics(objective.model, self.coords.di)
        self.kin.evaluate(entry)
        self.bodies = objective.envelope_bodies
        self.entry_rotations = self.kin.xmat[:, self.bodies].copy()
        self.points = self.point_groups()
        self.anchors = [point.copy() for point, _, _ in self.points]
        self.original_anchors = [point.copy() for point in self.anchors]
        self.movable_body = None
        # Frames the current repair group may move (None: all), see repair_frames.
        self.frame_mask = None
        self.penetration_yield = np.zeros((self.frames, objective.model.nbody))
        self.penetration_yield.setflags(write=False)
        self._penetration_yield_frozen = False
        self.limb_scopes = endpoint_limb_scopes(objective)
        self.targets = [e["contact"][:, None] for e in objective.effectors] + [s["targets"] for s in objective.surfaces]
        self.contact_tolerances = ([min(CONTACT_DRIFT, objective.cfg["endpoint_regression_m"])] * len(objective.effectors)
                                   + [min(CONTACT_DRIFT, s["task"].cfg["regression"]) for s in objective.surfaces])
        if self.ablate_contacts:
            from .repair_contact_ablation import prepare
            self.contact_planes = prepare(self)
        self.target_budgets = [np.linalg.norm(point - target, axis=-1) + tolerance
                               for point, target, tolerance in zip(self.anchors, self.targets, self.contact_tolerances)]
        # A free endpoint is allowed to approach its source-derived target.
        # Freezing its erroneous acquired position would obstruct useful repair.
        self.tracking_budgets = [np.linalg.norm(error, axis=-1)
                                 for error, _, _ in self.tracking_rows(False)]
        slide = objective.model.jnt_type[objective.model.dof_jntid[self.coords.di]] == 2
        self.slide = slide
        local = np.where(slide, FREE_DRIFT, self.local_angle)
        self.lower = np.maximum(objective.lower, entry[:, self.coords.qi] - local)
        self.upper = np.minimum(objective.upper, entry[:, self.coords.qi] + local)
        self.differences = []
        if OPTIONS.reference_derivative_budget:
            for order in (1, 2, 3):
                if self.frames <= order:
                    continue
                op = sparse.kron(_difference_operator(self.frames, order), sparse.eye(self.width), format="csr")
                before = op @ entry[:, self.coords.qi].ravel()
                # Legacy regression envelope; production checks source chatter
                # after repair and does not forbid smooth low-frequency change.
                self.differences.append((op, windowed_budget(before.reshape(-1, self.width)).ravel()))
        self.aggregate_temporal = RepairTemporal(entry[:, self.coords.qi], self.differences, self.width)
        self.support_continuity = SupportContinuity(
            objective, self.kin, slack=NUMERICAL_SLACK, enabled=OPTIONS.repair_support_continuity)
        # Independent joint branches can backtrack independently. A saturated
        # ankle must not shrink an otherwise safe correction of an arm.
        groups = {}
        model = objective.model
        root = (0 if self.coords.root_d is None
                else int(model.jnt_bodyid[model.dof_jntid[self.coords.root_d]]))
        for column, dof in enumerate(self.coords.di):
            body = int(model.jnt_bodyid[model.dof_jntid[dof]])
            while model.body_parentid[body] not in (0, root):
                body = int(model.body_parentid[body])
            groups.setdefault(body, []).append(column)
        self.branches = list(groups.values())

    def freeze_penetration_yield(self, measurement):
        """Freeze reliable scene/object depths from the original-mesh entry.

        Like surface_presence.body_penetration, each frame/contact uses only
        its own welded body's maximum depth, never a sum or another body's
        allowance. Self-collision does not grant scene-contact yield. The
        repair already measures the original meshes; using that baseline
        avoids a second query and a grid approximation of the permitted exit.
        No candidate or group switch may enlarge this frozen allowance.
        """
        if self._penetration_yield_frozen:
            raise RuntimeError("Repair penetration yield is already frozen")
        from .frozen_depth import reliable_body_depth
        self.penetration_yield = reliable_body_depth(self.objective.model, self.frames, measurement)
        self._penetration_yield_frozen = True

    def contact_budgets(self, index, bodies, budget):
        """The same own-body allowance for target and anchor checks/cones.

        Keep point_groups' nominal budgets intact: they identify contacts for
        group selection and reporting. Managed endpoints and free tracking
        receive no yield. Existing immutable guards remain independent caps.
        """
        weld = self.objective.model.body_weldid[bodies]
        yielded = np.where(budget == CONTACT_DRIFT, self.penetration_yield[:, weld], 0.)
        if self.ablate_contacts:
            extra = np.where(budget == CONTACT_DRIFT, CONTACT_DRIFT, 0.)
            return budget + yielded + extra, self.target_budgets[index] + yielded + extra
        return budget + yielded, self.target_budgets[index] + yielded

    def backtrack_branches(self, poses, step):
        corrected = np.zeros_like(step)
        fractions = []
        for columns in self.branches:
            fraction = 0.
            for alpha in (1., .5, .25, .125, .0625, .03125, .015625):
                candidate = poses.copy()
                candidate[:, self.coords.qi[columns]] += alpha * step[:, columns]
                if self.check(candidate)[0]:
                    fraction = alpha
                    break
            corrected[:, columns] = fraction * step[:, columns]
            fractions.append(fraction)
        return corrected, fractions

    def angles(self, target):
        relative = self.kin.xmat[:, self.bodies] @ target.transpose(0, 1, 3, 2)
        return Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(self.frames, -1)

    def point_groups(self, *, include_surfaces=True):
        groups = []
        from .repair_frame_norm import enabled
        from .repair_points import points as cached_points
        fast = enabled()
        for index, eff in enumerate(self.objective.effectors):
            body = eff["body"]
            if fast:
                point = cached_points(self, ('endpoint', index), 'endpoint', np.array([body]), eff['offset'])
            else:
                point = self.kin.xpos[:, body] + np.einsum("fij,fj->fi", self.kin.xmat[:, body], eff["offset"])
                point = point[:, None]
            active = eff["active"] | eff["managed"]
            budget = np.where(active, CONTACT_DRIFT, np.inf)[:, None]
            # Surface-managed contacts use their actual material samples;
            # the legacy endpoint target is inactive in the acquiring solve.
            budget = np.where(eff["managed"][:, None], np.inf, budget)
            groups.append((point, np.array([eff["body"]]), budget))
        for index, surf in enumerate(self.objective.surfaces if include_surfaces else ()):
            point = (cached_points(self, ('surface', index), 'surface', surf['geoms'], surf['local'])
                     if fast else self.kin.geom_points(surf["geoms"], surf["local"]))
            budget = np.where(surf["active"], CONTACT_DRIFT, np.inf)
            groups.append((point, surf["bodies"], budget))
        return groups

    def tracking_rows(self, with_jacobian, bounds=None):
        """Unchanged world/body-relative tracking semantics from acquisition."""
        tasks, kin = self.objective.tasks, self.kin
        roots = np.asarray(tasks.roots)
        root = roots if tasks.root_body is None else kin.xpos[:, tasks.root_body]
        from .repair_frame_norm import enabled, point_jacobians, root_relative
        for eff, (point, bodies, _) in zip(self.objective.effectors,
                                         self.point_groups(include_surfaces=not enabled())):
            target = np.where(eff["world"][:, None], eff["tracking"],
                              root + eff["scale"] * (eff["tracking"] - roots))
            target += eff["share"][:, None] * (eff["approach"] - target)
            free = ~(eff["active"] | eff["managed"])
            jac = None
            if with_jacobian:
                jac = (kin.point_jacobians(bodies, point) if bounds is None else
                       point_jacobians(self, bounds, bodies, point))
                if tasks.root_body is not None:
                    share = np.where(eff["world"], 0., 1. - eff["share"])
                    jac = root_relative(jac, share, kin, tasks.root_body)
            yield point - target[:, None], jac, free[:, None]

    def correction_mask(self, bodies, budget):
        """Only the repaired contact group may approach its unchanged target."""
        if self.movable_body is None:
            return np.zeros_like(budget, dtype=bool)
        weld = self.objective.model.body_weldid[bodies]
        return (weld[None] == self.movable_body) & (budget == CONTACT_DRIFT)

    def constraints(self, poses, *, restore_layout=False, projection=None):
        if (OPTIONS.repair_defect_windows and not OPTIONS.abl_repair_restore_layout
                and not getattr(self, '_repair_frame_view', False)):
            from .repair_sparse_frames import constraints
            compact = constraints(self, poses, restore_layout=restore_layout, projection=projection)
            if compact is not None:
                return compact
        self.kin.evaluate(poses)
        if projection is None:
            bounds = LinearBounds(self.frames * self.width)
        else:
            from .repair_projected import ProjectedBounds
            from .repair_redundancy import PROJECTED_RESTORATION_REDUNDANCY, RestorationBounds
            kind = RestorationBounds if restore_layout and PROJECTED_RESTORATION_REDUNDANCY else ProjectedBounds
            bounds = kind(self.frames * self.width, projection, self.frames, self.width)
        values = poses[:, self.coords.qi].ravel()
        bounds.set_box(self.lower.ravel() - values - 1e-8, self.upper.ravel() - values + 1e-8)
        for op, budget in self.differences:
            current = op @ (values - self.entry[:, self.coords.qi].ravel()
                            if OPTIONS.correction_budget else values)
            # Keep zero feasible after another chain's accepted roundoff.
            # The immutable nonlinear cap remains NUMERICAL_SLACK throughout.
            slack = np.minimum(NUMERICAL_SLACK, np.maximum(NUMERICAL_SLACK / 4, np.abs(current) - budget))
            bounds.add(op, -budget - slack - current, budget + slack - current)
        if OPTIONS.repair_aggregate:
            self.aggregate_temporal.add(bounds, poses[:, self.coords.qi])
        self.support_continuity.add(bounds, self.kin)
        from .repair_frame_norm import point_jacobians, masked_norm
        for index, (anchor, (points, bodies, budget)) in enumerate(zip(self.anchors, self.point_groups())):
            jac = (self.kin.point_jacobians(bodies, points) if self.ablate_contacts else
                   point_jacobians(self, bounds, bodies, points))
            if self.ablate_contacts:
                from .repair_contact_ablation import add
                add(self, index, points, jac, bounds)
            correcting = self.correction_mask(bodies, budget)
            budget, target_budget = self.contact_budgets(index, bodies, budget)
            if correcting.any():
                target_budget = np.where(correcting, target_budget, 1.)
                masked_norm(bounds, jac, points - self.targets[index],
                            np.maximum(target_budget, 1e-8), correcting)
            budget = np.where(correcting, np.inf, budget)
            delta = points - anchor
            valid = np.isfinite(budget)
            masked_norm(bounds, jac, delta, budget, valid)
        for (error, jac, free), budget in zip(self.tracking_rows(True, bounds), self.tracking_budgets):
            masked_norm(bounds, jac, error, budget, free)
        if not OPTIONS.body_rotation_report_only:
            # Vector norm cones prevent tangent steps on an already saturated
            # local sphere from repeatedly failing the nonlinear check.
            for target, allowed in ((self.entry_rotations, self.local_angle),):
                relative = self.kin.xmat[:, self.bodies] @ target.transpose(0, 1, 3, 2)
                vector = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(self.frames, -1, 3)
                length = np.linalg.norm(vector, axis=-1)
                cross = skew(vector)
                coefficient = np.full_like(length, 1 / 12)
                away = length > 1e-4
                angle = length[away]
                coefficient[away] = 1 / angle**2 - (1 + np.cos(angle)) / (2 * angle * np.sin(angle))
                inverse = np.eye(3) - 0.5 * cross + coefficient[..., None, None] * (cross @ cross)
                jac = inverse @ self.kin.jacr[:, self.bodies]
                bounds.norm(jac, vector, allowed)
        # Secondary proposals must see the original acquisition/continuity
        # bounds, not merely fail them after solving an incompatible QP.
        # Only scalar joints are variables here; root tangent columns stay zero.
        from .repair_frame_norm import enabled
        from .repair_contact_rows import selector_key
        selector_state = getattr(self, '_repair_selector', None)
        selector_identity = self.frames, selector_key(self.coords.selector)
        if enabled() and selector_state is not None and selector_state[0] == selector_identity:
            selector = selector_state[1]
        else:
            selector = sparse.kron(sparse.eye(self.frames), self.coords.selector.T, format="csr")
            if enabled():
                self._repair_selector = selector_identity, selector
        guards = list(getattr(self.objective, "candidate_guards", ()))
        shape = getattr(self.objective, "source_shape", None)
        if shape is not None:
            guards.insert(0, shape)
        for guard in guards:
            if not hasattr(guard, "repair_bounds"):
                continue
            from .repair_contact_rows import contact_bounds
            from .repair_layout_rows import layout_bounds
            restore = restore_layout and guard is shape
            align_layout = restore and OPTIONS.abl_repair_restore_layout
            # Full rows keep the preceding six frames of sustained-flip
            # history; the legacy chunk cache is unchanged when disabled.
            early_selector = selector if enabled() else None
            cached = (None if projection is None else
                      contact_bounds(guard, poses, selector=early_selector))
            if cached is None and projection is not None and not align_layout:
                cached = layout_bounds(guard, poses, restore=restore, selector=early_selector)
            linear, norms = (cached if cached is not None else
                             guard.repair_bounds(poses, restore=True)
                             if restore and not align_layout else guard.repair_bounds(poses))
            for matrix, lower, upper in linear:
                bounds.add(matrix if cached is not None and early_selector is not None
                           else matrix @ selector, lower, upper)
            for matrix, residual, allowed in norms:
                bounds.norms.append((matrix if cached is not None and early_selector is not None
                                    else matrix @ selector, residual, allowed))
        return bounds

    def check(self, poses):
        if not np.isfinite(poses).all():
            return False, {"reason": "nonfinite"}
        shape = getattr(self.objective, "source_shape", None)
        if shape is not None and not shape.accepts(poses):
            return False, {"reason": "immutable source shape"}
        for guard in getattr(self.objective, "candidate_guards", ()):
            from .repair_guard_cache import accepts
            if not accepts(guard, poses):
                from .repair_diagnostics import guard_failure
                return False, guard_failure(guard, poses)
        qi = self.coords.qi
        untouched = np.ones(poses.shape[1], dtype=bool)
        untouched[qi] = False
        if not np.array_equal(poses[:, untouched], self.entry[:, untouched]):
            return False, {"reason": "root changed"}
        values = poses[:, qi]
        if np.any(values < self.lower - 1e-8) or np.any(values > self.upper + 1e-8):
            return False, {"reason": "joint envelope"}
        excess = [float((np.abs(op @ (values - self.entry[:, qi] if OPTIONS.correction_budget else values).ravel()) - bound).max(initial=0)) for op, bound in self.differences]
        if max(excess, default=0) > NUMERICAL_SLACK:
            return False, {"reason": "per-joint temporal budget", "excess": excess}
        valid, aggregate = self.aggregate_temporal.check(values) if OPTIONS.repair_aggregate else (True, [])
        if not valid:
            return False, {"reason": "per-joint secondary RMS/peak budget", "constraints": aggregate}
        self.kin.evaluate(poses, False)
        support_valid, support_report = self.support_continuity.check(self.kin)
        if not support_valid:
            return False, support_report
        local = self.angles(self.entry_rotations)
        if not OPTIONS.body_rotation_report_only and local.max(initial=0) > self.local_angle + 1e-7:
            frame, body = np.unravel_index(np.argmax(local), local.shape)
            return False, {"reason": "body posture", "frame": int(frame),
                           "body": self.objective.model.body(int(self.bodies[body])).name,
                           "excess": float(local.max() - self.local_angle)}
        tracking_excess = 0.
        for (error, _, free), budget in zip(self.tracking_rows(False), self.tracking_budgets):
            tracking_delta = np.linalg.norm(error, axis=-1) - budget
            tracking_excess = max(tracking_excess, float(tracking_delta[free].max(initial=0.)))
        if tracking_excess > 1e-7:
            return False, {"reason": "free endpoint fidelity", "excess_m": tracking_excess}
        free_drift = 0.
        drift = 0.0
        repaired_drift = 0.
        total_drift = 0.
        for index, (anchor, (points, bodies, budget)) in enumerate(zip(self.anchors, self.point_groups())):
            if self.ablate_contacts:
                from .repair_contact_ablation import check
                inward = check(self, index, points)
                if inward is not None:
                    return False, inward
            distance = np.linalg.norm(points - anchor, axis=-1)
            original_drift = np.linalg.norm(points - self.original_anchors[index], axis=-1)
            total_drift = max(total_drift, float(original_drift[budget == CONTACT_DRIFT].max(initial=0)))
            if index < len(self.objective.effectors):
                eff = self.objective.effectors[index]
                free = ~(eff["active"] | eff["managed"])
                free_drift = max(free_drift, float(original_drift[free].max(initial=0.)))
            correcting = self.correction_mask(bodies, budget)
            anchored_contact = (budget == CONTACT_DRIFT) & ~correcting
            budget, target_budget = self.contact_budgets(index, bodies, budget)
            error = np.linalg.norm(points - self.targets[index], axis=-1)
            if np.any(error[correcting] > target_budget[correcting] + 1e-7):
                return False, {"reason": "repaired contact regressed"}
            repaired_drift = max(repaired_drift, float(distance[correcting].max(initial=0)))
            budget = np.where(correcting, np.inf, budget)
            if np.any(distance > budget + 1e-7):
                return False, {"reason": "contact/endpoint drift"}
            drift = max(drift, float(distance[anchored_contact].max(initial=0)))
        change = np.abs(values - self.entry[:, qi])
        # Audit relation changes and the layout of the fixed reference.
        envelope = getattr(self.objective, "relation_envelope", None)
        relation = {} if envelope is None else envelope.summary(self.kin)
        bad = relation.get("layout_flip_frames", [])
        return True, {**support_report, "max_joint_change_rad": float(change[:, ~self.slide].max(initial=0)),
                      "max_slide_change_m": float(change[:, self.slide].max(initial=0)),
                      "max_body_change_rad": float(local.max(initial=0)),
                      "max_contact_drift_m": drift, "temporal_excess": excess,
                      "repaired_contact_displacement_m": repaired_drift,
                      "max_contact_point_displacement_m": total_drift,
                      "source_chatter_checked_by_final_audit": not OPTIONS.reference_derivative_budget,
                      "max_free_endpoint_displacement_m": free_drift,
                      "free_tracking_error_increase_m": tracking_excess,
                      "relation_envelope": relation,
                      "fixed_reference_layout_flip_frames": len(bad),
                      "original_layout_preserved": not bad,
                      "root_unchanged": True}
