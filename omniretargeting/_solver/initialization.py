"""Optional limb-only pose seed, using existing fixed orientation calibration.

This whole-body experiment is not a reproduction of TopoRetarget's hand loss.
No contact/collision policy or final objective is changed here.
"""

import itertools
from time import perf_counter

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .targets import _calibrated_orientation_target


class PoseInitializationMixin:
    def _pose_initialization_bindings(self):
        """Take the last two edges of existing bone paths; infer joints from MJCF.

        The nearest calibrated ancestor above each proximal bone is its fixed
        reference (normally torso for arms, pelvis for legs). A missing mapping
        fails closed, instead of guessing URDF/source axis conventions.
        """
        edges = [(a, b) for a, b, _ in self.pose_bones]
        parents = {a for a, _ in edges}
        chains = [
            (a, b, c)
            for a, b in edges
            for b2, c in edges
            if b == b2 and c not in parents and a != c
        ]
        if not chains:
            raise ValueError("missing_two_segment_bone_mapping")
        reference_bodies = {
            name: task["body_id"] for name, task in self.body_orientation_tasks.items()
        }
        reference_bodies.update(
            {
                task["parent_source"]: task["parent_body_id"]
                for task in self.body_orientation_tasks.values()
                if "parent_source" in task
            }
        )
        model = self.robot_model
        bindings, coordinates = [], set()
        for chain in chains:
            bodies = [model.body(self.joint_mapping[name]).id for name in chain]
            ancestor = int(model.body_parentid[bodies[0]])
            reference = None
            while ancestor > 0:
                matches = [
                    name
                    for name, body in reference_bodies.items()
                    if body == ancestor
                    and name not in chain
                    and name in self._source_orientation_references
                    and name in self._robot_orientation_references
                ]
                if matches:
                    if len(matches) != 1:
                        raise ValueError(
                            "ambiguous_calibrated_ancestor:" + ",".join(matches)
                        )
                    reference = min(matches)
                    break
                ancestor = int(model.body_parentid[ancestor])
            if reference is None:
                raise ValueError("missing_calibrated_ancestor:" + "->".join(chain))
            # Only descendants of this reference may move. Thus root and all
            # torso/pelvis joints remain exactly as registered by the old path.
            body, path = bodies[-1], []
            while body > 0 and body != ancestor:
                path.append(body)
                start = int(model.body_jntadr[body])
                for joint in range(start, start + int(model.body_jntnum[body])):
                    if model.jnt_type[joint] not in (
                        mujoco.mjtJoint.mjJNT_HINGE,
                        mujoco.mjtJoint.mjJNT_SLIDE,
                    ):
                        raise ValueError("unsupported_non_scalar_limb_joint")
                    coordinates.add(int(model.jnt_qposadr[joint]))
                body = int(model.body_parentid[body])
            if body != ancestor or not all(b in path for b in bodies):
                raise ValueError(
                    "mapped_bones_are_not_one_robot_chain:" + "->".join(chain)
                )
            bindings.append((chain, reference, ancestor))
        # Do not introduce variables excluded by the full solver or its bounds.
        indices = np.array(
            [
                i
                for i, q in enumerate(self.q_a_indices)
                if q in coordinates and self.q_a_lb[i] < self.q_a_ub[i]
            ],
            dtype=int,
        )
        if not len(indices):
            raise ValueError("no_free_limb_coordinates")
        return bindings, indices

    def _pose_seed_rows(self, q, targets, bindings, indices):
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        jac, positions, _ = self._compute_robot_jacobians(q)
        names = {name: i for i, name in enumerate(self.valid_joint_names)}
        residuals, rows = [], []
        for chain, _, reference_body in bindings:
            rotation = self.robot_data.xmat[reference_body].reshape(3, 3)
            for a, b in itertools.pairwise(chain):
                vector = positions[b] - positions[a]
                length = np.linalg.norm(vector)
                if not np.isfinite(length) or length < 1e-8:
                    raise ValueError("degenerate_robot_bone")
                direction = vector / length
                ja, jb = (jac[3 * names[n] : 3 * names[n] + 3, indices] for n in (a, b))
                residuals.append(rotation.T @ direction - targets[(a, b)])
                # Reference body is fixed with respect to all selected variables.
                rows.append(
                    rotation.T
                    @ (np.eye(3) - np.outer(direction, direction))
                    @ (jb - ja)
                    / length
                )
        return np.concatenate(residuals), np.vstack(rows)

    def _initialize_pose(self, q_init, human_joints, source_joint_orientations):
        """Return a single bounded seed or the untouched original, with a reason.

        The residual retains each bone's 3D direction and the two-bone bending
        plane. Axial twist, terminal orientation and straight-limb twist remain
        unobservable. Existing frame calibration and final orientation tasks
        retain responsibility for those freedoms. There is no temporal term or
        velocity bound: warm-start affects numerical initialization only.
        """
        # Disabled/fallback paths preserve even the caller's original dtype.
        original = np.array(q_init, copy=True)
        cfg = self.end_effector_config.get("pose_initialization", {}) or {}
        report = {"status": "disabled", "seconds": 0.0}
        self.last_pose_initialization = report
        if not cfg.get("enabled", False):
            return original
        started = perf_counter()
        saved = self.robot_data.qpos.copy()
        try:
            if (
                original.shape != (self.robot_model.nq,)
                or not np.isfinite(original).all()
            ):
                raise ValueError("invalid_initial_configuration")
            if source_joint_orientations is None:
                raise ValueError("missing_source_orientations")
            source = np.asarray(human_joints, dtype=float)
            orientations = np.asarray(source_joint_orientations, dtype=float)
            if (
                source.shape != (len(self.valid_joint_names), 3)
                or not np.isfinite(source).all()
            ):
                raise ValueError("invalid_source_positions")
            if (
                orientations.ndim != 2
                or orientations.shape[1] != 4
                or not np.isfinite(orientations).all()
            ):
                raise ValueError("invalid_source_orientations")
            bindings, indices = self._pose_initialization_bindings()
            # Fixed reference bodies must also be unaffected by other limbs.
            for _, _, body in bindings:
                while body > 0:
                    start = int(self.robot_model.body_jntadr[body])
                    joint_ids = range(
                        start, start + int(self.robot_model.body_jntnum[body])
                    )
                    if any(
                        self.robot_model.jnt_qposadr[j] in self.q_a_indices[indices]
                        for j in joint_ids
                    ):
                        raise ValueError("limb_coordinates_would_move_reference_body")
                    body = int(self.robot_model.body_parentid[body])
            names = {name: i for i, name in enumerate(self.valid_joint_names)}
            targets = {}
            for chain, reference, _ in bindings:
                quat = orientations[self._source_orientation_indices[reference]]
                if np.linalg.norm(quat) < 1e-12:
                    raise ValueError("zero_source_quaternion")
                rotation = _calibrated_orientation_target(
                    Rotation.from_quat(quat, scalar_first=True).as_matrix(),
                    self._source_orientation_references[reference],
                    self._robot_orientation_references[reference],
                )
                for a, b in itertools.pairwise(chain):
                    vector = source[names[b]] - source[names[a]]
                    length = np.linalg.norm(vector)
                    if not np.isfinite(length) or length < 1e-8:
                        raise ValueError("degenerate_source_bone:" + a + "->" + b)
                    targets[(a, b)] = rotation.T @ (vector / length)
            q_indices = self.q_a_indices[indices]
            lower, upper = self.q_a_lb[indices], self.q_a_ub[indices]
            if np.any(original[q_indices] < lower) or np.any(
                original[q_indices] > upper
            ):
                raise ValueError("initial_limb_outside_existing_limits")
            cache_x, cache_rows = None, None

            def evaluate(x):
                nonlocal cache_x, cache_rows
                if cache_x is None or not np.array_equal(cache_x, x):
                    q = original.astype(float, copy=True)
                    q[q_indices] = x
                    cache_rows = self._pose_seed_rows(q, targets, bindings, indices)
                    cache_x = x.copy()
                return cache_rows

            initial_error = float(np.linalg.norm(evaluate(original[q_indices])[0]))
            report.update(
                initial_error=initial_error,
                chains=[list(b[0]) for b in bindings],
                references=[b[1] for b in bindings],
                qpos_indices=q_indices.tolist(),
            )
            if initial_error < 1e-8:
                report.update(
                    status="unchanged", final_error=initial_error, evaluations=0
                )
                return original
            result = least_squares(
                lambda x: evaluate(x)[0],
                original[q_indices],
                jac=lambda x: evaluate(x)[1],
                bounds=(lower, upper),
                max_nfev=int(cfg.get("max_iterations", 20)),
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
            )
            if not result.success:
                raise ValueError("prefit_did_not_converge:" + result.message)
            error = float(np.linalg.norm(result.fun))
            if not np.isfinite(result.x).all() or not np.isfinite(error):
                raise ValueError("nonfinite_candidate")
            if error > initial_error + 1e-10:
                raise ValueError("candidate_increased_direction_error")
            candidate = original.astype(float, copy=True)
            candidate[q_indices] = result.x
            if np.any(result.x < lower) or np.any(result.x > upper):
                raise ValueError("candidate_outside_existing_limits")
            report.update(
                status="accepted", final_error=error, evaluations=int(result.nfev)
            )
            return candidate
        except (
            TypeError,
            FloatingPointError,
            OverflowError,
            ValueError,
            IndexError,
            KeyError,
            RuntimeError,
            np.linalg.LinAlgError,
        ) as exc:
            report.update(status="fallback", reason=str(exc))
            return original
        finally:
            self.robot_data.qpos[:] = saved
            mujoco.mj_forward(self.robot_model, self.robot_data)
            report["seconds"] = perf_counter() - started
