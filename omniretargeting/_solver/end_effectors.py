"""Contact and free-motion end-effector objectives."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import cvxpy as cp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..contact import _is_surface_only_contact_mode
from .targets import (
    _clamp_target_to_reach,
    _source_orientation_step,
    _swing_target_z_offset,
    _transform_swing_target,
)


class EndEffectorMixin:
    """Contact and free-motion end-effector objectives.

    Internal methods sharing the owning GenericInteractionRetargeter instance; no separate state.
    """

    def _setup_end_effectors(self) -> None:
        """Resolve configured robot palm/sole markers.

        Offsets are expressed in the corresponding MuJoCo body frame.  A
        contact task may expose several surface points (for example heel,
        toe, and sole corners); the point closest to the human-derived scene
        anchor is selected when a contact run starts and then kept for the
        whole run, so the active marker cannot chatter between neighbouring
        offsets across SQP iterations or frames.  This avoids incorrectly
        forcing the ankle or wrist joint centre onto the scene surface.
        """
        cfg = self.end_effector_config
        self.end_effector_enabled = bool(
            cfg.get("enabled", False)
            or cfg.get("world_tracking_only", False)
            or cfg.get("world_tracking_enabled", False)
        )
        self.end_effectors: Dict[str, Dict[str, Any]] = {}
        self._active_contact_offsets: Dict[str, int] = {}
        if not self.end_effector_enabled:
            return

        specs = dict(cfg.get("effectors", {}) or {})
        world_tracking_enabled = bool(
            cfg.get("world_tracking_only", False)
            or cfg.get("world_tracking_enabled", False)
        )
        semantic_contact_enabled = bool(cfg.get("enabled", False))
        if world_tracking_enabled and not semantic_contact_enabled:
            selected_names = set(
                cfg.get(
                    "world_tracking_effectors",
                    ["left_foot", "right_foot"],
                )
            )
            specs = {
                name: spec for name, spec in specs.items() if name in selected_names
            }
        for name, raw_spec in specs.items():
            spec = dict(raw_spec or {})
            body_name = spec.get("robot_body")
            if not body_name:
                raise ValueError(
                    f"End-effector '{name}' is missing required 'robot_body'."
                )
            body_id = mujoco.mj_name2id(
                self.robot_model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)
            )
            if body_id < 0:
                raise ValueError(
                    f"End-effector '{name}' refers to missing robot body "
                    f"'{body_name}'."
                )

            reach_reference_body = spec.get("reach_reference_body")
            reach_reference_body_id = -1
            if reach_reference_body:
                reach_reference_body_id = mujoco.mj_name2id(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    str(reach_reference_body),
                )
                if reach_reference_body_id < 0:
                    raise ValueError(
                        f"End-effector '{name}' refers to missing reach reference "
                        f"body '{reach_reference_body}'."
                    )

            tracking_offset = np.asarray(
                spec.get("tracking_offset", [0.0, 0.0, 0.0]), dtype=float
            ).reshape(3)
            contact_offsets = np.asarray(
                spec.get("contact_offsets", [tracking_offset.tolist()]), dtype=float
            )
            if contact_offsets.ndim == 1:
                contact_offsets = contact_offsets[None, :]
            if contact_offsets.ndim != 2 or contact_offsets.shape[1] != 3:
                raise ValueError(
                    f"End-effector '{name}' contact_offsets must have shape (K, 3)."
                )

            contact_joint_qa_indices: List[int] = []
            for joint_name in spec.get("contact_joint_names", []):
                joint_id = mujoco.mj_name2id(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    str(joint_name),
                )
                if joint_id < 0:
                    raise ValueError(
                        f"End-effector '{name}' contact chain refers to "
                        f"missing joint '{joint_name}'."
                    )
                qpos_idx = int(self.robot_model.jnt_qposadr[joint_id])
                matches = np.flatnonzero(self.q_a_indices == qpos_idx)
                if len(matches):
                    contact_joint_qa_indices.append(int(matches[0]))

            self.end_effectors[name] = {
                "body_name": str(body_name),
                "body_id": int(body_id),
                "tracking_offset": tracking_offset,
                "contact_offsets": contact_offsets,
                "contact_joint_qa_indices": np.asarray(
                    contact_joint_qa_indices, dtype=int
                ),
                "track_orientation": bool(spec.get("track_orientation", False)),
                "track_swing_heading": bool(
                    spec.get("track_swing_heading", False)
                ),
                "track_swing_orientation": bool(
                    spec.get("track_swing_orientation", False)
                ),
                "orientation_sigma": float(
                    spec.get(
                        "orientation_sigma",
                        cfg.get("orientation_sigma", np.deg2rad(7.5)),
                    )
                ),
                "max_orientation_step": float(
                    spec.get("max_orientation_step", np.deg2rad(6.0))
                ),
                "orientation_step_reference": str(spec.get("orientation_step_reference", "source")),
                "contact_sigma": float(
                    spec.get("contact_sigma", cfg.get("contact_sigma", 0.025))
                ),
                "swing_sigma": float(
                    spec.get("swing_sigma", cfg.get("swing_sigma", 0.06))
                ),
                "swing_morphology_scale": float(
                    spec.get(
                        "swing_morphology_scale",
                        cfg.get("swing_morphology_scale", 1.0),
                    )
                ),
                "preserve_world_target": bool(
                    spec.get("preserve_world_target", False)
                ),
                "track_swing": bool(spec.get("track_swing", True)),
                "track_sliding_tangent": bool(
                    spec.get("track_sliding_tangent", True)
                ),
                "track_transition_tangent": bool(
                    spec.get("track_transition_tangent", True)
                ),
                "reach_reference_body_id": int(reach_reference_body_id),
                "reach_reference_offset": np.asarray(
                    spec.get("reach_reference_offset", [0.0, 0.0, 0.0]),
                    dtype=float,
                ).reshape(3),
                "max_swing_reach": (
                    None
                    if spec.get("max_swing_reach") is None
                    else float(spec["max_swing_reach"])
                ),
            }

        if not self.end_effectors:
            raise ValueError(
                "end_effector_priority is enabled but no effectors were configured."
            )
        mode_label = (
            "Mixed contact/world-tracking end effectors"
            if semantic_contact_enabled and world_tracking_enabled
            else (
                "Contact-priority end effectors"
                if semantic_contact_enabled
                else "World-tracking end effectors"
            )
        )
        print(
            f"{mode_label}: "
            + ", ".join(
                f"{name}->{spec['body_name']}" for name, spec in self.end_effectors.items()
            )
        )


    def _build_end_effector_objectives(
        self,
        dqa: cp.Variable,
        targets: Optional[Mapping[str, Mapping[str, Any]]],
        last_rotations: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Tuple[
        Optional[cp.Expression],
        Optional[cp.Expression],
        Optional[cp.Expression],
        List[cp.Constraint],
        int,
        int,
        int,
    ]:
        """Linearise contact and swing tracking objectives for one SQP step.

        Contact and swing expressions are deliberately returned separately so
        the caller can solve them lexicographically.  The normalization by a
        metric tolerance ``sigma`` makes each squared residual dimensionless.
        """
        self._protected_endpoint_residuals = {}
        if not self.end_effector_enabled or not targets:
            return None, None, None, [], 0, 0, 0

        contact_terms: List[cp.Expression] = []
        swing_terms: List[cp.Expression] = []
        swing_orientation_terms: List[cp.Expression] = []
        contact_constraints: List[cp.Constraint] = []
        for name, target_data in targets.items():
            if name not in self.end_effectors:
                continue
            spec = self.end_effectors[name]
            body_id = spec["body_id"]
            confidence = float(np.clip(target_data.get("confidence", 1.0), 0.0, 1.0))
            transition_weight = float(
                np.clip(target_data.get("transition_weight", 1.0), 0.0, 1.0)
            )
            contact_mode = str(
                target_data.get(
                    "contact_mode",
                    "sticking" if target_data.get("contact", False) else "none",
                )
            )
            forward_direction = target_data.get("forward_direction")
            up_direction = target_data.get("up_direction")

            if bool(target_data.get("contact", False)):
                target = np.asarray(target_data["contact_position"], dtype=float).reshape(3)
                offset_index = self._active_contact_offsets.get(name)
                if offset_index is None:
                    offset_index = int(
                        np.argmin(
                            [
                                np.linalg.norm(
                                    self._point_position_and_jacobian(body_id, offset)[0]
                                    - target
                                )
                                for offset in spec["contact_offsets"]
                            ]
                        )
                    )
                    self._active_contact_offsets[name] = offset_index
                position, jacobian = self._point_position_and_jacobian(
                    body_id, spec["contact_offsets"][offset_index]
                )
                sigma = max(float(spec["contact_sigma"]), 1e-6)
                # Keep every detected contact in the primary level. Confidence
                # only ranks contacts within that same priority level.
                weight = max(confidence, 0.25) * max(transition_weight, 1e-3)
                residual = position + jacobian @ dqa - target
                self._protected_endpoint_residuals[name] = residual
                surface_normal = np.asarray(
                    target_data.get("surface_normal", [0.0, 0.0, 1.0]),
                    dtype=float,
                ).reshape(3).copy()
                normal_norm = float(np.linalg.norm(surface_normal))
                if normal_norm > 1e-8:
                    surface_normal /= normal_norm
                if (
                    _is_surface_only_contact_mode(contact_mode)
                    and normal_norm > 1e-8
                ):
                    self._protected_endpoint_residuals[name] = surface_normal @ residual
                    contact_terms.append(
                        np.sqrt(weight) * (surface_normal @ residual) / sigma
                    )
                    if (
                        contact_mode == "sliding"
                        and spec["track_sliding_tangent"]
                    ):
                        tangent_projection = np.eye(3) - np.outer(
                            surface_normal, surface_normal
                        )
                        tangent_sigma = max(
                            float(
                                spec.get(
                                    "sliding_tangent_sigma",
                                    self.end_effector_config.get(
                                        "sliding_tangent_sigma", 0.10
                                    ),
                                )
                            ),
                            sigma,
                        )
                        swing_terms.append(
                            np.sqrt(weight)
                            * (tangent_projection @ residual)
                            / tangent_sigma
                        )
                    elif (
                        contact_mode == "transition"
                        and spec["track_transition_tangent"]
                    ):
                        # A source-labelled touchdown/release provides useful
                        # contact timing even when video reconstruction puts
                        # the foot above the surface. Constrain its scene-normal
                        # component at contact priority, while retaining the
                        # captured world-space tangent as a lower swing task.
                        # This avoids both a false no-slip anchor and an
                        # unconstrained 20+ cm lateral foot drift.
                        tracking_target = np.asarray(
                            target_data["tracking_position"], dtype=float
                        ).reshape(3)
                        tangent_projection = np.eye(3) - np.outer(
                            surface_normal, surface_normal
                        )
                        transition_residual = (
                            position + jacobian @ dqa - tracking_target
                        )
                        transition_sigma = max(
                            float(spec["swing_sigma"]), sigma
                        )
                        swing_terms.append(
                            np.sqrt(weight)
                            * (tangent_projection @ transition_residual)
                            / transition_sigma
                        )
                else:
                    contact_terms.append(np.sqrt(weight) * residual / sigma)
                if normal_norm > 1e-8:
                    normal_tolerance = max(
                        float(
                            self.end_effector_config.get(
                                "contact_normal_tolerance", 0.005
                            )
                        ),
                        0.0,
                    )
                    # The detector keeps this normal on the pre-contact side
                    # even when noisy video reconstruction penetrates the
                    # scene. Keep the robot marker on that outside half-space.
                    contact_constraints.append(
                        surface_normal @ residual >= -normal_tolerance
                    )

                    if (
                        contact_mode == "sticking"
                        and spec.get("track_orientation", False)
                        and forward_direction is not None
                    ):
                        forward = np.asarray(forward_direction, dtype=float).reshape(3).copy()
                        orientation_up = np.asarray(
                            up_direction
                            if up_direction is not None
                            else surface_normal,
                            dtype=float,
                        ).reshape(3).copy()
                        orientation_up /= max(
                            float(np.linalg.norm(orientation_up)), 1e-10
                        )
                        # The contact detector has already verified that the
                        # captured sole is compatible with the surface.  Track
                        # the captured rigid foot frame rather than replacing
                        # its pitch/roll with a noisy mesh-triangle normal.
                        forward -= orientation_up * float(
                            np.dot(orientation_up, forward)
                        )
                        forward_norm = float(np.linalg.norm(forward))
                        if forward_norm > 1e-8:
                            forward /= forward_norm
                            lateral = np.cross(orientation_up, forward)
                            lateral /= max(float(np.linalg.norm(lateral)), 1e-10)
                            forward = np.cross(lateral, orientation_up)
                            target_rotation = np.column_stack(
                                [forward, lateral, orientation_up]
                            )
                            current_rotation = self.robot_data.xmat[body_id].reshape(
                                3, 3
                            )
                            rotation_error = Rotation.from_matrix(
                                target_rotation @ current_rotation.T
                            ).as_rotvec()
                            angular_jacobian = self._body_angular_jacobian(body_id)
                            orientation_sigma = max(
                                float(
                                    spec.get(
                                        "orientation_sigma",
                                        self.end_effector_config.get(
                                            "orientation_sigma", np.deg2rad(7.5)
                                        ),
                                    )
                                ),
                                1e-6,
                            )
                            contact_terms.append(
                                np.sqrt(weight)
                                * (angular_jacobian @ dqa - rotation_error)
                                / orientation_sigma
                            )
            else:
                self._active_contact_offsets.pop(name, None)
                if not spec["track_swing"]:
                    continue
                target = np.asarray(target_data["tracking_position"], dtype=float).reshape(3)
                source_root_position = target_data.get("source_root_position")
                swing_morphology_scale = float(spec["swing_morphology_scale"])
                root_z_offset = _swing_target_z_offset(
                    self.end_effector_config,
                    robot_root_z=float(self.robot_data.qpos[2]),
                    source_root_position=source_root_position,
                )
                # Contact positions remain exact world-space constraints.
                # Swing targets follow the registered robot root and may also
                # be scaled root-relatively for morphology.  Root registration
                # still applies when morphology_scale == 1.0.
                target = _transform_swing_target(
                    target,
                    source_root_position,
                    root_z_offset,
                    swing_morphology_scale,
                    bool(spec["preserve_world_target"]),
                )
                target_jacobian = np.zeros((3, self.nq_a), dtype=float)
                if (
                    not spec["preserve_world_target"]
                    and source_root_position is not None
                    and self.end_effector_config.get("swing_target_z_offset") is None
                ):
                    # The registered free-hand target moves with root height.
                    # Linearize both sides of the residual, otherwise SQP can
                    # chase a target that rises with the robot at every step.
                    target_jacobian[2, self.q_a_indices == 2] = 1.0
                reach_body_id = int(spec["reach_reference_body_id"])
                if reach_body_id >= 0:
                    reach_origin, origin_jacobian = self._point_position_and_jacobian(
                        reach_body_id, spec["reach_reference_offset"]
                    )
                    delta = target - reach_origin
                    distance = float(np.linalg.norm(delta))
                    reach = spec["max_swing_reach"]
                    if reach is not None and 0.0 < reach < distance and distance > 1e-10:
                        direction = delta / distance
                        projection = (reach / distance) * (
                            np.eye(3) - np.outer(direction, direction)
                        )
                        target_jacobian = origin_jacobian + projection @ (
                            target_jacobian - origin_jacobian
                        )
                    target = _clamp_target_to_reach(
                        target,
                        reach_origin,
                        spec["max_swing_reach"],
                    )
                position, jacobian = self._point_position_and_jacobian(
                    body_id, spec["tracking_offset"]
                )
                sigma = max(float(spec["swing_sigma"]), 1e-6)
                residual = position - target + (jacobian - target_jacobian) @ dqa
                if spec["preserve_world_target"] or name in ("left_foot", "right_foot"):
                    self._protected_endpoint_residuals[name] = residual
                swing_terms.append(residual / sigma)
                if (
                    spec.get("track_orientation", False)
                    and spec.get("track_swing_orientation", False)
                    and forward_direction is not None
                    and up_direction is not None
                ):
                    target_up = np.asarray(up_direction, dtype=float).reshape(3).copy()
                    target_up /= max(float(np.linalg.norm(target_up)), 1e-10)
                    target_forward = np.asarray(forward_direction, dtype=float).reshape(3).copy()
                    target_forward -= target_up * float(
                        np.dot(target_up, target_forward)
                    )
                    forward_norm = float(np.linalg.norm(target_forward))
                    if forward_norm > 1e-8:
                        target_forward /= forward_norm
                        target_lateral = np.cross(target_up, target_forward)
                        target_lateral /= max(
                            float(np.linalg.norm(target_lateral)), 1e-10
                        )
                        target_forward = np.cross(target_lateral, target_up)
                        target_rotation = np.column_stack(
                            [target_forward, target_lateral, target_up]
                        )
                        current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
                        rotation_error = Rotation.from_matrix(
                            target_rotation @ current_rotation.T
                        ).as_rotvec()
                        angular_jacobian = self._body_angular_jacobian(body_id)
                        orientation_sigma = max(float(spec["orientation_sigma"]), 1e-6)
                        swing_orientation_terms.append(
                            (angular_jacobian @ dqa - rotation_error)
                            / orientation_sigma
                        )
                elif (
                    spec.get("track_orientation", False)
                    and spec.get("track_swing_heading", False)
                    and forward_direction is not None
                ):
                    target_forward = np.asarray(forward_direction, dtype=float).reshape(3).copy()
                    target_forward[2] = 0.0
                    current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
                    current_forward = current_rotation[:, 0].copy()
                    current_forward[2] = 0.0
                    target_norm = float(np.linalg.norm(target_forward))
                    current_norm = float(np.linalg.norm(current_forward))
                    if target_norm > 1e-8 and current_norm > 1e-8:
                        target_forward /= target_norm
                        current_forward /= current_norm
                        yaw_error = float(
                            np.arctan2(
                                np.cross(current_forward, target_forward)[2],
                                np.dot(current_forward, target_forward),
                            )
                        )
                        angular_jacobian = self._body_angular_jacobian(body_id)
                        orientation_sigma = max(float(spec["orientation_sigma"]), 1e-6)
                        swing_orientation_terms.append(
                            cp.hstack([angular_jacobian[2] @ dqa - yaw_error])
                            / orientation_sigma
                        )

            # Bound added rotation error, preserving the source's intentional
            # fast rotation (e.g. airborne feet during a flip). "absolute"
            # remains available for an explicitly configured physical cap.
            if (
                spec.get("track_orientation", False)
                and last_rotations is not None
                and name in last_rotations
            ):
                current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
                rotation_since_last = Rotation.from_matrix(
                    current_rotation @ np.asarray(last_rotations[name]).T
                ).as_rotvec()
                angular_jacobian = self._body_angular_jacobian(body_id)
                max_orientation_step = max(
                    float(spec.get("max_orientation_step", np.deg2rad(6.0))),
                    1e-6,
                )
                step_reference = spec.get("orientation_step_reference", "source")
                if step_reference not in {"source", "absolute"}:
                    raise ValueError("orientation_step_reference must be 'source' or 'absolute'.")
                source_step = _source_orientation_step(target_data) if step_reference == "source" else None
                if source_step is None:
                    source_step = np.zeros(3)
                contact_constraints.append(
                    cp.norm(rotation_since_last + angular_jacobian @ dqa - source_step, 2)
                    <= max_orientation_step
                )

        contact_residual = cp.hstack(contact_terms) if contact_terms else None
        swing_residual = cp.hstack(swing_terms) if swing_terms else None
        swing_orientation_residual = (
            cp.hstack(swing_orientation_terms)
            if swing_orientation_terms
            else None
        )
        # A foot pose is one SE(3) task: translating the sole marker and
        # rotating its rigid frame are coupled by the ankle kinematics.  When
        # requested, optimize both in one dimensionless Pareto level using the
        # configured metre/radian sigmas.  Separating them lexicographically
        # makes the orientation level inherit an already saturated ankle and
        # can leave the toe axis tens of degrees from the captured foot.
        if (
            bool(self.end_effector_config.get("combine_swing_foot_pose", False))
            and swing_residual is not None
            and swing_orientation_residual is not None
        ):
            swing_residual = cp.hstack(
                [swing_residual, swing_orientation_residual]
            )
            swing_orientation_residual = None
        return (
            contact_residual,
            swing_residual,
            swing_orientation_residual,
            contact_constraints,
            len(contact_terms),
            len(swing_terms),
            len(swing_orientation_terms),
        )
