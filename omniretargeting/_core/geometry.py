"""Robot end-effector geometry and collision-point sampling."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..scene_geometry import geom_surface_offsets


class RobotGeometryMixin:
    """Robot end-effector geometry and collision-point sampling.

    Internal methods sharing the owning OmniRetargeter instance; no separate state.
    """

    def _detect_robot_height(self) -> float:
        """
        Detect robot height from URDF by calculating the vertical span of the robot in its default configuration.

        Since this is a floating-base robot, we assume the default configuration puts the robot
        in a nominal pose (e.g. standing). We calculate the difference between the highest
        and lowest points of all visual meshes to get the full height.
        """
        # Use MuJoCo to get robot height in default configuration
        # This is more reliable than parsing the URDF scene graph

        # Set robot to default configuration (zeros)
        mujoco.mj_resetData(self.robot_model, self.robot_data)
        self.robot_data.qpos[3:7] = [1, 0, 0, 0]  # Set base quaternion to identity
        mujoco.mj_forward(self.robot_model, self.robot_data)

        min_z = float('inf')
        max_z = float('-inf')

        # Iterate through all bodies and get their positions
        for body_idx in range(self.robot_model.nbody):
            body_pos = self.robot_data.xpos[body_idx]
            z = body_pos[2]
            min_z = min(min_z, z)
            max_z = max(max_z, z)

        # Also check geometry positions for more accuracy
        for geom_idx in range(self.robot_model.ngeom):
            geom_pos = self.robot_data.geom_xpos[geom_idx]
            z = geom_pos[2]

            # Get geometry size to account for extent
            geom_size = self.robot_model.geom_size[geom_idx]
            geom_type = self.robot_model.geom_type[geom_idx]

            # Estimate vertical extent based on geometry type
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius = geom_size[0]
                min_z = min(min_z, z - radius)
                max_z = max(max_z, z + radius)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius = geom_size[0]
                half_height = geom_size[1]
                min_z = min(min_z, z - half_height - radius)
                max_z = max(max_z, z + half_height + radius)
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                half_size = geom_size[2]  # Z dimension
                min_z = min(min_z, z - half_size)
                max_z = max(max_z, z + half_size)
            else:
                # For other types, just use the position
                min_z = min(min_z, z)
                max_z = max(max_z, z)

        height = max_z - min_z

        # Sanity check
        if height < 0.3 or height > 3.0:
            print(f"Warning: Detected height {height:.3f}m seems unreasonable. Using default 1.6m")
            return 1.6

        print(f"Detected robot height: {height:.4f}m (Min Z: {min_z:.4f}, Max Z: {max_z:.4f})")

        return height


    @staticmethod
    def _body_name_matches_side(body_name: str, side: str) -> bool:
        """Match an explicit left/right name token, never an interior substring."""
        normalized = str(body_name).lower().replace("-", "_")
        tokens = tuple(token for token in normalized.split("_") if token)
        aliases = {"left", "l"} if side == "left" else {"right", "r"}
        return bool(aliases.intersection(tokens))


    def _resolve_foot_body_ids(self, cfg: Dict[str, Any]) -> Dict[str, int]:
        """Resolve left/right foot body ids from config, mapping, or robot body names."""
        resolved = {}
        explicit = dict(cfg.get("body_names", {}) or {})

        for side, joint_candidates in {
            "left": ["L_Foot", "L_Ankle"],
            "right": ["R_Foot", "R_Ankle"],
        }.items():
            body_id = -1
            explicit_name = explicit.get(side)
            if explicit_name:
                body_id = self._body_name_to_id(explicit_name)

            if body_id < 0:
                for joint_name in joint_candidates:
                    body_name = getattr(self, "valid_joint_mapping", {}).get(joint_name)
                    if body_name:
                        body_id = self._body_name_to_id(body_name)
                        if body_id >= 0:
                            break

            if body_id < 0:
                body_id = self._search_body_id_by_keywords(side)

            resolved[side] = body_id

        return resolved


    def _body_name_to_id(self, body_name: str) -> int:
        """Convert a body name to a MuJoCo id."""
        try:
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except Exception:
            return -1
        return int(body_id) if body_id is not None and body_id >= 0 else -1


    def _search_body_id_by_keywords(self, side: str) -> int:
        """Fallback search for foot/ankle body ids when mappings are incomplete."""
        side_tokens = ("left", "l_") if side == "left" else ("right", "r_")
        candidates = []
        for body_idx in range(self.robot_model.nbody):
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_idx)
            if not body_name:
                continue
            name_lower = body_name.lower()
            if not any(token in name_lower for token in side_tokens):
                continue
            if "foot" in name_lower:
                return body_idx
            if "ankle" in name_lower:
                candidates.append(body_idx)
        return candidates[0] if candidates else -1


    def _infer_end_effector_geometry(self) -> None:
        """Fill omitted palm/sole markers from the robot's own geometry.

        Palm tracking uses the end-effector geom centroid. Contact candidates
        cover its forward, outward, lower, and upper surface extremes. A sole
        marker uses the lowest center of the collision primitive nearest the
        foot body's sagittal plane. Explicit profile values still win.
        """
        retargeting = dict(self.retargeting_config)
        priority = dict(retargeting.get("end_effector_priority", {}) or {})
        raw_effectors = dict(priority.get("effectors", {}) or {})
        if self.robot_model.ngeom and getattr(self, "valid_joint_mapping", None):
            from ..terminal_effectors import infer_terminal_effectors

            raw_effectors = infer_terminal_effectors(
                self.robot_model, self.valid_joint_mapping, raw_effectors
            )
            priority["effectors"] = raw_effectors
        if not raw_effectors:
            return

        # Resolve robot anatomy at the profile boundary. Solver scopes consume
        # this explicit grouping, never names of source rigs or effectors.
        feet = self._resolve_foot_body_ids(self._get_foot_stabilization_config())
        foot_welds = {int(self.robot_model.body_weldid[b]) for b in feet.values() if b > 0}
        effectors: Dict[str, Dict[str, Any]] = {}
        for name, raw_spec in raw_effectors.items():
            spec = dict(raw_spec or {})
            body_id = self._body_name_to_id(str(spec.get("robot_body", "")))
            ancestor = body_id
            while ancestor > 0 and int(self.robot_model.body_weldid[ancestor]) not in foot_welds:
                ancestor = int(self.robot_model.body_parentid[ancestor])
            if ancestor > 0 and len(foot_welds) == 2:
                # Terminal inference can follow an unmapped distal joint,
                # e.g. an ankle roll after the mapped ankle pitch body.
                spec.setdefault("local_repair_group", "lower_limbs")
                from ..foot_frames import robot_sole_directions
                try:
                    forward, up = robot_sole_directions(self.robot_model, body_id)
                except ValueError as error:
                    spec["sole_orientation_calibration"] = {"status": "unavailable", "reason": str(error)}
                else:
                    spec.setdefault("track_orientation", True)
                    spec.setdefault("orientation_local_forward", forward.tolist())
                    spec.setdefault("orientation_local_up", up.tolist())
                    spec["sole_orientation_calibration"] = {"status": "geometry"}
            tracking_offset = spec.get("tracking_offset")
            contact_offsets = spec.get("contact_offsets")
            if tracking_offset is not None and contact_offsets is not None:
                pass
            elif tracking_offset is not None:
                spec.setdefault("contact_offsets", [tracking_offset])
            elif contact_offsets is not None:
                contacts = np.asarray(contact_offsets, dtype=float).reshape(-1, 3)
                if len(contacts):
                    spec["tracking_offset"] = contacts[0].tolist()
            else:
                body_name = spec.get("robot_body")
                body_id = (
                    -1 if not body_name else self._body_name_to_id(str(body_name))
                )
                inferred = None
                semantic_name = str(name).lower()
                if body_id >= 0 and "hand" in semantic_name:
                    inferred = self._infer_hand_marker_offsets(
                        body_id, semantic_name
                    )
                elif body_id >= 0 and "foot" in semantic_name:
                    inferred = self._infer_foot_marker_offset(body_id)
                if inferred is None:
                    raise ValueError(
                        f"End-effector '{name}' has no explicit offsets and its "
                        "robot geometry could not provide them."
                    )
                spec["tracking_offset"], spec["contact_offsets"] = inferred

            semantic_name = str(name).lower()
            if "hand" in semantic_name:
                body_name = spec.get("robot_body")
                body_id = (
                    -1 if not body_name else self._body_name_to_id(str(body_name))
                )
                raw_arm_posture = dict(spec.get("arm_posture", {}) or {})
                needs_chain = bool(raw_arm_posture.get("enabled", False)) or bool(
                    spec.get("refine_orientation", False)
                )
                if body_id >= 0 and needs_chain:
                    inferred_chain = self._infer_arm_chain_geometry(body_id)
                    if inferred_chain is None:
                        raise ValueError(
                            f"End-effector '{name}' arm chain could not be "
                            "inferred from the robot model."
                        )
                    spec.setdefault(
                        "contact_joint_names",
                        inferred_chain["contact_joint_names"],
                    )
                    spec.setdefault(
                        "orientation_joint_tokens",
                        inferred_chain["orientation_joint_tokens"],
                    )
                    if inferred_chain.get("orientation_local_forward") is not None:
                        spec.setdefault(
                            "orientation_local_forward",
                            inferred_chain["orientation_local_forward"],
                        )
                    if inferred_chain.get("orientation_local_up") is not None:
                        spec.setdefault(
                            "orientation_local_up",
                            inferred_chain["orientation_local_up"],
                        )
                    for key, value in inferred_chain["arm_posture"].items():
                        raw_arm_posture.setdefault(key, value)
                    spec["arm_posture"] = raw_arm_posture
            effectors[name] = spec

        priority["effectors"] = effectors
        retargeting["end_effector_priority"] = priority
        self.retargeting_config = retargeting


    def _infer_hand_marker_offsets(
        self, body_id: int, semantic_name: str
    ) -> Tuple[List[float], List[List[float]]] | None:
        geom_ids = np.flatnonzero(self.robot_model.geom_bodyid == body_id)
        mesh_ids = [
            int(geom_id)
            for geom_id in geom_ids
            if int(self.robot_model.geom_type[geom_id])
            == mujoco.mjtGeom.mjGEOM_MESH
        ]
        selected_ids = mesh_ids or [int(geom_id) for geom_id in geom_ids]
        point_sets = [
            self._sample_geom_points_in_body_frame(geom_id)
            for geom_id in selected_ids
        ]
        point_sets = [points for points in point_sets if points.size]
        if not point_sets:
            return None

        points = np.vstack(point_sets)
        centroid = np.mean(points, axis=0)
        outward_index = (
            int(np.argmin(points[:, 1]))
            if "left" in semantic_name
            else int(np.argmax(points[:, 1]))
        )
        indices = [
            int(np.argmax(points[:, 0])),
            outward_index,
            int(np.argmin(points[:, 2])),
            int(np.argmax(points[:, 2])),
        ]
        tracking = np.round(centroid, decimals=4)
        contacts = np.round(
            np.vstack([tracking, points[indices]]), decimals=4
        )
        return tracking.tolist(), contacts.tolist()


    def _infer_arm_chain_geometry(self, hand_body_id: int) -> Dict[str, Any] | None:
        """Infer a hand's actuated arm chain and segment geometry."""
        distal_chain: List[Tuple[int, int, str]] = []
        body_id = int(hand_body_id)
        while body_id > 0:
            for joint_id in np.flatnonzero(self.robot_model.jnt_bodyid == body_id):
                joint_id = int(joint_id)
                if (
                    int(self.robot_model.jnt_type[joint_id])
                    == mujoco.mjtJoint.mjJNT_FREE
                ):
                    continue
                joint_name = mujoco.mj_id2name(
                    self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
                if joint_name:
                    distal_chain.append((body_id, joint_id, str(joint_name)))
            body_id = int(self.robot_model.body_parentid[body_id])

        shoulder_entries = [
            entry for entry in distal_chain if "shoulder" in entry[2].lower()
        ]
        elbow_entries = [
            entry for entry in distal_chain if "elbow" in entry[2].lower()
        ]
        wrist_entries = [
            entry for entry in distal_chain if "wrist" in entry[2].lower()
        ]
        if not shoulder_entries or not elbow_entries or not wrist_entries:
            return None

        shoulder_entry = shoulder_entries[-1]
        elbow_entry = elbow_entries[0]
        wrist_entry = wrist_entries[0]
        shoulder_index = distal_chain.index(shoulder_entry)
        arm_chain = list(reversed(distal_chain[: shoulder_index + 1]))
        contact_joint_names = [entry[2] for entry in arm_chain]
        orientation_joint_tokens = [
            joint_name
            for joint_name in contact_joint_names
            if "wrist" in joint_name.lower()
        ]

        nominal_data = mujoco.MjData(self.robot_model)
        nominal_data.qpos[:] = self.robot_model.qpos0
        mujoco.mj_forward(self.robot_model, nominal_data)
        shoulder_body_id = int(shoulder_entry[0])
        elbow_body_id = int(elbow_entry[0])
        wrist_body_id = int(wrist_entry[0])
        upper_arm_length = float(
            np.linalg.norm(
                nominal_data.xpos[elbow_body_id]
                - nominal_data.xpos[shoulder_body_id]
            )
        )
        forearm_length = float(
            np.linalg.norm(
                nominal_data.xpos[wrist_body_id]
                - nominal_data.xpos[elbow_body_id]
            )
        )
        hand_rotation = nominal_data.xmat[hand_body_id].reshape(3, 3)

        def wrist_axis(token: str, sign: float = 1.0) -> List[float] | None:
            matches = [
                entry
                for entry in wrist_entries
                if token in entry[2].lower()
            ]
            if not matches:
                return None
            local_axis = (
                hand_rotation.T
                @ nominal_data.xaxis[int(matches[0][1])]
                * float(sign)
            )
            local_axis /= max(float(np.linalg.norm(local_axis)), 1e-12)
            local_axis[np.abs(local_axis) < 1e-8] = 0.0
            return local_axis.tolist()

        return {
            "contact_joint_names": contact_joint_names,
            "orientation_joint_tokens": orientation_joint_tokens,
            "orientation_local_forward": wrist_axis("roll"),
            "orientation_local_up": wrist_axis("pitch", sign=-1.0),
            "arm_posture": {
                "shoulder_body": mujoco.mj_id2name(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    shoulder_body_id,
                ),
                "elbow_body": mujoco.mj_id2name(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    elbow_body_id,
                ),
                "wrist_body": mujoco.mj_id2name(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    wrist_body_id,
                ),
                "upper_arm_length": upper_arm_length,
                "forearm_length": forearm_length,
            },
        }


    def _infer_foot_marker_offset(
        self, body_id: int
    ) -> Tuple[List[float], List[List[float]]] | None:
        primitive_ids = [
            int(geom_id)
            for geom_id in np.flatnonzero(self.robot_model.geom_bodyid == body_id)
            if int(self.robot_model.geom_type[geom_id])
            != mujoco.mjtGeom.mjGEOM_MESH
        ]
        if primitive_ids:
            geom_id = min(
                primitive_ids,
                key=lambda candidate: (
                    abs(float(self.robot_model.geom_pos[candidate, 1])),
                    -float(self.robot_model.geom_size[candidate, 1]),
                ),
            )
            points = self._sample_geom_points_in_body_frame(geom_id)
        else:
            points = self._collect_body_contact_points(body_id)
        if not points.size:
            return None

        min_z = float(np.min(points[:, 2]))
        lowest = points[points[:, 2] <= min_z + 1e-5]
        marker = np.round(np.mean(lowest, axis=0), decimals=4)
        return marker.tolist(), [marker.tolist()]


    def _collect_body_contact_points(self, body_id: int) -> np.ndarray:
        """Collect representative support points for a foot body in body coordinates."""
        geom_ids = [geom_id for geom_id in range(self.robot_model.ngeom) if int(self.robot_model.geom_bodyid[geom_id]) == body_id]
        if not geom_ids:
            return np.zeros((0, 3), dtype=float)

        primitive_geom_ids = [
            geom_id for geom_id in geom_ids
            if int(self.robot_model.geom_type[geom_id]) != mujoco.mjtGeom.mjGEOM_MESH
        ]
        if primitive_geom_ids:
            geom_ids = primitive_geom_ids

        points = [self._sample_geom_points_in_body_frame(geom_id) for geom_id in geom_ids]
        points = [pts for pts in points if pts.size > 0]
        if not points:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(points)


    def _sample_geom_points_in_body_frame(self, geom_id: int) -> np.ndarray:
        """Sample support candidate points for one geom in the owning body frame."""
        points_local = geom_surface_offsets(self.robot_model, geom_id)

        geom_pos = np.asarray(self.robot_model.geom_pos[geom_id], dtype=float)
        geom_quat = np.asarray(self.robot_model.geom_quat[geom_id], dtype=float)
        geom_quat_xyzw = np.array([geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]], dtype=float)
        geom_rot = Rotation.from_quat(geom_quat_xyzw).as_matrix()

        return points_local @ geom_rot.T + geom_pos


    def _select_collision_probe_points(self, sample_points: np.ndarray) -> np.ndarray:
        """Reduce dense foot samples to a compact set of boundary probes for collision checks."""
        if sample_points.size == 0:
            return np.zeros((0, 3), dtype=float)
        if len(sample_points) <= 24:
            return np.array(sample_points, copy=True)

        directions = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ], dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        chosen = set()
        for direction in directions:
            chosen.add(int(np.argmax(sample_points @ direction)))

        local_min_z = np.min(sample_points[:, 2])
        bottom_band = np.where(sample_points[:, 2] <= local_min_z + 0.01)[0]
        chosen.update(int(idx) for idx in bottom_band)

        chosen_points = sample_points[sorted(chosen)]
        unique_points = np.unique(np.round(chosen_points, decimals=6), axis=0)
        return unique_points.astype(float, copy=False)
