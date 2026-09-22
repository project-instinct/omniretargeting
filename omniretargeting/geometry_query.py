"""Rigid collision-geometry queries, independent of the optimization model.

Algorithm reference: UMR 8c4db7db (see docs/umr_scene_interaction.md).
Independently implemented; no UMR code or assets are distributed here.
The query model has nq=nv=0. Robot geoms follow original FK through mocap
bodies; scene bodies have no joints or mocap. Jacobians belong to the original
robot's nv tangent space, never to a prefix of a recompiled model's qpos.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import cKDTree


class GeometryQueryError(RuntimeError):
    """A requested collision query cannot provide reliable geometry."""


def _box_corners(half_size):
    return np.asarray(np.meshgrid(*[[-h, h] for h in half_size])).reshape(3, -1).T


def _exact_box_half_size(vertices):
    """Recognize an existing axis-aligned box without convexifying a scene."""
    vertices = np.unique(np.asarray(vertices), axis=0)
    if len(vertices) != 8:
        return None
    half_size = np.ptp(vertices, axis=0) / 2
    if np.any(half_size <= 0):
        return None
    corners = _box_corners(half_size)
    if np.max(cKDTree(vertices).query(corners)[0]) > 1e-10:
        return None
    return half_size


def _cylinder_box_face_distance(model, data, cylinder, box):
    """Exact face witness when other box faces cannot affect the distance.

    The cylinder's complete projection must stay inside both tangential box
    extents, even under a translation as long as this exit. Otherwise an edge
    or corner may determine the distance and the general query remains needed.
    """
    if (model.geom_type[cylinder] != mujoco.mjtGeom.mjGEOM_CYLINDER
            or model.geom_type[box] != mujoco.mjtGeom.mjGEOM_BOX):
        return None
    rotation = data.geom_xmat[box].reshape(3, 3)
    center = rotation.T @ (data.geom_xpos[cylinder] - data.geom_xpos[box])
    axis = rotation.T @ data.geom_xmat[cylinder].reshape(3, 3)[:, 2]
    radius, half_length = model.geom_size[cylinder, :2]
    half_size = model.geom_size[box]
    extents = half_length * np.abs(axis) + radius * np.sqrt(np.maximum(0., 1. - axis**2))
    distances = np.stack((center - extents - half_size, -center - extents - half_size))
    side, coordinate = np.unravel_index(np.argmax(distances), distances.shape)
    distance = float(distances[side, coordinate])
    tangential = np.arange(3) != coordinate
    side_margin = half_size - np.abs(center) - extents
    if np.any(side_margin[tangential] <= abs(distance) + 1e-9):
        return None
    normal = np.zeros(3)
    normal[coordinate] = 1. if side == 0 else -1.
    # Cylinder support point in the direction opposite the box's outward normal.
    direction = -normal
    axial = float(direction @ axis)
    radial = direction - axial * axis
    radial_norm = float(np.linalg.norm(radial))
    point = center + half_length * np.sign(axial) * axis
    if radial_norm > 1e-12:
        point += radius * radial / radial_norm
    surface = point - distance * normal
    return (distance, np.r_[rotation @ point + data.geom_xpos[box],
                            rotation @ surface + data.geom_xpos[box]], rotation @ normal)


def distance_direction(distance, fromto, contact_normal=None):
    """Gradient with respect to the first geom's world translation.

    At coincident witnesses use a collision-engine normal (geom1 -> geom2),
    otherwise fail explicitly. Never invent an upward or centre-to-centre exit.
    """
    fromto = np.asarray(fromto, dtype=float)
    if not np.isfinite(distance) or not np.all(np.isfinite(fromto)):
        raise GeometryQueryError("nonfinite geom distance/witness")
    vector = fromto[:3] - fromto[3:]
    length = np.linalg.norm(vector)
    if length > 1e-10:
        return vector / length * (1 if distance >= 0 else -1)
    if contact_normal is not None:
        normal = np.asarray(contact_normal, dtype=float)
        length = np.linalg.norm(normal)
        if np.all(np.isfinite(normal)) and length > 1e-10:
            return -normal / length
    raise GeometryQueryError("coincident witnesses without a reliable contact normal")


def relative_point_jacobian(
    model, data, point, body, direction, other_point=None, other_body=0
):
    """Signed distance row in nv; a static scene contributes zero columns."""
    jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jac, None, np.asarray(point), int(body))
    if other_body:
        second = np.zeros_like(jac)
        mujoco.mj_jac(
            model, data, second, None, np.asarray(other_point), int(other_body)
        )
        jac -= second
    return np.asarray(direction) @ jac


class RigidGeometryQuery:
    """Exact convex-component supplement to the original triangle query.

    Concave/open components stay on the triangle path. No convexification of a
    room or staircase is permitted. Mesh collision geoms use MuJoCo's compiled
    convex representation; visual geometry is never substituted for collision.
    """

    def __init__(self, robot, geom_ids, scene_query):
        self.robot, self.scene_query = robot, scene_query
        self.failures, self.pair_queries = [], 0
        self.distance_retries = 0
        self.analytic_face_queries = 0
        self._last_distance_normal = None
        self.parts = []
        self.open_parts = [
            (cid, part, faces)
            for cid, (part, faces, closed, _tree) in enumerate(scene_query.components)
            if not closed
        ]
        self.surface_intersection_queries = 0
        self.ambiguous_surface_intersections = 0
        spec = mujoco.MjSpec()
        spec.compiler.fusestatic = False
        spec.option.ccd_tolerance = 1e-9
        spec.option.ccd_iterations = 100
        for cid, (part, _faces, closed, _tree) in enumerate(scene_query.components):
            if not closed or not part.is_convex:
                continue
            # Every original vertex lies on/in this convex component, and every
            # face bounds it. Float conversion error is verified after compile.
            name = f"scene_{cid}"
            center = part.vertices.mean(axis=0)
            vertices = np.asarray(part.vertices) - center
            half_size = _exact_box_half_size(vertices)
            body = spec.worldbody.add_body(name=name, pos=center)
            if half_size is not None:
                # Thin, wide mesh boxes can produce a spurious exit through
                # the opposite face for a separated cylinder. The native box is
                # the same solid and avoids that degenerate mesh query.
                body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                              size=half_size, contype=2, conaffinity=1)
            else:
                spec.add_mesh(name=name, uservert=vertices.ravel(),
                              userface=np.asarray(part.faces).ravel())
                body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                              meshname=name, contype=2, conaffinity=1)
            self.parts.append((cid, name, vertices))
        self.mapping = {}
        for gid in map(int, geom_ids):
            name = f"robot_geom_{gid}"
            body = spec.worldbody.add_body(name=name, mocap=True)
            kwargs = {
                "name": name,
                "type": int(robot.geom_type[gid]),
                "size": robot.geom_size[gid].copy(),
                "contype": 1,
                "conaffinity": 2,
            }
            if robot.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(robot.geom_dataid[gid])
                va, vn = int(robot.mesh_vertadr[mid]), int(robot.mesh_vertnum[mid])
                fa, fn = int(robot.mesh_faceadr[mid]), int(robot.mesh_facenum[mid])
                spec.add_mesh(
                    name=name,
                    uservert=robot.mesh_vert[va : va + vn].ravel(),
                    userface=robot.mesh_face[fa : fa + fn].ravel(),
                )
                kwargs["meshname"] = name
            body.add_geom(**kwargs)
        if not len(geom_ids):
            raise GeometryQueryError("robot has no active collision geoms")
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        if self.model.nq or self.model.nv or self.model.njnt:
            raise GeometryQueryError(
                "query model must have no optimization coordinates"
            )
        robot_deviations = []
        for gid in map(int, geom_ids):
            name = f"robot_geom_{gid}"
            qg, qb = self.model.geom(name).id, self.model.body(name).id
            mocap = int(self.model.body_mocapid[qb])
            if (
                int(self.model.geom_bodyid[qg]) != qb
                or mocap < 0
                or self.model.geom_type[qg] != robot.geom_type[gid]
            ):
                raise GeometryQueryError(f"invalid robot geom/body mapping: {gid}")
            self.mapping[gid] = (qg, qb, mocap)
            if robot.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(robot.geom_dataid[gid])
                va, vn = int(robot.mesh_vertadr[mid]), int(robot.mesh_vertnum[mid])
                qmid = int(self.model.geom_dataid[qg])
                qa, qn = (
                    int(self.model.mesh_vertadr[qmid]),
                    int(self.model.mesh_vertnum[qmid]),
                )
                rot = np.empty(9)
                mujoco.mju_quat2Mat(rot, self.model.geom_quat[qg])
                restored = (
                    self.model.mesh_vert[qa : qa + qn] @ rot.reshape(3, 3).T
                    + self.model.geom_pos[qg]
                )
                vertices = robot.mesh_vert[va : va + vn]
                error = max(
                    cKDTree(restored).query(vertices)[0].max(),
                    cKDTree(vertices).query(restored)[0].max(),
                )
                if error > 1e-6:
                    raise GeometryQueryError(
                        f"compiled robot geom {gid} proxy deviation {error} m"
                    )
                robot_deviations.append(float(error))
            elif not np.array_equal(self.model.geom_size[qg], robot.geom_size[gid]):
                raise GeometryQueryError(f"compiled robot primitive {gid} changed size")
        self.robot_max_vertex_deviation = max(robot_deviations, default=0.0)
        self.scene_ids, deviations = {}, []
        for cid, name, vertices in self.parts:
            gid, bid = self.model.geom(name).id, self.model.body(name).id
            if self.model.body_mocapid[bid] != -1 or self.model.body_jntnum[bid]:
                raise GeometryQueryError("scene collision body is movable")
            self.scene_ids[cid] = gid
            if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                restored = _box_corners(self.model.geom_size[gid]) + self.model.geom_pos[gid]
                deviation = max(cKDTree(restored).query(vertices)[0].max(),
                                cKDTree(vertices).query(restored)[0].max())
                if deviation > 1e-10:
                    raise GeometryQueryError(f"compiled scene box deviation {deviation} m")
                deviations.append(float(deviation))
                continue
            mid = int(self.model.geom_dataid[gid])
            va, vn = (
                int(self.model.mesh_vertadr[mid]),
                int(self.model.mesh_vertnum[mid]),
            )
            # Reconstruct in the original component frame, including the mesh
            # compiler's centroid/principal-axis transform.
            rot = np.empty(9)
            mujoco.mju_quat2Mat(rot, self.model.geom_quat[gid])
            restored = (
                self.model.mesh_vert[va : va + vn] @ rot.reshape(3, 3).T
                + self.model.geom_pos[gid]
            )
            deviation = max(
                cKDTree(restored).query(vertices)[0].max(),
                cKDTree(vertices).query(restored)[0].max(),
            )
            if deviation > 1e-6:
                raise GeometryQueryError(
                    f"compiled scene proxy deviation {deviation} m"
                )
            deviations.append(float(deviation))
        self.max_vertex_deviation = max(deviations, default=0.0)

    def describe(self):
        return {
            "backend": "MuJoCo rigid collision geom pairs",
            "scene_fixed": True,
            "status": "hybrid" if self.scene_ids else "triangle_fallback_only",
            "nq": self.model.nq,
            "nv": self.model.nv,
            "component_ids": sorted(self.scene_ids),
            "triangle_fallback_components": [
                i
                for i in range(len(self.scene_query.components))
                if i not in self.scene_ids
            ],
            "proxy_max_vertex_deviation": self.max_vertex_deviation,
            "robot_proxy_max_vertex_deviation": self.robot_max_vertex_deviation,
            "robot_geom_mapping": {str(k): list(v) for k, v in self.mapping.items()},
            "pair_queries": self.pair_queries,
            "distance_retries": self.distance_retries,
            "analytic_cylinder_box_face_queries": self.analytic_face_queries,
            "failures": self.failures,
            "open_surface_test": "exact capsule-axis / finite triangle distance",
            "open_surface_supported_geoms": [
                gid for gid in self.mapping
                if self.robot.geom_type[gid] in (
                    mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE
                )
            ],
            "surface_intersection_queries": self.surface_intersection_queries,
            "ambiguous_surface_intersections": self.ambiguous_surface_intersections,
            "open_surface_solid_side_inferred": False,
        }

    def _distance(self, first, second, cutoff, witness):
        """Reject inconsistent engine witnesses before forming a constraint."""
        self._last_distance_normal = None
        analytic = _cylinder_box_face_distance(self.model, self.data, first, second)
        if analytic is not None:
            self.analytic_face_queries += 1
            distance, witness[:], self._last_distance_normal = analytic
            return distance
        flags = int(self.model.opt.disableflags)
        native_bit = int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        try:
            for attempt in range(2):
                if attempt:
                    if flags & native_bit:
                        break
                    self.distance_retries += 1
                    self.model.opt.disableflags = flags | native_bit
                witness[:] = 0
                distance = mujoco.mj_geomDistance(
                    self.model, self.data, first, second, cutoff, witness
                )
                if not np.isfinite(distance):
                    continue
                if distance >= cutoff:
                    return distance
                length = np.linalg.norm(witness[:3] - witness[3:])
                # Some native CCD cylinder queries return zero with witness
                # points metres apart. That is not a contact or a usable normal.
                if np.isfinite(witness).all() and np.isclose(
                    length, abs(distance), rtol=1e-5, atol=1e-7
                ):
                    return distance
            raise GeometryQueryError("inconsistent geom distance/witness after retry")
        finally:
            self.model.opt.disableflags = flags

    def query(self, robot_data, activation_distance):
        for gid, (_qg, _qb, mocap) in self.mapping.items():
            self.data.mocap_pos[mocap] = robot_data.geom_xpos[gid]
            mujoco.mju_mat2Quat(self.data.mocap_quat[mocap], robot_data.geom_xmat[gid])
        mujoco.mj_forward(self.model, self.data)
        rows = []
        for gid, (qg, _qb, _mocap) in self.mapping.items():
            for cid, sg in self.scene_ids.items():
                # Conservative bounding spheres, independent of MuJoCo's
                # contact candidate filtering (which can omit contained geoms).
                gap = np.linalg.norm(self.data.geom_xpos[qg] - self.data.geom_xpos[sg])
                if (
                    gap
                    > self.model.geom_rbound[qg]
                    + self.model.geom_rbound[sg]
                    + activation_distance
                ):
                    continue
                self.pair_queries += 1
                witness = np.zeros(6)
                try:
                    distance = self._distance(qg, sg, activation_distance, witness)
                    if not np.isfinite(distance):
                        raise GeometryQueryError("nonfinite geom distance")
                    if distance >= activation_distance:
                        continue
                    normal = (-self._last_distance_normal if self._last_distance_normal is not None else None)
                    contact_point = None
                    for contact in self.data.contact:
                        if self._last_distance_normal is not None:
                            break
                        if tuple(contact.geom) == (qg, sg):
                            normal = contact.frame[:3]
                            contact_point = contact.pos
                            break
                        if tuple(contact.geom) == (sg, qg):
                            normal = -contact.frame[:3]
                            contact_point = contact.pos
                            break
                    direction = distance_direction(distance, witness, normal)
                    if (
                        np.linalg.norm(witness[:3] - witness[3:]) <= 1e-10
                        and contact_point is not None
                    ):
                        witness[:3] = witness[3:] = contact_point
                    rows.append(
                        {
                            "clearance": float(distance),
                            "directions": direction,
                            "points": witness[:3].copy(),
                            "surface_points": witness[3:].copy(),
                            "geom_ids": gid,
                            "body_ids": int(self.robot.geom_bodyid[gid]),
                            "obstacle_ids": cid,
                            "surface_ids": -1,
                            "confidence": 1.0,
                            "uncertain": False,
                            "unsigned_distance": abs(distance),
                        }
                    )
                except (ValueError, RuntimeError) as exc:
                    reason = f"robot geom {gid}, scene component {cid}: {exc}"
                    if reason not in self.failures:
                        self.failures.append(reason)
                    # Keep the sample fallback, but mark the failed full-geom
                    # query uncertain: it cannot be accepted as collision free.
                    rows.append(
                        {
                            "clearance": np.nan,
                            "directions": np.zeros(3),
                            "points": robot_data.geom_xpos[gid].copy(),
                            "surface_points": np.zeros(3),
                            "geom_ids": gid,
                            "body_ids": int(self.robot.geom_bodyid[gid]),
                            "obstacle_ids": cid,
                            "surface_ids": -1,
                            "confidence": 0.0,
                            "uncertain": True,
                            "unsigned_distance": np.nan,
                        }
                    )
        rows.extend(self._open_surface_rows(robot_data, activation_distance))
        return rows

    def _open_surface_rows(self, data, activation_distance):
        from .triangle_capsule import segment_triangle_witnesses, intersecting_capsule_exit

        rows = []
        for gid in self.mapping:
            kind = self.robot.geom_type[gid]
            if kind not in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE):
                continue  # Other shapes retain the existing triangle query.
            radius = float(self.robot.geom_size[gid, 0])
            half = float(self.robot.geom_size[gid, 1]) if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
            offset = data.geom_xmat[gid].reshape(3, 3)[:, 2] * half
            start, end = data.geom_xpos[gid] - offset, data.geom_xpos[gid] + offset
            lo = np.minimum(start, end) - radius - activation_distance
            hi = np.maximum(start, end) + radius + activation_distance
            for cid, part, faces in self.open_parts:
                if np.any(hi < part.bounds[0]) or np.any(lo > part.bounds[1]):
                    continue
                ids = np.asarray(sorted(part.triangles_tree.intersection(np.r_[lo, hi])), dtype=int)
                if not len(ids):
                    continue
                self.pair_queries += 1
                self.surface_intersection_queries += 1
                axis_points, surface_points = segment_triangle_witnesses(start, end, part.triangles[ids])
                distances = np.linalg.norm(axis_points - surface_points, axis=1)
                clearances = distances - radius
                exits = {}
                for hit in np.flatnonzero(distances <= 1e-10):
                    depth, direction = intersecting_capsule_exit(start, end, radius, part.triangles[ids[hit]])
                    clearances[hit] = -depth
                    exits[int(hit)] = direction
                k = int(np.argmin(clearances))
                distance = float(distances[k])
                clearance = float(clearances[k])
                if clearance >= activation_distance:
                    continue
                ambiguous = distance <= 1e-10 and exits[k] is None
                if ambiguous:
                    self.ambiguous_surface_intersections += 1
                if distance > 1e-10:
                    direction = (axis_points[k] - surface_points[k]) / distance
                elif not ambiguous:
                    direction = exits[k]
                    translated_axis, surface = segment_triangle_witnesses(
                        start - clearance * direction, end - clearance * direction,
                        part.triangles[ids[k]][None],
                    )
                    axis_points[k] = translated_axis[0] + clearance * direction
                    surface_points[k] = surface[0]
                else:
                    direction = np.zeros(3)
                rows.append(dict(
                    clearance=clearance, directions=direction,
                    points=axis_points[k] - radius * direction,
                    surface_points=surface_points[k], geom_ids=gid,
                    body_ids=int(self.robot.geom_bodyid[gid]), obstacle_ids=cid,
                    surface_ids=int(faces[ids[k]]), confidence=0.0 if ambiguous else 1.0,
                    uncertain=ambiguous, unsigned_distance=distance,
                ))
        return rows
