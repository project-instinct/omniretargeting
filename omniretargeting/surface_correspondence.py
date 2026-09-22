"""Deterministic anatomical surface bindings; not learned UMR correspondence.

Only explicitly mapped limb regions participate. The shared calibrated rest
pose gives anatomical axes; normalized regional coordinates select a point on
the robot's actual collision shell. Normal agreement rejects wrong-side maps.
Normalization selects correspondence only: neither motion nor robot is scaled.
"""

from dataclasses import dataclass

import igl
import mujoco
import numpy as np
import trimesh

from .scene_geometry import collision_geom_ids, geom_surface_offsets
from .surface_tasks import SurfaceBinding, _face_frames


@dataclass
class SampledPart:
    name: str
    effector: str | None
    points: np.ndarray
    normals: np.ndarray
    frames: np.ndarray
    valid: np.ndarray
    bindings: tuple
    robot_frames: np.ndarray
    report: dict


def _part_bodies(model, mapping, spec):
    start = model.body(mapping[spec["source_joint"]]).id
    end_name = spec.get("source_end_joint")
    if end_name is None:
        # Profiles may map the wrist's first axis, while the palm shell lives
        # below its pitch/yaw joints. Include that distal subtree, never the arm.
        bodies = set()
        for candidate in range(model.nbody):
            body = candidate
            while body:
                if body == start:
                    bodies.add(candidate)
                    break
                body = int(model.body_parentid[body])
        return bodies
    end = model.body(mapping[end_name]).id
    bodies, body = set(), int(model.body_parentid[end])
    while body:
        bodies.add(body)
        if body == start:
            return bodies
        body = int(model.body_parentid[body])
    raise ValueError("Mapped limb endpoints do not form an ancestor chain")


def bind_sampled_parts(model, source, mapping, robot_reference_qpos):
    """Bind once using known rest poses; return disabled regions with reasons."""
    if source.get("kind") != "barycentric_skin" or source.get("metric_scale") != 1.0:
        raise ValueError("Self contact requires real metric barycentric skin samples")
    qref = np.asarray(robot_reference_qpos, dtype=float)
    if qref.shape != (model.nq,) or not np.isfinite(qref).all():
        raise ValueError("Self contact requires a finite calibrated robot reference")
    data = mujoco.MjData(model)
    data.qpos[:] = qref
    mujoco.mj_forward(model, data)
    parts, reports = {}, {}
    collision_ids = collision_geom_ids(model)
    for name, spec in source["parts"].items():
        try:
            bodies = _part_bodies(model, mapping, spec)
        except (KeyError, ValueError) as exc:
            reports[name] = {"status": "disabled", "reason": str(exc)}
            continue
        gids = [int(g) for g in collision_ids if model.geom_bodyid[g] in bodies]
        if not gids:
            reports[name] = {
                "status": "disabled",
                "reason": "no_mapped_collision_surface",
            }
            continue
        meshes = []
        for gid in gids:
            # MuJoCo's mesh collision shell is convex. This also triangulates
            # the existing deterministic box/capsule/sphere shell samples.
            local = trimesh.convex.convex_hull(geom_surface_offsets(model, gid, 0.008))
            rotation = data.geom_xmat[gid].reshape(3, 3)
            world = local.copy()
            world.vertices = local.vertices @ rotation.T + data.geom_xpos[gid]
            meshes.append((gid, local, world))
        robot_vertices = np.concatenate([world.vertices for _, _, world in meshes])
        rb = np.array([robot_vertices.min(axis=0), robot_vertices.max(axis=0)])
        bounds = np.asarray(spec["reference_bounds"], dtype=float)
        triangles = np.asarray(spec["triangles"], dtype=float)
        reference = np.asarray(spec["reference_triangles"], dtype=float)
        bary = np.asarray(spec["barycentric"], dtype=float)
        n = len(reference)
        if (
            reference.shape != (n, 3, 3)
            or triangles.ndim != 4
            or triangles.shape[1:] != reference.shape
            or bary.shape != (n, 3)
            or bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or np.any(bounds[1] - bounds[0] < 1e-6)
            or not np.isfinite(bary).all()
            or np.any(bary < -1e-8)
            or not np.allclose(bary.sum(axis=1), 1)
        ):
            raise ValueError(f"Invalid persistent surface samples for {name}")
        rest_frames, rest_valid = _face_frames(reference)
        frames, valid = _face_frames(triangles)
        rest_points = np.einsum("nij,ni->nj", reference, bary)
        points = np.einsum("tnij,ni->tnj", triangles, bary)
        query = rb[0] + (rest_points - bounds[0]) / (bounds[1] - bounds[0]) * (
            rb[1] - rb[0]
        )
        query[~rest_valid] = rb.mean(axis=0)
        candidates = []
        for gid, local, world in meshes:
            squared, face, closest = igl.point_mesh_squared_distance(
                query,
                np.asarray(world.vertices),
                np.asarray(world.faces, dtype=np.int64),
            )
            candidates.append((squared, face, closest, gid, local, world))
        chosen = np.argmin(np.array([c[0] for c in candidates]), axis=0)
        bindings, robot_frames, good = [], [], rest_valid.copy()
        for i, candidate in enumerate(chosen):
            squared, face, closest, gid, local, world = candidates[candidate]
            rotation = data.geom_xmat[gid].reshape(3, 3)
            normal = world.face_normals[face[i]]
            # The point must be on the corresponding side of the body. Do not
            # silently map a palm to the back of a collision shell.
            good[i] &= bool(
                normal @ rest_frames[i, :, 2] > 0.25 and squared[i] < 0.05**2
            )
            tangent = rest_frames[i, :, 0] - normal * (normal @ rest_frames[i, :, 0])
            norm = np.linalg.norm(tangent)
            good[i] &= norm > 1e-6
            tangent /= max(norm, 1e-12)
            robot_frames.append(
                rotation.T
                @ np.column_stack([tangent, np.cross(normal, tangent), normal])
            )
            bindings.append(
                SurfaceBinding(
                    name,
                    int(spec["face_ids"][i]),
                    tuple(bary[i]),
                    gid,
                    tuple(rotation.T @ (closest[i] - data.geom_xpos[gid])),
                    tuple(local.face_normals[face[i]]),
                )
            )
        valid &= good[None, :]
        report = {
            "status": "bound",
            "samples": n,
            "valid_bindings": int(good.sum()),
            "robot_geoms": [model.geom(g).name or g for g in gids],
            "invalid_frame_samples": int((~valid).sum()),
            "correspondence": "anatomical_rest_region_projection",
        }
        reports[name] = report
        parts[name] = SampledPart(
            name,
            spec.get("effector"),
            points,
            frames[..., 2],
            frames,
            valid,
            tuple(bindings),
            np.array(robot_frames),
            report,
        )
    return parts, reports
