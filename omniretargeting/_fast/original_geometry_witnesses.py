"""Candidate clearances at frozen material samples/pairs, without row selection."""

import os

import mujoco
import numpy as np

from ..geometry_query import GeometryQueryError
from ..moving_scene import interpolate_object
from .original_geometry_window import validate_frames


def _identity(key, row):
    return key, row.get("sample_id"), row.get("facet_id")


def reevaluate_witnesses(geometry, poses, measurement, *, frames=None):
    if os.getpid() != geometry._query_owner:
        raise RuntimeError("Use fork_private_client before querying inherited geometry")
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (geometry.frames, geometry.model.nq) or not np.isfinite(poses).all():
        raise ValueError("Witness poses must be finite (frames, model.nq)")
    frames = validate_frames(geometry, range(geometry.frames) if frames is None else frames)
    if measurement.get("evidence_id", geometry.reference_observation_id) != geometry.reference_observation_id:
        raise ValueError("Witnesses belong to a different frozen reference")
    rows = [(key, row) for key, row in measurement["rows"] if key[0] in frames]
    values = np.full(len(rows), np.nan)
    reliable = np.zeros(len(rows), dtype=bool)
    failures, pending, forwards, frame = [], [], {}, None
    shell_map = {} if geometry.moving is None else {s["geom_id"]: s for s in geometry.moving.shells}
    for index, (key, witness) in enumerate(rows):
        if key[0] != frame:
            frame = key[0]
            data = geometry.data
            data.qpos[:] = poses[frame]
            mujoco.mj_forward(geometry.model, data)
            cache = geometry._window_frame_cache.get(frame, {}).get(
                (geometry.reference_observation_id, poses[frame].tobytes()))
            known = {} if cache is None else {_identity(k, r): r for k, r in cache[1]}
            if cache is not None and cache[3]:
                failures.extend(cache[3])
            forwards.clear()
        previous = known.get(_identity(key, witness))
        if previous is not None:
            values[index] = previous["clearance"]
            reliable[index] = cache is not None and not cache[3]
            continue
        geom = int(witness.get("geom_id", -1))
        try:
            if key[1] == "static":
                sample = int(witness["sample_id"])
                point = data.geom_xpos[geom] + np.einsum(
                    "nij,nj->ni", data.geom_xmat[geom].reshape(1, 3, 3),
                    geometry.objective.shell_local[sample:sample + 1])[0]
                pending.append((index, geometry.scene, point, None, 1.))
            elif key[1] == "moving":
                oid, kind, cid = key[2:5]
                obj, shell = geometry.moving.objects[oid], shell_map[geom]
                translation, rotation, scale = interpolate_object(obj["surface"], frame)
                robot_rotation = data.geom_xmat[geom].reshape(3, 3)
                sample = int(witness["sample_id"])
                if kind == "reverse_surface":
                    point = obj["samples"][sample:sample + 1] @ rotation.T * scale + translation
                    local = (point - data.geom_xpos[geom]) @ robot_rotation
                    planes = shell["planes"]
                    # The scalar row is the frozen exit facet at a fixed object sample.
                    all_planes = local @ planes[:, :3].T + planes[:, 3]
                    values[index] = all_planes[0, int(witness["facet_id"])]
                    reliable[index] = np.isfinite(values[index])
                else:
                    token = oid, geom
                    if token not in forwards:
                        world = shell["offsets"] @ robot_rotation.T + data.geom_xpos[geom]
                        forwards[token] = (world - translation) @ rotation / scale
                    pending.append((index, obj["query"], forwards[token][sample], [cid], scale))
            else:
                first, second = key[2:]
                if np.all(geometry.model.geom_type[[first, second]] == mujoco.mjtGeom.mjGEOM_MESH):
                    phi, _, _ = geometry.self_query.mesh_distance.query(first, second, data)
                else:
                    phi = mujoco.mj_geomDistance(geometry.model, data, first, second, 1e6, np.zeros(6))
                values[index], reliable[index] = phi, np.isfinite(phi)
        except (GeometryQueryError, KeyError, ValueError) as error:
            failures.append(f"unknown frozen witness {key}: {error}")
    groups = {}
    for index, scene, point, components, scale in pending:
        token = id(scene), None if components is None else tuple(components)
        groups.setdefault(token, (scene, components, []))[2].append((index, point, scale))
    for scene, components, items in groups.values():
        result = scene.query(np.array([item[1] for item in items]), component_ids=components)
        for local, (index, _, scale) in enumerate(items):
            if not result["uncertain"][local] and np.isfinite(result["clearance"][local]):
                values[index] = result["clearance"][local] * scale
                reliable[index] = True
            else:
                failures.append(f"unknown frozen witness {rows[index][0]}")
    return dict(clearance=values, reliable=reliable, failures=failures,
                keys=tuple(key for key, _ in rows), complete=bool(reliable.all() and not failures),
                evidence_id=geometry.reference_observation_id)
