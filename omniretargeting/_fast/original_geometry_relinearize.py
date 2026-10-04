"""Origin rows for selected trial material identities, without activation culling.

This is a selected-witness query, never a complete geometry audit. In particular,
a reverse row keeps its trial facet's scalar; that facet need not maximize the
origin's plane distances. Its current maximum/active facet are reported separately.
"""

from copy import copy
import hashlib
import os

import mujoco
import numpy as np

from ..geometry_query import GeometryQueryError
from ..moving_scene import interpolate_object
from .original_geometry_window import validate_frames


def witness_identity(key, row):
    return tuple(key), row.get("sample_id"), row.get("facet_id")


def pose_identity(poses, frames):
    value = hashlib.sha256(repr((frames.start, frames.stop, poses.shape[1])).encode())
    value.update(np.ascontiguousarray(poses[frames], dtype=np.float64).tobytes())
    return value.hexdigest()


def _poses(geometry, poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (geometry.frames, geometry.model.nq) or not np.isfinite(poses).all():
        raise ValueError("Witness poses must be finite (frames, model.nq)")
    return poses


def _body_metadata(geometry, row):
    model, data = geometry.model, geometry.data
    if "points" in row and "pair" in row:
        bodies = [int(model.body_weldid[body]) for body in row["body_ids"]]
        row["welded_body_ids"] = bodies
        row["local_points"] = np.array([(point - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
                                         for body, point in zip(bodies, row["points"])])
    else:
        body = int(model.body_weldid[model.geom_bodyid[row["geom_id"]]])
        row["body_id"] = body
        row["local_point"] = (row["point"] - data.xpos[body]) @ data.xmat[body].reshape(3, 3)


def relinearize_witnesses(geometry, origin_poses, trial_poses, trial_measurement, *, frames, keys=None):
    """Query the trial's identities at origin, retaining explicit failed rows.

    ``keys`` selects collision keys (all manifold witnesses for each selected
    key). Row order is the trial's order. NaN phi/J placeholders keep alignment
    on failure; callers must inspect request_complete before using the rows.
    """
    if os.getpid() != geometry._query_owner:
        raise RuntimeError("Use fork_private_client before querying inherited geometry")
    origin, trial = _poses(geometry, origin_poses), _poses(geometry, trial_poses)
    frames = validate_frames(geometry, frames)
    if not geometry._reference_observations_frozen:
        raise ValueError("Freeze reference observations before selected-witness queries")
    if trial_measurement.get("evidence_id", geometry.reference_observation_id) != geometry.reference_observation_id:
        raise ValueError("Trial witnesses belong to a different frozen reference")
    requested = None if keys is None else {tuple(key) for key in keys}
    if requested is not None and any(key[0] not in frames for key in requested):
        raise ValueError("Selected collision keys must belong to the requested frames")
    selected = [(tuple(key), row) for key, row in trial_measurement["rows"]
                if key[0] in frames and (requested is None or tuple(key) in requested)]
    failures = list(trial_measurement.get("failures", ()))
    if not trial_measurement.get("complete", True):
        failures.append("selected trial measurement is incomplete")
    if trial_measurement.get("reliable", True) is False:
        failures.append("selected trial measurement is unreliable")
    if requested is not None:
        failures.extend(f"missing trial witness collision key {key}"
                        for key in sorted(requested - {key for key, _ in selected}))
    rows = [(key, {**witness, "clearance": np.nan, "direction": np.full(3, np.nan),
                   "jacobian": np.full(geometry.width, np.nan), "reliable": False})
            for key, witness in selected]
    reliable = np.zeros(len(rows), bool)
    shell_map = {} if geometry.moving is None else {shell["geom_id"]: shell for shell in geometry.moving.shells}
    groups, forwards, frame = {}, {}, None
    queried = set()

    def failed(index, error):
        key, row = rows[index]
        row.update(clearance=np.nan, direction=np.full(3, np.nan),
                   jacobian=np.full(geometry.width, np.nan), reliable=False)
        row["points" if key[1] == "self" else "point"] = np.full((2, 3) if key[1] == "self" else 3, np.nan)
        reliable[index] = False
        failures.append(f"unknown selected witness {witness_identity(*selected[index])}: {error}")

    def finish(index):
        key, row = rows[index]
        if (not np.isfinite(row["clearance"]) or not np.isfinite(row["direction"]).all()
                or np.linalg.norm(row["direction"]) <= 0):
            raise GeometryQueryError("nonfinite clearance or unreliable exit direction")
        if key[1] == "self":
            row["jacobian"] = geometry.self_query.jacobian(row, geometry.data)[geometry.columns]
        else:
            jacobian = np.zeros((3, geometry.model.nv))
            mujoco.mj_jac(geometry.model, geometry.data, jacobian, None, row["point"],
                          int(geometry.model.geom_bodyid[row["geom_id"]]))
            row["jacobian"] = row["direction"] @ jacobian[:, geometry.columns]
        if not np.isfinite(row["jacobian"]).all():
            raise GeometryQueryError("nonfinite Jacobian")
        _body_metadata(geometry, row)
        reliable[index] = row["reliable"] = True

    for index, (key, row) in enumerate(rows):
        if key[0] != frame:
            frame = key[0]
            geometry.data.qpos[:] = origin[frame]
            mujoco.mj_forward(geometry.model, geometry.data)
            forwards.clear()
        queried.add(frame)
        geom = int(row.get("geom_id", -1))
        try:
            if key[1] == "self":
                first, second = key[2:]
                pair = int(row["pair_index"])
                if (tuple(geometry.self_query.pairs[pair]) != (first, second)
                        or row["pair_id"] != geometry.self_query.pair_ids[pair]
                        or tuple(row["pair"]) != (first, second)
                        or row["body_ids"] != list(map(int, geometry.model.geom_bodyid[[first, second]]))):
                    raise ValueError("self witness does not match the frozen pair registry")
                if np.all(geometry.model.geom_type[[first, second]] == mujoco.mjtGeom.mjGEOM_MESH):
                    phi, points, direction = geometry.self_query.mesh_distance.query(first, second, geometry.data)
                    row.update(clearance=float(phi), points=points.copy(), direction=direction.copy())
                else:
                    query = copy(geometry.self_query)
                    query.pairs, query.pair_ids = query.pairs[pair:pair + 1], query.pair_ids[pair:pair + 1]
                    current = query.query(origin[frame], data=geometry.data, distance=np.inf)
                    if len(current) != 1:
                        raise GeometryQueryError("requested self pair returned no witness")
                    row.update(current[0], pair_index=pair)
                row["pair_witness_reselected"] = True
                finish(index)
                continue
            if geom < 0 or geom != key[-1]:
                raise ValueError("witness geom does not match its collision key")
            sample = int(row["sample_id"])
            rotation = geometry.data.geom_xmat[geom].reshape(3, 3)
            center = geometry.data.geom_xpos[geom]
            if key[1] == "static":
                if not 0 <= sample < len(geometry.objective.shell_local):
                    raise ValueError("static sample is outside the frozen material registry")
                if geometry.objective.shell_sample_geom[sample] != geom or geometry.scene is None:
                    raise ValueError("static sample does not belong to this geom/scene")
                point = center + np.einsum("nij,nj->ni", rotation[None],
                    geometry.objective.shell_local[sample:sample + 1])[0]
                scene, components, scale, object_rotation = geometry.scene, None, 1., np.eye(3)
                row["material_local_point"] = geometry.objective.shell_local[sample].copy()
                local = point
            elif key[1] == "moving":
                oid, kind, cid = key[2:5]
                obj, shell = geometry.moving.objects[oid], shell_map[geom]
                translation, object_rotation, scale = interpolate_object(obj["surface"], frame)
                if kind == "reverse_surface":
                    if not 0 <= sample < len(obj["samples"]):
                        raise ValueError("reverse sample is outside the frozen material registry")
                    facet = int(row["facet_id"])
                    planes = shell["planes"]
                    if not 0 <= facet < len(planes):
                        raise ValueError("reverse facet is outside the frozen plane registry")
                    point = obj["samples"][sample:sample + 1] @ object_rotation.T * scale + translation
                    values = ((point - center) @ rotation) @ planes[:, :3].T + planes[:, 3]
                    active = int(np.argmax(values[0]))
                    phi = float(values[0, facet])
                    direction = -(rotation @ planes[facet, :3])
                    row.update(clearance=phi, point=point[0], direction=direction,
                               surface_point=point[0] + phi * direction,
                               material_local_point=obj["samples"][sample].copy(), material_owner="object",
                               current_max_plane=float(values[0, active]), current_active_facet=active,
                               frozen_facet_active=bool(facet == active),
                               current_point_depth=max(0., -float(values[0, active])))
                    finish(index)
                    continue
                if kind not in ("closed_surface", "observed_open_surface"):
                    raise ValueError("unsupported moving witness kind")
                if not 0 <= sample < len(shell["offsets"]):
                    raise ValueError("forward sample is outside the frozen material registry")
                token = oid, geom
                if token not in forwards:
                    world = shell["offsets"] @ rotation.T + center
                    forwards[token] = world, (world - translation) @ object_rotation / scale
                world, local_points = forwards[token]
                point, local = world[sample], local_points[sample]
                scene, components = obj["query"], [cid]
                row["material_local_point"] = shell["offsets"][sample].copy()
            else:
                raise ValueError("unsupported collision key kind")
            row["point"] = np.asarray(point).copy()
            token = id(scene), None if components is None else tuple(components)
            groups.setdefault(token, (scene, components, []))[2].append(
                (index, local, scale, object_rotation.copy(), None if key[1] == "static" else translation.copy()))
        except (GeometryQueryError, KeyError, IndexError, ValueError, TypeError) as error:
            failed(index, error)

    for scene, components, items in groups.values():
        try:
            measured = scene.query(np.array([item[1] for item in items]), component_ids=components)
        except (GeometryQueryError, ValueError, RuntimeError) as error:
            for index, *_ in items:
                failed(index, error)
            continue
        for local_index, (index, _, scale, rotation, translation) in enumerate(items):
            key, row = rows[index]
            try:
                if measured["uncertain"][local_index]:
                    raise GeometryQueryError("original query cannot decide the selected material point")
                row["clearance"] = float(measured["clearance"][local_index] * scale)
                row["direction"] = rotation @ measured["directions"][local_index]
                point = measured["surface_points"][local_index]
                row["surface_point"] = point.copy() if translation is None else point @ rotation.T * scale + translation
                row["surface_id"] = int(measured["surface_ids"][local_index])
                geometry.data.qpos[:] = origin[key[0]]
                mujoco.mj_forward(geometry.model, geometry.data)
                finish(index)
            except (GeometryQueryError, ValueError) as error:
                failed(index, error)
    matrix, phi = geometry.linear_rows(dict(rows=rows), frames=frames)
    scope = dict(requested_frames=list(frames), requested_frame_count=len(frames),
                 queried_frames=sorted(queried), queried_frame_count=len(queried),
                 requested_witness_count=len(selected), returned_witness_count=len(rows),
                 reliable_witness_count=int(reliable.sum()))
    return dict(rows=rows, J=matrix, phi=phi, clearance=phi, reliable=reliable,
                witness_ids=tuple(witness_identity(key, witness) for key, witness in selected),
                pose_id=pose_identity(origin, frames), origin_pose_id=pose_identity(origin, frames),
                trial_pose_id=pose_identity(trial, frames), evidence_id=geometry.reference_observation_id,
                columns=geometry.columns.copy(), frames=tuple(frames), failures=failures,
                request_complete=bool(reliable.all() and not failures),
                query_scope=scope, queried_scope=dict(scope), coverage="selected_witnesses_only")
