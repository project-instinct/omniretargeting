"""Amortize original-scene winding construction without dropping witnesses."""

import mujoco
import numpy as np

QUERY_BATCH_POINTS = 1 << 20  # ~128 MiB of witnesses, independent of sampling density.


def probe_batches(probe, poses, *, batch_points=QUERY_BATCH_POINTS):
    """Stream unchanged probe results with a bounded query working set."""
    batch_size = max(1, batch_points // max(1, probe.query_work_bound))
    for start in range(0, len(poses), batch_size):
        yield from batched_probe(probe, poses[start : start + batch_size])


def batched_probe(probe, poses):
    """Same culling, sample order, signs and narrow phase as individual probes.

    Callers bound batch memory and retain the sequential path for strict query
    budgets. This changes query grouping only, never geometric tolerances.
    """
    data = mujoco.MjData(probe.model)
    if probe.query.has_unsigned_only_components:
        near = np.ones((len(poses), len(probe.geom_ids)), bool)
    else:
        centers = []
        for pose in poses:
            data.qpos[:] = pose
            mujoco.mj_forward(probe.model, data)
            centers.append(data.geom_xpos[probe.geom_ids].copy())
        measured = probe.query.query(np.concatenate(centers))
        clearance = measured["clearance"].reshape(len(poses), -1)
        near = (
            measured["uncertain"].reshape(len(poses), -1)
            | ~np.isfinite(clearance)
            | (clearance <= probe.bounding_radii + probe.margin)
        )
    points, owners, geometry_rows = [], [], []
    for pose, selected in zip(poses, near):
        data.qpos[:] = pose
        mujoco.mj_forward(probe.model, data)
        groups, ids = [], []
        for local in np.flatnonzero(selected):
            gid = probe.geom_ids[local]
            groups.append(
                data.geom_xpos[gid]
                + probe.offsets[local] @ data.geom_xmat[gid].reshape(3, 3).T
            )
            ids.append(np.full(len(probe.offsets[local]), local, int))
        points.append(np.concatenate(groups) if groups else np.empty((0, 3)))
        owners.append(np.concatenate(ids) if ids else np.empty(0, int))
        if probe.geometry_backend is not None:
            before = probe.geometry_backend.pair_queries
            # Reuse the same conservative centre mask for exact rigid pairs.
            geometry_rows.append(probe.geometry_backend.query(
                data, probe.margin, active_geom_ids=probe.geom_ids[selected]
            ))
            probe.query.query_points += probe.geometry_backend.pair_queries - before
    measured = probe.query.query(np.concatenate(points))
    frames, start = [], 0
    for index, (point, owner) in enumerate(zip(points, owners)):
        stop = start + len(point)
        frame = {key: value[start:stop].copy() for key, value in measured.items()}
        frame.update(
            points=point, body_ids=probe.body_ids[owner], geom_ids=probe.geom_ids[owner]
        )
        if probe.geometry_backend is not None:
            rows = geometry_rows[index]
            if rows:
                frame = {
                    key: np.concatenate([value, np.asarray([row[key] for row in rows])])
                    for key, value in frame.items()
                }
        frames.append(frame)
        start = stop
    return frames
