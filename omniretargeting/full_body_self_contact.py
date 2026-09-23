"""Discover non-adjacent body contact pairs on learned surface identities.

The source alone chooses a pair. Keep it while it remains in contact, split it
on release, and never reassign it using the robot's optimized geometry. Both
moving robot surfaces contribute to the relative-vector Jacobian.
"""

from dataclasses import asdict

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .surface_learning import mesh_edge_graph
from .surface_motion import SurfaceMotion


def _topology(full, count):
    joints = np.asarray(full.get("sample_joint_indices", []), dtype=int)
    parents = np.asarray(full.get("joint_parents", []), dtype=int)
    if joints.shape != (count,) or not len(parents):
        return None
    if np.any(joints < 0) or np.any(joints >= len(parents)):
        raise ValueError("Self-contact sample joints must belong to source topology")
    if np.any(parents >= len(parents)):
        raise ValueError("Invalid source joint parents")
    for joint in range(len(parents)):
        seen = set()
        while joint >= 0:
            if joint in seen:
                raise ValueError("Cyclic source joint topology")
            seen.add(joint)
            joint = int(parents[joint])
    return joints, parents


def _scene_exclusions(full, scene_tasks, frames, count):
    labels = np.asarray(full.get("sample_effectors", [""] * count))
    if labels.shape != (count,):
        raise ValueError("Sample effector labels must follow surface identities")
    excluded = np.zeros((frames, count), bool)
    for name, task in (scene_tasks or {}).items():
        mask = np.asarray(task.contact_mask, bool)
        if mask.shape != (frames,):
            raise ValueError("Scene and self contacts must share the source timeline")
        excluded[:, labels == name] |= mask[:, None]
    return excluded


class SurfaceAdjacency:
    """Exclude continuous nearby skin, even across a semantic joint boundary."""

    def __init__(self, full, minimum):
        self.vertices = np.asarray(full["reference_vertices"])
        faces = np.asarray(full["faces"])
        self.triangles = faces[np.asarray(full["face_ids"])]
        self.points = np.einsum(
            "nij,ni->nj", self.vertices[self.triangles], full["barycentric"]
        )
        self.graph = mesh_edge_graph(self.vertices, faces)
        self.minimum, self.cache = minimum, {}
        self.rejected = set()

    def separate(self, a, b):
        # Euclidean distance lower-bounds geodesic distance; most candidates
        # need no mesh query. Local Dijkstra results are reused across frames.
        if np.linalg.norm(self.points[a] - self.points[b]) >= self.minimum:
            return True
        if a not in self.cache:
            anchors = self.triangles[a]
            offsets = np.linalg.norm(self.vertices[anchors] - self.points[a], axis=1)
            distances = dijkstra(self.graph, indices=anchors, limit=self.minimum)
            self.cache[a] = np.min(distances + offsets[:, None], axis=0)
        anchors = self.triangles[b]
        offsets = np.linalg.norm(self.vertices[anchors] - self.points[b], axis=1)
        separate = bool(np.min(self.cache[a][anchors] + offsets) >= self.minimum)
        if not separate:
            self.rejected.add((a, b))
        return separate


def _tracks(motion, joints, parents, excluded, distance, adjacency):
    """One persistent point pair per source segment pair; arbitrary segments."""
    model, points, normals = motion.model, motion.points, motion.frames[..., 2]
    bodies = motion.body_ids
    weld = model.body_weldid[bodies]
    weld_parent = model.body_weldid[model.body_parentid[weld]]
    active, finished = {}, []
    for frame in range(len(points)):
        valid = motion.valid[frame] & ~excluded[frame]
        indices = np.flatnonzero(valid)
        pairs = cKDTree(points[frame, indices]).query_pairs(
            distance, output_type="ndarray"
        )
        a, b = indices[pairs].T
        swap = joints[a] > joints[b]
        a, b = np.where(swap, b, a), np.where(swap, a, b)
        ja, jb = joints[a], joints[b]
        good = (
            (ja != jb)
            & (parents[ja] != jb)
            & (parents[jb] != ja)
            & (weld[a] != weld[b])
            & (weld_parent[a] != weld[b])
            & (weld_parent[b] != weld[a])
            & (np.einsum("ij,ij->i", normals[frame, a], normals[frame, b]) < -0.5)
        )
        a, b = a[good], b[good]
        separate = np.array(
            [adjacency.separate(int(i), int(j)) for i, j in zip(a, b)], dtype=bool
        )
        a, b = a[separate], b[separate]
        gap = np.linalg.norm(points[frame, a] - points[frame, b], axis=1)
        keys = joints[a] * len(parents) + joints[b]
        order = np.lexsort((b, a, gap, keys))
        _, first = np.unique(keys[order], return_index=True)
        candidates = {
            (int(joints[a[i]]), int(joints[b[i]])): (
                float(gap[i]),
                int(a[i]),
                int(b[i]),
            )
            for i in order[first]
        }
        for key, (start, a, b) in list(active.items()):
            if (
                valid[a]
                and valid[b]
                and np.linalg.norm(points[frame, a] - points[frame, b]) <= distance
                and normals[frame, a] @ normals[frame, b] < -0.5
            ):
                candidates.pop(key, None)
            else:
                finished.append((start, frame, a, b))
                del active[key]
        for key, (_, a, b) in candidates.items():
            active[key] = (frame, a, b)
    finished.extend((start, len(points), a, b) for start, a, b in active.values())
    return sorted(finished)


def build_full_body_self_contact(
    model, source, mapping, reference_qpos, scene_tasks, cfg, *, surface_motion=None
):
    from .self_contact import SelfContactTasks

    report = {"status": "disabled", "reason": "missing_source_topology", "events": []}
    full = source["full_body"]
    topology = _topology(full, len(full["face_ids"]))
    if topology is None:
        return None, report
    motion = surface_motion or SurfaceMotion(
        model, source, reference_qpos, cfg.get("correspondence"), mapping
    )
    frames, count = motion.valid.shape
    excluded = _scene_exclusions(full, scene_tasks, frames, count)
    adjacency = SurfaceAdjacency(full, cfg["minimum_geodesic_distance"])
    tracks = _tracks(motion, *topology, excluded, cfg["distance"], adjacency)
    events, bindings, targets, normals, confidences = [], [], [], [], []
    for start, stop, a, b in tracks:
        if stop - start < cfg["min_contact_frames"]:
            continue
        # Transport source vectors into the receiving robot geom's bind frame.
        # This preserves the full metric vector without making the target
        # depend on the robot root, a later nearest point, or limb length.
        receiver = (
            motion.robot_rotations[b].T
            @ motion.reference_frames[b]
            @ np.swapaxes(motion.frames[:, b], -1, -2)
        )
        target = np.einsum(
            "tij,tj->ti", receiver, motion.points[:, a] - motion.points[:, b]
        )
        normal = np.einsum("tij,tj->ti", receiver, motion.normal_targets[:, a])
        confidence = np.zeros(frames)
        confidence[start:stop] = 1.0
        bindings.extend((motion.bindings[a], motion.bindings[b]))
        targets.append(target)
        normals.append(normal)
        confidences.append(confidence)
        events.append(
            {
                "part_a": motion.bindings[a].part,
                "part_b": motion.bindings[b].part,
                "start": start,
                "stop": stop,
                "sample_a": a,
                "sample_b": b,
                "binding_a": asdict(motion.bindings[a]),
                "binding_b": asdict(motion.bindings[b]),
                "source_nearest_gap_mean_m": float(
                    np.linalg.norm(target[start:stop], axis=1).mean()
                ),
            }
        )
    report.update(
        configuration=dict(cfg),
        correspondence=motion.diagnostics,
        events=events,
        event_count=len(events),
        discovery="non_adjacent_source_segments_and_robot_welds",
        minimum_geodesic_distance_m=cfg["minimum_geodesic_distance"],
        excluded_geodesic_pairs=len(adjacency.rejected),
        scene_excluded_frame_samples=int(excluded.sum()),
        geometry_avoidance="independent_MuJoCo_collision_constraints",
    )
    if not events:
        return None, {**report, "reason": "no_supported_source_self_contact"}
    report.update(
        status="active",
        reason=None,
        active_frame_pairs=int(np.count_nonzero(confidences)),
    )
    return SelfContactTasks(
        model,
        bindings,
        np.stack(targets, axis=1),
        np.stack(confidences, axis=1),
        events,
        cfg,
        report,
        np.stack(normals, axis=1),
    ), report
