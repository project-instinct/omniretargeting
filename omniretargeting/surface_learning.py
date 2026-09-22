"""Learn indexed whole-body surface correspondence in calibrated reference poses.

Independent implementation of UMR III-B (arXiv:2609.02134): PointNet encoding,
indexed residual decoding, Chamfer coverage, repulsion and mesh-geodesic edge
regularization. Training fits this source/robot pair; no external weights are
required. Normalization is confined to correspondence setup, never motion.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

import igl
import mujoco
import numpy as np
import torch
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from torch import nn

from .scene_geometry import collision_geom_ids, geom_surface_offsets
from .surface_anatomy import surface_domains
from .surface_tasks import SurfaceBinding, _face_frames

DEFAULTS = {
    "iterations": 300,
    "learning_rate": 0.001,
    "seed": 17,
    "edge_weight": 1.0,
    "repulsion_weight": 0.01,
    "repulsion_radius": 0.04,
    "neighbors": 6,
    "projection_tolerance": 0.08,
    "normal_agreement": 0.0,
    "lateral_axis": [0.0, 1.0, 0.0],
    "midline_tolerance": 0.02,
    "cache_dir": "~/.cache/omniretargeting/surface_correspondence",
}


def correspondence_config(config=None):
    cfg = {**DEFAULTS, **(config or {})}
    for key in ("iterations", "neighbors"):
        if int(cfg[key]) != cfg[key] or cfg[key] < 1:
            raise ValueError(f"surface correspondence {key} must be a positive integer")
    for key in (
        "learning_rate",
        "edge_weight",
        "repulsion_weight",
        "repulsion_radius",
        "projection_tolerance",
    ):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(
                f"surface correspondence {key} must be finite and positive"
            )
    if not -1 <= cfg["normal_agreement"] <= 1:
        raise ValueError("surface correspondence normal_agreement must be in [-1, 1]")
    axis = np.asarray(cfg["lateral_axis"], dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
        raise ValueError(
            "surface correspondence lateral_axis must be a finite nonzero vector"
        )
    if not np.isfinite(cfg["midline_tolerance"]) or cfg["midline_tolerance"] < 0:
        raise ValueError(
            "surface correspondence midline_tolerance must be finite and nonnegative"
        )
    return cfg


def crossed_midline_mask(human, robot, config):
    """Reject reference-side reversals without a semantic human/robot body map."""
    axis = np.array(config["lateral_axis"], dtype=float, copy=True)
    axis /= np.linalg.norm(axis)
    human_side, robot_side = human @ axis, robot @ axis
    return (np.abs(human_side) > config["midline_tolerance"]) & (
        human_side * robot_side < 0
    )


def geodesic_edges(vertices, faces, face_ids, barycentric, neighbors=6):
    """Connect sampled indices along the source mesh, never across limb gaps."""
    edges = trimesh.geometry.faces_to_edges(faces)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    graph = sparse.csr_matrix(
        (np.maximum(lengths, 1e-12), (edges[:, 0], edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    )
    graph = graph + graph.T
    anchors = faces[face_ids, np.argmax(barycentric, axis=1)]
    result = []
    for start in range(0, len(anchors), 32):
        distances = dijkstra(
            graph, directed=False, indices=anchors[start : start + 32]
        )[:, anchors]
        for row, distance in enumerate(distances):
            index = start + row
            distance[index] = np.inf
            chosen = np.argsort(distance, kind="stable")[:neighbors]
            result.extend((index, int(j)) for j in chosen if np.isfinite(distance[j]))
    return np.unique(
        np.sort(np.asarray(result, dtype=int).reshape(-1, 2), axis=1), axis=0
    )


class IndexedSurfaceNetwork(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, count * 3)
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, robot, source):
        latent = self.encoder(robot).amax(dim=0)
        return source + self.decoder(latent).reshape_as(source)


@contextmanager
def _training_state(seed):
    # Do not leak RNG state or thread settings into the caller's motion pipeline.
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        torch.set_num_threads(min(threads, 4))
        try:
            yield
        finally:
            torch.set_num_threads(threads)


def learn_indexed_points(
    source, robot, edges, config=None, *, source_domains=None, target_domains=None
):
    """Fit one reusable correspondence; return source-indexed robot points."""
    cfg = correspondence_config(config)
    source, robot = np.asarray(source), np.asarray(robot)
    for points in (source, robot):
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 2
            or not np.isfinite(points).all()
        ):
            raise ValueError(
                "Correspondence needs finite point clouds with at least two points"
            )
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if not len(edges) or edges.min() < 0 or edges.max() >= len(source):
        raise ValueError("Correspondence needs valid source mesh edges")
    with _training_state(cfg["seed"]):
        human, target = (
            torch.tensor(source, dtype=torch.float32),
            torch.tensor(robot, dtype=torch.float32),
        )
        edge = torch.tensor(edges, dtype=torch.long)
        compatible = None
        if source_domains is not None:
            if np.shape(source_domains) != (len(source),) or np.shape(
                target_domains
            ) != (len(robot),):
                raise ValueError(
                    "Correspondence domains must identify every sampled point"
                )
            compatible = torch.tensor(
                np.asarray(source_domains)[:, None]
                == np.asarray(target_domains)[None, :]
            )
            # A domain with no sampled counterpart is excluded from the fit;
            # projection reports it as uncovered, rather than borrowing a limb.
            source_covered, target_covered = compatible.any(1), compatible.any(0)
            if not source_covered.any():
                raise ValueError("No corresponding kinematic surface domains")
        network = IndexedSurfaceNetwork(len(source))
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg["learning_rate"])
        best, best_loss, initial, final_terms = None, np.inf, None, None
        for iteration in range(int(cfg["iterations"]) + 1):
            predicted = network(target, human)
            distance = torch.cdist(predicted, target).square()
            if compatible is None:
                chamfer = distance.amin(1).mean() + distance.amin(0).mean()
            else:
                domain_distance = distance.masked_fill(~compatible, float("inf"))
                chamfer = (
                    domain_distance[source_covered].amin(1).mean()
                    + domain_distance[:, target_covered].amin(0).mean()
                )
            pair_distance = torch.cdist(predicted, predicted).square()
            pair_distance = pair_distance + torch.eye(len(source)) * 1e6
            near = pair_distance.topk(
                min(cfg["neighbors"], len(source) - 1), largest=False
            ).values
            repulsion = torch.exp(-near / cfg["repulsion_radius"] ** 2).mean()
            residual = predicted - human
            smooth = (
                (residual[edge[:, 0]] - residual[edge[:, 1]]).square().sum(1).mean()
            )
            loss = (
                chamfer
                + cfg["repulsion_weight"] * repulsion
                + cfg["edge_weight"] * smooth
            )
            value = float(loss.detach())
            if not np.isfinite(value):
                raise ValueError("Non-finite learned surface correspondence")
            if initial is None:
                initial = float(chamfer.detach())
            if value < best_loss:
                best, best_loss = predicted.detach().numpy().copy(), value
                final_terms = {
                    "chamfer": float(chamfer.detach()),
                    "repulsion": float(repulsion.detach()),
                    "geodesic_edge": float(smooth.detach()),
                }
            if iteration < cfg["iterations"]:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return best, {
        "initial_chamfer": initial,
        "loss": best_loss,
        **final_terms,
        "iterations": cfg["iterations"],
        "training": "per_reference_pair",
        "device": "cpu",
    }


def robot_surface_mesh(model, qref):
    """Reference collision shells, with a persistent geom identity per face."""
    data = mujoco.MjData(model)
    data.qpos[:] = qref
    mujoco.mj_forward(model, data)
    meshes, ids = [], []
    for gid in collision_geom_ids(model):
        if model.body_weldid[model.geom_bodyid[gid]] == 0:
            continue  # Exclude the scene/world from a combined MuJoCo model.
        local = geom_surface_offsets(model, int(gid), 0.015)
        if len(local) < 4 or np.linalg.matrix_rank(local - local.mean(0)) < 3:
            continue
        mesh = trimesh.convex.convex_hull(local)
        mesh.vertices = (
            mesh.vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        )
        meshes.append(mesh)
        ids.extend([int(gid)] * len(mesh.faces))
    if not meshes:
        raise ValueError(
            "No robot body collision surfaces for whole-body correspondence"
        )
    return trimesh.util.concatenate(meshes), np.asarray(ids), data


def bind_whole_body(
    model, source, robot_reference_qpos, config=None, *, joint_mapping=None
):
    """Learn once, project onto actual robot shells, and freeze both bindings."""
    cfg = correspondence_config(config)
    if source.get("metric_scale") != 1.0 or source.get("kind") != "barycentric_skin":
        raise ValueError("Whole-body contact requires unscaled metric skin surfaces")
    full = source["full_body"]
    vertices = np.asarray(full["reference_vertices"], dtype=float)
    faces = np.asarray(full["faces"], dtype=int)
    face_ids, bary = (
        np.asarray(full["face_ids"], dtype=int),
        np.asarray(full["barycentric"], dtype=float),
    )
    qref = np.asarray(robot_reference_qpos, dtype=float)
    if qref.shape != (model.nq,) or not np.isfinite(qref).all():
        raise ValueError("A calibrated finite robot reference pose is required")
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(vertices)
        or face_ids.ndim != 1
        or len(face_ids) < 2
        or face_ids.min() < 0
        or face_ids.max() >= len(faces)
        or bary.shape != (len(face_ids), 3)
        or not np.isfinite(bary).all()
        or np.any(bary < 0)
        or not np.allclose(bary.sum(1), 1)
    ):
        raise ValueError(
            "Invalid whole-body reference mesh or barycentric sample identities"
        )
    reference = np.einsum("nij,ni->nj", vertices[faces[face_ids]], bary)
    mesh, face_geoms, data = robot_surface_mesh(model, qref)
    sample_domains, face_domains, anatomy = surface_domains(
        model, source, face_geoms, joint_mapping
    )

    # Independent uniform normalization is only a fitting coordinate system.
    def normalize(points):
        center = (points.max(0) + points.min(0)) / 2
        extent = np.ptp(points, axis=0).max()
        if extent < 1e-6:
            raise ValueError("Degenerate whole-body reference surface")
        return (points - center) / extent, center, extent

    _, hc, hs = normalize(vertices)
    _, rc, rs = normalize(np.asarray(mesh.vertices))
    human = (reference - hc) / hs
    target, target_faces = trimesh.sample.sample_surface(
        mesh, len(reference), seed=int(cfg["seed"])
    )
    target = (target - rc) / rs
    digest = sha256(b"umr-independent-correspondence-v2-kinematic-domains")
    for array in (
        vertices,
        faces,
        face_ids,
        bary,
        np.asarray(mesh.vertices),
        mesh.faces,
        face_geoms,
        qref,
    ):
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    if sample_domains is not None:
        digest.update(np.ascontiguousarray(sample_domains).tobytes())
        digest.update(np.ascontiguousarray(face_domains).tobytes())
    digest.update(
        json.dumps(
            {k: v for k, v in cfg.items() if k != "cache_dir"}, sort_keys=True
        ).encode()
    )
    key = digest.hexdigest()
    path = (
        Path(cfg["cache_dir"]).expanduser() / f"{key}.npz" if cfg["cache_dir"] else None
    )
    predicted, training = None, None
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as cached:
            predicted = cached["points"].copy()
            training = json.loads(str(cached["training"]))
        if predicted.shape != human.shape or not np.isfinite(predicted).all():
            raise ValueError(f"Invalid surface correspondence cache: {path}")
    cache_hit = predicted is not None
    if predicted is None:
        edges = geodesic_edges(vertices, faces, face_ids, bary, cfg["neighbors"])
        domain_options = (
            {}
            if sample_domains is None
            else {
                "source_domains": sample_domains,
                "target_domains": face_domains[target_faces],
            }
        )
        predicted, training = learn_indexed_points(
            human, target, edges, cfg, **domain_options
        )
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=path.parent, suffix=".npz", delete=False
            ) as tmp:
                temporary = tmp.name
                np.savez_compressed(
                    tmp, points=predicted, training=json.dumps(training)
                )
            os.replace(temporary, path)
    proposed = predicted * rs + rc
    squared, triangles, points = igl.point_mesh_squared_distance(
        proposed, np.asarray(mesh.vertices), mesh.faces
    )
    wrong_domain = np.zeros(len(reference), dtype=bool)
    covered = np.ones(len(reference), dtype=bool)
    if sample_domains is not None:
        wrong_domain = sample_domains != face_domains[triangles]
        for domain in np.unique(sample_domains):
            indices = np.flatnonzero(sample_domains == domain)
            allowed = np.flatnonzero(face_domains == domain)
            if not len(allowed):
                covered[indices] = False
                continue
            d2, local_face, projected = igl.point_mesh_squared_distance(
                proposed[indices], np.asarray(mesh.vertices), mesh.faces[allowed]
            )
            squared[indices], triangles[indices], points[indices] = (
                d2,
                allowed[local_face],
                projected,
            )
    frames, valid = _face_frames(vertices[faces[face_ids]])
    valid &= covered
    normals = mesh.face_normals[triangles]
    valid &= np.sqrt(squared) <= cfg["projection_tolerance"] * rs
    valid &= np.einsum("ij,ij->i", normals, frames[:, :, 2]) > cfg["normal_agreement"]
    # Coverage and local edge loss can fold a few foot points onto the other
    # foot. A reversed reference-side binding must not drive another limb.
    crossed = crossed_midline_mask(human, (points - rc) / rs, cfg)
    valid &= ~crossed
    bindings = []
    labels = full.get("labels", ["body"] * len(face_ids))
    if np.shape(labels) != (len(face_ids),):
        raise ValueError("Whole-body surface labels must identify every sample")
    for i, face in enumerate(triangles):
        gid = int(face_geoms[face])
        rotation = data.geom_xmat[gid].reshape(3, 3)
        bindings.append(
            SurfaceBinding(
                str(labels[i]),
                int(face_ids[i]),
                tuple(bary[i]),
                gid,
                tuple(rotation.T @ (points[i] - data.geom_xpos[gid])),
                tuple(rotation.T @ normals[i]),
            )
        )
    return (
        tuple(bindings),
        valid,
        {
            "method": "learned_indexed_surface_correspondence",
            "algorithm": "UMR III-B independent implementation",
            "cache_key": key,
            "cache_hit": cache_hit,
            "samples": len(bindings),
            "valid_bindings": int(valid.sum()),
            "invalid_binding_indices": np.flatnonzero(~valid).tolist(),
            "crossed_midline_indices": np.flatnonzero(crossed).tolist(),
            "anatomy": anatomy,
            "projection_domain_corrections": int(wrong_domain.sum()),
            "valid_bindings_per_region": {
                str(label): {
                    "valid": int(
                        np.count_nonzero(valid & (np.asarray(labels) == label))
                    ),
                    "total": int(np.count_nonzero(np.asarray(labels) == label)),
                }
                for label in np.unique(labels)
            },
            "training": training,
            "normalization_scope": "reference_correspondence_only",
            "surface": "robot_collision_shells",
            "robot_scale": 1.0,
            "scene_scale": 1.0,
        },
    )
