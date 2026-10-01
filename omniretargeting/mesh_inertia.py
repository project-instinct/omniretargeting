"""Segment COMs and inertias from a labelled reference surface mesh.

Open anatomical cuts are capped independently along each boundary loop. The
uniform-density approximation is explicit; no joint origin substitutes for a
segment COM and no robot properties enter the estimate.
"""

import numpy as np
import trimesh


def close_segment(vertices, faces):
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.remove_unreferenced_vertices()
    edges = mesh.edges
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("Inertia needs a manifold segment surface")
    boundary = edges[counts[inverse] == 1]
    if not len(boundary):
        return mesh
    outgoing, incoming = {}, {}
    for a, b in boundary:
        outgoing.setdefault(int(a), set()).add(int(b))
        incoming.setdefault(int(b), set()).add(int(a))
    if any(
        len(outgoing.get(v, ())) != len(incoming.get(v, ()))
        for v in set(outgoing) | set(incoming)
    ):
        raise ValueError("Anatomical cuts must have consistently oriented boundaries")
    vertices, triangles = list(mesh.vertices), list(mesh.faces)
    # At a segmentation junction two loops may touch at a vertex. Decompose
    # the directed boundary into simple cycles instead of capping a figure 8
    # with one centre (which would create nonmanifold cap edges).
    while any(outgoing.values()):
        path = [min(v for v, edges in outgoing.items() if edges)]
        while outgoing[path[-1]]:
            nxt = min(outgoing[path[-1]])
            outgoing[path[-1]].remove(nxt)
            if nxt in path:
                index = path.index(nxt)
                cycle, path = path[index:], path[: index + 1]
                center = len(vertices)
                vertices.append(mesh.vertices[cycle].mean(axis=0))
                triangles.extend(
                    [a, center, b] for a, b in zip(cycle, cycle[1:] + cycle[:1])
                )
            else:
                path.append(nxt)
    closed = trimesh.Trimesh(vertices, triangles, process=False)
    if not closed.is_watertight or not closed.is_winding_consistent:
        raise ValueError("Segment closure failed; cannot invent its mass properties")
    return closed


def segment_inertias(vertices, faces, face_regions, *, total_mass=1.0):
    """Unit total mass is sufficient for homogeneous wrench-cone feasibility.

    Absolute mass cancels when there are no force/torque caps. Returned centres
    and tensors use the supplied reference mesh axes, with inertia about COM.
    """
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
    labels = np.asarray(face_regions)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or labels.shape != (len(faces),)
        or not np.isfinite(vertices).all()
        or not np.isfinite(total_mass)
        or total_mass <= 0
    ):
        raise ValueError(
            "Finite reference vertices and one region per face are required"
        )
    regions = np.unique(labels)
    centers, inertias, volumes = [], [], []
    for region in regions:
        mesh = close_segment(vertices, faces[labels == region])
        if mesh.volume < 0:
            mesh.invert()
        props = mesh.mass_properties
        if not np.isfinite(props["volume"]) or props["volume"] <= 0:
            raise ValueError(f"Nonpositive closed volume for source region {region}")
        centers.append(props["center_mass"])
        inertias.append(props["inertia"])
        volumes.append(props["volume"])
    density = float(total_mass) / sum(volumes)
    return {
        "regions": regions,
        "centers": np.asarray(centers),
        "inertias": np.asarray(inertias) * density,
        "masses": np.asarray(volumes) * density,
        "volumes": np.asarray(volumes),
        "assumptions": {
            "density": "uniform_reference_mesh",
            "mass_scale": float(total_mass),
            "segment_closure": "oriented_boundary_loop_caps",
            "subject_measurement": False,
        },
    }
