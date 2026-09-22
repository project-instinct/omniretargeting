"""Interaction graph adjacency, source edge weights, and Laplacian operators."""

from __future__ import annotations

import numpy as np


def get_adjacency_list(tetrahedra, num_vertices):
    """Creates an adjacency list from the tetrahedra."""
    adj = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = tet[i], tet[j]
                adj[u].add(v)
                adj[v].add(u)
    return [list(s) for s in adj]


def calculate_laplacian_coordinates(vertices, adj_list, epsilon=1e-6, uniform_weight=True):
    """
    Calculates the Laplacian coordinates for each vertex in the mesh.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        epsilon (float): Small value to prevent division by zero.
        uniform_weight (bool): Whether to use uniform weights.

    Returns:
        np.ndarray: (N, 3) array of Laplacian coordinates.
    """
    laplacian = np.zeros_like(vertices)

    for i in range(len(vertices)):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            vi = vertices[i]
            neighbor_positions = vertices[neighbors_indices]
            distances = np.linalg.norm(vi - neighbor_positions, axis=1)

            if uniform_weight:
                weights = np.ones_like(distances)
            else:
                weights = 1.0 / (1.5 * distances + epsilon)

            sum_of_weights = np.sum(weights)
            weighted_sum_of_neighbors = np.sum(weights[:, np.newaxis] * neighbor_positions, axis=0)
            center_of_neighbors = weighted_sum_of_neighbors / sum_of_weights
            laplacian[i] = vi - center_of_neighbors

    return laplacian


def calculate_laplacian_matrix(vertices, adj_list, epsilon=1e-6, uniform_weight=True):
    """
    Calculates the Laplacian matrix for the mesh with optional weight schemes.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        epsilon (float): Small value to prevent division by zero.
        uniform_weight (bool): If True, use uniform weights; if False, use distance-based weights.

    Returns:
        np.ndarray: (N, N) Laplacian matrix.
    """
    N = len(vertices)
    laplacian_matrix = np.zeros((N, N))

    for i in range(N):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            if uniform_weight:
                weights = np.ones(len(neighbors_indices)) / len(neighbors_indices)
            else:
                vi = vertices[i]
                neighbor_positions = vertices[neighbors_indices]
                distances = np.linalg.norm(vi - neighbor_positions, axis=1)
                weights = 1.0 / (distances + epsilon)
                sum_weights = np.sum(weights)
                weights = weights / sum_weights

            laplacian_matrix[i, i] = 1.0

            for j, neighbor_idx in enumerate(neighbors_indices):
                laplacian_matrix[i, neighbor_idx] = -weights[j]

    return laplacian_matrix


def calculate_exponential_edge_weights(
    vertices: np.ndarray,
    adj_list: list[list[int]],
    kappa: float = 30.0,
):
    """
    Distance-dependent adjacency weights from TopoRetarget (arXiv:2606.16272), Eq. (5).

    For each edge (i, j): w_tilde_ij = exp(-kappa * ||v_i - v_j||), then
    row-normalized so that sum_j w_ij = 1 for every vertex i.

    Per the paper, these weights are computed once on the *source* configuration
    and reused unchanged for the robot-side Laplacian computation.

    Args:
        vertices (np.ndarray): (N, 3) array of source vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        kappa (float): Nonnegative inverse-distance decay. For metre inputs its
            unit is 1/metre. The helper's legacy default is not a full-body preset.

    Returns:
        list of np.ndarray: edge_weights[i] holds the normalized weights for the
            neighbors of vertex i, aligned with the order of adj_list[i].
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Exponential weights require finite (N, 3) source vertices.")
    if len(adj_list) != len(vertices):
        raise ValueError("Adjacency rows must match source vertex order and count.")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative.")
    edge_weights = []
    for i in range(len(vertices)):
        neighbors_indices = np.asarray(adj_list[i])
        if len(neighbors_indices) > 0:
            if (neighbors_indices.ndim != 1 or neighbors_indices.dtype.kind not in "iu"
                    or np.any(neighbors_indices < 0) or np.any(neighbors_indices >= len(vertices))
                    or i in neighbors_indices or len(np.unique(neighbors_indices)) != len(neighbors_indices)):
                raise ValueError("Adjacency must contain distinct, in-range neighbor indices (no self edges).")
            # Extended arithmetic also protects finite, extreme float64 coordinates
            # against overflow in subtraction/norm. Shift before exponentiation:
            # the nearest edge always has weight 1, even if exp(-k*d) underflows.
            delta = vertices[i].astype(np.longdouble) - vertices[neighbors_indices].astype(np.longdouble)
            distances = np.hypot.reduce(delta, axis=1)
            logits = -np.longdouble(kappa) * (distances - distances.min())
            weights = np.asarray(np.exp(np.maximum(logits, -745)), dtype=float)
            weights = weights / np.sum(weights)
            edge_weights.append(weights)
        else:
            edge_weights.append(np.zeros(0))
    return edge_weights


def calculate_source_laplacian(vertices, adj_list, *, kappa):
    """Freeze Eq. (5)'s source weights into L; isolated vertices have zero rows.

    Node order is exactly [mapped source joints; existing scene samples]. No
    scaling, resampling or robot-pose-dependent weighting occurs here.
    """
    weights = calculate_exponential_edge_weights(vertices, adj_list, kappa=kappa)
    matrix = np.zeros((len(vertices), len(vertices)), dtype=float)
    for i, (neighbors, row) in enumerate(zip(adj_list, weights)):
        if len(neighbors):
            matrix[i, i] = 1.0
            matrix[i, neighbors] = -row
    matrix.setflags(write=False)
    return matrix
