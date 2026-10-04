"""Full conic-QP KKT certificate, including constant rows and complete cones."""

import numpy as np
from scipy import sparse


def _norm(value):
    return float(np.max(np.abs(value), initial=0.))


def cone_violation(vector, nonneg, soc):
    result = float(np.max(-vector[:nonneg], initial=0.))
    sizes = np.asarray(soc, int)
    starts = nonneg + np.r_[0, np.cumsum(sizes)[:-1]]
    for size in np.unique(sizes):
        block = vector[starts[sizes == size, None] + np.arange(size)]
        result = max(result, float(np.max(np.linalg.norm(block[:, 1:], axis=1) - block[:, 0], initial=0.)))
    return max(0., result)


def _project_cones(vector, nonneg, soc):
    """Euclidean cone projection for a window iterate without global slack data."""
    result = np.array(vector, copy=True)
    result[:nonneg] = np.maximum(result[:nonneg], 0.)
    sizes = np.asarray(soc, int)
    starts = nonneg + np.r_[0, np.cumsum(sizes)[:-1]]
    for size in np.unique(sizes):
        indices = starts[sizes == size, None] + np.arange(size)
        values = result[indices]
        norm = np.linalg.norm(values[:, 1:], axis=1)
        outside = norm > values[:, 0]
        vertex = norm <= -values[:, 0]
        curved = outside & ~vertex
        scale = .5 * (norm[curved] + values[curved, 0])
        values[curved, 1:] *= (scale / norm[curved])[:, None]
        values[curved, 0] = scale
        values[vertex] = 0.
        result[indices] = values
    return result


def clarabel_residual(P, c, A, b, nonneg, soc, x, z, settings, slack=None):
    """Clarabel 0.11.1 original-unit L2 stopping residuals, plus cone membership.

    Definitions: Clarabel.rs v0.11.1 default/info.rs:149-160 and
    algebra/vecmath.rs:107-116. The numerical tolerances are unchanged.
    Complementarity is reported; Clarabel tests objective gap, not an extra
    unnormalized dot(s,z) threshold in the presence of primal/dual residuals.
    """
    x, z = np.asarray(x), np.asarray(z)
    if (x.shape != np.shape(c) or z.shape != np.shape(b)
            or not np.isfinite(x).all() or not np.isfinite(z).all()):
        return dict(certified=False, reason='invalid_primal_dual')
    P = sparse.csr_matrix(P)
    P = sparse.triu(P) + sparse.triu(P, 1).T
    px, ax, atz = P @ x, A @ x, A.T @ z
    s = _project_cones(b - ax, nonneg, soc) if slack is None else np.asarray(slack)
    if s.shape != np.shape(b) or not np.isfinite(s).all():
        return dict(certified=False, reason='invalid_slack')
    norm = np.linalg.norm
    # ProblemData caches infinity norms for b/q; variable/residual norms are L2.
    primal = float(norm(ax + s - b) / max(1., _norm(b) + norm(x) + norm(s)))
    stationary = float(norm(px + c + atz) / max(1., _norm(c) + norm(x) + norm(z)))
    primal_cone = cone_violation(s, nonneg, soc) / max(1., norm(s))
    dual_cone = cone_violation(z, nonneg, soc) / max(1., norm(z))
    primal_cost = float(.5 * x @ px + c @ x)
    dual_cost = float(-.5 * x @ px - b @ z)
    gap = abs(primal_cost - dual_cost)
    gap_limit = max(settings.tol_gap_abs, settings.tol_gap_rel *
                    max(1., min(abs(primal_cost), abs(dual_cost))))
    certified = max(primal, stationary, primal_cone, dual_cone) <= settings.tol_feas and gap <= gap_limit
    return dict(certified=bool(certified), primal=primal, stationarity=stationary,
                primal_cone=float(primal_cone), dual=float(dual_cone), gap=gap, gap_limit=gap_limit,
                complementarity=abs(float(s @ z)))


def kkt_residual(P, c, A, b, nonneg, soc, x, z, settings):
    """Conservative residuals in original (pre-equilibration) program units."""
    x, z = np.asarray(x), np.asarray(z)
    if (x.shape != np.shape(c) or z.shape != np.shape(b)
            or not np.isfinite(x).all() or not np.isfinite(z).all()):
        return dict(certified=False, reason='invalid_primal_dual')
    P = sparse.csr_matrix(P)
    P = sparse.triu(P) + sparse.triu(P, 1).T
    px, ax, atz = P @ x, A @ x, A.T @ z
    slack = b - ax
    primal = cone_violation(slack, nonneg, soc) / max(1., _norm(b), _norm(ax))
    dual = cone_violation(z, nonneg, soc) / max(1., _norm(z))
    stationary = _norm(px + c + atz) / max(1., _norm(px), _norm(c), _norm(atz))
    primal_cost = float(.5 * x @ px + c @ x)
    dual_cost = float(-.5 * x @ px - b @ z)
    gap = abs(primal_cost - dual_cost)
    complementarity = abs(float(slack @ z))
    gap_limit = max(settings.tol_gap_abs,
                    settings.tol_gap_rel * min(abs(primal_cost), abs(dual_cost)))
    certified = (max(primal, dual, stationary) <= settings.tol_feas
                 and max(gap, complementarity) <= gap_limit)
    return dict(certified=bool(certified), primal=primal, dual=dual, stationarity=stationary,
                gap=gap, complementarity=complementarity, gap_limit=gap_limit)


def certified_solution(program, P, c, result):
    from .qp_types import settings
    return kkt_residual(P, c, program.A, program.b, program.nonneg, program.soc,
                        result.x, result.z, settings())['certified']
