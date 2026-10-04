"""Experimental acquisition cone solver with cached trajectory-local assembly.

Solves Clarabel standard-form convex QPs over an orthant and small SOCs by
Mehrotra predictor-corrector steps with Nesterov-Todd scaling. Early iterations
use banded Cholesky. Late iterations factor a regularized quasi-definite
augmented KKT matrix by frame-interleaved LDLᵀ, keeping small cone eigenvalues
out of normal equations. Refinement uses the unregularized augmented equations.

This CPU experiment is separate from the pipeline. Unsupported large cones,
structure or convergence are reported explicitly for a caller to use Clarabel.
"""

from __future__ import annotations

import time
import ctypes
import importlib.util
import os
import subprocess
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import LinAlgError, cho_solve_banded, cholesky_banded
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import splu

# Only small, trajectory-local cones are supported by the acquisition experiment.
DENSE_CONE = 16
# A reduced matrix wider than this is not treated as banded.
MAX_BANDWIDTH = 1200
STEP_FRACTION = 0.99
KKT_REFINEMENT = 10
KKT_TOLERANCE = 1e-10
# A Newton direction with a larger relative residual ends the solve.
DIRECTION_ACCURACY = 1e-6
REDUCED_FEAS = 1e-4
REDUCED_GAP = 5e-5
# Static regularization of the scaled problem (Clarabel uses 1e-8 on its KKT diagonal).
DELTA = 1e-9
RHO = 1e-9
# Relative diagonal boosts tried when the banded factorization breaks down;
# the refinement on the unregularized system removes their effect.
DYNAMIC_REGULARIZATION = (0.0, 1e-12, 1e-10, 1e-8, 1e-6)
EQUILIBRATION_ITERATIONS = 10
DEBUG = False
USE_AUGMENTED = True
SWITCH_MU = 1e-6
INITIAL_ZERO = False
INITIAL_MARGIN = 1.0
INITIAL_CENTRE = False


@dataclass
class Result:
    x: np.ndarray
    s: np.ndarray
    z: np.ndarray
    status: str
    iterations: int
    solve_time: float
    obj_val: float
    bandwidth: int


class _Cones:
    """Index layout of R₊ˡ × Q^{m₁} × … in a stacked slack vector."""

    def __init__(self, nonneg, soc):
        self.l = int(nonneg)
        self.soc = np.asarray(soc, dtype=int)
        self.starts = self.l + np.r_[0, np.cumsum(self.soc)[:-1]].astype(int)
        self.m = self.l + int(self.soc.sum())
        self.degree = self.l + len(self.soc)
        self.small = np.flatnonzero(self.soc <= DENSE_CONE)
        self.large = np.flatnonzero(self.soc > DENSE_CONE)
        # Small cones grouped by dimension for vectorized arithmetic.
        self.groups = []
        for size in np.unique(self.soc[self.small]) if len(self.small) else []:
            which = self.small[self.soc[self.small] == size]
            index = (self.starts[which, None] + np.arange(size)).astype(np.int32)
            self.groups.append((int(size), which, index))

    def identity(self):
        e = np.zeros(self.m)
        e[: self.l] = 1.0
        e[self.starts] = 1.0
        return e

    def max_step(self, v, dv):
        """Largest α ≤ 1 with v + α dv in the cone (v interior)."""
        alpha = 1.0
        if self.l:
            neg = dv[: self.l] < 0
            if neg.any():
                alpha = min(alpha, float(np.min(-v[: self.l][neg] / dv[: self.l][neg])))
        for size, _, index in self.groups:
            alpha = min(alpha, _numeric_library()[0].maxstep(len(index), size, index.ravel(), v, dv))
        for c in self.large:
            index = self.starts[c] + np.arange(self.soc[c])
            alpha = min(alpha, _soc_step(v[index][None], dv[index][None]))
        return alpha

    def violation(self, v):
        """Smallest cone 'margin' of v (negative: outside)."""
        margin = np.inf
        if self.l:
            margin = float(v[: self.l].min())
        for size, _, index in self.groups:
            block = v[index]
            margin = min(margin, float((block[:, 0] - np.linalg.norm(block[:, 1:], axis=1)).min()))
        for c in self.large:
            block = v[self.starts[c]:self.starts[c] + self.soc[c]]
            margin = min(margin, float(block[0] - np.linalg.norm(block[1:])))
        return margin


def _soc_step(v, dv):
    """Largest α ∈ [0, 1] keeping each row of v + α dv in the second-order cone."""
    a = dv[:, 0] ** 2 - np.sum(dv[:, 1:] ** 2, axis=1)
    b = v[:, 0] * dv[:, 0] - np.sum(v[:, 1:] * dv[:, 1:], axis=1)
    radius = np.linalg.norm(v[:, 1:], axis=1)
    c = (v[:, 0] - radius) * (v[:, 0] + radius)
    alpha = np.ones(len(v))
    # Boundary of t² − ‖u‖² = 0 on the ray, plus the t ≥ 0 half.
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = b * b - a * c
        root = np.sqrt(np.maximum(disc, 0.0))
        # Roots of a α² + 2 b α + c = 0.
        stable = -b - np.copysign(root, b)
        r1 = np.where(a != 0, stable / a, np.where(b != 0, -c / (2 * b), np.inf))
        r2 = np.where(stable != 0, c / stable, np.inf)
    for r in (r1, r2):
        hit = np.isfinite(r) & (r > 0)
        alpha[hit] = np.minimum(alpha[hit], r[hit])
    t_neg = dv[:, 0] < 0
    with np.errstate(divide="ignore"):
        tt = np.where(t_neg, -v[:, 0] / dv[:, 0], np.inf)
    alpha = np.minimum(alpha, tt)
    return float(alpha.min()) if len(alpha) else 1.0


def _jnorm2(block):
    """tᵀJt = (t₀ − ‖t₁‖)(t₀ + ‖t₁‖), accurate near the cone boundary."""
    tail = np.linalg.norm(block[:, 1:], axis=1)
    return np.maximum((block[:, 0] - tail) * (block[:, 0] + tail), np.finfo(float).tiny)


def _jprod(block):
    out = -block
    out[..., 0] = block[..., 0]
    return out


class _Scaling:
    """Nesterov-Todd scaling W (symmetric) with Wz = W⁻¹s = λ."""

    def __init__(self, cones, s, z):
        self.cones = cones
        l = cones.l
        self.d = np.sqrt(s[:l] / z[:l])            # W = diag(d) on R₊
        self.lam = np.empty(cones.m)
        self.lam_det = np.empty(cones.m)
        self.lam[:l] = np.sqrt(s[:l] * z[:l])
        self.small = []
        for size, which, index in cones.groups:
            self.small.append(self._soc(s[index], z[index], index))
        self.large = []
        for c in cones.large:
            index = (cones.starts[c] + np.arange(cones.soc[c]))[None]
            self.large.append(self._soc(s[index], z[index], index))

    def _soc(self, s, z, index):
        """NT point w̄ (W² = η²(2w̄w̄ᵀ − J)) and its Jordan square root v (W = η(2vvᵀ − J))."""
        count, size = s.shape
        wb, root, eta = np.empty_like(s), np.empty_like(s), np.empty(count)
        _numeric_library()[0].scaling(count, size, s.ravel(), z.ravel(), index.ravel(),
                                     wb.ravel(), root.ravel(), eta, self.lam, self.lam_det)
        return index, root, wb, eta

    def apply(self, v, inverse=False):
        out = np.empty_like(v)
        l = self.cones.l
        out[:l] = v[:l] / self.d if inverse else v[:l] * self.d
        for index, root, _, eta in self.small + self.large:
            out[index] = _soc_w(root, eta, v[index], inverse)
        return out

    def square(self, v):
        """W²v, using the same entries as the augmented factorization."""
        out = np.empty_like(v)
        l = self.cones.l
        out[:l] = self.d ** 2 * v[:l]
        blocks = getattr(self, "_square_blocks", None)
        if blocks is None:
            blocks = []
            for index, _, wb, eta in self.small:
                size = wb.shape[1]
                block = 2 * wb[:, :, None] * wb[:, None, :]
                block -= np.diag(np.r_[1., -np.ones(size - 1)])[None]
                blocks.append((index, (eta ** 2)[:, None, None] * block))
            self._square_blocks = blocks
        for index, block in blocks:
            out[index] = np.einsum("kij,kj->ki", block, v[index])
        return out

    def regularized_blocks(self, delta):
        """(W_c² + δI)⁻¹ for every small cone, as (count, size, size) arrays (cached)."""
        cached = getattr(self, "_blocks", None)
        if cached is None or cached[0] != delta:
            blocks = []
            for index, root, wb, eta in self.small:
                radius = np.linalg.norm(wb[:, 1:], axis=1)
                u = np.divide(wb[:, 1:], radius[:, None], out=np.zeros_like(wb[:, 1:]),
                              where=radius[:, None] > 0)
                stretch = wb[:, 0] + radius
                plus = 1.0 / (eta**2 * stretch**2 + delta)
                minus = 1.0 / (eta**2 / stretch**2 + delta)
                middle = 1.0 / (eta**2 + delta)
                average = (plus + minus) * 0.5
                block = np.zeros((len(wb), wb.shape[1], wb.shape[1]))
                block[:, 0, 0] = average
                block[:, 0, 1:] = block[:, 1:, 0] = ((plus - minus) * 0.5)[:, None] * u
                block[:, 1:, 1:] = (average - middle)[:, None, None] * u[:, :, None] * u[:, None, :]
                diagonal = np.arange(1, wb.shape[1])
                block[:, diagonal, diagonal] += middle[:, None]
                blocks.append((index, block))
            self._blocks = cached = (delta, blocks)
        return cached[1]


def _soc_w(root, eta, v, inverse):
    """W v = η(2 r rᵀ v − J v);  W⁻¹ v = (1/η)(2 Jr (Jr)ᵀ v − J v), r the Jordan root of w̄."""
    if inverse:
        u = _jprod(root)
        return (2.0 * u * np.sum(u * v, axis=1)[:, None] - _jprod(v)) / eta[:, None]
    return eta[:, None] * (2.0 * root * np.sum(root * v, axis=1)[:, None] - _jprod(v))


def _circ(cones, u, v):
    """Jordan product u ∘ v."""
    out = np.empty_like(u)
    l = cones.l
    out[:l] = u[:l] * v[:l]
    for size, which, index in cones.groups:
        a, b = u[index], v[index]
        out[index[:, 0]] = np.sum(a * b, axis=1)
        out[index[:, 1:]] = a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:]
    for c in cones.large:
        index = cones.starts[c] + np.arange(cones.soc[c])
        a, b = u[index], v[index]
        out[index[0]] = a @ b
        out[index[1:]] = a[0] * b[1:] + b[0] * a[1:]
    return out


def _circ_solve(cones, lam, v, determinants=None):
    """x with λ ∘ x = v."""
    out = np.empty_like(v)
    l = cones.l
    out[:l] = v[:l] / lam[:l]
    def solve(a, b, det=None):
        # λ∘x = v with λ = (a0, a1):  x0 = (a0 b0 − a1ᵀb1)/det,  x1 = (b1 − a1 x0)/a0
        det = _jnorm2(a) if det is None else det
        x0 = (a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1)) / det
        x1 = (b[:, 1:] - a[:, 1:] * x0[:, None]) / a[:, :1]
        return np.concatenate([x0[:, None], x1], axis=1)
    for size, which, index in cones.groups:
        out[index] = solve(lam[index], v[index],
                           None if determinants is None else determinants[index[:, 0]])
    for c in cones.large:
        index = cones.starts[c] + np.arange(cones.soc[c])
        out[index] = solve(lam[index][None], v[index][None],
                           None if determinants is None else determinants[index[:1]])[0]
    return out


class _Banded:
    """Banded Cholesky of a symmetric sparse matrix in a bandwidth-reducing order."""

    def __init__(self, pattern):
        n = pattern.shape[0]
        natural = _bandwidth(pattern, np.arange(n))
        perm = reverse_cuthill_mckee(sparse.csr_matrix(pattern), symmetric_mode=True)
        reduced = _bandwidth(pattern, perm)
        self.perm = np.arange(n) if natural <= reduced else np.asarray(perm)
        self.width = min(natural, reduced)
        self.inverse = np.empty(n, dtype=int)
        self.inverse[self.perm] = np.arange(n)
        self.n = n

    def factor(self, matrix, shift):
        # CVXPY's local epigraph stars can have a wide bandwidth even though
        # sparse elimination stays cheap. Do not allocate a mostly empty band.
        if self.width > 256:
            self.sparse_factor = splu((matrix + shift * sparse.eye(self.n)).tocsc(),
                                     permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                                     options={"SymmetricMode": True})
            return
        coo = sparse.triu(matrix, format="coo")
        row, col = self.inverse[coo.row], self.inverse[coo.col]
        upper = row <= col
        r = np.where(upper, row, col)
        c = np.where(upper, col, row)
        ab = np.zeros((self.width + 1, self.n))
        np.add.at(ab, (self.width - (c - r), c), coo.data)
        ab[self.width] += shift
        self.chol = cholesky_banded(ab, lower=False, check_finite=False)

    def solve(self, rhs):
        if self.width > 256:
            return self.sparse_factor.solve(rhs)
        permuted = rhs[self.perm]
        out = cho_solve_banded((self.chol, False), permuted, check_finite=False)
        result = np.empty_like(out)
        result[self.perm] = out
        return result


def _bandwidth(pattern, order):
    coo = sparse.coo_matrix(pattern)
    inverse = np.empty(len(order), dtype=int)
    inverse[order] = np.arange(len(order))
    if not coo.nnz:
        return 0
    return int(np.abs(inverse[coo.row] - inverse[coo.col]).max())


@lru_cache(maxsize=1)
def _numeric_library():
    """Compile the cached elimination recurrence into an anonymous memory file."""
    source = r"""
    #include <math.h>
    int numeric(int n, const int *ap, const int *ai, const double *ax,
                const int *lp, const int *li, double *lx, double *d, double *di,
                const int *rp, const int *rj, const int *pos, double *y,
                const int *perm, int primal) {
        int positive = 0;
        for (int k = 0; k < n; ++k) y[k] = 0.;
        for (int k = 0; k < n; ++k) {
            double pivot = 0.;
            for (int a = ap[k]; a < ap[k+1]; ++a) {
                if (ai[a] == k) pivot = ax[a]; else y[ai[a]] = ax[a];
            }
            for (int p = rp[k]; p < rp[k+1]; ++p) {
                int j = rj[p], entry = pos[p];
                double value = y[j], l = value * di[j];
                for (int q = lp[j]; q < entry; ++q) y[li[q]] -= lx[q] * value;
                lx[entry] = l;
                pivot -= value * l;
                y[j] = 0.;
            }
            double sign = perm[k] < primal ? 1. : -1.;
            if (sign * pivot < 1e-13) pivot = sign * 2e-7;
            if (!isfinite(pivot)) return -1;
            d[k] = pivot; di[k] = 1. / pivot;
            positive += pivot > 0.;
        }
        return positive;
    }
    void scaling(int count, int dim, const double *s, const double *z, const int *ix,
                 double *wb, double *root, double *eta, double *lam, double *det) {
        for (int c = 0; c < count; ++c) {
            int off = c*dim;
            double rs = 0., rz = 0., sb[16], zb[16], product = 0., radius = 0.;
            for (int j = 1; j < dim; ++j) { rs += s[off+j]*s[off+j]; rz += z[off+j]*z[off+j]; }
            rs = sqrt(rs); rz = sqrt(rz);
            double sn = sqrt(fmax((s[off]-rs)*(s[off]+rs), 2.2250738585072014e-308));
            double zn = sqrt(fmax((z[off]-rz)*(z[off]+rz), 2.2250738585072014e-308));
            for (int j = 0; j < dim; ++j) { sb[j] = s[off+j]/sn; zb[j] = z[off+j]/zn; product += sb[j]*zb[j]; }
            double gamma = sqrt((1.+product)*.5), denom = 2.*gamma+sb[0]+zb[0];
            for (int j = 1; j < dim; ++j) {
                wb[off+j] = (sb[j]-zb[j])/(2.*gamma);
                radius += wb[off+j]*wb[off+j];
            }
            wb[off] = sqrt(1.+radius);
            double divisor = sqrt(2.*(wb[off]+1.)), magnitude = sqrt(sn*zn);
            root[off] = (wb[off]+1.)/divisor;
            for (int j = 1; j < dim; ++j) {
                root[off+j] = wb[off+j]/divisor;
                lam[ix[off+j]] = magnitude*((gamma+zb[0])*sb[j]+(gamma+sb[0])*zb[j])/denom;
            }
            eta[c] = sqrt(sn/zn); lam[ix[off]] = magnitude*gamma; det[ix[off]] = sn*zn;
        }
    }
    double maxstep(int count, int dim, const int *ix, const double *v, const double *dv) {
        double alpha = 1.;
        for (int c = 0; c < count; ++c) {
            double a = 0., b = 0., radius = 0.;
            int off = c*dim, i0 = ix[off];
            for (int j = 1; j < dim; ++j) {
                int k = ix[off+j]; a += dv[k]*dv[k]; b += v[k]*dv[k]; radius += v[k]*v[k];
            }
            a = dv[i0]*dv[i0]-a; b = v[i0]*dv[i0]-b; radius = sqrt(radius);
            double det = (v[i0]-radius)*(v[i0]+radius);
            double q = -b-copysign(sqrt(fmax(b*b-a*det, 0.)), b);
            double r1 = a != 0. ? q/a : (b != 0. ? -det/(2.*b) : INFINITY);
            double r2 = q != 0. ? det/q : INFINITY;
            if (r1 > 0. && isfinite(r1)) alpha = fmin(alpha, r1);
            if (r2 > 0. && isfinite(r2)) alpha = fmin(alpha, r2);
            if (dv[i0] < 0.) alpha = fmin(alpha, -v[i0]/dv[i0]);
        }
        return alpha;
    }
    void linear(int count, const int *map, const int *rows, const double *coeff,
                const double *weights, double *ab) {
        for (int i = 0; i < count; ++i) ab[map[i]] += coeff[i] * weights[rows[i]];
    }
    void conic(int count, int dim, const int *ptr, const int *map,
               const double *a, const double *e, double *ab) {
        int pair = 0;
        for (int c = 0; c < count; ++c) {
            const double *block = e + c*dim*dim;
            for (int i = ptr[c]; i < ptr[c+1]; ++i) {
                double temp[16];
                for (int r = 0; r < dim; ++r) {
                    temp[r] = 0.;
                    for (int t = 0; t < dim; ++t) temp[r] += block[r*dim+t]*a[i*dim+t];
                }
                for (int j = ptr[c]; j <= i; ++j) {
                    double value = 0.;
                    for (int r = 0; r < dim; ++r) value += temp[r]*a[j*dim+r];
                    ab[map[pair++]] += value;
                }
            }
        }
    }
    """
    create = ctypes.CDLL(None).memfd_create
    create.argtypes, create.restype = [ctypes.c_char_p, ctypes.c_uint], ctypes.c_int
    fd = create(b"trajectory_ldl", 0)
    if fd < 0:
        raise OSError("Could not create the numeric kernel's memory file")
    path = f"/proc/self/fd/{fd}"
    subprocess.run(["cc", "-O3", "-march=native", "-fPIC", "-shared", "-x", "c", "-", "-o", path],
                   input=source.encode(), pass_fds=(fd,), check=True, capture_output=True, timeout=30)
    lib = ctypes.CDLL(path)
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    fp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.numeric.argtypes = [ctypes.c_int, ip, ip, fp, ip, ip, fp, fp, fp,
                            ip, ip, ip, fp, ip, ctypes.c_int]
    lib.numeric.restype = ctypes.c_int
    lib.scaling.argtypes = [ctypes.c_int, ctypes.c_int, fp, fp, ip, fp, fp, fp, fp, fp]
    lib.scaling.restype = None
    lib.maxstep.argtypes = [ctypes.c_int, ctypes.c_int, ip, fp, fp]
    lib.maxstep.restype = ctypes.c_double
    lib.linear.argtypes = [ctypes.c_int, ip, ip, fp, fp, fp]
    lib.conic.argtypes = [ctypes.c_int, ctypes.c_int, ip, ip, fp, fp, fp]
    lib.linear.restype = lib.conic.restype = None
    return lib, fd


class _ReducedAssembly:
    """Scatter local Gram contributions into a fixed LAPACK band, without CSR rebuilds."""

    def __init__(self, P, A, cones, banded):
        n, width = banded.n, banded.width
        inverse = banded.inverse
        def locations(row, col):
            return ((width - abs(row - col)) * n + np.maximum(row, col)).astype(np.int32)
        self.base = np.zeros((width + 1, n))
        pc = sparse.triu(P, format="coo")
        np.add.at(self.base.ravel(), locations(inverse[pc.row], inverse[pc.col]), pc.data)
        self.base[-1] += RHO
        linear = A[:cones.l]
        counts = np.diff(linear.indptr)
        maps, rows, coefficients = [], [], []
        for k in np.unique(counts[counts > 0]):
            which = np.flatnonzero(counts == k)
            offsets = linear.indptr[which, None] + np.arange(k)
            cols, vals = inverse[linear.indices[offsets]], linear.data[offsets]
            r, c = np.tril_indices(k)
            maps.append(locations(cols[:, r], cols[:, c]).ravel())
            rows.append(np.repeat(which, len(r)).astype(np.int32))
            coefficients.append((vals[:, r] * vals[:, c]).ravel())
        self.linear = (np.concatenate(maps) if maps else np.empty(0, np.int32),
                       np.concatenate(rows) if rows else np.empty(0, np.int32),
                       np.concatenate(coefficients) if coefficients else np.empty(0))
        self.groups = []
        for size, _, index in cones.groups:
            local = A[index.ravel()].tocoo()
            cone, component = local.row // size, local.row % size
            keys, mapping = np.unique(cone.astype(np.int64) * n + inverse[local.col],
                                      return_inverse=True)
            counts = np.bincount(keys // n, minlength=len(index))
            ptr = np.r_[0, np.cumsum(counts)].astype(np.int32)
            cols = keys % n
            packed = np.zeros((len(keys), size))
            np.add.at(packed, (mapping, component), local.data)
            pairptr = np.r_[0, np.cumsum(counts * (counts + 1) // 2)]
            pairmap = np.empty(pairptr[-1], np.int32)
            for k in np.unique(counts):
                which = np.flatnonzero(counts == k)
                variables = cols[ptr[which, None] + np.arange(k)]
                r, c = np.tril_indices(k)
                pairmap[pairptr[which, None] + np.arange(len(r))] = locations(variables[:, r], variables[:, c])
            self.groups.append((size, ptr, pairmap, packed.ravel()))
        self.kernel = _numeric_library()[0]

    def assemble(self, scaling):
        ab = self.base.copy()
        mapping, rows, coeff = self.linear
        weights = 1.0 / (scaling.d ** 2 + DELTA)
        self.kernel.linear(len(mapping), mapping, rows, coeff, weights, ab.ravel())
        for (_, blocks), (size, ptr, mapping, packed) in zip(scaling.regularized_blocks(DELTA), self.groups):
            self.kernel.conic(len(ptr) - 1, size, ptr, mapping, packed,
                              np.ascontiguousarray(blocks).ravel(), ab.ravel())
        return ab


class _Augmented:
    """Cached sparse LDLᵀ of the frame-interleaved quasi-definite KKT matrix.

    The installed QOCO extension exports its int32/float64 QDLDL kernel with
    sign-aware dynamic regularization. Only this kernel is used, not QOCO's IPM.
    """

    @staticmethod
    def _library():
        spec = importlib.util.find_spec("qoco.qoco_ext")
        if spec is None:
            raise ImportError("The augmented experiment requires QOCO's QDLDL kernel")
        lib = ctypes.CDLL(spec.origin)
        ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        fp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        bp = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        lib.QDLDL_etree.argtypes = [ctypes.c_int, ip, ip, ip, ip, ip]
        lib.QDLDL_etree.restype = ctypes.c_int
        lib.QDLDL_factor.argtypes = [ctypes.c_int, ip, ip, fp, ip, ip, fp, fp, fp,
                                         ip, ip, bp, ip, fp, ip, ctypes.c_int, ctypes.c_double]
        lib.QDLDL_factor.restype = ctypes.c_int
        lib.QDLDL_solve.argtypes = [ctypes.c_int, ip, ip, fp, fp, fp]
        lib.QDLDL_solve.restype = None
        return lib

    def __init__(self, P, A, cones, frame_width=35):
        self.lib = self._library()
        self.primal = A.shape[1]
        self.n = self.primal + cones.m
        # A constraint/cone is visited at its earliest touched frame. Local
        # future-frame couplings then fill only the trajectory frontier.
        ac = A.tocoo()
        anchors = np.full(cones.m, self.primal // frame_width + 1, dtype=int)
        np.minimum.at(anchors, ac.row, ac.col // frame_width)
        for _, _, ix in cones.groups:
            anchors[ix] = anchors[ix].min(axis=1)[:, None]
        keys = np.r_[2 * (np.arange(self.primal) // frame_width) + 1, 2 * anchors]
        self.perm = np.argsort(keys, kind="stable").astype(np.int32)
        inverse = np.empty(self.n, dtype=int)
        inverse[self.perm] = np.arange(self.n)
        pc = sparse.triu(P, format="coo")
        rows = [pc.row, ac.col, self.primal + np.arange(cones.l)]
        cols = [pc.col, self.primal + ac.row, self.primal + np.arange(cones.l)]
        values = [pc.data, ac.data, -np.ones(cones.l)]
        self.cone_layout = []
        offset = pc.nnz + ac.nnz + cones.l
        for size, _, ix in cones.groups:
            r, c = np.triu_indices(size)
            rows.append((self.primal + ix[:, r]).ravel())
            cols.append((self.primal + ix[:, c]).ravel())
            values.append(np.zeros(len(ix) * len(r)))
            self.cone_layout.append((r, c, offset, len(ix) * len(r)))
            offset += len(ix) * len(r)
        # Primal diagonals are present even for a semidefinite objective.
        rows.append(np.arange(self.primal))
        cols.append(np.arange(self.primal))
        values.append(np.full(self.primal, RHO))
        row, col = inverse[np.concatenate(rows)], inverse[np.concatenate(cols)]
        rr, cc = np.minimum(row, col), np.maximum(row, col)
        # Duplicate primal diagonal entries need a many-to-one assembly map.
        key = cc.astype(np.int64) * self.n + rr
        unique, mapping = np.unique(key, return_inverse=True)
        self.mapping = mapping
        self.base = np.concatenate(values)
        self.linear_offset = pc.nnz + ac.nnz
        self.linear_count = cones.l
        self.indices = (unique % self.n).astype(np.int32)
        counts = np.bincount(unique // self.n, minlength=self.n)
        self.indptr = np.r_[0, np.cumsum(counts)].astype(np.int32)
        self.data = np.zeros(len(unique))
        self.diagonal = np.flatnonzero(self.indices == np.repeat(np.arange(self.n), counts))
        self._allocate()

    def _allocate(self):
        self.lnz, self.etree = np.empty(self.n, np.int32), np.empty(self.n, np.int32)
        self.iwork = np.empty(3 * self.n, np.int32)
        nz = self.lib.QDLDL_etree(self.n, self.indptr, self.indices, self.iwork,
                                 self.lnz, self.etree)
        if nz < 0:
            raise LinAlgError("Invalid augmented sparsity pattern")
        self.lp = np.empty(self.n + 1, np.int32)
        self.li, self.lx = np.empty(nz, np.int32), np.empty(nz)
        self.d, self.di = np.empty(self.n), np.empty(self.n)
        self.bwork, self.fwork = np.empty(self.n, np.uint8), np.empty(self.n)
        self.row_structure = None

    def factor(self, scaling):
        data = self.base.copy()
        start = self.linear_offset
        data[start:start + self.linear_count] = -(scaling.d ** 2)
        for (_, _, wb, eta), (r, c, start, length) in zip(scaling.small, self.cone_layout):
            blocks = 2 * wb[:, r] * wb[:, c]
            diagonal = r == c
            signs = np.where(r[diagonal] == 0, 1., -1.)
            blocks[:, diagonal] -= signs
            data[start:start + length] = (-(eta ** 2)[:, None] * blocks).ravel()
        self.data[:] = np.bincount(self.mapping, weights=data, minlength=len(self.data))
        self.data[self.diagonal[self.perm >= self.primal]] -= DELTA
        self._factor_data()

    def _factor_data(self):
        if self.row_structure is not None:
            rp, rj, positions = self.row_structure
            positive = _numeric_library()[0].numeric(
                self.n, self.indptr, self.indices, self.data, self.lp, self.li,
                self.lx, self.d, self.di, rp, rj, positions, self.fwork, self.perm, self.primal)
        else:
            positive = self.lib.QDLDL_factor(self.n, self.indptr, self.indices, self.data,
                                        self.lp, self.li, self.lx, self.d, self.di,
                                        self.lnz, self.etree, self.bwork, self.iwork,
                                        self.fwork, self.perm, self.primal, 2e-7)
            rows = sparse.csc_matrix((np.arange(len(self.li), dtype=np.int32), self.li, self.lp),
                                     shape=(self.n, self.n)).tocsr()
            self.row_structure = (rows.indptr, rows.indices, rows.data)
        if positive != self.primal or not np.isfinite(self.d).all():
            raise LinAlgError("Augmented LDLᵀ factorization failed")

    def solve(self, rx, rz):
        rhs = np.r_[rx, rz]
        out = rhs[self.perm].copy()
        self.lib.QDLDL_solve(self.n, self.lp, self.li, self.lx, self.di, out)
        answer = np.empty_like(out)
        answer[self.perm] = out
        return answer[:self.primal], answer[self.primal:]



def _equilibrate(P, q, A, b, cones, iterations=EQUILIBRATION_ITERATIONS):
    """Ruiz scaling D (variables), E (rows; one value per second-order cone), cost c.

    Returns the scaled data P̂ = cDPD, q̂ = cDq, Â = EAD, b̂ = Eb with
    x = D x̂, s = E⁻¹ ŝ, z = E ẑ / c. Cones are invariant under E.
    """
    n, m = A.shape[1], A.shape[0]
    d, e = np.ones(n), np.ones(m)
    P = P.tocsc(copy=True)
    A = A.tocsc(copy=True)
    pcols = np.repeat(np.arange(n), np.diff(P.indptr))
    acols = np.repeat(np.arange(n), np.diff(A.indptr))
    # Fixed CSR row layout gives row maxima without repeated CSC conversions.
    rows = A.copy()
    rows.data = np.arange(A.nnz, dtype=np.int32)
    rows = rows.tocsr()
    row_nonempty = np.diff(rows.indptr) > 0
    cone_of_row = np.full(m, -1)
    for c, (start, size) in enumerate(zip(cones.starts, cones.soc)):
        cone_of_row[start:start + size] = c
    in_cone = cone_of_row >= 0
    for _ in range(iterations):
        col = np.maximum(_column_inf(P), _column_inf(A))
        delta = 1.0 / np.sqrt(np.clip(col, 1e-4, 1e4))
        delta[col == 0] = 1.0
        row = np.zeros(m)
        row[row_nonempty] = np.maximum.reduceat(np.abs(A.data[rows.data]),
                                                 rows.indptr[:-1][row_nonempty])
        eps = 1.0 / np.sqrt(np.clip(row, 1e-4, 1e4))
        eps[row == 0] = 1.0
        if in_cone.any():
            # One factor per cone: the geometric mean of its rows' factors.
            sums = np.bincount(cone_of_row[in_cone], np.log(eps[in_cone]), minlength=len(cones.soc))
            eps[in_cone] = np.exp(sums / cones.soc)[cone_of_row[in_cone]]
        P.data *= delta[P.indices] * delta[pcols]
        A.data *= eps[A.indices] * delta[acols]
        d *= delta
        e *= eps
    q = d * q
    scale = 1.0 / max(np.abs(P.diagonal()).mean() if P.nnz else 0.0, np.abs(q).max(initial=0.0), 1e-4)
    scale = min(scale, 1e4)
    return (scale * P).tocsr(), scale * q, A.tocsr(), e * b, d, e, scale


def _column_inf(matrix):
    """Infinity norm of every column of a CSC matrix."""
    matrix = sparse.csc_matrix(matrix)
    out = np.zeros(matrix.shape[1])
    counts = np.diff(matrix.indptr)
    nonempty = counts > 0
    if matrix.nnz:
        out[nonempty] = np.maximum.reduceat(np.abs(matrix.data), matrix.indptr[:-1][nonempty])
    return out


def solve(P, q, A, b, nonneg, soc, *, max_iter=100, tol_feas=1e-8, tol_gap_abs=1e-8,
          tol_gap_rel=1e-8, verbose=False):
    """Interior-point solve of the standard-form cone program; see module docstring."""
    tick = time.perf_counter()
    q0 = np.asarray(q, float)
    b0 = np.asarray(b, float)
    n = len(q0)
    cones = _Cones(nonneg, soc)
    A0 = sparse.csr_matrix(A)
    if A0.shape != (cones.m, n):
        raise ValueError("Cone dimensions do not match the constraint matrix")
    P0 = sparse.csr_matrix(P if P is not None else (n, n))
    P0 = (sparse.triu(P0) + sparse.triu(P0, 1).T).tocsr()      # Clarabel passes the upper triangle
    if len(cones.large):
        return Result(np.zeros(n), np.zeros(cones.m), np.zeros(cones.m), "UnsupportedCones", 0,
                      time.perf_counter() - tick, np.nan, 0)
    P, q, A, b, dvar, erow, cost = _equilibrate(P0, q0, A0, b0, cones)
    AT = A.T.tocsr()
    nn = A[: cones.l]
    small_rows = (np.concatenate([index.ravel() for _, _, index in cones.groups])
                  if cones.groups else np.zeros(0, int))
    a_small = A[small_rows] if len(small_rows) else None
    block_rows, block_cols, offset = [], [], 0
    for size, which, _ in cones.groups:
        local = np.arange(len(which))[:, None, None] * size + offset
        block_rows.append(np.broadcast_to(local + np.arange(size)[None, :, None], (len(which), size, size)).ravel())
        block_cols.append(np.broadcast_to(local + np.arange(size)[None, None, :], (len(which), size, size)).ravel())
        offset += len(which) * size
    block_rows = np.concatenate(block_rows) if block_rows else np.zeros(0, int)
    block_cols = np.concatenate(block_cols) if block_cols else np.zeros(0, int)
    # Structure of P + AᵀA (all scalings keep it, the rank-two terms are separate).
    pattern = (abs(P) + abs(AT) @ abs(A)).tocsr()
    if a_small is not None:
        dense_pattern = sparse.csr_matrix((np.ones(len(block_rows)), (block_rows, block_cols)),
                                          shape=(len(small_rows),) * 2)
        pattern = pattern + abs(a_small.T) @ dense_pattern @ abs(a_small)
    pattern.data[:] = 1.0
    banded = _Banded(pattern)
    if banded.width > MAX_BANDWIDTH:
        return Result(np.zeros(n), np.zeros(cones.m), np.zeros(cones.m), "NotBanded", 0,
                      time.perf_counter() - tick, np.nan, banded.width)
    assembly = _ReducedAssembly(P, A, cones, banded)
    e = cones.identity()
    augmented = None

    def apply_reg(scaling, v):
        """(W² + δI)⁻¹ v for the supported small cones."""
        out = np.empty_like(v)
        out[: cones.l] = v[: cones.l] / (scaling.d ** 2 + DELTA)
        for index, blocks in scaling.regularized_blocks(DELTA):
            out[index] = np.einsum("kij,kj->ki", blocks, v[index])
        return out

    def kkt_solve(solver, scaling, rx, rz, scaled_rhs):
        """Refine the complete, unregularized augmented linearized system."""
        rhs = rz - scaling.apply(scaled_rhs)
        def direct(ax, az):
            if isinstance(solver, _Augmented):
                dx, dz = solver.solve(ax, az)
                return dx, dz, A @ dx
            dx = solver(ax + AT @ apply_reg(scaling, az))
            adx = A @ dx
            return dx, apply_reg(scaling, adx - az), adx
        dx, dz, adx = direct(rx, rhs)
        norm = max(np.linalg.norm(rx), np.linalg.norm(rhs), 1e-300)
        best_error, best_direction = np.inf, (dx, dz, adx)
        for _ in range(KKT_REFINEMENT + 1):
            if not np.isfinite(dx).all() or not np.isfinite(dz).all():
                dx, dz, adx = best_direction
                break
            ex = rx - P @ dx - AT @ dz
            ez = rhs - adx + scaling.square(dz)
            error = max(np.linalg.norm(ex), np.linalg.norm(ez))
            if not np.isfinite(error) or error >= best_error:
                dx, dz, adx = best_direction
                break
            best_error, best_direction = error, (dx, dz, adx)
            if error <= KKT_TOLERANCE * norm:
                break
            cx, cz, acx = direct(ex, ez)
            dx, dz, adx = dx + cx, dz + cz, adx + acx
        quality = best_error / norm
        if DEBUG:
            print(f"      augmented residual {quality:.1e}")
        return dx, rz - adx, dz, quality

    # Initial point (CVXOPT coneqp): x minimizes ½xᵀPx + qᵀx + ½‖b − Ax‖², s = b − Ax,
    # z starts at the cone identity; each slack is shifted locally into the interior.
    if INITIAL_ZERO:
        x = np.zeros(n)
    else:
        unit = _Scaling(cones, e.copy(), e.copy())
        try:
            banded.chol = cholesky_banded(assembly.assemble(unit), check_finite=False)
        except LinAlgError:
            return Result(np.zeros(n), np.zeros(cones.m), np.zeros(cones.m), "FactorizationFailed", 0,
                          time.perf_counter() - tick, np.nan, banded.width)
        x = banded.solve(-q + AT @ b)
    s = b - A @ x
    z = e.copy()

    def shift(v):
        v = v.copy()
        v[:cones.l] = np.maximum(v[:cones.l], INITIAL_MARGIN)
        for _, _, index in cones.groups:
            block = v[index]
            v[index[:, 0]] = np.maximum(block[:, 0],
                                        np.linalg.norm(block[:, 1:], axis=1) + INITIAL_MARGIN)
        for c in cones.large:
            start, size = cones.starts[c], cones.soc[c]
            v[start] = max(v[start], np.linalg.norm(v[start + 1:start + size]) + 1.0)
        return v

    s, z = shift(s), shift(z)
    if INITIAL_CENTRE:
        z[:cones.l] = 1.0 / s[:cones.l]
        for _, _, ix in cones.groups:
            z[ix] = _jprod(s[ix]) / _jnorm2(s[ix])[:, None]
        mu0 = min(1.0, max(np.linalg.norm(q, np.inf), 1e-6)
                  / max(np.linalg.norm(AT @ z, np.inf), 1e-6))
        z *= mu0

    def unscaled(x, s, z):
        return dvar * x, s / erow, erow * z / cost

    status = "MaxIterations"
    best = (np.inf, None, None, None, 0, np.inf, np.inf, np.inf)
    nb, nq = max(1.0, np.linalg.norm(b0)), max(1.0, np.linalg.norm(q0))
    iteration = 0
    for iteration in range(1, max_iter + 1):
        Px = P @ x
        rx = Px + q + AT @ z
        rz = A @ x + s - b
        gap = float(s @ z)
        xu = dvar * x
        res_p = np.linalg.norm(rz / erow) / max(nb, np.linalg.norm(xu), 1.0)
        res_d = np.linalg.norm(rx / dvar) / (cost * nq)
        quadratic = 0.5 * x @ Px
        primal = (quadratic + q @ x) / cost
        dual = (-quadratic - b @ z) / cost
        gap_u = gap / cost
        if verbose:
            print(f"{iteration:3d} pobj {primal:.8e} gap {gap_u:.2e} rp {res_p:.1e} rd {res_d:.1e}")
        gap_abs = abs(primal - dual)
        gap_rel = gap_abs / max(min(abs(primal), abs(dual)), 1e-300)
        if res_p <= tol_feas and res_d <= tol_feas and (gap_abs <= tol_gap_abs or gap_rel <= tol_gap_rel):
            # Verify the original coordinates before accepting strict accuracy.
            xu, su, zu = unscaled(x, s, z)
            exact_p = np.linalg.norm(A0 @ xu + su - b0) / max(nb, np.linalg.norm(xu), 1.)
            exact_d = np.linalg.norm(P0 @ xu + q0 + A0.T @ zu) / nq
            if exact_p <= tol_feas and exact_d <= tol_feas:
                status = "Solved"
                break
        merit = max(res_p / tol_feas, res_d / tol_feas, min(gap_abs, gap_rel * tol_gap_abs / tol_gap_rel) / tol_gap_abs)
        if merit < best[0]:
            best = (merit, x.copy(), s.copy(), z.copy(), iteration, res_p, res_d, min(gap_abs, gap_rel))
        mu = gap / cones.degree
        scaling = _Scaling(cones, s, z)
        if augmented is None and USE_AUGMENTED and not len(cones.large) and mu <= SWITCH_MU:
            augmented = _Augmented(P, A, cones)
        if augmented is not None:
            try:
                augmented.factor(scaling)
                solver = augmented
            except LinAlgError:
                solver = None
        else:
            ab = assembly.assemble(scaling)
            solver = None
            for boost in DYNAMIC_REGULARIZATION:
                try:
                    shifted = ab.copy()
                    shifted[-1] += boost * np.maximum(abs(ab[-1]), 1.)
                    banded.chol = cholesky_banded(shifted, check_finite=False)
                    solver = banded.solve
                    break
                except LinAlgError:
                    continue
        if solver is None:
            status = "FactorizationFailed"
            break
        def direction(factor):
            lam = scaling.lam
            dx_a, ds_a, dz_a, quality_a = kkt_solve(factor, scaling, -rx, -rz, -lam)
            if not np.isfinite(quality_a) or quality_a > DIRECTION_ACCURACY:
                return None
            alpha = min(cones.max_step(s, ds_a), cones.max_step(z, dz_a))
            sigma = (float((s + alpha * ds_a) @ (z + alpha * dz_a)) / gap) ** 3 if gap > 0 else 0.0
            sigma = min(max(sigma, 0.0), 1.0)
            corr = _circ(cones, scaling.apply(ds_a, inverse=True), scaling.apply(dz_a))
            dx, ds, dz, quality = kkt_solve(factor, scaling, -rx, -rz,
                                           -lam + _circ_solve(cones, lam, -corr + sigma * mu * e,
                                                                        scaling.lam_det))
            if not np.isfinite(quality) or quality > DIRECTION_ACCURACY:
                return None
            return dx, ds, dz
        step = direction(solver)
        if step is None and augmented is None and USE_AUGMENTED:
            # Retry this same Newton step in augmented form before any update.
            augmented = _Augmented(P, A, cones)
            try:
                augmented.factor(scaling)
                step = direction(augmented)
            except LinAlgError:
                pass
        if step is None:
            status = "InsufficientProgress"
            break
        dx, ds, dz = step
        alpha = min(1.0, STEP_FRACTION * min(cones.max_step(s, ds), cones.max_step(z, dz)))
        x = x + alpha * dx
        s = s + alpha * ds
        z = z + alpha * dz
    if status != "Solved" and best[1] is not None:
        _, x, s, z, _, bp, bd, bg = best
        # Clarabel's reduced accuracy ("AlmostSolved"): feasibility 1e-4, gap 5e-5.
        if bp <= REDUCED_FEAS and bd <= REDUCED_FEAS and bg <= REDUCED_GAP:
            status = "AlmostSolved"
    xu, su, zu = unscaled(x, s, z)
    obj = float(0.5 * xu @ (P0 @ xu) + q0 @ xu)
    return Result(xu, su, zu, status, iteration, time.perf_counter() - tick, obj, banded.width)
