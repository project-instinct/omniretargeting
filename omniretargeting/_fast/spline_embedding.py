"""All-frame, fixed-reference tangent corrections with cubic local support."""

import copy

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

from .coarse_embedding import _right_jacobian


class CorrectionSpline:
    """Represent ``q = Integrate(reference, basis @ coefficients)``.

    Coefficients have shape (controls, coords.width), flattened control first.
    The reference is copied once and read-only; refinement never recenters it.
    Jacobians map coefficient increments to right-local pose increments, the
    coordinates consumed by MuJoCo kinematics and ``Coordinates.difference``.
    All frames participate; spline controls are not sampled trajectory poses.
    """

    degree = 3

    def __init__(self, coords, reference, framerate, spacing=.12,
                 identity=False, extra_times=()):
        self.coords = coords
        self.reference = np.array(reference, dtype=float, copy=True)
        if (self.reference.ndim != 2 or not len(self.reference)
                or self.reference.shape[1] != coords.model.nq
                or not np.all(np.isfinite(self.reference))):
            raise ValueError('reference must be finite (nonempty frames, model.nq)')
        self.reference.setflags(write=False)
        if not np.isfinite(framerate) or framerate <= 0:
            raise ValueError('framerate must be finite and positive')
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError('spacing must be finite and positive')
        self.framerate, self.spacing = float(framerate), float(spacing)
        self.count, self.identity = len(reference), bool(identity)
        self.times = np.arange(self.count, dtype=float) / self.framerate
        self.times.setflags(write=False)
        # A singleton still has a nondegenerate cubic domain, with its only
        # observation at zero. No frames are invented or sampled away.
        self.end_time = max(self.times[-1], 1 / self.framerate)
        extra_times = self._validate_times(extra_times)
        if self.identity:
            if len(extra_times):
                raise ValueError('identity embedding has no spline knots')
            self.knot_vector = None
            self.basis = sparse.eye(self.count, format='csr')
        else:
            interior = np.arange(spacing, self.end_time, spacing)
            interior = np.unique(np.r_[interior, extra_times])
            self.knot_vector = np.r_[np.zeros(4), interior,
                                     np.full(4, self.end_time)]
            self.knot_vector.setflags(write=False)
            self.basis = self.basis_at(self.times)
        self.controls = self.basis.shape[1]
        self.coefficients = np.zeros((self.controls, coords.width))
        self.angular = ([] if coords.root_d is None else
                        [coords.inverse.get(coords.root_d + axis) for axis in (3, 4, 5)])
        if self.angular and any(column is None for column in self.angular):
            if not all(column is None for column in self.angular):
                raise ValueError('root angular coordinates must be all active or all locked')
            self.angular = []

    def _validate_times(self, times):
        times = np.asarray(tuple(times), dtype=float)
        if (times.ndim != 1 or not np.all(np.isfinite(times))
                or np.any(times <= 0) or np.any(times >= self.end_time)):
            raise ValueError('extra knots must lie strictly inside the time domain')
        return np.unique(times)

    def _coefficients(self, a):
        a = np.asarray(a, dtype=float)
        if a.shape != (self.controls, self.coords.width) or not np.all(np.isfinite(a)):
            raise ValueError('coefficients must be finite (controls, coords.width)')
        return a

    def basis_at(self, times, derivative=0):
        """Sparse correction basis (or time derivative) at arbitrary times.

        Derivatives are in seconds, and available through order three. In
        identity mode only the original observation times have a definition.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError('evaluation times must be a finite vector')
        if derivative not in (0, 1, 2, 3):
            raise ValueError('derivative must be 0, 1, 2 or 3')
        if self.identity:
            if derivative or not np.array_equal(times, self.times):
                raise ValueError('identity mode only defines original frame values')
            return sparse.eye(self.count, format='csr')
        knots, degree = self.knot_vector, self.degree
        count = len(knots) - degree - 1
        transform = sparse.eye(count, format='csr')
        for _ in range(derivative):
            denominator = knots[degree + 1:degree + count] - knots[1:count]
            scale = degree / denominator
            diff = sparse.diags((-scale, scale), (0, 1), shape=(count - 1, count))
            transform = diff @ transform
            knots, degree, count = knots[1:-1], degree - 1, count - 1
        weights = sparse.csr_matrix(BSpline.design_matrix(times, knots, degree,
                                                          extrapolate=False))
        result = (weights @ transform).tocsr()
        # Derivative rows annihilate constant coefficients. Evaluate the last
        # entry as the complementary sum to avoid cancellation drift in short
        # domains (second derivatives scale as 1 / duration**2).
        if derivative:
            result.sort_indices()
            for row in range(len(times)):
                lo, hi = result.indptr[row:row + 2]
                if hi > lo:
                    total = 0.
                    for value in result.data[lo:hi - 1]:
                        total += value
                    result.data[hi - 1] = -total
        result.eliminate_zeros()
        return result

    @property
    def support_times(self):
        """Closed time support of each control; identity rows are singletons."""
        if self.identity:
            return np.column_stack((self.times, self.times))
        return np.column_stack((self.knot_vector[:self.controls],
                                self.knot_vector[4:4 + self.controls]))

    def tangent_differentials(self, a):
        """Per-frame local differential used to pull back frame Hessians."""
        delta = self.correction(a)
        result = np.broadcast_to(np.eye(self.coords.width),
                                 (self.count, self.coords.width, self.coords.width)).copy()
        if self.angular:
            angular = np.asarray(self.angular)
            result[:, angular[:, None], angular[None, :]] = _right_jacobian(
                delta[:, angular])
        return result

    def correction(self, a, times=None, derivative=0):
        """Evaluate tangent correction, velocity or acceleration in seconds."""
        a = self._coefficients(a)
        if times is None and derivative == 0:
            return self.basis @ a
        return self.basis_at(self.times if times is None else times, derivative) @ a

    def embed(self, a):
        """Integrate local tangent corrections on the unchanged reference."""
        delta = self.correction(a)
        coords = self.coords
        result = self.reference.copy()
        result[:, coords.qi] += delta[:, coords.scalar_cols]
        q, d = coords.root_q, coords.root_d
        if q is not None:
            for axis in range(3):
                column = coords.inverse.get(d + axis)
                if column is not None:
                    result[:, q + axis] += delta[:, column]
            if self.angular:
                rotation = Rotation.from_quat(self.reference[:, q + 3:q + 7], scalar_first=True)
                result[:, q + 3:q + 7] = (rotation * Rotation.from_rotvec(
                    delta[:, self.angular])).as_quat(scalar_first=True)
                zero = np.all(delta[:, self.angular] == 0, axis=1)
                result[zero, q + 3:q + 7] = self.reference[zero, q + 3:q + 7]
        return result

    def jacobian(self, a):
        """CSR d(right-local pose)/d(coefficients), frame/control major."""
        delta = self.correction(a)
        width = self.coords.width
        if not self.angular:
            return sparse.kron(self.basis, sparse.eye(width), format='csr')
        weights = self.basis.tocoo()
        blocks = _right_jacobian(delta[:, self.angular])[weights.row]
        blocks = blocks * weights.data[:, None, None]
        diagonal = np.eye(width)
        diagonal[np.ix_(self.angular, self.angular)] = 0
        linear = sparse.kron(self.basis, diagonal, format='csr')
        rows = weights.row[:, None, None] * width + np.asarray(self.angular)[None, :, None]
        cols = weights.col[:, None, None] * width + np.asarray(self.angular)[None, None, :]
        angular = sparse.coo_matrix((blocks.ravel(),
            (np.broadcast_to(rows, blocks.shape).ravel(),
             np.broadcast_to(cols, blocks.shape).ravel())), shape=linear.shape).tocsr()
        result = linear + angular
        result.eliminate_zeros()
        return result.tocsr()

    def insert_knots(self, times, a):
        """Return ``(new_spline, inherited_a)`` without modifying this instance.

        Boehm insertion preserves the complete continuous correction curve,
        including its derivatives, not just sampled frame values. Existing
        knots are ignored so refinement retains simple interior knots and C2
        continuity. The returned coefficients are also stored on new_spline.
        """
        a = self._coefficients(a)
        times = self._validate_times(times)
        if self.identity:
            raise ValueError('identity embedding cannot insert spline knots')
        curve = BSpline(self.knot_vector, a, self.degree, extrapolate=False)
        for time in times:
            if not np.any(curve.t == time):
                curve = curve.insert_knot(time)
        refined = copy.copy(self)
        refined.knot_vector = curve.t.copy()
        refined.knot_vector.setflags(write=False)
        refined.basis = refined.basis_at(self.times)
        refined.controls = refined.basis.shape[1]
        inherited = curve.c[:refined.controls].copy()
        refined.coefficients = inherited.copy()
        return refined, inherited

    def affected_frames(self, control_indices):
        """Sorted observed frame indices with nonzero selected basis values."""
        indices = self._control_indices(control_indices)
        return np.flatnonzero(np.asarray(self.basis[:, indices].getnnz(axis=1)).ravel())

    def _control_indices(self, indices):
        indices = np.asarray(tuple(indices), dtype=int)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.controls):
            raise ValueError('control indices out of range')
        return indices

    def local_controls(self, start, stop):
        """Controls fully supported inside frame window ``[start, stop)``.

        For cubic mode support must lie between the first and last window
        sample times. Global clamped endpoint controls are excluded: all
        returned basis functions and their first two derivatives vanish at
        window boundaries, preserving the exterior continuous curve exactly.
        Identity mode preserves exterior frames but has no time derivatives.
        """
        if (not isinstance(start, (int, np.integer))
                or not isinstance(stop, (int, np.integer))
                or not 0 <= start <= stop <= self.count):
            raise ValueError('window must be valid half-open frame indices')
        if self.identity:
            return np.arange(start, stop, dtype=int)
        if stop - start < 2:
            return np.empty(0, dtype=int)
        left = self.knot_vector[:self.controls]
        right = self.knot_vector[4:4 + self.controls]
        inside = ((left >= self.times[start]) & (right <= self.times[stop - 1])
                  & (left > 0) & (right < self.end_time))
        return np.flatnonzero(inside)
