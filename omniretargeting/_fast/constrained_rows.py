"""Proposal-only inner margins on the complete frozen acquisition domain.

Physical cost and exact acceptance remain the original objective/guards. Fixed
neighbours participate in residuals; only window tangent columns can move.
"""

import os

import numpy as np
from scipy import sparse

from .barrier_rows import BarrierRows, ball, _locations
from .rows import _block_rows


class ConstrainedRows(BarrierRows):
    def __init__(self, window):
        super().__init__(window)
        from .continuity import derivatives
        self.inner = {}
        # Ten percent of each signal's own 0.1 RMS relaxation, clipped by
        # immutable reference room. The reference has zero hinge residual.
        for kind, guard, _ in self.guards:
            if kind == 'sequence':
                for (name, order), bound in guard.bounds.items():
                    values = np.diff(guard.reference[name][0], n=order, axis=0)
                    size = np.abs(values) if name == 'joints' else np.linalg.norm(values, axis=1)
                    rms = np.sqrt(np.mean(size**2, axis=0))
                    reference = np.zeros_like(size) if guard.correction else size
                    self.inner[type(guard).__name__, name, order] = np.minimum(
                        .01 * rms, .1 * np.maximum(bound - reference, 0.))
            elif kind == 'correction':
                for key, values in derivatives(guard.coords, guard.fixed).items():
                    name, order = key.rsplit('_order_', 1)
                    size = np.abs(values)
                    reference = np.zeros_like(size) if guard.correction else size
                    self.inner[type(guard).__name__, name, int(order)] = np.minimum(
                        .01 * np.sqrt(np.mean(size**2, axis=0)),
                        .1 * np.maximum(guard.bounds[key] - reference, 0.))
                    if key in guard.norm_bounds:
                        size = np.linalg.norm(values, axis=1)
                        reference = np.zeros_like(size) if guard.correction else size
                        self.inner[type(guard).__name__, name, int(order), 'norm'] = np.minimum(
                            .01 * np.sqrt(np.mean(size**2)),
                            .1 * np.maximum(guard.norm_bounds[key] - reference, 0.))
            elif kind == 'shape':
                for i, env in enumerate(guard.envelopes):
                    value = env.layout_sign * np.einsum('fpk,fak->fpa', env.reference, env.axes)
                    self.inner['ReferenceShape', i, 'layout'] = .01 * np.maximum(value, 0.)

    def margins(self, f):
        label = f.label
        if label[0] == 'AcquisitionNormalGuard':
            from . import acquisition_guard
            hinge_margin = getattr(acquisition_guard, 'HINGE_MARGIN_M',
                float(os.environ.get('OMNIRETARGETING_FAST_HINGE_MARGIN', '0')))
            guard = next(g for kind, g, _ in self.guards if kind == 'contact')
            result = []
            for identity in f.locations:
                _, group, frame, sample, _ = identity
                bound = guard.groups[group][7][frame, sample]
                result.append(min(max(hinge_margin, .00025), .5 * bound))
            return np.asarray(result) + 1e-7
        if label[0] == 'ReferenceShape':
            values = self.inner[label]
            indices = np.asarray(f.locations).T
            return values[tuple(indices)] + 1e-7
        if 'Continuity' in label[0]:
            key = label[:3] + (('norm',) if label[-1] == 'norm' else ())
            values = self.inner[key]
            indices = np.asarray(f.locations).T
            return values[tuple(indices)] + 1e-7  # Original half-tolerance reserve.
        return np.full(len(f.slack), 1e-8)

    def evaluate(self, poses, jacobian=False):
        result = super().evaluate(poses, jacobian)
        # Actual root projection balls (rather than the extra proposal-only
        # ReferenceShape 3D balls). Zero domains are held as equalities.
        window, owner = self.window, self.window.global_objective
        coords, q = window.coords, window.coords.root_q
        if q is None:
            return result
        select = np.zeros((3, window.width))
        for k in range(3):
            col = coords.inverse.get(coords.root_d + k)
            if col is not None:
                select[k, col] = 1.
        matrix = _block_rows(np.broadcast_to(select, (window.frames, 3, window.width)), window.width) if jacobian else None
        budget = getattr(owner, 'root_budget', None)
        if budget is not None and budget > 0:
            value = poses[:, q:q + 3] - owner.reference[window.span, q:q + 3]
            result.append(ball(value, np.full(window.frames, budget), matrix,
                ('RootDomain', 'reference'), _locations(window.variable_frames.start, (window.frames,))))
        if owner.source_shape.xy > 0:
            value = poses[:, q:q + 2] - owner.source_shape.seed[window.span, q:q + 2]
            xy = None if matrix is None else matrix[(np.arange(window.frames)[:, None] * 3 + [0, 1]).ravel()]
            result.append(ball(value, np.full(window.frames, owner.source_shape.xy), xy,
                ('RootDomain', 'xy'), _locations(window.variable_frames.start, (window.frames,))))
        return result

    def inequalities(self, families, direction, *, inner=True):
        """A d <= h: scalar faces and supporting halfspaces for vector balls.

        Norm cuts use the predicted vector, so a zero residual does not lose
        its cone. All cuts still span no more than four consecutive frames.
        """
        matrices, bounds, keys = [], [], []
        for number, f in enumerate(families):
            margin = self.margins(f)
            if not inner:
                gap = f.slack
                if f.ball:
                    radius = np.sqrt(np.maximum(f.slack+np.square(f.residual).sum(axis=1),0.))
                    gap = radius-np.linalg.norm(f.residual,axis=1)
                # B solves the original feasible domain, with only a tiny
                # inward roundoff reserve. A's hinge margins are numerical
                # springs and need not be reached by an already feasible QP.
                reserve=1e-8 if f.label[0]=='RootDomain' else 1e-7
                margin = np.minimum(reserve,.5*np.maximum(gap,0.))
            if not f.ball:
                matrices.append(-f.matrix)
                bounds.append(f.slack - margin)
            else:
                dim, count = f.residual.shape[1], len(f.slack)
                predicted = f.residual + (f.matrix @ direction).reshape(f.residual.shape)
                length = np.linalg.norm(predicted, axis=1)
                normals = predicted / np.maximum(length[:, None], 1e-30)
                radius = np.sqrt(np.maximum(f.slack + np.einsum('ij,ij->i', f.residual, f.residual), 0.))
                operator = sparse.csr_matrix((normals.ravel(),
                    (np.repeat(np.arange(count), dim), np.arange(count * dim))), shape=(count, count * dim))
                matrices.append(operator @ f.matrix)
                bounds.append(radius - margin - np.einsum('ij,ij->i', normals, f.residual))
            keys.extend((number, i) for i in range(len(f.slack)))
        return sparse.vstack(matrices, format='csr'), np.concatenate(bounds), keys
