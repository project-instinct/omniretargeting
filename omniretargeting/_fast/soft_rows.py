"""One-sided frozen continuity penalties for whole-trajectory banded LM.

Contact/layout hinges remain in the parallel frame objective. Root-anchor
signals reuse that evaluation's FK/Jacobians; no second trajectory FK runs.
Weights/margins are frozen at entry, and never change physical guard budgets.
"""

from collections import OrderedDict
import copy

import numpy as np
from scipy import sparse

from .barrier_assembly import scatter
from .constrained_rows import ConstrainedRows
from .legacy_window_qp import WindowObjective
from .rows import _block_rows

STIFFNESS = 100.


class SoftTemporalRows:
    def __init__(self, objective, poses, stiffness=STIFFNESS):
        context = ConstrainedRows(WindowObjective(objective, poses, range(len(poses))))
        self.inner = context.inner
        self.guards = [item for item in context.guards if item[0] in ('sequence', 'correction')]
        self.objective, self.weights, self.operators = objective, {}, {}
        self.stiffness = stiffness
        self.sequence = {}
        for kind, guard, _ in self.guards:
            if kind == 'sequence':
                local = copy.copy(guard)
                local.tasks = copy.copy(guard.tasks)
                local.tasks.root_body = None
                local._signals_cache = OrderedDict()
                self.sequence[id(guard)] = local

    def difference(self, count, order, dim):
        key = count, order, dim
        if key not in self.operators:
            coeff = {1: [-1., 1.], 2: [1., -2., 1.], 3: [-1., 3., -3., 1.]}[order]
            difference = sparse.diags(coeff, np.arange(order + 1),
                shape=(count - order, count), format='csr')
            self.operators[key] = sparse.kron(difference, sparse.eye(dim), format='csr')
        return self.operators[key]

    def blocks(self, poses, signals, jacobian):
        from .continuity import derivatives, correction_derivatives, ROUND_OFF
        from .._sequence.continuity_guard import ACCEPTANCE_SLACK
        for kind, guard, _ in self.guards:
            if kind == 'sequence':
                values = self.sequence[id(guard)].signals(poses)
                if guard.tasks.root_body is not None:
                    anchor = signals['root_point'] - guard.tasks.roots
                    if guard.correction:
                        anchor = anchor - guard.reference['root_anchor_offset'][0]
                    values['root_anchor_offset'] = anchor, signals['root_jacobian']
                for (name, order), bound in guard.bounds.items():
                    value, jac = values[name]
                    residual = np.diff(value, n=order, axis=0)
                    matrix = None
                    if jacobian:
                        if name == 'joints':
                            dim = len(guard.tasks.coords.qi)
                            matrix = self.difference(len(poses), order, 1)
                            matrix = sparse.kron(matrix, guard.tasks.coords.selector, format='csr')
                        else:
                            matrix = self.difference(len(poses), order, value.shape[1]) @ _block_rows(jac, self.objective.width)
                    label = type(guard).__name__, name, order
                    yield label, residual, matrix, bound + ACCEPTANCE_SLACK - self.inner[label] - 1e-7, name != 'joints'
            else:
                if jacobian:
                    values = guard.linearizations(poses)
                else:
                    raw = (correction_derivatives(guard.coords, guard.fixed, poses) if guard.correction
                           else derivatives(guard.coords, poses))
                    values = {key: (value, None) for key, value in raw.items()}
                for key, bound in guard.bounds.items():
                    residual, matrix = values[key]
                    name, order = key.rsplit('_order_', 1)
                    label = type(guard).__name__, name, int(order)
                    yield label, residual, matrix, bound + ROUND_OFF - self.inner[label] - 1e-7, False
                    if key in guard.norm_bounds:
                        norm_label = *label, 'norm'
                        yield norm_label, residual, matrix, guard.norm_bounds[key] + ROUND_OFF - self.inner[norm_label] - 1e-7, True

    def accumulate(self, poses, signals, diagonal, system=None, *, blocks=None, direction=None):
        penalty, gradient, active_count, maxima = 0., np.zeros(len(poses) * self.objective.width), 0, {}
        source = self.blocks(poses, signals, system is not None) if blocks is None else blocks
        for label, value, matrix, allowed, norm in source:
            if direction is not None:
                value = value + (matrix @ direction).reshape(value.shape)
            size = np.linalg.norm(value, axis=1) if norm else np.abs(value).ravel()
            residual = np.maximum(size - np.asarray(allowed).ravel(), 0.)
            active = residual > 0.
            if label not in self.weights:
                if matrix is None:
                    raise RuntimeError('Initialize soft row weights with an entry linearization')
                metric = np.asarray(matrix.power(2) @ (1. / diagonal)).ravel()
                if norm:
                    metric = metric.reshape(value.shape).mean(axis=1)
                self.weights[label] = self.stiffness / np.maximum(metric, 1e-30)
            weights = self.weights[label]
            penalty += float(weights @ np.square(residual))
            active_count += int(active.sum())
            maxima[str(label)] = float(residual.max(initial=0.))
            if system is None or not active.any():
                continue
            if norm:
                count, dim = value.shape
                normals = value / np.maximum(size[:, None], 1e-30)
                operator = sparse.csr_matrix((normals.ravel(),
                    (np.repeat(np.arange(count), dim), np.arange(count * dim))), shape=(count, count * dim))
                derivative = (operator @ matrix)[active]
            else:
                derivative = matrix[active].multiply(np.sign(value.ravel()[active])[:, None]).tocsr()
            intercept = residual[active]
            if direction is not None:
                intercept = intercept - derivative @ direction
            gradient += derivative.T @ (weights[active] * intercept)
            scatter(system, derivative, weights[active])
        return penalty, gradient, dict(soft_continuity_active_rows=active_count,
                                       soft_continuity_excess=maxima)
