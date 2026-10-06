"""Opt-in joint acquisition: original witnesses and source-support split bounds."""
from collections import Counter
import time

import cvxpy as cp
import numpy as np
from scipy import sparse

from .joint_geometry import JointGeometry
from .support_split import SupportSplit
from .joint_model import JointModel, pair_indices, maxima, worst_rows


class JointWitness:
    def __init__(self, objective, reference, objects, scene):
        self.objective = objective
        self.geometry = JointGeometry(objective, objects, scene, reference)
        self.support = SupportSplit(objective, reference, self.geometry)
        # Reuse the existing acquisition clearance priority. Neither profile
        # weights nor any acceptance threshold is changed by this experiment.
        self.sigma = objective.cfg['collision_sigma_m'] * objective.clearance_scale
        self.queries, self.iterations = [], []
        self.rejections = Counter()
        self.geometry.start_workers(objective.frames)
        try:
            self.origin = self.measure(reference)
        except BaseException:
            self.geometry.stop_workers()
            raise
        self.fixed_failures = set(self.origin['failures'])
        self.matrix = None
        self.corrections = 0

    def measure(self, poses, jacobian=False):
        tick = time.monotonic()
        measured = self.geometry.measure(poses, frames=range(len(poses)), jacobians=jacobian)
        # Self is already exact in FastObjective. Do not charge it twice.
        measured['rows'] = [(key, row) for key, row in measured['rows'] if key[1] != 'self']
        measured['depths'] = {key: d for key, d in measured['depths'].items() if key[1] != 'self'}
        self.queries.append(dict(frames=measured['query_scope']['queried_frame_count'],
                                 witnesses=len(measured['rows']), seconds=time.monotonic() - tick,
                                 unknown=len(measured['failures'])))
        return measured

    def rows(self, measured):
        keys, residual, jac, frames = [], [], [], []
        for key, row in measured['rows']:
            allowance = self.support.allowance(key, row)
            keys.append(key)
            residual.append(-row['clearance'] - allowance)
            frames.append(key[0])
            derivative = np.asarray(row.get('jacobian', []))
            # Non-Jacobian moving queries explicitly return an empty vector;
            # cached/static rows may retain a full derivative in the same call.
            if not derivative.size:
                derivative = np.zeros(self.objective.width)
            if derivative.shape != (self.objective.width,):
                raise ValueError(f'Unexpected witness Jacobian shape: {derivative.shape}')
            jac.append(-derivative)
        return keys, np.asarray(residual), np.array(jac).reshape(-1, self.objective.width), np.asarray(frames, int)

    def begin(self, poses, iteration):
        self.query_start = len(self.queries)
        self.origin_poses = poses.copy()
        self.origin = self.measure(poses, True)
        self.keys, self.raw, self.jac, self.frames = self.rows(self.origin)
        self.pairs = pair_indices(self.keys)
        width, size = self.objective.width, len(poses) * self.objective.width
        columns = self.frames[:, None] * width + np.arange(width)
        self.matrix = sparse.csr_matrix((self.jac.ravel(), (np.repeat(np.arange(len(self.raw)), width), columns.ravel())),
                                        shape=(len(self.raw), size))
        self.matrix.eliminate_zeros()
        self.support.begin(poses)
        self.corrections = 0
        self.iteration = iteration

    def finish_iteration(self, accepted):
        calls = self.queries[self.query_start:]
        self.iterations.append(dict(iteration=self.iteration, accepted=accepted,
                                    original_query_frames=sum(q['frames'] for q in calls),
                                    witness_count=len(self.raw), query_calls=len(calls),
                                    original_query_seconds=sum(q['seconds'] for q in calls)))

    def add_cost(self, poses, accumulator, stats):
        measured = self.measure(poses, accumulator.jacobian)
        keys, raw, jac, frames = self.rows(measured)
        pairs = pair_indices(keys)
        if accumulator.jacobian:
            width = self.objective.width
            columns = frames[:, None] * width + np.arange(width)
            matrix = sparse.csr_matrix((jac.ravel(), (np.repeat(np.arange(len(raw)), width), columns.ravel())),
                                       shape=(len(raw), len(poses) * width))
            accumulator.joint_model = JointModel(raw, matrix, pairs, self.sigma)
        selected = worst_rows(raw, pairs)
        raw, jac, frames = raw[selected], jac[selected], frames[selected]
        residual = np.maximum(raw, 0.) / self.sigma
        jac *= ((raw > 0) / self.sigma)[:, None]
        for f in np.unique(frames):
            select = frames == f
            accumulator.add_frame_rows(f, residual[select], jac[select] if accumulator.jacobian else None)
        stats['joint_witness_excess_m'] = float(np.maximum(raw, 0.).max(initial=0.))
        stats['joint_witness_unknown_frames'] = len({failure.split('frame ')[-1].split()[0] for failure in measured['failures']})

    def linear_feasible(self, poses):
        if self.matrix is None:
            return True
        delta = self.objective.coords.difference(self.origin_poses, poses).ravel()
        # A banded step must include every activated witness. Existing
        # infeasibility can shrink; slack is never a historical-depth ratchet.
        slack = maxima(self.raw, self.pairs)[self.pairs]
        good = bool(np.all(self.raw + self.matrix @ delta <= slack + 1e-7))
        if not good:
            self.rejections['banded_witness_rows'] += 1
        return good

    def proposal(self, flat, matrix, gradient, scale):
        # Replace the hinge's local GN quadratic by its actual elastic model,
        # rather than counting the same penetration twice in the conic cost.
        worst = worst_rows(self.raw, self.pairs)
        active = worst[self.raw[worst] > 0]
        local = self.matrix[active] / self.sigma
        residual = self.raw[active] / self.sigma
        matrix = matrix - local.T @ local
        gradient = gradient - local.T @ residual
        elastic = cp.Variable(int(self.pairs.max()) + 1, nonneg=True) if len(self.raw) else None
        constraints = [] if elastic is None else [self.raw + self.matrix @ flat <= elastic[self.pairs]]
        objective = cp.quad_form(flat, cp.psd_wrap(matrix)) + 2 * gradient @ flat
        if elastic is not None:
            objective += cp.sum_squares(elastic / self.sigma)
        return objective / scale, constraints

    def accepts(self, poses):
        good = self.support.accepts(poses)
        if not good:
            self.rejections['support_split_curvature_or_bound'] += 1
        return good

    def constraints(self, poses, flat):
        return self.support.constraints(poses, flat)

    def correct(self, poses, step):
        """At most one second-order correction per LM iteration, all rows kept."""
        if self.corrections:
            return None
        self.corrections += 1
        self.rejections['second_order_correction'] += 1
        from .joint_proposal import correct
        return correct(self, poses, step)

    def diagnostics(self, poses):
        measured = self.measure(poses)
        summary = self.geometry.summary(measured)
        _, raw, _, _ = self.rows(measured)
        for kind in ('static', 'moving'):
            selected = [i for i, (key, _) in enumerate(measured['rows']) if key[1] == kind]
            worst = max(selected, key=lambda i: raw[i], default=None)
            summary[kind]['max_excess_m'] = 0. if worst is None else max(float(raw[worst]), 0.)
            if worst is not None:
                summary[kind]['worst_excess_key'] = measured['rows'][worst][0]
        return dict(iterations=self.iterations, queries=self.queries, before_free_repair=summary,
                    support=self.support.diagnostics(poses), rejected_reasons=dict(self.rejections),
                    physical_validation='unverified', unknown_is_pass=False)

    def close(self):
        self.geometry.stop_workers()
