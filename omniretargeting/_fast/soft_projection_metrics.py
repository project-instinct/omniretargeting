"""Original physical-cost measurements and natural projection metric."""
from contextlib import contextmanager
import time

import numpy as np
from scipy import sparse


class Measurements:
    def __init__(self, objective, poses):
        self.objective, self.seconds, self.history = objective, 0., []
        value = self.evaluate(poses, True)
        diagonal = np.diagonal(value[0].hessian, axis1=1, axis2=2).copy()
        if value[2] is not None:
            diagonal += np.asarray(value[2].power(2).sum(axis=0)).reshape(diagonal.shape)
        self.metric = np.maximum(diagonal, 1.)
        self.metric /= np.median(self.metric)
        from .barrier_assembly import physical_system
        band = physical_system(value[0], value[2])
        count = band.shape[1]
        lower = sparse.diags([band[k, :count-k] for k in range(band.shape[0])],
            -np.arange(band.shape[0]), shape=(count,count), format='csr')
        self.system = lower + lower.T - sparse.diags(band[0], format='csr')
        self.system += sparse.diags(1e-6*np.maximum(self.system.diagonal(),1.), format='csr')
        self.system.eliminate_zeros()
        self.last = value[0].cost
        self.history.append(dict(stage='bulk', cost=self.last, stats=value[3]))

    def evaluate(self, poses, jacobian=False):
        started = time.monotonic()
        value = self.objective.evaluate(poses, jacobian)
        self.seconds += time.monotonic() - started
        return value

    def record(self, stage, families, poses):
        value = self.evaluate(poses)
        self.history.append(dict(stage=stage, guard_families=families,
            before_cost=self.last, cost=value[0].cost, stats=value[3]))
        self.last = value[0].cost


@contextmanager
def measurements(objective, poses):
    backend = getattr(objective, 'evaluation_backend', 'legacy')
    running = getattr(objective, 'pool', None) is not None
    objective.stop_workers()
    try:
        objective.evaluation_backend = 'cpu'
        objective.start_workers()
        yield Measurements(objective, poses)
    finally:
        objective.stop_workers()
        objective.evaluation_backend = backend
        if running:
            objective.start_workers()
