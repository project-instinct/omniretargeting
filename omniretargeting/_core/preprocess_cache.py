"""Solve-local caches of immutable source graphs and support certificates."""

from copy import deepcopy

import numpy as np

from ..contact_mechanics import ContactCandidate


class SourceGraphCache:
    """Reuse graph construction; each task set still owns writable operators."""

    def __init__(self):
        self.human = self.environment = self.operators = None

    def get(self, human, environment, builder):
        def same(first, second):
            return (second is not None and first.shape == second.shape and first.dtype == second.dtype
                    and first.tobytes() == second.tobytes())

        if self.operators is None or not same(human, self.human) or not same(environment, self.environment):
            self.human = human.copy()
            self.environment = environment.copy()
            self.operators = [builder(np.vstack([human[f], environment[f]]), human.shape[1])
                              for f in range(len(human))]
        return [operator.copy() for operator in self.operators]


class CertificateMap:
    """Deduplicate only a nominated stateless certificate function, in order.

    Candidate snapshots must be frozen, including their numerical arrays.
    Keys keep all scalar request bits and contact order. Cache entries hold
    candidates alive so object identities cannot be recycled during a solve.
    Stateful nominal explanations always use the original ordered map.
    """

    def __init__(self, mapper, certifier):
        self.mapper, self.certifier = mapper, certifier
        self.cache = {}
        self.requests = self.hits = 0

    @staticmethod
    def key(task):
        contacts, origin, wrench, uncertainty = task
        for contact in contacts:
            if type(contact) is not ContactCandidate:
                return None
            for name in ('observed_points', 'surface_points', 'normals', 'sliding_velocity'):
                value = getattr(contact, name)
                if value is not None and (not isinstance(value, np.ndarray) or value.flags.writeable):
                    return None
        arrays = [np.asarray(value, dtype=float) for value in (origin, wrench, uncertainty)]
        return (tuple(id(contact) for contact in contacts),
                tuple((array.shape, array.tobytes()) for array in arrays))

    def __call__(self, function, items):
        items = list(items)
        if function is not self.certifier:
            return self.mapper(function, items)
        keys, pending, missing = [], [], {}
        for index, task in enumerate(items):
            self.requests += 1
            key = self.key(task)
            if key is None:
                key = ('uncached', index)
            if key in self.cache or key in missing:
                self.hits += 1
            else:
                missing[key] = len(pending)
                pending.append(task)
            keys.append(key)
        results = list(self.mapper(function, pending)) if pending else []
        fresh = {key: results[index] for key, index in missing.items()}
        for key, index in missing.items():
            if key[0] != 'uncached':
                self.cache[key] = (tuple(pending[index][0]), fresh[key])
        return [deepcopy(self.cache[key][1] if key in self.cache else fresh[key]) for key in keys]
