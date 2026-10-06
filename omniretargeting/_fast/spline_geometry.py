"""Original collider witnesses in the all-frame correction problem.

The existing geometry service owns filtering, tolerances, uncertainty and
caches. SDFs remain objective proxies, never acceptance certificates.
"""
import time

import numpy as np
from scipy import sparse

from .mesh_repair import COLLISION_SIGMA, _regressed, REGRESSION, PENETRATING_REGRESSION_M
from .priority_geometry import PriorityGeometry
from .qp_guard_rows import intervals


class SplineGeometry:
    def __init__(self, objective, objects, scene, resources):
        self.provider = PriorityGeometry(objective, objects, scene)
        resources.callback(self.provider.close)
        self.provider.freeze_reference(objective.reference)
        self.provider.start()
        tick = time.monotonic()
        self.current = self.provider.measure(objective.reference)
        self.query_seconds = time.monotonic() - tick
        self.reference_summary = self.provider.summary(self.current)
        self.best = dict(self.current['depths'])
        self.rejections = 0
        self.unknown_trials = 0
        self.history = []
        self.witness_memory = {}
        self.witness_failures = []
        self.witness_batches = self.witness_queries = 0

    def learn(self, origin, candidate, measured, keys):
        """Requery new trial material witnesses at the current accepted pose."""
        tick = time.monotonic()
        rows = self.provider.relinearize_witnesses(origin, candidate, measured,
            frames=range(len(origin)), keys=keys)
        self.query_seconds += time.monotonic() - tick
        self.witness_batches += 1
        self.witness_queries += len(rows['rows'])
        self.witness_failures.extend(rows.get('failures', ()))
        added = 0
        saved_origin = np.array(origin, copy=True)
        saved_origin.setflags(write=False)
        from .original_geometry_relinearize import witness_identity
        for key, row in rows['rows']:
            if not np.isfinite(row['clearance']) or not np.isfinite(row['jacobian']).all():
                continue
            identity = repr(witness_identity(key, row))
            added += identity not in self.witness_memory
            self.witness_memory[identity] = (saved_origin, key, dict(row))
        return added

    def _remembered_rows(self, poses, measured=None):
        from .original_geometry_relinearize import witness_identity
        measured = self.current if measured is None else measured
        known = {repr(witness_identity(key, row)) for key, row in measured['rows']}
        result = []
        groups, equal = {}, {}
        for identity, (origin, key, row) in list(self.witness_memory.items()):
            if identity in known:
                continue
            token = id(origin)
            if token not in equal:
                equal[token] = np.array_equal(origin, poses)
            if equal[token]:
                result.append((key, row))
            else:
                groups.setdefault(token, (origin, []))[1].append((identity, key, row))
        # A material witness registry permits a batched original query. Keep
        # each old origin's provenance, instead of rehashing/querying the whole
        # clip separately for every remembered sample.
        saved_origin = None
        for origin, items in groups.values():
            items.sort(key=lambda item: item[1][0])
            value = self.provider.relinearize_witnesses(poses, origin,
                {'rows': [(key, row) for _, key, row in items], 'failures': []},
                frames=range(len(poses)), keys={key for _, key, _ in items})
            self.witness_batches += 1
            self.witness_queries += len(value['rows'])
            self.witness_failures.extend(value.get('failures', ()))
            if (not value.get('request_complete', False)
                    or len(value.get('witness_ids', ())) != len(items)
                    or len(value['rows']) != len(items)):
                self.witness_failures.append('unknown incomplete remembered-witness batch')
            for identity, (key, row) in zip(value.get('witness_ids', ()), value['rows']):
                if not np.isfinite(row['clearance']) or not np.isfinite(row['jacobian']).all():
                    continue
                if saved_origin is None:
                    saved_origin = np.array(poses, copy=True)
                    saved_origin.setflags(write=False)
                self.witness_memory[repr(identity)] = (saved_origin, key, dict(row))
                result.append((key, row))
        return result

    def measure(self, poses, jacobians=False):
        tick = time.monotonic()
        value = self.provider.measure(poses, jacobians=jacobians)
        failures_before = len(self.witness_failures)
        remembered = self._remembered_rows(poses, value)
        failures = value['failures'] + self.witness_failures[failures_before:]
        if remembered or len(failures) != len(value['failures']):
            depths = dict(value['depths'])
            for key, row in remembered:
                depths[key] = max(depths.get(key, 0.), -row['clearance'], 0.)
            value = dict(value, rows=value['rows'] + remembered, depths=depths,
                         failures=failures, complete=value.get('complete', True) and not failures)
        self.query_seconds += time.monotonic() - tick
        return value

    def blocks(self, poses):
        self.current = self.measure(poses, True)
        rows = self.current['rows']
        self.linearized = dict(self.current, rows=rows)
        matrix, phi = self.provider.linear_rows(self.linearized)
        allowance = np.array([self.best.get(key, 0.) + REGRESSION
                              + (PENETRATING_REGRESSION_M if self.best.get(key, 0.) > 0 else 0.)
                              for key, _ in rows])
        return [intervals(matrix, -allowance - phi, np.full(len(phi), np.inf))]

    @staticmethod
    def _deepest(measured):
        selected = {}
        for index, (key, row) in enumerate(measured['rows']):
            if row['clearance'] < 0 and (key not in selected or
                    row['clearance'] < measured['rows'][selected[key]][1]['clearance']):
                selected[key] = index
        return np.array(list(selected.values()), dtype=int)

    def residual_rows(self, poses):
        measured = self.measure(poses, True)
        matrix, phi = self.provider.linear_rows(measured)
        selected = self._deepest(measured)
        return matrix[selected] / COLLISION_SIGMA, phi[selected] / COLLISION_SIGMA

    def cost(self, measured):
        return float(sum((value / COLLISION_SIGMA)**2 for value in measured['depths'].values()))

    def accepts(self, poses):
        measured = self.measure(poses)
        from .spline_collision_priority import measurement_collision_layer
        if not measurement_collision_layer(measured, frame_count=len(poses))['valid']:
            self.unknown_trials += 1
        bad = [key for key, value in measured['depths'].items()
               if _regressed(value, self.best.get(key, 0.))]
        self.rejections += bool(bad)
        return not bad, measured, bad

    def commit(self, measured):
        # Unknown/missing keys must never acquire a zero-depth certificate.
        for key, value in measured['depths'].items():
            self.best[key] = min(self.best.get(key, value), value)
        self.current = measured
        self.history.append(self.provider.summary(measured))

    def focus(self, count):
        result = np.zeros(count, bool)
        for key, value in self.current['depths'].items():
            if value > .001:  # Existing repair activation, not an acceptance cap.
                result[key[0]] = True
        return result

    def report(self):
        return dict(fixed_reference=self.reference_summary,
                    final=self.provider.summary(self.current), query_seconds=self.query_seconds,
                    rejected_trials=self.rejections, unknown_trials=self.unknown_trials,
                    history=self.history, original_filter_and_tolerances_unchanged=True,
                    remembered_witnesses=len(self.witness_memory),
                    selected_witness_batches=self.witness_batches,
                    selected_witness_queries=self.witness_queries,
                    selected_witness_failures=self.witness_failures,
                    collision_free_certificate=False)
