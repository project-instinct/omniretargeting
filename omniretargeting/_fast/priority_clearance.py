"""Original-geometry priority and transactional contact-yield accounting.

Trial evaluation is pure with respect to accepted depths and yield credit.
Only ``commit`` advances that ledger. The solver still owns exact hard-guard
acceptance and the certified, ordered L1/LM merit comparison.
"""

import os

import numpy as np
from scipy import sparse

from .priority_contacts import PriorityContacts, _rank
from .priority_geometry import body_depths

CONTACT_SLACK = 1e-7
DEPTH_PROGRESS = 1e-7


class PriorityClearance:
    def __init__(self, objective, provider, reference):
        self.objective, self.provider = objective, provider
        self.owner_pid = os.getpid()
        self.reference = np.array(reference, copy=True)
        self.reference.setflags(write=False)
        freeze = getattr(provider, 'freeze_reference', None)
        if freeze is not None:
            freeze(self.reference)
        initial = provider.measure(reference)
        depth, _ = body_depths(objective.model, len(reference), initial)
        self.initial_depth = depth.copy()
        self.initial_depth.setflags(write=False)
        self.best_depth = depth.copy()
        self.current_depth = depth.copy()
        self.best_keys = dict(initial['depths'])
        self.current = np.array(reference, copy=True)
        self.initial_measurement = initial
        self.current_measurement = initial
        self.owns_clearance = False
        self.margin_preferences = {}
        self.clearance_coverage = None
        from .runtime_options import OPTIONS
        if OPTIONS.windowed_priority:
            from .priority_margins import preferences, coverage
            settings = preferences(objective)
            self.clearance_coverage = coverage(provider, objective, settings)
            self.margin_preferences = settings
            self.owns_clearance = True
        self.contacts = []
        self.spent = {}
        self.yields = []
        self.descent_certificates = []
        self.epoch = 0
        self.certificate = None
        self._trial = None
        self._prepared = False
        self._current_contacts = {}
        from .priority_stall import StalledSearch
        self._stalled_search = StalledSearch()

    def register(self, guard):
        self._stalled_search.invalidate()
        adapter = PriorityContacts(guard, f'{type(guard).__name__}:{len(self.contacts)}')
        self.contacts.append(adapter)
        self._current_contacts.update(self._contact_values(adapter.evaluate(self.current, jacobian=False)))
        self._window_state = None
        self._prepared = False
        self._trial = None
        return adapter

    def window_dependencies(self, frames):
        # Original geometry and every ledger donor/contact are same-frame.
        # Scheduler separately unions objective and hard-guard stencil halos.
        from .priority_window_contacts import checked_frames
        checked_frames(frames, len(self.current))
        return frames

    def fork_window(self, poses=None, *, frames=None):
        from .priority_window import fork_window
        if isinstance(poses, range) and frames is None:
            frames, poses = poses, None
        return fork_window(self, poses, frames)

    def advance_local(self, poses, *, frames=None):
        from .priority_window import commit_window, WindowState
        frames = self._fork_frames if frames is None else frames
        if self._funding_window is None:
            self._funding_window = WindowState(self._funding_baseline, frames)
        return commit_window(self, poses, frames, local=True)

    def export_transactions(self, *, frames=None):
        from .priority_window import export_transactions
        return export_transactions(self, self._fork_frames if frames is None else frames)

    def repeated_stall(self, model, policy):
        return self._stalled_search.repeated(self, model, policy)

    def record_stall(self, model, policy, record):
        self._stalled_search.record(self, model, policy, record)

    @staticmethod
    def _contact_values(blocks):
        return {identity: float(value) for block in blocks
                for identity, value in zip(block.identities, block.values)}

    def frozen_depth(self, poses):
        measured = self.provider.measure(poses)
        result = body_depths(self.objective.model, len(poses), measured)[0]
        result.setflags(write=False)
        return result

    def collision_diagnostics(self, poses, *, frames=None):
        from .priority_margins import collision_diagnostics
        measured = self.provider.measure(poses, **({} if frames is None else {'frames': frames}))
        return collision_diagnostics(self, measured)

    def begin_iteration(self, poses, model=None, *, frames=None):
        if frames is not None:
            from .priority_window import window_state, WindowState
            self.model = {} if model is None else model
            self.certificate = None
            if hasattr(self, '_funding_baseline') and self._funding_window is None:
                self._funding_window = WindowState(self._funding_baseline, frames)
            window_state(self, poses, frames, fresh=True)
            return
        if not np.array_equal(poses, self.current):
            raise ValueError('Priority iteration must start at the committed trajectory')
        self.model = {} if model is None else model
        self.certificate = None
        self._prepare()

    def _prepare(self):
        self.current_measurement = self.provider.measure(self.current, jacobians=True)
        _, self.owners = body_depths(self.objective.model, len(self.current), self.current_measurement)
        self.matrix, self.phi = self.provider.linear_rows(self.current_measurement)
        self.blocks = [b for adapter in self.contacts for b in adapter.evaluate(self.current)]
        self._prepared = True
        self._trial = None

    def _at(self, poses):
        if not np.array_equal(poses, self.current):
            raise ValueError('Priority rows must be linearized at the committed trajectory')
        if not self._prepared:
            self._prepare()

    def soft_limits(self, poses, *, frames=None):
        """Window-only necessary funding caps; legacy soft terms are unchanged."""
        if frames is None:
            return {}
        from .priority_window import window_state
        return dict(window_state(self, poses, frames).contact_domain.limits)

    def geometry_refinement(self, origin, candidate, *, frames, known_witness_ids, max_new_witnesses):
        from .priority_refinement import geometry_refinement
        return geometry_refinement(self, origin, candidate, frames=frames,
                                   known_witness_ids=known_witness_ids, max_new_witnesses=max_new_witnesses)

    def qp_bound_sources(self, poses, *, frames=None):
        """Frozen diagnostic labels; never remeasure a prepared window."""
        if frames is None:
            linear, cones = self.qp_bounds(poses)
            return ['priority_nonregression'] * len(linear), ['priority_nonregression'] * len(cones)
        from .priority_window import window_state
        state = window_state(self, poses, frames)
        linear = ['penetration_nonregression'] if len(state.measured['rows']) else []
        cones = []
        for block in state.blocks:
            label = 'contact_cap:' + '+'.join(sorted({str(key[-1]) for key in block.identities}))
            (linear if block.scalar else cones).append(label)
        return linear, cones

    def qp_values(self, poses, *, frames=None):
        """Contact-only hard SOC values; unknown geometry batches stay None."""
        if frames is None:
            linear, cones = self.qp_bounds(self.current)
            return [None] * len(linear), [None] * len(cones)
        from .priority_window import window_state
        return window_state(self, self.current, frames).qp_values(poses)

    def soft_rows(self, poses, *, frames=None):
        if frames is not None:
            from .priority_window import window_state
            return window_state(self, poses, frames).soft_rows()
        self._at(poses)
        indices, keys = [], []
        for index, (key, _) in enumerate(self.current_measurement['rows']):
            for body in self.owners[key]:
                indices.append(index)
                keys.append(('penetration', int(key[0]), body))
        result = []
        if indices:
            selected = np.asarray(indices)
            result.append((-self.matrix[selected], self.phi[selected].copy(), np.ones(len(selected)), tuple(keys)))
        for block in self.blocks:
            if block.scalar:
                result.append((block.matrix, block.bound - block.residual,
                               np.ones(len(block.bound)), block.keys))
        return result

    def soft_cones(self, poses, *, frames=None):
        if frames is not None:
            from .priority_window import window_state
            return window_state(self, poses, frames).soft_cones()
        self._at(poses)
        return [(b.matrix, b.residual.copy(), b.bound.copy(), np.ones(len(b.bound)), b.keys)
                for b in self.blocks if not b.scalar]

    def soft_values(self, poses, *, frames=None):
        if frames is not None:
            from .priority_window import window_state
            return window_state(self, self.current, frames).soft_values(poses)
        if not self._prepared:
            self._prepare()
        blocks = [b for adapter in self.contacts for b in adapter.evaluate(poses, jacobian=False)]
        if [b.identities for b in blocks] != [b.identities for b in self.blocks]:
            raise ValueError('SOC contact channels changed identity')
        return ([None] * bool(len(self.current_measurement['rows']))
                + [b.bound - b.residual for b in blocks if b.scalar],
                [(b.residual.copy(), b.bound.copy()) for b in blocks if not b.scalar])

    def constraints(self, poses, flat):
        """Non-regression rows; yielding contacts themselves are never hard."""
        linear, _ = self.repair_bounds(poses)
        return [matrix @ flat <= upper for matrix, _, upper in linear]

    def repair_bounds(self, poses):
        """Full tangent columns, identical reserve to ``constraints``."""
        self._at(poses)
        from .mesh_repair import REGRESSION, PENETRATING_REGRESSION_M
        protected = np.array([self.best_keys.get(key, 0.) for key, _ in self.current_measurement['rows']])
        tolerance = REGRESSION + (protected > 0) * PENETRATING_REGRESSION_M
        allowed = np.minimum(np.maximum(protected + tolerance / 2, -self.phi), protected + tolerance)
        linear = [] if not len(allowed) else [(-self.matrix, np.full(len(allowed), -np.inf), self.phi + allowed)]
        return linear, []

    def qp_bounds(self, poses, *, frames=None):
        if frames is not None:
            from .priority_window import window_state
            return window_state(self, poses, frames).qp_bounds()
        from .runtime_options import OPTIONS
        linear, cones = self.repair_bounds(poses)
        if getattr(OPTIONS, 'priority_zero_yield_descent', False):
            extra_linear, extra_cones = self.zero_yield_bounds(poses)
            linear += extra_linear
            cones += extra_cones
        return linear, cones

    def zero_yield_bounds(self, poses):
        """Each original channel keeps its committed metric, with half reserve."""
        self._at(poses)
        linear, cones = [], []
        for block in self.blocks:
            bound = np.array([self._current_contacts[key] for key in block.identities]) + CONTACT_SLACK / 2
            if block.scalar:
                linear.append((block.matrix, np.full(len(bound), -np.inf), bound - block.residual))
            else:
                cones.append((block.matrix, block.residual.copy(), bound))
        return linear, cones

    def certify_zero_yield(self, poses, predicted_slacks=None):
        from .priority_descent import certify
        return certify(self, poses, predicted_slacks)

    def _prediction(self, step):
        predicted = np.zeros_like(self.current_depth)
        columns = {}
        moved = np.abs(step) > 1e-12
        values = np.maximum(-self.phi - self.matrix @ step, 0.)
        for index, (key, _) in enumerate(self.current_measurement['rows']):
            row = self.matrix.getrow(index)
            used = set(int(c) for c in row.indices[(np.abs(row.data) > 1e-12) & moved[row.indices]])
            for body in self.owners[key]:
                predicted[key[0], body] = max(predicted[key[0], body], values[index])
                columns.setdefault((key[0], body), set()).update(used)
        return predicted, columns

    @staticmethod
    def _levels(blocks):
        groups = {}
        for block in blocks:
            for key, value in zip(block.keys, np.maximum(block.values - block.bound, 0.)):
                groups[key] = max(groups.get(key, 0.), float(value))
        return [sum(value for key, value in groups.items() if _rank(key[1]) == rank) for rank in range(3)]

    @staticmethod
    def _merit(report, slacks=None):
        report['valid'] = report['accepted']
        report['tolerances'] = (DEPTH_PROGRESS, CONTACT_SLACK, CONTACT_SLACK, CONTACT_SLACK)
        report['slacks'] = {} if slacks is None else slacks
        report['details'] = {k: v for k, v in report.items() if k not in ('valid', 'levels', 'tolerances', 'slacks')}
        return report

    def _evaluate_trial(self, poses):
        poses = np.asarray(poses)
        if self._trial is not None and np.array_equal(poses, self._trial[0]):
            return self._trial[1]
        if not self._prepared:
            self._prepare()
        report = dict(accepted=False, epoch=self.epoch, reliable=False, levels=None,
                      violating_frames=[], yields=[], requires_priority_certificate=False)
        if poses.shape != self.current.shape or not np.isfinite(poses).all():
            return self._merit(dict(report, reason='nonfinite_or_wrong_shape', violating_frames=list(range(len(self.current)))))
        measured = self.provider.measure(poses)
        try:
            depth, _ = body_depths(self.objective.model, len(poses), measured)
        except ValueError as error:
            return self._merit(dict(report, reason=str(error), violating_frames=list(range(len(poses)))))
        report['reliable'] = True
        from .mesh_repair import _regressed, REGRESSION, PENETRATING_REGRESSION_M
        regressions = [(key, float(value), self.best_keys.get(key, 0.))
                       for key, value in measured['depths'].items()
                       if _regressed(value, self.best_keys.get(key, 0.))]
        body_bad = [(int(f), int(b)) for f, b in zip(*np.nonzero(
            depth > self.best_depth + REGRESSION + (self.best_depth > 0) * PENETRATING_REGRESSION_M))]
        step = self.objective.coords.difference(self.current, poses).ravel()
        predicted, donor_columns = self._prediction(step)
        # A tolerated regression, followed by recovery, earns nothing twice.
        gains = np.maximum(np.minimum(self.best_depth - depth, self.current_depth - predicted), 0.)
        candidate_blocks = [b for adapter in self.contacts for b in adapter.evaluate(poses, jacobian=False)]
        current_values = self._current_contacts
        records, unfunded = [], []
        for block, linear in zip(candidate_blocks, self.blocks):
            if block.identities != linear.identities:
                raise ValueError('Contact evidence changed during priority acquisition')
            for index, identity in enumerate(block.identities):
                value = float(block.values[index])
                old = current_values[identity]
                limit = float(block.bound[index] + block.own[index])
                frame, body = int(block.frames[index]), int(block.bodies[index])
                own_used = max(0., min(value, limit) - max(old, float(block.bound[index])))
                if own_used > CONTACT_SLACK:
                    records.append(dict(contact=identity, frame=frame, body=body, donor_body=body,
                                        gap_m=own_used, contact_error_m=value, rank=int(block.ranks[index]),
                                        authorization='frozen_own_body',
                                        measured_reduction_m=float(self.current_depth[frame, body] - depth[frame, body]),
                                        cumulative_reduction_m=float(self.initial_depth[frame, body] - depth[frame, body]),
                                        predicted_reduction_m=float(self.current_depth[frame, body] - predicted[frame, body]),
                                        available_m=float(block.own[index])))
                # Credit is only needed beyond the immutable nominal + own
                # allowance. Re-entering an old yielded gap still requires a
                # same-step donor and consumes cumulative credit again.
                needed = max(0., value - max(old, limit))
                if needed <= CONTACT_SLACK:
                    continue
                start = index if block.scalar else 3 * index
                stop = start + (1 if block.scalar else 3)
                jac = linear.matrix[start:stop]
                contact_columns = set(int(c) for c in jac.indices[
                    (np.abs(jac.data) > 1e-12) & (np.abs(step[jac.indices]) > 1e-12)])
                donors = []
                for donor in np.flatnonzero(gains[frame] > DEPTH_PROGRESS):
                    donor = int(donor)
                    if donor == body:
                        continue  # Own depth was already granted in the frozen allowance.
                    if not contact_columns.intersection(donor_columns.get((frame, donor), ())):
                        continue
                    spent = self.spent.get((identity, donor), 0.)
                    total = max(0., self.initial_depth[frame, donor] - min(self.best_depth[frame, donor], depth[frame, donor]))
                    budget = min(float(gains[frame, donor]), max(0., total - spent))
                    if budget > DEPTH_PROGRESS:
                        donors.append((budget, donor))
                # A single explicit donor suffices; never add unrelated bodies'
                # depths together to invent an allowance for one contact.
                donors.sort(reverse=True)
                if not donors or donors[0][0] + CONTACT_SLACK < needed:
                    unfunded.append(dict(contact=identity, frame=frame, body=body, gap_m=needed,
                                         available_m=0. if not donors else donors[0][0]))
                    continue
                budget, donor = donors[0]
                records.append(dict(contact=identity, frame=frame, body=body, donor_body=donor,
                                    gap_m=needed, contact_error_m=value, rank=int(block.ranks[index]),
                                    authorization='same_step_cross_body',
                                    measured_reduction_m=float(self.current_depth[frame, donor] - depth[frame, donor]),
                                    predicted_reduction_m=float(self.current_depth[frame, donor] - predicted[frame, donor]),
                                    available_m=budget))
        frames = sorted({k[0] for k, _, _ in regressions} | {f for f, _ in body_bad}
                        | {r['frame'] for r in unfunded})
        levels = [float(depth.sum()), *self._levels(candidate_blocks)]
        before = [float(self.current_depth.sum()), *self._levels(self.blocks)]
        predicted_blocks = []
        from dataclasses import replace
        for b in self.blocks:
            change = b.matrix @ step
            residual = b.residual + (change if b.scalar else change.reshape(b.residual.shape))
            predicted_blocks.append(replace(b, residual=residual))
        predicted_levels = [float(predicted.sum()), *self._levels(predicted_blocks)]
        report.update(accepted=not frames, levels=levels, before_levels=before,
                      predicted_levels=predicted_levels,
                      actual_reductions=(np.asarray(before) - levels).tolist(),
                      predicted_reductions=(np.asarray(before) - predicted_levels).tolist(),
                      strict_penetration_decrease=bool(np.any(gains > DEPTH_PROGRESS)),
                      violating_frames=frames, penetration_regressions=regressions,
                      body_regressions=body_bad, unfunded_contacts=unfunded, yields=records,
                      requires_priority_certificate=bool(records))
        if frames:
            report['reason'] = 'penetration_regression' if regressions or body_bad else 'unfunded_contact_yield'
        slacks = {('penetration', int(f), int(b)): float(depth[f, b])
                  for f, b in zip(*np.nonzero(depth))}
        for block in candidate_blocks:
            for key, value in zip(block.keys, np.maximum(block.values - block.bound, 0.)):
                slacks[key] = max(slacks.get(key, 0.), float(value))
        report = self._merit(report, slacks)
        self._trial = (poses.copy(), report, measured, depth, candidate_blocks)
        return report

    def evaluate_trial(self, poses, *, frames=None):
        if frames is not None:
            from .priority_window import window_state
            report = window_state(self, self.current, frames).evaluate(poses)
            return dict(valid=report["valid"], levels=tuple(report["levels"] or (0., 0., 0., 0.)),
                        tolerances=report["tolerances"], slacks=report["slacks"], details=report["details"])
        """The exact five-field R2 PriorityMerit protocol."""
        report = self._evaluate_trial(poses)
        return dict(valid=report['valid'], levels=tuple(report['levels'] or (0., 0., 0., 0.)),
                    tolerances=report['tolerances'], slacks=report['slacks'], details=report['details'])

    def accepts(self, poses, *, frames=None):
        if frames is not None:
            from .priority_window import window_state
            report = window_state(self, self.current, frames).evaluate(poses)
            return report["accepted"]
        return self._evaluate_trial(poses)['accepted']

    def violation_frames(self, poses, *, frames=None):
        if frames is not None:
            from .priority_window import window_state
            report = window_state(self, self.current, frames).evaluate(poses)
            result = np.zeros(len(self.current), bool)
            result[report["violating_frames"]] = True
            return result
        result = np.zeros(len(self.current), bool)
        result[self._evaluate_trial(poses)['violating_frames']] = True
        return result

    def commit(self, poses, *, frames=None, transactions=None):
        if frames is not None:
            from .priority_window import commit_window
            return commit_window(self, poses, frames, transactions=transactions)
        if os.getpid() != self.owner_pid:
            raise RuntimeError('A speculative worker cannot commit priority credit')
        from .priority_descent import KIND, pose_digest
        certificate = self.certificate or {}
        kind = certificate.get('kind')
        if kind not in (None, KIND):
            raise ValueError('Unknown priority certificate kind')
        descent = None
        if kind == KIND:
            if (certificate.get('epoch') != self.epoch
                    or certificate.get('pose_sha256') != pose_digest(poses)
                    or certificate.get('version') != 1
                    or certificate.get('descent_certified') is not True
                    or certificate.get('optimality_certified') is not False
                    or certificate.get('zero_additional_yield') is not True):
                raise ValueError('Stale or mismatched descent certificate')
            # Recheck against CURRENT geometry, contacts, hard guards and the
            # prepared model. Cached trial bookkeeping is not commit authority.
            self._trial = None
            descent = self.certify_zero_yield(poses)
            if descent is None:
                raise ValueError('Zero-yield descent certificate failed commit recheck')
        report = self._evaluate_trial(poses)
        if not report['accepted']:
            raise ValueError('Cannot commit rejected priority candidate')
        if report['requires_priority_certificate'] and (kind == KIND or not certificate.get('certified', False)):
            raise ValueError('Cross-body yield requires a certified priority proposal')
        _, _, measured, depth, blocks = self._trial
        for row in report['yields']:
            if row['authorization'] == 'same_step_cross_body':
                key = (row['contact'], row['donor_body'])
                self.spent[key] = self.spent.get(key, 0.) + row['gap_m']
            self.yields.append(dict(row, iteration=self.epoch))
        for key, value in measured['depths'].items():
            self.best_keys[key] = min(self.best_keys.get(key, 0.), value)
        # Vanished witnessed collisions have cleared; future re-entry gets no
        # old depth allowance. Complete geometry/evidence was checked above.
        for key in self.best_keys.keys() - measured['depths'].keys():
            self.best_keys[key] = 0.
        self.best_depth = np.minimum(self.best_depth, depth)
        self.current_depth = depth.copy()
        self.current = np.array(poses, copy=True)
        self.current_measurement = measured
        self._current_contacts = self._contact_values(blocks)
        if descent is not None:
            self.descent_certificates.append(descent)
        self.epoch += 1
        self._stalled_search.invalidate()
        self._prepared = False
        self._trial = None
        self.certificate = None

    def diagnostics(self, poses, *, solver_history=None):
        from .priority_reporting import solver_progress
        from .runtime_options import OPTIONS
        if OPTIONS.windowed_priority:
            from .priority_window import window_state
            result = dict(window_state(self, self.current, range(len(self.current))).evaluate(poses))
        else:
            result = dict(self._evaluate_trial(poses))
        # Tuple-keyed slacks are a solver interface, not a JSON object schema.
        result['slacks'] = [dict(key=key, value=value) for key, value in result['slacks'].items()]
        result.update(committed_steps=self.epoch, committed_yields=list(self.yields),
                      committed_descent_certificates=list(self.descent_certificates),
                      pose_admissible=bool(result['accepted']),
                      accepted_semantics='priority_pose_admissibility_only_not_solver_success',
                      repeated_stalled_searches=list(self._stalled_search.skips),
                      immutable_reference=True, targets_modified=False,
                      yield_rule='same-step measured and predicted reduction; fixed cumulative donor ledger')
        result.update(solver_progress(solver_history, accepted_update_count=self.epoch))
        return result
