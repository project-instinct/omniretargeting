"""Window-local nonlinear priority checks and parent-owned net transactions.

QP rows are proposals. Neither a child's history nor its epigraph multipliers
can authorize an accepted yield. The parent verifies the final window afresh.
"""

from copy import copy
import hashlib
import inspect
import os

import numpy as np

from .priority_geometry import body_depths, window_body_depths
from .priority_window_contacts import checked_frames
from .continuity import ROUND_OFF

TOL = 1e-7
# Window step acceptance only, not a residual-defect threshold or yield credit.
# References, hard guards, contact allowances and the best-depth ledger do not
# inherit this tolerance. The final audit still reports defects above 1 cm.
WINDOW_PENETRATION_STEP_TOLERANCE = 0.0005


def _digest(value):
    a = np.asarray(value)
    return hashlib.sha256(repr((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def hard_guards(coordinator):
    obj = coordinator.objective
    items = [getattr(obj, 'source_shape', None), *getattr(obj, 'candidate_guards', ())]
    seen = set()
    for guard in items:
        if (guard is None or guard is coordinator or id(guard) in seen
                or getattr(guard, 'priority', None) is not None
                or isinstance(guard, type(coordinator))):
            continue
        seen.add(id(guard))
        yield guard


def dependencies(coordinator, frames):
    checked_frames(frames, len(coordinator.current))
    lo, hi = frames.start, frames.stop
    for component in [coordinator.objective, *hard_guards(coordinator)]:
        hook = getattr(component, 'window_dependencies', None)
        if hook is None:
            return None
        part = hook(frames)
        if part is None:
            return None
        lo, hi = min(lo, part.start), max(hi, part.stop)
    return range(lo, hi)


def hard_failures(coordinator, poses, frames):
    obj, failures = coordinator.objective, []
    coords = obj.coords
    part = checked_frames(frames, len(poses))
    if hasattr(coords, 'qi') and hasattr(obj, 'lower'):
        q = poses[part][:, coords.qi]
        low, high = np.asarray(obj.lower), np.asarray(obj.upper)
        low = low[part] if low.ndim == 2 else low
        high = high[part] if high.ndim == 2 else high
        if not (np.all(q >= low - 1e-9) and np.all(q <= high + 1e-9)):
            failures.append('joint_box')
    fixed_root = getattr(obj, 'fixed_root', None)
    if fixed_root is not None and coords.root_q is not None:
        q = coords.root_q
        if not np.array_equal(poses[part, q:q + 7], fixed_root[part]):
            failures.append('fixed_root')
    if getattr(coords, 'root_q', None) is not None:
        q = coords.root_q
        if getattr(obj, 'root_budget', None) is not None:
            if np.any(np.linalg.norm(poses[part, q:q + 3] - obj.reference[part, q:q + 3], axis=1)
                      > obj.root_budget + ROUND_OFF):
                failures.append('objective_root_ball')
        shape = getattr(obj, 'source_shape', None)
        if shape is not None and hasattr(shape, 'xy'):
            references = [shape.seed]
            if getattr(shape, 'acquired_poses', None) is not None:
                references.append(shape.acquired_poses)
            for index, reference in enumerate(references):
                if np.any(np.linalg.norm(poses[part, q:q + 3] - reference[part, q:q + 3], axis=1)
                          > shape.xy + ROUND_OFF):
                    failures.append('reference_root_ball:' + str(index))
    for guard in hard_guards(coordinator):
        # An unknown legacy guard is checked globally and forces serial work.
        scoped = 'frames' in inspect.signature(guard.accepts).parameters
        good = guard.accepts(poses, frames=frames) if scoped else guard.accepts(poses)
        if not good:
            failures.append(type(guard).__name__)
    return failures


class WindowState:
    """One frozen local linearization; full accepted references remain shared."""
    def __init__(self, coordinator, frames):
        self.coordinator, self.frames = coordinator, frames
        if (getattr(coordinator, '_funding_baseline', None) is not None
                and coordinator._funding_window is None):
            coordinator._funding_window = WindowState(coordinator._funding_baseline, frames)
        self.part = checked_frames(frames, len(coordinator.current))
        self.origin = coordinator.current.copy()
        self.epoch = coordinator.epoch
        self.measured = coordinator.provider.measure(self.origin, frames=frames, jacobians=True)
        _, self.owners = body_depths(coordinator.objective.model, len(self.origin), self.measured)
        self.matrix, self.phi = coordinator.provider.linear_rows(self.measured, frames=frames)
        expected = len(frames) * coordinator.objective.width
        if self.matrix.shape != (len(self.measured['rows']), expected):
            raise ValueError('Window witness rows must use local frame-major columns')
        self.blocks = [b for adapter in coordinator.contacts
                       for b in adapter.evaluate(self.origin, frames=frames)]
        self.identities = tuple(key for b in self.blocks for key in b.identities)
        if len(set(self.identities)) != len(self.identities):
            raise ValueError('Duplicated contact channel identities')
        from .window_contact_domain import WindowContactDomain
        self.contact_domain = WindowContactDomain(self)
        self.trial = None

    @property
    def baseline(self):
        """One immutable net transaction domain for private and parent checks."""
        return getattr(self.coordinator, '_funding_window', None) or self

    def soft_rows(self):
        result, indices, keys = [], [], []
        for index, (key, _) in enumerate(self.measured['rows']):
            for body in self.owners[key]:
                indices.append(index)
                keys.append(('penetration', int(key[0]), body))
        if indices:
            result.append((-self.matrix[indices], self.phi[indices].copy(), np.ones(len(indices)), tuple(keys)))
        from .priority_margins import margin_rows
        result += margin_rows(self.matrix, self.phi, self.measured['rows'],
                              self.coordinator.margin_preferences)
        result += self.contact_domain.soft_rows()
        return result

    def soft_cones(self):
        return self.contact_domain.soft_cones()

    def contact_values(self, poses):
        c = self.coordinator
        if np.array_equal(poses, self.origin):
            return self.blocks
        if self.trial is not None and np.array_equal(poses, self.trial[0]):
            return self.trial[4]
        cached = getattr(self, '_contact_values_cache', None)
        if cached is not None and np.array_equal(poses, cached[0]):
            return cached[1]
        blocks = [b for adapter in c.contacts for b in adapter.evaluate(poses, frames=self.frames, jacobian=False)]
        if tuple(key for b in blocks for key in b.identities) != self.identities:
            raise ValueError('SOC contact channels changed identity')
        self._contact_values_cache = (np.array(poses, copy=True), blocks)
        return blocks

    def soft_values(self, poses):
        c = self.coordinator
        blocks = self.contact_values(poses)
        # Original mesh witnesses may change at a candidate. A missing frozen
        # witness evaluator means no collision SOC coverage, not a new witness.
        linear = [None] if len(self.measured['rows']) else []
        from .priority_margins import margin_rows
        linear += [None] * len(margin_rows(self.matrix, self.phi, self.measured['rows'], c.margin_preferences))
        contact_linear, cones = self.contact_domain.soft_values(blocks)
        linear += contact_linear
        return linear, cones

    def qp_bounds(self):
        c = self.baseline.coordinator
        protected = np.array([c.current_measurement['depths'].get(key, 0.)
                              for key, _ in self.measured['rows']])
        # Keep half of the step allowance for nonlinear linearization error.
        tol = WINDOW_PENETRATION_STEP_TOLERANCE
        allowed = np.minimum(np.maximum(protected + tol / 2, -self.phi), protected + tol)
        linear = ([(-self.matrix, np.full(len(allowed), -np.inf), self.phi + allowed)]
                  if len(allowed) else [])
        contacts, cones = self.contact_domain.hard_bounds(self.blocks)
        return linear + contacts, cones

    def qp_values(self, poses):
        # No same-witness geometry evaluator is enabled here. Contact values
        # use frozen caps and identities, without selecting a new donor/domain.
        linear = [None] if len(self.measured['rows']) else []
        cones = []
        for block, upper in zip(self.contact_values(poses), self.contact_domain.hard_upper):
            if block.scalar:
                linear.append((np.full(len(upper), -np.inf), upper - block.residual))
            else:
                cones.append((block.residual.copy(), upper.copy()))
        return linear, cones

    def prediction(self, poses):
        c = self.coordinator
        step = c.objective.coords.difference(self.origin, poses)[self.part].ravel()
        values = np.maximum(-self.phi - self.matrix @ step, 0.)
        result = np.zeros_like(c.current_depth)
        columns = {}
        moved = np.abs(step) > 1e-12
        for index, (key, _) in enumerate(self.measured['rows']):
            row = self.matrix.getrow(index)
            used = set(int(i) for i in row.indices[(np.abs(row.data) > 1e-12) & moved[row.indices]])
            for body in self.owners[key]:
                result[key[0], body] = max(result[key[0], body], values[index])
                columns.setdefault((key[0], body), set()).update(used)
        return result, columns, step

    def evaluate(self, poses):
        c, part, frames = self.coordinator, self.part, self.frames
        poses = np.asarray(poses)
        report = dict(accepted=False, reliable=False, epoch=c.epoch, levels=None,
                      violating_frames=[], yields=[], requires_priority_certificate=False,
                      optimality_certified=False, convergence_certified=False,
                      hard_infeasibility_certified=False, contract='windowed_priority_net')
        if (poses.shape != self.origin.shape or not np.isfinite(poses).all()
                or not np.array_equal(poses[:frames.start], self.origin[:frames.start])
                or not np.array_equal(poses[frames.stop:], self.origin[frames.stop:])):
            return c._merit(dict(report, reason='invalid_window_candidate', violating_frames=list(frames)))
        measured = c.provider.measure(poses, frames=frames)
        report.update(query_scope=dict(measured.get('query_scope', {})),
                      origin_query_scope=dict(self.measured.get('query_scope', {})))
        try:
            depth = window_body_depths(c.objective.model, len(poses), frames, measured)
        except ValueError as error:
            return c._merit(dict(report, reason=str(error), violating_frames=list(frames)))
        predicted, _, _ = self.prediction(poses)
        tolerance = WINDOW_PENETRATION_STEP_TOLERANCE
        base = self.baseline.coordinator
        protection = base.current_depth
        regressions = [(key, float(value), base.current_measurement['depths'].get(key, 0.))
                       for key, value in measured['depths'].items()
                       if value > base.current_measurement['depths'].get(key, 0.) + tolerance]
        body_bad = [(int(f + frames.start), int(b)) for f, b in zip(*np.nonzero(
            depth[part] > protection[part] + tolerance))]
        blocks = [b for adapter in c.contacts for b in adapter.evaluate(poses, frames=frames, jacobian=False)]
        identities = tuple(key for b in blocks for key in b.identities)
        if identities != self.identities or any(not np.isfinite(b.values).all() for b in blocks):
            return c._merit(dict(report, reason='changed_or_nonfinite_contact_evidence',
                                 violating_frames=list(frames)))
        # A private child funds its whole current excursion against its fork
        # baseline, never by temporary local yields or local spent resets.
        funding = self.baseline
        base = funding.coordinator
        credit_depth = depth
        credit_predicted, donor_columns, credit_step = funding.prediction(poses)
        gains = np.maximum(np.minimum(base.best_depth - credit_depth,
                                      base.current_depth - credit_predicted), 0.)
        old_values = base._contact_values(funding.blocks)
        linear_by_id = {key: (b, i) for b in funding.blocks for i, key in enumerate(b.identities)}
        records, unfunded = [], []
        for block in blocks:
            for index, identity in enumerate(block.identities):
                value = float(block.values[index])
                old, limit = old_values[identity], float(block.bound[index] + block.own[index])
                frame, body = int(block.frames[index]), int(block.bodies[index])
                own_used = max(0., min(value, limit) - max(old, float(block.bound[index])))
                if own_used > TOL:
                    records.append(dict(contact=identity, frame=frame, body=body, donor_body=body,
                                        gap_m=own_used, contact_error_m=value, rank=int(block.ranks[index]),
                                        authorization='frozen_own_body', available_m=float(block.own[index]),
                                        measured_reduction_m=float(base.current_depth[frame, body] - depth[frame, body])))
                needed = max(0., value - max(old, limit))
                if needed <= TOL:
                    continue
                linear, i = linear_by_id[identity]
                start, width = (i, 1) if linear.scalar else (3 * i, 3)
                jac = linear.matrix[start:start + width]
                columns = set(int(k) for k in jac.indices[(np.abs(jac.data) > 1e-12)
                                                         & (np.abs(credit_step[jac.indices]) > 1e-12)])
                donors = []
                for donor in np.flatnonzero(gains[frame] > TOL):
                    donor = int(donor)
                    if donor == body or not columns.intersection(donor_columns.get((frame, donor), ())):
                        continue
                    total = max(0., base.initial_depth[frame, donor]
                                - min(base.best_depth[frame, donor], depth[frame, donor]))
                    available = min(float(gains[frame, donor]),
                                    max(0., total - base.spent.get((identity, donor), 0.)))
                    if available > TOL:
                        donors.append((available, donor))
                donors.sort(reverse=True)
                if not donors or donors[0][0] + TOL < needed:
                    unfunded.append(dict(contact=identity, frame=frame, body=body, gap_m=needed,
                                         available_m=0. if not donors else donors[0][0]))
                else:
                    budget, donor = donors[0]
                    records.append(dict(contact=identity, frame=frame, body=body, donor_body=donor,
                                        gap_m=needed, contact_error_m=value, rank=int(block.ranks[index]),
                                        authorization='same_step_cross_body', available_m=budget,
                                        measured_reduction_m=float(base.current_depth[frame, donor] - depth[frame, donor]),
                                        predicted_reduction_m=float(base.current_depth[frame, donor] - credit_predicted[frame, donor])))
        before = [float(c.current_depth[part].sum()), *self.contact_domain.levels(self.blocks)]
        levels = [float(depth[part].sum()), *self.contact_domain.levels(blocks)]
        net_before = float(base.current_depth[part].sum())
        total_regression = levels[0] > net_before + tolerance
        failures = hard_failures(c, poses, frames)
        bad = sorted({k[0] for k, *_ in regressions} | {f for f, _ in body_bad}
                     | {r['frame'] for r in unfunded} | (set(frames) if failures or total_regression else set()))
        actual = before[0] - levels[0]
        predicted_reduction = before[0] - float(predicted[part].sum())
        report.update(accepted=not bad, reliable=True, levels=levels, before_levels=before,
                      query_scope=dict(measured.get('query_scope', {})),
                      predicted_levels=[float(predicted[part].sum()), *self.contact_domain.levels(self.blocks)],
                      actual_reductions=[actual, *(np.asarray(before[1:]) - levels[1:]).tolist()],
                      predicted_reductions=[predicted_reduction, 0., 0., 0.],
                      strict_penetration_decrease=bool(actual > TOL),
                      violating_frames=bad, penetration_regressions=regressions,
                      body_regressions=body_bad, total_penetration_regression=total_regression,
                      penetration_step_tolerance_m=tolerance,
                      penetration_step_reference=('fork_baseline' if self.baseline is not self
                                                  else 'current_committed'),
                      net_before_penetration_m=net_before,
                      unfunded_contacts=unfunded, hard_failures=failures, yields=records,
                      contact_metrics_checked=len(identities))
        if bad:
            report['reason'] = ('hard_guard' if failures else 'penetration_regression'
                                if regressions or body_bad or total_regression else 'unfunded_contact_yield')
        from .priority_rejections import hard_details, collision_details
        rejection_details = collision_details(c, regressions, body_bad, depth, protection, tolerance)
        if failures:
            rejection_details += hard_details(c, poses, frames, failures)
        if total_regression:
            rejection_details.append(dict(guard='original_geometry', family='window_total',
                                          frames=list(frames), actual_m=levels[0], budget_m=net_before,
                                          tolerance_m=tolerance, excess_m=levels[0] - net_before - tolerance))
        rejection_details += [dict(row, guard='contact_funding', family='unfunded_yield',
                                   excess_m=row['gap_m'] - row['available_m'] - TOL) for row in unfunded]
        report['rejection_details'] = rejection_details
        slacks = {('penetration', int(f), int(b)): float(depth[f, b])
                  for f, b in zip(*np.nonzero(depth))}
        from .priority_margins import margin_cost, collision_diagnostics
        margin_slacks, tracking_margin_cost = margin_cost(measured, c.margin_preferences)
        slacks.update(margin_slacks)
        report.update(tracking_margin_cost=tracking_margin_cost,
                      clearance_preferences_level='tracking',
                      collision_diagnostics=collision_diagnostics(c, measured))
        slacks.update(self.contact_domain.values(blocks))
        report = c._merit(report, slacks)
        self.trial = (poses.copy(), report, measured, depth, blocks)
        return report


def window_state(coordinator, poses, frames, *, fresh=False):
    checked_frames(frames, len(coordinator.current))
    if not np.array_equal(poses, coordinator.current):
        raise ValueError('Window rows must start at the committed trajectory')
    state = getattr(coordinator, '_window_state', None)
    if (fresh or state is None or state.frames != frames or state.epoch != coordinator.epoch
            or not np.array_equal(state.origin, coordinator.current)):
        state = WindowState(coordinator, frames)
        coordinator._window_state = state
    return state


def fork_window(coordinator, poses, frames):
    if poses is not None and not np.array_equal(poses, coordinator.current):
        raise ValueError('Fork must start at the committed trajectory')
    part = checked_frames(frames, len(coordinator.current))
    child = copy(coordinator)
    for name in ('current', 'current_depth', 'best_depth'):
        setattr(child, name, getattr(coordinator, name).copy())
    for name in ('spent', 'best_keys', '_current_contacts'):
        setattr(child, name, dict(getattr(coordinator, name)))
    # Measurements are read-only here; installation creates a new mapping.
    # Sharing these witness arrays retains OS copy-on-write under real fork.
    child.current_measurement = coordinator.current_measurement
    child.contacts = list(coordinator.contacts)
    child.yields, child.descent_certificates = list(coordinator.yields), list(coordinator.descent_certificates)
    child.provider = coordinator.provider.fork_private_client()
    footprint = dependencies(coordinator, frames) or range(len(coordinator.current))
    child._fork_frames = frames
    child._fork_footprint = footprint
    child._fork_origin = coordinator.current.copy()
    child._fork_digest = _digest(coordinator.current[slice(footprint.start, footprint.stop)])
    child._fork_steps = []
    child._funding_window = None
    child._window_state = None
    child._prepared, child._trial, child.certificate = False, None, None
    from .priority_stall import StalledSearch
    child._stalled_search = StalledSearch()
    # Baseline clone is data-only: no constructor, FK, reference or depth query.
    child._funding_baseline = copy(child)
    for name in ('current_depth', 'best_depth'):
        setattr(child._funding_baseline, name, getattr(child, name).copy())
    child._funding_baseline.best_keys = dict(child.best_keys)
    child._funding_baseline._funding_baseline = None
    child._funding_baseline._funding_window = None
    return child


def export_transactions(child, frames):
    if getattr(child, '_fork_frames', None) != frames:
        raise ValueError('Transactions require the original private window')
    footprint = child._fork_footprint
    return dict(kind='window_net_claim', frames=(frames.start, frames.stop),
                footprint=(footprint.start, footprint.stop), baseline_sha256=child._fork_digest,
                final_sha256=_digest(child.current[slice(frames.start, frames.stop)]),
                local_steps=list(child._fork_steps), credit_authority=False)


def commit_window(coordinator, poses, frames, *, transactions=None, local=False):
    if not local and os.getpid() != coordinator.owner_pid:
        raise RuntimeError('Only the parent can commit window credit')
    part = checked_frames(frames, len(coordinator.current))
    if local and getattr(coordinator, '_fork_frames', None) != frames:
        raise ValueError('Local advancement requires a private window coordinator')
    if not local and getattr(coordinator, '_fork_frames', None) is not None:
        raise RuntimeError('Private coordinators must advance_local, never commit')
    poses = np.asarray(poses)
    if poses.shape != coordinator.current.shape:
        raise ValueError('Window commit requires full poses')
    candidate = coordinator.current.copy()
    candidate[part] = poses[part]
    if transactions is None:
        if not np.array_equal(candidate, poses):
            raise ValueError('A serial candidate changed poses outside its window')
    else:
        footprint = dependencies(coordinator, frames) or range(len(candidate))
        expected = (footprint.start, footprint.stop)
        if (transactions.get('kind') != 'window_net_claim'
                or tuple(transactions.get('frames', ())) != (frames.start, frames.stop)
                or tuple(transactions.get('footprint', ())) != expected
                or transactions.get('baseline_sha256') != _digest(coordinator.current[slice(*expected)])
                or transactions.get('final_sha256') != _digest(candidate[part])):
            raise ValueError('Stale or mismatched window transaction')
    state = window_state(coordinator, coordinator.current, frames, fresh=True)
    report = state.evaluate(candidate)
    if not report['accepted']:
        raise ValueError('Cannot commit rejected window: ' + report.get('reason', 'unknown'))
    _, _, measured, depth, blocks = state.trial
    # Everything below is installation after complete net verification.
    if not local:
        for row in report['yields']:
            if row['authorization'] == 'same_step_cross_body':
                key = (row['contact'], row['donor_body'])
                coordinator.spent[key] = coordinator.spent.get(key, 0.) + row['gap_m']
            coordinator.yields.append(dict(row, iteration=coordinator.epoch, verification='parent_net'))
    for key in list(coordinator.best_keys):
        if key[0] in frames:
            coordinator.best_keys[key] = min(coordinator.best_keys[key], measured['depths'].get(key, 0.))
    for key, value in measured['depths'].items():
        coordinator.best_keys[key] = min(coordinator.best_keys.get(key, 0.), value)
    coordinator.best_depth[part] = np.minimum(coordinator.best_depth[part], depth[part])
    coordinator.current_depth[part] = depth[part]
    old = coordinator.current_measurement
    coordinator.current_measurement = dict(old,
        depths={**{k: v for k, v in old['depths'].items() if k[0] not in frames}, **measured['depths']},
        rows=[(k, v) for k, v in old['rows'] if k[0] not in frames] + list(measured['rows']))
    coordinator._current_contacts.update(coordinator._contact_values(blocks))
    coordinator.current = candidate
    coordinator.epoch += 1
    coordinator._window_state = None
    coordinator._prepared, coordinator._trial, coordinator.certificate = False, None, None
    coordinator._stalled_search.invalidate()
    if local:
        coordinator._fork_steps.append(dict(accepted=True, actual_reduction_m=report['actual_reductions'][0],
                                           temporary_yields=len(report['yields'])))
