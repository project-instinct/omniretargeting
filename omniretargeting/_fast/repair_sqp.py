"""Baseline group repair with interior witnesses and second-order candidates.

Acceptance is still RepairLimits.check and repair_acceptance.measure_candidate.
No production module is patched; the baseline schedule and gates are reused.
"""

import time
from contextlib import closing
from copy import copy

import numpy as np
from scipy import sparse

from .mesh_repair import (
    OriginalGeometry as BaseGeometry, _repair_groups, _activation_depth, _joint_projection as _baseline_projection,
    ITERATIONS, MIN_GROUP_PROGRESS_M, DEEP_RESIDUAL_M, COLLISION_SIGMA,
    LIMB_COLLISION_SIGMA, WITNESS_BAND_CAP,
)
from .repair_limits import RepairLimits, LinearBounds, LOCAL_ANGLE
from .limb_scope import scope_for_body
from .repair_schedule import RepairSchedule
from .repair_frames import halo_frames, restrict
from .repair_proposals import gate_predictions, transition_group
from .local_qp import witness_lower, cut_lower, CLEARANCE_MARGIN_M
from .local_repair import _inside_box
from .local_chatter import JointChatter

WITNESS_BAND = 3e-3
LIMIT_REFRESHES = 0
GEOMETRY_REFRESHES = 2
SELF_PAIR_SCOPE_DEPTH_M = .005
AUDIT_WITNESSES_PER_KEY = 48
COLLISION_ROW_SCALE = 1e3  # Measure collision rows in mm, including their RHS.


def _audit_key(key):
    return key[1] == "static" and len(key) == 4 and key[2] == -1


def _physical_key(key):
    return (key[0], "static", key[-1]) if _audit_key(key) else key


def _joint_projection(limits, model, measured):
    """A deep self pair can recruit its other chain under the same guards.

    A paired-limb scope otherwise fixes the colliding hand while only moving
    the hip. Add that pair's existing ancestor/compensator columns; never add
    floating-root coordinates or change contact anchors or their budgets.
    """
    scope = scope_for_body(limits, limits.movable_body)
    limits._sqp_widened_self = scope is not None and any(
        key[1] == "self" and depth > SELF_PAIR_SCOPE_DEPTH_M and any(
            model.body_weldid[model.geom_bodyid[g]] == limits.movable_body for g in key[2:])
        for key, depth in measured["depths"].items())
    if not limits._sqp_widened_self:
        return _baseline_projection(limits, model, measured)
    original = limits.limb_scopes
    limits.limb_scopes = ()
    try:
        return _baseline_projection(limits, model, measured)
    finally:
        limits.limb_scopes = original


class OriginalGeometry(BaseGeometry):
    """Query the union once, retaining independent original/audit budgets.

    More probes must not enlarge an old per-key budget or contact yield.
    Original probes and the full union retain fixed budgets and use the unchanged
    regression check. Their shared epigraph avoids doubling objective weight.
    """

    def _query_static_batch(self, pending):
        from .repair_static_query import prepare
        from .collision_manifold import select_collision_manifold
        scene, native = self.scene, prepare(self)
        captured = []
        old = {id(summary): list(summary["rows"]) for summary, _, _ in pending}

        class Capture:
            def query(self, points):
                result = native.query(points)
                captured.append(result)
                return result

            def __getattr__(self, name):
                value = getattr(scene, name)
                if name == "query_with_sign_hints":
                    def hints(*args):
                        result = value(*args)
                        captured.append(result)
                        return result
                    return hints
                return value

        self.scene = Capture()
        try:
            super()._query_static_batch(pending)
        finally:
            self.scene = scene
        if not self.manifold:
            return
        assert len(captured) == 1
        result, first, gathered = captured[0], 0, {}
        for summary, points, ids in pending:
            last = first + len(ids)
            phi, direction = result["clearance"][first:last], result["directions"][first:last]
            valid = (np.isfinite(phi) & ~result["uncertain"][first:last]
                     & np.isfinite(direction).all(axis=1) & np.any(direction != 0, axis=1) & (phi < .008))
            bucket = gathered.setdefault(id(summary), (summary, list(old[id(summary)])))[1]
            bucket.extend(dict(point=p.copy(), direction=d.copy(), clearance=float(v), sample_id=int(s))
                          for p, d, v, s in zip(points[valid], direction[valid], phi[valid], ids[valid]))
            first = last
        original = self.objective._sqp_original_probes
        for summary, rows in gathered.values():
            retained = []
            for partition in (True, False):
                family = [r for r in rows if bool(original[r["sample_id"]]) == partition]
                if not family:
                    continue
                points = np.array([r["point"] for r in family])
                phi = np.array([r["clearance"] for r in family])
                directions = np.array([r["direction"] for r in family])
                selected = select_collision_manifold(points, phi, directions, origin=summary["origin"])
                band = getattr(self, "witness_band", None)
                if band is not None:
                    near = np.flatnonzero(phi <= phi.min() + band)
                    selected = np.union1d(selected, near[np.argsort(phi[near], kind="stable")[:WITNESS_BAND_CAP]])
                ids = np.array([family[i]["sample_id"] for i in selected])
                selected = selected[np.unique(ids, return_index=True)[1]]
                retained.extend(family[i] for i in selected)
            summary["rows"] = retained

    def _record(self, rows, depths, key, row):
        if key[1] == "static":
            sample = row.get("sample_id", -1)
            if sample is not None and sample >= 0 and not self.objective._sqp_original_probes[sample]:
                key = (key[0], "static", -1, key[-1])
        BaseGeometry._record(rows, depths, key, row)
        if key[1] == "static":
            union = (key[0], "static", -1, key[-1])
            original = (key[0], "static", key[-1])
            depths[union] = max(depths.get(union, 0.), depths.get(original, 0.))


def _physical_depths(depths):
    result = {}
    for key, depth in depths.items():
        physical = _physical_key(key)
        result[physical] = max(result.get(physical, 0.), depth)
    return result


def measure_candidate(objective, geometry, candidate, measured, protected, group, before, *, frames):
    """Preserve old-sample descent and the complete material union's descent."""
    from .repair_acceptance import measure_candidate as original_acceptance
    checked, bad, _, _ = original_acceptance(
        objective, geometry, candidate, measured, protected, group, before, frames=frames)
    original_before = sum(v*v for k, v in measured["depths"].items() if not _audit_key(k))
    original_after = sum(v*v for k, v in checked["depths"].items() if not _audit_key(k))
    after = sum(v*v for v in _physical_depths(checked["depths"]).values())
    return checked, bad, after, after < before - 1e-10 and original_after < original_before - 1e-10


class Ledger:
    def __init__(self):
        self.counts = dict(qp=0, correction=0, fallback_fractions=0,
                           geometry_measures=0, rejections=0,
                           limit_rejections=0, geometry_rejections=0, cuts=0, corrected_resolves=0, geometry_resolves=0, chatter_objective_resolves=0)
        self.seconds = dict(qp=0., correction=0., limits=0., geometry=0., fallback=0.)
        self.rejection_reasons = {}
        self.qps = []
        self.corrections = []
        self.setup_seconds = self.entry_measure_seconds = 0.

    def report(self):
        return dict(counts=self.counts, seconds=self.seconds,
                    setup_seconds=self.setup_seconds,
                    entry_measure_seconds=self.entry_measure_seconds,
                    rejection_reasons=self.rejection_reasons, qps=self.qps)


def _cleanup_chatter(objective, limits, geometry, current, measured, protected, *, introduced_only=False):
    """Clean only audited excess, keeping contacts and achieved depths fixed."""
    from .local_chatter import JointChatter
    from .local_repair import Ledger as LocalLedger, State, _Context
    from .local_rounds import waves, run_wave
    from .local_windows import ClusterLimits, chatter_windows

    started = time.monotonic()
    chatter = JointChatter(objective.coords, objective.reference)

    def summary(poses):
        excess = np.maximum(chatter.excess(poses), 0.)
        return dict(violations=int((excess > 0).sum()), excess=float(excess.sum()))

    before = summary(current)
    cleanup = dict(status="skipped_no_chatter", chatter_before=before,
                   chatter_after=before.copy(), counts={}, seconds_by_task={}, history=[])
    if not before["violations"]:
        cleanup["seconds"] = time.monotonic() - started
        return current, measured, cleanup
    if introduced_only and np.all(np.maximum(chatter.excess(current), 0.)
                                  <= np.maximum(chatter.excess(limits.entry), 0.) + 1e-9):
        cleanup.update(status="preserved_entry_chatter", seconds=time.monotonic() - started)
        return current, measured, cleanup
    # Complete the final group's established anchor transition before all
    # chatter contacts become fixed. Otherwise unmoving contacts of that
    # group still refer to its old anchor and fail the whole-domain check.
    transition_group(limits, current, None)
    ClusterLimits.adopt(limits)
    if any("jacobian" not in row for _, row in measured["rows"]):
        measured = geometry.linearize(current, measured)
    local_ledger = LocalLedger()
    state = State(current, measured, protected["depths"])
    ctx = _Context(objective, limits, geometry, chatter, local_ledger, cleanup["history"])
    windows = chatter_windows(chatter, current, limits.width)
    local_ledger.counts["chatter_windows"] = len(windows)
    for wave in waves(windows):
        run_wave(ctx, wave, state)
    history = cleanup["history"]
    ends = [row["started"] for row in history[1:]] + [time.monotonic()]
    for row, end in zip(history, ends):
        row["seconds"] = row.get("elapsed", end - row["started"])
        row.pop("started", None)
        row.setdefault("stop_reason", "accepted" if row["alpha"] else "chatter_rejected")
        row.setdefault("solver_status", "chatter_cleanup")
        row["contact_group"] = None
        row["stage"] = "chatter_cleanup"
    cleanup.update(status="corrected" if local_ledger.counts["accepted"] else "retained",
                   chatter_after=summary(state.current), counts=local_ledger.counts,
                   seconds_by_task=local_ledger.seconds, corrections=local_ledger.corrections,
                   seconds=time.monotonic() - started)
    return state.current, state.measured, cleanup


def _rows(geometry, measured, protected):
    matrix, phi = geometry.linear_rows(measured)
    keys = [key for key, _ in measured["rows"]]
    budgets = np.array([protected["depths"].get(key, 0.) for key in keys])
    return matrix, phi, budgets, keys


def _soc(proposal, trial):
    """Pure minimum norm; screen with one fixed ball and check it afterwards.

    Its radius cone is unnecessary for the minimum-norm optimum inside that
    ball. Removing this global cone avoids cross-frame KKT fill; the complete
    residual and physical norm checks retain the screening certificate.
    """
    import clarabel
    from .local_correction import _Problem, STEP_SCALE, MAX_ITERATIONS
    from .local_repair import _reserve
    from .local_qp import _settings
    from .working_set import solve as active_solve
    limits, p = proposal.limits, proposal.projection
    offset = (trial[:, limits.coords.qi] - proposal.current[:, limits.coords.qi]).ravel()
    bounds = limits.constraints(trial, projection=p)
    _reserve(bounds)
    matrix, phi, budgets, _ = _rows(proposal.geometry, proposal.measured, proposal.protected)
    bounds.add(matrix, witness_lower(phi, budgets, matrix @ offset), np.inf)
    chatter = getattr(limits, "_sqp_chatter", None)
    if chatter is not None:
        moved = np.asarray(p.getnnz(axis=1)).reshape(limits.frames, limits.width) > 0
        m, lo, hi = chatter.rows(proposal.current, chatter.stencil_frames(moved.any(axis=1)),
                                np.flatnonzero(moved.any(axis=0)), elastic=False)
        shift = m @ offset
        bounds.add(m, lo - shift, hi - shift)
    problem = _Problem(bounds, p)
    if problem.constants:
        return None
    radius = proposal.soc_radius = max(getattr(proposal, "soc_radius", .02),
                                       .5 * float(np.linalg.norm(offset)))
    program, info = problem.program(radius, True)
    h, g, a, b, cones = program
    first = info["rows"]
    trust_index = int(first > 0)
    size = p.shape[1] + 1 if problem.selector else np.count_nonzero(p.getnnz(axis=1)) + 1
    keep = np.r_[np.arange(first), np.arange(first + size, len(b))]
    del cones[trust_index]
    def settings():
        value = _settings(); value.direct_solve_method = "qdldl"
        value.max_threads = 1; value.max_iter = MAX_ITERATIONS
        return value
    tick = time.monotonic()
    proposal.ledger.counts["qp"] += 1
    proposal.ledger.counts["correction"] += 1
    result = active_solve(h, g, a[keep].tocsc(), b[keep], cones, settings, diagnostics=info)
    if result is None and len(b[keep]) <= 4096:
        # A tiny complete projection costs less than an active-set restart.
        info["direct_fallback"] = True
        result = clarabel.DefaultSolver(h, g, a[keep].tocsc(), b[keep], cones, settings()).solve()
    elapsed = time.monotonic() - tick
    proposal.ledger.seconds["correction"] += elapsed
    info.update(type="soc", group=limits.movable_body, seconds=elapsed,
                status="uncertified" if result is None else str(result.status), radius=radius)
    proposal.ledger.qps.append(info)
    proposal.ledger.corrections.append(info)
    if result is None or result.x is None or not np.isfinite(result.x).all():
        return None
    delta = np.asarray(result.x) * min(radius, STEP_SCALE)
    info["linearized_feasible"] = problem.feasible(delta, radius, info)
    if not info["linearized_feasible"] or np.max(np.abs(delta), initial=0.) < 1e-12:
        return None
    return np.asarray(p @ delta).reshape(limits.frames, limits.width)


class SQPProposal:
    def __init__(self, limits, geometry, current, measured, protected, fixed_q, depth, ledger):
        self.limits, self.geometry, self.current = limits, geometry, current
        self.measured, self.protected, self.fixed_q = measured, protected, fixed_q
        self.depth, self.ledger = depth, ledger
        self.projection = restrict(_joint_projection(limits, geometry.model, measured), limits.frame_mask)
        self.feedback, self.cut_rows = None, 0
        self.cuts = []
        self.limit_cuts = []
        self.prediction, fast = gate_predictions(limits, current, geometry, measured, protected, depth)
        self.step, self.stationary = None, False
        if self.prediction is not None:
            self.status = "no_local_dofs" if self.prediction["reason"] == "no_local_dofs" else "pre_qp_certified_small_progress"
            if self.status != "no_local_dofs":
                self.step = np.zeros((limits.frames, limits.width))
            return
        from .repair_fast_gate import should_skip
        if fast is not None and should_skip(fast):
            self.prediction = dict(fast, reason="reduced_predicted_small_progress", reduced_gate=fast)
            self.step = np.zeros((limits.frames, limits.width))
            self.status = "reduced_predicted_small_progress"
            return
        self.step, self.status = self.solve()
        self.stationary = self.step is not None and np.max(np.abs(self.step), initial=0.) < 1e-9
        from .repair_prediction import linear_prediction, should_gate
        if self.step is not None:
            self.prediction = linear_prediction(geometry, measured, self.step, limits.movable_body, depth)
            if self.prediction is not None and fast is not None:
                self.prediction["reduced_gate"] = fast

    def solve(self):
        tick = time.monotonic()
        self.ledger.counts["qp"] += 1
        info = {}
        try:
            result = solve_step(self.limits, self.current, self.geometry, self.measured,
                              self.protected, self.projection,
                              collision_sigma=(LIMB_COLLISION_SIGMA if scope_for_body(
                                  self.limits, self.limits.movable_body) is not None else COLLISION_SIGMA),
                              local_angle=getattr(self.limits, "local_angle", LOCAL_ANGLE), cuts=self.cuts,
                              limit_cuts=self.limit_cuts, diagnostics=info,
                              restore_geometry=getattr(self, "restore_geometry", False),
                              omit_chatter_objective=getattr(self, "omit_chatter_objective", False))
            info.update(status=result[1], group=self.limits.movable_body,
                        witness_cut_round=len(self.cuts), limit_cut_round=len(self.limit_cuts),
                        geometry_refresh_round=getattr(self, "geometry_refresh_round", 0))
            return result
        finally:
            info["seconds"] = time.monotonic() - tick
            self.ledger.seconds["qp"] += info["seconds"]
            self.ledger.qps.append(info)

    @property
    def has_direction(self):
        return self.step is not None

    def check(self, candidate, record, alpha):
        tick = time.monotonic()
        if self.fixed_q is not None and not np.array_equal(candidate[:, self.fixed_q], self.current[:, self.fixed_q]):
            valid, check = False, dict(reason="joint outside limb scope changed")
        else:
            valid, check = self.limits.check(candidate)
        chatter = getattr(self.limits, "_sqp_chatter", None)
        if valid and chatter is not None:
            before = np.maximum(chatter.excess(self.current), 0.)
            after = np.maximum(chatter.excess(candidate), 0.)
            if np.any(after > before + 1e-9):
                valid, check = False, dict(reason="source-relative chatter floor",
                                          excess_rad=float(np.max(after - before)))
        self.ledger.seconds["limits"] += time.monotonic() - tick
        record["check"] = check
        if not valid:
            self.ledger.counts["rejections"] += 1
            self.ledger.counts["limit_rejections"] += 1
            record.setdefault("limit_rejection_history", []).append(dict(alpha=alpha, **check))
            record["rejected_limits"] = check
            reason = check.get("guard", check.get("reason", "unknown"))
            self.ledger.rejection_reasons[reason] = self.ledger.rejection_reasons.get(reason, 0) + 1
        return valid

    def full_candidate(self, step, record):
        candidate = _inside_box(self.limits, self.current, step)
        if self.check(candidate, record, 1.):
            return candidate
        while getattr(self, "soc_attempts", 0) < 2:
            self.soc_attempts = getattr(self, "soc_attempts", 0) + 1
            delta = _soc(self, candidate)
            if delta is None:
                break
            candidate = _inside_box(self.limits, candidate, delta)
            if self.check(candidate, record, 1.):
                return candidate
        if len(self.limit_cuts) >= LIMIT_REFRESHES or self.cuts:
            record["stop_reason"] = "corrected_direction_rejected"
            return None
        # Rejected nonlinear rows are linearized at the trial, then translated
        # back to the current origin. Re-solve with a fixed refresh cap;
        # no global trust cone, radius sequence or fraction ladder is needed.
        offset = (candidate[:, self.limits.coords.qi] - self.current[:, self.limits.coords.qi]).ravel()
        bounds = self.limits.constraints(candidate, projection=self.projection)
        linear, norms = [], []
        for matrix, lower, upper in zip(bounds.matrices, bounds.lower, bounds.upper):
            selected = sparse.csr_matrix(matrix)
            shift = selected @ offset
            linear.append((selected, np.where(lower > 0, lower + 1e-6, lower) + shift,
                           np.where(upper < 0, upper - 1e-6, upper) + shift))
        for matrix, residual, allowed in bounds.norms:
            selected = sparse.csr_matrix(matrix)
            value = residual - (selected @ offset).reshape(residual.shape)
            violated = np.linalg.norm(residual, axis=1) > allowed + 1e-12
            norms.append((selected, value, np.maximum(allowed - violated * 1e-6, 0.)))
        if not linear and not norms:
            record["stop_reason"] = "no_variable_limit_cut"
            return None
        self.limit_cuts.append((linear, norms))
        self.ledger.counts["corrected_resolves"] += 1
        step, status = self.solve()
        record["corrected_resolve_status"] = status
        if step is None:
            return None
        return self.full_candidate(step, record)

    def candidates(self, record):
        """Guard-feasible SQP trials with bounded exact-witness refreshes."""
        candidate = self.full_candidate(self.step, record)
        for cut_round in range(GEOMETRY_REFRESHES + 1):
            if candidate is None:
                break
            self.feedback = None
            yield 1., candidate
            if cut_round == GEOMETRY_REFRESHES or self.feedback is None:
                break
            trial, checked, regressed = self.feedback
            # Prefix rejection is useful witness evidence, never acceptance.
            if not regressed or checked["failures"]:
                break
            # A newly penetrating sample can have zero origin derivative.
            # Its origin tangent then cannot model the second-order collision.
            # Advance the SQP linearization to the guard-valid trial, query the
            # complete geometry there, and solve on those exact witnesses.
            tick = time.monotonic()
            self.ledger.counts["geometry_measures"] += 1
            full = self.geometry.measure(trial, frames=range(len(trial)))
            self.ledger.seconds["geometry"] += time.monotonic() - tick
            if not full.get("complete", True) or full["failures"]:
                record["stop_reason"] = "unknown_trial_geometry"
                break
            full = self.geometry.linearize(trial, full)
            added = sum(key in set(regressed) for key, _ in full["rows"])
            self.cut_rows += added
            self.ledger.counts["cuts"] += added
            self.ledger.counts["geometry_resolves"] += 1
            self.geometry_refresh_round = cut_round + 1
            origin, origin_measurement = self.current, self.measured
            self.current, self.measured = trial, full
            self.restore_geometry = True
            self.limit_cuts, self.cuts = [], []
            try:
                step, status = self.solve()
                record["cut_solver_status"] = status
                if step is not None:
                    candidate = self.full_candidate(step, record)
                else:
                    candidate = None
            finally:
                self.current, self.measured = origin, origin_measurement
                self.restore_geometry = False
            if candidate is not None:
                # Check the exact chatter floor against the accepted origin,
                # then let the usual complete geometry acceptance decide.
                if not self.check(candidate, record, 1.):
                    candidate = None
        record.setdefault("stop_reason", "rejected_direction")

    def close(self):
        pass


class SerialProposals:
    """Established group/gate policy without speculative CPU work."""
    def __init__(self, limits, geometry, ledger):
        self.limits, self.geometry, self.ledger = limits, geometry, ledger
        self.stats = dict(local=0, worker_count=0, disabled_reason="serial SQP candidates")

    def prefetch(self, *args):
        pass

    def take(self, current, measured, protected, fixed_q, depth):
        self.stats["local"] += 1
        return SQPProposal(self.limits, self.geometry, current, measured, protected, fixed_q, depth, self.ledger)

    def note_outcome(self, changed):
        pass

    def close(self, *, abort=False):
        pass


def solve_step(limits, poses, geometry, measurement, baseline, projection, *,
               collision_sigma, local_angle, cuts=(), limit_cuts=(), diagnostics=None, restore_geometry=False, omit_chatter_objective=False):
    import clarabel

    variables = projection.shape[1]
    if not variables:
        return None, "no_local_dofs"
    if limit_cuts:
        # Refresh every nonlinear row at the trial, rather than intersecting
        # stale origin tangents with only the first violated family.
        constraints = LinearBounds(limits.frames * limits.width)
        constraints.set_box((limits.lower - poses[:, limits.coords.qi]).ravel(),
                            (limits.upper - poses[:, limits.coords.qi]).ravel())
        linear, norms = limit_cuts[-1]
        for matrix, lower, upper in linear:
            constraints.add(matrix, lower, upper)
        constraints.norms.extend(norms)
    else:
        constraints = limits.constraints(poses, projection=projection)
    chatter_term = None
    chatter = getattr(limits, "_sqp_chatter", None)
    if chatter is not None:
        moved = np.asarray(projection.getnnz(axis=1)).ravel().reshape(limits.frames, limits.width) > 0
        columns = np.flatnonzero(moved.any(axis=0))
        frames = chatter.stencil_frames(moved.any(axis=1))
        if len(columns) and len(frames):
            chatter_rows = chatter.rows(poses, frames, columns, elastic=False)
            constraints.add(*chatter_rows)
            # The hard current-value floor prevents new excess. A sparse
            # least-squares term also removes pre-existing excess on this
            # group's movable stencil, without an elastic constraint/slack.
            content = chatter.content(poses)[np.ix_(frames, columns)].ravel()
            limit = chatter.limit[np.ix_(frames, columns)].ravel()
            violated = np.abs(content) > limit
            if violated.any():
                matrix = (chatter_rows[0][violated] @ projection).tocsr() / 2e-4
                target = np.sign(content[violated]) * np.maximum(limit[violated] - 2e-4, 0.)
                chatter_term = matrix, (content[violated] - target) / 2e-4
    collision, phi = geometry.linear_rows(measurement)
    budgets = np.array([baseline["depths"].get(key, 0.) for key, _ in measurement["rows"]])
    # Keep the nonlinear guard's tiny tolerance in reserve. Spending it in
    # the linear subproblem leaves no room for curvature and roundoff.
    # Accepted roundoff on a different fixed chain must leave zero feasible.
    # Final acceptance still uses the original protected depths + REGRESSION.
    lower = witness_lower(phi, budgets, np.zeros(len(phi)))
    if restore_geometry:
        violated = -phi > budgets
        # A new contact has no penetration budget. Put its trial tangent
        # inside free space as well: targeting exactly zero leaves the next
        # nonlinear evaluation penetrating by the remaining curvature.
        lower[violated] = -(budgets[violated] - CLEARANCE_MARGIN_M) - phi[violated]
    constraints.add(collision * COLLISION_ROW_SCALE, lower * COLLISION_ROW_SCALE, np.inf)
    for matrix, value, budget, displacement in cuts:
        constraints.add(matrix * COLLISION_ROW_SCALE,
                        cut_lower(matrix, value, budget, displacement) * COLLISION_ROW_SCALE, np.inf)
    # One epigraph per collision group: all witnesses constrain its maximum
    # depth. Duplicating samples never duplicates objective weight, and a flat
    # patch cannot improve by exchanging its deepest corner for another one.
    keys, physical_depths = [], _physical_depths(measurement["depths"])
    varying_collision = np.asarray((collision @ projection).getnnz(axis=1)).ravel() > 0
    coupled = {}
    for (key, _), varying in zip(measurement["rows"], varying_collision):
        if (getattr(limits, "_sqp_coupled_geometry", False) and chatter_term is None
                and varying and measurement["depths"].get(key, 0.) > _activation_depth()):
            # The upper tail of every coupled collider includes the frames
            # deciding its max/p95. Other frames retain all protection rows.
            physical = _physical_key(key)
            coupled.setdefault(physical[1:], set()).add(physical)
    tail = max(3, int(np.ceil(limits.frames * .1)))
    coupled_keys = set()
    for values in coupled.values():
        coupled_keys.update(sorted(values, key=lambda k: (-physical_depths[k], k[0]))[:tail])
    for key, depth in physical_depths.items():
        if depth <= 0:
            continue
        if key in coupled_keys:
            keys.append(key)
            continue
        if limits.movable_body is not None:
            geoms = key[2:] if key[1] == "self" else key[-1:]
            if not any(geometry.model.body_weldid[geometry.model.geom_bodyid[g]] == limits.movable_body
                       for g in geoms):
                continue
        keys.append(key)
    if not keys:
        return None, "no_active_collision"
    lookup = {key: index for index, key in enumerate(keys)}
    selected = np.array([i for i, (key, _) in enumerate(measurement["rows"])
                         if _physical_key(key) in lookup], dtype=int)
    group = np.array([lookup[_physical_key(measurement["rows"][i][0])] for i in selected], dtype=int)
    joint_variables = variables
    count = len(keys)
    variables += count
    shift = (poses[:, limits.coords.qi] - limits.entry[:, limits.coords.qi]).ravel()
    joint_hessian = projection.T @ projection / local_angle**2
    joint_gradient = projection.T @ shift / local_angle**2
    if diagnostics is not None:
        diagnostics["chatter_soft_rows"] = 0 if chatter_term is None else chatter_term[0].shape[0]
    if chatter_term is not None and not omit_chatter_objective:
        chatter_matrix, chatter_offset = chatter_term
        joint_hessian = joint_hessian + chatter_matrix.T @ chatter_matrix
        joint_gradient = joint_gradient + chatter_matrix.T @ chatter_offset
    hessian = sparse.block_diag((joint_hessian, sparse.eye(count) / collision_sigma**2), format="csc")
    # Use t = attained_depth + delta_t. This exact variable translation makes
    # zero a feasible collision seed: the working set starts with deepest
    # witnesses rather than every penetrating point in the 3 mm witness band.
    # All omitted witnesses are still checked by the complete working-set QP.
    attained = np.array([physical_depths[key] for key in keys])
    gradient = np.r_[joint_gradient, attained / collision_sigma**2]
    scale = 1 / np.sqrt(np.maximum(hessian.diagonal(), 1e-12))
    diagonal = sparse.diags(scale)
    joint_diagonal = sparse.diags(scale[:joint_variables])
    a, lower, upper = constraints.packed()
    upper_mask, lower_mask = np.isfinite(upper), np.isfinite(lower)
    constrained = sparse.hstack((a @ projection, sparse.csc_matrix((a.shape[0], count))), format="csc") @ diagonal
    constrained.eliminate_zeros()
    variable = np.asarray(constrained.getnnz(axis=1)).ravel() > 0
    upper_mask &= variable | (upper < 0)
    lower_mask &= variable | (lower > 0)
    matrix = sparse.vstack([constrained[upper_mask], -constrained[lower_mask]], format="csc")
    rhs = np.r_[upper[upper_mask], -lower[lower_mask]]
    epigraph = sparse.csc_matrix((-np.ones(len(selected)), (np.arange(len(selected)), group)),
                                  shape=(len(selected), count))
    maximum_rows = sparse.hstack((-collision[selected] @ projection, epigraph), format="csc") @ diagonal
    positive_rows = sparse.hstack((sparse.csc_matrix((count, joint_variables)), -sparse.eye(count)),
                                 format="csc") @ diagonal
    blocks = [matrix, maximum_rows * COLLISION_ROW_SCALE, positive_rows * COLLISION_ROW_SCALE]
    targets = [rhs, (phi[selected] + attained[group]) * COLLISION_ROW_SCALE, attained * COLLISION_ROW_SCALE]
    nonneg = len(rhs) + len(selected) + count
    cones = [clarabel.NonnegativeConeT(nonneg)]
    soc_sizes = []
    for jac, residual, allowed in constraints.cones():
        dimension = residual.shape[1]
        jac = (jac @ projection).tocsr()
        jac.eliminate_zeros()
        variable = np.asarray(jac.getnnz(axis=1)).reshape(-1, dimension).sum(axis=1) > 0
        constant_valid = np.linalg.norm(residual, axis=1) <= allowed + 1e-12
        keep = variable | ~constant_valid
        jac, residual, allowed = jac[np.repeat(keep, dimension)], residual[keep], allowed[keep]
        norm_count, dimension = residual.shape
        if not norm_count:
            continue
        # Normalize every cone by its unchanged geometric budget.
        normalization = np.maximum(allowed, 1e-3)
        jac = (sparse.diags(np.repeat(1 / normalization, dimension)) @ jac @ joint_diagonal).tocoo()
        rows = jac.row + jac.row // dimension + 1
        blocks.append(sparse.csc_matrix((-jac.data, (rows, jac.col)),
                                        shape=(norm_count * (dimension + 1), variables)))
        targets.append(np.c_[allowed / normalization, residual / normalization[:, None]].ravel())
        cones.extend(clarabel.SecondOrderConeT(dimension + 1) for _ in range(norm_count))
        soc_sizes += [dimension + 1] * norm_count
    matrix, rhs = sparse.vstack(blocks, format="csc"), np.concatenate(targets)

    def settings():
        value = clarabel.DefaultSettings()
        value.verbose = False
        value.direct_solve_method = "qdldl"
        value.max_threads = 1
        value.max_iter = 100
        value.tol_gap_abs = value.tol_gap_rel = value.tol_feas = 1e-9
        return value

    # Numerical termination is only a proposal. Exact nonlinear limits and
    # original geometry below decide acceptance, including after backtracking.
    accepted = {"Solved", "AlmostSolved", "InsufficientProgress", "MaxIterations"}
    quadratic = sparse.triu(diagonal @ hessian @ diagonal, format="csc")
    columns_per_frame = joint_variables // limits.frames
    solution = None
    # Same optimum on the few rows that can bind (``working_set``).
    from .working_set import solve as solve_working_set
    solution = solve_working_set(quadratic, gradient * scale, matrix, rhs, cones, settings, diagnostics=diagnostics)
    if solution is None:
        if matrix.shape[0] > 4096:
            return None, "working_set_uncertified"
        # Small programs cost less to solve directly than cold working-set
        # restarts. Large uncertified programs stop instead of spending a
        # second complete solve on a numerically degenerate direction.
        solution = clarabel.DefaultSolver(quadratic, gradient * scale, matrix, rhs, cones, settings()).solve()
    if diagnostics is not None:
        diagnostics.update(variables=int(variables), nonzeros=int(matrix.nnz), complete_rows=int(matrix.shape[0]),
                           iterations=int(solution.iterations))
    status = str(solution.status)
    if status not in accepted or solution.x is None or not np.isfinite(solution.x).all():
        return None, status
    return (projection @ (np.asarray(solution.x[:joint_variables]) * scale[:joint_variables])).reshape(limits.frames, limits.width), status



def _geometry_objective(objective):
    """Freeze a static material registry containing the independent audit's probes."""
    from ..scene_geometry import geom_surface_offsets
    geometry_objective = copy(objective)
    local, owners, original_probes = [], [], []
    for geom in objective.shell_geoms:
        original = objective.shell_local[objective.shell_sample_geom == geom]
        samples = np.unique(np.vstack((original, geom_surface_offsets(objective.model, int(geom), .02),
                                       np.zeros((1, 3)))), axis=0)
        local.append(samples)
        owners.append(np.full(len(samples), geom, dtype=int))
        original_set = set(map(tuple, original))
        original_probes.append(np.array([tuple(point) in original_set for point in samples]))
    geometry_objective.shell_local = np.concatenate(local)
    geometry_objective.shell_sample_geom = np.concatenate(owners)
    geometry_objective._sqp_original_probes = np.concatenate(original_probes)
    geometry_objective._sqp_original_probes.setflags(write=False)
    return geometry_objective

def repair_original_meshes_sqp(objective, poses, scene_objects=(), scene_query=None):
    """Retain only small corrections satisfying immutable nonlinear limits."""
    started = time.monotonic()
    entry = np.array(poses, copy=True)
    if not len(objective.coords.qi):
        return entry, {"status": "no_joint_dofs", "seconds": time.monotonic() - started}
    limits = RepairLimits(objective, entry)
    limits._sqp_chatter = JointChatter(objective.coords, objective.reference)
    # Existing chatter makes the two priorities one coupled joint problem:
    # clear its sparse excess first, then improve the affected collider tails.
    # A motion entering without chatter retains the original group objective.
    limits._sqp_coupled_geometry = bool(np.any(limits._sqp_chatter.excess(entry) > 0.))
    geometry = OriginalGeometry(_geometry_objective(objective), scene_objects, scene_query, manifold=True)
    from .repair_geometry_setup import prepare_static_bounds, prepare_moving_reduction
    prepare_static_bounds(geometry, entry)
    prepare_moving_reduction(geometry, entry)
    geometry.witness_band = WITNESS_BAND
    if geometry.moving is not None:
        geometry.moving.witness_band = WITNESS_BAND
    geometry._sqp_ledger = Ledger()
    geometry._sqp_ledger.setup_seconds = time.monotonic() - started
    # Frame-parallel measurement returns the serial rows; see parallel_geometry.
    geometry.start_workers(len(entry))
    try:
        return _repair(objective, entry, limits, geometry, started)
    finally:
        proposals = getattr(geometry, "_repair_proposals", None)
        if proposals is not None:
            proposals.close(abort=True)
        geometry.stop_workers()


def _repair(objective, entry, limits, geometry, started):
    ledger = geometry._sqp_ledger
    tick = time.monotonic()
    baseline = geometry.measure(entry, jacobians=True)
    ledger.counts["geometry_measures"] += 1
    ledger.entry_measure_seconds = time.monotonic() - tick
    ledger.seconds["geometry"] += ledger.entry_measure_seconds
    current, measured = entry.copy(), baseline
    protected = {"depths": dict(baseline["depths"])}
    model = objective.model
    ordered = _repair_groups(objective, baseline)
    groups = ordered or [None]
    report = {"before": geometry.summary(baseline), "history": [],
              "entry_limits": limits.check(entry)[1],
              "local_angle_budget_rad": getattr(limits, 'local_angle', LOCAL_ANGLE), "root_fixed": True,
              "limb_scopes": [{"fixed_anchor": model.body(scope.anchor).name,
                               "movable_joints": [model.joint(int(model.dof_jntid[limits.coords.di[c]])).name
                                                  for c in sorted(scope.columns)]}
                              for scope in limits.limb_scopes],
              "scheduled_groups": [model.body(body).name for body in ordered],
              "deferred_groups": [], "iterations_per_group": ITERATIONS,
              "scope": "saved-frame original triangles/reverse samples; no collision-free certificate",
              "collision_objective": "per_group_maximum_depth", "static_manifold_max_witnesses": 15,
              "witness_band_m": WITNESS_BAND, "witness_band_cap": WITNESS_BAND_CAP,
              "static_audit_witnesses_per_key": AUDIT_WITNESSES_PER_KEY,
              "static_audit_spacing_m": .02, "static_audit_centers_included": True}
    report["ledger"] = ledger.report()
    if baseline["failures"]:
        report.update(status="unknown_query", after=report["before"], seconds=time.monotonic() - started)
        return current, report
    original_baseline = dict(baseline, depths={k: v for k, v in baseline["depths"].items() if not _audit_key(k)})
    limits.freeze_penetration_yield(original_baseline)
    original_group_count = max(1, len(_repair_groups(objective, baseline, activation=.001)))
    total_limit = ITERATIONS * original_group_count
    report['total_attempt_budget'] = total_limit
    report['activation_depth_m'] = _activation_depth()
    schedule = RepairSchedule(groups, ITERATIONS, total_limit=total_limit)
    proposals = geometry._repair_proposals = SerialProposals(limits, geometry, ledger)
    iteration = -1
    while (scheduled := schedule.next()) is not None:
        group, group_iteration = scheduled
        iteration += 1
        # Preserve baseline history numbering, including unused attempts
        # after an earlier group stopped. Scheduling changes are opt-in.
        iteration = groups.index(group) * ITERATIONS + group_iteration
        transition_group(limits, current, group)
        columns = limits.coords.qi[np.unique(_joint_projection(limits, model, measured).nonzero()[0] % limits.width)]
        scope = scope_for_body(limits, group)
        fixed_q = np.ones(current.shape[1], dtype=bool)
        if scope is not None:
            fixed_q[limits.coords.qi[sorted(scope.columns)]] = False
            if limits._sqp_widened_self:
                fixed_q[columns] = False
        def group_depths():
            for key, value in measured["depths"].items():
                geoms = key[2:] if key[1] == "self" else key[-1:]
                if group is None or any(model.body_weldid[model.geom_bodyid[g]] == group for g in geoms):
                    yield value
        entry_depth = max(group_depths(), default=0.)
        if entry_depth <= _activation_depth():
            continue
        tick = time.monotonic()
        # Only the group's penetrating frames and their halo move.
        from .repair_defects import frames as attempt_frames
        frame_mask = limits.frame_mask = attempt_frames(limits, measured, model, group,
                                                        padding=halo_frames(objective.framerate))
        subproblem = measured
        proposals.prefetch(schedule, current, measured, protected)
        proposal = proposals.take(current, subproblem, protected,
                                  fixed_q if scope is not None else None, entry_depth)
        status = proposal.status
        record = {"iteration": iteration, "group_iteration": group_iteration, "started": tick,
                  "solver_status": status, "alpha": 0.0,
                  "contact_group": None if group is None else model.body(group).name,
                  "limb_only": scope is not None,
                  "expanded_self_collision_scope": bool(limits._sqp_widened_self),
                  "collision_sigma_m": LIMB_COLLISION_SIGMA if scope is not None else COLLISION_SIGMA}
        record["chatter_objective_fallback"] = bool(getattr(proposal, "omit_chatter_objective", False))
        record["cluster"] = [] if group is None else [model.body(group).name]
        report["history"].append(record)
        prediction = getattr(proposal, "prediction", None)
        if prediction is not None:
            record["model_progress"] = dict(prediction)
        if not proposal.has_direction:
            proposal.close()
            proposals.note_outcome(False)
            continue
        if proposal.stationary:
            proposal.close()
            proposals.note_outcome(False)
            record["stationary"] = True
            record["stop_reason"] = "stationary_proposal"
            continue
        from .repair_prediction import should_gate
        if should_gate(prediction, MIN_GROUP_PROGRESS_M):
            proposal.close()
            proposals.note_outcome(False)
            record["model_progress"]["realized_decrease_m"] = 0.
            record["stop_reason"] = "predicted_small_progress"
            continue
        # A pair and its contact compensators form one coupled proposal.
        # Backtracking branches independently can destroy that compensation.
        before = sum(value**2 for value in _physical_depths(measured["depths"]).values())
        first_geometry_candidate = None
        new_witnesses = 0
        with closing(proposal.candidates(record)) as candidates:
            for alpha, candidate in candidates:
                if first_geometry_candidate is None:
                    first_geometry_candidate = candidate.copy()
                record["geometry_coverage"] = dict(
                    changed_frames=np.flatnonzero(np.any(candidate != current, axis=1)).tolist(),
                    requested_frame_count=len(candidate), bodies="all collision geoms")
                # Unmoved frames reuse their exact per-pose cache entries.
                measure_tick = time.monotonic()
                ledger.counts["geometry_measures"] += 1
                checked, regressions, after, descent = measure_candidate(
                    objective, geometry, candidate, measured, protected, group, before,
                    frames=None if frame_mask is None else range(len(candidate)))
                ledger.seconds["geometry"] += time.monotonic() - measure_tick
                if not checked.get("complete", True) or checked["failures"] or regressions or not descent:
                    ledger.counts["rejections"] += 1
                    ledger.counts["geometry_rejections"] += 1
                    reason = ("unknown_query" if checked["failures"] else
                              "per_key_geometry" if regressions else "no_geometry_descent")
                    ledger.rejection_reasons[reason] = ledger.rejection_reasons.get(reason, 0) + 1
                    proposal.feedback = (candidate, checked, regressions)
                    if not checked.get("complete", True):
                        record["rejected_geometry_prefix_frame"] = checked.get("rejection_frame")
                    record["query_failures"] = checked["failures"][:3]
                    record["collision_regressions"] = len(regressions)
                    record["regression_examples"] = [{
                        "key": list(key),
                        "limit_m": protected["depths"].get(key, 0.),
                        "actual_m": checked["depths"][key],
                        "increase": checked["depths"][key] - protected["depths"].get(key, 0.),
                    } for key in regressions[:3]]
                    continue
                if prediction is not None:
                    from .repair_prediction import linear_prediction
                    candidate_prediction = linear_prediction(
                        geometry, subproblem, candidate[:, limits.coords.qi] - current[:, limits.coords.qi],
                        group, entry_depth)
                    if candidate_prediction is not None:
                        record["model_progress"]["predicted_candidate_decrease_m"] = candidate_prediction["predicted_decrease_m"]
                current = candidate
                measured = geometry.linearize(current, checked)
                for key in protected["depths"].keys() | measured["depths"].keys():
                    protected["depths"][key] = min(protected["depths"].get(key, 0.), measured["depths"].get(key, 0.))
                record.update(alpha=alpha, before=before, after=after)
                break
        proposals.note_outcome(bool(record["alpha"]))
        if prediction is not None:
            record["model_progress"]["realized_decrease_m"] = entry_depth - max(group_depths(), default=0.)
            record["model_progress"]["accepted_alpha"] = record["alpha"]
        small = (MIN_GROUP_PROGRESS_M > 0
                 and entry_depth - max(group_depths(), default=0.) < MIN_GROUP_PROGRESS_M)
        if not record["alpha"] or small:
            reason = "no_admissible_descent" if not record["alpha"] else "small_progress"
            record["stop_reason"] = reason
        else:
            schedule.retry(group)
        record["new_witnesses"] = proposal.cut_rows
        if group_iteration == ITERATIONS - 1:
            record["stop_reason"] = "iteration_limit"
    proposals.close()
    report["proposal_workers"] = proposals.stats
    report["geometry_cache"] = dict(frames=getattr(geometry, "window_totals", {}),
                                     jacobians=getattr(geometry, "linearization_totals", {}))
    if getattr(geometry, '_repair_static_bounds', False):
        report['geometry_cache']['static_bounds'] = geometry.static_bounds_stats
    report["deferred_groups"] = []
    report["remembered_witnesses"] = 0
    # Wall time of each proposal: until the next one (or the end) started.
    ends = [r["started"] for r in report["history"][1:]] + [time.monotonic()]
    for record, end in zip(report["history"], ends):
        record["seconds"] = end - record.pop("started")
        record.setdefault("stop_reason", "accepted" if record["alpha"] else record["solver_status"])
    report["iteration_limited_groups"] = [r["contact_group"] for r in report["history"]
                                           if r.get("stop_reason") == "iteration_limit"]
    certified = [r for r in report["history"]
                 if r.get("model_progress", {}).get("model") == "pre_qp_upper_bound"]
    report["triage"] = {"enabled": True, "pre_qp_skips": len(certified),
                        "groups": [{"body": r["contact_group"],
                                    "certificate": r["model_progress"]} for r in certified]}
    gated = [r for r in report["history"]
             if r["solver_status"] == "reduced_predicted_small_progress"]
    report["fast_gate"] = {"enabled": True, "pre_qp_skips": len(gated),
                           "iterations": [r["iteration"] for r in gated]}
    current, measured, cleanup = _cleanup_chatter(
        objective, limits, geometry, current, measured, protected, introduced_only=True)
    report["chatter_cleanup"] = cleanup
    report["chatter_before"] = cleanup["chatter_before"]
    report["chatter_after"] = cleanup["chatter_after"]
    report["history"].extend(cleanup["history"])
    counts = cleanup["counts"]
    ledger.counts["qp"] += counts.get("qp_solves", 0)
    ledger.counts["correction"] += counts.get("correction_solves", 0)
    ledger.counts["geometry_measures"] += (counts.get("geometry_measures", 0)
                                            + counts.get("geometry_remeasures", 0))
    ledger.counts["limit_rejections"] += counts.get("limit_rejections", 0)
    geometry_rejections = sum(counts.get(k, 0) for k in
                              ("geometry_rejections", "unknown_rejections", "no_descent"))
    ledger.counts["geometry_rejections"] += geometry_rejections
    ledger.counts["rejections"] += counts.get("limit_rejections", 0) + geometry_rejections
    ledger.counts["cuts"] += counts.get("cuts", 0)
    for name in ledger.seconds:
        ledger.seconds[name] += cleanup["seconds_by_task"].get(name, 0.)
    ledger.corrections.extend(cleanup.get("corrections", ()))
    # The entry full-domain query does not call measure_window. Include it
    # explicitly rather than returning an empty cache counter on retained clips.
    frame_counts = dict(calls=0, requested_frames=0, cached_frames=0, queried_frames=0,
                        rejected=0, cached_rejections=0)
    frame_counts.update(getattr(geometry, "window_totals", {}))
    frame_counts["calls"] += 1
    frame_counts["requested_frames"] += len(entry)
    frame_counts["queried_frames"] += len(entry)
    report["geometry_cache"].update(frames=frame_counts,
                                    jacobians=getattr(geometry, "linearization_totals", {}))
    report["ledger"] = ledger.report()
    report["corrections"] = ledger.corrections
    report.update(status="corrected" if np.any(current != entry) else "retained",
                  after=geometry.summary(measured), limits=limits.check(current)[1],
                  seconds=time.monotonic() - started)
    return current, report
