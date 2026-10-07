"""Waves of independent local-repair windows, solved together.

Windows of one wave have disjoint movable frames and temporal stencils
(high-pass chatter rows, support differences, sustained layout runs). Each round, every
active window of the wave proposes one model step, its curvature correction
and its nonlinear limit check in a forked worker (the same computation as a
serial call, so the result does not depend on the worker count). The parent
merges all trials, measures the original geometry once, and decides each
window on its own frames: per-key protection, its own collision-depth descent
and chatter. Accepted windows commit together.
"""

import os
import pickle
import signal
import threading
import time
import traceback

import numpy as np

from ..cpu_budget import available_cpus
from .local_chatter import KERNEL_RADIUS
from .mesh_repair import MIN_GROUP_PROGRESS_M, _regressed
from .repair_limits import LOCAL_ANGLE
from . import local_windows as lw


def _fork_map(function, items, parallel):
    """Ordered map with bounded fork workers and unconditional resource cleanup."""
    if not parallel or len(items) < 2 or threading.active_count() != 1 or not hasattr(os, "fork"):
        return [function(item) for item in items]
    from threadpoolctl import threadpool_limits

    cap = max(1, available_cpus() // 2)
    results, children, descriptors = [], set(), set()
    context = "launch"
    try:
        for start in range(0, len(items), cap):
            batch = []
            for index in range(start, min(start + cap, len(items))):
                context = f"launch item {index}"
                read, write = os.pipe()
                descriptors.update((read, write))
                pid = os.fork()
                if pid == 0:
                    exit_code = 1
                    try:
                        for fd in descriptors - {write}:
                            os.close(fd)
                        try:
                            with threadpool_limits(1):
                                value = function(items[index])
                            payload = pickle.dumps(("ok", value), protocol=5)
                        except BaseException:
                            payload = pickle.dumps(("error", traceback.format_exc()), protocol=5)
                        with os.fdopen(write, "wb") as handle:
                            handle.write(payload)
                        exit_code = 0
                    finally:
                        os._exit(exit_code)
                children.add(pid)
                os.close(write)
                descriptors.remove(write)
                batch.append((index, pid, read))
            for index, pid, read in batch:
                context = f"read item {index} (pid {pid})"
                chunks = []
                while True:
                    chunk = os.read(read, 65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                os.close(read)
                descriptors.remove(read)
                _, exit_status = os.waitpid(pid, 0)
                children.remove(pid)
                if not os.WIFEXITED(exit_status) or os.WEXITSTATUS(exit_status) != 0:
                    raise RuntimeError(f"worker exited with status {os.waitstatus_to_exitcode(exit_status)}")
                data = b"".join(chunks)
                if not data:
                    raise RuntimeError("worker returned an empty payload")
                context = f"deserialize item {index} (pid {pid})"
                status, value = pickle.loads(data)
                if status != "ok":
                    raise RuntimeError(f"worker failed:\n{value}")
                results.append(value)
        return results
    except Exception as error:
        raise RuntimeError(f"Local repair {context}: {error}") from error
    finally:
        for fd in descriptors:
            try:
                os.close(fd)
            except OSError:
                pass
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in children:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass


class WindowRun:
    def __init__(self, ctx, window):
        self.window = window
        self.projection = lw.projection(ctx.limits, window)
        self.stencil = ctx.chatter.stencil_frames(window["mask"])
        self.frames = np.flatnonzero(window["mask"])
        self.cuts, self.trust = [], None
        self.cut_rounds = self.shrinks = self.iteration = 0
        self.done = False


def _record(ctx, run, tick):
    model = ctx.objective.model
    window = run.window
    return dict(kind=window["kind"], cluster=[model.body(b).name for b in window["bodies"]],
                iteration=run.iteration, block=window.get("block"), alpha=0.0, started=tick,
                frames=int(window["mask"].sum()), variables=run.projection.shape[1])


def _positive(values):
    return float(np.maximum(values, 0.).sum())


def propose(ctx, run, state):
    """Time this window's own CPU proposal, independently of sibling workers."""
    tick = time.monotonic()
    out = _propose(ctx, run, state, tick)
    out["record"]["elapsed"] = time.monotonic() - tick
    return out


def _propose(ctx, run, state, tick):
    """One model step of one window: QP, curvature correction, limit check.

    Pure CPU work on the inherited state; safe in a forked worker. Returns a
    picklable dict; the parent applies every state change.
    """
    from .local_qp import solve
    from .local_repair import CORRECTIONS, MIN_CHATTER_PROGRESS, _inside_box, correction

    objective, limits, geometry, chatter = ctx.objective, ctx.limits, ctx.geometry, ctx.chatter
    model, qi = objective.model, limits.coords.qi
    window, mask = run.window, run.window["mask"]
    kind, bodies = window["kind"], window["bodies"]
    current, measured, protected = state.current, state.measured, state.protected
    out = dict(record=_record(ctx, run, tick), counts={}, seconds={}, action="stop")
    record = out["record"]
    chatter_before = _positive(chatter.excess(current)[run.stencil])
    if kind == "penetration":
        depths = lw.body_depths(model, measured["depths"], bodies, mask)
        if max(depths.values(), default=0.) <= 0.:
            record["stop_reason"] = "no_penetration"
            return out
        keys, sigmas = lw.objective_keys(model, measured, mask, bodies, limits)
    else:
        if chatter_before <= 0.:
            record["stop_reason"] = "chatter_clear"
            return out
        keys, sigmas = [], []
    matrix, phi, key_of_row = lw.witness_rows(geometry, measured, mask)
    budgets = np.array([protected.get(key, 0.) for key in key_of_row])
    rows = (matrix, phi, budgets, key_of_row)
    chatter_rows = chatter.rows(current, run.stencil, window["columns"], elastic=kind == "chatter")

    def cut_list(origin):
        return [(m, v, b, trial - origin) for m, v, b, trial in run.cuts]

    def count(name, value=1):
        out["counts"][name] = out["counts"].get(name, 0) + value

    def spend(name, since):
        out["seconds"][name] = out["seconds"].get(name, 0.) + time.monotonic() - since

    from .local_slip import slip_term
    solve_tick = time.monotonic()
    count("qp_solves")
    try:
        step, status, info = solve(limits, current, run.projection, rows, keys, sigmas,
                                   local_angle=LOCAL_ANGLE, cuts=cut_list(current[:, qi].ravel()),
                                   trust=run.trust,
                                   linear=[] if kind == "chatter" else [chatter_rows],
                                   elastic=chatter_rows if kind == "chatter" else None,
                                   quadratic=lambda: slip_term(limits, limits.entry_surface_points, mask,
                                                               run.projection))
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        count("qp_failures")
        record.update(stop_reason="qp_error", qp_error=f"{type(error).__name__}: {error}")
        return out
    finally:
        spend("qp", solve_tick)
    record.update(solver_status=status,
                  qp=info and {k: v for k, v in info.items() if k not in ("predicted_depths", "slack")})
    if step is None:
        count("qp_failures")
        record["stop_reason"] = "qp_" + status
        return out
    trial = _inside_box(limits, current, step)
    if kind == "penetration":
        predicted = lw.progress(depths, lw.predicted_depths(model, keys, info["predicted_depths"], bodies))
        enough = predicted >= MIN_GROUP_PROGRESS_M
    else:
        # Chatter rows are exactly linear: the model value is the audit value.
        predicted = chatter_before - _positive(chatter.excess(trial)[run.stencil])
        enough = predicted >= MIN_CHATTER_PROGRESS
    record["predicted_decrease"] = predicted
    if not enough or np.max(np.abs(step), initial=0.) < 1e-9:
        record["stop_reason"] = "predicted_small_progress"
        return out
    tick_limits = time.monotonic()
    valid, check = limits.check(trial)
    spend("limits", tick_limits)
    corrections, correction_norm = 0, 0.
    radius = 0.5 * float(np.linalg.norm(step))
    log = []

    class _Log:
        corrections = log

    while not valid and corrections < CORRECTIONS:
        corrections += 1
        count("limit_rejections")
        solve_tick = time.monotonic()
        delta = correction(limits, current, trial, run.projection, rows, cut_list(trial[:, qi].ravel()),
                           [chatter.rows(trial, run.stencil, window["columns"])], radius, _Log)
        spend("correction", solve_tick)
        count("correction_solves")
        if delta is None:
            break
        correction_norm += float(np.linalg.norm(delta))
        trial = _inside_box(limits, trial, delta)
        tick_limits = time.monotonic()
        valid, check = limits.check(trial)
        spend("limits", tick_limits)
    record.update(corrections=corrections, step_norm=float(np.linalg.norm(step)),
                  correction_norm=correction_norm)
    out["corrections"] = log
    if not valid:
        count("limit_rejections")
        record["rejected_limits"] = {k: v for k, v in check.items() if k in ("reason", "guard")}
        out.update(action="rejected_limits", step_max=float(np.max(np.abs(step), initial=0.)))
        return out
    out.update(action="trial", trial=trial[np.ix_(run.frames, qi)].copy(), keys=keys,
               step_max=float(np.max(np.abs(step), initial=0.)))
    return out


def _merge_counts(ledger, out):
    for name, value in out["counts"].items():
        ledger.counts[name] = ledger.counts.get(name, 0) + value
    for name, value in out["seconds"].items():
        ledger.seconds[name] = ledger.seconds.get(name, 0.) + value
    ledger.corrections.extend(out.get("corrections", ()))


def run_wave(ctx, windows, state, *, parallel=True):
    """Grant a shared per-frame permission snapshot for the whole wave."""
    ctx.limits.grant(windows)
    try:
        _run_wave(ctx, windows, state, parallel=parallel)
    finally:
        ctx.limits.release(state.current)


def _run_wave(ctx, windows, state, *, parallel):
    """Solve one wave of mutually independent windows to completion."""
    from .local_repair import CUT_ROUNDS, ITERATIONS, MAX_SHRINKS, MIN_CHATTER_PROGRESS, TRUST_SHRINK
    from .repair_linearize import linearize

    objective, limits, geometry, chatter, ledger = ctx.objective, ctx.limits, ctx.geometry, ctx.chatter, ctx.ledger
    model, qi = objective.model, limits.coords.qi
    runs = [WindowRun(ctx, window) for window in windows]
    ledger.counts["waves"] = ledger.counts.get("waves", 0) + 1
    while True:
        active = [run for run in runs if not run.done and run.iteration < ITERATIONS]
        if not active:
            break
        round_tick = time.monotonic()
        try:
            ledger.counts["rounds"] = ledger.counts.get("rounds", 0) + 1
            outputs = _fork_map(lambda run: propose(ctx, run, state), active, parallel)
            trials = []
            for run, out in zip(active, outputs):
                _merge_counts(ledger, out)
                ctx.history.append(out["record"])
                run.iteration += 1
                if out["action"] == "stop":
                    run.done = True
                elif out["action"] == "rejected_limits":
                    if run.shrinks >= MAX_SHRINKS:
                        out["record"]["stop_reason"] = "limits_rejected"
                        run.done = True
                    else:
                        run.shrinks += 1
                        run.trust = TRUST_SHRINK * out["step_max"]
                        ledger.counts["trust_shrinks"] = ledger.counts.get("trust_shrinks", 0) + 1
                else:
                    trials.append((run, out))
            if not trials:
                continue
            current, measured, protected = state.current, state.measured, state.protected
            merged = current.copy()
            for run, out in trials:
                merged[np.ix_(run.frames, qi)] = out["trial"]
            tick = time.monotonic()
            checked = geometry.measure(merged, frames=range(len(merged)))
            ledger.seconds["geometry"] += time.monotonic() - tick
            ledger.counts["geometry_measures"] += 1
            # Unknown original queries reject only the windows owning those frames.
            unreliable = np.zeros(len(merged), bool)
            if checked["failures"] or not checked.get("complete", True):
                depth_table = checked.get("max_depth")
                if depth_table is None or not checked.get("complete", True):
                    unreliable[:] = True
                else:
                    unreliable[np.asarray(checked["queried_frames"])] = np.isnan(depth_table).any(axis=1)
            accepted = []
            for run, out in trials:
                mask, record = run.window["mask"], out["record"]
                if unreliable[mask].any():
                    record["query_failures"] = checked["failures"][:3]
                    ledger.counts["unknown_rejections"] = ledger.counts.get("unknown_rejections", 0) + 1
                    if run.shrinks >= MAX_SHRINKS:
                        record["stop_reason"] = "unknown_query"
                        run.done = True
                    else:
                        run.shrinks += 1
                        run.trust = TRUST_SHRINK * out["step_max"]
                        ledger.counts["trust_shrinks"] = ledger.counts.get("trust_shrinks", 0) + 1
                    continue
                regressed = [key for key, value in checked["depths"].items()
                             if mask[key[0]] and _regressed(value, protected.get(key, 0.))]
                before = sum(v**2 for k, v in measured["depths"].items() if mask[k[0]])
                after = sum(v**2 for k, v in checked["depths"].items() if mask[k[0]])
                if regressed:
                    ledger.counts["geometry_rejections"] += 1
                    ledger.counts["regressions"] += len(regressed)
                    record["collision_regressions"] = len(regressed)
                    record["regression_examples"] = [
                        dict(key=list(k), limit_m=protected.get(k, 0.), actual_m=checked["depths"][k],
                             increase=checked["depths"][k] - protected.get(k, 0.)) for k in regressed[:3]]
                    if run.cut_rounds >= CUT_ROUNDS:
                        record["stop_reason"] = "geometry_rejected"
                        run.done = True
                        continue
                    selected = set(regressed)
                    tick = time.monotonic()
                    subset = linearize(geometry, merged, dict(rows=[(k, r) for k, r in checked["rows"]
                                                                  if k in selected]))
                    ledger.seconds["linearize"] += time.monotonic() - tick
                    if not subset["rows"]:
                        record["stop_reason"] = "geometry_rejected"
                        run.done = True
                        continue
                    cut_matrix, cut_phi = geometry.linear_rows(subset)
                    cut_budget = np.array([protected.get(k, 0.) for k, _ in subset["rows"]])
                    trial_pose = current.copy()
                    trial_pose[np.ix_(run.frames, qi)] = out["trial"]
                    run.cuts.append((cut_matrix, cut_phi, cut_budget, trial_pose[:, qi].ravel().copy()))
                    ledger.counts["cuts"] += len(cut_phi)
                    run.cut_rounds += 1
                    continue
                alone = current.copy()
                alone[np.ix_(run.frames, qi)] = out["trial"]
                excess_before = np.maximum(chatter.excess(current)[run.stencil], 0.)
                excess_after = np.maximum(chatter.excess(alone)[run.stencil], 0.)
                elementwise = bool(np.all(excess_after <= excess_before + 1e-9))
                chatter_before, chatter_after = float(excess_before.sum()), float(excess_after.sum())
                if run.window["kind"] == "penetration":
                    descent = elementwise and after < before - 1e-10
                else:
                    descent = (elementwise and after <= before
                               and chatter_before - chatter_after >= MIN_CHATTER_PROGRESS)
                if not descent:
                    ledger.counts["no_descent"] += 1
                    record["stop_reason"] = "no_descent"
                    run.done = True
                    continue
                record.update(alpha=1.0, before=before, after=after,
                              chatter_before=chatter_before, chatter_after=chatter_after)
                accepted.append((run, out))
            if not accepted:
                continue
            updated = current.copy()
            for run, out in accepted:
                updated[np.ix_(run.frames, qi)] = out["trial"]
            tick = time.monotonic()
            remeasured = geometry.measure(updated, frames=range(len(updated)))
            ledger.counts["geometry_remeasures"] += 1
            ledger.seconds["geometry"] += time.monotonic() - tick
            tick = time.monotonic()
            state.measured = geometry.linearize(updated, remeasured)
            ledger.seconds["linearize"] += time.monotonic() - tick
            state.current = updated
            for key in protected.keys() | state.measured["depths"].keys():
                protected[key] = min(protected.get(key, 0.), state.measured["depths"].get(key, 0.))
            for run, out in accepted:
                ledger.counts["accepted"] += 1
                window, mask, record = run.window, run.window["mask"], out["record"]
                if window["kind"] == "penetration":
                    before_depths = lw.body_depths(model, measured["depths"], window["bodies"], mask)
                    realized = lw.progress(before_depths, lw.body_depths(model, state.measured["depths"],
                                                                         window["bodies"], mask))
                    small = realized < MIN_GROUP_PROGRESS_M
                else:
                    realized = record["chatter_before"] - record["chatter_after"]
                    small = realized < MIN_CHATTER_PROGRESS
                record["realized_decrease"] = realized
                run.cuts, run.trust, run.cut_rounds, run.shrinks = [], None, 0, 0
                if small:
                    record["stop_reason"] = "small_progress"
                    run.done = True
        finally:
            ledger.seconds["round_seconds"] += time.monotonic() - round_tick


def waves(windows):
    """Schedule conflicts in input order; independent windows share a wave."""
    result, placed = [], []
    reach = 2 * KERNEL_RADIUS + 2
    for window in windows:
        mask = window["mask"]
        # Expand only one mask: frame distance <= reach is a conflict.
        expanded = lw._dilate(mask, reach)
        conflicts = [wave for previous, wave in placed if (expanded & previous["mask"]).any()]
        wave = 1 + max(conflicts) if conflicts else 0
        while len(result) <= wave:
            result.append([])
        result[wave].append(window)
        placed.append((window, wave))
    return result
