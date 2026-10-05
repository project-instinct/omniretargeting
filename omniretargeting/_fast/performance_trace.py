"""Non-overlapping performance records for the fast sequence backend.

The trace is diagnostic only.  It never changes a guard, budget, objective or
acceptance decision.  Stage timers are exclusive by construction: setup starts
only after source fitting, and each later solver/audit phase is timed around
its own call.  Child-window records are returned to the parent and merged, so
parallel workers do not have to share mutable state while solving.
"""

from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from functools import wraps
import importlib
import sys
from unittest.mock import patch
import time

import numpy as np


_ACTIVE_TRACE = ContextVar("fast_performance_trace", default=None)
PROPOSAL_PHASES = (
    "constraint_row_construction", "canonicalization", "solver_setup",
    "numerical_solve", "backtracking", "restoration",
)
COUNTER_SEMANTICS = {
    "objective_jacobian_evaluations": {
        "included_in": "objective_evaluations",
        "additive": False,
        "description": "Same call and seconds when with_jacobian=True; never sum these counters.",
    },
    "constraint_proposals": {
        "additive": False,
        "description": "Inclusive proposal elapsed time; do not sum with exclusive proposal phases.",
    },
    "recoveries": {
        "additive": False,
        "description": "Legacy inclusive recovery blocks; actual restoration calls are recorded separately.",
    },
    "proposal_phases": {
        "exclusive": list(PROPOSAL_PHASES),
        "description": "Nested phase time is subtracted from its parent; restoration excludes construction and solves.",
    },
}


def timed_phase(name, *, call_kind=None, call_name=None, outer_model=False, auto_proposal=False):
    """Instrument a callable only inside a live trace context."""
    def decorate(function):
        @wraps(function)
        def timed(*args, **kwargs):
            trace = _ACTIVE_TRACE.get()
            if trace is None or not trace.enabled:
                return function(*args, **kwargs)
            if outer_model and trace._automatic_iterations and trace._iteration is None:
                start = trace._phase_start
                trace.begin_iteration(start["label"], len(trace.iterations) - start["iteration_start"])
            if auto_proposal and trace._automatic_iterations:
                trace.record_call("proposal", call_name or function.__name__)
            if call_kind:
                trace.record_call(call_kind, call_name or function.__name__)
            with trace.operation(name):
                try:
                    result = function(*args, **kwargs)
                except Exception:
                    if auto_proposal and trace._automatic_iterations and call_name == "equilibrated_solve":
                        trace.banded_alpha(None)
                    raise
                if auto_proposal and trace._automatic_iterations and call_name == "equilibrated_solve":
                    trace.banded_alpha(1.)
                return result
        return timed
    return decorate


STAGES = (
    "source_pose", "grid_setup", "contact_acquisition", "reweight",
    "free_bodies", "original_mesh_repair", "final_audit",
)
ALPHA_BUCKETS = (
    ("1", lambda value: value >= 1.0 - 1e-12),
    (".5-.999", lambda value: value >= .5),
    (".25-.499", lambda value: value >= .25),
    (".125-.249", lambda value: value >= .125),
    ("0-.124", lambda value: value >= 0.0),
)


def _zero():
    return {"calls": 0, "seconds": 0.0}


def _copy_counters(value):
    return {
        stage: {name: dict(row) for name, row in rows.items()}
        for stage, rows in value.items()
    }


def _merge_counters(target, source):
    for stage, rows in source.items():
        into = target.setdefault(stage, {})
        for name, row in rows.items():
            current = into.setdefault(name, _zero())
            current["calls"] += int(row.get("calls", 0))
            current["seconds"] += float(row.get("seconds", 0.0))


def _diff_counters(before, after):
    result = {}
    for stage, rows in after.items():
        old = before.get(stage, {})
        result[stage] = {
            name: {
                "calls": int(row["calls"] - old.get(name, _zero())["calls"]),
                "seconds": float(row["seconds"] - old.get(name, _zero())["seconds"]),
            }
            for name, row in rows.items()
        }
    return result


def guard_name(guard):
    return type(guard).__name__


def explain_guard_rows(objective, guard, poses):
    """Best-effort exact rejection record; never replaces the boolean guard.

    Temporal and acquisition guards expose their frozen limit, actual value and
    excess through the existing diagnostic contracts.  Other guards at least
    report the violating frame interval, rather than hiding behind a generic
    rejection count.
    """
    frame_offset = 0
    window = getattr(guard, "window", None)
    if window is not None:
        frame_offset = 0  # Embedded diagnostics already use global frame indices.
        poses = window.embed(poses)
        objective, guard = window.global_objective, guard.guard

    name = guard_name(guard)
    result = {
        "guard": name,
        "frames": [frame_offset, frame_offset + len(poses)],
        "joint_or_contact": None,
        "limit": None,
        "actual": None,
        "excess": None,
    }
    try:
        from .coarse_diagnostics import failure_details
        from .proposals import guards_of

        guards = guards_of(objective)[1]
        index = guards.index(guard) if guard in guards else 0
        key = f"{name}:{index}"
        details = failure_details(objective, poses, {key: True}).get(key)
        if details and not (isinstance(details, dict) and
                            ("worst_constraint" in details or "groups" in details)):
            if isinstance(details, dict):
                details = [details]
            rows = []
            for row in details:
                frame = row.get("frame", row.get("stencil_start"))
                if frame is not None:
                    frame = int(frame) + frame_offset
                rows.append({
                    "guard": name,
                    "frame": frame,
                    "channel": row.get("channel"),
                    "coordinate": row.get("coordinate"),
                    "joint_or_contact": row.get("joint", row.get("group", row.get("channel"))),
                    "limit": row.get("frozen_bound", row.get("bound")),
                    "actual": row.get("value"),
                    "excess": row.get("excess", row.get("excess_m", row.get("max_excess_m"))),
                })
            return rows if rows else [result]
    except (AttributeError, ValueError, TypeError):
        pass

    diagnostics = getattr(guard, "diagnostics", None)
    report, violations = None, getattr(guard, "violation_frames", None)
    if callable(diagnostics):
        try:
            report = diagnostics(poses)
            if isinstance(report, dict) and report.get("binding_rows"):
                return [dict(result, **row) for row in report["binding_rows"]]
            worst = report.get("worst_constraint") if isinstance(report, dict) else None
            if isinstance(worst, dict):
                result.update({
                    "joint_or_contact": worst.get("group"),
                    "limit": 0.0,
                    "actual": worst.get("excess_m"),
                    "excess": worst.get("excess_m"),
                })
                if worst.get("frame") is not None:
                    frame = frame_offset + int(worst["frame"])
                    result["frames"] = [frame, frame + 1]
                return [result]
        except (AttributeError, ValueError, TypeError):
            pass

        if isinstance(report, dict) and report.get("groups"):
            groups = [group for group in report["groups"]
                      if float(group.get("max_excess", 0.0)) > 0.0]
            if groups:
                rows = []
                for group in groups:
                    excess = float(group.get("max_excess", 0.0))
                    actual = float(group.get("max_regression", excess))
                    frames = group.get("violating_frames") or report.get("violating_frames") or []
                    row = dict(result, joint_or_contact=group.get("name"),
                               limit=max(0.0, actual - excess), actual=actual, excess=excess)
                    if frames:
                        row["frames"] = [frame_offset + int(frames[0]),
                                         frame_offset + int(frames[-1]) + 1]
                    rows.append(row)
                return rows

    if callable(violations):
        try:
            bad = np.flatnonzero(np.asarray(violations(poses)))
            if len(bad):
                result["frames"] = [frame_offset + int(bad[0]),
                                    frame_offset + int(bad[-1]) + 1]
        except (AttributeError, ValueError, TypeError):
            pass
    return [result]


def explain_guard(objective, guard, poses):
    """Compatibility helper returning the first available binding row."""
    return explain_guard_rows(objective, guard, poses)[0]


class PerformanceTrace:
    """Collect exclusive stage timings and solver diagnostics.

    ``install`` wraps only the objective's evaluation method on that instance.
    Solver proposals and guards are counted by the solver while their objects
    are live, then restored even on an exception.
    """

    def __init__(self, enabled=True, *, stage=None, frames=(0, None)):
        self.enabled = bool(enabled)
        self.current_stage = stage
        self.frames = frames
        self.stage_seconds = {name: 0.0 for name in STAGES}
        self.operations = {}
        self.guard_checks = {}
        self.guard_rejections = {}
        self.alphas = []
        self.failed = {"count": 0, "seconds": 0.0, "reasons": {}}
        self.stage_alphas = {}
        self.stage_failed = {}
        self.stage_stop_reasons = {}
        self.stage_details = {}
        self.slow_windows = []
        self.stop_reasons = {}
        self._phase_start = None
        self.iterations = []
        self._iteration = None
        self._operation_stack = []
        self._automatic_iterations = False

    @contextmanager
    def operation(self, name):
        """Exclusive wall time even for nested construction/solve/restoration."""
        if not self.enabled:
            yield
            return
        entry = [name, time.monotonic(), 0.0]
        self._operation_stack.append(entry)
        try:
            yield
        finally:
            elapsed = time.monotonic() - entry[1]
            self._operation_stack.pop()
            if self._operation_stack:
                self._operation_stack[-1][2] += elapsed
            self.add_operation(name, max(0.0, elapsed - entry[2]))

    def begin_iteration(self, label, iteration):
        self._iteration = {
            "phase": label, "iteration": int(iteration), "stage": self._stage(),
            "frames": list(self.frames), "banded_feasible_alpha": [],
            "accepted_step_backend": None, "fallback_reason": [],
            "proposal_calls": {}, "restore_calls": {}, "binding_rows": [],
        }
        self._iteration_operations = _copy_counters(self.operations)

    def record_call(self, kind, name):
        self.add_operation(kind + "_" + name, 0.0)
        if self._iteration is not None:
            calls = self._iteration[kind + "_calls"]
            calls[name] = calls.get(name, 0) + 1

    def banded_alpha(self, value):
        if self._iteration is not None:
            self._iteration["banded_feasible_alpha"].append(value)

    def temporal_direction(self, diagnostics, seconds, *, accepted):
        """Compact per-attempt record of the opt-in banded temporal direction."""
        keys = ("fallback_reason", "feasible_alpha", "ip_iterations", "ip_status", "domain_rows",
                "temporal_rows", "temporal_cones", "linear_excess", "cone_excess", "ratio", "error",
                "rejecting_guards")
        record = {key: diagnostics[key] for key in keys if key in diagnostics}
        record.update(accepted=bool(accepted), seconds=float(seconds))
        if self._iteration is not None:
            self._iteration.setdefault("banded_temporal", []).append(record)
        attempts = self.stage_details.setdefault(self._stage(), {}).setdefault(
            "banded_temporal_directions", {"accepted": 0, "fallbacks": {}, "seconds": 0.0})
        attempts["seconds"] += float(seconds)
        if accepted:
            attempts["accepted"] += 1
        else:
            reason = record.get("fallback_reason", "unknown")
            attempts["fallbacks"][reason] = attempts["fallbacks"].get(reason, 0) + 1

    def fallback(self, reason):
        if self._iteration is not None:
            self._iteration["fallback_reason"].append(reason)

    def finish_iteration(self, history_row, backend):
        if self._iteration is None:
            return
        record = self._iteration
        record.update(accepted_step_backend=backend, alpha=float(history_row.get("alpha", 0.)),
                      operations=_diff_counters(self._iteration_operations, self.operations))
        record["proposal_call_count"] = sum(record["proposal_calls"].values())
        record["restore_call_count"] = sum(record["restore_calls"].values())
        self.iterations.append(record)
        history_row["iteration_performance"] = record
        self._iteration = None

    def add_binding_rows(self, rows):
        if self._iteration is not None:
            for row in rows:
                if row not in self._iteration["binding_rows"]:
                    self._iteration["binding_rows"].append(dict(row))

    @contextmanager
    def instrument(self, *, automatic_iterations=False):
        """Scoped diagnostic hooks; original callables/results survive intact.

        Patch aliases too: legacy windows and restoration import their solve
        helpers directly. A ContextVar confines collection to the solving thread.
        """
        import clarabel
        import cvxpy as cp
        token = _ACTIVE_TRACE.set(self)
        previous_automatic = self._automatic_iterations
        self._automatic_iterations = automatic_iterations
        imported_aliases = {}
        try:
            with ExitStack() as stack:
                specs = {
                    "proposals": {"backtrack": ("backtracking", None)},
                    "solver": {"backtrack": ("backtracking", None),
                               "_shorten_conic": ("backtracking", None)},
                    "shape_step": {"bounded_shape_step": ("constraint_row_construction", None),
                                   "restore_shape_step": ("restoration", "restore")},
                    "window_canonical": {"canonical_data": ("canonicalization", None)},
                    "local_restoration": {"backtrack_with_restoration": ("backtracking", None),
                                          "restore_local": ("restoration", "restore"),
                                          "_bounds": ("constraint_row_construction", None),
                                          "_program": ("canonicalization", None)},
                    "proposal_curvature": {"restore_curvature": ("restoration", "restore")},
                    "qp_assembly": {"prepare_model": ("constraint_row_construction", None),
                                    "assemble_program": ("canonicalization", None),
                                    "solve_model": ("solver_setup", None)},
                    "qp_types": {"solve_program": ("solver_setup", None)},
                    "barrier_assembly": {"physical_system": ("constraint_row_construction", None),
                                         "equilibrated_solve": ("banded_solves", None)},
                    "qp_reduction": {"solve_restricted": ("solver_setup", None)},
                    "temporal_direction": {"_temporal_rows": ("constraint_row_construction", None)},
                    "constrained_ip": {"solve": ("numerical_solve", None)},
                }
                replacements = {}
                for module_name, functions in specs.items():
                    module = importlib.import_module("omniretargeting._fast." + module_name)
                    for name, (phase, kind) in functions.items():
                        original = getattr(module, name)
                        replacements[id(original)] = timed_phase(
                            phase, call_kind=kind, call_name=name,
                            outer_model=name in {"prepare_model", "physical_system"},
                            auto_proposal=name in {"solve_model", "solve_restricted", "equilibrated_solve"})(original)
                from .. import conic_solver
                for name in ("solve_conic", "solve_conic_working_set"):
                    original = getattr(conic_solver, name)
                    replacements[id(original)] = timed_phase("solver_setup")(original)
                from .shape_step import _ShapeRestoration
                stack.enter_context(patch.object(_ShapeRestoration, "restoration_constraints",
                    timed_phase("constraint_row_construction")(_ShapeRestoration.restoration_constraints)))
                from .constrained_ip import QuadraticDomain
                stack.enter_context(patch.object(QuadraticDomain, "__init__",
                    timed_phase("constraint_row_construction")(QuadraticDomain.__init__)))
                originals = {id(value): value for module_name, module in list(sys.modules.items())
                             if module is not None and module_name.startswith("omniretargeting.")
                             for value in list(vars(module).values()) if callable(value)}
                imported_aliases = {id(wrapped): originals[key] for key, wrapped in replacements.items()}
                # Imported aliases refer to the same original function object.
                for module_name, module in list(sys.modules.items()):
                    if module is None or not module_name.startswith("omniretargeting."):
                        continue
                    for name, original in list(vars(module).items()):
                        if callable(original):
                            replacement = replacements.get(id(original))
                            if replacement is not None:
                                stack.enter_context(patch.object(module, name, replacement))
                stack.enter_context(patch.object(cp.Problem, "get_problem_data",
                    timed_phase("canonicalization")(cp.Problem.get_problem_data)))
                constructor = clarabel.DefaultSolver
                stack.enter_context(patch.object(clarabel, "DefaultSolver",
                    timed_phase("solver_setup")(constructor)))
                for name, phase in (("solve", "numerical_solve"), ("update", "solver_setup")):
                    stack.enter_context(patch.object(constructor, name,
                        timed_phase(phase)(getattr(constructor, name))))
                from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL
                stack.enter_context(patch.object(CLARABEL, "solve_via_data",
                    timed_phase("solver_setup")(CLARABEL.solve_via_data)))
                try:
                    import qoco
                except ImportError:
                    pass
                else:
                    for name, phase in (("setup", "solver_setup"), ("solve", "numerical_solve")):
                        stack.enter_context(patch.object(qoco.QOCO, name,
                            timed_phase(phase)(getattr(qoco.QOCO, name))))
                yield
        finally:
            # A module first imported inside the context can retain a wrapped
            # helper. Restore those aliases as well as the pre-existing ones.
            for module_name, module in list(sys.modules.items()):
                if module is None or not module_name.startswith("omniretargeting."):
                    continue
                for name, value in list(vars(module).items()):
                    original = imported_aliases.get(id(value))
                    if original is not None:
                        setattr(module, name, original)
            self._automatic_iterations = previous_automatic
            _ACTIVE_TRACE.reset(token)

    def _stage(self, name=None):
        return name or self.current_stage or "unattributed"

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        previous = self.current_stage
        self.current_stage = name
        tick = time.monotonic()
        try:
            yield
        finally:
            self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + time.monotonic() - tick
            self.current_stage = previous

    def add_stage_seconds(self, name, seconds):
        self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + float(seconds)

    def add_stage_detail(self, stage, key, value):
        if self.enabled:
            self.stage_details.setdefault(stage, {})[key] = value

    def add_operation(self, name, seconds, *, calls=1):
        if not self.enabled:
            return
        stage = self._stage()
        row = self.operations.setdefault(stage, {}).setdefault(name, _zero())
        row["calls"] += int(calls)
        row["seconds"] += float(seconds)

    def add_evaluation(self, seconds, *, with_jacobian):
        self.add_operation("objective_evaluations", seconds)
        if with_jacobian:
            self.add_operation("objective_jacobian_evaluations", seconds)

    def add_guard(self, guard, accepted, seconds, explanation=None):
        if not self.enabled:
            return
        name = guard_name(guard)
        row = self.guard_checks.setdefault(self._stage(), {}).setdefault(name, _zero())
        row["calls"] += 1
        row["seconds"] += float(seconds)
        if not accepted:
            rejects = self.guard_rejections.setdefault(self._stage(), {}).setdefault(name, 0)
            rejects += 1
            self.guard_rejections[self._stage()][name] = rejects
            if explanation is not None:
                rows = explanation if isinstance(explanation, list) else [explanation]
                self.add_binding_rows(rows)
                for row in rows:
                    self.add_slow_window(row, seconds)

    def add_alpha(self, alpha):
        if self.enabled and alpha > 0.0:
            value = float(alpha)
            self.alphas.append(value)
            self.stage_alphas.setdefault(self._stage(), []).append(value)

    def add_failed(self, seconds, reason):
        if not self.enabled:
            return
        self.failed["count"] += 1
        self.failed["seconds"] += float(seconds)
        self.failed["reasons"][reason] = self.failed["reasons"].get(reason, 0) + 1
        row = self.stage_failed.setdefault(self._stage(), {"count": 0, "seconds": 0.0, "reasons": {}})
        row["count"] += 1
        row["seconds"] += float(seconds)
        row["reasons"][reason] = row["reasons"].get(reason, 0) + 1

    def add_slow_window(self, row, seconds):
        if not self.enabled:
            return
        record = dict(row)
        record["stage"] = record.get("stage", self._stage())
        record.setdefault("stop_reason", "guard_rejected")
        frame = record.get("frame")
        if frame is not None and record.get("frames") is None:
            frame = int(frame)
            record["frames"] = [frame, frame + 1]
        key = (record["stage"], record.get("guard"), tuple(record.get("frames") or ()),
               record.get("joint_or_contact"))
        for existing in self.slow_windows:
            if (existing["stage"], existing.get("guard"), tuple(existing.get("frames") or ()),
                    existing.get("joint_or_contact")) == key:
                existing["seconds"] = float(existing.get("seconds", 0.0)) + float(seconds)
                existing["rejection_count"] = int(existing.get("rejection_count", 0)) + 1
                for field in ("limit", "actual", "excess"):
                    first, value = existing.get(field), record.get(field)
                    if first is None or (value is not None and value > first):
                        existing[field] = value
                if record.get("stop_reason") is not None:
                    existing["stop_reason"] = record["stop_reason"]
                return
        record.update(seconds=float(record.get("seconds", seconds)), rejection_count=1)
        self.slow_windows.append(record)
        self.slow_windows.sort(key=lambda value: (-value.get("seconds", 0.0), str(value.get("guard"))))
        del self.slow_windows[12:]

    def install(self, objective):
        if not self.enabled:
            return lambda: None
        if getattr(objective, "_performance_trace_installed", False):
            objective.performance_trace = self
            return lambda: None
        objective.performance_trace = self
        original = objective.evaluate

        def evaluate(*args, **kwargs):
            tick = time.monotonic()
            try:
                return original(*args, **kwargs)
            finally:
                with_jacobian = kwargs.get(
                    "with_jacobian", args[1] if len(args) > 1 else True)
                self.add_evaluation(time.monotonic() - tick,
                                    with_jacobian=bool(with_jacobian))

        objective.evaluate = evaluate
        objective._performance_trace_installed = True

        def restore():
            objective.__dict__.pop("_performance_trace_installed", None)
            objective.__dict__.pop("performance_trace", None)
            objective.__dict__.pop("evaluate", None)

        return restore

    def begin_phase(self, label):
        if not self.enabled:
            return
        self._phase_start = {
            "label": label,
            "iteration_start": len(self.iterations),
            "operations": _copy_counters(self.operations),
            "guards": _copy_counters(self.guard_checks),
            "alphas": list(self.alphas),
            "failed": dict(self.failed, reasons=dict(self.failed["reasons"])),
        }

    def finish_phase(self, stop_reason):
        if not self.enabled or self._phase_start is None:
            return None
        start = self._phase_start
        self._phase_start = None
        reason = str(stop_reason)
        self.stop_reasons[reason] = self.stop_reasons.get(reason, 0) + 1
        stage_reasons = self.stage_stop_reasons.setdefault(self._stage(), {})
        stage_reasons[reason] = stage_reasons.get(reason, 0) + 1
        return {
            "phase": start["label"],
            "counter_semantics": COUNTER_SEMANTICS,
            "iterations": self.iterations[start["iteration_start"]:],
            "stage": self._stage(),
            "frames": list(self.frames),
            "stop_reason": reason,
            "operations": _diff_counters(start["operations"], self.operations),
            "guard_checks": _diff_counters(start["guards"], self.guard_checks),
            "alphas": self.alphas[len(start["alphas"]):],
            "slow_windows": [row for row in self.slow_windows
                             if row.get("stage") == self._stage()],
            "accepted_step_count": len(self.alphas) - len(start["alphas"]),
            "failed_candidates": {
                "count": self.failed["count"] - start["failed"]["count"],
                "seconds": self.failed["seconds"] - start["failed"]["seconds"],
                "reasons": {
                    key: value - start["failed"]["reasons"].get(key, 0)
                    for key, value in self.failed["reasons"].items()
                    if value > start["failed"]["reasons"].get(key, 0)
                },
            },
        }

    def absorb(self, report):
        if not self.enabled or not isinstance(report, dict):
            return
        self.iterations.extend(report.get("iterations", ()))
        _merge_counters(self.operations, report.get("operations", {}))
        for stage, rows in report.get("guard_checks", {}).items():
            into = self.guard_checks.setdefault(stage, {})
            for name, row in rows.items():
                current = into.setdefault(name, _zero())
                current["calls"] += int(row.get("calls", 0))
                current["seconds"] += float(row.get("seconds", 0.0))
        stage = report.get("stage", self._stage())
        previous_stage = self.current_stage
        self.current_stage = stage
        try:
            for alpha in report.get("alphas", ()):
                self.add_alpha(alpha)
        finally:
            self.current_stage = previous_stage
        failed = report.get("failed_candidates")
        if failed:
            self.failed["count"] += int(failed.get("count", 0))
            self.failed["seconds"] += float(failed.get("seconds", 0.0))
            row = self.stage_failed.setdefault(stage, {"count": 0, "seconds": 0.0, "reasons": {}})
            row["count"] += int(failed.get("count", 0))
            row["seconds"] += float(failed.get("seconds", 0.0))
            for reason, count in failed.get("reasons", {}).items():
                self.failed["reasons"][reason] = (
                    self.failed["reasons"].get(reason, 0) + int(count))
                row["reasons"][reason] = row["reasons"].get(reason, 0) + int(count)
        reason = report.get("stop_reason")
        if reason is not None:
            self.stop_reasons[reason] = self.stop_reasons.get(reason, 0) + 1
            stage_reasons = self.stage_stop_reasons.setdefault(stage, {})
            stage_reasons[reason] = stage_reasons.get(reason, 0) + 1
        for row in report.get("slow_windows", ()):
            self.add_slow_window(row, row.get("seconds", 0.0))

    def absorb_history(self, history):
        for row in history if isinstance(history, list) else ():
            self.absorb(row.get("window_performance") or row.get("solver_performance"))

    def absorb_repair(self, report):
        """Merge the original-mesh repair ledger into the requested counters."""
        if not self.enabled or not isinstance(report, dict):
            return
        for row in report.get("history", ()):
            seconds = float(row.get("seconds", 0.0))
            self.add_operation("constraint_proposals", seconds)
            alpha = float(row.get("alpha", 0.0))
            if alpha > 0.0:
                self.add_alpha(alpha)
            else:
                reason = row.get("stop_reason", row.get("solver_status", "repair_rejected"))
                self.add_failed(seconds, str(reason))
            examples = row.get("regression_examples") or []
            if examples:
                worst = max(examples, key=lambda value: float(value.get("increase", 0.0)))
                key = worst.get("key", ())
                frame = (int(key[2]) if len(key) > 2 and str(key[0]) == "field"
                         else int(key[1]) if len(key) > 1 and isinstance(key[1], int)
                         else None)
                self.add_slow_window({
                    "guard": "protected_original_depth",
                    "frames": None if frame is None else [frame, frame + 1],
                    "joint_or_contact": worst.get("key"),
                    "limit": worst.get("limit_m"),
                    "actual": worst.get("actual_m"),
                    "excess": float(worst.get("increase", 0.0)),
                    "stop_reason": row.get("stop_reason", row.get("solver_status")),
                }, seconds)

    @staticmethod
    def _alpha_distribution(values):
        counts = {label: 0 for label, _ in ALPHA_BUCKETS}
        for value in values:
            for label, contains in ALPHA_BUCKETS:
                if contains(value):
                    counts[label] += 1
                    break
        return counts

    def report(self, *, total_seconds=None):
        if not self.enabled:
            return None
        table = [{"stage": name, "seconds": self.stage_seconds.get(name, 0.0)}
                 for name in STAGES]
        slowest = max(table, key=lambda row: row["seconds"])["stage"] if table else None
        return {
            "counter_semantics": COUNTER_SEMANTICS,
            "iterations": self.iterations,
            "measurement_mode": {
                "speculative_attempt_forks_disabled": True,
                "parallel_shortening_disabled": True,
                "reason": "exact parent-side proposal, evaluation and guard counters",
            },
            "stage_timings": table,
            "slowest_stage": slowest,
            "total_stage_seconds": float(sum(row["seconds"] for row in table)),
            "unattributed_seconds": (
                None if total_seconds is None else
                max(0.0, float(total_seconds) - sum(row["seconds"] for row in table))
            ),
            "slowest_stage_details": {
                "iterations": [row for row in self.iterations if row.get("stage") == slowest],
                "stage_details": self.stage_details.get(slowest, {}),
                "operations": self.operations.get(slowest, {}),
                "guard_checks": self.guard_checks.get(slowest, {}),
                "guard_rejections": self.guard_rejections.get(slowest, {}),
                "accepted_step_distribution": self._alpha_distribution(
                    self.stage_alphas.get(slowest, [])),
                "failed_candidates": self.stage_failed.get(
                    slowest, {"count": 0, "seconds": 0.0, "reasons": {}}),
                "stop_reasons": self.stage_stop_reasons.get(slowest, {}),
                "slow_windows": [row for row in self.slow_windows
                                 if row.get("stage", slowest) == slowest],
            },
            "operations": self.operations,
            "stage_details": self.stage_details,
            "guard_checks": self.guard_checks,
            "guard_rejections": self.guard_rejections,
            "accepted_step_count": len(self.alphas),
            "accepted_step_distribution": self._alpha_distribution(self.alphas),
            "failed_candidates": self.failed,
            "stop_reasons": self.stop_reasons,
            "slow_windows": self.slow_windows,
        }
