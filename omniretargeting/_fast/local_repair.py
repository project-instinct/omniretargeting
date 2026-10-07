"""Local SQP repair of original-mesh penetration and source-relative chatter.

Replaces the per-body group sweep of ``mesh_repair._repair`` while keeping
immutable limits, guards, per-key protection and strict penetration descent.
Chatter windows require elementwise nonincrease, summed excess progress and
nonincreasing squared collision depth:

1. Active sets (``local_windows``): penetrating bodies are batched per connected
   defect run with the joints their defects in that run actually couple, in the
   established contact-rank order; joint chatter forms its own windows.
2. Each iteration solves one program linearized at the accepted pose
   (``local_qp``): every limit and guard, witness protection with a small
   interior margin, exact linear chatter rows, and a soft term against added
   sticking-contact slip (``local_slip``). The linear witness model keeps
   every sample a small step can make the deepest (``WITNESS_BAND``), not
   only the fixed support probes.
3. A trial failing the nonlinear limits by the second-order remainder of its
   step is not shortened: a minimum-norm correction linearized at the trial
   removes the remainder (``local_correction``).
4. Original geometry is measured at merged trials that pass every nonlinear
   limit, then remeasured for the accepted subset (``local_rounds``). Witnesses
   that still regress become one batch of cutting planes and the direction
   is solved again; unknown queries reject only their own window.
"""

import time

import numpy as np

from .mesh_repair import OriginalGeometry, _activation_depth
from .repair_limits import LOCAL_ANGLE
from . import local_windows as lw

# SQP iterations of one window and curvature corrections of one trial.
ITERATIONS = 12
CORRECTIONS = 3
# Rounds of missed-witness cuts and of trust shrinking per accepted step.
CUT_ROUNDS = 2
MAX_SHRINKS = 1
# Per-coordinate trust box after a model failure, as a fraction of the
# largest coordinate change of the failed step.
TRUST_SHRINK = 0.25
# Interior margin (in each row's own unit) the correction aims for on rows
# the trial violates; acceptance itself is unchanged.
CORRECTION_RESERVE = 1e-6
# Joint-space radius (rad) first tried by the second-order correction.
CORRECTION_RADIUS = 2e-3
# Witness band of the local linear model (metres below the deepest sample).
WITNESS_BAND = 3e-3
# A chatter window must remove at least this much summed excess (rad).
MIN_CHATTER_PROGRESS = 1e-5


class Ledger:
    def __init__(self):
        self.counts = dict(qp_solves=0, correction_solves=0, qp_failures=0, limit_rejections=0,
                           geometry_measures=0, geometry_remeasures=0, geometry_rejections=0, regressions=0, cuts=0,
                           no_descent=0, accepted=0, trust_shrinks=0, chatter_windows=0)
        self.seconds = dict(qp=0., correction=0., limits=0., geometry=0., linearize=0., round_seconds=0.)
        self.corrections = []

    def time(self, name, tick):
        self.seconds[name] += time.monotonic() - tick


def _reserve(bounds):
    """Ask rows violated at the trial to come back CORRECTION_RESERVE inside.

    The guards accept 1e-7 beyond their caps; aiming exactly there leaves an
    inexact interior-point solution on the wrong side. Only already violated
    rows move, so the zero correction of a satisfied row stays feasible.
    """
    for index, (lower, upper) in enumerate(zip(bounds.lower, bounds.upper)):
        if bounds.box is not None and index == bounds.box[0]:
            continue
        bounds.lower[index] = np.where(lower > 0, lower + CORRECTION_RESERVE, lower)
        bounds.upper[index] = np.where(upper < 0, upper - CORRECTION_RESERVE, upper)
    for index, (jacobian, residual, allowed) in enumerate(bounds.norms):
        violated = np.linalg.norm(residual, axis=1) > allowed + 1e-12
        if violated.any():
            bounds.norms[index] = (jacobian, residual,
                                   np.where(violated, np.maximum(allowed - CORRECTION_RESERVE, 0.), allowed))


def correction(limits, current, trial, projection, rows, cuts, linear, radius, ledger):
    """Second-order correction: the smallest step back onto every linearized limit.

    All limits are linearized at ``trial`` with their normal margins (not the
    stricter restoration layout floor); witness rows keep their linearization
    at ``current``; ``linear`` rows are exact. No objective term, so the
    correction is only as large as the second-order remainder it removes.
    """
    from .local_correction import correction as correct

    return correct(limits, current, trial, projection, rows, cuts, linear, radius, ledger)


def _inside_box(limits, poses, step):
    """Apply a joint step, projected onto the fixed joint box.

    The box rows are in every subproblem; an interior-point solution may still
    cross them by more than the box check's 1e-8. Projection onto a box is
    exact for the box, and only coordinates the step moved are touched.
    """
    qi = limits.coords.qi
    result = poses.copy()
    moved = step != 0
    values = poses[:, qi] + step
    # Unmoved coordinates keep their exact bytes (geometry caches use them).
    values[moved] = np.clip(values, limits.lower, limits.upper)[moved]
    result[:, qi] = values
    return result


def _positive(values):
    return float(np.maximum(values, 0.).sum())


class State:
    def __init__(self, current, measured, protected):
        self.current, self.measured, self.protected = current, measured, protected


class _Context:
    def __init__(self, objective, limits, geometry, chatter, ledger, history):
        self.objective, self.limits, self.geometry = objective, limits, geometry
        self.chatter, self.ledger, self.history = chatter, ledger, history

    def __iter__(self):
        return iter((self.objective, self.limits, self.geometry, self.chatter, self.ledger))


def repair_local(objective, poses, scene_objects=(), scene_query=None):
    """Drop-in replacement of ``mesh_repair.repair_original_meshes``."""
    started = time.monotonic()
    entry = np.array(poses, copy=True)
    if not len(objective.coords.qi):
        return entry, {"status": "no_joint_dofs", "seconds": time.monotonic() - started}
    limits = lw.ClusterLimits(objective, entry)
    geometry = OriginalGeometry(objective, scene_objects, scene_query, manifold=True)
    # The local model keeps every witness a small step can make the deepest
    # (``mesh_repair.WITNESS_BAND_CAP``); measured depths are unchanged.
    geometry.witness_band = WITNESS_BAND
    if geometry.moving is not None:
        geometry.moving.witness_band = WITNESS_BAND
    from .repair_geometry_setup import prepare_static_bounds, prepare_moving_reduction
    prepare_static_bounds(geometry, entry)
    prepare_moving_reduction(geometry, entry)
    geometry.start_workers(len(entry))
    try:
        return _repair(objective, entry, limits, geometry, started)
    finally:
        geometry.stop_workers()


def _repair(objective, entry, limits, geometry, started):
    from .local_chatter import JointChatter

    setup = time.monotonic() - started
    tick = time.monotonic()
    baseline = geometry.measure(entry, jacobians=True)
    entry_measure = time.monotonic() - tick
    chatter = JointChatter(objective.coords, objective.reference)
    report = {"method": "local_sqp", "before": geometry.summary(baseline), "history": [],
              "root_fixed": True, "local_angle_budget_rad": LOCAL_ANGLE,
              "chatter_before": dict(violations=int((chatter.excess(entry) > 0).sum()),
                                     excess=_positive(chatter.excess(entry)))}
    if baseline["failures"]:
        report.update(status="unknown_query", after=report["before"], seconds=time.monotonic() - started)
        return entry, report
    limits.freeze_penetration_yield(baseline)
    state = State(entry.copy(), baseline, dict(baseline["depths"]))
    ledger = Ledger()
    ctx = _Context(objective, limits, geometry, chatter, ledger, report["history"])
    windows = lw.penetration_windows(objective, limits, baseline, _activation_depth())
    report["windows"] = [dict(kind=w["kind"], bodies=[objective.model.body(b).name for b in w["bodies"]],
                              block=w["block"], frames=int(w["mask"].sum()), columns=len(w["columns"]),
                              depth_m=w["depth"]) for w in windows]
    from .local_rounds import run_wave, waves
    for wave in waves(windows):
        run_wave(ctx, wave, state)
    chatter_windows = lw.chatter_windows(chatter, state.current, limits.width)
    ledger.counts["chatter_windows"] += len(chatter_windows)
    for wave in waves(chatter_windows):
        run_wave(ctx, wave, state)
    limits.release(state.current)
    current, measured = state.current, state.measured
    ends = [r["started"] for r in report["history"][1:]] + [time.monotonic()]
    for record, end in zip(report["history"], ends):
        record["seconds"] = record.get("elapsed", end - record["started"])
        record.pop("started")
    report.update(status="corrected" if np.any(current != entry) else "retained",
                  after=geometry.summary(measured), limits=limits.check(current)[1],
                  chatter_after=dict(violations=int((chatter.excess(current) > 0).sum()),
                                     excess=_positive(chatter.excess(current))),
                  ledger=dict(counts=ledger.counts, seconds=ledger.seconds,
                              setup_seconds=setup, entry_measure_seconds=entry_measure),
                  corrections=ledger.corrections,
                  geometry_cache=dict(frames=getattr(geometry, "window_totals", {}),
                                      jacobians=getattr(geometry, "linearization_totals", {})),
                  seconds=time.monotonic() - started)
    return current, report
