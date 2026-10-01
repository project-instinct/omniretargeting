"""Opt-in, source-only review of separated, uninterrupted near passes.

This does not infer intention or declare a contact unloaded. It identifies a
restricted case: all observed skin samples remain outside an explicitly supplied
geometric error budget, while an affine motion explains the entire candidate
event and its immediate neighbours within that same budget. Anything touching,
stationary, turning, unobserved, explicitly labelled, or protected by support
evidence is retained. No body name, duration cutoff, or robot reach enters the
decision. A caller must retain scene collision checks after removing attraction.
"""

from dataclasses import dataclass, replace

import numpy as np

from .contact_schedule import _iter_true_runs, approach_blend


@dataclass(frozen=True)
class PassReview:
    candidate_mask: np.ndarray
    pass_mask: np.ndarray
    events: tuple
    position_uncertainty_m: float


def review_contact_passes(
    positions,
    signed_clearance,
    candidate_mask,
    *,
    times,
    position_uncertainty_m,
    environment_ids=None,
    protected_mask=None,
):
    """Review source material coordinates and minimum signed SKIN clearance.

    ``positions`` follows one material point/semantic centre, with recorded
    object motion removed. ``signed_clearance`` is the minimum over the whole
    observed patch, positive on the known free side; unknown signs are NaN.
    ``position_uncertainty_m`` is a caller-declared combined reconstruction and
    registration budget, NOT the distance threshold or numerical SMPL-X error.
    True authored contacts and every plausible support alternative must be
    included in ``protected_mask``. Missing mechanics is not negative evidence.
    The resulting pass label remains an experimental geometric hypothesis.
    """
    points = np.asarray(positions, float)
    gap = np.asarray(signed_clearance, float)
    mask = np.asarray(candidate_mask, bool)
    time = np.asarray(times, float)
    frames = len(mask)
    env = (
        np.zeros(frames, int)
        if environment_ids is None
        else np.asarray(environment_ids)
    )
    protected = (
        np.zeros(frames, bool)
        if protected_mask is None
        else np.asarray(protected_mask, bool)
    )
    budget = float(position_uncertainty_m)
    if (
        points.shape != (frames, 3)
        or any(v.shape != (frames,) for v in (gap, mask, time, env, protected))
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
        or not np.isfinite(budget)
        or budget <= 0
    ):
        raise ValueError(
            "Pass review needs a shared timeline and positive metric uncertainty"
        )
    passes = np.zeros(frames, bool)
    events = []
    for start, stop in _iter_true_runs(mask):
        event = {"start": start, "stop": stop, "decision": "retain"}
        events.append(event)
        if protected[start:stop].any():
            event["reason"] = "protected_contact_evidence"
            continue
        # Never invent an outside trajectory across a recording boundary or a
        # change of object/material frame. An incomplete event stays protected.
        if (
            start == 0
            or stop == frames
            or np.any(env[start - 1 : stop + 1] != env[start])
        ):
            event["reason"] = "incomplete_or_changing_environment"
            continue
        window = points[start - 1 : stop + 1]
        clearance = gap[start:stop]
        if not np.isfinite(window).all() or not np.isfinite(clearance).all():
            event["reason"] = "unknown_observation"
            continue
        event["min_skin_clearance_m"] = float(clearance.min())
        if np.any(clearance <= budget):
            event["reason"] = "touch_or_gap_within_uncertainty"
            continue
        # Normalize time for numerical conditioning, not a recording-dependent
        # velocity cutoff. A rigid change of coordinates leaves both norms equal.
        clock = time[start - 1 : stop + 1]
        phase = (clock - clock[0]) / (clock[-1] - clock[0])
        design = np.column_stack((np.ones(len(phase)), phase))
        coefficient = np.linalg.lstsq(design, window - window[0], rcond=None)[0]
        fitted = design @ coefficient + window[0]
        error = float(np.linalg.norm(window - fitted, axis=1).max())
        travel = float(np.linalg.norm(coefficient[1]))
        event.update(affine_error_m=error, explained_travel_m=travel)
        if error > budget:
            event["reason"] = "motion_change_or_curvature"
        elif travel <= 2 * budget:
            # Two endpoints can each move by the observation budget. Below
            # their combined uncertainty, stationary contact remains plausible.
            event["reason"] = "stationary_within_uncertainty"
        else:
            event.update(decision="unconfirmed_pass", reason="separated_affine_pass")
            passes[start:stop] = True
    original = mask.copy()
    original.setflags(write=False)
    passes.setflags(write=False)
    return PassReview(original, passes, tuple(events), budget)


def without_pass_attraction(task, review):
    """Derive a trial task without modifying observed masks/anchors/source.

    Retained contacts keep their original anchors AND approach references.
    Recomputing approach progress after dropping a run could extend attraction
    from its neighbours over the removed event, defeating the intended policy.
    Call before role attachment and retain ``review`` beside the original plan.
    """
    if not np.array_equal(task.contact_mask, review.candidate_mask):
        raise ValueError("Pass review must refer to the exact original contact plan")
    if task.contact_phase is not None:
        raise ValueError("Review candidates before attaching shared contact roles")
    if not review.pass_mask.any():
        return task
    _, _, assigned = approach_blend(
        task.tracking_positions,
        task.contact_positions,
        task.contact_mask,
        return_contact_frames=True,
    )
    weights, targets = (np.array(v, copy=True) for v in task.approach)
    influenced = review.pass_mask | review.pass_mask[assigned]
    # An imported handover may not use this module's ownership convention.
    if task.approach_reference is not None:
        raise ValueError(
            "Review candidates on the original full timeline before slicing"
        )
    weights[influenced] = 0.0
    targets[influenced] = task.tracking_positions[influenced]
    mask = task.contact_mask & ~review.pass_mask
    modes = np.array(
        task.contact_modes
        if task.contact_modes is not None
        else np.where(task.contact_mask, "sticking", "none"),
        copy=True,
    )
    modes[review.pass_mask] = "none"
    return replace(
        task,
        contact_mask=mask,
        contact_modes=modes,
        approach_reference=(weights, targets),
    )
