"""Bound the coupled whole-body conic solve to fixed-reference time windows."""

import time

import numpy as np

from .scene_witness_guard import preserve_known_scene_samples
from .whole_body_guard import WholeBodyGuard, interpolated_pose


def _moving_rows(validator, moving, poses, objective, guard, cfg):
    if moving is None:
        return {"poses": 0, "reverse_available_facets": 0, "reverse_selected_facets": 0}
    n = len(poses)
    objective.moving_grid = []
    available = selected = 0
    for frame in range(n):
        steps = 1 if frame + 1 == n else max(
            moving.interval_steps(
                path[frame], path[frame + 1], validator.frame_start + frame,
                cfg["motion_spacing"], cfg["max_substeps"],
            )
            for path in (validator.reference, poses)
        )
        for part in range(steps):
            alpha = part / steps
            objective.moving_grid.append((frame, alpha))
            pose = interpolated_pose(validator.model, poses, frame, alpha)
            query = moving.query(pose, validator.frame_start + frame + alpha, validator.di)
            if query["failures"]:
                raise ValueError("Moving geometry query failed: " + "; ".join(query["failures"]))
            guard.collision_rows.extend((frame, alpha, row) for row in query["rows"])
            available += query["available_rows"]
            selected += query["selected_rows"]
    immovable = [
        {
            "frame": validator.frame_start + frame + alpha,
            "body": validator.model.body(
                int(validator.model.geom_bodyid[row["geom_id"]])
            ).name,
            "geom_id": row["geom_id"],
            "depth_m": -row["clearance"],
            "kind": row["kind"],
        }
        for frame, alpha, row in guard.collision_rows
        if row["clearance"] < 0 and np.linalg.norm(row["jacobian"]) < 1e-10
    ]
    return {
        "poses": len(objective.moving_grid),
        "reverse_available_facets": available,
        "reverse_selected_facets": selected,
        "acceptance_checks_all_original_samples": True,
        "immovable_witnesses": immovable,
    }


def solve_windowed(validator, moving, original, cfg, active, full_guard,
                   *, core_frames=96, boundary_frames=3,
                   candidate_acceptance=None, baseline_moving=None):
    """Optimize local variables, then audit their full composed trajectory.

    Three fixed samples on either side cover every changed velocity,
    acceleration and jerk stencil. All windows retain the same immutable
    robot reference, contact marker choices and world scene.
    ``baseline_moving`` is the completed moving check of that same reference,
    obtained by the caller before solving. Reuse it only for identical poses.
    """
    from .whole_body_correction import WholeBodyObjective

    result = np.asarray(original).copy()
    events = []
    accepted_windows = []
    before = full_guard._baseline_geometry()
    for sweep in range(int(cfg["iterations"])):
        if moving is not None:
            moving.prepare_trajectory(result)
        accepted = 0
        for first in range(0, len(result), core_frames):
            last = min(len(result), first + core_frames)
            start = max(0, first - boundary_frames)
            stop = min(len(result), last + boundary_frames)
            tick = time.perf_counter()
            with validator.segment(start, stop) as view:
                current = result[start:stop].copy()
                guard = WholeBodyGuard(view, moving)
                objective = WholeBodyObjective(view, moving)
                coverage = _moving_rows(view, moving, current, objective, guard, cfg)
                moving_seconds = time.perf_counter() - tick
                matrix, rhs = objective.linearize(current)
                linearized = time.perf_counter()
                editable = np.zeros(stop - start, dtype=bool)
                editable[first - start:last - start] = True
                task_rows = matrix.shape[0] - sum(
                    regularizer.shape[0] for regularizer in objective.regularizers
                )
                if (
                    task_rows == 0
                    and not guard.collision_rows
                    and np.array_equal(
                        current[editable], view.reference[editable]
                    )
                ):
                    events.append({
                        "iteration": sweep,
                        "window": [first, last - 1],
                        "solve": {"status": "no_active_rows"},
                        "moving_coverage": coverage,
                        "accepted_alpha": 0.0,
                        "seconds": time.perf_counter() - tick,
                    })
                    print(
                        f"Whole-body window sweep={sweep} frames={first}:{last} "
                        "skipped=no_active_rows",
                        flush=True,
                    )
                    continue
                change, solve = guard.solve(
                    matrix, rhs, current, active, editable_frames=editable
                )
                solved = time.perf_counter()
                event = {
                    "iteration": sweep,
                    "window": [first, last - 1],
                    "solve": solve,
                    "moving_coverage": coverage,
                    "moving_seconds": moving_seconds,
                    "linearization_seconds": linearized - tick - moving_seconds,
                    "solve_seconds": solved - linearized,
                }
                if change is not None and np.max(np.abs(change), initial=0) >= 1e-7:
                    def accepts_window(
                        local, *, first=first, last=last, start=start,
                        accepted=result, check=candidate_acceptance,
                    ):
                        if check is None:
                            return True
                        composed = accepted.copy()
                        composed[first:last] = local[first - start:last - start]
                        return check(composed)

                    candidate, acceptance = guard.accept(
                        current, change, objective,
                        candidate_acceptance=accepts_window,
                    )
                    event.update(acceptance)
                    if acceptance["accepted_alpha"]:
                        result[first:last] = candidate[first - start:last - start]
                        accepted_windows.append((event, first, last, result[first:last].copy()))
                        accepted += 1
                else:
                    event["accepted_alpha"] = 0.0
                event["seconds"] = time.perf_counter() - tick
                event["acceptance_seconds"] = time.perf_counter() - solved
                events.append(event)
                print(
                    f"Whole-body window sweep={sweep} frames={first}:{last} "
                    f"solve={solve['status']} alpha={event['accepted_alpha']:g} "
                    f"seconds={event['seconds']:.1f}",
                    flush=True,
                )
        if not accepted:
            break

    reused = {"static": 0, "moving": 0}

    def full_check(candidate):
        reasons = (
            ["fixed_sequence_budget"]
            if candidate_acceptance is not None and not candidate_acceptance(candidate)
            else []
        )
        if not reasons:
            reasons.extend(validator.fidelity(candidate)["reasons"])
        geometry = None
        moving_stats = None
        unchanged = np.array_equal(candidate, validator.reference)
        if not reasons:
            if unchanged:
                geometry = before
                reused["static"] += 1
            else:
                geometry = validator.geometry(candidate)
            reasons.extend(validator.compare_geometry(
                before, geometry, allow_existing_uncertainty=True
            ))
            if not reasons:
                reasons.extend(preserve_known_scene_samples(validator, candidate, before))
        if not reasons:
            if unchanged and baseline_moving is not None and baseline_moving.get("complete"):
                moving_stats = baseline_moving
                reused["moving"] += 1
            else:
                reasons, moving_stats = full_guard.moving_check(candidate)
        return sorted(set(reasons)), geometry, moving_stats

    print("Whole-body windows: checking composed trajectory", flush=True)
    reasons, geometry, moving_stats = full_check(result)
    if reasons:
        # Keep independently safe windows if their composition introduced a
        # cross-window regression. Their fixed boundary frames stay unchanged.
        accepted_result = np.asarray(original).copy()
        for event, first, last, accepted_poses in accepted_windows:
            trial = accepted_result.copy()
            trial[first:last] = accepted_poses
            failure, checked, checked_moving = full_check(trial)
            if failure:
                event["composition_rejected"] = failure
            else:
                accepted_result, geometry, moving_stats = trial, checked, checked_moving
            print(
                f"Whole-body windows: replay frames={first}:{last} "
                f"accepted={not failure}",
                flush=True,
            )
        result = accepted_result
        reasons, geometry, moving_stats = full_check(result)
    full_guard.last_geometry = geometry if geometry is not None else before
    full_guard.last_moving = moving_stats
    return result, {
        "iterations": events,
        "accepted_update_count": sum(
            bool(event.get("accepted_alpha")) and not event.get("composition_rejected")
            for event in events
        ),
        "status": "accepted_updates" if np.any(result != original) else "no_feasible_update",
        "final_reasons": reasons,
        "window_frames": core_frames,
        "boundary_frames": boundary_frames,
        "reused_baseline_checks": reused,
    }
