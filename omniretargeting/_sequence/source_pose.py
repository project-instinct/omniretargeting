"""Acquire a source-shaped reference before scene correction can warp it."""

from dataclasses import replace

from ..sequence_retargeting import retarget_sequence
from .orientation_seed import source_orientation_seed


def acquire_source_pose(
    model,
    initial,
    positions,
    names,
    mapping,
    roots,
    tasks,
    priority,
    *,
    config,
    framerate,
    root_orientation_targets,
    body_orientation_targets,
    callback=None,
    solve=None,
):
    """Fit kinematic source evidence without competing scene-clearance rows.

    Contact locations are immutable observed world points. Relaxed endpoint
    fitting chooses a reachable IK branch before whole-body contact acquisition;
    it does not classify support or edit the source. Subsequent acquisition must
    stay within this fixed posture and derivative envelope.
    """
    cfg = dict(
        config,
        contact_sigma_m=0.05,
        contact_normal_sigma_m=0.05,
        contact_tangent_sigma_m=0.05,
        source_side_support=True,
        self_collision_enabled=False,
        source_shape_priority=True,
    )
    source_tasks = {
        name: replace(task, contact_phase=None, surface_managed=False)
        for name, task in tasks.items()
    }
    initial = source_orientation_seed(
        model, initial, roots, priority, body_orientation_targets, config
    )
    solve = retarget_sequence if solve is None else solve
    return solve(
        model,
        initial,
        positions,
        names,
        mapping,
        roots,
        source_tasks,
        priority,
        config=cfg,
        framerate=framerate,
        root_orientation_targets=root_orientation_targets,
        body_orientation_targets=body_orientation_targets,
        callback=callback,
        scene_query=None,
        extra_tasks=(),
        environment_points=None,
    )
