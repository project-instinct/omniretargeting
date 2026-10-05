"""Reuse immutable model query preparation with fresh repair-owned buffers."""

from ..self_collision import SelfCollisionQuery, self_collision_config


def self_query(objective):
    """Acquisition's default query has the same structural-pair classification.

    Custom subclasses/configuration and another model cannot be substituted.
    fresh_state resets FK, counters and segment state while retaining hulls.
    """
    previous = getattr(objective, 'self_collision', None)
    if (type(previous) is SelfCollisionQuery and previous.model is objective.model
            and previous.cfg == self_collision_config(None)):
        return previous.fresh_state()
    return SelfCollisionQuery(objective.model)


def prepare_moving(geometry):
    """Resident original winding/AABB state for each moving mesh component.

    The static repair already uses this exact backend. Moving forward samples
    otherwise rebuild winding trees at every frame/component, even when the
    object only changes its prescribed transform. Prepare once before fork;
    custom callbacks and missing optional native headers retain their path.
    """
    from types import SimpleNamespace
    from .repair_static_query import prepare
    if geometry.moving is None:
        return
    geometry.moving.objects = [
        {**obj, 'query': prepare(SimpleNamespace(scene=obj['query']))}
        for obj in geometry.moving.objects]


def prepare_moving_reduction(geometry, poses):
    """Build repair-only cones for objects reachable from its entry poses.

    The inexpensive sphere/box test only decides whether to prepare an
    acceleration structure. Objects without one still execute full queries,
    even if a later candidate approaches them.
    """
    import subprocess
    import mujoco
    import numpy as np
    from ..moving_scene import interpolate_object
    from .repair_static_query import _QUERY
    from .runtime_options import OPTIONS
    moving = geometry.moving
    if not OPTIONS.repair_geometry_reduction or moving is None:
        return
    from .original_geometry_reduction import ClosedComponentReduction
    eligible = {i for i, obj in enumerate(moving.objects)
                if getattr(obj['query'], '_r4_serial', False) and obj.get('closed')
                and getattr(getattr(obj['query'].query, '__func__', None), '__code__', None)
                is _QUERY.__code__}
    if not eligible:
        return
    geoms = [shell['geom_id'] for shell in moving.shells]
    radii = np.array([shell['radius'] for shell in moving.shells])
    reachable = set()
    band = moving.cfg['activation_distance']
    for frame, pose in enumerate(poses):
        moving.data.qpos[:] = pose
        mujoco.mj_forward(moving.model, moving.data)
        for oid in eligible - reachable:
            obj = moving.objects[oid]
            translation, rotation, scale = interpolate_object(obj['surface'], frame)
            local = (moving.data.geom_xpos[geoms] - translation) @ rotation / scale
            for cid in obj['closed']:
                bounds = obj['query']._component_query_data[cid][2]
                gap = np.maximum(np.maximum(bounds[0] - local, local - bounds[1]), 0.)
                if np.any(np.linalg.norm(gap, axis=1) <= (radii + band) / scale):
                    reachable.add(oid)
                    break
        if reachable == eligible:
            break
    for oid in reachable:
        obj = moving.objects[oid]
        scene = obj['query']
        if obj.get('repair_reduction') is not None:
            continue
        try:
            obj['repair_reduction'] = ClosedComponentReduction(scene, band)
        except (OSError, RuntimeError, subprocess.CalledProcessError):
            pass  # Optional screening never replaces the exact fallback.


def prepare_static_bounds(geometry, poses):
    """Prepare the existing exact distance cones before a cold CPU audit.

    Only closed, winding-signed scenes are eligible. Query original triangles
    at grid nodes once; interpolation is never a clearance certificate. The
    existing bound takes max(node_clearance - distance_to_node), and only
    removes points proved outside the witness band. No depth/witness is
    approximated. GPU-supported scenes retain their established screen.
    """
    if not geometry._static_bound_reuse or geometry._gpu_allowed:
        return
    from copy import copy
    import subprocess
    from ..scene_geometry import SceneQuery
    from .original_geometry_bounds import build_scene_bounds
    from .repair_static_query import prepare, _QUERY
    from ..geometry_query import GeometryQueryError
    if (type(geometry.scene) is not SceneQuery
            or (not getattr(geometry.scene, '_r4_serial', False)
                and getattr(geometry.scene.query, '__func__', None) is not _QUERY)):
        return
    kin = getattr(geometry.objective, 'kin', None)
    if kin is None:
        from .kinematics import TrajectoryKinematics
        kin = TrajectoryKinematics(geometry.model, geometry.columns)
    kin.evaluate(poses, with_jacobian=False)
    view = copy(geometry)
    view.scene = prepare(geometry)
    # build_scene_bounds needs only queried placement envelopes for its region.
    # These are a private region description, never inserted in witness caches.
    view._static_cache = {(frame, int(geom)): (None, None, kin.geom_xpos[frame, geom], None)
                          for frame in range(len(poses)) for geom in geometry.objective.shell_geoms}
    try:
        bounds = build_scene_bounds(view)
    except GeometryQueryError:
        return  # Exact measurement remains responsible for reporting unknowns.
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        return  # Optional native acceleration may be unavailable.
    geometry._static_coarse_bounds = bounds
    geometry._repair_static_bounds = bounds is not None
    geometry.static_bounds_stats = getattr(view, 'static_bounds_stats', {})
