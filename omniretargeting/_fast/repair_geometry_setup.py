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
