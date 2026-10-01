"""Finish original-collider acquisition before attained contacts are frozen."""

from .local_reach import unreachable_geoms
from .original_clearance import OriginalClearance
from .repair_limits import LOCAL_ANGLE, FREE_DRIFT


def complete_original_clearance(objective, poses, provider, conflicts, optimize, callback=None):
    """Add newly proved local-box conflicts to acquisition, never to a new reference.

    Initial reference probes cannot predict every overlap a contact-acquiring
    step may introduce. Check again before freezing contacts. Every extra pass
    adds at least one collider to a monotonically growing set, so a stalled
    solve with an unchanged witness set cannot request more iterations.
    """
    selected = {row['geom_id'] for row in conflicts}
    history = []
    conflicts = list(conflicts)
    while True:
        provider(poses)
        geometry = provider.geometry
        found = unreachable_geoms(objective.model, poses, provider.last_measurement,
                                  angle=LOCAL_ANGLE, slide=FREE_DRIFT,
                                  closed_static=geometry._static_bound_reuse)
        added = {row['geom_id'] for row in found} - selected
        if not added:
            return poses, history, conflicts
        selected.update(added)
        conflicts += [dict(row, detected='before_contact_freeze') for row in found if row['geom_id'] in added]
        objective.stop_workers()
        objective.original_clearance = OriginalClearance(geometry, selected)
        objective.start_workers()
        poses, more = optimize(objective, poses, iterations=objective.cfg['iterations'],
                               callback=callback, label='contact_original_meshes')
        history += more
