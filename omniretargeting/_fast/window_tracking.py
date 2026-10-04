"""Explicit ownership transfer of collision proxies in windowed priority mode."""


def skips_proxy_collision(objective):
    from .runtime_options import OPTIONS
    coordinator = getattr(objective, 'priority_clearance', None)
    return (OPTIONS.windowed_priority and coordinator is not None
            and getattr(coordinator, 'owns_clearance', False) is True)


def collision_stats(objective, poses, stats, *, frames=None):
    """Merge original-geometry diagnostics only where full poses are available.

    A missing/unreliable family is unknown, never zero penetration. The
    coordinator validates original shell/self/object coverage before taking
    ownership; its measured diagnostics retain their own provenance.
    """
    if not skips_proxy_collision(objective):
        return stats
    stats = dict(stats)
    stats.update(proxy_collision_terms_skipped=True,
                 clearance_diagnostics_owner='priority_clearance_original_geometry',
                 scene_depth_m=None, moving_depth_m=None, self_depth_m=None)
    coordinator = objective.priority_clearance
    measure = getattr(coordinator, 'collision_diagnostics', None)
    if measure is not None:
        stats.update(measure(poses, frames=frames))
    else:
        stats['collision_diagnostics_reliable'] = False
        stats['collision_diagnostics_failure'] = 'coordinator_diagnostics_unavailable'
    return stats
