"""Original-geometry acceptance with frozen per-key penetration budgets."""

def measure_candidate(objective, geometry, candidate, measured, protected, group, before, *, frames):
    """Measure each candidate against its frozen collision-depth budgets."""
    from .mesh_repair import _regressed
    checked = geometry.measure(candidate,
        protected_depths=protected['depths'], frames=frames)
    bad = [key for key, value in checked['depths'].items()
            if _regressed(value, protected['depths'].get(key, 0.))]
    after = sum(value**2 for value in checked['depths'].values())
    descent = not (after >= before - 1e-10)
    return checked, bad, after, descent
