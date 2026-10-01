"""Remove arbitrary contact groups without inferring individual contact forces."""

from .wrench_feasibility import DualWrenchCone


def removal_evidence(contacts, wrench, groups, *, origin=(0, 0, 0), uncertainty=0.):
    """Distinguish necessary groups, redundant explanations and uncertainty.

    A group is certified necessary only if the full set explains the complete
    uncertainty box and the set without it cannot explain any demand in that
    box. Merely losing robust feasibility does not prove necessity.
    """
    full = DualWrenchCone(contacts, origin=origin).check(wrench, uncertainty)
    names = {c.name for c in contacts}
    result = {}
    for group, members in groups.items():
        selected = names.intersection(members)
        if not selected:
            result[group] = {"role": "absent", "removed": []}
            continue
        reduced = DualWrenchCone([c for c in contacts if c.name not in selected], origin=origin).check(wrench, uncertainty)
        role = "unexplained"
        if full["status"] == "feasible":
            role = {"infeasible": "necessary_support_group", "feasible": "optional_explanation", "uncertain": "uncertain_necessity"}[reduced["status"]]
        result[group] = {"role": role, "removed": sorted(selected), "without": reduced}
    return {"full": full, "groups": result, "force_distribution_inferred": False}
