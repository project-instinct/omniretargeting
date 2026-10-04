"""Conservative support evidence shared by explicit physical scenarios."""

from copy import deepcopy

import numpy as np

from .contact_support import SourceSupportEvidence
from .support_patches import support_patches
from .wrench_feasibility import DualWrenchCone


def _certify(task):
    contacts, origin, wrench, uncertainty = task
    return DualWrenchCone(contacts, origin=origin).check(wrench, uncertainty)


def _patch_keep(entry):
    """Names a contracted frame keeps as support: necessary plus restored."""
    return (
        {name for patch in entry["necessary"] for name in patch}
        | set(entry.get("restored", ()))
    )


def _restore_short_demotions(plan, min_frames):
    """Restore demotion runs shorter than the source derivative window.

    A demotion (support contracted to optional) is only trusted once it
    holds for at least ``min_frames`` consecutive frames: shorter changes
    sit below the temporal resolution of the source dynamics itself (the
    Savitzky-Golay derivative window behind ``demand["derivative_window"]``),
    so the candidate stays protected as a possible support instead of being
    released. The rule is strictly one-directional: short support segments
    are never demoted, and frames the union never contracted are untouched.
    """
    restored = {}
    if min_frames <= 1 or not plan:
        return restored
    demotion = {}
    for frame, entry in plan.items():
        if entry["contracted"]:
            for name in entry["demoted"]:
                demotion.setdefault(name, []).append(frame)
    for name in sorted(demotion):
        frames = sorted(demotion[name])
        runs, start, previous = [], frames[0], frames[0]
        for frame in frames[1:]:
            if frame != previous + 1:
                runs.append((start, previous))
                start = frame
            previous = frame
        runs.append((start, previous))
        for first, last in runs:
            if last - first + 1 >= min_frames:
                continue
            for frame in range(first, last + 1):
                restored.setdefault(frame, []).append(name)
    for frame, names in restored.items():
        entry = plan[frame]
        entry["restored"] = sorted(set(names))
        entry["demoted"] = sorted(set(entry["demoted"]) - set(names))
    return restored


def _patch_necessity(scenarios, unions, by_frame, wrench, uncertainty, origin,
                     skin_spacing, map_function):
    """Contract certified unions to their necessary support patches.

    The union of one frame splits into support patches (``support_patches``):
    connected contact regions on one surface. A patch is necessary when
    removing it makes every scenario's remaining union provably infeasible
    over the whole uncertainty box; merely uncertain is not proof. When the
    necessary patches alone stay feasible in every scenario, the union
    contracts to them -- by monotonicity of the contact cone every other
    patch is then provably an optional explanation -- and those contacts
    become optional (touching, not carrying). Otherwise the union semantics
    keep protecting every alternative, including genuinely ambiguous
    supports; no redundant support is ever deleted outright.
    """
    plan, removals = {}, []
    for f, union in enumerate(unions):
        checks = by_frame.get(f, [])
        if not union or len(checks) != len(scenarios) or not all(
            c["status"] == "feasible" for c in checks
        ):
            continue
        contacts, surfaces = {}, {}
        for evidence in scenarios:
            for candidate in evidence.candidates[f]:
                if candidate.name in union and candidate.name not in contacts:
                    contacts[candidate.name] = candidate
                    surfaces[candidate.name] = evidence.regions[candidate.name][1]
        link = (
            max((c.reconstruction_bound_m for c in contacts.values()), default=0.0)
            + skin_spacing
        )
        patches = support_patches(contacts, surfaces, link)
        plan[f] = {
            "patches": patches,
            "necessary": [],
            "restored": [],
            "demoted": [],
            "contracted": False,
            "link_distance_m": float(link),
        }
        for index, patch in enumerate(patches):
            gone = set(patch)
            for s, evidence in enumerate(scenarios):
                reduced = [
                    c for c in evidence.candidates[f]
                    if c.name in union and c.name not in gone
                ]
                removals.append(
                    ((f, index, s), (tuple(reduced), origin, wrench[f], uncertainty[f]))
                )
    statuses = {}
    if removals:
        results = list(map_function(_certify, [task for _, task in removals]))
        statuses = {key: result["status"] for key, result in zip(
            (key for key, _ in removals), results)}
    needed = {}
    for f, entry in plan.items():
        necessary = [
            patch for index, patch in enumerate(entry["patches"])
            if all(statuses.get((f, index, s)) == "infeasible"
                   for s in range(len(scenarios)))
        ]
        entry["necessary"] = necessary
        if necessary:
            needed[f] = [name for patch in necessary for name in patch]
    sufficiency = []
    for f, names in needed.items():
        keep = set(names)
        for s, evidence in enumerate(scenarios):
            subset = [c for c in evidence.candidates[f] if c.name in keep]
            sufficiency.append(
                ((f, s), (tuple(subset), origin, wrench[f], uncertainty[f])))
    if sufficiency:
        results = list(map_function(_certify, [task for _, task in sufficiency]))
        for (f, s), result in zip((key for key, _ in sufficiency), results):
            if result["status"] != "feasible":
                # Necessary patches that cannot stand alone do not contract
                # the union; the finding stays in the diagnostics.
                needed.pop(f, None)
    for f, names in needed.items():
        keep = set(names)
        plan[f]["demoted"] = sorted(set(unions[f]) - keep)
        plan[f]["contracted"] = True
    return plan


def combine_support_scenarios(
    scenarios, wrench, *, origin, uncertainty=0.0, map_function=map, skin_spacing=0.0,
    min_demotion_frames=0,
):
    """Protect a feasible union, never pick a convenient friction/gap assumption.

    A support union is certified for the entire supplied uncertainty box in
    every scenario. It need not be minimal and it is labelled accordingly.
    When the union is certified it is additionally split into support patches
    (same surface, environment points linked within the observation's own
    resolution: the contact reconstruction bound plus ``skin_spacing``, the
    measured source skin sampling distance). If exactly the necessary patches
    are feasible alone in every scenario, the union contracts to them and the
    remaining contacts are optional; otherwise the whole union keeps its
    alternative-protecting semantics. Demotion is additionally guarded by the
    source dynamics' own temporal resolution: a candidate demoted for fewer
    than ``min_demotion_frames`` consecutive frames -- the Savitzky-Golay
    derivative window, ``demand["derivative_window"]``, supplied by
    ``source_support_evidence`` -- is restored to support, because the
    evidence cannot resolve changes that fast. That guard never removes
    support: short support segments and uncontracted frames stay untouched.
    Disagreement remains unknown; geometric holds are still protected by the
    downstream policy. No new contact is created by a nominal-only
    explanation. The independent certificates may be computed by
    ``map_function`` (e.g. a process pool's ``map``); the result does not
    depend on it.
    """
    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("Support sensitivity requires explicit scenarios")
    frames = scenarios[0].frames
    wrench = np.asarray(wrench, float)
    uncertainty = np.broadcast_to(np.asarray(uncertainty, float), (frames, 6))
    if any(e.frames != frames for e in scenarios) or wrench.shape != (frames, 6):
        raise ValueError("Physical scenarios must address the same source timeline")
    regions = {k: v for evidence in scenarios for k, v in evidence.regions.items()}
    if any(regions[k] != v for e in scenarios for k, v in e.regions.items()):
        raise ValueError("Scenario contact identities have inconsistent source regions")
    unions, tasks = [], []
    for f in range(frames):
        rows = [e.explanations[f] for e in scenarios]
        # Preserve the source temporal explanation. Reintroducing every
        # framewise alternative here would promote a passing hand that the
        # source-only continuity test had already excluded from support paths.
        alternatives = [
            e.temporal["alternatives"][f]
            if e.temporal is not None
            else row.get("support_sets", row["minimal_feasible_sets"])
            for e, row in zip(scenarios, rows)
        ]
        union = set().union(*(set(s) for sets in alternatives for s in sets))
        unions.append(union)
        if all(row["full"]["status"] == "feasible" for row in rows):
            for evidence in scenarios:
                contacts = [c for c in evidence.candidates[f] if c.name in union]
                tasks.append((f, (contacts, origin, wrench[f], uncertainty[f])))
    results = list(map_function(_certify, [task for _, task in tasks]))
    by_frame = {}
    for index, (f, _) in enumerate(tasks):
        by_frame.setdefault(f, []).append(results[index])
    plan = _patch_necessity(
        scenarios, unions, by_frame, wrench, uncertainty, origin,
        float(skin_spacing), map_function)
    _restore_short_demotions(plan, int(min_demotion_frames))
    explanations, candidates, certificates = [], [], []
    for f in range(frames):
        rows = [e.explanations[f] for e in scenarios]
        union = unions[f]
        checks = by_frame.get(f, [])
        known = len(checks) == len(scenarios) and all(
            c["status"] == "feasible" for c in checks
        )
        entry = plan.get(f)
        contracted = bool(known and entry and entry["contracted"])
        row = deepcopy(rows[0])
        if contracted:
            row["support_sets"] = [sorted(_patch_keep(entry))]
            row["support_set_semantics"] = "necessary_patches_of_certified_union"
        else:
            row["support_sets"] = [sorted(union)] if known else []
            row["support_set_semantics"] = (
                "certified_union_of_alternatives_not_a_minimal_set"
            )
        row["full"]["status"] = "feasible" if known else "uncertain"
        # Necessary means necessary in every specified model. Missing evidence
        # can never demote a geometric hold to an optional touch. Only the
        # candidates observed in this frame get a role: an identity absent
        # here is not an unexplained contact overriding the present ones.
        present = set().union(*(set(r["roles"]) for r in rows))
        roles = {
            name: (
                "necessary_support_candidate"
                if known
                and all(
                    r["roles"].get(name) == "necessary_support_candidate" for r in rows
                )
                # The certified union explains the demand without every
                # other contact: those are optional (touching, not carrying).
                else "optional_contact"
                if known
                else "unexplained"
            )
            for name in regions
            if name in present
        }
        if contracted:
            # The necessary patches alone explain the demand in every
            # scenario and over the whole uncertainty box; every other
            # patch is provably an optional explanation. Restored names
            # rejoin the support set and its necessary roles together.
            keep = _patch_keep(entry)
            roles = {
                name: (
                    "necessary_support_candidate" if name in keep else "optional_contact"
                )
                for name in roles
            }
        row["roles"] = roles
        explanations.append(row)
        candidates.append(scenarios[0].candidates[f])
        certificates.append(checks)
    combined = SourceSupportEvidence(explanations, regions, candidates=candidates)
    combined.diagnostics.update(
        {
            "scenario_count": len(scenarios),
            "certification": "every_supplied_scenario_and_wrench_box",
            "certificates": certificates,
            "support_sets_are_minimal": False,
            "support_patches": {
                "semantics": "same_surface_single_linkage_of_environment_contact_points",
                "skin_spacing_m": float(skin_spacing),
                "min_demotion_frames": int(min_demotion_frames),
                "frames_grouped": len(plan),
                "frames_contracted": sum(
                    1 for entry in plan.values() if entry["contracted"]),
                "demoted_candidate_frames": sum(
                    len(entry["demoted"]) for entry in plan.values()),
                "restored_candidate_frames": sum(
                    len(entry["restored"]) for entry in plan.values()),
                "per_frame": {
                    str(f): {
                        "patches": entry["patches"],
                        "necessary": entry["necessary"],
                        "restored": entry["restored"],
                        "demoted": entry["demoted"],
                        "contracted": entry["contracted"],
                        "link_distance_m": entry["link_distance_m"],
                    }
                    for f, entry in sorted(plan.items())
                },
            },
        }
    )
    return combined
