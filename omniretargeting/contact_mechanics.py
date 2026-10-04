"""Independent source-contact explanations using net wrench cones.

No contact-force variables enter retargeting, and no load percentages are
inferred. All points, normals and wrenches use world axes; moments use the
explicit, fixed origin. Convex-hull conversion is cached per contact geometry.
See docs/contact_mechanics.md for assumptions and uncertainty semantics.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class ContactCandidate:
    name: str
    observed_points: np.ndarray
    surface_points: np.ndarray
    normals: np.ndarray
    friction: float
    reconstruction_bound_m: float
    mode: str = "sticking"
    sliding_velocity: np.ndarray | None = None
    friction_sides: int = 8
    # Optional stable indices into the source's full-body surface samples.
    # These identify observations; they are not robot-specific parameters.
    sample_indices: np.ndarray | None = None

    def rays(self, origin, length_scale):
        """Wrench rays (force, moment/length_scale) of this contact, or None across a gap.

        Every subset test of one frame asks for the same rays: they are
        computed once per origin and scale and then shared (read-only).
        """
        key = (np.asarray(origin, float).tobytes(), float(length_scale))
        cache = self.__dict__.get("_ray_cache")
        if cache is None:
            cache = {}
            object.__setattr__(self, "_ray_cache", cache)
        if key not in cache:
            cache[key] = self._rays(origin, length_scale)
        return cache[key]

    def _rays(self, origin, length_scale):
        observed = np.asarray(self.observed_points, float).reshape(-1, 3)
        points = np.asarray(self.surface_points, float).reshape(-1, 3)
        normals = np.broadcast_to(np.asarray(self.normals, float), points.shape).copy()
        if (
            not len(points)
            or observed.shape != points.shape
            or not all(np.isfinite(a).all() for a in (observed, points, normals))
            or not np.isfinite(self.friction)
            or self.friction < 0
            or not np.isfinite(self.reconstruction_bound_m)
            or self.reconstruction_bound_m < 0
            or np.any(np.linalg.norm(normals, axis=1) < 1e-10)
        ):
            raise ValueError(
                "Contact candidates require finite paired skin/surface geometry"
            )
        if self.mode not in ("sticking", "sliding"):
            raise ValueError("A candidate must specify sticking or sliding friction")
        if not isinstance(self.friction_sides, int) or self.friction_sides < 3:
            raise ValueError("A friction polygon needs at least three sides")
        if (
            np.max(np.linalg.norm(observed - points, axis=1))
            > self.reconstruction_bound_m
        ):
            return None  # mechanics cannot create a force across a real gap
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        if self.mode == "sticking":
            axes = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
            first = np.cross(normals, axes)
            # Each cross product has one zero component. This rowwise norm
            # retains the scalar norm's sum while avoiding per-point calls.
            first /= np.linalg.norm(first, axis=1, keepdims=True)
            second = np.cross(normals, first)
            angles = np.arange(self.friction_sides) * (2 * np.pi / self.friction_sides)
            directions = normals[:, None] + self.friction * (
                np.cos(angles)[None, :, None] * first[:, None]
                + np.sin(angles)[None, :, None] * second[:, None])
            force = directions.reshape(-1, 3)
            locations = np.repeat(points, self.friction_sides, axis=0)
            moment = np.cross(locations - origin, force) / length_scale
            return np.c_[force, moment]
        forces, locations = [], []
        for i, (point, normal) in enumerate(zip(points, normals)):
            if self.mode == "sliding":
                if self.sliding_velocity is None:
                    raise ValueError(
                        "Sliding requires the observed relative slip direction"
                    )
                velocity = np.broadcast_to(
                    np.asarray(self.sliding_velocity, float), points.shape
                )[i]
                tangent = velocity - normal * (normal @ velocity)
                speed = np.linalg.norm(tangent)
                if not np.isfinite(speed) or speed < 1e-10:
                    raise ValueError(
                        "Sliding velocity must have a finite nonzero tangent"
                    )
                directions = [normal - self.friction * tangent / speed]
            else:
                axis = np.eye(3)[np.argmin(np.abs(normal))]
                first = np.cross(normal, axis)
                first /= np.linalg.norm(first)
                second = np.cross(normal, first)
                # Inscribed polygon: conservative relative to circular Coulomb friction.
                angles = np.arange(self.friction_sides) * (2 * np.pi / self.friction_sides)
                directions = normal + self.friction * (
                    np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second)
            forces.extend(directions)
            locations.extend([point] * len(directions))
        force = np.asarray(forces)
        moment = np.cross(np.asarray(locations) - origin, force) / length_scale
        return np.c_[force, moment]


@lru_cache(maxsize=256)
def _cone_faces(shape, serialized):
    """Convert ray rows to A w <= 0, including a lower-dimensional span.

    A facet through zero of conv({0} union normalized rays) is a cone facet.
    Facets away from zero bound the artificial finite hull and are discarded.
    A force-closure cone has no such facets within its span. SVD handles point
    contacts, frictionless contacts, and the empty (flight) cone explicitly.
    """
    rays = np.frombuffer(serialized, dtype=np.float64).reshape(shape)
    if not len(rays):
        result = np.r_[np.eye(6), -np.eye(6)]
    else:
        normalized = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        # Only the six-dimensional right basis is needed. Avoid a quadratic
        # left singular-vector allocation when contact patches have many rays.
        _, values, basis = np.linalg.svd(normalized, full_matrices=len(normalized) < 6)
        rank = int(np.count_nonzero(values > max(normalized.shape) * values[0] * 1e-12))
        span, null = basis[:rank], basis[rank:]
        reduced = normalized @ span.T
        if rank == 1:
            inequalities = []
            if reduced.min() >= -1e-10:
                inequalities.append(-span[0])
            if reduced.max() <= 1e-10:
                inequalities.append(span[0])
            faces = np.asarray(inequalities).reshape(-1, 6)
        else:
            hull = ConvexHull(np.r_[np.zeros((1, rank)), reduced])
            through_origin = np.abs(hull.equations[:, -1]) < 1e-9
            faces = hull.equations[through_origin, :-1] @ span
        result = np.r_[faces, null, -null]
        result = np.unique(np.round(result, 12), axis=0)
    result.setflags(write=False)
    return result


class WrenchCone:
    def __init__(self, contacts, *, origin=(0.0, 0.0, 0.0), length_scale_m=1.0):
        self.origin = np.array(origin, float, copy=True)
        self.length_scale = float(length_scale_m)
        if (
            self.origin.shape != (3,)
            or not np.isfinite(self.origin).all()
            or not np.isfinite(self.length_scale)
            or self.length_scale <= 0
        ):
            raise ValueError(
                "A wrench cone needs a finite origin and positive length scale"
            )
        self.accepted, self.excluded = [], []
        rays = []
        for contact in contacts:
            block = contact.rays(self.origin, self.length_scale)
            if block is None:
                self.excluded.append(contact.name)
            else:
                self.accepted.append(contact.name)
                rays.extend(block)
        matrix = np.asarray(rays, dtype=np.float64).reshape(-1, 6)
        scaled_faces = _cone_faces(matrix.shape, matrix.tobytes())
        self.A = scaled_faces / np.array(
            [1.0, 1.0, 1.0, self.length_scale, self.length_scale, self.length_scale]
        )
        self.A.setflags(write=False)

    def check(self, wrench, uncertainty=0.0, *, tolerance=1e-8):
        wrench = np.asarray(wrench, float)
        radius = np.broadcast_to(np.asarray(uncertainty, float), (6,))
        if (
            wrench.shape != (6,)
            or not np.isfinite(wrench).all()
            or not np.isfinite(radius).all()
            or np.any(radius < 0)
            or not np.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError("Net wrench and nonnegative uncertainty must be finite")
        violation = self.A @ wrench
        spread = np.abs(self.A) @ radius
        robust = bool(np.all(violation + spread <= tolerance))
        impossible = bool(np.any(violation - spread > tolerance))
        return {
            "nominal_feasible": bool(np.all(violation <= tolerance)),
            "status": "feasible"
            if robust
            else "infeasible"
            if impossible
            else "uncertain",
            "max_scaled_violation": float(violation.max(initial=0)),
            "excluded_geometry": list(self.excluded),
        }


def _minimal_dual_subsets(indices, check, seeds=()):
    """Enumerate minimal explanations by excluding a member of each solution.

    Any different minimal support must omit at least one member of a known
    minimal support. Branching on that fact avoids visiting every irrelevant
    subset when a few contacts are necessary among many geometric candidates.
    """
    pending, visited, minimal = [tuple(indices)], set(), set()
    valid_seeds = {}
    while pending:
        chosen = pending.pop()
        if chosen in visited:
            continue
        visited.add(chosen)
        status = check(chosen)["status"]
        if status == "infeasible":
            continue
        if status != "feasible":
            # Uncertainty is not proof of impossibility. Preserve the original
            # exhaustive semantics when no feasible certificate is available.
            pending.extend(tuple(i for i in chosen if i != k) for k in chosen)
            continue
        reduced = None
        for seed in seeds:
            if not set(seed) <= set(chosen):
                continue
            if seed not in valid_seeds:
                valid_seeds[seed] = (
                    check(seed)["status"] == "feasible"
                    and all(check(tuple(i for i in seed if i != k))["status"] == "infeasible"
                            for k in seed)
                )
            if valid_seeds[seed]:
                reduced = seed
                break
        if reduced is None:
            reduced = chosen
            for index in chosen:
                trial = tuple(i for i in reduced if i != index)
                if check(trial)["status"] == "feasible":
                    reduced = trial
        minimal.add(reduced)
        pending.extend(tuple(i for i in chosen if i != k) for k in reduced)
    return sorted(minimal)


def explain_contacts(
    contacts,
    wrench,
    *,
    origin=(0.0, 0.0, 0.0),
    uncertainty=0.0,
    max_candidates=8,
    method="facets",
    detail="full",
    prior_minimal=(),
):
    """Compare sets, retain alternatives, and never delete all redundant supports."""
    contacts = tuple(contacts)
    names = [c.name for c in contacts]
    if len(names) != len(set(names)) or len(names) > max_candidates:
        raise ValueError("Candidate names must be unique and their number bounded")
    if method not in {"facets", "dual"}:
        raise ValueError("Wrench feasibility method must be facets or dual")
    if detail not in {"full", "source", "object"}:
        raise ValueError("Contact explanation detail must be full, source or object")
    from .wrench_feasibility import DualWrenchCone

    cone_type = WrenchCone if method == "facets" else DualWrenchCone
    checks = {}
    geometry_checks = {}
    feasible_masks, infeasible_masks = [], []
    full = tuple(range(len(contacts)))

    def check(chosen, *, certify=False):
        if chosen not in checks or (certify and checks[chosen].get("_inferred")):
            mask = sum(1 << i for i in chosen)
            if detail != "full" and not certify:
                known_feasible = any(mask & known == known for known in feasible_masks)
                known_infeasible = any(mask & known == mask for known in infeasible_masks)
                if known_feasible != known_infeasible:
                    checks[chosen] = {"status": "feasible" if known_feasible else "infeasible",
                                      "excluded_geometry": [], "_inferred": True}
                    return checks[chosen]
            cone = cone_type([contacts[i] for i in chosen], origin=origin)
            if method == "dual":
                # Rejected contacts contribute no rays. Different subsets can
                # therefore ask exactly the same LP/uncertainty-box question.
                # Keep row order, duplicates and every bit (including signed
                # zero); only the excluded-name diagnostic is subset-specific.
                key = cone.rays.tobytes()
                if key not in geometry_checks:
                    geometry_checks[key] = cone.check(wrench, uncertainty)
                checks[chosen] = dict(geometry_checks[key], excluded_geometry=list(cone.excluded))
            else:
                checks[chosen] = cone.check(wrench, uncertainty)
            if detail != "full":
                result = checks[chosen]
                if result["status"] == "feasible" and not any(
                    mask & known == known for known in feasible_masks
                ):
                    feasible_masks[:] = [known for known in feasible_masks if known & mask != mask]
                    feasible_masks.append(mask)
                elif (result["status"] == "infeasible"
                      and result.get("max_scaled_violation", 0.0) > 1e-6
                      and not any(mask & known == mask for known in infeasible_masks)):
                    infeasible_masks[:] = [known for known in infeasible_masks if known & mask != known]
                    infeasible_masks.append(mask)
        return checks[chosen]

    full_result = check(full)
    if detail != "full" and (
        full_result["status"] == "uncertain"
        or (full_result["status"] == "infeasible"
            and full_result["max_scaled_violation"] > 1e-6)
    ):
        # A subset's cone is contained in the full cone. If the full cone
        # cannot certify feasibility of the wrench box, neither can a
        # subset. Keep borderline infeasibility on the exhaustive path to
        # avoid amplifying numerical classification at the LP tolerance.
        roles = {name: ("geometry_rejected" if name in full_result["excluded_geometry"]
                        else "unexplained") for name in names} if detail == "source" else {}
        return {"roles": roles, "full": full_result, "flight": None,
                "minimal_feasible_sets": [], "hypotheses": [],
                "physical_execution_verified": False, "force_distribution_inferred": False}
    if detail == "full":
        check(())
    if detail == "full":
        for i in full:
            check(tuple(j for j in full if j != i))
    if method == "dual":
        accepted = tuple(
            i for i in full if names[i] not in checks[full]["excluded_geometry"]
        )
        seeds = []
        if detail != "full" and full_result["status"] == "feasible":
            by_name = {names[i]: i for i in accepted}
            for prior in prior_minimal:
                if any(name not in by_name for name in prior):
                    continue
                seed = tuple(sorted(by_name[name] for name in prior))
                seeds.append(seed)
            # A feasible S whose every member is necessary in the full set
            # is the only minimal set. All strict subsets of full omitting an
            # S member are infeasible, so no other alternative can exist.
            if (len(seeds) == 1
                    and check(seeds[0])["status"] == "feasible"
                    and all(check(tuple(i for i in seeds[0] if i != k))["status"] == "infeasible"
                            for k in seeds[0])
                    and all(check(tuple(i for i in full if i != k))["status"] == "infeasible"
                            for k in seeds[0])):
                minimal = [seeds[0]]
            else:
                minimal = _minimal_dual_subsets(accepted, check, seeds)
        else:
            minimal = _minimal_dual_subsets(accepted, check)
    else:
        for count in range(len(contacts) + 1):
            for chosen in combinations(range(len(contacts)), count):
                check(chosen)
        minimal = None
    if detail != "full":
        if detail == "source":
            roles = {}
            minimal_names = [set(names[i] for i in indices) for indices in minimal]
            for i, name in enumerate(names):
                if name in full_result["excluded_geometry"]:
                    roles[name] = "geometry_rejected"
                elif full_result["status"] == "feasible" and minimal_names:
                    roles[name] = ("necessary_support_candidate"
                                   if all(name in support for support in minimal_names)
                                   else "optional_contact")
                else:
                    roles[name] = "unexplained"
        else:
            roles = {}
        return {
            "roles": roles,
            "full": full_result,
            "flight": None,
            "minimal_feasible_sets": [[names[i] for i in indices] for indices in minimal],
            "hypotheses": [],
            "physical_execution_verified": False,
            "force_distribution_inferred": False,
        }
    roles = {}
    for i, name in enumerate(names):
        removed = checks[tuple(j for j in full if j != i)]
        all_status = checks[full]["status"]
        if name in checks[full]["excluded_geometry"]:
            roles[name] = "geometry_rejected"
        elif all_status == "feasible" and removed["status"] == "infeasible":
            roles[name] = "necessary_support_candidate"
        elif all_status == "feasible" and removed["status"] == "feasible":
            roles[name] = "optional_contact"
        else:
            roles[name] = "unexplained"
    feasible = [
        indices
        for indices, result in checks.items()
        if result["status"] == "feasible" and not result["excluded_geometry"]
    ]
    feasible_sets = [set(indices) for indices in feasible]
    minimal = [
        indices
        for indices, members in zip(feasible, feasible_sets)
        if not any(other < members for other in feasible_sets)
    ]
    return {
        "roles": roles,
        "full": checks[full],
        "flight": checks[()],
        "minimal_feasible_sets": [[names[i] for i in indices] for indices in sorted(minimal)],
        "hypotheses": [
            dict(contacts=[names[i] for i in indices], **result)
            for indices, result in checks.items()
        ],
        "physical_execution_verified": False,
        "force_distribution_inferred": False,
    }
