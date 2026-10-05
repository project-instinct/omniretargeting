"""Exact rigid-geom fallback for unsigned static shell-sample sides."""

from types import SimpleNamespace
import os

import mujoco
import numpy as np

from ..geometry_query import OPEN_SURFACE_GEOMS, GeometryQueryError, RigidGeometryQuery
from ..scene_geometry import SceneQuery


WITNESS_CLEARANCE = 0.008


def rigid_robot_data(data):
    """Snapshot only the original collision transforms used by rigid queries."""
    return SimpleNamespace(geom_xpos=data.geom_xpos.copy(), geom_xmat=data.geom_xmat.copy())


def enabled(scene):
    # A custom source query owns its uncertainty semantics; only the repository's
    # immutable SceneQuery is known to have unsigned components.
    return (type(scene) is SceneQuery
            and bool(getattr(scene, "has_unsigned_only_components", False)))


def _backend(geometry):
    owner, backend = getattr(geometry, "_rigid_static_backend", None) or (None, None)
    if backend is not None and owner == os.getpid():
        return backend
    backend = RigidGeometryQuery(geometry.model, geometry.objective.shell_geoms, geometry.scene)
    geometry._rigid_static_backend = (os.getpid(), backend)
    return backend


def uncovered_components(scene, backend):
    """Components whose signed information the rigid fallback cannot reproduce.

    Closed convex parts are exact MuJoCo pairs; an unsigned-only open part has
    no side at all, so its finite-triangle crossing test is its whole answer.
    A closed concave, evidence-backed or nearly closed part keeps a side the
    crossing test would ignore: such a scene stays unknown.
    """
    covered = set(backend.scene_ids) | {
        cid for cid, _part, _faces in backend.open_parts if scene._unsigned_only(cid)}
    return [cid for cid in range(len(scene.components)) if cid not in covered]


def _failure_category(rows, failures, unsupported):
    if unsupported:
        return "rigid_unsupported_geom_type"
    if failures:
        return "rigid_backend_failure"
    # Only intersecting exits can tie; a separated uncertain row lacks a certificate.
    if any(row["clearance"] >= 0. for row in rows):
        return "rigid_uncertified_witness"
    return "rigid_ambiguous_exit"


def resolve_unknown_static(geometry, frame, geom, summary, signature, robot_data,
                           witness_clearance):
    """Resolve an unsigned shell-sample side only with an exact geom query."""
    geom, frame = int(geom), int(frame)
    coverage = geometry.static_query_coverage
    coverage["sample_unknown"] += 1
    body = geometry.model.body(int(geometry.model.geom_bodyid[geom])).name
    kind = int(geometry.model.geom_type[geom])
    record_example(geometry, {
        "category": "shell_sample_unknown_side", "frame": frame, "geom": geom,
        "body": body, "geom_type": kind, "backend": "SceneQuery shell samples",
        "detail": "open scene component without free-side evidence or near closure",
    })
    key = (frame, geom, signature)
    cached = geometry._rigid_static_cache.get(key)
    if cached is None:
        try:
            backend = _backend(geometry)
            uncovered = uncovered_components(geometry.scene, backend)
            unsupported = bool(backend.open_parts and kind not in OPEN_SURFACE_GEOMS)
            before = len(backend.failures)
            rows = [] if unsupported or uncovered else backend.query(
                robot_data, witness_clearance, active_geom_ids=[geom])
            failures = [str(value) for value in backend.failures[before:]]
            uncertain = [row for row in rows if row.get("uncertain")]
            cached = {"rows": rows, "failures": failures,
                      "unknown": bool(uncovered or unsupported or failures or uncertain)}
            if uncovered:
                cached.update(category="rigid_uncovered_scene_components",
                              detail=f"signed components {uncovered[:8]} need the sample query")
            elif cached["unknown"]:
                cached["category"] = _failure_category(uncertain, failures, unsupported)
                cached["detail"] = (
                    f"no exact open-surface query for MuJoCo geom type {kind}" if unsupported
                    else "; ".join(failures) if failures else
                    "; ".join(
                        f"component {row['obstacle_ids']} clearance {row['clearance']:.6g} m"
                        + (f" witness gap {row['witness_gap']:.3g} m" if "witness_gap" in row else "")
                        for row in uncertain))
            geometry._rigid_static_cache[key] = cached
        except (GeometryQueryError, ValueError, RuntimeError) as error:
            cached = {"rows": [], "failures": [f"{type(error).__name__}: {error}"], "unknown": True,
                      "category": "rigid_backend_failure",
                      "detail": f"{type(error).__name__}: {error}"}
            geometry._rigid_static_cache[key] = cached
    rows = [dict(row) for row in cached["rows"]]
    if cached["unknown"]:
        coverage["rigid_unknown"] += 1
        coverage[cached["category"]] = coverage.get(cached["category"], 0) + 1
        summary["query_failures"] = [f"{cached['category']}: {cached['detail']}"]
        record_example(geometry, {
            "category": cached["category"], "frame": frame, "geom": geom,
            "body": body, "geom_type": kind, "backend": "RigidGeometryQuery",
            "detail": cached["detail"],
        })
        return
    coverage["rigid_resolved"] += 1
    coverage["rigid_rows"] += len(rows)
    summary.update(unknown=False, clearance=witness_clearance, row=None, rows=[])
    rows.sort(key=lambda row: float(row["clearance"]))
    witnesses = [{
        "clearance": float(row["clearance"]),
        "point": np.asarray(row["points"], dtype=float).copy(),
        "direction": np.asarray(row["directions"], dtype=float).copy(),
        "sample_id": -1, "obstacle_id": int(row.get("obstacle_ids", -1)),
        "surface_id": int(row.get("surface_ids", -1)),
    } for row in rows]
    if witnesses:
        summary["clearance"] = min(witness_clearance, witnesses[0]["clearance"])
        summary["row"] = None if witnesses[0]["clearance"] >= witness_clearance else witnesses[0]
        summary["rows"] = witnesses[:15] if geometry.manifold else []
    record_example(geometry, {
        "category": "rigid_geom_query_resolved", "frame": frame, "geom": geom,
        "body": body, "geom_type": kind, "backend": "RigidGeometryQuery",
        "clearance_m": float(summary["clearance"]), "rows": len(witnesses),
    })


EXAMPLES_PER_CATEGORY = 3


def record_example(geometry, row):
    """Keep the first few examples of every category (all frames, all geoms)."""
    same = sum(example["category"] == row["category"] for example in geometry.static_query_examples)
    if same < EXAMPLES_PER_CATEGORY:
        geometry.static_query_examples.append(row)


def query_summary(geometry):
    describe = getattr(geometry.scene, "describe", None)
    return {
        "query_coverage": dict(geometry.static_query_coverage),
        "query_failure_examples": list(geometry.static_query_examples),
        "original_geometry_backend": {
            "sample_backend": describe()["backend"] if callable(describe) else type(geometry.scene).__name__,
            "unsigned_only_components": enabled(geometry.scene),
            "rigid_fallback": "exact MuJoCo geom pairs / capsule-axis and solid-cylinder finite triangles",
        },
    }
