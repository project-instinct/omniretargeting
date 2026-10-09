"""Canonical manifests for the collision geometry of a compiled robot model.

The manifest deliberately describes MuJoCo collision semantics rather than file
provenance.  A source path is retained for diagnostics but does not enter the
semantic hash, so moving an unchanged XML/URDF does not change its identity.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .scene_geometry import collision_geom_ids

_SCHEMA_VERSION = 1


def _object_name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str | None:
    name = mujoco.mj_id2name(model, kind, int(index))
    return str(name) if name else None


def _body_name(model: mujoco.MjModel, index: int) -> str | None:
    return _object_name(model, mujoco.mjtObj.mjOBJ_BODY, index)


def _array_hash(values: np.ndarray) -> str:
    """Hash a canonical little-endian float64 representation of an array."""
    canonical = np.asarray(values, dtype="<f8")
    canonical = canonical.copy()
    canonical[canonical == 0.0] = 0.0  # Give -0.0 and +0.0 the same identity.
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _exclude_pairs(model: mujoco.MjModel) -> list[list[str | None]]:
    """Decode MuJoCo exclude signatures into stable body-name pairs."""
    pairs: set[tuple[str | None, str | None]] = set()
    for signature in np.asarray(model.exclude_signature, dtype=np.int64):
        value = int(signature)
        first, second = value >> 16, value & 0xFFFF
        if first >= model.nbody or second >= model.nbody:
            # Malformed signatures must be visible rather than silently dropped.
            pairs.add((f"<invalid-body:{first}>", f"<invalid-body:{second}>"))
        else:
            pairs.add((_body_name(model, first), _body_name(model, second)))
    return [list(pair) for pair in sorted(pairs, key=lambda pair: (pair[0] or "", pair[1] or ""))]


def _mesh_record(model: mujoco.MjModel, geom_id: int) -> dict[str, Any] | None:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[start : start + count]
    return {
        "id": mesh_id,
        "name": _object_name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id),
        "vertex_count": count,
        "compiled_vertex_sha256": _array_hash(vertices),
    }


def _geom_record(model: mujoco.MjModel, geom_id: int) -> dict[str, Any]:
    body_id = int(model.geom_bodyid[geom_id])
    parent_id = int(model.body_parentid[body_id])
    weld_id = int(model.body_weldid[body_id])
    geom_type = mujoco.mjtGeom(int(model.geom_type[geom_id])).name
    return {
        "id": int(geom_id),
        "name": _object_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        "type": geom_type,
        "body": {
            "id": body_id,
            "name": _body_name(model, body_id),
            "parent_id": parent_id,
            "parent_name": _body_name(model, parent_id),
            "weld_id": weld_id,
            "weld_name": _body_name(model, weld_id),
        },
        "size": [float(value) for value in model.geom_size[geom_id]],
        "pos_in_body": [float(value) for value in model.geom_pos[geom_id]],
        "quat_in_body": [float(value) for value in model.geom_quat[geom_id]],
        "mesh": _mesh_record(model, geom_id),
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
        "condim": int(model.geom_condim[geom_id]),
        "margin": float(model.geom_margin[geom_id]),
        "gap": float(model.geom_gap[geom_id]),
    }


def _hash_manifest(manifest: dict[str, Any]) -> str:
    # source_path is provenance, not collision semantics.  Every other field is
    # JSON scalar data and participates in the canonical semantic hash.
    payload = {key: value for key, value in manifest.items() if key != "source_path"}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def collision_manifest(model: mujoco.MjModel, *, source_path: str | Path | None = None) -> dict[str, Any]:
    """Return a canonical manifest for all active robot collision geoms.

    Active follows :func:`omniretargeting.scene_geometry.collision_geom_ids`:
    non-world geoms with a nonzero ``contype`` or ``conaffinity``.  The manifest
    and its ``sha256`` cover compiled geom geometry and placement, contact
    parameters, mesh compiled-vertex hashes, body tree/weld names used by
    parent filtering, explicit excludes, and collision-related disable flags.

    It does **not** cover scenes, moving objects, task CollisionCfg patches,
    training interpolation/reset, actuators, visual assets, or file paths.
    ``source_path`` is retained only as diagnostic provenance and is excluded
    from ``sha256``.
    """
    geom_ids = collision_geom_ids(model)
    disableflags = int(model.opt.disableflags)
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "source_path": None if source_path is None else str(source_path),
        "mujoco_version": mujoco.__version__,
        "active_geom_count": len(geom_ids),
        "geoms": [_geom_record(model, geom_id) for geom_id in geom_ids],
        "exclude_pairs": _exclude_pairs(model),
        "disableflags": {
            "raw": disableflags,
            "filterparent_disabled": bool(
                disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
            ),
            "nativeccd_disabled": bool(
                disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
            ),
        },
        "coverage": (
            "compiled active robot collision geoms, parent/weld topology, "
            "explicit excludes, and collision-related MuJoCo disable flags"
        ),
        "not_covered": (
            "scene and moving-object geometry, downstream task collision "
            "patches, training interpolation/reset, physics solver settings, "
            "and source-file paths"
        ),
    }
    manifest["sha256"] = _hash_manifest(manifest)
    return manifest


def collision_manifest_difference(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    """Summarize count/type/body differences between two collision manifests."""
    first_types = Counter(geom["type"] for geom in first.get("geoms", []))
    second_types = Counter(geom["type"] for geom in second.get("geoms", []))
    first_bodies = Counter(
        geom["body"]["name"] for geom in first.get("geoms", [])
    )
    second_bodies = Counter(
        geom["body"]["name"] for geom in second.get("geoms", [])
    )

    def count_differences(left: Counter, right: Counter) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for key in sorted(set(left) | set(right)):
            before, after = int(left.get(key, 0)), int(right.get(key, 0))
            if before != after:
                result[key] = {"first": before, "second": after, "delta": after - before}
        return result

    return {
        "geom_count": {
            "first": int(first.get("active_geom_count", len(first.get("geoms", [])))),
            "second": int(second.get("active_geom_count", len(second.get("geoms", [])))),
            "delta": int(second.get("active_geom_count", len(second.get("geoms", []))))
            - int(first.get("active_geom_count", len(first.get("geoms", [])))),
        },
        "geom_type_counts": {
            "first": dict(sorted(first_types.items())),
            "second": dict(sorted(second_types.items())),
        },
        "geom_type_differences": count_differences(first_types, second_types),
        "body_geom_count_differences": count_differences(first_bodies, second_bodies),
    }


def compare_collision_manifests(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    """Compare two manifests and return one of matched/mismatched."""
    matched = first.get("sha256") == second.get("sha256")
    return {
        "status": "matched" if matched else "mismatched",
        "first_sha256": first.get("sha256"),
        "second_sha256": second.get("sha256"),
        "difference": None if matched else collision_manifest_difference(first, second),
    }
