"""Immutable summaries of exact-transform moving-surface samples.

Sample identities are shared across transforms and frame times. Summaries
retain the old iteration/reduction order; only point Jacobians depend on the
caller's current configuration and are deliberately not stored here.
"""

import numpy as np


def summarize(owner, oid, obj, gid, values, rotation, scale, band):
    world, _, components = values
    keys = owner.sample_keys.get((oid, gid))
    if keys is None:
        keys = tuple((oid, gid, i) for i in range(len(world)))
        owner.sample_keys[oid, gid] = keys
    result = dict(known=set(), penetrating=set(), unknown=0, failures=[], rows=[], depths=[])
    for cid, entry in components.items():
        if cid == "hull":
            result["known"].update(keys[int(i)] for i in entry)
            continue
        ids, signed = entry
        reliable = ~signed["uncertain"] & np.isfinite(signed["clearance"])
        closed = cid in obj["closed"]
        if closed and not reliable.all():
            result["failures"].append(f"uncertain closed object {oid} component {cid} geom {gid}")
            continue
        if not closed:
            result["unknown"] += int((~reliable).sum())
            result["known"].update(keys[int(i)] for i in ids[reliable])
            penetrating = reliable & (signed["clearance"] * scale < -1e-9)
            result["penetrating"].update(keys[int(i)] for i in ids[penetrating])
        if not reliable.any():
            continue
        k = int(np.argmin(np.where(reliable, signed["clearance"], np.inf)))
        clearance = float(signed["clearance"][k] * scale)
        result["depths"].append((cid, closed, max(0., -clearance)))
        if clearance < band:
            result["rows"].append((world[ids[k]], rotation @ signed["directions"][k],
                                   clearance, ids[k], closed, cid))
    result["known"] = frozenset(result["known"])
    result["penetrating"] = frozenset(result["penetrating"])
    return result
