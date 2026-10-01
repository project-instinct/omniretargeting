"""Transfer bounded, necessary source support proposals to surface targets.

Observed skin stays immutable and separate from the inferred scene target.
Optional load explanations alone never create a previously unobserved contact.
"""

import numpy as np


def apply_support_observations(observation, points, valid, evidence):
    out = {
        k: (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in observation.items()
    }
    targets = np.array(points, float, copy=True)
    if evidence.frames != len(points) or len(evidence.candidates) != len(points):
        raise ValueError("Support proposals must share the source surface timeline")
    confirmed, unmapped, assignments = [], [], {}
    for f, candidates in enumerate(evidence.candidates):
        for candidate in candidates:
            if (
                evidence.explanations[f]["roles"][candidate.name]
                != "necessary_support_candidate"
            ):
                continue
            if candidate.sample_indices is None:
                unmapped.append({"frame": f, "candidate": candidate.name})
                continue
            ids = np.asarray(candidate.sample_indices)
            observed = np.asarray(candidate.observed_points, float).reshape(-1, 3)
            surface = np.asarray(candidate.surface_points, float).reshape(-1, 3)
            if (
                ids.shape != (len(observed),)
                or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0)
                or np.any(ids >= points.shape[1])
                or len(np.unique(ids)) != len(ids)
            ):
                raise ValueError(
                    "Support sample identities must match the source patch"
                )
            if not valid[f, ids].all() or not np.allclose(
                points[f, ids], observed, atol=1e-8, rtol=0
            ):
                raise ValueError(
                    "Support evidence does not match immutable source observations"
                )
            normals = np.broadcast_to(
                np.asarray(candidate.normals, float), surface.shape
            )
            region, surface_id = evidence.regions[candidate.name]
            for k, index in enumerate(ids):
                key = (f, int(index))
                assignment = (surface_id, tuple(surface[k]), tuple(normals[k]))
                if key in assignments and assignments[key] != assignment:
                    raise ValueError(
                        "Conflicting source support targets for one material sample"
                    )
                assignments[key] = assignment
            out["active"][f, ids] = True
            out["environment"][f, ids] = surface
            out["normals"][f, ids] = normals / np.linalg.norm(
                normals, axis=1, keepdims=True
            )
            out["surface_ids"][f, ids] = surface_id
            out["modes"][f, ids] = candidate.mode
            out["confidence"][f, ids] = 1.0
            targets[f, ids] = surface
            confirmed.append(
                {
                    "frame": f,
                    "candidate": candidate.name,
                    "region": region,
                    "samples": ids.tolist(),
                    "surface_id": surface_id,
                }
            )
    out["support_inference"] = {
        "confirmed": confirmed,
        "unmapped": unmapped,
        "observations_modified": False,
    }
    return out, targets
