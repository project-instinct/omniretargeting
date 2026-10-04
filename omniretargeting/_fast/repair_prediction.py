"""Stop attempts whose predicted group-depth decrease is below useful progress.

The coordinator approved default gating after four long and six OMOMO clips:
1209 repair fell from 873 to 350 seconds; worst per-item penetration/contact
increases were 3.22/3.10 mm, with no new body/category defect above 1 cm and
zero windowed-budget violations. It applies the existing progress threshold
to the linear witness model, trading small residual changes for lower cost.
A linear prediction is not a nonlinear clearance certificate.
"""

import numpy as np

PREDICTIVE_GATING = True
RECORD_PREDICTIONS = False


def linear_prediction(geometry, measurement, step, body, entry_depth):
    """Maximum unilateral depth at the full QP step, in the witness model.

    Keep unrepresented penetrating keys at their measured depth. Missing
    witnesses must not invent clearance. Include every relevant witness,
    rather than averaging rows or allowing sample multiplicity to vote.
    """
    collision, phi = geometry.linear_rows(measurement)
    clearance = phi + np.asarray(collision @ step.ravel()).ravel()
    if not np.isfinite(clearance).all():
        return None
    model = geometry.model

    def relevant(key):
        geoms = key[2:] if key[1] == "self" else key[-1:]
        return body is None or any(model.body_weldid[model.geom_bodyid[g]] == body for g in geoms)

    represented = set()
    predicted = 0.
    for (key, _), value in zip(measurement["rows"], clearance, strict=True):
        if relevant(key):
            represented.add(key)
            predicted = max(predicted, -float(value))
    for key, depth in measurement["depths"].items():
        if relevant(key) and key not in represented:
            predicted = max(predicted, float(depth))
    decrease = float(entry_depth) - predicted
    return {"entry_depth_m": float(entry_depth), "predicted_depth_m": predicted,
            "predicted_decrease_m": decrease, "model": "full_qp_linear_witness_maximum"}


def should_gate(prediction, threshold):
    return bool(PREDICTIVE_GATING and prediction is not None and threshold > 0
                and np.isfinite(prediction["predicted_decrease_m"])
                and prediction["predicted_decrease_m"] < threshold)
