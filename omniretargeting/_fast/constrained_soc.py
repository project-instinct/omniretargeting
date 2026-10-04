"""Second-order guard remainders for the banded quadratic step model."""

from dataclasses import replace

import numpy as np


def shifted(original, measured, step, diagnostics):
    """Retain original Jacobians and measured nonlinear row remainders.

    This adjusts the proposal, never the physical guard. Complete vector
    remainders are needed for balls. Scalar joint stencils have zero remainder.
    Every original row, including crossing temporal rows, remains in the QP.
    """
    result, count, maximum = [], 0, 0.
    for before, after in zip(original, measured):
        if before.ball:
            predicted = before.residual + (before.matrix @ step).reshape(before.residual.shape)
            radius = np.sqrt(np.maximum(before.slack + np.square(before.residual).sum(axis=1), 0.))
            remainder = after.residual - predicted
            bad = np.linalg.norm(remainder, axis=1) > 1e-12
            value = before.residual + np.where(bad[:, None], remainder * 1.01, 0.)
            result.append(replace(before, residual=value,
                                  slack=radius**2 - np.square(value).sum(axis=1)))
            count += int(bad.sum())
            maximum = max(maximum, float(np.linalg.norm(remainder[bad], axis=1).max(initial=0.)))
        else:
            predicted = before.slack + before.matrix @ step
            remainder = predicted - after.slack
            # A previously corrected face may now pass. Its remainder must
            # remain, or the next solve simply returns to the failing step.
            bad = remainder > 1e-12
            result.append(replace(before, slack=before.slack - np.where(bad, remainder * 1.01, 0.)))
            count += int(bad.sum())
            maximum = max(maximum, float(remainder[bad].max(initial=0.)))
    diagnostics['soc_rounds'] = diagnostics.get('soc_rounds', 0) + 1
    diagnostics['soc_rows'] = count
    diagnostics['soc_maximum_remainder'] = maximum
    return result if count else None
