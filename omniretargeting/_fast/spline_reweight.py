"""Necessary reweighting for the shared spline/free-body iteration."""

import numpy as np


class SharedReweight:
    """Callable accepted-update reweight hook used by the shared spline loop.

    Posture rows are weighted once per originally flagged frame.  Chatter is
    normally already a hard guard; if it nevertheless appears, retain the old
    global rewriter for at most its original number of rounds.
    """

    def __init__(self, objective, fixed, shape, record_list):
        self.objective = objective
        self.fixed = np.array(fixed, dtype=float, copy=True)
        self.fixed.setflags(write=False)
        self.shape = shape
        self.records = record_list
        self.posture_weighted = np.zeros(len(self.fixed), bool)
        self.chatter_rounds = 0

    def _weights(self):
        objective = self.objective
        return tuple(np.array(value, copy=True) for value in (
            objective.envelope_weight,
            *objective.joint_temporal_weight,
            *objective.root_translation_weight,
            *objective.root_rotation_weight,
        ))

    def __call__(self, motion, phase):
        # Local imports preserve the production definitions and honor test or
        # runtime monkeypatches without making spline_reweight import the
        # pipeline at module-import time.
        from .pipeline import REWEIGHT_FACTOR, REWEIGHT_ROUNDS, verify_budgets
        from .reweight import reweight_temporal_signals

        before = verify_budgets(self.objective.model, self.objective.coords,
                                self.fixed, motion,
                                self.objective.envelope_bodies, self.shape)
        budgets, chatter, posture = before
        new_posture = np.asarray(posture, bool) & ~self.posture_weighted
        chatter_used = False
        old_weights = self._weights()

        if new_posture.any():
            self.objective.envelope_weight[new_posture] *= REWEIGHT_FACTOR
            self.posture_weighted |= new_posture

        if chatter.any() and self.chatter_rounds < REWEIGHT_ROUNDS:
            chatter_used = True
            self.chatter_rounds += 1
            reweighted = reweight_temporal_signals(
                self.objective, self.fixed, motion, REWEIGHT_FACTOR)
        else:
            reweighted = None

        changed = any(not np.array_equal(left, right) for left, right
                      in zip(old_weights, self._weights()))
        if not changed:
            return False

        self.records.append(dict(
            phase=phase,
            flagged_frames=np.flatnonzero(new_posture).tolist(),
            posture_flagged_frames=np.flatnonzero(new_posture).tolist(),
            chatter_flagged_frames=np.flatnonzero(np.asarray(chatter, bool)).tolist(),
            before=budgets,
            necessary_rows_only=not chatter_used,
            no_reference_rebase=True,
            chatter_global_fallback=chatter_used,
            chatter_reweighted_signals=reweighted,
        ))
        return True
