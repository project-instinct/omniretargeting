"""One sparse CVXPY constraint for exact scene collision halfspaces."""

import cvxpy as cp
import numpy as np
from scipy.sparse import coo_matrix


class SceneCollisionRows:
    def __init__(self, frame_count, joint_count):
        self.frame_count = frame_count
        self.joint_count = joint_count
        self.indices = []
        self.columns = []
        self.values = []
        self.clearances = []
        self.targets = []
        self.slack_caps = []
        self.slack = None

    def append(self, gradient, frame_weights, clearance, target, max_slack):
        row = len(self.clearances)
        gradient = np.asarray(gradient, dtype=float)
        for frame, weight in frame_weights:
            nonzero = np.flatnonzero(gradient)
            self.indices.extend([row] * len(nonzero))
            self.columns.extend((int(frame) * self.joint_count + nonzero).tolist())
            self.values.extend((float(weight) * gradient[nonzero]).tolist())
        self.clearances.append(float(clearance))
        self.targets.append(float(target))
        self.slack_caps.append(float(max_slack))

    def add_to_problem(self, step, constraints, objective_terms, sigma):
        count = len(self.clearances)
        if not count:
            return
        matrix = coo_matrix(
            (self.values, (self.indices, self.columns)),
            shape=(count, self.frame_count * self.joint_count),
        ).tocsr()
        self.slack = cp.Variable(count, nonneg=True)
        constraints.extend([
            matrix @ cp.vec(step, order="C") + np.asarray(self.clearances)
            >= np.asarray(self.targets) - self.slack,
            self.slack <= np.asarray(self.slack_caps),
        ])
        objective_terms.append(cp.sum_squares(self.slack / sigma))

    def solved_slacks(self):
        return [] if self.slack is None or self.slack.value is None else self.slack.value.tolist()
