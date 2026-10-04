"""Experimental float64 device accumulation of the original row families.

CUDA reductions change round-off enough to affect sensitive conic proposals.
The validated evaluator therefore uses the CPU accumulator by default.
"""

import numpy as np
import torch


class GPUAccumulator:
    """Per-frame normal equations, retaining the original family/body order."""

    def __init__(self, frames, width, with_jacobian, kin):
        self.jacobian = with_jacobian
        self.device = 'cuda'
        self.frame_cost = torch.zeros(frames, dtype=torch.float64, device=self.device)
        self.hessian = self.gradient = None
        if with_jacobian:
            self.hessian = torch.zeros((frames, width, width), dtype=torch.float64, device=self.device)
            self.gradient = torch.zeros((frames, width), dtype=torch.float64, device=self.device)
            self.jacp, self.jacr = self.tensor(kin.jacp), self.tensor(kin.jacr)

    def tensor(self, value):
        if torch.is_tensor(value):
            return value
        array = np.ascontiguousarray(value)
        if not array.flags.writeable:
            array = array.copy()
        return torch.as_tensor(array, dtype=torch.float64, device=self.device)

    def _add_cost(self, residual):
        residual = self.tensor(residual)
        self.frame_cost += residual.square().reshape(len(residual), -1).sum(dim=1)
        return residual

    def add(self, residual, jacobian=None):
        residual = self._add_cost(residual)
        if self.jacobian:
            jacobian = self.tensor(jacobian)
            self.hessian += jacobian.transpose(1, 2) @ jacobian
            self.gradient += (residual[:, None, :] @ jacobian)[:, 0]

    def add_body_rows(self, residual, coefficients, bodies, kin):
        residual = self._add_cost(residual)
        if not self.jacobian:
            return
        coefficients = self.tensor(coefficients)
        for body in np.unique(bodies):
            index = torch.as_tensor(np.flatnonzero(bodies == body), device=self.device)
            rows = coefficients[:, index]
            spatial = torch.cat((self.jacp[:, body], self.jacr[:, body]), dim=1)
            moment = rows.transpose(1, 2) @ rows
            pull = residual[:, None, index] @ rows
            self.hessian += spatial.transpose(1, 2) @ (moment @ spatial)
            self.gradient += (pull @ spatial)[:, 0]

    def add_normals(self, frame_cost, hessian, gradient):
        """Insert independently measured self-collision rows at their original position."""
        self.frame_cost += self.tensor(frame_cost)
        if self.jacobian:
            self.hessian += self.tensor(hessian)
            self.gradient += self.tensor(gradient)

    def finish(self, accumulator_type):
        result = accumulator_type(len(self.frame_cost), 0, False)
        result.jacobian = self.jacobian
        result.frame_cost = self.frame_cost.cpu().numpy()
        result.cost = float(result.frame_cost.sum())
        if self.jacobian:
            result.hessian = self.hessian.cpu().numpy()
            result.gradient = self.gradient.cpu().numpy()
        return result
