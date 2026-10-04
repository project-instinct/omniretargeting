"""Certified seed supports for the reference convex-distance expansion.

Select local vertices on CUDA, but form their world coordinates with the
reference CPU BLAS. A margin covers both algebraic rearrangement and all
coordinate/three-term dot rounding, including pair-dependent recentering.
Ties retain the original recentered CPU support query.
"""

import os

import numpy as np
import torch


class GPUSelfSeeds:
    def __init__(self):
        self.pid = os.getpid()
        self.vertices = {}

    def __call__(self, batch, kin):
        frames, geoms = kin.geom_xpos.shape[:2]
        choices = np.zeros((frames, geoms, 52), np.int64)
        reliable = np.zeros(choices.shape, np.uint8)
        # Unit rotations give |R v|_inf <= |v|_2. Means and pair origins
        # lie within the same global coordinate bound. 256 eps covers the
        # two transformed dots, subtraction/addition, and both competitors.
        world_bound = np.abs(kin.geom_xpos).max(axis=(1, 2)) + batch.radius.max(initial=0) + 1.
        slack = 256 * np.finfo(float).eps * world_bound
        axes = np.r_[batch.seed, -batch.seed]
        for geom in np.flatnonzero(batch.counts):
            mesh = int(batch.model.geom_dataid[geom])
            if mesh not in self.vertices:
                local = batch.vertices[batch.starts[geom]:batch.starts[geom] + batch.counts[geom]]
                self.vertices[mesh] = torch.as_tensor(local, device='cuda')
            vertices = self.vertices[mesh]
            directions = np.matmul(kin.geom_xmat[:, geom].transpose(0, 2, 1), axes.T).transpose(0, 2, 1)
            tensor = torch.as_tensor(np.ascontiguousarray(directions.reshape(-1, 3)), device='cuda')
            geom_choices = np.empty(len(tensor), np.int64)
            geom_reliable = np.empty(len(tensor), np.uint8)
            for start in range(0, len(tensor), 8192):
                score = tensor[start:start + 8192] @ vertices.T
                index = score.argmax(dim=1).cpu().numpy()
                gap = score.topk(2, dim=1).values
                gap = (gap[:, 0] - gap[:, 1]).cpu().numpy()
                stop = start + len(index)
                geom_choices[start:stop] = index
                geom_reliable[start:stop] = gap > slack[np.arange(start, stop) // 52]
            choices[:, geom] = geom_choices.reshape(frames, 52)
            reliable[:, geom] = geom_reliable.reshape(frames, 52)
        return choices, reliable
