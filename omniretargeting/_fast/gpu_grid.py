"""Trilinear grid evaluation using the reference's arithmetic and dtype.

Coordinates are formed on the CPU exactly as GridSDF.query forms them.
The eight double-precision corner products follow SciPy's axis/corner order;
the final scalar is rounded to float32 just as map_coordinates rounds it.
Unknown corners participate even at zero weight. No fused multiply/add or
replacement distance/sign approximation is used.
"""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _lookup(grid, x, y, z, active, sx: tl.constexpr, sy: tl.constexpr,
            sz: tl.constexpr, block: tl.constexpr):
    x = tl.minimum(tl.maximum(x, 0), sx - 1)
    y = tl.minimum(tl.maximum(y, 0), sy - 1)
    z = tl.minimum(tl.maximum(z, 0), sz - 1)
    fx, fy, fz = tl.floor(x), tl.floor(y), tl.floor(z)
    bx, by, bz = fx.to(tl.int64), fy.to(tl.int64), fz.to(tl.int64)
    tx, ty, tz = x - fx, y - fy, z - fz
    total = tl.full((block,), 0, tl.float64)
    for corner in tl.static_range(8):
        dx, dy, dz = (corner >> 2) & 1, (corner >> 1) & 1, corner & 1
        ix = tl.minimum(bx + dx, sx - 1)
        iy = tl.minimum(by + dy, sy - 1)
        iz = tl.minimum(bz + dz, sz - 1)
        value = tl.load(grid + (ix * sy + iy) * sz + iz, active, other=0).to(tl.float64)
        value = value * (tx if dx else 1.0 - tx)
        value = value * (ty if dy else 1.0 - ty)
        value = value * (tz if dz else 1.0 - tz)
        total = total + value
    return total.to(tl.float32)


@triton.jit(do_not_specialize=['count'])
def _trilinear(grid, coordinates, output, count,
               sx: tl.constexpr, sy: tl.constexpr, sz: tl.constexpr,
               block: tl.constexpr):
    row = tl.program_id(0) * block + tl.arange(0, block)
    active = row < count
    x = tl.load(coordinates + row * 3, active, other=0)
    y = tl.load(coordinates + row * 3 + 1, active, other=0)
    z = tl.load(coordinates + row * 3 + 2, active, other=0)
    total = _lookup(grid, x, y, z, active, sx, sy, sz, block)
    tl.store(output + row, total, active)


class GPUGrid:
    """Query accelerator; analytical gradients keep the reference implementation."""

    def __init__(self, reference, device='cuda'):
        self.reference, self.device = reference, device
        self.lower, self.spacing, self.shape = reference.lower, reference.spacing, reference.shape
        self.grid = torch.as_tensor(np.ascontiguousarray(reference.clearance), device=device)
        self.flat = self.grid.flatten()
        self.maximum = torch.as_tensor(self.shape - 1, device=device)

    def query(self, points, gradient=True):
        points = np.asarray(points, dtype=float)
        flat = points.reshape(-1, 3)
        index = ((flat - self.lower) / self.spacing).T
        inside = np.all((index >= 0) & (index <= (self.shape - 1)[:, None]), axis=0)
        result = np.empty(len(flat), np.float32)
        # Bound temporary tensors for the widest collision shells.
        for first in range(0, len(flat), 1048576):
            chunk = np.ascontiguousarray(index[:, first:first + 1048576].T)
            if not np.isfinite(chunk).all():
                chunk = np.nan_to_num(chunk, nan=0., posinf=0., neginf=0.)
            coordinates = torch.as_tensor(chunk,
                                          device=self.device)
            values = torch.empty(len(coordinates), dtype=torch.float32, device=self.device)
            _trilinear[(triton.cdiv(len(coordinates), 256),)](
                self.flat, coordinates, values, len(coordinates),
                *(int(s) for s in self.shape), 256, enable_fp_fusion=False)
            result[first:first + len(coordinates)] = values.cpu().numpy()
        result = np.where(inside, result, np.nan).reshape(points.shape[:-1])
        return (result, self.gradient(points)) if gradient else result

    def gradient(self, points):
        return self.reference.gradient(points)

    def value_gradient(self, points):
        return self.reference.value_gradient(points)
