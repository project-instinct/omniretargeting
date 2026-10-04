"""Fused shell transformation, exact grid interpolation and first minimum.

The small centre broadphase and selected-point reconstruction use the
reference NumPy definitions. Dense sample coordinates never cross PCIe.
All arithmetic is float64 with fusion disabled; grid values round to
float32 before object scaling, as in GridSDF.query.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from .gpu_grid import GPUGrid, _lookup
from .terms import NEAR_MARGIN


@triton.jit
def _div64(a, b):
    return tl.inline_asm_elementwise('div.rn.f64 $0, $1, $2;', constraints='=d,d,d',
                                    args=[a, b], dtype=tl.float64, is_pure=True, pack=1)


@triton.jit(do_not_specialize=['count', 'samples', 'geom'])
def _shell_values(grid, local, frames, position, rotation, field_rotation,
                  field_translation, scales, bounds, output, count,
                  samples, geom, ngeom: tl.constexpr,
                  moving: tl.constexpr, scaled: tl.constexpr,
                  sx: tl.constexpr, sy: tl.constexpr, sz: tl.constexpr,
                  block: tl.constexpr):
    row = tl.program_id(0) * block + tl.arange(0, block)
    active = row < count
    frame = tl.load(frames + row // samples, active, other=0)
    sample = row % samples
    a = tl.load(local + sample * 3, active, other=0)
    b = tl.load(local + sample * 3 + 1, active, other=0)
    c = tl.load(local + sample * 3 + 2, active, other=0)
    base = (frame * ngeom + geom) * 9
    x = ((tl.load(rotation + base, active, other=0) * a
          + tl.load(rotation + base + 1, active, other=0) * b)
         + tl.load(rotation + base + 2, active, other=0) * c)
    y = ((tl.load(rotation + base + 3, active, other=0) * a
          + tl.load(rotation + base + 4, active, other=0) * b)
         + tl.load(rotation + base + 5, active, other=0) * c)
    z = ((tl.load(rotation + base + 6, active, other=0) * a
          + tl.load(rotation + base + 7, active, other=0) * b)
         + tl.load(rotation + base + 8, active, other=0) * c)
    base = (frame * ngeom + geom) * 3
    x += tl.load(position + base, active, other=0)
    y += tl.load(position + base + 1, active, other=0)
    z += tl.load(position + base + 2, active, other=0)
    if moving:
        x -= tl.load(field_translation + frame * 3, active, other=0)
        y -= tl.load(field_translation + frame * 3 + 1, active, other=0)
        z -= tl.load(field_translation + frame * 3 + 2, active, other=0)
        base = frame * 9
        u = ((tl.load(field_rotation + base, active, other=0) * x
              + tl.load(field_rotation + base + 3, active, other=0) * y)
             + tl.load(field_rotation + base + 6, active, other=0) * z)
        v = ((tl.load(field_rotation + base + 1, active, other=0) * x
              + tl.load(field_rotation + base + 4, active, other=0) * y)
             + tl.load(field_rotation + base + 7, active, other=0) * z)
        w = ((tl.load(field_rotation + base + 2, active, other=0) * x
              + tl.load(field_rotation + base + 5, active, other=0) * y)
             + tl.load(field_rotation + base + 8, active, other=0) * z)
        x, y, z = u, v, w
        if scaled:
            scale = tl.load(scales + frame, active, other=1)
            x, y, z = _div64(x, scale), _div64(y, scale), _div64(z, scale)
    x = _div64(x - tl.load(bounds), tl.load(bounds + 3))
    y = _div64(y - tl.load(bounds + 1), tl.load(bounds + 4))
    z = _div64(z - tl.load(bounds + 2), tl.load(bounds + 5))
    inside = (x >= 0) & (x <= sx - 1) & (y >= 0) & (y <= sy - 1) & (z >= 0) & (z <= sz - 1)
    value = _lookup(grid, x, y, z, active & inside, sx, sy, sz, block)
    if moving and scaled:
        value = value.to(tl.float64) * scale
    value = tl.where(inside & (tl.abs(value) < float('inf')), value, float('inf'))
    tl.store(output + row, value, active)


class GPUShell:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.local = {}

    def prepare(self, kin):
        self.position = torch.as_tensor(np.array(kin.geom_xpos), device='cuda')
        self.rotation = torch.as_tensor(np.array(kin.geom_xmat), device='cuda')

    def deepest(self, sdf, rotation, translation, scales=None):
        obj = self.evaluator.objective
        if not isinstance(sdf, GPUGrid):
            return obj.__class__._deepest_samples(obj, sdf, rotation, translation, scales)
        kin, geoms = obj.kin, np.asarray(obj.shell_geoms)

        def to_field(points, frame):
            if rotation is None:
                return points
            local = np.einsum('nji,nsj->nsi', rotation[frame], points - translation[frame, None])
            return local if scales is None else local / scales[frame, None, None]

        frame_all = np.arange(obj.frames).repeat(len(geoms))
        centres = kin.geom_xpos[:, geoms].reshape(-1, 1, 3)
        centre_value = sdf.reference.query(
            to_field(centres, frame_all)[:, 0].reshape(obj.frames, len(geoms), 3), gradient=False)
        if scales is not None:
            centre_value = centre_value * scales[:, None]
        near = np.isfinite(centre_value) & (centre_value < obj.shell_radius + obj.cfg['scene_clearance_m'] + NEAR_MARGIN)
        clearance = np.full(near.shape, np.inf)
        frame_index, geom_index = np.nonzero(near)
        world, field = np.zeros((len(frame_index), 3)), np.zeros((len(frame_index), 3))
        if not len(frame_index):
            return clearance, frame_index, geom_index, world, field
        bounds = torch.as_tensor(np.r_[sdf.lower, np.full(3, sdf.spacing)], device='cuda')
        field_r = torch.as_tensor(rotation, device='cuda') if rotation is not None else bounds
        field_t = torch.as_tensor(translation, device='cuda') if translation is not None else bounds
        field_s = torch.as_tensor(scales, device='cuda') if scales is not None else bounds
        for geom in np.unique(geom_index):
            pairs = np.flatnonzero(geom_index == geom)
            samples = obj.shell_index[geom]
            samples = samples[samples >= 0]
            if int(geom) not in self.local:
                self.local[int(geom)] = torch.as_tensor(obj.shell_local[samples], device='cuda')
            frames = frame_index[pairs]
            frame_tensor = torch.as_tensor(frames, device='cuda')
            values = torch.empty((len(pairs), len(samples)), dtype=(torch.float64 if scales is not None else torch.float32), device='cuda')
            _shell_values[(triton.cdiv(values.numel(), 256),)](
                sdf.flat, self.local[int(geom)], frame_tensor, self.position, self.rotation,
                field_r, field_t, field_s, bounds, values, values.numel(),
                len(samples), int(geoms[geom]), obj.model.ngeom, rotation is not None, scales is not None,
                *(int(s) for s in sdf.shape), 256, enable_fp_fusion=False)
            choice = values.argmin(dim=1)
            selected = values.gather(1, choice[:, None])[:, 0].cpu().numpy()
            choice = choice.cpu().numpy()
            local = obj.shell_local[samples[choice]][:, None]
            points = kin.geom_xpos[frames, geoms[geom]][:, None] + np.einsum(
                'nij,nsj->nsi', kin.geom_xmat[frames, geoms[geom]], local)
            clearance[frames, geom] = selected
            world[pairs], field[pairs] = points[:, 0], to_field(points, frames)[:, 0]
        return clearance, frame_index, geom_index, world, field
