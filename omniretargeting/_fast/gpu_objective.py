"""Float64 batch evaluation, with the exact native collision kernel in parallel.

MuJoCo FK and all residual definitions remain the reference definitions.
Device work accelerates exact grid interpolation and certified hull support
selection. CPU reductions preserve the reference normal equations bitwise.
Near-tied support scores go through the original BLAS.
CUDA is used only by the owning process, never an inherited fork worker.
"""

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import torch
import mujoco

from .._sequence.proximity import _hull_vertices, support_points
from .gpu_grid import GPUGrid
from .gpu_rows import GPUAccumulator
from .gpu_shell import GPUShell
from .gpu_guards import acquisition_rows
from .native_point_rows import NativePointRows
from .cpu_frames import ReferenceFrames
from .gpu_self_seeds import GPUSelfSeeds
from .rows import _Accumulator
from .terms import ENVELOPE_SIGMA


class GPUFrameEvaluator:
    def __init__(self, objective):
        self.objective, self.pid = objective, os.getpid()
        self.grids = {}
        self.groups = {}
        self.vertices = {}
        self.proximity_data = {}
        self.diagnostics = {}
        self.shell = GPUShell(self)
        mode = os.environ.get('OMNIRETARGETING_FAST_MOMENTS', 'reference')
        if mode not in ('reference', 'gpu'):
            raise ValueError('OMNIRETARGETING_FAST_MOMENTS must be reference or gpu')
        # Retain NumPy's reduction order for sensitive conic linearizations.
        self.device_moments = mode == 'gpu'
        self.reference_frames = ReferenceFrames(self)
        # The CPU seed cache is faster on the 3090's limited float64 units.
        # Retain the certified device implementation for optional assessment.
        self.self_seeds = (GPUSelfSeeds() if os.environ.get('OMNIRETARGETING_FAST_SELF_SEEDS') == 'gpu'
                           else None)
        torch.backends.cuda.matmul.allow_tf32 = False

    def _group(self, prox, geom):
        key = id(prox), geom
        if key not in self.groups:
            index = np.flatnonzero(prox['geoms'] == geom)
            normals = np.ascontiguousarray(prox['normals'][:, index])
            frames, count = normals.shape[:2]
            keys = np.empty((frames * count, 4), np.uint64)
            keys[:, 0] = np.arange(frames).repeat(count)
            keys[:, 1:] = normals.reshape(-1, 3).view(np.uint64)
            # Immutable identical normals in one frame give bit-identical
            # directions in the original matmul. Reuse the same support point.
            _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
            self.groups[key] = first, inverse
        return self.groups[key]

    def support(self, prox, geom, directions):
        first, inverse = self._group(prox, geom)
        return self.support_unique(geom, np.asarray(directions, float)[first])[inverse]

    def support_unique(self, geom, directions):
        if self.objective.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            return support_points(self.objective.model, geom, directions)
        lengths = np.linalg.norm(directions, axis=1, keepdims=True)
        normalized = directions / np.maximum(lengths, 1e-12)
        mesh = int(self.objective.model.geom_dataid[geom])
        if mesh not in self.vertices:
            local = _hull_vertices(self.objective.model, mesh)
            vertices = torch.as_tensor(local, device='cuda')
            self.vertices[mesh] = local, vertices, vertices.float()
        local, vertices, vertices32 = self.vertices[mesh]
        tensor = torch.as_tensor(normalized, device='cuda')
        result = np.empty((len(normalized), 3))
        ambiguous_count = 0
        for start in range(0, len(normalized), 16384):
            block = tensor[start:start + 16384]
            # Screen only the discrete support choice in float32. Both
            # normalized directions and mesh vertices have components <=1
            # and max|v| respectively. Conversion plus a three-term dot has
            # error <8*eps32*max|v|; a 32*eps32 margin covers both competitors
            # and the reference float64 error. Uncertified rows use float64.
            score = block.float() @ vertices32.T
            index = score.argmax(dim=1)
            if len(local) > 1:
                extrema = score.topk(2, dim=1).values
                screen_slack = 32 * np.finfo(np.float32).eps * (np.abs(local).max() + 1e-12)
                uncertain = torch.nonzero(extrema[:, 0] - extrema[:, 1] <= screen_slack)[:, 0]
                exact = block[uncertain] @ vertices.T
                index[uncertain] = exact.argmax(dim=1)
                exact_extrema = exact.topk(2, dim=1).values
                # Bound both CPU and GPU three-term dot-product round-off.
                slack = 64 * np.finfo(float).eps * (np.abs(local).max() + 1)
                ambiguous_device = torch.zeros(len(block), dtype=torch.bool, device='cuda')
                ambiguous_device[uncertain] = exact_extrema[:, 0] - exact_extrema[:, 1] <= slack
                ambiguous = ambiguous_device.cpu().numpy()
                selected = vertices[index].cpu().numpy()
                zero = lengths[start:start + len(block), 0] == 0
                selected[zero] = local[0]
                ambiguous &= ~zero
                if ambiguous.any():
                    selected[ambiguous] = support_points(
                        self.objective.model, geom, directions[start:start + len(block)][ambiguous])
                    ambiguous_count += int(ambiguous.sum())
            else:
                selected = vertices[index].cpu().numpy()
            result[start:start + len(block)] = selected
        self.diagnostics['support_tie_fallbacks'] = self.diagnostics.get('support_tie_fallbacks', 0) + ambiguous_count
        return result

    def proximity(self, acc, stats):
        """Transfer each geom's supports, then form its dense rows on device.

        Support directions use the original CPU matmul. Grouping identical
        observed normals saves duplicate queries without changing ties.
        """
        obj, kin = self.objective, self.objective.kin
        for prox in obj.proximities:
            key = id(prox)
            if key not in self.proximity_data:
                data = {name: acc.tensor(prox[name]) for name in ('normals', 'environment', 'gaps', 'scale')}
                data['groups'] = []
                for geom in np.unique(prox['geoms']):
                    index = np.flatnonzero(prox['geoms'] == geom)
                    first, inverse = self._group(prox, int(geom))
                    frame, sample = first // len(index), index[first % len(index)]
                    data['groups'].append((int(geom), frame, sample, inverse,
                                           torch.as_tensor(index, device='cuda')))
                self.proximity_data[key] = data
            data = self.proximity_data[key]
            local = torch.empty((*prox['normals'].shape,), dtype=torch.float64, device='cuda')
            for geom, frame, sample, inverse, index in data['groups']:
                directions = -np.matmul(prox['normals'][frame, sample, None, :],
                                        kin.geom_xmat[frame, geom])[:, 0]
                chosen = self.support_unique(geom, directions)
                local[:, index] = acc.tensor(chosen[inverse].reshape(obj.frames, len(index), 3))
            rotation = self.shell.rotation[:, prox['geoms']]
            points = self.shell.position[:, prox['geoms']] + (rotation @ local[..., None])[..., 0]
            normal, scale = data['normals'], data['scale']
            # Explicit three-term sum follows the residual's einsum ordering.
            difference = (points - data['environment']) * normal
            error = (difference[..., 0] + difference[..., 1]) + difference[..., 2] - data['gaps']
            lever = points - acc.tensor(kin.xpos[:, prox['bodies']])
            rows = torch.cat((normal, torch.cross(lever, normal, dim=-1)), dim=-1) * scale[..., None]
            acc.add_body_rows(error * scale, rows, prox['bodies'], kin)
            active = scale > 0
            if np.any(prox['scale'] > 0):
                worst = float(torch.where(active, error.abs(), 0.).max())
                stats['proximity_gap_error_max_m'] = max(stats.get('proximity_gap_error_max_m', 0.), worst)

    def grid(self, field):
        if field is None or not hasattr(field, 'clearance'):
            return field
        key = id(field)
        if key not in self.grids:
            self.grids[key] = GPUGrid(field)
        return self.grids[key]

    def envelope(self, acc, stats):
        """Keep residuals and hinge decisions exact; assemble large Jacobians on device."""
        obj, kin = self.objective, self.objective.kin
        if obj.temporal_reference is None:
            return
        envelope = obj.relation_envelope
        residual, _, excess = envelope.hinge_rows(kin, obj.envelope_weight, ENVELOPE_SIGMA, False)
        stats['layout_flip_max'] = excess
        jacobian = None
        if acc.jacobian:
            points = kin.points(envelope.bodies, envelope.offsets)
            lever = acc.tensor(points - kin.xpos[:, envelope.bodies])
            jacp, jacr = acc.jacp[:, envelope.bodies], acc.jacr[:, envelope.bodies]
            x, y, z = lever.unbind(-1)
            a, b, c = jacr.unbind(-2)
            cross = torch.stack((y[..., None] * c - z[..., None] * b,
                                 z[..., None] * a - x[..., None] * c,
                                 x[..., None] * b - y[..., None] * a), dim=-2)
            point_jac = jacp - cross
            first, second = envelope.graph.pairs.T
            pairs = (point_jac[:, second] - point_jac[:, first]) / acc.tensor(envelope.graph.robot_lengths)[None, :, None, None]
            count = len(first)
            jacs = []
            if envelope.layout_sign is not None:
                layout = torch.einsum('fak,fpkw->fpaw', acc.tensor(envelope.axes), pairs)
                layout *= acc.tensor(envelope.layout_sign)[..., None]
                _, _, (change, _) = envelope.measure(kin)
                value, _ = envelope.layout(change)
                flip = np.where(envelope.layout_sign != 0, np.maximum(-value, 0.0), 0.0)
                active = (flip.reshape(obj.frames, -1) > 0) * (obj.envelope_weight[:, None] / ENVELOPE_SIGMA)
                jacs.append(-layout.reshape(obj.frames, -1, obj.width) * acc.tensor(active)[..., None])
            soft = 1.0 / (np.maximum(envelope.soft_scale, 1e-9) * np.sqrt(max(count, 1)))
            jacs.append(pairs.reshape(obj.frames, -1, obj.width) * acc.tensor(soft)[:, None, None])
            jacobian = torch.cat(jacs, dim=1)
        acc.add(residual, jacobian)
        frames = getattr(obj, 'frame_slice', slice(0, obj.frames))
        for guard in getattr(obj, 'hinge_guards', ()):
            from .acquisition_guard import AcquisitionNormalGuard
            if type(guard) is AcquisitionNormalGuard and obj.frames >= 64:
                acquisition_rows(guard, kin, frames, obj.envelope_weight, ENVELOPE_SIGMA, acc)
            else:
                rows, jac = guard.hinge_rows(kin, frames, obj.envelope_weight, ENVELOPE_SIGMA, acc.jacobian)
                if rows is not None:
                    acc.add(rows, jac)

    def envelope_reference(self, acc, stats, native):
        obj = self.objective
        if obj.temporal_reference is None:
            return
        rows, jac, excess = obj.relation_envelope.hinge_rows(
            obj.kin, obj.envelope_weight, ENVELOPE_SIGMA, acc.jacobian)
        stats['layout_flip_max'] = excess
        acc.add(rows, jac)
        frames = getattr(obj, 'frame_slice', slice(0, obj.frames))
        from .acquisition_guard import AcquisitionNormalGuard
        for guard in getattr(obj, 'hinge_guards', ()):
            if type(guard) is AcquisitionNormalGuard and native is not None:
                acquisition_rows(guard, obj.kin, frames, obj.envelope_weight, ENVELOPE_SIGMA, acc, native)
            else:
                rows, jac = guard.hinge_rows(obj.kin, frames, obj.envelope_weight, ENVELOPE_SIGMA, acc.jacobian)
                if rows is not None:
                    acc.add(rows, jac)

    def evaluate(self, poses, with_jacobian, envelope_weight):
        objective = self.objective
        if os.getpid() != self.pid:
            raise RuntimeError('CUDA frame evaluation cannot run in an inherited process')
        self.diagnostics['support_tie_fallbacks'] = 0
        objective.envelope_weight = envelope_weight
        objective.kin.evaluate(poses, with_jacobian, native=True)
        self.shell.prepare(objective.kin)
        if not self.device_moments:
            return self.reference_frames.evaluate(poses, with_jacobian, envelope_weight)
        native = NativePointRows(objective.kin) if with_jacobian else None
        if native is not None:
            objective.kin.point_jacobians = native.jacobians

        def collisions():
            acc = _Accumulator(objective.frames, objective.width, with_jacobian)
            stats = {}
            objective._self_collision_terms(poses, acc, stats)
            return acc, stats

        scene, objects = objective.scene_sdf, objective.objects
        objective.scene_sdf = self.grid(scene)
        objective.objects = [dict(item, sdf=self.grid(item['sdf'])) for item in objects]
        objective._proximity_support = self.support
        objective._deepest_samples = self.shell.deepest
        stats = {}
        self.diagnostics['support_tie_fallbacks'] = 0
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(collisions)
                # Small row batches cost less on CPU than repeated device
                # transfers. Dense geometry can still use the device kernels.
                short = objective.frames < 64 or not self.device_moments
                acc = (_Accumulator(objective.frames, objective.width, with_jacobian) if short
                       else GPUAccumulator(objective.frames, objective.width, with_jacobian, objective.kin))
                objective._mapped_terms(acc, stats)
                objective._orientation_terms(acc, stats)
                objective._effector_terms(acc, stats)
                objective._surface_terms(acc, stats)
                if short:
                    objective._proximity_terms(acc, stats)
                else:
                    self.proximity(acc, stats)
                objective._clearance_terms(acc, stats, poses)
                objective._plane_terms(acc, stats)
                collision, collision_stats = future.result()
                stats.update(collision_stats)
                if short:
                    acc.frame_cost += collision.frame_cost
                    if with_jacobian:
                        acc.hessian += collision.hessian
                        acc.gradient += collision.gradient
                else:
                    acc.add_normals(collision.frame_cost,
                                    collision.hessian if with_jacobian else None,
                                    collision.gradient if with_jacobian else None)
                objective._reference_terms(poses, acc, stats)
                if short:
                    self.envelope_reference(acc, stats, native)
                else:
                    self.envelope(acc, stats)
                    acc = acc.finish(_Accumulator)
        finally:
            objective.scene_sdf, objective.objects = scene, objects
            del objective._proximity_support
            del objective._deepest_samples
            if native is not None:
                del objective.kin.point_jacobians
        effector_point, effector_jacobian = objective._effector_points(with_jacobian)
        return dict(cost=acc.cost, frame_cost=acc.frame_cost,
                    hessian=acc.hessian if with_jacobian else None,
                    gradient=acc.gradient if with_jacobian else None, stats=stats,
                    root_point=objective._root_points(),
                    root_jacobian=objective._root_jacobian() if with_jacobian else None,
                    effector_point=effector_point, effector_jacobian=effector_jacobian)
