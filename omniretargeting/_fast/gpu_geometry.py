"""Resident Warp prefilter for original-geometry measurement.

OMNIRETARGETING_GPU_GEOMETRY defaults to 1; 0 forces exact CPU measurement.
Missing CUDA/Warp or unsupported scenes automatically use the CPU. A fresh
interpreter owns CUDA, so the solver's forked CPU workers never initialize it.
Float32 results only select points for the existing float64 SceneQuery; no GPU
witness enters repair. Open, multiple-component and sliver scenes use the CPU.
Calibration of the distance/winding bands is empirical, not a global proof.
"""

from __future__ import annotations

import atexit
import hashlib
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import igl
import numpy as np
import warp as wp

DISTANCE_BAND = 1e-5
WINDING_BAND = 1e-3
# 979,968 actual shell positions over all 957 benchmark frames: maximum
# unsigned error 4.72e-7 m, no sign/threshold disagreements outside this band.
# OMNIRETARGETING_GPU_GEOMETRY_DEBUG=1 checks a deterministic 1% of screened
# samples against SceneQuery at runtime; failure raises, never relaxes a gate.


@wp.func_native("return wp::mesh_query_winding_number(mid, point, accuracy);")
def _winding(mid: wp.uint64, point: wp.vec3, accuracy: float) -> float:
    ...


@wp.kernel
def _closest(
    mid: wp.uint64, points: wp.array(dtype=wp.vec3),
    distance: wp.array(dtype=float), faces: wp.array(dtype=int),
    winding: wp.array(dtype=float), signed: int,
):
    i = wp.tid()
    p = points[i]
    q = wp.mesh_query_point_no_sign(mid, p, 1e6)
    if q.result:
        distance[i] = wp.length(p - wp.mesh_eval_position(mid, q.face, q.u, q.v))
        faces[i] = q.face
    else:
        distance[i] = 1e6
        faces[i] = -1
    if signed != 0:
        winding[i] = _winding(mid, p, 2.0)


@wp.kernel
def _resident_candidates(
    mid: wp.uint64, local: wp.array(dtype=wp.vec3d),
    sample_ids: wp.array(dtype=int), starts: wp.array(dtype=int), counts: wp.array(dtype=int),
    position: wp.array(dtype=wp.vec3d), rotation: wp.array(dtype=wp.mat33d),
    mesh_center: wp.vec3d, lower: wp.vec3d, upper: wp.vec3d,
    invert: int, band: float, eps: float, width: int, debug: int, minimum: int,
    counter: wp.array(dtype=int), packed: wp.array(dtype=int),
    values: wp.array(dtype=float), minima: wp.array(dtype=float),
    uncertain: wp.array(dtype=int), robust_minima: wp.array(dtype=float),
):
    job, sample = wp.tid()
    if sample >= counts[job]:
        return
    sid = sample_ids[starts[job] + sample]
    world = position[job] + rotation[job] * local[sid]
    p64 = world - mesh_center
    p = wp.vec3(wp.float32(p64[0]), wp.float32(p64[1]), wp.float32(p64[2]))
    q = wp.mesh_query_point_no_sign(mid, p, 1e6)
    phi = float(1e6)
    exact = int(1)
    unknown = int(1)
    if q.result:
        distance = wp.length(p - wp.mesh_eval_position(mid, q.face, q.u, q.v))
        winding = float(0.)
        if (world[0] >= lower[0] and world[0] <= upper[0]
            and world[1] >= lower[1] and world[1] <= upper[1]
            and world[2] >= lower[2] and world[2] <= upper[2]):
            winding = wp.abs(_winding(mid, p, 2.0))
        sign = float(1.)
        if winding > .5:
            sign = -1.
        if invert != 0:
            sign = -sign
        phi = sign * distance
        ambiguous = ((winding >= .009 and wp.abs(winding - 1.) >= .009)
                     or distance <= eps or not wp.isfinite(winding))
        unknown = int(ambiguous)
        if not ambiguous and minimum != 0:
            wp.atomic_min(robust_minima, job, phi)
        if not ambiguous and phi >= band + eps:
            exact = 0
        wp.atomic_min(minima, job, phi - eps)
    index = job * width + sample
    if exact != 0 or (debug != 0 and index % 100 == 0):
        out = wp.atomic_add(counter, 0, 1)
        packed[out] = index
        if exact == 0:
            packed[out] = -index - 1
        values[out] = phi
        uncertain[out] = unknown


@wp.kernel
def _minimum_candidates(
    packed: wp.array(dtype=int), values: wp.array(dtype=float), uncertain: wp.array(dtype=int),
    minima: wp.array(dtype=float), width: int, eps: float, debug: int,
    counter: wp.array(dtype=int), selected: wp.array(dtype=int),
    selected_values: wp.array(dtype=float), selected_uncertain: wp.array(dtype=int),
):
    i = wp.tid()
    encoded = packed[i]
    index = encoded
    if encoded < 0:
        index = -encoded - 1
    keep = encoded < 0 or uncertain[i] != 0 or values[i] <= minima[index // width] + 3. * eps
    if not keep and debug != 0 and (index % width) % 100 == 0:
        keep = True
        encoded = -index - 1
    if keep:
        out = wp.atomic_add(counter, 0, 1)
        selected[out] = encoded
        selected_values[out] = values[i]
        selected_uncertain[out] = uncertain[i]


class WarpScenePrefilter:
    """Conservative candidate screening; exact CPU values for retained points.

    ``query_reduction`` is only for minimum/8-mm-manifold reductions. Values
    of omitted points are intentionally unspecified, so this is not a general
    replacement for SceneQuery.query. Minimum candidates include a 2-epsilon
    overlap, preserving CPU argmin ties. Every negative point and every point
    within the witness band is exact, so all downstream depth thresholds and
    manifold force/moment probes still use the original CPU data.
    """

    def __init__(self, scene, *, device="cuda:0", cpu_sign=False, eps=DISTANCE_BAND):
        self.scene, self.device, self.cpu_sign, self.eps = scene, device, cpu_sign, float(eps)
        self.stats = dict(points=0, exact_points=0, gpu_seconds=0., cpu_seconds=0.,
                          ambiguous_points=0, sliver_points=0)
        self.enabled = (len(scene.components) == 1 and scene.components[0][2]
                        and scene.closed_sign_method == "winding")
        self.mesh = None
        if not self.enabled:
            return
        wp.init()
        part = scene.components[0][0]
        self.vertices = np.asarray(part.vertices, dtype=np.float64)
        self.faces = np.asarray(part.faces, dtype=np.int64)
        self.center = (self.vertices.min(0) + self.vertices.max(0)) / 2
        centered = (self.vertices - self.center).astype(np.float32)
        tri = centered[self.faces]
        e0, e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], tri[:, 2] - tri[:, 1]
        area = np.linalg.norm(np.cross(e0, e1), axis=1)
        lengths = np.sum(e0 * e0 + e1 * e1 + e2 * e2, axis=1)
        bad = area <= 2e-6 * lengths
        self.sliver_count = int(bad.sum())
        self.sliver_tree = None
        if bad.any():
            self.sliver_tree = igl.AABB()
            self.sliver_faces = np.ascontiguousarray(self.faces[bad])
            self.sliver_tree.init(self.vertices, self.sliver_faces)
        self.mesh = wp.Mesh(
            points=wp.array(centered, dtype=wp.vec3, device=device),
            indices=wp.array(self.faces.astype(np.int32).ravel(), dtype=wp.int32, device=device),
            support_winding_number=not cpu_sign)
        # Compile outside the timed queries.
        self.raw(self.vertices[:1])
        self.stats.update(gpu_seconds=0.)

    def raw(self, points):
        start = time.perf_counter()
        local = wp.array((np.asarray(points) - self.center).astype(np.float32), dtype=wp.vec3, device=self.device)
        distance = wp.empty(len(points), dtype=float, device=self.device)
        faces = wp.empty(len(points), dtype=int, device=self.device)
        winding = wp.empty(len(points), dtype=float, device=self.device)
        wp.launch(_closest, dim=len(points),
                  inputs=[self.mesh.id, local, distance, faces, winding, int(not self.cpu_sign)],
                  device=self.device)
        result = distance.numpy().astype(np.float64), faces.numpy(), winding.numpy().astype(np.float64)
        self.stats["gpu_seconds"] += time.perf_counter() - start
        return result

    def lower_bounds(self, points):
        """Calibrated signed bounds at coarse nodes; ambiguous nodes stay NaN."""
        distance, faces, winding = self.raw(points)
        winding = np.abs(winding)
        bounds = self.scene.components[0][0].bounds
        winding[~np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)] = 0.
        known = ((faces >= 0) & np.isfinite(distance) & np.isfinite(winding)
                 & (distance > self.eps)
                 & ((winding < .009) | (np.abs(winding - 1.) < .009)))
        if self.sliver_tree is not None:
            d2, _, _ = self.sliver_tree.squared_distance(self.vertices, self.sliver_faces, points)
            known &= np.sqrt(np.maximum(d2, 0.)) > distance + 2 * self.eps
        sign = np.where(winding > .5, -1., 1.)
        if self.scene.enclosed_free_space:
            sign = -sign
        return np.where(known, sign * distance - self.eps, np.nan)

    def candidates(self, points, groups, *, band=.008):
        """Select original point indices; caller may batch their exact query."""
        points = np.asarray(points, dtype=np.float64)
        self.stats["points"] += len(points)
        if not self.enabled:
            self.stats["exact_points"] += len(points)
            return np.arange(len(points))
        distance, faces, winding = self.raw(points)
        start = time.perf_counter()
        bounds = self.scene.components[0][0].bounds
        in_bounds = np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
        if self.cpu_sign:
            winding = np.zeros(len(points))
            winding[in_bounds] = np.abs(igl.fast_winding_number(
                self.vertices, self.faces, points[in_bounds]))
        else:
            winding = np.abs(winding)
            winding[~in_bounds] = 0.
        ambiguous = ((winding >= .01 - WINDING_BAND)
                     & (np.abs(winding - 1.) >= .01 - WINDING_BAND))
        # Zero-distance reliability, nearest-face failures and any skipped face
        # which could be nearer than Warp's result all use the CPU path.
        ambiguous |= (distance <= self.eps) | (faces < 0) | ~np.isfinite(winding)
        sign = np.where(winding > .5, -1., 1.)
        if self.scene.enclosed_free_space:
            sign = -sign
        phi = sign * distance
        exact = ambiguous | (phi < band + self.eps)
        if self.sliver_tree is not None:
            d2, _, _ = self.sliver_tree.squared_distance(self.vertices, self.sliver_faces, points)
            sliver = np.sqrt(np.maximum(d2, 0.)) <= distance + 2 * self.eps
            exact |= sliver
            self.stats["sliver_points"] += int(sliver.sum())
        for begin, end in groups:
            minimum = phi[begin:end].min(initial=np.inf)
            exact[begin:end] |= phi[begin:end] <= minimum + 2 * self.eps
        selected = np.flatnonzero(exact)
        self.stats["exact_points"] += len(selected)
        self.stats["ambiguous_points"] += int(ambiguous.sum())
        self.stats["cpu_seconds"] += time.perf_counter() - start
        return selected

    def query_reduction(self, points, groups, *, band=.008):
        """Exact query all possible witnesses/minima and uncertain signs."""
        points = np.asarray(points, dtype=np.float64)
        selected = self.candidates(points, groups, band=band)
        start = time.perf_counter()
        measured = self.scene.query(points[selected])
        # Nonselected points are known, nonwinning, and outside the witness
        # band under the calibrated interval assumption. Their directions never
        # enter a returned row. Exact minima retain original sample ordering.
        result = {key: np.empty((len(points), *value.shape[1:]), dtype=value.dtype)
                  for key, value in measured.items()}
        # Scene reduction treats nonfinite clearances as unknown. A finite
        # nonwinning placeholder preserves coverage while exact candidates
        # supply every minimum; it is never returned as a witness.
        result["clearance"][:] = 1e6
        result["directions"][:] = 0.
        result["uncertain"][:] = False
        for key, values in measured.items():
            result[key][selected] = values
        self.stats["cpu_seconds"] += time.perf_counter() - start
        return result


class ResidentScene:
    """GPU-resident immutable shells; only placements and candidates cross IPC."""

    def __init__(self, scene, local, sample_ids):
        self.prefilter = WarpScenePrefilter(scene)
        if not self.prefilter.enabled or self.prefilter.sliver_count:
            raise ValueError("Resident screening requires one closed non-sliver winding component")
        self.local = wp.array(local, dtype=wp.vec3d, device="cuda:0")
        self.sample_ids = wp.array(sample_ids, dtype=int, device="cuda:0")

    def candidates(self, positions, rotations, starts, counts, *, debug=False, band=.008, minimum=False):
        device, query = "cuda:0", self.prefilter
        started = time.perf_counter()
        width, capacity = int(np.max(counts)), int(np.sum(counts))
        if capacity > 1 << 28 or len(counts) * width >= 2**31:
            raise ValueError("GPU candidate tile index exceeds int32")
        # At most one record per original point, including debug samples.
        packed = wp.empty(capacity, dtype=int, device=device)
        values = wp.empty(capacity, dtype=float, device=device)
        uncertain = wp.empty(capacity, dtype=int, device=device)
        minima = wp.full(len(counts), 1e6, dtype=float, device=device)
        robust_minima = wp.full(len(counts), 1e6, dtype=float, device=device)
        counter = wp.zeros(1, dtype=int, device=device)
        pos = wp.array(positions, dtype=wp.vec3d, device=device)
        rot = wp.array(rotations.reshape(-1, 3, 3), dtype=wp.mat33d, device=device)
        begin = wp.array(starts, dtype=int, device=device)
        sizes = wp.array(counts, dtype=int, device=device)
        bounds = query.scene.components[0][0].bounds
        wp.launch(_resident_candidates, dim=(len(counts), width), device=device, inputs=[
            query.mesh.id, self.local, self.sample_ids, begin, sizes, pos, rot,
            wp.vec3d(*query.center), wp.vec3d(*bounds[0]), wp.vec3d(*bounds[1]),
            int(query.scene.enclosed_free_space), band, query.eps, width, int(debug), int(minimum),
            counter, packed, values, minima, uncertain, robust_minima])
        count = int(counter.numpy()[0])
        if minimum and count:
            selected = wp.empty(count, dtype=int, device=device)
            selected_values = wp.empty(count, dtype=float, device=device)
            selected_uncertain = wp.empty(count, dtype=int, device=device)
            selected_count = wp.zeros(1, dtype=int, device=device)
            # The extra epsilon covers float32 rounding in this reduction.
            # The caller still applies its original 2-epsilon CPU selection.
            wp.launch(_minimum_candidates, dim=count, device=device, inputs=[
                packed, values, uncertain, robust_minima, width, query.eps, int(debug),
                selected_count, selected, selected_values, selected_uncertain])
            count = int(selected_count.numpy()[0])
            packed, values, uncertain = selected, selected_values, selected_uncertain
        # Slice before D2H: only compacted candidates, never the full shell.
        # Warp 1.17 cannot convert a zero-length CUDA slice to numpy.
        indices = packed[:count].numpy() if count else np.empty(0, np.int32)
        phi = values[:count].numpy() if count else np.empty(0, np.float32)
        ambiguous = uncertain[:count].numpy().astype(bool) if count else np.empty(0, bool)
        lower_bounds = minima.numpy().astype(np.float64)
        debug_mask = indices < 0
        indices[debug_mask] = -indices[debug_mask] - 1
        order = np.argsort(indices, kind="stable")
        indices, phi, debug_mask = indices[order], phi[order], debug_mask[order]
        return dict(jobs=indices // width, samples=indices % width, phi=phi,
                    debug=debug_mask, ambiguous=ambiguous[order], lower_bounds=lower_bounds,
                    seconds=time.perf_counter() - started, points=capacity,
                    candidates=int(np.count_nonzero(~debug_mask)))


def _gpu_server(connection):
    """Spawned CUDA service: the solver and its forked CPU workers never init CUDA."""
    scenes = {}
    try:
        while True:
            message = connection.recv()
            if message is None:
                break
            command, token, args = message
            try:
                if command == "build":
                    vertices, faces, enclosed, local, sample_ids = args
                    import trimesh
                    from ..scene_geometry import SceneQuery
                    scene = SceneQuery(trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
                                       enclosed_free_space=enclosed)
                    scenes[token] = ResidentScene(scene, local, sample_ids)
                    value = True
                elif command == "centers":
                    value = scenes[token].prefilter.raw(args)
                elif command == "bounds":
                    value = scenes[token].prefilter.lower_bounds(args)
                else:
                    positions, rotations, starts, counts, debug, *band = args
                    value = scenes[token].candidates(positions, rotations, starts, counts,
                                                     debug=debug, minimum=command == "query_min",
                                                     band=.008 if not band else band[0])
                connection.send((True, value))
            except Exception as error:
                import traceback
                error.add_note(traceback.format_exc())
                connection.send((False, error))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class GeometryService:
    def __init__(self):
        self.pid = os.getpid()
        self.connection, child = multiprocessing.Pipe()
        # spawn imports __main__, which CLI entry points need not guard. Exec
        # this module instead: no inherited CUDA context or repeated CLI work.
        self.process = subprocess.Popen(
            [sys.executable, "-m", __name__, "--serve-fd", str(child.fileno())],
            pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
            cwd=str(Path(__file__).resolve().parents[2]))
        child.close()
        self.tokens = set()
        atexit.register(self.close)

    def request(self, command, token, args):
        self.connection.send((command, token, args))
        ok, value = self.connection.recv()
        if not ok:
            raise value
        return value

    def register(self, scene, local, sample_ids):
        part = scene.components[0][0]
        digest = hashlib.sha256()
        for array in (part.vertices, part.faces, local, sample_ids):
            digest.update(np.asarray(array).tobytes())
        digest.update(bytes([scene.enclosed_free_space]))
        token = digest.hexdigest()
        if token not in self.tokens:
            self.request("build", token, (np.asarray(part.vertices), np.asarray(part.faces),
                                          scene.enclosed_free_space, local, sample_ids))
            self.tokens.add(token)
        return token

    def candidates(self, token, positions, rotations, starts, counts, debug=False, *,
                   minimum=False, band=.008):
        return self.request("query_min" if minimum else "query", token,
                            (positions, rotations, starts, counts, debug, band))

    def centers(self, token, positions):
        return self.request("centers", token, positions)

    def close(self):
        if os.getpid() != self.pid or self.process.poll() is not None:
            return
        try:
            self.connection.send(None)
        except (OSError, EOFError):
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
        self.connection.close()


_SERVICE = None


def geometry_service():
    global _SERVICE
    if (_SERVICE is None or _SERVICE.pid != os.getpid()
            or _SERVICE.process.poll() is not None):
        _SERVICE = GeometryService()
    return _SERVICE


def initialize_static(geometry):
    """Build the immutable backend during geometry construction, before measurement."""
    scene, objective = geometry.scene, geometry.objective
    if (scene is None or not geometry._static_bound_reuse or len(scene.components) != 1
            or not scene.components[0][2]):
        return False
    groups = {int(g): np.flatnonzero(objective.shell_sample_geom == g)
              for g in objective.shell_geoms}
    geometry._gpu_groups = groups
    geometry._gpu_sample_ids = np.concatenate(list(groups.values())).astype(np.int32)
    starts = np.r_[0, np.cumsum([len(v) for v in groups.values()])]
    geometry._gpu_starts = {g: int(start) for g, start in zip(groups, starts)}
    geometry._gpu_token = geometry_service().register(scene, objective.shell_local,
                                                    geometry._gpu_sample_ids)
    return True


class StaticPlan:
    """Screen one worker's changed geoms while earlier workers query exactly.

    Shells and the scene remain in the CUDA interpreter. Each worker receives
    only its changed placements and retained original sample IDs. A rejection
    stops unsent batches; complete candidates still measure every original row.
    """

    def __init__(self, geometry, updates, debug, scanned, centered, begun):
        self.geometry, self.updates, self.debug = geometry, updates, debug
        geometry.gpu_stats = dict(seconds=0., points=0, candidates=0,
                                  sphere_culled_geoms=int(sum(np.isfinite(u[2]["clearance"]) for u in updates)),
                                  scan_seconds=scanned - begun, center_seconds=centered - scanned,
                                  setup_seconds=0., prepare_seconds=centered - begun)
        if debug:
            points = []
            for update in updates:
                if not np.isfinite(update[2]["clearance"]):
                    continue
                ids = geometry._gpu_groups[update[0][1]][::100]
                points.append(update[3] + np.einsum(
                    "nij,nj->ni", np.broadcast_to(update[4].reshape(3, 3), (len(ids), 3, 3)),
                    geometry.objective.shell_local[ids]))
            if points:
                exact = geometry.scene.query(np.concatenate(points))
                assert not exact["uncertain"].any() and np.all(exact["clearance"] >= .008), "Sphere bound hid a witness/unknown"

    def query_worker(self, worker, count):
        geometry = self.geometry
        updates = [u for u in self.updates if u[0][0] % count == worker]
        if not geometry._gpu_allowed:
            return None
        active = [u for u in updates if not np.isfinite(u[2]["clearance"])]
        if not active:
            return updates
        try:
            service = geometry_service()
            groups = geometry._gpu_groups
            positions = np.array([u[3] for u in active])
            rotations = np.array([u[4] for u in active])
            starts = np.array([geometry._gpu_starts[u[0][1]] for u in active], np.int32)
            counts = np.array([len(groups[u[0][1]]) for u in active], np.int32)
            result = service.candidates(geometry._gpu_token, positions, rotations, starts, counts, self.debug)
        except (ImportError, RuntimeError, ValueError, OSError, EOFError) as error:
            geometry._gpu_allowed = False
            geometry.gpu_unavailable = str(error)
            return None  # Remaining workers retain their exact CPU query path.
        for field in ("seconds", "points", "candidates"):
            geometry.gpu_stats[field] += result[field]
        jobs, samples = result["jobs"], result["samples"]
        begin = 0
        debug_points, debug_phi = [], []
        for job in np.unique(jobs):
            end = int(np.searchsorted(jobs, job, side="right"))
            update = active[job]
            ids = groups[update[0][1]][samples[begin:end]]
            audit = result["debug"][begin:end]
            if audit.any():
                chosen = ids[audit]
                debug_points.append(update[3] + np.einsum(
                    "nij,nj->ni", np.broadcast_to(update[4].reshape(3, 3), (len(chosen), 3, 3)),
                    geometry.objective.shell_local[chosen]))
                debug_phi.append(result["phi"][begin:end][audit])
            update[2]["_gpu_ids"] = ids[~audit].astype(np.int32)
            begin = end
        for job, update in enumerate(active):
            update[2]["_gpu_lower_bound"] = float(result["lower_bounds"][job])
        if debug_points:
            exact = geometry.scene.query(np.concatenate(debug_points))
            predicted = np.concatenate(debug_phi)
            assert not exact["uncertain"].any(), "GPU screening hid an unknown sample"
            assert np.all(exact["clearance"] >= .008), "GPU screening hid a witness"
            assert np.all(np.abs(exact["clearance"] - predicted) <= DISTANCE_BAND), "GPU error band exceeded"
        return updates


def prepare_static(geometry, poses, *, frames=None):
    """Parent-only exact summaries, with GPU screening and no dense world arrays.

    Far summaries store a LOWER BOUND, not an invented measured minimum. For
    a closed signed-distance component, d(center)-radius <= every shell d.
    A bound above .008 implies no row, no depth key, and unchanged coverage.
    OriginalGeometry uses this field only for further Lipschitz cache reuse;
    a smaller bound can cause extra exact work, never skip a possible witness.
    """
    import mujoco

    begun = time.perf_counter()
    scene, objective, data = geometry.scene, geometry.objective, geometry.data
    if (scene is None or not geometry._static_bound_reuse or len(scene.components) != 1
            or not scene.components[0][2]):
        return None
    groups = getattr(geometry, "_gpu_groups", None)
    if groups is None:
        groups = {int(g): np.flatnonzero(objective.shell_sample_geom == g)
                  for g in objective.shell_geoms}
        geometry._gpu_groups = groups
        geometry._gpu_sample_ids = np.concatenate(list(groups.values())).astype(np.int32)
        starts = np.r_[0, np.cumsum([len(v) for v in groups.values()])]
        geometry._gpu_starts = {g: int(start) for g, start in zip(groups, starts)}
    updates = []
    for frame in range(len(poses)) if frames is None else frames:
        pose = poses[frame]
        data.qpos[:] = pose
        mujoco.mj_forward(geometry.model, data)
        for geom in groups:
            signature = (data.geom_xpos[geom].tobytes(), data.geom_xmat[geom].tobytes())
            cached = geometry._static_cache.get((frame, geom))
            if cached is not None and cached[0] == signature:
                continue
            if cached is not None and geometry._clear_after_motion(cached, geom):
                # The normal summary pass counts this reuse once.
                continue
            position, rotation = data.geom_xpos[geom].copy(), data.geom_xmat[geom].copy()
            summary = dict(row=None, unknown=False, clearance=np.inf, rows=[], origin=position.copy(),
                           _screen_band=.008)
            updates.append(((frame, geom), signature, summary, position, rotation))
    if not updates:
        return []
    scanned = time.perf_counter()
    positions = np.array([u[3] for u in updates])
    rotations = np.array([u[4] for u in updates])
    radii = np.array([geometry._static_radii[u[0][1]] for u in updates])
    if os.environ.get("OMNIRETARGETING_GPU_GEOMETRY_CENTERS", "0") != "0":
        distance, faces, winding = geometry_service().centers(geometry._gpu_token, positions)
        winding = np.abs(winding)
        bounds = scene.components[0][0].bounds
        winding[~np.all((positions >= bounds[0]) & (positions <= bounds[1]), axis=1)] = 0.
        known = ((faces >= 0) & np.isfinite(distance) & np.isfinite(winding)
                 & (distance > DISTANCE_BAND)
                 & ((winding < .009) | (np.abs(winding - 1.) < .009)))
        sign = np.where(winding > .5, -1., 1.)
        if scene.enclosed_free_space:
            sign = -sign
        lower = sign * distance - DISTANCE_BAND - radii
        far = known & np.isfinite(lower) & (lower >= .008 + 1e-9)
    else:
        from .repair_static_query import query_scene
        centers = query_scene(geometry, positions)
        lower = centers["clearance"] - radii
        far = ~centers["uncertain"] & np.isfinite(lower) & (lower >= .008 + 1e-9)
    centered = time.perf_counter()
    for index in np.flatnonzero(far):
        updates[index][2]["clearance"] = float(lower[index])
    active = np.flatnonzero(~far)
    debug = os.environ.get("OMNIRETARGETING_GPU_GEOMETRY_DEBUG", "0") != "0"
    if (getattr(geometry, "parallel", None) is not None
            and os.environ.get("OMNIRETARGETING_GPU_GEOMETRY_STREAM", "0") != "0"):
        return StaticPlan(geometry, updates, debug, scanned, centered, begun)
    if len(active):
        service = geometry_service()
        token = getattr(geometry, "_gpu_token", None)
        if token is None:
            token = service.register(scene, objective.shell_local, geometry._gpu_sample_ids)
            geometry._gpu_token = token
        registered = time.perf_counter()
        starts = np.array([geometry._gpu_starts[updates[i][0][1]] for i in active], np.int32)
        counts = np.array([len(groups[updates[i][0][1]]) for i in active], np.int32)
        result = service.candidates(token, positions[active], rotations[active], starts, counts, debug)
        geometry.gpu_stats = {k: result[k] for k in ("seconds", "points", "candidates")}
        geometry.gpu_stats["sphere_culled_geoms"] = int(far.sum())
        geometry.gpu_stats.update(scan_seconds=scanned - begun, center_seconds=centered - scanned,
                                  setup_seconds=registered - centered)
        jobs, samples = result["jobs"], result["samples"]
        debug_points, debug_phi = [], []
        begin = 0
        for job in np.unique(jobs):
            end = int(np.searchsorted(jobs, job, side="right"))
            update = updates[active[job]]
            ids = groups[update[0][1]][samples[begin:end]]
            audit = result["debug"][begin:end]
            if audit.any():
                chosen = ids[audit]
                debug_points.append(update[3] + np.einsum(
                    "nij,nj->ni", np.broadcast_to(update[4].reshape(3, 3), (len(chosen), 3, 3)),
                    objective.shell_local[chosen]))
                debug_phi.append(result["phi"][begin:end][audit])
            update[2]["_gpu_ids"] = ids[~audit].astype(np.int32)
            begin = end
        for job, index in enumerate(active):
            summary = updates[index][2]
            summary["_gpu_lower_bound"] = float(result["lower_bounds"][job])
        if debug_points:
            exact = scene.query(np.concatenate(debug_points))
            predicted = np.concatenate(debug_phi)
            assert not exact["uncertain"].any(), "GPU screening hid an unknown sample"
            assert np.all(exact["clearance"] >= .008), "GPU screening hid a witness"
            assert np.all(np.abs(exact["clearance"] - predicted) <= DISTANCE_BAND), "GPU error band exceeded"
    if debug and far.any():
        points = []
        for index in np.flatnonzero(far):
            update = updates[index]
            ids = groups[update[0][1]][::100]
            points.append(update[3] + np.einsum(
                "nij,nj->ni", np.broadcast_to(update[4].reshape(3, 3), (len(ids), 3, 3)),
                objective.shell_local[ids]))
        exact = scene.query(np.concatenate(points))
        assert not exact["uncertain"].any() and np.all(exact["clearance"] >= .008), "Sphere bound hid a witness/unknown"
    if getattr(geometry, "parallel", None) is None:
        complete_static(geometry, updates)
    if hasattr(geometry, "gpu_stats"):
        geometry.gpu_stats["prepare_seconds"] = time.perf_counter() - begun
    return updates


def complete_static(geometry, updates):
    """Float64 witnesses in existing CPU workers; their immutable shells stay local."""
    begun = time.perf_counter()
    pending, count = [], 0
    for key, signature, summary, position, rotation in updates:
        ids = summary.pop("_gpu_ids", np.empty(0, np.int32))
        if not len(ids):
            continue
        points = position + np.einsum(
            "nij,nj->ni", np.broadcast_to(rotation.reshape(3, 3), (len(ids), 3, 3)),
            geometry.objective.shell_local[ids])
        first = 0
        while first < len(ids):
            take = min(len(ids) - first, geometry.SCENE_BATCH_POINTS - count)
            pending.append((summary, points[first:first + take], ids[first:first + take]))
            first, count = first + take, count + take
            if count == geometry.SCENE_BATCH_POINTS:
                geometry._query_static_batch(pending)
                pending, count = [], 0
    if pending:
        geometry._query_static_batch(pending)
    for key, signature, summary, position, rotation in updates:
        lower = summary.pop("_gpu_lower_bound", np.inf)
        if summary["row"] is None and not summary["rows"] and not summary["unknown"]:
            summary["clearance"] = min(summary["clearance"], lower)
        geometry._static_cache[key] = (signature, summary, position, rotation)
    geometry.gpu_exact_seconds = time.perf_counter() - begun


def measure_lazy_static(geometry, poses, frames, updates, jacobians, protected, stop):
    """Keep GPU candidates pending until moving/self queries need static rows.

    A rejected candidate needs no later exact static work. Accepted candidates
    still query every retained sample with the original CPU reduction.
    """
    completed = []
    geometry._gpu_pending_static = (updates, completed, stop)
    geometry.gpu_exact_seconds = 0.
    try:
        part = geometry.measure_frames(poses, frames, jacobians=jacobians,
                                       protected_depths=protected, stop=stop)
        return part, completed
    finally:
        del geometry._gpu_pending_static


def lazy_static_frames(geometry, poses, frames):
    from .mesh_repair import OriginalGeometry

    updates, completed, _ = geometry._gpu_pending_static
    by_frame = {}
    for update in updates:
        by_frame.setdefault(update[0][0], []).append(update)
    frames = list(frames)
    # Bound how much exact work a rejected candidate can leave unused, while
    # amortizing libigl's winding-data construction over several frames.
    chunk_size = 8
    for first in range(0, len(frames), chunk_size):
        # measure_frames checks the shared stop flag before starting a frame.
        # A peer can reject while this frame's moving/self query is running;
        # its already started frame still needs a static summary to finish.
        chunk = frames[first:first + chunk_size]
        pending = [u for frame in chunk for u in by_frame.get(frame, ())]
        elapsed = geometry.gpu_exact_seconds
        complete_static(geometry, pending)
        geometry.gpu_exact_seconds += elapsed
        completed.extend(pending)
        yield from OriginalGeometry._static_frames(geometry, poses, chunk, _gpu_ready=True)


class MovingForwardCache:
    """The existing MovingSceneQuery.forward_cache protocol, without CUDA."""

    def __init__(self):
        self.cached = {}

    def get(self, pose, frame, object_id):
        entry = self.cached.get(frame)
        signature = np.ascontiguousarray(pose).tobytes()
        return entry[1][object_id] if entry is not None and entry[0] == signature else None

    def update(self, updates):
        self.cached.update({frame: (signature, groups) for frame, signature, groups in updates})


class SparseWorld:
    """Original moving sample IDs with only queried world points resident.

    Closed-only moving summaries index a world's winning samples; they never
    inspect other points or the local array. Preserve its original length for
    sample-key construction, and retain exact original matmul coordinates.
    """

    def __init__(self, size):
        self.size, self.points = size, {}

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.points[int(index)]


def _moving_exact_batch(query, cid, batch):
    """Keep first exact minima and closed-component failures across batches."""
    if not batch:
        return
    exact = query.query(np.concatenate([item[2] for item in batch]), component_ids=[cid])
    begin = 0
    for components, ids, points, worlds, sparse in batch:
        end = begin + len(ids)
        signed = {key: value[begin:end] for key, value in exact.items()}
        reliable = ~signed["uncertain"] & np.isfinite(signed["clearance"])
        winner = int(np.argmin(signed["clearance"]) if reliable.all()
                     else np.flatnonzero(~reliable)[0])
        take = cid not in components
        if not take:
            _, old = components[cid]
            old_valid = not old["uncertain"][0] and np.isfinite(old["clearance"][0])
            take = old_valid and (not reliable.all() or signed["clearance"][winner] < old["clearance"][0])
        if take:
            sample = int(ids[winner])
            sparse.points[sample] = worlds[winner].copy()
            components[cid] = (ids[winner:winner + 1].copy(),
                               {key: value[winner:winner + 1].copy() for key, value in signed.items()})
        begin = end
    batch.clear()


def prepare_moving(geometry, poses, *, frames=None):
    """Batch closed-object minima; every possible winning row is float64 CPU.

    Moving forward summaries use only the first minimum per geom/component,
    its depth and its 8-mm eligibility. Retain all ambiguous samples and all
    robust candidates within 2*eps of their approximate minimum. Positive
    minima beyond the witness band contribute zero depth and no row. Open
    components, pseudonormal signing and nonunit prescribed scales fall back.
    Reverse samples and self collision keep their existing parallel CPU paths.
    """
    import mujoco
    from ..moving_scene import interpolate_object
    from ..scene_geometry import SceneQuery

    moving = geometry.moving
    if (moving is None or not moving.objects or moving.unknown_components
            or moving.cfg["activation_distance"] != .008
            or any(obj["query"].closed_sign_method != "winding"
                   or np.any(obj["surface"].scales != 1.) for obj in moving.objects)):
        return None
    cache = getattr(geometry, "_gpu_moving_cache", None)
    if cache is None:
        cache = geometry._gpu_moving_cache = MovingForwardCache()
    moving.forward_cache = cache
    changed = [frame for frame in (range(len(poses)) if frames is None else frames)
               if frame not in cache.cached or cache.cached[frame][0] != np.ascontiguousarray(poses[frame]).tobytes()]
    if not changed:
        geometry.gpu_moving_stats = dict(changed_frames=0, reused_geoms=0, screened_geoms=0)
        return []
    local = np.concatenate([shell["offsets"] for shell in moving.shells])
    starts = np.r_[0, np.cumsum([len(s["offsets"]) for s in moving.shells])]
    shell_ranges = {s["geom_id"]: (int(start), len(s["offsets"]))
                    for s, start in zip(moving.shells, starts)}
    tokens = getattr(geometry, "_gpu_moving_tokens", None)
    service = geometry_service()
    if tokens is None:
        tokens = {}
        for oid, obj in enumerate(moving.objects):
            for cid in obj["closed"]:
                part = SceneQuery(obj["query"].components[cid][0])
                tokens[oid, cid] = service.register(part, local, np.arange(len(local), dtype=np.int32))
        geometry._gpu_moving_tokens = tokens
    signatures = dict(getattr(geometry, "_gpu_moving_signatures", {}))
    updates, jobs, reused = [], [[] for _ in moving.objects], 0
    data = moving.data
    for frame in changed:
        pose = poses[frame]
        data.qpos[:] = pose
        mujoco.mj_forward(moving.model, data)
        old = cache.cached.get(frame)
        current = {shell["geom_id"]: (data.geom_xpos[shell["geom_id"]].tobytes(),
                                      data.geom_xmat[shell["geom_id"]].tobytes())
                   for shell in moving.shells}
        groups = []
        for oid, obj in enumerate(moving.objects):
            translation, rotation, scale = interpolate_object(obj["surface"], frame)
            samples = {}
            for shell in moving.shells:
                gid = shell["geom_id"]
                if np.linalg.norm(data.geom_xpos[gid] - translation) > obj["radius"] + shell["radius"] + .008:
                    continue
                if old is not None and signatures.get((frame, gid)) == current[gid]:
                    # Prescribed object motion and frame time are immutable.
                    # Keep the exact winner when this material transform is
                    # byte-identical, even if another joint changed the pose.
                    samples[gid] = old[1][oid][gid]
                    reused += 1
                    continue
                robot_rotation = data.geom_xmat[gid].reshape(3, 3)
                samples[gid] = (SparseWorld(len(shell["offsets"])), None, {})
                jobs[oid].append((samples[gid], shell_ranges[gid],
                                  (data.geom_xpos[gid] - translation) @ rotation / scale,
                                  rotation.T @ robot_rotation / scale, shell["offsets"],
                                  data.geom_xpos[gid].copy(), robot_rotation.copy(),
                                  translation, rotation, scale))
            groups.append(samples)
        signatures.update({(frame, gid): value for gid, value in current.items()})
        updates.append((frame, np.ascontiguousarray(pose).tobytes(), groups))
    debug = os.environ.get("OMNIRETARGETING_GPU_GEOMETRY_DEBUG", "0") != "0"
    for oid, obj in enumerate(moving.objects):
        active = jobs[oid]
        if not active:
            continue
        positions = np.array([job[2] for job in active])
        rotations = np.array([job[3] for job in active])
        begins = np.array([job[1][0] for job in active], np.int32)
        counts = np.array([job[1][1] for job in active], np.int32)
        choices = {}
        for cid in obj["closed"]:
            result = service.candidates(tokens[oid, cid], positions, rotations, begins, counts, debug,
                                        minimum=True)
            first = 0
            for job in np.unique(result["jobs"]):
                last = int(np.searchsorted(result["jobs"], job, side="right"))
                ids = result["samples"][first:last]
                phi = result["phi"][first:last]
                audit = result["debug"][first:last]
                ambiguous = result["ambiguous"][first:last]
                robust = ~audit & ~ambiguous & np.isfinite(phi)
                minimum = float(np.min(phi[robust], initial=np.inf))
                selected = ~audit & (ambiguous | (phi <= minimum + 2 * DISTANCE_BAND))
                if debug:
                    audit |= ~selected & ~ambiguous & (ids % 100 == 0)
                if not selected.any() and not audit.any():
                    first = last
                    continue
                choices.setdefault(int(job), {})[cid] = (ids[selected].astype(np.int64),
                                                         ids[audit], phi[audit])
                first = last
        batches = {cid: [] for cid in obj["closed"]}
        totals = dict.fromkeys(obj["closed"], 0)
        audit_points = {cid: [] for cid in obj["closed"]}
        audit_phi = {cid: [] for cid in obj["closed"]}
        for job, components in sorted(choices.items()):
            values, _, _, _, offsets, position, robot_rotation, translation, rotation, scale = active[job]
            # Preserve the original FULL matmul's float64 operation order.
            # Transform once per geom/object, shared by all closed components;
            # only this geom's dense arrays and compact candidates are live.
            world = offsets @ robot_rotation.T + position
            local_points = (world - translation) @ rotation / scale
            for cid, (chosen, audit, predicted) in components.items():
                if len(chosen):
                    bounds = obj["query"].components[cid][0].bounds
                    points = local_points[chosen]
                    gap = np.maximum(np.maximum(bounds[0] - points, points - bounds[1]), 0)
                    chosen = chosen[np.linalg.norm(gap, axis=1) <= .008 / scale]
                    first = 0
                    while first < len(chosen):
                        take = min(len(chosen) - first, geometry.SCENE_BATCH_POINTS - totals[cid])
                        ids = chosen[first:first + take]
                        batches[cid].append((values[2], ids, local_points[ids], world[ids], values[0]))
                        first, totals[cid] = first + take, totals[cid] + take
                        if totals[cid] == geometry.SCENE_BATCH_POINTS:
                            _moving_exact_batch(obj["query"], cid, batches[cid])
                            totals[cid] = 0
                if len(audit):
                    audit_points[cid].append(local_points[audit])
                    audit_phi[cid].append(predicted)
        for cid in obj["closed"]:
            if debug and audit_points[cid]:
                exact = obj["query"].query(np.concatenate(audit_points[cid]), component_ids=[cid])
                assert not exact["uncertain"].any(), "Moving GPU screening hid an unknown sample"
                error = np.abs(exact["clearance"] - np.concatenate(audit_phi[cid]))
                assert np.all(error <= DISTANCE_BAND), "Moving GPU error band exceeded"
                stats = getattr(geometry, "gpu_moving_audit", dict(points=0, max_error=0.))
                stats["points"] += len(error)
                stats["max_error"] = max(stats["max_error"], float(error.max(initial=0.)))
                geometry.gpu_moving_audit = stats
            _moving_exact_batch(obj["query"], cid, batches[cid])
        for values, *_ in active:
            ordered = {cid: values[2][cid] for cid in obj["closed"] if cid in values[2]}
            values[2].clear()
            values[2].update(ordered)
            wanted = {int(ids[0]) for ids, _ in values[2].values()}
            values[0].points = {sample: values[0].points[sample] for sample in wanted}
    cache.update(updates)
    geometry._gpu_moving_signatures = signatures
    geometry.gpu_moving_stats = dict(changed_frames=len(changed), reused_geoms=reused,
                                     screened_geoms=sum(map(len, jobs)))
    return updates


if __name__ == "__main__":
    from multiprocessing.connection import Connection
    _gpu_server(Connection(int(sys.argv[2])))
