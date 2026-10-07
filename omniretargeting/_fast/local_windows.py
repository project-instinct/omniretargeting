"""Active sets of the local repair: which frames, joints and bodies move together.

A window is a time range x joint scope whose variables are coupled by the
defects inside it. Penetration windows come from original-mesh collision keys;
chatter windows from source-relative high-frequency excess of the joints.
"""

import numpy as np
from scipy import sparse

from .limb_scope import scope_for_body
from .mesh_repair import COLLISION_SIGMA, LIMB_COLLISION_SIGMA, _group_rank, _joint_projection
from .repair_frames import halo_frames, restrict
from .repair_limits import CONTACT_DRIFT, RepairLimits

# A run of defect frames longer than this is split into even windows (frames).
WINDOW_MAX_FRAMES = 240


class ClusterLimits(RepairLimits):
    """RepairLimits whose correcting contacts are granted per (body, frame).

    A window may move its own bodies' contacts toward their original targets
    on its own frames, as one repair group did; every other contact keeps its
    anchor. Windows of one wave grant disjoint frame ranges, so each worker
    checks the whole trajectory under the same permissions the parent holds.
    """

    def __init__(self, objective, entry):
        from .local_slip import entry_points
        super().__init__(objective, entry)
        self.permission = {}
        self.entry_surface_points = entry_points(self)

    @classmethod
    def adopt(cls, limits):
        """Use accumulated repair limits without rebuilding their evidence.

        Entry sample positions are measured temporarily; the caller's current
        FK arrays, Jacobians, cache and MuJoCo data retain their original state.
        """
        import mujoco
        from .local_slip import entry_points

        limits.__class__ = cls
        limits.permission = {}
        kin = limits.kin
        saved = kin.__dict__.copy()
        try:
            kin.data = mujoco.MjData(kin.model)
            kin.cache = kin.cache.copy()
            kin.evaluate(limits.entry, False)
            limits.entry_surface_points = entry_points(limits)
        finally:
            kin.__dict__.clear()
            kin.__dict__.update(saved)
        return limits

    def grant(self, windows):
        """Correcting permission of every window of a wave on its frames."""
        self.permission = {}
        for window in windows:
            for body in window["bodies"]:
                self.permission[body] = self.permission.get(body, np.zeros(self.frames, bool)) | window["mask"]

    def correction_mask(self, bodies, budget):
        mask = np.zeros_like(budget, dtype=bool)
        if not self.permission:
            return mask
        weld = self.objective.model.body_weldid[bodies]
        for body, frames in self.permission.items():
            mask |= frames[:, None] & (weld == body)[None]
        return mask & (budget == CONTACT_DRIFT)

    def release(self, current):
        """Anchor the finished wave's contacts where they now are (cf. transition_group)."""
        if self.permission:
            self.kin.evaluate(current, False)
            for index, (points, contact_bodies, budget) in enumerate(self.point_groups()):
                acquired = self.correction_mask(contact_bodies, budget)
                self.anchors[index][acquired] = points[acquired]
                error = np.linalg.norm(points - self.targets[index], axis=-1)
                self.target_budgets[index][acquired] = np.minimum(
                    self.target_budgets[index][acquired], error[acquired] + self.contact_tolerances[index])
        self.permission = {}
        self.movable_body = None


def key_bodies(model, key):
    geoms = key[2:] if key[1] == "self" else key[-1:]
    return {int(model.body_weldid[model.geom_bodyid[g]]) for g in geoms}


def body_columns(limits, model, measured, body):
    """Joints the established group projection moves for ``body``."""
    saved = limits.movable_body
    limits.movable_body = body
    try:
        projection = _joint_projection(limits, model, measured)
    finally:
        limits.movable_body = saved
    return set(np.unique(projection.nonzero()[0] % limits.width).tolist())


def _dilate(mask, halo):
    frames = len(mask)
    return np.convolve(np.r_[np.zeros(halo), mask, np.zeros(halo)],
                       np.ones(2 * halo + 1), "valid")[:frames] > 0


def _runs(mask):
    padded = np.r_[False, mask, False].astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2], edges[1::2]))


def _pieces(hit, halo, longest):
    """Haloed runs of ``hit``; a run longer than ``longest`` is split evenly."""
    frames = len(hit)
    result = []
    for start, stop in _runs(_dilate(hit, halo)):
        count = int(np.ceil((stop - start) / longest))
        edges = np.linspace(start, stop, count + 1).round().astype(int)
        for first, last in zip(edges[:-1], edges[1:]):
            inside = np.zeros(frames, bool)
            inside[first:last] = hit[first:last]
            if count > 1:
                inside = _dilate(inside, halo)
                inside[:max(start, first - halo)] = False
                inside[min(stop, last + halo):] = False
            else:
                inside[first:last] = True
            if inside.any():
                result.append(inside)
    return result


def penetration_windows(objective, limits, measured, activation, longest=WINDOW_MAX_FRAMES):
    """Batches: a coupled run of defect frames x the joints its bodies need.

    A body with a defect deeper than ``activation`` is repaired wherever it
    penetrates at all (the established group frame domain). Each run of its
    penetrating frames, widened by the temporal halo, is one unit; a run
    longer than ``longest`` frames is split evenly. A unit's joints come from
    the defects inside it (a self-collision partner at another time does not
    couple it). Units of one contact rank merge when their frames overlap and
    their joints intersect. Batches are ordered by contact rank, then depth.
    """
    model, frames = objective.model, objective.frames
    halo = halo_frames(objective.framerate)
    hits, depth = {}, {}
    for key, value in measured["depths"].items():
        if value <= 0:
            continue
        for body in key_bodies(model, key):
            hits.setdefault(body, np.zeros(frames, bool))[key[0]] = True
            depth[body] = max(depth.get(body, 0.), value)
    hits = {body: hit for body, hit in hits.items() if depth[body] >= activation}
    units = []
    for body, hit in hits.items():
        for mask in _pieces(hit, halo, longest):
            local = dict(measured, depths={k: v for k, v in measured["depths"].items() if mask[k[0]]})
            cols = body_columns(limits, model, local, body)
            if cols:
                units.append((body, mask, cols, _group_rank(objective, body)))
    parent = list(range(len(units)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, a in enumerate(units):
        for j in range(i + 1, len(units)):
            b = units[j]
            if a[3] == b[3] and (a[1] & b[1]).any() and a[2] & b[2]:
                parent[find(j)] = find(i)
    groups = {}
    for index in range(len(units)):
        groups.setdefault(find(index), []).append(units[index])
    result = []
    for members in groups.values():
        bodies = sorted({m[0] for m in members}, key=lambda b: -depth[b])
        mask = np.logical_or.reduce([m[1] for m in members])
        result.append(dict(kind="penetration", bodies=bodies, block=int(np.flatnonzero(mask)[0]),
                           mask=mask, columns=sorted(set().union(*(m[2] for m in members))),
                           rank=members[0][3], depth=max(depth[b] for b in bodies)))
    return sorted(result, key=lambda w: (w["rank"], -w["depth"], w["block"]))


def chatter_windows(chatter, poses, width, halo=2):
    """Runs of frames with audited joint chatter, widened by the kernel reach."""
    from .local_chatter import KERNEL_RADIUS
    bad = (chatter.excess(poses) > 0).any(axis=1)
    if not bad.any():
        return []
    mask = _dilate(bad, KERNEL_RADIUS + halo)
    padded = np.r_[False, mask, False].astype(int)
    edges = np.flatnonzero(np.diff(padded))
    result = []
    for start, stop in zip(edges[0::2], edges[1::2]):
        window = np.zeros(len(mask), bool)
        window[start:stop] = True
        result.append(dict(kind="chatter", bodies=[], block=int(start), mask=window,
                           columns=list(range(width)), rank=-1, depth=0.))
    return result


def projection(limits, window):
    selector = sparse.eye(limits.width, format="csc")[:, window["columns"]]
    return restrict(sparse.kron(sparse.eye(limits.frames), selector, format="csc"), window["mask"])


def witness_rows(geometry, measured, mask):
    rows = [(key, row) for key, row in measured["rows"] if mask[key[0]]]
    matrix, phi = geometry.linear_rows(dict(rows=rows))
    return matrix, phi, [key for key, _ in rows]


def objective_keys(model, measured, mask, bodies, limits):
    keys, sigmas = [], []
    for key, value in measured["depths"].items():
        if value <= 0 or not mask[key[0]]:
            continue
        owners = key_bodies(model, key) & set(bodies)
        if not owners:
            continue
        keys.append(key)
        sigmas.append(min(LIMB_COLLISION_SIGMA if scope_for_body(limits, b) is not None
                          else COLLISION_SIGMA for b in owners))
    return keys, sigmas


def body_depths(model, depths, bodies, mask):
    """Deepest key of each batch body inside the window."""
    result = {body: 0. for body in bodies}
    for key, value in depths.items():
        if mask[key[0]]:
            for body in key_bodies(model, key) & set(bodies):
                result[body] = max(result[body], value)
    return result


def predicted_depths(model, keys, values, bodies):
    result = {body: 0. for body in bodies}
    for key, value in zip(keys, values):
        for body in key_bodies(model, key) & set(bodies):
            result[body] = max(result[body], float(value))
    return result


def progress(before, after):
    """Largest decrease of any batch body's deepest key (one body may be stuck)."""
    return max((before[b] - after.get(b, 0.) for b in before), default=0.)
