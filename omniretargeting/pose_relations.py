"""Morphology-normalized relations between mapped body landmarks.

A pose is described by every pair of mapped landmarks,
``E_ij = (p_j - p_i) / l_ij``, where ``l_ij`` is the path length between the
two landmarks along that body's own landmark tree. For adjacent landmarks this
is the unit bone direction, so ``|dE_ij| = 2 sin(angle / 2)``; for distant
pairs it is their relative displacement in units of the chain between them.
One dimensionless quantity therefore covers directions and positions, and a
proximal change (trunk, shoulder, hip) alters more pairs than a distal one.

Changes are reported as equivalent angles: rotating a pose rigidly by ``theta``
about a uniformly random axis changes each relation on average by
``2 sin(theta / 2) * (pi / 4) * |E_ij|``. Inverting this for the whole-body mean
(or one landmark's mean) states "the shape changed as much as turning the whole
body by theta", independent of robot size and proportions.

References: Fu et al., "Morphology-agnostic humanoid retargeting via
perception-motivated graph similarity", Frontiers in Neurorobotics 2026
(all-pair edges normalized per body); Bernardin et al., "Normalized Euclidean
distance matrices for human motion retargeting", MIG 2017.
"""

import numpy as np

# Mean |n x e| of a unit vector e over uniformly random unit axes n.
RANDOM_AXIS_FACTOR = np.pi / 4
# Pairs whose landmarks coincide on either body have no direction to compare.
MIN_PATH_LENGTH_M = 1e-3


def landmark_parents(model, bodyids):
    """Nearest mapped ancestor of each mapped body (-1 for the tree root)."""
    mapped = {int(body): index for index, body in enumerate(bodyids)}
    parents = []
    for body in bodyids:
        parent = int(model.body_parentid[body])
        while parent and parent not in mapped:
            parent = int(model.body_parentid[parent])
        parents.append(mapped.get(parent, -1) if parent != int(body) else -1)
    return parents


def _tree_path(parents, first, second):
    def chain(node):
        nodes = [node]
        while parents[nodes[-1]] >= 0:
            nodes.append(parents[nodes[-1]])
        return nodes

    up, down = chain(first), chain(second)
    common = next(node for node in up if node in down)
    return up[: up.index(common) + 1], down[: down.index(common) + 1]


def path_lengths(parents, positions, pairs):
    """Tree path length of every pair; bone lengths are medians over frames."""
    positions = np.asarray(positions, dtype=float).reshape(-1, len(parents), 3)
    bones = np.zeros(len(parents))
    for child, parent in enumerate(parents):
        if parent >= 0:
            bones[child] = np.median(
                np.linalg.norm(positions[:, child] - positions[:, parent], axis=-1)
            )
    lengths = np.zeros(len(pairs))
    for index, (first, second) in enumerate(pairs):
        up, down = _tree_path(parents, first, second)
        lengths[index] = bones[up[:-1]].sum() + bones[down[:-1]].sum()
    return lengths


class PoseRelations:
    """All-pair normalized landmark relations of one robot and its source."""

    def __init__(self, parents, robot_positions, source_positions=None):
        """Without source positions only robot-side changes can be measured."""
        self.parents = list(parents)
        count = len(self.parents)
        pairs = np.array(
            [(i, j) for i in range(count) for j in range(i + 1, count)], dtype=int
        ).reshape(-1, 2)
        robot = path_lengths(self.parents, robot_positions, pairs)
        source = (robot if source_positions is None
                  else path_lengths(self.parents, source_positions, pairs))
        valid = (robot > MIN_PATH_LENGTH_M) & (source > MIN_PATH_LENGTH_M)
        self.pairs = pairs[valid]
        self.robot_lengths = robot[valid]
        self.source_lengths = source[valid]
        self.adjacent = np.array(
            [self.parents[j] == i or self.parents[i] == j for i, j in self.pairs], bool
        )
        # Mean over a landmark's pairs: one row per landmark, one column per pair.
        incidence = np.zeros((count, len(self.pairs)))
        incidence[self.pairs[:, 0], np.arange(len(self.pairs))] = 1.0
        incidence[self.pairs[:, 1], np.arange(len(self.pairs))] = 1.0
        self.incidence = incidence / np.maximum(incidence.sum(1, keepdims=True), 1.0)

    def relations(self, positions, *, source=False):
        positions = np.asarray(positions, dtype=float)
        lengths = self.source_lengths if source else self.robot_lengths
        delta = positions[..., self.pairs[:, 1], :] - positions[..., self.pairs[:, 0], :]
        return delta / lengths[:, None]

    def changes(self, first, second):
        """Per-pair ``|E_first - E_second|`` for relation arrays."""
        return np.linalg.norm(np.asarray(first) - np.asarray(second), axis=-1)

    def summary(self, changes):
        """Whole-body mean and per-landmark mean of per-pair changes."""
        changes = np.asarray(changes, dtype=float)
        # One product per frame: a frame's value must not depend on how many
        # frames share the call (a single GEMM blocks rows by their count).
        parts = np.matmul(changes[..., None, :], self.incidence.T)[..., 0, :]
        return changes.mean(-1), parts

    def equivalent_angles(self, changes, reference):
        """Whole-body and per-landmark equivalent rotation angles (radians).

        ``changes`` are per-pair changes from ``reference`` relations; the
        response of a rigid rotation is measured on the same reference.
        """
        changes = np.asarray(changes, dtype=float)
        scale = RANDOM_AXIS_FACTOR * np.linalg.norm(reference, axis=-1)
        whole = changes.mean(-1) / (2 * scale.mean(-1))
        parts = (changes @ self.incidence.T) / (2 * scale @ self.incidence.T)
        return (2 * np.arcsin(np.clip(whole, 0.0, 1.0)),
                2 * np.arcsin(np.clip(parts, 0.0, 1.0)))


def _block_matrix(blocks, width, selected=None):
    """Block-diagonal sparse matrix of per-frame (frames, rows, width) blocks."""
    from scipy import sparse

    frames, rows, _ = blocks.shape
    if selected is not None:
        chosen = np.flatnonzero(selected)
        column = (chosen[:, None] // rows * width + np.arange(width)).ravel()
        pointer = np.arange(len(chosen) + 1) * width
        values = blocks.reshape(-1, width)[chosen].ravel()
        return sparse.csr_matrix((values, column, pointer),
                                 shape=(len(chosen), frames * width))
    # Every row holds its frame's ``width`` columns in order: build CSR directly.
    column = (np.arange(frames)[:, None, None] * width + np.arange(width)).repeat(rows, 1).ravel()
    pointer = np.arange(frames * rows + 1) * width
    return sparse.csr_matrix((blocks.ravel(), column, pointer), shape=(frames * rows, frames * width))


def _sustained(mask, window):
    """Mark entries only after ``window`` consecutive True frames."""
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros_like(mask)
    if not mask.shape[0]:
        return result
    if not mask.any():
        return result
    if window <= 1:
        return mask.copy()
    run = np.zeros(mask.shape[1:], dtype=np.int32)
    for frame, row in enumerate(mask):
        np.multiply(run, row, out=run)
        np.add(run, row, out=run)
        np.minimum(run, window, out=run)
        result[frame] = run >= window
    return result


# A reference layout component within this chord of zero is ambiguous and may
# take either sign. Collapse requires a similarly clear opposite side, sustained
# for long enough that a brief transition or contact-avoidance crossing is not
# mistaken for a changed layout.
LAYOUT_MARGIN = 2 * np.sin(0.1)
LAYOUT_FLIP_MARGIN = LAYOUT_MARGIN
LAYOUT_FLIP_WINDOW_FRAMES = 7
# Largest normalized trunk-placement difference for a landmark to judge layout
# against the source (see ``comparable_trunk``).
TRUNK_PLACEMENT_TOLERANCE = 0.25
# Unit of the soft posture regularizer: a whole-body relation change equal to
# turning the reference by this angle (or by the robot's morphology angle to
# the source, when larger) costs one. It weights posture against the other
# tasks and never limits how far the pose may change.
POSTURE_SCALE_RAD = 0.2


class RelationEnvelope:
    """Fixed-reference posture preference on normalized landmark relations.

    Landmarks are the mapped bodies and one tip point per end effector. Posture
    is the lowest-priority task: a soft regularizer keeps relations closest to
    the fixed, source-shaped reference, in units of ``soft_angle``
    (``from_tasks`` uses the robot's own morphology angle to the source when it
    exceeds ``POSTURE_SCALE_RAD``). No angle limits how far the whole body or
    one landmark may change; changes are reported as equivalent angles and
    judged against the source motion. Layout is also a collapse criterion, not
    a per-frame posture cap: between different limbs, or a limb and the trunk,
    a source-clear relation component may spend brief or near-zero intervals on
    either side, but a sustained component on the clearly opposite side means
    the action is no longer recognizable. Bends within one limb are not layout.
    """

    SLACK = 1e-7

    def __init__(self, bodies, offsets, parents, reference_positions, names=None, axes=None,
                 source_positions=None):
        self.bodies = np.asarray(bodies, dtype=int)
        self.offsets = np.asarray(offsets, dtype=float).reshape(len(self.bodies), 3)
        self.names = [str(b) for b in self.bodies] if names is None else list(names)
        self.graph = PoseRelations(parents, reference_positions)
        self.reference = self.graph.relations(reference_positions)
        self.source_components = None
        if source_positions is not None:
            source_positions = np.asarray(source_positions, dtype=float)
            if source_positions.ndim != 3 or source_positions.shape[0] != len(reference_positions):
                raise ValueError("Source positions must have shape (frames, landmarks, 3)")
            count = len(source_positions[0])
            self.source_landmark_count = count
            source_graph = PoseRelations(
                parents[:count], np.asarray(reference_positions)[:, :count], source_positions
            )
            source_relations = source_graph.relations(source_positions, source=True)
            lookup = {tuple(pair): index for index, pair in enumerate(source_graph.pairs)}
            selected = [lookup.get(tuple(pair)) for pair in self.graph.pairs]
            mask = np.array([index is not None for index in selected])
            components = np.zeros((len(reference_positions), len(self.graph.pairs), 3))
            components[:, mask] = source_relations[:, [i for i in selected if i is not None]]
            self.source_components = components
            self._source_fit = _fit_rotations(np.asarray(reference_positions)[:, :count], source_positions)
            # Solver hinges select pairs with these historical axes (bit-identical
            # solves); they are not the source body frame used by source_flips.
            self._hinge_source_axes = heading_axes(_hinge_rotations(
                np.asarray(reference_positions)[:, :count], source_positions))
            self._source_comparable = comparable_trunk(parents, np.asarray(reference_positions), source_positions)
            self.source_axes = None
        scale = RANDOM_AXIS_FACTOR * np.linalg.norm(self.reference, axis=-1)
        self.whole_scale = scale.mean(-1)
        self.part_scale = scale @ self.graph.incidence.T
        self.morphology_angle = None
        self.set_soft_angle(POSTURE_SCALE_RAD)
        self.axes = self.layout_sign = None
        self.source_axes = getattr(self, "source_axes", None)
        self.source_layout_sign = None
        if axes is not None:
            self.axes = np.asarray(axes, dtype=float)
            components = np.einsum("fpk,fak->fpa", self.reference, self.axes)
            limb = limb_ids(self.graph.parents)
            first, second = self.graph.pairs.T
            across = (limb[first] < 0) | (limb[second] < 0) | (limb[first] != limb[second])
            reference_sign = np.where(
                (np.abs(components) >= LAYOUT_MARGIN) & across[None, :, None],
                np.sign(components), 0.0,
            )
            self.layout_sign = reference_sign
            if self.source_components is not None:
                # The source's own body axes: the robot reference heading carried
                # into the source by the per-frame best-fit rotation.
                self.source_axes = heading_axes(
                    np.swapaxes(self._source_fit, 1, 2) @ np.swapaxes(self.axes, 1, 2))
                def signs(axes):
                    values = np.einsum("fpk,fak->fpa", self.source_components, axes)
                    return np.where((np.abs(values) >= LAYOUT_MARGIN) & across[None, :, None],
                                    np.sign(values), 0.0)
                # Effector tips have no mapped source landmark. Pairs without a
                # source observation remain soft posture terms only.
                source_count = self.source_landmark_count
                observed = ((first < source_count) & (second < source_count))[None, :, None]
                hinge_sign = signs(self._hinge_source_axes)
                self.layout_sign = np.where(
                    observed & (hinge_sign == reference_sign) & (hinge_sign != 0),
                    reference_sign, 0.0,
                )
                comparable = (self._source_comparable[first] & self._source_comparable[second])[None, :, None]
                source_sign = signs(self.source_axes)
                self.source_layout_sign = np.where(observed & comparable & (source_sign != 0),
                                                   source_sign, 0.0)

    def set_soft_angle(self, angle):
        self.soft_angle = float(angle)
        self.soft_scale = 2 * np.sin(self.soft_angle / 2) * self.whole_scale

    @classmethod
    def from_tasks(cls, tasks, kin):
        """Mapped bodies plus effector tips of a sequence task set, on ``kin``."""
        model = tasks.model
        bodies = [int(b) for b in tasks.bodyids]
        parents = landmark_parents(model, bodies)
        offsets = [np.zeros(3) for _ in bodies]
        names = list(tasks.names)
        mapped = list(bodies)
        for name, spec in tasks.effectors.items():
            body = anchor = int(spec["body_id"])
            while anchor and anchor not in mapped:
                anchor = int(model.body_parentid[anchor])
            if anchor not in mapped:
                continue
            bodies.append(body)
            offsets.append(np.asarray(spec.get("tracking_offset", np.zeros(3)), dtype=float))
            parents.append(mapped.index(anchor))
            names.append(f"{name}_tip")
        points = kin.points(np.asarray(bodies), np.asarray(offsets))
        envelope = cls(bodies, offsets, parents, points, names,
                       axes=heading_axes(kin.xmat[:, mapped[parents.index(-1)]]),
                       source_positions=getattr(tasks, "human", None))
        human = getattr(tasks, "human", None)
        if human is not None:
            envelope.morphology_angle = float(np.median(
                source_relation_angles(envelope, kin, np.asarray(human))))
            envelope.set_soft_angle(max(POSTURE_SCALE_RAD, envelope.morphology_angle))
        return envelope

    def for_frames(self, frames):
        import copy

        chunk = copy.copy(self)
        for name in ("reference", "source_components", "whole_scale", "part_scale", "soft_scale",
                     "axes", "source_axes", "layout_sign", "source_layout_sign"):
            value = getattr(self, name)
            if value is not None:
                setattr(chunk, name, value[frames])
        return chunk

    def measure(self, kin, with_jacobian=False):
        """Per-frame whole and per-landmark mean changes, with the per-pair
        ``(change, pair_jacobians)``; the Jacobians are None unless requested."""
        points = kin.points(self.bodies, self.offsets)
        change = self.graph.relations(points) - self.reference
        whole, parts = self.graph.summary(np.linalg.norm(change, axis=-1))
        if not with_jacobian:
            return whole, parts, (change, None)
        jac = kin.point_jacobians(self.bodies, points)
        first, second = self.graph.pairs.T
        pairs = (jac[:, second] - jac[:, first]) / self.graph.robot_lengths[:, None, None]
        return whole, parts, (change, pairs)

    def layout(self, change, pairs=None):
        """Signed source-layout components (frames, pairs, 3), with Jacobians.

        Positive values are on the source side; values above the negative flip
        margin remain in the transient/near-zero deadband.
        """
        if self.layout_sign is None:
            return None, None
        components = np.einsum("fpk,fak->fpa", change + self.reference, self.axes)
        value = self.layout_sign * components
        jac = None
        if pairs is not None:
            jac = self.layout_sign[..., None] * np.einsum("fak,fpkw->fpaw", self.axes, pairs)
        return value, jac

    def equivalent(self, whole, parts):
        """Raw mean changes as equivalent rigid-rotation angles (radians)."""
        def angle(value, scale):
            ratio = np.divide(value, 2 * scale, out=np.zeros_like(value), where=scale > 0)
            return 2 * np.arcsin(np.clip(ratio, 0.0, 1.0))

        return angle(whole, self.whole_scale), angle(parts, self.part_scale)

    def flips(self, kin):
        """Largest sustained clear-layout collapse per frame."""
        _, _, (change, _) = self.measure(kin)
        value, _ = self.layout(change)
        if value is None:
            return np.zeros(len(change))
        return self._collapse_excess(value).max(axis=(1, 2), initial=0.0)

    def _collapse_excess(self, value):
        deep = value < -LAYOUT_FLIP_MARGIN
        excess = np.maximum(-value - LAYOUT_FLIP_MARGIN, 0.0)
        return np.where(_sustained(deep, LAYOUT_FLIP_WINDOW_FRAMES), excess, 0.0)

    def violations(self, kin):
        return self.flips(kin) > self.SLACK

    def source_flips(self, kin):
        """Sustained clear flips against the original source landmarks."""
        _, _, (change, _) = self.measure(kin)
        if self.source_layout_sign is None:
            return np.zeros(len(change))
        axes = self.source_axes if self.source_axes is not None else self.axes
        components = np.einsum("fpk,fak->fpa", change + self.reference, axes)
        value = self.source_layout_sign * components
        return self._collapse_excess(value).max(axis=(1, 2), initial=0.0)

    def accepts(self, kin):
        return not self.violations(kin).any()

    def summary(self, kin):
        whole, parts, _ = self.measure(kin)
        whole_angle, part_angle = self.equivalent(whole, parts)
        frame, part = np.unravel_index(np.argmax(part_angle), part_angle.shape)
        flip = self.flips(kin)
        _, _, (change, _) = self.measure(kin)
        value, _ = self.layout(change)
        source_flip = self.source_flips(kin)
        return {
            "soft_angle_rad": self.soft_angle,
            "soft_angle_rule": "max(posture scale, robot morphology angle to source); a weight, not a limit",
            "morphology_angle_rad": self.morphology_angle,
            "whole_angle_max_rad": float(whole_angle.max(initial=0)),
            "whole_angle_frame": int(np.argmax(whole_angle)) if len(whole_angle) else 0,
            "part_angle_max_rad": float(part_angle.max(initial=0)),
            "part_angle_landmark": self.names[int(part)] if part_angle.size else None,
            "part_angle_frame": int(frame),
            "layout_relations": 0 if self.layout_sign is None else int(np.count_nonzero(self.layout_sign)),
            "layout_flip_frames": np.flatnonzero(flip > self.SLACK).tolist(),
            "source_layout_flip_frames": np.flatnonzero(source_flip > self.SLACK).tolist(),
            "layout_clear_opposite_frames": np.flatnonzero(
                (value < -LAYOUT_FLIP_MARGIN).any(axis=(1, 2))
                if value is not None else []
            ).tolist(),
        }

    def hinge_rows(self, kin, weight, sigma, with_jacobian):
        """Stiff hinges on layout flips, plus the soft minimum-change
        regularizer (scaled by each frame's ``soft_scale``).

        Returns rows, their Jacobians (or None) and the largest layout flip.
        """
        _, _, (change, pairs) = self.measure(kin, with_jacobian)
        frames, count = change.shape[:2]
        rows, jacs, excess = [], [], 0.0
        value, layout_jac = self.layout(change, pairs)
        if value is not None:
            stiff = np.asarray(weight, dtype=float)[:, None] / sigma
            flip = self._collapse_excess(value).reshape(frames, -1)
            rows.append(flip * stiff)
            excess = float(flip.max(initial=0))
            if with_jacobian:
                active = (flip > 0) * stiff
                jacs.append(-layout_jac.reshape(frames, -1, layout_jac.shape[-1]) * active[..., None])
        soft = 1.0 / (np.maximum(self.soft_scale, 1e-9) * np.sqrt(max(count, 1)))
        rows.append(change.reshape(frames, -1) * soft[:, None])
        if with_jacobian:
            jacs.append(pairs.reshape(frames, -1, pairs.shape[-1]) * soft[:, None, None])
        rows = np.concatenate(rows, axis=1)
        return rows, (np.concatenate(jacs, axis=1) if with_jacobian else None), excess

    def linear_bounds(self, kin, width, box=None, *, restore=False):
        """Halfspaces for source layout, with transient and near-zero freedom.

        A component may cross zero until it is clearly on the opposite side.
        Once that opposite side is sustained, nonlinear acceptance rejects it
        and these rows restore it to the source side. With ``box``, a step that
        cannot reach the allowed floor is omitted.
        """
        _, _, (change, pairs) = self.measure(kin, with_jacobian=True)
        value, jac = self.layout(change, pairs)
        if value is None:
            return []
        keep = self.layout_sign != 0
        deep = value < -LAYOUT_FLIP_MARGIN
        sustained = _sustained(deep, LAYOUT_FLIP_WINDOW_FRAMES)
        floor = (np.zeros_like(value) if restore
                 else np.where(sustained, 0.0, -LAYOUT_FLIP_MARGIN))
        if box is not None:
            # A component whose distance to the floor exceeds its largest
            # change cannot violate this linearization.
            reach = np.einsum("fpaw,fw->fpa", np.abs(jac), np.asarray(box, float))
            keep &= reach >= value - floor
        keep = keep.ravel()
        if not keep.any():
            return []
        matrix = _block_matrix(jac.reshape(len(value), -1, width), width, keep)
        # Preserve the deadband; a numerically touching component stays feasible.
        margin = value.ravel()[keep]
        lower = floor.ravel()[keep] - margin - self.SLACK / 2
        return [(matrix, lower, np.full(len(lower), np.inf))]


def layout_constraints(bounds, flat):
    """CVXPY constraints of several envelopes' ``linear_bounds`` on one step;
    identical halfspace rows keep their tightest bound."""
    from scipy import sparse

    layout = [item for items in bounds for item in items]
    if not layout:
        return []
    rows = sparse.vstack([item[0] for item in layout], format="csr")
    lower = np.concatenate([item[1] for item in layout])
    rows, lower = _tightest_lower(rows, lower)
    return [rows @ flat >= lower]


def _row_groups(matrix, extra=None):

    """Group ids of bitwise identical sparse rows (and ``extra`` columns)."""
    from scipy import sparse

    matrix = sparse.csr_matrix(matrix, copy=True)
    matrix.sum_duplicates()
    rows = matrix.shape[0]
    # A fixed probe only proposes candidates; equality is then checked exactly.
    key = np.c_[matrix @ np.random.default_rng(0).uniform(1., 2., matrix.shape[1]),
                np.diff(matrix.indptr), np.zeros((rows, 0)) if extra is None else extra]
    order = np.lexsort(key.T[::-1])
    same = np.all(key[order[1:]] == key[order[:-1]], axis=1)
    pairs = np.flatnonzero(same)
    if len(pairs):
        difference = matrix[order[pairs]] - matrix[order[pairs + 1]]
        difference.eliminate_zeros()
        same[pairs] = np.diff(difference.indptr) == 0
    group = np.empty(rows, int)
    group[order] = np.cumsum(np.r_[True, ~same]) - 1
    return group


def _tightest_lower(matrix, lower):
    """Identical halfspace rows collapse to their largest lower bound."""
    group = _row_groups(matrix)
    count = group.max() + 1
    first = np.full(count, len(lower))
    np.minimum.at(first, group, np.arange(len(lower)))
    tight = np.full(count, -np.inf)
    np.maximum.at(tight, group, lower)
    order = np.argsort(first)
    return matrix[first[order]], tight[order]


def limb_ids(parents):
    """Limb of every landmark from the tree alone; -1 on the trunk.

    The trunk is every landmark that is an ancestor of (or is) a branching
    landmark. A limb starts at a non-trunk child of the trunk and contains all
    of its descendants. No landmark names are used.
    """
    parents = list(parents)
    children = [[] for _ in parents]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(child)
    trunk = set()
    for node, kids in enumerate(children):
        if len(kids) >= 2:
            while node >= 0 and node not in trunk:
                trunk.add(node)
                node = parents[node]
    limb = np.full(len(parents), -1)
    for node in range(len(parents)):
        if node in trunk:
            continue
        start = node
        while parents[start] >= 0 and parents[start] not in trunk:
            start = parents[start]
        limb[node] = start
    return limb


def heading_axes(rotations):
    """Per-frame (forward, left, up) rows: the root's heading and world up."""
    rotations = np.asarray(rotations, dtype=float).reshape(-1, 3, 3)
    up = np.array([0.0, 0.0, 1.0])
    forward = rotations[:, :, 0] - np.outer(rotations[:, 2, 0], up)
    side = rotations[:, :, 1] - np.outer(rotations[:, 2, 1], up)
    weak = np.linalg.norm(forward, axis=1) < 1e-6
    forward[weak] = np.cross(side[weak], up)
    forward /= np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-12)
    left = np.cross(up, forward)
    return np.stack([forward, left, np.broadcast_to(up, forward.shape)], axis=1)


def _fit_rotations(reference, source):
    """Per-frame proper rotations R minimizing |R source - reference| (Kabsch)."""
    reference, source = np.asarray(reference, float), np.asarray(source, float)
    centered_reference = reference - reference.mean(axis=1, keepdims=True)
    centered_source = source - source.mean(axis=1, keepdims=True)
    u, _, vt = np.linalg.svd(np.einsum("fki,fkj->fij", centered_source, centered_reference))
    v = np.swapaxes(vt, 1, 2).copy()
    d = np.sign(np.linalg.det(v @ np.swapaxes(u, 1, 2)))
    v[:, :, -1] *= d[:, None]
    return v @ np.swapaxes(u, 1, 2)


def _hinge_rotations(reference, source):
    """Historical per-frame fit used only to select solver hinge pairs.

    It returns V^T U^T rather than the Kabsch V U^T; kept so that solves stay
    bit-identical while layout judgement uses ``_fit_rotations``.
    """
    reference, source = np.asarray(reference, float), np.asarray(source, float)
    centered_reference = reference - reference.mean(axis=1, keepdims=True)
    centered_source = source - source.mean(axis=1, keepdims=True)
    left, _, right = np.linalg.svd(np.einsum("fki,fkj->fij", centered_source, centered_reference))
    result = np.einsum("fij,fkj->fik", right, left)
    reflect = np.linalg.det(result) < 0
    right[reflect, :, -1] *= -1
    result[reflect] = np.einsum("fij,fkj->fik", right[reflect], left[reflect])
    return result


def comparable_trunk(parents, reference, source, tolerance=TRUNK_PLACEMENT_TOLERANCE):
    """Trunk landmarks whose placement along the trunk matches the source.

    The trunk spans the two farthest-apart groups of limb roots (e.g. hips and
    shoulders). Each trunk landmark is placed by its normalized position on
    that span. When robot and source positions differ by more than
    ``tolerance`` (e.g. a chest point mapped to a waist-level torso origin),
    its relations do not describe the same geometry: a hanging elbow lies
    below the source chest but above the robot point. Such landmarks cannot
    judge layout against the source. Uses the tree only, no names.
    """
    parents = list(parents)
    count = source.shape[1]
    limb = limb_ids(parents[:count])
    groups = {}
    for child in range(count):
        parent = parents[child]
        if parent >= 0 and limb[child] >= 0 and limb[parent] < 0:
            groups.setdefault(parent, []).append(child)
    keep = np.ones(len(parents), bool)
    if len(groups) < 2:
        return keep
    reference = np.asarray(reference)[:, :count]
    centres = {k: reference[:, v].mean(axis=1) for k, v in groups.items()}
    keys = list(groups)
    lower, upper = max(((i, j) for i in keys for j in keys if i < j),
                       key=lambda ij: np.median(np.linalg.norm(centres[ij[0]] - centres[ij[1]], axis=1)))
    def placement(points, node):
        start, stop = points[:, groups[lower]].mean(axis=1), points[:, groups[upper]].mean(axis=1)
        axis = stop - start
        return float(np.median(np.einsum("fk,fk->f", points[:, node] - start, axis)
                               / np.maximum(np.einsum("fk,fk->f", axis, axis), 1e-12)))
    for node in np.flatnonzero(limb < 0):
        keep[node] = abs(placement(reference, node) - placement(source, node)) <= tolerance
    return keep


def source_relation_angles(envelope, kin, source_positions):
    """Whole-body equivalent angle between mapped robot and source relations.

    Uses the mapped landmarks only (effector tips have no source point). This
    is the posture measure against the original motion; it is reported, not
    bounded.
    """
    mapped = len(source_positions[0])
    points = kin.points(envelope.bodies[:mapped], envelope.offsets[:mapped])
    parents = envelope.graph.parents[:mapped]
    graph = PoseRelations(parents, points, source_positions)
    robot = graph.relations(points)
    source = graph.relations(np.asarray(source_positions), source=True)
    change = np.linalg.norm(robot - source, axis=-1).mean(-1)
    scale = RANDOM_AXIS_FACTOR * np.linalg.norm(source, axis=-1).mean(-1)
    return 2 * np.arcsin(np.clip(change / (2 * scale), 0.0, 1.0))
