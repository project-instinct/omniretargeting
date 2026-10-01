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


def _block_matrix(blocks, width):
    """Block-diagonal sparse matrix of per-frame (frames, rows, width) blocks."""
    from scipy import sparse

    frames, rows, _ = blocks.shape
    # Every row holds its frame's ``width`` columns in order: build CSR directly.
    column = (np.arange(frames)[:, None, None] * width + np.arange(width)).repeat(rows, 1).ravel()
    pointer = np.arange(frames * rows + 1) * width
    return sparse.csr_matrix((blocks.ravel(), column, pointer), shape=(frames * rows, frames * width))


# A reference layout component within this chord of zero is ambiguous and may
# take either sign; the no-flip rule applies to clear components only.
LAYOUT_MARGIN = 2 * np.sin(0.1)
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
    judged against the source motion. The only hard rule is the layout: between
    different limbs, or a limb and the trunk, relation components that are
    clear in the reference's heading frame (front/back, left/right, up/down)
    may approach zero but not flip. Bends within one limb are not layout.
    """

    SLACK = 1e-7

    def __init__(self, bodies, offsets, parents, reference_positions, names=None, axes=None):
        self.bodies = np.asarray(bodies, dtype=int)
        self.offsets = np.asarray(offsets, dtype=float).reshape(len(self.bodies), 3)
        self.names = [str(b) for b in self.bodies] if names is None else list(names)
        self.graph = PoseRelations(parents, reference_positions)
        self.reference = self.graph.relations(reference_positions)
        scale = RANDOM_AXIS_FACTOR * np.linalg.norm(self.reference, axis=-1)
        self.whole_scale = scale.mean(-1)
        self.part_scale = scale @ self.graph.incidence.T
        self.morphology_angle = None
        self.set_soft_angle(POSTURE_SCALE_RAD)
        self.axes = self.layout_sign = None
        if axes is not None:
            self.axes = np.asarray(axes, dtype=float)
            components = np.einsum("fpk,fak->fpa", self.reference, self.axes)
            limb = limb_ids(self.graph.parents)
            first, second = self.graph.pairs.T
            across = (limb[first] < 0) | (limb[second] < 0) | (limb[first] != limb[second])
            self.layout_sign = np.where((np.abs(components) >= LAYOUT_MARGIN) & across[None, :, None],
                                        np.sign(components), 0.0)

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
                       axes=heading_axes(kin.xmat[:, mapped[parents.index(-1)]]))
        human = getattr(tasks, "human", None)
        if human is not None:
            envelope.morphology_angle = float(np.median(
                source_relation_angles(envelope, kin, np.asarray(human))))
            envelope.set_soft_angle(max(POSTURE_SCALE_RAD, envelope.morphology_angle))
        return envelope

    def for_frames(self, frames):
        import copy

        chunk = copy.copy(self)
        for name in ("reference", "whole_scale", "part_scale", "soft_scale", "axes", "layout_sign"):
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
        """Signed layout components (frames, pairs, 3) that must not flip:
        ``sign * component >= 0``, with their Jacobians when ``pairs`` given."""
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
        """Largest layout flip per frame; positive where a clear component changed sign."""
        _, _, (change, _) = self.measure(kin)
        value, _ = self.layout(change)
        if value is None:
            return np.zeros(len(change))
        return np.max(np.where(self.layout_sign != 0, -value, -np.inf), axis=(1, 2), initial=-np.inf)

    def violations(self, kin):
        return self.flips(kin) > self.SLACK

    def accepts(self, kin):
        return not self.violations(kin).any()

    def summary(self, kin):
        whole, parts, _ = self.measure(kin)
        whole_angle, part_angle = self.equivalent(whole, parts)
        frame, part = np.unravel_index(np.argmax(part_angle), part_angle.shape)
        flip = self.flips(kin)
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
            flip = np.where(self.layout_sign != 0, np.maximum(-value, 0.0), 0.0).reshape(frames, -1)
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
        """Linearized layout halfspaces ``(matrix, lower, upper=inf)``: clear
        components keep their current side; exact nonlinear acceptance follows
        every proposal. With ``box`` (frames, width), the largest absolute step
        per coordinate, a component no step in the box can flip is omitted."""
        _, _, (change, pairs) = self.measure(kin, with_jacobian=True)
        value, jac = self.layout(change, pairs)
        if value is None:
            return []
        keep = self.layout_sign != 0
        if box is not None:
            # A component whose margin exceeds its largest change cannot flip.
            reach = np.einsum("fpaw,fw->fpa", np.abs(jac), np.asarray(box, float))
            keep &= reach >= value
        keep = keep.ravel()
        if not keep.any():
            return []
        matrix = _block_matrix(jac.reshape(len(value), -1, width), width)[keep]
        # Keep the current side; a numerically touching component stays feasible.
        margin = value.ravel()[keep]
        # An outside restoration must return to the immutable legal side.
        # Clamping a negative margin only forbids further worsening.
        lower = -(margin if restore else np.maximum(margin, 0.0)) - self.SLACK / 2
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
