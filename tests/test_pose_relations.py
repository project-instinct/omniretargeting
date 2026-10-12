"""Normalized landmark relations: one unit for directions and positions."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast.reference import ReferenceShape
from omniretargeting.pose_relations import PoseRelations
from tests.test_fast_backend import _objective

# A small tree: pelvis -> spine -> (left arm, right arm), pelvis -> leg.
PARENTS = [-1, 0, 1, 2, 1, 4, 0, 6]
POINTS = np.array([
    [0, 0, 1.0], [0, 0, 1.4], [0, .2, 1.4], [0, .2, 1.1],
    [0, -.2, 1.4], [0, -.2, 1.1], [0, .1, .5], [0, .1, 0.0],
])


def test_adjacent_pair_change_is_the_bone_angle_chord():
    graph = PoseRelations(PARENTS, POINTS)
    moved = POINTS.copy()
    angle = .3
    # Turn the left forearm (landmark 3 about landmark 2) by ``angle``.
    moved[3] = POINTS[2] + Rotation.from_rotvec([angle, 0, 0]).apply(POINTS[3] - POINTS[2])
    change = graph.changes(graph.relations(moved), graph.relations(POINTS))
    bone = np.flatnonzero((graph.pairs == [2, 3]).all(1))[0]
    assert np.isclose(change[bone], 2 * np.sin(angle / 2))


def test_relations_are_invariant_to_size_and_translation():
    graph = PoseRelations(PARENTS, POINTS)
    large = PoseRelations(PARENTS, 2.5 * POINTS + [3, -1, .2])
    np.testing.assert_allclose(
        large.relations(2.5 * POINTS + [3, -1, .2]), graph.relations(POINTS), atol=1e-12)


def test_rigid_rotation_reads_as_its_angle_on_average():
    graph = PoseRelations(PARENTS, POINTS)
    reference = graph.relations(POINTS)
    axes = np.random.default_rng(0).normal(size=(400, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = []
    for axis in axes:
        turned = Rotation.from_rotvec(.2 * axis).apply(POINTS)
        whole, _ = graph.equivalent_angles(graph.changes(graph.relations(turned), reference), reference)
        angles.append(whole)
    assert abs(np.mean(angles) - .2) < .01


def test_proximal_changes_cost_more_than_distal_ones():
    objective, poses, _ = _objective(temporal=True)
    shape = ReferenceShape(objective, poses)
    envelope = shape.envelopes[0]
    coords = objective.coords

    def whole_angle(column):
        moved = poses.copy()
        moved[:, coords.qi[column]] -= .3
        shape.kin.evaluate(moved, with_jacobian=False)
        whole, parts, _ = envelope.measure(shape.kin)
        return envelope.equivalent(whole, parts)[0].max()

    hip, knee = whole_angle(0), whole_angle(1)
    assert hip > knee > 0


def test_soft_and_layout_rows_match_finite_differences():
    objective, poses, _ = _objective(temporal=True)
    envelope = objective.relation_envelope
    coords = objective.coords
    moved = poses.copy()
    moved[:, coords.qi] -= .25
    kin = objective.kin
    kin.evaluate(moved)
    weight = np.ones(len(moved))
    rows, jac, _ = envelope.hinge_rows(kin, weight, 1.0, True)
    step = np.zeros((len(moved), coords.width))
    step[:, coords.scalar_cols] = np.random.default_rng(1).normal(
        size=(len(moved), len(coords.scalar_cols))) * 1e-6
    kin.evaluate(coords.integrate(moved, step, 1.0), with_jacobian=False)
    after, _, _ = envelope.hinge_rows(kin, weight, 1.0, False)
    np.testing.assert_allclose(after, rows + np.einsum("frw,fw->fr", jac, step), atol=1e-9)


class _Positions:
    """Minimal kinematics stand-in: fixed landmark positions per frame."""

    def __init__(self, positions):
        self.positions = np.asarray(positions, dtype=float)

    def points(self, bodies, offsets):
        return self.positions


def test_layout_component_may_approach_zero_but_not_flip():
    from omniretargeting.pose_relations import RelationEnvelope, heading_axes

    # pelvis, chest, a hand clearly in front of the chest (x forward) and a
    # second hand, so the chest is a branch point of the trunk.
    one = np.array([[0, 0, 1.0], [0, 0, 1.5], [.3, 0, 1.4], [0, -.3, 1.2]])
    reference = np.repeat(one[None], 8, axis=0)
    envelope = RelationEnvelope([0, 1, 2, 3], np.zeros((4, 3)), [-1, 0, 1, 1], reference,
                                axes=heading_axes(np.eye(3)[None]))
    assert envelope.accepts(_Positions(reference))
    far = reference.copy()
    far[:, 2] = [.9, 0, 1.45]  # a large change that keeps every clear side
    assert envelope.accepts(_Positions(far))
    closer = reference.copy()
    closer[:, 2, 0] = .01  # crosses the numerical zero, but stays ambiguous
    assert envelope.accepts(_Positions(closer))
    behind = reference.copy()
    behind[:, 2, 0] = -.4  # clearly behind the chest for the whole interval
    assert not envelope.accepts(_Positions(behind))
    assert envelope.summary(_Positions(behind))["layout_flip_frames"] == list(range(6, 8))


def test_brief_clear_layout_crossing_is_not_collapse():
    from omniretargeting.pose_relations import RelationEnvelope, heading_axes

    one = np.array([[0, 0, 1.0], [0, 0, 1.5], [.3, 0, 1.4], [0, -.3, 1.2]])
    reference = np.repeat(one[None], 8, axis=0)
    envelope = RelationEnvelope([0, 1, 2, 3], np.zeros((4, 3)), [-1, 0, 1, 1], reference,
                                axes=heading_axes(np.repeat(np.eye(3)[None], 8, axis=0)))
    brief = reference.copy()
    brief[3, 2, 0] = -.4
    assert envelope.accepts(_Positions(brief))


def test_source_disagreement_is_not_a_hard_reference_layout_row():
    from omniretargeting.pose_relations import RelationEnvelope, heading_axes

    robot = np.array([[[0, 0, 1.0], [0, 0, 1.5], [.4, 0, 1.4], [0, -.3, 1.2]]])
    source = robot.copy()
    source[0, 2, 0] = -.4
    axes = heading_axes(np.eye(3)[None])
    envelope = RelationEnvelope([0, 1, 2, 3], np.zeros((4, 3)), [-1, 0, 1, 1],
                                robot, axes=axes, source_positions=source)
    reference_only = RelationEnvelope([0, 1, 2, 3], np.zeros((4, 3)), [-1, 0, 1, 1],
                                      robot, axes=axes)
    assert np.count_nonzero(envelope.layout_sign) < np.count_nonzero(reference_only.layout_sign)
    assert envelope.accepts(_Positions(robot))


def test_bends_within_one_limb_are_not_layout_flips():
    from omniretargeting.pose_relations import limb_ids

    # pelvis -> (left hip -> knee), (right hip -> knee), spine -> (arm, arm)
    parents = [-1, 0, 1, 0, 3, 0, 5, 5]
    limb = limb_ids(parents)
    assert limb[0] == limb[5] == -1  # trunk: pelvis and spine
    assert limb[1] == limb[2] != limb[3] == limb[4]
    assert limb[6] != limb[7]


def test_soft_posture_scale_follows_the_robot_morphology_angle():
    objective, poses, _ = _objective(temporal=True)
    envelope = objective.relation_envelope
    # Identical robot and source bodies: the base posture scale applies.
    assert envelope.soft_angle == pytest.approx(.2)
    human = np.array(objective.tasks.human, copy=True)
    human[..., 0] *= 4.0  # a source body of very different proportions
    human[..., 2] *= .5
    objective.tasks.human = human
    objective.kin.evaluate(poses, with_jacobian=False)
    from omniretargeting.pose_relations import RelationEnvelope

    wider = RelationEnvelope.from_tasks(objective.tasks, objective.kin)
    assert wider.morphology_angle > .2
    assert wider.soft_angle == pytest.approx(wider.morphology_angle)


def test_pruned_conic_constraints_keep_the_same_optimum():
    import cvxpy as cp
    from omniretargeting._fast.reference import ReferenceShape

    objective, poses, _ = _objective(temporal=True)
    shape = ReferenceShape(objective, poses)
    coords = objective.coords
    moved = poses.copy()
    moved[:, coords.qi] -= .15  # close to the limits in some frames
    count = len(poses) * coords.width
    target = np.random.default_rng(2).normal(size=count) * .2
    box = np.full((len(poses), coords.width), .05)

    def solve(pruned):
        flat = cp.Variable(count)
        constraints = [flat >= -box.ravel(), flat <= box.ravel()]
        constraints += shape.constraints(moved, flat, box if pruned else None)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(flat - target)), constraints)
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
        return flat.value, problem.value

    full, full_value = solve(False)
    pruned, pruned_value = solve(True)
    # Same strictly convex optimum up to the interior-point tolerance.
    assert pruned_value == pytest.approx(full_value, rel=1e-8)
    np.testing.assert_allclose(pruned, full, atol=1e-5)


def test_identical_layout_rows_of_both_references_merge_with_the_same_optimum():
    import cvxpy as cp
    from omniretargeting._fast.reference import ReferenceShape
    from omniretargeting.pose_relations import layout_constraints

    objective, poses, _ = _objective(temporal=True)
    coords = objective.coords
    shape = ReferenceShape(objective, poses)
    frozen = poses.copy()
    # A roundoff-level change in a few frames, as quaternion normalization
    # leaves between the seed and the acquired reference, and one real change.
    frozen[:3, coords.qi[0]] += 1e-15
    frozen[5, coords.qi[-1]] += .05
    shape.freeze_reference(frozen)
    assert len(shape.envelopes) == 2
    moved = poses.copy()
    moved[:, coords.qi] -= .15
    count = len(poses) * coords.width
    target = np.random.default_rng(3).normal(size=count) * .2
    box = np.full((len(poses), coords.width), .05)

    def solve(shared):
        flat = cp.Variable(count)
        shape.kin.evaluate(moved)
        bounds = [envelope.linear_bounds(shape.kin, coords.width) for envelope in shape.envelopes]
        layout = (layout_constraints(bounds, flat) if shared
                  else [c for b in bounds for c in layout_constraints([b], flat)])
        constraints = [flat >= -box.ravel(), flat <= box.ravel()] + layout
        problem = cp.Problem(cp.Minimize(cp.sum_squares(flat - target)), constraints)
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
        return flat.value, problem.value, sum(c.size for c in layout)

    separate, separate_value, separate_rows = solve(False)
    shared, shared_value, shared_rows = solve(True)
    assert 0 < shared_rows < separate_rows
    assert shared_value == pytest.approx(separate_value, rel=1e-8)
    np.testing.assert_allclose(shared, separate, atol=1e-5)


def test_fit_rotations_is_the_kabsch_rotation():
    from omniretargeting.pose_relations import _fit_rotations

    rng = np.random.default_rng(4)
    source = rng.normal(size=(5, 9, 3))
    truth = Rotation.random(5, random_state=1).as_matrix()
    reference = np.einsum("fij,fkj->fki", truth, source) + rng.normal(size=(5, 1, 3))
    fitted = _fit_rotations(reference, source)
    np.testing.assert_allclose(fitted, truth, atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(fitted), 1.0)


def _trunk_figure(chest_height):
    """pelvis(root) -> chest -> two shoulders -> elbows; pelvis -> two hips."""
    parents = [-1, 0, 1, 2, 1, 4, 0, 0]
    points = np.array([
        [0, 0, 1.0], [0, 0, chest_height], [0, .2, 1.45], [0, .25, 1.15],
        [0, -.2, 1.45], [0, -.25, 1.15], [0, .1, .95], [0, -.1, .95],
    ])
    return parents, points[None]


def test_misplaced_trunk_landmark_does_not_judge_source_layout():
    from omniretargeting.pose_relations import RelationEnvelope, comparable_trunk, heading_axes

    parents, source = _trunk_figure(1.3)      # chest point near the shoulders
    _, robot = _trunk_figure(1.04)            # robot origin at waist height
    keep = comparable_trunk(parents, robot, source)
    assert not keep[1] and keep[0]
    axes = heading_axes(np.eye(3)[None])
    envelope = RelationEnvelope(range(8), np.zeros((8, 3)), parents, robot, axes=axes,
                                source_positions=source)
    first, second = envelope.graph.pairs.T
    involved = (first == 1) | (second == 1)
    assert not envelope.source_layout_sign[:, involved].any()
    assert envelope.source_layout_sign[:, ~involved].any()
    # Hanging elbows: below the source chest, above the robot waist point.
    assert not envelope.source_flips(_Positions(robot)).any()
    _, matched = _trunk_figure(1.3)
    assert comparable_trunk(parents, matched, source).all()


def test_source_layout_flip_still_detects_an_arm_behind_the_body():
    from omniretargeting.pose_relations import LAYOUT_FLIP_WINDOW_FRAMES, RelationEnvelope, heading_axes

    frames = LAYOUT_FLIP_WINDOW_FRAMES + 3
    parents, source = _trunk_figure(1.3)
    source = np.repeat(source, frames, axis=0)
    source[:, 3, 0] = .3                      # left elbow clearly in front
    robot = source.copy()
    axes = heading_axes(np.repeat(np.eye(3)[None], frames, axis=0))
    envelope = RelationEnvelope(range(8), np.zeros((8, 3)), parents, robot, axes=axes,
                                source_positions=source)
    assert not envelope.source_flips(_Positions(robot)).any()
    behind = robot.copy()
    behind[:, 3, 0] = -.3
    flagged = envelope.source_flips(_Positions(behind)) > envelope.SLACK
    assert flagged[LAYOUT_FLIP_WINDOW_FRAMES - 1:].all()
