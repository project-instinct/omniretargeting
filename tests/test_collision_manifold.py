"""Physical coverage and merge invariance of bounded collision witnesses."""

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.collision_manifold import select_collision_manifold


def _plane():
    x, y = np.meshgrid(np.linspace(-.1, .1, 11), np.linspace(-.05, .05, 13))
    points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -.01)])
    return points, np.full(len(points), -.01), np.tile([0., 0., 1.], (len(points), 1))


def _physical(points, clearance, directions, selected):
    normals = directions[selected] / np.linalg.norm(directions[selected], axis=1, keepdims=True)
    return sorted(map(tuple, np.column_stack([points[selected], clearance[selected], normals])))


def test_same_normal_keeps_all_planar_corners_and_the_deepest_point():
    points, clearance, directions = _plane()
    center = np.argmin(np.linalg.norm(points[:, :2], axis=1))
    clearance[center] = -.011
    chosen = select_collision_manifold(points, clearance, directions)
    assert chosen[0] == center
    corners = {(x, y) for x in (-.1, .1) for y in (-.05, .05)}
    assert corners <= set(map(tuple, points[chosen, :2]))
    assert len(chosen) == 5
    # Force normals alone are identical, while moments about the origin differ.
    assert len(np.unique(directions[chosen], axis=0)) == 1
    assert len(np.unique(np.cross(points[chosen], directions[chosen]), axis=0)) == 5


def test_distinct_force_normals_are_covered_even_without_lever_arms():
    directions = np.vstack([np.eye(3), -np.eye(3)])
    points, clearance = np.zeros((6, 3)), np.full(6, -.01)
    chosen = select_collision_manifold(points, clearance, directions)
    assert set(chosen) == set(range(6))


def test_repetition_reordering_and_normal_magnitude_do_not_weight_the_manifold():
    rng = np.random.default_rng(92)
    points = rng.normal(size=(200, 3))
    clearance = rng.uniform(-.03, .006, len(points))
    directions = rng.normal(size=points.shape)
    chosen = select_collision_manifold(points, clearance, directions)
    expected = _physical(points, clearance, directions, chosen)
    repeats = np.r_[np.arange(len(points)), np.repeat(np.arange(17), 83)]
    rng.shuffle(repeats)
    p, c, n = points[repeats], clearance[repeats], directions[repeats]
    selected = select_collision_manifold(p, c, n)
    assert _physical(p, c, n, selected) == expected
    # Powers of two preserve exact normalization while changing raw magnitudes.
    scaled = directions * 8.
    np.testing.assert_array_equal(select_collision_manifold(points, clearance, scaled), chosen)


@pytest.mark.parametrize("limit", [1, 4, 8, 12, 16])
def test_incremental_summaries_match_full_union_for_arbitrary_batches(limit):
    rng = np.random.default_rng(31)
    points = rng.normal(size=(601, 3))
    clearance = rng.uniform(-.01, .02, len(points))
    clearance[-1] = -.2  # A late deepest point must not change the candidate rule.
    directions = rng.normal(size=points.shape)
    parameters = {"max_points": limit, "activation_distance": .008, "origin": [.3, -.2, .7]}
    full = select_collision_manifold(points, clearance, directions, **parameters)
    expected = _physical(points, clearance, directions, full)
    blocks = np.array_split(rng.permutation(len(points)), 17)
    survivors = np.empty(0, dtype=int)
    partials = []
    for block in blocks:
        pool = np.r_[survivors, block]
        survivors = pool[select_collision_manifold(points[pool], clearance[pool], directions[pool], **parameters)]
        partials.extend(block[select_collision_manifold(points[block], clearance[block], directions[block], **parameters)])
        assert len(survivors) <= limit
    assert _physical(points, clearance, directions, survivors) == expected
    partials = np.asarray(partials)
    combined = partials[select_collision_manifold(points[partials], clearance[partials], directions[partials], **parameters)]
    assert _physical(points, clearance, directions, combined) == expected
    assert full[0] == len(points) - 1


def test_large_dense_face_keeps_extreme_corners_without_density_weighting():
    rng = np.random.default_rng(21)
    xy = rng.uniform([-.1, -.05], [.1, .05], size=(200_000, 2))
    xy = np.vstack([xy, [[x, y] for x in (-.1, .1) for y in (-.05, .05)], [0., 0.]])
    points = np.column_stack([xy, np.full(len(xy), -.01)])
    clearance = np.full(len(points), -.01)
    clearance[-1] = -.011
    directions = np.tile([0., 0., 1.], (len(points), 1))
    chosen = select_collision_manifold(points, clearance, directions)
    assert set(chosen) == set(range(200_000, len(points)))
    assert chosen[0] == len(points) - 1


def test_degenerate_points_and_collinear_samples_need_no_convex_hull():
    points = np.zeros((30, 3))
    clearance = np.full(30, -.01)
    directions = np.tile([0., 0., 1.], (30, 1))
    assert len(select_collision_manifold(points, clearance, directions)) == 1
    points[:, 0] = np.linspace(-1., 1., len(points))
    chosen = select_collision_manifold(points, clearance, directions)
    assert set(chosen) == {0, 29}
    points[:, 0] = 0.
    points[:, 2] = np.arange(len(points))
    assert len(select_collision_manifold(points, clearance, directions)) == 1


def test_unreliable_nonfinite_zero_normals_and_inactive_points_are_excluded():
    points = np.zeros((8, 3))
    clearance = np.array([-.02, -.01, -.03, -.04, np.nan, -.05, .008, .007])
    directions = np.tile([0., 0., 1.], (len(points), 1))
    valid = np.ones(len(points), dtype=bool)
    valid[0] = False
    points[2, 0] = np.nan
    directions[3] = 0.
    directions[5, 0] = np.inf
    chosen = select_collision_manifold(points, clearance, directions, valid=valid)
    assert chosen.tolist() == [1]
    assert select_collision_manifold(points, clearance, directions, valid=np.zeros(8, bool)).size == 0
    assert select_collision_manifold(np.empty((0, 3)), np.empty(0), np.empty((0, 3))).size == 0
    assert select_collision_manifold(np.zeros((2, 3)), np.ones(2), np.tile([0., 0., 1.], (2, 1))).size == 0
    # A fixed trust-distance band can include a future contact; no active-depth
    # cutoff later removes it when a deeper sample arrives in another batch.
    assert select_collision_manifold(points[7:8], clearance[7:8], directions[7:8]).tolist() == [0]
    assert select_collision_manifold(points[7:8], clearance[7:8], directions[7:8], activation_distance=0.).size == 0


@pytest.mark.parametrize("kwargs", [{"max_points": 0}, {"max_points": -1}, {"max_points": 2.5},
                                    {"activation_distance": -.1}, {"activation_distance": np.nan},
                                    {"origin": [1., 2.]}, {"origin": [0., np.inf, 0.]}])
def test_invalid_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        select_collision_manifold(np.zeros((1, 3)), [-.01], [[0., 0., 1.]], **kwargs)


def _solve_pitch_epigraph(jacobian, clearance):
    clarabel = pytest.importorskip("clarabel")
    # q is a fixed-root ankle rotation; one shared depth variable t is used for
    # the whole geometry, independent of the number of retained witnesses.
    matrix = np.vstack([[[1., 0.], [-1., 0.], [0., -1.]],
                        np.column_stack([-jacobian, np.zeros(len(jacobian))]),
                        np.column_stack([-jacobian, -np.ones(len(jacobian))])])
    rhs = np.r_[.005, .005, 0., .01 + clearance, clearance]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-10
    solution = clarabel.DefaultSolver(
        sparse.diags([1. / .05**2, 1. / .01**2], format="csc"), np.zeros(2),
        sparse.csc_matrix(matrix), rhs, [clarabel.NonnegativeConeT(len(rhs))], settings).solve()
    assert str(solution.status) in {"Solved", "AlmostSolved"}
    return float(solution.x[0])


def test_manifold_prevents_flat_sole_wrong_direction_in_a_real_constrained_qp():
    points, clearance, directions = _plane()
    selected = select_collision_manifold(points, clearance, directions)
    jacobian = -points[:, 0]  # d(clearance)/d(pitch) about the ankle's Y axis.
    single = _solve_pitch_epigraph(jacobian[:1], clearance[:1])
    multiple = _solve_pitch_epigraph(jacobian[selected], clearance[selected])
    assert single > .0049
    assert abs(multiple) < 1e-8
    # The ankle center stays at z=.04, and the rigid sole is .05 below it.
    # This is the exact nonlinear plane clearance of every original sample.
    exact_single = .04 - .05 * np.cos(single) - points[:, 0] * np.sin(single)
    exact_multiple = .04 - .05 * np.cos(multiple) - points[:, 0] * np.sin(multiple)
    assert clearance[0] + jacobian[0] * single > -.00951
    assert exact_single.min() < -.01049
    assert exact_multiple.min() >= -.010000001
    # The rejected single-witness proposal still fits a 1 mm contact tube and
    # introduces no temporal derivatives when applied to a constant sequence.
    assert 2. * np.hypot(.1, .05) * np.sin(abs(single) / 2.) < .001
    assert np.count_nonzero(np.diff(np.full(4, single))) == 0
