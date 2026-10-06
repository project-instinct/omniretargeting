"""Fixed-reference all-frame spline correction and local SO(3) derivatives."""

import mujoco
import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.transform import Rotation

from omniretargeting._fast.spline_embedding import CorrectionSpline
from omniretargeting._sequence.coordinates import Coordinates


def make_reference(count=121, **locks):
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody><body><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
        <body pos="0 0 .2"><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".05 .1" mass="1"/>
        <body pos="0 0 .2"><joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".05" mass="1"/>
        </body></body></body></worldbody></mujoco>
        """)
    coords = Coordinates(model, **locks)
    times = np.arange(count) / 60
    reference = np.zeros((count, model.nq))
    reference[:, :3] = np.column_stack((times, .1 * times**2, 1 + .1 * times))
    reference[:, 3:7] = Rotation.from_euler('xyz', np.column_stack(
        (.5 * times, -.3 * times**2, .8 * times))).as_quat(scalar_first=True)
    reference[1::2, 3:7] *= -1
    reference[:, 7:] = np.column_stack((.2 * np.sin(times), .03 * times))
    return coords, reference


@pytest.mark.parametrize('identity', [False, True])
@pytest.mark.parametrize('locks', [{}, {'lock_root_xy': True},
                                    {'lock_root_orientation': True},
                                    {'lock_root_z': True}])
def test_all_frame_embedding_matches_integrate_and_freezes_reference(identity, locks):
    coords, reference = make_reference(**locks)
    spline = CorrectionSpline(coords, reference, 60, identity=identity)
    assert sparse.isspmatrix_csr(spline.basis)
    assert spline.basis.shape[0] == len(reference)
    np.testing.assert_array_equal(spline.embed(spline.coefficients), reference)
    a = np.random.default_rng(4).normal(scale=.25, size=spline.coefficients.shape)
    expected = coords.integrate(reference, spline.basis @ a, 1.)
    actual = spline.embed(a)
    np.testing.assert_allclose(coords.difference(expected, actual), 0, atol=2e-15)
    np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=2e-15)
    np.testing.assert_allclose(actual[:, 7:], expected[:, 7:], atol=2e-15)
    saved = spline.reference.copy()
    reference[:] = 0
    np.testing.assert_array_equal(spline.reference, saved)
    assert not spline.reference.flags.writeable
    if identity:
        np.testing.assert_array_equal(spline.basis.toarray(), np.eye(len(reference)))


@pytest.mark.parametrize('count', [1, 2, 7, 121])
def test_partition_of_unity_and_cubic_local_support(count):
    coords, reference = make_reference(count)
    spline = CorrectionSpline(coords, reference, 60)
    np.testing.assert_allclose(np.asarray(spline.basis.sum(axis=1)).ravel(), 1, atol=5e-16)
    assert np.max(spline.basis.getnnz(axis=1)) <= 4
    dense_times = np.linspace(0, spline.end_time, 313)
    np.testing.assert_allclose(spline.basis_at(dense_times) @ np.ones(spline.controls), 1,
                               atol=6e-16)
    for order in (1, 2):
        np.testing.assert_allclose(spline.basis_at(dense_times, order)
                                   @ np.ones(spline.controls), 0, atol=4e-12)


@pytest.mark.parametrize('identity', [False, True])
def test_right_local_rotation_jacobian_and_scalar_objective_gradient(identity):
    coords, reference = make_reference(37)
    spline = CorrectionSpline(coords, reference, 60, identity=identity)
    rng = np.random.default_rng(13)
    a = rng.normal(scale=.7, size=spline.coefficients.shape)
    a[:, spline.angular] += np.array([2.8, -.9, 1.2])
    direction = rng.normal(size=a.shape)
    epsilon = 2e-7
    poses = spline.embed(a)
    plus, minus = spline.embed(a + epsilon * direction), spline.embed(a - epsilon * direction)
    finite_difference = (coords.difference(poses, plus)
                         - coords.difference(poses, minus)) / (2 * epsilon)
    jacobian = spline.jacobian(a)
    assert sparse.isspmatrix_csr(jacobian)
    assert jacobian.shape == (spline.count * coords.width, spline.controls * coords.width)
    analytic = (jacobian @ direction.ravel()).reshape(finite_difference.shape)
    differential = spline.tangent_differentials(a)
    factored = np.einsum('nij,nj->ni', differential, spline.basis @ direction)
    np.testing.assert_allclose(factored, analytic, atol=2e-15)
    np.testing.assert_allclose(analytic, finite_difference, rtol=3e-7, atol=2e-8)
    # An independently evaluated rotation/translation/scalar objective exercises
    # the adjoint chain used by downstream full-frame residuals.
    target = spline.embed(rng.normal(scale=.2, size=a.shape))
    def objective(q):
        displacement = coords.difference(target, q)
        return .5 * np.sum(displacement**2)
    local = rng.normal(size=(spline.count, coords.width))
    # Compute the pose-local gradient without quaternion subtraction.
    local_gradient = np.zeros_like(local)
    for column in range(coords.width):
        step = np.zeros_like(local)
        step[:, column] = epsilon
        p = coords.difference(target, coords.integrate(poses, step, 1.))
        m = coords.difference(target, coords.integrate(poses, step, -1.))
        local_gradient[:, column] = (.5 * np.sum(p**2 - m**2, axis=1)) / (2 * epsilon)
    analytic_scalar = (jacobian.T @ local_gradient.ravel()) @ direction.ravel()
    finite_scalar = (objective(plus) - objective(minus)) / (2 * epsilon)
    np.testing.assert_allclose(analytic_scalar, finite_scalar, rtol=2e-7, atol=2e-6)


def test_knot_insertion_inherits_curve_and_first_two_derivatives_everywhere():
    coords, reference = make_reference()
    spline = CorrectionSpline(coords, reference, 60)
    rng = np.random.default_rng(17)
    a = rng.normal(scale=.3, size=spline.coefficients.shape)
    times = np.unique(np.r_[np.linspace(0, spline.end_time, 537),
                            spline.knot_vector, [.231, .713, 1.417]])
    before = spline.embed(a)
    refined, inherited = spline.insert_knots([.231, .713, 1.417, .231], a)
    assert refined.controls == spline.controls + 3
    assert refined.reference is spline.reference
    assert spline.controls == len(a)
    np.testing.assert_array_equal(refined.coefficients, inherited)
    for order in range(3):
        np.testing.assert_allclose(refined.correction(inherited, times, order),
                                   spline.correction(a, times, order), rtol=3e-13, atol=6e-12)
    np.testing.assert_allclose(coords.difference(before, refined.embed(inherited)), 0,
                               atol=3e-15)
    unchanged, same = refined.insert_knots([.231], inherited)
    assert unchanged.controls == refined.controls
    np.testing.assert_array_equal(same, inherited)


def test_fully_supported_controls_preserve_exterior_and_window_boundary_derivatives():
    coords, reference = make_reference(241)
    spline = CorrectionSpline(coords, reference, 60)
    start, stop = 43, 187
    controls = spline.local_controls(start, stop)
    assert len(controls) > 0
    rng = np.random.default_rng(19)
    a = rng.normal(scale=.3, size=spline.coefficients.shape)
    changed = a.copy()
    changed[controls] += rng.normal(scale=.4, size=(len(controls), coords.width))
    affected = spline.affected_frames(controls)
    assert np.all((affected >= start) & (affected < stop))
    outside = np.r_[np.arange(start), np.arange(stop, spline.count)]
    np.testing.assert_array_equal(spline.embed(a)[outside], spline.embed(changed)[outside])
    left, right = spline.times[start], spline.times[stop - 1]
    exterior = np.r_[np.linspace(0, left, 53), np.linspace(right, spline.end_time, 61)]
    for order in range(3):
        np.testing.assert_array_equal(spline.correction(a, exterior, order),
                                      spline.correction(changed, exterior, order))
    all_local = spline.local_controls(0, spline.count)
    assert not np.isin([0, 1, 2, spline.controls - 3, spline.controls - 2,
                       spline.controls - 1], all_local).any()
    assert len(spline.affected_frames([])) == 0
    assert len(spline.local_controls(0, 0)) == 0


@pytest.mark.parametrize('identity', [False, True])
def test_invalid_parameters_and_coefficient_shape(identity):
    coords, reference = make_reference(3)
    for rate in (0, -1, np.nan):
        with pytest.raises(ValueError):
            CorrectionSpline(coords, reference, rate)
    for spacing in (0, -1, np.nan):
        with pytest.raises(ValueError):
            CorrectionSpline(coords, reference, 60, spacing=spacing)
    spline = CorrectionSpline(coords, reference, 60, identity=identity)
    with pytest.raises(ValueError):
        spline.embed(np.zeros((1, 1)))
    with pytest.raises(ValueError):
        spline.local_controls(-1, 2)
    with pytest.raises(ValueError):
        spline.insert_knots([0], spline.coefficients)
    if identity:
        with pytest.raises(ValueError):
            spline.insert_knots([.01], spline.coefficients)
