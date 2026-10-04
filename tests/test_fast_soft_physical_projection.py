"""Physical projection uses the actual window objective and crossing rows."""

import numpy as np
from scipy.linalg import block_diag

from omniretargeting._fast.legacy_window_qp import WindowObjective
from omniretargeting._fast.soft_projection_physical import model, PROXIMITY
from tests.test_fast_legacy_window import problem


def test_physical_projection_gradient_and_psd_with_nonzero_fixed_neighbours():
    objective, fixed = problem(extra=True)
    poses = fixed.copy()
    poses[:, 7:] += .001*np.sin(np.arange(len(poses))[:, None]/3.)
    window = WindowObjective(objective, poses, range(8, 26))
    local = poses[window.span]
    columns = np.arange(window.frames*window.width)
    report = {}
    hessian, gradient, cost, _ = model(window, local, local, columns, report)
    acc, residual, temporal, _, _ = window.evaluate(local)
    dense = block_diag(*acc.hessian)+(temporal.T@temporal).toarray()
    expected = dense+np.diag(PROXIMITY*np.maximum(np.diag(dense), 1.))
    np.testing.assert_allclose(hessian.toarray(), expected, atol=1e-8, rtol=1e-12)
    assert np.linalg.eigvalsh(hessian.toarray()).min() > 0.
    direction = np.random.default_rng(7).normal(size=(window.frames, window.width))
    epsilon = 1e-6
    values = [window.evaluate(window.coords.integrate(local, direction, sign*epsilon), False)[0].cost
              for sign in (-1, 1)]
    finite = (values[1]-values[0])/(4*epsilon)
    np.testing.assert_allclose(gradient@direction.ravel(), finite, rtol=1e-6, atol=1e-4)
    assert cost == acc.cost
    assert report['physical_model_evaluations'] == 1


def test_proximity_gradient_is_separate_from_the_physical_rows():
    objective, poses = problem(extra=True)
    window = WindowObjective(objective, poses, range(8, 26))
    local = poses[window.span]
    target = local.copy()
    target[:, 7] += .02
    columns = np.arange(window.frames*window.width)
    hessian, centered, _, _ = model(window, local, local, columns, {})
    moved, gradient, _, _ = model(window, local, target, columns, {})
    np.testing.assert_array_equal(hessian.toarray(), moved.toarray())
    diagonal = hessian.diagonal()/(1+PROXIMITY)
    desired = window.coords.difference(local, target).ravel()
    np.testing.assert_allclose(gradient-centered, -PROXIMITY*diagonal*desired, atol=1e-6)


def test_equilibrated_diagonal_majorizer_bounds_every_physical_direction():
    from omniretargeting._fast.soft_projection_physical import majorizer, curvature
    objective, poses = problem(extra=True)
    acc, _, temporal, _, _ = objective.evaluate(poses)
    physical = curvature(acc, temporal)
    diagonal = majorizer(physical)
    delta = np.diag(diagonal)-physical.toarray()
    assert np.linalg.eigvalsh(delta).min() > -1e-7


def test_first_order_model_retains_physical_gradient_in_natural_units():
    objective, poses = problem(extra=True)
    window = WindowObjective(objective, poses, range(8, 26))
    local = poses[window.span]
    columns = np.arange(window.frames*window.width)
    full, gradient, *_ = model(window,local,local,columns,{})
    diagonal, first_order, *_ = model(window,local,local,columns,{'physical_first_order':True})
    np.testing.assert_array_equal(first_order,gradient)
    np.testing.assert_allclose(diagonal.toarray(),np.diag(full.diagonal()),atol=1e-8)
