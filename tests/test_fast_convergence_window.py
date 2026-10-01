"""Guard clipping cannot turn two small curvature corrections into convergence."""

from omniretargeting._fast.solver import _progress_converged


def test_short_steps_after_material_descent_wait_for_the_existing_window():
    costs, slow = [100.], 0
    for cost, alpha in ((80., 1.), (79.9, .5), (79.8, .5)):
        slow, done = _progress_converged(costs, cost, slow, alpha, True, .003)
        assert not done
    # A genuinely flat three-step window still stops without increasing the
    # iteration cap or requiring a full conic step at a curved boundary.
    slow, done = _progress_converged(costs, 79.7, slow, .5, True, .003)
    assert done


def test_two_small_full_steps_still_stop_an_unconstrained_pass():
    costs, slow = [100.], 0
    slow, done = _progress_converged(costs, 99.9, slow, 1., False, .003)
    assert not done
    slow, done = _progress_converged(costs, 99.8, slow, 1., False, .003)
    assert done


def test_zero_threshold_does_not_force_early_termination():
    costs, slow = [100.], 0
    for cost in (80., 79.9, 79.8, 79.7, 79.7):
        slow, done = _progress_converged(costs, cost, slow, .5, True, 0.)
        assert not done
